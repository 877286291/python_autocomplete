import os
import re
from pathlib import Path
from typing import List

import torch
from labml import logger, lab, monit

from dataset import tokenizer
from evaluate import Predictor
from evaluate.beam_search import NextWordPredictionComplete
from evaluate.factory import get_predictor


# from create_dataset import get_python_files


def _fix_indentation(parsed: List[tokenizer.ParsedToken]) -> List[tokenizer.ParsedToken]:
    """
    Change indentation tokens. Remove `DEDENT` tokens and
    add `INDENT` tokens to each line.
    This is easier for prediction.
    """
    res: List[tokenizer.ParsedToken] = []
    indentation = 0
    indented = False
    for t in parsed:
        if t.type == tokenizer.TokenType.indent:
            indentation += 1
        elif t.type == tokenizer.TokenType.dedent:
            indentation -= 1
        elif t.type in [tokenizer.TokenType.new_line,
                        tokenizer.TokenType.eof]:
            indented = False
            res.append(t)
        else:
            if not indented:
                for _ in range(indentation):
                    res.append(tokenizer.ParsedToken(tokenizer.TokenType.indent, 0))
                indented = True

            res.append(t)

    return res


def _remove_comments(parsed: List[tokenizer.ParsedToken]) -> List[tokenizer.ParsedToken]:
    """
    Remove comment tokens
    """
    res = []
    for p in parsed:
        if p.type == tokenizer.TokenType.comment:
            continue
        else:
            res.append(p)

    return res


def _remove_empty_lines(parsed: List[tokenizer.ParsedToken]) -> List[tokenizer.ParsedToken]:
    """
    Remove empty lines
    """

    tokens = [tokenizer.TokenType.new_line, tokenizer.TokenType.new_line]
    res = []
    for p in parsed:
        for i in range(1):
            tokens[i] = tokens[i + 1]
        tokens[-1] = p.type
        all_new_line = True
        for t in tokens:
            if t != tokenizer.TokenType.new_line:
                all_new_line = False

        if all_new_line:
            continue
        else:
            res.append(p)

    return res


def evaluate(predictor: Predictor, text: str):
    stripped, prompt = predictor.rstrip(text)
    rest = text[len(stripped):]
    prediction_complete = NextWordPredictionComplete(rest, 3)
    prompt = torch.tensor(prompt, dtype=torch.long).unsqueeze(-1)

    predictions = predictor.get_next_word(prompt, None, rest, [1.], prediction_complete, 10)
    predictions.sort(key=lambda x: -x.prob)
    results = [pred.text[len(rest):] for pred in predictions]
    # print(results)
    return results


def get_python_files(file_dir):
    file_list = []
    for file in file_dir.glob("**/*.py"):
        file_list.append(file)
    return file_list


def get_caller(rec):
    nrec = re.sub(r'\(.*\)', '', rec)
    pindex = nrec.rfind('.')
    return nrec[:pindex]


def get_callee(rec):
    nrec = re.sub(r'\(.*\)', '', rec)
    pindex = nrec.rfind('.')
    return nrec[pindex + 1:]


def main():
    predictor = get_predictor()
    files = os.listdir(lab.get_data_path() / "pyart/valid")
    for file in files:
        top1 = 0
        top5 = 0
        top10 = 0
        total = 0
        reciprocal_rank = 0
        project_name = file
        sample_files = get_python_files(Path(lab.get_data_path() / 'pyart' / 'valid' / project_name))
        reg = re.compile(r'[a-zA-Z0-9_\.\[\]]+\.[a-zA-Z0-9\_]+\(.*\)')
        for _, sample_file in monit.enum("Process:", sample_files):
            with open(str(sample_file), 'r', encoding='iso-8859-1') as f:
                sample = f.read()
            # parsed = parse_string(sample)
            # parsed = _remove_comments(parsed)
            # parsed = _remove_empty_lines(parsed)
            # parsed = _fix_indentation(parsed)
            # sample = to_string(parsed)
            points = re.finditer(reg, sample)
            for point in points:
                start, end = point.span()
                caller = get_caller(point.group())
                label = get_callee(point.group())
                # if label.startswith('_'):
                #     continue
                data = sample[:start + len(caller)]
                if len(data) > 2048:
                    data = data[-2048:]
                result = evaluate(predictor, data + ".")
                if label in result:
                    rank_i = result.index(label) + 1
                    reciprocal_rank += 1 / rank_i
                    if rank_i == 1:
                        top1 += 1
                    if rank_i < 6:
                        top5 += 1
                    top10 += 1
                # else:
                #     print(caller, label, result)
                total += 1
        logger.inspect(Project=project_name, Top1=top1 / total, Top5=top5 / total, Top10=top10 / total,
                       MRR=reciprocal_rank / total, Total=total)


if __name__ == '__main__':
    main()
