from labml import experiment, lab
from labml.utils.pytorch import get_modules
from evaluate import Predictor
from train import Configs


def load_experiment() -> Configs:
    conf = Configs()
    experiment.evaluate()

    # This will download a pretrained model checkpoint and some cached files.
    # It will download the archive as `saved_checkpoint.tar.gz` and extract it.
    #
    # If you have a locally trained model load it directly with
    # run_uuid = 'RUN_UUID'
    # And for latest checkpoint
    # checkpoint = None

    run_uuid = '8e929d1fcc2f11eca22718c04d1fbab2'
    checkpoint = None
    # run_uuid, checkpoint = experiment.load_bundle(
    #     lab.get_path() / 'bundle.tar.gz',
    #     url='https://github.com/lab-ml/python_autocomplete/releases/download/0.0.5/bundle.tar.gz')

    conf_dict = experiment.load_configs(run_uuid)
    conf_dict['text.is_load_data'] = False
    conf_dict['device.cuda_device'] = 1
    experiment.configs(conf, conf_dict)
    experiment.add_pytorch_models(get_modules(conf))
    experiment.load(run_uuid, checkpoint)

    experiment.start()

    return conf


def get_predictor() -> Predictor:
    conf = load_experiment()
    conf.model.eval()
    return Predictor(conf.model, conf.text.tokenizer,
                     state_updater=conf.state_updater,
                     is_token_by_token=conf.is_token_by_token)
