from labml import experiment, lab

if __name__ == '__main__':
    experiment.save_bundle(lab.get_path() / 'bundle.tar.gz', '248c453ad0e311ec84dd35759c7d3eba',
                           data_files=['cache/bpe.json'])
