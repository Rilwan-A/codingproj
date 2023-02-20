from train import ImputationTrainer
import os
from argparse import ArgumentParser
import copy
from __init__ import MAP_NAME_DIFFUSION, MAP_NAME_DSET, MAP_MNAME_MODEL


if __name__ == "__main__":
    
    parent_parser = ArgumentParser(add_help=False, allow_abbrev=False)

    # args trainer
    config_trainer = ImputationTrainer.parse_config(parent_parser)
    
    # add model specific args
    config_model = MAP_MNAME_MODEL[config_trainer.model_name].parse_config(parent_parser)
    
    # add data specific args
    config_data = MAP_NAME_DSET[config_trainer.dataset_name].parse_config(parent_parser)

    # add diffusion specific args
    config_diffusion = MAP_NAME_DIFFUSION[config_trainer.diffusion_method].parse_config(parent_parser)
    
    ImputationTrainer.train(config_trainer, config_data, config_diffusion, config_model )
    
    # Holiday Test Run
    config_trainer_hol = copy.deepcopy(config_trainer)
    config_diffusion_hol = copy.deepcopy(config_diffusion)

    config_trainer_hol.test_set_method = 'holidays_only'
    config_diffusion_hol.mask_method = 'bm_channelgroup'
    ImputationTrainer.test(config_trainer_hol, config_data, config_diffusion_hol, config_model, )
    
    #Hacky renaming of the yaml file containing test results to avoid it being overwritten by the next test run
    dir_outp = os.path.join(os.path.join(config_trainer.dir_ckpt, config_trainer.exp_name, 'test' ))
    
    os.rename(
        os.path.join(dir_outp,'pred_outp.pkl'),
        os.path.join(dir_outp,f'pred_outp_{config_trainer_hol.test_set_method}_.pkl')    )
    os.rename(
        os.path.join(dir_outp,'pred_metrics.yaml'),
        os.path.join(dir_outp,f'pred_metrics_{config_trainer_hol.test_set_method}_.yaml')    )
    
    
    # Batch Missing Test Run
    config_trainer_bm = copy.deepcopy(config_trainer)
    config_diffusion_bm = copy.deepcopy(config_diffusion)
    
    config_trainer_bm.test_set_method = 'normal'
    config_diffusion_bm.mask_method = 'bm_channelgroup'
    ImputationTrainer.test(config_trainer_bm, config_data, config_diffusion_bm, config_model)
    
    os.rename(
        os.path.join(dir_outp,'pred_outp.pkl'),
        os.path.join(dir_outp,f'pred_outp_{config_trainer_bm.test_set_method}_.pkl')    )
    os.rename(
        os.path.join(dir_outp,'pred_metrics.yaml'),
        os.path.join(dir_outp,f'pred_metrics_{config_trainer_bm.test_set_method}_.yaml')    )