# main_src_dtcc_resnet18_hust.py

import hydra
import omegaconf

from omegaconf import open_dict

import Dataset
from Dataset.HUST_Bearing import HUST

Dataset.HUST = HUST


import main_src_dtcc_resnet18_b as base



def force_hust(cfg):

    with open_dict(cfg):

        # Dataset
        cfg.Dataset.data_name="HUST"
        cfg.Dataset.TL_list=[0,1,2,3]


        # Model
        cfg.Model.model_name="ResNet18"
        cfg.Model.bottleneck=True
        cfg.Model.bottleneck_num=128
        cfg.Model.temp=1
        cfg.Model.model_type="linear"


        # training
        cfg.src_epoch=50
        cfg.batch_size=64
        cfg.num_workers=0

        cfg.process_wandb=False



@hydra.main(
    version_base=None,
    config_path="./Configs",
    config_name="defaults"
)
def run(cfg:omegaconf.DictConfig):

    force_hust(cfg)

    base.run.__wrapped__(cfg)



if __name__=="__main__":
    run()