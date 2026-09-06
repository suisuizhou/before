# HUST DtCC TTA wrapper
import hydra, omegaconf
from omegaconf import open_dict
import main_tta_dtcc_resnet18_b as base

@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg:omegaconf.DictConfig):
    with open_dict(cfg):
        cfg.Dataset.data_name="HUST"
        cfg.Dataset.TL_list=[0,1,2,3]
    base.run.__wrapped__(cfg)

if __name__=="__main__":
    run()
