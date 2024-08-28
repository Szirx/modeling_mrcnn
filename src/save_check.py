from lightning_module import MaskRCNNLightning
from train import arg_parse
import torch
from config import Config

if __name__ == '__main__':
    args = arg_parse()
    config = Config.from_yaml(args.config_file)
    
    trained_model = MaskRCNNLightning.load_from_checkpoint(config.load_from_ckpt, config=config)
    model_weights = trained_model.model.state_dict()

    torch.save(model_weights, "weights/mrcnn-weights.pt")