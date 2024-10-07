import os
import torch
import torchvision
from torchvision import transforms
from easydict import EasyDict
from omegaconf import DictConfig
import hydra
from lafo.utils.data_utils import get_loader_in, get_loader_out


def res18_loader(cfg: DictConfig) -> torch.nn.Module:
    from lafo.modules.resnet import resnet18_cifar

    model = resnet18_cifar(num_classes=cfg.num_classes)
    checkpoint = torch.load(cfg.model_path)
    checkpoint = {"state_dict": {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}}
    model.load_state_dict(checkpoint["state_dict"])
    return model


def res18_supcon_loader(cfg: DictConfig) -> torch.nn.Module:
    from lafo.modules.resnet_ss import resnet18_cifar

    model = resnet18_cifar(num_classes=cfg.num_classes)
    checkpoint = torch.load(cfg.model_path)
    checkpoint = {"state_dict": {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}}
    checkpoint_linear = torch.load(cfg.model_path[:-4] + "_linear" + cfg.model_path[-4:])
    checkpoint["state_dict"]["fc.weight"] = checkpoint_linear["model"]["fc.weight"]
    checkpoint["state_dict"]["fc.bias"] = checkpoint_linear["model"]["fc.bias"]
    model.load_state_dict(checkpoint["state_dict"])
    return model


def vit_loader(cfg: DictConfig) -> torch.nn.Module:
    from pytorch_pretrained_vit import ViT

    model = ViT("B_16_imagenet1k", pretrained=True)
    return model


def res50_loader(cfg: DictConfig) -> torch.nn.Module:
    from lafo.modules.resnet import resnet50

    model = resnet50(num_classes=cfg.num_classes, pretrained=True)
    return model


def res50_supcon_loader(cfg: DictConfig) -> torch.nn.Module:
    from lafo.modules.resnet_supcon import SupConResNet

    model = SupConResNet(num_classes=cfg.num_classes)
    checkpoint = torch.load(cfg.model_path)
    state_dict = {str.replace(k, "module.", ""): v for k, v in checkpoint["model"].items()}
    checkpoint_linear = torch.load(cfg.model_path[:-4] + "_linear" + cfg.model_path[-4:])
    state_dict["fc.weight"] = checkpoint_linear["model"]["fc.weight"]
    state_dict["fc.bias"] = checkpoint_linear["model"]["fc.bias"]
    model.load_state_dict(state_dict)
    return model


def get_model(cfg: DictConfig) -> torch.nn.Module:
    if cfg.in_dataset == "imagenet":
        if cfg.model_name == "resnet50":
            model = res50_loader(cfg)
        elif cfg.model_name == "resnet50-supcon":
            model = res50_supcon_loader(cfg)
        elif cfg.model_name == "vit":
            model = vit_loader(cfg)
        else:
            assert False, "Not supported model arch: {}".format(cfg.model_name)
    else:
        if cfg.model_name == "resnet18":
            model = res18_loader(cfg)
        elif cfg.model_name == "resnet18-supcon":
            model = res18_supcon_loader(cfg)
        else:
            assert False, "Not supported model arch: {}".format(cfg.model_name)

    model = model.cuda()
    model.eval()
    print("Number of model parameters: {}".format(sum([p.data.nelement() for p in model.parameters()])))
    return model


@hydra.main(config_path="../cfgs", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig):
    print(cfg)
    cfg.benchmark = "imagenet"

    model_dict = {"imagenet": ["resnet50", "resnet50-supcon", "vit"], "cifar10": ["resnet18", "resnet18-supcon"]}
    
    for b in model_dict.keys():
        benchmark = b + "_cfgs"
        for model_name in model_dict[b]:
            cfg[benchmark].model_name = model_name
            model = get_model(cfg=cfg[benchmark])
            print(benchmark, model_name)
            print(model)


if __name__ == "__main__":
    main()
