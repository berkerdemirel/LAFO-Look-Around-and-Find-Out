import hydra
import torch
from omegaconf import DictConfig


def res18_loader(model_path: str, num_classes: int) -> torch.nn.Module:
    from lafo.modules.resnet import resnet18_cifar

    model = resnet18_cifar(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    checkpoint = {"state_dict": {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}}
    model.load_state_dict(checkpoint["state_dict"])
    return model


def res18_supcon_loader(model_path: str, num_classes: int) -> torch.nn.Module:
    from lafo.modules.resnet_ss import resnet18_cifar

    model = resnet18_cifar(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    checkpoint = {"state_dict": {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}}
    checkpoint_linear = torch.load(
        model_path[:-4] + "_linear" + model_path[-4:], map_location="cpu", weights_only=False
    )
    checkpoint["state_dict"]["fc.weight"] = checkpoint_linear["model"]["fc.weight"]
    checkpoint["state_dict"]["fc.bias"] = checkpoint_linear["model"]["fc.bias"]
    model.load_state_dict(checkpoint["state_dict"])
    return model


def vit_loader() -> torch.nn.Module:
    from pytorch_pretrained_vit import ViT

    model = ViT("B_16_imagenet1k", pretrained=True)
    return model


def res50_loader(num_classes: int) -> torch.nn.Module:
    from lafo.modules.resnet import resnet50

    model = resnet50(num_classes=num_classes, pretrained=True)
    return model


def res50_supcon_loader(model_path: str, num_classes: int) -> torch.nn.Module:
    from lafo.modules.resnet_supcon import SupConResNet

    model = SupConResNet(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = {str.replace(k, "module.", ""): v for k, v in checkpoint["model"].items()}
    checkpoint_linear = torch.load(
        model_path[:-4] + "_linear" + model_path[-4:], map_location="cpu", weights_only=False
    )
    state_dict["fc.weight"] = checkpoint_linear["model"]["fc.weight"]
    state_dict["fc.bias"] = checkpoint_linear["model"]["fc.bias"]
    model.load_state_dict(state_dict)
    return model


def get_model(in_dataset: str, model_name: str, model_path: str, num_classes: int) -> torch.nn.Module:
    if in_dataset == "imagenet":
        if model_name == "resnet50":
            model = res50_loader(num_classes)
        elif model_name == "resnet50-supcon":
            model = res50_supcon_loader(model_path, num_classes)
        elif model_name == "vit":
            model = vit_loader()
        else:
            assert False, "Not supported model arch: {}".format(model_name)
    else:
        if model_name == "resnet18":
            model = res18_loader(model_path, num_classes)
        elif model_name == "resnet18-supcon":
            model = res18_supcon_loader(model_path, num_classes)
        else:
            assert False, "Not supported model arch: {}".format(model_name)

    print("Number of model parameters: {}".format(sum([p.data.nelement() for p in model.parameters()])))
    return model


@hydra.main(config_path="../cfgs", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig):
    print(cfg)
    model_dict = {"imagenet": ["resnet50", "resnet50-supcon", "vit"], "cifar10": ["resnet18", "resnet18-supcon"]}

    for b in model_dict.keys():
        for model_name in model_dict[b]:
            cfg_temp = cfg[b + "_cfgs"]
            print(b, model_name, "." + cfg_temp.model_path, cfg_temp.num_classes)
            model = get_model(b, model_name, "." + cfg_temp.model_path, cfg_temp.num_classes)
            print(model)


if __name__ == "__main__":
    main()
