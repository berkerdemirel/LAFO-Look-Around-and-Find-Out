import torch

from lafo.utils.model_utils import (
    get_model,
    res18_loader,
    res18_supcon_loader,
    res50_loader,
    res50_supcon_loader,
    vit_loader,
)


def test_get_model_vit():
    in_dataset = "imagenet"
    model_name = "vit"
    model_path = None
    num_classes = 1000

    model = get_model(in_dataset, model_name, model_path, num_classes)
    assert isinstance(model, torch.nn.Module), "Model is not a torch.nn.Module instance"
    assert isinstance(model, vit_loader().__class__), "Loaded model is not a ViT instance"


def test_get_model_resnet50():
    in_dataset = "imagenet"
    model_name = "resnet50"
    model_path = None
    num_classes = 1000

    model = get_model(in_dataset, model_name, model_path, num_classes)
    assert isinstance(model, torch.nn.Module), "Model is not a torch.nn.Module instance"
    assert isinstance(model, res50_loader(num_classes).__class__), "Loaded model is not a ResNet50 instance"


def test_get_model_resnet50_supcon():
    in_dataset = "imagenet"
    model_name = "resnet50-supcon"
    model_path = "./src/lafo/ckpt/ImageNet_resnet50_supcon.pth"
    num_classes = 1000

    model = get_model(in_dataset, model_name, model_path, num_classes)
    assert isinstance(model, torch.nn.Module), "Model is not a torch.nn.Module instance"
    assert isinstance(
        model, res50_supcon_loader(model_path, num_classes).__class__
    ), "Loaded model is not a ResNet50 SupCon instance"


def test_get_model_resnet18():
    in_dataset = "cifar10"
    model_name = "resnet18"
    model_path = "./src/lafo/ckpt/CIFAR10_resnet18.pth"
    num_classes = 10

    model = get_model(in_dataset, model_name, model_path, num_classes)
    assert isinstance(model, torch.nn.Module), "Model is not a torch.nn.Module instance"
    assert isinstance(model, res18_loader(model_path, num_classes).__class__), "Loaded model is not a ResNet18 instance"


def test_get_model_resnet18_supcon():
    in_dataset = "cifar10"
    model_name = "resnet18-supcon"
    model_path = "./src/lafo/ckpt/CIFAR10_resnet18_supcon.pth"
    num_classes = 10

    model = get_model(in_dataset, model_name, model_path, num_classes)
    assert isinstance(model, torch.nn.Module), "Model is not a torch.nn.Module instance"
    assert isinstance(
        model, res18_supcon_loader(model_path, num_classes).__class__
    ), "Loaded model is not a ResNet18 SupCon instance"
