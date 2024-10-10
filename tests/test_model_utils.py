from unittest.mock import MagicMock, patch

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
    model_path = "dummy"
    num_classes = 1000
    mock_state_dict = {
        "state_dict": MagicMock(),
        "model": MagicMock(),
    }

    with patch("torch.load", return_value=mock_state_dict), patch(
        "lafo.modules.resnet_supcon.SupConResNet.load_state_dict", return_value=None
    ) as mock_load_state_dict:
        model = get_model(in_dataset, model_name, model_path, num_classes)
        assert isinstance(model, torch.nn.Module), "Model is not a torch.nn.Module instance"
        assert isinstance(
            model, res50_supcon_loader(None, num_classes).__class__
        ), "Loaded model is not a ResNet50 SupCon instance"
        mock_load_state_dict.assert_called_once()


def test_get_model_resnet18():
    in_dataset = "cifar10"
    model_name = "resnet18"
    model_path = "dummy"
    num_classes = 10

    with patch("torch.load", return_value={"model": MagicMock(), "state_dict": MagicMock()}):
        model = get_model(in_dataset, model_name, model_path, num_classes)
        assert isinstance(model, torch.nn.Module), "Model is not a torch.nn.Module instance"
        assert isinstance(model, res18_loader(None, num_classes).__class__), "Loaded model is not a ResNet18 instance"


def test_get_model_resnet18_supcon():
    mock_state_dict = {"model": MagicMock(), "state_dict": MagicMock()}
    in_dataset = "cifar10"
    model_name = "resnet18-supcon"
    model_path = "dummy"
    num_classes = 10
    with patch("torch.load", return_value=mock_state_dict):
        model = get_model(in_dataset, model_name, model_path, num_classes)
        assert isinstance(model, torch.nn.Module), "Model is not a torch.nn.Module instance"
        assert isinstance(
            model, res18_supcon_loader(None, num_classes).__class__
        ), "Loaded model is not a ResNet18 SupCon instance"
