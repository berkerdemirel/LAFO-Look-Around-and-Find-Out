from unittest.mock import MagicMock, patch

import torch

from lafo.utils.feature_extraction_utils import feature_extract_helper, vit_features


# Mock model for vit_features
class MockModelViT:
    def __init__(self):
        self.patch_embedding = MagicMock(return_value=torch.randn(2, 768, 14, 14))
        self.class_token = torch.randn(1, 1, 768)
        self.positional_embedding = MagicMock(return_value=torch.randn(2, 197, 768))
        self.transformer = MagicMock(return_value=torch.randn(2, 197, 768))
        self.norm = MagicMock(return_value=torch.randn(2, 768, 768))
        self.eval = MagicMock()  # Add eval method
        self.to = MagicMock()  # Add to method
        self.fc = MagicMock()  # Add fc method


def test_vit_features():
    model = MockModelViT()
    x = torch.randn(2, 3, 224, 224)
    features = vit_features(model, x)
    assert features.shape == (2, 768), "Feature extraction failed for ViT model"


# Mock dataloader for feature_extract_helper
class MockDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return torch.randn(3, 224, 224), torch.tensor(0)


def test_feature_extract_helper_vit():
    model = MockModelViT()
    dataloader = torch.utils.data.DataLoader(MockDataset(), batch_size=2)
    feat_dim = 768
    in_dataset = "imagenet"
    model_name = "vit"
    num_classes = 1000
    device = "cpu"
    ood_dataset = None

    with patch("os.makedirs"), patch("numpy.memmap") as mock_memmap:
        feature_extract_helper(
            model, dataloader, feat_dim, in_dataset, model_name, num_classes, device, ood_dataset, "train"
        )
    # Check if the model was set to eval mode
    model.eval.assert_called_once()

    # Check if the model was moved to the correct device
    model.to.assert_called_with(device)

    # Check if the memmap files were created with the correct shapes
    split = "train"
    mock_memmap.assert_any_call(
        f"./cache/{in_dataset}_{split}_{model_name}_in/feat.mmap", dtype=float, mode="w+", shape=(100, feat_dim)
    )
    mock_memmap.assert_any_call(
        f"./cache/{in_dataset}_{split}_{model_name}_in/score.mmap", dtype=float, mode="w+", shape=(100, num_classes)
    )
    mock_memmap.assert_any_call(
        f"./cache/{in_dataset}_{split}_{model_name}_in/label.mmap", dtype=float, mode="w+", shape=(100,)
    )


class MockModelResNet50:
    def __init__(self):
        self.conv1 = MagicMock(return_value=torch.randn(2, 64, 112, 112))
        self.bn1 = MagicMock(return_value=torch.randn(2, 64, 112, 112))
        self.relu = MagicMock(return_value=torch.randn(2, 64, 112, 112))
        self.layer1 = MagicMock(return_value=torch.randn(2, 256, 56, 56))
        self.layer2 = MagicMock(return_value=torch.randn(2, 512, 28, 28))
        self.layer3 = MagicMock(return_value=torch.randn(2, 1024, 14, 14))
        self.layer4 = MagicMock(return_value=torch.randn(2, 2048, 7, 7))
        self.fc = MagicMock(return_value=torch.randn(2, 10))
        self.eval = MagicMock()  # Add eval method
        self.to = MagicMock()  # Add to method
        self.features = MagicMock(return_value=torch.randn(2, 2048))


def test_feature_extract_helper_resnet50():
    model = MockModelResNet50()
    dataloader = torch.utils.data.DataLoader(MockDataset(), batch_size=2)
    feat_dim = 2048  # ResNet50 feature dimension
    in_dataset = "imagenet"
    model_name = "resnet50"
    num_classes = 1000
    device = "cpu"
    ood_dataset = None

    with patch("os.makedirs"), patch("numpy.memmap") as mock_memmap:
        feature_extract_helper(
            model, dataloader, feat_dim, in_dataset, model_name, num_classes, device, ood_dataset, "train"
        )
    # Check if the model was set to eval mode
    model.eval.assert_called_once()

    # Check if the model was moved to the correct device
    model.to.assert_called_with(device)

    # Check if the memmap files were created with the correct shapes
    split = "train"
    mock_memmap.assert_any_call(
        f"./cache/{in_dataset}_{split}_{model_name}_in/feat.mmap", dtype=float, mode="w+", shape=(100, feat_dim)
    )
    mock_memmap.assert_any_call(
        f"./cache/{in_dataset}_{split}_{model_name}_in/score.mmap", dtype=float, mode="w+", shape=(100, num_classes)
    )
    mock_memmap.assert_any_call(
        f"./cache/{in_dataset}_{split}_{model_name}_in/label.mmap", dtype=float, mode="w+", shape=(100,)
    )


# Mock dataloader for feature_extract_helper
class MockDatasetCifar(torch.utils.data.Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return torch.randn(3, 32, 32), torch.tensor(0)


class MockModelResNet18:
    def __init__(self):
        self.fc = MagicMock(return_value=torch.randn(2, 10))
        self.eval = MagicMock()  # Add eval method
        self.to = MagicMock()  # Add to method
        self.penult_feature = MagicMock(return_value=torch.randn(2, 512))


def test_feature_extract_helper_resnet18():
    model = MockModelResNet18()
    dataloader = torch.utils.data.DataLoader(MockDatasetCifar(), batch_size=2)
    feat_dim = 512  # ResNet50 feature dimension
    in_dataset = "cifar10"
    model_name = "resnet18-supcon"
    num_classes = 10
    device = "cpu"
    ood_dataset = None

    with patch("os.makedirs"), patch("numpy.memmap") as mock_memmap:
        feature_extract_helper(
            model, dataloader, feat_dim, in_dataset, model_name, num_classes, device, ood_dataset, "train"
        )
    # Check if the model was set to eval mode
    model.eval.assert_called_once()

    # Check if the model was moved to the correct device
    model.to.assert_called_with(device)

    # Check if the memmap files were created with the correct shapes
    split = "train"
    mock_memmap.assert_any_call(
        f"./cache/{in_dataset}_{split}_{model_name}_in/feat.mmap", dtype=float, mode="w+", shape=(100, feat_dim)
    )
    mock_memmap.assert_any_call(
        f"./cache/{in_dataset}_{split}_{model_name}_in/score.mmap", dtype=float, mode="w+", shape=(100, num_classes)
    )
    mock_memmap.assert_any_call(
        f"./cache/{in_dataset}_{split}_{model_name}_in/label.mmap", dtype=float, mode="w+", shape=(100,)
    )
