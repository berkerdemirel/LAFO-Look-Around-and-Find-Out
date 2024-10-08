from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def LAFO(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    use_cache: bool = False,
    device: Optional[str] = None,
) -> np.ndarray:
    device = device if device else "cpu"
    if use_cache:
        return LAFO_cache(model, test_loader, num_classes, class_means, device)
    else:
        return LAFO_no_cache(model, test_loader, num_classes, class_means, device)


def LAFO_cache(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
) -> np.ndarray:
    model = model.to(device)
    model.eval()

    all_scores = []
    total_size = 0

    class_idx = np.arange(num_classes)

    with torch.inference_mode():
        for feats_batch_initial, logits_batch_initial in test_loader:
            feats_batch_initial = feats_batch_initial.to(device)
            logits_batch_initial = logits_batch_initial.to(device)
            preds_initial = logits_batch_initial.argmax(1)
            max_logits = logits_batch_initial.max(dim=1).values
            total_size += feats_batch_initial.size(0)
            trajectory_list = torch.zeros(feats_batch_initial.size(0), num_classes, device=device)
            for class_id in class_idx:
                logit_diff = max_logits - logits_batch_initial[:, class_id]
                weight_diff = model.fc.weight[preds_initial] - model.fc.weight[class_id]
                weight_diff_norm = torch.linalg.norm(weight_diff, dim=1)

                feats_batch_db = (
                    feats_batch_initial - torch.divide(logit_diff, weight_diff_norm**2).view(-1, 1) * weight_diff
                )

                centered_feats = feats_batch_initial - torch.mean(class_means, dim=0)
                centered_feats_db = feats_batch_db - torch.mean(class_means, dim=0)

                norm_centered_feats = F.normalize(centered_feats, p=2, dim=1)
                norm_centered_feats_db = F.normalize(centered_feats_db, p=2, dim=1)

                cos_sim_origin_perspective = torch.sum(norm_centered_feats * norm_centered_feats_db, dim=1)
                angles_origin = torch.arccos(cos_sim_origin_perspective) / torch.pi

                # # fdbd original
                # distance_to_db = torch.linalg.norm(feats_batch_initial - feats_batch_db, dim=1)
                # fdbd_score = distance_to_db / torch.linalg.norm(feats_batch_initial - torch.mean(class_means, dim=0), dim=1)

                # # fdbd our derivation
                # feats_centered_db = feats_batch_initial - feats_batch_db
                # mean_centered_db = torch.mean(class_means, dim=0) - feats_batch_db
                # cos_sim = F.cosine_similarity(feats_centered_db, mean_centered_db, dim=1)
                # angles_db = torch.arccos(cos_sim) / torch.pi
                # our_derivation = torch.sin(angles_origin * torch.pi) / torch.sin(angles_db * torch.pi)

                # check our derivation is same as fdbd
                # fdbd_score[torch.isnan(fdbd_score)] = 0
                # our_derivation[torch.isnan(our_derivation)] = 0
                # # print(torch.allclose(fdbd_score, our_derivation))

                trajectory_list[:, class_id] = angles_origin

            trajectory_list[torch.isnan(trajectory_list)] = 0
            ood_score = torch.max(trajectory_list, dim=1).values
            # ood_score = torch.topk(trajectory_list, 2, largest=False, dim=1).values[:, 1]
            # ood_score = torch.mean(trajectory_list, dim=1)
            all_scores.append(ood_score)
    scores = np.asarray(torch.cat(all_scores).detach().cpu().numpy(), dtype=np.float32)
    return scores


def LAFO_no_cache(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
) -> np.ndarray:
    model = model.to(device)
    model.eval()

    all_scores = []
    total_size = 0

    class_idx = np.arange(num_classes)

    with torch.inference_mode():
        for data, _ in test_loader:
            data = data.to(device)
            feats_batch_initial = model.penult_feature(data)
            logits_batch_initial = model.fc(feats_batch_initial)
            preds_initial = logits_batch_initial.argmax(1)
            max_logits = logits_batch_initial.max(dim=1).values
            total_size += feats_batch_initial.size(0)
            trajectory_list = torch.zeros(feats_batch_initial.size(0), num_classes, device=device)
            for class_id in class_idx:
                logit_diff = max_logits - logits_batch_initial[:, class_id]
                weight_diff = model.fc.weight[preds_initial] - model.fc.weight[class_id]
                weight_diff_norm = torch.linalg.norm(weight_diff, dim=1)

                feats_batch_db = (
                    feats_batch_initial - torch.divide(logit_diff, weight_diff_norm**2).view(-1, 1) * weight_diff
                )

                centered_feats = feats_batch_initial - torch.mean(class_means, dim=0)
                centered_feats_db = feats_batch_db - torch.mean(class_means, dim=0)

                norm_centered_feats = F.normalize(centered_feats, p=2, dim=1)
                norm_centered_feats_db = F.normalize(centered_feats_db, p=2, dim=1)

                cos_sim_origin_perspective = torch.sum(norm_centered_feats * norm_centered_feats_db, dim=1)
                angles_origin = torch.arccos(cos_sim_origin_perspective) / torch.pi

                # # fdbd original
                # distance_to_db = torch.linalg.norm(feats_batch_initial - feats_batch_db, dim=1)
                # fdbd_score = distance_to_db / torch.linalg.norm(feats_batch_initial - torch.mean(class_means, dim=0), dim=1)

                # # fdbd our derivation
                # feats_centered_db = feats_batch_initial - feats_batch_db
                # mean_centered_db = torch.mean(class_means, dim=0) - feats_batch_db
                # cos_sim = F.cosine_similarity(feats_centered_db, mean_centered_db, dim=1)
                # angles_db = torch.arccos(cos_sim) / torch.pi
                # our_derivation = torch.sin(angles_origin * torch.pi) / torch.sin(angles_db * torch.pi)

                # check our derivation is same as fdbd
                # fdbd_score[torch.isnan(fdbd_score)] = 0
                # our_derivation[torch.isnan(our_derivation)] = 0
                # # print(torch.allclose(fdbd_score, our_derivation))

                trajectory_list[:, class_id] = angles_origin

            trajectory_list[torch.isnan(trajectory_list)] = 0
            ood_score = torch.max(trajectory_list, dim=1).values
            # ood_score = torch.topk(trajectory_list, 2, largest=False, dim=1).values[:, 1]
            # ood_score = torch.mean(trajectory_list, dim=1)
            all_scores.append(ood_score)
    scores = np.asarray(torch.cat(all_scores).detach().cpu().numpy(), dtype=np.float32)
    return scores


def fDBD(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
) -> np.ndarray:
    model = model.to(device)
    model.eval()

    all_scores = []
    total_size = 0

    class_idx = np.arange(num_classes)

    with torch.inference_mode():
        for feats_batch_initial, logits_batch_initial in test_loader:
            feats_batch_initial = feats_batch_initial.to(device)
            logits_batch_initial = logits_batch_initial.to(device)
            preds_initial = logits_batch_initial.argmax(1)
            max_logits = logits_batch_initial.max(dim=1).values
            total_size += feats_batch_initial.size(0)
            trajectory_list = torch.zeros(feats_batch_initial.size(0), num_classes, device=device)
            for class_id in class_idx:
                logit_diff = max_logits - logits_batch_initial[:, class_id]
                weight_diff = model.fc.weight[preds_initial] - model.fc.weight[class_id]
                weight_diff_norm = torch.linalg.norm(weight_diff, dim=1)

                feats_batch_db = (
                    feats_batch_initial - torch.divide(logit_diff, weight_diff_norm**2).view(-1, 1) * weight_diff
                )
                distance_to_db = torch.linalg.norm(feats_batch_initial - feats_batch_db, dim=1)
                # fdbd
                trajectory_list[:, class_id] = distance_to_db / torch.linalg.norm(
                    feats_batch_initial - torch.mean(class_means, dim=0), dim=1
                )

            trajectory_list[torch.isnan(trajectory_list)] = 0
            ood_score = torch.mean(trajectory_list, dim=1)
            all_scores.append(ood_score)
    scores = np.asarray(torch.cat(all_scores).detach().cpu().numpy(), dtype=np.float32)
    return scores


def knn_score(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    train_features: torch.Tensor,
    device: str,
    k: int = 50,
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    all_scores = []
    norm_train_feats = F.normalize(train_features, p=2, dim=1)

    with torch.inference_mode():
        for feats_batch_initial, logits_batch_initial in test_loader:
            feats_batch_initial = feats_batch_initial.to(device)
            logits_batch_initial = logits_batch_initial.to(device)
            # normalize
            norm_feats_batch_initial = F.normalize(feats_batch_initial, p=2, dim=1)
            # calculate the distance
            # get the kth nearest neighbor distance
            distances = torch.cdist(
                norm_feats_batch_initial, norm_train_feats, p=2, compute_mode="donot_use_mm_for_euclid_dist"
            )
            # get the kth nearest neighbor distance
            kth_distances = torch.topk(distances, k, largest=False)[0][:, -1]
            all_scores.append(-kth_distances)
    scores = np.asarray(torch.cat(all_scores).cpu().numpy(), dtype=np.float32)
    return scores


def get_class_means(train_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: str) -> torch.Tensor:
    model = model.to(device)
    model.eval()
    all_features = []
    all_labels = []
    with torch.inference_mode():
        for data, labels in train_loader:
            data = data.to(device)
            features = model.penult_feature(data)
            all_features.append(features)
            all_labels.append(labels)
    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    class_means = torch.zeros(all_labels.max() + 1, all_features.size(1)).to(all_features.device)
    for i in range(all_labels.max() + 1):
        class_means[i] = torch.mean(all_features[all_labels == i], dim=0)
    return class_means
