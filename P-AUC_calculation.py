import os
import sys
import gc
import subprocess
from glob import glob
from collections import deque
from typing import Tuple, List, Dict

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import beta
from skimage import io
from skimage.util import img_as_float
from skimage.filters import sobel
from skimage.color import rgb2gray, rgb2lab
from skimage.segmentation import slic, find_boundaries
from skimage.metrics import structural_similarity as ssim
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

# =============================
# Config
# =============================
mean_PAUC = 0.0
for sample_test in range(10):
    sample_name = "samples_00{sample_test}".format(sample_test=sample_test)
    sample = sample_name.split("_")[1:]
    sample = "_".join(sample)
    test_origin = "Combined_half_sigma_batched"
    patches = 24

    # =============================
    # Step 0: Run super_pixel_generation.py
    # =============================
    subprocess.run([
        "python3", "super_pixel_generation.py",
        f"--input_image=/data/akheirandish3/mvtec_ad/cable/test/combined/{sample}.png",
        "--output_dir=/home/akheirandish3/diffusion-posterior-sampling/figures"
    ], check=True)

    # =============================
    # Step 1: Load reconstruction images
    # =============================
    images_all = []
    images_recon_dict = {}

    for item in range(patches):
        directory = (
            f"/home/akheirandish3/diffusion-posterior-sampling/results_patches/"
            f"{sample_name}/{test_origin}_{item}_4/inpainting/recon"
        )
        image_paths = sorted(glob(os.path.join(directory, "*.png")))
        images_list = []
        for path in image_paths:
            img = io.imread(path)
            if img.ndim == 3 and img.shape[-1] == 4:
                img = img[..., :3]
            images_list.append(img)

        images_all.extend(images_list)
        if not images_list:
            # print(f"No images found in {directory}, skipping...")
            continue

        images = np.array(images_list)
        if item not in images_recon_dict:
            images_recon_dict[item] = images
        else:
            images_recon_dict[item] = np.concatenate(
                [images_recon_dict[item], images], axis=0
            )

    # for k, v in images_recon_dict.items():
    #     print(f"images_recon_dict[{k}]: shape = {v.shape}")

    arr_list = [images_recon_dict[k] for k in sorted(images_recon_dict.keys()) if images_recon_dict[k].size > 0]
    shapes = [a.shape[1:] for a in arr_list]
    if len(set(shapes)) != 1:
        raise ValueError(f"Inconsistent shapes after batch dim: {set(shapes)}")

    images_recon_all = np.concatenate(arr_list, axis=0)
    # print("images_recon_all shape:", images_recon_all.shape)

    mask_directory = "/home/akheirandish3/diffusion-posterior-sampling/figures/mask.png"
    mask = io.imread(mask_directory)


    # =============================
    # Superpixel helpers
    # =============================
    def rgb_variance(img: np.ndarray, mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return 0.0
        pix = img[mask]
        if pix.ndim == 1:
            return float(np.var(pix))
        return float(np.mean([np.var(pix[:, c]) for c in range(pix.shape[1])]))


    def slic_features_lab(img: np.ndarray, alpha_grad: float = 10.0) -> np.ndarray:
        img_f = img_as_float(img).astype(np.float32)
        if img_f.ndim == 2:
            g = sobel(img_f)
            return np.dstack([img_f, alpha_grad * g])
        lab = rgb2lab(img_f).astype(np.float32)
        gray = rgb2gray(img_f)
        g = sobel(gray).astype(np.float32)
        return np.dstack([lab, alpha_grad * g[..., None]])


    def split_region(img, full_mask, region_id, n_sub=4, compactness=8.0, alpha_grad=10.0):
        m = (full_mask == region_id)
        if m.sum() < n_sub:
            return full_mask
        rows, cols = np.where(m)
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1
        img_crop = img[r0:r1, c0:c1]
        m_crop = m[r0:r1, c0:c1]
        feats = slic_features_lab(img_crop, alpha_grad=alpha_grad)
        sub = slic(feats, n_segments=n_sub, compactness=compactness, sigma=0.5,
                start_label=0, mask=m_crop, channel_axis=-1)
        new_labels = full_mask.copy()
        base = int(full_mask.max()) + 1
        rr, cc = np.where(m_crop)
        used = np.unique(sub[rr, cc])
        used = used[used >= 0]
        for i, u in enumerate(used):
            sel = (sub == u) & m_crop
            r_sel, c_sel = np.where(sel)
            new_labels[r0 + r_sel, c0 + c_sel] = base + i
        return new_labels


    def recursive_subdivide(img, labels, var_threshold=150.0, min_pixels=16,
                            max_sub=6, max_depth=4, compactness=12.0,
                            alpha_grad=10.0, target_size=200):
        labels_out = labels.copy().astype(np.int32)
        orig_ids = list(np.unique(labels_out))
        parent_map = {int(oid): [int(oid)] for oid in orig_ids}

        def _update_parent(orig_parent, old_child, new_children):
            children = parent_map[int(orig_parent)]
            children = [c for c in children if c != old_child]
            children.extend(new_children)
            parent_map[int(orig_parent)] = children

        queue = deque()
        for oid in orig_ids:
            queue.append((int(oid), 0, int(oid)))

        while queue:
            rid, depth, orig_parent = queue.popleft()
            m = (labels_out == rid)
            area = int(m.sum())
            if area < min_pixels or depth >= max_depth:
                continue
            v = rgb_variance(img, m)
            if v < var_threshold:
                continue
            n_sub = max(2, min(max_sub, area // target_size))
            if n_sub < 2:
                continue
            old_max = int(labels_out.max())
            labels_out = split_region(img, labels_out, rid, n_sub=n_sub,
                                    compactness=compactness, alpha_grad=alpha_grad)
            new_max = int(labels_out.max())
            if new_max == old_max:
                continue
            new_children = list(range(old_max + 1, new_max + 1))
            if (labels_out == rid).sum() > 0:
                new_children.append(rid)
            _update_parent(orig_parent, rid, new_children)
            for child in new_children:
                if child != rid:
                    queue.append((child, depth + 1, orig_parent))

        final_ids = sorted(list(np.unique(labels_out).astype(int)))
        return labels_out, final_ids, parent_map


    # =============================
    # Load label image & subdivide
    # =============================
    label_image = io.imread(
        f"/home/akheirandish3/diffusion-posterior-sampling/results_patches/"
        f"{sample_name}/{test_origin}_8_4/inpainting/label/0_00000.png"
    )
    label_image = label_image[:, :, :3]

    labels0 = mask

    labels_fine, final_ids, parent_map = recursive_subdivide(
        img=label_image, labels=labels0,
        var_threshold=0.0, min_pixels=20, max_sub=6, max_depth=4,
        compactness=12.0, alpha_grad=10.0, target_size=10,
    )
    # print(f"Initial superpixels: {len(np.unique(labels0))}")
    # print(f"Final superpixels:   {len(final_ids)}")


    # =============================
    # ResNet Pixel Embedder
    # =============================
    def patchify_context(features, patchsize=3, stride=1):
        padding = (patchsize - 1) // 2
        unfolder = torch.nn.Unfold(kernel_size=patchsize, stride=stride, padding=padding)
        B, C, H, W = features.shape
        unfolded = unfolder(features)
        unfolded = unfolded.view(B, C, patchsize * patchsize, H * W)
        pooled = unfolded.mean(dim=2)
        return pooled.view(B, C, H, W)


    class ResNetPixelEmbedder(nn.Module):
        def __init__(self, resnet_name="resnet18", layers=("layer1", "layer2", "layer3"),
                    out_size=None, use_imagenet_norm=True, use_patch_context=True,
                    proj_dim_per_layer=None):
            super().__init__()
            if resnet_name == "resnet18":
                try:
                    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                except AttributeError:
                    net = models.resnet18(pretrained=True)
            elif resnet_name == "resnet50":
                try:
                    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                except AttributeError:
                    net = models.resnet50(pretrained=True)
            else:
                raise ValueError("resnet_name must be resnet18 or resnet50")
            net.eval()
            return_nodes = {ln: ln for ln in layers}
            self.extractor = create_feature_extractor(net, return_nodes=return_nodes)
            self.layers = layers
            self.out_size = out_size
            self.use_patch_context = use_patch_context
            self.proj = nn.ModuleDict()
            self.proj_dim_per_layer = proj_dim_per_layer
            if proj_dim_per_layer is not None:
                ch_map = {}
                if resnet_name == "resnet18":
                    ch_map = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}
                else:
                    ch_map = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
                for ln in layers:
                    self.proj[ln] = nn.Conv2d(ch_map[ln], proj_dim_per_layer, kernel_size=1, bias=False)
            self.use_imagenet_norm = use_imagenet_norm
            if use_imagenet_norm:
                self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
                self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

        @torch.no_grad()
        def forward(self, x):
            assert x.ndim == 4 and x.shape[1] == 3
            B, _, H, W = x.shape
            if x.min() < 0:
                x = (x + 1) / 2.0
            if self.use_imagenet_norm:
                x = (x - self.mean) / self.std
            feats = self.extractor(x)
            outH, outW = (self.out_size, self.out_size) if self.out_size is not None else (H, W)
            ups = []
            for ln in self.layers:
                f = feats[ln]
                if self.use_patch_context:
                    f = patchify_context(f, patchsize=3, stride=1)
                if self.proj_dim_per_layer is not None:
                    f = self.proj[ln](f)
                f = F.interpolate(f, size=(outH, outW), mode="bilinear", align_corners=False)
                f = F.normalize(f, dim=1)
                ups.append(f)
            return torch.cat(ups, dim=1)


    # =============================
    # Extract features
    # =============================
    embedder = ResNetPixelEmbedder(
        resnet_name="resnet18",
        layers=("layer1", "layer2", "layer3"),
        out_size=None,
        use_patch_context=True,
        proj_dim_per_layer=None,
    ).to("cuda").eval()

    # print("images_recon_all shape:", images_recon_all.shape)

    x_all = torch.from_numpy(images_recon_all).float().permute(0, 3, 1, 2)
    if x_all.max() > 1.0:
        x_all = x_all / 255.0

    feat_list = []
    with torch.no_grad():
        for i in range(x_all.shape[0]):
            xi = x_all[i:i + 1].to("cuda")
            fi = embedder(xi)
            feat_list.append(fi.cpu())
            del xi, fi
            if i % 10 == 0:
                torch.cuda.empty_cache()

    feat_map_10 = torch.cat(feat_list, dim=0)
    # print("feat_map_10:", feat_map_10.shape)

    del feat_list
    gc.collect()
    torch.cuda.empty_cache()


    # =============================
    # Helper: numpy image to tensor
    # =============================
    def to_tensor01(img_np, device="cuda"):
        t = torch.from_numpy(img_np).float()
        if t.ndim == 2:
            t = t[..., None].repeat(1, 1, 3)
        if t.max() > 1.0:
            t = t / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
        return t


    # =============================
    # PCA on label image features
    # =============================
    model_device = next(embedder.parameters()).device
    n_pca = 3

    with torch.no_grad():
        label_x = to_tensor01(label_image, device=model_device)
        label_feat = embedder(label_x).squeeze(0).cpu()  # (C, H, W)

    C_feat, H_feat, W_feat = label_feat.shape
    X_label = label_feat.permute(1, 2, 0).reshape(-1, C_feat).numpy()  # (H*W, C)

    # PCA basis
    mu_pca = X_label.mean(axis=0, keepdims=True)
    X_centered = X_label - mu_pca
    _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components_pca = Vt[:n_pca]
    explained = (S[:n_pca] ** 2) / (S ** 2).sum()
    # print(f"PCA explained variance ({n_pca} components): {explained.sum() * 100:.1f}%")


    def project_to_pca(feat_hwc, mu, components):
        proj = (feat_hwc - mu) @ components.T
        return np.clip(proj, -1.0, 1.0)


    label_pca_map = project_to_pca(X_label, mu_pca, components_pca).reshape(H_feat, W_feat, n_pca)
    # print("label_pca_map range:", label_pca_map.min(), label_pca_map.max())

    # Project all recon features
    B = feat_map_10.shape[0]
    feat_np = feat_map_10.permute(0, 2, 3, 1).numpy()
    feat_flat = feat_np.reshape(B, -1, C_feat)
    feat_centered = feat_flat - mu_pca[None]
    proj_flat = feat_centered @ components_pca.T
    pca_feats_recon = np.clip(proj_flat, -1.0, 1.0).reshape(B, H_feat, W_feat, n_pca)
    # print("pca_feats_recon range:", pca_feats_recon.min(), pca_feats_recon.max())


    # =============================
    # Typical set helpers
    # =============================
    def quantize_u8_to_bins(x_u8, bins):
        x = x_u8.astype(np.uint16)
        return ((x * bins) // 256).astype(np.uint8)


    def quantize_pca_to_bins(x_pca, bins):
        x = np.clip(x_pca, -1.0, 1.0)
        x_shifted = (x + 1.0) / 2.0
        return np.clip((x_shifted * bins).astype(np.int32), 0, bins - 1).astype(np.uint8)


    def joint_6d_pmf(rgb_q, pca_q, bins_rgb, bins_pca, smooth_sigma=1.0, eps=1e-12):
        r, g, b = rgb_q[:, 0].astype(np.int64), rgb_q[:, 1].astype(np.int64), rgb_q[:, 2].astype(np.int64)
        idx_rgb = np.ravel_multi_index((r, g, b), dims=(bins_rgb,) * 3)
        hist_rgb = np.bincount(idx_rgb, minlength=bins_rgb ** 3).astype(np.float64).reshape((bins_rgb,) * 3)
        if smooth_sigma > 0:
            hist_rgb = gaussian_filter(hist_rgb, sigma=smooth_sigma, mode="nearest")
        hist_rgb += eps
        pmf_rgb = hist_rgb / hist_rgb.sum()

        p1, p2, p3 = pca_q[:, 0].astype(np.int64), pca_q[:, 1].astype(np.int64), pca_q[:, 2].astype(np.int64)
        idx_pca = np.ravel_multi_index((p1, p2, p3), dims=(bins_pca,) * 3)
        hist_pca = np.bincount(idx_pca, minlength=bins_pca ** 3).astype(np.float64).reshape((bins_pca,) * 3)
        if smooth_sigma > 0:
            hist_pca = gaussian_filter(hist_pca, sigma=smooth_sigma, mode="nearest")
        hist_pca += eps
        pmf_pca = hist_pca / hist_pca.sum()

        return pmf_rgb, pmf_pca


    def entropy_bits_combined(pmf_rgb, pmf_pca, eps=1e-12):
        def H(pmf):
            p = pmf.ravel()
            return float(-np.sum(p * np.log2(p + eps)))
        return H(pmf_rgb) + H(pmf_pca)


    def avg_neg_logp_bits_combined(rgb_q, pca_q, pmf_rgb, pmf_pca, eps=1e-12):
        r, g, b = rgb_q[:, 0].astype(np.int64), rgb_q[:, 1].astype(np.int64), rgb_q[:, 2].astype(np.int64)
        p_rgb = pmf_rgb[r, g, b]
        nll_rgb = float(-np.mean(np.log2(p_rgb + eps)))

        p1, p2, p3 = pca_q[:, 0].astype(np.int64), pca_q[:, 1].astype(np.int64), pca_q[:, 2].astype(np.int64)
        p_pca = pmf_pca[p1, p2, p3]
        nll_pca = float(-np.mean(np.log2(p_pca + eps)))

        return nll_rgb + nll_pca


    # =============================
    # Compute typical set with PCA
    # =============================
    def compute_typical_set_with_pca(
        labels_fine, parent_map, images_recon_dict, pca_feats_recon,
        images_patch_dict, pca_feats_patch,
        bins_rgb=16, bins_pca=16, smooth_sigma=1.0, min_pixels=2, eps=1e-12,
        use_label_as_target=True, label_image=None, label_pca_map=None,
    ):
        H_img, W_img = labels_fine.shape
        delta_map = np.full((H_img, W_img), np.nan, dtype=np.float32)
        info = {}
        labels_used = []

        child_to_parent = {}
        for orig_id, children in parent_map.items():
            for child_id in children:
                child_to_parent[child_id] = orig_id

        all_refined_ids = sorted(np.unique(labels_fine).astype(int).tolist())

        for refined_id in all_refined_ids:
            orig_id = child_to_parent.get(refined_id, None)
            sp_mask = (labels_fine == refined_id)
            n_pix = int(sp_mask.sum())

            if n_pix < min_pixels or orig_id is None:
                continue

            # Build PMFs from recon
            recon_rgb = images_recon_dict[:, sp_mask, :3].reshape(-1, 3).astype(np.uint8)
            if recon_rgb.shape[0] == 0:
                continue
            recon_rgb_q = quantize_u8_to_bins(recon_rgb, bins=bins_rgb)

            N = images_recon_dict.shape[0]
            recon_pca = pca_feats_recon[:N, sp_mask, :]
            recon_pca_flat = recon_pca.reshape(-1, recon_pca.shape[-1])
            recon_pca_q = quantize_pca_to_bins(recon_pca_flat, bins=bins_pca)

            pmf_rgb, pmf_pca = joint_6d_pmf(
                recon_rgb_q, recon_pca_q, bins_rgb=bins_rgb, bins_pca=bins_pca,
                smooth_sigma=smooth_sigma, eps=eps,
            )
            H_bits = entropy_bits_combined(pmf_rgb, pmf_pca, eps=eps)

            # Get target
            if use_label_as_target and label_image is not None and label_pca_map is not None:
                target_rgb = label_image[sp_mask, :3].astype(np.uint8)
                target_pca = label_pca_map[sp_mask]
            else:
                if orig_id not in images_patch_dict:
                    continue
                patch_samples = images_patch_dict[orig_id]
                if patch_samples.ndim != 4 or patch_samples.shape[-1] < 3:
                    continue
                target_rgb = np.mean(patch_samples[:, sp_mask, :3], axis=0).astype(np.uint8)
                Np = patch_samples.shape[0]
                target_pca = pca_feats_patch[:Np, sp_mask, :].mean(axis=0)

            if target_rgb.shape[0] == 0:
                continue

            target_rgb_q = quantize_u8_to_bins(target_rgb, bins=bins_rgb)
            target_pca_q = quantize_pca_to_bins(target_pca, bins=bins_pca)

            avg_nlogp = avg_neg_logp_bits_combined(
                target_rgb_q, target_pca_q, pmf_rgb, pmf_pca, eps=eps
            )
            delta_sp = float(np.abs(avg_nlogp - H_bits))
            delta_map[sp_mask] = delta_sp

            info[int(refined_id)] = {
                "orig_parent": int(orig_id),
                "num_pixels": n_pix,
                "H_bits": float(H_bits),
                "avg_neg_logp_bits": float(avg_nlogp),
                "delta_sp": float(delta_sp),
            }
            labels_used.append(int(refined_id))

        return delta_map, info, sorted(labels_used)


    # =============================
    # Run delta computation
    # =============================
    delta_map_pca, info_pca, labels_used_pca = compute_typical_set_with_pca(
        labels_fine=labels_fine,
        parent_map=parent_map,
        images_recon_dict=images_recon_all,
        pca_feats_recon=pca_feats_recon,
        images_patch_dict=images_recon_all,
        pca_feats_patch=pca_feats_recon,
        bins_rgb=64,
        bins_pca=16,
        smooth_sigma=0.01,
        min_pixels=2,
        use_label_as_target=True,
        label_image=label_image,
        label_pca_map=label_pca_map,
    )

    # print(f"Superpixels processed: {len(labels_used_pca)}")
    # print(f"Delta max: {np.nanmax(delta_map_pca):.4f}, mean: {np.nanmean(delta_map_pca):.4f}")


    # =============================
    # Load ground truth mask
    # =============================
    # Patch numpy._core for torch.load compatibility
    if not hasattr(np, '_core'):
        np._core = np.core
        for submod in ['multiarray', 'numeric', 'umath', 'fromnumeric', '_methods']:
            full_old = f'numpy.core.{submod}'
            full_new = f'numpy._core.{submod}'
            if full_old in sys.modules and full_new not in sys.modules:
                sys.modules[full_new] = sys.modules[full_old]
            elif hasattr(np.core, submod):
                sys.modules[full_new] = getattr(np.core, submod)

    gt_mask_path = f"/data/akheirandish3/mvtec_ad/cable/ground_truth/combined/{sample}_mask.png"
    gt_mask = np.array(Image.open(gt_mask_path).convert("L"))
    gt_mask = gt_mask[::4, ::4]
    gt_mask_binary = (gt_mask > 0).astype(np.uint8)

    # print(f"GT mask shape: {gt_mask_binary.shape}")
    # print(f"Anomalous pixels: {gt_mask_binary.sum()} / {gt_mask_binary.size} "
    #       f"({100 * gt_mask_binary.mean():.2f}%)")


    # =============================
    # Manual ROC / AUC
    # =============================
    def manual_roc_curve(y_true, y_score):
        desc_idx = np.argsort(-y_score)
        y_score = y_score[desc_idx]
        y_true = y_true[desc_idx]
        distinct_idx = np.where(np.diff(y_score))[0]
        threshold_idx = np.concatenate([distinct_idx, [len(y_true) - 1]])
        tps = np.cumsum(y_true)[threshold_idx]
        fps = (threshold_idx + 1) - tps
        tps = np.concatenate([[0], tps])
        fps = np.concatenate([[0], fps])
        fpr = fps / fps[-1] if fps[-1] > 0 else fps
        tpr = tps / tps[-1] if tps[-1] > 0 else tps
        thresholds = y_score[threshold_idx]
        return fpr, tpr, thresholds


    def manual_auc(x, y):
        order = np.argsort(x)
        x, y = x[order], y[order]
        return float(np.trapz(y, x))


    def manual_precision_recall_curve(y_true, y_score):
        desc_idx = np.argsort(-y_score)
        y_score = y_score[desc_idx]
        y_true = y_true[desc_idx]
        tps = np.cumsum(y_true)
        fps = np.cumsum(1 - y_true)
        total_pos = y_true.sum()
        precision = tps / (tps + fps)
        recall = tps / total_pos if total_pos > 0 else tps
        precision = np.concatenate([[1.0], precision])
        recall = np.concatenate([[0.0], recall])
        return precision, recall, y_score


    def manual_average_precision(y_true, y_score):
        precision, recall, _ = manual_precision_recall_curve(y_true, y_score)
        return float(-np.sum(np.diff(recall) * precision[:-1]))


    # =============================
    # Compute AUC (superpixel-level and pixel-level)
    # =============================
    delta_map_smooth = gaussian_filter(np.nan_to_num(delta_map_pca, nan=0.0), sigma=10)

    all_sp_ids = sorted(np.unique(labels_fine).astype(int).tolist())

    sp_scores = []
    sp_labels = []

    for sp_id in all_sp_ids:
        sp_mask = (labels_fine == sp_id)
        n_pix = int(sp_mask.sum())
        if n_pix < 1:
            continue
        score = float(np.nanmean(delta_map_smooth[sp_mask]))
        if np.isnan(score):
            continue
        frac_anomalous = gt_mask_binary[sp_mask].mean()
        label = int(frac_anomalous > 0.5)
        sp_scores.append(score)
        sp_labels.append(label)

    sp_scores = np.array(sp_scores)
    sp_labels = np.array(sp_labels)

    # Superpixel-level
    fpr, tpr, _ = manual_roc_curve(sp_labels, sp_scores)
    roc_auc_sp = manual_auc(fpr, tpr)
    pr_auc_sp = manual_average_precision(sp_labels, sp_scores)

    # Pixel-level
    valid_mask = ~np.isnan(delta_map_smooth) if np.isnan(delta_map_smooth).any() else np.ones_like(delta_map_smooth, dtype=bool)
    pixel_scores = delta_map_smooth[valid_mask].ravel()
    pixel_labels = gt_mask_binary[valid_mask].ravel()

    fpr_px, tpr_px, _ = manual_roc_curve(pixel_labels, pixel_scores)
    roc_auc_px = manual_auc(fpr_px, tpr_px)
    pr_auc_px = manual_average_precision(pixel_labels, pixel_scores)


    # =============================
    # Print results
    # =============================
    print()
    print("=" * 50)
    print(f"SUPERPIXEL-LEVEL:  ROC-AUC = {roc_auc_sp:.4f}  |  AP = {pr_auc_sp:.4f}")
    print(f"PIXEL-LEVEL:       ROC-AUC = {roc_auc_px:.4f}  |  AP = {pr_auc_px:.4f}")
    print("=" * 50)
    mean_PAUC += roc_auc_px
print(f"MEAN PIXEL AUC OVER 10 SAMPLES: {mean_PAUC / 10:.4f}")
