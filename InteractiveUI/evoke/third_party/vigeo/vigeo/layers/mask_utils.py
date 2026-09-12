import torch
def depth_masking(
    x,
    patch_num_h,
    patch_num_w,
    depth_values,
    depth_mask_threshold_ratio=None,
    depth_mask_threshold_num=None,
    valid_depth_range=(0.1, 10.0),
):


    B, N, D = x.shape
    device = x.device

    assert N == patch_num_h * patch_num_w, \
        f"N={N} must equal patch_num_h * patch_num_w = {patch_num_h * patch_num_w}"


    depth_invalid_mask = _compute_depth_invalid_mask(
        depth_values,
        patch_num_h,
        patch_num_w,
        depth_mask_threshold_ratio,
        depth_mask_threshold_num,
        valid_depth_range
    )


    visible_list = []
    mask_info = {
        'visible_indices': [],
        'mask_indices': [],
        'num_visible': [],
    }

    for i in range(B):

        valid_mask = ~depth_invalid_mask[i]
        visible_indices = torch.where(valid_mask)[0]
        masked_indices = torch.where(depth_invalid_mask[i])[0]


        visible = x[i, visible_indices]
        visible_list.append(visible)


        mask_info['visible_indices'].append(visible_indices)
        mask_info['mask_indices'].append(masked_indices)
        mask_info['num_visible'].append(len(visible_indices))

    return visible_list, mask_info

def _compute_depth_invalid_mask(
    depth_values,
    H_patch,
    W_patch,
    threshold_ratio,
    threshold_num,
    valid_range
):


    B, _, H_img, W_img = depth_values.shape
    N = H_patch * W_patch
    device = depth_values.device

    min_depth, max_depth = valid_range


    patch_h = H_img // H_patch
    patch_w = W_img // W_patch

    assert H_img % H_patch == 0 and W_img % W_patch == 0, \
        f"Image size ({H_img}, {W_img}) must be divisible by patch grid ({H_patch}, {W_patch})"


    depth_reshaped = depth_values.view(B, 1, H_patch, patch_h, W_patch, patch_w)


    depth_reshaped = depth_reshaped.permute(0, 2, 4, 1, 3, 5).reshape(B, N, -1)


    valid_depth = (depth_reshaped >= min_depth) & (depth_reshaped <= max_depth)
    valid_depth_ratio = valid_depth.float().mean(dim=-1)
    valid_depth_num = valid_depth.float().sum(dim=-1)


    if isinstance(threshold_ratio, list) or isinstance(threshold_num, list):
        invalid_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        for i in range(B):
            tr = threshold_ratio[i] if isinstance(threshold_ratio, list) else threshold_ratio
            tn = threshold_num[i] if isinstance(threshold_num, list) else threshold_num

            sample_mask = torch.zeros(N, dtype=torch.bool, device=device)
            if tr is not None:
                sample_mask |= (valid_depth_ratio[i] < tr)
            if tn is not None:
                sample_mask |= (valid_depth_num[i] < tn)

            invalid_mask[i] = sample_mask
    else:

        invalid_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        if threshold_ratio is not None:
            invalid_mask |= (valid_depth_ratio < threshold_ratio)
        if threshold_num is not None:
            invalid_mask |= (valid_depth_num < threshold_num)

    return invalid_mask
