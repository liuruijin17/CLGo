import cv2
import numpy as np

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

def draw_paf(pafmap, tl, br, radius, variable_width=False):
    tl = np.array(tl, dtype=float)
    br = np.array(br, dtype=float)
    # print("pafmap.shape: {}".format(pafmap.shape))
    height, width = pafmap.shape[1:]
    if tl[0] >= 0 and tl[1] >= 0 and br[0] <= width and br[1] <= height:
        part_line_segment = br - tl
        l = np.linalg.norm(part_line_segment)
        # print('l: {}'.format(l))

        if l > 1e-2:
            sigma = radius
            if variable_width:
                sigma = radius * l * 0.025
            v = part_line_segment / l
            # print('v: {}'.format(v))
            v_per = v[1], -v[0]
            x, y = np.meshgrid(np.arange(width), np.arange(height))

            dist_along_part = v[0] * (x - tl[0]) + v[1] * (y - tl[1])
            dist_per_part = np.abs(v_per[0] * (x - tl[0]) + v_per[1] * (y - tl[1]))

            mask1 = dist_along_part >= 0
            mask2 = dist_along_part <= l
            mask3 = dist_per_part <= sigma
            mask = mask1 & mask2 & mask3
            pafmap[0] = pafmap[0] + mask.astype('float32') * v[0]
            pafmap[1] = pafmap[1] + mask.astype('float32') * v[1]
            # indicator = np.where(mask, 255, 0)
            # indicator = indicator.astype('uint8')
            # cv2.imshow('figure1', indicator)
            # cv2.waitKey(0)
    # raise NotImplementedError("draw paf is not implemented")

def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)

def _get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

def random_crop(image, detections, joints, random_scales, view_size, border=64):
    view_height, view_width   = view_size
    image_height, image_width = image.shape[0:2]
    mask = np.zeros((image_height, image_width, 1), dtype=np.float32)  # init a mask for raw image (all set to zeros)
    # print('image_height: {}'.format(image_height)) # 720
    # print('image_width: {}'.format(image_width)) # 1280

    scale  = np.random.choice(random_scales)
    height = int(view_height * scale)
    width  = int(view_width  * scale)
    # print('random_scales: {}'.format(random_scales)) # [0.6 0.7 0.8 0.9 1.  1.1 1.2 1.3]
    # print('scale: {}'.format(scale))
    # print('height: {}'.format(height)) # 511 * scale
    # print('width: {}'.format(width)) # 511 * scale

    cropped_image = np.zeros((height, width, 3), dtype=image.dtype)
    cropped_mask  = np.ones((height, width, 1), dtype=np.float32)


    w_border = _get_border(border, image_width)
    h_border = _get_border(border, image_height)
    # print('w_border: {}'.format(w_border)) # 128 (always)
    # print('h_border: {}'.format(h_border)) # 128 (always)
    # h_border *= 2
    # exit()

    w_low  = image_width // 2 - (width // 2 - image_width // 2)
    w_high = image_width // 2 + (width // 2 - image_width // 2) + 1
    h_low  = image_height // 2 - (height // 2 - image_height // 2)
    h_high = image_height // 2 + (height // 2 - image_height // 2) + 1
    # print(w_low, w_high)
    # print(h_low, h_high)
    # exit()


    # ctx = np.random.randint(low=w_border, high=image_width - w_border)
    # cty = np.random.randint(low=h_border, high=image_height - h_border)

    ctx = np.random.randint(low=w_low, high=w_high)
    cty = np.random.randint(low=h_low, high=h_high)


    x0, x1 = max(ctx - width // 2, 0),  min(ctx + width // 2, image_width)
    y0, y1 = max(cty - height // 2, 0), min(cty + height // 2, image_height)

    left_w, right_w = ctx - x0, x1 - ctx
    top_h, bottom_h = cty - y0, y1 - cty

    # crop image
    cropped_ctx, cropped_cty = width // 2, height // 2
    x_slice = slice(cropped_ctx - left_w, cropped_ctx + right_w)
    y_slice = slice(cropped_cty - top_h, cropped_cty + bottom_h)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    # crop mask
    cropped_mask[y_slice, x_slice, :]  = mask[y0:y1, x0:x1, :]

    # crop detections
    cropped_detections = detections.copy()
    cropped_detections[:, 0:4:2] -= x0
    cropped_detections[:, 1:4:2] -= y0
    cropped_detections[:, 0:4:2] += cropped_ctx - left_w
    cropped_detections[:, 1:4:2] += cropped_cty - top_h

    cropped_joints = joints.copy()
    cropped_joints[:, :,  0] -= x0
    cropped_joints[:, :,  1] -= y0
    cropped_joints[:, :,  0] += cropped_ctx - left_w
    cropped_joints[:, :,  1] += cropped_cty - top_h


    return cropped_image, cropped_mask, cropped_detections, cropped_joints

def resize_hm(heatmap, hm_size):
    if np.isscalar(hm_size):
        hm_size = (hm_size, hm_size)
    heatmap = cv2.resize(heatmap.transpose(1, 2, 0), hm_size,interpolation=cv2.INTER_CUBIC)
    return heatmap.transpose(2, 0, 1)

def resize_hm_paf(heatmap, paf, hm_size):
    heatmap = resize_hm(heatmap, hm_size)
    # print(paf.shape) # 1 2 192 352
    paf = paf.transpose(2,3,0,1)
    # print(paf.shape) # 192 352 1 2
    paf = paf.reshape(paf.shape[0], paf.shape[1], paf.shape[2] * paf.shape[3])
    # print(paf.shape) # 192 352 2
    paf = cv2.resize(paf, hm_size,interpolation=cv2.INTER_CUBIC)
    # print(paf.shape) # 767 1407 2
    paf = paf.transpose(2, 0, 1)
    # print(paf.shape) # 2 767 1407
    return heatmap, paf

def visualize_heatmap(img, heat_maps, displayname='heatmaps'):
    heat_maps = heat_maps.max(axis=0)
    heat_maps = (heat_maps / heat_maps.max() * 255.).astype('uint8')
    img = img.copy()
    # print(type(img), type(heat_maps)) # np array
    # print(img.shape, heat_maps.shape) #
    colored = cv2.applyColorMap(heat_maps, cv2.COLORMAP_JET)
    # img = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
    img = img * 0.6 + colored * 0.4
    img = (img / img.max() * 255.).astype('uint8')
    cv2.imshow(displayname, img)
    cv2.waitKey()

colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]]

def visualize_paf(img, pafs,name='pafs'):
    img = img.copy()
    for i in range(pafs.shape[0]):
        paf_x = pafs[i,0,:,:]
        paf_y = pafs[i,1,:,:]
        len_paf = np.sqrt(paf_x**2 + paf_y**2)
        for x in range(0, img.shape[0], 8):
            for y in range(0, img.shape[1], 8):
                if len_paf[x,y]>0.25:
                    img = cv2.arrowedLine(img, (y,x), (int(y + 6*paf_x[x,y]), int(x + 6*paf_y[x,y])), colors[i], 1)
    cv2.imshow(name, img)
    cv2.waitKey()

def visualize_heatmap_return(img, heat_maps, displayname = 'heatmaps'):
    heat_maps = heat_maps.max(axis=0)
    heat_maps = (heat_maps / heat_maps.max() * 255.).astype('uint8')
    img = img.copy()
    # print(type(img), type(heat_maps)) # np array
    # print(img.shape, heat_maps.shape) #
    colored = cv2.applyColorMap(heat_maps, cv2.COLORMAP_JET)
    # img = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
    img = img * 0.6 + colored * 0.5
    img = (img / img.max() * 255.).astype('uint8')
    return img


def visualize_paf_return(img, pafs,name='pafs'):
    img = img.copy()
    for i in range(pafs.shape[0]):
        paf_x = pafs[i,0,:,:]
        paf_y = pafs[i,1,:,:]
        len_paf = np.sqrt(paf_x**2 + paf_y**2)
        for x in range(0, img.shape[0], 8):
            for y in range(0, img.shape[1], 8):
                # if len_paf[x,y] > 0.25:
                    img = cv2.arrowedLine(img, (y,x), (int(y + 6*paf_x[x,y]), int(x + 6*paf_y[x,y])), [0, 255, 255], 1)
    return img

def generate_target(joints, joints_vis, num_joints, heatmap_size, image_size, sigma, target_type='gaussian'):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    assert target_type == 'gaussian', \
        'Only support gaussian map now!'

    if target_type == 'gaussian':
        target = np.zeros((num_joints,
                           heatmap_size[1],
                           heatmap_size[0]),
                          dtype=np.float32)

        tmp_size = sigma * 3

        for joint_id in range(num_joints):
            feat_stride = image_size / heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0 # if out-bounds, set target_weight = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

            if joints_vis[joint_id, -1]:
                target[joint_id][:, :] = np.zeros((heatmap_size[1], heatmap_size[0]), dtype=np.float32)


    return target, target_weight


