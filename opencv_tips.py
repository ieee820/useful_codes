  #将大图分割成小图  
    img_name = os.path.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            # var = np.var(crop_img / 255)
            # if var > 0.008:
            #     print(img_name, index_str, var)
            cv2.imwrite(
                os.path.join(save_folder, img_name.replace('.png', '_s{:03d}.png'.format(index))),
                crop_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    
    return 'Processing {:s} ...'.format(img_name)
