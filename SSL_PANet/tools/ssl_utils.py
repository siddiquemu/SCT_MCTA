def get_dirs_params(params):
    if params['baseline'] == 'ResNet50':
        cfg_file = '/home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml'
        load_ckpt = '/home/siddique/PANet/panet_mask_step179999.pth'  # baseline
        data = 'coco'
        class_list = [1, 25, 27, 29]  # list(np.arange(1,81)) #[1, 25, 27, 29]
        load_detectron = None

    if params['baseline'] == 'ResNet50AICity':
        cfg_file = '/home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml'
        load_ckpt = '/home/siddique/PANet/panet_mask_step179999.pth'  # baseline
        data = 'coco'
        class_list = [3, 6, 8]
        load_detectron = None

    if params['baseline'] == 'ResNet50_box_mask_tuned':
        cfg_file = '/home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml'

        if params['database'] in ['clasp2', 'clasp2_30fps']:
            if alpha_loss:
                load_ckpt = mask_model + '/clasp2/modified_loss/iter{}/ckpt/model_step19999.pth'.format(iter)
                # load_ckpt = mask_model + '/Jul24-00-31-16_siddiquemu_step/ckpt/model_step19999.pth'
                print('modified loss model: {}'.format(load_ckpt))

            elif alpha_loss_sigm:
                # load_ckpt =  mask_model + '/clasp2/loss_sigmoid/iter{}/ckpt/model_step19999.pth'.format(iter)
                load_ckpt = mask_model + '/Jul29-01-14-32_siddiquemu_step/ckpt/model_step9999.pth'
                print('modified loss model: {}'.format(load_ckpt))

            else:
                # default loss
                load_ckpt = storage + 'PANet_Models/clasp2/wo_score_loss/iter{}/ckpt/model_step19999.pth'.format(iter)

            data = 'clasp2_2021'
            class_list = [1, 2]

        if params['database'] in ['clasp1', 'clasp1_30fps']:
            if modified_loss:
                load_ckpt = mask_model + '/clasp1/modified_loss/iter{}/ckpt/model_step19999.pth'.format(iter)
                # load_ckpt = mask_model + '/Jul24-00-31-16_siddiquemu_step/ckpt/model_step19999.pth'
                print('alpha loss model: {}'.format(load_ckpt))
            elif modified_loss_sigm:
                load_ckpt = mask_model + '/clasp1/loss_sigmoid/iter{}/ckpt/model_step19999.pth'.format(iter)
                # load_ckpt = mask_model + '/Jul29-01-14-32_siddiquemu_step/ckpt/model_step4854.pth'
                print('sigmoid alpha loss model: {}'.format(load_ckpt))
            else:
                load_ckpt = mask_model + '/clasp1/iter{}/ckpt/model_step19999.pth'.format(iter)

            data = 'clasp1_2021'
            class_list = [1, 2]
        if params['database'] == 'PVD':
            load_ckpt = mask_model + '/PVD_tuned00/ckpt/model_step12898.pth'
            data = 'clasp1_2021'
            class_list = [1, 2]
        if params['database'] == 'MOT20':
            load_ckpt = mask_model + '/MOT20Det_iter1/ckpt/model_step29842.pth'
            data = 'mot_2020'
            class_list = [1]
        load_detectron = None

    if baseline == 'ResNet50DOTA':
        cfg_file = '/home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml'
        load_ckpt = '/home/siddique/PANet/panet_mask_step179999.pth'  # baseline
        data = 'coco'
        class_list = list(np.arange(0, 81))
        load_detectron = None

    images = False
    set_cfgs = None
    cuda = True
    maskRCNN, dataset = configure_detector(data, pred_score, gpu_id)
    # cams = ['C3']#,'C2','C3', 'C4', 'C5', 'C6', 'C7']
    # cams = ['group_A', 'group_B'] , 'C4']

    if params['database'] == 'AICity':
        cams = ['c007']  # , 'C2', 'C3', 'C4']
    if params['database'] == 'iSAID':
        cams = ['part1-002']  # , 'C2', 'C3', 'C4']

    if params['database'] == 'COCO':
        cams = ['val2017']
    if params['database'] == 'DAVIS':
        cams = glob.glob(storage + 'tracking_wo_bnw/data/DAVIS/DAVIS/JPEGImages/480p/' + '*')
        # cams.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
        cams = [cam.split('/')[-1] for cam in cams]
    if params['database'] == 'MOT20':
        cams = ['C4']
    if params['database'] == 'clasp1':
        cams = ['A_11', 'A_9', 'B_11', 'B_9', 'C_11', 'C_9', 'D_11', 'D_9', 'E_11', 'E_9']

    if params['database'] in ['clasp2_30fps']:
        cams = ['G_2']  # , 'G_5', 'G_9', 'G_11', 'H_2', 'H_5', 'H_9', 'H_11', 'I_2', 'I_5', 'I_9', 'I_11']
    if params['database'] in ['clasp2']:
        cams = ['G_9', 'G_11', 'H_9', 'H_11', 'I_9', 'I_11']

    if params['database'] == 'PVD':
        cams = ['C1']  # ,'C2','C3', 'C4', 'C5', 'C6']
    if params['database'] == 'clasp1_30fps':
        server = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/' \
                 'ourdataset/exp5a/'
        # cams = glob.glob(server+'imgs/*')
        cams = ['cam05exp5a.mp4', 'cam09exp5a.mp4', 'cam02exp5a.mp4', 'cam11exp5a.mp4']

    if params['test_aug']:
        if params['database'] in ['clasp1', 'clasp2']:
            # angleSet = [0, 6, 12, 78, 84, 90, 96, 102, 168, 174, 180, 186, 192, 258, 264, 270, 276, 342, 348, 354] #[0, 6, 12, 90, 96, 180, 186, 270, 348, 354]#
            # angleSet = [0, 6, 12, 18, 24, 72, 78, 84, 90, 96, 102, 162, 168, 174, 180, 186, 192, 252, 258, 264, 270, 276, 272,336, 342, 348, 354]
            angleSet = [0, 6, 12, 78, 84, 90, 96, 102, 168, 174, 180, 186, 192, 258, 264, 270, 276, 342, 348, 354]
            # angleSet = list(np.arange(0, 360, 3))
        elif params['database'] == 'MOT20':
            angleSet = [0, 3, 6, 9, 12, 15, 18, 21, 336, 339, 342, 345, 348, 351, 354]
            angleSet = [0, 6, 12, 18, 24, 78, 174, 180, 186, 192, 330, 336, 342, 348, 354]
        else:
            angleSet = [0, 6, 12, 18, 24, 78, 174, 180, 186, 192, 330, 336, 342, 348, 354]
    else:
        angleSet = [0]

    return maskRCNN, dataset

def get_cam_dirs(params):
    for cam in cams:
        if params['database'] == 'wild-track':
            # image_dir = storage+'tracking_wo_bnw/data/wild-track/imgs_30fps/' + cam
            image_dir = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/wildtrack/Wildtrack_dataset/Image_subsets/' + cam
            output_dir = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/PANet_Results/wild-track/' + cam + '/resnet50-baseline'
            out_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/PANet_Results/wild-track/' + cam + '/resnet50-baseline-aug'  # det result path
            img_fmt = 'png'

        if params['database'] == 'kri':
            image_dir = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp1/imgs/' + cam
            output_dir = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp1/Results/' + cam + '/panet_tuned_rotationv2_angle0_180'
            out_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp1/Results/' + cam + '/box_mask_aug_panetv2/'
        if params['database'] == 'logan':
            save_dir = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/tracking_wo_bnw'
            image_dir = save_dir + '/data/logan-data/exp1-train/' + cam
            output_dir = save_dir + '/output/PANet-det/logan/' + cam + '/panet_tuned_rotationv2_angle0'
            out_path = save_dir + '/output/PANet-det/logan/' + cam + '/box_mask_aug_panetv2/'
        if params['database'] == 'clasp-5k':
            save_dir = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw'
            image_dir = save_dir + '/data/5k-data/' + cam
            output_dir = save_dir + '/output/PANet-det/5k-data/' + cam + '/panet_tuned_rotationv2_angle0'
            out_path = save_dir + '/output/PANet-det/5k-data/' + cam + '/box_mask_aug_panetv2/'
            img_fmt = 'jpg'

        if params['database'] == 'clasp1':
            image_dir = storage + 'tracking_wo_bnw/data/CLASP1/train_gt/' + cam + '/img1'
            if test_aug:
                output_dir = storage + 'PANet_Results/CLASP1/PANet_mask/' + cam + '/resnet50-aug'
            else:
                output_dir = storage + 'PANet_Results/CLASP1/PANet_mask/' + cam + '/resnet50-infer'
            result_path = storage + 'PANet_Results/CLASP1/PANet_mask'  # det result path
            img_fmt = 'png'

        if params['database'] in ['clasp2', 'clasp2_30fps']:
            image_dir = storage + 'tracking_wo_bnw/data/CLASP/train_gt_all/PB_gt/' + cam + '/img1'
            if test_aug:
                output_dir = storage + 'PANet_Results/CLASP2/PANet_mask/' + cam + '/resnet50-aug'
            else:
                output_dir = storage + 'PANet_Results/CLASP2/PANet_mask/' + cam + '/resnet50-tunedms'
            result_path = storage + 'PANet_Results/CLASP2/PANet_mask'  # det result path

            if params['database'] == 'clasp2_30fps':
                if alpha_loss:
                    result_folder = 'ssl_alpha_dets_30fps'
                else:
                    result_folder = 'ssl_dets_30fps'
                result_path = storage + 'PANet_Results/CLASP2/PANet_mask/' + result_folder
                output_dir = result_path + '/' + cam
                print('results will be svaed to: {}'.format(output_dir))

            img_fmt = 'jpg'

        if params['database'] == 'clasp1_30fps':
            image_dir = os.path.join(server, 'imgs', cam)
            output_dir = storage + 'PANet_Results/CLASP1/PANet_mask/dets_30fps/' + cam
            result_path = output_dir
            img_fmt = 'png'

        if params['database'] == 'PVD':
            image_dir = storage + 'tracking_wo_bnw/data/PVD/HDPVD/cams/' + cam
            if params['test_aug']:
                output_dir = storage + 'PANet_Results/PVD/HDPVD/PANet_mask/' + cam + '/resnet50-base'
            else:
                output_dir = storage + 'PANet_Results/PVD/HDPVD/PANet_mask/' + cam + '/resnet50-aug'
            result_path = storage + 'PANet_Results/PVD/HDPVD/PANet_mask'  # det result path
            img_fmt = 'png'

        if params['database'] == 'AICity':
            image_dir = storage + 'tracking_wo_bnw/data/AICityChallenge/AIC21_Track3_MTMC_Tracking/' \
                                  'AIC21_Track3_MTMC_Tracking/validation/S02/' + cam + '/img1'
            if params['test_aug']:
                output_dir = storage + 'PANet_Results/AICity/PANet_mask/' + cam + '/resnet50-base'
            else:
                output_dir = storage + 'PANet_Results/AICity/PANet_mask/' + cam + '/resnet50-tuned'
            result_path = storage + 'PANet_Results/AICity/PANet_mask'  # det result path
            img_fmt = 'png'

        if params['database'] == 'COCO':
            image_dir = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/coco_images/coco_2017/' + cam
            if params['test_aug']:
                output_dir = storage + 'PANet_Results/COCO/PANet_mask/' + cam + '/resnet50-base'
            else:
                output_dir = storage + 'PANet_Results/COCO/PANet_mask/' + cam + '/resnet50-tuned'
            result_path = storage + 'PANet_Results/COCO/PANet_mask'  # det result path
            img_fmt = 'jpg'

        if params['database'] == 'DAVIS':
            image_dir = storage + 'tracking_wo_bnw/data/DAVIS/DAVIS/JPEGImages/480p/' + cam
            if params['test_aug']:
                output_dir = storage + 'PANet_Results/DAVIS/PANet_mask/' + cam + '/resnet50-base'
            else:
                output_dir = storage + 'PANet_Results/DAVIS/PANet_mask/' + cam + '/resnet50-tuned'
            result_path = storage + 'PANet_Results/DAVIS/PANet_mask'  # det result path
            img_fmt = 'jpg'

        if params['database'] == 'iSAID':
            image_dir = storage + 'tracking_wo_bnw/data/iSAID/' + cam + '/images'
            if params['test_aug']:
                output_dir = storage + 'PANet_Results/iSAID/PANet_mask/' + cam + '/resnet50-base'
            else:
                output_dir = storage + 'PANet_Results/iSAID/PANet_mask/' + cam + '/resnet50-tuned'
            result_path = storage + 'PANet_Results/iSAID/PANet_mask'  # det result path
            img_fmt = 'png'

        if params['database'] == 'MOT20':
            image_dir = storage + 'tracking_wo_bnw/data/MOT/MOT20Det/test/' + cam + '/img1'
            if test_aug:
                output_dir = storage + 'PANet_Results/MOT20/PANet_mask/' + cam + '/resnet50-base'
            else:
                output_dir = storage + 'PANet_Results/MOT20/PANet_mask/' + cam + '/resnet50-tuned'
            result_path = storage + 'PANet_Results/MOT20/PANet_mask'  # det result path
            img_fmt = 'jpg'
    return image_dir, output_dir, img_fmt