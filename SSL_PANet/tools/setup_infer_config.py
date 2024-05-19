import numpy as np
def get_infer_config(args):
    config = {}
    # model and loss
    config['iter'] = args.ssl_iter
    config['gpu_id'] = args.gpu_id
     #'ResNet50'#'ResNet50_box_mask_tuned'
    config['baseline'] = 'ResNet50_box_mask_tuned' 
    config['SSL_alpha_loss'] = 1
    config['SSL_loss'] = 0
    config['SL_loss'] = 0
    # regression
    config['regress_augment'] = 1
    # test aug
    config['test_aug'] = 0
    config['save_aug'] = config['test_aug']

    config['pred_score'] = 0.5
    config['soft_nms'] = 1
    config['nms_thr'] = 0.3
    config['save_mask'] = 0

    config['mask_model'] = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62/PANet_Models'
    config['det_model'] = '/home/siddique/PANet/Outputs/e2e_panet_R-50-FPN_1x_det'

    # vis
    config['vis'] = 1
    config['cluster_mode_vis'] = 0
    config['vis_rate'] = 1

    if config['baseline'] == 'ResNet50':
        config['cfg_file'] = './configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml'
        config['load_ckpt'] = './models/R50-MRCNN/panet_mask_step179999.pth'  # baseline
        config['data'] = 'coco'
        config['class_list'] = [25, 27, 29] #list(np.arange(1,81)) #[1, 25, 27, 29]
        config['load_detectron'] = None

    if config['baseline'] == 'ResNet50_box_mask_tuned':
        config['class_list'] = [1, 2]
        config['cfg_file'] = './configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml'

        if args.dataset in ['clasp2', 'clasp2_30fps']:
            config['data'] = 'clasp2_2021'
            if config['SSL_alpha_loss']:
                # SSL-alpha
                config['load_ckpt'] =  f"{args.model_dir}/iter{args.ssl_iter}/ckpt/model_step19999.pth"
                #load_ckpt = mask_model + '/clasp2/modified_loss_mixed_aug/iter{}/ckpt/model_step19999.pth'.format(iter)
                print('alpha loss model: {}'.format(config['load_ckpt']))

            elif config['SSL_loss']:
                # SL loss
                config['load_ckpt'] = f"{config['det_model']}/PANet_box_tuned/ckpt/model_step19999.pth"
                print('SL loss model: {}'.format(config['load_ckpt']))

            elif config['SL_loss']:
                # SSL loss
                config['load_ckpt'] = f"{config['mask_model']}/clasp2/wo_score_regress_loss/iter{args.ssl_iter}/ckpt/model_step19999.pth"
                print('default loss model: {}'.format(config['load_ckpt']))
            else:
                print('base model read separately')

        if args.dataset in ['clasp1', 'clasp1_30fps']:
            config['data'] = 'clasp1_2021'
            if config['SSL_alpha_loss']:
                #config['load_ckpt'] =f"{config['mask_model']}/clasp1/modified_loss_wo_reg/iter{args.ssl_iter}/ckpt/model_step19999.pth"
                config['load_ckpt'] =  f"{args.model_dir}/iter{args.ssl_iter}/ckpt/model_step19999.pth"
                #config['load_ckpt'] = f"{config['mask_model']}/clasp1/modified_loss_mixed_aug/iter{args.ssl_iter}/ckpt/model_step9999.pth"
                print('alpha loss model: {}'.format(config['load_ckpt']))
            elif config['SSL_loss']:
                config['load_ckpt'] = f"{config['mask_model']}/clasp1/wo_score_regress_loss/iter{args.ssl_iter}/ckpt/model_step19999.pth"
            else:
                print('base model read separately')

    config['images'] = False
    config['set_cfgs'] = None
    config['cuda'] = True
    config['load_detectron'] = False

    if args.dataset == 'clasp1':
        config['cams'] = ['A_11', 'A_9', 'B_11', 'B_9', 'C_11', 'C_9', 'D_11', 'D_9', 'E_11', 'E_9']

    if args.dataset in ['clasp2_30fps']:
        config['cams'] = ['G_2', 'G_5', 'H_2', 'H_5', 'I_2','I_5']  # ['G_9', 'G_11', 'H_9', 'H_11', 'I_9', 'I_11'] #['G_2', 'G_5', 'H_2', 'H_5', 'I_2', 'I_5']

    if args.dataset in ['clasp2']:
        config['cams'] = ['G_9', 'G_11', 'H_9', 'H_11', 'I_9', 'I_11']

    if args.dataset == 'clasp1_30fps':
        server = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/' \
                 'ourdataset/exp6a/'
        # cams = glob.glob(server+'imgs/*')
        config['cams'] = ['cam05exp6a.mp4', 'cam09exp6a.mp4', 'cam02exp6a.mp4', 'cam11exp6a.mp4']

    if config['test_aug']:
        if args.dataset in ['clasp1', 'clasp2']:
            #angleSet = [0, 6, 12, 78, 84, 90, 96, 102, 168, 174, 180, 186, 192, 258, 264, 270, 276, 342, 348, 354]
            #angleSet = [0, 12, 84, 90, 174, 180, 192, 270, 348, 354]
            config['angleSet'] = [0, 90, 180, 270, 354]
        else:
            #angleSet = [0, 6, 12, 78, 84, 90, 96, 102, 168, 174, 180, 186, 192, 258, 264, 270, 276, 342, 348, 354]
            config['angleSet'] = [0, 12, 84, 90, 174, 180, 192, 270, 348, 354]
    else:
        config['angleSet'] = [0]
    return config