import os

def init_all_params(init_params, database, storage):
    if database == 'CLASP2':
        camlist = [1, 2, 3, 4, 5, 6]#1: 'cam02exp2.mp4',
        init_params['cam_map'] = {1: 'G_9', 2: 'G_11', 3: 'H_11', 4: 'H_9'
                                  ,5: 'I_9', 6: 'I_11'}
        benchmark = storage + 'tracking_wo_bnw/data/CLASP/train_gt_all/PB_gt'
        out_dir = storage + 'tracking_wo_bnw/output/self-supervise-{}-det/'.format(database.lower())
        init_params['dataset'] = 'C'
        init_params['dataset_map'] = 'C1'
        init_params['frame_rate'] = '10FPS'
        init_params['frame_rate'] = 30
        init_params['server_loc'] = 'CLASP2'

        init_params['img_HW'] = [1080.0, 1920.0]
        init_params['cam'] = camlist
        init_params['start_frame'] = 1000

    if database == 'KRI_exp2_train':
        camlist = [2, 5, 9, 11, 13, 14]#1: 'cam02exp2.mp4',
        init_params['cam_map'] = {2: 'cam02exp2.mp4', 9: 'cam09exp2.mp4', 5: 'cam05exp2.mp4', 11: 'cam11exp2.mp4'
                                  ,13: 'cam13exp2.mp4', 14: 'cam14exp2.mp4'} #, 4: 'cam04exp2.mp4'
        benchmark = storage + 'tracking_wo_bnw/data/CLASP/train_gt'
        out_dir = storage + 'tracking_wo_bnw/output/semi_SL_{}_det/'.format(database.lower())
        init_params['dataset'] = 'C'
        init_params['dataset_map'] = 'C1'
        init_params['frame_rate'] = 30
        init_params['server_loc'] = 'KRI_exp2_train'

        init_params['img_HW'] = [1080.0, 1920.0]
        init_params['cam'] = camlist
        init_params['start_frame'] = 1000
        init_params['fr_offset'] = {init_params['cam_map'][2]: 5, init_params['cam_map'][9]: 10, 
                                    init_params['cam_map'][5]: 10, init_params['cam_map'][11]: 15, 
                                    init_params['cam_map'][13]: 10, init_params['cam_map'][14]: 18}

    if database == 'CLASP1':
        camlist = [1, 2, 3, 4, 5]
        benchmark = storage + 'tracking_wo_bnw/data/CLASP1/train_gt'
        init_params['dataset'] = 'A_9'
        init_params['frame_rate'] = '1FPS'
        init_params['server_loc'] = 'CLASP1'
        init_params['img_HW'] = [1080.0, 1920.0]
        init_params['cam'] = [9]

    if database == 'PVD':
        benchmark = os.path.join(storage, 'data/HDPVD_new/train_gt')
        init_params['dataset'] = 'C'
        init_params['frame_rate'] = 10
        init_params['server_loc'] = 'PVD'
        init_params['img_HW'] = [800.0, 1280.0]
        camlist = [330,  360, 300, 340, 440,]
        init_params['cam_map'] = None
        out_dir = os.path.join(storage, 'self-supervise-pvd-det')
        init_params['cam'] = 1
        init_params['train_cams'] = [1,3]
        init_params['batch_end'] = 5700


    if database == 'LOGAN':
        camlist = [1, 2, 3, 4, 5]
        init_params['cam_map'] = None
        benchmark = storage + 'tracking_wo_bnw/data/logan-data/exp1_logan/train_gt'
        out_dir = storage + 'tracking_wo_bnw/output/self-supervise-logan-det/'
        init_params['dataset'] = 'C1'
        init_params['frame_rate'] = 10
        init_params['server_loc'] = 'LOGAN'
        init_params['img_HW'] = [960.0, 1280.0]
        init_params['cam'] = 1


    if database == 'MOT20':
        benchmark = storage + 'tracking_wo_bnw/data/MOT/MOT20Det/test'
        out_dir = storage + 'tracking_wo_bnw/output/self-supervise-mot20-det/'
        camlist = [1, 2, 3, 4]
        init_params['dataset'] = 'C1'
        init_params['frame_rate'] = 10
        init_params['server_loc'] = 'MOT20'
        init_params['img_HW'] = [960.0, 1280.0]
        init_params['cam'] = 1

    init_params['backbone'] = 'ResNet50FPN' #'ResNet34FPN' #'ResNet50FPN'
    init_params['regress_remap'] =1
    init_params['soft_nms'] = 1
    init_params['skip_nms'] = 0
    init_params['pred_thr'] = 0.5
    init_params['nms_thr'] = 0.4
    init_params['print_stats'] = False
    init_params['isTrain'] = 1
    init_params['save_scores'] = 0
    init_params['cluster_score_thr'] = [0.5, 0.2]  # [0.05, 0.1]
    init_params['det_selection_thr'] = 0.7
    init_params['gpu_test'] = 0
    init_params['gpu_train'] = 0
    init_params['num_epochs'] = 100


    init_params['fr_step'] = 1
    init_params['batch_start'] = 1
    init_params['start_frame'] = 1
    init_params['num_class'] = 2
    init_params['class_id'] = 1

    if init_params['test_aug']:
        if database in ['CLASP1', 'CLASP2', 'PVD']:
            # angleSet =[0, 12, 84, 90, 180, 186, 264, 270, 348, 354]
            init_params['angleSet'] = [0, 6, 12, 18, 24, 84, 90, 96, 168, 174, 180, 186, 192, 264, 270, 330, 336, 342,
                                       348, 354]
        elif database == 'MOT20':
            init_params['angleSet'] = [0, 3, 6, 9, 12, 15, 18, 21, 24, 180, 186, 330, 333, 336, 339, 342, 345, 348, 351,
                                       354]
        elif database=='KRI_exp2_train':
            init_params['angle_ranges'] = [[1, 24], [78, 96], [168, 192], [258, 276], [342, 354]]
            init_params['angleSet'] = [0]
    else:
        init_params['angleSet'] = [0]
    
    return init_params, benchmark, out_dir, camlist