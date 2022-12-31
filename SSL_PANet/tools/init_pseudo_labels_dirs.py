from clasp2coco import load_clasp_json, load_flower_json
import os
import numpy as np
import pdb
import glob

def delete_all(demo_path, fmt='png'):
    try:
        filelist = glob.glob(os.path.join(demo_path, '*'))
        if len(filelist) > 0:
            for f in filelist:
                os.remove(f)
    except:
        print(f'{demo_path} is already empty')

def get_all_dirs(args, exp, init_params, storage, models_dir, model_type, percent_gt=None,):
    # datasets and model catalogs for Self-SL or Semi-SL
    
    if init_params['database'] in ['clasp1', 'clasp2']:
        if init_params['database'] == 'clasp2':
            benchmark = storage + '/tracking_wo_bnw/data/CLASP/'
            init_params['train_img_dir'] = storage + '/tracking_wo_bnw/data/CLASP/train_gt_det/img1/'
        else:
            benchmark = storage + '/tracking_wo_bnw/data/CLASP1/'
            init_params['train_img_dir'] = storage + '/tracking_wo_bnw/data/CLASP1/train_gt_det/img1/'

        if model_type=='modified_loss_semi':
            init_params['output_dir'] = benchmark + f'test_augMS_gt_score/iter{exp}'
            if not os.path.exists(init_params['output_dir']):
                os.makedirs(init_params['output_dir'])

            init_params['dataJSON'] = init_params['output_dir'] + f'/{args.database}_test_aug_{exp}.json'
            init_params['AugImgDir'] = init_params['output_dir'] + f'/img1_{exp}'
            if not os.path.exists(init_params['AugImgDir']):
                os.makedirs(init_params['AugImgDir'])

            init_params['model_path'] = os.path.join(storage, models_dir, init_params['database'],
                                                     model_type, f'{percent_gt}_percent',
                                                     f'iter{exp - 1}', 'ckpt/model_step19999.pth')
            if not os.path.exists(init_params['model_path']):
                init_params['model_path'] = os.path.join(storage, models_dir, init_params['database'],
                                                         model_type, f'{percent_gt}_percent',
                                                         f'iter{exp - 1}', 'ckpt/model_step9999.pth')

            init_params['det_score_file'] = init_params['output_dir'] + f'/det_scores_{args.database}_test_aug_{exp}.csv'
            init_params['cluster_score_file'] = init_params['output_dir'] + f'/cluster_scores_{args.database}_test_aug_{exp}.csv'

        elif model_type=='modified_loss_wo_reg':
            init_params['output_dir'] = benchmark + f'test_augMS_gt_wo_regress/iter{exp}'
            if not os.path.exists(init_params['output_dir']):
                os.makedirs(init_params['output_dir'])

            init_params['dataJSON'] = init_params['output_dir'] + f'/{args.database}_test_aug_{exp}.json'
            init_params['AugImgDir'] = init_params['output_dir'] + f'/img1_{exp}'
            if not os.path.exists(init_params['AugImgDir']):
                os.makedirs(init_params['AugImgDir'])

            init_params['model_path'] = os.path.join(models_dir, init_params['database'], model_type,
                                                     f'iter{exp - 1}', 'ckpt/model_step19999.pth')

            init_params['det_score_file'] = init_params['output_dir'] + f'/det_scores_{args.database}_test_aug_{exp}.csv'
            init_params['cluster_score_file'] = init_params['output_dir'] + f'/cluster_scores_{args.database}_test_aug_{exp}.csv'
        
        elif model_type=='modified_loss':
            init_params['output_dir'] = benchmark + f'test_augMS_gt_score/iter{exp}'
            if not os.path.exists(init_params['output_dir']):
                os.makedirs(init_params['output_dir'])
            #pseudo labels and data
            init_params['dataJSON'] = init_params['output_dir'] + f'/{args.database}_test_aug_{exp}.json'
            init_params['AugImgDir'] = init_params['output_dir'] + f'/img1_{exp}'
            if not os.path.exists(init_params['AugImgDir']):
                os.makedirs(init_params['AugImgDir'])
            #model used to generate pseudo-labels prediction
            if exp>0:
                #prev model
                init_params['model_path'] = os.path.join(models_dir, init_params['database'], model_type,
                                                        f'iter{exp - 1}', 'ckpt/model_step19999.pth')
            else:
                init_params['model_path'] = f'{storage}/PANet/panet_mask_step179999.pth'
            assert os.path.exists(init_params['model_path'])

            init_params['det_score_file'] = init_params['output_dir'] + f'/det_scores_{args.database}_test_aug_{exp}.csv'
            init_params['cluster_score_file'] = init_params['output_dir'] + f'/cluster_scores_{args.database}_test_aug_{exp}.csv'

        if init_params['semi_supervised']:

            # load gt json for percent gt data: used as manual annotations
            # Data instances_train_clasp2.1@10.json saved (138 images 554 annotations)
            json_path = storage + '/SoftTeacher/data/{}/annotations/semi_supervised/instances_train_{}.1@{}.json'.format(
            args.database, args.database, percent_gt)
            init_params['percent_clasp_gt'], init_params['semi_frames'] = load_clasp_json(json_path, percent=percent_gt)
            semi_size = (len(init_params['semi_frames']))

            # load full gt data
            json_path = storage + '/SoftTeacher/data/{}/annotations/instances_train_{}.json'.format(args.database, args.database)

            _, all_frames = load_clasp_json(json_path, percent=100)
            print(f'total gt: {len(all_frames)}\t semi gt: {semi_size}')
            init_params['labeled_frames'] = all_frames

            labeled_max_fr = max(all_frames)
            init_params['labeled_max_fr'] = labeled_max_fr
            all_frames_unlabeled = None
        else: #fully self-supervised
            # load full gt data as unlabeled frames
            json_path = storage + '/SoftTeacher/data/{}/annotations/instances_train_{}.json'.format(args.database, args.database)
            _, all_frames = load_clasp_json(json_path, percent=100)
            init_params['semi_frames'] = []
            init_params['labeled_frames'] = all_frames
            print(f'total gt: {len(all_frames)}\t semi gt: {None}')
            init_params['labeled_max_fr'] = None

    elif init_params['database']=='flower':
        # source images dirs
        #init_params['train_img_dir'] = storage + f'/SoftTeacher/data/{args.database}/trainFlower/'
        #init_params['unlabeled_img_dir'] = storage + f'/SoftTeacher/data/{args.database}/unlabeledFlower/'
        # init_params['train_img_dir'] = storage + f'/tracking_wo_bnw/data/{args.database}/train_gt_sw/trainFlowerAug/'
        # init_params['unlabeled_img_dir'] = storage + f'/tracking_wo_bnw/data/{args.database}/train_gt_sw/unlabeledFlower/'
        init_params['train_img_dir'] = storage + f'/tracking_wo_bnw/data/{args.database}/train_gt_sw/trainFlowerAug/'
        init_params['unlabeled_img_dir'] = storage + f'/tracking_wo_bnw/data/{args.database}/train_gt_sw/unlabeledFlowerAppleA/'
        


        if model_type=='modified_loss_semi':
            # training data dirs
            benchmark = storage + f'/tracking_wo_bnw/data/{args.database}/'
            init_params['output_dir'] = benchmark + f'test_augMS_gt_score/iter{exp}'
            if not os.path.exists(init_params['output_dir']):
                os.makedirs(init_params['output_dir'])


            # images and JSON file dirs
            init_params['dataJSON'] = init_params['output_dir'] + f'/{args.database}_test_aug_{exp}.json'
            init_params['AugImgDir'] = init_params['output_dir'] + f'/img1_{exp}'
            if not os.path.exists(init_params['AugImgDir']):
                os.makedirs(init_params['AugImgDir'])
            else:
                delete_all(init_params['AugImgDir'])

            # previous model dir
            init_params['model_path'] = os.path.join(storage, models_dir, init_params['database'],
                                                     model_type, f'{percent_gt}_percent',
                                                     f'iter{exp - 1}', 'ckpt/model_step39999.pth')
            if not os.path.exists(init_params['model_path']):                                         
                init_params['model_path'] = os.path.join(storage, models_dir, init_params['database'],
                                                        model_type, f'{percent_gt}_percent',
                                                        f'iter{exp - 1}', 'ckpt/model_step19999.pth')
            if not os.path.exists(init_params['model_path']):
                init_params['model_path'] = os.path.join(storage, models_dir, init_params['database'],
                                                         model_type, f'{percent_gt}_percent',
                                                         f'iter{exp - 1}', 'ckpt/model_step9999.pth')
            else:
                assert os.path.exists(init_params['model_path']), '{} is not available'.format(init_params['model_path'])
                

            # pseudo labels score file dir
            init_params['det_score_file'] = init_params['output_dir'] + f'/det_scores_{args.database}_test_aug_{exp}.csv'
            init_params['cluster_score_file'] = init_params['output_dir'] + f'/cluster_scores_{args.database}_test_aug_{exp}.csv'

        if init_params['semi_supervised']:

            # # load gt json for percent gt data: used as manual annotations
            # json_path = f'{storage}/tracking_wo_bnw/data/{args.database}/train_gt_sw/instances_train_2021.json'
            # #json_path = f'{storage}/SoftTeacher/data/{args.database}/annotations/semi_supervised/instances_train_2021.1@{percent_gt}.json'
            # init_params['percent_clasp_gt'], init_params['semi_frames'] = load_clasp_json(json_path, percent=percent_gt)
            # semi_size = (len(init_params['semi_frames']))
            # labeled_frames = init_params['semi_frames']
            init_params['semi_frames'] = []
            semi_size = (len(init_params['semi_frames']))
            init_params['labeled_frames'] = init_params['semi_frames']

            # # load full gt data
            # json_path = f'{storage}/tracking_wo_bnw/data/{args.database}/train_gt_sw/instances_train_2021.json'
            # #json_path = f'{storage}/SoftTeacher/data/{args.database}/annotations/instances_train_2021.json'
            # _, labeled_frames = load_clasp_json(json_path, percent=100)
            # print(f'total gt: {len(labeled_frames)}\t semi gt: {semi_size}')
            # init_params['labeled_frames'] = labeled_frames
            # if percent_gt==100:
            #     assert init_params['semi_frames'].sort() == labeled_frames.sort()

            #load all unlabeled frames
            #json_path_unlabeled = f'{storage}/tracking_wo_bnw/data/{args.database}/train_gt_sw/instances_unlabeled_2021.json'
            json_path_unlabeled = f'{storage}/tracking_wo_bnw/data/{args.database}/train_gt_sw/instances_unlabeled_2021_AppleA.json'
            #json_path_unlabeled = storage + '/SoftTeacher/data/{}/annotations/instances_unlabeled_2021.json'.format(args.database)
            _, unlabeled_frames = load_clasp_json(json_path_unlabeled, percent=100)
            print(f'total unlabeled: {len(unlabeled_frames)}\t semi gt: {semi_size}')
            init_params['unlabeled_frames'] = unlabeled_frames
            

            # labeled_max_fr = max(labeled_frames)
            # init_params['labeled_max_fr'] = labeled_max_fr
            #all_frames = all_frames + list((labeled_max_fr + np.array(all_frames_unlabeled)))
            # all_frames = labeled_frames + unlabeled_frames
            # #check labeled and unlabeled frame ids are unique
            # assert len(set(all_frames))==len(all_frames)
            # assert len(all_frames)==len(set(labeled_frames)) + len(set(unlabeled_frames))
            all_frames = unlabeled_frames
            assert len(all_frames)== len(set(unlabeled_frames))
            print(f'total frames for training: {len(all_frames)}')
            print(f'labeled frames anns json will read directly during training')
            #pdb.set_trace()

        else:
            all_frames = 'all frames ids'

    return init_params, all_frames