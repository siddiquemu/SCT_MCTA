#########################################
# Still ugly file with helper functions #
#########################################

import os
from collections import defaultdict
from os import path as osp

import numpy as np
import torch
from cycler import cycler as cy

import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import motmetrics as mm
import pdb
import glob
matplotlib.use('Agg')

# https://matplotlib.org/cycler/
# get all colors with
# colors = []
#	for name,_ in matplotlib.colors.cnames.items():
#		colors.append(name)

colors = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque',
    'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan',
    'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
    'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
    'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
    'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo',
    'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
    'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
    'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
    'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
    'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
    'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
    'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
    'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
    'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
    'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'
]


# From frcnn/utils/bbox.py
def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = lambda x: x.numpy()  # If input is ndarray, turn the overlaps back to ndarray when return
    else:
        out_fn = lambda x: x

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1],
                                                                        query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2],
                                                                        query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return out_fn(overlaps)


def delete_all(demo_path, fmt='png'):
    try:
        filelist = glob.glob(os.path.join(demo_path, '*'))
        if len(filelist) > 0:
            for f in filelist:
                os.remove(f)
    except:
        print(f'{demo_path} is already empty')

def center_distances(inactive_pos, new_pos):
    #cent: 2, bottom_center: [2,1]
    inactive_cents = inactive_pos[:, 0:2] + (inactive_pos[:, 2:4] - inactive_pos[:, 0:2]) / torch.tensor([2.,1.]).cuda()
    new_cents =new_pos[:, 0:2] + (new_pos[:, 2:4] - new_pos[:, 0:2]) / torch.tensor([2.,1.]).cuda()
    return torch.cdist(inactive_cents, new_cents)

def ReID_search_distances(new_pos, max_dist=300, max_patience=40):
    norm_factor = torch.tensor([1280, 800], dtype=torch.float).cuda()
    fov_cent = torch.tensor([1280 // 2, 800 // 2], dtype=torch.float).cuda()
    fov_cent = fov_cent.reshape(1, 2)

    new_cents = new_pos[:, 0:2] + (new_pos[:, 2:4] - new_pos[:, 0:2]) / torch.tensor([2., 1.]).cuda()
    cdist = torch.cdist(new_cents/norm_factor, fov_cent/norm_factor)
    rads, pats = torch.exp(-2*cdist)*max_dist, torch.floor(torch.exp(-2*cdist)*max_patience)
    return rads, pats

def ReID_search_distances_linear(new_pos, max_dist=300, max_patience=40):
    norm_factor = torch.tensor([1280, 800], dtype=torch.float).cuda()
    fov_cent = torch.tensor([1280 // 2, 800 // 2], dtype=torch.float).cuda()
    fov_cent = fov_cent.reshape(1, 2)

    new_cents = new_pos[:, 0:2] + (new_pos[:, 2:4] - new_pos[:, 0:2]) / torch.tensor([2., 2.]).cuda()
    new_cents_invert = norm_factor - new_cents
    t_poses = torch.cat((new_cents, new_cents_invert), dim=1)
    print('poses from borders: {}'.format(t_poses))

    pos_mins, _ = torch.min(t_poses, 1)
    print('min cxs or cys: {}'.format(pos_mins))
    rads = torch.div(pos_mins, 2.).reshape(pos_mins.shape[0], 1)
    pats = rads
    assert rads.shape[0] == new_cents.shape[0]
    return rads, pats

def filter_tracklets(num_frames, trackers, start_frame=None, batch_length=120,
                     min_trklt_size=30, look_back_batches=3, reid_patience=30):
    # confirmed tracklets with tl.size >= scan_intrvl-reid_patience if tl.end not in
    # trackers has items from the begining of the video
    keep = []
    keep_tracks = {}
    batch_frames = np.arange(num_frames-batch_length+1, num_frames+1)
    look_back_frames = np.arange(num_frames - look_back_batches*batch_length + 1, num_frames+1)
    for j, tl in trackers.items():
        # check if trklt ends in reid_patience
        #keep tracks for at least 1 second life
        #TODO: get tracks only available in current batch but consider the previous batches life if available
        if len(set(np.array(list(tl.keys()))+start_frame).intersection(set(list(batch_frames))))>0 and len(tl.keys())>=min_trklt_size:
            #tracks key might be initiated with empty values: batch results will have only keys but no values
            look_back_batches = {k+start_frame:v for k,v in tl.items() if k+start_frame in look_back_frames}
            #keep track as global key if it is available in look back frames otherwise return empty keep_tracks
            if len(look_back_batches.keys())>0:
                keep_tracks[j] = look_back_batches
    return keep_tracks

def filter_tracklets_app(num_frames, trackers=None, start_frame=None, batch_length=120,
                     min_trklt_size=30, look_back_batches=3, reid_patience=30, track_start_time=None):

    keep_tracks = {'pose':{}, 'appearance':{}}
    batch_frames = np.arange(num_frames - batch_length + 1, num_frames + 1)
    look_back_frames = np.arange(num_frames - look_back_batches*batch_length + 1, num_frames+1)
    tl_del = []
    for j, tl in trackers['pose'].items():
        #print('{}:{}'.format(j, np.array(list(tl.keys()))+start_frame))
        if len(set(np.array(list(tl.keys()))+start_frame).intersection(set(list(batch_frames))))>0 and len(tl.keys())>=min_trklt_size:
            #tracks key might be initiated with empty values: batch results will have only keys but no values
            look_back_batches = {k+start_frame:v for k,v in tl.items() if k+start_frame in look_back_frames}
            #keep track as global key if it is available in look back frames otherwise return empty keep_tracks
            if len(look_back_batches.keys())>0:
                keep_tracks['pose'][j] = look_back_batches
                assert j in trackers['appearance'].keys(), 'pose track {} found no appearance track'.format(j)
                keep_tracks['appearance'][j] = trackers['appearance'][j]
                if track_start_time:
                    track_start_time[j] = min(np.array(list(tl.keys()))+start_frame)
        else:
            tl_del.append(j)
    return keep_tracks, track_start_time, tl_del

def plot_scanned_track(num_frames, tracks, im_path, output_dir, im_wh=None,
                       cam=None, reid_map=None, reid_rads=None):
    """Plots a whole sequence

    Args:
        tracks (dict): The dictionary containing the track dictionaries in the form tracks[track_id][frame] = bb
        db (torch.utils.data.Dataset): The dataset with the images belonging to the tracks (e.g. MOT_Sequence object)
        output_dir (String): Directory where to save the resulting images
    """

    print("[*] Plotting scanned tracks at {}".format(osp.basename(im_path)))

    # infinite color loop
    cyl = cy('ec', colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    im_name = osp.basename(im_path)
    assert cam != None
    im_output = osp.join(output_dir, im_name)
    im = cv2.imread(im_path)
    if im_wh is not None and im_wh[0]!=im.shape[0]:
        im = cv2.resize(im, (im_wh))
    im = im[:, :, (2, 1, 0)]

    sizes = np.shape(im)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / 100, height / 100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(im)

    for j, t in tracks.items():
        #pdb.set_trace()
        #if (j in [5,6] and cam==1) or (j in [31,34,128,132,153,180] and cam==3):
        #if j in [106,121,159,190,198]:
        if num_frames-1 in t.keys():
            if reid_map:
                if j in reid_map.keys():
                    j=reid_map[j]
            t_i = t[num_frames-1]
            center = (t_i[0] + (t_i[2] - t_i[0]) / 2.0, t_i[1] + (t_i[3] - t_i[1]) / 2.0)
            ax.add_patch(
                plt.Rectangle(
                    (t_i[0], t_i[1]),
                    t_i[2] - t_i[0],
                    t_i[3] - t_i[1],
                    fill=False, edgecolor='blue',
                    linewidth=3.0, alpha=0.8
                ))
            if reid_rads and j in reid_rads:
                #TODO: add color id
                ax.add_patch(plt.Circle(center, radius=reid_rads[j][0], color='blue', fill=True, alpha=0.2))
                plt.text(center[0]-55, center[1]+25, 'patience: {}'.format(int(reid_rads[j][1])),
                         bbox=dict(fill=False, edgecolor='red', linewidth=2), color='white')

            ax.annotate(j, center,
                    color='magenta', weight='bold', fontsize=20, ha='center', va='center')

    plt.axis('off')
    # plt.tight_layout()
    #DefaultSize = plt.get_size_inches()
    #plt.set_figsize_inches((DefaultSize[0] // 2, DefaultSize[1] // 2))
    plt.draw()
    plt.savefig(im_output, dpi=100)
    plt.close()

def plot_dets(num_frames, all_dets, im_path, output_dir, im_wh=None, cam=None, labels=None, dataset='PVD'):
    """Plots a whole sequence

    Args:
        tracks (dict): The dictionary containing the track dictionaries in the form tracks[track_id][frame] = bb
        db (torch.utils.data.Dataset): The dataset with the images belonging to the tracks (e.g. MOT_Sequence object)
        output_dir (String): Directory where to save the resulting images
    """
    if dataset in ['PVD']:
        category = {0: 'background', 1: 'PAX', 2: 'TSO'}
    else:
        category = {0: 'background', 1: 'Pax', 2: 'Bag'}


    print("[*] Plotting dets at {}".format(osp.basename(im_path)))

    # infinite color loop
    cyl = cy('ec', colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    im_name = osp.basename(im_path)
    assert cam != None
    im_output = osp.join(output_dir, im_name)
    im = cv2.imread(im_path)
    if im_wh is not None and im_wh[0]!=im.shape[0]:
        im = cv2.resize(im, (im_wh))
    im = im[:, :, (2, 1, 0)]

    sizes = np.shape(im)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / 100, height / 100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(im)
    all_dets = all_dets.cpu().numpy()
    for i, t_i in enumerate(all_dets):
        ax.add_patch(
            plt.Rectangle(
                (t_i[0], t_i[1]),
                t_i[2] - t_i[0],
                t_i[3] - t_i[1],
                fill=False, edgecolor='green',
                linewidth=2.0, alpha=0.8
            ))
        if labels:
            label_text = '{}: {}'.format(category[labels], round(float(t_i[4]),2))
        else:
            label_text = '{}: {}'.format(category[int(t_i[-1])], round(float(t_i[4]), 2))
        ax.annotate(label_text, (t_i[0] + (t_i[2] - t_i[0]) / 2.0, t_i[1] + (t_i[3] - t_i[1]) / 2.0),
                    color='red', fontsize=20, ha='center', va='center')

    plt.axis('off')
    # plt.tight_layout()
    #DefaultSize = plt.get_size_inches()
    #plt.set_figsize_inches((DefaultSize[0] // 2, DefaultSize[1] // 2))
    plt.draw()
    plt.savefig(im_output, dpi=100)
    plt.close()

def plot_sequence(tracks, db, output_dir):
    """Plots a whole sequence

    Args:
        tracks (dict): The dictionary containing the track dictionaries in the form tracks[track_id][frame] = bb
        db (torch.utils.data.Dataset): The dataset with the images belonging to the tracks (e.g. MOT_Sequence object)
        output_dir (String): Directory where to save the resulting images
    """

    print("[*] Plotting whole sequence to {}".format(output_dir))
    delete_all(demo_path=output_dir)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    # infinite color loop
    cyl = cy('ec', colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    for i, v in enumerate(db):
        if i%30==0:
            im_path = v['img_path']
            im_name = osp.basename(im_path)
            im_output = osp.join(output_dir, im_name)
            im = cv2.imread(im_path)
            im = im[:, :, (2, 1, 0)]

            sizes = np.shape(im)
            height = float(sizes[0])
            width = float(sizes[1])

            fig = plt.figure()
            fig.set_size_inches(width / 100, height / 100)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(im)

            for j, t in tracks.items():
                #pdb.set_trace()
                if i in t.keys():
                    t_i = t[i]
                    ax.add_patch(
                        plt.Rectangle(
                            (t_i[0], t_i[1]),
                            t_i[2] - t_i[0],
                            t_i[3] - t_i[1],
                            fill=False,
                            linewidth=2.0, **styles[j]
                        ))

                    ax.annotate(j, (t_i[0] + (t_i[2] - t_i[0]) / 2.0, t_i[1] + (t_i[3] - t_i[1]) / 2.0),
                                color=styles[j]['ec'], weight='bold', fontsize=20, ha='center', va='center')

            plt.axis('off')
            # plt.tight_layout()
            #DefaultSize = plt.get_size_inches()
            #plt.set_figsize_inches((DefaultSize[0] // 2, DefaultSize[1] // 2))
            plt.draw()
            plt.savefig(im_output, dpi=100)
            plt.close()


def plot_tracks(blobs, tracks, gt_tracks=None, output_dir=None, name=None):
    # output_dir = get_output_dir("anchor_gt_demo")
    im_paths = blobs['im_paths']
    if not name:
        im0_name = osp.basename(im_paths[0])
    else:
        im0_name = str(name) + ".jpg"
    im0 = cv2.imread(im_paths[0])
    im1 = cv2.imread(im_paths[1])
    im0 = im0[:, :, (2, 1, 0)]
    im1 = im1[:, :, (2, 1, 0)]

    im_scales = blobs['im_info'][0, 2]

    tracks = tracks.data.cpu().numpy() / im_scales
    num_tracks = tracks.shape[0]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(im0, aspect='equal')
    ax[1].imshow(im1, aspect='equal')

    # infinte color loop
    cyl = cy('ec', colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    ax[0].set_title(('{} tracks').format(num_tracks), fontsize=14)

    for i, t in enumerate(tracks):
        t0 = t[0]
        t1 = t[1]
        ax[0].add_patch(
            plt.Rectangle(
                (t0[0], t0[1]),
                t0[2] - t0[0],
                t0[3] - t0[1], fill=False,
                linewidth=1.0, **styles[i]
            ))

        ax[1].add_patch(
            plt.Rectangle(
                (t1[0], t1[1]),
                t1[2] - t1[0],
                t1[3] - t1[1], fill=False,
                linewidth=1.0, **styles[i]
            ))

    if gt_tracks:
        for gt in gt_tracks:
            for i in range(2):
                ax[i].add_patch(
                    plt.Rectangle(
                        (gt[i][0], gt[i][1]),
                        gt[i][2] - gt[i][0],
                        gt[i][3] - gt[i][1], fill=False,
                        edgecolor='blue', linewidth=1.0
                    ))

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    image = None
    if output_dir:
        im_output = osp.join(output_dir, im0_name)
        plt.savefig(im_output)
    else:
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image


def interpolate(tracks):
    interpolated = {}
    for i, track in tracks.items():
        interpolated[i] = {}
        frames = []
        x0 = []
        y0 = []
        x1 = []
        y1 = []

        for f, bb in track.items():
            frames.append(f)
            x0.append(bb[0])
            y0.append(bb[1])
            x1.append(bb[2])
            y1.append(bb[3])

        if len(frames) > 1:
            x0_inter = interp1d(frames, x0)
            y0_inter = interp1d(frames, y0)
            x1_inter = interp1d(frames, x1)
            y1_inter = interp1d(frames, y1)

            for f in range(min(frames), max(frames) + 1):
                bb = np.array([x0_inter(f), y0_inter(f), x1_inter(f), y1_inter(f)])
                interpolated[i][f] = bb
        else:
            interpolated[i][frames[0]] = np.array([x0[0], y0[0], x1[0], y1[0]])

    return interpolated


def bbox_transform_inv(boxes, deltas):
    # Input should be both tensor or both Variable and on the same device
    if len(boxes) == 0:
        return deltas.detach() * 0

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    pred_boxes = torch.cat(
        [_.unsqueeze(2) for _ in [pred_ctr_x - 0.5 * pred_w,
                                pred_ctr_y - 0.5 * pred_h,
                                pred_ctr_x + 0.5 * pred_w,
                                pred_ctr_y + 0.5 * pred_h]], 2).view(len(boxes), -1)
    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    boxes must be tensor or Variable, im_shape can be anything but Variable
    """
    if not hasattr(boxes, 'data'):
        boxes_ = boxes.numpy()

    boxes = boxes.view(boxes.size(0), -1, 4)
    boxes = torch.stack([
        boxes[:, :, 0].clamp(0, im_shape[1] - 1),
        boxes[:, :, 1].clamp(0, im_shape[0] - 1),
        boxes[:, :, 2].clamp(0, im_shape[1] - 1),
        boxes[:, :, 3].clamp(0, im_shape[0] - 1)
    ], 2).view(boxes.size(0), -1)

    return boxes


def get_center(pos):
    x1 = pos[0, 0]
    y1 = pos[0, 1]
    x2 = pos[0, 2]
    y2 = pos[0, 3]
    return torch.Tensor([(x2 + x1) / 2, (y2 + y1) / 2]).cuda()


def get_width(pos):
    return pos[0, 2] - pos[0, 0]


def get_height(pos):
    return pos[0, 3] - pos[0, 1]


def make_pos(cx, cy, width, height):
    return torch.Tensor([[
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2
    ]]).cuda()


def warp_pos(pos, warp_matrix):
    p1 = torch.Tensor([pos[0, 0], pos[0, 1], 1]).view(3, 1)
    p2 = torch.Tensor([pos[0, 2], pos[0, 3], 1]).view(3, 1)
    p1_n = torch.mm(warp_matrix, p1).view(1, 2)
    p2_n = torch.mm(warp_matrix, p2).view(1, 2)
    return torch.cat((p1_n, p2_n), 1).view(1, -1).cuda()


def get_mot_accum(results, seq):
    mot_accum = mm.MOTAccumulator(auto_id=True)    

    for i, data in enumerate(seq):
        gt = data['gt']
        gt_ids = []
        if gt:
            gt_boxes = []
            for gt_id, box in gt.items():
                gt_ids.append(gt_id)
                gt_boxes.append(box)
        
            gt_boxes = np.stack(gt_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            gt_boxes = np.stack((gt_boxes[:, 0],
                                 gt_boxes[:, 1],
                                 gt_boxes[:, 2] - gt_boxes[:, 0],
                                 gt_boxes[:, 3] - gt_boxes[:, 1]),
                                axis=1)
        else:
            gt_boxes = np.array([])
        
        track_ids = []
        track_boxes = []
        for track_id, frames in results.items():
            if i in frames:
                track_ids.append(track_id)
                # frames = x1, y1, x2, y2, score
                track_boxes.append(frames[i][:4])

        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            track_boxes = np.stack((track_boxes[:, 0],
                                    track_boxes[:, 1],
                                    track_boxes[:, 2] - track_boxes[:, 0],
                                    track_boxes[:, 3] - track_boxes[:, 1]),
                                    axis=1)
        else:
            track_boxes = np.array([])
        
        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)
        
        mot_accum.update(
            gt_ids,
            track_ids,
            distance)

    return mot_accum

    
def evaluate_mot_accums(accums, names, generate_overall=False):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums, 
        metrics=mm.metrics.motchallenge_metrics, 
        names=names,
        generate_overall=generate_overall,)

    str_summary = mm.io.render_summary(
        summary, 
        formatters=mh.formatters, 
        namemap=mm.io.motchallenge_metric_names,)
    print(str_summary)