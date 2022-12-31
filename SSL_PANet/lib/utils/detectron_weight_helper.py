"""Helper functions for loading pretrained weights from Detectron pickle files
"""

import pickle
import re
import torch


def load_detectron_weight(net, detectron_weight_file, isTrain):
    name_mapping, orphan_in_detectron = net.detectron_weight_mapping

    with open(detectron_weight_file, 'rb') as fp:
        src_blobs = pickle.load(fp, encoding='latin1')
    if 'blobs' in src_blobs:
        src_blobs = src_blobs['blobs']

    params = net.state_dict()
    # TODO: delete the loop for displaying the layers name
    i=0
    for p_name, p_tensor in params.items():
        print(i,p_name)

        i+=1
        '''
        if isTrain and p_name not in ['Box_Outs.cls_score.weight', 'Box_Outs.cls_score.bias', 'Box_Outs.bbox_pred.weight', 'Box_Outs.bbox_pred.bias', 'Mask_Outs.classify.weight', 'Mask_Outs.classify.bias']:
            d_name = name_mapping[p_name]
            if isinstance(d_name, str):  # maybe str, None or True
                p_tensor.copy_(torch.Tensor(src_blobs[d_name]))
        '''
    i = 0
    rest = set(params.keys()) - set(name_mapping.keys())
    print(rest)
    print(src_blobs.keys())
    for p_name, p_tensor in params.items():
        # last layers weights are ignored since we are training using different number of classes
        d_name = name_mapping[p_name]
        print(i,d_name)
        if isinstance(d_name, str):  # maybe str, None or True, d_name not in rest
            print(p_name)
            p_tensor.copy_(torch.Tensor(src_blobs[d_name]))
        i+=1

def load_detectron_weight_org(net, detectron_weight_file):
    name_mapping, orphan_in_detectron = net.detectron_weight_mapping

    with open(detectron_weight_file, 'rb') as fp:
        src_blobs = pickle.load(fp, encoding='latin1')
    if 'blobs' in src_blobs:
        src_blobs = src_blobs['blobs']

    params = net.state_dict()
    for p_name, p_tensor in params.items():
        d_name = name_mapping[p_name]
        if isinstance(d_name, str):  # maybe str, None or True
            p_tensor.copy_(torch.Tensor(src_blobs[d_name]))

def resnet_weights_name_pattern():
    pattern = re.compile(r"conv1_w|conv1_gn_[sb]|res_conv1_.+|res\d+_\d+_.+")
    return pattern


if __name__ == '__main__':
    """Testing"""
    from pprint import pprint
    import sys
    sys.path.insert(0, '..')
    from modeling.model_builder import Generalized_RCNN
    from core.config import cfg, cfg_from_file

    cfg.MODEL.NUM_CLASSES = 81
    cfg_from_file('../../cfgs/res50_mask.yml')
    net = Generalized_RCNN()

    # pprint(list(net.state_dict().keys()), width=1)

    mapping, orphans = net.detectron_weight_mapping
    state_dict = net.state_dict()

    for k in mapping.keys():
        assert k in state_dict, '%s' % k

    rest = set(state_dict.keys()) - set(mapping.keys())
    assert len(rest) == 0
