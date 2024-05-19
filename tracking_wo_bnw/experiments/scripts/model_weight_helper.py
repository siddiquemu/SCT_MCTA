import torch
import pickle
import torchvision

FRCNN_ResNet101FPN_weight_file = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/resnet101/boxAP42/model_final_f6e8b1.pkl'

model = torchvision.models.detection.fasterrcnn_resnet101_fpn(pretrained=False)
i=0
for p_name in model.state_dict():
    print(i, p_name)
    i+=1

with open(FRCNN_ResNet101FPN_weight_file, 'rb') as fp:
    src_blobs = pickle.load(fp, encoding='latin1')
i=0
for p_name in src_blobs['model'].keys():
    print(i, p_name)
    i+=1

if 'blobs' in src_blobs:
    src_blobs = src_blobs['blobs']

params = model.state_dict()

for p_name, p_tensor in params.items():
    d_name = name_mapping[p_name]
    if isinstance(d_name, str):  # maybe str, None or True
        p_tensor.copy_(torch.Tensor(src_blobs[d_name]))