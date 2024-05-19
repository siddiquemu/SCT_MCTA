import glob
import pandas as pd
import os

generateOverlay = 0
# '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/GlobalScoringTool/GT'
# gt_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/CLASP/GTOutputs'
gt_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/logan-data/exp1_logan/GTA'
# gt_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/GlobalScoringTool/GT'
# gt_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/CLASP/10fps_gt/clasp_format'
cams = glob.glob(gt_path + '/*')

# cams.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f))))
print(cams)
gt = {}
id_count = 0
id_unique = []
mapped_id = {}
for file in os.listdir(gt_path):
    print(file)
    gt[file] = []
    for entry in open(gt_path + "/" + file, 'r').read().split('\n'):
        d = entry.split(' ')

        if len(entry) > 0:
            if d[0] == '#' and (d[2].isdigit()) and ("total" not in entry):
                continue

        try:
            idx = d.index("frame:") + 1
            frame = d[idx].zfill(5)

            if generateOverlay:
                fileNameParts = ((file.split("."))[0]).split("-")
                frameset = RemoveLeadingZeros(fileNameParts[0])
                jpgFrame = str(int(frame) + 1)
                srcJPG = str(pathlib.Path().absolute()) + '/' + frameset + '/' + jpgFrame.zfill(5) + '.jpg'
                destJPG = str(pathlib.Path().absolute()) + '/Overlay_' + frameset + '/' + jpgFrame.zfill(5) + '.jpg'
                destPath = str(pathlib.Path().absolute()) + '/Overlay_' + frameset
                if not (os.path.isfile(destJPG)):
                    shutil.copy2(srcJPG, destPath)

            print(d[0])
            if d[0] == "LOC:":  # and d[d.index("type:") + 1] in ("TSO", "DVI", "PAX")
                temp_dict = {"Entry-Type": d[0], "Type": d[d.index("type:") + 1],
                             "Camera-Num": d[d.index("camera-num:") + 1],
                             "Frame": d[d.index("frame:") + 1], "Time-Offset": d[d.index("time-offset:") + 1],
                             "BB": d[d.index("BB:") + 1: d.index("BB:") + 5], "ID": d[d.index("ID:") + 1],
                             "Partial-Complete": d[d.index("partial-complete:") + 1],
                             "Description": ' '.join(d[d.index("description:") + 1: len(d)])}

                # For tracking gt TSO and PAX should have different label at same time
                # For detection gt, id does not matter
                print(temp_dict)
                if d[d.index("type:") + 1] in ["TSO", "PAX"]:
                    # print(d[d.index("ID:") + 1])
                    # print(temp_dict["BB"])
                    # save as gt file using format:[frame,id,x,y,w,h]
                    fr = float(temp_dict["Frame"])
                    x = float(temp_dict["BB"][0].split(',')[0])
                    y = float(temp_dict["BB"][1].split(',')[0])
                    w = float(temp_dict["BB"][2].split(',')[0]) - x
                    h = float(temp_dict["BB"][3]) - y
                    score = 1
                    classID = 1
                    visibility = 1
                    if temp_dict["ID"] not in id_unique:
                        id_count += 1
                        mapped_id[temp_dict["ID"]] = id_count
                        id_unique.append(temp_dict["ID"])

                    gt[file].append([fr, mapped_id[temp_dict["ID"]], x, y, w, h, score, classID, visibility])
                    print('Frame {}: label {} for {}'.format(fr, mapped_id[temp_dict["ID"]], d[d.index("ID:") + 1]))


        except:
            continue

import csv
#gt_path_save = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/CLASP/10fps_gt/mot_format/'
print(gt.keys())
print(gt['camBeltStartexp1-logfile.txt'])
gt_path_save = gt_path+'/MOT_format/'
if not os.path.exists(gt_path_save):
    os.makedirs(gt_path_save)
for k, v in gt.items():
    with open(gt_path_save+k, "w") as of:
        writer = csv.writer(of, delimiter=',')
        for d in v:
            writer.writerow(d)