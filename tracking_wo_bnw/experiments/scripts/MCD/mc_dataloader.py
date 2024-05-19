import torch
class MulticamObjDetect(torch.utils.data.Dataset):  # inherit torch dataset class
    """ Data class for the Multi-camera datsets
    """

    def __init__(self, root, transforms=None, vis_threshold=0.0):
        self.root = root
        self.transforms = transforms
        self._vis_threshold = vis_threshold
        self._classes = ('background', 'pedestrian')
        self._img_paths = []

        for f in os.listdir(root):
            path = os.path.join(root, f)
            config_file = os.path.join(path, 'seqinfo.ini')
            GT = np.loadtxt(os.path.join(path, 'gt/gt.txt'), delimiter=',')
            # gt_frameset = GT[:,0][GT[:,6]>=0.9].astype('int')+5
            # print(gt_frameset)
            assert os.path.exists(config_file), \
                'Path does not exist: {}'.format(config_file)

            config = configparser.ConfigParser()
            config.read(config_file)
            seq_len = int(config['Sequence']['seqLength'])
            im_width = int(config['Sequence']['imWidth'])
            im_height = int(config['Sequence']['imHeight'])
            im_ext = config['Sequence']['imExt']
            im_dir = config['Sequence']['imDir']

            _imDir = os.path.join(path, im_dir)

            for i in range(0, seq_len):  # 0 start for wild-track

                if i % 5 == 0:  # logan i%3
                    img_path = os.path.join(_imDir, f"{i:08d}{im_ext}")  # i for wild-track
                    if not os.path.exists(img_path):
                        continue
                    assert os.path.exists(img_path), \
                        'Path does not exist: {img_path}'
                    # self._img_paths.append((img_path, im_width, im_height))
                    # save image path if image has detection (some frames in the video may have no detection)
                    self._img_paths.append(img_path)
                    print(img_path)

    @property
    def num_classes(self):
        return len(self._classes)

    def _get_annotation(self, idx):
        """
        """

        if 'test' in self.root:
            num_objs = 0
            boxes = torch.zeros((num_objs, 4), dtype=torch.float32)

            return {'boxes': boxes,
                    'labels': torch.ones((num_objs,), dtype=torch.int64),
                    'image_id': torch.tensor([idx]),
                    'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                    'iscrowd': torch.zeros((num_objs,), dtype=torch.int64),
                    'visibilities': torch.zeros((num_objs), dtype=torch.float32)}

        img_path = self._img_paths[idx]
        file_index = int(os.path.basename(img_path).split('.')[0])

        gt_file = os.path.join(os.path.dirname(
            os.path.dirname(img_path)), 'gt', 'gt.txt')

        assert os.path.exists(gt_file), \
            'GT file does not exist: {}'.format(gt_file)

        bounding_boxes = []
        # print(gt_file)
        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=',')
            for row in reader:
                # print(row)
                visibility = float(row[8])
                if int(float(row[0])) == file_index:
                    bb = {}
                    # print(row)
                    bb['bb_left'] = int(float(row[2]))
                    bb['bb_top'] = int(float(row[3]))
                    bb['bb_width'] = int(float(row[4]))
                    bb['bb_height'] = int(float(row[5]))
                    bb['visibility'] = 1.0  # float(row[8])

                    bounding_boxes.append(bb)

        num_objs = len(bounding_boxes)

        boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
        visibilities = torch.zeros((num_objs), dtype=torch.float32)

        for i, bb in enumerate(bounding_boxes):
            # Make pixel indexes 0-based, should already be 0-based (or not)
            x1 = bb['bb_left'] - 1
            y1 = bb['bb_top'] - 1
            # This -1 accounts for the width (width of 1 x1=x2)
            x2 = x1 + bb['bb_width'] - 1
            y2 = y1 + bb['bb_height'] - 1

            boxes[i, 0] = x1
            boxes[i, 1] = y1
            boxes[i, 2] = x2
            boxes[i, 3] = y2
            visibilities[i] = bb['visibility']

        return {'boxes': boxes,
                'labels': torch.ones((num_objs,), dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': torch.zeros((num_objs,), dtype=torch.int64),
                'visibilities': visibilities, }

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self._img_paths[idx]

        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        target = self._get_annotation(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self._img_paths)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import transforms as T

    data_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/tracking_wo_bnw/data/wild-track'
    dataset = MulticamObjDetect(data_path + '/train_gt')


    def plot(img, boxes):
        fig, ax = plt.subplots(1, dpi=96)

        img = img.mul(255).permute(1, 2, 0).byte().numpy()
        width, height, _ = img.shape

        ax.imshow(img, cmap='gray')
        fig.set_size_inches(width / 80, height / 80)

        for box in boxes:
            rect = plt.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                fill=False,
                linewidth=1.0)
            ax.add_patch(rect)
    #visualize
    for i in range(50, 100):
        img, target = dataset[i]
        img, target = T.ToTensor()(img, target)
        plot(img, target['boxes'])

        plt.axis('off')
        plt.show()
