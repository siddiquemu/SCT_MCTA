from torch.utils.data import Dataset
import torch

from .mot_reid import MOTreID


class MOTreIDWrapper(Dataset):
	"""A Wrapper class for MOTSiamese.

	Wrapper class for combining different sequences into one dataset for the MOTreID
	Dataset.
	"""

	def __init__(self, split, dataloader):

		train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
				         'MOT17-11', 'MOT17-13']

		self._dataloader = MOTreID(None, split=split, **dataloader)

		for seq in train_folders:
			d = MOTreID(seq, split=split, **dataloader)
			for sample in d.data:
				self._dataloader.data.append(sample)

	def __len__(self):
		return len(self._dataloader.data)

	def __getitem__(self, idx):
		return self._dataloader[idx]

class CLASPreIDWrapper(Dataset):
	"""A Wrapper class for MOTSiamese.

	Wrapper class for combining different sequences into one dataset for the CLASPreID
	Dataset.
	"""

	def __init__(self, split, dataloader):
		#split='train_reid'
		#clasp2
		#train_folders = ['cam02exp2.mp4', 'cam09exp2.mp4', 'cam05exp2.mp4','cam11exp2.mp4','cam13exp2.mp4', 'cam14exp2.mp4']
		#frame_offset = {'cam02exp2.mp4': 5, 'cam05exp2.mp4': 10, 'cam09exp2.mp4': 10,
							 #'cam11exp2.mp4': 15, 'cam13exp2.mp4': 10, 'cam14exp2.mp4': 18}
		#train_folders = ['G_9', 'G_11', 'H_9', 'H_11', 'I_9', 'I_11']
		#frame_offset = {'G_9':0, 'G_11':0, 'H_9':0, 'H_11':0, 'I_9':0, 'I_11':0}

		#clasp1
		#train_folders = ['A_9', 'A_11', 'B_9', 'B_11', 'C_9', 'C_11', 'D_9', 'D_11', 'E_9', 'E_11']
		#frame_offset = {'A_9':0, 'A_11':0, 'B_9':0, 'B_11':0, 'C_9':0, 'C_11':0, 'D_9':0, 'D_11':0, 'E_9':0, 'E_11':0}

		#PVD
		train_folders = ['C330', 'C360']# 'C2','C3', 'C4','C5', 'C6']
		frame_offset = {'C330':0, 'C360':0} #,'C3':0, 'C4':0,'C5':0 ,'C6':0}

		self._dataloader = MOTreID(None, None, split=split, **dataloader)

		# loop iver all the monocular tracks
		for seq in train_folders:
			d = MOTreID(seq, frame_offset[seq], split=split, **dataloader)
			for sample in d.data:
				self._dataloader.data.append(sample)

	def __len__(self):
		return len(self._dataloader.data)

	def __getitem__(self, idx):
		return self._dataloader[idx]

