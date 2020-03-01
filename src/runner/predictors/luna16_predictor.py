import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F

from src.runner.utils import EpochLog
from src.runner.predictors.base_predictor import BasePredictor

LOGGER = logging.getLogger(__name__.split('.')[-1])


class Luna16Predictor(BasePredictor):
    def __init__(self, candidate_csv_path, submission_csv_path, **kwargs):
        super().__init__(**kwargs)
        self.candidate_df = pd.read_csv(candidate_csv_path)
        self.submission_csv_path = Path(submission_csv_path)
        if not self.submission_csv_path.parent.exists():
            self.submission_csv_path.parent.mkdir(parents=True)
        
        self.seriesuid, self.prob = [], []
        self.coordX, self.coordY, self.coordZ = [], [], []
        
    def predict(self):
        """The testing process.
        """
        super().predict()
        submission_df = pd.DataFrame(data={'seriesuid': self.seriesuid, 'coordX': self.coordX,
                                           'coordY': self.coordY, 'coordZ': self.coordZ, 'probability': self.prob})
        submission_df.to_csv(self.submission_csv_path, index=False)
        
    def _test_step(self, batch):
        data, label = batch['data'].to(self.device), batch['label'].to(self.device)
        cid = batch['cid'].squeeze().numpy()
        
        logits = self.net(data)
        prob = F.softmax(logits, dim=1).squeeze()[:, 1].cpu().numpy()
        cross_entropy_loss = self.loss_fns.cross_entropy_loss(logits, label.squeeze())
        accuracy = self.metric_fns.accuracy(F.softmax(logits, dim=1), label)
        
        candidate_list = np.array(self.candidate_df.loc[cid].values.tolist())
        self.seriesuid.extend(candidate_list[:, 0])
        self.coordX.extend(candidate_list[:, 1])
        self.coordY.extend(candidate_list[:, 2])
        self.coordZ.extend(candidate_list[:, 3])
        self.prob.extend(prob)
        
        return {
            'losses': {
                'cross_entropy_loss': cross_entropy_loss
            },
            'metrics': {
                'accuracy': accuracy
            }
        }