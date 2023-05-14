from time import time
import numpy as np
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
import os
from recbole.trainer import Trainer
from recbole.utils import early_stopping, dict2str, set_color, get_gpu_usage

class GCFMCLTrainer(Trainer):

    def __init__(self, config, model):
        super(GCFMCLTrainer, self).__init__(config, model)

        self.num_m_step = config['m_step']
        assert self.num_m_step is not None

    def fit(self, train_data, valid_data=None,dataset=None):
        self.eval_collector.data_collect(train_data)

        for epoch_idx in range(self.start_epoch, self.epochs):

            if epoch_idx % self.num_m_step == 0:
                self.model.e_step()
            train_loss = self._train_epoch(train_data)
        
        embedding1, embedding2 = self.model.predict(dataset)
        np.savetxt(fname='./mlp/data/embedding1.txt', X=embedding1, newline='\n', encoding='UTF-8')
        np.savetxt(fname='./mlp/data/embedding2.txt', X=embedding2, newline='\n', encoding='UTF-8')
        os.system('python ./mlp/test.py')
    def _train_epoch(self, train_data):
        self.model.train()
        loss_func =self.model.calculate_loss
        total_loss = None
        for batch_idx, interaction in enumerate(train_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
        return total_loss
