# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:05:11 2023

@author: yaobv
"""

import pandas as pd
from sklearn.metrics import brier_score_loss

folder = './submissions/'

pred_df = pd.read_csv(f'{folder}late_sub_03_28.csv')
results_df = pd.read_csv(f'{folder}results_to_date.csv')

preds_dict = pred_df.set_index('ID')['Pred'].to_dict()
results_df = results_df[results_df['Usage'] == 'Public'].copy()

results_df['my_pred'] = [preds_dict[id_]
                         if id_ in preds_dict else 0.5 for id_ in results_df.ID]


print('my model would have a brier score loss of',
      brier_score_loss(results_df['Pred'], results_df['my_pred']))