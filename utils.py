# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:42:39 2023

@author: yaobv
"""

import pandas as pd
from sklearn.linear_model import RidgeCV


def get_season_pace(season=2023, df=None):

    winners = df[['Season', 'WTeamID', 'LTeamID', 'WLoc', 'WPoss']].copy()
    winners.columns = ['Season', 'OffTeamID', 'DefTeamID', 'Home', 'Poss']

    losers = df[['Season', 'LTeamID', 'WTeamID', 'WLoc', 'LPoss']].copy()
    losers.columns = ['Season', 'OffTeamID', 'DefTeamID', 'Home', 'Poss']
    losers['Home'] = losers['Home'].map({'H': 'A', 'N': 'N', 'A': 'H'})

    pacedf = pd.concat([winners, losers], axis=0).reset_index(drop=True)

    pace_22 = pacedf[pacedf['Season'] == season].copy()
    pace_22.drop(columns=['Season'], inplace=True)

    pace_22 = pd.get_dummies(data=pace_22, columns=[
                             'OffTeamID', 'DefTeamID', 'Home'])

    pace_model_cols = [x for x in pace_22.columns if 'Poss' not in x]
    target = 'Poss'

    X = pace_22[pace_model_cols].copy()
    y = pace_22[target]

    model = RidgeCV(alphas=[2.3, 2.5, 2.7])
    model.fit(X=X, y=y)

    coefs_pace = pd.DataFrame(
        {'col': pace_model_cols, 'coef': model.coef_.round(1)})

    off_pace = coefs_pace[coefs_pace['col'].str.contains('Off')].copy()
    off_pace['teamid'] = [int(x.split('_')[1]) for x in off_pace.col]

    def_pace = coefs_pace[coefs_pace['col'].str.contains('Def')].copy()
    def_pace['teamid'] = [int(x.split('_')[1]) for x in def_pace.col]

    off_pace.set_index('teamid', inplace=True)
    def_pace.set_index('teamid', inplace=True)

    final_pace = off_pace.join(def_pace, rsuffix='_def')

    final_pace['pace'] = (final_pace['coef'] + final_pace['coef_def']) / 2
    final_pace['season'] = season

    return final_pace[['pace', 'season']].copy()


def get_season_eff(season=2022, df=None, teams_dict=None):

    winners = df[['Season', 'WTeamID', 'LTeamID', 'WLoc', 'WPer100']].copy()
    winners.columns = ['Season', 'OffTeamID', 'DefTeamID', 'Home', 'OPer100']

    losers = df[['Season', 'LTeamID', 'WTeamID', 'WLoc', 'LPer100']].copy()
    losers.columns = ['Season', 'OffTeamID', 'DefTeamID', 'Home', 'OPer100']
    losers['Home'] = losers['Home'].map({'H': 'A', 'N': 'N', 'A': 'H'})

    off_eff_ridge = pd.concat([winners, losers], axis=0).reset_index(drop=True)

    offeff_22 = off_eff_ridge[off_eff_ridge['Season'] == season].copy()
    offeff_22.drop(columns=['Season'], inplace=True)

    offeff_22 = pd.get_dummies(data=offeff_22, columns=[
                               'OffTeamID', 'DefTeamID', 'Home'])

    # building the model
    model_cols = [x for x in offeff_22.columns if 'OPer' not in x]
    target = 'OPer100'

    X = offeff_22[model_cols].copy()
    y = offeff_22[target]

    model = RidgeCV(alphas=[0.8, 1, 1.2])

    model.fit(X=X, y=y)

    # creating the tables
    coefs_df = pd.DataFrame({'col': model_cols, 'coef': model.coef_.round(2)})

    off_coefs = coefs_df[coefs_df['col'].str.contains('Off')].copy()
    off_coefs['teamid'] = [int(x.split('_')[1]) for x in off_coefs.col]

    def_coefs = coefs_df[coefs_df['col'].str.contains('Def')].copy()
    def_coefs['teamid'] = [int(x.split('_')[1]) for x in def_coefs.col]

    off_coefs.set_index('teamid', inplace=True)
    def_coefs.set_index('teamid', inplace=True)

    total_df = off_coefs.join(def_coefs, rsuffix='_def')
    total_df['team'] = [teams_dict[t] for t in total_df.index]
    total_df = total_df[['coef', 'coef_def']].copy()

    total_df['season'] = season

    return total_df

def elo_predict(elo_a=1500, elo_b=1500):
    
    prob_a = 1 / (1 + 10 ** ((elo_b - elo_a)/400))
    
    return prob_a


class GameElo:

    def __init__(self, df: pd.DataFrame, base_k: int = 42):
        self.df = df
        self.names = set(
            pd.concat([self.df['WTeamID'], self.df['LTeamID']]))
        self.winners = self.df.WTeamID
        self.losers = self.df.LTeamID
        self.elo_dict = {name: 1500 for name in self.names}
        self.base_k = base_k
        self.processed = False
        self.winner_elo = []
        self.loser_elo = []
        self.winner_probs = []

    def update_elo(self, winner='Rafael Nadal', loser='Pete Sampras'):

        prematch_winner_elo = self.elo_dict[winner]
        prematch_loser_elo = self.elo_dict[loser]

        exp_a = 1 / \
            (1 + 10 ** ((prematch_loser_elo - prematch_winner_elo)/400))
        exp_b = 1 - exp_a

        winner_delta = self.base_k * (1 - exp_a)
        loser_delta = self.base_k * (0 - exp_b)

        self.elo_dict[winner] = prematch_winner_elo + winner_delta
        self.elo_dict[loser] = prematch_loser_elo + loser_delta

        return exp_a, prematch_winner_elo, prematch_loser_elo

    def process_elo(self):

        if self.processed:
            return "elo already processed"

        else:
            for w, l in zip(self.winners, self.losers):
                win_prob_, pm_w, pm_l = self.update_elo(winner=w, loser=l)

                self.winner_elo.append(pm_w)
                self.loser_elo.append(pm_l)
                self.winner_probs.append(win_prob_)

            self.processed = True
            print("elo ratings processed successfully")
            return self
        
def get_season_four_factors(df=None):
    
    win_cols = [x for x in df.columns if x.startswith('W')]
    win_cols.append('Season')
    win_cols.append('DayNum')
    lose_cols = [x for x in df.columns if x.startswith('L')]
    lose_cols.append('Season')
    lose_cols.append('DayNum')

    other_cols = ['Season', 'DayNum']

    win_det = df[win_cols].copy()
    lose_det = df[lose_cols].copy()
    win_det.columns = [x[1:] if x not in other_cols else x for x in win_det.columns]
    lose_det.columns = [x[1:] if x not in other_cols else x for x in lose_det.columns]

    season_stats = (
        pd.concat([win_det, lose_det])
        .sort_values(by=['Season', 'DayNum'])
        .reset_index(drop=True)
        .drop(columns='Loc')
    )

    season_sums = (
        season_stats
        .groupby(['Season', 'TeamID'])
        .sum()
        .reset_index()
        .drop(columns=['DayNum'])
    )

    season_sums['3PRate'] = season_sums['FGA3'].div(season_sums['FGA']).round(3)
    season_sums['FGA2'] = season_sums['FGA'].sub(season_sums['FGA3'])
    season_sums['FGM2'] = season_sums['FGM'].sub(season_sums['FGM3'])
    season_sums['eFG'] = (season_sums['FGM2'] + (1.5 * season_sums['FGM3']))/ season_sums['FGA']
    season_sums['3Perc'] = season_sums['FGM3'].div(season_sums['FGA3']).round(3)
    season_sums['2Perc'] = season_sums['FGM2'].div(season_sums['FGA2']).round(3)
    season_sums['FTr'] = season_sums['FTA'].div(season_sums['FGA'])
    season_sums['TOVperc'] = season_sums['TO'] / (season_sums['FGA'] + season_sums['FTA'] * 0.44 + season_sums['TO'])
    
    # calculating the season rate stats for each team
    season_rates = season_sums[['Season', 'TeamID', '3PRate', 'FTr', 'eFG', '3Perc', '2Perc', 'TOVperc']].round(3).copy()
    season_rates.columns = season_rates.columns.str.lower()
    
    return season_rates

class NCAAELO:

    def __init__(self, df: pd.DataFrame, base_k: int = 49, home_adv: int = 105):
        self.df = df
        self.teams = set(
            pd.concat([self.df['WTeamID'], self.df['LTeamID']]))
        self.winners = self.df.WTeamID
        self.losers = self.df.LTeamID
        self.seasons = self.df.Season
        self.win_loc = self.df.WLoc
        self.elo_dict = {name: 1500 for name in self.teams}
        self.base_k = base_k
        self.home_advantage = home_adv
        self.processed = False
        self.winner_elo = []
        self.loser_elo = []
        self.winner_probs = []

    def update_elo(self, winner=None, loser=None, win_loc=None):

        if win_loc == 'H':
            adv = self.home_advantage
        elif win_loc == 'A':
            adv = -self.home_advantage
        else:
            adv = 0

        prematch_winner_elo = self.elo_dict[winner] + adv
        prematch_loser_elo = self.elo_dict[loser]

        exp_a = 1 / \
            (1 + 10 ** ((prematch_loser_elo - prematch_winner_elo)/400))
        exp_b = 1 - exp_a

        winner_delta = self.base_k * (1 - exp_a)
        loser_delta = self.base_k * (0 - exp_b)

        self.elo_dict[winner] = self.elo_dict[winner] + winner_delta
        self.elo_dict[loser] = self.elo_dict[loser] + loser_delta

        return exp_a

    def process_elo(self):
        
        if self.processed == True:
            return "already processed"
        
        season_ = min(self.seasons)

        for s, w, l, h in zip(self.seasons, self.winners, self.losers, self.win_loc):

            if s != season_:
                self.elo_dict = {team: ((0.87 * value + 0.13 * 1500)) for team, value in self.elo_dict.items()}
                season_ += 1

            winner_prob = self.update_elo(winner=w, loser=l, win_loc=h)

            self.winner_probs.append(winner_prob)
        
        self.processed=True
        
        return self