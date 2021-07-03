from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

__all__ = ['compute_improvement']


def get_res(df:pd.DataFrame, bins:np.ndarray=np.linspace(100,4000,21),
            pred_name:str='pred', targ_name:str='target') -> pd.DataFrame:
    r'''
    Computes resolution in bins of true energy based on predictions which are already bias-corrected

    Arguments:
        df: DataFrame with true and corrected predicted energy per muon
        bins: Array of bin edges
        pred_name: name of column to use as predictions
        targ_name: name of column to use as targets

    Returns:
        Aggregated DataFrame with resolution in bins of true energy
    '''
    
    def _percentile(n:float) -> Callable[[np.ndarray], float]:
        def __percentile(x): return np.nanpercentile(x, n)
        __percentile.__name__ = f'{n}'
        return __percentile
    
    grps = df.groupby(pd.cut(df[targ_name], bins))
    df.loc[:,'error'] = df[pred_name].values/df[targ_name].values
    agg_func = dict(pred_p16=pd.NamedAgg(column=pred_name, aggfunc=_percentile(15.865)),
                    pred_p84=pd.NamedAgg(column=pred_name, aggfunc=_percentile(84.135)),
                    pred_med=pd.NamedAgg(column=pred_name, aggfunc='median'),
                    true_med=pd.NamedAgg(column=targ_name, aggfunc='median'))
    
    agg = grps.agg(**agg_func).reset_index()
    agg['pred_c68'] = (agg.pred_p84-agg.pred_p16)/2
    agg['rmse_med'] = np.sqrt((agg.pred_c68**2)+((agg.pred_med-agg.true_med)**2))
    agg['frac_rmse_med'] = agg['rmse_med'].values/agg.true_med.values
    return agg


def compute_improvement(df:pd.DataFrame, plot:bool=True, pred_name:str='pred', targ_name:str='target') -> float:
    r'''
    Computes the overall improvement in RMSE over tracker due to the calo measurement.

    Arguments:
        df: Pandas DataFrame containing predictions and targets
        plot: if true, will plot resolutions
        pred_name: name of column to use as predictions
        targ_name: name of column to use as targets
        
    Returns:
        Improvment metric (higher = better)
    '''
    
    n_bins = 20
    res_bins = np.linspace(100,4000,n_bins+1)
    targ_bins = np.array([res_bins[i]+(res_bins[i+1]-res_bins[i])/2 for i in range(n_bins)])
    tracker_res = (2e-4)*targ_bins
    
    res = get_res(df, bins=res_bins, pred_name=pred_name, targ_name=targ_name)
    rmse = np.sqrt(1/((1/(res.frac_rmse_med**2))+(1/(tracker_res**2))))
    improv = (tracker_res-rmse).mean()
    
    if plot:
        with sns.axes_style(** {'style':'whitegrid', 'rc':{'patch.edgecolor':'none'}}), sns.color_palette('tab10'):
            plt.figure(figsize=(8*16/9, 8))
            plt.plot(targ_bins, 100*res['frac_rmse_med'], label='Calorimeter')
            plt.plot(targ_bins, 100*tracker_res, label='Tracker')
            plt.plot(targ_bins, 100*rmse, label=f'Combined frac. RMSE,\nmean improvement={improv*100:.1f}%')

            plt.xlabel(r'$E_{\mathrm{True}}\ [GeV]$', fontsize=24)
            plt.ylabel('Percentage RMSE [%]', fontsize=24)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(fontsize=16)
            plt.show()
    return improv
