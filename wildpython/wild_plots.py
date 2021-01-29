import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.offline import init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots

margins = {'t': 20, 'r': 10, 'l': 80, 'b': 20}
dark2 = px.colors.qualitative.Dark2

def create_stats_figure(results, stat_name, p_name, alpha=0.05, log_stats=True, diverging=False, correction=None, vertline=4):
    """ Creates a matrix figure to summarize multple tests/scores. Each cell represents a contrast
        (or model comparison) for a specific effect (rows) for a given score (columns). Also
        draws asterisks on cells for which there is a statistically significant effect.
        
    Args:
        results (Pandas dataframe): a dataframe that contains the statistics to display. Should
            be a rectangular dataframe with tests as rows and effects as columns (i.e., the 
            transpose of the resulting image). The dataframe index and column labels are used
            as labels for the resulting figure.
        stat_name (string): Which statistic to plot. There might be multiple columns for each
            effect (e.g., Likelihood Ratio, BFs, F-stats, etc.)
        p_name (string): The name of the column to use for p-values.
        alpha (float): what is the alpha for significant effects?
        log_stats (boolean): Should we take the logarithm of statistic values before creating 
            the image? Probably yes, if there is a large variance in value across tests and
            effects.
        correction (string): indicates how the alpha was corrected (e.g., FDR or bonferroni) so
            the legend can be labelled appropriately.
            
    Returns:
        A matplotlib figure.
        
    """

    score_index = results.index.get_level_values(1).unique()
    contrast_index = results.index.get_level_values(0).unique()
    stat_values = results.loc[:, stat_name].unstack().T.reindex(
        index=score_index, columns=contrast_index)
    p_values = results.loc[:, p_name].unstack().T.reindex(
        index=score_index, columns=contrast_index)
    num_scores = stat_values.shape[0]
    num_contrasts = stat_values.shape[1]
    image_values = stat_values.values.astype('float32')

    if diverging:
        log_stats = False

    image_values = np.log10(image_values) if log_stats else image_values

    imax = np.max(image_values)
    if diverging:
        irange = [-1*imax, imax]
        cmap = 'coolwarm'
    else:
        irange = [0, np.min([3, imax])]
        cmap = 'viridis'

    figure = plt.figure(figsize=[num_scores*0.6, num_contrasts*0.6])
    plt_axis = figure.add_subplot(1, 1, 1)
    imgplot = plt_axis.imshow(image_values.T, aspect='auto', clim=irange, cmap=cmap)

    if vertline is not None:
        plt_axis.plot([num_scores-(vertline+.5), num_scores-(vertline+.5)],
                    [-0.5, num_contrasts-0.5], c='w')
    plt_axis.set_yticks(np.arange(0, num_contrasts))
    plt_axis.set_yticklabels(list(contrast_index))
    plt_axis.set_xticks(np.arange(0, num_scores))
    plt_axis.set_xticklabels(list(score_index), rotation=45, ha='right')
    cbar = figure.colorbar(imgplot, ax=plt_axis, pad=0.2/num_scores)
    if log_stats:
        cbar.ax.set_ylabel('$Log_{10}$'+stat_name)
    else:
        cbar.ax.set_ylabel(f"{stat_name}")

    reject_h0 = (p_values.values.T < alpha).nonzero()
    legend_label = "p < %.04f" % alpha
    legend_label += f" ({'unc' if correction is None else correction})"
    plt_axis.plot(reject_h0[1], reject_h0[0], 'r*',
                  markersize=10, label=legend_label)

    plt.legend(bbox_to_anchor=(1, 1.1), loc=4, borderaxespad=0.)
    plt.show()
    return figure


def create_bayes_factors_figure(results, log_stats=True):
    """ Creates a matrix figure to summarize Bayesian stats for multiple scores & tests.
        Each cell indicates the Bayes Factor (BF associated with a model comparison) for 
        a specific effect (rows) for a given score (columns). Also draws symbols on cells
        to indicate the interpretation of that BF.
        
    Args:
        results (Pandas dataframe): a dataframe that contains the statistics to display. Should
            be a rectangular dataframe with tests as rows and effects as columns (i.e., the 
            transpose of the resulting image). The dataframe index and column labels are used
            as labels for the resulting figure.
        log_stats (boolean): Should we take the logarithm of BF values before creating 
            the image? Probably yes, if there is a large variance in value across scores and
            effects.
            
    Returns:
        A matplotlib figure
    
    """

    score_index = results.index.get_level_values(1).unique()
    contrast_index = results.index.get_level_values(0).unique()
    num_scores = len(score_index)
    num_contrasts = len(contrast_index)
    bf_values = results.loc[:, 'BF_01'].unstack().T.reindex(
        index=score_index, columns=contrast_index).values.astype('float32')
    # Too small values cause problems for the image scaling
    np.place(bf_values, bf_values < 0.00001, 0.00001)

    figure = plt.figure(figsize=[num_scores*0.6, num_contrasts*0.6])
    plt_axis = figure.add_subplot(1, 1, 1)
    imgplot = plt_axis.imshow(np.log10(bf_values.T),
                              aspect='auto', cmap='coolwarm', clim=[-6.0, 6.0])
    plt_axis.plot([num_scores-4.5, num_scores-4.5],
                  [-0.5, num_contrasts-0.5], c='w')
    plt_axis.set_yticks(np.arange(0, num_contrasts))
    plt_axis.set_yticklabels(list(contrast_index))
    plt_axis.set_xticks(np.arange(0, num_scores))
    plt_axis.set_xticklabels(list(score_index), rotation=45, ha='right')

    # Add a colour bar
    cbar = figure.colorbar(imgplot, ax=plt_axis, pad=0.2/num_scores)
    cbar.ax.set_ylabel('$Log_{10}(BF_{01})$')
    cbar.ax.text(0, 1.05, "$H_0$")
    cbar.ax.text(0, -0.12, "$H_1$")

    # Use absolute BFs for determining weight of evidence
    abs_bfs = bf_values
    abs_bfs[abs_bfs == 0] = 0.000001
    abs_bfs[abs_bfs < 1] = 1/abs_bfs[abs_bfs < 1]

    # Custom markers for the grid
    markers = [(2+i, 1+i % 2, i/4*90.0) for i in range(1, 5)]
    markersize = 10

    # Positive evidence BF 3 - 20
    positive = (abs_bfs >= 3) & (abs_bfs < 20)
    xy = positive.nonzero()
    plt_axis.plot(xy[0], xy[1], 'r', linestyle='none',
                  marker=markers[0], label='positive', markersize=markersize)

    # Strong Evidence BF 20 - 150
    strong = (abs_bfs >= 20) & (abs_bfs < 150)
    xy = strong.nonzero()
    plt_axis.plot(xy[0], xy[1], 'r', linestyle='none',
                  marker=markers[1], label='strong', markersize=markersize)

    # Very strong evidence BF > 150
    very_strong = (abs_bfs >= 150)
    xy = very_strong.nonzero()
    plt_axis.plot(xy[0], xy[1], 'r', linestyle='none',
                  marker=markers[2], label='very strong', markersize=markersize)

    plt.legend(bbox_to_anchor=(0.5, 1.05), loc='lower center',
               borderaxespad=0., ncol=4, title='Bayes\' evidence')
    plt.show()
    return figure

def pie_plot(df, group_var, hole=0.3, marker=None, width=400, height=250):
    c = df.groupby(group_var).agg(['count']).iloc[:, 0]
    f = go.Figure(
        go.Pie(
                labels=c.index, 
                values=c.values, 
                hole=hole, 
                textinfo='value+percent',
                marker={} if marker is None else marker
        ))
    f.update_layout(width=400, height=250, margin=margins)
    return f

def pca_loading_plot(loadings_matrix, n_comps, feature_names, write_img=False,
                     height=900, width=400):
    """ Visalize PCA loadings.
    """
    loadings = pd.DataFrame(loadings_matrix[:, 0:n_comps],
                            index=feature_names,
                            columns=[f"PC{i:02d}" for i in np.arange(0, n_comps)+1])

    lpic = go.Heatmap(z=loadings.values,
                      y=loadings.index.to_list(),
                      x=loadings.columns,
                      colorscale='RdBu', zmid=0)
    fig = go.Figure(data=[lpic])
    fig.update_layout(
        font=dict(size=10),
        yaxis=dict(tickfont=dict(size=8)),
        height=height, width=width)

    if write_img:
        write_image(fig, f"{age_str}_{test}")

    return fig