import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.offline import init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
from .wild_statsmodels import f_1way_pval, tstat
from .wild_colors import D1_CMAP, D2_CMAP, D3_CMAP, D4_CMAP
from os import path
from copy import deepcopy
from mergedeep import merge

idx = pd.IndexSlice

dark2 = px.colors.qualitative.Dark2

_LINE_COLOUR = 'rgb(16, 16, 16)'
_MARGINS = {'t': 20, 'r': 10, 'l': 80, 'b': 20}

from matplotlib import rc
plt.rcParams['figure.dpi'] = 100
plt.rcParams.update({'font.size': 10})
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rcParams['svg.fonttype'] = 'none'

def plotly_template():
    return {
        'layout': go.Layout(
            plot_bgcolor = 'rgba(1,1,1,0.1)',
            font_family = 'sans-serif',
            font = {'size': 10},
            xaxis = {
                'zeroline': True,
                'zerolinecolor': _LINE_COLOUR,
                'zerolinewidth': 1,
                'gridcolor': 'white',
                'gridwidth': 1
            },
            yaxis = {
                'zeroline': True, 
                'zerolinewidth': 1,
                'gridwidth': 1,
                'zerolinecolor': 'white',
                'gridcolor': 'white',
            },
        ),
        'data': {
            'bar': [go.Bar(
                marker_line_color = _LINE_COLOUR, 
                marker_line_width = 1.5,
                error_y = {
                    'color': _LINE_COLOUR,
                    'thickness': 1.5,
                }
            )],
            'scatter': [go.Scatter(
                marker_line_color = _LINE_COLOUR, 
                marker_line_width = 1.5,
                error_y = {
                    'color': _LINE_COLOUR,
                    'thickness': 1.5,
                }
            )]
        }
    }

def create_stats_figure(
        results, stat_name, p_name, alpha=0.05, log_stats=True, 
        diverging=False, stat_range=None, correction=None, vertline=4, 
        marker_color=None, reverse=False
    ):
    """ Creates a matrix figure to summarize multple tests/scores. Each cell 
        represents a contrast (or model comparison) for a specific effect (rows)
        for a given score (columns). Also draws asterisks on cells for which 
        there is a statistically significant effect.
        
    Args:
        results (Pandas dataframe): a dataframe that contains the statistics to 
            display. Should be a rectangular dataframe with tests as rows and 
            effects as columns (i.e., the  transpose of the resulting image). 
            The dataframe index and column labels are used as labels for the 
            resulting figure.
        stat_name (string): Which statistic to plot. There might be multiple 
            columns for each effect (e.g., Likelihood Ratio, BFs, F-stats, etc.)
        p_name (string): The name of the column to use for p-values.
        alpha (float): what is the alpha for significant effects?
        log_stats (boolean): Should we take the logarithm of statistic values 
            before creating the image? Probably yes, if there is a large 
            variance in value across tests and effects.
        correction (string): indicates how the alpha was corrected (e.g., FDR 
            or bonferroni) so the legend can be labelled appropriately.
            
    Returns:
        A matplotlib figure.
        
    """

    score_index = results.index.unique('score')
    contrast_index = results.index.unique('contrast')
    stat_values = (results
        .loc[:, stat_name]
        .unstack('contrast')
        .loc[score_index, contrast_index]
    )
    p_values = (results
        .loc[:, p_name]
        .unstack('contrast')
        .loc[score_index, contrast_index]
    )
    num_scores = stat_values.shape[0]
    num_contrasts = stat_values.shape[1]
    image_values = stat_values.values.astype('float32')

    # If it's a diverging scale, it's probably a t-stat or something. Don't
    # know why I have this here. There is a better solution.
    if diverging:
        log_stats = False

    image_values = np.log10(image_values) if log_stats else image_values

    imax = np.max(np.abs(image_values))
    if diverging:
        irange = [-1*imax, imax] if stat_range is None else stat_range
        cmap = D2_CMAP
    else:
        irange = [0, np.min([3, imax])] if stat_range is None else stat_range
        cmap = 'viridis'
        image_values = np.clip(image_values, 0, 100)

    figure = plt.figure(figsize=[num_scores*0.6, num_contrasts*0.6])
    plt_axis = figure.add_subplot(1, 1, 1)
    imgplot = plt_axis.imshow(
                image_values.T, aspect='auto', clim=irange, cmap=cmap)

    if vertline is not None:
        plt_axis.plot([num_scores-(vertline+.5), num_scores-(vertline+.5)],
                    [-0.5, num_contrasts-0.5], c='w')

    if marker_color is None:
        marker_color = 'whitesmoke' 

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
    legend_label = "p < %.02f" % alpha
    legend_label += f" ({'unc' if correction is None else correction})"
    plt_axis.plot(reject_h0[1], reject_h0[0], marker_color, linestyle='none',
                  marker='$\u2217$', label=legend_label, markersize=10)
    # plt_axis.plot(reject_h0[1], reject_h0[0], '*',
    #               markersize=10, label=legend_label)

    plt.legend(bbox_to_anchor=(1, 1.1), loc=4, borderaxespad=0.,
        facecolor='lightgray', edgecolor='lightgray')

    return figure


def create_bayes_factors_figure(results, log_stats=True, 
        vertline=None, cmap=None, cell_scale=0.6, suppress_h0=False):
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

    
    score_index = results.index.unique('score')
    contrast_index = results.index.unique('contrast')
    num_scores = len(score_index)
    num_contrasts = len(contrast_index)
    bf_values = results.loc[:, 'BF10'].unstack('contrast').reindex(
        index=score_index, columns=contrast_index).values.astype('float32')
    # Too small values cause problems for the image scaling

    np.place(bf_values, bf_values < 0.00001, 0.00001)

    if cmap is None:
        cmap = D2_CMAP

    figure = plt.figure(figsize=[num_scores*cell_scale, num_contrasts*cell_scale])
    plt_axis = figure.add_subplot(1, 1, 1)
    imgplot = plt_axis.imshow(np.log10(bf_values.T),
                              aspect='auto', cmap=cmap, clim=[-6.0, 6.0])

    if vertline is not None:
        plt_axis.plot([num_scores-(vertline+.5), num_scores-(vertline+.5)],
                    [-0.5, num_contrasts-0.5], c='w')

    plt_axis.set_yticks(np.arange(0, num_contrasts))
    plt_axis.set_yticklabels(list(contrast_index))
    plt_axis.set_xticks(np.arange(0, num_scores))
    plt_axis.set_xticklabels(list(score_index), rotation=45, ha='right')

    # Add a colour bar
    cbar = figure.colorbar(imgplot, ax=plt_axis, pad=0.2/num_scores)
    cbar.ax.set_ylabel('$H_0$   '+'$Log(BF_{10})$'+'   $H_1$')
    # cbar.ax.text(75,  4, "$H_1$")
    # cbar.ax.text(75, -5, "$H_0$")

    # Use absolute BFs for determining weight of evidence
    abs_bfs = bf_values
    abs_bfs[abs_bfs == 0] = 0.000001
    if not suppress_h0:
        abs_bfs[abs_bfs < 1] = 1/abs_bfs[abs_bfs < 1]

    # Custom markers for the grid
    # markers = [(2+i, 1+i % 2, i/4*90.0) for i in range(1, 3)]
    markers = [(3, 2, 22.5), '$\u2727$', '$\u2736$']
    markersize = 10 * cell_scale *2

    # Positive evidence BF 3 - 20
    positive = (abs_bfs >= 3) & (abs_bfs < 20)
    xy = positive.nonzero()
    plt_axis.plot(xy[0], xy[1], 'whitesmoke', linestyle='none',
                  marker=markers[0], label='positive', markersize=markersize)

    # Strong Evidence BF 20 - 150
    strong = (abs_bfs >= 20) & (abs_bfs < 150)
    xy = strong.nonzero()
    plt_axis.plot(xy[0], xy[1], 'whitesmoke', linestyle='none',
                  marker=markers[1], label='strong', markersize=markersize)

    # Very strong evidence BF > 150
    very_strong = (abs_bfs >= 150)
    xy = very_strong.nonzero()
    plt_axis.plot(xy[0], xy[1], 'whitesmoke', linestyle='none',
                  marker=markers[2], label='v. strong', markersize=markersize)

    plt.legend(bbox_to_anchor=(0.5, 1.05), loc='lower center',
               borderaxespad=0., ncol=4, title='Bayesian Evidence',
               facecolor='lightgray', edgecolor='lightgray')

    return figure

def pie_plot(
        df, group_var, hole=0.3, marker=None, width=400, height=250,
        layout_args={}, pie_args={}
    ):
    c = df.groupby(group_var).agg(['count']).iloc[:, 0]
    f = go.Figure(
            go.Pie(
                labels=c.index, 
                values=c.values, 
                hole=hole, 
                textinfo='value+percent',
                marker={} if marker is None else marker,
                **pie_args
            ))
    f.update_layout(
        width=width, height=250, 
        margin=margins,
        legend=dict(title=group_var),
        **merge(_PLOTLY_LAYOUT_DEFAULTS, layout_args))

    return f

def histogram(
        df, var, bins, centres=None, x_title=None, y_title=None, height=300,
        width=400, layout_args={}, bar_args={}
    ):

    counts, bins = np.histogram(df[var], bins=bins)
    if centres is None: 
        centres = bins = 0.5 * (bins[:-1] + bins[1:])
    f = px.bar(
            x=centres, y=counts, 
            labels={ 
                'x': x_title if x_title is not None else var,
                'y': y_title if y_title is not None else '# of Participants'
            },
            **bar_args)

    f.update_layout(
        margin=margins,
        barmode='group', bargap=0.0, 
        width=width, height=height,
        **merge(_PLOTLY_LAYOUT_DEFAULTS(), layout_args))

    return f


def means_plot(
        df, vars, vars_name,
        bar_args={}, layout_args={}, trace_args={},
        group=None, group_order=None, 
        group_color_sequence=px.colors.sequential.Plasma,
        group_tests=False, bar_tests=False, bar_correction='group',
    ):

    stats = ['mean', 'std', 'count']
    order = {vars_name: vars}
    if group is None:
        means = df[vars].agg(stats).T
        means.index.name = vars_name
        means.columns.name = 'stat'
    else:
        if group_order is not None:
            order = {**order, group: group_order}
            group_nms = group_order
        else:
            group_nms = list(df[group].unique())

        ngrps = len(group_nms)
        means = df[vars+[group]].groupby(group).agg(stats)
        means.columns.names = [vars_name, 'stat']
        means = means.stack(vars_name)
    
    means['mean_se'] = means['std']/np.sqrt(means['count'])

    f = px.bar(
            means.reset_index(),
            x=vars_name, y='mean', error_y='mean_se', 
            color=group,
            category_orders=order,
            color_discrete_sequence=group_color_sequence,
            barmode='group',
            **bar_args)
            
    if group_tests:
        for v in vars:
            p = f_1way_pval(df, group, v)
            if  p < 0.001:
                txt = "***"
            elif p < 0.01:
                txt = "**"
            elif p < 0.05:
                txt = "*"
            else:
                txt = ""
            f.add_annotation(
                x=v, text=txt, showarrow=False,
                y=means.loc[idx[:, v], :][['mean', 'mean_se']].sum(axis=1).max(),
                xanchor='center', yanchor='bottom')


    f.update_traces(**trace_args)
    f.update_layout(
        template=plotly_template(),
        # margin=_MARGINS,
        **layout_args)
    
    return f, means

def qq_plots(qq_results, titles, marker_size=5, lims=[-4,4], layout_args={}):
    """ Assumes a A x B matrix of results from a statsmodels probplot function.
    """
    from plotly.subplots import make_subplots
    assert(isinstance(qq_results, np.ndarray))
    assert(qq_results.shape == titles.shape)

    nrows, ncols = qq_results.shape

    fig = make_subplots(rows=nrows, cols=ncols, 
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing = 0.04, vertical_spacing = 0.04)

    ii = 1
    for ir in range(nrows):
        for ic in range(ncols):
            qq = qq_results[ir, ic]
            if qq is not None:
                fig.add_trace(
                    go.Scatter(
                        x=qq[0][0], y=qq[0][1],
                        mode='markers', 
                        marker={'size': marker_size, 'opacity': 0.5, 'color': 'black'},
                        showlegend=False,
                    ),
                    row=ir+1, col=ic+1,
                )

                xx = np.array(lims)
                yy = xx*qq[1][0] + qq[1][1]
                fig.add_trace(
                    go.Scatter(
                        x=xx, y=yy, 
                        mode='lines', line={'color': 'white', 'width': 2},
                        showlegend=False,
                    ),
                    row=ir+1, col=ic+1,
                )
                fig.update_xaxes(
                    range=lims, 
                    zerolinecolor='white', 
                    row=ir+1, col=ic+1
                )
                fig.update_yaxes(
                    range=lims, row=ir+1, col=ic+1
                )

                fig.add_annotation(
                    yanchor='top', xanchor='left', 
                    x=lims[0]+0.5, y=lims[1]-0.5,
                    xref=f"x{ii}", yref=f"y{ii}",
                    text=titles[ir, ic], showarrow=False
                )
            ii += 1

            if ic == 0:
                fig.update_yaxes(
                    title={'text': 'observed'},
                    row=ir+1, col=ic+1,
                )
            if ir == nrows-1:
                fig.update_xaxes(
                    title={'text': 'theoretical'},
                    row=ir+1, col=ic+1,
                )

    fig.update_layout(
        template=plotly_template(),
        margin={'b': 75, 't': 20, 'r': 30},
        **layout_args)

    return fig

def correlogram(
        df, subset=None, mask_diag=True, thresh=None, 
        width=475, height=350, colormap='Picnic', layout_args={}):
    """ Description here
    Args: 
        df (dataframe): The dataframe with rows as observations and columns
            as variables.
        subset (list-like): Which variables (columns) to subselect. If None, all
            columns are used. (default: None)
        
    """
    if subset is None:
        subset = df.columns

    r = df[subset].corr()

    if thresh is not None:
        df[np.abs(df)<thresh] = 0

    if mask_diag:
        np.fill_diagonal(r.values, 0)

    r = (r
        .stack()
        .reset_index()
        .rename(columns={'level_0': 'x', 'level_1': 'y', 0: 'r'})
    )

    f = px.scatter(r, x='x', y='y', size=np.abs(r['r']), 
        color='r', range_color=[-1,1], opacity = 1,
        color_continuous_scale=getattr(px.colors.diverging, colormap))

    f.update_layout(
        xaxis={'title': None},
        yaxis={'title': None},
        width=width, height=height,
        coloraxis={'colorbar': 
            {'thickness': 10, 'tickmode': 'array', 'tickvals': [-1, 0, 1],
            'title': 
                {'text': 'correlation (r)', 'side': 'right'}
            },
        },
        font={'size': 8},
        margin={'t': 20, 'r': 10, 'l': 80, 'b': 20},
        **layout_args)
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

    return fig

def jittered_helper(df, xvar, xgrp, gap=8):
    df[xvar] = df[xvar].astype('category')
    df[xgrp] = df[xgrp].astype('category')
    n_grps = len(df[xgrp].cat.categories)
    dx = (n_grps-1)/2
    x_jitt = (df[xgrp].cat.codes - dx) / gap / dx
    df['new_x'] = df[xvar].cat.codes+1+x_jitt
    return df

def jittered_scatter(df, xvar, xgrp, yvar, yerr, colormap, horizontal=False,
                     yrange=[0, 1], chance_line=True, 
                     ytitle='Probability (negative outcome)', 
                     xtitle='Measure'):
    df = jittered_helper(df, xvar, xgrp) 
    n_cats = len(df[xvar].cat.categories)
    f = px.scatter(
        df, x='new_x', y=yvar, color=xgrp, error_y=yerr, 
        color_discrete_sequence=colormap)

    f.update_layout(
        xaxis={
            'tickmode': 'array',
            'tickvals': np.arange(1, n_cats+1),
            'ticktext': list(df[xvar].cat.categories),
        }
    )

    if chance_line:
        f.add_trace(
            go.Scatter(x=[0.5, n_cats+0.5], y=[0.5, 0.5], line={'dash': 'dot', 'width': 2},
                    mode='lines', marker_color='gray', showlegend=False)
        )

    f.update_yaxes(zeroline=False, range=yrange,
                   title=ytitle)
    f.update_xaxes(title=xtitle)

    return f


def linear_mean_prediction_plot(
        df, xvar, xgrp, yvar, yerr, colormap, ymap=None, ytitle="",
        xtitle="", width=400, height=250, margins = {'t': 20, 'r': 10, 'l': 80, 'b': 20}):
    """
        ymap is a list of lables to apply to the y-axis, instead of numbers.
        Assumes 0 -> n_values-1 (integers)
    """
    n_cats = df['DV'].value_counts().shape[0]
    df[xgrp] = df[xgrp].astype('category')
    f = px.bar(
        df, x=xvar, y=yvar, color=xgrp, error_y=yerr,
        color_discrete_sequence=colormap, barmode='group')

    if ymap is not None:
        f.update_layout(
            yaxis={
                'tickmode': 'array',
                'tickvals': np.arange(0, len(ymap)),
                'ticktext': ymap,
                'range': [0, len(ymap)-0.75]
            }
        )

    f.update_layout(
        yaxis={'title': ytitle},
        xaxis={'showgrid': False, 'title': xtitle, 'tickangle': 30},
        bargap=.25, bargroupgap=0.1,
        margin=margins, width=width, height=height)

    return f


def odds_plot(df, xvar, xgrp, yvar, yerr, colormap):
    df[xvar] = df[xvar].astype('category')
    df[xgrp] = df[xgrp].astype('category')
    n_cats = len(df[xvar].cat.categories)
    n_parm = len(df[xgrp].cat.categories)
    fig = make_subplots(rows=1, cols=n_cats, subplot_titles=df[xvar].cat.categories,
                        shared_yaxes=True, horizontal_spacing=0.01)

    df['y_pls'] = df['CI_hi'] - df[yvar]
    df['y_min'] = df[yvar] - df['CI_lo']

    for i, (_, d) in enumerate(df.groupby(xvar)):
        fig.add_trace(
            go.Scatter(
                x=d[yvar].values, y=d[xgrp].values,
                error_x={
                    'type': 'data', 'symmetric': False, 
                    'array': d['y_pls'], 'arrayminus': d['y_min']},
                mode='markers', showlegend=False),
            row=1, col=i+1)
        fig.add_trace(
            go.Scatter(
                y=d[xgrp].values, x=[1]*n_parm,
                line={'dash': 'dot', 'width': 2},
                mode='lines', marker_color='gray', showlegend=False
            ),
            row=1, col=i+1)

    fig.update_xaxes(
        zeroline=False,
        tickvals=[0.2, 0.5, 1.0, 2.0, 5.0], range=[-1, 1], type="log",
        title='Odds Ratio')
    fig.update_yaxes(
        zeroline=False, showgrid=False)
    return fig
    
def write_image(figure, name, directory, format='png', scale=2):
    img_bytes = figure.to_image(format=format, scale=scale)
    out_file = path.join(directory, name)
    with open(out_file, 'wb') as file:

        file.write(img_bytes)
