import pandas as pd
import numpy as np
import itertools
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.formula.api import logit, mnlogit, ols
from statsmodels.stats.anova import anova_lm

from scipy import stats
from pandas import CategoricalDtype
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots

idx = pd.IndexSlice


def build_model_expression(regressors):
    """ Given a list (or set) of strings that are variables in a dataframe, builds an expression
        (i.e., a string) that specifies a model for multiple regression. The dependent variable (y)
        is filled with '%s' so that it can replaced as in a format string.
    
    Args:
        regressors (list or set of strings): names of independent variables that will be used 
            to model the dependent variable (score).
        
    Returns:
        string: the model expression
                
    Example:
        >>> build_model_expression(['age', 'gender', 'other'])
            '%s ~ age+gender+other'
    """
    if len(regressors) == 0:
        regressors = ['1']
    return '%s ~ '+'+'.join(regressors)


def build_interaction_terms(*regressors):
    """ Given multiple lists (or sets) of regressors, returns a set of all interaction terms.
        The set of interaction terms can then be used to build a model expression.
    
    Args:
        *regressors (mutiple list or sets of strings)
            
    Returns:
        the set of all interaction terms.
            
    Examples:
        >>> build_interaction_terms(['age'], ['sleep_1', 'sleep_2'])
            {'age:sleep_1', 'age:sleep_2'}
        >>> build_interaction_terms(['age'], ['sleep_1', 'sleep_2'], ['gender_male', 'gender_other'])
            {'age:sleep_1:gender_male',
             'age:sleep_1:gender_other',
             'age:sleep_2:gender_male',
             'age:sleep_2:gender_other'}
    """
    return set([':'.join(pair) for pair in itertools.product(*regressors)])

def build_exog_pred(estimated_results, at):
    """
        at: (dict) keys are factor / variable names, with either a range of
            values, or a single specified value. All other variables will be 
            filled with the mean value.
    """
    from patsy import dmatrix
    model = estimated_results.model
    exog_mn = model.exog.mean(0)
    design = model.data.design_info
    factors = {f.name():i for f, i in design.factor_infos.items()}
    factor_slices = {f: design.term_name_slices[f] for f,_ in factors.items()}
    cats_not_specced = [f for f, i in factors.items() if i.type=='categorical' and f not in at.keys()]
    int_terms = [term for term in design.terms if len(term.factors) > 1]

    # Verify names of specified variables
    for var, val in at.items():
        assert(var in factors.keys())

    # Does cross product of all items in the "at" dict
    at = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                at.values(), names=at.keys())).reset_index()

    # Make sure categories codes are consistent
    for col in at:
        if factors[col].type == 'categorical':
            at[col] = at[col].astype(CategoricalDtype(factors[col].categories))

    at = at.set_index(list(at.columns), drop=False)

    # Names of factors to replace
    for f, i in factors.items():
        if f not in at.keys():
            if i.type == 'categorical':
                at[f] = i.categories[0]
            else:
                at[f] = exog_mn[factor_slices[f]][0]
    
    # Convert to functional design matrix
    exog = np.asarray(dmatrix(design, at))
    
    # Now replace the missing categories with column means
    for f in cats_not_specced:
        exog[:, factor_slices[f]] = exog_mn[factor_slices[f]]

    # Then update any interaction columns...
    for term in int_terms:
        int_factors = [factor.name() for factor in term.factors]
        if set(int_factors) & set(cats_not_specced):
            assert(len(int_factors) < 3)
            print(term)
            s0 = factor_slices[int_factors[0]]
            s1 = factor_slices[int_factors[1]]
            si = design.term_slices[term]
            exog[:, si] = exog[:, s0] * exog[:, s1]

    return pd.DataFrame(exog, index=at.index, columns=model.data.orig_exog.columns)


def tstat(alpha, df):
    assert((alpha < 1.0) & (alpha > 0.0))
    from scipy.stats import t
    return t.ppf((1+alpha)/2, df)

def logit_prediction_w_CIs(lr, at, confidence=0.95, linear=False):
    X = build_exog_pred(lr, at)

    prob = lr.predict(X.values, transform=False, linear=linear)
    cov = lr.cov_params()

    c = tstat(confidence, lr.df_resid-1)  # multiplier for confidence interval

    if not linear:
        # matrix of gradients for each observation
        gradient = (prob * (1 - prob) * X.values.T).T
        se = np.array([np.sqrt(np.dot(np.dot(g, cov), g))
                            for g in gradient])
        ci_lo = np.maximum(0, np.minimum(1, prob + se * c))
        ci_hi = np.maximum(0, np.minimum(1, prob - se * c))
    else:
        se = np.sqrt((X * np.dot(X, cov)).sum(1))
        ci_hi = prob + se * c
        ci_lo = prob - se * c
    
    return pd.DataFrame(
        data = {
            'prob': prob,
            'CI_hi': ci_hi,
            'CI_lo': ci_lo,
        }, index = X.index)

def linear_prediction_w_CIs(lr, at, confidence=0.95):
    X = build_exog_pred(lr, at)
    r = lr.get_prediction(X, transform=False, row_labels=X.index)
    return r.summary_frame(alpha=1-confidence)

def convert_logits(lr, confidence=.95):
    """ Given a estimated logit model (results) object, convert the regression
        parameters (logits) into odds rations (ORs), with standard error and
        confidence intervals (based on t-distribution). 
            OR = exp(B)
        The CIS are calculated as specified here:
            https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Confidence_Intervals_for_the_Odds_Ratio_in_Logistic_Regression_with_One_Binary_X.pdf
    """
    from scipy.stats import t
    c = tstat(confidence, lr.df_resid-1)
    r = pd.DataFrame(index = lr.params.index,
                     columns=['OR', 'CI_lo', 'CI_hi'])
    r.OR = np.exp(lr.params)
    r.CI_lo = np.exp(lr.params - c * lr.bse)
    r.CI_hi = np.exp(lr.params + c * lr.bse)
    return r

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

def plot_data_helper(models, at, type_='logit'):
    """ Utility function for collecting and organizing predictions from
        estimated models. Just makes the following analyses cleaner.
    """
    preds = []
    names = []
    for name, model in models.items():
        names.append(name)
        if type_ == 'logit':
            preds.append(logit_prediction_w_CIs(model, at))
        elif type_ == 'linear':
            preds.append(linear_prediction_w_CIs(model, at))
        else:
            raise ArgumentError("Error!")

    df = pd.concat(
            preds, axis=0, keys=names, names=['DV']
         ).reset_index()

    # PlotLY likes the errors bars as (+/- err), rather than mean (+/- err)
    if type_ == 'logit':
        df['CI_c'] = df.CI_hi - df.prob
    elif type_ == 'linear':
        df['CI_c'] = df.mean_ci_upper - df['mean']
    return df


def cohens_f_squared(full_model, restricted_model, type_='logit'):
    """ Calculate Cohen's f squared effect size statistic. See this reference:
    
        Selya, A. S., Rose, J. S., Dierker, L. C., Hedeker, D., & Mermelstein, R. J. (2012). 
            A practical guide to calculating Cohen’s f 2, a measure of local effect size, from PROC 
            MIXED. Frontiers in Psychology, 3, 1–6.
    
    """
    return (r2_from(full_model)-r2_from(restricted_model))/(1-r2_from(full_model))


def bayes_factor_01_approximation(full_model, restricted_model, min_value=0.001, max_value=1000):
    """ Estimate Bayes Factor using the BIC approximation outlined here:
        Wagenmakers, E.-J. (2007). A practical solution to the pervasive problems of p values.
            Psychonomic Bulletin & Review, 14, 779–804.
        
    Args:
        full_model (statsmodels.regression.linear_model.RegressionResultsWrapper):
            The estimated multiple regression model that represents H1 - the alternative hypothesis
        restricted_model (statsmodels.regression.linear_model.RegressionResultsWrapper):
            The estimated multiple regression model that represents H0 - the null hypothesis
        min_value (float): a cutoff to prevent values from getting too small.
        max_value (float): a cutoff to prevent values from getting too big
    
    Returns:
        A float - the approximate Bayes Factor in support of the null hypothesis
    """
    bf = np.exp((full_model.bic - restricted_model.bic)/2)
    return np.clip(bf, min_value, max_value)


def likelihood_ratio_test(df, h1, h0, lr):
    h0 = lr(h0, df).fit(disp=False)
    h1 = lr(h1, df).fit(disp=False)
    return likelihood_ratio_test_calc(h0, h1)


def likelihood_ratio_test_calc(h0, h1):
    llf_full = h1.llf
    llf_restr = h0.llf
    df_full = h1.df_resid
    df_restr = h0.df_resid
    lrdf = (df_restr - df_full)
    lrstat = -2*(llf_restr - llf_full)
    lr_pvalue = stats.chi2.sf(lrstat, lrdf)
    return (lrstat, lr_pvalue, lrdf)

def r2_from(estimated_model):
    r2 = getattr(estimated_model, 'rsquared', None)
    if r2 is None:
        r2 = getattr(estimated_model, 'prsquared', None)
    if r2 is None:
        raise AttributeError("Can't find R2 function for model")
    return r2


def compare_models(model_comparisons, 
        data, score_columns, model_type, alpha=0.05, 
        num_comparisons=None, adj_across=None, adj_type='bonferroni'):
    #UPDATE THIS    
    """ Performs statistical analyses to compare fully specified regression models to (nested)
        restricted models. Uses a likelihood ratio test to compare the models, and also a 
        Bayesian model comparison method.
        
    References:
        http://www.statsmodels.org/dev/regression.html
        http://www.statsmodels.org/dev/generated/statsmodels.sandbox.stats.multicomp.multipletests.html
        http://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.RegressionResults.compare_lr_test.html
        
    Args:
        model_comparisons (list of dicts): each dict in the list represents a contrast
            between two models. The dict must have: a 'name' field; a 'h0' field that 
            is the model expression for the restricted (or null) model; and a 'h1' field
            that is the model expression string for the fully specified model.
        data (panads data frame): the full data frame with all independent variables
            (predictors) and dependent variables (scores). Model expression strings in
            the previous arg are built from names of columns in this data frame. 
        score_columns (list of strings): which columns of the data frame are dependent
            variables to be modelled using expressions in model_comparisons?
        alpha (float): What considered a significant p-value after correction.
        n_comparisons (float): If specified, this is used to correct the p-values returned
            by the LR tests (i.e., multiply each p-value by the number of comparisons).
            The resulting adjusted p-value is what is compared to the alpha argument.
            Note: this take precendence over the multiple correction type argument.
        correction (string): method for correcting for multiple comparisons. Can be
            None, or any of the options listed in the above documentation for multipletests.
                
    Returns:
        A pandas dataframe with one row per score, and a multindex column structure that 
        follows a (contrast_name, statistic_name) convention.
    
    """
    score_names = [score for score in score_columns]  # no prproc for now
    contrasts = [comparison['name'] for comparison in model_comparisons]
    statistics = ['LR', 'p', 'p_adj', 'df',
                  'Delta R^2', 'Cohen f^2', 'BF_01', 'BF_10']
    results_df = pd.DataFrame(
                    index=pd.MultiIndex.from_product(
                        [contrasts, score_names], names=['contrast', 'score']),
                              columns=statistics)
    for contrast in model_comparisons:
        for score_index, score in enumerate(score_columns):
            score_name = score_names[score_index]

            # Fit the fully specified model (h1) and the nested restricted model (h0) using OLS
            h1 = model_type(contrast['h1'] % score, data=data).fit(disp=False)

            h0data = data.loc[h1.model.data.orig_exog.index, :]
            h0 = model_type(contrast['h0'] % score, data=h0data).fit(disp=False)

            # Perform Likelihood Ratio test to compare h1 (full) model to h0 (restricted) one
            lr_test_result = likelihood_ratio_test_calc(h0, h1)
            bayesfactor_01 = bayes_factor_01_approximation(
                h1, h0, min_value=0.0000001, max_value=10000000)
            all_statistics = [lr_test_result[0],
                              lr_test_result[1],
                              np.nan,
                              lr_test_result[2],
                              r2_from(h1) - r2_from(h0),
                              cohens_f_squared(h1, h0),
                              bayesfactor_01,
                              1/bayesfactor_01]
            results_df.loc[(contrast['name'], score_name), :] = all_statistics

    # Correct p-values for multiple comparisons across all tests of this contrast?
    results_df = adjust_pvals(results_df, adj_across=adj_across, adj_type=adj_type)

    return results_df

def adjust_pvals(results, num_comparisons=None, adj_across=None, adj_type=None):
    # if n_comparisons is not None:
    #     adjusted_p_vals = np.clip(contrast_p_vals * n_comparisons, 0, 1)
    #     results_df.loc[idx[contrast['name'], :], 'p_adj'] = adjusted_p_vals
    if adj_across is not None and adj_type is not None:
        if adj_across == 'all':
            p_vals = results[['p']]
        elif adj_across in ['scores', 'score', 's']:
            p_vals = results['p'].unstack('contrast')
        elif adj_across in ['contrasts', 'contrast', 'con', 'cons', 'c']:
            p_vals = results['p'].unstack('score')
        else:
            raise ValueError(f"Invalid adjust across = {adj_across}")
          
        p_adj = [multipletests(p_vals[col], method=adj_type)[1] for col in p_vals]
        p_adj = pd.DataFrame(np.column_stack(p_adj), index=p_vals.index, columns=p_vals.columns)
        if p_adj.shape[1] > 1:
            p_adj = p_adj.stack()
        p_adj = p_adj.reorder_levels(results.index.names).reindex(results.index)
    else:
        p_adj = results['p']

    results['p_adj'] = p_adj.values
    return results

def anova_analyses(formula, DVs, data, type_=2, adj_across=None, adj_type=None):
    results = []
    for dv in DVs:
        r = ols(formula%(dv), data).fit()
        results.append(anova_lm(r, typ=type_).drop('Residual'))
    results = pd.concat(results, keys=DVs, names=['score', 'contrast'])
    results = results.swaplevel('contrast', 'score')
    results = results.rename(columns={'PR(>F)': 'p'})
    results = adjust_pvals(results, adj_across=adj_across, adj_type=adj_type)
    return results

def regression_analyses(formula, DVs, data, adj_across=None, adj_type=None):
    results = []
    for dv in DVs:
        model = ols(formula % dv, data).fit()
        r = pd.concat([model.params, model.tvalues, model.pvalues], axis=1)
        r.columns = ['value', 'tstat', 'p']
        r.index.name = 'contrast'
        results.append(r)
    results = pd.concat(results, names=['score'], keys=DVs)
    results = results.swaplevel('contrast', 'score')
    results = adjust_pvals(results, adj_across=adj_across, adj_type=adj_type)
    return results

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

def pca_loadings(pca):
    L = np.sqrt(np.diag(pca.explained_variance_))
    A = np.dot(pca.components_.T, L)
    return A

def stderr(X):
    N = X.shape[0]
    return X.std(ddof=1)/np.sqrt(N)

def rm_sems(X):
    """ Calculates standard error of the mean (SEM) for repeated-measures data.
        One row per subject, one column per condition. 
        The average subject-specific differences are subtracted out.
    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    assert isinstance(X, np.ndarray)

    print('asdfasdfasdf')
    X = X[~np.isnan(X).any(axis=1)]
    grand_mean = X.mean()
    subj_means = X.mean(axis=1)[:, np.newaxis]
    subj_diffs = subj_means - grand_mean
    X_corrected = X - subj_diffs
    cond_stdevs = X_corrected.std(axis=0, ddof=1)
    cond_stders = cond_stdevs / np.sqrt(X.shape[0])
    return cond_stders

def filter_df(df, sds = [6,4], subset=None):
    if subset is None:
        subset = df.columns
    
    df_ = df[subset].copy()

    for sd in sds:
        stats = df_.describe()
        outliers = (
            abs(df_ - stats.loc['mean', :]) > sd*stats.loc['std', :])
        df_[outliers] = np.nan
        
    df[subset] = df_    
    return df

from scipy import stats
## Helper functions for running ch2, 1-way ANOVA, or t-tests on a Pandas datafame.
def chi2_pval(df, grouper, var):
    """ Computes Chi2 stat and pvalue for a variable, given a grouping variable.
    """
    tabs = df[[grouper, var]].groupby([grouper])[var].value_counts()
    chi2 = stats.chi2_contingency(tabs.unstack(grouper))
    return chi2[1]

def f_1way_pval(df, grouper, var):
    """
    """
    g = [d[var].dropna() for i, d in df[[grouper, var]].groupby(grouper)]
    f = stats.f_oneway(*g)
    return f[1]

def t_pval(df, grouper, var):
    """ Assumes only two groups present! Otherwise use f_1way_pval
    """
    g = [d[var].dropna() for i, d in df[[grouper, var]].groupby(grouper)]
    t = stats.ttest_ind(g[0], g[1], equal_var=False)
    return t[1]



