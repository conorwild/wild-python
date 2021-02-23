import pandas as pd
import numpy as np
import itertools
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.formula.api import logit, mnlogit, ols
from statsmodels.stats.anova import anova_lm

from scipy import stats
from pandas import CategoricalDtype

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
            raise AttributeError(f"Invalid model type: {type_}")

    df = pd.concat(
            preds, axis=0, keys=names, names=['DV']
         ).reset_index()

    # PlotLY likes the errors bars as (+/- err), rather than mean (+/- err)
    if type_ == 'logit':
        df['CI_c'] = df.CI_hi - df.prob
    elif type_ == 'linear':
        df['CI_c'] = df.mean_ci_upper - df['mean']
    return df


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

def compare_all_models(
        data, IVs, DVs, null=[], max_n_IVs=None, 
        do_cell_figure=False, do_line_figure=False):
    """

    Args:
        data (dataframe):
        IVs (list-like): names of columns to use as predictors in linear models
        DVs (list-like): names of columns to as outcome measures (to be 
            predicted).
        null (list-like): names of columns to be included in the null model.
        max_n_IVs (integer): maximum number of variables to be included in a 
            the models. e.g., if equal to 1, then only models that contain 
            one of the IVs will be tested. If None, then no limit (can use all
            IVs in the same model)
        do_cell_figure (boolean): generate a nifty figure.

    """
    from itertools import combinations
    from .wild_plots import create_bayes_factors_figure

    data = data.dropna(subset=[iv for iv in IVs if not ':' in iv]+DVs+null)

    if max_n_IVs is None:
        max_n_IVs = len(IVs)

    all_combs = [comb for n_var in range(max_n_IVs) for comb in combinations(IVs, r=n_var+1)]
    n_combs = len(all_combs)

    BFs = (pd
        .DataFrame(index=DVs, columns=all_combs)
        .rename_axis('score', axis=0)
        .rename_axis('model', axis=1)
    )

    for i_dv, dv in enumerate(DVs):
        expr0 = build_model_expression(null)%dv
        m0 = ols(expr0, data=data).fit()
        for i_m, comb in enumerate(all_combs):
            expr1 = build_model_expression(list(comb)+null)%dv
            m1 = ols(expr1, data=data).fit()
            BFs.iloc[i_dv, i_m] = 1./ bayes_factor_01_approximation(
                m1, m0, min_value=0, max_value=None)

    if do_cell_figure:
        plot_data = (pd
            .concat([BFs.rename_axis('contrast', axis=1)], keys=['BF10'], axis=1)
            .stack('contrast')
        )
        fig = create_bayes_factors_figure(plot_data, cell_scale=0.4)
    else:
        fig = None

    return BFs, fig

def compare_models(model_comparisons, 
        data, score_columns, model_type, **correction_args):
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
    results_df = (results_df
        .pipe(adjust_pvals, **correction_args)
        .swaplevel('contrast', 'score')
        .loc[idx[score_names,:], :]
    )

    return results_df

def corrected_alpha_from(**correction_args):
    if 'alpha' in correction_args:
        return correction_args['alpha']
    elif 'n_comparisons' in correction_args:
        return 0.05 / correction_args['n_comparisons']
    else:
        return 0.05

def adjust_pvals(results, 
        n_comparisons=None, adj_across=None, adj_type=None, alpha=None):

    if n_comparisons is not None:
        p_adj = results['p'] * n_comparisons

    elif alpha is not None:
        p_adj = results['p'] * 0.05/alpha

    elif adj_across is not None and adj_type is not None:
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

    results['p_adj'] = np.clip(p_adj.values, 0, 1)
    return results

def anova_analyses(formula, DVs, data, type_=2, **correction_args):
    results = []
    for dv in DVs:
        r = ols(formula%(dv), data).fit()
        results.append(anova_lm(r, typ=type_).drop('Residual'))
    results = pd.concat(results, keys=DVs, names=['score', 'contrast'])
    results = results.swaplevel('contrast', 'score')
    results = results.rename(columns={'PR(>F)': 'p'})
    results = adjust_pvals(results, **correction_args)
    return results

def JSZ_BFs_from_ts(tvalues, N):
    """
    """
    from pingouin import bayesfactor_ttest
    BFs = { name: bayesfactor_ttest(t, N, paired=True, tail='two-sided') for 
                name, t in tvalues.items() }
    return pd.Series(BFs)


def regression_analyses(formula, DVs, data, IVs=None, **correction_args):
    """ Insert description here.

    Args:
        Formula (string): the regression formula used for building models.
            Must begind with '%s' because the DV gets interpolated when looping
            over all the variables to be evaluated.
        DVs (list-like): the names of dependent variables. One regression model
            will be estimated for each DV.
        IVs (list-like): the independent variables to do statistics on. If 
            None then all variables in the formula are tested. (default: None)
        alpha (float): the alpha (e.g., 0.05) for determining statistical
            significance. You can etiher specify this here, or leave it and
            specify a method for correcting formultiple comparisons (see
            next two parameters). If set, this option takes priority.
        n_comparisons: like alpha, but instead it's the number of comparisons.
        adj_across (string): when doing an automatic correction for multiple
            comparisons, do you correct across DVs ("scores"), IVs ("contrasts")
            or all of them ("all")? 
        adj_type (string): the method for correcting for multiple comparisons,
            should be one of the options available in <insert function here>.
            For example 'fdr_bh', 'sidak', 'bonferroni', etc.

    """
    results = []

    for dv in DVs:
        model = ols(formula % dv, data).fit()
        n_obs  = model.df_resid+model.df_model+1
        r = pd.concat(
                [model.params, 
                 model.tvalues,
                 model.pvalues, 
                 model.conf_int(corrected_alpha_from(**correction_args)),
                 JSZ_BFs_from_ts(model.tvalues, n_obs)], 
                axis=1)
        r.columns = ['value', 'tstat', 'p', 'CI_lower', 'CI_upper', 'BF10']
        r['df'] = model.df_resid
        r.index.name = 'contrast'
        results.append(r)
    results = pd.concat(results, names=['score'], keys=DVs)

    if IVs is not None:
        results = results.loc[idx[:, IVs], :]

    results = adjust_pvals(results, **correction_args)

    return results


def ttests(
        group_var, DVs, data, paired=False, tails='two-sided',
        test_name='Mean Difference', **correction_args):
    """ Insert description here.

    Args:
        group_var (string): The variables that will be used to group/split
            the dataset. Bust have only two levels.
        DVs (list-like): the names of dependent variables. One regression model
            will be estimated for each DV.
        alpha (float): the alpha (e.g., 0.05) for determining statistical
            significance. You can etiher specify this here, or leave it and
            specify a method for correcting formultiple comparisons (see
            next two parameters). If set, this option takes priority.
        adj_across (string): when doing an automatic correction for multiple
            comparisons, do you correct across DVs ("scores"), IVs ("contrasts")
            or all of them ("all")? 
        adj_type (string): the method for correcting for multiple comparisons,
            should be one of the options available in <insert function here>.
            For example 'fdr_bh', 'sidak', 'bonferroni', etc.

    """
    from pingouin import ttest
    grp_data = [d for _, d in data.groupby(group_var)]
    assert(len(grp_data)==2)

    results = []
    for dv in DVs:
        t = ttest(
            grp_data[0][dv], grp_data[1][dv], 
            confidence=(1-corrected_alpha_from(**correction_args)),
            paired=paired, tail=tails)
        t.index.names = ['contrast']
        t['value'] = grp_data[0][dv].mean() - grp_data[1][dv].mean()
        results.append(t)

    results = (pd
        .concat(results, names=['score'], keys=DVs)
        .rename(columns={'p-val': 'p', 'T': 'tstat', 'dof': 'df'},
                index={'T-test': test_name})
    )

    # unpack the CIs provided by pingouin
    ci_col = [c for c in results.columns if c[0:2]=='CI'][0]
    results = (results
        .assign(CI_lower=results[ci_col].apply(lambda x: x[0]))
        .assign(CI_upper=results[ci_col].apply(lambda x: x[1]))
        .drop(columns=[ci_col])
    )

    results = adjust_pvals(results, **correction_args)

    return results


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



