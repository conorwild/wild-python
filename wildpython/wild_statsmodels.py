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
    factors = {f.name(): i for f, i in design.factor_infos.items()}
    factor_slices = {f: design.term_name_slices[f] for f, _ in factors.items()}
    cats_not_specced = [f for f, i in factors.items(
    ) if i.type == 'categorical' and f not in at.keys()]
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


def logit_prediction_w_CIs(lr, at, confidence=0.95):
    X = build_exog_pred(lr, at)

    prob = lr.predict(X.values, transform=False)  # predicted probability
    cov = lr.cov_params()

    # matrix of gradients for each observation
    gradient = (prob * (1 - prob) * X.values.T).T
    se = np.array([np.sqrt(np.dot(np.dot(g, cov), g))
                   for g in gradient])

    c = tstat(confidence, lr.df_resid-1)  # multiplier for confidence interval
    return pd.DataFrame(
        data={
            'prob': prob,
            'CI_hi': np.maximum(0, np.minimum(1, prob + se * c)),
            'CI_lo': np.maximum(0, np.minimum(1, prob - se * c))
        }, index=X.index)


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
    r = pd.DataFrame(index=lr.params.index,
                     columns=['OR', 'CI_lo', 'CI_hi'])
    r.OR = np.exp(lr.params)
    r.CI_lo = np.exp(lr.params - c * lr.bse)
    r.CI_hi = np.exp(lr.params + c * lr.bse)
    return r
