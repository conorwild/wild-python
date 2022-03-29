#%%
import cbspython as cbs
from scipy.stats import probplot # for QQ-plots
from cbsdata.sleep_study import SleepStudy as SS
import wildpython as wp
import re

nc = cbs.abbrev_features([t, 'num_correct'] for t in cbs.TESTS.keys())
hc = cbs.abbrev_features(cbs.hc_feature_list())
df = cbs.abbrev_features(cbs.domain_feature_list())

data = (SS
    .score_data()
    .pipe(cbs.abbrev_columns)
    .groupby('user')
    .nth(0)
    .query('&'.join([f"({c} > 1)" for c in nc]))
    .pipe(wp.filter_df, subset=df, sds=[6,4], drop=False)
    .join(SS.questionnaire.data)
    .query('(age >= 18) & (age < 80)')
    .dropna(subset=hc)
)

res, estimated_models = wp.regression_analyses('%s ~ gender * age', df, data)
contrasts = ['age', 'gender', 'age*gender']

r = wp.f_tests(estimated_models, contrasts, adj_across='c', adj_type='bonferroni')
r
