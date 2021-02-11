# helper functions for generating tables

import pandas as pd

def categorical_variables_summary(df, grp_var, vars=None):

    df = df.reset_inex()

    var_tables = {}

    if vars is None:
        vars = df.select_dtypes('category').columns

    grp_labels = df[grp_var].unique()
    grp_counts = df.groupby(grp_var).agg(['count']).iloc[:, 0]
    for col in vars:
        var_counts = (df
            .groupby(grp_var)[col]
            .value_counts()
            .unstack(grp_var)
        )

        var_counts.index = var_counts.index.to_list()

        var_pcts = (var_counts / grp_counts * 100)
        
        var_tables[col] = (pd
            .concat([var_counts, var_pcts], axis=1, keys=['N', '%'])
            .swaplevel(axis=1)
            .loc[:, pd.MultiIndex.from_product([grp_labels, ['N', '%']])]
            .rename_axis('value')
        )
    grp_tables = pd.concat(
        var_tables.values(), axis=0, names=['Q'], keys=var_tables.keys())

    return grp_tables