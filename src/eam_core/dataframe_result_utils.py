def group_by(df, pattern_list=None, group_name=None, metadata=None, categories=None):
    """
    Combine process results. The original DataFrame is returned with new columns added and combined columns dropped.
    Only columns that are included in the simulation metadata will possibly be combined into a group.

    If `included_types` is set, only columns of which the device type is included will be considered for grouping.

    :param df:
    :type df: pd.DataFrame
    :param pattern_list: if present, a list of patterns to check against. If a column fits to any of these patterns it is included
    :type pattern_list: List[str]
    :param group_name: the name of the group to create in the dataframe
    :type group_name: str
    :param metadata: Dict
    :type metadata:
    :param included_types: a list of (metadata) device type names to be considered for inclusion
    :type included_types: List[str]
    :return: the original DataFrame, potentially altered with new columns added and combined columns dropped
    :rtype: pd.DataFrame
    """
    ud_cols = []
    for col in df.columns:
        if col == group_name:
            raise Exception(f"Process name and group name identical ({group_name})")

        if not col in metadata:
            # a column might be the result of a previous call to group_by. We do not want to include them in future operations.
            # thus exclude them by checking that the name is present in the simulation metadata
            continue
        md = metadata[col]

        # check if in relevant category
        if not categories or all([ cat_name in md and md[cat_name] == cat_val for cat_name, cat_val in categories.items()]):

            # 'device_type' in md and md['device_type'] in included_types:
            # check the name pattern matches
            if not pattern_list or any(ud_name.lower() in col.lower() for ud_name in pattern_list):
                ud_cols.append(col)

    df[group_name] = df[ud_cols].sum(axis=1)
    ud_df = df.drop(ud_cols, axis=1)
    return ud_df


def group_data(data, metadata, plot_def):
    for group in plot_def['groups']:
        group_name = group['name']

        kwargs = {}
        if 'categories' in group:
            categories = group['categories']
            kwargs.update({'categories': categories})

        data = group_by(data, metadata=metadata, group_name=group_name, **kwargs)
    return data
