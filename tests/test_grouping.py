import unittest
import pandas as pd
import numpy as np
import pint

from eam_core.dataframe_result_utils import group_data


class GroupingTestCase(unittest.TestCase):

    def test_basic(self):
        start_date = '2010-01-01'
        end_date = '2010-02-01'
        date_range = pd.date_range(start_date, end_date, freq='MS')
        samples = 2
        iterables = [date_range, range(samples)]
        index_names = ['time', 'samples']
        _multi_index = pd.MultiIndex.from_product(iterables, names=index_names)

        A = pd.Series(np.full((len(date_range), samples), 1).ravel(), index=_multi_index)
        B = pd.Series(np.full((len(date_range), samples), 2).ravel(), index=_multi_index)
        C = pd.Series(np.full((len(date_range), samples), 2).ravel(), index=_multi_index)

        data = pd.DataFrame({'A': A, "B": B, "C": C})

        metadata = {'A': {'some category': 'type 1'}, 'B': {'some category': 'type 2'},
                    'C': {'some category': 'type 2'}}
        plot_def = {
            "name": "all_area",
            "variable": "energy",

            "groups": [
                {
                    "name": "Type 1",
                    "categories": {
                        "some category": "type 1"
                    }
                },
                {
                    "name": "Type 2",
                    "categories": {
                        "some category": "type 2"
                    }
                }
            ]
        }

        grouped_data = group_data(data, metadata, plot_def)

        assert set(grouped_data.columns) == set(['Type 1', 'Type 2'])
        assert (grouped_data['Type 1'].ravel() == 1).all()
        assert (grouped_data['Type 2'].ravel() == 4).all()

        if __name__ == '__main__':
            unittest.main()
