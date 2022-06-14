# Internal Datastructures

We are using pandas multiindex DataFrames throughout, in order to represent time and sampled random variables.

Eg.
```text
time        samples
2016-01-01  0          2
            1          2
2016-02-01  0          2
            1          2
2016-03-01  0          2
            1          2
2016-04-01  0          2
            1          2
2016-05-01  0          2
            1          2
2016-06-01  0          2
            1          2
2016-07-01  0          2
            1          2
2016-08-01  0          2
            1          2
2016-09-01  0          2
            1          2
2016-10-01  0          2
            1          2
2016-11-01  0          2
            1          2
2016-12-01  0          2
            1          2
2017-01-01  0          2
            1          2
Name: test, dtype: int64
```

For unit support, we wrap the DFs into [pint](https://github.com/hgrecco/pint) Quantities.

Pint converts the DFs into numpy ndparrays (bug report [here](https://github.com/hgrecco/pint/issues/678)). Thus, after 
sampled random variables have been turned into `Quantities`, assume that you work with np arrays that are flattended multiindex DFs.

When the results are stored to disc, they are converted back to pandas multiindex DFs. 