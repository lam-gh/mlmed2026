How do I determine the input size?
> Just look at the data

```
ValueError: Input 0 of layer "conv1d" is incompatible with the layer:
expected min_ndim=3, found ndim=2. Full shape received: (None, 186)
```

Following III.A. from 1805.00794
1) Splitting the continuous ECG signal to 10s windows and select a 10s
   window from an ECG signal.
2) Normalizing the amplitude values to the range of between zero and
   one.
3) Finding the set of all local maximums based on zero-crossings of the
   first derivative.
4) Finding the set of ECG R-peak candidates by applying a threshold of
   0.9 on the normalized value of the local maximums.
5) Finding the median of R-R time intervals as the nominal heartbeat
   period of that window (T).
6) For each R-peak, selecting a signal part with the length equal to 1.2T
7) Padding each selected part with zeros to make its length equal to a
   predefined fixed length.

```
File "/Volumes/Reg/Users/faputa/Developer/mlmed2026/prac1/main.py", line 36, in my_model
    x = layers.Conv1D(filters=32, kernel_size=5, activation="relu")(x)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nix/store/jqf98n3m69gf9j20xb6q11v22vzml2gx-devenv-profile/lib/python3.12/site-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/nix/store/jqf98n3m69gf9j20xb6q11v22vzml2gx-devenv-profile/lib/python3.12/site-packages/keras/src/ops/operation_utils.py", line 221, in compute_conv_output_shape
    raise ValueError(
ValueError: Computed output size would be negative. Received `inputs shape=(None, 2, 32)`, `kernel shape=(5, 32, 32)`, `dilation_rate=[1]`.
```

I need to format the data 
