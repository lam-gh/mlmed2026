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

```
Epoch 1/12
1095/1095 ━━━━━━━━━━━━━━━━━━━━ 25s 22ms/step - accuracy: 0.9989 - loss: 0.0104 - val_accuracy: 0.1387 - val_loss: 49.9532
Epoch 2/12
1095/1095 ━━━━━━━━━━━━━━━━━━━━ 21s 19ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 0.1387 - val_loss: 49.9532
Epoch 3/12
1095/1095 ━━━━━━━━━━━━━━━━━━━━ 21s 19ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 0.1387 - val_loss: 49.9532
Epoch 4/12
1095/1095 ━━━━━━━━━━━━━━━━━━━━ 20s 18ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 0.1387 - val_loss: 49.9532
Epoch 5/12
1095/1095 ━━━━━━━━━━━━━━━━━━━━ 21s 19ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 0.1387 - val_loss: 49.9532
Epoch 6/12
1095/1095 ━━━━━━━━━━━━━━━━━━━━ 21s 19ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 0.1387 - val_loss: 49.9532
Epoch 7/12
1095/1095 ━━━━━━━━━━━━━━━━━━━━ 22s 20ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 0.1387 - val_loss: 49.9532
Epoch 8/12
1095/1095 ━━━━━━━━━━━━━━━━━━━━ 23s 21ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 0.1387 - val_loss: 49.9532
Epoch 9/12
1095/1095 ━━━━━━━━━━━━━━━━━━━━ 39s 19ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 0.1387 - val_loss: 49.9532
Epoch 10/12
1095/1095 ━━━━━━━━━━━━━━━━━━━━ 22s 21ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 0.1387 - val_loss: 49.9532
Epoch 11/12
1095/1095 ━━━━━━━━━━━━━━━━━━━━ 25s 22ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 0.1387 - val_loss: 49.9531
Epoch 12/12
1095/1095 ━━━━━━━━━━━━━━━━━━━━ 24s 22ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 0.1387 - val_loss: 49.9531
685/685 - 2s - 3ms/step - accuracy: 0.8276 - loss: 9.9785
Test loss: 9.978506088256836
Test accuracy: 0.8276082873344421
```
