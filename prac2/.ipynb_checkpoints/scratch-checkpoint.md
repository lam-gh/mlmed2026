# The dataset

- Bunch of images
- training csv contains pixel size and circum
- test csv contains only pixel
- predicting circumference
- measure the quality by uploading to website
- what the hell is circum for -> that's the target

Input: image
output: pixel_size

I will use a convolution neural network

Regressional problem

first method

- Simple segmentation method
- opencv to find longest
- estimate ellipse


second method

- thresholding 
- deep learning?

I will go with the first method.

- Keras U net
- train using original ultrasound and annotated as truth
- use this model to predict the mask for images in the test set
- find the HC by calculate_hc(mask, pixel_size)

My training is too slow!!! 

What the hell is THIS!? 
```
 35%|█████████████████████████████████████████████▌                                                                                    | 14/40 [1:14:49<3:37:09, 501.13s/it]

[INFO] EPOCH: 14/40
Train loss: 0.044076, Test loss: 0.0434

 38%|████████████████████████████████████████████████▊                                                                                 | 15/40 [1:17:17<2:44:29, 394.77s/it]

[INFO] EPOCH: 15/40
Train loss: 0.044082, Test loss: 0.0432

 40%|████████████████████████████████████████████████████                                                                              | 16/40 [1:19:45<2:08:14, 320.59s/it]

[INFO] EPOCH: 16/40
Train loss: 0.043987, Test loss: 0.0432

 42%|███████████████████████████████████████████████████████▎                                                                          | 17/40 [1:22:14<1:43:04, 268.89s/it]

[INFO] EPOCH: 17/40
Train loss: 0.043862, Test loss: 0.0432
```

Potential cause??? 

I asked Gemini, with context of the above snippet, and that I am running segmentation task, unet architecture, and on a mac mini. 

Per Gemini: 

1. Memory swapping
    * batch_size=6; already small value
    * already using DataLoader
3. not using GPU
    * already using mps
5. Data loading bottleneck
    * num_workers=cpu_count()
7. Thermal throttling
    * not hot to touch
9. Loss is flat
    * using BCEWithLogitsLoss
  
Given these information, Gemini suggested to use num_workers=0

But whyyyyyyyy??? 

Because macOS uses `spawn`.
> Changed in version 3.8: On macOS, the spawn start method is now the default. The fork start method should be considered unsafe as it can lead to crashes of the subprocess as macOS system libraries may start threads. See bpo-33725.

ref: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods

But why does this matter? 

`spawn` starts a fresh Python interpreter and imports everything, slows things down! 




```
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[25], line 9
      6 unet = torch.load(MODEL_PATH, weights_only=False).to(DEVICE)
      8 for path in imagePaths:
----> 9     make_predictions(unet, path)

Cell In[24], line 24, in make_predictions(model, imagePath)
     21 image = np.expand_dims(image, 0)
     22 image = torch.from_numpy(image).to(DEVICE)
---> 24 predMask = model(image).squeeze()
     25 predMask = torch.sigmoid(predMask)
     26 predMask = predMask.cpu().numpy()

File ~/Developer/mlmed2026/.devenv/state/venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1776, in Module._wrapped_call_impl(self, *args, **kwargs)
   1774     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1775 else:
-> 1776     return self._call_impl(*args, **kwargs)

File ~/Developer/mlmed2026/.devenv/state/venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1787, in Module._call_impl(self, *args, **kwargs)
   1782 # If we don't have any hooks, we want to skip the rest of the logic in
   1783 # this function, and just call forward.
   1784 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1785         or _global_backward_pre_hooks or _global_backward_hooks
   1786         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1787     return forward_call(*args, **kwargs)
   1789 result = None
   1790 called_always_called_hooks = set()

Cell In[7], line 20, in UNet.forward(self, x)
     18 def forward(self, x):
     19     encFeatures = self.encoder(x) 
---> 20     decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])
     21     mapper = self.head(decFeatures)
     23     if self.retainDim: 

File ~/Developer/mlmed2026/.devenv/state/venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1776, in Module._wrapped_call_impl(self, *args, **kwargs)
   1774     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1775 else:
-> 1776     return self._call_impl(*args, **kwargs)

File ~/Developer/mlmed2026/.devenv/state/venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1787, in Module._call_impl(self, *args, **kwargs)
   1782 # If we don't have any hooks, we want to skip the rest of the logic in
   1783 # this function, and just call forward.
   1784 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1785         or _global_backward_pre_hooks or _global_backward_hooks
   1786         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1787     return forward_call(*args, **kwargs)
   1789 result = None
   1790 called_always_called_hooks = set()

Cell In[6], line 17, in Decoder.forward(self, x, encFeatures)
     14 for i in range(len(self.channels) - 1):
     15     x = self.upconvs[i](x) 
---> 17     encFeat = self.crop(encFeatures[i], x)
     18     x = torch.cat([x, encFeat], dim=1)
     19     x = self.dec_blocks[i](x) 

Cell In[6], line 24, in Decoder.crop(self, encFeatures, x)
     23 def crop(self, encFeatures, x):
---> 24     (_, _, H, W) = x.shape
     25     encFeatures = CenterCrop([H, W])(encFeatures)
     27     return encFeatures

ValueError: not enough values to unpack (expected 4, got 3)
```

I dont't have enough dimension. 

Should I increase dimension? 

I didn't have dimension

Gemini suggested `unsqueeze` to expand dimension. 

This has worked successfully

```
image.unsqueeze(0).unsqueeze(0)
```