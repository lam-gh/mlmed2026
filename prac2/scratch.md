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

My masks show NOTHING of value!!! Absolute TOTAL PITCH DARK! Whyyyy~~~~~~~

I asked Gemini :DDDD 

Gemini says... 
> Dude your loss function is ass.
> You shouldn't be using BCEWithLogitsLoss
> You should use Dice Loss instead
> It's not implemented so DIY!
> Also, dilate the pixels of ground truth mask in the first few iteration so the model has a larger target to hit

BCEWithLogitsLoss: Binary Cross Entropy With Logits Loss
Dice Loss: 

Me thinks... 
- Why not BCEWithLogitsLoss?
    - It performs badly on semantic segmentation tasks because of class imbalance.
    - most of the time we only have 10~20% of target against majority background.
    - this loss function will just default to pitch black 
- Why Dice Loss?
    - solves the above problems
    - it combines 
- How to dilate the pixels?
- When do I stop dilation?

ref: https://www.kaggle.com/discussions/getting-started/133156
ref: https://arxiv.org/html/2312.05391v1
ref: https://discuss.pytorch.org/t/implementation-of-dice-loss/53552

```
class DiceLoss(Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        smooth = self.smooth
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
```

HAHAHA Dice LOSS FAILS COMPLETELY 😭

Maybe my implementation is wrong? 

Also, why isn't it merged into main yet!? ref: https://github.com/pytorch/pytorch/issues/1249

I asked Gemini what's wrong with the implementation above: 
- No activation
- expecting probabilities as input

What I should do to make it work properly:
- Add sigmoid
- expect logits as input 

I tried these, and realized the first implemetation is a squared dice too. 
So far, its still showing disappointing result. 

I should try dilating the pixels next 

Gemini suggested the following: 

```
# inside Dataset __getitem__
kernel = np.ones((3, 3), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)
```

After all that... 

```
[INFO] EPOCH: 40/40
Train loss: 0.809125, Test loss: 0.8432
[INFO] total time taken to train the model: 186.36s
```

This is so ass ... 

I asked Gemini what the problem might be...

- shrinking image whenever passing 3x3 convolution...
    - solution: add padding
- lack of normalization
    - because the image I use have high variance in intensity and noise,
    - without batchnorm, the gradient explode and vanish (what?)
    - solution: add batchnorm
- class imbalance
    - I should use both BCEWithLogitsLoss and Dice loss

Consider adding more layers 
- (1, 64, 128, 256, 512)
- (1, 16, 32, 64)
- (512, 256, 128, 64)
- (64, 32, 16)

**THERE ARE MORE PROBLEMS!!!**

1. SegmentationDataset class is using transform.Resize(). which..... **USES interpolation=InterpolationMode.BILINEAR**!!!!
    - Issues here, bilinear interpolation blurs the pixels,
    - so my thin thin ring of circumference now just disappears when I do `ToTensor()`
2. Not only that, I should separate transfrom for base and mask.
3. And add back dilation!

Adding pos_weight to BCE to help with class imbalance 

```
pos_weight = torch.tensor([50]).to(DEVICE)
```

Okay, I actually have something in the output now. I think I can move onto "skeletonizing" the output mask and estiamting an ellipse on it?

It's actually still very ass.

How ABOUT !!!!

- Fill in the ring to get a circle,
- learn that circle,
- then get the ring in post processing
- ???

--------------------------------------------------------------------

Organize myself... 

- Task: Head circumference prediction given ultrasound image.
- How to get there?
    - Segment first to find the skull.
        - Ground truth mask is a bunch of rings of head circumference.
        - It's hard to predict just the ring (not many target)
        - Fill inside the ring to get blob
        - Then train to predict the blob
        - It's easy to get the ring from the blob now
        - Erode the blob
        - Get the difference of the blob and the erosion
        - that is the ring!
        - Use cv2 to estimate the ellipse on this ring
        - find the circumference. <- this is what we want.

- HC18 dataset:
    - 999 original image, 999 (annotated) = 1998 total
    - 335 images in test set
    - images are 540x800 (height, width)
    - masks pixel values are 0 or 255 (binary)
    - original image value is between 0, 255

- My current architecture:
    - Block
        - Conv2d
        - BatchNorm2d
        - ReLU
        - Conv2d
        - BatchNorm2d
    - Encoder
        - Block (1, 64)
        - MaxPool2d
        - Block (64, 128)
        - MaxPool2d
        - Block (128, 256)
        - MaxPool2d
        - Block (256, 512)
        - MaxPool2d
    - Decoder
        - UpConv
        - Crop
        - cat
        - Block

- Training:
    - Loss func: BCEWithLogitsLoss
    - Optimizer: Adam
    - NUM of EPOCH: 40
    - BATCH size: 8


I asked Gemini to identify the causes of failure with my current setup: 

> Resize (smoothing loses information): solved
- Used two different transformation for original image and for mask
> Not enough block in Decoder: solved
- added another block in Decoder
> Missing non linearity: solved
- added another ReLU after bn2 in Block
> Poor loss function choice:
    > BCE... might predict all black
- This isn't a problem because I fill in the circumference so the model has a bigger target to hit.
- Another problem arises is that now the model just predict a big white square in the middle instead of an ellipse
Current loss:
```
[INFO] EPOCH: 40/40
Train loss: 0.131643, Test loss: 0.4412
[INFO] total time taken to train the model: 430.81s
```
> Lack of generalization
    > Add random rotational augmentation to image
- If I want to do this I need to use the same transformation for both base and mask
- However, my current transforms are separated because Resize needs to use different interpolation
- I need to unify this transform so the same random transformation is applied to both base and mask
> Ellipse fitting instability:
    > neglegible, forget about it.
> Geometric Disregard in Loss Function
- Then I will use Dice Loss instead. This should fix it.

I changed the loss function to Dice Loss 
```
[INFO] EPOCH: 40/40
Train loss: 0.047851, Test loss: 0.2060
[INFO] total time taken to train the model: 535.87s
```
Loss looks a bit more promising 
But it still shows a big white square. 
I need to get the segmentation part correctly to calculate the circumference arithmetically. Only discuss the segmentation part for now. 

> Arithmetic incompatibility Input

What should I do about this?

> About transformation

I will apply resize outside the transformation function. 
And define a unified transformation for both original and mask with the following method
```
base, mask = transforms(base, mask)
```

Well this SUCKS! 

- I am going to explicitly define each layer instead of using Blocks
- I get more fine grain control of dimensions

```
class Block(Module): 
    def __init__(self, inChannels, outChannels): 
        super().__init__()
        self.conv1 = Conv2d(inChannels, outChannels, 3, padding=1)
        self.bn1 = BatchNorm2d(outChannels)
        self.relu1 = ReLU() 
        self.conv2 = Conv2d(outChannels, outChannels, 3, padding=1)
        self.bn2 = BatchNorm2d(outChannels)
        self.relu2 = ReLU() 

        self.net = Sequential(
            self.conv1,
            self.bn1,
            self.relu1,
            self.conv2,
            self.bn2,
            self.relu2
        )

    def forward(self, x): 
        return self.net(x)
```

Try focal loss 

```
Cell In[34], line 73, in UNet_ALT.forward(self, x)
     71 # Decode
     72 xu1 = self.upconv1(xe52)
---> 73 xu11 = torch.cat([xu1, xe42], dim=1)
     74 xd11 = relu(self.d11(xu11))
     75 xd12 = relu(self.d12(xd11))

RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 24 but got size 16 for tensor number 1 in the list.
```

```
Cell In[50], line 73, in UNet_ALT.forward(self, x)
     71 # Decode
     72 xu1 = self.upconv1(xe52)
---> 73 xu11 = torch.cat([xu1, xe42], dim=1)
     74 xd11 = relu(self.d11(xu11))
     75 xd12 = relu(self.d12(xd11))

RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 9 but got size 10 for tensor number 1 in the list.
```

```
Cell In[68], line 73, in UNet_ALT.forward(self, x)
     71 # Decode
     72 xu1 = self.upconv1(xe52)
---> 73 xu11 = torch.cat([xu1, xe42], dim=1)
     74 xd11 = relu(self.d11(xu11))
     75 xd12 = relu(self.d12(xd11))

RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 66 but got size 67 for tensor number 1 in the list.
```

My training strategy sucks ass 

Some of the caveat of working with medical images:

- avoid downscale images and reduce resolution because lose detail :(
- avoid slimming down the model because can't learn !!!

Then what the hell do we dooooo??? 

Gemini response:
> Patch based training
> Gradient Accumulation
> Use ResNet as encoder instead of normal encoder

> Clearly it's a pain in the ass to train from scratch with no libraries.
> Fine, I'll use SMP


Albumentation docs: https://explore.albumentations.ai/transform/ShiftScaleRotate/docs

file name is wrong for mask in block