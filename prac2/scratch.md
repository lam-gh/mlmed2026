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

