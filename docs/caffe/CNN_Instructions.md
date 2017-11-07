# CNN Conversion Instructions

Instructions from here are taken from [Andrew
Janowyzck](http://www.andrewjanowczyk.com/efficient-pixel-wise-deep-learning-on-large-images/).
This document provides a concise TL;DR to that blog post.

Preliminaries: 
- You've already trained a patch-based classifier using Digits on the server.
- Download the model by clicking on its entry in the main "Models" tab, then
  scrolling down past the performance curves and click on the button that says
  "Download Model". This will save a `.tar.gz` file that you can unzip and
  untar.
- Also scroll up to the top of this page, under `Job Directory`, and click on
  the link that says `caffe_output.log` (the last entry in that box). Save it to
  the same directory you just saved the model to.

Procedure:
1. Copy deploy.prototxt to deploy_full.prototxt
2. Find and Replace "fc6", "fc7", "fc8" with "fc6-conv", "fc7-conv", "fc8-conv",
   respectively.
3. Find and Replace "InnerProduct" with "Convolution"
4. Find and Replace "inner_product_param" with "convolution_param"
5. Add "kernel_size: 1" to "convolutional_param for fc7-conv and fc8-conv
6. Look at the output log for the "Top shape: 128 32 6 6" (or whatever) before
   fc6 -- final layer of convolutional body. Last two numbers are the size of
   the kernel; add that to fc6-conv in "convolutional_param": "kernel_size: 6"
   (in this example)
7. Change input shape at top to the size of the desired input image, plus the
   trained patch size
8. Ensure the final layer is a softmax layer for probability:

layer {
    name: "prob"
    type: "Softmax"
    bottom: "fc8-conv"
    top: "prob"
}

9. Run the notebook "Convert Patch to FCN", adjusting the model directory and
   name of the caffe model "snapshot_iter_xxxxx.caffemodel" as needed

- This will transpose the weights from the old model into the new
  fully-connected one
