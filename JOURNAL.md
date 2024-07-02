## 7/1/24

No progress since Winter.
After looking through Skia's [FEConvolveMatrix source](https://github.com/google/skia/blob/main/src/effects/imagefilters/SkMatrixConvolutionImageFilter.cpp), there are definitely shenanigans when kernal is large.

Theories:

- SVG is fine, but some parsing in Chromium/Skia is messing up kernel and thus the outputs
- Upstream: The actual MNIST images have something wrong with them
