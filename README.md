# max_cnn

Simplest example CNN design for Maxeler Summer School 2017.

Please clone this repository and import it into your MaxIDE workspace.

## Design Details

There are one convolution layer (implemened as `MaxCNNConvKernel`) and one pooling layer (implemented as `MaxCNNPoolKernel`)
in this design.
The pooling kernel is appended after the convolution layer.

The convolution layer can take a `10 x 10` image as input, and the pooling layer takes `8 x 8`.
The convolution layer has a `3 x 3` kernel filter, which is initialized through scalar input.
