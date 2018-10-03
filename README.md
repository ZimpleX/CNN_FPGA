## A Framework for Generating High Throughput CNN Accelerators

### Approach

To accelerate the inferencing of large-scale convolutional neural networks via a
hardware-software co-design methodology.

* Target platforms:
  * FPGAs (done)
  * GPUs (TODO)
* Algorithmic optimizations:
  * Frequency domain convolution
  * Overlap-and-Add
  * Concatenate-and-Pad
  * Loop tiling / unrolling
  * Complex image computation
* Architectural optimizations:
  * Abstraction of device coefficient
  * Fast design space exploration
* TODO
  * Precision study
  * CNN model classification
  * GPU mapping
* Remaining issues
  * Neural network model compression
  * Sparsity in weight matrices

