Convolutional Neural Networks (CNNs) have achieved state-of-the-art results in many computer
vision tasks, including image classification. However, with the increasing complexity of network
architectures and the size of datasets, training CNNs can become computationally expensive and
time-consuming. Therefore, it is essential to identify bottlenecks and optimization opportunities in
the network and training process to improve the performance of CNNs. In this project, we aim to
evaluate the performance of a CNN model for image classification and examine different performance
optimization techniques to assess their value and potential. We investigate ResNets of varying sizes and implementations, 
and study how the execution times and memory footprints are affected as the networks grow larger. We compare ResNet with VGG16,
we profile memory and execution times, and we profile data loading. This is done on CPU as well as on GPU for comparison.

To try our project, execute any of the notebooks in this repository, or simply read them for our results.
Make sure that you read the analysis in our notebooks at the end of every file. Suggested notebook order:
- cpu_forward.ipynb
- gpu_forward.ipynb
- memory_profiling.ipynb
- resnet_profiling.ipynb
- dataloaders.ipynb
- model_training.ipynb

Additionally, see attached trace files for more detailed view of execution and kernel computations.
