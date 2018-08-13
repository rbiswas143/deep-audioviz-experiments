# Deep Audio Viz Experiemnts

Conventionally, audio visualizations are created using techniques in Digital Signal Processing.
The approach is limited to the ability of hand-designing audio features.
We address this problem by using Convolutional Deep Neural Network architectures, 
in both supervised and unsupervised learning setups, to extract features from songs 
and explore several techniques for mapping the extracted audio features to visual parameters 
that are used to drive audio visualizations. We have demonstrated the use of autoencoder
for generating visualizations that are dynamic and synchronous with music and further explored 
techniques for improving the quality of the visualizations. We have also shown that
it is possible to use a genre classifier to create visualizations that vary across musical genres. 

* [Read more about our work](https://github.com/rbiswas143/deep-audioviz-experiments/blob/master/rsrcs/paper.pdf)
* [Visit the PlayGround](https://rbiswas143.github.io/deep-audioviz-experiments/)

**Credits** to [BGodefroyFR](https://github.com/BGodefroyFR/Deep-Audio-Visualization) for inspiring us to work in this domain.

## Feature Extractors

![Autoencoder Architectur](https://github.com/rbiswas143/deep-audioviz-experiments/blob/master/rsrcs/ae.png)
An Autoendoer was employed to extract features from small segments of audio tracks. The output of the encoder, of size 10,
is extracted, processed further and finally mapped to parameters that drive audio visualizations. The features obtained
from autoencoders produce dynamic visualizations changing througout the track

![Autoencoder Architectur](https://github.com/rbiswas143/deep-audioviz-experiments/blob/master/rsrcs/alexnet.png)
Genre Classifiers were also used to extract genre specifiec features and create audio visualizations that vary across
musical genres. For ease of computation, features were extracted from the final few layers of the genre classifiers
which are relatively small in size.

## Audio Visualizations

![VIZ-BAR](https://github.com/rbiswas143/deep-audioviz-experiments/blob/master/rsrcs/random2.png)
![VIZ-REAL-1](https://github.com/rbiswas143/deep-audioviz-experiments/blob/master/rsrcs/deterministic-vizreal1.png)
![VIZ-REAL-2](https://github.com/rbiswas143/deep-audioviz-experiments/blob/master/rsrcs/vizreal3.png)

## Code Setup Notes

### Installation
* Clone this Git Repository
* For training, [download](https://github.com/mdeff/fma) the FMA DataSet at the following locations
..* /datasets/fma/fma_metadata
..* /datasets/fma/fma_small
..* /datasets/fma/fma_medium (optional)

#### Training
* **Models** - All models are defined

### Analysis

### Server

### Client
