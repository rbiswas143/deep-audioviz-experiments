# Deep Audio Viz Experiemnts

Conventionally, audio visualizations are created using techniques in Digital Signal Processing.
The approach is limited to the ability of hand-designing audio features.
We address this problem by using Convolutional Deep Neural Network architectures, 
in both supervised and unsupervised learning setups, to extract features from songs 
and explore several techniques for mapping the extracted audio features to visual parameters 
that are used to drive audio visualizations. We have demonstrated the use of autoencoder
for generating visualizations that are dynamic and synchronous with music and further explored 
techniques for improving the quality of the visualizations. We have also shown that
it is possible to use a genre classifier to create visualizations that vary across musical genres

* [Read more about our work](https://github.com/rbiswas143/deep-audioviz-experiments/blob/master/rsrcs/paper.pdf)
* [Visit the PlayGround](https://rbiswas143.github.io/deep-audioviz-experiments/)


## Feature Extractors

![Autoencoder Architectur](https://github.com/rbiswas143/deep-audioviz-experiments/blob/master/rsrcs/ae.png)
An Autoendoer was employed to extract features from small segments of audio tracks. The output of the encoder, of size 10,
is extracted, processed further and finally mapped to parameters that drive audio visualizations. The Autoencoder architecture
has been experimented i three setups: normal, shared weights between the encoder and decoder, and skip connections from the encoder
to the decoder. The features obtained from Autoencoders produce dynamic visualizations changing throughout the track

![Autoencoder Architectur](https://github.com/rbiswas143/deep-audioviz-experiments/blob/master/rsrcs/alexnet.png)
Genre Classifiers were also used to extract genre-specific features and create audio visualizations that vary across
musical genres. For ease of computation, features were extracted from the final few layers of the genre classifiers
which are relatively small in size. Pre-trained state-of-the-art models lke VGG-16 and AlexNet have been used

## Audio Visualizations

![VIZ-BAR](https://github.com/rbiswas143/deep-audioviz-experiments/blob/master/rsrcs/random2.png)
![VIZ-REAL-1](https://github.com/rbiswas143/deep-audioviz-experiments/blob/master/rsrcs/deterministic-vizreal1.png)
![VIZ-REAL-2](https://github.com/rbiswas143/deep-audioviz-experiments/blob/master/rsrcs/deterministic-vizreal3.png)

## Code Setup Notes

#### Installation
* Clone this Git Repository
* Create a Python [Virtual Environment](https://docs.python.org/3/tutorial/venv.html)
and install all dependencies from _reqirements.txt_

#### Data Pre-Processing
* The FMA DataSet can be converted into MFCCs or kept as audio frames for end to end training.
The DataSet can, optionally, be split into partitions if it is too big to fit in memory
* All pre-processing code is defined in _data_processor.py_
* [Download](https://github.com/mdeff/fma) the FMA DataSet at the following locations
  * /datasets/fma/fma_metadata
  * /datasets/fma/fma_small
  * /datasets/fma/fma_medium (optional)
* Create a JSON file to configure pre-processing (See example files at _datasets/processed/samples_)
* Run pre-processing using the CLI provided by _data_processor.py_

#### Training

* Two kinds of models available, CNN Classifier and CNN Autoencoder, with a variety of
configurable options in each. All models are defined in _models.py_
* Create a JSON file to configure training (See example files at _models/test_)
* Run training using the CLI provided by _train.py_
* For hyper-parameter tuning, use the helpers provided in _helpers.py_ to  auto-generate 
hyper-tuning configurations and train a batch of models using the CLI provided by
_hp_tune.py_

#### Analysis
* Experiments on all the trained models and all feature mapping techniques are available in
the Ipython Notebook, _analysis.ipynb_
* During analysis, an analysis directory is created in the respective model directory where all
the results are saved
* The feature mapping utilities have also been defined in _mapping_utils.py_

#### PlayGround Server
* A playground has been created where you can upload songs or choose samples form the FMA DataSet
to visualize them
* Features are extracted in the server in real-time and processed using the feature mapping techniques
* Execute _api.py_ to run the PlayGround Server

#### Client
* The PlayGround code has been built upon [three-seed](https://github.com/edwinwebb/three-seed/)
It is written in ES6, managed with NPM and bundled with WebPack
* [three.js](https://threejs.org/) has been used to create all the visualizations
* The code resides in [demo](https://github.com/rbiswas143/deep-audioviz-experiments/blob/master/demo)

### Credits

* [BGodefroyFR](https://github.com/BGodefroyFR/Deep-Audio-Visualization), for inspiring us to work in this domain
* Amanda Glosson, for the visualization [Spinning Ball of Crystal](https://codepen.io/aglosson/pen/rVyRGm)
* Sean Dempsy, for the visualization [Three Js Point Cloud Experiment](https://codepen.io/seanseansean/pen/EaBZEY)
