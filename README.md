## HMS - Harmful Brain Activity Classification

[This](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview) Kaggle Competition asked the competitors to classify EEG and Spectrogram segments across six categories of harmful brain activity. Thank you to Kaggle, Harvard Medical School, Sunstella Foundation, Jazz Pharmaceuticals, and the Clinal Data Animation Center (CDAC) for sponsoring the competition.

My goal from the beginning was to focus on only the raw EEG data. This way, I could learn the state-of-the-art of signal processing, including preprocessing techniques (Wavelet transforms, Butterworth filtering, etc.) and 1D deep learning models (e.g., ResNet, WaveNet, GRU, etc.). My final EEG model architecture was a ResNet model, followed by a GeM Pooling layer, a GRU, and a fully-connected classification head. After ensembling with some publicly-available spectrogram models, I was able to finish 387/2767, within the top 14%.

This repository excludes the EEGs and spectrograms themselves and other preprocessed data. On my local machine, the full folder is 35GB.

## Preprocessing and Training

See [DEFINITIONS.md](./DEFINITIONS.md) for information on the labels. Our goal was to predict class probabilities across the six labels in order to minimize the Kullback-Leibler Divergence (a ubiquitous function in information theory measuring the information gain or "surprise" on learning the true distribution given some other distribution; here, this other distribution is the model's output).

The labels given are integers corresponding to the number of experts who believed the data showed the type of harmful activity represented by that label. The problem, however, was that the number of experts voting, in total, ranged from 3-20. Note that a mixed response example across 20 votes is less likely to be mixed when any 3 votes are taken. Likewise, we can be much more confident with 20 votes all pointing to one label, whereas 3 votes all pointing to one label is less convincing. To address this, I used 2-stage training: for the first stage, the labels were averaged with constant probabilities of 1/6 to make the labels less confident; in the second stage, I used the true labels.

For preprocessing, I used Butterworth low-pass and high-pass filters with the scipy signals module. The frequency ranges that remained were comfortably within the diagnostic frequencies of the harmful brain activity I aimed to classify. I also tested Morlet Wavelets, but this did not perform better. I used the "double banana" EEG montage along the four principal chains as input to my model, and I downsampled by a factor of 5, i.e. 10,000 observations (50 seconds at 200 Hz) became 2,000 observations (50 seconds at 40 Hz). I performed Butterworth filtering prior to downsampling to better capture the essential frequencies.

## Model Notes

### [WaveNet](https://doi.org/10.48550/arXiv.1609.03499)*
This deep learning model was the first to show real promise on the raw EEG data; I saw it first in [this notebook](https://www.kaggle.com/code/cdeotte/wavenet-starter-lb-0-52) of [Chris Deotte](https://www.kaggle.com/cdeotte). WaveNet came out of Google in 2016 and was an important leap in deep learning for audio data. Most notably, it led to major improvements in the naturalness of artificially-generated speech. The model consists of residual blocks with convolutional kernels of increasing dilations, and these blocks are stacked together with increasing channel sizes, as is common with CNNs. Each dilation in a residual block is really two: a tanh-activated filter and sigmoid-activated filter. These are multiplied together (sigmoid acts as a soft gate for the tanh-activated filter) and added to the residual stream. 

One WaveNet model was used for all EEG streams, in parallel, followed by global average pooling and a fully-connected classification head.

While this approach works, training is slow and infeasible on a single GPU for more than, say, 10 epochs for 5 folds.

Personally, it seems wrong that a model well-suited for producing and understanding speech would be a good choice for EEG classification. The former task needs to observe much more complicated features and much more of them. The task of identifying EEG patterns is more about counting relatively simple, consistent patterns over a certain interval of time. To test this idea, I changed the last residual block's Tanh activation to ReLU to allow for "counting". With all things equal, this tiny change showed some improvement.

The important advantage of WaveNet compared to other CNN architectures was its sensitivity to both the time and frequency domains and its ability to see large scales. This is due to its stacked dilations.

### [Graph WaveNet](https://doi.org/10.48550/arXiv.1906.00121)*

Inspired by the previous WaveNet model, I wanted a model that could see long-range temporal dependencies, but further, could learn deep connections between the EEG nodes.

Graph WaveNet allows nodes of homogeneous temporal data to be integrated into a graph convolutional network (GCN). Each node is processed by a WaveNet architecture, and the subsequent "features" are taken as input to the GCN. The model is given a graph during initialization (I used the EEG 10-20 graph), but it also has an adaptive adjacency matrix for further connectivity information to be learned.

The Graph WaveNet trained very slowly, and there was no evidence in terms of performance that larger, more costly training runs would be worthwhile.

### [ResNet GRU](https://www.kaggle.com/code/nischaydnk/hms-submission-1d-eegnet-pipeline-lightning)*

The ResNet GRU architecture worked very well and was the brainchild of [Nischay Dhankhar](https://www.kaggle.com/nischaydnk). The EEG streams pass, in parallel, through the ResNet architecture and then through a bi-directional GRU. The outputs *and final hidden states* are concatenated and fed through the fully-connected, dense, classification head.

Processing real-world EEG data for classification happens in two parts. First, the data is very noisy, so a good proportion of the model should be spent cleaning and smoothing the data. This is where ResNet shines; it slowly compresses the data by smoothing over unimportant details and accentuating predictive attributes. Second, classification happens by counting certain waveforms over a particular duration. GRU accomplishes this easily. The hidden state can easily store long-term frequency information. This model performed exceptionally well and was the backbone for my final model.

### ResNet GRU with Generalized Mean (GeM) Pooling

In the ResNet GRU mentioned above, output of the ResNet portion is downsampled before being fed into the GRU. This was done with an average pooling layer.

Because the data coming from the ResNet block is the residual stream of the input data, my hypothesis was that the channels would each represent certain features whose intensity would vary along the signal. For rare or sparse features, a max pool may work better because "is present" is more useful than "average presence". Likewise, features that are dense or common may benefit more from an average pool. With this intuition, I added a Generalized Mean Pooling layer.

The GeM Pooling method comes from the $L_p$ norm in real analysis, defined as 

$$||v||_p = \left( \sum_{i=1}^n |v_i|^p \right)^{1/p}$$

for a vector $v = (v_1, \ldots, v_n)$. As $p \to \infty$, $||v||_p \to \max_i \{|v_i|\},$
and for $p = 1$, $||v||_1$ is just the sum, which is proportional to the average. By making $p$ learnable and initializing it with a value of, say, 2, we allow each channel to pool in a flexible, learnable way, while only adding 1 new parameter per channel.

This performed better than the ResNet GRU with average pooling. On the private leaderboard, it single-handedly allowed me to break a final KL-Divergence score of 0.38.

## Files

**DEFINITIONS.md**:
These are my notes from reading [Hirsch et al.](https://doi.org/10.1097/WNP.0000000000000806) I write out the explicit criteria for classifying EEG data.

**EDA**:
This folder contains numerous notebooks where I experiment with different ways of representing the data, understand class differences, try basic methods, etc.

**cleaning.ipynb**:
This notebook is where I used parallel processing to clean/preprocess the EEGs for local storage.

**paths.py**:
A simple class to store global variables.

**hjorth.py**:
A simple class to generate Hjorth parameters, which I did not ultimately use. Motivating discussion with personal comments available [here](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/466718#2629094).

**resnet-gru.ipynb**:
This is the ResNet-GRU with GeM pooling training notebook from Kaggle. The code won't all work locally, but it's here for viewing.

**csv data/train.csv**:
This csv file gives ids, eeg ids, patient ids, eeg offset for labelling, the label votes, as well as spectrogram information. From this, we have the information to find and slice the EEG for any particular observation.

## References

[[1]](https://kaggle.com/competitions/hms-harmful-brain-activity-classification) Jin Jing, Zhen Lin, Chaoqi Yang, Ashley Chow, Sohier Dane, Jimeng Sun, M. Brandon Westover. (2024). HMS - Harmful Brain Activity Classification. Kaggle.

[[2]](https://doi.org/10.1097/WNP.0000000000000806) Hirsch, L. J., Fong, M. W. K., Leitinger, M., LaRoche, S. M., Beniczky, S., Abend, N. S., Lee, J. W., Wusthoff, C. J., Hahn, C. D., Westover, M. B., Gerard, E. E., Herman, S. T., Haider, H. A., Osman, G., Rodriguez-Ruiz, A., Maciel, C. B., Gilmore, E. J., Fernandez, A., Rosenthal, E. S., Claassen, J., … Gaspard, N. (2021). American Clinical Neurophysiology Society's Standardized Critical Care EEG Terminology: 2021 Version. *Journal of clinical neurophysiology : official publication of the American Electroencephalographic Society, 38*(1), 1–29.

[[3]](https://doi.org/10.48550/arXiv.1512.03385) Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. (2015). Deep Residual Learning for Image Recognition. arXiv.

[[4]](https://doi.org/10.48550/arXiv.1609.03499) Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu. (2016). WaveNet: A Generative Model for Raw Audio. arXiv.

[[5]](https://doi.org/10.48550/arXiv.1906.00121) Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Chengqi Zhang. (2019). Graph WaveNet for Deep Spatial-Temporal Graph Modeling. arXiv.
