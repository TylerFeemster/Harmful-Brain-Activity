## HMS - Harmful Brain Activity Classification

The link to this Kaggle competition is [here](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview).  

I finished 387/2767, within the top 14%.

## Model Notes

### [WaveNet](https://doi.org/10.48550/arXiv.1609.03499)
This deep learning model was the first to show real promise on the raw EEG data; I saw it first in [this notebook](https://www.kaggle.com/code/cdeotte/wavenet-starter-lb-0-52) of [Chris Deotte](https://www.kaggle.com/cdeotte). WaveNet came out of Google in 2016 and was an important step in the deep learning of audio data. Most notably, it led to major improvements in the naturalness of artificially-generated speech. The model consists of residual blocks with convolutional kernels of increasing dilations, and these blocks are stacked together with increasing channel sizes, as is common with CNNs. Each dilation in a residual block is really two: a tanh-activated filter and sigmoid-activated filter. These are multiplied together (sigmoid acts to softgate the tanh-activated filter) and added to the residual stream.

While this approach works, training is slow and infeasible on a single GPU for more than, say, 10 epochs for 5 folds.

Personally, it seems wrong that a model well-suited for producing and understanding speech would be a good choice for EEG classification. The former task needs to observe much more complicated features and much more of them. The task of identifying EEG patterns is more about counting relatively simple, consistent patterns over a certain interval of time. To test this idea, I changed the last residual block's Tanh activation to ReLU to allow for "counting". With all things equal, this tiny change showed some improvement.

The important advantage of WaveNet compared to other CNN architectures was its sensitivity to both the time and frequency domains and its ability to see large scales. This is due to its stacked dilations.

### GraphWaveNet

### ResNet

### ResNet GRU

### ResNet GRU with Geometric Pooling

## References

[[1]](https://kaggle.com/competitions/hms-harmful-brain-activity-classification) Jin Jing, Zhen Lin, Chaoqi Yang, Ashley Chow, Sohier Dane, Jimeng Sun, M. Brandon Westover. (2024). HMS - Harmful Brain Activity Classification. Kaggle.

[[2]](https://doi.org/10.1097/WNP.0000000000000806) Hirsch, L. J., Fong, M. W. K., Leitinger, M., LaRoche, S. M., Beniczky, S., Abend, N. S., Lee, J. W., Wusthoff, C. J., Hahn, C. D., Westover, M. B., Gerard, E. E., Herman, S. T., Haider, H. A., Osman, G., Rodriguez-Ruiz, A., Maciel, C. B., Gilmore, E. J., Fernandez, A., Rosenthal, E. S., Claassen, J., … Gaspard, N. (2021). American Clinical Neurophysiology Society's Standardized Critical Care EEG Terminology: 2021 Version. *Journal of clinical neurophysiology : official publication of the American Electroencephalographic Society, 38*(1), 1–29.

[[3]](https://doi.org/10.48550/arXiv.1512.03385) Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. (2015). Deep Residual Learning for Image Recognition. arXiv.

[[4]](https://doi.org/10.48550/arXiv.1609.03499) Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu. (2016). WaveNet: A Generative Model for Raw Audio. arXiv.