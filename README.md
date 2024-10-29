# Fault Detection in Gearbox Vibration Data ISAV_2023


## Context

This repository addresses the scarcity of mechanical engineering datasets tailored for applying Machine Learning techniques in an industrial environment. The dataset provided here was not previously available on Kaggle, making it a valuable resource for the community.

## Content

The [Gearbox Fault Diagnosis dataset](https://www.kaggle.com/datasets/brjapon/gearbox-fault-diagnosis) consists of vibration data recorded using SpectraQuest's Gearbox Fault Diagnostics Simulator. The dataset captures vibrations using four sensors placed in different directions under varying loads from 0% to 90%. It encompasses two distinct scenarios:

1. **Healthy Condition**
2. **Broken Tooth Condition**

There are 20 files, with 10 corresponding to a healthy gearbox and ten from a gearbox with a broken tooth. Each file corresponds to a specific load, ranging from 0% to 90% in 10% increments.

## Repository Structure

This repository is organized into two main branches, each addressing a different approach to fault detection:

### 1. LSTM Branch

This branch focuses on applying Long Short-Term Memory (LSTM) networks for fault detection. LSTMs are a recurrent neural network well-suited for sequential data, making them a promising choice for analyzing time series vibration data.

### 2. Continuous Wavelet Transform with CNN Branch

This branch employs a combination of Continuous Wavelet Transform (CWT) and Convolutional Neural Networks (CNNs) for fault detection. The CWT provides a time-frequency representation of the data, which is then fed into a CNN for feature extraction and classification.

## Getting Started

For detailed instructions on setting up and running the code in each branch, please look at the respective branch's README.md file.

## Citation

If you find this work helpful or build upon it in your research, please consider citing the following paper:

```
[Navidreza Ghanbari, Yassin Riyazi, Farzad A. Shirazi*, and Ahmad Kalhor. 2023. "Enhanced Gearbox Fault Diagnosis with Fusion LSTM CNN Network." ISAV, 2023, Page Numbers. DOI]
```

## License

This project is licensed under the terms of the MIT License. Please look at the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to SpectraQuest for providing the Gearbox Fault Diagnosis dataset.

