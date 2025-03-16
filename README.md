# Risk-Based Thresholding for Reliable Anomaly Detection in Concentrated Solar Power Plants

This is the official repository for our paper "Risk-Based Thresholding for Reliable Anomaly Detection in Concentrated Solar Power Plants"

## Overview

Efficient and reliable operation of Concentrated Solar Power (CSP) plants is essential for meeting the growing demand for sustainable energy. However, high-temperature solar receivers face severe operational risks, such as freezing, deformation, and corrosion, which can result in costly downtime and maintenance. To monitor CSP plants, cameras mounted on solar receivers record infrared images at irregular intervals ranging from one to five minutes throughout the day. Anomaly detection can be performed by assigning an anomaly score to an image and applying a threshold to classify instances as normal or anomalous. Existing methods determine this decision threshold by optimizing anomaly detection metrics, such as F1-score or G-Mean. This work proposes a framework for generating more reliable decision thresholds with finite-sample guarantees on any chosen risk function (e.g., the false positive rate). Our framework also incorporates an abstention mechanism, allowing high-risk predictions to be deferred to domain experts. Second, to compute the anomaly score of an observed image, we propose a density forecasting method that estimates its likelihood conditional on a sequence of previously observed images. Third, we analyze the deployment results of our framework across multiple training scenarios over several months from two CSP plants, offering valuable insights to our industry partner for maintenance operations. Finally, given the confidential nature of our dataset, we provide an extended simulated dataset, leveraging recent advancements in generative modeling to create diverse thermal images that simulate multiple CSP plants.

## Installation

This code requires the packages listed in [`environment.yml`](environment.yml).

To run the code, set up a virtual environment using `conda`:

```
cd <path-to-cloned-directory>
conda env create --file environment.yml
conda activate reladcspenv
```

## Dataset

The simulated dataset can be downloaded using this [link](TODO). After downloading simulated_dataset.zip, extract the contents into the [`data`](data/) folder. The pickle files should thus be located at `<path-to-cloned-directory>\data\<csp_name>\<name>.pickle`.

## Running experiments

To run an experiment create a new configuration file in the [`configs`](configs/) directory. The experiments can be can run using the following command:

```
cd <path-to-cloned-directory>
python src/main.py ./configs/<config-file-name>.json
```

We provide the configuration files for running DensityAD.

## License

This project is under the MIT license.