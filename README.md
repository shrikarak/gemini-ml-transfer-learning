# Transfer Learning for Image Classification with TensorFlow

This repository contains a Jupyter Notebook that provides a step-by-step guide to performing image classification using transfer learning. The project demonstrates how to take a powerful, pre-trained model and adapt it for a new, custom task, which is a common and effective technique in modern machine learning.

## Overview

The goal of this project is to classify images as either 'cats' or 'dogs' using a pre-trained convolutional neural network (CNN). We leverage the **MobileNetV2** architecture, which was trained on the extensive ImageNet dataset, and fine-tune it for our specific binary classification problem.

This approach saves significant time and computational resources compared to training a model from scratch.

## Key Concepts Demonstrated

- **Transfer Learning**: Reusing a pre-trained model for a new problem.
- **Data Augmentation and Preprocessing**: Using `tf.data` to build an efficient input pipeline for image data.
- **Model Customization**: Removing the original classification head of a pre-trained model and adding new, trainable layers.
- **Feature Extraction**: The initial training phase where we only train the new classification layers.
- **Fine-Tuning**: Unfreezing some of the top layers of the base model and continuing training with a low learning rate to improve accuracy.
- **Visualization**: Plotting training history (accuracy and loss) to evaluate model performance.

## Tech Stack

- **TensorFlow / Keras**: The core deep learning framework used to build and train the model.
- **NumPy**: For numerical operations.
- **Matplotlib**: For plotting and visualizing images and results.
- **Jupyter Notebook**: For interactive development and documentation.

## Setup and Installation

Follow these steps to set up the project environment.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/<your-username>/ml-transfer-learning-notebook.git
    cd ml-transfer-learning-notebook
    ```

2.  **Create and Activate a Virtual Environment** (Recommended)
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## How to Use the Notebook

1.  **Download the Dataset**
    - The notebook is designed to work with the [Kaggle "Dogs vs. Cats"](https://www.kaggle.com/c/dogs-vs-cats/data) dataset.
    - Download the `train.zip` file from the competition page.

2.  **Organize the Data**
    - Create a `data` directory inside the cloned repository.
    - Inside `data`, create a `train` directory.
    - Unzip the `train.zip` file. You will get thousands of `cat.xxxx.jpg` and `dog.xxxx.jpg` files.
    - Create `cat` and `dog` subdirectories inside `data/train`.
    - Move all cat images into `data/train/cat/` and all dog images into `data/train/dog/`.

    Your final directory structure should look like this:
    ```
    ml-transfer-learning-notebook/
    ├── data/
    │   └── train/
    │       ├── cat/
    │       │   ├── cat.0.jpg
    │       │   ├── cat.1.jpg
    │       │   └── ...
    │       └── dog/
    │           ├── dog.0.jpg
    │           ├── dog.1.jpg
    │           └── ...
    ├── image_classification_transfer_learning.ipynb
    ├── requirements.txt
    └── README.md
    ```

3.  **Launch Jupyter Notebook**
    ```bash
    jupyter notebook
    ```

4.  **Run the Cells**
    - Open the `image_classification_transfer_learning.ipynb` file in your browser.
    - Execute the cells sequentially from top to bottom to train the model and see the results.
