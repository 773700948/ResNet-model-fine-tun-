# ResNet-model-fine-tun-


# 🧠 ResNet.ipynb
Title: Training and Evaluating Xception Model (title might be a copy-paste oversight — it's likely about ResNet based on the filename)

### Purpose:
This notebook is focused on training a convolutional neural network model, possibly ResNet, despite the title mentioning Xception.

### Key Components:

*  Imports: TensorFlow, Keras layers and models, metrics, optimizers, image preprocessing, and other utility libraries.

*  Image Preprocessing: Uses ImageDataGenerator to handle training/testing image input.

*  Model Architecture (likely): It will likely define and train a ResNet model using the tensorflow.keras.applications module, which includes a prebuilt ResNet50, ResNet101, etc.

This notebook is most likely training a ResNet-based model on an image classification dataset.

# 📊 Predict.ipynb
### Purpose:
This one is very similar to pridect.ipynb, and is clearly focused on loading a trained model and predicting outcomes on a test dataset.

### Key Components:

*  Model Loading: Uses model_from_json to restore a model architecture.

*  Performance Metrics: Imports precision_score, recall_score, classification_report, and f1_score to evaluate the predictions.

*  Test Dataset: Loads the test images from the path E:/ourProject/newmodel/Dataset/test, resizes them to 224x224 pixels (ResNet size standard), and feeds them into the model.

# Dataset used to train the model:
## https://www.kaggle.com/datasets/rehabalsaby/yemeni-sign-language
