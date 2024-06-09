# 🌱 Plant Disease Classification with ResNet-9

## 🎯 Goal
To develop a deep learning model that can identify plant diseases from images. Early detection of plant diseases can help farmers take timely actions, reducing crop losses and promoting sustainable agriculture.

## 🗂️ Dataset
- **Source**: [New Plant Diseases Dataset(Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Content**:
  - 38 classes (plant-disease pairs)
  - 14 unique plants
  - 87,867 images for training

## 📊 Data Exploration
- Visualized distribution of images per class
- Displayed sample images from different classes

## 🛠️ Data Preparation
- Used `transforms.ToTensor()` for normalization and tensor conversion
- Utilized `DataLoader` for efficient batch processing
- Implemented `DeviceDataLoader` for GPU acceleration

## 🏗️ Model Architecture: ResNet-9
- **ResNet** (Residual Network): A type of convolutional neural network that uses "skip connections" to allow training of very deep networks without the problem of vanishing gradients.
- **Components**:
  - **Convolutional Blocks**: Consist of `Conv2d`, `BatchNorm2d`, `ReLU`, and optionally `MaxPool2d`.
    - `Conv2d`: 2D convolution for feature extraction
    - `BatchNorm2d`: Normalizes the outputs of the previous layer to stabilize training
    - `ReLU` (Rectified Linear Unit): Activation function that introduces non-linearity
    - `MaxPool2d`: Reduces spatial dimensions
  - **Residual Blocks**: Contains two conv blocks with a skip connection.
  - **Classifier**: `MaxPool2d`, `Flatten`, and `Linear` layers for final classification.

## 🏋️‍♀️ Training
- **Loss Function**: `CrossEntropyLoss` for multi-class classification
- **Optimizer**: `Adam` (Adaptive Moment Estimation) for per-parameter adaptive learning rates
- **Learning Rate**: One Cycle policy for faster convergence
- **Gradient Clipping**: To prevent exploding gradients

## 📈 Results
- Plotted training and validation losses to monitor learning and check for overfitting
- Plotted validation accuracy to track performance improvement
- Plotted learning rates to visualize the One Cycle policy

## 🧪 Testing
- Tested on separate images to qualitatively assess performance
- For production, would calculate metrics like accuracy, precision, recall on full test set

## 🚀 Future Improvements
1. More data or data augmentation for better generalization
2. Transfer learning with pre-trained models like ResNet50 or EfficientNet
3. Hyperparameter tuning for learning rate, batch size, etc.
4. Mobile app for farmers to use in the field
5. Integration with drone imagery for large-scale monitoring

## 💡 Conclusion
We've developed a ResNet-9 model that can accurately classify plant diseases from images. This project demonstrates the power of deep learning in agriculture, with potential for significant impact on food security and sustainability.

## 🛠️ Tech Stack
- **Python**: Primary programming language
- **PyTorch**: Deep learning framework
- **torchvision**: For computer vision utilities
- **matplotlib**: For data visualization
- **Jupyter Notebook**: For interactive development and presentation

## 📝 Definitions
- **Deep Learning**: A subset of machine learning based on artificial neural networks.
- **Convolutional Neural Network (CNN)**: A type of neural network particularly effective for image analysis.
- **Tensor**: A multi-dimensional array, fundamental to PyTorch.
- **Normalization**: Scaling input data to a common range, often [0, 1] for images.
- **Batch**: A set of samples processed together for efficiency.
- **Epoch**: One complete pass through the training dataset.
- **Overfitting**: When a model learns the training data too well, including its noise and fluctuations.
- **Learning Rate**: Determines the step size at each iteration while moving toward a minimum of the loss function.
- **Gradient**: The direction and magnitude of the steepest increase in the loss function.
- **Vanishing Gradient**: When gradients become very small, making learning slow or stopping altogether.
- **Transfer Learning**: Using a pre-trained model as a starting point, often faster and more accurate than training from scratch.

## 🙌 Acknowledgements
- Dataset creators and Kaggle for making data accessible
- PyTorch and torchvision developers
- Anthropic for creating Claude, an AI assistant that helped in drafting this README

Feel free to explore the code, open issues, or contribute to make this project even better! Together, we can leverage AI for a more sustainable future. 🌍
