Age Prediction from Facial Images using CNN
This project implements a Convolutional Neural Network (CNN) to predict age groups from facial images using the UTKFace dataset. The model classifies faces into 21 discrete 5-year age intervals ranging from 0–4 up to 100+, framing the problem as a multi-class classification task.

Dataset
The UTKFace dataset contains over 23,000 labeled facial images, each named in the format:

css
Copy code
[age]_[gender]_[race]_[date&time].jpg.chip.jpg
All images are preprocessed into a 64x64 grayscale format to serve as input to the model.

Preprocessing
Skipped unreadable or corrupt files using error handling

Resized all images to 64x64x1

Normalized pixel values to the [0, 1] range

Mapped continuous age values into 21 discrete classes (e.g., 0–4, 5–9, ..., 100+)

Model Architecture
Three convolutional layers with ReLU activation

Max pooling and dropout layers for regularization

Fully connected dense layers

Final softmax output layer with 21 class probabilities

Training Configuration
Loss function: Categorical Crossentropy

Optimizer: Adam

Batch size: 128

Epochs: 25

Evaluation: Accuracy and loss metrics on the test split

Results
The model performs consistently across different age ranges and effectively manages class imbalance in the dataset. It provides a robust approach to age group classification based on facial features.
