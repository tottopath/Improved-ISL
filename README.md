This project is based on the tutorial "Sign language detection with Python and Scikit Learn | Landmark detection | Computer vision tutorial" by a Computer Vision Engineer from YouTube. The original tutorial focused on American Sign Language (ASL) using a Random Forest classifier for single-hand detection. However, this project extends the concept to support Indian Sign Language (ISL), detection for both hands simultaneously, and saving and translating detected words into Hindi. This also compares the accuracy of Random Forest vs Convolutional Neural Network.

Modifications:
  1. Expanded support for Indian Sign Language (ISL).
  2. Enabled detection for both hands simultaneously.
  3. Implemented saving and translation of detected words into Hindi.
  4. Compared the performance of Random Forest and Convolutional Neural Network (CNN) classifiers.

Key Features:
  1. Collection of hand gesture images via webcam for dataset creation.
  2. Training of machine learning models (Random Forest and Convolutional Neural Network) for gesture classification.
  3. Real-time detection and recognition of hand gestures using the trained models.
  4. Translation of detected words into Hindi for improved accessibility and understanding.

Technologies Used:
  1. Python
  2. OpenCV
  3. Mediapipe
  4. Scikit-learn
  5. TensorFlow
  6. EnglisttoHindi

Instructions:
  1. Run collect_imgs.py to capture hand gesture images for dataset creation.
  2. Execute create_dataset.py to extract hand landmarks from the collected images and create the dataset.
  3. Train the machine learning models by running train_classifier.py or hyperparameters for random forest.py.
  3. Optionally, compare the performance of Random Forest and CNN classifiers using CNN vs RF.py.
  4. Utilize inference_classifier.py for real-time detection and translation of hand gestures.

This project provides a foundation for building applications related to sign language interpretation and accessibility.
