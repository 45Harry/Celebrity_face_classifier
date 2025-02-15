# Sports Celebrity Image Classification

## Description

This project implements a sports celebrity image classification model using machine learning and computer vision techniques. The model detects and classifies images based on facial features, using wavelet transformations and support vector machines (SVM). It follows a structured preprocessing pipeline, including face detection, cropping, wavelet transformation, feature extraction, and model training using different classifiers.

## Prerequisites

Before running the code, ensure you have a properly configured Python environment with all necessary dependencies installed.

## Setting Up the Virtual Environment

1. Open a terminal or command prompt.
2. Navigate to the project directory where the script files are located.
3. Run the following command to create a virtual environment:
   ```sh
   python -m venv venv
   ```
4. Activate the virtual environment:
   - **Windows:**
     ```sh
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```sh
     source venv/bin/activate
     ```

## Installing Dependencies

Once the virtual environment is activated, install the required dependencies using the following command:

```sh
pip install -r requirements.txt
```

If `requirements.txt` is not available, manually install the following libraries:

```sh
pip install numpy pandas matplotlib scikit-learn opencv-python seaborn pywavelets joblib
```

## Steps for Image Classification

### (1) Preprocessing: Detect Face and Eyes
- The model loads an image and converts it to grayscale.
- It uses pre-trained Haar cascade classifiers to detect faces and eyes.
- If two eyes are detected, the image is retained; otherwise, it is discarded.

### (2) Preprocessing: Crop the Facial Region
- Once a face is detected, the region containing the face is cropped and saved.

### (3) Preprocessing: Apply Wavelet Transform
- The cropped face image undergoes wavelet transformation using the Haar wavelet.
- This transformation enhances the detection of facial features such as eyes, nose, and lips.

### (4) Image Processing and Dataset Preparation
- The script processes all images in the dataset directory, detects faces, and creates a cropped image dataset.
- The cropped images are stored in a separate folder inside the dataset directory.

### (5) Feature Extraction and Model Training
- The model extracts features from images using:
  - Raw pixel values resized to (32x32) pixels.
  - Wavelet transformed images resized to (32x32) pixels.
  - A combination of both as input features.
- These features are converted into a single vector of size 4096 for training.
- The dataset is split into training and test sets.
- Standard scaling is applied to normalize the data.

### (6) Model Selection and Training
- The project trains multiple models, including:
  - **Support Vector Machine (SVM)** with an RBF kernel.
  - **Random Forest Classifier**.
  - **Logistic Regression**.
- GridSearchCV is used to fine-tune hyperparameters for the best performance.

### (7) Model Evaluation
- The trained models are tested on the test dataset.
- A classification report and confusion matrix are generated to evaluate performance.
- The best-performing model is selected for saving.

### (8) Saving the Model
- The final trained model is saved using joblib for future inference:
  ```python
  import joblib
  joblib.dump(best_clf, 'saved_model.pkl')
  ```
- A class dictionary mapping celebrity names to labels is stored in a JSON file:
  ```python
  import json
  with open("class_dictionary.json", 'w') as f:
      f.write(json.dumps(class_dict))
  ```

## Running the Project

1. Activate the virtual environment.
2. Run the script using:
   ```sh
   python script.py
   ```
3. Examine the `cropped` folder inside `datasets` to verify correctly processed images.
4. The model's classification results and performance metrics will be displayed.

## Output
- The trained model is saved as `saved_model.pkl`.
- The class dictionary is stored in `class_dictionary.json`.
- Cropped and preprocessed images are stored inside the dataset folder.

## Libraries Used

- **numpy** - Numerical computations.
- **pandas** - Data manipulation.
- **matplotlib & seaborn** - Data visualization.
- **scikit-learn** - Machine learning models and preprocessing.
- **opencv-python** - Image processing and computer vision.
- **pywavelets** - Wavelet transformation for feature extraction.
- **joblib** - Model saving and loading.
- **json** - Handling feature column data.

## Author

[Harry Sah]

