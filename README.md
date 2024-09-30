# Smart Glove for Object Recognition

This project is focused on using **Support Vector Machine (SVM)** for object recognition based on sensor data. The project demonstrates how machine learning algorithms, in particular, Principal Component Analysis (PCA) for feature extraction and SVM as a classifier, can be used to recognize objects of different shapes and sizes with high accuracy.

For more detailed explanations and results, please refer to our [paper](link-to-paper).

## About The Project

In this project, we have utilized sensor data to train and evaluate an SVM model for object recognition. The data includes different objects such as spheres, cubes, and complex shapes like dodecahedrons. The model achieved a prediction accuracy of 98.75%.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

- Python 3.12
- Required libraries: 
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `scipy`
  - `time`
  - `serial`


# Installation
1. git clone https://github.com/your-username/your-repo-name.git
2. cd your-repo-name
3. Run the Python scripts as needed.

   
# Usage
 1. **Prepare the Data**
   - Ensure that your sensor data is in a CSV format similar to the one used in the project.
   - Download and import the dataset to appropriate directory.

2. **Run the Python Script**
   - After ensuring the dependencies and data are in place, run the provided Python scripts to preprocess the data, extract features, train the model, and evaluate its performance.
   - You can visualize the confusion matrix and see the prediction accuracy to understand how well the model is performing.
  
3. **Modify Parameters**
   - If you want to tweak the program (e.g., change the window size for filtering, or adjust SVM parameters), open the script file (e.g., main.py) and adjust the relevant variables.

4. **View and Save Outputs**
   - The results (confusion matrix, prediction accuracy) will be displayed on the terminal and in graphical windows.
   - Filtered, normalized data and prediction results can be saved automatically to a CSV file by configuring the file saving paths in the script.
     
5. **Test with New Data**
   - Once the model is trained, you can test it with new sensor data by following the object recognition step outlined above. You can load new sensor data and predict the shape or object type.


# Contributing
 - Pei-Song Chee
 - Cao Guan
 - Kok-Tong Lee
 - Eng-Hock Lim
 - Chun-Hui Tan
 - Jen-Hahn Low
 - Kwong-Long Wong


# Contact
Pei-Song Chee

Email: cheeps@utar.edu.my


