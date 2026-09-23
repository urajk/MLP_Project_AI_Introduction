#-----------------------------------------------------------------------------------------------------------------------------#
# CS-340 Final Project, Build your own MLP!
# Supervisor: Alexandros Astaras
# Author: Ioannis Karantounias 20240177@student.act.edu
# Licensed under the GNU open source agreement.
# Disclaimer - NONE of this code was written using AI.
# Though AI was used so that I can understand the material needed for its implementation since I was behind in the lectures.
# Prompt questions included things like:
# What is a synaptic weight?
# What does a sigmoid function do?
#-----------------------------------------------------------------------------------------------------------------------------#
import numpy as np
import pandas
import warnings
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Tweakable values
epochs = 2000
testDataSize = 0.2
hiddenLayerSizesArr = [(10,), (20,10)]
randomness = 0
learningRateValueArr = [0.001, 0.02, 0.1]
learningRateArr = ['constant', 'adaptive', 'invscaling']

# Read and set up the dataframe.
# We do this because every variable in the dataset is categorical.
dataFrame = pandas.read_csv("car.data")

# The dataset currently has no column headers, but we know what
# they are called cause they are described in the UCI repo.
columnHeaders = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "target"]
dataFrame.columns = columnHeaders
trainingColumnHeaders = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

# This is a dataset meant for supervised learning,
# meaning it includes the desired output along with the inputs.
# Thus, we have to split the two.
inputData = dataFrame.drop("target", axis=1)
desiredOutput = dataFrame["target"]

# We now have to turn these categorical values into numbers somehow.
inputData = pandas.get_dummies(inputData)
desiredOutput = LabelEncoder().fit_transform(desiredOutput)

# These are all the parameters we will have to keep for the weight changing graph
highestAccuracy = 0
highestAccuracyModel = None
bestHiddenLayerSize = None
bestRandomness = 0
bestLearningRate = ""
bestLearningRateValue = 0
bestEpochs = 0
bestX_Train = None
bestY_Train = None

for lr_type in learningRateArr:
    for lr_value in learningRateValueArr:
        for hiddenLayerSize in hiddenLayerSizesArr:
            # Trait / Test Split
            x_Train, x_Test, y_Train, y_Test = train_test_split(inputData, desiredOutput, test_size=testDataSize, random_state=randomness)
            scaler = StandardScaler()

            # Normalizing:
            x_Train = scaler.fit_transform(x_Train)
            x_Test = scaler.transform(x_Test)

            # Creating the model:
            model = MLPClassifier(hidden_layer_sizes=hiddenLayerSize, activation="relu", solver="adam", max_iter=epochs, random_state=randomness, learning_rate_init=lr_value, learning_rate=lr_type)
            model.fit(x_Train, y_Train)

            # Predicting + accuracy testing:
            y_Prediction = model.predict(x_Test)
            accuracy = accuracy_score(y_Test, y_Prediction)
            accuracy = round(accuracy*100, 3)
            print(f"Accuracy: {accuracy}%")

            # Saving this data so we can mess with the best model later
            if accuracy > highestAccuracy:
                highestAccuracyModel = model
                bestHiddenLayerSize = hiddenLayerSize
                bestRandomness = randomness
                bestLearningRate = lr_type
                bestLearningRateValue = lr_value
                bestEpochs = epochs
                bestX_Train = x_Train
                bestY_Train = y_Train

                highestAccuracy = accuracy

            # Output_txt generation
            with open("output_data.txt", "a") as f:
                f.write(f"\n===== EXPERIMENT =====\n")
                f.write(f"Topology: {hiddenLayerSize}\n")
                f.write(f"Learning rate: {lr_value}\n")
                f.write(f"Split: {testDataSize*100}% for testing | {(1-testDataSize)*100}% for training\n")
                f.write(f"Accuracy: {accuracy}\n")

            # Loss Curve (used your code for the implementation):
            fig, axes = plt.subplots(figsize=(7, 4))
            axes.plot(model.loss_curve_, color="steelblue", linewidth=2)
            axes.set_xlabel("Epoch", fontsize=12)
            axes.set_ylabel("Training loss (cross-entropy)", fontsize=12)
            axes.set_title("MLP training loss curve — Cars Dataset", fontsize=13)
            axes.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(f"loss_{lr_type}_{lr_value}_{hiddenLayerSize}.png", dpi=150)
            print(f"Loss curve saved to: loss_{lr_type}_{lr_value}_{hiddenLayerSize}.png")
            plt.close()

# Loss Curve for the Best performing model:
figHighest, axesHighest = plt.subplots(figsize=(7, 4))
axesHighest.plot(highestAccuracyModel.loss_curve_, color="steelblue", linewidth=2)
axesHighest.set_xlabel("Epoch", fontsize=12)
axesHighest.set_ylabel("Training loss (cross-entropy)", fontsize=12)
axesHighest.set_title("MLP training loss curve — Cars Dataset", fontsize=13)
axesHighest.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"loss_BestPerformingModel.png", dpi=150)
print(f"Loss curve saved to: loss_BestPerformingModel.png")
plt.close()

# Here will go the weight history of 1 weight of the best performing model
weight_history = []

bestModel = MLPClassifier(
    hidden_layer_sizes=bestHiddenLayerSize,
    max_iter=1,
    warm_start=True,
    random_state=bestRandomness
)

for epoch in range(bestEpochs):
    bestModel.fit(bestX_Train, bestY_Train)
    weight_history.append(bestModel.coefs_[0][0,0])

figBest, axesBest = plt.subplots(figsize=(7, 4))
axesBest.plot(weight_history, color="steelblue", linewidth=2)
axesBest.set_xlabel("Epoch", fontsize=12)
axesBest.set_ylabel("Weight Value", fontsize=12)
axesBest.set_title("Weight Change Over Time (Best Model)", fontsize=13)
axesBest.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"weight_BestPerformingModelWeight.png", dpi=150)
print(f"Weight curve saved to: weight_BestPerformingModel.png")
plt.close()

# Here we will be setting up all the required output files for the project
inputData.to_csv("testing_data_labeled.txt")