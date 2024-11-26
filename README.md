# COMP 472 Final Project

## Description
by Tudor Cosmin Suciu 40179863, submitted the 26th of November, 2024


## Instructions
1. First please install all the dependencies from the requirements.txt file.
```
pip install -r requirements.txt --user
```

2. Run the scripts/data_preprocessing.py script
This will download and pre-process the dataset for the models to use later.

3. To train a model go to scripts/ and choose your preffered model and run it's script
This will save all the trained models with their variants in the models/ folder

4. If you do not wish to train the models but just want to evaluate them, run scripts/evaluate_models.py
The saved models from /models will each be evaluated and you will get the following outputs:

    - A terminal output of all the confusion matrices, clasification report and accuracy for each model
    - A evaluation_results.txt file found in results/ that prints the same metrics but in a text file for easy readability 
    - A set of confusion matrix images (heatmap) saved in results/confusion_matrices/ to help visualize the confusion matrix of each model

5. For a summarized view of the model evaluations please refer to the metrics_table.csv in the results/ folder

6. Open report.pdf to view the full report of the models

7. To view this project on GitHub, go to:
https://github.com/cuscus555/COMP472-AI-Project.git
