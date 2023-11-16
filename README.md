# Credit Risk Classification
Module 20 challenge - Supervised Learning

### Guidelines
The guidelines for this Challenge are categorized into the subsequent sections:

* Data Segmentation into Training and Testing Sets

* Development of a Logistic Regression Model using the Initial Data

* Compilation of a Credit Risk Assessment Report

#### Data Segmentation into Training and Testing Sets
* Access the provided starter code notebook and execute the outlined actions below:

* Import the lending_data.csv dataset from the Resources directory into a Pandas DataFrame.

* Construct the labels set (y) based on the “loan_status” column, and subsequently, generate the features (X) DataFrame from the remaining columns.

* Divide the data into training and testing datasets utilizing the train_test_split method.

#### Develop a Logistic Regression Model with the Original Data

* Apply your understanding of logistic regression to accomplish the subsequent tasks:

* Train a logistic regression model using the training data (X_train and y_train).

* Utilize the fitted model to predict the labels for the testing data by employing the testing feature data (X_test).

* Assess the model's performance by undertaking the following steps:

    * Create a confusion matrix.

    * Display the classification report.

* Address the following inquiry: How effective is the logistic regression model in predicting both the 0 (healthy loan) and 1 (high-risk loan) labels?

#### Compose a Credit Risk Analysis Report
Compose a concise report integrating a summary and assessment of the performance exhibited by the machine learning models utilized in this assignment. The report should be crafted as the README.md file embedded within your GitHub repository.

* Organize your report adhering to the report template, ensuring its inclusion of the following components:

* An outline of the analysis: Elucidate the objective behind conducting this analysis.

* The outcomes: Enumerate the accuracy score, precision score, and recall score of the machine learning model using a bulleted format.

* A synopsis: Condense the outcomes derived from the machine learning model. Provide reasoning to endorse the model for adoption within the company. Alternatively, if not recommending the model, justify the rationale behind this decision.

## Credit Risk Analysis Report

In this section, I describe the analysis completed for the machine learning models used in this Challenge. 

* The purpose of this analysis is to determine good and bad candidates for loan approval. We use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.<br>

* By adopting a statistical model to make predictions, we eliminate human error and bias, thus improving the odds of loan approval based solely on an individual's finances and good credit score. Some of the key metrics taken into consideration were, loan_size, interest_rate,	borrower_income,debt_to_income,	num_of_accounts, derogatory_marks and total_debt <br>


* We conducted a thorough training of a model based on lending activity from a peer-to-peer lending services company to assess credit worthiness of an individual. We fed the dataset to our model, designating loan_status as the target and the value_counts() of the target variable are as follows, <br>
    **0:    75036** <br>
    **1:     2500** <br>
    Where **0 stands for healthy loan** and **1 stands for high-risk loan** 

* The stages of machine learning process include:
    * Assessing the dataset and identifying the target and feature values
    * Checking the target value_counts() to determine how balanced our dataset is and if any modifications might be needed.
    * We split the data into two sets for training and testing. The purpose of this step is to have the data divided into an approximate split of 70-80% and 30-20% for training and testing, so we can train our model with the training data and assess its predicting capabilities with the test data.
    * Post that step, we determine the type of model we want to train our data with and initiate an instance of it. For this challenge, we opted to use the Logistic Regression model since we are classifying data.
    * After initiating an instance of the model, we fit it with the training data.
    * Make predictions using the test data and finally evaluate the model's ability to predict.
    * To evaluate the model, we calculate quantifiable metrics such as Accuracy Score, Classification Report (that summarizes the ability of model w.r.t Precision, Recall and Accuracy in predicting) and confusion matrix (that shows how many incorrect predictions were made based on the overall test data.)
* To check if the model can be made to perform better, we use Random Oversampler
    * Since our data is clearly imbalanced with **0: 75036** and **1: 2500**, we use Random Oversampler to remedy this.
    * Random Oversampler increases the number of **1** or high-risk loan samples to correct the imbalance in the dataset. 
    * This step allows us to make apples to apples comparision of target and features values.
    * Once the data is balanced, we repeat the above steps all the way to the evaluation, which clearly shows an increase in the accuracy score from **0.95** to **0.99** which indicates that using Random Oversampler was beneficial in improving the model's predictions.



## Results

Evaluation metrics: balanced accuracy scores, precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores. <br>
    * Precison (0): 1.00
    * Precison (1): 0.85
    * Recall (0): 0.99
    * Recall (1): 0.91
    * Balanced Accuracy (overall): 0.95

* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores. <br>
    * Precison (0): 1.00
    * Precison (1): 0.84
    * Recall (0): 0.99
    * Recall (1): 0.99
    * Balanced Accuracy (overall): 0.99

## Summary

* Using random oversampler, we even out the imabalances in our dataset, thus improving the balanced accuracy score to 0.99. 
The recall for predicting '1' also went up from 0.91 to 0.99. <br>
* But, an overall balanced accuracy score of 0.99 indicates that this model further improves the predictions for Credit Risk.
* Thus, I would recommend using Model 2 for its improved efficiency in predicting candidates correctly. 
* It is equally important to predict both **0 and 1** for Credit Risk as the bank could lose a huge sum of money if **1/high-rish loan candidates** are predicting incorrectly and it would lose valuable customers if **0/healthy loan candidates** are predicted incorrectly. In both cases, the bank stands to lose money and good business.  



## How to Install and Run the script

To run this script:
1. Copy the git link in your local git repository
2. Ensure *Credit_Risk* directory is present, inside the *Credit_Risk* directory, *credit_risk_classification.ipynb* Jupyter Notebook and *lending_data.csv* file is present in the *Resources* folder. 
3. Run the script using **Jupyter Notebook** from *Credit_Risk* directory and view results
4. Be sure to run the entire script using *Restart & Run All option* from *Kernel* in *Jupyter Notebook* to get error free results
5. Or, you could alternatively run each block of code individually starting from the very top 

## Credits

To write this script, I used the starter code provided and followed the challenge requirements meticulously