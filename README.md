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



## How to Install and Run the script

To run this script:
1. Copy the git link in your local git repository
2. Ensure *Starter_Code* directory is present, inside the *Starter_Code* directory, *part_1_mars_news.ipynb* and *part_2_mars_weather.ipynb* Jupyter Notebook files are present. 
3. Additionally, *mars_weather_df.csv* the output file (from *part_2_mars_weather.ipynb* file) is present
4. Run the script using **Jupyter Notebook** from *Starter_Code* directory and view results
5. Be sure to run the entire script using *Restart & Run All option* from *Kernel* in *Jupyter Notebook* to get error free results
6. Or, you could alternatively run each block of code individually starting from the very top 

## Credits

To write this script, I used the starter code provided and followed the challenge requirements meticulously