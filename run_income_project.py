"""
Zhi Ye
CSE 163 AD

A file that runs the functions for my income project, which are
filtered_df_train_test, dtree_fit_and_predict_income,
calc_dtree_test_assignment, knn_fit_and_predict_income,
compare_gender_race, compare_gender_occupation, and
compare_education_income. The dataset used in this file is called
adult.data, which comes from UCI Machine Learning Repository that
provides data on the income of adult individuals. In the dataset,
there are 48,842 rows and 14 columns. More specifically, the 14 columns
included represents age, workclass, fnlwgt, education, education-num,
marital-status, occupation, relationship, race, sex, capital-gain,
capital-loss, hours-per-week, and native-country. Lastly, this file
imports income_project_functions and pandas.
"""


import income_project_functions
import pandas as pd


def main():
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race',
               'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
               'native-country', 'income']
    data = pd.read_csv('./adult.data', names=columns, na_values=' ?')
    train_test_data = income_project_functions.filtered_df_train_test(data)
    y_test_data = income_project_functions. \
        dtree_fit_and_predict_income(train_test_data)
    income_project_functions.calc_dtree_test_assignment(y_test_data)
    income_project_functions.knn_fit_and_predict_income(train_test_data)
    income_project_functions.compare_gender_race(data)
    income_project_functions.compare_gender_occupation(data)
    income_project_functions.compare_education_income(data)


if __name__ == '__main__':
    main()
