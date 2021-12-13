"""
Zhi Ye
CSE 163 AD

A file that test functions written in income_project_functions.py,
which are filtered_df_train_test, dtree_fit_and_predict_income,
calc_dtree_test_assignment, knn_fit_and_predict_income,
compare_gender_race, compare_gender_occupation, and
compare_education_income by using my own custom test dataset called
adult.test. Additionally, this file include helper functions such as
helper_length_train_test_data and helper_length_y_pred_data to aid in
testing functions. Lastly, this file imports income_project_functions,
pandas, and assert_equals from cse163_utils.
"""


import income_project_functions
import pandas as pd
from cse163_utils import assert_equals


def helper_length_train_test_data(train_test_data):
    """
    This helper function takes the train_test_data as a parameter
    to help test the filtered_df_train_test function by returning
    the length of the train_test_data.
    """
    X_train, X_test, y_train, y_test = train_test_data
    len_X_train = len(X_train)
    len_X_test = len(X_test)
    len_y_train = len(y_train)
    len_y_test = len(y_test)
    return len_X_train, len_X_test, len_y_train, len_y_test


def test_filtered_df_train_test(len_train_test_data):
    """
    Takes len_train_test_data as a parameter to test the
    filtered_df_train_test function using my own test file
    to ensure that it is working properly using the imported
    assert_equals function from cse163_utils. By doing so, I can
    confirm if the expected value matches the received value. If they
    do not match, assert_equals will crash the program and tell me what
    was wrong.
    """
    assert_equals(14, len_train_test_data[0])
    assert_equals(6, len_train_test_data[1])
    assert_equals(14, len_train_test_data[2])
    assert_equals(6, len_train_test_data[3])


def helper_length_y_pred_data(y_test_data):
    """
    This helper function takes the y_test_data as a parameter
    to help test the dtree_fit_and_predict_income function by
    returning the length of the y_test_data.
    """
    return len(y_test_data)


def test_dtree_fit_and_predict_income(helper_yp_data):
    """
    Takes helper_yp_data as a parameter to test the
    dtree_fit_and_predict_income function using my own test file to
    ensure that it is working properly using the imported
    assert_equals function from cse163_utils. By doing so, I can
    confirm if the expected value matches the received value.
    If they do not match, assert_equals will crash the program and
    tell me what was wrong.
    """
    assert_equals(6, helper_yp_data)


def test_calc_dtree_test_assignment(y_test_data):
    """
    Takes y_test_data as a parameter to test the
    calc_dtree_test_assignment function using my own test file to
    ensure that it is working properly using the imported
    assert_equals function from cse163_utils. By doing so, I can
    confirm if the expected value matches the received value.
    If they do not match, assert_equals will crash the program and
    tell me what was wrong.
    """
    assert_equals(6, len(y_test_data))


def test_knn_fit_and_predict_income(knn_y_test_pred):
    """
    Takes knn_y_test_pred as a parameter to test the
    knn_fit_and_predict_income function using my own test file to
    ensure that it is working properly using the imported
    assert_equals function from cse163_utils. By doing so, I can
    confirm if the expected value matches the received value.
    If they do not match, assert_equals will crash the program and
    tell me what was wrong.
    """
    assert_equals(6, len(knn_y_test_pred))


def test_compare_gender_race(merged_fe_race_data):
    """
    Takes merged_fe_race_data as a parameter to test the
    compare_gender_race function using my own test file to ensure
    that it is working properly using the imported assert_equals
    function from cse163_utils. By doing so, I can confirm if the
    expected value matches the received value. If they do not match,
    assert_equals will crash the program and tell me what was wrong.
    """
    assert_equals(' White', merged_fe_race_data.loc[0, 'race'])
    assert_equals(2, merged_fe_race_data.loc[0, 'Count (>50K)'])
    assert_equals(4, merged_fe_race_data.loc[0, 'Total'])
    assert_equals(50.0, merged_fe_race_data.loc[0, 'Percent (<=50K)'])


def test_compare_gender_occupation(merged_male_occ_data):
    """
    Takes merged_male_occ_data as a parameter to test the
    compare_gender_occupation function using my own test file to
    ensure that it is working properly using the imported
    assert_equals function from cse163_utils. By doing so, I can
    confirm if the expected value matches the received value. If
    they do not match, assert_equals will crash the program and
    tell me what was wrong.
    """
    assert_equals(' Exec-managerial',
                  merged_male_occ_data.loc[0, 'occupation'])
    assert_equals(3, merged_male_occ_data.loc[0, 'Count (>50K)'])
    assert_equals(4, merged_male_occ_data.loc[0, 'Total'])
    assert_equals(25.0, merged_male_occ_data.loc[0, 'Percent (<=50K)'])


def test_compare_education_income(merged_edu_data):
    """
    Takes merged_edu_data as a parameter to test the
    compare_education_income function using my own test file
    to ensure that it is working properly using the imported
    assert_equals function from cse163_utils. By doing so, I can
    confirm if the expected value matches the received value.
    If they do not match, assert_equals will crash the program and
    tell me what was wrong.
    """
    assert_equals(' Bachelors', merged_edu_data.loc[0, 'education'])
    assert_equals(3, merged_edu_data.loc[1, 'Count (<=50K)'])
    assert_equals(' Masters', merged_edu_data.loc[2, 'education'])
    assert_equals(33.33, merged_edu_data.loc[2, 'Percent (<=50K)'])


def main():
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race',
               'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
               'native-country', 'income']
    data_test = pd.read_csv('./adult.test', names=columns, na_values=' ?')
    train_test_data = income_project_functions. \
        filtered_df_train_test(data_test)
    helper_tt_data = helper_length_train_test_data(train_test_data)
    test_filtered_df_train_test(helper_tt_data)
    y_test_data = income_project_functions. \
        dtree_fit_and_predict_income(train_test_data)
    helper_yp_data = helper_length_y_pred_data(y_test_data)
    test_dtree_fit_and_predict_income(helper_yp_data)
    income_project_functions.calc_dtree_test_assignment(y_test_data)
    test_calc_dtree_test_assignment(y_test_data)
    knn_y_test_pred = income_project_functions. \
        knn_fit_and_predict_income(train_test_data)
    test_knn_fit_and_predict_income(knn_y_test_pred)
    merged_fe_race_data = income_project_functions. \
        compare_gender_race(data_test)
    test_compare_gender_race(merged_fe_race_data)
    merged_male_occ_data = income_project_functions. \
        compare_gender_occupation(data_test)
    test_compare_gender_occupation(merged_male_occ_data)
    merged_edu_data = income_project_functions. \
        compare_education_income(data_test)
    test_compare_education_income(merged_edu_data)


if __name__ == '__main__':
    main()
