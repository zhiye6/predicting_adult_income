"""
Zhi Ye
CSE 163 AD

A file that define functions for my income project, which are
filtered_df_train_test, dtree_fit_and_predict_income,
calc_dtree_test_assignment, knn_fit_and_predict_income,
compare_gender_race, compare_gender_occupation, and
compare_education_income. Additionally, this file imports
necessary libraries such as pandas, accuracy_score from
sklearn.metrics, DecisionTreeClassifier from sklearn.tree,
train_test_split from sklearn.model_selection, StandardScaler
from sklearn.preprocessing, KNeighborsClassifier from
sklearn.neighbors, classification_report from sklearn.metrics,
plotly.graph_objects, and make_subplots from plotly.subplots.
"""


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def filtered_df_train_test(data):
    """
    Takes the adult dataset as a parameter to create a filtered
    dataframe of the necessary columns to prep for machine learning
    and returning the X_train, X_test, y_train, y_test.
    """
    filtered_df = data[['education', 'occupation', 'race', 'sex', 'income']]
    filtered_df = filtered_df.dropna()
    X = filtered_df.loc[:, filtered_df.columns != 'income']
    X = pd.get_dummies(X)
    y = filtered_df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test


def dtree_fit_and_predict_income(train_test_data):
    """
    Takes the train test data and performs machine learning using
    the DecisionTreeClassifier. This function will print the
    train and test accuracy while returning y_test predictions.
    """
    X_train, X_test, y_train, y_test = train_test_data
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print('Train Accuracy:', accuracy_score(y_train, y_train_pred))
    print('Test Accuracy:', accuracy_score(y_test, y_test_pred))
    return y_test_pred


def calc_dtree_test_assignment(y_test_data):
    """
    Takes the y_test prediction data as a parameter to calculate
    and print the number of adults with income greater than 50K,
    number of adults with income less than or equal to 50K, and the
    total number of adults in the y_test prediction data. Lastly, this
    function will return the length of the y_test_data.
    """
    greater_than_50k = 0
    less_than_or_equal_to_50k = 0
    for i in y_test_data:
        if i == ' >50K':
            greater_than_50k += 1
        elif i == ' <=50K':
            less_than_or_equal_to_50k += 1
    print('# of Adults with income >50K:', greater_than_50k)
    print('# of Adults with income <=50K:', less_than_or_equal_to_50k)
    print('Total # of Adults:', len(y_test_data))
    return len(y_test_data)


def knn_fit_and_predict_income(train_test_data):
    """
    Takes the train test data as a parameter performs machine learning
    using the KNeighborsClassifier. This function will print a
    classification report, where insight such as precision, recall,
    f1-score, etc. are available to evaluate the models accuracy. Lastly,
    this function will return knn_y_test_pred.
    """
    X_train, X_test, y_train, y_test = train_test_data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    knn_y_test_pred = classifier.predict(X_test)
    print(classification_report(y_test, knn_y_test_pred))
    return knn_y_test_pred


def compare_gender_race(data):
    """
    Takes the adult dataset as a parameter and creates a filtered
    dataframe containing the columns race, sex, and income. Then,
    using the newly filtered dataframe, this function creates a
    filtered dataframe for males with >50k income and a filtered
    dataframe for males with <=50k income. Then, these dataframes
    are merged together, where computed columns are also created to
    showcase the percentage of males >50k income and males with
    <=50k income for each race. This process is then repeated for
    females. Once male and female dataframes are created with
    computed columns, a plotly group bar chart is made to compare
    percentages between males and females with income >50K by race.
    Lastly, this function returns the merged female dataframe for
    testing purposes.
    """
    gender_race_income = data[['race', 'sex', 'income']]
    gender_race_income = gender_race_income.dropna()
    is_gt_50k = gender_race_income['income'] == ' >50K'
    is_le_50k = gender_race_income['income'] == ' <=50K'
    is_male = gender_race_income['sex'] == ' Male'
    # Filter and create dataframe for males with >50k income
    me_gt_50k = gender_race_income[is_gt_50k & is_male]
    me_race_gt_50k = me_gt_50k.groupby('race')['income'].count()
    male_race_gt_50k = me_race_gt_50k.reset_index(name='Count (>50K)')
    # Filter and create dataframe for males with <=50k income
    me_le_50k = gender_race_income[is_le_50k & is_male]
    male_race_le_50k = me_le_50k.groupby('race')['income'].count()
    male_race_le_50k = male_race_le_50k.reset_index(name='Count (<=50K)')
    # Merge the two male dataframes
    merged_male_race = male_race_gt_50k.merge(male_race_le_50k, left_on='race',
                                              right_on='race', how='inner')
    # Make computed columns
    merged_male_race.loc[:, 'Total'] = merged_male_race.sum(axis=1)
    merged_male_race['Percent (>50K)'] = ((merged_male_race['Count (>50K)']
                                           / merged_male_race['Total'])) * 100
    merged_male_race['Percent (<=50K)'] = ((merged_male_race['Count (<=50K)']
                                            / merged_male_race['Total'])) * 100
    merged_male_race = merged_male_race.round(2)
    is_female = gender_race_income['sex'] == ' Female'
    # Filter and create dataframe for females with >50k income
    fe_gt_50k = gender_race_income[is_gt_50k & is_female]
    fe_race_gt_50k = fe_gt_50k.groupby('race')['income'].count()
    female_race_gt_50k = fe_race_gt_50k.reset_index(name='Count (>50K)')
    # Filter and create dataframe for females with <=50k income
    fe_le_50k = gender_race_income[is_le_50k & is_female]
    female_race_le_50k = fe_le_50k.groupby('race')['income'].count()
    female_race_le_50k = female_race_le_50k.reset_index(name='Count (<=50K)')
    # Merge the two female dataframes
    merged_fe_race = female_race_gt_50k.merge(female_race_le_50k,
                                              left_on='race', right_on='race',
                                              how='inner')
    # Make computed columns
    merged_fe_race.loc[:, 'Total'] = merged_fe_race.sum(axis=1)
    merged_fe_race['Percent (>50K)'] = ((merged_fe_race['Count (>50K)']
                                         / merged_fe_race['Total'])) * 100
    merged_fe_race['Percent (<=50K)'] = ((merged_fe_race['Count (<=50K)']
                                          / merged_fe_race['Total'])) * 100
    merged_fe_race = merged_fe_race.round(2)
    # Plot percentages for males and females with income >50K
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=merged_male_race['race'],
        y=merged_male_race['Percent (>50K)'],
        name='Male',
        marker_color='mediumseagreen'
    ))
    fig.add_trace(go.Bar(
        x=merged_fe_race['race'],
        y=merged_fe_race['Percent (>50K)'],
        name='Female',
        marker_color='mediumpurple'
    ))
    fig.update_layout(barmode='group', xaxis_tickangle=-45,
                      title="Percentage of Males and Females with "
                            "Income >50K by Race",
                      xaxis_title="Race",
                      yaxis_title="Percentage")
    fig.write_html("./pct_mf_income_>50k_race.html")
    return merged_fe_race


def compare_gender_occupation(data):
    """
    Takes the adult dataset as a parameter and creates a filtered
    dataframe containing the columns occupation, sex, and income.
    Then, using the newly filtered dataframe, this function creates
    a filtered dataframe for males with >50k income and a filtered
    dataframe for males with <=50k income. Then, these dataframes
    are merged together, where computed columns are also created
    to showcase the percentage of males with >50k income and males with
    <=50k income for each occupation. This process is then repeated
    for females. Once male and female dataframes are created with
    computed columns, subplots are made using plotly to compare
    percentages between males and females with income >50K or <=50k
    by occupation. Lastly, this function returns the merged male
    dataframe for testing purposes.
    """
    gender_occ_income = data[['occupation', 'sex', 'income']]
    gender_occ_income = gender_occ_income.dropna()
    is_gt_50k = gender_occ_income['income'] == ' >50K'
    is_le_50k = gender_occ_income['income'] == ' <=50K'
    is_male = gender_occ_income['sex'] == ' Male'
    # Filter and create dataframe for males with >50k income
    me_gt_50k = gender_occ_income[is_gt_50k & is_male]
    me_occ_gt_50k = me_gt_50k.groupby('occupation')['income'].count()
    male_occ_gt_50k = me_occ_gt_50k.reset_index(name='Count (>50K)')
    # Fill in missing occupation in >50k income dataframe
    male_occ_count_gt_50k = pd.concat([male_occ_gt_50k.loc[0:7],
                                      pd.DataFrame({'occupation':
                                       ' Priv-house-serv', 'Count (>50K)': 0},
                                      index=[0]),
                                      male_occ_gt_50k.loc[8:]],
                                      ignore_index=True)
    # Filter and create dataframe for males with <=50k income
    me_le_50k = gender_occ_income[is_le_50k & is_male]
    me_occ_le_50k = me_le_50k.groupby('occupation')['income'].count()
    male_occ_count_le_50k = me_occ_le_50k.reset_index(name='Count (<=50K)')
    # Merge the two male dataframes
    merged_male_occ = male_occ_count_gt_50k.merge(male_occ_count_le_50k,
                                                  left_on='occupation',
                                                  right_on='occupation',
                                                  how='inner')
    # Make computed columns
    merged_male_occ.loc[:, 'Total'] = merged_male_occ.sum(axis=1)
    merged_male_occ['Percent (>50K)'] = ((merged_male_occ['Count (>50K)']
                                          / merged_male_occ['Total'])) * 100
    merged_male_occ['Percent (<=50K)'] = ((merged_male_occ['Count (<=50K)']
                                           / merged_male_occ['Total'])) * 100
    merged_male_occ = merged_male_occ.round(2)
    is_female = gender_occ_income['sex'] == ' Female'
    # Filter and create dataframe for females with >50k income
    fe_gt_50k = gender_occ_income[is_gt_50k & is_female]
    fe_occ_gt_50k = fe_gt_50k.groupby('occupation')['income'].count()
    female_occ_gt_50k = fe_occ_gt_50k.reset_index(name='Count (>50K)')
    # Filter and create dataframe for females with <=50k income
    fe_le_50k = gender_occ_income[is_le_50k & is_female]
    fe_occ_le_50k = fe_le_50k.groupby('occupation')['income'].count()
    female_occ_count_le_50k = fe_occ_le_50k.reset_index(name='Count (<=50K)')
    # Merge the two female dataframes
    merged_fe_occ = female_occ_gt_50k.merge(female_occ_count_le_50k,
                                            left_on='occupation',
                                            right_on='occupation',
                                            how='inner')
    # Make computed columns
    merged_fe_occ.loc[:, 'Total'] = merged_fe_occ.sum(axis=1)
    merged_fe_occ['Percent (>50K)'] = ((merged_fe_occ['Count (>50K)']
                                        / merged_fe_occ['Total'])) * 100
    merged_fe_occ['Percent (<=50K)'] = ((merged_fe_occ['Count (<=50K)']
                                         / merged_fe_occ['Total'])) * 100
    merged_fe_occ = merged_fe_occ.round(2)
    # Create subplots for both males and females
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Percentage of "
                                                        "Males with "
                                                        "Income >50K "
                                                        "or <=50K by "
                                                        "Occupation",
                                                        "Percentage of "
                                                        "Females with "
                                                        "Income >50K "
                                                        "or <=50K by "
                                                        "Occupation"))
    # Male plot on left
    male_x = go.Bar(name='<=50K Male', x=merged_male_occ['occupation'],
                    y=merged_male_occ['Percent (<=50K)'],
                    marker_color='mediumpurple')
    male_y = go.Bar(name='>50K Male', x=merged_male_occ['occupation'],
                    y=merged_male_occ['Percent (>50K)'],
                    marker_color='mediumseagreen')
    fig.add_trace(male_x, row=1, col=1)
    fig.add_trace(male_y, row=1, col=1)
    fig.update_xaxes(title_text="Occupation", row=1, col=1)
    fig.update_yaxes(title_text="Percentage", row=1, col=1)
    # Female plot on right
    female_x = go.Bar(name='<=50K Female', x=merged_fe_occ['occupation'],
                      y=merged_fe_occ['Percent (<=50K)'],
                      marker_color='mediumpurple')
    female_y = go.Bar(name='>50K Female', x=merged_fe_occ['occupation'],
                      y=merged_fe_occ['Percent (>50K)'],
                      marker_color='mediumseagreen')
    fig.add_trace(female_x, row=1, col=2)
    fig.add_trace(female_y, row=1, col=2)
    fig.update_xaxes(title_text="Occupation", row=1, col=2)
    fig.update_yaxes(title_text="Percentage", row=1, col=2)
    fig.update_layout(barmode='stack')
    fig.update_xaxes(tickangle=-45)
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=12)
    fig.write_html("./pct_mf_income_>50k_<=50k_occupation.html")
    return merged_male_occ


def compare_education_income(data):
    """
    Takes the adult dataset as a parameter and creates a filtered
    dataframe containing the columns education and income. Then,
    using the newly filtered dataframe, this function creates
    a filtered dataframe for adults with >50k income and a filtered
    dataframe for adults with <=50k income. Then, these dataframes
    are merged together, where computed columns are also created
    to showcase the percentage of adults with >50k income and adults with
    <=50k income for each education level. Once the merged education
    dataframe is made, plotly is used to make stacked bar chart to
    compare percentages of adults with income >50K or <=50K by education.
    Lastly, this function returns the merged education dataframe for
    testing purposes.
    """
    education_income = data[['education', 'income']]
    education_income = education_income.dropna()
    # Filter and create dataframe for education with >50k income
    is_gt_50k = education_income['income'] == ' >50K'
    edu_gt_50k = education_income[is_gt_50k]
    edu_count_gt_50k = edu_gt_50k.groupby('education')['income'].count()
    edu_count_gt_50k_df = edu_count_gt_50k.reset_index(name='Count (>50K)')
    # Fill in missing education in >50k income dataframe
    education_count_gt_50k = pd.concat([edu_count_gt_50k_df.loc[0:12],
                                       pd.DataFrame({'education': ' Preschool',
                                                     'Count (>50K)': 0},
                                                    index=[0]),
                                        edu_count_gt_50k_df.loc[13:]],
                                       ignore_index=True)
    # Filter and create dataframe for education with <=50k income
    is_le_50k = education_income['income'] == ' <=50K'
    edu_le_50K = education_income[is_le_50k]
    edu_count_le_50k = edu_le_50K.groupby('education')['income'].count()
    education_count_le_50k = edu_count_le_50k.reset_index(name='Count (<=50K)')
    # Merge the two education dataframes
    merged_edu = education_count_gt_50k.merge(education_count_le_50k,
                                              left_on='education',
                                              right_on='education',
                                              how='inner')
    # Make computed columns
    merged_edu.loc[:, 'Total'] = merged_edu.sum(axis=1)
    merged_edu['Percent (>50K)'] = ((merged_edu['Count (>50K)']
                                     / merged_edu['Total'])) * 100
    merged_edu['Percent (<=50K)'] = ((merged_edu['Count (<=50K)']
                                      / merged_edu['Total'])) * 100
    merged_edu = merged_edu.round(2)
    # Plot percentages for adults with income >50K or <=50k by education
    fig = go.Figure(data=[
        go.Bar(name='<=50K', x=merged_edu['education'],
               y=merged_edu['Percent (<=50K)'],
               marker_color='steelblue'),
        go.Bar(name='>50K', x=merged_edu['education'],
               y=merged_edu['Percent (>50K)'],
               marker_color='indianred')
    ])
    fig.update_layout(barmode='stack', title="Percentage of Adults "
                                             "with Income >50K or <=50K "
                                             "by Education",
                      xaxis_title="Education Level",
                      yaxis_title="Percentage")
    fig.write_html("./pct_income_>50k_<=50k_education.html")
    return merged_edu
