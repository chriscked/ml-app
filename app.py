import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from pycaret.datasets import get_data
from pycaret.classification import *

from IPython.display import display



def main():
    
    st.title('ML App')
    
    activities = ['EDA', 'Plot', 'Model Building', 'Alternative ML', 'ML']
    choice = st.sidebar.selectbox('Select Activity', activities)

    if choice == 'EDA':
        st.subheader('Exploratory Data Analysis')

        data = st.file_uploader('Upload Dataset', type=['csv'])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox('Show shape'):
                st.write(df.shape)

            if st.checkbox('Show Columns'):
                all_columns = df.columns.to_list()
                st.write(all_columns)

            if st.checkbox('Select Columns To Show'):
                all_columns = df.columns.to_list()
                selected_columns = st.multiselect('Select Columns', all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)

            if st.checkbox('Show Summary'):
                st.write(df.describe())

            if st.checkbox('Show Value Counts'):             
                st.write(df.iloc[:, 3].value_counts())

    elif choice == 'Plot':
        st.subheader('Data Visualisation')

        data = st.file_uploader('Upload Dataset', type=['csv'])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox('Correlation with Seaborn'):
                st.write(sns.heatmap(df.corr(), annot=True))
                st.pyplot()

            if st.checkbox('Pie Chart'):
                all_columns = df.columns.to_list()
                columns_to_plot = st.selectbox('Select 1 Column', all_columns)
                pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct='%1.1f%%')
                st.write(pie_plot)
                st.pyplot()
            
            all_columns_names = df.columns.to_list()
            type_of_plot = st.selectbox('Select Type of Pot', ['area', 'bar', 'line', 'hist', 'box', 'kde'])
            selected_columns_names = st.multiselect('Select Columns To Plot', all_columns_names)

            if st.button('Generate Plot'):
                st.success('Generating Customizable Plot of {} for {}'.format(type_of_plot, selected_columns_names ))

                if type_of_plot == 'area':
                    cust_data = df[selected_columns_names]
                    st.area_chart(cust_data)

                elif type_of_plot == 'bar':
                    cust_data = df[selected_columns_names]
                    st.bar_chart(cust_data)

                elif type_of_plot == 'line':
                    cust_data = df[selected_columns_names]
                    st.line_chart(cust_data)

                elif type_of_plot:
                    cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
                    st.write(cust_data)
                    st.pyplot()
                

    elif choice == 'Model Building':
        st.subheader('Building ML Model')

        data = st.file_uploader('Upload Dataset', type=['csv'])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))

            X = df.iloc[:,0:-1]

            Y = df.iloc[:, -1]            

            seed = 7

            models = []
            models.append(('LR', LogisticRegression()))
            models.append(('LDA', LinearDiscriminantAnalysis()))
            models.append(('KNN', KNeighborsClassifier()))
            models.append(('CART', DecisionTreeClassifier()))
            models.append(('NB', GaussianNB()))
            models.append(('SVM', SVC()))

            model_names = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = 'accuracy'

            for name, model in models:
                kfold = model_selection.KFold(n_splits=10, random_state=seed)
                cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
                model_names.append(name)
                model_mean.append(cv_results.mean())
                model_std.append(cv_results.std())
                accuracy_results = {'model_name': name, 'model_accuracy':cv_results.mean(), 'standard_deviation':cv_results.std()}
                all_models.append(accuracy_results)

            if st.checkbox('Metrics as Table'):
                st.dataframe(pd.DataFrame(zip(model_names, model_mean, model_std)))

    elif choice == 'Alternative ML':
        st.subheader('Alternative ML')

        data = st.file_uploader('Upload Dataset', type=['csv'])
        if data is not None:

            df = pd.read_csv(data)
            st.dataframe(df.head(10))
            df1=df.drop(columns=['Species'])
            data_types = df1.dtypes.value_counts()
            st.write(df1)
            if data_types.object:
                st.write('yes')

            else:
                st.write('no')

    
    elif choice == 'ML':
        st.subheader('ML')
                 

        uploaded_data = st.file_uploader('Upload Dataset', type=['csv'])
        if uploaded_data is not None:
            dataset = pd.read_csv(uploaded_data)
            st.dataframe(dataset.head(10))

            #data preparation
            #data = dataset.sample(frac=0.95, random_state=786).reset_index(drop=True)
            #data_unseen = dataset.drop(data.index).reset_index(drop=True)                    

            all_columns_names = dataset.columns.to_list()
            select_feature = st.selectbox('Select Feature', all_columns_names)

            model_name = ['Gaussian Process',
                            'K Nearest Neighbour', 
                            'Logistic Regression',
                            'Naives Bayes',
                            'Decision Tree',
                            'SVM (Linear)',
                            'Random Forest'
                            ]
            model_pycaret = ['gpc',
                            'knn', 
                            'lr',
                            'nb',
                            'dt',
                            'svm',
                            'rf'
                            ]
            dic=dict(zip(model_pycaret, model_name))
            select_model = st.selectbox('Select a prediction model',model_pycaret, format_func=lambda x: dic[x])            
            
            if st.button('Start Prediction Process'):
                data_load_state = st.text('Processing Data...')
                st.success('Predicting {} as target'.format(select_feature))
                
                # 1 setup function
                exp_clf101 = setup(data = dataset, target = select_feature, session_id=123,silent = True) 
                
                # 2 compare_models() - AttributeError: 'NoneType' object has no attribute 'display_id'
                #compare_models()
                # 3 create model -  choose a specific model
                lr = create_model(select_model, verbose=False)
 
                # 4 tune a model
                tuned_lr = tune_model(select_model, verbose=False)

                # 5 plot_model(tuned_dt, plot = 'auc'), plot_model(tuned_dt, plot = 'confusion_matrix')
                #st.plot_model(tuned_dt, plot = 'auc')

                # 6 evaluate model
                #t = evaluate_model(tuned_dt)

                # 7 predict model - split data - checking 70/30 split
                #pred = predict_model(tuned_dt);      
                #st.write('Initial prediction check:', pred)           
 
                #8 Finalise model - full dataset training
                #final_lr = finalize_model(tuned_lr)
                final_pred_lr = predict_model(tuned_lr)
                st.markdown('## Results: ##')
                
                st.dataframe(display(final_pred_lr[1]))              
                #st.dataframe(final_pred_lr[0])

                data_load_state.text("Done! Process Complete")
                #8 finalise model - full data without

                #9 Predict unseen data - 5% from original data
                #unseen_predictions = predict_model(final_dt, data=data_unseen)                
                #st.dataframe(unseen_predictions.head(20))                        

if __name__ == '__main__':
    main()
