import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import randint,loguniform
import shap
import joblib


df = pd.read_csv('insurance.csv',encoding='latin1')

null_values = df.isnull().sum()


bins = [0,20,30,40,np.inf]
labels = ['UnderWeight','Normal weight','Over weight','obesity']

df['bmi_category'] = pd.cut(df['bmi'],labels=labels,bins=bins,right=False)

x = df.drop('charges',axis=1)
y = df['charges']


numerical_features = x.select_dtypes(include=np.number).columns.to_list()
categorial_features = x.select_dtypes(include=['object','category']).columns.to_list()



categorical_data = df[categorial_features]
numerical_data = df[numerical_features]


df_encoded = pd.get_dummies(df,columns=categorial_features,drop_first=False)

correlation = df_encoded.corr(numeric_only=True)



#ModeL Evalutaion and Training

numerical_transformer  = StandardScaler()
categorial_transformer = OneHotEncoder(handle_unknown='ignore')




preprocessor = ColumnTransformer(
    transformers=[
        ('num',numerical_transformer,numerical_features),
        ('cat',categorial_transformer,categorial_features)
    ],
    remainder='passthrough'
)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)

#Linear Regression

pipeline_lr = Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('regressor',LinearRegression())
    ]
)

pipeline_lr.fit(x_train,y_train)

y_pred_lr  = pipeline_lr.predict(x_test)

print(f"Linear Regression Training R^2 score: {pipeline_lr.score(x_train, y_train):.4f}")
print(f"Linear Regression Test R^2 score: {pipeline_lr.score(x_test, y_test):.4f}")

mae_lr = mean_absolute_error(y_test,y_pred_lr)
rmse_lr = root_mean_squared_error(y_test,y_pred_lr)

r2_lr = r2_score(y_test,y_pred_lr)

print('MAE of Linear Regression :-', mae_lr)
print('RMSE of Linear Regression :-', rmse_lr)
print('R2 Score of Linear Regression :-',r2_lr)

#Random Forest Regressor with RandomSearchCV

pipeline_rf = Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('regressor',RandomForestRegressor(random_state=42))
    ]
)

params_rf = {
    'regressor__n_estimators' : randint(10,200),
    'regressor__max_depth' : randint(3,20),
    'regressor__min_samples_split' : randint(2,11),
    'regressor__min_samples_leaf' : randint(1,11)
}

random_search_rf = RandomizedSearchCV(
    pipeline_rf,
    param_distributions=params_rf,
    n_iter=20,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)

random_search_rf.fit(x_train,y_train)


print(f'\nBest Parameters for RandomSearch : {random_search_rf.best_params_}')
print(f"Best cross-validation score (negative MSE): {random_search_rf.best_score_:.4f}")

best_model_rf = random_search_rf.best_estimator_
y_pred_rfc_tuned = best_model_rf.predict(x_test)

model_score = best_model_rf.score(x_test,y_test)

print(f"Random Forest Regressor Test R^2 score: {model_score:.4f}")

mae_rf = mean_absolute_error(y_test,y_pred_rfc_tuned)
rmse_rf = root_mean_squared_error(y_test,y_pred_rfc_tuned)
r2_rf = r2_score(y_test,y_pred_rfc_tuned)

print('MAE of Random forest regressor :-', mae_rf)
print('RMSE of Random forest regressor :-', rmse_rf)
print('R2 score of Random Forest :-', r2_rf)

preprocessor_shap = best_model_rf.named_steps['preprocessor']
regressor_shap = best_model_rf.named_steps['regressor']

x_train_processed = preprocessor_shap.transform(x_train)

feature_names = numerical_features + list(preprocessor_shap.named_transformers_['cat'].get_feature_names_out(categorial_features))

explainer = shap.TreeExplainer(regressor_shap)

shap_values = explainer.shap_values(x_train_processed)

print('Generating Summary plot for Random Forest....')
shap.summary_plot(shap_values,x_train_processed,feature_names=feature_names,show=False)
plt.title('Shap summary plot random forest')
plt.show()

print('Generating Dependence plot for random forest...')
shap.dependence_plot(
    'smoker_yes',
    shap_values,
    x_train_processed,
    feature_names=feature_names,
    show=False
)
plt.title('Shap Dependence plot for smoker status')
plt.show()

print('Generating shap force plot for a single prediction')
shap.initjs()

shap_value_test = explainer.shap_values(preprocessor_shap.transform(x_test.iloc[[0]]))

force_plot = shap.force_plot(
    explainer.expected_value,
    shap_value_test,
    features=preprocessor_shap.transform(x_test.iloc[[0]]),
    feature_names=feature_names
)

force_plot

#Ridge Regression with tuning
pipeline_rr = Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('regressor',Ridge(random_state=42))
    ]
)

param_rr = {
    'regressor__alpha' : loguniform(1e-4,1e2)
}
random_search_rr = RandomizedSearchCV(
    pipeline_rr,
    param_distributions=param_rr,
    n_iter=10,
    cv=5,
    n_jobs=-1,
    random_state=42
)


random_search_rr.fit(x_train,y_train)

y_pred_rr = random_search_rr.predict(x_test)

print(f'Best Parameters for Ridge :- {random_search_rr.best_params_}')
print(f'Best cross validation score :- {random_search_rr.best_score_}')

best_model = random_search_rr.best_estimator_


model_score_rr = best_model.score(x_test,y_test)

print(f"Ridge Test R^2 score: {model_score_rr}")

mae_rr = mean_absolute_error(y_test,y_pred_rr)
rmse_rr = root_mean_squared_error(y_test,y_pred_rr)
r2_rr = r2_score(y_test,y_pred_rr)


print('Ridge MAE :-', mae_rr)
print('Ridge RMSE :-', rmse_rr)
print('R2 Score of Ridge :-', r2_rr)


print('Null Values :-', null_values)
print('Final:-',df.head())



results = pd.DataFrame({
    'Model' : ['Linear Regression','Random Forest','Ridge'],
    'R-squared' : [r2_lr,r2_rf,r2_rr],
    'MAE' : [mae_lr,mae_rf,mae_rr],
    'RMSE' : [rmse_lr,rmse_rf,rmse_rr]
})

results_melted = results.melt('Model',var_name='Metric',value_name='Score')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# R-squared Plot
sns.barplot(ax=axes[0], x='Model', y='Score', data=results_melted[results_melted['Metric'] == 'R-squared'], palette='viridis')
axes[0].set_title('R-squared Score', fontsize=14)
axes[0].set_ylabel('Score')
axes[0].set_ylim(0, 1.0)

# MAE Plot
sns.barplot(ax=axes[1], x='Model', y='Score', data=results_melted[results_melted['Metric'] == 'MAE'], palette='plasma')
axes[1].set_title('Mean Absolute Error (MAE)', fontsize=14)
axes[1].set_ylabel('Error')
# Adjust ylim for MAE to better show differences
axes[1].set_ylim(0, max(results['MAE']) * 1.1)


# RMSE Plot
sns.barplot(ax=axes[2], x='Model', y='Score', data=results_melted[results_melted['Metric'] == 'RMSE'], palette='magma')
axes[2].set_title('Root Mean Squared Error (RMSE)', fontsize=14)
axes[2].set_ylabel('Error')
# Adjust ylim for RMSE
axes[2].set_ylim(0, max(results['RMSE']) * 1.1)

plt.suptitle('Model Performance Comparison', fontsize=18, y=1.02)
plt.tight_layout()
plt.show()



#Visuals

plt.figure()
sns.histplot(data=df,x='age')
plt.title('Age distribution')

plt.figure()
sns.histplot(data=df,x='bmi')
plt.title('BMI distribution')

plt.figure()
sns.histplot(data=df,x='charges')
plt.title('Charges distribution')

plt.figure()
sns.heatmap(correlation,annot=True)
plt.title('Correlation heatmap')

plt.figure()
sns.barplot(data=df,x='age',y='charges',palette='viridis')
plt.title('Charges Vs Age distribution')

plt.figure()
sns.boxplot(data=df,x='smoker',hue='charges',palette='rainbow')



plt.show()



#Exporting the RANDOM FOREST REGRESSOR and Features Columns to Joblib

joblib.dump(best_model_rf,'best_model_rf.joblib')
joblib.dump(x.columns.to_list(),'feature_columns.joblib')