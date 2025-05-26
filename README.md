**🚗 Car Price Predictor**
A machine learning project that predicts the selling price of a used car based on its features such as year of manufacture, kilometers driven, fuel type, transmission, and more. This project aims to assist sellers and buyers in making informed decisions in the used car market.


**🎯 Features**
- Preprocessing of real-world used car dataset
- Exploratory Data Analysis (EDA)
- Feature engineering and transformation
- Model training using multiple regression algorithms
- Hyperparameter tuning
- Model evaluation with metrics such as R² Score
- Persistent model using joblib for future deployment

  **🧰 Tech Stack**
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Joblib


  📊 Dataset
 The dataset includes columns such as:

- Year
- Present_Price (in lakhs)
- Kms_Driven
- Fuel_Type
- Seller_Type
- Transmission
- Owner
- Selling_Price (target variable)

The data is preprocessed by:
- Converting categorical variables using one-hot encoding
- Creating a new feature car_age = current_year - year
- Dropping redundant columns


  🧠 Model Training
The following steps are used:

- Train-Test Split: Using train_test_split with 80-20 ratio
- Model: Random Forest Regressor
- Hyperparameter Tuning: Grid Search / RandomizedSearchCV
- Evaluation Metrics: R² Score, MAE, MSE


🛠️ Future Work
- Deploy the model using Streamlit or Flask
- Integrate with a web frontend for real-time prediction
- Improve model accuracy with larger datasets
- Add more relevant features (e.g., region, brand reputation)
