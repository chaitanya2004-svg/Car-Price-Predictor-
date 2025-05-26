import streamlit as st

# Set page config (must be the first Streamlit command)
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import other libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Set pandas option to handle the warnings
pd.set_option('future.no_silent_downcasting', True)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .contact-info {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 30px;
        text-align: center;
    }
    .contact-info h3 {
        color: #4CAF50;
        margin-bottom: 10px;
    }
    .contact-info p {
        margin: 5px 0;
    }
    .metrics-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/car.png", width=100)
    st.title("Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Home", "📊 Data Analysis", "📈 Model Performance", "ℹ️ About", "📞 Contact"]
    )

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')
    
    # Convert categorical variables to numeric using proper method
    df['fuel'] = df['fuel'].map({'Diesel': 0, 'Petrol': 1, 'CNG': 2, 'LPG': 3, 'Electric': 4})
    df['seller_type'] = df['seller_type'].map({'Individual': 0, 'Dealer': 1, 'Trustmark Dealer': 2})
    df['transmission'] = df['transmission'].map({'Manual': 0, 'Automatic': 1})
    df['owner'] = df['owner'].map({
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4,
        'Test Drive Car': 5
    })
    
    return df

# Train model
@st.cache_resource
def train_model(df):
    # Extract car company name
    df['company'] = df['name'].apply(lambda x: x.split()[0])
    
    # Create dummy variables for categorical columns
    df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner', 'company'], drop_first=True)
    
    # Prepare features and target
    X = df.drop(['name', 'selling_price'], axis=1)
    y = df['selling_price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate model performance
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    return model, X.columns, r2, rmse, mae, mape, X

# Load data and train model
df = load_data()
model, feature_columns, r2_score, rmse, mae, mape, X_train = train_model(df)

# Get unique values for dropdowns
companies = sorted(df['name'].unique())
years = sorted(df['year'].unique(), reverse=True)

# Create mapping dictionaries for categorical variables
fuel_mapping = {0: 'Diesel', 1: 'Petrol', 2: 'CNG', 3: 'LPG', 4: 'Electric'}
seller_type_mapping = {0: 'Individual', 1: 'Dealer', 2: 'Trustmark Dealer'}
transmission_mapping = {0: 'Manual', 1: 'Automatic'}
owner_mapping = {
    1: 'First Owner',
    2: 'Second Owner',
    3: 'Third Owner',
    4: 'Fourth & Above Owner',
    5: 'Test Drive Car'
}

# Get unique values for dropdowns using the original data
fuels = sorted(df['fuel'].map(fuel_mapping).unique())
seller_types = sorted(df['seller_type'].map(seller_type_mapping).unique())
transmissions = sorted(df['transmission'].map(transmission_mapping).unique())
owners = sorted(df['owner'].map(owner_mapping).unique())

# Main content based on navigation
if page == "🏠 Home":
    st.title("🚗 Car Price Prediction")
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;'>
        <h2>Predict Your Car's Value</h2>
        <p>Enter the details of your car below to get an accurate price estimate.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            car_name = st.selectbox("Car Model", companies)
            year = st.selectbox("Year of Purchase", years)
            km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, step=1000)
            fuel = st.selectbox("Fuel Type", fuels)
        
        with col2:
            seller_type = st.selectbox("Seller Type", seller_types)
            transmission = st.selectbox("Transmission", transmissions)
            owner = st.selectbox("Owner", owners)
        
        submit_button = st.form_submit_button("Predict Price")

    # Make prediction when form is submitted
    if submit_button:
        # Create input data with the same structure as training data
        input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # Set the values
        input_data['year'] = year
        input_data['km_driven'] = km_driven
        
        # Set categorical variables using reverse mapping
        fuel_value = next(k for k, v in fuel_mapping.items() if v == fuel)
        seller_type_value = next(k for k, v in seller_type_mapping.items() if v == seller_type)
        transmission_value = next(k for k, v in transmission_mapping.items() if v == transmission)
        owner_value = next(k for k, v in owner_mapping.items() if v == owner)
        
        # Extract company name from full car name
        company = car_name.split()[0]
        
        # Set categorical variables
        for col in feature_columns:
            if col.startswith('fuel_') and col == f'fuel_{fuel_value}':
                input_data[col] = 1
            elif col.startswith('seller_type_') and col == f'seller_type_{seller_type_value}':
                input_data[col] = 1
            elif col.startswith('transmission_') and col == f'transmission_{transmission_value}':
                input_data[col] = 1
            elif col.startswith('owner_') and col == f'owner_{owner_value}':
                input_data[col] = 1
            elif col.startswith('company_') and col == f'company_{company}':
                input_data[col] = 1
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display result with enhanced styling
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: #4CAF50; color: white; border-radius: 10px; margin: 20px 0;'>
            <h2>Estimated Price</h2>
            <h1>₹{prediction:,.2f}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Display prediction accuracy metrics
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin: 20px 0;'>
            <h3>Prediction Accuracy</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for accuracy metrics
        acc_col1, acc_col2, acc_col3, acc_col4 = st.columns(4)
        
        with acc_col1:
            st.metric(
                "Model Accuracy",
                f"{r2_score*100:.1f}%",
                help="R² score indicates the model's overall accuracy"
            )
        
        with acc_col2:
            st.metric(
                "Average Error",
                f"₹{rmse:,.2f}",
                help="Root Mean Square Error shows the average prediction error"
            )
        
        with acc_col3:
            st.metric(
                "Absolute Error",
                f"₹{mae:,.2f}",
                help="Mean Absolute Error shows the average absolute difference"
            )
        
        with acc_col4:
            st.metric(
                "Percentage Error",
                f"{mape:.1%}",
                help="Mean Absolute Percentage Error shows the average percentage difference"
            )
        
        # Add explanation of accuracy metrics
        st.markdown("""
        <div style='padding: 15px; background-color: #f8f9fa; border-radius: 5px; margin: 10px 0;'>
            <h4>Understanding the Accuracy Metrics:</h4>
            <ul>
                <li><strong>Model Accuracy:</strong> Shows how well the model fits the data (higher is better)</li>
                <li><strong>Average Error:</strong> The typical difference between predicted and actual prices</li>
                <li><strong>Absolute Error:</strong> The average absolute difference between predictions and actual values</li>
                <li><strong>Percentage Error:</strong> The average percentage difference between predictions and actual values</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Display statistics in cards
        st.subheader("Price Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Minimum Price", f"₹{df['selling_price'].min():,.2f}")
        with col2:
            st.metric("Average Price", f"₹{df['selling_price'].mean():,.2f}")
        with col3:
            st.metric("Maximum Price", f"₹{df['selling_price'].max():,.2f}")

elif page == "📊 Data Analysis":
    st.title("📊 Data Analysis")
    
    # Price Distribution
    st.write("### Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='selling_price', bins=50, color='#4CAF50')
    plt.title('Distribution of Car Prices', pad=20)
    plt.xlabel('Price (₹)')
    plt.ylabel('Count')
    st.pyplot(fig)
    
    # Price vs Year
    st.write("### Price vs Year")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='year', y='selling_price', color='#4CAF50')
    plt.title('Car Prices by Year', pad=20)
    plt.xlabel('Year')
    plt.ylabel('Price (₹)')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Price by Fuel Type
    st.write("### Price by Fuel Type")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='fuel', y='selling_price', color='#4CAF50')
    plt.title('Car Prices by Fuel Type', pad=20)
    plt.xlabel('Fuel Type')
    plt.ylabel('Price (₹)')
    st.pyplot(fig)

elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    
    # Model Performance Metrics
    st.write("### Model Performance")
    st.markdown("""
    <div class="metrics-container">
        <h3>Accuracy Metrics</h3>
        <p>The model's performance is evaluated using multiple metrics to ensure reliable predictions:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{r2_score:.4f}", 
                 help="R² score indicates how well the model fits the data. Higher is better.")
        st.metric("RMSE", f"₹{rmse:,.2f}", 
                 help="Root Mean Square Error shows the average prediction error in rupees.")
    with col2:
        st.metric("MAE", f"₹{mae:,.2f}", 
                 help="Mean Absolute Error shows the average absolute difference between predictions and actual values.")
        st.metric("MAPE", f"{mape:.2%}", 
                 help="Mean Absolute Percentage Error shows the average percentage difference between predictions and actual values.")
    
    # Feature Importance
    st.write("### Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': abs(model.feature_importances_)
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature', color='#4CAF50')
    plt.title('Top 10 Most Important Features', pad=20)
    st.pyplot(fig)

elif page == "ℹ️ About":
    st.title("ℹ️ About")
    st.markdown("""
    <div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
        <h2>About the Car Price Prediction Model</h2>
        <p>This application uses a Random Forest Regressor to predict car prices based on various features. 
        The model was trained on a dataset of used car sales from CarDekho.</p>
        
        <h3>Features Used:</h3>
        <ul>
            <li>Car Company</li>
            <li>Year of Purchase</li>
            <li>Kilometers Driven</li>
            <li>Fuel Type</li>
            <li>Seller Type</li>
            <li>Transmission</li>
            <li>Owner</li>
        </ul>
        
        <h3>Model Performance:</h3>
        <ul>
            <li>R² Score: Indicates how well the model fits the data (higher is better)</li>
            <li>RMSE: Root Mean Square Error, shows the average prediction error in rupees</li>
            <li>MAE: Mean Absolute Error, shows the average absolute difference between predictions and actual values</li>
            <li>MAPE: Mean Absolute Percentage Error, shows the average percentage difference between predictions and actual values</li>
        </ul>
        
        <h3>Data Source:</h3>
        <p>The data used for training this model comes from CarDekho, a leading car marketplace in India.</p>
    </div>
    """, unsafe_allow_html=True)

else:  # Contact page
    st.title("📞 Contact")
    st.markdown("""
    <div class="contact-info">
        <h3>Get in Touch</h3>
        <p>For any queries or feedback about the Car Price Prediction application, please contact:</p>
        <p><strong>Name:</strong> Chaitanya</p>
        <p><strong>Email:</strong> <a href="mailto:chaitanya.ghanghas@gmail.com">chaitanya.ghanghas@gmail.com</a></p>
        <p>Feel free to reach out for any questions about the model, suggestions for improvements, or to report any issues.</p>
    </div>
    """, unsafe_allow_html=True) 