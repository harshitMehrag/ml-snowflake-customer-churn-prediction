from snowflake.snowpark import Session
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

def get_session():
    connection_parameters = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA")
    }
    return Session.builder.configs(connection_parameters).create()

def generate_churn_data():
    print("🎲 Generating 5,000 synthetic customers...")
    np.random.seed(42)
    n_rows = 5000
    
    data = {
        'CUSTOMER_ID': range(1, n_rows + 1),
        'TENURE_MONTHS': np.random.randint(1, 72, n_rows),
        'MONTHLY_BILL': np.random.uniform(30, 150, n_rows).round(2),
        'DATA_USAGE_GB': np.random.uniform(0, 100, n_rows).round(1),
        'CS_CALLS': np.random.randint(0, 10, n_rows),
        # Random contract type: 0=Month-to-Month, 1=1 Year, 2=2 Year
        'CONTRACT_TYPE': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_rows),
    }
    
    df = pd.DataFrame(data)
    
    # Logic: More calls + High Bill + Month-to-Month = High Churn Probability
    df['CHURN_SCORE'] = (
        (df['CS_CALLS'] * 0.1) + 
        (df['MONTHLY_BILL'] / 200) - 
        (df['TENURE_MONTHS'] / 100) +
        (df['CONTRACT_TYPE'].apply(lambda x: 0.3 if x == 'Month-to-Month' else 0))
    )
    
    # Convert score to Binary Churn (0 or 1)
    df['CHURN'] = (df['CHURN_SCORE'] + np.random.normal(0, 0.1, n_rows)) > 0.6
    df['CHURN'] = df['CHURN'].astype(int)
    
    return df.drop(columns=['CHURN_SCORE'])

if __name__ == "__main__":
    session = get_session()
    
    # Generate Data
    pdf = generate_churn_data()
    
    # Create Snowpark DataFrame
    sdf = session.create_dataframe(pdf)
    
    # Write to Snowflake
    print("❄️  Uploading data to Snowflake table: CUSTOMER_CHURN_DATA...")
    sdf.write.mode("overwrite").save_as_table("CUSTOMER_CHURN_DATA")
    print("✅ Success! Data is ready for training.")