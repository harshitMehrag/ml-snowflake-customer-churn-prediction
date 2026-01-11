from snowflake.snowpark import Session
from snowflake.ml.modeling.xgboost import XGBClassifier
from snowflake.ml.modeling.preprocessing import OrdinalEncoder
from snowflake.snowpark.functions import col
import os
from dotenv import load_dotenv

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

def main():
    session = get_session()
    print("❄️  Connected. Fetching data reference...")

    # 1. Get Reference to the Table (No data is downloaded yet)
    df = session.table("CUSTOMER_CHURN_DATA")
    
    # 2. Preprocessing: Convert 'CONTRACT_TYPE' (Text) to Numbers
    print("🛠️  Preprocessing data inside Snowflake...")
    encoder = OrdinalEncoder(input_cols=["CONTRACT_TYPE"], output_cols=["CONTRACT_TYPE_ENCODED"])
    df_encoded = encoder.fit(df).transform(df)

    # Drop the original text column and rename the new one
    df_ready = df_encoded.drop("CONTRACT_TYPE").rename(col("CONTRACT_TYPE_ENCODED"), "CONTRACT_TYPE")

    # 3. Split Data (80% Train, 20% Test)
    train_df, test_df = df_ready.random_split(weights=[0.8, 0.2], seed=42)
    print(f"📊 Training Set: {train_df.count()} rows | Test Set: {test_df.count()} rows")

    # 4. Train the Model (XGBoost)
    print("🧠 Training XGBoost Model (This runs on Snowflake Compute)...")
    
    # Define features and label
    feature_cols = ["TENURE_MONTHS", "MONTHLY_BILL", "DATA_USAGE_GB", "CS_CALLS", "CONTRACT_TYPE"]
    label_col = "CHURN"

    xgb_model = XGBClassifier(
        input_cols=feature_cols,
        label_cols=label_col,
        n_estimators=100,  # Number of trees
        learning_rate=0.1
    )

    # .fit() pushes the training logic to Snowflake
    xgb_model.fit(train_df)

    # 5. Evaluate Accuracy
    print("📝 Testing model accuracy...")
    result = xgb_model.predict(test_df)
    
    # Compare Prediction vs Actual
    accuracy = result.filter(col("CHURN") == col("OUTPUT_CHURN")).count() / result.count()
    print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

    # 6. Save the Model to a local file (for now) to prove it exists
    # In a real enterprise setup, we would save this to the Snowflake Model Registry
    print("💾 Model training complete.")

if __name__ == "__main__":
    main()