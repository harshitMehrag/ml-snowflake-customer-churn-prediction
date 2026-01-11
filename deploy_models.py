from snowflake.snowpark import Session
from snowflake.ml.modeling.xgboost import XGBClassifier
from snowflake.ml.modeling.preprocessing import OrdinalEncoder
from snowflake.snowpark.functions import col
# FIXED IMPORTS: Use the generic types, not the 'Type' classes
from snowflake.snowpark.types import PandasDataFrame, PandasSeries 
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
    print("❄️  Connected to Snowflake.")

    # 1. Prepare Data
    df = session.table("CUSTOMER_CHURN_DATA")
    encoder = OrdinalEncoder(input_cols=["CONTRACT_TYPE"], output_cols=["CONTRACT_TYPE_ENCODED"])
    df_ready = encoder.fit(df).transform(df).drop("CONTRACT_TYPE").rename(col("CONTRACT_TYPE_ENCODED"), "CONTRACT_TYPE")

    # 2. Train Model
    print("🧠 Training Final Model...")
    xgb_snowflake = XGBClassifier(
        input_cols=["TENURE_MONTHS", "MONTHLY_BILL", "DATA_USAGE_GB", "CS_CALLS", "CONTRACT_TYPE"],
        label_cols="CHURN",
        n_estimators=100
    )
    xgb_snowflake.fit(df_ready)

    # CRITICAL FIX: Extract the actual native XGBoost model to use inside the UDF
    # The 'xgb_snowflake' object generates SQL; 'native_model' runs Python.
    native_model = xgb_snowflake.to_xgboost()

    # 3. Register as SQL Function
    print("🚀 Deploying model as SQL Function: PREDICT_CHURN...")
    
    session.add_packages("snowflake-snowpark-python", "xgboost", "pandas", "scikit-learn")
    from snowflake.snowpark.functions import udf
    
    # We use a temporary stage for the UDF artifacts to avoid path errors
    session.sql("CREATE STAGE IF NOT EXISTS MODEL_STAGE").collect()

    @udf(name="PREDICT_CHURN", is_permanent=True, stage_location="@MODEL_STAGE", replace=True)
    def predict_churn_udf(df: PandasDataFrame[int, float, float, int, int]) -> PandasSeries[int]:
        # Rename columns to match what the model expects
        df.columns = ["TENURE_MONTHS", "MONTHLY_BILL", "DATA_USAGE_GB", "CS_CALLS", "CONTRACT_TYPE"]
        
        # Use the native model for prediction
        return native_model.predict(df)

    print("✅ Model Deployed! You can now use PREDICT_CHURN() in SQL.")

if __name__ == "__main__":
    main()