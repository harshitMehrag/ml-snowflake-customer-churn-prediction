 **Enterprise-grade Machine Learning pipeline built entirely within Snowflake's Data Cloud.**

This is an end-to-end Machine Learning workflow that predicts customer churn for a company. Unlike traditional workflows that pull data to a local machine, this project utilizes **Snowpark ML** to push all data processing and model training to the Snowflake cloud.

The system trains an **XGBoost** classifier on 5,000+ customer records and deploys the model as a **Vectorized User Defined Function (UDF)**, allowing business users to predict churn risk directly via SQL.

This project follows the **Compute Pushdown** pattern:
1.  **Data Generation:** Synthetic telecom data is generated and uploaded to Snowflake.
2.  **Preprocessing:** Feature engineering (Ordinal Encoding, Scaling) is executed inside Snowflake using `snowflake.ml.modeling`.
3.  **Training:** An XGBoost model is trained on Snowflake's compute clusters (preventing data egress).
4.  **Deployment:** The trained model is wrapped as a Python UDF for scalable inference.

We use - 
* **Cloud Data Platform:** Snowflake (Snowpark Python API)
* **Machine Learning:** Snowpark ML, XGBoost, Scikit-Learn
* **Language:** Python 3.11
* **Environment:** VS Code, Virtual Environment
