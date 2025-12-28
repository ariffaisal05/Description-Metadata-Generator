import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Function to export DataFrame to Greenplum database
def export_to_greenplum(df: pd.DataFrame, table_name="DescAI_AutoWrite", schema="public", if_exists="append", index=False) -> None:
    """
    Exports a DataFrame to a Greenplum database table.
    """
    
    # Load environment variables from .env file if it exists
    load_dotenv() 
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    host = os. getenv("DB_HOST")
    port = os.getenv("DB_PORT") 
    dbname = os.getenv("DB_NAME")

    # Greenplum uses PostgreSQL driver
    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")
    print(f"Connecting to Greenplum...")

    # Write dataframe to table 
    df.to_sql(
        table_name,    # target table name
        engine,
        schema=schema,                # schema name
        if_exists=if_exists,  # "append" if you don’t want to overwrite
        index=index
    )
    print("✅ DataFrame exported to Greenplum")