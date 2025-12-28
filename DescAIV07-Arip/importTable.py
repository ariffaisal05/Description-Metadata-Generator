import pandas as pd
import os
import jaydebeapi
from dotenv import load_dotenv
from typing import overload, Optional

@overload
def import_from_denodo(table_name:str, schema:str) -> pd.DataFrame: ...

@overload
def import_from_denodo(table_name:str, schema:str, head:int) -> pd.DataFrame: ...

def import_from_denodo(table_name:str, schema="public", head: Optional[int]=None) -> pd.DataFrame:
    load_dotenv()
    user = os.getenv("DB_USER1")
    password = os.getenv("DB_PASS1")
    
    # JDBC connection
    host = os. getenv("DB_HOST1")
    port = os.getenv("DB_PORT1") 
    dbname = os.getenv("DB_NAME1")

    jar = ".drivers/denodo-vdp-jdbcdriver-8.0-update-20240926.jar"  
    url = f"jdbc:vdb://{host}:{port}/{dbname}"
    driver = "com.denodo.vdp.jdbc.Driver"

    # Open connection
    conn = jaydebeapi.connect(driver, url, [user, password], jar)

    # Run query
    if head is not None:
        query = f"SELECT * FROM {table_name} LIMIT {head}"
    else:
        query = f"SELECT * FROM {table_name}"  

    df = pd.read_sql(query, conn)

    # print(df.head())
    print("✅ DataFrame imported from Denodo")
    return df
