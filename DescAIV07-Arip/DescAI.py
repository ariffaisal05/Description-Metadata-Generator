import pandas as pd
from typing import overload, Optional

# Load LLM Client
def get_client():
    """
    Returns a new AzureOpenAI client, 
    reading environment variables at runtime.
    """
    import os
    from openai import AzureOpenAI
    from dotenv import load_dotenv

    load_dotenv() # Load environment variables from .env file if it exists

    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )


# Generate table description----------------------------------------------------------------------------
def read_table_context(df: pd.DataFrame, strategy="random", max_columns=8, max_rows=4):
    import random
    if strategy == "consecutive":
        num_cols = len(df.columns)
        N = min(max_columns, num_cols)
        start_col_idx = max(0, (num_cols - N) // 2)
        context_columns = df.columns[start_col_idx:start_col_idx + N]
    elif strategy == "random":
        shuffled_columns = df.columns.to_list()
        random.shuffle(shuffled_columns)
        context_columns = shuffled_columns[:max_columns]
    else:
        raise ValueError("strategy must be either 'consecutive' or 'random'")
    context_df = df[context_columns]

    non_empty_mask = context_df.notnull() & (context_df.astype(str).apply(lambda x: x.str.strip() != ""))
    non_empty_rows = non_empty_mask.all(axis=1)

    best_start, best_len = None, 0
    current_start, current_len = None, 0

    for idx, is_valid in enumerate(non_empty_rows):
        if is_valid:
            if current_start is None:
                current_start = idx
            current_len += 1
            if current_len > best_len:
                best_start, best_len = current_start, current_len
                if best_len >= max_rows:
                    break
        else:
            current_start, current_len = None, 0

    if best_start is not None:
        end_idx = best_start + min(best_len, max_rows)
        sampled_context = context_df.iloc[best_start:end_idx].to_string(index=False)
    else:
        sampled_context = "(No sufficient non-empty rows found in these columns.)"
        
    return sampled_context, context_columns, strategy

@overload
def generate_table_description(sampled_context:str, context_columns_used:list, file_name: str): ...

@overload
def generate_table_description(sampled_context:str, context_columns_used:list): ...
    
def generate_table_description(sampled_context:str, context_columns_used:list, file_name: Optional[str]=None):
    import datetime, sys, time

    if file_name is not None:
        system_prompt = (
        "You are an expert data documentation assistant.\n\n"
        f"You are analyzing a dataset file named '{file_name}'.\n\n"
        f"Below is a sample of {len(context_columns_used)} columns from this dataset (up to 4 rows):\n\n"
        f"{sampled_context}\n\n"
    )
    else:
        system_prompt = (
        "You are an expert data documentation assistant.\n\n"
        "You are analyzing a dataset.\n\n"
        f"Below is a sample of {len(context_columns_used)} columns from this dataset (up to 4 rows):\n\n"
        f"{sampled_context}\n\n"
    )

    user_prompt = (
        "Based on the sample, briefly describe the overall nature or topic of the dataset. "
        "You may mention the types of columns, but do not describe them individually.\n\n"
        "Return only the summary. Avoid any introductory phrases like 'Here is the description'."
    )


    for attempt in range(3):
        try:
            client = get_client()
            response = client.chat.completions.create(
                # model="deepseek/deepseek-r1-0528-qwen3-8b:free",
                model='gpt-4o-mini',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            try:
                if hasattr(e, "response") and e.response is not None:
                    error_info = e.response.json()
                    reset_ts_ms = int(error_info['error']['metadata']['headers'].get('X-RateLimit-Reset', '0'))
                    if reset_ts_ms > 0:
                        reset_ts_s = reset_ts_ms / 1000
                        reset_time = datetime.datetime.fromtimestamp(reset_ts_s)
                        now = datetime.datetime.now()
                        wait_seconds = (reset_time - now).total_seconds()
                        print(f"⚠️ Rate limit exceeded. Reset at {reset_time}")
                        if wait_seconds > 0:
                            print(f"⏳ Waiting {int(wait_seconds)}s... Exiting.")
                            sys.exit(0)
            except Exception:
                pass
            time.sleep(2 ** attempt)
# Generate table description----------------------------------------------------------------------------

# Vector RAG--------------------------------------------------------------------------------------------
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer

def search_batch(batch_cols, metadf):
    # docs = metadf.apply(lambda row: f"{str(row['column_name']).lower()}: {str(row['definition']).lower()}", axis=1).tolist()
    docs = metadf.apply(lambda row: str(row['column_name']).lower(), axis=1).tolist()

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    embeddings = model.encode(docs)

    client_qdrant = QdrantClient(host="localhost", port=6333)

    client_qdrant.recreate_collection(
        collection_name="my_collection",
        vectors_config={"size": len(embeddings[0]), "distance": "Cosine"}
    )

    points = [
        PointStruct(id=i, vector=embeddings[i], payload={"definition": str(metadf.iloc[i]['definition'])})
        for i in range(len(embeddings))
    ]
    client_qdrant.upsert(collection_name="my_collection", points=points)

    results = []
    for col in batch_cols:
        q_vector = model.encode(col.lower()).tolist()
        res = client_qdrant.search(
            collection_name="my_collection",
            query_vector=q_vector,
            limit=1
        )
        if res and res[0].score > 0.7:
            results.append(res[0].payload["definition"])
        else:
            results.append(f"- (No info found for '{col}' on RAG context)")
        
    context = "\n".join(results)
    print("=== Context for LLM ===")
    print(batch_cols)
    print(context)

    return context
# Vector RAG--------------------------------------------------------------------------------------------

# Generate column descriptions--------------------------------------------------------------------------
def extract_descriptions2(response_text, batch_cols):
    lines = [line.strip() for line in response_text.strip().split('\n') if ':' in line]
    desc_map = {}
    for line in lines:
        parts = line.split(':', 1)
        if len(parts) == 2:
            col_name, desc = parts[0].strip(), parts[1].strip()
            desc_map[col_name] = desc
    return [desc_map.get(col, "[MISSING]") for col in batch_cols]

@overload
def generate_column_descriptions(df: pd.DataFrame, metadf: pd.DataFrame, table_context: str, csv_file_path: str, strategy:str, batch_size:int): ...
  
@overload
def generate_column_descriptions(df: pd.DataFrame, metadf: pd.DataFrame, table_context: str, strategy:str, batch_size:int): ...

def generate_column_descriptions(df: pd.DataFrame, metadf: pd.DataFrame, table_context: str, csv_file_path: Optional[str] = None, strategy="random", batch_size=10):
  """
  Generates descriptions for each column in the DataFrame using an LLM.
  """

  import os, time, datetime
  import pandas as pd
  import tiktoken
  from tqdm import tqdm
  from deep_translator import GoogleTranslator

  from VectorRAG import search_batch

  # Timer
  start_time = time.time()

  # Approximate tokenizer
  encoder = tiktoken.get_encoding("o200k_base")  # compatible with o-series models
  total_tokens = 0

  descriptions = []
  raw_outputs= []
  raw_output_lines = 0
  translated_descriptions = []

  # Search for rows with N(5) consecutive nonempty cells from the sampled columns
  for batch_start in tqdm(range(0, len(df.columns), batch_size), desc="Generating descriptions"):
      batch_cols = df.columns[batch_start:batch_start + batch_size] # Sample columns based on batch size
      batch_prompts = [] # Array to store prompts for the sampled columns

      for i, col_name in enumerate(batch_cols):
          col_idx = batch_start + i

          best_start, best_len = None, 0 # Index, count to mark earliest best N consecutive rows
          current_start, current_len = None, 0 # Index, count current best consecutive rows

          # Create mask to filter empty rows from the sampeld columns
          col_series = df[col_name]
          non_empty_mask = col_series.notnull() & (col_series.astype(str).str.strip() != "")

          # Algorithm to find the earliest N consecutive nonempty rows
          for idx_row, is_valid in enumerate(non_empty_mask):
              if is_valid:
                  if current_start is None:
                      current_start = idx_row
                  current_len += 1
                  if current_len > best_len:
                      best_start, best_len = current_start, current_len
                      if best_len >= 5:
                          break
              else:
                  current_start, current_len = None, 0

          # Index the sampled rows (N consecutive nonempty)
          if best_start is not None:
              end_idx = best_start + min(best_len, 5)
              current_values = df[col_name].iloc[best_start:end_idx].astype(str).tolist() # Cell values at col_name;sampled rows
          else:
              current_values = ["[no non-empty examples found]"]

          # Write N example values of each sampled columns
          batch_prompts.append(
              f"[{i+1}]\n"
              f"Target Column: {col_name}\n"
              f"Example values: {current_values}\n"
          )

      RAG_context = search_batch(batch_cols, metadf)

      system_prompt = (
        "You are an expert data documentation assistant.\n\n"
        "You are to describe target columns prompted by the user.\n\n"
        "The user will give you RAG_context, table context, target columns, and example values\n\n"
        "\n\n### When answering:"
        "\n- Prioritize knowledge from RAG_context."
        "\n- If RAG_context has no relevant match, fallback to using only Table_context."
        "\n- If both are relevant, you may combine them."
        "Below are the rules you have to follow in describing the target columns\n\n"
        "\n\n### Format Rules:"
        "\n- Respond with **exactly one line per target column**, in the given order."
        f"\n- Your output **must contain {len(batch_cols)} lines**, no more, no less."
        "\n- IMPORTANT Each line **must begin with the column name**, followed by a colon and a brief description (e.g., `column_name: ...`)."
        "\n- Do **not** skip any column, even if uncertain — make your best guess from the examples."
        "\n- Each line must be **1 sentence maximum**."
        "\n- Do **not** include numbering, bullets, or any header/intro."
        "\n- Do **not** skip any column, even if uncertain — make your best guess from the examples."
        "\n\n### Description Guidelines:"
        "\n- Use the column name and its sample values to infer meaning."
        "\n- IMPORTANT Mention example values only if helpful to support your description."
        "\n- IMPORTANT Identify relationships between columns in the batch, if relevant and mention them"
      )

      user_prompt = (
        "Please describe the following columns:\n\n"
        f"RAG context:\n{RAG_context}\n\n"
        f"Table context:\n{table_context}\n\n"
        f"There are {len(batch_cols)} target columns to describe, listed below:\n\n"
        + "\n---\n".join(batch_prompts) + "\n\n"
      )

      # Estimate number of token used in Input
      token_count_in = len(encoder.encode(system_prompt)) + len(encoder.encode(user_prompt))
      total_tokens += token_count_in

      # Attempt to get output from LLM
      for attempt in range(3):
          try:
              # Call LLM
              client = get_client()
              response = client.chat.completions.create(
                  model="gpt-4o-mini",
                  messages=[
                      {"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}
                  ],
              )
              raw_response = response.choices[0].message.content
              # Estimate number of token used in Output
              token_count_out = len(encoder.encode(raw_response))
              total_tokens += token_count_out

              expected_count = len(batch_cols)
              # batch_descriptions = extract_descriptions(raw_response, expected_count)
              batch_descriptions = extract_descriptions2(raw_response, batch_cols)

              # TroubleShooting-----------------------
              # Split raw_response into cleaned lines
              raw_output_lines = [line.strip() for line in raw_response.strip().split('\n') if line.strip()]
              # Normalize length to match number of columns
              if len(raw_output_lines) < expected_count:
                  raw_output_lines += ['-'] * (expected_count - len(raw_output_lines))  # pad with blanks
              elif len(raw_output_lines) > expected_count:
                  raw_output_lines = raw_output_lines[:expected_count]  # truncate extra lines
              #----------------------------------------
              break

          # Excception handler: LLM insufficient token quota
          except Exception as e:
              print(f"Error on batch starting at column {batch_start}, attempt {attempt+1}: {e}")
              try:
                  if hasattr(e, "response") and e.response is not None:
                      error_info = e.response.json()
                      reset_ts_ms = int(error_info['error']['metadata']['headers'].get('X-RateLimit-Reset', '0'))
                      if reset_ts_ms > 0:
                          reset_ts_s = reset_ts_ms / 1000
                          reset_time = datetime.datetime.fromtimestamp(reset_ts_s)
                          now = datetime.datetime.now()
                          wait_seconds = (reset_time - now).total_seconds()
                          print(f"⚠️ Rate limit exceeded. Reset at {reset_time.strftime('%Y-%m-%d %H:%M:%S')}")
                          if wait_seconds > 0:
                              print(f"⏳ Waiting for {int(wait_seconds)} seconds before retrying...")
              except Exception as parse_err:
                  print("⚠️ Could not parse rate limit reset time:", parse_err)

              time.sleep(2 ** attempt)
      else:
          # LLM failed to respond after 3 tries
          batch_descriptions = ["[ERROR: Failed to get description after retries]"] * len(batch_cols)

      # Append output to descriptions array
      for col, desc in zip(batch_cols, batch_descriptions):
          descriptions.append(desc)

      for raw_output_line in raw_output_lines:
          raw_outputs.append(raw_output_line)
       
  for desc in descriptions:
    try:
        translated = GoogleTranslator(source='auto', target='id').translate(desc)
    except Exception as e:
        print(f"Translation error: {e}")
        translated = "[ERROR_TRANSLATING]"
    translated_descriptions.append(translated)

  end_time = time.time()
  elapsed = end_time - start_time

  # Automate file naming
  suffix = "_consec" if strategy == "consecutive" else "_random"
  timestamp = timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") #Unique timestamp
  if csv_file_path is not None:
    filename = os.path.splitext(os.path.basename(csv_file_path))[0]
    output_path = f"Output/0_Consec/outputAIV07_{filename}{suffix}_{batch_size}batch_{timestamp}.csv" if strategy == "consecutive" else f"Output/1_Random/outputAIV07_{filename}{suffix}_{batch_size}batch_{timestamp}.csv"
  else:
    output_path = f"Output/0_Consec/outputAIV07_{suffix}_{batch_size}batch_{timestamp}.csv" if strategy == "consecutive" else f"Output/1_Random/outputAIV07_{suffix}_{batch_size}batch_{timestamp}.csv"
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  output_df = pd.DataFrame({
    "column_name": df.columns,
    "description": descriptions,
    "description_bahasa": translated_descriptions,
    "raw_output": raw_outputs
  })
  # Add a stat row for total tokens and elapsed time
  summary_rows = pd.DataFrame({
      "column_name": ["[TOTAL_TOKENS]", "[TIME_ELAPSED_SECONDS]"],
      "description": [str(total_tokens), f"{elapsed:.2f}"]
  })
  # Append stat rows to the DataFrame
#   output_df = pd.concat([output_df, summary_rows], ignore_index=True)
  output_df.to_csv(output_path, index=False)

  print("=============================================================")
  print(f"\033[92mDone! Descriptions saved to {output_path}\033[0m")
  print(f"\033[93mTotal tokens used (approx): {total_tokens}\033[0m")
  print(f"\033[93mTime taken: {elapsed:.2f} seconds\033[0m")

  return output_df
# Generate column descriptions--------------------------------------------------------------------------



# Export------------------------------------------------------------------------------------------------
def export_to_greenplum(df: pd.DataFrame, table_name="DescAI_test", schema="public", if_exists="append", index=False):
    import os
    from sqlalchemy import create_engine
    from dotenv import load_dotenv

    load_dotenv() # Load environment variables from .env file if it exists
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
# Export------------------------------------------------------------------------------------------------



# Import -----------------------------------------------------------------------------------------------
@overload
def import_from_denodo(table_name:str, schema:str): ...

@overload
def import_from_denodo(table_name:str, schema:str, head:int): ...

def import_from_denodo(table_name:str, schema="public", head: Optional[int]=None) -> pd.DataFrame:
    import os
    import jaydebeapi
    from dotenv import load_dotenv
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

    print(df.head())
    print("✅ DataFrame imported from Denodo")
    return df
# Import -----------------------------------------------------------------------------------------------

def DescAI(df: pd.DataFrame, metadf: pd.DataFrame, table_name="DescAI_test", schema="public", if_exist="append", index=False):
    # Example to import from Denodo
    # df = import_from_denodo(table_name="bv_wins_insera_app_fd_ticket_unmask", schema="public", head=2)
    # print (df)
    
    sampled_context, context_columns, strategy_used = read_table_context(df)

    table_context = generate_table_description(sampled_context, context_columns)
    print(f"\n📄 Table Description:\n{table_context}\n")

    df_desc=generate_column_descriptions(df, metadf, table_context, strategy=strategy_used)

    # Ask user to write output to Greenplum database
    printDB = input("Export to Greenplum Database? (y/n)")
    if printDB != "y":
        return
    else:
        export_to_greenplum(df_desc, table_name=table_name, schema=schema, if_exists=if_exist, index=index)

