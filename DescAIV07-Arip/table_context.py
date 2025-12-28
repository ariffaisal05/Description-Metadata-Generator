import datetime, sys, time
from ai_client import get_client
from typing import overload, Optional

def read_table_context(df, strategy="random", max_columns=8, max_rows=4):
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
def generate_table_description(sampled_context:str, context_columns_used:list, file_name: str) -> str: ...

@overload
def generate_table_description(sampled_context:str, context_columns_used:list) -> str: ...
    
def generate_table_description(sampled_context:str, context_columns_used:list, file_name: Optional[str]=None) -> str:
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
