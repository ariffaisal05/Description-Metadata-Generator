import argparse
import pandas as pd
from dotenv import load_dotenv
from table_context import read_table_context, generate_table_description
from column_descriptions import generate_column_descriptions
from exportDescriptions import export_to_greenplum
from importTable import import_from_denodo

def main():
    parser = argparse.ArgumentParser(description="Dataset Documentation Generator")
    parser.add_argument("--csv_path", help="Path to CSV file")
    parser.add_argument("--strategy", choices=["consecutive", "random"], default="random")
    parser.add_argument("--max_columns", type=int, default=8)
    parser.add_argument("--max_rows", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=5)
    args = parser.parse_args()

    # # Example to import from Denodo
    # df = import_from_denodo(table_name="bv_wins_insera_app_fd_ticket_unmask", schema="public", head=2)
    # print (df.head())
    
    if not args.csv_path:
        df = pd.read_csv("/Users/mariff/VScode/Intern/DescAIV07/Data/data_validation.csv", delimiter=";")
        metadf = pd.read_excel("/Users/mariff/VScode/Intern/DescAIV07/Data/definition_metadata.xlsx")
    else:
        df = pd.read_csv(args.csv_path, delimiter=";")

    # Sample cells from dataset to get context
    sampled_context, context_columns, strategy_used = read_table_context(
        df, strategy=args.strategy, max_columns=args.max_columns, max_rows=args.max_rows
    )

    # Generate table description based on context from sampled cells & file name
    table_desc = generate_table_description(
        sampled_context, context_columns, args.csv_path
    )
    print(f"\n📄 Table Description:\n{table_desc}\n")

    if not args.csv_path:
        df_desc=generate_column_descriptions(
            df, metadf, table_desc, args.strategy, batch_size=args.batch_size
        )
    else:
        # Generate column descriptions based on the table description and sampled columns/cells
        df_desc=generate_column_descriptions(
            df, metadf, table_desc, args.strategy, args.csv_path, batch_size=args.batch_size
        )
    
    # Ask user to write output to Greenplum database
    printDB = input("Export to Greenplum Database? (y/n)")
    if printDB != "y":
        return
    else:
        export_to_greenplum(df_desc, table_name="descriptions", schema="public")

if __name__ == "__main__":
    main()

