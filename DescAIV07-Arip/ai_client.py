from openai import AzureOpenAI

# Load LLM Client
def get_client() -> AzureOpenAI:
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
