import os

def get_lightrag_working_dir() -> str:
    """
    Returns the resolved working directory for LightRAG based on environment variables.
    """
    data_path = os.getenv("DATA_PATH", "/app/data")
    lightrag_dir = os.getenv("LIGHTRAG_DIR", "lightrag_cache")
    return os.path.join(data_path, lightrag_dir)

def get_naiverag_working_dir() -> str:
    """
    Returns the resolved working directory for NaiveRAG based on the environment configuration.
    """
    data_path = os.getenv("DATA_PATH", "/app/data")
    lightrag_dir = os.getenv("LIGHTRAG_DIR", "lightrag_cache")
    naive_dir = "corporate_naive_data" if lightrag_dir == "corporate_lightrag_cache" else "naive_data"
    return os.path.join(data_path, naive_dir)
