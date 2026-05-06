import os
import sys
from datasets import load_dataset

def download_datasets():
    print("Descargando datasets de FlashRAG: MSMarco-QA y Hotpot-QA")
    # Este es un script de inicialización de ejemplo. 
    # FlashRAG normalmente gestiona sus propios métodos de descarga o usa HuggingFace datasets.
    # Aquí se pueden añadir las llamadas específicas para inicializar la carpeta compartida '/app/data'
    
    data_dir = "/app/data"
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Ejemplo con la librería datasets de HuggingFace si están subidos ahí
        # from datasets import load_dataset
        load_dataset("ms_marco", "v1.1", cache_dir=os.path.join(data_dir, "msmarco-qa"))
        load_dataset("hotpot_qa", "distractor", cache_dir=os.path.join(data_dir, "hotpot-qa"))
        print(f"Los datos se almacenarán en: {data_dir}")
        print("La descarga de MSMarco-QA y Hotpot-QA ha sido orquestada.")
    except Exception as e:
        print(f"Error al descargar datos: {e}")

if __name__ == "__main__":
    download_datasets()
