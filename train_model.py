import os
import sys

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.model import train_and_save_model

if __name__ == '__main__':
    # Define rutas relativas al proyecto para ejecutar el entrenamiento desde la raíz.
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, '..', 'DEMRE_1.csv')
    models_dir = os.path.join(base_dir, 'models')
    
    # Inicia el pipeline completo de lectura, entrenamiento y persistencia.
    print(f"Iniciando entrenamiento del modelo desde: {data_path}")
    train_and_save_model(data_path, models_dir)
    print("¡Entrenamiento completado!")
