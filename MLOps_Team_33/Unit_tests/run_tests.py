import sys
import pytest

# Añadir el directorio Unit_tests al PYTHONPATH
sys.path.insert(0, '.')

# Configurar las opciones de pytest
pytest_args = [
    '--log-cli-level=INFO',  # Level of logs to show on the console
    '--log-file=pytest_log.txt',  # File to log the output
    '--log-file-level=INFO',  # Level of logs to log to the file
]

# Ejecutar pytest con los argumentos
pytest.main(pytest_args)