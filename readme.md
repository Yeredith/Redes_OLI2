# Instrucciones para ejecutar los códigos

Este documento proporciona instrucciones para ejecutar los códigos mencionados.

## Preparación

### Requisitos previos

Asegúrate de tener Python instalado en tu sistema. También es recomendable utilizar un entorno virtual para el proyecto.

1. **Instalación de requisitos**: Desde la terminal, navega al directorio del proyecto y ejecuta el siguiente comando para instalar las dependencias necesarias:

    ```bash
    pip install -r requirements.txt
    ```

    Asegúrate de que el archivo `requirements.txt` contenga todas las bibliotecas necesarias y sus versiones correspondientes.

## Código

### Entrenamiento

1. **Preparación del modelo**: Antes de ejecutar el código de entrenamiento, asegúrate de tener el modelo (`DRN`) y los datos necesarios preparados.

2. **Ejecución del entrenamiento**: Una vez que todo esté configurado correctamente, puedes ejecutar el script de entrenamiento con el siguiente comando:

    ```bash
    python entrenamiento_drn.py
    ```

    Asegúrate de que el archivo `entrenamiento_drn.py` esté presente en tu directorio de trabajo y contenga el código necesario para el entrenamiento del modelo.

### Test

1. **Preparación de los datos de prueba**: Asegúrate de tener los datos de prueba y el modelo entrenado listos antes de ejecutar el test.

2. **Ejecución del test**: Utiliza el siguiente comando para ejecutar el script de test:

    ```bash
    python test_drn.py
    ```

    Asegúrate de que el archivo `test_drn.py` esté presente en tu directorio de trabajo y contenga el código necesario para el test del modelo.

---

Antes de ejecutar revisa que tengas todos los archivos necesarios y los requisitos instalados.

**Versión de Python utilizada:** Python 3.9.16

