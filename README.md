# Clasificación de poblaciones celulares mediante marcadores obtenidos por Citometría de Flujo aplicando técnicas de Inteligencia Artificial
Este repositorio se presenta de forma complementaria a mi Trabajo Fin de Grado en Ingeniería Informática.
En él se recogen el código y los recursos utilizados para el desarrollo de los modelos y la aplicación web descrita en la memoria.
## Estructura del repositorio
- [`APP/`](./APP): Código de la aplicación web y Dockerfile para el despliegue.
    - [`APP/static/`](./APP/static/): Archivos estáticos (CSS, JS, imágenes).
    - [`APP/templates/`](./APP/templates/): Plantilla HTML.
- [`Notebooks/`](./Notebooks/): Notebooks utilizados durante el desarrollo y evaluación.
    - [`Notebooks/SOM/`](./Notebooks/SOM/): Clase som y neurona y código desarrollado para su implementación y evaluación.
    - [`Notebooks/MLP/`](./Notebooks/MLP/): Código para la optimización de los hiperparámetros y para el desarrollo del modelo jerárquico completo.
- [`diagramas_app.asta`](./diagramas_app.asta): Diagramas de análisis y diseño.

## Instalación y despliegue
1. Asegúrase de tener **Docker** instalado. Consulte: https://docs.docker.com/engine/install/
2. Acceda al directorio `APP/`, donde se encuentra el fichero Dockerfile.
3. Construya la imagen Docker:
```bash
docker build -t <nombre_imagen> .
```
4. Una vez construida la imagen, proceda a su ejecución:
```bash
docker run -p 5000:5000 <nombre_imagen>
```
5. Acceda al servicio web al introducir la siguiente ruta en nuestro navegador: http://localhost:5000
