# JAV_MLOps_Taller1

## Objetivos del taller
Descargue los datos asociados en el aula virtual.


- Cree un archivo en Python consuma estos datos y realice las dos etapas, procesamiento de datos y creación de modelo. Considere usar como guía las sub-etapas listadas. 
- Cree un API usando FastAPI permita hacer inferencia al modelo entrenado.
- Cree la imagen del contenedor con el API creada. Exponga el API en puerto 8989.

![Nivel 0](img/lvl0.svg)

Bono.

El proceso de entrenamiento de un modelo busca encontrar el mejor modelo y ajustarlo a los datos, este proceso de experimentación en ocasiones resulta en multiples modelos con muy buenos resultados. Como bono entregue en el API un método adicional que permita seleccionar cual modelo será usado en el proceso de inferencia.

## Instrucciones para despliegue

- instalar docker descargando este link docker https://get.docker.com/ como sh y ejecutarlo
- clonar repositorio
- ejecutar estos comandos
- docker build -t PenguinModelAPI -f docker/Dockerfile .
- docker run --rm -d -p 8989:8989 --name modelAPI PenguinModelAPI:latest

La API se levantara en localhost:8989

Documentación de la API en localhost:8989/docs