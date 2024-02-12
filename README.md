# Taller1: MLOps

## Por Brayan Carvajal, Juan Pablo Nieto y Nicolas Rojas

## Objetivos del taller
Descargue los datos asociados en el aula virtual.

- Cree un archivo en Python consuma estos datos y realice las dos etapas, procesamiento de datos y creación de modelo. Considere usar como guía las sub-etapas listadas. 
- Cree un API usando FastAPI permita hacer inferencia al modelo entrenado.
- Cree la imagen del contenedor con el API creada. Exponga el API en puerto 8989.

![Nivel 0](img/lvl0.svg)

Bono.

El proceso de entrenamiento de un modelo busca encontrar el mejor modelo y ajustarlo a los datos, este proceso de experimentación en ocasiones resulta en multiples modelos con muy buenos resultados. Como bono entregue en el API un método adicional que permita seleccionar cual modelo será usado en el proceso de inferencia.

## Instrucciones para despliegue

- Instale docker siguiendo las instrucciones en la [documentación oficial](https://docs.docker.com/get-docker/).
- Clone este repositorio con el siguiente comando:
    ```shell
    git clone https://github.com/GacelyML/JAV_MLOps_Taller1
    ```
- Ubíquese en la carpeta recién creada:
    ```shell
    cd JAV_MLOps_Taller1
    ```
- Cree el contenedor con el siguiente comando:
    ```shell
    docker build -t penguinmodelapi -f docker/Dockerfile .
    ```
- Ejecute el contenedor con el siguiente comando:
    ```shell
    docker run --rm -d -p 8989:8989 --name modelapi penguinmodelapi:latest
    ```

La API se levantará en `localhost:8989`, y
puede acceder a la documentación de la misma en `localhost:8989/docs`.

## Uso

Puede probar la API con el siguiente input de ejemplo:

```json
{
    "model": "knn",
    "culmenLen": [39.1, 39.5],
    "culmenDepth": [18.7, 17.4],
    "flipperLen": [181, 186],
    "bodyMass": [3750, 3800],
    "sex": ["MALE", "FEMALE"],
    "delta15N": [8.94, 8.37],
    "delta13C": [-24.69, -25.33]
}
```

Lo cual retornará la siguiente salida:

```json
{
    "specie": [
        "Adelie Penguin (Pygoscelis adeliae)",
        "Adelie Penguin (Pygoscelis adeliae)"
    ]
}
```

Para cambiar el modelo seleccionado, modificar el argumento `model` entre las siguientes opciones: `knn`, `lda`, `lr`. 

