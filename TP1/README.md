# TP1 SIA - Métodos de Búsqueda

### Requisitos
 * Python 3.10.*
 * Si se cuenta con `pipenv`, es posible instalar todo usando un Pipfile con el siguiente comando:
 ```py
 pipenv install
 ```
 * Luego, ejecutar el comando para tener la versión correcta de python y todas sus dependencias:
 ```py
 pipenv shell
 ```

### Creación del Estado inicial
 El proyecto puede ejecutar un mapa por defecto si no se le envían datos de configuración.
 También acepta un archivo `.json` con las siguientes configuraciones:
 * Para un mapa random:
 ```json
 {
     "random": {
        "isRandom": bool,
        "seed": int,
        "level": level
     },
     "times": int
 }
 ```
 * Para un mapa custom:
 ```json
 {
     "custom": {
        "isCustom": bool,
        "player": {x: int, y: int},
        "boxes": [{x: int, y: int}],
        "goals": [{x: int, y: int}],
        "blocks": [{x: int, y: int}],
        "n_rows": int,
        "n_cols": int,
     },
     "times": int
 }
 ```
 Notar que solamente el atributo `times` es un atributo común, que
 indica la cantidad de veces que se ejecutará cada algoritmo,
 mejorando la precisión de algunas métricas (como el tiempo
 de ejecución).
 De haber un archivo con el siguiente layout:
 ```json
 {
     "random": {
        "isRandom": bool,
        "seed": int,
        "level": level
     },
     "custom": {
        "isCustom": bool,
        "player": {x: int, y: int},
        "boxes": [{x: int, y: int}],
        "goals": [{x: int, y: int}],
        "blocks": [{x: int, y: int}],
        "n_rows": int,
        "n_cols": int,
     },
     "times": int
 }
 ```
 Se priorizará primero `random`, de fallar por tener parámetros inválidos,
 se intentará con `custom`.

 Debido a la cantidad de algoritmos que se prueban y la cantidad de
 heurísticas utilizadas, se recomienda no utilizar mapas muy grandes
 para las pruebas, dejando niveles como el random en `1`, o mapas que
 no sean mucho mayores a un `10x10`

 Una vez está todo listo, se puede correr el programa con el comando:
 ```shell
 python main.py [config.json]
 ```
