# TP3 SIA - Perceptrón Simple y Multicapa - Grupo 6

## Integrantes

| Nombre | Legajo | Correo electrónico |
| :--- | ---: | :--- |
| Francisco Sendot | 62351 | [fsendot@itba.edu.ar](mailto:fsendot@itba.edu.ar) |
| Lucía Digon | 59030 | [ldigon@itba.edu.ar](mailto:ldigon@itba.edu.ar) |
| Juan Ignacio Fernández Dinardo | 62466 | [jfernandezdinardo@itba.edu.ar](mailto:jfernandezdinardo@itba.edu.ar) |
| Martín E. Zahnd | 60401 | [mzahnd@itba.edu.ar](mailto:mzahnd@itba.edu.ar) |

## Requisitos
 * Python 3.12
 * [Pipenv](https://pipenv.pypa.io/en/latest/)
 * Instalar las dependencias descriptas en el Pipfile con el siguiente comando:
 ```py
 pipenv install
 ```
 * Luego, ejecutar el comando para tener la versión correcta de python y todas sus dependencias:
 ```py
 pipenv shell
 ```

## Configuración del proyecto

El proyecto cuenta con un archivo TOML que se utiliza para la configuración del mismo, llamado `config.toml`.

Los parámetros son tenidos en cuenta solamente en los casos en que se selecciona el método que los utiliza.
Por ejemplo, `alpha` en `[multi_layer.momentum]` solamente se utilizará si `optimizer` en `[multi_layer]` es `"momentum"`.

Los parámetros son descriptivos e indican en qué rango de valores se deben introducir utilizando notación estándar:
- Los símbolos `[` y `]` incluyen el límite izquierdo y derecho (por ejemplo: "[0; 1]" significa que el valor debe estar entre 0 y 1, ambos inclusive).
- Los símbolos `(` y `)` indican que NO se incluye el límite izquierdo y derecho (por ejemplo: "[0; 1)" significa que el valor debe estar entre 0, inclusive, y un valor estrictamente menor a 1).

La opción especial `plot` es un bool ("true" o "false") que permite o no la impresión de gráficos.

Por otra parte, la opción `seed` es opcional y de estar configurada permite una ejecución determinística del algoritmo.

Una vez configurada la ejecución, el script se corre mediante:
```shell
python main.py
```

