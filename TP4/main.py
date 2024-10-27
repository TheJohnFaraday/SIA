import csv
import logging
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.grapher import (
    Coordinates,
    Rotation,
    scatter_pca,
    boxplot_pca,
    pc1_pca,
    three_dimension_pca,
)

from sklearn.decomposition import PCA

from src.configuration import read_configuration, ConfigurationToRead, OjaConfig
from src.OjaPerceptron import OjaSimplePerceptron


@dataclass(frozen=True)
class EuropeInput:
    country: list[str]
    area: list[int]
    gdp: list[int]
    inflation: list[float]
    life_expectancy: list[float]
    military: list[float]
    population_growth: list[float]
    unemployment: list[float]


@dataclass(frozen=True)
class Europe:
    country: list[str]
    data: np.array


def load_csv(path: str):
    europe: EuropeInput | None = None
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not europe:
                europe = EuropeInput(
                    country=[row["Country"]],
                    area=[int(row["Area"])],
                    gdp=[int(row["GDP"])],
                    inflation=[float(row["Inflation"])],
                    life_expectancy=[float(row["Life.expect"])],
                    military=[float(row["Military"])],
                    population_growth=[float(row["Pop.growth"])],
                    unemployment=[float(row["Unemployment"])],
                )
            else:
                europe.country.append(row["Country"])
                europe.area.append(int(row["Area"]))
                europe.gdp.append(int(row["GDP"]))
                europe.inflation.append(float(row["Inflation"]))
                europe.life_expectancy.append(float(row["Life.expect"]))
                europe.military.append(float(row["Military"]))
                europe.population_growth.append(float(row["Pop.growth"]))
                europe.unemployment.append(float(row["Unemployment"]))

    return europe


def europe_input_to_dataframe(europe_input: EuropeInput):
    # return pd.DataFrame(
    #     np.array(
    #         [
    #             europe_input.area,
    #             europe_input.gdp,
    #             europe_input.inflation,
    #             europe_input.life_expectancy,
    #             europe_input.military,
    #             europe_input.population_growth,
    #             europe_input.unemployment,
    #         ]
    #     ),
    #     columns=europe_input.country,
    # )

    matrix = np.array(
        [
            europe_input.area,
            europe_input.gdp,
            europe_input.inflation,
            europe_input.life_expectancy,
            europe_input.military,
            europe_input.population_growth,
            europe_input.unemployment,
        ]
    )

    return pd.DataFrame(
        matrix.transpose(),
        columns=[
            "Area",
            "GDP",
            "Inflation",
            "Life Expectancy",
            "Military",
            "Population Growth",
            "Unemployment",
        ],
    )


def standardize_dataframe(df: pd.DataFrame):
    return df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)


def pca_two(europe_input: EuropeInput, europe: pd.DataFrame, europe_std: pd.DataFrame):
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(europe_std)
    pca_components = pca.components_

    df_pca_dict = {}
    for idx, component in enumerate(pca_components):
        df_pca_dict[f"PC{idx+1}"] = component

    df = pd.DataFrame.from_dict(
        df_pca_dict,
    )
    df["Variable"] = europe_std.columns
    df = df.set_index("Variable")

    scatter_pca(
        "Análisis PCA",
        "pca",
        "sklearn",
        pca_data,
        df,
        europe_input.country,
        scale_factor=3,
        text_scale_factor=3.2,
        arrow_text_offset=Coordinates(x=0, y=0.1),
    )

    boxplot_pca(
        "Variables sin normalizar",
        "pca",
        "not_normalized",
        europe,
        label_x="Variable",
        label_y="Valor",
    )
    boxplot_pca(
        "Variables normalizadas",
        "pca",
        "normalized",
        europe_std,
        label_x="Variable",
        label_y="Valor",
    )


def pca_three(europe_input: EuropeInput, europe_std: pd.DataFrame):
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(europe_std)
    pca_components = pca.components_

    df_pca_dict = {}
    for idx, component in enumerate(pca_components):
        df_pca_dict[f"PC{idx+1}"] = component

    df = pd.DataFrame.from_dict(
        df_pca_dict,
    )
    df["Variable"] = europe_std.columns
    df = df.set_index("Variable")

    three_dimension_pca(
        "Análisis PCA",
        "pca",
        "3d",
        pca_data,
        df,
        europe_input.country,
        scale_factor=3,
        text_scale_factor=3.2,
        arrow_text_offset=Coordinates(x=0, y=0.1, z=0.2),
        rotation=Rotation(elevation=10, azimuth=-25),
    )


def pca_one(europe_input: EuropeInput, europe_std: pd.DataFrame):
    pca = PCA(n_components=1)
    pca_data = pca.fit_transform(europe_std)

    df_pca_dict = {"PC1": pca_data[:, 0]}

    df_country = pd.DataFrame.from_dict(
        df_pca_dict,
    )
    df_country["Country"] = europe_input.country
    df_country = df_country.set_index("Country")

    pc1_pca(
        "Análisis PC1",
        "pca",
        "pc1_countries",
        df_country.index.values,
        df_country["PC1"].values,
        "País",
        "PC1",
    )

    pca_components = pca.components_

    df_pca_dict = {}
    for idx, component in enumerate(pca_components):
        df_pca_dict[f"PC{idx + 1}"] = component

    df_variables = pd.DataFrame.from_dict(
        df_pca_dict,
    )
    df_variables["Variable"] = europe_std.columns
    df_variables = df_variables.set_index("Variable")

    pc1_pca(
        "Análisis PC1",
        "pca",
        "pc1_vars",
        df_variables.index.values,
        df_variables["PC1"].values,
        "Variables",
        "PC1",
    )


def ej_pca():
    europe_input = load_csv("./dataset/europe.csv")
    if not europe_input:
        logging.error("No Input parsed. Is the filepath OK?")
        sys.exit(1)

    europe = europe_input_to_dataframe(europe_input)
    europe_std = standardize_dataframe(europe)

    pca_one(europe_input, europe_std)
    pca_two(europe_input, europe, europe_std)
    pca_three(europe_input, europe_std)


def ej_oja():
    configuration: OjaConfig = read_configuration(ConfigurationToRead.OJA)

    europe_input = load_csv("./dataset/europe.csv")
    if not europe_input:
        logging.error("No Input parsed. Is the filepath OK?")
        sys.exit(1)

    europe = europe_input_to_dataframe(europe_input)
    europe_std = standardize_dataframe(europe)

    oja_perceptron = OjaSimplePerceptron(
        configuration=configuration,
        input_data=europe_std
    )

    oja_perceptron.train()
    print(f"Oja weights: {oja_perceptron.weights}")


if __name__ == "__main__":
    # ej_pca()
    ej_oja()
