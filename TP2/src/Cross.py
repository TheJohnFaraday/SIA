import math
import random as rnd
from enum import Enum
from dataclasses import dataclass
from decimal import Decimal

from .Player import Player
from .PlayerAttributes import PlayerAttributes
from .InvalidCrossoverPoint import InvalidCrossoverPoint
from .EVE import EVE


class CrossoverMethod(Enum):
    ONE_POINT = "one_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ANNULAR = "annular"


@dataclass(frozen=True)
class Configuration:
    method: CrossoverMethod
    pc: Decimal


class Cross:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration

        # Number of PlayerAttributes plus Height. Must be calculated
        self.total_points = len(Player(Decimal(0), Player.p_class.WARRIOR, PlayerAttributes(100, 1, 1, 1, 1),
                                       Decimal(0)).attributes_as_list())

    def perform(self, father: Player, mother: Player):
        # Crossover happens with a probability of Pc
        if (self.configuration.pc < 1 and
                self.configuration.pc < Decimal(rnd.uniform(0.0, float(self.configuration.pc)))):
            return [father, mother]

        crossover_locus = rnd.randint(0, self.total_points)

        match self.configuration.method:
            case CrossoverMethod.ONE_POINT:
                return self.single_point(father, mother, crossover_locus)
            case CrossoverMethod.TWO_POINT:
                crossover_locus_2 = rnd.randint(0, self.total_points)
                return self.double_point(
                    father, mother, min(crossover_locus, crossover_locus_2), max(crossover_locus, crossover_locus_2)
                )
            case CrossoverMethod.UNIFORM:
                return self.uniform(
                    father, mother, crossover_locus
                )
            case CrossoverMethod.ANNULAR:
                # Annular requires crossover_locus = [0; total_points - 1]
                if crossover_locus == self.total_points:
                    crossover_locus -= 1

                segment_len = rnd.randint(0, math.ceil(self.total_points / 2))
                return self.annular(
                    father, mother, crossover_locus, segment_len
                )

    def single_point(
        self, player1: Player, player2: Player, p: int
    ) -> [Player]:
        attributes_player1 = player1.attributes_as_list()
        attributes_player2 = player2.attributes_as_list()

        if p >= len(attributes_player1):
            raise InvalidCrossoverPoint
        child1 = attributes_player1[:p] + attributes_player2[p:]
        child2 = attributes_player1[p:] + attributes_player2[:p]
        normalized_child1 = self.__normalize_attr(child1)
        normalized_child2 = self.__normalize_attr(child2)

        p_class = player1.p_class
        p1_attr = PlayerAttributes(
            strength=normalized_child1[1],
            dexterity=normalized_child1[2],
            intelligence=normalized_child1[3],
            endurance=normalized_child1[4],
            physique=normalized_child1[5],
        )
        p2_attr = PlayerAttributes(
            strength=normalized_child2[1],
            dexterity=normalized_child2[2],
            intelligence=normalized_child2[3],
            endurance=normalized_child2[4],
            physique=normalized_child2[5],
        )
        return [
            Player(
                height=normalized_child1[0],
                p_class=p_class,
                p_attr=p1_attr,
                fitness=EVE(normalized_child1[0], p_class, p1_attr).performance,
            ),
            Player(
                height=normalized_child2[0],
                p_class=p_class,
                p_attr=p2_attr,
                fitness=EVE(normalized_child2[0], p_class, p2_attr).performance,
            ),
        ]

    def double_point(
        self, player1: Player, player2: Player, locus_1: int, locus_2: int
    ) -> [Player]:
        attributes_player1 = player1.attributes_as_list()
        attributes_player2 = player2.attributes_as_list()

        if locus_1 >= len(attributes_player1) or locus_2 >= len(attributes_player2):
            raise InvalidCrossoverPoint

        if locus_1 > locus_2:
            locus_1, locus_2 = locus_2, locus_1

        child1 = attributes_player1[:locus_1] + attributes_player2[locus_1:locus_2] + attributes_player1[locus_2:]
        child2 = attributes_player2[:locus_1] + attributes_player1[locus_1:locus_2] + attributes_player2[locus_2:]
        normalized_child1 = self.__normalize_attr(child1)
        normalized_child2 = self.__normalize_attr(child2)

        p_class = player1.p_class
        p1_attr = PlayerAttributes(
            strength=normalized_child1[1],
            dexterity=normalized_child1[2],
            intelligence=normalized_child1[3],
            endurance=normalized_child1[4],
            physique=normalized_child1[5],
        )
        p2_attr = PlayerAttributes(
            strength=normalized_child2[1],
            dexterity=normalized_child2[2],
            intelligence=normalized_child2[3],
            endurance=normalized_child2[4],
            physique=normalized_child2[5],
        )
        return [
            Player(
                height=normalized_child1[0],
                p_class=p_class,
                p_attr=p1_attr,
                fitness=EVE(normalized_child1[0], p_class, p1_attr).performance,
            ),
            Player(
                height=normalized_child2[0],
                p_class=p_class,
                p_attr=p2_attr,
                fitness=EVE(normalized_child2[0], p_class, p2_attr).performance,
            ),
        ]

    def annular(
        self, player1: Player, player2: Player, locus: int, segment_length: int
    ) -> [Player]:
        attributes_player1 = player1.attributes_as_list()
        attributes_player2 = player2.attributes_as_list()
        if locus >= len(attributes_player1) or segment_length > int(len(attributes_player1) / 2):
            raise InvalidCrossoverPoint

        if (locus + segment_length) >= len(attributes_player1):
            segment_length = len(attributes_player1) - 1

        child1 = attributes_player1[:locus] + attributes_player2[locus: locus + segment_length] + attributes_player1[locus + segment_length:]
        child2 = attributes_player2[:locus] + attributes_player1[locus: locus + segment_length] + attributes_player2[locus + segment_length:]
        normalized_child1 = self.__normalize_attr(child1)
        normalized_child2 = self.__normalize_attr(child2)

        p_class = player1.p_class
        p1_attr = PlayerAttributes(
            strength=normalized_child1[1],
            dexterity=normalized_child1[2],
            intelligence=normalized_child1[3],
            endurance=normalized_child1[4],
            physique=normalized_child1[5],
        )
        p2_attr = PlayerAttributes(
            strength=normalized_child2[1],
            dexterity=normalized_child2[2],
            intelligence=normalized_child2[3],
            endurance=normalized_child2[4],
            physique=normalized_child2[5],
        )
        return [
            Player(
                height=normalized_child1[0],
                p_class=p_class,
                p_attr=p1_attr,
                fitness=EVE(normalized_child1[0], p_class, p1_attr).performance,
            ),
            Player(
                height=normalized_child2[0],
                p_class=p_class,
                p_attr=p2_attr,
                fitness=EVE(normalized_child2[0], p_class, p2_attr).performance,
            ),
        ]

    def uniform(
        self, player1: Player, player2: Player, pm: float
    ) -> [Player]:
        attributes_player1 = player1.attributes_as_list()
        attributes_player2 = player2.attributes_as_list()
        if pm > 1:
            raise InvalidCrossoverPoint

        child1 = []
        child2 = []
        for i in range(len(attributes_player1)):
            if rnd.random() > pm:
                child1[i] = attributes_player1[i]
                child2[i] = attributes_player2[i]
            else:
                child1[i] = attributes_player2[i]
                child2[i] = attributes_player1[i]
        normalized_child1 = self.__normalize_attr(child1)
        normalized_child2 = self.__normalize_attr(child2)

        p_class = player1.p_class
        p1_attr = PlayerAttributes(
            strength=normalized_child1[1],
            dexterity=normalized_child1[2],
            intelligence=normalized_child1[3],
            endurance=normalized_child1[4],
            physique=normalized_child1[5],
        )
        p2_attr = PlayerAttributes(
            strength=normalized_child2[1],
            dexterity=normalized_child2[2],
            intelligence=normalized_child2[3],
            endurance=normalized_child2[4],
            physique=normalized_child2[5],
        )
        return [
            Player(
                height=normalized_child1[0],
                p_class=p_class,
                p_attr=p1_attr,
                fitness=EVE(normalized_child1[0], p_class, p1_attr).performance,
            ),
            Player(
                height=normalized_child2[0],
                p_class=p_class,
                p_attr=p2_attr,
                fitness=EVE(normalized_child2[0], p_class, p2_attr).performance,
            ),
        ]

    @staticmethod
    def __normalize_attr(player_attr: list) -> []:
        player_attr_isolated = player_attr[1:]
        current_total = sum(player_attr_isolated)
        factor = PlayerAttributes.TOTAL_POINTS_MAX / current_total
        normalized_attrs = [int(factor * x) for x in player_attr_isolated]

        adjusted_total = sum(normalized_attrs)
        difference = PlayerAttributes.TOTAL_POINTS_MAX - adjusted_total
        for i in range(abs(difference)):
            if difference < 0:
                normalized_attrs[i % len(normalized_attrs)] -= 1
            if difference > 0:
                normalized_attrs[i % len(normalized_attrs)] += 1

        return [player_attr[0]] + normalized_attrs
