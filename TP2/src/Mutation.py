import random as rnd
from enum import Enum
from decimal import Decimal
from dataclasses import dataclass

from .EVE import EVE
from .InvalidGenProbabilityValue import InvalidGenProbabilityValue
from .Player import Player
from .PlayerAttributes import PlayerAttributes


class MutationMethod(Enum):
    SINGLE = "single"
    MULTI = "multi"


class GenMutation(Enum):
    HEIGHT = "height"
    STRENGTH = "strength"
    DEXTERITY = "dexterity"
    INTELLIGENCE = "intelligence"
    ENDURANCE = "endurance"
    PHYSIQUE = "physique"


@dataclass
class Configuration:
    mutation: MutationMethod
    pm: Decimal
    is_uniform: bool
    generational_increment: Decimal = Decimal("0")  # For not-uniform mutation method
    max_genes: int = 1  # For limited_multi mutation method
    gen_mutation: GenMutation = GenMutation.HEIGHT
    lower_bound: Decimal = Decimal("-0.2")
    higher_bound: Decimal = Decimal("0.2")


class Mutation:
    def __init__(self, configuration: Configuration, total_points: int):
        self.config = configuration
        self.pm = configuration.pm
        max_genes = configuration.max_genes
        if max_genes < 1 or max_genes > len(list(GenMutation)):
            max_genes = 1
        self.max_genes = max_genes

    def mutate(self, population: list[Player]):
        res = map(lambda pl: self.perform(pl), population)
        if not self.config.is_uniform:
            self.update()
        return list(res)

    def update(self):
        if self.config.is_uniform:
            self.pm += self.config.generational_increment

    def perform(self, player: Player):
        match self.config.mutation:
            case MutationMethod.SINGLE:
                return self.gen(
                    player,
                    self.config.gen_mutation,
                    self.pm,
                    (self.config.lower_bound, self.config.higher_bound),
                )
            case MutationMethod.MULTI:
                return self.multigen(
                    player,
                    self.pm,
                    (self.config.lower_bound, self.config.higher_bound),
                )

    def gen(
        self,
        player: Player,
        gen_mut: GenMutation,
        pm: Decimal,
        interval: (Decimal, Decimal),
    ) -> Player:
        if pm > 1 or interval[0] > interval[1]:
            raise InvalidGenProbabilityValue
        if rnd.random() > pm:
            mutation = rnd.uniform(interval[0], interval[1])
            match gen_mut:
                case GenMutation.HEIGHT:
                    new_height = Decimal(player.height * Decimal(1 + mutation))
                    new_height = max(
                        Player.MIN_HEIGHT, min(Player.MAX_HEIGHT, new_height)
                    )
                    return Player(
                        height=new_height,
                        p_class=player.p_class,
                        p_attr=player.p_attr,
                        fitness=(
                            EVE(
                                height=new_height,
                                p_class=player.p_class,
                                attributes=player.p_attr,
                            ).performance
                        ),
                    )
                case _:
                    attributes_player = player.attributes_as_list()
                    match gen_mut:
                        case GenMutation.STRENGTH:
                            attributes_player[1] = int(
                                player.p_attr.strength * (1 + mutation)
                            )
                        case GenMutation.DEXTERITY:
                            attributes_player[2] = int(
                                player.p_attr.dexterity * (1 + mutation)
                            )
                        case GenMutation.INTELLIGENCE:
                            attributes_player[3] = int(
                                player.p_attr.intelligence * (1 + mutation)
                            )
                        case GenMutation.ENDURANCE:
                            attributes_player[4] = int(
                                player.p_attr.endurance * (1 + mutation)
                            )
                        case GenMutation.PHYSIQUE:
                            attributes_player[5] = int(
                                player.p_attr.physique * (1 + mutation)
                            )
                    normalized_p = self.__normalize_attr(
                        attributes_player, self.max_points
                    )
                    normalized_p_attr = PlayerAttributes(
                        strength=normalized_p[1],
                        dexterity=normalized_p[2],
                        intelligence=normalized_p[3],
                        endurance=normalized_p[4],
                        physique=normalized_p[5],
                    )
                    return Player(
                        height=player.height,
                        p_attr=normalized_p_attr,
                        p_class=player.p_class,
                        fitness=(
                            EVE(
                                height=player.height,
                                p_class=player.p_class,
                                attributes=normalized_p_attr,
                            ).performance
                        ),
                    )

        return player

    def multigen(
        self, player: Player, pm: Decimal, interval: (Decimal, Decimal)
    ) -> Player:
        gen_list = list(GenMutation)
        if pm > 1 or interval[0] > interval[1] or self.max_genes > len(gen_list):
            raise InvalidGenProbabilityValue

        attributes_player = player.attributes_as_list()
        gens_mutated = set()
        for i in range(self.max_genes - 1):
            if rnd.random() > pm:
                mutation = rnd.uniform(interval[0], interval[1])
                mut = gen_list[rnd.randint(0, len(gen_list) - 1)]
                while mut not in gens_mutated:
                    mut = gen_list[rnd.randint(0, len(gen_list) - 1)]
                match mut:
                    case GenMutation.HEIGHT:
                        new_height = Decimal(player.height * Decimal(1 + mutation))
                        attributes_player[0] = max(
                            Player.MIN_HEIGHT, min(Player.MAX_HEIGHT, new_height)
                        )
                    case GenMutation.STRENGTH:
                        attributes_player[1] = int(
                            player.p_attr.strength * (1 + mutation)
                        )
                    case GenMutation.DEXTERITY:
                        attributes_player[2] = int(
                            player.p_attr.dexterity * (1 + mutation)
                        )
                    case GenMutation.INTELLIGENCE:
                        attributes_player[3] = int(
                            player.p_attr.intelligence * (1 + mutation)
                        )
                    case GenMutation.ENDURANCE:
                        attributes_player[4] = int(
                            player.p_attr.endurance * (1 + mutation)
                        )
                    case GenMutation.PHYSIQUE:
                        attributes_player[5] = int(
                            player.p_attr.physique * (1 + mutation)
                        )
                gens_mutated.add(mut)

        normalized_p = self.__normalize_attr(attributes_player, self.max_points)
        normalized_p_attr = PlayerAttributes(
            strength=normalized_p[1],
            dexterity=normalized_p[2],
            intelligence=normalized_p[3],
            endurance=normalized_p[4],
            physique=normalized_p[5],
        )
        return Player(
            height=normalized_p[0],
            p_attr=normalized_p_attr,
            p_class=player.p_class,
            fitness=(
                EVE(
                    height=player.height,
                    p_class=player.p_class,
                    attributes=normalized_p_attr,
                ).performance
            ),
        )

    @staticmethod
    def __normalize_attr(player_attr: list, total_points: int) -> []:
        player_attr_isolated = player_attr[1:]
        current_total = sum(player_attr_isolated)
        factor = total_points / current_total
        normalized_attrs = [int(factor * x) for x in player_attr_isolated]

        adjusted_total = sum(normalized_attrs)
        difference = total_points - adjusted_total
        for i in range(abs(difference)):
            if difference < 0:
                normalized_attrs[i % len(normalized_attrs)] -= 1
            if difference > 0:
                normalized_attrs[i % len(normalized_attrs)] += 1

        return [player_attr[0]] + normalized_attrs
