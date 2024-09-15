import random as rnd
import math
from enum import Enum
from typing import Decimal
from dataclasses import dataclass
from .Player import Player
from .EVE import EVE
from .PlayerAttributes import PlayerAttributes
from .InvalidGenProbabilityValue import InvalidGenProbabilityValue

min_height = Decimal('1.3')
max_height = Decimal('2.0')


class MutationMethod(Enum):
    SINGLE = "single"
    MULTI = "multi"


class GenMutation(Enum):
    HEIGHT = "height"
    STRENGHT = "strength"
    DEXTERITY = "dexterity"
    INTELLIGENCE = "intelligence"
    ENDURANCE = "endurance"
    PHYSIQUE = "physique"


@dataclass
class Configuration:
    mutation: MutationMethod
    pm: Decimal
    is_uniform: bool
    generational_increment: Decimal = Decimal(0)  # For not-uniform mutation method
    max_genes: int = 1  # For limited_multi mutation method


class Mutation:
    def __init__(self, configuration: Configuration, max_points: int):
        self.config = configuration
        self.max_points = max_points
        self.pm = configuration.pm

    def mutate(self, generation: int, population: list[Player]):
        return list(map(lambda pl: self.perform(pl),
                        population))

    def update(self):
        if self.config.is_uniform:
            self.pm += self.config.generational_increment

    def perform(self, player: Player):
        match self.config.mutation_method:
            case MutationMethod.SINGLE:
                return Mutation.gen(player,
                                    self.config.gen_mut,
                                    self.pm,
                                    (self.config.lower_bound,
                                     self.config.higher_bound))
            case MutationMethod.MULTI:
                return Mutation.multigen(player,
                                         self.config.max_genes,
                                         self.pm,
                                         (self.config.lower_bound,
                                          self.config.higher_bound))

    @staticmethod
    def gen(self, player: Player, gen_mut: GenMutation,
            p: float, interval: (float, float)) -> Player:
        if p > 1 or interval[0] > interval[1]:
            raise InvalidGenProbabilityValue
        if rnd.random() > p:
            mutation = rnd.uniform(interval[0], interval[1])
            match gen_mut:
                case GenMutation.HEIGHT:
                    new_height = Decimal(player.height*(Decimal('1')+mutation))
                    new_height = math.max(min_height,
                                          math.min(max_height, new_height))
                    return Player(height=new_height,
                                  p_class=player.p_class,
                                  p_attr=player.p_attr,
                                  fitness=(EVE(height=new_height,
                                               p_class=player.p_class,
                                               attributes=player.p_attr)
                                           .performance))
                case _:
                    player_list = self.__get_player_attr_list(player)
                    match gen_mut:
                        case GenMutation.STRENGHT:
                            player_list[1] = int(player
                                                 .p_attr.strength*(1+mutation))
                        case GenMutation.DEXTERITY:
                            player_list[2] = int(player
                                                 .p_attr
                                                 .dexterity*(1+mutation))
                        case GenMutation.INTELLIGENCE:
                            player_list[3] = int(player
                                                 .p_attr
                                                 .intelligence*(1+mutation))
                        case GenMutation.ENDURANCE:
                            player_list[4] = int(player
                                                 .p_attr
                                                 .endurance*(1+mutation))
                        case GenMutation.PHYSIQUE:
                            player_list[5] = int(player
                                                 .p_attr.physique*(1+mutation))
                    normalized_p = self.__normalize_attr(player_list,
                                                         self.max_points)
                    normalized_p_attr = PlayerAttributes(
                        strength=normalized_p[1],
                        dexterity=normalized_p[2],
                        intelligence=normalized_p[3],
                        endurance=normalized_p[4],
                        physique=normalized_p[5])
                    return Player(height=player.height,
                                  p_attr=normalized_p_attr,
                                  p_class=player.p_class,
                                  fitness=(EVE(height=player.height,
                                               p_class=player.p_class,
                                               attributes=normalized_p_attr)
                                           .performance))
            return player

    @staticmethod
    def multigen(self, player: Player, max_genes: int,
                 p: float, interval: (float, float)) -> Player:
        gen_list = list(GenMutation)
        if p > 1 or interval[0] > interval[1] or max_genes > len(gen_list):
            raise InvalidGenProbabilityValue
        player_list = self.__get_player_attr_list(player)
        gens_mutated = set()
        for i in range(max_genes-1):
            if rnd.random() > p:
                mutation = rnd.uniform(interval[0], interval[1])
                mut = gen_list[rnd.randint(0, len(gen_list) - 1)]
                while mut not in gens_mutated:
                    mut = gen_list[rnd.randint(0, len(gen_list) - 1)]
                match mut:
                    case GenMutation.HEIGHT:
                        new_height = Decimal(player
                                             .height*(Decimal('1')+mutation))
                        player_list[0] = math.max(min_height,
                                                  math.min(max_height,
                                                           new_height))
                    case GenMutation.STRENGHT:
                        player_list[1] = int(player
                                             .p_attr.strength*(1+mutation))
                    case GenMutation.DEXTERITY:
                        player_list[2] = int(player
                                             .p_attr
                                             .dexterity*(1+mutation))
                    case GenMutation.INTELLIGENCE:
                        player_list[3] = int(player
                                             .p_attr
                                             .intelligence*(1+mutation))
                    case GenMutation.ENDURANCE:
                        player_list[4] = int(player
                                             .p_attr
                                             .endurance*(1+mutation))
                    case GenMutation.PHYSIQUE:
                        player_list[5] = int(player
                                             .p_attr.physique*(1+mutation))
                gens_mutated.add(mut)

        normalized_p = self.__normalize_attr(player_list, self.max_points)
        normalized_p_attr = PlayerAttributes(
                                strength=normalized_p[1],
                                dexterity=normalized_p[2],
                                intelligence=normalized_p[3],
                                endurance=normalized_p[4],
                                physique=normalized_p[5])
        return Player(height=normalized_p[0],
                      p_attr=normalized_p_attr,
                      p_class=player.p_class,
                      fitness=(EVE(height=player.height,
                                   p_class=player.p_class,
                                   attributes=normalized_p_attr)
                               .performance))

    '''
    get_player_attr_list: Returns a list containing the attributes of a
                         player in the following order:
        [Height, Strength, Dexterity, Intelligence, Endurance, Physique]
    '''
    def __get_player_attr_list(player: Player) -> []:
        return [player.height, player.p_attr.strength, player.p_attr.dexterity,
                player.p_attr.intelligence, player.p_attr.endurance,
                player.p_attr.physique]

    def __normalize_attr(player_attr: [], max_points: int) -> []:
        player_attr_isolated = player_attr[1:]
        current_total = sum(player_attr_isolated)
        factor = max_points / current_total
        normalized_attrs = [int(factor * x) for x in player_attr_isolated]

        adjusted_total = sum(normalized_attrs)
        difference = max_points - adjusted_total
        for i in range(abs(difference)):
            if difference < 0:
                normalized_attrs[i % len(normalized_attrs)] -= 1
            if difference > 0:
                normalized_attrs[i % len(normalized_attrs)] += 1

        return [player_attr[0]] + normalized_attrs
