import random as rnd
import math
from enum import Enum, auto
from typing import Decimal
from .Player import Player
from .EVE import EVE
from .PlayerAttributes import PlayerAttributes
from .InvalidGenProbabilityValue import InvalidGenProbabilityValue

min_height = Decimal('1.3')
max_height = Decimal('2.0')


class MutationMethod(Enum):
    SINGLE = "single"
    LIMITED_MULTI = "limited_multi"
    UNIFORM_MULTI = "uniform_multi"
    COMPLETE = "complete"


class GenMutation(Enum):
    HEIGHT = auto()
    STRENGHT = auto()
    DEXTERITY = auto()
    INTELLIGENCE = auto()
    ENDURANCE = auto()
    PHYSIQUE = auto()


class Mutation:
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
                    normalized_p = self.__normalize_attr(player_list)
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
