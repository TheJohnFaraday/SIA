from .Player import Player
from .PlayerAttributes import PlayerAttributes
from .InvalidCrossoverPoint import InvalidCrossoverPoint
from .EVE import EVE
import random as rnd


class Cross:
    def __init__(self, configuration):
        self.configuration = configuration

    def perform(self, parent, mother) -> list[Player]:
        pass

    @staticmethod
    def single_point(self, player1: Player, player2: Player,
                     p: int, total_points: int) -> (Player, Player):
        player1_list = self.__get_player_attr_list(player1)
        player2_list = self.__get_player_attr_list(player2)
        if p >= len(player1_list):
            raise InvalidCrossoverPoint
        child1 = player1_list[:p]+player2_list[p:]
        child2 = player1_list[p:]+player2_list[:p]
        normalized_child1 = self.__normalize_attr(child1)
        normalized_child2 = self.__normalize_attr(child2)

        p_class = player1.p_class
        p1_attr = PlayerAttributes(
                        strength=normalized_child1[1],
                        dexterity=normalized_child1[2],
                        intelligence=normalized_child1[3],
                        endurance=normalized_child1[4],
                        physique=normalized_child1[5])
        p2_attr = PlayerAttributes(
                        strength=normalized_child2[1],
                        dexterity=normalized_child2[2],
                        intelligence=normalized_child2[3],
                        endurance=normalized_child2[4],
                        physique=normalized_child2[5])
        return (Player(height=normalized_child1[0],
                       p_class=p_class,
                       p_attr=p1_attr,
                       fitness=EVE(normalized_child1[0],
                                   p_class,
                                   p1_attr).performance),
                Player(height=normalized_child2[0],
                       p_class=p_class,
                       p_attr=p2_attr,
                       fitness=EVE(normalized_child2[0],
                                   p_class,
                                   p2_attr).performance))

    @staticmethod
    def double_point(self, player1: Player, player2: Player,
                     p1: int, p2: int, total_points: int) -> (Player, Player):
        player1_list = self.__get_player_attr_list(player1)
        player2_list = self.__get_player_attr_list(player2)
        if p1 >= len(player1_list) or p2 >= len(player1_list):
            raise InvalidCrossoverPoint
        if p1 > p2:
            aux = p1
            p1 = p2
            p2 = aux
        child1 = player1_list[:p1]+player2_list[p1:p2]+player1_list[p2:]
        child2 = player2_list[:p1]+player1_list[p1:p2]+player2_list[p2:]
        normalized_child1 = self.__normalize_attr(child1)
        normalized_child2 = self.__normalize_attr(child2)

        p_class = player1.p_class
        p1_attr = PlayerAttributes(
                        strength=normalized_child1[1],
                        dexterity=normalized_child1[2],
                        intelligence=normalized_child1[3],
                        endurance=normalized_child1[4],
                        physique=normalized_child1[5])
        p2_attr = PlayerAttributes(
                        strength=normalized_child2[1],
                        dexterity=normalized_child2[2],
                        intelligence=normalized_child2[3],
                        endurance=normalized_child2[4],
                        physique=normalized_child2[5])
        return (Player(height=normalized_child1[0],
                       p_class=p_class,
                       p_attr=p1_attr,
                       fitness=EVE(normalized_child1[0],
                                   p_class,
                                   p1_attr).performance),
                Player(height=normalized_child2[0],
                       p_class=p_class,
                       p_attr=p2_attr,
                       fitness=EVE(normalized_child2[0],
                                   p_class,
                                   p2_attr).performance))

    @staticmethod
    def anular(self, player1: Player, player2: Player,
               p: int, len: int, total_points: int) -> (Player, Player):
        player1_list = self.__get_player_attr_list(player1)
        player2_list = self.__get_player_attr_list(player2)
        if p >= len(player1_list) or len > int(len(player1_list)/2):
            raise InvalidCrossoverPoint
        if (p+len) >= len(player1_list):
            len = len(player1_list) - 1
        child1 = player1_list[:p]+player2_list[p:p+len]+player1_list[p+len:]
        child2 = player2_list[:p]+player1_list[p:p+len]+player2_list[p+len:]
        normalized_child1 = self.__normalize_attr(child1)
        normalized_child2 = self.__normalize_attr(child2)

        p_class = player1.p_class
        p1_attr = PlayerAttributes(
                        strength=normalized_child1[1],
                        dexterity=normalized_child1[2],
                        intelligence=normalized_child1[3],
                        endurance=normalized_child1[4],
                        physique=normalized_child1[5])
        p2_attr = PlayerAttributes(
                        strength=normalized_child2[1],
                        dexterity=normalized_child2[2],
                        intelligence=normalized_child2[3],
                        endurance=normalized_child2[4],
                        physique=normalized_child2[5])
        return (Player(height=normalized_child1[0],
                       p_class=p_class,
                       p_attr=p1_attr,
                       fitness=EVE(normalized_child1[0],
                                   p_class,
                                   p1_attr).performance),
                Player(height=normalized_child2[0],
                       p_class=p_class,
                       p_attr=p2_attr,
                       fitness=EVE(normalized_child2[0],
                                   p_class,
                                   p2_attr).performance))

    @staticmethod
    def uniform(self, player1: Player, player2: Player,
                p: float, total_points: int) -> (Player, Player):
        player1_list = self.__get_player_attr_list(player1)
        player2_list = self.__get_player_attr_list(player2)
        if p > 1:
            raise InvalidCrossoverPoint
        child1 = []
        child2 = []
        for i in range(len(player1_list)):
            if rnd.random() > p:
                child1[i] = player1_list[i]
                child2[i] = player2_list[i]
            else:
                child1[i] = player2_list[i]
                child2[i] = player1_list[i]
        normalized_child1 = self.__normalize_attr(child1)
        normalized_child2 = self.__normalize_attr(child2)

        p_class = player1.p_class
        p1_attr = PlayerAttributes(
                        strength=normalized_child1[1],
                        dexterity=normalized_child1[2],
                        intelligence=normalized_child1[3],
                        endurance=normalized_child1[4],
                        physique=normalized_child1[5])
        p2_attr = PlayerAttributes(
                        strength=normalized_child2[1],
                        dexterity=normalized_child2[2],
                        intelligence=normalized_child2[3],
                        endurance=normalized_child2[4],
                        physique=normalized_child2[5])
        return (Player(height=normalized_child1[0],
                       p_class=p_class,
                       p_attr=p1_attr,
                       fitness=EVE(normalized_child1[0],
                                   p_class,
                                   p1_attr).performance),
                Player(height=normalized_child2[0],
                       p_class=p_class,
                       p_attr=p2_attr,
                       fitness=EVE(normalized_child2[0],
                                   p_class,
                                   p2_attr).performance))

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
