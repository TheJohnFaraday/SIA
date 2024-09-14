from .Player import Player
from .PlayerAttributes import PlayerAttributes
from .InvalidCrossoverPoint import InvalidCrossoverPoint


class Cross:
    @staticmethod
    def single_point(self, player1: Player, player2: Player,
                     p: int, total_points: int) -> (Player, Player):
        player1_list = self.__get_player_attr_list(player1)
        player2_list = self.__get_player_attr_list(player2)
        if p >= player1_list:
            raise InvalidCrossoverPoint
        child1 = player1_list[:p] + player2_list[p:]
        child2 = player1_list[p:] + player2_list[:p]
        normalized_child1 = self.__normalize_attr(child1)
        normalized_child2 = self.__normalize_attr(child2)

        return (Player(height=normalized_child1[0],
                       p_class=player1.p_class,
                       p_attr=PlayerAttributes(
                                    strength=normalized_child1[1],
                                    dexterity=normalized_child1[2],
                                    intelligence=normalized_child1[3],
                                    endurance=normalized_child1[4],
                                    physique=normalized_child1[5])),
                Player(height=normalized_child2[0],
                       p_class=player1.p_class,
                       p_attr=PlayerAttributes(
                                    strength=normalized_child2[1],
                                    dexterity=normalized_child2[2],
                                    intelligence=normalized_child2[3],
                                    endurance=normalized_child2[4],
                                    physique=normalized_child2[5])))

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
