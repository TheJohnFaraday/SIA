from .InvalidTotalPoints import InvalidTotalPoints


class PlayerAttributes:
    TOTAL_POINTS_MIN = 100
    TOTAL_POINTS_MAX = 200
    NUMBER_OF_ATTRIBUTES = 5  # Count of attributes

    def __init__(
        self,
        strength: int,
        dexterity: int,
        intelligence: int,
        endurance: int,
        physique: int,
    ):
        total = strength + dexterity + intelligence + endurance + physique
        cond = (
            PlayerAttributes.TOTAL_POINTS_MIN
            <= total
            <= PlayerAttributes.TOTAL_POINTS_MAX
        )
        if not cond:
            raise InvalidTotalPoints(f"Failed because total points is {total}")

        self.strength = strength
        self.dexterity = dexterity
        self.intelligence = intelligence
        self.endurance = endurance
        self.physique = physique

    @staticmethod
    def normalize_attr(player_attr: list[int], total_points: int) -> list[int]:
        player_attr_isolated = player_attr[1:]  # Remove Height
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
