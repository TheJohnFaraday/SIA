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
        if not (
            PlayerAttributes.TOTAL_POINTS_MIN
            <= total
            <= PlayerAttributes.TOTAL_POINTS_MAX
        ):
            raise InvalidTotalPoints

        self.strength = strength
        self.dexterity = dexterity
        self.intelligence = intelligence
        self.endurance = endurance
        self.physique = physique
