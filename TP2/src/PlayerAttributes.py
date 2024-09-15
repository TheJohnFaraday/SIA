from .InvalidTotalPoints import InvalidTotalPoints


class PlayerAttributes:
    TOTAL_POINTS = 100

    def __init__(
        self,
        strength: int,
        dexterity: int,
        intelligence: int,
        endurance: int,
        physique: int,
    ):
        total = strength + dexterity + intelligence + endurance + physique
        if total != PlayerAttributes.TOTAL_POINTS:
            raise InvalidTotalPoints

        self.strength = strength
        self.dexterity = dexterity
        self.intelligence = intelligence
        self.endurance = endurance
        self.physique = physique
