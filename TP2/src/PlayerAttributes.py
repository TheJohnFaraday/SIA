from .InvalidTotalPoints import InvalidTotalPoints


class PlayerAttributes:
    def __init__(
        self,
        strength: int,
        dexterity: int,
        intelligence: int,
        endurance: int,
        physique: int,
    ):
        total = strength + dexterity + intelligence + endurance + physique
        if total < 100 or total > 100:
            raise InvalidTotalPoints
        self.strength = strength
        self.dexterity = dexterity
        self.intelligence = intelligence
        self.endurance = endurance
        self.physique = physique
