from .PlayerClass import PlayerClass
from .Attributes import Attributes
from .InvalidHeight import InvalidHeight
from decimal import Decimal
import numpy as np


class EVE:
    def __init__(self, height: Decimal,
                 p_class: PlayerClass, attributes: Attributes):
        if height < Decimal('1.3') or height > Decimal('2'):
            raise InvalidHeight
        self.height = height
        self.p_class = p_class
        self.atm = (Decimal('0.5')
                    - (Decimal(3)*height-Decimal(5))**4
                    + (Decimal(3)*height-Decimal(5))**2
                    + height/Decimal(2))
        self.dem = (Decimal('2')
                    - (Decimal(3)*height-Decimal(5))**4
                    - (Decimal(3)*height-Decimal(5))**2
                    - height/Decimal(2))
        self.strength = (Decimal('100')
                         * np.tanh(Decimal('0.01')
                                   * Decimal(attributes.strength)))
        self.dexterity = np.tanh(Decimal('0.01')
                                 * Decimal(attributes.dexterity))
        self.intelligence = (Decimal('0.6')
                             * np.tanh(Decimal('0.01')
                                       * Decimal(attributes.intelligence)))
        self.endurance = np.tanh(Decimal('0.01')
                                 * Decimal(attributes.endurance))
        self.physique = (Decimal('100')
                         * np.tanh(Decimal('0.01')
                                   * Decimal(attributes.physique)))
        self.attack = ((self.dexterity + self.intelligence)
                       * self.strength * self.atm)
        self.defense = ((self.endurance + self.intelligence)
                        * self.physique * self.dem)
        match p_class:
            case PlayerClass.WARRIOR:
                self.performance = (Decimal('0.6') * self.attack
                                    + Decimal('0.4') * self.defense)
            case PlayerClass.ARCHER:
                self.performance = (Decimal('0.9') * self.attack
                                    + Decimal('0.1') * self.defense)
            case PlayerClass.GUARDIAN:
                self.performance = (Decimal('0.1') * self.attack
                                    + Decimal('0.9') * self.defense)
            case PlayerClass.WIZARD:
                self.performance = (Decimal('0.8') * self.attack
                                    + Decimal('0.3') * self.defense)
