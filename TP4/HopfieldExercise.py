import numpy as np
from src.Hopfield import Hopfield

FONT = [
    [0x7C, 0x44, 0x44, 0x7C, 0x44],
    [0x7C, 0x44, 0x78, 0x44, 0x7C],
    [0x7C, 0x40, 0x40, 0x40, 0x7C],
    [0x78, 0x44, 0x44, 0x44, 0x78],
    [0x7C, 0x40, 0x78, 0x40, 0x7C],
    [0x7C, 0x40, 0x70, 0x40, 0x40],
    [0x7C, 0x40, 0x4C, 0x44, 0x7C],
    [0x44, 0x44, 0x7C, 0x44, 0x44],
    [0x7C, 0x10, 0x10, 0x10, 0x7C],
    [0x0C, 0x04, 0x04, 0x44, 0x7C],
    [0x44, 0x48, 0x70, 0x48, 0x44],
    [0x40, 0x40, 0x40, 0x40, 0x7C],
    [0x44, 0x6C, 0x54, 0x44, 0x44],
    [0x44, 0x64, 0x54, 0x4C, 0x44],
    [0x38, 0x44, 0x44, 0x44, 0x38],
    [0x78, 0x44, 0x78, 0x40, 0x40],
    [0x7C, 0x44, 0x44, 0x7C, 0x10],
    [0x78, 0x44, 0x78, 0x44, 0x44],
    [0x7C, 0x40, 0x7C, 0x04, 0x7C],
    [0x7C, 0x10, 0x10, 0x10, 0x10],
    [0x44, 0x44, 0x44, 0x44, 0x7C],
    [0x44, 0x44, 0x28, 0x28, 0x10],
    [0x44, 0x44, 0x54, 0x54, 0x28],
    [0x44, 0x28, 0x10, 0x28, 0x44],
    [0x44, 0x44, 0x28, 0x10, 0x10],
    [0x7C, 0x08, 0x10, 0x20, 0x7C],
]


def create_pattern():
    mask = [0, 0, 4, 8, 16, 32, 64, 0, 0]
    patterns = []
    for f in FONT[0:4]:
        pattern = []
        for row in f:
            for i in range(2, 7):
                if row & mask[i] > 0:
                    pattern.append(1)
                else:
                    pattern.append(-1)
        patterns.append(np.array(pattern))

    patterns = np.array(patterns)

    noisy_patterns = np.copy(patterns)
    for p in noisy_patterns:
        for i in range(0, 5):
            if np.random.normal(0.5, 0.6) > 0.5:
                p[i] *= -1

    return (patterns, noisy_patterns)


if __name__ == "__main__":
    patterns, noisy_patterns = create_pattern()
    hopfield = Hopfield(len(patterns[0]))
    for p in patterns:
        hopfield.set_patterns(p)
    print(f'Noisy Pattern to train:\n{noisy_patterns[0]}')
    hopfield.set_initial_state(noisy_patterns[0])
    hopfield.train()
