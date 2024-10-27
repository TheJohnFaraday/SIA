import matplotlib.pyplot as plt
import numpy as np
from src.Hopfield import Hopfield

FONT = [
    [0x7C, 0x44, 0x44, 0x7C, 0x44],
    [0x7C, 0x44, 0x7C, 0x44, 0x7C],
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
    # mask = [0, 0, 4, 8, 16, 32, 64, 0, 0]
    # patterns = []
    # for f in FONT[0:4]:
    #     pattern = []
    #     for row in f:
    #         for i in range(2, 7):
    #             if row & mask[i] > 0:
    #                 pattern.append(1)
    #             else:
    #                 pattern.append(-1)
    #     patterns.append(np.array(pattern))
    patterns = [
        [
            [-1, 1, 1, 1, -1],
            [1, -1, -1, -1, 1],
            [1, 1, 1, 1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
        ],
        [
            [1, 1, 1, 1, -1],
            [1, -1, -1, -1, 1],
            [1, 1, 1, 1, 1],
            [1, -1, -1, -1, 1],
            [1, 1, 1, 1, -1],
        ],
        [
            [-1, 1, 1, 1, 1],
            [1, -1, -1, -1, -1],
            [1, -1, -1, -1, -1],
            [1, -1, -1, -1, -1],
            [-1, 1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 1, -1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, 1, 1, 1, -1],
        ],
    ]
    patterns = np.array(patterns)
    patterns = np.reshape(patterns, (4, 25))

    noisy_patterns = np.copy(patterns)
    for p in noisy_patterns:
        for i in range(25):
            if np.random.normal(0.3, 0.15) > 0.5:
                p[i] *= -1

    return (patterns, noisy_patterns)


if __name__ == "__main__":
    # np.random.seed(101)
    patterns, noisy_patterns = create_pattern()
    hopfield = Hopfield(len(patterns[0]))
    print(f"Train Patterns: \n{patterns[:4]}")
    for p in patterns[:4]:
        hopfield.set_patterns(p)
    print(f"Original Pattern:\n{patterns[3]}")
    print(f"Noisy Pattern to train:\n{noisy_patterns[3]}")
    hopfield.set_initial_state(noisy_patterns[3])
    final_result, trace = hopfield.train()
    print(f"Final Result:\n{final_result}")
    print(f"Number of steps:{len(trace)}")

    plt.subplot(211)
    plt.imshow(np.reshape(patterns[3], (5, 5)))
    plt.subplot(212)
    plt.imshow(np.reshape(patterns[3], (5, 5)), cmap="Greys", interpolation="nearest")
    plt.savefig("./plots/original-Hopfield.png")
    plt.subplot(211)
    plt.imshow(np.reshape(noisy_patterns[3], (5, 5)))
    plt.subplot(212)
    plt.imshow(
        np.reshape(noisy_patterns[3], (5, 5)), cmap="Greys", interpolation="nearest"
    )
    plt.savefig("./plots/noisy_test-Hopfield.png")
    plt.subplot(211)
    plt.imshow(np.reshape(final_result, (5, 5)))
    plt.subplot(212)
    plt.imshow(np.reshape(final_result, (5, 5)), cmap="Greys", interpolation="nearest")
    plt.savefig("./plots/final_result-Hopfield.png")
    i = 0
    for t in trace:
        plt.subplot(211)
        plt.imshow(np.reshape(final_result, (5, 5)))
        plt.subplot(212)
        plt.imshow(
            np.reshape(final_result, (5, 5)), cmap="Greys", interpolation="nearest"
        )
        plt.savefig(f"./plots/trace-{i}-Hopfield.png")
        i += 1
