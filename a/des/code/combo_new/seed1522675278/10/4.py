import random
import numpy as np


BG = 0
WALL = 5


def rot(mask: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(mask, k=k)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move quadrant objects across the central cross and rotate each by a quadrant-specific angle.

    Input structure:
    - The grid is an odd-sized square split by a grey center row and center column.
    - Each quadrant contains exactly one small asymmetric colored object on a black background.

    Rule:
    - Copy the top-left object to the bottom-right quadrant after a 180-degree rotation.
    - Copy the top-right object to the bottom-left quadrant after a 90-degree clockwise rotation.
    - Copy the bottom-right object to the top-left quadrant after a 180-degree rotation.
    - Copy the bottom-left object to the top-right quadrant after a 90-degree counterclockwise rotation.
    - Clear the original objects and keep the grey cross unchanged.

    Key insight:
    - The transform acts on whole quadrant canvases: first swap to the opposite quadrant,
      then rotate by the angle associated with the source quadrant.
    """
    n = grid.shape[0]
    mid = n // 2
    q = mid
    out = np.zeros_like(grid)
    out[mid, :] = WALL
    out[:, mid] = WALL

    quads = {
        "tl": grid[:q, :q],
        "tr": grid[:q, mid + 1 :],
        "bl": grid[mid + 1 :, :q],
        "br": grid[mid + 1 :, mid + 1 :],
    }
    out[:q, :q] = rot(quads["br"], 2)
    out[:q, mid + 1 :] = rot(quads["bl"], 1)
    out[mid + 1 :, :q] = rot(quads["tr"], 3)
    out[mid + 1 :, mid + 1 :] = rot(quads["tl"], 2)
    return out


SHAPES = [
    np.array([[1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=int),
]


def place_shape(canvas: np.ndarray, shape: np.ndarray, color: int) -> None:
    h, w = shape.shape
    for _ in range(200):
        r = random.randint(0, canvas.shape[0] - h)
        c = random.randint(0, canvas.shape[1] - w)
        if np.all(canvas[r : r + h, c : c + w][shape == 1] == BG):
            canvas[r : r + h, c : c + w][shape == 1] = color
            return


def generate_input() -> np.ndarray:
    q = random.randint(4, 7)
    n = 2 * q + 1
    grid = np.zeros((n, n), dtype=int)
    mid = q
    grid[mid, :] = WALL
    grid[:, mid] = WALL
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    quads = [
        grid[:q, :q],
        grid[:q, mid + 1 :],
        grid[mid + 1 :, :q],
        grid[mid + 1 :, mid + 1 :],
    ]
    for quad, color in zip(quads, colors):
        shape = rot(random.choice(SHAPES), random.randint(0, 3))
        place_shape(quad, shape, color)
    return grid
