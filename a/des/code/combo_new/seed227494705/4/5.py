import random
import numpy as np


BG = 0
GREY = 5
MAROON = 9


def _receiver_boxes(grid):
    h, w = grid.shape
    boxes = []
    for r in range(h - 2):
        for c in range(w - 2):
            block = grid[r:r + 3, c:c + 3]
            if block[1, 1] == BG and np.all(block[[0, 2], :] == GREY) and np.all(block[:, [0, 2]] == GREY):
                boxes.append((r, c))
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """Broadcast colored rays from singleton towers and fill receiver centers.

    Input structure: Colored singleton signal towers and grey 3x3 hollow
    receiver boxes sit on a black background without overlap.
    Rule: Each tower casts rays in the four cardinal directions, coloring only
    black cells and stopping before any non-black obstacle. A receiver records
    rays that stop adjacent to its grey border; one touching ray fills the
    center with that ray color, while two or more touching rays fill the center
    maroon. Untouched centers remain black.
    Key insight: Ray propagation and receiver counting are separate stages; ray
    collisions affect which receiver touches are counted.
    """
    out = grid.copy()
    h, w = grid.shape
    boxes = _receiver_boxes(grid)
    border_to_box = {}
    for idx, (r, c) in enumerate(boxes):
        for rr in range(r, r + 3):
            for cc in range(c, c + 3):
                if rr in (r, r + 2) or cc in (c, c + 2):
                    border_to_box[(rr, cc)] = idx
    touches = {i: [] for i in range(len(boxes))}
    box_cells = set(border_to_box) | {(r + 1, c + 1) for r, c in boxes}
    towers = []
    for r in range(h):
        for c in range(w):
            col = int(grid[r, c])
            if col not in (BG, GREY) and (r, c) not in box_cells:
                towers.append((r, c, col))
    for r, c, col in towers:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cr, cc = r + dr, c + dc
            last = None
            while 0 <= cr < h and 0 <= cc < w:
                if out[cr, cc] != BG:
                    if (cr, cc) in border_to_box and last is not None:
                        touches[border_to_box[(cr, cc)]].append(col)
                    break
                out[cr, cc] = col
                last = (cr, cc)
                cr += dr
                cc += dc
    for idx, cols in touches.items():
        r, c = boxes[idx]
        if len(cols) == 1:
            out[r + 1, c + 1] = cols[0]
        elif len(cols) >= 2:
            out[r + 1, c + 1] = MAROON
    return out


def generate_input() -> np.ndarray:
    h = random.choice([15, 17, 19, 21])
    w = random.choice([15, 17, 19, 21])
    grid = np.zeros((h, w), dtype=int)
    recs = [(h // 2 - 1, w // 2 - 1), (3, w // 2 - 1), (h // 2 - 1, 3)]
    random.shuffle(recs)
    for r, c in recs[: random.randint(2, 3)]:
        grid[r:r + 3, c:c + 3] = GREY
        grid[r + 1, c + 1] = BG
    colors = random.sample([1, 2, 3, 4, 6, 7, 8], random.randint(3, 5))
    candidates = [(1, w // 2), (h - 2, w // 2), (h // 2, 1), (h // 2, w - 2), (1, 1), (h - 2, w - 2)]
    random.shuffle(candidates)
    for col, (r, c) in zip(colors, candidates):
        if grid[r, c] == BG:
            grid[r, c] = col
    return grid
