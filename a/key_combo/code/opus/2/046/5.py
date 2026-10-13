import random
import numpy as np

Color = {
    "BLACK": 0,
    "ORANGE": 7,
    "BLUE": 1,
    "RED": 2,
    "YELLOW": 4,
}


def _nearest_corner(r: int, c: int, h: int, w: int):
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    best = corners[0]
    best_key = (h + w, h, w)
    for cr, cc in corners:
        d = abs(r - cr) + abs(c - cc)
        key = (d, cr, cc)
        if key < best_key:
            best_key = key
            best = (cr, cc)
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill rectangles from orange sources to nearest corners and color by overlap count.

    Input structure:
    - Black background.
    - Three to four isolated orange source pixels.

    Rule:
    - For each source, find the nearest corner using Manhattan distance (row tie, then col tie).
    - Fill the axis-aligned rectangle to that corner.
    - Count for each cell how many rectangles cover it.
    - Recolor overlap counts: 1->blue, 2->red, 3+->yellow.
    - Keep original orange source pixels unchanged.

    Key insight:
    - The final state depends on the global overlap field; local per-pixel fill is only the intermediate step.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sources = [(r, c) for r, c in np.argwhere(g == Color["ORANGE"])]
    if not sources:
        return g.copy()

    cover = np.zeros((h, w), dtype=int)
    for r, c in sources:
        cr, cc = _nearest_corner(int(r), int(c), h, w)
        r0, r1 = sorted((r, cr))
        c0, c1 = sorted((c, cc))
        for rr in range(r0, r1 + 1):
            for cc2 in range(c0, c1 + 1):
                cover[rr, cc2] += 1

    out = np.zeros_like(g)
    for r in range(h):
        for c in range(w):
            cnt = int(cover[r, c])
            if cnt == 0:
                out[r, c] = Color["BLACK"]
            elif cnt == 1:
                out[r, c] = Color["BLUE"]
            elif cnt == 2:
                out[r, c] = Color["RED"]
            else:
                out[r, c] = Color["YELLOW"]

    for r, c in sources:
        out[r, c] = Color["ORANGE"]
    return out


def _candidate_cells_for_corner(h: int, w: int, corner: tuple[int, int]):
    if corner == (0, 0):
        r_range = range(0, h // 2)
        c_range = range(0, w // 2)
    elif corner == (0, w - 1):
        r_range = range(0, h // 2)
        c_range = range((w + 1) // 2, w)
    elif corner == (h - 1, 0):
        r_range = range((h + 1) // 2, h)
        c_range = range(0, w // 2)
    else:
        r_range = range((h + 1) // 2, h)
        c_range = range((w + 1) // 2, w)

    coords = [(r, c) for r in r_range for c in c_range]
    random.shuffle(coords)
    return coords


def generate_input() -> np.ndarray:
    for _ in range(500):
        h = random.randint(10, 14)
        w = random.randint(10, 14)
        g = np.zeros((h, w), dtype=int)

        n_src = random.randint(3, 4)
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        random.shuffle(corners)
        selected = corners[:n_src]

        sources = []
        used = set()
        ok = True
        for cr, cc in selected:
            found = None
            for r, c in _candidate_cells_for_corner(h, w, (cr, cc)):
                if (r, c) in used:
                    continue
                if _nearest_corner(r, c, h, w) == (cr, cc):
                    found = (r, c)
                    break
            if found is None:
                ok = False
                break
            sources.append(found)
            used.add(found)

        if not ok or len(sources) != n_src:
            continue

        for r, c in sources:
            g[r, c] = Color["ORANGE"]
        return g

    raise RuntimeError("generate_input: failed to sample valid case for puzzle 5")


if __name__ == "__main__":
    for _ in range(3):
        x = generate_input()
        y = main(x)
        print(x.shape, y.max(), np.bincount(y.ravel()))
