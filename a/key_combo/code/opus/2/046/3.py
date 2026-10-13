import random
from collections import deque
import numpy as np

Color = {
    "BLACK": 0,
    "BLUE": 1,
    "RED": 2,
    "GREEN": 3,
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bfs_components(grid: np.ndarray, color: int, r0: int, r1: int, c0: int, c1: int):
    h, w = grid.shape
    seen = set()
    comps = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            if grid[r, c] != color or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if r0 <= nr < r1 and c0 <= nc < c1 and grid[nr, nc] == color and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_of_coords(coords):
    rs = [p[0] for p in coords]
    cs = [p[1] for p in coords]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the missing fourth mirrored red sprite, then slide blue pixels right against all obstacles.

    Input structure:
    - A green separator cross that partitions a central 2x2 cell lattice into four quadrants.
    - Exactly three quadrants contain a 4-8-cell red sprite; one quadrant is empty.
    - Column 16 is a blue wall; left side contains independent blue pixels.

    Rule:
    - Infer the missing quadrant by taking the diagonal opposite sprite and applying horizontal+vertical flip.
    - Remove all green separator pixels.
    - Slide each blue pixel to the right until just before blue wall, red sprite cells, or a previously settled blue pixel.
    - Process each row right-to-left (from wall side outward).

    Key insight:
    - The symmetry completion is not cosmetic; the completed sprite becomes a real obstacle for subsequent collision-driven sliding.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    row_sep = int(np.argmax(np.sum(g == Color["GREEN"], axis=1)))
    col_sep = int(np.argmax(np.sum(g == Color["GREEN"], axis=0)))

    quadrants = [
        (0, row_sep, 0, col_sep),
        (0, row_sep, col_sep + 1, w),
        (row_sep + 1, h, 0, col_sep),
        (row_sep + 1, h, col_sep + 1, w),
    ]

    present = {}
    for qi, (r0, r1, c0, c1) in enumerate(quadrants):
        comps = _bfs_components(g, Color["RED"], r0, r1, c0, c1)
        if comps:
            # pick the largest component in that quadrant
            comp = max(comps, key=lambda x: len(x))
            present[qi] = comp

    if len(present) != 3:
        return g.copy()

    missing = ({0, 1, 2, 3} - set(present.keys())).pop()
    opp = missing ^ 3
    opp_comp = present.get(opp)
    if opp_comp is None:
        return g.copy()

    r0q = quadrants[opp][0]
    c0q = quadrants[opp][2]
    rq0, cq0 = _bbox_of_coords(opp_comp)
    hr = (max(r for r, _ in opp_comp) - min(r for r, _ in opp_comp)) + 1
    wc = (max(c for _, c in opp_comp) - min(c for _, c in opp_comp)) + 1
    patch = np.zeros((hr, wc), dtype=bool)
    for r, c in opp_comp:
        patch[r - rq0, c - cq0] = True

    inferred_patch = np.flipud(np.fliplr(patch))
    inf_coords = []
    mr0, mc0 = quadrants[missing][0], quadrants[missing][2]
    for rr in range(hr):
        for cc in range(wc):
            if inferred_patch[rr, cc]:
                gr = mr0 + (rr + (rq0 - r0q))
                gc = mc0 + (cc + (cq0 - c0q))
                if 0 <= gr < h and 0 <= gc < w:
                    inf_coords.append((gr, gc))

    out = g.copy()
    # fill missing sprite
    for r, c in inf_coords:
        out[r, c] = Color["RED"]

    # remove separators
    for r in range(h):
        out[r, col_sep] = Color["BLACK"]
    for c in range(w):
        out[row_sep, c] = Color["BLACK"]

    wall = w - 1
    for r in range(h):
        blues = [c for c in range(w - 1) if out[r, c] == Color["BLUE"]]
        for c in sorted(blues, reverse=True):
            if out[r, c] != Color["BLUE"]:
                continue
            out[r, c] = Color["BLACK"]
            cc = c
            while cc + 1 < w and out[r, cc + 1] == Color["BLACK"]:
                cc += 1
            if cc >= wall:
                cc = wall - 1
            out[r, cc] = Color["BLUE"]

    return out


def _random_connected_shape(size: int, h: int, w: int):
    shape = {(random.randrange(h), random.randrange(w))}
    frontier = list(shape)
    while len(shape) < size:
        r, c = random.choice(frontier)
        nbrs = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        random.shuffle(nbrs)
        for nr, nc in nbrs:
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in shape:
                shape.add((nr, nc))
                frontier.append((nr, nc))
                if len(shape) >= size:
                    break
        else:
            frontier = list(shape)
    return shape


def _transform_coords(coords, h, w, mode):
    out = []
    for r, c in coords:
        rr, cc = r, c
        if "h" in mode:
            cc = w - 1 - cc
        if "v" in mode:
            rr = h - 1 - rr
        out.append((rr, cc))
    return out


def generate_input() -> np.ndarray:
    for _ in range(400):
        h, w = 17, 15
        cell = 5
        wall = w - 1
        source_w = wall - cell * 2 - 1
        if source_w <= 0:
            continue

        row_sep = 8
        col_sep = 8
        quadrants = [
            (3, 3 + cell, 3, 3 + cell),
            (3, 3 + cell, col_sep + 1, col_sep + 1 + cell),
            (row_sep + 1, row_sep + 1 + cell, 3, 3 + cell),
            (row_sep + 1, row_sep + 1 + cell, col_sep + 1, col_sep + 1 + cell),
        ]

        base = random.randint(0, 3)
        missing = base ^ 3

        base_shape = _random_connected_shape(random.randint(4, 8), cell, cell)
        rmin, cmin = min(r for r, _ in base_shape), min(c for _, c in base_shape)
        hbox = max(r for r, _ in base_shape) - rmin + 1
        wbox = max(c for _, c in base_shape) - cmin + 1
        base_local = [(r - rmin, c - cmin) for r, c in base_shape]

        g = np.zeros((h, w), dtype=int)

        for qi in range(4):
            if qi == missing:
                continue
            q_r0, q_c0, _, _ = 0, 0, 0, 0
            q_r0, q_r1, q_c0, q_c1 = quadrants[qi]
            if qi == base:
                mode = ""
            elif qi == (base ^ 1):
                mode = "v"
            elif qi == (base ^ 2):
                mode = "h"
            else:
                continue

            tr = _transform_coords(base_local, hbox, wbox, mode)
            for rr, cc in tr:
                r = q_r0 + rmin + rr
                c = q_c0 + cmin + cc
                g[r, c] = Color["RED"]

        # infer missing for guaranteed non-duplicate and valid completion
        inf_mode = "hv"
        inf = _transform_coords(base_local, hbox, wbox, inf_mode)
        if missing in range(4):
            q_r0, q_r1, q_c0, q_c1 = quadrants[missing]
            for rr, cc in inf:
                g[q_r0 + rmin + rr, q_c0 + cmin + cc] = Color["RED"]

        # green separators and wall
        g[row_sep, :] = Color["GREEN"]
        g[:, col_sep] = Color["GREEN"]
        g[:, wall] = Color["BLUE"]

        # choose source rows influenced by inferred shape
        influence_rows = [p[0] for p in inf]
        if not influence_rows:
            continue
        n_blue = random.randint(3, 5)
        placed_rows = set()
        for i in range(n_blue):
            if i == 0:
                r = random.choice(influence_rows)
            else:
                r = random.randint(0, h - 1)
            if r in placed_rows:
                continue
            c = random.randint(0, source_w - 1)
            g[r, c] = Color["BLUE"]
            placed_rows.add(r)

        # safety: at least one blue exists in a row hit by inferred block
        valid = False
        for r, c in np.argwhere(g[:, :wall] == Color["BLUE"]):
            rr, cc = int(r), int(c)
            if rr in influence_rows:
                valid = True
                break
        if not valid:
            continue

        out = main(g)
        if out.shape == g.shape:
            return g

    raise RuntimeError("generate_input: failed to sample a valid case for puzzle 3")


if __name__ == "__main__":
    for _ in range(3):
        x = generate_input()
        y = main(x)
        print(x.shape, y.sum(), y.max())
