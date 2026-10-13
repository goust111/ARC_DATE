import random
from collections import deque

import numpy as np


BG = 0
TEMPLATE = 8
IND = 6


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _norm(comp):
    r0, c0, _, _ = _bbox(comp)
    return tuple(sorted((r - r0, c - c0) for r, c in comp))


def _sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def _adjacent_indicator(g, comp):
    found = set()
    cells = set(comp)
    for r, c in comp:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in cells and 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and g[nr, nc] == IND:
                found.add((nr, nc))
    assert len(found) == 1, "each object must have one adjacent indicator"
    return next(iter(found))


def _centroid_dir(comp, indicator):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    cr = int(round(sum(rs) / len(rs)))
    cc = int(round(sum(cs) / len(cs)))
    return _sign(indicator[0] - cr), _sign(indicator[1] - cc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor template-matching objects cyan and activate only their adjacent magenta ray indicators.

    Input structure:
    - A black grid contains one cyan template object and several blue or red candidate objects.
    - Every object has exactly one 4-adjacent magenta indicator pixel.
    - Some candidates have the same normalized shape as the cyan template; others do not.

    Rule:
    - Normalize the cyan template mask and compare every non-cyan, non-magenta object to it.
    - Candidate objects matching the template are recolored cyan; nonmatching candidates keep their color.
    - Indicators adjacent to cyan objects, including the original template and recolored matches, emit cyan rays in the centroid-to-indicator direction.
    - Indicators next to nonmatching objects are erased to black; rays stop before the first non-black cell or the grid boundary.

    Key insight:
    - The visible magenta indicators are conditionally activated by shape matching; matching controls both recoloring and ray emission.
    """
    g = np.asarray(grid, dtype=int)
    template_comps = _components(g == TEMPLATE)
    assert len(template_comps) == 1, "expected one cyan template"
    tmpl = _norm(template_comps[0])

    objects = []
    for color in sorted(set(map(int, g.ravel())) - {BG, IND, TEMPLATE}):
        for comp in _components(g == color):
            objects.append({"color": color, "comp": comp, "match": _norm(comp) == tmpl})

    out = g.copy()
    active = [(template_comps[0], _adjacent_indicator(g, template_comps[0]))]
    for obj in objects:
        comp = obj["comp"]
        ind = _adjacent_indicator(g, comp)
        if obj["match"]:
            for r, c in comp:
                out[r, c] = TEMPLATE
            active.append((comp, ind))
        else:
            out[ind] = BG

    for comp, ind in sorted(active, key=lambda x: x[1]):
        dr, dc = _centroid_dir(comp, ind)
        r, c = ind[0] + dr, ind[1] + dc
        while 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            if out[r, c] != BG:
                break
            out[r, c] = TEMPLATE
            r += dr
            c += dc
    return out


MATCH_SHAPE = [(0, 0), (1, 0), (1, 1), (2, 1)]
OTHER_SHAPES = [
    [(0, 0), (0, 1), (1, 0), (2, 0)],
    [(0, 1), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _indicator(shape, r0, c0, direction):
    pts = [(r0 + r, c0 + c) for r, c in shape]
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    cr = int(round(sum(rs) / len(rs)))
    cc = int(round(sum(cs) / len(cs)))
    dr, dc = direction
    candidates = []
    for r, c in pts:
        nr, nc = r + dr, c + dc
        if (nr, nc) not in pts:
            candidates.append((abs((nr - cr) - dr) + abs((nc - cc) - dc), nr, nc))
    if not candidates:
        return None
    _, nr, nc = min(candidates)
    return nr, nc


def _can_place(grid, cells, ind):
    all_cells = set(cells) | {ind}
    for r, c in all_cells:
        if not (1 <= r < grid.shape[0] - 1 and 1 <= c < grid.shape[1] - 1):
            return False
        if grid[r, c] != BG:
            return False
    for r, c in all_cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in all_cells and grid[nr, nc] != BG:
                return False
    return True


def _place(grid, shape, color):
    for _ in range(200):
        direction = random.choice(DIRS)
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        r0 = random.randint(2, grid.shape[0] - max_r - 3)
        c0 = random.randint(2, grid.shape[1] - max_c - 3)
        ind = _indicator(shape, r0, c0, direction)
        if ind is None:
            continue
        cells = [(r0 + r, c0 + c) for r, c in shape]
        if not _can_place(grid, cells, ind):
            continue
        for r, c in cells:
            grid[r, c] = color
        grid[ind] = IND
        return True
    return False


def _gen_attempt():
    h = w = random.randint(15, 20)
    grid = np.zeros((h, w), dtype=int)
    if not _place(grid, MATCH_SHAPE, TEMPLATE):
        return None
    specs = [(1, MATCH_SHAPE), (1, MATCH_SHAPE), (1, random.choice(OTHER_SHAPES)), (2, random.choice(OTHER_SHAPES)), (2, random.choice(OTHER_SHAPES))]
    if random.random() < 0.5:
        specs.append((1, random.choice(OTHER_SHAPES)))
    random.shuffle(specs)
    for color, shape in specs:
        if not _place(grid, shape, color):
            return None
    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
