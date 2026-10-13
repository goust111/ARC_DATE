import random
from collections import Counter, deque

import numpy as np


GRAY = 5
BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MISSING_BY_DIR = {
    "up": (0, 1),
    "right": (1, 2),
    "down": (2, 1),
    "left": (1, 0),
}
QUADRANTS = {
    "up": (0, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (1, 1),
}
SIDES = ["left", "right", "up", "down"]
MAX_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    (r0, c0), (r1, c1) = pts.min(axis=0), pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_templates(grid: np.ndarray) -> dict[str, np.ndarray]:
    box = _bbox(grid == GRAY)
    assert box is not None, "gray template library is missing"
    r0, c0, r1, c1 = box
    lib_h = r1 - r0 + 1
    lib_w = c1 - c0 + 1
    assert lib_h >= 5 and lib_w >= 5
    assert (lib_h - 3) % 2 == 0 and (lib_w - 3) % 2 == 0
    stamp_h = (lib_h - 3) // 2
    stamp_w = (lib_w - 3) // 2
    split_r = r0 + 1 + stamp_h
    split_c = c0 + 1 + stamp_w

    assert np.all(grid[r0, c0 : c1 + 1] == GRAY)
    assert np.all(grid[r1, c0 : c1 + 1] == GRAY)
    assert np.all(grid[r0 : r1 + 1, c0] == GRAY)
    assert np.all(grid[r0 : r1 + 1, c1] == GRAY)
    assert np.all(grid[split_r, c0 : c1 + 1] == GRAY)
    assert np.all(grid[r0 : r1 + 1, split_c] == GRAY)

    return {
        "up": grid[r0 + 1 : split_r, c0 + 1 : split_c] == GRAY,
        "right": grid[r0 + 1 : split_r, split_c + 1 : c1] == GRAY,
        "down": grid[split_r + 1 : r1, c0 + 1 : split_c] == GRAY,
        "left": grid[split_r + 1 : r1, split_c + 1 : c1] == GRAY,
    }


def _seed_direction(grid: np.ndarray, color: int, comp: list[tuple[int, int]]) -> str:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 == 2 and c1 - c0 == 2 and len(comp) == 8
    block = grid[r0 : r1 + 1, c0 : c1 + 1] == color
    miss = np.argwhere(~block)
    assert miss.shape == (1, 2)
    missing = tuple(map(int, miss[0]))
    for direction, pos in MISSING_BY_DIR.items():
        if pos == missing:
            return direction
    raise AssertionError("seed gap must lie on one side center")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    templates = _extract_templates(g)
    out = g.copy()

    for color in sorted(set(map(int, np.unique(g))) - {bg, GRAY}):
        comps = _components_of_color(g, color)
        seed = None
        anchor = None
        for comp in comps:
            if len(comp) == 1:
                anchor = comp[0]
            elif len(comp) == 8:
                rows = [r for r, _ in comp]
                cols = [c for _, c in comp]
                if max(rows) - min(rows) == 2 and max(cols) - min(cols) == 2:
                    seed = comp
        assert seed is not None and anchor is not None, f"color {color} needs one seed and one anchor"

        direction = _seed_direction(g, color, seed)
        mask = templates[direction]
        ar, ac = anchor
        mh, mw = mask.shape
        assert ar + mh <= g.shape[0] and ac + mw <= g.shape[1]
        rr, cc = np.nonzero(mask)
        out[ar + rr, ac + cc] = color

    return out


def _reserve(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _is_free(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _repeat_balanced(rng: random.Random, items: list[str], n: int) -> list[str]:
    out: list[str] = []
    while len(out) < n:
        chunk = items[:]
        rng.shuffle(chunk)
        out.extend(chunk)
    return out[:n]


def _connected_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    total = h * w
    target = rng.randint(3, total - 1)
    cells = {(0, 0)}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_templates(rng: random.Random, h: int, w: int) -> dict[str, np.ndarray]:
    seen: set[tuple[int, ...]] = set()
    templates: dict[str, np.ndarray] = {}
    for direction in ["up", "right", "down", "left"]:
        for _ in range(100):
            mask = _connected_mask(rng, h, w)
            flat = tuple(int(x) for x in mask.ravel())
            if flat in seen:
                continue
            seen.add(flat)
            templates[direction] = mask
            break
        else:
            raise RuntimeError("failed to create distinct templates")
    return templates


def _draw_library(
    grid: np.ndarray,
    top: int,
    left: int,
    templates: dict[str, np.ndarray],
) -> tuple[int, int, int, int]:
    stamp_h, stamp_w = next(iter(templates.values())).shape
    lib_h = 2 * stamp_h + 3
    lib_w = 2 * stamp_w + 3
    bottom = top + lib_h - 1
    right = left + lib_w - 1

    grid[top : bottom + 1, left] = GRAY
    grid[top : bottom + 1, right] = GRAY
    grid[top, left : right + 1] = GRAY
    grid[bottom, left : right + 1] = GRAY
    split_r = top + 1 + stamp_h
    split_c = left + 1 + stamp_w
    grid[split_r, left : right + 1] = GRAY
    grid[top : bottom + 1, split_c] = GRAY

    for direction, (qr, qc) in QUADRANTS.items():
        r0 = top + 1 + qr * (stamp_h + 1)
        c0 = left + 1 + qc * (stamp_w + 1)
        mask = templates[direction]
        rr, cc = np.nonzero(mask)
        grid[r0 + rr, c0 + cc] = GRAY

    return top, left, bottom, right


def _seed_cells(direction: str) -> list[tuple[int, int]]:
    missing = MISSING_BY_DIR[direction]
    cells = []
    for r in range(3):
        for c in range(3):
            if (r, c) != missing:
                cells.append((r, c))
    return cells


def _side_gap(rng: random.Random, side: str, stamp_h: int, stamp_w: int) -> int | None:
    if side == "left":
        low = stamp_w
        return None if low > 4 else rng.randint(low, 4)
    if side == "up":
        low = stamp_h
        return None if low > 4 else rng.randint(low, 4)
    return rng.randint(2, 4)


def _place_object(
    rng: random.Random,
    reserved: np.ndarray,
    stamp_h: int,
    stamp_w: int,
    side: str,
) -> tuple[int, int, int, int] | None:
    h, w = reserved.shape
    gap = _side_gap(rng, side, stamp_h, stamp_w)
    if gap is None:
        return None

    for _ in range(200):
        if side == "right":
            max_sc = w - gap - stamp_w - 3
            max_sr = h - max(3, stamp_h)
            if max_sc < 0 or max_sr < 0:
                return None
            sr = rng.randint(0, max_sr)
            sc = rng.randint(0, max_sc)
            ar, ac = sr, sc + 3 + gap
        elif side == "left":
            min_sc = gap + 1
            max_sr = h - max(3, stamp_h)
            if min_sc > w - 3 or max_sr < 0:
                return None
            sr = rng.randint(0, max_sr)
            sc = rng.randint(min_sc, w - 3)
            ar, ac = sr, sc - gap - 1
        elif side == "down":
            max_sr = h - gap - stamp_h - 3
            max_sc = w - max(3, stamp_w)
            if max_sr < 0 or max_sc < 0:
                return None
            sr = rng.randint(0, max_sr)
            sc = rng.randint(0, max_sc)
            ar, ac = sr + 3 + gap, sc
        else:
            min_sr = gap + 1
            max_sc = w - max(3, stamp_w)
            if min_sr > h - 3 or max_sc < 0:
                return None
            sr = rng.randint(min_sr, h - 3)
            sc = rng.randint(0, max_sc)
            ar, ac = sr - gap - 1, sc

        seed_r1, seed_c1 = sr + 2, sc + 2
        stamp_r1, stamp_c1 = ar + stamp_h - 1, ac + stamp_w - 1
        if stamp_r1 >= h or stamp_c1 >= w:
            continue
        r0 = min(sr, ar)
        c0 = min(sc, ac)
        r1 = max(seed_r1, stamp_r1)
        c1 = max(seed_c1, stamp_c1)
        if _is_free(reserved, r0, c0, r1, c1, pad=1):
            _reserve(reserved, r0, c0, r1, c1, pad=1)
            return sr, sc, ar, ac
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    stamp_h = rng.randint(3, 4)
    stamp_w = rng.randint(3, 4)
    templates = _random_templates(rng, stamp_h, stamp_w)
    lib_h = 2 * stamp_h + 3
    lib_w = 2 * stamp_w + 3

    count = rng.randint(4, 6)
    min_h = max(18, lib_h + 8)
    min_w = max(18, lib_w + 8)
    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    corner = rng.choice(["tl", "tr", "bl", "br"])
    row_margin = rng.randint(0, 1)
    col_margin = rng.randint(0, 1)
    if corner == "tl":
        lib_r0, lib_c0 = row_margin, col_margin
    elif corner == "tr":
        lib_r0, lib_c0 = row_margin, w - lib_w - col_margin
    elif corner == "bl":
        lib_r0, lib_c0 = h - lib_h - row_margin, col_margin
    else:
        lib_r0, lib_c0 = h - lib_h - row_margin, w - lib_w - col_margin

    lib_box = _draw_library(grid, lib_r0, lib_c0, templates)
    _reserve(reserved, *lib_box, pad=1)

    colors = rng.sample(COLORS, count)
    directions = _repeat_balanced(rng, ["up", "right", "down", "left"], count)
    sides = _repeat_balanced(rng, SIDES, count)
    placements = []

    for color, direction, side in zip(colors, directions, sides):
        pos = _place_object(rng, reserved, stamp_h, stamp_w, side)
        if pos is None:
            return None
        sr, sc, ar, ac = pos
        for dr, dc in _seed_cells(direction):
            grid[sr + dr, sc + dc] = color
        grid[ar, ac] = color
        placements.append((color, direction, ar, ac))

    expected = grid.copy()
    for color, direction, ar, ac in placements:
        mask = templates[direction]
        rr, cc = np.nonzero(mask)
        expected[ar + rr, ac + cc] = color

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
