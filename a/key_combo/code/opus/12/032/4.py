import random
from collections import deque

import numpy as np


BG = 0
CTRL = 2
KEEP_RECOLOR = 7
FG = 8

QUADRANTS = ("NW", "NE", "SW", "SE")
EDGE_PAIRS = (
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _count_holes(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True

    holes = 0
    seen = np.zeros_like(local, dtype=bool)
    h, w = local.shape
    for r in range(h):
        for c in range(w):
            if local[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_boundary = False
            while q:
                cr, cc = q.popleft()
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_boundary = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not local[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_boundary:
                holes += 1
    return holes


def _control_splits(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(grid == CTRL)]
    if len(ctrl_pts) != 2:
        raise ValueError("expected exactly two control pixels")

    split_row = None
    split_col = None
    for r, c in ctrl_pts:
        if r == 0 or r == h - 1:
            split_col = c
        elif c == 0 or c == w - 1:
            split_row = r
    if split_row is None or split_col is None:
        raise ValueError("expected one horizontal and one vertical control ray")
    return split_row, split_col


def _cut_fragments(grid: np.ndarray) -> list[np.ndarray]:
    split_row, split_col = _control_splits(grid)
    fg_mask = grid == FG
    cut_mask = fg_mask.copy()
    cut_mask[split_row, :] = False
    cut_mask[:, split_col] = False
    return _components(cut_mask)


def _closest_corner(coords: np.ndarray, shape: tuple[int, int]) -> str:
    h, w = shape
    r0, c0, r1, c1 = _bbox(coords)
    center_r2 = r0 + r1
    center_c2 = c0 + c1
    candidates = [
        ("TL", 0, 0),
        ("TR", 0, w - 1),
        ("BL", h - 1, 0),
        ("BR", h - 1, w - 1),
    ]
    best_name = None
    best_key = None
    for name, cr, cc in candidates:
        key = (abs(center_r2 - 2 * cr) + abs(center_c2 - 2 * cc), cr, cc)
        if best_key is None or key < best_key:
            best_key = key
            best_name = name
    return str(best_name)


def _move_to_corner(coords: np.ndarray, shape: tuple[int, int], corner: str) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = _bbox(coords)
    if corner == "TL":
        dr, dc = -r0, -c0
    elif corner == "TR":
        dr, dc = -r0, (w - 1) - c1
    elif corner == "BL":
        dr, dc = (h - 1) - r1, -c0
    elif corner == "BR":
        dr, dc = (h - 1) - r1, (w - 1) - c1
    else:
        raise ValueError(f"unknown corner {corner}")
    moved = coords.copy()
    moved[:, 0] += dr
    moved[:, 1] += dc
    return moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut a holed foreground object by two edge rays, rank the fragments by hole count, and keep only the extremes.

    Input structure:
    - Black background contains one teal 4-connected object near the center.
    - The object may contain several enclosed black holes.
    - Exactly two red control pixels lie on perpendicular grid edges and define one horizontal and one vertical ray line.

    Rule:
    - Use the control pixels to determine a split row and split column.
    - Temporarily remove all teal pixels on that row or column; the remaining teal 4-connected components are the cut fragments.
    - Count enclosed black holes inside each fragment's own bounding box.
    - Recolor the unique minimum-hole fragment orange in place.
    - Slide the unique maximum-hole fragment, unchanged, so its bounding-box corner is flush with the nearest grid corner.
    - Remove every other fragment, all red ray pixels, and all cut-line bridge pixels.

    Key insight:
    - The red rays do not paint through the object, but their row/column positions define a topological cut: deleting cut-line foreground bridges reveals the three fragments, and the hole counts decide which two survive.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g, dtype=int)

    fragments = _cut_fragments(g)
    if len(fragments) < 2:
        return out

    hole_counts = [_count_holes(fragment) for fragment in fragments]
    min_idx = int(np.argmin(hole_counts))
    max_idx = int(np.argmax(hole_counts))

    min_fragment = fragments[min_idx]
    max_fragment = fragments[max_idx]

    out[min_fragment[:, 0], min_fragment[:, 1]] = KEEP_RECOLOR
    corner = _closest_corner(max_fragment, g.shape)
    moved = _move_to_corner(max_fragment, g.shape, corner)
    out[moved[:, 0], moved[:, 1]] = FG
    return out


def _tpl(rows: list[str]) -> np.ndarray:
    return np.array([[1 if ch == "1" else 0 for ch in row] for row in rows], dtype=int)


TEMPLATES = {
    0: (
        _tpl(["111", "100", "111"]),
        _tpl(["1111", "1111", "1111"]),
        _tpl(["111", "110", "011"]),
    ),
    1: (
        _tpl(["111", "101", "111"]),
        _tpl(["1111", "1001", "1111"]),
        _tpl(["1111", "1001", "1001", "1111"]),
    ),
    2: (
        _tpl(["11111", "10101", "11111"]),
        _tpl(["111", "101", "111", "101", "111"]),
    ),
}


def _random_template(rng: random.Random, hole_count: int) -> np.ndarray:
    tpl = np.array(rng.choice(TEMPLATES[hole_count]), copy=True)
    if rng.random() < 0.5:
        tpl = np.flipud(tpl)
    if rng.random() < 0.5:
        tpl = np.fliplr(tpl)
    if tpl.shape[0] == tpl.shape[1] and rng.random() < 0.5:
        tpl = tpl.T
    return tpl


def _place_template(tpl: np.ndarray, quadrant: str) -> np.ndarray:
    h, w = tpl.shape
    if quadrant == "NW":
        r0, c0 = -h, -w
    elif quadrant == "NE":
        r0, c0 = -h, 1
    elif quadrant == "SW":
        r0, c0 = 1, -w
    elif quadrant == "SE":
        r0, c0 = 1, 1
    else:
        raise ValueError(f"unknown quadrant {quadrant}")

    pts = []
    for r in range(h):
        for c in range(w):
            if tpl[r, c]:
                pts.append((r0 + r, c0 + c))
    return np.array(pts, dtype=int)


def _object_hole_count(mask: np.ndarray) -> int:
    comps = _components(mask)
    if len(comps) != 1:
        return -1
    return _count_holes(comps[0])


def _build_expected(shape: tuple[int, int], fragments_by_holes: dict[int, np.ndarray]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    min_fragment = fragments_by_holes[0]
    max_fragment = fragments_by_holes[max(fragments_by_holes)]
    out[min_fragment[:, 0], min_fragment[:, 1]] = KEEP_RECOLOR
    corner = _closest_corner(max_fragment, shape)
    moved = _move_to_corner(max_fragment, shape, corner)
    out[moved[:, 0], moved[:, 1]] = FG
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(16, 22)
    w = rng.randint(16, 22)
    vertical_edge, horizontal_edge = rng.choice(EDGE_PAIRS)

    missing = rng.choice(QUADRANTS)
    occupied = [q for q in QUADRANTS if q != missing]
    rng.shuffle(occupied)
    assignment = dict(zip(occupied, (0, 1, 2)))

    fragments_local: dict[int, np.ndarray] = {}
    object_pts = {(0, 0)}
    if "NW" in occupied:
        object_pts.add((0, -1))
    if "NE" in occupied:
        object_pts.add((0, 1))
    if "SW" in occupied:
        object_pts.add((1, 0))

    for quadrant, hole_count in assignment.items():
        coords = _place_template(_random_template(rng, hole_count), quadrant)
        fragments_local[hole_count] = coords
        object_pts.update((int(r), int(c)) for r, c in coords)

    object_local = np.array(sorted(object_pts), dtype=int)
    r0, c0, r1, c1 = _bbox(object_local)
    obj_h = r1 - r0 + 1
    obj_w = c1 - c0 + 1
    if obj_h < 7 or obj_w < 7 or obj_h > 10 or obj_w > 10:
        return None

    top_lo = 3
    left_lo = 3
    top_hi = h - obj_h - 3
    left_hi = w - obj_w - 3
    if top_lo > top_hi or left_lo > left_hi:
        return None

    top_center = (h - obj_h) // 2
    left_center = (w - obj_w) // 2
    top = min(max(top_center + rng.randint(-1, 1), top_lo), top_hi)
    left = min(max(left_center + rng.randint(-1, 1), left_lo), left_hi)

    shift_r = top - r0
    shift_c = left - c0
    split_row = shift_r
    split_col = shift_c

    grid = np.zeros((h, w), dtype=int)
    for r, c in object_local:
        grid[r + shift_r, c + shift_c] = FG

    if vertical_edge == "top":
        grid[0, split_col] = CTRL
    else:
        grid[h - 1, split_col] = CTRL
    if horizontal_edge == "left":
        grid[split_row, 0] = CTRL
    else:
        grid[split_row, w - 1] = CTRL

    fragments_global = {
        holes: coords + np.array([shift_r, shift_c], dtype=int)
        for holes, coords in fragments_local.items()
    }
    expected = _build_expected(grid.shape, fragments_global)

    fg_components = _components(grid == FG)
    if len(fg_components) != 1:
        return None
    if _object_hole_count(grid == FG) not in (3, 4, 5):
        return None

    cut_fragments = _cut_fragments(grid)
    if len(cut_fragments) != 3:
        return None
    measured = sorted(_count_holes(fragment) for fragment in cut_fragments)
    if measured != [0, 1, 2]:
        return None
    if any(fragment.shape[0] < 4 for fragment in cut_fragments):
        return None

    if np.any(expected == KEEP_RECOLOR) and np.any(expected == FG):
        orange_set = set(map(tuple, np.argwhere(expected == KEEP_RECOLOR)))
        teal_set = set(map(tuple, np.argwhere(expected == FG)))
        if orange_set & teal_set:
            return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
