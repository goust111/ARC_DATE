import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ORIENTATIONS = ("up", "down", "left", "right")
GENERATE_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _ordered_axis(axis_cells: list[tuple[int, int]], orientation: str) -> list[tuple[int, int]]:
    if orientation == "up":
        return sorted(axis_cells)
    if orientation == "down":
        return sorted(axis_cells, reverse=True)
    if orientation == "left":
        return sorted(axis_cells, key=lambda p: p[1])
    return sorted(axis_cells, key=lambda p: p[1], reverse=True)


def _parse_arrow(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    local = np.zeros((h, w), dtype=bool)
    for r, c in comp:
        local[r - r0, c - c0] = True

    orientation = None
    axis_cells: list[tuple[int, int]] = []
    search_start = None
    step = None

    if w == 3 and h >= 7 and local[0, 1] and np.all(local[1, :]):
        orientation = "up"
        axis_cells = [(r, c0 + 1) for r in range(r0 + 3, r1)]
        search_start = (r0 - 1, c0 + 1)
        step = (-1, 0)
    elif w == 3 and h >= 7 and local[-1, 1] and np.all(local[-2, :]):
        orientation = "down"
        axis_cells = [(r, c0 + 1) for r in range(r0 + 1, r1 - 2)]
        search_start = (r1 + 1, c0 + 1)
        step = (1, 0)
    elif h == 3 and w >= 7 and local[1, 0] and np.all(local[:, 1]):
        orientation = "left"
        axis_cells = [(r0 + 1, c) for c in range(c0 + 3, c1)]
        search_start = (r0 + 1, c0 - 1)
        step = (0, -1)
    elif h == 3 and w >= 7 and local[1, -1] and np.all(local[:, -2]):
        orientation = "right"
        axis_cells = [(r0 + 1, c) for c in range(c0 + 1, c1 - 2)]
        search_start = (r0 + 1, c1 + 1)
        step = (0, 1)

    assert orientation is not None, "unrecognized arrow geometry"
    ordered_axis = _ordered_axis(axis_cells, orientation)
    divider_cells = [p for p in ordered_axis if int(grid[p]) not in (BG, GREY)]
    divider_idx_to_color = {i: int(grid[p]) for i, p in enumerate(ordered_axis) if int(grid[p]) not in (BG, GREY)}
    assert len(divider_idx_to_color) in (1, 2), "expected one or two divider cells"

    return {
        "bbox": (r0, c0, r1, c1),
        "orientation": orientation,
        "axis": ordered_axis,
        "dividers": divider_idx_to_color,
        "search_start": search_start,
        "step": step,
    }


def _find_target(grid: np.ndarray, arrow: dict) -> tuple[list[tuple[int, int]], int]:
    h, w = grid.shape
    n = len(arrow["axis"])
    sr, sc = arrow["search_start"]
    dr, dc = arrow["step"]

    gap = 0
    r, c = sr, sc
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == BG:
        gap += 1
        r += dr
        c += dc
    assert gap in (1, 2), "target gap must be one or two cells"
    assert 0 <= r < h and 0 <= c < w, "target must exist in front of arrow"

    color = int(grid[r, c])
    assert color not in (BG, GREY), "target color must be visible and non-grey"

    target: list[tuple[int, int]] = []
    rr, cc = r, c
    while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == color:
        target.append((rr, cc))
        rr += dr
        cc += dc
    assert len(target) == n, "target length must match shaft interior length"
    assert 0 > rr or rr >= h or 0 > cc or cc >= w or int(grid[rr, cc]) == BG, "target must end cleanly"
    return target, color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    out = g.copy()
    grey_comps = _components(g == GREY)
    assert grey_comps, "expected at least one grey arrow"

    arrows = [_parse_arrow(g, comp) for comp in grey_comps]
    for arrow in arrows:
        target_cells, target_color = _find_target(g, arrow)
        for r, c in target_cells:
            out[r, c] = BG

        divider_items = sorted(arrow["dividers"].items())
        axis = arrow["axis"]
        first_divider = divider_items[0][0]
        for i, (r, c) in enumerate(axis):
            if i in arrow["dividers"]:
                continue
            new_color = target_color
            if i > first_divider:
                for idx, color in divider_items:
                    if i > idx:
                        new_color = color
            out[r, c] = new_color

    return out


def _make_vertical_arrow(body_len: int, orientation: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    h = body_len + 2
    arr = np.zeros((h, 3), dtype=int)
    arr[1, :] = GREY
    arr[0, 1] = GREY
    arr[2:, 0] = GREY
    arr[2:, 2] = GREY
    arr[2, 1] = GREY
    arr[-1, 1] = GREY
    axis = [(r, 1) for r in range(3, h - 1)]
    if orientation == "up":
        return arr, axis
    return np.flipud(arr), [(h - 1 - r, c) for r, c in axis]


def _make_horizontal_arrow(body_len: int, orientation: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    w = body_len + 2
    arr = np.zeros((3, w), dtype=int)
    arr[:, 1] = GREY
    arr[1, 0] = GREY
    arr[0, 2:] = GREY
    arr[2, 2:] = GREY
    arr[1, 2] = GREY
    arr[1, -1] = GREY
    axis = [(1, c) for c in range(3, w - 1)]
    if orientation == "left":
        return arr, axis
    return np.fliplr(arr), [(r, w - 1 - c) for r, c in axis]


def _build_arrow(body_len: int, orientation: str, divider_positions: list[int], divider_colors: list[int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if orientation in ("up", "down"):
        arr, axis = _make_vertical_arrow(body_len, orientation)
    else:
        arr, axis = _make_horizontal_arrow(body_len, orientation)
    for idx, color in zip(divider_positions, divider_colors):
        rr, cc = axis[idx]
        arr[rr, cc] = color
    return arr, axis


def _front_step(orientation: str) -> tuple[int, int]:
    return {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[orientation]


def _arrow_front_anchor(orientation: str, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if orientation == "up":
        return (0, 1)
    if orientation == "down":
        return (h - 1, 1)
    if orientation == "left":
        return (1, 0)
    return (1, w - 1)


def _target_cells(top: int, left: int, shape: tuple[int, int], orientation: str, gap: int, interior_len: int) -> list[tuple[int, int]]:
    fr, fc = _arrow_front_anchor(orientation, shape)
    dr, dc = _front_step(orientation)
    start_r = top + fr + dr * (gap + 1)
    start_c = left + fc + dc * (gap + 1)
    return [(start_r + dr * i, start_c + dc * i) for i in range(interior_len)]


def _stamp(grid: np.ndarray, top: int, left: int, patch: np.ndarray) -> None:
    h, w = patch.shape
    mask = patch != BG
    grid[top : top + h, left : left + w][mask] = patch[mask]


def _fits(occ: np.ndarray, cells: list[tuple[int, int]], margin: int = 1) -> bool:
    h, w = occ.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - margin)
        r1 = min(h, r + margin + 1)
        c0 = max(0, c - margin)
        c1 = min(w, c + margin + 1)
        if np.any(occ[r0:r1, c0:c1]):
            return False
    return True


def _paint_occ(occ: np.ndarray, cells: list[tuple[int, int]], margin: int = 1) -> None:
    h, w = occ.shape
    for r, c in cells:
        r0 = max(0, r - margin)
        r1 = min(h, r + margin + 1)
        c0 = max(0, c - margin)
        c1 = min(w, c + margin + 1)
        occ[r0:r1, c0:c1] = True


def _choose_dividers(interior_len: int) -> list[int]:
    if interior_len >= 5 and random.random() < 0.5:
        candidates = []
        for a in range(1, interior_len - 3):
            for b in range(a + 2, interior_len - 1):
                candidates.append([a, b])
        if candidates:
            return random.choice(candidates)
    choices = list(range(1, interior_len - 1))
    return [random.choice(choices)]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    grid = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    count_choices = [1, 2, 3]
    weights = [0.3, 0.4, 0.3]
    n_arrows = random.choices(count_choices, weights=weights, k=1)[0]

    for _ in range(n_arrows):
        placed = False
        for _attempt in range(120):
            orientation = random.choice(ORIENTATIONS)
            body_len = random.randint(5, 8)
            interior_len = body_len - 2
            divider_positions = _choose_dividers(interior_len)
            k = len(divider_positions)

            divider_colors = random.sample(PALETTE, k)
            remaining = [c for c in PALETTE if c not in divider_colors]
            target_color = random.choice(remaining)
            gap = random.choice([1, 2])

            arrow_patch, axis = _build_arrow(body_len, orientation, divider_positions, divider_colors)
            ah, aw = arrow_patch.shape

            top = random.randint(0, h - ah)
            left = random.randint(0, w - aw)
            arrow_cells = [(top + r, left + c) for r in range(ah) for c in range(aw) if int(arrow_patch[r, c]) != BG]
            target_cells = _target_cells(top, left, arrow_patch.shape, orientation, gap, interior_len)
            all_cells = arrow_cells + target_cells

            if not _fits(occ, all_cells, margin=1):
                continue

            _stamp(grid, top, left, arrow_patch)
            for r, c in target_cells:
                grid[r, c] = target_color
            _paint_occ(occ, all_cells, margin=1)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
