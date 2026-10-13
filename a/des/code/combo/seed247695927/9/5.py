from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
ORANGE = 7

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GRID_SIDE_RANGE = (18, 28)
BODY_MARGIN_RANGE = (3, 5)
BODY_TRIM_RANGE = (0, 3)
SMALL_DEFECT_COUNT_RANGE = (1, 3)
LARGE_DEFECT_COUNT_RANGE = (1, 3)
SPIKE_COUNT_RANGE = (0, 2)
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
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
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _blue_count_8(mask: np.ndarray, r: int, c: int) -> int:
    total = 0
    for dr, dc in NEIGHBORS8:
        nr, nc = r + dr, c + dc
        if _in_bounds(mask.shape, nr, nc) and mask[nr, nc]:
            total += 1
    return total


def _has_blue_2x2(mask: np.ndarray, r: int, c: int) -> bool:
    for dr in (-1, 0):
        for dc in (-1, 0):
            cells = [(r + dr + rr, c + dc + cc) for rr in (0, 1) for cc in (0, 1)]
            if all(_in_bounds(mask.shape, rr, cc) and mask[rr, cc] for rr, cc in cells):
                return True
    return False


def _absorb_blue_spikes(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    while True:
        blue = out == BLUE
        remove: list[tuple[int, int]] = []
        for r, c in np.argwhere(blue):
            rr = int(r)
            cc = int(c)
            if _has_blue_2x2(blue, rr, cc):
                continue
            if _blue_count_8(blue, rr, cc) <= 2:
                remove.append((rr, cc))
        if not remove:
            return out
        for r, c in remove:
            out[r, c] = BLACK


def _leaf_pixels(comp: list[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    comp_set = set(comp)
    leaves: list[tuple[int, int]] = []
    for r, c in comp:
        degree = 0
        for dr, dc in NEIGHBORS8:
            if (r + dr, c + dc) in comp_set:
                degree += 1
        if degree <= 1:
            leaves.append((r, c))
    return leaves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Highlight black defects inside a blue foreground by haloing them in orange.

    Rule:
    - Absorb thin blue spikes that are mostly surrounded by black.
    - Every black 4-connected component of size <= 3 is a defect.
    - For larger black components, only 8-neighbor leaf pixels are defects.
    - Defect pixels stay black and their 8-neighborhood becomes orange.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(np.unique(g)).issubset({BLACK, BLUE}), "input must use only black and blue"

    cleaned = _absorb_blue_spikes(g)
    defects: set[tuple[int, int]] = set()
    for comp in _components(cleaned == BLACK):
        if len(comp) <= 3:
            defects.update(comp)
        else:
            defects.update(_leaf_pixels(comp, cleaned.shape))

    assert defects, "expected at least one black defect"

    out = cleaned.copy()
    for r, c in defects:
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if _in_bounds(out.shape, nr, nc) and (nr, nc) not in defects:
                out[nr, nc] = ORANGE
    for r, c in defects:
        out[r, c] = BLACK
    return out


def _carve_rect(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top : top + height, left : left + width] = color


def _paint_path(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    color: int,
) -> list[tuple[int, int]]:
    r, c = start
    pts: list[tuple[int, int]] = []
    for _ in range(length):
        if not _in_bounds(grid.shape, r, c):
            return []
        pts.append((r, c))
        r += direction[0]
        c += direction[1]
    for rr, cc in pts:
        grid[rr, cc] = color
    return pts


def _body_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _make_body(h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    top = random.randint(*BODY_MARGIN_RANGE)
    left = random.randint(*BODY_MARGIN_RANGE)
    bottom = h - 1 - random.randint(*BODY_MARGIN_RANGE)
    right = w - 1 - random.randint(*BODY_MARGIN_RANGE)
    if bottom - top < 9:
        bottom = top + 9
    if right - left < 9:
        right = left + 9
    grid[top : bottom + 1, left : right + 1] = BLUE

    trim = random.randint(*BODY_TRIM_RANGE)
    for _ in range(trim):
        side = random.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            row = top if side == "top" else bottom
            seg_w = random.randint(2, max(2, (right - left + 1) // 3))
            seg_c = random.randint(left, right - seg_w + 1)
            grid[row, seg_c : seg_c + seg_w] = BLACK
        else:
            col = left if side == "left" else right
            seg_h = random.randint(2, max(2, (bottom - top + 1) // 3))
            seg_r = random.randint(top, bottom - seg_h + 1)
            grid[seg_r : seg_r + seg_h, col] = BLACK
    return grid


def _place_small_defect(grid: np.ndarray) -> bool:
    blue = grid == BLUE
    r0, c0, r1, c1 = _body_bbox(blue)
    shapes = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (0, 2)],
    ]
    shape = random.choice(shapes)
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    for _ in range(PLACEMENT_ATTEMPTS):
        base_r = random.randint(r0 + 2, r1 - max_r - 2)
        base_c = random.randint(c0 + 2, c1 - max_c - 2)
        pts = [(base_r + dr, base_c + dc) for dr, dc in shape]
        halo = [(r + dr, c + dc) for r, c in pts for dr, dc in NEIGHBORS8 + [(0, 0)]]
        if any(not _in_bounds(grid.shape, r, c) or grid[r, c] != BLUE for r, c in pts):
            continue
        if any(
            _in_bounds(grid.shape, r, c) and grid[r, c] == BLACK and (r, c) not in pts
            for r, c in halo
        ):
            continue
        for r, c in pts:
            grid[r, c] = BLACK
        return True
    return False


def _place_internal_large_defect(grid: np.ndarray) -> bool:
    blue = grid == BLUE
    r0, c0, r1, c1 = _body_bbox(blue)
    chamber_h = random.randint(2, 3)
    chamber_w = random.randint(2, 3)
    for _ in range(PLACEMENT_ATTEMPTS):
        top = random.randint(r0 + 2, r1 - chamber_h - 3)
        left = random.randint(c0 + 2, c1 - chamber_w - 3)
        chamber = [(top + dr, left + dc) for dr in range(chamber_h) for dc in range(chamber_w)]
        if any(grid[r, c] != BLUE for r, c in chamber):
            continue
        branch_side = random.choice(["top", "bottom", "left", "right"])
        length = random.randint(2, 4)
        if branch_side == "top":
            start = (top - 1, left + chamber_w // 2)
            direction = (-1, 0)
        elif branch_side == "bottom":
            start = (top + chamber_h, left + chamber_w // 2)
            direction = (1, 0)
        elif branch_side == "left":
            start = (top + chamber_h // 2, left - 1)
            direction = (0, -1)
        else:
            start = (top + chamber_h // 2, left + chamber_w)
            direction = (0, 1)
        branch = [(start[0] + i * direction[0], start[1] + i * direction[1]) for i in range(length)]
        pts = chamber + branch
        if any(not _in_bounds(grid.shape, r, c) or grid[r, c] != BLUE for r, c in pts):
            continue
        if any(r <= r0 or r >= r1 or c <= c0 or c >= c1 for r, c in branch):
            continue
        halo = [(r + dr, c + dc) for r, c in pts for dr, dc in NEIGHBORS8 + [(0, 0)]]
        if any(
            _in_bounds(grid.shape, r, c) and grid[r, c] == BLACK and (r, c) not in pts
            for r, c in halo
        ):
            continue
        for r, c in pts:
            grid[r, c] = BLACK
        return True
    return False


def _place_external_large_defect(grid: np.ndarray) -> bool:
    blue = grid == BLUE
    r0, c0, r1, c1 = _body_bbox(blue)
    options = [
        ("top", (r0, None), (1, 0), lambda: random.randint(c0 + 2, c1 - 2), lambda: random.randint(4, min(7, r1 - r0 - 2))),
        ("bottom", (r1, None), (-1, 0), lambda: random.randint(c0 + 2, c1 - 2), lambda: random.randint(4, min(7, r1 - r0 - 2))),
        ("left", (None, c0), (0, 1), lambda: random.randint(r0 + 2, r1 - 2), lambda: random.randint(4, min(7, c1 - c0 - 2))),
        ("right", (None, c1), (0, -1), lambda: random.randint(r0 + 2, r1 - 2), lambda: random.randint(4, min(7, c1 - c0 - 2))),
    ]
    random.shuffle(options)
    for _name, anchor, direction, pos_fn, len_fn in options:
        try:
            length = len_fn()
        except ValueError:
            continue
        pos = pos_fn()
        if anchor[0] is None:
            start = (pos, anchor[1])
        else:
            start = (anchor[0], pos)
        pts = [(start[0] + i * direction[0], start[1] + i * direction[1]) for i in range(length)]
        if any(not _in_bounds(grid.shape, r, c) or grid[r, c] != BLUE for r, c in pts):
            continue
        for r, c in pts:
            grid[r, c] = BLACK
        return True
    return False


def _place_blue_spike(grid: np.ndarray) -> bool:
    blue = grid == BLUE
    r0, c0, r1, c1 = _body_bbox(blue)
    options = [
        ((r0 - 1, None), (-1, 0), lambda: random.randint(c0 + 2, c1 - 2), lambda: random.randint(2, max(2, r0 - 1))),
        ((r1 + 1, None), (1, 0), lambda: random.randint(c0 + 2, c1 - 2), lambda: random.randint(2, max(2, grid.shape[0] - r1 - 2))),
        ((None, c0 - 1), (0, -1), lambda: random.randint(r0 + 2, r1 - 2), lambda: random.randint(2, max(2, c0 - 1))),
        ((None, c1 + 1), (0, 1), lambda: random.randint(r0 + 2, r1 - 2), lambda: random.randint(2, max(2, grid.shape[1] - c1 - 2))),
    ]
    random.shuffle(options)
    for anchor, direction, pos_fn, len_fn in options:
        try:
            length = len_fn()
        except ValueError:
            continue
        pos = pos_fn()
        if anchor[0] is None:
            pts = [(pos, anchor[1] + i * direction[1]) for i in range(length)]
            join = (pos, anchor[1] - direction[1])
        else:
            pts = [(anchor[0] + i * direction[0], pos) for i in range(length)]
            join = (anchor[0] - direction[0], pos)
        if not _in_bounds(grid.shape, *join) or grid[join] != BLUE:
            continue
        if any(not _in_bounds(grid.shape, r, c) or grid[r, c] != BLACK for r, c in pts):
            continue
        if any(abs(r - rr) <= 1 and abs(c - cc) <= 1 and grid[rr, cc] == BLACK for r, c in pts for rr in range(max(0, r - 1), min(grid.shape[0], r + 2)) for cc in range(max(0, c - 1), min(grid.shape[1], c + 2)) if (rr, cc) not in pts):
            continue
        for r, c in pts:
            grid[r, c] = BLUE
        return True
    return False


def _attempt_grid() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    grid = _make_body(h, w)

    for _ in range(random.randint(*SMALL_DEFECT_COUNT_RANGE)):
        if not _place_small_defect(grid):
            return None

    large_count = random.randint(*LARGE_DEFECT_COUNT_RANGE)
    branch_modes = [random.choice(["internal", "external"]) for _ in range(large_count)]
    if "internal" not in branch_modes:
        branch_modes[0] = "internal"
    if "external" not in branch_modes and large_count > 1:
        branch_modes[1] = "external"
    for mode in branch_modes:
        ok = _place_internal_large_defect(grid) if mode == "internal" else _place_external_large_defect(grid)
        if not ok:
            return None

    spike_count = random.randint(*SPIKE_COUNT_RANGE)
    for _ in range(spike_count):
        _place_blue_spike(grid)

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if ORANGE not in out:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
