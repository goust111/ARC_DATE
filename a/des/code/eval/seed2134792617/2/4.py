from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

INNER = 5
CELL = 7
GAP = 1
FRAME_MASK = np.ones((CELL, CELL), dtype=bool)
FRAME_MASK[1:-1, 1:-1] = False
FOUR_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (2, 5)
TOP_PAD_RANGE = (1, 3)
LEFT_PAD_RANGE = (1, 3)
BOTTOM_PAD_RANGE = (3, 5)
RIGHT_PAD_RANGE = (1, 3)
GEN_ATTEMPTS = 400
MASK_ATTEMPTS = 200

_CLASS_QUEUE: list[str] = []
_AXIS_QUEUE: list[str] = []


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in FOUR_DIRS:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.asarray(coords, dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    return coords.size > 0 and len(_components(mask)) == 1


def _has_horizontal_symmetry(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.flipud(mask)))


def _has_vertical_symmetry(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.fliplr(mask)))


def _kind_from_mask(mask: np.ndarray) -> str:
    hsym = _has_horizontal_symmetry(mask)
    vsym = _has_vertical_symmetry(mask)
    if hsym and vsym:
        return "dual"
    if hsym or vsym:
        return "single"
    return "none"


def _border_color(mask: np.ndarray) -> int:
    kind = _kind_from_mask(mask)
    if kind == "dual":
        return RED
    if kind == "single":
        return BLUE
    return GREEN


def _orbits_for(flips: tuple[str, ...]) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(INNER):
        for c in range(INNER):
            if (r, c) in seen:
                continue
            orbit = {(r, c)}
            frontier = [(r, c)]
            while frontier:
                cr, cc = frontier.pop()
                variants = {(cr, cc)}
                for flip in flips:
                    if flip == "h":
                        variants.add((INNER - 1 - cr, cc))
                    else:
                        variants.add((cr, INNER - 1 - cc))
                if "h" in flips and "v" in flips:
                    variants.add((INNER - 1 - cr, INNER - 1 - cc))
                for vr, vc in variants:
                    if (vr, vc) not in orbit:
                        orbit.add((int(vr), int(vc)))
                        frontier.append((int(vr), int(vc)))
            seen |= orbit
            orbits.append(sorted(orbit))
    return orbits


DUAL_ORBITS = _orbits_for(("h", "v"))
HORIZONTAL_ORBITS = _orbits_for(("h",))
VERTICAL_ORBITS = _orbits_for(("v",))


def _next_class() -> str:
    global _CLASS_QUEUE
    if not _CLASS_QUEUE:
        _CLASS_QUEUE = ["dual", "single", "none"]
        random.shuffle(_CLASS_QUEUE)
    return _CLASS_QUEUE.pop()


def _next_axis() -> str:
    global _AXIS_QUEUE
    if not _AXIS_QUEUE:
        _AXIS_QUEUE = ["h", "v"]
        random.shuffle(_AXIS_QUEUE)
    return _AXIS_QUEUE.pop()


def _balanced_kinds(count: int) -> list[str]:
    kinds = ["dual", "single", "none"]
    while len(kinds) < count:
        kinds.append(_next_class())
    random.shuffle(kinds)
    return kinds


def _random_symmetric_mask(orbits: list[list[tuple[int, int]]], want: str, axis: str | None = None) -> np.ndarray | None:
    for _ in range(MASK_ATTEMPTS):
        mask = np.zeros((INNER, INNER), dtype=bool)
        p = random.uniform(0.25, 0.65)
        shuffled = orbits[:]
        random.shuffle(shuffled)
        for orbit in shuffled:
            if random.random() >= p:
                continue
            for r, c in orbit:
                mask[r, c] = True

        count = int(mask.sum())
        if count < 2 or count > 15 or not _is_connected(mask):
            continue

        hsym = _has_horizontal_symmetry(mask)
        vsym = _has_vertical_symmetry(mask)
        if want == "dual" and hsym and vsym:
            return mask
        if want == "single":
            if axis == "h" and hsym and not vsym:
                return mask
            if axis == "v" and vsym and not hsym:
                return mask
    return None


def _random_asymmetric_mask() -> np.ndarray | None:
    for _ in range(MASK_ATTEMPTS):
        target = random.randint(2, 10)
        cells = {(random.randrange(INNER), random.randrange(INNER))}
        while len(cells) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in FOUR_DIRS:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < INNER and 0 <= nc < INNER and (nr, nc) not in cells:
                        frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(random.choice(sorted(frontier)))

        mask = np.zeros((INNER, INNER), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if int(mask.sum()) < 2 or not _is_connected(mask):
            continue
        if _has_horizontal_symmetry(mask) or _has_vertical_symmetry(mask):
            continue
        return mask
    return None


def _make_pattern(kind: str) -> np.ndarray | None:
    if kind == "dual":
        return _random_symmetric_mask(DUAL_ORBITS, "dual")
    if kind == "single":
        axis = _next_axis()
        orbits = HORIZONTAL_ORBITS if axis == "h" else VERTICAL_ORBITS
        return _random_symmetric_mask(orbits, "single", axis=axis)
    return _random_asymmetric_mask()


def _draw_frame(grid: np.ndarray, top: int, left: int, color: int) -> None:
    patch = grid[top : top + CELL, left : left + CELL]
    patch[FRAME_MASK] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify each 5x5 orange motif by axis symmetry and recolor its grey frame.

    Input structure:
    - Black background.
    - A regular array of independent grey 7x7 cell frames, each with a 5x5 interior.
    - Each interior contains one connected orange motif on black.
    - A 3x1 legend stack (red, blue, green) sits at the lower-right and stays unchanged.

    Rule:
    - If a motif has both horizontal and vertical symmetry, recolor that cell frame red.
    - If it has exactly one of the two symmetries, recolor that cell frame blue.
    - Otherwise recolor that cell frame green.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    out = g.copy()
    found = 0

    for comp in _components(g == GREY):
        if len(comp) != int(FRAME_MASK.sum()):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (CELL, CELL):
            continue

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if not bool(np.all(patch[FRAME_MASK] == GREY)):
            continue
        inner = patch[1:-1, 1:-1]
        if not bool(np.isin(inner, [BLACK, ORANGE]).all()):
            continue

        motif = inner == ORANGE
        if int(motif.sum()) < 2:
            continue

        patch_out = out[r0 : r1 + 1, c0 : c1 + 1]
        patch_out[FRAME_MASK] = _border_color(motif)
        found += 1

    assert found > 0, "expected at least one valid cell frame"
    return out


def _grid_shape(n_rows: int, n_cols: int, top_pad: int, left_pad: int, bottom_pad: int, right_pad: int) -> tuple[int, int]:
    cell_h = n_rows * CELL + (n_rows - 1) * GAP
    cell_w = n_cols * CELL + (n_cols - 1) * GAP
    height = 2 + top_pad + cell_h + bottom_pad
    width = 2 + left_pad + cell_w + right_pad
    return int(height), int(width)


def _cell_positions(n_rows: int, n_cols: int, top_pad: int, left_pad: int) -> list[tuple[int, int]]:
    start_r = 1 + top_pad
    start_c = 1 + left_pad
    step = CELL + GAP
    return [
        (int(start_r + r * step), int(start_c + c * step))
        for r in range(n_rows)
        for c in range(n_cols)
    ]


def _validate_instance(grid: np.ndarray, positions: list[tuple[int, int]], kinds: list[str]) -> bool:
    solved = main(grid)

    legend = solved[-4:-1, -2]
    if not bool(np.array_equal(legend, np.array([RED, BLUE, GREEN], dtype=int))):
        return False

    expected_colors = {"dual": RED, "single": BLUE, "none": GREEN}
    seen_kinds: set[str] = set()

    for (top, left), expected_kind in zip(positions, kinds):
        original = grid[top + 1 : top + 1 + INNER, left + 1 : left + 1 + INNER]
        solved_patch = solved[top : top + CELL, left : left + CELL]
        motif = original == ORANGE
        actual_kind = _kind_from_mask(motif)
        if actual_kind != expected_kind:
            return False
        if not bool(np.array_equal(solved_patch[1:-1, 1:-1], original)):
            return False
        if not bool(np.all(solved_patch[FRAME_MASK] == expected_colors[expected_kind])):
            return False
        seen_kinds.add(actual_kind)

    return seen_kinds == {"dual", "single", "none"}


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(*GRID_ROWS_RANGE)
    n_cols = random.randint(*GRID_COLS_RANGE)
    top_pad = random.randint(*TOP_PAD_RANGE)
    left_pad = random.randint(*LEFT_PAD_RANGE)
    bottom_pad = random.randint(*BOTTOM_PAD_RANGE)
    right_pad = random.randint(*RIGHT_PAD_RANGE)

    height, width = _grid_shape(n_rows, n_cols, top_pad, left_pad, bottom_pad, right_pad)
    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY
    grid[-4:-1, -2] = np.array([RED, BLUE, GREEN], dtype=int)

    positions = _cell_positions(n_rows, n_cols, top_pad, left_pad)
    kinds = _balanced_kinds(len(positions))

    for (top, left), kind in zip(positions, kinds):
        pattern = _make_pattern(kind)
        if pattern is None or _kind_from_mask(pattern) != kind:
            return None
        _draw_frame(grid, top, left, GREY)
        inner = grid[top + 1 : top + 1 + INNER, left + 1 : left + 1 + INNER]
        inner[pattern] = ORANGE

    if not _validate_instance(grid, positions, kinds):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated retries")
