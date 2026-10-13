import random
from collections import deque

import numpy as np


PANEL_SIZE = 5
GRID_SHAPE = (5, 17)
PANEL_STARTS = (0, 6, 12)
SEPARATOR_COLS = (5, 11)
BG = 8
SEP = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
GEN_ATTEMPTS = 400
LOCAL_ATTEMPTS = 200


def _mode_color(values: np.ndarray) -> int:
    counts = np.bincount(values.astype(int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _shift_score(dr: int, dc: int) -> tuple[int, int, int, int, int]:
    return (abs(dr) + abs(dc), max(abs(dr), abs(dc)), dr * dr + dc * dc, dr, dc)


def _nearest_non_overlapping_shift(coords: np.ndarray, occupied: np.ndarray) -> tuple[int, int]:
    h, w = occupied.shape
    best = None
    best_shift = None
    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            rr = coords[:, 0] + dr
            cc = coords[:, 1] + dc
            if rr.min() < 0 or rr.max() >= h or cc.min() < 0 or cc.max() >= w:
                continue
            if np.any(occupied[rr, cc]):
                continue
            score = _shift_score(dr, dc)
            if best is None or score < best:
                best = score
                best_shift = (dr, dc)
    if best_shift is None:
        raise AssertionError("no valid non-overlapping shift exists")
    return best_shift


def _normalize(coords: set[tuple[int, int]]) -> np.ndarray:
    arr = np.array(sorted(coords), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    return arr


def _random_connected_shape(rng: random.Random, size: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < size:
        r, c = rng.choice(tuple(cells))
        dr, dc = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        cells.add((r + dr, c + dc))
    shape = _normalize(cells)
    if shape[:, 0].max() >= PANEL_SIZE or shape[:, 1].max() >= PANEL_SIZE:
        return _random_connected_shape(rng, size)
    return shape


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([dr, dc], dtype=int)


def _can_place(coords: np.ndarray, forbidden: np.ndarray | None = None) -> bool:
    if coords[:, 0].min() < 0 or coords[:, 0].max() >= PANEL_SIZE:
        return False
    if coords[:, 1].min() < 0 or coords[:, 1].max() >= PANEL_SIZE:
        return False
    if forbidden is None:
        return True
    return not np.any(forbidden[coords[:, 0], coords[:, 1]])


def _touches_4n(a: np.ndarray, bset: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)):
            if (int(r + dr), int(c + dc)) in bset:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE, "expected a 5x17 grid"
    assert all(np.all(g[:, c] == SEP) for c in SEPARATOR_COLS), "separator columns must be black"

    panel_pixels = np.concatenate([g[:, 0:5].ravel(), g[:, 6:11].ravel(), g[:, 12:17].ravel()])
    bg = _mode_color(panel_pixels)

    out = np.full((PANEL_SIZE, PANEL_SIZE), bg, dtype=int)
    occupied = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)

    for start in PANEL_STARTS:
        panel = g[:, start : start + PANEL_SIZE]
        colors = [int(c) for c in np.unique(panel) if int(c) != bg]
        assert len(colors) == 1, "each panel must have exactly one foreground color"
        color = colors[0]

        comps = _components_of_color(panel, color)
        assert len(comps) == 2, "each panel must contain exactly two components"
        sizes = [len(comp) for comp in comps]
        largest = max(sizes)
        assert sizes.count(largest) == 1, "largest component must be unique"

        sprite = comps[sizes.index(largest)]
        dr, dc = _nearest_non_overlapping_shift(sprite, occupied)
        shifted = _translate(sprite, dr, dc)
        out[shifted[:, 0], shifted[:, 1]] = color
        occupied[shifted[:, 0], shifted[:, 1]] = True

    return out


def _generate_panel_pair(
    rng: random.Random,
    occupied: np.ndarray,
    require_shift: bool,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(LOCAL_ATTEMPTS):
        size = rng.randint(3, 6)
        shape = _random_connected_shape(rng, size)
        max_r = PANEL_SIZE - 1 - int(shape[:, 0].max())
        max_c = PANEL_SIZE - 1 - int(shape[:, 1].max())
        src = _translate(shape, rng.randint(0, max_r), rng.randint(0, max_c))
        try:
            shift = _nearest_non_overlapping_shift(src, occupied)
        except AssertionError:
            continue
        if require_shift and shift == (0, 0):
            continue
        if not require_shift and shift != (0, 0):
            continue
        dst = _translate(src, shift[0], shift[1])
        return src, dst
    raise RuntimeError("failed to generate a panel sprite")


def _generate_distractor(
    rng: random.Random,
    main_coords: np.ndarray,
) -> np.ndarray:
    main_set = {tuple(map(int, xy)) for xy in main_coords}
    max_size = min(3, len(main_coords) - 1)
    for _ in range(LOCAL_ATTEMPTS):
        size = rng.randint(1, max_size)
        shape = _random_connected_shape(rng, size)
        max_r = PANEL_SIZE - 1 - int(shape[:, 0].max())
        max_c = PANEL_SIZE - 1 - int(shape[:, 1].max())
        placed = _translate(shape, rng.randint(0, max_r), rng.randint(0, max_c))
        placed_set = {tuple(map(int, xy)) for xy in placed}
        if _touches_4n(placed, main_set):
            continue
        if placed_set & main_set:
            continue
        return placed
    raise RuntimeError("failed to place distractor")


def _assemble_panel(color: int, major: np.ndarray, minor: np.ndarray) -> np.ndarray:
    panel = np.full((PANEL_SIZE, PANEL_SIZE), BG, dtype=int)
    panel[major[:, 0], major[:, 1]] = color
    panel[minor[:, 0], minor[:, 1]] = color
    comps = _components_of_color(panel, color)
    assert len(comps) == 2
    sizes = sorted(len(comp) for comp in comps)
    assert sizes[0] < sizes[1]
    return panel


def generate_input() -> np.ndarray:
    rng = random.Random()
    shift_pattern = rng.choice(
        [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]
    )

    for _ in range(GEN_ATTEMPTS):
        colors = rng.sample(PALETTE, 3)
        panels = []
        occupied = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
        ok = True
        for idx in range(3):
            require_shift = idx > 0 and shift_pattern[idx - 1]
            try:
                src, dst = _generate_panel_pair(rng, occupied, require_shift)
                distractor = _generate_distractor(rng, src)
            except RuntimeError:
                ok = False
                break
            panels.append(_assemble_panel(colors[idx], src, distractor))
            occupied[dst[:, 0], dst[:, 1]] = True
        if not ok:
            continue

        grid = np.full(GRID_SHAPE, BG, dtype=int)
        grid[:, SEPARATOR_COLS[0]] = SEP
        grid[:, SEPARATOR_COLS[1]] = SEP
        for start, panel in zip(PANEL_STARTS, panels):
            grid[:, start : start + PANEL_SIZE] = panel

        out = main(grid)
        if out.shape != (5, 5):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
