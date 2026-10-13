from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


BG = 0
LIGHT = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
_RNG = random.Random()
_HOLE_SEQUENCE = (0, 1, 2, 3)
_HOLE_CURSOR = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _trim(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _count_holes(mask: np.ndarray) -> int:
    bg_parts = _components(~mask)
    h, w = mask.shape
    holes = 0
    for coords in bg_parts:
        if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in coords):
            holes += 1
    return holes


def _component_hole_count(coords: list[tuple[int, int]]) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return _count_holes(mask)


def _most_common_color(grid: np.ndarray) -> int:
    return Counter(map(int, grid.ravel())).most_common(1)[0][0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Light the right-side signal cells according to each object's enclosed-hole count.

    Input structure:
    - The most common color is the black background.
    - Several distinct-color 4-connected objects sit on the left; each object may enclose 0 to 3 background holes.
    - On the right, one 1x3 gray signal bar corresponds to each object, ordered from top to bottom.

    Rule:
    - Split the non-background pixels into connected components.
    - Treat every 1x3 component as a signal bar; treat every other component as a colored object.
    - Sort objects and signal bars by their topmost row. For each object, count its enclosed background holes and recolor the leftmost N cells of the matching signal bar with the object's color.

    Key insight:
    - The relevant feature is the object's topology: the number of enclosed background regions, not its size, color id, or outer silhouette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    objects: list[list[tuple[int, int]]] = []
    signal_rows: list[list[tuple[int, int]]] = []
    for coords in _components(g != bg):
        r0, c0, r1, c1 = _bbox(coords)
        if len(coords) == 3 and r0 == r1 and c1 - c0 == 2:
            signal_rows.append(coords)
        else:
            objects.append(coords)

    assert objects, "expected at least one object"
    assert len(objects) == len(signal_rows), "object and signal counts must match"

    objects.sort(key=lambda pts: (_bbox(pts)[0], _bbox(pts)[1]))
    signal_rows.sort(key=lambda pts: (_bbox(pts)[0], _bbox(pts)[1]))

    out = g.copy()
    for obj_coords, signal_coords in zip(objects, signal_rows):
        hole_count = _component_hole_count(obj_coords)
        color = int(g[obj_coords[0]])
        ordered_signal = sorted(signal_coords, key=lambda rc: rc[1])
        for idx in range(hole_count):
            r, c = ordered_signal[idx]
            out[r, c] = color
    return out


def _pairwise_nonadjacent(cells: tuple[tuple[int, int], ...]) -> bool:
    for (r1, c1), (r2, c2) in itertools.combinations(cells, 2):
        if abs(r1 - r2) + abs(c1 - c2) <= 1:
            return False
    return True


def _build_base_options() -> dict[int, list[tuple[int, int, tuple[tuple[int, int], ...]]]]:
    options: dict[int, list[tuple[int, int, tuple[tuple[int, int], ...]]]] = {k: [] for k in range(4)}
    for h in range(3, 7):
        for w in range(3, 8):
            area = h * w
            if 8 <= area <= 20:
                options[0].append((h, w, ()))
            interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
            for hole_count in range(1, 4):
                if not (8 <= area - hole_count <= 20):
                    continue
                for holes in itertools.combinations(interior, hole_count):
                    if _pairwise_nonadjacent(holes):
                        options[hole_count].append((h, w, holes))
    return options


BASE_OPTIONS = _build_base_options()


def _is_single_component(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r, c in map(tuple, np.argwhere(mask)):
        if any(not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc] for nr, nc in _neighbors4(r, c)):
            cells.append((r, c))
    return cells


def _maybe_add_bar(mask: np.ndarray, hole_count: int, rng: random.Random) -> np.ndarray:
    if int(mask.sum()) >= 20:
        return mask
    for _ in range(12):
        padded = np.pad(mask, 2, constant_values=False)
        coords = np.argwhere(padded)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        remaining = 20 - int(mask.sum())
        side = rng.choice(("top", "bottom", "left", "right"))
        trial = padded.copy()
        if side in ("top", "bottom"):
            span = c1 - c0 + 1
            length = rng.randint(1, min(2, span, remaining))
            start = rng.randint(c0, c1 - length + 1)
            row = r0 - 1 if side == "top" else r1 + 1
            trial[row, start : start + length] = True
        else:
            span = r1 - r0 + 1
            length = rng.randint(1, min(2, span, remaining))
            start = rng.randint(r0, r1 - length + 1)
            col = c0 - 1 if side == "left" else c1 + 1
            trial[start : start + length, col] = True
        trial = _trim(trial)
        if _is_single_component(trial) and _count_holes(trial) == hole_count and int(trial.sum()) <= 20:
            return trial
    return mask


def _maybe_remove_boundary_cell(mask: np.ndarray, hole_count: int, rng: random.Random) -> np.ndarray:
    if int(mask.sum()) <= 8:
        return mask
    cells = _boundary_cells(mask)
    rng.shuffle(cells)
    for r, c in cells:
        trial = mask.copy()
        trial[r, c] = False
        trial = _trim(trial)
        if int(trial.sum()) < 8:
            continue
        if _is_single_component(trial) and _count_holes(trial) == hole_count:
            return trial
    return mask


def _make_object_mask(hole_count: int, rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h, w, holes = rng.choice(BASE_OPTIONS[hole_count])
        mask = np.ones((h, w), dtype=bool)
        for r, c in holes:
            mask[r, c] = False

        for _ in range(rng.randint(0, 2)):
            mask = _maybe_add_bar(mask, hole_count, rng)
        for _ in range(rng.randint(0, 2)):
            mask = _maybe_remove_boundary_cell(mask, hole_count, rng)

        mask = np.rot90(mask, rng.randrange(4))
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        mask = _trim(mask)

        if 8 <= int(mask.sum()) <= 20 and _is_single_component(mask) and _count_holes(mask) == hole_count:
            return mask.astype(bool)
    raise RuntimeError(f"failed to build an object with {hole_count} holes")


def _planned_hole_counts(n_objects: int, rng: random.Random) -> list[int]:
    global _HOLE_CURSOR
    counts = [_HOLE_SEQUENCE[(_HOLE_CURSOR + i) % len(_HOLE_SEQUENCE)] for i in range(n_objects)]
    rng.shuffle(counts)
    return counts


def _commit_hole_counts(n_objects: int) -> None:
    global _HOLE_CURSOR
    _HOLE_CURSOR += n_objects


def _distribute_vertical_spacing(n_objects: int, heights: list[int], rng: random.Random) -> tuple[int, list[int], int] | None:
    total_height = sum(heights)
    min_spacing = n_objects + 1
    slack = 30 - total_height - min_spacing
    if slack < 0:
        return None

    top_margin = 1
    gaps = [1] * (n_objects - 1)
    bottom_margin = 1
    buckets = ["top", "bottom", *range(n_objects - 1)]
    while slack > 0:
        pick = rng.choice(buckets)
        if pick == "top":
            top_margin += 1
        elif pick == "bottom":
            bottom_margin += 1
        else:
            gaps[pick] += 1
        slack -= 1
    return top_margin, gaps, bottom_margin


def _render_case(
    masks: list[np.ndarray],
    colors: list[int],
    top_margin: int,
    gaps: list[int],
    bottom_margin: int,
    left_cols: list[int],
    signal_col: int,
    right_margin: int,
    signal_rows: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    row_starts: list[int] = []
    cur_r = top_margin
    for idx, mask in enumerate(masks):
        row_starts.append(cur_r)
        cur_r += mask.shape[0]
        if idx < len(gaps):
            cur_r += gaps[idx]
    height = cur_r + bottom_margin
    width = signal_col + 3 + right_margin

    grid = np.full((height, width), BG, dtype=int)
    expected = grid.copy()

    for mask, color, r0, c0, signal_r in zip(masks, colors, row_starts, left_cols, signal_rows):
        for dr, dc in map(tuple, np.argwhere(mask)):
            grid[r0 + dr, c0 + dc] = color
            expected[r0 + dr, c0 + dc] = color

        grid[signal_r, signal_col : signal_col + 3] = LIGHT
        expected[signal_r, signal_col : signal_col + 3] = LIGHT
        expected[signal_r, signal_col : signal_col + _count_holes(mask)] = color

    return grid, expected


def _gen_attempt(counts: list[int]) -> np.ndarray | None:
    masks = [_make_object_mask(hole_count, _RNG) for hole_count in counts]
    heights = [mask.shape[0] for mask in masks]
    spacing = _distribute_vertical_spacing(len(masks), heights, _RNG)
    if spacing is None:
        return None
    top_margin, gaps, bottom_margin = spacing

    colors = list(_RNG.sample(OBJECT_COLORS, len(masks)))
    left_cols = [_RNG.randint(1, 4) for _ in masks]
    max_right = max(c0 + mask.shape[1] for c0, mask in zip(left_cols, masks))
    signal_col = max_right + _RNG.randint(4, 7)
    right_margin = _RNG.randint(1, 3)

    row_starts: list[int] = []
    cur_r = top_margin
    for idx, mask in enumerate(masks):
        row_starts.append(cur_r)
        cur_r += mask.shape[0]
        if idx < len(gaps):
            cur_r += gaps[idx]

    signal_rows = [r0 + _RNG.randrange(mask.shape[0]) for r0, mask in zip(row_starts, masks)]
    if any(b - a < 2 for a, b in zip(signal_rows, signal_rows[1:])):
        return None

    grid, expected = _render_case(
        masks,
        colors,
        top_margin,
        gaps,
        bottom_margin,
        left_cols,
        signal_col,
        right_margin,
        signal_rows,
    )
    if max(grid.shape) > 30:
        return None
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        n_objects = _RNG.randint(2, 4)
        counts = _planned_hole_counts(n_objects, _RNG)
        grid = _gen_attempt(counts)
        if grid is not None:
            _commit_hole_counts(n_objects)
            return grid
    raise RuntimeError("failed to generate a valid input")
