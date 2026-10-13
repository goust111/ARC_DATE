from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


BG = 0
MID = 1
TOP = 2
BOT = 3
PATH = 4
START = 5
DIVIDER = 6

LAYER_COLORS = (TOP, MID, BOT)
AREAS = (3, 4, 5, 6, 7)

AREA_TO_TEMPLATE = {
    3: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
    4: np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=int),
    5: np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=int),
    7: np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=int),
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    idx = sorted(map(int, indices))
    if not idx:
        return []
    out: list[tuple[int, int]] = []
    start = prev = idx[0]
    for val in idx[1:]:
        if val == prev + 1:
            prev = val
            continue
        out.append((start, prev))
        start = prev = val
    out.append((start, prev))
    return out


def _find_divider_col(grid: np.ndarray) -> int:
    for c in range(int(grid.shape[1])):
        col = grid[:, c]
        if np.all(col == DIVIDER):
            return int(c)
    raise AssertionError("expected one full-height divider column")


def _segment_cells(anchor: tuple[int, int], area: int) -> list[tuple[int, int]]:
    r, c = map(int, anchor)
    sr, sc = r + 1, c
    if area == 3:
        return [(sr, sc + dc) for dc in range(2)]
    if area == 4:
        return [(sr + dr, sc) for dr in range(2)]
    if area == 5:
        return [(sr, sc - dc) for dc in range(3)]
    if area == 6:
        return [(sr - dr, sc) for dr in range(3)]
    if area == 7:
        return [(sr, sc + dc) for dc in range(4)]
    raise AssertionError(f"unsupported area: {area}")


def _trace_path(shape: tuple[int, int], start: tuple[int, int], areas: list[int]) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), BG, dtype=int)
    sr, sc = map(int, start)
    out[sr, sc] = START
    anchor = (sr, sc)
    for area in areas:
        cells = _segment_cells(anchor, int(area))
        for r, c in cells:
            assert 0 <= r < h and 0 <= c < w, "path exits canvas"
            if (r, c) != (sr, sc):
                out[r, c] = PATH
        anchor = cells[-1]
    out[sr, sc] = START
    return out


def _overlay_left(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h % 3 == 0, "stacked panel height must be divisible by 3"
    div_c = _find_divider_col(g)
    panel_h = h // 3
    left = g[:, :div_c]
    top = left[:panel_h]
    middle = left[panel_h : 2 * panel_h]
    bottom = left[2 * panel_h :]
    composite = middle.copy()
    composite[top != BG] = top[top != BG]
    composite[bottom != BG] = bottom[bottom != BG]
    return composite


def _decode_areas_from_composite(composite: np.ndarray) -> list[int]:
    fg_rows = np.where(np.any(composite != BG, axis=1))[0]
    fg_cols = np.where(np.any(composite != BG, axis=0))[0]
    row_bands = _runs(fg_rows)
    col_bands = _runs(fg_cols)
    assert row_bands and col_bands, "expected at least one glyph"
    assert all(r1 - r0 == 2 for r0, r1 in row_bands), "glyph rows must be 3 cells tall"
    assert all(c1 - c0 == 2 for c0, c1 in col_bands), "glyph cols must be 3 cells wide"

    areas: list[int] = []
    for c0, c1 in col_bands:
        for r0, r1 in row_bands:
            tile = composite[r0 : r1 + 1, c0 : c1 + 1]
            area = int(np.count_nonzero(tile != BG))
            if area:
                assert 3 <= area <= 7, "glyph area must map to an instruction"
                areas.append(area)
    assert areas, "expected at least one non-empty glyph tile"
    return areas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the three stacked panels, decode each restored 3x3 glyph by pixel area, and draw the resulting path on the start canvas.

    Input structure:
    - A full-height magenta divider splits the grid into a left instruction half and a right drawing half.
    - The left half is stacked into three equal-height black panels whose foreground fragments use red, blue, and green.
    - The right half is otherwise black and contains exactly one gray start pixel.

    Rule:
    - Composite the left panels with middle as base, top overwriting middle, and bottom overwriting both.
    - In the composite, read every non-empty 3x3 glyph tile in column-major order using the occupied 3-row and 3-column bands.
    - Decode each glyph by its non-background area: 3 -> right 2, 4 -> down 2, 5 -> left 3, 6 -> up 3, 7 -> right 4.
    - Starting from the gray marker on the right half, step down one row before every segment, draw the yellow segment, and keep the gray start pixel unchanged.

    Key insight:
    - No single panel contains the instruction by itself; only the three-layer overlay restores the full 3x3 occupancy pattern whose area encodes each movement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    div_c = _find_divider_col(g)
    composite = _overlay_left(g)
    areas = _decode_areas_from_composite(composite)

    right = g[:, div_c + 1 :]
    start_pts = np.argwhere(right == START)
    assert start_pts.shape[0] == 1, "expected exactly one gray start pixel"
    start = tuple(map(int, start_pts[0]))
    return _trace_path(right.shape, start, areas)


def _band_start_options(length: int, count: int) -> list[tuple[int, ...]]:
    starts = range(0, length - 2)
    out = []
    for combo in itertools.combinations(starts, count):
        if combo[-1] + 2 >= length:
            continue
        if all(combo[i + 1] >= combo[i] + 4 for i in range(count - 1)):
            out.append(tuple(map(int, combo)))
    return out


def _pick_path_sequence(
    h: int,
    w: int,
    n_steps: int,
    start: tuple[int, int],
    rng: random.Random,
) -> list[int] | None:
    required_groups = (
        {3, 7},  # rightward
        {5},     # leftward
        {4, 6},  # vertical
    )

    for _ in range(600):
        anchor = tuple(map(int, start))
        used_cells = {anchor}
        areas: list[int] = []
        ok = True

        for step in range(n_steps):
            valid: list[tuple[int, list[tuple[int, int]]]] = []
            for area in AREAS:
                cells = _segment_cells(anchor, area)
                if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                    continue
                if any((r, c) in used_cells for r, c in cells):
                    continue
                valid.append((int(area), cells))
            if not valid:
                ok = False
                break

            missing_groups = [group for group in required_groups if not any(a in group for a in areas)]
            valid.sort(
                key=lambda item: (
                    int(item[0] not in areas),
                    int(any(item[0] in group for group in missing_groups)),
                    rng.random(),
                ),
                reverse=True,
            )
            top_score = (
                int(valid[0][0] not in areas),
                int(any(valid[0][0] in group for group in missing_groups)),
            )
            best = [
                item
                for item in valid
                if (
                    int(item[0] not in areas),
                    int(any(item[0] in group for group in missing_groups)),
                )
                == top_score
            ]
            area, cells = rng.choice(best)
            areas.append(int(area))
            used_cells.update(cells)
            anchor = cells[-1]

        if not ok:
            continue
        if len(set(areas)) < 3:
            continue
        if not all(any(area in group for area in areas) for group in required_groups):
            continue
        if n_steps >= 6 and not ({6, 7} & set(areas)):
            continue
        return areas
    return None


def _assign_visible_layers(area: int, rng: random.Random) -> list[int]:
    if area <= 4:
        chosen = rng.sample(LAYER_COLORS, 2)
    else:
        chosen = list(LAYER_COLORS) if rng.random() < 0.65 else rng.sample(LAYER_COLORS, 2)
    rng.shuffle(chosen)
    return chosen


def _split_counts(total: int, parts: int, rng: random.Random) -> list[int]:
    cuts = sorted(rng.sample(range(1, total), parts - 1)) if parts > 1 else []
    prev = 0
    out = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _render_panels(
    panel_h: int,
    left_w: int,
    row_starts: tuple[int, int],
    col_starts: tuple[int, ...],
    areas: list[int],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    top = np.full((panel_h, left_w), BG, dtype=int)
    middle = np.full((panel_h, left_w), BG, dtype=int)
    bottom = np.full((panel_h, left_w), BG, dtype=int)
    visible_counts = Counter()
    missing_visible = set(LAYER_COLORS)

    idx = 0
    for c0 in col_starts:
        for r0 in row_starts:
            area = int(areas[idx])
            idx += 1
            template = AREA_TO_TEMPLATE[area]
            coords = [(r0 + r, c0 + c) for r, c in zip(*np.where(template == 1))]
            rng.shuffle(coords)
            visible_layers = _assign_visible_layers(area, rng)
            if missing_visible and not (missing_visible & set(visible_layers)):
                visible_layers[0] = rng.choice(sorted(missing_visible))
                rng.shuffle(visible_layers)
            counts = _split_counts(len(coords), len(visible_layers), rng)

            offset = 0
            for layer, count in zip(visible_layers, counts):
                for rr, cc in coords[offset : offset + count]:
                    if layer == TOP:
                        top[rr, cc] = TOP
                    elif layer == MID:
                        middle[rr, cc] = MID
                    else:
                        bottom[rr, cc] = BOT
                    visible_counts[layer] += 1
                offset += count
            missing_visible -= set(visible_layers)

            for rr, cc in coords:
                if bottom[rr, cc] == BOT:
                    if rng.random() < 0.55:
                        top[rr, cc] = TOP
                    if rng.random() < 0.55:
                        middle[rr, cc] = MID
                elif top[rr, cc] == TOP and rng.random() < 0.6:
                    middle[rr, cc] = MID

    if not all(visible_counts[color] > 0 for color in LAYER_COLORS):
        raise AssertionError("all three colors must be visible in the composite")

    composite = middle.copy()
    composite[top != BG] = top[top != BG]
    composite[bottom != BG] = bottom[bottom != BG]
    return top, middle, bottom, composite


def _build_input_from_plan(
    panel_h: int,
    left_w: int,
    right_w: int,
    row_starts: tuple[int, int],
    col_starts: tuple[int, ...],
    start: tuple[int, int],
    areas: list[int],
    rng: random.Random,
) -> np.ndarray:
    top, middle, bottom, _composite = _render_panels(panel_h, left_w, row_starts, col_starts, areas, rng)
    h = 3 * panel_h
    w = left_w + 1 + right_w
    grid = np.full((h, w), BG, dtype=int)
    grid[:panel_h, :left_w] = top
    grid[panel_h : 2 * panel_h, :left_w] = middle
    grid[2 * panel_h :, :left_w] = bottom
    grid[:, left_w] = DIVIDER
    sr, sc = map(int, start)
    grid[sr, left_w + 1 + sc] = START
    return grid


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    panel_h = rng.randint(7, 9)
    row_options = _band_start_options(panel_h, 2)
    if not row_options:
        return None
    row_starts = rng.choice(row_options)

    n_cols = rng.choice((2, 3))
    min_left_w = 7 if n_cols == 2 else 11
    left_w = rng.randint(min_left_w, min_left_w + 2)
    col_options = _band_start_options(left_w, n_cols)
    if not col_options:
        return None
    col_starts = rng.choice(col_options)

    right_w = rng.randint(7, 10)
    h = 3 * panel_h
    n_steps = 2 * n_cols

    start_row = rng.randint(2, h - 4)
    start_col = rng.randint(2, right_w - 4)
    start = (start_row, start_col)

    areas = _pick_path_sequence(h, right_w, n_steps, start, rng)
    if areas is None:
        return None

    inp = _build_input_from_plan(panel_h, left_w, right_w, row_starts, col_starts, start, areas, rng)

    try:
        composite = _overlay_left(inp)
        decoded = _decode_areas_from_composite(composite)
        if decoded != areas:
            return None
        expected = _trace_path((h, right_w), start, areas)
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(800):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
