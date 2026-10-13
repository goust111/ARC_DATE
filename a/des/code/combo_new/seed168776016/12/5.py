from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

STAMP_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE)
DIR_OFFSETS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TURNS_BY_OFFSET = {offset: turns for turns, offset in enumerate(DIR_OFFSETS)}


def _mask(rows: tuple[str, str, str]) -> np.ndarray:
    return np.array([[1 if ch == "1" else 0 for ch in row] for row in rows], dtype=int)


STAMP_MASK_LIBRARY = [
    _mask(("111", "010", "010")),
    _mask(("111", "011", "001")),
    _mask(("111", "110", "100")),
    _mask(("100", "111", "001")),
    _mask(("001", "111", "100")),
    _mask(("110", "010", "011")),
    _mask(("011", "010", "110")),
    _mask(("010", "111", "010")),
    _mask(("101", "111", "001")),
    _mask(("100", "111", "101")),
    _mask(("101", "111", "100")),
    _mask(("001", "111", "101")),
]


def _is_rotation_symmetric(mask: np.ndarray) -> bool:
    return any(np.array_equal(mask, np.rot90(mask, k)) for k in range(1, 4))


ASYMMETRIC_LIBRARY_INDICES = [
    idx for idx, mask in enumerate(STAMP_MASK_LIBRARY) if not _is_rotation_symmetric(mask)
]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rotate_clockwise(pattern: np.ndarray, quarter_turns: int) -> np.ndarray:
    turns = int(quarter_turns) % 4
    return np.rot90(pattern, -turns)


def _stamp(out: np.ndarray, pattern: np.ndarray, center_r: int, center_c: int) -> None:
    ph, pw = pattern.shape
    anchor_r = ph // 2
    anchor_c = pw // 2
    for pr in range(ph):
        for pc in range(pw):
            color = int(pattern[pr, pc])
            if color == BLACK:
                continue
            rr = center_r + pr - anchor_r
            cc = center_c + pc - anchor_c
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and int(out[rr, cc]) != GREY:
                out[rr, cc] = color


def _separator_row(g: np.ndarray) -> int:
    rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == MAROON)]
    assert len(rows) == 1, "expected exactly one full-width maroon separator row"
    return int(rows[0])


def _extract_stamps(top: np.ndarray) -> dict[int, np.ndarray]:
    stamps: dict[int, np.ndarray] = {}
    for comp in _components(top != BLACK):
        colors = {int(top[r, c]) for r, c in comp}
        assert len(colors) == 1, "each top stamp must be monochrome"
        color = colors.pop()
        r0, c0, r1, c1 = _bbox(comp)
        stamp = top[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert max(stamp.shape) <= 3, "stamps must be 3x3 or smaller"
        assert color not in stamps, "each stamp color must be unique"
        stamps[color] = stamp
    assert stamps, "expected at least one top stamp"
    return stamps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read color-keyed stamps from the upper half and print rotated copies onto the lower grey template.

    Input structure:
    - A full-width maroon row splits the grid into an upper stamp area and a lower template area.
    - Above the separator, black background contains several disconnected monochrome stamp shapes, each 3x3 or smaller and using a unique color.
    - Below the separator, black background contains one grey rectangular grid template made of one-pixel lines.
    - Some template cells contain a colored center pixel matching one upper stamp color, plus exactly one adjacent azure pixel above, right, below, or left of that center.

    Rule:
    - Match each lower colored center pixel to the upper stamp with the same color.
    - Decode rotation from the azure marker: above = 0 degrees, right = clockwise 90 degrees, below = 180 degrees, left = clockwise 270 degrees.
    - Remove the control markers, then stamp the rotated pattern onto the lower area centered on that colored pixel, painting only non-grey cells.
    - Discard the entire upper stamp area and keep only the transformed lower area as output.

    Key insight:
    - The colored center is a lookup key, the azure neighbor is a rotation code, and the grey template is a protected mask that the printed stamp must avoid.
    """
    g = np.asarray(grid, dtype=int)
    sep = _separator_row(g)
    top = g[:sep, :]
    bottom = g[sep + 1 :, :].copy()
    stamps = _extract_stamps(top)

    markers: list[tuple[int, int, int, int, int]] = []
    for r in range(bottom.shape[0]):
        for c in range(bottom.shape[1]):
            color = int(bottom[r, c])
            if color in {BLACK, GREY, AZURE}:
                continue
            assert color in stamps, "every lower marker color must reference an upper stamp"
            azure_neighbors = []
            for dr, dc in DIR_OFFSETS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < bottom.shape[0] and 0 <= nc < bottom.shape[1] and int(bottom[nr, nc]) == AZURE:
                    azure_neighbors.append((dr, dc, nr, nc))
            assert len(azure_neighbors) == 1, "each colored marker needs exactly one adjacent azure direction marker"
            dr, dc, ar, ac = azure_neighbors[0]
            markers.append((r, c, color, ar, ac))

    for r, c, _color, ar, ac in markers:
        bottom[r, c] = BLACK
        bottom[ar, ac] = BLACK

    for r, c, color, ar, ac in markers:
        turns = TURNS_BY_OFFSET[(ar - r, ac - c)]
        _stamp(bottom, _rotate_clockwise(stamps[color], turns), r, c)

    return bottom


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=int)
    out[mask.astype(bool)] = int(color)
    return out


def _draw_template(canvas: np.ndarray, top: int, left: int, cell_rows: int, cell_cols: int, inner: int) -> None:
    step = inner + 1
    height = cell_rows * step + 1
    width = cell_cols * step + 1
    for rr in range(0, height, step):
        canvas[top + rr, left : left + width] = GREY
    for cc in range(0, width, step):
        canvas[top : top + height, left + cc] = GREY


def _cell_center(top: int, left: int, cell_r: int, cell_c: int, inner: int) -> tuple[int, int]:
    step = inner + 1
    return (
        top + cell_r * step + 1 + inner // 2,
        left + cell_c * step + 1 + inner // 2,
    )


def _top_layout_width(count: int, gaps: list[int]) -> int:
    return 3 * count + sum(gaps)


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    top_height = random.randint(3, 4)
    inner = random.choice((3, 5))

    row_choices = [2, 3, 4] if inner == 5 else [2, 3, 4, 5]
    col_choices = [2, 3, 4] if inner == 5 else [2, 3, 4, 5]
    cell_rows = random.choice(row_choices)
    cell_cols = random.choice(col_choices)

    stamp_count = random.randint(2, min(4, cell_rows * cell_cols))
    template_height = cell_rows * (inner + 1) + 1
    template_width = cell_cols * (inner + 1) + 1

    lower_limit = 30 - top_height - 1
    assert template_height <= lower_limit
    lower_height = random.randint(template_height, min(lower_limit, template_height + 3))

    gaps = [random.randint(1, 2) for _ in range(max(0, stamp_count - 1))]
    stamps_width = _top_layout_width(stamp_count, gaps)
    width_floor = max(template_width, stamps_width)
    width = random.randint(width_floor, min(30, width_floor + 4))

    shape_indices = random.sample(range(len(STAMP_MASK_LIBRARY)), stamp_count)
    if not any(idx in ASYMMETRIC_LIBRARY_INDICES for idx in shape_indices):
        replace_at = random.randrange(stamp_count)
        current = set(shape_indices)
        choices = [idx for idx in ASYMMETRIC_LIBRARY_INDICES if idx not in current]
        if not choices:
            choices = list(ASYMMETRIC_LIBRARY_INDICES)
        shape_indices[replace_at] = random.choice(choices)

    colors = random.sample(STAMP_COLORS, stamp_count)
    color_to_mask = {color: STAMP_MASK_LIBRARY[idx] for color, idx in zip(colors, shape_indices)}
    asymmetric_colors = {
        color for color, idx in zip(colors, shape_indices) if idx in ASYMMETRIC_LIBRARY_INDICES
    }

    marker_count = random.randint(stamp_count, min(cell_rows * cell_cols, stamp_count + 3))
    all_cells = [(r, c) for r in range(cell_rows) for c in range(cell_cols)]
    chosen_cells = random.sample(all_cells, marker_count)

    marker_colors = list(colors)
    while len(marker_colors) < marker_count:
        marker_colors.append(random.choice(colors))
    random.shuffle(marker_colors)

    marker_turns = [random.randrange(4) for _ in range(marker_count)]
    if marker_count >= 4 and random.random() < 0.5:
        marker_turns[:4] = [0, 1, 2, 3]
        random.shuffle(marker_turns)
    elif marker_count > 1 and len(set(marker_turns)) == 1:
        idx = random.randrange(marker_count)
        marker_turns[idx] = (marker_turns[idx] + random.randint(1, 3)) % 4

    if asymmetric_colors:
        asym_indices = [i for i, color in enumerate(marker_colors) if color in asymmetric_colors]
        if asym_indices and all(marker_turns[i] == 0 for i in asym_indices):
            pick = random.choice(asym_indices)
            marker_turns[pick] = random.randint(1, 3)

    lower = np.full((lower_height, width), BLACK, dtype=int)
    template_top = random.randint(0, lower_height - template_height)
    template_left = random.randint(0, width - template_width)
    _draw_template(lower, template_top, template_left, cell_rows, cell_cols, inner)

    expected = lower.copy()
    lower_with_markers = lower.copy()
    marker_specs: list[tuple[int, int, int, int]] = []
    for (cell_r, cell_c), color, turns in zip(chosen_cells, marker_colors, marker_turns):
        rr, cc = _cell_center(template_top, template_left, cell_r, cell_c, inner)
        dr, dc = DIR_OFFSETS[turns]
        lower_with_markers[rr, cc] = color
        lower_with_markers[rr + dr, cc + dc] = AZURE
        pattern = _rotate_clockwise(_colorize(color_to_mask[color], color), turns)
        _stamp(expected, pattern, rr, cc)
        marker_specs.append((rr, cc, color, turns))

    top = np.full((top_height, width), BLACK, dtype=int)
    stamp_top = random.randint(0, top_height - 3)
    cursor = random.randint(0, width - stamps_width)
    top_order = list(colors)
    random.shuffle(top_order)
    for idx, color in enumerate(top_order):
        colored = _colorize(color_to_mask[color], color)
        top[stamp_top : stamp_top + 3, cursor : cursor + 3] = np.maximum(
            top[stamp_top : stamp_top + 3, cursor : cursor + 3], colored
        )
        if idx < len(gaps):
            cursor += 3 + gaps[idx]

    full = np.full((top_height + 1 + lower_height, width), BLACK, dtype=int)
    full[:top_height, :] = top
    full[top_height, :] = MAROON
    full[top_height + 1 :, :] = lower_with_markers

    assert len(_extract_stamps(top)) == stamp_count
    assert marker_specs
    assert np.array_equal(main(full), expected)
    return full, expected


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            grid, expected = _build_case()
        except AssertionError:
            continue
        if max(grid.shape) > 30 or max(expected.shape) > 30:
            continue
        return grid
    raise AssertionError("failed to generate a valid stamp-template puzzle")
