from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np


BG = 0
TAB_COLOR = 7
NOTCH_COLOR = 8
SPECIAL_COLORS = {TAB_COLOR, NOTCH_COLOR}
BODY_COLOR_CHOICES = [1, 2, 3, 4, 5, 6, 9]

PUZZLE_LAYOUTS = [(2, 2), (2, 3), (3, 2), (3, 3)]
CELL_SIZE_OPTIONS = [5, 6]
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _get_bg(grid: np.ndarray) -> int:
    vals = np.asarray(grid, dtype=int).reshape(-1)
    return int(np.bincount(vals).argmax())


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _strict_mode(values: list[int]) -> int:
    counts = Counter(map(int, values))
    best = max(counts.values())
    modes = [value for value, count in counts.items() if int(count) == int(best)]
    assert len(modes) == 1, "expected a unique mode"
    return int(modes[0])


def _is_flat(profile: tuple[int, ...]) -> bool:
    return all(int(v) == 0 for v in profile)


def _complement(profile: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(-v) for v in profile)


def _resolve_marker_color(cells: dict[tuple[int, int], int], coord: tuple[int, int], fallback: int) -> int:
    r, c = coord
    neighbor_colors = [
        int(cells[(nr, nc)])
        for nr, nc in _neighbors4(r, c)
        if (nr, nc) in cells and int(cells[(nr, nc)]) not in SPECIAL_COLORS
    ]
    if not neighbor_colors:
        return int(fallback)
    counts = Counter(neighbor_colors)
    best = max(counts.values())
    tied = [value for value, count in counts.items() if int(count) == int(best)]
    return int(min(tied))


def _analyze_piece(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))

    local_cells: dict[tuple[int, int], int] = {}
    row_to_cols: dict[int, list[int]] = {}
    col_to_rows: dict[int, list[int]] = {}
    piece_colors: list[int] = []

    for r, c in coords:
        lr = int(r) - int(r0)
        lc = int(c) - int(c0)
        value = int(grid[int(r), int(c)])
        local_cells[(lr, lc)] = value
        row_to_cols.setdefault(int(lr), []).append(int(lc))
        col_to_rows.setdefault(int(lc), []).append(int(lr))
        if value not in SPECIAL_COLORS:
            piece_colors.append(value)

    assert piece_colors, "piece must contain at least one body pixel"
    body_color = _strict_mode(piece_colors)

    base_left = _strict_mode([min(cols) for cols in row_to_cols.values()])
    base_right = _strict_mode([max(cols) for cols in row_to_cols.values()])
    base_top = _strict_mode([min(rows) for rows in col_to_rows.values()])
    base_bottom = _strict_mode([max(rows) for rows in col_to_rows.values()])

    base_h = int(base_bottom - base_top + 1)
    base_w = int(base_right - base_left + 1)
    assert base_h >= 3 and base_w >= 3, "base tile must be at least 3x3"

    norm_cells = {
        (int(lr - base_top), int(lc - base_left)): int(value)
        for (lr, lc), value in local_cells.items()
    }

    left_profile: list[int] = []
    right_profile: list[int] = []
    for rr in range(base_h):
        cols = [cc for (r, cc) in norm_cells if int(r) == int(rr)]
        assert cols, "each base row must contain piece pixels"
        left_profile.append(int(-min(cols)))
        right_profile.append(int(max(cols) - (base_w - 1)))

    top_profile: list[int] = []
    bottom_profile: list[int] = []
    for cc in range(base_w):
        rows = [rr for (rr, c) in norm_cells if int(c) == int(cc)]
        assert rows, "each base column must contain piece pixels"
        top_profile.append(int(-min(rows)))
        bottom_profile.append(int(max(rows) - (base_h - 1)))

    profiles = {
        "top": tuple(top_profile),
        "right": tuple(right_profile),
        "bottom": tuple(bottom_profile),
        "left": tuple(left_profile),
    }
    for profile in profiles.values():
        assert all(int(v) in (-1, 0, 1) for v in profile), "unexpected edge displacement"

    return {
        "body_color": int(body_color),
        "base_h": int(base_h),
        "base_w": int(base_w),
        "cells": norm_cells,
        "profiles": profiles,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered tab-and-notch puzzle fragments into their seamless rectangular picture.

    Input structure:
    - The background is black.
    - The foreground is split into separated 4-connected puzzle pieces that keep a fixed upright orientation.
    - Each piece is a rectangle-like tile whose sides may contain one-cell-deep orange tabs or one-cell-deep azure notches.
    - The remaining non-marker pixels are the pieces' body colors.

    Rule:
    - Separate the pieces and recover, for each one, its flat base rectangle and its four side profiles.
    - Treat all-flat sides as the outer border of the final picture.
    - Match non-flat right/left profiles and bottom/top profiles only when one side is the exact cellwise complement of the other.
    - Place the matched pieces on the implied rectangular lattice and repaint every orange/azure marker pixel with the neighboring body color.

    Key insight:
    - The special edge colors do not encode piece order directly; they encode complementary boundary profiles, so the rectangle is recovered by reconstructing the pieces' adjacency graph.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    comps = _components4(g != bg)
    assert comps, "expected at least one puzzle piece"

    pieces = [_analyze_piece(g, comp) for comp in comps]
    base_heights = {int(piece["base_h"]) for piece in pieces}
    base_widths = {int(piece["base_w"]) for piece in pieces}
    assert len(base_heights) == 1 and len(base_widths) == 1, "all pieces must share one base size"
    base_h = int(next(iter(base_heights)))
    base_w = int(next(iter(base_widths)))

    opposite = {"top": "bottom", "right": "left", "bottom": "top", "left": "right"}
    deltas = {"top": (-1, 0), "right": (0, 1), "bottom": (1, 0), "left": (0, -1)}
    neighbors: list[dict[str, int | None]] = []

    for i, piece in enumerate(pieces):
        piece_neighbors: dict[str, int | None] = {}
        for side, profile in piece["profiles"].items():
            if _is_flat(profile):
                piece_neighbors[side] = None
                continue
            target_side = opposite[side]
            wanted = _complement(profile)
            matches = [
                j
                for j, other in enumerate(pieces)
                if j != i and tuple(other["profiles"][target_side]) == wanted
            ]
            assert len(matches) == 1, "each non-flat side must have exactly one complement"
            piece_neighbors[side] = int(matches[0])
        neighbors.append(piece_neighbors)

    for i, piece_neighbors in enumerate(neighbors):
        for side, j in piece_neighbors.items():
            if j is None:
                continue
            opposite_side = opposite[side]
            assert neighbors[int(j)][opposite_side] == i, "side matches must be symmetric"

    top_left_candidates = [
        i
        for i, piece in enumerate(pieces)
        if _is_flat(piece["profiles"]["top"]) and _is_flat(piece["profiles"]["left"])
    ]
    assert len(top_left_candidates) == 1, "expected exactly one top-left corner piece"

    positions: dict[int, tuple[int, int]] = {int(top_left_candidates[0]): (0, 0)}
    q = deque([int(top_left_candidates[0])])
    while q:
        i = int(q.popleft())
        r, c = positions[i]
        for side, j in neighbors[i].items():
            if j is None:
                continue
            dr, dc = deltas[side]
            dest = (int(r + dr), int(c + dc))
            j = int(j)
            if j in positions:
                assert positions[j] == dest, "piece placement must be consistent"
                continue
            positions[j] = dest
            q.append(j)

    assert len(positions) == len(pieces), "all pieces must be connected by matches"

    min_r = min(r for r, _ in positions.values())
    min_c = min(c for _, c in positions.values())
    norm_positions = {i: (int(r - min_r), int(c - min_c)) for i, (r, c) in positions.items()}

    max_r = max(r for r, _ in norm_positions.values())
    max_c = max(c for _, c in norm_positions.values())
    out_rows = int(max_r + 1)
    out_cols = int(max_c + 1)
    occupied_slots = {(int(r), int(c)) for r, c in norm_positions.values()}
    assert len(occupied_slots) == len(pieces) == out_rows * out_cols, "pieces must fill a full rectangle"

    for i, piece in enumerate(pieces):
        pr, pc = norm_positions[i]
        border_expectations = {
            "top": pr == 0,
            "bottom": pr == out_rows - 1,
            "left": pc == 0,
            "right": pc == out_cols - 1,
        }
        for side, should_be_flat in border_expectations.items():
            assert _is_flat(piece["profiles"][side]) == bool(should_be_flat), "flat sides must land on the border"

    out = np.full((out_rows * base_h, out_cols * base_w), -1, dtype=int)
    for i, piece in enumerate(pieces):
        pr, pc = norm_positions[i]
        origin_r = int(pr * base_h)
        origin_c = int(pc * base_w)
        for coord, value in piece["cells"].items():
            rr, cc = coord
            gr = int(origin_r + rr)
            gc = int(origin_c + cc)
            assert 0 <= gr < out.shape[0] and 0 <= gc < out.shape[1], "piece pixel left rectangle bounds"
            paint = (
                _resolve_marker_color(piece["cells"], coord, int(piece["body_color"]))
                if int(value) in SPECIAL_COLORS
                else int(value)
            )
            assert out[gr, gc] == -1, "assembled pieces must not overlap"
            out[gr, gc] = int(paint)

    assert np.all(out != -1), "assembled rectangle must have no gaps"
    return out


def _feature_patterns(length: int) -> list[tuple[int, ...]]:
    internal = list(range(1, length - 1))
    patterns: list[tuple[int, ...]] = []
    for size in (1, 2):
        for combo in combinations(internal, size):
            patterns.append(tuple(sorted(int(x) for x in combo)))
    return patterns


def _make_piece_cells(cell_h: int, cell_w: int, body_color: int, spec: dict[str, tuple[str, tuple[int, ...]]]) -> dict[tuple[int, int], int]:
    cells = {(r, c): int(body_color) for r in range(cell_h) for c in range(cell_w)}

    def set_notch_wall(coord: tuple[int, int]) -> None:
        if coord in cells:
            cells[coord] = NOTCH_COLOR

    kind, pattern = spec["left"]
    if kind == "tab":
        for r in pattern:
            cells[(int(r), -1)] = TAB_COLOR
    elif kind == "notch":
        for r in pattern:
            cells.pop((int(r), 0), None)
            set_notch_wall((int(r), 1))

    kind, pattern = spec["right"]
    if kind == "tab":
        for r in pattern:
            cells[(int(r), cell_w)] = TAB_COLOR
    elif kind == "notch":
        for r in pattern:
            cells.pop((int(r), cell_w - 1), None)
            set_notch_wall((int(r), cell_w - 2))

    kind, pattern = spec["top"]
    if kind == "tab":
        for c in pattern:
            cells[(-1, int(c))] = TAB_COLOR
    elif kind == "notch":
        for c in pattern:
            cells.pop((0, int(c)), None)
            set_notch_wall((1, int(c)))

    kind, pattern = spec["bottom"]
    if kind == "tab":
        for c in pattern:
            cells[(cell_h, int(c))] = TAB_COLOR
    elif kind == "notch":
        for c in pattern:
            cells.pop((cell_h - 1, int(c)), None)
            set_notch_wall((cell_h - 2, int(c)))

    return cells


def _render_piece_patch(cells: dict[tuple[int, int], int]) -> np.ndarray:
    rows = [int(r) for r, _ in cells]
    cols = [int(c) for _, c in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
    for (r, c), value in cells.items():
        patch[int(r - min_r), int(c - min_c)] = int(value)
    return patch


def _render_output(piece_grid: list[list[dict]], cell_h: int, cell_w: int) -> np.ndarray:
    rows = len(piece_grid)
    cols = len(piece_grid[0])
    out = np.full((rows * cell_h, cols * cell_w), -1, dtype=int)
    for pr in range(rows):
        for pc in range(cols):
            piece = piece_grid[pr][pc]
            origin_r = int(pr * cell_h)
            origin_c = int(pc * cell_w)
            for (rr, cc), value in piece["cells"].items():
                gr = int(origin_r + rr)
                gc = int(origin_c + cc)
                assert 0 <= gr < out.shape[0] and 0 <= gc < out.shape[1]
                paint = int(piece["body_color"]) if int(value) in SPECIAL_COLORS else int(value)
                assert out[gr, gc] == -1
                out[gr, gc] = paint
    assert np.all(out != -1)
    return out


def _slot_layout_options(piece_count: int) -> list[tuple[int, int]]:
    options = [(2, 2), (2, 3), (3, 2), (3, 3)]
    return [shape for shape in options if int(shape[0] * shape[1]) >= int(piece_count)]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    puzzle_rows, puzzle_cols = rng.choice(PUZZLE_LAYOUTS)
    cell_h = rng.choice(CELL_SIZE_OPTIONS)
    cell_w = rng.choice(CELL_SIZE_OPTIONS)

    vertical_needed = int(puzzle_rows * (puzzle_cols - 1))
    horizontal_needed = int((puzzle_rows - 1) * puzzle_cols)
    vertical_patterns = _feature_patterns(cell_h)
    horizontal_patterns = _feature_patterns(cell_w)
    if len(vertical_patterns) < vertical_needed or len(horizontal_patterns) < horizontal_needed:
        return None

    rng.shuffle(vertical_patterns)
    rng.shuffle(horizontal_patterns)
    vertical_patterns = vertical_patterns[:vertical_needed]
    horizontal_patterns = horizontal_patterns[:horizontal_needed]

    piece_count = int(puzzle_rows * puzzle_cols)
    palette_size = rng.randint(2, min(4, len(BODY_COLOR_CHOICES)))
    palette = rng.sample(BODY_COLOR_CHOICES, palette_size)
    body_colors = [palette[i % palette_size] for i in range(piece_count)]
    rng.shuffle(body_colors)

    piece_grid: list[list[dict]] = [
        [
            {
                "body_color": int(body_colors[r * puzzle_cols + c]),
                "spec": {
                    "top": ("flat", ()),
                    "right": ("flat", ()),
                    "bottom": ("flat", ()),
                    "left": ("flat", ()),
                },
            }
            for c in range(puzzle_cols)
        ]
        for r in range(puzzle_rows)
    ]

    v_idx = 0
    for r in range(puzzle_rows):
        for c in range(puzzle_cols - 1):
            pattern = tuple(vertical_patterns[v_idx])
            v_idx += 1
            if rng.random() < 0.5:
                piece_grid[r][c]["spec"]["right"] = ("tab", pattern)
                piece_grid[r][c + 1]["spec"]["left"] = ("notch", pattern)
            else:
                piece_grid[r][c]["spec"]["right"] = ("notch", pattern)
                piece_grid[r][c + 1]["spec"]["left"] = ("tab", pattern)

    h_idx = 0
    for r in range(puzzle_rows - 1):
        for c in range(puzzle_cols):
            pattern = tuple(horizontal_patterns[h_idx])
            h_idx += 1
            if rng.random() < 0.5:
                piece_grid[r][c]["spec"]["bottom"] = ("tab", pattern)
                piece_grid[r + 1][c]["spec"]["top"] = ("notch", pattern)
            else:
                piece_grid[r][c]["spec"]["bottom"] = ("notch", pattern)
                piece_grid[r + 1][c]["spec"]["top"] = ("tab", pattern)

    piece_payloads: list[dict] = []
    for r in range(puzzle_rows):
        for c in range(puzzle_cols):
            piece = piece_grid[r][c]
            cells = _make_piece_cells(cell_h, cell_w, int(piece["body_color"]), piece["spec"])
            patch = _render_piece_patch(cells)
            piece["cells"] = cells
            piece["patch"] = patch
            piece_payloads.append(piece)

    expected = _render_output(piece_grid, cell_h, cell_w)

    slot_rows, slot_cols = rng.choice(_slot_layout_options(piece_count))
    slot_h = int(cell_h + 4)
    slot_w = int(cell_w + 4)
    grid_h = int(slot_rows * slot_h)
    grid_w = int(slot_cols * slot_w)
    if max(grid_h, grid_w) > 30:
        return None

    slots = [(r, c) for r in range(slot_rows) for c in range(slot_cols)]
    rng.shuffle(slots)
    chosen_slots = slots[:piece_count]
    rng.shuffle(piece_payloads)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for piece, (slot_r, slot_c) in zip(piece_payloads, chosen_slots):
        patch = piece["patch"]
        ph, pw = map(int, patch.shape)
        max_local_r = int(slot_h - ph - 1)
        max_local_c = int(slot_w - pw - 1)
        if max_local_r < 1 or max_local_c < 1:
            return None
        local_r = rng.randint(1, max_local_r)
        local_c = rng.randint(1, max_local_c)
        top = int(slot_r * slot_h + local_r)
        left = int(slot_c * slot_w + local_c)
        window = grid[top : top + ph, left : left + pw]
        if np.any((patch != BG) & (window != BG)):
            return None
        window[patch != BG] = patch[patch != BG]

    if len(_components4(grid != BG)) != piece_count:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
