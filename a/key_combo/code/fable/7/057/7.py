from __future__ import annotations

from collections import Counter
from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_7_057_7_rework",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()
Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components


BG = Color.BLACK
GREY = Color.GREY
PALETTE = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]


FAMILIES = (
    {
        "lengths": (3, 4),
        "superset": (2, 3, 4),
        "motifs": (
            ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (2, 1), (3, 1)),
            ((0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0)),
            ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)),
            ((0, 0), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (1, 4)),
        ),
    },
    {
        "lengths": (2, 5),
        "superset": (2, 3, 5),
        "motifs": (
            ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 2)),
            ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1)),
            ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 3), (1, 4)),
            ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (2, 0)),
        ),
    },
    {
        "lengths": (2, 3),
        "superset": (2, 3, 4),
        "motifs": (
            ((0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0)),
            ((0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 4)),
            ((0, 0), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (1, 4)),
            ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (1, 3), (1, 4)),
        ),
    },
)


def _row_segments(row: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    c = 0
    while c < row.size:
        color = int(row[c])
        if color == bg:
            c += 1
            continue
        c0 = c
        while c + 1 < row.size and int(row[c + 1]) == color:
            c += 1
        segments.append((c0, c, color))
        c += 1
    return segments


def _separator_row(grid: np.ndarray) -> int:
    assert grid.ndim == 2 and grid.shape[0] >= 3
    sep_r = int(grid.shape[0] - 2)
    sep_color = int(grid[sep_r, 0])
    assert bool(np.all(grid[sep_r] == sep_color)), "separator row must be uniform"
    return sep_r


def _parse_dictionary(grid: np.ndarray) -> tuple[int, int, dict[int, int]]:
    sep_r = _separator_row(grid)
    work = np.asarray(grid[:sep_r], dtype=int)
    bg = int(Counter(map(int, work.ravel())).most_common(1)[0][0])
    dict_row = np.asarray(grid[sep_r + 1], dtype=int)
    segments = _row_segments(dict_row, bg)
    assert len(segments) >= 2, "dictionary row must contain multiple separated segments"

    sep_color = int(grid[sep_r, 0])
    len_to_color: dict[int, int] = {}
    seen_colors: set[int] = set()
    for c0, c1, color in segments:
        length = int(c1 - c0 + 1)
        assert color != bg and color != GREY and color != sep_color
        assert length not in len_to_color, "dictionary lengths must be distinct"
        assert color not in seen_colors, "dictionary colors must be distinct"
        len_to_color[length] = color
        seen_colors.add(color)
    return sep_r, bg, len_to_color


def _component_mask(component: dict) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = map(int, component["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _cells_to_bits(cells: tuple[tuple[int, int], ...], width: int) -> int:
    bits = 0
    for r, c in cells:
        bits |= 1 << (int(r) * width + int(c))
    return int(bits)


def _candidate_strips(mask: np.ndarray, lengths: tuple[int, ...]) -> tuple[list[dict], list[list[int]], list[int]]:
    h, w = map(int, mask.shape)
    occupied = [int(r * w + c) for r, c in np.argwhere(mask)]
    by_cell: list[list[int]] = [[] for _ in range(h * w)]
    candidates: list[dict] = []

    for r in range(h):
        for c0 in range(w):
            if not bool(mask[r, c0]):
                continue
            for length in lengths:
                c1 = c0 + length
                if c1 <= w and bool(mask[r, c0:c1].all()):
                    cells = tuple((r, c) for c in range(c0, c1))
                    idx = len(candidates)
                    candidates.append(
                        {
                            "bits": _cells_to_bits(cells, w),
                            "length": int(length),
                            "orientation": "h",
                            "cells": cells,
                        }
                    )
                    for _r, _c in cells:
                        by_cell[_r * w + _c].append(idx)

    for c in range(w):
        for r0 in range(h):
            if not bool(mask[r0, c]):
                continue
            for length in lengths:
                r1 = r0 + length
                if r1 <= h and bool(mask[r0:r1, c].all()):
                    cells = tuple((r, c) for r in range(r0, r1))
                    idx = len(candidates)
                    candidates.append(
                        {
                            "bits": _cells_to_bits(cells, w),
                            "length": int(length),
                            "orientation": "v",
                            "cells": cells,
                        }
                    )
                    for _r, _c in cells:
                        by_cell[_r * w + _c].append(idx)

    return candidates, by_cell, occupied


def _decompose_mask(mask: np.ndarray, lengths: tuple[int, ...]) -> tuple[int, list[dict]]:
    candidates, by_cell, occupied = _candidate_strips(mask, lengths)
    full_state = 0
    for idx in occupied:
        full_state |= 1 << idx
        assert by_cell[idx], "every grey cell must belong to at least one allowed strip"

    memo: dict[int, tuple[int, tuple[int, ...]]] = {}

    def solve(state: int) -> tuple[int, tuple[int, ...]]:
        if state == 0:
            return 1, ()
        if state in memo:
            return memo[state]

        pivot_options: list[int] | None = None
        for idx in occupied:
            if not (state >> idx) & 1:
                continue
            options = [k for k in by_cell[idx] if (candidates[k]["bits"] & state) == candidates[k]["bits"]]
            if not options:
                memo[state] = (0, ())
                return memo[state]
            if pivot_options is None or len(options) < len(pivot_options):
                pivot_options = options
                if len(options) == 1:
                    break

        assert pivot_options is not None
        count = 0
        best: tuple[int, ...] = ()
        for k in pivot_options:
            subcount, subsolution = solve(state ^ candidates[k]["bits"])
            if subcount == 0:
                continue
            if count == 0:
                best = (k,) + subsolution
            count += subcount
            if count > 1:
                memo[state] = (2, best)
                return memo[state]

        memo[state] = (count, best)
        return memo[state]

    count, ids = solve(full_state)
    return int(count), [candidates[k] for k in ids]


def _paint_max_runs(grid: np.ndarray, len_to_color: dict[int, int], axis: str) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    sep_r, bg, _ = _parse_dictionary(grid)
    work = out[:sep_r]
    if axis == "h":
        for r in range(work.shape[0]):
            c = 0
            while c < work.shape[1]:
                if int(work[r, c]) != GREY:
                    c += 1
                    continue
                c0 = c
                while c + 1 < work.shape[1] and int(work[r, c + 1]) == GREY:
                    c += 1
                length = int(c - c0 + 1)
                if length in len_to_color:
                    work[r, c0 : c + 1] = int(len_to_color[length])
                c += 1
        return out

    for c in range(work.shape[1]):
        r = 0
        while r < work.shape[0]:
            if int(work[r, c]) != GREY:
                r += 1
                continue
            r0 = r
            while r + 1 < work.shape[0] and int(work[r + 1, c]) == GREY:
                r += 1
            length = int(r - r0 + 1)
            if length in len_to_color:
                work[r0 : r + 1, c] = int(len_to_color[length])
            r += 1
    assert bg == BG or bg != GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose grey composite blocks by the strip-length dictionary at the bottom and recolor each strip by its dictionary color.

    Input structure:
    - A full-width separator row splits the grid into an upper grey work area and a bottom dictionary row.
    - The dictionary row contains separated colored horizontal segments with distinct lengths and distinct colors.
    - The work area contains one or more grey 4-connected composite blocks on a plain background.

    Rule:
    - Read the dictionary so each segment length maps to that segment's color.
    - For each grey block, find the unique partition into one-cell-wide horizontal or vertical strips whose lengths are in the dictionary.
    - Recolor every strip by the color for its length, while leaving the separator row and dictionary row unchanged.

    Key insight:
    - The bottom row is a visible length dictionary: it does not just choose output colors, it also determines which strip lengths are legal inside each grey block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_r, bg, len_to_color = _parse_dictionary(g)
    work = g[:sep_r]
    assert np.all((work == bg) | (work == GREY)), "work area must contain only background and grey cells"

    components = find_components(work, bg=bg, connectivity=4, monochromatic=True, colors=[GREY])
    assert components, "expected at least one grey component"

    out = g.copy()
    lengths = tuple(sorted(len_to_color))
    for component in components:
        mask, (r0, c0, _r1, _c1) = _component_mask(component)
        count, strips = _decompose_mask(mask, lengths)
        assert count == 1, "each component must have a unique strip decomposition"
        for strip in strips:
            color = int(len_to_color[strip["length"]])
            for rr, cc in strip["cells"]:
                out[r0 + rr, c0 + cc] = color
    return out


def _patch_from_cells(cells: tuple[tuple[int, int], ...]) -> np.ndarray:
    max_r = 1 + max(r for r, _ in cells)
    max_c = 1 + max(c for _, c in cells)
    patch = np.full((max_r, max_c), BG, dtype=int)
    for r, c in cells:
        patch[int(r), int(c)] = GREY
    return patch


def _transform_patch(patch: np.ndarray, code: int) -> np.ndarray:
    out = np.asarray(patch, dtype=int)
    if code >= 4:
        out = np.fliplr(out)
    return np.rot90(out, int(code % 4)).copy()


def _choose_family() -> dict:
    return random.choice(FAMILIES)


def _dictionary_row(width: int, lengths: tuple[int, ...], colors: tuple[int, ...]) -> np.ndarray:
    order = list(range(len(lengths)))
    random.shuffle(order)
    ordered_lengths = [int(lengths[i]) for i in order]
    ordered_colors = [int(colors[i]) for i in order]

    base = sum(ordered_lengths) + (len(ordered_lengths) - 1)
    assert base <= width
    extra = width - base
    slots = [0] * (len(ordered_lengths) + 1)
    for _ in range(extra):
        slots[random.randrange(len(slots))] += 1

    row = np.full(width, BG, dtype=int)
    c = slots[0]
    for i, (length, color) in enumerate(zip(ordered_lengths, ordered_colors)):
        row[c : c + length] = color
        c += length
        if i + 1 < len(ordered_lengths):
            c += 1 + slots[i + 1]
    return row


def _solve_patch(patch: np.ndarray, lengths: tuple[int, ...]) -> tuple[int, list[dict]]:
    return _decompose_mask(np.asarray(patch == GREY, dtype=bool), lengths)


def _layout_positions(patches: list[np.ndarray]) -> tuple[int, int, list[tuple[int, int]]]:
    n = len(patches)
    split = n if n <= 2 or random.random() < 0.5 else random.randint(1, n - 1)
    rows = [patches[:split], patches[split:]]

    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    row_gap = random.randint(1, 3)

    positions: list[tuple[int, int]] = []
    row_widths: list[int] = []
    row_heights: list[int] = []
    row_positions: list[list[tuple[int, int]]] = []

    for row_patches in rows:
        if not row_patches:
            row_widths.append(0)
            row_heights.append(0)
            row_positions.append([])
            continue
        left_margin = random.randint(1, 4)
        right_margin = random.randint(1, 4)
        gaps = [random.randint(2, 4) for _ in range(max(0, len(row_patches) - 1))]
        x = left_margin
        pos_row: list[tuple[int, int]] = []
        row_height = max(int(p.shape[0]) for p in row_patches)
        for i, patch in enumerate(row_patches):
            slack = row_height - int(patch.shape[0])
            y = random.randint(0, slack)
            pos_row.append((y, x))
            x += int(patch.shape[1])
            if i < len(gaps):
                x += gaps[i]
        row_widths.append(x + right_margin)
        row_heights.append(row_height)
        row_positions.append(pos_row)

    work_w = max(row_widths) if row_widths else 0
    work_h = top_margin + sum(h for h in row_heights if h > 0)
    if row_heights[0] > 0 and row_heights[1] > 0:
        work_h += row_gap
    work_h += bottom_margin

    y_cursor = top_margin
    flat_index = 0
    for row_idx, row_patches in enumerate(rows):
        if not row_patches:
            continue
        for local_idx, patch in enumerate(row_patches):
            dy, x = row_positions[row_idx][local_idx]
            positions.append((y_cursor + dy, x))
            flat_index += 1
        y_cursor += row_heights[row_idx]
        if row_idx == 0 and row_heights[1] > 0:
            y_cursor += row_gap

    assert len(positions) == len(patches)
    return int(work_h), int(work_w), positions


def _place_patches(patches: list[np.ndarray], positions: list[tuple[int, int]], work_h: int, work_w: int) -> np.ndarray:
    grid = np.full((work_h, work_w), BG, dtype=int)
    for patch, (r0, c0) in zip(patches, positions):
        ph, pw = map(int, patch.shape)
        sub = grid[r0 : r0 + ph, c0 : c0 + pw]
        mask = patch == GREY
        assert not bool(sub[mask].any()), "patches must not overlap"
        sub[mask] = GREY
    return grid


def _semantic_checks(grid: np.ndarray, family: dict, patches: list[np.ndarray]) -> None:
    out = main(grid)
    sep_r, bg, len_to_color = _parse_dictionary(grid)
    lengths = tuple(sorted(len_to_color))
    assert tuple(sorted(family["lengths"])) == lengths
    assert np.array_equal(out[sep_r:], grid[sep_r:]), "separator and dictionary rows must stay unchanged"
    assert out.shape == grid.shape
    assert bg == BG

    used_lengths: set[int] = set()
    used_orientations: set[str] = set()
    has_dictionary_sensitive = False
    for patch in patches:
        count, strips = _solve_patch(patch, lengths)
        assert count == 1
        used_lengths.update(int(strip["length"]) for strip in strips)
        used_orientations.update(str(strip["orientation"]) for strip in strips)
        extra_count, _ = _solve_patch(patch, tuple(sorted(family["superset"])))
        has_dictionary_sensitive |= extra_count > 1

    assert used_lengths == set(lengths), "every dictionary length must appear in the work area"
    assert used_orientations == {"h", "v"}, "sample set must require both strip orientations"
    assert has_dictionary_sensitive, "at least one component must rely on the exact dictionary lengths"
    assert not np.array_equal(_paint_max_runs(grid, len_to_color, "h"), out)
    assert not np.array_equal(_paint_max_runs(grid, len_to_color, "v"), out)


def generate_input() -> np.ndarray:
    family = _choose_family()
    lengths = tuple(sorted(int(x) for x in family["lengths"]))

    separator_color, *dict_colors = random.sample(PALETTE, len(lengths) + 1)
    dict_colors_t = tuple(int(c) for c in dict_colors)

    motif_cells = [family["motifs"][0]]
    extra_count = random.randint(1, 2)
    motif_cells.extend(random.choice(family["motifs"]) for _ in range(extra_count))

    patches: list[np.ndarray] = []
    for cells in motif_cells:
        patch = _patch_from_cells(cells)
        patch = _transform_patch(patch, random.randrange(8))
        patches.append(patch)

    work_h, work_w, positions = _layout_positions(patches)
    dict_min_w = sum(lengths) + len(lengths) - 1
    extra_w = random.randint(2, 4)
    work_w = max(work_w, dict_min_w + extra_w)
    assert work_h + 2 <= 30 and work_w <= 30

    work = _place_patches(patches, positions, work_h, work_w)
    grid = np.full((work_h + 2, work_w), BG, dtype=int)
    grid[:work_h] = work
    grid[work_h] = int(separator_color)
    grid[work_h + 1] = _dictionary_row(work_w, lengths, dict_colors_t)

    _semantic_checks(grid, family, patches)
    return grid
