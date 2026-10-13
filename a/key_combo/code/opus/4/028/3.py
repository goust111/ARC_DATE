from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg


BG = Color.BLACK
DIVIDER = Color.GREY
BETWEEN = Color.PINK
MATCH = Color.ORANGE
LEGEND_SHAPE = (2, 2)
SLOT_SIZE = 3
CELL_STRIDE = SLOT_SIZE + 1
INPUT_COLOR_POOL = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.TEAL, Color.MAROON)
MAX_GEN_ATTEMPTS = 400


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in mask.astype(int))


def _generate_shape_catalog() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1, 1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=int).reshape(3, 3)
        if int(mask.sum()) > 5:
            continue
        if len(components_mask(mask.astype(bool), connectivity=4)) != 1:
            continue
        masks.append(mask.astype(bool))
    masks.sort(key=lambda m: (int(m.sum()), _mask_key(m)))
    return masks


SHAPE_CATALOG = _generate_shape_catalog()


def _parse_legend(grid: np.ndarray) -> tuple[list[tuple[int, int]], set[frozenset[int]], set[int]]:
    legend = np.asarray(grid[: LEGEND_SHAPE[0], : LEGEND_SHAPE[1]], dtype=int)
    assert legend.shape == LEGEND_SHAPE
    pairs: list[tuple[int, int]] = []
    colors: set[int] = set()
    for r in range(2):
        a, b = map(int, legend[r])
        assert a not in (BG, DIVIDER, BETWEEN, MATCH)
        assert b not in (BG, DIVIDER, BETWEEN, MATCH)
        assert a != b
        pairs.append((a, b))
        colors.update((a, b))
    assert len(colors) == 4
    pair_set = {frozenset(pair) for pair in pairs}
    assert len(pair_set) == 2
    return pairs, pair_set, colors


def _slot_spans(dividers: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for a, b in zip(dividers, dividers[1:]):
        if b - a <= 1:
            continue
        spans.append((a + 1, b))
    return spans


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BG
    _, legal_pairs, input_colors = _parse_legend(g)

    gray = np.argwhere(g == DIVIDER)
    assert gray.size > 0
    r0, c0 = gray.min(axis=0)
    r1, c1 = gray.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    row_divs = [r for r in range(r0, r1 + 1) if np.all(g[r, c0 : c1 + 1] == DIVIDER)]
    col_divs = [c for c in range(c0, c1 + 1) if np.all(g[r0 : r1 + 1, c] == DIVIDER)]
    row_spans = _slot_spans(row_divs)
    col_spans = _slot_spans(col_divs)
    assert len(row_spans) in (4, 5)
    assert len(col_spans) in (4, 5)
    assert all(re - rs == SLOT_SIZE for rs, re in row_spans)
    assert all(ce - cs == SLOT_SIZE for cs, ce in col_spans)

    slots: dict[tuple[int, int], dict] = {}
    shape_groups: dict[tuple[tuple[int, ...], ...], list[tuple[int, int]]] = defaultdict(list)
    for sr, (rs, re) in enumerate(row_spans):
        for sc, (cs, ce) in enumerate(col_spans):
            sub = g[rs:re, cs:ce]
            mask = sub != BG
            assert sub.shape == (SLOT_SIZE, SLOT_SIZE)
            assert mask.any()
            assert not np.any(sub == DIVIDER)
            colors = {int(v) for v in sub[mask]}
            assert len(colors) == 1
            color = colors.pop()
            assert color in input_colors
            assert len(components_mask(mask, connectivity=4)) == 1
            key = _mask_key(mask)
            slot = {
                "slot_rc": (sr, sc),
                "pixel_rc": (rs, cs),
                "mask": mask,
                "color": color,
                "shape": key,
            }
            slots[(sr, sc)] = slot
            shape_groups[key].append((sr, sc))

    valid_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    cross_row_illegal_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    matched_slots: set[tuple[int, int]] = set()
    for positions in shape_groups.values():
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i + 1 :]:
                c1 = slots[pos1]["color"]
                c2 = slots[pos2]["color"]
                if c1 == c2:
                    continue
                if frozenset((c1, c2)) in legal_pairs:
                    valid_pairs.append((pos1, pos2))
                    matched_slots.add(pos1)
                    matched_slots.add(pos2)
                else:
                    cross_row_illegal_pairs.append((pos1, pos2))

    row_matches: dict[int, list[int]] = defaultdict(list)
    col_matches: dict[int, list[int]] = defaultdict(list)
    for sr, sc in matched_slots:
        row_matches[sr].append(sc)
        col_matches[sc].append(sr)
    for cols in row_matches.values():
        cols.sort()
    for rows in col_matches.values():
        rows.sort()

    between_slots: set[tuple[int, int]] = set()
    row_intervals: list[tuple[int, int, int]] = []
    for sr, cols in row_matches.items():
        for left, right in zip(cols, cols[1:]):
            if right - left > 1:
                row_intervals.append((sr, left, right))
                for sc in range(left + 1, right):
                    between_slots.add((sr, sc))
    col_intervals: list[tuple[int, int, int]] = []
    for sc, rows in col_matches.items():
        for top, bottom in zip(rows, rows[1:]):
            if bottom - top > 1:
                col_intervals.append((sc, top, bottom))
                for sr in range(top + 1, bottom):
                    between_slots.add((sr, sc))

    return {
        "grid": g,
        "slots": slots,
        "shape_groups": shape_groups,
        "valid_pairs": valid_pairs,
        "cross_row_illegal_pairs": cross_row_illegal_pairs,
        "matched_slots": matched_slots,
        "between_slots": between_slots,
        "row_intervals": row_intervals,
        "col_intervals": col_intervals,
        "slot_rows": len(row_spans),
        "slot_cols": len(col_spans),
        "input_colors": input_colors,
    }


def _recolor_slot(out: np.ndarray, slot: dict, color: int) -> None:
    rs, cs = slot["pixel_rc"]
    sub = out[rs : rs + SLOT_SIZE, cs : cs + SLOT_SIZE]
    sub[slot["mask"]] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor same-shape slot pairs only when their colors form a legend-approved pair, then fill the slots between adjacent matches.

    Input structure:
    - A 2x2 color legend sits at the upper-left; each legend row contains two different colors.
    - To the right, gray divider lines split the main area into a 4x4, 4x5, 5x4, or 5x5 grid of 3x3 symbol slots.
    - Each slot contains one 4-connected monochrome symbol on black background, using one of the four legend colors.

    Rule:
    - Read each legend row as one legal two-color pairing.
    - Compare every pair of slots: they are an effective match iff their 3x3 symbol masks are exactly equal and their two colors are the two different colors from the same legend row.
    - Recolor every slot that participates in any effective match to orange, then for each slot-row and slot-column fill the slots strictly between adjacent matched slots with pink.

    Key insight:
    - The legend does not tell you how to recolor directly; it restricts which same-shape color combinations are allowed to count as matches.
    """
    analysis = _analyze_grid(grid)
    out = np.array(grid, copy=True)

    for pos in analysis["matched_slots"]:
        _recolor_slot(out, analysis["slots"][pos], MATCH)
    for pos in analysis["between_slots"]:
        _recolor_slot(out, analysis["slots"][pos], BETWEEN)
    return out


def _choose_gap_pair(rng: random.Random, values: list[int], forbidden: set[tuple[int, int]], fixed_axis: int, is_row: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for i, a in enumerate(values):
        for b in values[i + 1 :]:
            if b - a < 2:
                continue
            p1 = (fixed_axis, a) if is_row else (a, fixed_axis)
            p2 = (fixed_axis, b) if is_row else (b, fixed_axis)
            if p1 in forbidden or p2 in forbidden:
                continue
            candidates.append((p1, p2))
    assert candidates
    return rng.choice(candidates)


def _draw_main_grid(rows: int, cols: int) -> np.ndarray:
    h = rows * CELL_STRIDE + 1
    w = cols * CELL_STRIDE + 1
    main = np.zeros((h, w), dtype=int)
    main[::CELL_STRIDE, :] = DIVIDER
    main[:, ::CELL_STRIDE] = DIVIDER
    return main


def _slot_origin(slot_r: int, slot_c: int) -> tuple[int, int]:
    return slot_r * CELL_STRIDE + 1, slot_c * CELL_STRIDE + 1


def _place_symbol(grid: np.ndarray, slot_r: int, slot_c: int, mask: np.ndarray, color: int) -> None:
    rs, cs = _slot_origin(slot_r, slot_c)
    sub = grid[rs : rs + SLOT_SIZE, cs : cs + SLOT_SIZE]
    sub[:] = BG
    sub[mask] = int(color)


def _build_input(
    rows: int,
    cols: int,
    legend_pairs: list[tuple[int, int]],
    assignments: dict[tuple[int, int], tuple[np.ndarray, int]],
    *,
    bottom_pad: int,
    right_pad: int,
) -> np.ndarray:
    main = _draw_main_grid(rows, cols)
    for (sr, sc), (mask, color) in assignments.items():
        _place_symbol(main, sr, sc, mask, color)

    grid = np.zeros((main.shape[0] + bottom_pad, main.shape[1] + 3 + right_pad), dtype=int)
    grid[: main.shape[0], 3 : 3 + main.shape[1]] = main
    for r, pair in enumerate(legend_pairs):
        grid[r, 0] = int(pair[0])
        grid[r, 1] = int(pair[1])
    return grid


def _unused_shapes(rng: random.Random, needed: int) -> list[np.ndarray]:
    sample = rng.sample(SHAPE_CATALOG, needed)
    return [mask.copy() for mask in sample]


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if np.any(np.isin(grid, [BETWEEN, MATCH])):
        return False

    info = _analyze_grid(grid)
    if len(info["valid_pairs"]) < 2:
        return False
    if not info["row_intervals"] or not info["col_intervals"]:
        return False
    if not info["cross_row_illegal_pairs"]:
        return False
    if len(info["matched_slots"]) != 4:
        return False
    if len(info["between_slots"]) < 2:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    colors = set(map(int, out.ravel()))
    if BETWEEN not in colors or MATCH not in colors:
        return False

    changed_slots = {
        pos
        for pos, slot in info["slots"].items()
        if not np.array_equal(
            grid[slot["pixel_rc"][0] : slot["pixel_rc"][0] + SLOT_SIZE, slot["pixel_rc"][1] : slot["pixel_rc"][1] + SLOT_SIZE],
            out[slot["pixel_rc"][0] : slot["pixel_rc"][0] + SLOT_SIZE, slot["pixel_rc"][1] : slot["pixel_rc"][1] + SLOT_SIZE],
        )
    }
    if changed_slots != info["matched_slots"] | info["between_slots"]:
        return False
    if len(changed_slots) >= info["slot_rows"] * info["slot_cols"]:
        return False

    unchanged = (info["slot_rows"] * info["slot_cols"]) - len(changed_slots)
    return unchanged >= 2


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.choice((4, 5))
    cols = rng.choice((4, 5))

    chosen_colors = list(rng.sample(INPUT_COLOR_POOL, 4))
    rng.shuffle(chosen_colors)
    legend_pairs = [
        (chosen_colors[0], chosen_colors[1]),
        (chosen_colors[2], chosen_colors[3]),
    ]
    legend_pairs = [tuple(pair if rng.random() < 0.5 else pair[::-1]) for pair in legend_pairs]
    if rng.random() < 0.5:
        legend_pairs.reverse()
    legal_pair_set = {frozenset(pair) for pair in legend_pairs}

    total_slots = rows * cols
    chosen_shapes = _unused_shapes(rng, total_slots)
    valid_row_shape = chosen_shapes[0]
    valid_col_shape = chosen_shapes[1]
    invalid_shape = chosen_shapes[2]
    filler_shapes = chosen_shapes[3:]

    row_r = rng.randrange(rows)
    row_pair = _choose_gap_pair(rng, list(range(cols)), set(), row_r, True)
    row_between = {(row_r, c) for c in range(min(row_pair[0][1], row_pair[1][1]) + 1, max(row_pair[0][1], row_pair[1][1]))}

    forbidden_for_col = {row_pair[0], row_pair[1], *row_between}
    col_c = rng.randrange(cols)
    try:
        col_pair = _choose_gap_pair(rng, list(range(rows)), forbidden_for_col, col_c, False)
    except AssertionError:
        return None
    col_between = {(r, col_c) for r in range(min(col_pair[0][0], col_pair[1][0]) + 1, max(col_pair[0][0], col_pair[1][0]))}

    forbidden_for_invalid = {row_pair[0], row_pair[1], col_pair[0], col_pair[1], *row_between, *col_between}
    invalid_is_row = rng.random() < 0.5
    invalid_pair = None
    if invalid_is_row:
        row_options = list(range(rows))
        rng.shuffle(row_options)
        for invalid_row in row_options:
            try:
                invalid_pair = _choose_gap_pair(rng, list(range(cols)), forbidden_for_invalid, invalid_row, True)
                break
            except AssertionError:
                continue
    else:
        col_options = list(range(cols))
        rng.shuffle(col_options)
        for invalid_col in col_options:
            try:
                invalid_pair = _choose_gap_pair(rng, list(range(rows)), forbidden_for_invalid, invalid_col, False)
                break
            except AssertionError:
                continue
    if invalid_pair is None:
        return None

    assignments: dict[tuple[int, int], tuple[np.ndarray, int]] = {}
    pair1 = legend_pairs[0]
    pair2 = legend_pairs[1]

    row_colors = list(pair1)
    col_colors = list(pair2)
    rng.shuffle(row_colors)
    rng.shuffle(col_colors)
    assignments[row_pair[0]] = (valid_row_shape, row_colors[0])
    assignments[row_pair[1]] = (valid_row_shape, row_colors[1])
    assignments[col_pair[0]] = (valid_col_shape, col_colors[0])
    assignments[col_pair[1]] = (valid_col_shape, col_colors[1])

    cross_colors = [rng.choice(pair1), rng.choice(pair2)]
    assignments[invalid_pair[0]] = (invalid_shape, cross_colors[0])
    assignments[invalid_pair[1]] = (invalid_shape, cross_colors[1])

    all_positions = [(r, c) for r in range(rows) for c in range(cols)]
    filler_positions = [pos for pos in all_positions if pos not in assignments]
    rng.shuffle(filler_positions)
    filler_colors = [color for pair in legend_pairs for color in pair]
    for pos, shape in zip(filler_positions, filler_shapes):
        assignments[pos] = (shape, rng.choice(filler_colors))

    bottom_pad = rng.randint(3, 6)
    right_pad = rng.randint(1, 6)
    grid = _build_input(rows, cols, legend_pairs, assignments, bottom_pad=bottom_pad, right_pad=right_pad)
    info = _analyze_grid(grid)
    if len(info["valid_pairs"]) != 2:
        return None
    if {frozenset((info["slots"][a]["color"], info["slots"][b]["color"])) for a, b in info["valid_pairs"]} != legal_pair_set:
        return None
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(random)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-row pairing ARC input")
