from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import components_mask, get_bg, paste
except ImportError:
    utils_dir = Path("/home/lixinyu/snap/new/solvers")
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
    from arc_utils import components_mask, get_bg, paste


BG = 0
OBSTACLE = 5
PALETTE = (1, 2, 3, 4, 6, 7)
GEN_TRIES = 400


def _neighbors4(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _parse_grid(g: np.ndarray):
    bg = int(get_bg(g))
    h, w = map(int, g.shape)
    legends = []
    markers = []
    obstacles = []

    for coords in components_mask(g != bg, connectivity=4):
        if coords.shape[0] == 1:
            r, c = map(int, coords[0])
            color = int(g[r, c])
            if color == OBSTACLE:
                obstacles.append((r, c))
            else:
                markers.append((r, c, color))
            continue

        rows = coords[:, 0]
        cols = coords[:, 1]
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        assert r0 == r1, "multi-cell foreground components must be horizontal legends"
        assert r0 == 0, "legends must sit on the top edge"
        assert coords.shape[0] == c1 - c0 + 1, "legend cells must form a contiguous run"
        seq = tuple(int(g[r0, c]) for c in range(c0, c1 + 1))
        assert 4 <= len(seq) <= 6, "legend length must be between 4 and 6"
        assert len(set(seq)) == len(seq), "legend colors must be distinct within a legend"
        assert all(color in PALETTE for color in seq), "legend colors must come from the visible palette"
        legends.append({"row": r0, "c0": c0, "seq": seq})

    legends.sort(key=lambda item: (item["row"], item["c0"]))
    markers.sort()
    obstacles.sort()

    assert 1 <= len(legends) <= 2, "expected one or two top-edge legends"
    assert 2 <= len(markers) <= 3, "expected two or three isolated markers"
    return legends, markers, obstacles, bg, h, w


def _legend_occurrence(legends, color: int) -> tuple[tuple[int, ...], int, int]:
    matches = []
    for legend_i, legend in enumerate(legends):
        seq = legend["seq"]
        for idx, value in enumerate(seq):
            if int(value) == int(color):
                matches.append((seq, idx, legend_i))
    assert len(matches) == 1, "each marker color must identify one legend position"
    return matches[0]


def _direction_indices(anchor: int, seq_len: int, direction: int) -> list[int]:
    if direction < 0:
        return list(range(anchor - 1, -1, -1))
    return list(range(anchor + 1, seq_len))


def _simulate_direction(
    out: np.ndarray,
    g: np.ndarray,
    row: int,
    col: int,
    seq: tuple[int, ...],
    anchor: int,
    direction: int,
    bg: int,
) -> None:
    h, w = map(int, g.shape)
    folded = False

    for step, idx in enumerate(_direction_indices(anchor, len(seq), direction), start=1):
        target_col = col + direction * step
        target_row = row + (1 if folded else 0)

        if not (0 <= target_col < w and 0 <= target_row < h):
            break

        if not folded and int(g[row, target_col]) == OBSTACLE:
            folded = True
            target_row = row + 1
            if target_row >= h:
                break
            if int(g[target_row, target_col]) == OBSTACLE:
                break
            value = int(seq[len(seq) - 1 - idx])
        else:
            if folded and int(g[target_row, target_col]) == OBSTACLE:
                break
            value = int(seq[idx] if not folded else seq[len(seq) - 1 - idx])

        if int(out[target_row, target_col]) in (bg, value):
            out[target_row, target_col] = value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each isolated marker into its matching legend, folding around grey obstacles and reversing only the folded remainder's colors.

    Input structure:
    - The background is black, the top edge contains one or two horizontal legend strings,
      and every legend is a contiguous run of distinct colors from a fixed visible palette.
    - Interior foreground cells are isolated single-pixel markers whose colors occur in
      exactly one legend position, plus isolated grey obstacle pixels.
    - Obstacles and markers are separated so they remain single-pixel components in the input.

    Rule:
    - For each marker, find the unique legend containing the marker color and align that
      legend index over the marker.
    - Expand left and right from the marker: normal cells keep the legend colors, but if the
      next target cell on the marker row is grey, skip that obstacle, move the remaining cells
      for that direction to the next row, and recolor only that folded remainder by reversing
      its legend-index mapping.
    - Out-of-bounds expansion stops, later folded cells do not fold again, grey obstacles stay grey,
      and earlier marker rows keep priority if a later expansion would conflict.

    Key insight:
    - The top legend gives the original color order, while a grey pixel on a marker's path
      forces the unfinished suffix onto the next row and simultaneously mirrors its colors
      within the same legend palette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    legends, markers, _obstacles, bg, _h, _w = _parse_grid(g)

    out = g.copy()
    for row, col, color in sorted(markers, key=lambda item: (item[0], item[1])):
        seq, anchor, _legend_i = _legend_occurrence(legends, color)
        out[row, col] = int(color)
        _simulate_direction(out, g, row, col, seq, anchor, -1, bg)
        _simulate_direction(out, g, row, col, seq, anchor, 1, bg)

    return out


def _arrange_legend(length: int, shared: list[int], unique: list[int], rng: random.Random) -> tuple[tuple[int, ...], dict[int, int]]:
    positions = [None] * length
    unique_positions = rng.sample(list(range(1, length - 1)), len(unique))
    unique_positions.sort()

    unique_map = {}
    unique_colors = unique[:]
    rng.shuffle(unique_colors)
    for pos, color in zip(unique_positions, unique_colors):
        positions[pos] = int(color)
        unique_map[int(color)] = int(pos)

    remaining = shared[:]
    rng.shuffle(remaining)
    it = iter(remaining)
    for i in range(length):
        if positions[i] is None:
            positions[i] = int(next(it))

    return tuple(positions), unique_map


def _build_legends(rng: random.Random):
    if rng.random() < 0.5:
        length = rng.randint(4, 6)
        seq = tuple(rng.sample(PALETTE, length))
        candidates = {
            int(color): (0, int(idx), seq)
            for idx, color in enumerate(seq)
            if 0 < idx < length - 1
        }
        return [seq], candidates

    length_pairs = [(4, 4), (4, 5), (5, 4), (5, 5)]
    len_a, len_b = rng.choice(length_pairs)
    colors = list(PALETTE)
    rng.shuffle(colors)

    unique_a_n = 6 - len_b
    unique_b_n = 6 - len_a
    shared_n = len_a + len_b - 6

    unique_a = colors[:unique_a_n]
    unique_b = colors[unique_a_n : unique_a_n + unique_b_n]
    shared = colors[unique_a_n + unique_b_n : unique_a_n + unique_b_n + shared_n]

    seq_a, map_a = _arrange_legend(len_a, shared, unique_a, rng)
    seq_b, map_b = _arrange_legend(len_b, shared, unique_b, rng)
    candidates = {
        int(color): (0, int(pos), seq_a) for color, pos in map_a.items()
    }
    candidates.update(
        {int(color): (1, int(pos), seq_b) for color, pos in map_b.items()}
    )
    return [seq_a, seq_b], candidates


def _sample_rows(height: int, count: int, rng: random.Random) -> list[int] | None:
    rows = list(range(2, height - 1))
    for _ in range(200):
        picked = sorted(rng.sample(rows, count))
        if all(b - a >= 3 for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _choose_marker_specs(legends, candidates, rng: random.Random):
    colors = list(candidates)
    rng.shuffle(colors)
    max_markers = min(3, len(colors))
    marker_count = 3 if max_markers >= 3 and rng.random() < 0.6 else 2
    marker_count = min(marker_count, max_markers)
    assert marker_count >= 2

    if len(legends) == 2:
        by_legend = {0: [], 1: []}
        for color in colors:
            by_legend[candidates[color][0]].append(color)
        if by_legend[0] and by_legend[1] and marker_count >= 2:
            chosen = [rng.choice(by_legend[0]), rng.choice(by_legend[1])]
            for color in colors:
                if color not in chosen and len(chosen) < marker_count:
                    chosen.append(color)
        else:
            chosen = colors[:marker_count]
    else:
        chosen = colors[:marker_count]

    specs = []
    for color in chosen:
        legend_i, anchor, seq = candidates[int(color)]
        specs.append(
            {
                "color": int(color),
                "legend_i": int(legend_i),
                "anchor": int(anchor),
                "seq": tuple(int(v) for v in seq),
            }
        )
    return specs


def _fold_visible(seq: tuple[int, ...], indices: list[int]) -> bool:
    last = len(seq) - 1
    return any(int(seq[i]) != int(seq[last - i]) for i in indices)


def _assign_patterns(specs, rng: random.Random):
    for _ in range(200):
        folded_sides = 0
        clean_sides = 0
        planned = []
        for spec in specs:
            left_n = spec["anchor"]
            right_n = len(spec["seq"]) - spec["anchor"] - 1
            options = [{"left": False, "right": False}]
            if left_n >= 2:
                options.append({"left": True, "right": False})
            if right_n >= 2:
                options.append({"left": False, "right": True})
            if left_n >= 2 and right_n >= 2:
                options.append({"left": True, "right": True})
            pattern = rng.choice(options)
            folded_sides += int(pattern["left"]) + int(pattern["right"])
            clean_sides += int(not pattern["left"]) + int(not pattern["right"])
            planned.append(pattern)
        if folded_sides >= 1 and clean_sides >= 1:
            return planned
    return None


def _plan_direction(
    seq: tuple[int, ...],
    anchor: int,
    row: int,
    col: int,
    direction: int,
    fold_step: int | None,
    shape: tuple[int, int],
):
    h, w = shape
    indices = _direction_indices(anchor, len(seq), direction)
    cells = []
    clipped = False
    obstacle = None

    if fold_step is not None:
        obstacle_col = col + direction * fold_step
        if not (0 <= obstacle_col < w and row + 1 < h):
            return None
        folded_indices = indices[fold_step - 1 :]
        if not folded_indices or not _fold_visible(seq, folded_indices):
            return None
        obstacle = (row, obstacle_col)

    for step, idx in enumerate(indices, start=1):
        target_col = col + direction * step
        if not (0 <= target_col < w):
            clipped = True
            break
        if fold_step is not None and step >= fold_step:
            target_row = row + 1
            value = int(seq[len(seq) - 1 - idx])
            folded = True
        else:
            target_row = row
            value = int(seq[idx])
            folded = False
        cells.append(
            {
                "r": int(target_row),
                "c": int(target_col),
                "value": value,
                "idx": int(idx),
                "folded": folded,
            }
        )

    clipped |= len(cells) < len(indices)
    return {"cells": cells, "obstacle": obstacle, "clipped": clipped}


def _is_isolated(cell: tuple[int, int], occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    if not (0 <= r < h and 0 <= c < w):
        return False
    if cell in occupied:
        return False
    for nr, nc in _neighbors4(cell):
        if 0 <= nr < h and 0 <= nc < w and (nr, nc) in occupied:
            return False
    return True


def _enumerate_marker_placements(
    shape: tuple[int, int],
    row: int,
    spec,
    pattern,
    input_occ: set[tuple[int, int]],
    output_occ: set[tuple[int, int]],
):
    h, w = shape
    left_indices = _direction_indices(spec["anchor"], len(spec["seq"]), -1)
    right_indices = _direction_indices(spec["anchor"], len(spec["seq"]), 1)

    left_steps = [None]
    if pattern["left"]:
        left_steps = [
            step
            for step in range(2, len(left_indices) + 1)
            if _fold_visible(spec["seq"], left_indices[step - 1 :])
        ]
    right_steps = [None]
    if pattern["right"]:
        right_steps = [
            step
            for step in range(2, len(right_indices) + 1)
            if _fold_visible(spec["seq"], right_indices[step - 1 :])
        ]

    if not left_steps or not right_steps:
        return []

    candidates = []
    for col in range(w):
        marker = (row, col)
        for left_step in left_steps:
            left_plan = _plan_direction(spec["seq"], spec["anchor"], row, col, -1, left_step, shape)
            if left_plan is None:
                continue
            for right_step in right_steps:
                right_plan = _plan_direction(spec["seq"], spec["anchor"], row, col, 1, right_step, shape)
                if right_plan is None:
                    continue

                obstacles = {
                    coord
                    for coord in (left_plan["obstacle"], right_plan["obstacle"])
                    if coord is not None
                }
                if len(obstacles) != int(left_step is not None) + int(right_step is not None):
                    continue

                proposed_input = set(input_occ)
                valid = _is_isolated(marker, proposed_input | obstacles, shape)
                if not valid:
                    continue

                for obstacle in obstacles:
                    if not _is_isolated(obstacle, proposed_input | {marker} | (obstacles - {obstacle}), shape):
                        valid = False
                        break
                if not valid:
                    continue

                cells = left_plan["cells"] + right_plan["cells"]
                coords = {(cell["r"], cell["c"]) for cell in cells}
                if len(coords) != len(cells):
                    continue
                if not cells:
                    continue
                if any(coord in output_occ for coord in coords):
                    continue
                if any(coord in obstacles for coord in coords):
                    continue
                if marker in coords:
                    continue

                candidates.append(
                    {
                        "marker": marker,
                        "obstacles": tuple(sorted(obstacles)),
                        "cells": cells,
                        "clipped": bool(left_plan["clipped"] or right_plan["clipped"]),
                        "left_fold": left_step is not None,
                        "right_fold": right_step is not None,
                    }
                )
    return candidates


def _render_legends(shape: tuple[int, int], legends: list[tuple[int, ...]], starts: list[int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for seq, start in zip(legends, starts):
        paste(grid, np.asarray([seq], dtype=int), 0, int(start), clip=False)
    return grid


def _legend_starts(width: int, legends: list[tuple[int, ...]], rng: random.Random) -> list[int] | None:
    if len(legends) == 1:
        return [rng.randint(0, width - len(legends[0]))]

    left, right = legends
    min_gap = 2
    total = len(left) + len(right) + min_gap
    if total > width:
        return None
    start_a = rng.randint(0, width - total)
    start_b = rng.randint(start_a + len(left) + min_gap, width - len(right))
    return [start_a, start_b]


def _independent_expected(shape: tuple[int, int], legends: list[tuple[int, ...]], starts: list[int], plans) -> np.ndarray:
    out = _render_legends(shape, legends, starts)
    for plan in plans:
        row, col = plan["marker"]
        out[row, col] = int(plan["color"])
        for obstacle in plan["obstacles"]:
            out[obstacle] = OBSTACLE
        for cell in plan["cells"]:
            out[cell["r"], cell["c"]] = int(cell["value"])
    return out


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, plans) -> bool:
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    legends, markers, obstacles, _bg, _h, _w = _parse_grid(inp)
    marker_map = {(r, c): color for r, c, color in markers}
    obstacle_set = set(obstacles)

    saw_fold = False
    saw_clean = False
    for plan in plans:
        if plan["marker"] not in marker_map:
            return False
        if marker_map[plan["marker"]] != plan["color"]:
            return False

        seq, anchor, legend_i = _legend_occurrence(legends, plan["color"])
        if legend_i != plan["legend_i"] or anchor != plan["anchor"] or tuple(seq) != tuple(plan["seq"]):
            return False

        for obstacle in plan["obstacles"]:
            if obstacle not in obstacle_set or int(actual[obstacle]) != OBSTACLE:
                return False

        for cell in plan["cells"]:
            if int(actual[cell["r"], cell["c"]]) != int(cell["value"]):
                return False
            if cell["folded"]:
                saw_fold = True
            else:
                saw_clean = True

    return saw_fold and saw_clean


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    legends, candidates = _build_legends(rng)
    specs = _choose_marker_specs(legends, candidates, rng)
    patterns = _assign_patterns(specs, rng)
    if patterns is None:
        return None

    max_legend_width = sum(len(seq) for seq in legends) + (2 if len(legends) == 2 else 0)
    width = rng.randint(max(17, max_legend_width + 2), 24)
    height = rng.randint(13, 18)
    rows = _sample_rows(height, len(specs), rng)
    if rows is None:
        return None

    starts = _legend_starts(width, legends, rng)
    if starts is None:
        return None

    inp = _render_legends((height, width), legends, starts)
    input_occ = {(0, c) for start, seq in zip(starts, legends) for c in range(start, start + len(seq))}
    output_occ = set(input_occ)

    want_clipping = rng.random() < 0.35
    got_clipping = False
    plans = []

    paired_specs = []
    for row, spec, pattern in zip(rows, specs, patterns):
        combined = dict(spec)
        combined["row"] = int(row)
        combined["pattern"] = pattern
        paired_specs.append(combined)
    paired_specs.sort(key=lambda item: item["row"])

    for spec in paired_specs:
        choices = _enumerate_marker_placements(
            (height, width),
            spec["row"],
            spec,
            spec["pattern"],
            input_occ,
            output_occ,
        )
        if not choices:
            return None

        if want_clipping and not got_clipping:
            clipped = [choice for choice in choices if choice["clipped"]]
            if clipped:
                choices = clipped

        choice = rng.choice(choices)
        got_clipping |= bool(choice["clipped"])

        marker = choice["marker"]
        inp[marker] = int(spec["color"])
        input_occ.add(marker)
        output_occ.add(marker)

        for obstacle in choice["obstacles"]:
            inp[obstacle] = OBSTACLE
            input_occ.add(obstacle)
            output_occ.add(obstacle)

        for cell in choice["cells"]:
            output_occ.add((cell["r"], cell["c"]))

        plans.append(
            {
                "marker": marker,
                "color": int(spec["color"]),
                "legend_i": int(spec["legend_i"]),
                "anchor": int(spec["anchor"]),
                "seq": tuple(int(v) for v in spec["seq"]),
                "obstacles": choice["obstacles"],
                "cells": choice["cells"],
                "left_fold": bool(choice["left_fold"]),
                "right_fold": bool(choice["right_fold"]),
            }
        )

    if want_clipping and not got_clipping:
        return None

    expected = _independent_expected((height, width), legends, starts, plans)
    if not _validate_semantics(inp, expected, plans):
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_TRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid obstacle-fold legend-expansion input")
