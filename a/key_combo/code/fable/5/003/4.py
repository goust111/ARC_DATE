from __future__ import annotations

from collections import Counter
from itertools import permutations
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _find_baseline(g: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for r in range(g.shape[0]):
        row_vals = np.unique(g[r])
        if row_vals.size == 1 and int(row_vals[0]) != int(bg):
            candidates.append((r, int(row_vals[0])))
    assert len(candidates) == 1, "expected exactly one full-width non-background baseline"
    return candidates[0]


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = _get_bg(g)
    baseline_r, baseline_color = _find_baseline(g, bg)
    assert 1 <= baseline_r < height - 1, "baseline must have room above and below"

    slots = []
    bar_cells: set[tuple[int, int]] = set()
    tooth_cells: set[tuple[int, int]] = set()

    for c in range(width):
        top_color = int(g[baseline_r - 1, c])
        if top_color in (bg, baseline_color):
            continue

        r = baseline_r - 1
        while r >= 0 and int(g[r, c]) == top_color:
            bar_cells.add((r, c))
            r -= 1
        assert np.all(g[: r + 1, c] == bg), "bar columns must be clean above the bar"
        bar_height = baseline_r - r - 1

        tooth = None
        below_color = int(g[baseline_r + 1, c])
        if below_color != bg:
            assert below_color != baseline_color, "tooth color must differ from the baseline"
            rr = baseline_r + 1
            while rr < height and int(g[rr, c]) == below_color:
                tooth_cells.add((rr, c))
                rr += 1
            assert np.all(g[rr:, c] == bg), "tooth columns must be clean below the tooth"
            tooth = {
                "length": int(rr - baseline_r - 1),
                "color": int(below_color),
            }

        slots.append(
            {
                "col": int(c),
                "height": int(bar_height),
                "color": int(top_color),
                "tooth": tooth,
            }
        )

    assert len(slots) >= 3, "expected several bar slots"

    actual_above = {
        tuple(map(int, p))
        for p in np.argwhere(g[:baseline_r, :] != bg)
    }
    assert actual_above == bar_cells, "all foreground above the baseline must be bars"

    actual_below = {
        (int(r + baseline_r + 1), int(c))
        for r, c in np.argwhere(g[baseline_r + 1 :, :] != bg)
    }
    assert actual_below == tooth_cells, "all foreground below the baseline must be teeth"

    bar_heights = [slot["height"] for slot in slots]
    bar_colors = [slot["color"] for slot in slots]
    assert len(set(bar_heights)) == len(bar_heights), "bar heights must be unique"
    assert len(set(bar_colors)) == len(bar_colors), "bar colors must be unique"

    toothed = [slot for slot in slots if slot["tooth"] is not None]
    locked = [slot for slot in slots if slot["tooth"] is None]
    assert len(locked) <= 1, "at most one slot may omit a tooth"
    assert len(toothed) >= 2, "expected at least two ranked tooth slots"

    tooth_lengths = [slot["tooth"]["length"] for slot in toothed]
    tooth_colors = [slot["tooth"]["color"] for slot in toothed]
    assert len(set(tooth_lengths)) == len(tooth_lengths), "tooth lengths must be unique"
    assert len(set(tooth_colors)) == len(tooth_colors), "tooth colors must be unique"
    assert set(tooth_lengths) == set(range(1, len(toothed) + 1)), (
        "tooth lengths must be exactly the visible rank indexes"
    )

    return {
        "bg": int(bg),
        "baseline_r": int(baseline_r),
        "baseline_color": int(baseline_color),
        "slots": slots,
    }


def _solve_from_scene(shape: tuple[int, int], scene: dict) -> np.ndarray:
    bg = int(scene["bg"])
    baseline_r = int(scene["baseline_r"])
    baseline_color = int(scene["baseline_color"])
    slots = scene["slots"]

    out = np.full(shape, bg, dtype=int)
    out[baseline_r, :] = baseline_color

    ranked_heights = sorted(int(slot["height"]) for slot in slots if slot["tooth"] is not None)
    for slot in slots:
        col = int(slot["col"])
        tooth = slot["tooth"]
        if tooth is None:
            bar_height = int(slot["height"])
            color = int(slot["color"])
        else:
            bar_height = int(ranked_heights[int(tooth["length"]) - 1])
            color = int(tooth["color"])
        assert 1 <= bar_height <= baseline_r, "assigned bar must fit above the baseline"
        out[baseline_r - bar_height : baseline_r, col] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Redistribute vertical bar heights by the rank indexes encoded in the teeth, then recolor by tooth color.

    Input structure:
    - A single-color horizontal baseline spans the full grid width.
    - Above the baseline, several one-cell-wide monochrome vertical bars rise from distinct slot columns; their heights and colors are unique.
    - Below the baseline, every slot except possibly one has a one-cell-wide monochrome tooth; tooth lengths are the visible rank indexes and tooth colors are unique.

    Rule:
    - The slot with no tooth keeps its original bar shape and color, and that bar is removed from the ranking pool.
    - Each toothed slot with a tooth of length k receives the kth shortest remaining bar height.
    - Repaint each received bar with that slot's tooth color, keep the baseline, and erase all teeth.

    Key insight:
    - The underground tooth is both an index and a paint source; a missing tooth locks its own bar and forces the height ranks to be recomputed without it.
    """
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)
    return _solve_from_scene(g.shape, scene)


COLOR_COUNT = 10
GEN_ATTEMPTS = 500


def _sample_columns(width: int, count: int) -> list[int] | None:
    usable = list(range(1, width - 1))
    for _ in range(100):
        cols = sorted(random.sample(usable, count))
        if all(cols[i + 1] - cols[i] >= 2 for i in range(count - 1)):
            return cols
    return None


def _sample_deranged_lengths(indices: list[int], heights: list[int]) -> dict[int, int] | None:
    sorted_indices = sorted(indices, key=lambda i: heights[i])
    own_rank = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
    ranks = tuple(range(1, len(indices) + 1))
    choices = [
        perm
        for perm in permutations(ranks)
        if all(int(perm[pos]) != int(own_rank[idx]) for pos, idx in enumerate(indices))
    ]
    if not choices:
        return None
    chosen = random.choice(choices)
    return {idx: int(chosen[pos]) for pos, idx in enumerate(indices)}


def _render_input(
    shape: tuple[int, int],
    bg: int,
    baseline_r: int,
    baseline_color: int,
    cols: list[int],
    heights: list[int],
    bar_colors: list[int],
    tooth_lengths: dict[int, int],
    tooth_colors: dict[int, int],
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    grid[baseline_r, :] = int(baseline_color)
    for idx, col in enumerate(cols):
        h = int(heights[idx])
        grid[baseline_r - h : baseline_r, col] = int(bar_colors[idx])
        if idx in tooth_lengths:
            length = int(tooth_lengths[idx])
            grid[baseline_r + 1 : baseline_r + 1 + length, col] = int(tooth_colors[idx])
    return grid


def _expected_from_spec(
    shape: tuple[int, int],
    bg: int,
    baseline_r: int,
    baseline_color: int,
    cols: list[int],
    heights: list[int],
    bar_colors: list[int],
    tooth_lengths: dict[int, int],
    tooth_colors: dict[int, int],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    out[baseline_r, :] = int(baseline_color)
    ranked = sorted(int(heights[idx]) for idx in tooth_lengths)
    for idx, col in enumerate(cols):
        if idx not in tooth_lengths:
            h = int(heights[idx])
            color = int(bar_colors[idx])
        else:
            h = int(ranked[int(tooth_lengths[idx]) - 1])
            color = int(tooth_colors[idx])
        out[baseline_r - h : baseline_r, col] = color
    return out


def _is_semantically_strong(
    grid: np.ndarray,
    expected: np.ndarray,
    heights: list[int],
    bar_colors: list[int],
    tooth_lengths: dict[int, int],
    tooth_colors: dict[int, int],
) -> bool:
    if _get_bg(grid) != int(grid[0, 0]):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(grid, expected):
        return False

    participating = list(tooth_lengths)
    ranked_participating = sorted(int(heights[idx]) for idx in participating)

    changed_height = False
    for idx in participating:
        assigned = int(ranked_participating[int(tooth_lengths[idx]) - 1])
        if assigned != int(heights[idx]):
            changed_height = True
        if int(tooth_colors[idx]) == int(bar_colors[idx]):
            return False
    if not changed_height:
        return False

    locked = [idx for idx in range(len(heights)) if idx not in tooth_lengths]
    if locked:
        all_ranked = sorted(map(int, heights))
        naive = [all_ranked[int(tooth_lengths[idx]) - 1] for idx in participating]
        correct = [ranked_participating[int(tooth_lengths[idx]) - 1] for idx in participating]
        if naive == correct:
            return False

    if len(participating) >= 3:
        by_slot = [idx for idx in range(len(heights)) if idx in tooth_lengths]
        original_ranks = {
            idx: rank + 1
            for rank, idx in enumerate(sorted(by_slot, key=lambda i: heights[i]))
        }
        ranks_in_slot_order = [original_ranks[idx] for idx in by_slot]
        teeth_in_slot_order = [int(tooth_lengths[idx]) for idx in by_slot]
        if ranks_in_slot_order == sorted(ranks_in_slot_order):
            return False
        if teeth_in_slot_order == sorted(teeth_in_slot_order):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    slot_count = random.randint(3, 4)
    use_lock = random.random() < 0.5
    tooth_count = slot_count - (1 if use_lock else 0)
    if tooth_count < 2:
        return None

    top_capacity = random.randint(max(slot_count + 2, 6), 13)
    baseline_r = top_capacity + random.randint(0, 2)
    bottom_room = tooth_count + random.randint(1, 4)
    height = baseline_r + 1 + bottom_room
    if height > 30:
        return None

    width = random.randint(max(2 * slot_count + 5, 11), 30)
    cols = _sample_columns(width, slot_count)
    if cols is None:
        return None

    heights = random.sample(range(2, top_capacity + 1), slot_count)
    if use_lock:
        lock_candidates = [i for i, h in enumerate(heights) if int(h) != max(heights)]
        if not lock_candidates:
            return None
        locked_idx = random.choice(lock_candidates)
    else:
        locked_idx = None

    participating = [i for i in range(slot_count) if i != locked_idx]
    tooth_lengths = _sample_deranged_lengths(participating, heights)
    if tooth_lengths is None:
        return None

    bg = random.randrange(COLOR_COUNT)
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    baseline_color = palette[0]
    needed = slot_count + tooth_count
    if needed > len(palette) - 1:
        return None
    paint = palette[1 : 1 + needed]
    bar_colors = paint[:slot_count]
    tooth_palette = paint[slot_count:]
    tooth_colors = {idx: int(tooth_palette[pos]) for pos, idx in enumerate(participating)}

    grid = _render_input(
        (height, width),
        bg,
        baseline_r,
        baseline_color,
        cols,
        heights,
        bar_colors,
        tooth_lengths,
        tooth_colors,
    )
    expected = _expected_from_spec(
        grid.shape,
        bg,
        baseline_r,
        baseline_color,
        cols,
        heights,
        bar_colors,
        tooth_lengths,
        tooth_colors,
    )
    if _is_semantically_strong(grid, expected, heights, bar_colors, tooth_lengths, tooth_colors):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
