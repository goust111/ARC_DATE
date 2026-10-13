from __future__ import annotations

from collections import Counter
import random

import numpy as np


MAX_BREAKS = 2
GEN_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _stripe_rows_and_cols(grid: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    row_counts = np.sum(grid != bg, axis=1)
    col_counts = np.sum(grid != bg, axis=0)

    rows = [int(r) for r, count in enumerate(row_counts) if int(count) >= w - MAX_BREAKS]
    cols = [int(c) for c, count in enumerate(col_counts) if int(count) >= h - MAX_BREAKS]

    assert rows, "expected at least one horizontal stripe row"
    assert cols, "expected at least one vertical stripe column"
    return rows, cols


def _direct_row_color(grid: np.ndarray, row: int, stripe_cols: set[int], bg: int) -> int | None:
    vals = [
        int(grid[row, c])
        for c in range(grid.shape[1])
        if c not in stripe_cols and int(grid[row, c]) != bg
    ]
    if not vals:
        return None
    return vals[0] if len(set(vals)) == 1 else None


def _direct_col_color(grid: np.ndarray, col: int, stripe_rows: set[int], bg: int) -> int | None:
    vals = [
        int(grid[r, col])
        for r in range(grid.shape[0])
        if r not in stripe_rows and int(grid[r, col]) != bg
    ]
    if not vals:
        return None
    return vals[0] if len(set(vals)) == 1 else None


def _mix_maps(
    grid: np.ndarray,
    row_colors: dict[int, int],
    col_colors: dict[int, int],
    bg: int,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], set[int]], dict[tuple[int, int], set[int]]]:
    pair_to_mix: dict[tuple[int, int], int] = {}
    inv_row: dict[tuple[int, int], set[int]] = {}
    inv_col: dict[tuple[int, int], set[int]] = {}

    for r, row_color in row_colors.items():
        for c, col_color in col_colors.items():
            mix = int(grid[r, c])
            if mix == bg:
                continue
            pair = (int(row_color), int(col_color))
            prev = pair_to_mix.setdefault(pair, mix)
            assert prev == mix, "same color pair must produce a stable mixed color"

    for (row_color, col_color), mix in pair_to_mix.items():
        inv_row.setdefault((int(col_color), int(mix)), set()).add(int(row_color))
        inv_col.setdefault((int(row_color), int(mix)), set()).add(int(col_color))

    return pair_to_mix, inv_row, inv_col


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore interrupted horizontal/vertical color bands by using crossing colors as a lookup table.

    Input structure:
    - The background is the most common color.
    - Several 1-pixel horizontal bands and several 1-pixel vertical bands cross.
    - An intact crossing uses a deterministic mixed color distinct from both band colors.
    - Some band pixels are replaced by background, creating 1-2 pixel breaks.

    Rule:
    - Identify the stripe rows and stripe columns.
    - Read any directly visible stripe colors away from crossings.
    - Build a mapping from (horizontal color, vertical color) to crossing color from known crossings.
    - Reverse that mapping to infer broken stripe colors, then fill only the broken pixels.
    - Crossing pixels stay mixed; if an intersection itself is missing, restore the mixed color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _bg_color(g)
    stripe_rows, stripe_cols = _stripe_rows_and_cols(g, bg)
    row_set = set(stripe_rows)
    col_set = set(stripe_cols)

    row_colors: dict[int, int] = {}
    col_colors: dict[int, int] = {}

    for r in stripe_rows:
        color = _direct_row_color(g, r, col_set, bg)
        if color is not None:
            row_colors[int(r)] = int(color)
    for c in stripe_cols:
        color = _direct_col_color(g, c, row_set, bg)
        if color is not None:
            col_colors[int(c)] = int(color)

    for _ in range(len(stripe_rows) + len(stripe_cols) + 2):
        pair_to_mix, inv_row, inv_col = _mix_maps(g, row_colors, col_colors, bg)
        changed = False

        for r in stripe_rows:
            if r in row_colors:
                continue
            candidates: list[int] = []
            for c, col_color in col_colors.items():
                mix = int(g[r, c])
                opts = inv_row.get((int(col_color), int(mix)), set())
                if len(opts) == 1:
                    candidates.append(next(iter(opts)))
            if candidates and len(set(candidates)) == 1:
                row_colors[int(r)] = int(candidates[0])
                changed = True

        for c in stripe_cols:
            if c in col_colors:
                continue
            candidates = []
            for r, row_color in row_colors.items():
                mix = int(g[r, c])
                opts = inv_col.get((int(row_color), int(mix)), set())
                if len(opts) == 1:
                    candidates.append(next(iter(opts)))
            if candidates and len(set(candidates)) == 1:
                col_colors[int(c)] = int(candidates[0])
                changed = True

        if not changed:
            break

    assert len(row_colors) == len(stripe_rows), "failed to infer every horizontal stripe color"
    assert len(col_colors) == len(stripe_cols), "failed to infer every vertical stripe color"

    pair_to_mix, _, _ = _mix_maps(g, row_colors, col_colors, bg)
    out = g.copy()

    for r in stripe_rows:
        row_color = int(row_colors[r])
        for c in range(g.shape[1]):
            if c in col_set:
                continue
            if int(out[r, c]) == bg:
                out[r, c] = row_color

    for c in stripe_cols:
        col_color = int(col_colors[c])
        for r in range(g.shape[0]):
            if r in row_set:
                continue
            if int(out[r, c]) == bg:
                out[r, c] = col_color

    for r in stripe_rows:
        for c in stripe_cols:
            if int(out[r, c]) == bg:
                out[r, c] = int(pair_to_mix[(int(row_colors[r]), int(col_colors[c]))])

    return out


def _sample_spaced_positions(length: int, n: int) -> list[int] | None:
    candidates = list(range(1, length - 1))
    if n <= 0 or len(candidates) < n:
        return None
    for _ in range(120):
        picked = sorted(random.sample(candidates, n))
        if all(b - a >= 2 for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _mix_color(row_color: int, col_color: int, h_palette: list[int], v_palette: list[int], mix_palette: list[int]) -> int:
    i = h_palette.index(int(row_color))
    j = v_palette.index(int(col_color))
    return int(mix_palette[(i + j) % len(mix_palette)])


def _build_full_grid(
    height: int,
    width: int,
    bg: int,
    row_specs: list[dict[str, int | bool]],
    col_specs: list[dict[str, int | bool]],
    h_palette: list[int],
    v_palette: list[int],
    mix_palette: list[int],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)

    for spec in row_specs:
        grid[int(spec["pos"]), :] = int(spec["color"])
    for spec in col_specs:
        grid[:, int(spec["pos"])] = int(spec["color"])
    for row_spec in row_specs:
        r = int(row_spec["pos"])
        row_color = int(row_spec["color"])
        for col_spec in col_specs:
            c = int(col_spec["pos"])
            col_color = int(col_spec["color"])
            grid[r, c] = _mix_color(row_color, col_color, h_palette, v_palette, mix_palette)

    return grid


def _counts_ok(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    if int(counts[bg]) != max(counts.values()):
        return False
    return list(counts.values()).count(int(counts[bg])) == 1


def _gen_attempt() -> np.ndarray | None:
    palette = list(range(10))
    random.shuffle(palette)

    k = random.choice([2, 3])
    bg = int(palette[0])
    h_palette = [int(x) for x in palette[1 : 1 + k]]
    v_palette = [int(x) for x in palette[1 + k : 1 + 2 * k]]
    mix_palette = [int(x) for x in palette[1 + 2 * k : 1 + 3 * k]]

    for _ in range(80):
        height = random.randint(10, 17)
        width = random.randint(10, 17)
        max_rows = (height - 1) // 2
        max_cols = (width - 1) // 2
        min_needed = k + 1
        if max_rows < min_needed or max_cols < min_needed:
            continue

        n_rows = random.randint(min_needed, min(max_rows, k + 3))
        n_cols = random.randint(min_needed, min(max_cols, k + 3))

        row_positions = _sample_spaced_positions(height, n_rows)
        col_positions = _sample_spaced_positions(width, n_cols)
        if row_positions is None or col_positions is None:
            continue

        row_specs: list[dict[str, int | bool]] = []
        col_specs: list[dict[str, int | bool]] = []

        row_anchor_colors = h_palette.copy()
        col_anchor_colors = v_palette.copy()
        random.shuffle(row_anchor_colors)
        random.shuffle(col_anchor_colors)

        for pos, color in zip(row_positions[:k], row_anchor_colors):
            row_specs.append({"pos": int(pos), "color": int(color), "anchor": True})
        for pos, color in zip(col_positions[:k], col_anchor_colors):
            col_specs.append({"pos": int(pos), "color": int(color), "anchor": True})

        for pos in row_positions[k:]:
            row_specs.append({"pos": int(pos), "color": int(random.choice(h_palette)), "anchor": False})
        for pos in col_positions[k:]:
            col_specs.append({"pos": int(pos), "color": int(random.choice(v_palette)), "anchor": False})

        random.shuffle(row_specs)
        random.shuffle(col_specs)

        full = _build_full_grid(height, width, bg, row_specs, col_specs, h_palette, v_palette, mix_palette)
        if not _counts_ok(full, bg):
            continue

        inp = full.copy()
        stripe_rows = {int(spec["pos"]) for spec in row_specs}
        stripe_cols = {int(spec["pos"]) for spec in col_specs}

        row_breakables = [spec for spec in row_specs if not bool(spec["anchor"])]
        col_breakables = [spec for spec in col_specs if not bool(spec["anchor"])]

        row_break_count = random.randint(0, min(2, len(row_breakables)))
        col_break_count = random.randint(0, min(2, len(col_breakables)))
        if row_break_count == 0 and col_break_count == 0:
            if row_breakables and col_breakables:
                if random.random() < 0.5:
                    row_break_count = 1
                else:
                    col_break_count = 1
            elif row_breakables:
                row_break_count = 1
            elif col_breakables:
                col_break_count = 1
            else:
                continue

        for spec in random.sample(row_breakables, row_break_count):
            r = int(spec["pos"])
            options = [c for c in range(width) if c not in stripe_cols]
            if len(options) < 1:
                return None
            n_breaks = random.choice([1, 2]) if len(options) >= 2 else 1
            for c in random.sample(options, n_breaks):
                inp[r, c] = bg

        for spec in random.sample(col_breakables, col_break_count):
            c = int(spec["pos"])
            options = [r for r in range(height) if r not in stripe_rows]
            if len(options) < 1:
                return None
            n_breaks = random.choice([1, 2]) if len(options) >= 2 else 1
            for r in random.sample(options, n_breaks):
                inp[r, c] = bg

        if np.array_equal(main(inp), full):
            return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stripe-crossing puzzle")
