from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

MAX_GENERATION_ATTEMPTS = 400
PALETTE_COUNT_CHOICES = (2, 3, 4)
PALETTE_COUNT_WEIGHTS = (1, 2, 2)
CROP_HEIGHT_RANGE = (5, 12)
RUN_WIDTH_RANGE = (1, 3)
MAX_TARGET_WIDTH = 8


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _is_monotone(values: list[int]) -> bool:
    return all(a <= b for a, b in zip(values, values[1:])) or all(
        a >= b for a, b in zip(values, values[1:])
    )


def _mirror_mosaic(seed: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [np.flipud(np.fliplr(seed)), np.flipud(seed)],
            [np.fliplr(seed), seed],
        ]
    )


def _palette_columns(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    out: list[tuple[int, int]] = []
    for c in range(w):
        col = grid[:, c]
        vals = np.unique(col)
        if len(vals) == 1:
            color = int(vals[0])
            if color not in (BLACK, GREY):
                out.append((c, color))
    assert 2 <= len(out) <= 4
    cols = [c for c, _ in out]
    assert all(cols[i + 1] - cols[i] == 2 for i in range(len(cols) - 1))
    return out


def _target_runs(mask: np.ndarray) -> list[dict]:
    active_cols = list(map(int, np.flatnonzero(mask.any(axis=0))))
    assert active_cols
    assert active_cols == list(range(active_cols[0], active_cols[-1] + 1))

    column_spans: list[tuple[int, int, int]] = []
    for c in active_cols:
        rows = np.flatnonzero(mask[:, c])
        assert rows.size > 0
        column_spans.append((c, int(rows[0]), int(rows[-1])))

    runs: list[dict] = []
    start_c, top, bottom = column_spans[0]
    prev_c = start_c
    for c, cur_top, cur_bottom in column_spans[1:]:
        if (cur_top, cur_bottom) == (top, bottom):
            prev_c = c
            continue
        runs.append({"c0": start_c, "c1": prev_c, "top": top, "bottom": bottom})
        start_c, top, bottom = c, cur_top, cur_bottom
        prev_c = c
    runs.append({"c0": start_c, "c1": prev_c, "top": top, "bottom": bottom})
    return runs


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    palette = _palette_columns(g)
    palette_cols = [c for c, _ in palette]
    palette_colors = [color for _, color in palette]

    present = set(map(int, np.unique(g)))
    assert present <= {BLACK, GREY, *palette_colors}

    target_mask = g == GREY
    assert np.any(target_mask)
    assert _component_count(target_mask) == 1

    grey_cols = list(map(int, np.flatnonzero(target_mask.any(axis=0))))
    assert grey_cols[0] > palette_cols[-1]
    spacer = g[:, palette_cols[-1] + 1 : grey_cols[0]]
    assert spacer.size > 0 and np.all(spacer == BLACK)

    target_box = _bbox(target_mask)
    assert target_box is not None
    runs = _target_runs(target_mask)
    assert len(runs) == len(palette_colors)
    spans = [(run["top"], run["bottom"]) for run in runs]
    assert len(set(spans)) == len(spans)

    return {
        "grid": g,
        "shape": (h, w),
        "palette": palette,
        "palette_cols": palette_cols,
        "palette_colors": palette_colors,
        "target_mask": target_mask,
        "target_box": target_box,
        "runs": runs,
        "spans": spans,
    }


def _recolored_seed(info: dict) -> np.ndarray:
    g = info["grid"]
    target_mask = info["target_mask"]
    recolored = np.zeros_like(g)
    for color, run in zip(info["palette_colors"], info["runs"]):
        sl = slice(run["c0"], run["c1"] + 1)
        view = recolored[:, sl]
        view[target_mask[:, sl]] = int(color)
    return _crop_bbox(recolored, info["target_box"])


def _recolored_seed_with_run_order(info: dict, run_order: list[int]) -> np.ndarray:
    g = info["grid"]
    target_mask = info["target_mask"]
    recolored = np.zeros_like(g)
    for color, run_idx in zip(info["palette_colors"], run_order):
        run = info["runs"][run_idx]
        sl = slice(run["c0"], run["c1"] + 1)
        view = recolored[:, sl]
        view[target_mask[:, sl]] = int(color)
    return _crop_bbox(recolored, info["target_box"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each target column-run from the left palette order, then mirror the cropped result into a four-quadrant mosaic.

    Input structure:
    - Black background with 2-4 full-height one-cell palette columns on the left, each separated by one black column.
    - A single connected grey target figure sits to the right, also separated from the palette by black space.
    - The grey figure spans consecutive columns; each maximal group of adjacent columns sharing the same top and bottom foreground rows is one column-run.

    Rule:
    - Scan the grey figure left to right and partition it into maximal column-runs with identical vertical spans.
    - Match palette columns to those runs in left-to-right order and recolor each run's grey pixels with its matched palette color.
    - Erase the palette area, crop the recolored target to its tight bounding box, and use that crop as the lower-right quadrant of a mirror mosaic.
    - Fill the lower-left with a left-right mirror, the upper-right with an up-down mirror, and the upper-left with the 180-degree rotation.

    Key insight:
    - The palette does not match runs by geometry or color id; its only role is to provide an ordered color list that is applied to the target's ordered column-runs before the mirrored assembly step.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))
    seed = _recolored_seed(info)
    return _mirror_mosaic(seed)


def _sample_palette_count() -> int:
    return random.choices(PALETTE_COUNT_CHOICES, weights=PALETTE_COUNT_WEIGHTS, k=1)[0]


def _sample_palette_colors(n: int) -> list[int]:
    for _ in range(100):
        colors = random.sample(PALETTE_POOL, n)
        if n < 3 or not _is_monotone(colors):
            return colors
    raise RuntimeError("failed to sample a non-monotone palette order")


def _sample_widths(n: int) -> list[int]:
    for _ in range(200):
        widths = [random.randint(*RUN_WIDTH_RANGE) for _ in range(n)]
        if sum(widths) > MAX_TARGET_WIDTH:
            continue
        if max(widths) == 1:
            continue
        if len(set(widths)) == 1:
            continue
        if n >= 3 and _is_monotone(widths):
            continue
        return widths
    raise RuntimeError("failed to sample non-degenerate run widths")


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _sample_run_intervals(n: int, crop_h: int) -> list[tuple[int, int]]:
    all_intervals = [
        (top, bottom)
        for top in range(crop_h - 1)
        for bottom in range(top + 1, crop_h)
    ]
    require_duplicate_height = n >= 3 and random.random() < 0.5

    for _ in range(1500):
        chosen = random.sample(all_intervals, n)
        if min(top for top, _ in chosen) != 0:
            continue
        if max(bottom for _, bottom in chosen) != crop_h - 1:
            continue

        heights = [bottom - top + 1 for top, bottom in chosen]
        if require_duplicate_height and len(set(heights)) == len(heights):
            continue
        if not require_duplicate_height and len(set(heights)) == 1:
            continue

        for perm in itertools.permutations(chosen):
            if any(not _intervals_overlap(perm[i], perm[i + 1]) for i in range(n - 1)):
                continue
            tops = [top for top, _ in perm]
            bottoms = [bottom for _, bottom in perm]
            if n >= 3:
                if _is_monotone(tops) or _is_monotone(bottoms):
                    continue
            return list(perm)

    raise RuntimeError("failed to sample valid run intervals")


def _build_seed(intervals: list[tuple[int, int]], widths: list[int], colors: list[int]) -> np.ndarray:
    crop_h = max(bottom for _, bottom in intervals) + 1
    crop_w = sum(widths)
    seed = np.zeros((crop_h, crop_w), dtype=int)
    c = 0
    for (top, bottom), width, color in zip(intervals, widths, colors):
        seed[top : bottom + 1, c : c + width] = int(color)
        c += width
    return seed


def _build_input_from_seed(seed: np.ndarray, colors: list[int]) -> np.ndarray:
    crop_h, crop_w = seed.shape
    h = crop_h + 2
    n = len(colors)
    palette_cols = [1 + 2 * i for i in range(n)]
    target_start = 2 * n + 1
    w = target_start + crop_w + 1

    grid = np.zeros((h, w), dtype=int)
    for c, color in zip(palette_cols, colors):
        grid[:, c] = int(color)

    grey_seed = np.where(seed != BLACK, GREY, BLACK)
    grid[1 : 1 + crop_h, target_start : target_start + crop_w] = grey_seed
    return grid


def _has_seed_symmetry(seed: np.ndarray) -> bool:
    return (
        np.array_equal(seed, np.fliplr(seed))
        or np.array_equal(seed, np.flipud(seed))
        or np.array_equal(seed, np.rot90(seed, 2))
    )


def _validate_instance(grid: np.ndarray) -> None:
    info = _analyze_grid(grid)
    seed = _recolored_seed(info)
    out = main(grid)

    assert seed.shape[0] <= 15 and seed.shape[1] <= 15
    assert out.shape == (2 * seed.shape[0], 2 * seed.shape[1])
    assert np.array_equal(out, _mirror_mosaic(seed))
    assert not _has_seed_symmetry(seed)
    assert set(map(int, np.unique(seed))) <= {BLACK, *info["palette_colors"]}
    assert len(set(map(int, np.unique(seed))) - {BLACK}) == len(info["palette_colors"])

    h, w = info["shape"]
    r0, c0, r1, c1 = info["target_box"]
    assert (r0, r1) == (1, h - 2)
    assert c0 > info["palette_cols"][-1]
    assert seed.shape != grid.shape

    widths = [run["c1"] - run["c0"] + 1 for run in info["runs"]]
    assert max(widths) > 1
    assert len(set(widths)) > 1

    if len(info["palette_colors"]) >= 3:
        top_order = sorted(range(len(info["runs"])), key=lambda i: info["runs"][i]["top"])
        bottom_order = sorted(range(len(info["runs"])), key=lambda i: info["runs"][i]["bottom"])
        if top_order != list(range(len(info["runs"]))):
            assert not np.array_equal(seed, _recolored_seed_with_run_order(info, top_order))
        if bottom_order != list(range(len(info["runs"]))):
            assert not np.array_equal(seed, _recolored_seed_with_run_order(info, bottom_order))

    reversed_seed = _build_seed(
        [(run["top"] - 1, run["bottom"] - 1) for run in info["runs"]],
        widths,
        list(reversed(info["palette_colors"])),
    )
    assert not np.array_equal(seed, reversed_seed)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        n = _sample_palette_count()
        colors = _sample_palette_colors(n)
        widths = _sample_widths(n)
        crop_h = random.randint(*CROP_HEIGHT_RANGE)
        intervals = _sample_run_intervals(n, crop_h)
        seed = _build_seed(intervals, widths, colors)
        if _has_seed_symmetry(seed):
            continue
        grid = _build_input_from_seed(seed, colors)
        if max(grid.shape) > 30 or max(_mirror_mosaic(seed).shape) > 30:
            continue
        _validate_instance(grid)
        return grid
    raise RuntimeError("failed to generate a valid ARC instance")
