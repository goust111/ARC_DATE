import random

import numpy as np


BG = 0
FRAME = 9
NON_SPECIAL_COLORS = tuple(c for c in range(10) if c not in (BG, FRAME))
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 500
PLACE_RETRIES = 200


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _monochrome_components(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((color, cells))
    return comps


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color, cells in _monochrome_components(grid, bg):
        r0, c0, r1, c1 = _bbox(cells)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        expected = set()
        for cc in range(c0, c1 + 1):
            expected.add((r0, cc))
            expected.add((r1, cc))
        for rr in range(r0 + 1, r1):
            expected.add((rr, c0))
            expected.add((rr, c1))
        if len(cells) != len(expected):
            continue
        if set(cells) != expected:
            continue
        frames.append({"color": color, "bbox": (r0, c0, r1, c1), "cells": expected})
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _find_legends(grid: np.ndarray, bg: int, frames: list[dict]) -> dict[int, list[int]]:
    covered = np.zeros(grid.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        covered[r0 : r1 + 1, c0 : c1 + 1] = True

    legends: dict[int, list[int]] = {}
    mask = (grid != bg) & ~covered
    for cells in _components(mask):
        if not (2 <= len(cells) <= 4):
            continue
        r0, c0, r1, c1 = _bbox(cells)
        if r0 != r1 and c0 != c1:
            continue
        if r0 == r1:
            cols = sorted(c for _, c in cells)
            if cols != list(range(c0, c1 + 1)):
                continue
            seq = [int(grid[r0, cc]) for cc in cols]
        else:
            rows = sorted(r for r, _ in cells)
            if rows != list(range(r0, r1 + 1)):
                continue
            seq = [int(grid[rr, c0]) for rr in rows]
        if len(set(seq)) != len(seq):
            continue
        legends[int(seq[0])] = [int(v) for v in seq]
    return legends


def _infer_sweep(seed_idx: int, lo: int, hi: int, extra_steps: int) -> int | None:
    candidates = []
    for sign in (-1, 1):
        forward = hi - seed_idx if sign > 0 else seed_idx - lo
        other = seed_idx - lo if sign > 0 else hi - seed_idx
        if forward >= extra_steps and other < extra_steps:
            candidates.append(sign)
    if len(candidates) != 1:
        return None
    return int(candidates[0])


def _segment_plan(
    frame_bbox: tuple[int, int, int, int],
    seed: tuple[int, int],
    indicator: tuple[int, int],
    seq: list[int],
) -> list[tuple[list[tuple[int, int]], int]] | None:
    r0, c0, r1, c1 = frame_bbox
    inner_r0, inner_c0, inner_r1, inner_c1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
    seed_r, seed_c = seed
    ind_r, ind_c = indicator
    dr, dc = ind_r - seed_r, ind_c - seed_c
    if abs(dr) + abs(dc) != 1:
        return None

    extra_steps = len(seq) - 1
    plan: list[tuple[list[tuple[int, int]], int]] = []

    if dc != 0:
        sweep = _infer_sweep(seed_r, inner_r0, inner_r1, extra_steps)
        if sweep is None:
            return None
        if dc > 0:
            initial = [(seed_r, cc) for cc in range(seed_c, inner_c1 + 1)]
        else:
            initial = [(seed_r, cc) for cc in range(inner_c0, seed_c + 1)]
        plan.append((initial, int(seq[0])))
        for step, color in enumerate(seq[1:], start=1):
            rr = seed_r + sweep * step
            if not (inner_r0 <= rr <= inner_r1):
                return None
            plan.append(([(rr, cc) for cc in range(inner_c0, inner_c1 + 1)], int(color)))
        return plan

    sweep = _infer_sweep(seed_c, inner_c0, inner_c1, extra_steps)
    if sweep is None:
        return None
    if dr > 0:
        initial = [(rr, seed_c) for rr in range(seed_r, inner_r1 + 1)]
    else:
        initial = [(rr, seed_c) for rr in range(inner_r0, seed_r + 1)]
    plan.append((initial, int(seq[0])))
    for step, color in enumerate(seq[1:], start=1):
        cc = seed_c + sweep * step
        if not (inner_c0 <= cc <= inner_c1):
            return None
        plan.append(([(rr, cc) for rr in range(inner_r0, inner_r1 + 1)], int(color)))
    return plan


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow striped bands inside framed boxes.

    The seed color is identified by the outside legend whose first color matches it.
    The seed→indicator offset gives the initial growth direction, and the only
    perpendicular side with enough room for the remaining legend colors determines
    which neighboring rows/columns the bounced segments occupy.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)
    legends = _find_legends(g, bg, frames)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = []
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                value = int(g[rr, cc])
                if value != bg:
                    interior.append((rr, cc, value))
        if len(interior) != 2:
            continue
        if abs(interior[0][0] - interior[1][0]) + abs(interior[0][1] - interior[1][1]) != 1:
            continue
        if interior[0][2] == interior[1][2]:
            continue

        candidates = []
        for seed_idx, ind_idx in ((0, 1), (1, 0)):
            seed_r, seed_c, seed_color = interior[seed_idx]
            ind_r, ind_c, _ = interior[ind_idx]
            if seed_color not in legends:
                continue
            seq = legends[seed_color]
            plan = _segment_plan(frame["bbox"], (seed_r, seed_c), (ind_r, ind_c), seq)
            if plan is not None:
                candidates.append((seed_idx, ind_idx, seq, plan))
        if len(candidates) != 1:
            continue

        for rr, cc, _ in interior:
            out[rr, cc] = bg
        _seed_idx, _ind_idx, _seq, plan = candidates[0]
        for cells, color in plan:
            for rr, cc in cells:
                out[rr, cc] = color
    return out


def _can_place(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rows, cols = reserved.shape
    if r0 < 1 or c0 < 1 or r0 + h >= rows or c0 + w >= cols:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(rows, r0 + h + pad)
    cc1 = min(cols, c0 + w + pad)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _reserve(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rows, cols = reserved.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(rows, r0 + h + pad)
    cc1 = min(cols, c0 + w + pad)
    reserved[rr0:rr1, cc0:cc1] = True


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _sample_seq(seed_color: int, seed_colors: set[int]) -> list[int]:
    length = random.randint(2, 4)
    pool = [c for c in NON_SPECIAL_COLORS if c not in seed_colors and c != seed_color]
    return [seed_color] + random.sample(pool, length - 1)


def _sample_box_spec(seed_color: int, seed_colors: set[int]) -> dict:
    seq = _sample_seq(seed_color, seed_colors)
    extra_steps = len(seq) - 1
    direction = random.choice(DIRS)
    legend_axis = random.choice(("h", "v"))
    indicator_pool = [c for c in NON_SPECIAL_COLORS if c != seed_color and c not in seed_colors]
    indicator_color = random.choice(indicator_pool)

    if direction[0] == 0:
        inner_h = random.randint(extra_steps + 1, max(extra_steps + 1, min(8, extra_steps + 5)))
        inner_w = random.randint(4, 8)
        sweep = random.choice((-1, 1))
        if sweep > 0:
            hi = min(extra_steps - 1, inner_h - 1 - extra_steps)
            seed_r = random.randint(0, hi)
        else:
            lo = max(extra_steps, inner_h - extra_steps)
            seed_r = random.randint(lo, inner_h - 1)
        if direction[1] > 0:
            seed_c = random.randint(0, inner_w - 2)
            indicator = (seed_r, seed_c + 1)
        else:
            seed_c = random.randint(1, inner_w - 1)
            indicator = (seed_r, seed_c - 1)
    else:
        inner_h = random.randint(4, 8)
        inner_w = random.randint(extra_steps + 1, max(extra_steps + 1, min(8, extra_steps + 5)))
        sweep = random.choice((-1, 1))
        if sweep > 0:
            hi = min(extra_steps - 1, inner_w - 1 - extra_steps)
            seed_c = random.randint(0, hi)
        else:
            lo = max(extra_steps, inner_w - extra_steps)
            seed_c = random.randint(lo, inner_w - 1)
        if direction[0] > 0:
            seed_r = random.randint(0, inner_h - 2)
            indicator = (seed_r + 1, seed_c)
        else:
            seed_r = random.randint(1, inner_h - 1)
            indicator = (seed_r - 1, seed_c)

    return {
        "seq": seq,
        "direction": direction,
        "sweep": sweep,
        "legend_axis": legend_axis,
        "indicator_color": indicator_color,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "seed_local": (seed_r, seed_c),
        "indicator_local": indicator,
    }


def _place_box(reserved: np.ndarray, spec: dict) -> tuple[int, int] | None:
    box_h = spec["inner_h"] + 2
    box_w = spec["inner_w"] + 2
    rows, cols = reserved.shape
    for _ in range(PLACE_RETRIES):
        r0 = random.randint(1, rows - box_h - 1)
        c0 = random.randint(1, cols - box_w - 1)
        if _can_place(reserved, r0, c0, box_h, box_w, pad=1):
            _reserve(reserved, r0, c0, box_h, box_w, pad=1)
            return r0, c0
    return None


def _place_legend(reserved: np.ndarray, spec: dict) -> tuple[int, int] | None:
    length = len(spec["seq"])
    legend_h = 1 if spec["legend_axis"] == "h" else length
    legend_w = length if spec["legend_axis"] == "h" else 1
    rows, cols = reserved.shape
    for _ in range(PLACE_RETRIES):
        r0 = random.randint(1, rows - legend_h - 1)
        c0 = random.randint(1, cols - legend_w - 1)
        if _can_place(reserved, r0, c0, legend_h, legend_w, pad=1):
            _reserve(reserved, r0, c0, legend_h, legend_w, pad=1)
            return r0, c0
    return None


def _render_input(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        r0, c0 = spec["box_pos"]
        box_h = spec["inner_h"] + 2
        box_w = spec["inner_w"] + 2
        _draw_frame(grid, r0, c0, box_h, box_w, FRAME)
        seed_r = r0 + 1 + spec["seed_local"][0]
        seed_c = c0 + 1 + spec["seed_local"][1]
        ind_r = r0 + 1 + spec["indicator_local"][0]
        ind_c = c0 + 1 + spec["indicator_local"][1]
        grid[seed_r, seed_c] = int(spec["seq"][0])
        grid[ind_r, ind_c] = int(spec["indicator_color"])

        lr0, lc0 = spec["legend_pos"]
        if spec["legend_axis"] == "h":
            for idx, color in enumerate(spec["seq"]):
                grid[lr0, lc0 + idx] = int(color)
        else:
            for idx, color in enumerate(spec["seq"]):
                grid[lr0 + idx, lc0] = int(color)
    return grid


def _apply_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0, c0 = spec["box_pos"]
        inner_r0, inner_c0 = r0 + 1, c0 + 1
        inner_r1 = inner_r0 + spec["inner_h"] - 1
        inner_c1 = inner_c0 + spec["inner_w"] - 1
        seed_r = inner_r0 + spec["seed_local"][0]
        seed_c = inner_c0 + spec["seed_local"][1]
        ind_r = inner_r0 + spec["indicator_local"][0]
        ind_c = inner_c0 + spec["indicator_local"][1]
        out[seed_r, seed_c] = BG
        out[ind_r, ind_c] = BG

        dr, dc = spec["direction"]
        seq = spec["seq"]
        if dc != 0:
            if dc > 0:
                out[seed_r, seed_c : inner_c1 + 1] = int(seq[0])
            else:
                out[seed_r, inner_c0 : seed_c + 1] = int(seq[0])
            for step, color in enumerate(seq[1:], start=1):
                rr = seed_r + spec["sweep"] * step
                out[rr, inner_c0 : inner_c1 + 1] = int(color)
        else:
            if dr > 0:
                out[seed_r : inner_r1 + 1, seed_c] = int(seq[0])
            else:
                out[inner_r0 : seed_r + 1, seed_c] = int(seq[0])
            for step, color in enumerate(seq[1:], start=1):
                cc = seed_c + spec["sweep"] * step
                out[inner_r0 : inner_r1 + 1, cc] = int(color)
    return out


def _valid_background(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    lookup = {int(v): int(c) for v, c in zip(values, counts)}
    bg_count = lookup.get(BG, 0)
    return all(color == BG or bg_count > count for color, count in lookup.items())


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(18, 28)
    cols = random.randint(18, 28)
    box_count = random.randint(1, 3)
    seed_colors = set(random.sample(list(NON_SPECIAL_COLORS), box_count))
    specs = [_sample_box_spec(seed_color, seed_colors) for seed_color in seed_colors]
    specs.sort(key=lambda item: (item["inner_h"] + 2) * (item["inner_w"] + 2), reverse=True)

    reserved = np.zeros((rows, cols), dtype=bool)
    for spec in specs:
        box_pos = _place_box(reserved, spec)
        if box_pos is None:
            return None
        spec["box_pos"] = box_pos
    for spec in specs:
        legend_pos = _place_legend(reserved, spec)
        if legend_pos is None:
            return None
        spec["legend_pos"] = legend_pos

    grid = _render_input(specs, (rows, cols))
    if not _valid_background(grid):
        return None
    expected = _apply_expected(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
