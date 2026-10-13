import random
from collections import Counter

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRECTIONS = ("up", "down", "left", "right")
FRAME_COLOR = 5
MAX_SPIKE_LEN = 4
GEN_RETRIES = 300


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular_frame(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    if not np.all(grid[r0, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == color):
        return False
    if r1 - r0 > 1 and c1 - c0 > 1 and np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == color):
        return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in sorted(set(map(int, grid.flat)) - {bg}):
        mask = grid == color
        for comp in _components(mask):
            if _is_rectangular_frame(grid, color, comp):
                frames.append({"color": color, "bbox": _bbox(comp)})
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _mean_coord(coords: list[tuple[int, int]]) -> tuple[float, float]:
    rows = np.array([r for r, _ in coords], dtype=float)
    cols = np.array([c for _, c in coords], dtype=float)
    return float(rows.mean()), float(cols.mean())


def _classify_direction(coords: list[tuple[int, int]], center_r: float, center_c: float) -> str:
    mean_r, mean_c = _mean_coord(coords)
    dr = mean_r - center_r
    dc = mean_c - center_c
    if abs(dr) > abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _projection_index(
    coords: list[tuple[int, int]],
    use_cols: bool,
    axis_center: float,
) -> int:
    values = [c if use_cols else r for r, c in coords]
    counts = Counter(values)
    best_count = max(counts.values())
    centroid = sum(values) / len(values)
    candidates = [value for value, count in counts.items() if count == best_count]
    return min(candidates, key=lambda value: (abs(value - centroid), abs(value - axis_center), value))


def _draw_spike(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    coords: list[tuple[int, int]],
    color: int,
    direction: str,
) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = bbox
    length = len(coords)
    frame_center_r = (r0 + r1) / 2.0
    frame_center_c = (c0 + c1) / 2.0
    if direction == "up":
        col = _projection_index(coords, use_cols=True, axis_center=frame_center_c)
        for step in range(1, length + 1):
            rr = r0 - step
            if 0 <= rr < h:
                out[rr, col] = color
    elif direction == "down":
        col = _projection_index(coords, use_cols=True, axis_center=frame_center_c)
        for step in range(1, length + 1):
            rr = r1 + step
            if 0 <= rr < h:
                out[rr, col] = color
    elif direction == "left":
        row = _projection_index(coords, use_cols=False, axis_center=frame_center_r)
        for step in range(1, length + 1):
            cc = c0 - step
            if 0 <= cc < w:
                out[row, cc] = color
    else:
        row = _projection_index(coords, use_cols=False, axis_center=frame_center_r)
        for step in range(1, length + 1):
            cc = c1 + step
            if 0 <= cc < w:
                out[row, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow exterior spikes on a rectangular frame according to the side-bias and size of interior seed blocks.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for frame in _find_frames(g, bg):
        r0, c0, r1, c1 = frame["bbox"]
        center_r = (r0 + r1) / 2.0
        center_c = (c0 + c1) / 2.0
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        for color in sorted(set(map(int, inner.flat)) - {bg, frame["color"]}):
            mask = inner == color
            for local_comp in _components(mask):
                coords = [(r0 + 1 + lr, c0 + 1 + lc) for lr, lc in local_comp]
                direction = _classify_direction(coords, center_r, center_c)
                _draw_spike(out, frame["bbox"], coords, color, direction)
    return out


def _local_direction(cells: list[tuple[int, int]], ih: int, iw: int) -> str:
    return _classify_direction(cells, (ih - 1) / 2.0, (iw - 1) / 2.0)


def _candidate_bars(direction: str, length: int, ih: int, iw: int) -> list[list[tuple[int, int]]]:
    candidates: list[list[tuple[int, int]]] = []
    if direction in {"up", "down"}:
        for c in range(1, iw - 1):
            for r0 in range(1, ih - length):
                cells = [(r0 + i, c) for i in range(length)]
                if _local_direction(cells, ih, iw) == direction:
                    candidates.append(cells)
    else:
        for r in range(1, ih - 1):
            for c0 in range(1, iw - length):
                cells = [(r, c0 + i) for i in range(length)]
                if _local_direction(cells, ih, iw) == direction:
                    candidates.append(cells)
    return candidates


def _touches_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return any(blocked[r, c] for r, c in cells)


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _place_seeds_in_frame(rng: random.Random, ih: int, iw: int) -> list[dict] | None:
    for _ in range(80):
        n_seeds = rng.choice([1, 2, 2, 3, 3])
        directions = rng.sample(list(ALL_DIRECTIONS), n_seeds)
        blocked = np.zeros((ih, iw), dtype=bool)
        seeds: list[dict] = []
        ok = True
        for direction in directions:
            length = rng.randint(2, 4)
            candidates = _candidate_bars(direction, length, ih, iw)
            rng.shuffle(candidates)
            placed = None
            for cells in candidates:
                if not _touches_blocked(blocked, cells):
                    placed = cells
                    break
            if placed is None:
                ok = False
                break
            _mark_blocked(blocked, placed)
            seeds.append({"direction": direction, "color": None, "cells": placed})
        if ok:
            return seeds
    return None


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _draw_seed(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], seed: dict) -> None:
    r0, c0, _r1, _c1 = frame_bbox
    color = seed["color"]
    for lr, lc in seed["cells"]:
        grid[r0 + 1 + lr, c0 + 1 + lc] = color


def _draw_expected_spike(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], seed: dict) -> None:
    r0, c0, r1, c1 = frame_bbox
    color = seed["color"]
    cells = [(r0 + 1 + lr, c0 + 1 + lc) for lr, lc in seed["cells"]]
    _draw_spike(grid, (r0, c0, r1, c1), cells, color, seed["direction"])


def _place_frames(
    rng: random.Random,
    h: int,
    w: int,
    frame_specs: list[dict],
) -> list[tuple[int, int, int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int, int, int]] = []
    for spec in frame_specs:
        fh = spec["ih"] + 2
        fw = spec["iw"] + 2
        placed = None
        row_choices = list(range(MAX_SPIKE_LEN, h - MAX_SPIKE_LEN - fh + 1))
        col_choices = list(range(MAX_SPIKE_LEN, w - MAX_SPIKE_LEN - fw + 1))
        rng.shuffle(row_choices)
        rng.shuffle(col_choices)
        for r0 in row_choices:
            for c0 in col_choices:
                er0 = r0 - MAX_SPIKE_LEN
                ec0 = c0 - MAX_SPIKE_LEN
                er1 = r0 + fh - 1 + MAX_SPIKE_LEN
                ec1 = c0 + fw - 1 + MAX_SPIKE_LEN
                if occupied[er0 : er1 + 1, ec0 : ec1 + 1].any():
                    continue
                occupied[er0 : er1 + 1, ec0 : ec1 + 1] = True
                placed = (r0, c0, r0 + fh - 1, c0 + fw - 1)
                break
            if placed is not None:
                break
        if placed is None:
            return None
        placements.append(placed)
    return placements


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    h = rng.randint(22, 32)
    w = rng.randint(22, 32)
    bg_choices = [color for color in range(10) if color != FRAME_COLOR]
    bg = rng.choice(bg_choices)
    n_frames = rng.choice([1, 1, 2, 2, 3])

    frame_specs: list[dict] = []
    for _ in range(n_frames):
        ih = rng.randint(8, 12)
        iw = rng.randint(8, 12)
        seeds = _place_seeds_in_frame(rng, ih, iw)
        if seeds is None:
            return None
        frame_specs.append({"ih": ih, "iw": iw, "seeds": seeds})

    placements = _place_frames(rng, h, w, frame_specs)
    if placements is None:
        return None

    inp = np.full((h, w), bg, dtype=int)
    out = inp.copy()

    palette = [color for color in range(10) if color not in {bg, FRAME_COLOR}]
    for spec, bbox in zip(frame_specs, placements):
        if len(spec["seeds"]) > len(palette):
            return None
        seed_colors = rng.sample(palette, len(spec["seeds"]))
        for seed, color in zip(spec["seeds"], seed_colors):
            seed["color"] = color

        _draw_frame(inp, bbox, FRAME_COLOR)
        _draw_frame(out, bbox, FRAME_COLOR)
        for seed in spec["seeds"]:
            _draw_seed(inp, bbox, seed)
            _draw_seed(out, bbox, seed)
        for seed in spec["seeds"]:
            _draw_expected_spike(out, bbox, seed)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        pair = _gen_attempt(rng)
        if pair is not None:
            return pair[0]
    raise RuntimeError("generate_input: exhausted retries")
