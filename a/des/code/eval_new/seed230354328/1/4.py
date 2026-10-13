from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SEED = 4
PALETTE = [1, 2, 3, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


def _same_color_component(grid: np.ndarray, start: tuple[int, int], visited: np.ndarray) -> list[tuple[int, int]]:
    color = int(grid[start])
    h, w = grid.shape
    q = deque([start])
    visited[start] = True
    pts: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        pts.append((r, c))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                visited[nr, nc] = True
                q.append((nr, nc))
    return pts


def _ring_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _paint_ring_if_black(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    h, w = out.shape
    for r, c in _ring_cells(r0, c0, r1, c1):
        if 0 <= r < h and 0 <= c < w and int(out[r, c]) == BG:
            out[r, c] = int(color)


def _frame_from_component(grid: np.ndarray, pts: list[tuple[int, int]], color: int) -> dict | None:
    pset = set(pts)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3 or height % 2 == 0 or width % 2 == 0:
        return None
    if len(pts) != 2 * (height + width) - 4:
        return None
    if pset != set(_ring_cells(r0, c0, r1, c1)):
        return None

    seed_r = (r0 + r1) // 2
    seed_c = (c0 + c1) // 2
    if int(grid[seed_r, seed_c]) != SEED:
        return None

    sequence: list[int] = []
    seq_cells: list[tuple[int, int]] = []
    rr = seed_r - 1
    while rr > r0:
        value = int(grid[rr, seed_c])
        if value in (BG, SEED, color):
            break
        sequence.append(value)
        seq_cells.append((rr, seed_c))
        rr -= 1
    if not (1 <= len(sequence) <= 3):
        return None

    allowed = {(seed_r, seed_c), *seq_cells}
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(grid[r, c]) != BG and (r, c) not in allowed:
                return None

    return {
        "bbox": (r0, c0, r1, c1),
        "area": height * width,
        "frame_color": int(color),
        "seed": (seed_r, seed_c),
        "sequence": [int(v) for v in sequence],
        "seq_cells": seq_cells,
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    frames: list[dict] = []
    for r in range(h):
        for c in range(w):
            value = int(grid[r, c])
            if visited[r, c] or value in (BG, SEED):
                continue
            comp = _same_color_component(grid, (r, c), visited)
            info = _frame_from_component(grid, comp, value)
            if info is not None:
                frames.append(info)
    return frames


def _render_from_frames(grid: np.ndarray, frames: list[dict], reverse: bool = False) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for frame in frames:
        for r, c in frame["seq_cells"]:
            out[r, c] = BG

    order = sorted(
        range(len(frames)),
        key=lambda i: (
            int(frames[i]["area"]),
            int(frames[i]["bbox"][0]),
            int(frames[i]["bbox"][1]),
        ),
        reverse=reverse,
    )

    for idx in order:
        frame = frames[idx]
        r0, c0, r1, c1 = frame["bbox"]
        for depth, color in enumerate(frame["sequence"], start=1):
            _paint_ring_if_black(out, r0 - depth, c0 - depth, r1 + depth, c1 + depth, color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each rectangular frame outward using the color command stack above its central seed.

    Input structure:
    - The background is black.
    - Several single-color, one-pixel-thick closed rectangular frames appear on the grid; each frame has a black interior.
    - The exact center of each frame contains one yellow seed pixel, and directly above that seed there is a contiguous vertical stack of 1-3 colored command pixels.

    Rule:
    - Treat every rectangular frame as a template and read the command stack from nearest-above the seed to farthest-above.
    - Erase those command pixels in the output, but keep the original frame and the yellow seed unchanged.
    - For command color k, paint the kth outward rectangular ring around the frame with that color.
    - If two grown rings compete for the same black cell, the frame with smaller rectangle area claims it first.

    Key insight:
    - The pixels above the yellow center are not decoration: they encode both the number of outward growth layers and the colors of those layers, while the growth shape is the original rectangle expanded ring by ring rather than a generic flood.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    if not frames:
        return g.copy()
    return _render_from_frames(g, frames, reverse=False)


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _spec_bbox(spec: dict, pos: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = pos
    return (r0, c0, r0 + spec["height"] - 1, c0 + spec["width"] - 1)


def _expanded_bbox(spec: dict, pos: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _spec_bbox(spec, pos)
    depth = spec["depth"]
    return (r0 - depth, c0 - depth, r1 + depth, c1 + depth)


def _candidate_positions(spec: dict, H: int, W: int) -> list[tuple[int, int]]:
    depth = spec["depth"]
    r_lo = depth
    r_hi = H - depth - spec["height"]
    c_lo = depth
    c_hi = W - depth - spec["width"]
    if r_lo > r_hi or c_lo > c_hi:
        return []
    return [(r0, c0) for r0 in range(r_lo, r_hi + 1) for c0 in range(c_lo, c_hi + 1)]


def _build_specs(rng: random.Random, count: int) -> list[dict] | None:
    lengths = [1, 2, 3] if count == 3 else list(rng.choice([(1, 2), (1, 3), (2, 3)]))
    rng.shuffle(lengths)
    frame_colors = rng.sample(PALETTE, count)
    width_cap = 9 if count == 2 else 7
    specs: list[dict] = []
    used_areas: set[int] = set()

    for idx, depth in enumerate(lengths):
        min_h = 2 * depth + 3
        height_choices = [h for h in range(min_h, min(11, min_h + 2) + 1, 2)]
        width_choices = [w for w in range(5, width_cap + 1, 2)]
        candidates = [(h, w) for h in height_choices for w in width_choices]
        rng.shuffle(candidates)
        chosen = None
        for height, width in candidates:
            area = height * width
            if area not in used_areas:
                chosen = (height, width)
                used_areas.add(area)
                break
        if chosen is None:
            return None
        frame_color = frame_colors[idx]
        seq_colors = rng.sample([c for c in PALETTE if c != frame_color], depth)
        specs.append(
            {
                "height": chosen[0],
                "width": chosen[1],
                "depth": depth,
                "frame_color": frame_color,
                "sequence": seq_colors,
                "area": chosen[0] * chosen[1],
            }
        )
    return specs


def _place_nonoverlap(specs: list[dict], H: int, W: int, rng: random.Random) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int] | None] = [None] * len(specs)
    placed_boxes: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(specs)), key=lambda i: (specs[i]["height"] + 2 * specs[i]["depth"]) * (specs[i]["width"] + 2 * specs[i]["depth"]), reverse=True)

    for idx in order:
        candidates = _candidate_positions(specs[idx], H, W)
        rng.shuffle(candidates)
        chosen = None
        for pos in candidates:
            box = _expanded_bbox(specs[idx], pos)
            if any(_boxes_intersect(box, other) for other in placed_boxes):
                continue
            chosen = pos
            placed_boxes.append(box)
            break
        if chosen is None:
            return None
        positions[idx] = chosen
    return [pos for pos in positions if pos is not None]


def _choose_overlap_pair(specs: list[dict], rng: random.Random) -> tuple[int, int]:
    ordered = sorted(range(len(specs)), key=lambda i: specs[i]["area"])
    if len(ordered) == 2:
        return ordered[0], ordered[1]
    return ordered[0], rng.choice(ordered[1:])


def _place_overlap_pair(
    specs: list[dict], H: int, W: int, rng: random.Random
) -> tuple[dict[int, tuple[int, int]], tuple[int, int]] | None:
    a_idx, b_idx = _choose_overlap_pair(specs, rng)
    spec_a = specs[a_idx]
    spec_b = specs[b_idx]

    depth_pairs = [
        (da, db)
        for da in range(1, spec_a["depth"] + 1)
        for db in range(1, spec_b["depth"] + 1)
        if spec_a["sequence"][da - 1] != spec_b["sequence"][db - 1]
    ]
    if not depth_pairs:
        return None

    for _ in range(80):
        da, db = rng.choice(depth_pairs)
        orientation = rng.choice(["horizontal", "vertical"])
        swap_order = rng.choice([False, True])
        first_idx, second_idx = (a_idx, b_idx) if not swap_order else (b_idx, a_idx)
        first_spec = specs[first_idx]
        second_spec = specs[second_idx]
        df, ds = (da, db) if not swap_order else (db, da)
        gap = df + ds - 1

        if orientation == "horizontal":
            center_min = max(
                first_spec["depth"] + first_spec["height"] // 2,
                second_spec["depth"] + second_spec["height"] // 2,
            )
            center_max = min(
                H - 1 - (first_spec["depth"] + first_spec["height"] // 2),
                H - 1 - (second_spec["depth"] + second_spec["height"] // 2),
            )
            c0_min = max(first_spec["depth"], second_spec["depth"] - first_spec["width"] - gap)
            c0_max = W - (first_spec["width"] + gap + second_spec["width"] + second_spec["depth"])
            if center_min > center_max or c0_min > c0_max:
                continue
            center = rng.randint(center_min, center_max)
            c0_first = rng.randint(c0_min, c0_max)
            c0_second = c0_first + first_spec["width"] + gap
            r0_first = center - first_spec["height"] // 2
            r0_second = center - second_spec["height"] // 2
        else:
            center_min = max(
                first_spec["depth"] + first_spec["width"] // 2,
                second_spec["depth"] + second_spec["width"] // 2,
            )
            center_max = min(
                W - 1 - (first_spec["depth"] + first_spec["width"] // 2),
                W - 1 - (second_spec["depth"] + second_spec["width"] // 2),
            )
            r0_min = max(first_spec["depth"], second_spec["depth"] - first_spec["height"] - gap)
            r0_max = H - (first_spec["height"] + gap + second_spec["height"] + second_spec["depth"])
            if center_min > center_max or r0_min > r0_max:
                continue
            center = rng.randint(center_min, center_max)
            r0_first = rng.randint(r0_min, r0_max)
            r0_second = r0_first + first_spec["height"] + gap
            c0_first = center - first_spec["width"] // 2
            c0_second = center - second_spec["width"] // 2

        pos_first = (r0_first, c0_first)
        pos_second = (r0_second, c0_second)
        bbox_first = _spec_bbox(first_spec, pos_first)
        bbox_second = _spec_bbox(second_spec, pos_second)
        if _boxes_intersect(bbox_first, bbox_second):
            continue
        exp_first = _expanded_bbox(first_spec, pos_first)
        exp_second = _expanded_bbox(second_spec, pos_second)
        if not _boxes_intersect(exp_first, exp_second):
            continue

        positions = {first_idx: pos_first, second_idx: pos_second}
        return positions, (a_idx, b_idx)
    return None


def _place_with_overlap(specs: list[dict], H: int, W: int, rng: random.Random) -> list[tuple[int, int]] | None:
    pair_result = _place_overlap_pair(specs, H, W, rng)
    if pair_result is None:
        return None
    positions_by_idx, _pair = pair_result
    positions: list[tuple[int, int] | None] = [None] * len(specs)
    for idx, pos in positions_by_idx.items():
        positions[idx] = pos

    occupied_boxes = [_expanded_bbox(specs[idx], pos) for idx, pos in positions_by_idx.items()]
    remaining = [i for i in range(len(specs)) if positions[i] is None]
    remaining.sort(key=lambda i: (specs[i]["height"] + 2 * specs[i]["depth"]) * (specs[i]["width"] + 2 * specs[i]["depth"]), reverse=True)

    for idx in remaining:
        candidates = _candidate_positions(specs[idx], H, W)
        rng.shuffle(candidates)
        chosen = None
        for pos in candidates:
            box = _expanded_bbox(specs[idx], pos)
            if any(_boxes_intersect(box, other) for other in occupied_boxes):
                continue
            chosen = pos
            occupied_boxes.append(box)
            break
        if chosen is None:
            return None
        positions[idx] = chosen

    return [pos for pos in positions if pos is not None]


def _draw_input_grid(H: int, W: int, specs: list[dict], positions: list[tuple[int, int]]) -> tuple[np.ndarray, list[dict]]:
    grid = np.zeros((H, W), dtype=int)
    frames: list[dict] = []
    for spec, pos in zip(specs, positions):
        r0, c0, r1, c1 = _spec_bbox(spec, pos)
        for r, c in _ring_cells(r0, c0, r1, c1):
            grid[r, c] = spec["frame_color"]
        seed_r = r0 + spec["height"] // 2
        seed_c = c0 + spec["width"] // 2
        grid[seed_r, seed_c] = SEED
        seq_cells: list[tuple[int, int]] = []
        for depth, color in enumerate(spec["sequence"], start=1):
            grid[seed_r - depth, seed_c] = color
            seq_cells.append((seed_r - depth, seed_c))
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "area": spec["area"],
                "frame_color": spec["frame_color"],
                "seed": (seed_r, seed_c),
                "sequence": list(spec["sequence"]),
                "seq_cells": seq_cells,
            }
        )
    return grid, frames


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.choice([2, 3])
    overlap_mode = rng.choice([False, True])
    specs = _build_specs(rng, count)
    if specs is None:
        return None

    if count == 3 and not overlap_mode:
        H = rng.randint(26, 30)
        W = rng.randint(26, 30)
    elif count == 3:
        H = rng.randint(24, 30)
        W = rng.randint(24, 30)
    else:
        H = rng.randint(20, 30)
        W = rng.randint(20, 30)

    positions = _place_with_overlap(specs, H, W, rng) if overlap_mode else _place_nonoverlap(specs, H, W, rng)
    if positions is None:
        return None

    grid, frames = _draw_input_grid(H, W, specs, positions)
    expected = _render_from_frames(grid, frames, reverse=False)
    alternate = _render_from_frames(grid, frames, reverse=True)

    if overlap_mode:
        if np.array_equal(expected, alternate):
            return None
    else:
        if not np.array_equal(expected, alternate):
            return None

    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
