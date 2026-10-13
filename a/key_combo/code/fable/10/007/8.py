from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BOX = 3
BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 2000
MIN_SWITCHES = 1


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _rotate_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, -int(turns))


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mask.ravel())


def _connected_4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(coords)


def _shape_pool() -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    for bits in range(1, 1 << (BOX * BOX)):
        mask = np.zeros((BOX, BOX), dtype=bool)
        for bit in range(BOX * BOX):
            if (bits >> bit) & 1:
                mask[bit // BOX, bit % BOX] = True
        count = int(mask.sum())
        if not (4 <= count <= 6):
            continue
        if not _connected_4(mask):
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        rotations = [_rotate_cw(mask, k) for k in range(4)]
        if len({_mask_key(rot) for rot in rotations}) != 4:
            continue
        pool.append(mask)
    assert pool, "expected at least one usable footprint shape"
    return pool


SHAPE_POOL = _shape_pool()
GEN_SHAPE = SHAPE_POOL[0]
GEN_TEMPLATES = [
    (0, 8, 0, 0),
    (0, 8, 1, 2),
    (0, 8, 2, 3),
    (0, 8, 3, 1),
    (0, 9, 0, 2),
    (0, 9, 1, 3),
    (0, 9, 2, 0),
    (0, 9, 3, 1),
    (1, 9, 0, 1),
    (1, 9, 2, 3),
    (2, 10, 1, 1),
    (2, 10, 3, 0),
]


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0 = min(rs)
            r1 = max(rs)
            c0 = min(cs)
            c1 = max(cs)
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            components.append(
                {
                    "color": color,
                    "cells": tuple(cells),
                    "bbox": (r0, c0, r1, c1),
                    "patch": patch.copy(),
                }
            )
    return components


def _canonical_orientation_map(mask: np.ndarray) -> dict[tuple[int, ...], int]:
    rotations = [_rotate_cw(mask, k) for k in range(4)]
    canonical = min(rotations, key=_mask_key)
    return {_mask_key(_rotate_cw(canonical, k)): k for k in range(4)}


def _parse_stamps(grid: np.ndarray) -> tuple[int, int, list[dict[str, int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    components = _find_components(g, bg)
    assert len(components) >= 4, "expected multiple footprint stamps"

    colors = {int(comp["color"]) for comp in components}
    assert len(colors) == 1, "expected all footprints to share one color"
    (color,) = tuple(colors)

    first_patch = np.asarray(components[0]["patch"])
    assert first_patch.shape == (BOX, BOX), "expected a 3x3 footprint frame"
    orientation_map = _canonical_orientation_map(first_patch != bg)
    canonical_mask = min([_rotate_cw(first_patch != bg, k) for k in range(4)], key=_mask_key)

    stamps: list[dict[str, int]] = []
    top_rows = set()
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]  # type: ignore[misc]
        patch = np.asarray(comp["patch"])
        assert patch.shape == (BOX, BOX), "all footprints must occupy a full 3x3 frame"
        mask = patch != bg
        key = _mask_key(mask)
        assert key in orientation_map, "all components must be rotated copies of one footprint"
        top_rows.add(int(r0))
        stamps.append({"r": int(r0), "c": int(c0), "o": int(orientation_map[key])})

    assert len(top_rows) == 1, "all footprints must lie on one row band"
    stamps.sort(key=lambda item: (item["c"], item["o"]))
    return bg, color, stamps, canonical_mask.astype(bool)


def _ordered_indices(stamps: list[dict[str, int]], indices: tuple[int, ...], sign: int) -> list[int]:
    return sorted(indices, key=lambda idx: sign * stamps[idx]["c"])


def _sequence_matches(
    stamps: list[dict[str, int]],
    indices: tuple[int, ...],
    sign: int,
    *,
    require_spacing: bool,
    require_rotation: bool,
) -> bool:
    if len(indices) < 2:
        return False
    order = _ordered_indices(stamps, indices, sign)
    gaps = [
        sign * (stamps[order[i + 1]]["c"] - stamps[order[i]]["c"]) - BOX
        for i in range(len(order) - 1)
    ]
    if any(gap <= 0 for gap in gaps):
        return False
    if require_spacing and any(gaps[i + 1] != gaps[i] + 1 for i in range(len(gaps) - 1)):
        return False

    turns = [stamps[idx]["o"] for idx in order]
    if require_rotation and any((turns[i + 1] - turns[i]) % 4 != 1 for i in range(len(turns) - 1)):
        return False
    return True


def _next_stamp(
    grid_shape: tuple[int, int],
    stamps: list[dict[str, int]],
    indices: tuple[int, ...],
    sign: int,
) -> dict[str, int] | None:
    order = _ordered_indices(stamps, indices, sign)
    gaps = [
        sign * (stamps[order[i + 1]]["c"] - stamps[order[i]]["c"]) - BOX
        for i in range(len(order) - 1)
    ]
    if any(gap <= 0 for gap in gaps):
        return None
    next_gap = gaps[-1] + 1
    last = stamps[order[-1]]
    next_c = last["c"] + sign * (BOX + next_gap)
    next_r = last["r"]
    h, w = grid_shape
    if not (0 <= next_r <= h - BOX and 0 <= next_c <= w - BOX):
        return None
    return {"r": next_r, "c": next_c, "o": (last["o"] + 1) % 4}


def _enumerate_partitions(
    stamps: list[dict[str, int]],
    *,
    require_spacing: bool,
    require_rotation: bool,
) -> list[dict[str, object]]:
    n = len(stamps)
    assert n >= 4
    solutions: list[dict[str, object]] = []
    ascending = tuple(range(n))
    for sign in (1, -1):
        for mask in range(1 << (n - 1)):
            seq0 = [ascending[0]]
            seq1: list[int] = []
            for bit_index, idx in enumerate(ascending[1:]):
                if (mask >> bit_index) & 1:
                    seq1.append(idx)
                else:
                    seq0.append(idx)
            if min(len(seq0), len(seq1)) < 2:
                continue
            tup0 = tuple(sorted(seq0))
            tup1 = tuple(sorted(seq1))
            if not _sequence_matches(
                stamps, tup0, sign, require_spacing=require_spacing, require_rotation=require_rotation
            ):
                continue
            if not _sequence_matches(
                stamps, tup1, sign, require_spacing=require_spacing, require_rotation=require_rotation
            ):
                continue
            solutions.append({"sign": sign, "seqs": (tup0, tup1)})
    return solutions


def _full_solutions(grid_shape: tuple[int, int], stamps: list[dict[str, int]]) -> list[dict[str, object]]:
    occupied = {(stamp["r"], stamp["c"]) for stamp in stamps}
    solutions: list[dict[str, object]] = []
    for part in _enumerate_partitions(stamps, require_spacing=True, require_rotation=True):
        sign = int(part["sign"])
        seq0, seq1 = part["seqs"]  # type: ignore[misc]
        nxt0 = _next_stamp(grid_shape, stamps, seq0, sign)
        nxt1 = _next_stamp(grid_shape, stamps, seq1, sign)
        if nxt0 is None or nxt1 is None:
            continue
        if (nxt0["r"], nxt0["c"]) in occupied or (nxt1["r"], nxt1["c"]) in occupied:
            continue
        if abs(nxt0["c"] - nxt1["c"]) < BOX:
            continue
        solutions.append({"sign": sign, "seqs": (seq0, seq1), "nexts": (nxt0, nxt1)})
    return solutions


def _paint_patch(grid: np.ndarray, mask: np.ndarray, r: int, c: int, color: int, orientation: int) -> None:
    patch = _rotate_cw(mask, orientation)
    grid[r : r + BOX, c : c + BOX][patch] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Split the mixed row of footprints into two rotating accelerating walks and stamp one next footprint for each.

    Input structure:
    - The background is black, and every foreground component is the same single-color footprint sprite.
    - Each footprint occupies an implicit 3x3 frame, all frames lie on the same row band, and the four possible orientations are quarter-turn rotations of one non-rotationally-symmetric shape.
    - The visible row is a mixture of two left-to-right or two right-to-left walking sequences using the same sprite color.

    Rule:
    - Find the unique split into two groups such that, within each group and along its walking direction, consecutive frame gaps increase by exactly 1 and orientations advance by one clockwise quarter-turn.
    - For each recovered sequence, place the next footprint in the same direction using the next larger gap and the next clockwise rotation.
    - Keep all existing footprints unchanged and return the full grid with the two new non-overlapping stamps added.

    Key insight:
    - The row only looks irregular when read as one timeline; the hidden regularity appears only after jointly splitting the mixed footprints into two sequences that agree on both spacing growth and rotation cycle.
    """
    bg, color, stamps, canonical_mask = _parse_stamps(grid)
    solutions = _full_solutions(np.asarray(grid).shape, stamps)
    assert len(solutions) == 1, "expected a unique valid split"
    out = np.asarray(grid, dtype=int).copy()
    for nxt in solutions[0]["nexts"]:  # type: ignore[index]
        _paint_patch(out, canonical_mask, nxt["r"], nxt["c"], color, nxt["o"])
    return out


def _sequence_positions(start: int, first_gap: int, length: int) -> list[int]:
    positions = [int(start)]
    for i in range(length - 1):
        positions.append(positions[-1] + BOX + first_gap + i)
    return positions


def _switch_count(labels: list[str]) -> int:
    return sum(labels[i] != labels[i + 1] for i in range(len(labels) - 1))


def _actual_c(t: int, *, sign: int, t_min: int, t_max: int, margin: int) -> int:
    if sign == 1:
        return int(t - t_min + margin)
    return int(t_max - t + margin)


def _render_instance(
    shape_mask: np.ndarray,
    color: int,
    sign: int,
    seq_a: tuple[list[int], list[int]],
    seq_b: tuple[list[int], list[int]],
    *,
    row_margin: int,
    side_margin: int,
    bottom_margin: int,
    canvas_positions: list[int] | None = None,
) -> np.ndarray:
    pos_a, ori_a = seq_a
    pos_b, ori_b = seq_b
    all_t = list(pos_a + pos_b) if canvas_positions is None else list(canvas_positions)
    t_min = min(all_t)
    t_max = max(all_t)
    width = (t_max - t_min) + BOX + side_margin * 2
    height = row_margin + BOX + bottom_margin
    grid = np.zeros((height, width), dtype=int)
    row = row_margin

    for t, o in zip(pos_a, ori_a):
        c = _actual_c(t, sign=sign, t_min=t_min, t_max=t_max, margin=side_margin)
        _paint_patch(grid, shape_mask, row, c, color, o)
    for t, o in zip(pos_b, ori_b):
        c = _actual_c(t, sign=sign, t_min=t_min, t_max=t_max, margin=side_margin)
        _paint_patch(grid, shape_mask, row, c, color, o)
    return grid


def _rotation_ambiguity(stamps: list[dict[str, int]]) -> bool:
    return len(_enumerate_partitions(stamps, require_spacing=False, require_rotation=True)) >= 2


def _global_orientation_misdirection(left_to_right: list[tuple[int, str, int]]) -> bool:
    for i in range(len(left_to_right) - 1):
        _, label_a, ori_a = left_to_right[i]
        _, label_b, ori_b = left_to_right[i + 1]
        if label_a != label_b and (ori_b - ori_a) % 4 == 1:
            return True
    return False


def _overlapping_gap_ranges(first_gap_a: int, length_a: int, first_gap_b: int, length_b: int) -> bool:
    range_a = set(range(first_gap_a, first_gap_a + length_a - 1))
    range_b = set(range(first_gap_b, first_gap_b + length_b - 1))
    return bool(range_a & range_b)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    shape_mask = np.array(GEN_SHAPE, copy=True)
    color = int(rng.choice(PALETTE))
    sign = rng.choice([1, -1])
    start_a, start_b, start_ori_a, start_ori_b = rng.choice(GEN_TEMPLATES)
    len_a, len_b = 2, 3
    first_gap_a = 1
    first_gap_b = 1
    current_gap_a = 1
    current_gap_b = 2
    pos_a = _sequence_positions(start_a, first_gap_a, len_a)
    pos_b = _sequence_positions(start_b, first_gap_b, len_b)
    ori_a = [(start_ori_a + i) % 4 for i in range(len_a)]
    ori_b = [(start_ori_b + i) % 4 for i in range(len_b)]
    canonical_shape = min([_rotate_cw(shape_mask, k) for k in range(4)], key=_mask_key)
    existing = pos_a + pos_b
    next_a = pos_a[-1] + BOX + current_gap_a + 1
    next_b = pos_b[-1] + BOX + current_gap_b + 1
    future = [next_a, next_b]
    abstract_all = existing + future

    row_margin = rng.randint(0, 2)
    bottom_margin = rng.randint(0, 2)
    side_margin = 1
    t_min = min(abstract_all)
    t_max = max(abstract_all)
    width = (t_max - t_min) + BOX + side_margin * 2
    height = row_margin + BOX + bottom_margin
    if width > 30 or height > 30:
        return None

    grid = _render_instance(
        shape_mask,
        color,
        sign,
        (pos_a, ori_a),
        (pos_b, ori_b),
        row_margin=row_margin,
        side_margin=side_margin,
        bottom_margin=bottom_margin,
        canvas_positions=abstract_all,
    )
    try:
        bg, parsed_color, stamps, parsed_shape = _parse_stamps(grid)
    except Exception:
        return None
    if bg != BG or parsed_color != color or not np.array_equal(parsed_shape, canonical_shape):
        return None
    if not _rotation_ambiguity(stamps):
        return None
    solutions = _full_solutions(grid.shape, stamps)
    if len(solutions) != 1:
        return None
    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if np.count_nonzero(out != grid) != int(shape_mask.sum()) * 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wrong-footprints instance")
