from __future__ import annotations

import random
from typing import Iterable

import numpy as np


ARC_MAX = 30
ANCHOR_COLOR = 4
RING_COLOR_POOL = [1, 2, 3, 5, 6, 7, 8, 9]
SIDE_ORDER = ("top", "right", "bottom", "left")
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _border_positions(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    positions: list[tuple[int, int]] = []

    for c in range(c0, c1 + 1):
        positions.append((r0, c))
    for r in range(r0 + 1, r1):
        positions.append((r, c1))
    if r1 > r0:
        for c in range(c1, c0 - 1, -1):
            positions.append((r1, c))
    if c1 > c0:
        for r in range(r1 - 1, r0, -1):
            positions.append((r, c0))
    return positions


def _side_midpoint_positions(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    positions: list[tuple[int, int]] = []
    if side in ("top", "bottom"):
        row = r0 if side == "top" else r1
        center = (c0 + c1) / 2.0
        best = None
        for c in range(c0 + 1, c1):
            dist = abs(c - center)
            if best is None or dist < best:
                best = dist
                positions = [(row, c)]
            elif dist == best:
                positions.append((row, c))
    else:
        col = c1 if side == "right" else c0
        center = (r0 + r1) / 2.0
        best = None
        for r in range(r0 + 1, r1):
            dist = abs(r - center)
            if best is None or dist < best:
                best = dist
                positions = [(r, col)]
            elif dist == best:
                positions.append((r, col))
    return positions


def _gap_side(bbox: tuple[int, int, int, int], gap_pos: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    r, c = gap_pos
    if r == r0 and c0 < c < c1:
        return "top"
    if c == c1 and r0 < r < r1:
        return "right"
    if r == r1 and c0 < c < c1:
        return "bottom"
    if c == c0 and r0 < r < r1:
        return "left"
    raise AssertionError(f"gap must be on a side, got {gap_pos} for bbox {bbox}")


def _rotate_choice(length: int, gap_idx: int, target_indices: Iterable[int]) -> tuple[int, int]:
    best_key = None
    best_shift = 0
    best_target = gap_idx
    for target in sorted(set(int(x) for x in target_indices)):
        cw = (target - gap_idx) % length
        ccw = (gap_idx - target) % length
        if cw <= ccw:
            shift = int(cw)
            cost = int(cw)
            ccw_flag = 0
        else:
            shift = -int(ccw)
            cost = int(ccw)
            ccw_flag = 1
        key = (cost, ccw_flag, target)
        if best_key is None or key < best_key:
            best_key = key
            best_shift = shift
            best_target = target
    return best_shift, best_target


def _extract_structure(grid: np.ndarray) -> tuple[dict, list[dict]]:
    bg = _get_bg(grid)
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]

    anchor_candidates: list[dict] = []
    rings: list[dict] = []
    for color in colors:
        pts = np.argwhere(grid == color)
        assert pts.size > 0
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        bbox = (r0, c0, r1, c1)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        mask = grid[r0 : r1 + 1, c0 : c1 + 1] == color

        if bool(mask.all()):
            anchor_candidates.append({"color": color, "bbox": bbox, "shape": (h, w)})
            continue

        border = _border_positions(bbox)
        border_set = set(border)
        pts_set = {tuple(map(int, p)) for p in pts}
        if pts_set.issubset(border_set) and len(pts_set) == len(border) - 1:
            gap_pos = next(pos for pos in border if pos not in pts_set)
            index_of = {pos: idx for idx, pos in enumerate(border)}
            rings.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "positions": border,
                    "index_of": index_of,
                    "gap_idx": index_of[gap_pos],
                    "gap_pos": gap_pos,
                }
            )
            continue

        raise AssertionError(f"unexpected non-background object for color {color}")

    assert len(anchor_candidates) == 1, f"expected one anchor, got {len(anchor_candidates)}"
    anchor = anchor_candidates[0]
    rings = sorted(rings, key=lambda ring: (ring["bbox"][2] - ring["bbox"][0], ring["bbox"][3] - ring["bbox"][1]))
    assert rings, "expected at least one ring"
    return anchor, rings


def _inner_target_positions(ring_bbox: tuple[int, int, int, int], anchor_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    ar0, ac0, ar1, ac1 = anchor_bbox
    center_r = (ar0 + ar1) / 2.0
    center_c = (ac0 + ac1) / 2.0
    r0, c0, r1, c1 = ring_bbox

    side_distance = {
        "top": abs(center_r - r0),
        "right": abs(center_c - c1),
        "bottom": abs(center_r - r1),
        "left": abs(center_c - c0),
    }
    min_distance = min(side_distance.values())
    positions: list[tuple[int, int]] = []
    for side in SIDE_ORDER:
        if side_distance[side] == min_distance:
            positions.extend(_side_midpoint_positions(ring_bbox, side))
    assert positions
    return positions


def _aligned_target_positions(ring_bbox: tuple[int, int, int, int], ref_gap: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = ring_bbox
    rr, cc = ref_gap
    positions: list[tuple[int, int]] = []
    if r0 <= rr <= r1:
        positions.append((rr, c0))
        positions.append((rr, c1))
    if c0 <= cc <= c1:
        positions.append((r0, cc))
        positions.append((r1, cc))

    border_set = set(_border_positions(ring_bbox))
    unique = []
    seen = set()
    for pos in positions:
        if pos in border_set and pos not in seen:
            seen.add(pos)
            unique.append(pos)
    assert unique
    return unique


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each nested rectangular ring so its one-cell gap aligns step by step from the center outward.

    Input structure:
    - The background is black.
    - Several one-pixel-thick colored rectangular rings are nested concentrically with black spacing between rings.
    - Every ring has exactly one single-cell gap on one of its sides.
    - The middle object is a solid yellow anchor block of size 2x2 or 3x3 with no gap.

    Rule:
    - Detect the anchor block and order the rings from inner to outer.
    - For the innermost ring, treat the anchor center as the reference and rotate the ring by the fewest steps so the gap reaches the nearest side midpoint candidate.
    - For each later ring, use the already-moved inner gap as the new reference and rotate the current ring by the fewest steps so its gap lands on a border cell in the same row or the same column.
    - Ring motion is cyclic along the clockwise border sequence that starts at the ring's top-left corner; if clockwise and counterclockwise are tied, choose clockwise.

    Key insight:
    - The rings cannot be solved independently: each outer rotation depends on the inner ring's moved gap, so the alignment propagates outward as a chained sequence of border-cycle shifts.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    anchor, rings = _extract_structure(g)

    out = g.copy()
    ref_gap: tuple[int, int] | None = None

    for ring_idx, ring in enumerate(rings):
        positions = ring["positions"]
        gap_idx = int(ring["gap_idx"])
        sequence = np.full(len(positions), int(ring["color"]), dtype=int)
        sequence[gap_idx] = int(bg)

        if ring_idx == 0:
            targets = _inner_target_positions(ring["bbox"], anchor["bbox"])
        else:
            assert ref_gap is not None
            targets = _aligned_target_positions(ring["bbox"], ref_gap)
        target_indices = [ring["index_of"][target] for target in targets]
        shift, _target_idx = _rotate_choice(len(positions), gap_idx, target_indices)
        rotated = np.roll(sequence, shift)

        for (r, c), value in zip(positions, rotated):
            out[r, c] = int(value)

        new_gap_idx = int(np.where(rotated == bg)[0][0])
        ref_gap = positions[new_gap_idx]
        _gap_side(ring["bbox"], ref_gap)

    return out


def _random_gap_index(bbox: tuple[int, int, int, int], rng: random.Random) -> int:
    r0, c0, r1, c1 = bbox
    side = rng.choice(SIDE_ORDER)
    if side == "top":
        gap_pos = (r0, rng.randint(c0 + 1, c1 - 1))
    elif side == "right":
        gap_pos = (rng.randint(r0 + 1, r1 - 1), c1)
    elif side == "bottom":
        gap_pos = (r1, rng.randint(c0 + 1, c1 - 1))
    else:
        gap_pos = (rng.randint(r0 + 1, r1 - 1), c0)
    positions = _border_positions(bbox)
    index_of = {pos: idx for idx, pos in enumerate(positions)}
    return int(index_of[gap_pos])


def _render_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, gap_idx: int) -> None:
    for idx, (r, c) in enumerate(_border_positions(bbox)):
        if idx != gap_idx:
            grid[r, c] = int(color)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    anchor_size = rng.choice([2, 3])
    max_rings = min(6, (ARC_MAX - anchor_size) // 4)
    ring_count = rng.randint(2, max_rings)
    outer_size = anchor_size + 4 * ring_count

    extra_h = rng.randint(0, ARC_MAX - outer_size)
    extra_w = rng.randint(0, ARC_MAX - outer_size)
    top_pad = rng.randint(0, extra_h)
    bottom_pad = extra_h - top_pad
    left_pad = rng.randint(0, extra_w)
    right_pad = extra_w - left_pad

    h = outer_size + top_pad + bottom_pad
    w = outer_size + left_pad + right_pad
    grid = np.zeros((h, w), dtype=int)

    anchor_r0 = top_pad + 2 * ring_count
    anchor_c0 = left_pad + 2 * ring_count
    grid[anchor_r0 : anchor_r0 + anchor_size, anchor_c0 : anchor_c0 + anchor_size] = ANCHOR_COLOR

    ring_colors = rng.sample(RING_COLOR_POOL, ring_count)
    for depth in range(1, ring_count + 1):
        size = anchor_size + 4 * depth
        r0 = anchor_r0 - 2 * depth
        c0 = anchor_c0 - 2 * depth
        bbox = (r0, c0, r0 + size - 1, c0 + size - 1)
        gap_idx = _random_gap_index(bbox, rng)
        _render_ring(grid, bbox, ring_colors[depth - 1], gap_idx)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
