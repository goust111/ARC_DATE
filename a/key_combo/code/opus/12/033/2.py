from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


BLACK = 0
GREEN = 3
FRAME_COLORS = (1, 2, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

_RNG = np.random.default_rng()
_GEN_ATTEMPTS = 500


@dataclass(frozen=True)
class Slot:
    color: int
    bbox: tuple[int, int, int, int]
    hole_count: int


@dataclass(frozen=True)
class FrameSpec:
    color: int
    bbox: tuple[int, int, int, int]
    hole_masks: tuple[np.ndarray, ...]


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_rectangular_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    patch = np.zeros((h, w), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros_like(patch)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(patch, border)


def _find_slots(grid: np.ndarray) -> list[Slot]:
    g = np.asarray(grid, dtype=int)
    slots: list[Slot] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BLACK):
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        bbox = _bbox(coords)
        r0, c0, r1, c1 = bbox
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        if not (
            np.all(g[r0, c0 : c1 + 1] == color)
            and np.all(g[r1, c0 : c1 + 1] == color)
            and np.all(g[r0 : r1 + 1, c0] == color)
            and np.all(g[r0 : r1 + 1, c1] == color)
        ):
            continue
        hole_count = len(_hole_components_in_bbox(g, bbox))
        slots.append(Slot(color=color, bbox=bbox, hole_count=hole_count))
    slots.sort(key=lambda slot: (slot.bbox[1], slot.bbox[0]))
    return slots


def _hole_components_in_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0:
        return []
    holes: list[np.ndarray] = []
    h, w = inner.shape
    for coords in _components4(inner == BLACK):
        touches_boundary = (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == h - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == w - 1)
        )
        if not touches_boundary:
            holes.append(coords + np.array([r0 + 1, c0 + 1], dtype=int))
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each hollow frame by a cyclic shift whose offset equals its interior hole count, then fill those holes green.

    Input structure:
    - The grid has a black background and 3-5 horizontally ordered hollow rectangular frames of distinct colors.
    - Each frame is a one-cell-thick outer rectangle; inside it may contain extra same-color wall pixels that enclose 0, 1, or 2 small black holes.
    - The slot order is determined by the outer frame's left column, not by color value.

    Rule:
    - For each slot i, read its input frame color C_i and count H_i of black 4-connected hole components strictly inside the outer frame that do not touch the frame interior boundary.
    - Recolor every non-black pixel belonging to slot i with the input color from slot (i - H_i) mod N.
    - Fill every counted hole pixel in every slot with green.
    - Keep the grid shape, slot positions, and all remaining black interior/background pixels unchanged.

    Key insight:
    - The decisive signal is a topological count of enclosed black regions inside each frame; after the output fills those holes green, the evidence for the color shift disappears.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    slots = _find_slots(g)
    assert 3 <= len(slots) <= 5

    colors = [slot.color for slot in slots]
    holes_by_slot = [_hole_components_in_bbox(g, slot.bbox) for slot in slots]

    out = g.copy()
    n = len(slots)
    for i, slot in enumerate(slots):
        new_color = colors[(i - len(holes_by_slot[i])) % n]
        r0, c0, r1, c1 = slot.bbox
        sub_in = g[r0 : r1 + 1, c0 : c1 + 1]
        sub_out = out[r0 : r1 + 1, c0 : c1 + 1]
        sub_out[sub_in == slot.color] = new_color
        for hole in holes_by_slot[i]:
            out[hole[:, 0], hole[:, 1]] = GREEN

    return out


_BASE_HOLE_SHAPES = (
    np.array([[1], [1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1], [1], [1], [1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
)


def _random_hole_shape(max_h: int, max_w: int) -> np.ndarray | None:
    candidates: list[np.ndarray] = []
    for base in _BASE_HOLE_SHAPES:
        for k in range(4):
            rotated = np.rot90(base, k)
            for variant in (rotated, np.fliplr(rotated)):
                if variant.shape[0] <= max_h and variant.shape[1] <= max_w:
                    if not any(np.array_equal(variant, old) for old in candidates):
                        candidates.append(variant.copy())
    if not candidates:
        return None
    return candidates[int(_RNG.integers(len(candidates)))]


def _loop_patch_from_hole(hole_mask: np.ndarray, color: int) -> tuple[np.ndarray, np.ndarray]:
    hh, hw = hole_mask.shape
    hole_patch = np.zeros((hh + 2, hw + 2), dtype=bool)
    hole_patch[1 : hh + 1, 1 : hw + 1] = hole_mask

    wall_patch = np.zeros_like(hole_patch)
    for r, c in np.argwhere(hole_patch):
        for dr, dc in DIR4:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < hole_patch.shape[0] and 0 <= nc < hole_patch.shape[1] and not hole_patch[nr, nc]:
                wall_patch[nr, nc] = True

    patch = np.zeros_like(hole_patch, dtype=int)
    patch[wall_patch] = color
    return patch, hole_patch


def _place_holes(inner_h: int, inner_w: int, count: int, color: int) -> tuple[np.ndarray, tuple[np.ndarray, ...]] | None:
    if count == 0:
        return np.zeros((inner_h, inner_w), dtype=int), ()

    if count == 2 and inner_w == 3:
        if inner_h < 9:
            return None
        loop_patch, hole_patch = _loop_patch_from_hole(np.array([[1], [1]], dtype=bool), color)
        patch = np.zeros((inner_h, inner_w), dtype=int)
        extra = inner_h - 9
        top_pad = int(_RNG.integers(0, extra + 1))
        bottom_pad = int(_RNG.integers(0, extra - top_pad + 1))
        middle_gap = 1 + (extra - top_pad - bottom_pad)
        top_r = top_pad
        bottom_r = top_r + loop_patch.shape[0] + middle_gap
        patch[top_r : top_r + loop_patch.shape[0], 0:3] += loop_patch
        patch[bottom_r : bottom_r + loop_patch.shape[0], 0:3] += loop_patch
        holes = (
            np.argwhere(hole_patch) + np.array([top_r, 0], dtype=int),
            np.argwhere(hole_patch) + np.array([bottom_r, 0], dtype=int),
        )
        return patch, tuple(np.array(hole, dtype=int) for hole in holes)

    patch = np.zeros((inner_h, inner_w), dtype=int)
    reserved = np.zeros((inner_h, inner_w), dtype=bool)
    holes: list[np.ndarray] = []

    for _index in range(count):
        placed = False
        for _ in range(80):
            hole_shape = _random_hole_shape(max_h=max(1, inner_h - 2), max_w=max(1, inner_w - 2))
            if hole_shape is None:
                return None
            loop_patch, hole_patch = _loop_patch_from_hole(hole_shape, color)
            ph, pw = loop_patch.shape
            if ph > inner_h or pw > inner_w:
                continue

            positions = [
                (int(r0), int(c0))
                for r0 in range(inner_h - ph + 1)
                for c0 in range(inner_w - pw + 1)
            ]
            if not positions:
                continue
            order = _RNG.permutation(len(positions))
            for idx in order:
                r0, c0 = positions[int(idx)]
                if reserved[r0 : r0 + ph, c0 : c0 + pw].any():
                    continue
                patch[r0 : r0 + ph, c0 : c0 + pw] += loop_patch
                abs_hole = np.argwhere(hole_patch) + np.array([r0, c0], dtype=int)
                holes.append(abs_hole)
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(inner_h, r0 + ph + 1)
                cc1 = min(inner_w, c0 + pw + 1)
                reserved[rr0:rr1, cc0:cc1] = True
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    return patch, tuple(np.array(hole, dtype=int) for hole in holes)


def _sample_hole_counts(n: int) -> tuple[int, ...]:
    valid: list[tuple[int, ...]] = []
    for counts in np.ndindex(*(3,) * n):
        if len(set(counts)) < 2:
            continue
        if all(count == 0 for count in counts):
            continue
        if n == 5 and sum(count > 0 for count in counts) > 2:
            continue
        valid.append(tuple(int(v) for v in counts))

    rich = [counts for counts in valid if set(counts) == {0, 1, 2}]
    with_zero = [counts for counts in valid if 0 in counts]
    if rich and _RNG.random() < 0.45:
        pool = rich
    elif with_zero and _RNG.random() < 0.75:
        pool = with_zero
    else:
        pool = valid
    return pool[int(_RNG.integers(len(pool)))]


def _frame_size(n: int, hole_count: int) -> tuple[int, int]:
    if hole_count == 0:
        height = int(_RNG.integers(5, 9))
        width = 4 if n == 5 or _RNG.random() < 0.65 else 5
    elif hole_count == 1:
        height = int(_RNG.integers(6, 10))
        width = 5 if n >= 4 or _RNG.random() < 0.6 else 6
    else:
        height = int(_RNG.integers(11, 14))
        width = 5 if n >= 4 or _RNG.random() < 0.75 else 6
    return height, width


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _expected_output_from_specs(grid: np.ndarray, specs: list[FrameSpec]) -> np.ndarray:
    ordered = sorted(specs, key=lambda spec: (spec.bbox[1], spec.bbox[0]))
    colors = [spec.color for spec in ordered]
    counts = [len(spec.hole_masks) for spec in ordered]
    out = grid.copy()
    n = len(ordered)
    for i, spec in enumerate(ordered):
        new_color = colors[(i - counts[i]) % n]
        r0, c0, r1, c1 = spec.bbox
        sub_in = grid[r0 : r1 + 1, c0 : c1 + 1]
        sub_out = out[r0 : r1 + 1, c0 : c1 + 1]
        sub_out[sub_in == spec.color] = new_color
        for hole in spec.hole_masks:
            out[hole[:, 0], hole[:, 1]] = GREEN
    return out


def _generate_attempt() -> np.ndarray | None:
    n = int(_RNG.integers(3, 6))
    hole_counts = _sample_hole_counts(n)
    colors = _RNG.choice(np.array(FRAME_COLORS, dtype=int), size=n, replace=False).tolist()

    frame_dims: list[tuple[int, int]] = []
    max_h = 0
    for hole_count in hole_counts:
        dims = _frame_size(n, hole_count)
        frame_dims.append(dims)
        max_h = max(max_h, dims[0])

    gaps = [int(_RNG.integers(2, 4)) for _ in range(n - 1)]
    total_w = sum(width for _height, width in frame_dims) + sum(gaps)
    if total_w > 30:
        return None

    height = max(max_h + int(_RNG.integers(0, 3)), int(_RNG.integers(12, 19)))
    if height > 30:
        return None

    top_limit = height - max_h
    if top_limit < 0:
        return None
    grid = np.zeros((height, total_w), dtype=int)

    specs: list[FrameSpec] = []
    col = 0
    for idx, ((frame_h, frame_w), color, hole_count) in enumerate(zip(frame_dims, colors, hole_counts)):
        max_r0 = height - frame_h
        if max_r0 < 0:
            return None
        if idx == 0:
            r0 = int(_RNG.integers(0, max_r0 + 1))
        else:
            prev_r0 = specs[-1].bbox[0]
            low = max(0, prev_r0 - 2)
            high = min(max_r0, prev_r0 + 2)
            if low > high:
                r0 = int(_RNG.integers(0, max_r0 + 1))
            else:
                r0 = int(_RNG.integers(low, high + 1))
        c0 = col
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1
        bbox = (r0, c0, r1, c1)

        _draw_frame(grid, bbox, int(color))
        inner_h = frame_h - 2
        inner_w = frame_w - 2
        placement = _place_holes(inner_h, inner_w, int(hole_count), int(color))
        if placement is None:
            return None
        inner_patch, local_holes = placement
        grid[r0 + 1 : r1, c0 + 1 : c1] = np.maximum(grid[r0 + 1 : r1, c0 + 1 : c1], inner_patch)

        abs_holes = tuple(hole + np.array([r0 + 1, c0 + 1], dtype=int) for hole in local_holes)
        specs.append(FrameSpec(color=int(color), bbox=bbox, hole_masks=abs_holes))
        if idx < n - 1:
            col = c1 + 1 + gaps[idx]

    slots = _find_slots(grid)
    if len(slots) != n:
        return None
    extracted_counts = tuple(slot.hole_count for slot in slots)
    if extracted_counts != tuple(hole_counts):
        return None
    if [slot.color for slot in slots] != [spec.color for spec in sorted(specs, key=lambda spec: spec.bbox[1])]:
        return None

    expected = _expected_output_from_specs(grid, specs)
    produced = main(grid)
    if np.array_equal(produced, grid):
        return None
    if not np.array_equal(produced, expected):
        return None

    changed_slots = sum(
        int(colors[i] != colors[(i - hole_counts[i]) % n])
        for i in range(n)
    )
    if changed_slots == 0:
        return None

    for slot in slots:
        if len(_hole_components_in_bbox(expected, slot.bbox)) != 0:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
