from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
OBJECT_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_HEIGHT_RANGE = (9, 17)
GRID_WIDTH_RANGE = (9, 17)
PALETTE_SIZE_RANGE = (2, 4)
MATCH_COUNT_RANGE = (1, 2)
MISMATCH_COUNT_RANGE = (1, 2)
GENERATION_ATTEMPTS = 400

BASE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return tuple(sorted((r - min_r, c - min_c) for r, c in shape))


def _rotate_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    height = max(r for r, _ in shape) + 1
    return tuple((c, height - 1 - r) for r, c in shape)


def _reflect_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    width = max(c for _, c in shape) + 1
    return tuple((r, width - 1 - c) for r, c in shape)


def _shape_variants() -> tuple[tuple[tuple[int, int], ...], ...]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    for base in BASE_SHAPES:
        current = _normalize_shape(base)
        for _ in range(4):
            variants.add(_normalize_shape(current))
            variants.add(_normalize_shape(_reflect_shape(current)))
            current = _rotate_shape(current)
    return tuple(
        sorted(
            variants,
            key=lambda shape: (
                len(shape),
                max(r for r, _ in shape),
                max(c for _, c in shape),
                shape,
            ),
        )
    )


SHAPES = _shape_variants()


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            parts.append(cells)

    return parts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use edge colors as an implicit row/column lookup table.

    Each interior object is recolored only when all top-row colors across its
    columns agree, all left-column colors across its rows agree, and the two
    unanimous colors are the same.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 3 and g.shape[1] >= 3, "grid must contain axes and an interior"

    object_color = int(g[0, 0])
    out = g.copy()

    interior = g[1:, 1:]
    for comp in _components(interior == object_color):
        cells = [(r + 1, c + 1) for r, c in comp]
        top_colors = {int(g[0, c]) for _r, c in cells}
        left_colors = {int(g[r, 0]) for r, _c in cells}

        if len(top_colors) != 1 or len(left_colors) != 1:
            continue

        color = next(iter(top_colors))
        if color != next(iter(left_colors)):
            continue

        for r, c in cells:
            out[r, c] = color

    return out


def _segmented_axis(length: int, palette: tuple[int, ...], anchor_color: int) -> list[int]:
    n_segments = random.randint(2, min(5, length))
    cuts = sorted(random.sample(range(1, length), n_segments - 1))
    bounds = [0, *cuts, length]
    labels: list[int] = []
    prev = None

    for start, end in zip(bounds, bounds[1:], strict=False):
        choices = [color for color in palette if color != prev]
        if not choices:
            choices = list(palette)
        color = random.choice(choices)
        labels.extend([color] * (end - start))
        prev = color

    anchor_len = random.randint(2, min(5, length - 1))
    anchor_start = random.randint(0, length - anchor_len)
    labels[anchor_start : anchor_start + anchor_len] = [anchor_color] * anchor_len

    if len(set(labels)) == 1:
        swap_index = random.choice(
            [idx for idx in range(length) if not (anchor_start <= idx < anchor_start + anchor_len)]
        )
        labels[swap_index] = random.choice([color for color in palette if color != anchor_color])

    return labels


def _classify_cells(
    cells: tuple[tuple[int, int], ...],
    top_axis: np.ndarray,
    left_axis: np.ndarray,
) -> tuple[bool, int]:
    top_colors = {int(top_axis[c]) for _r, c in cells}
    left_colors = {int(left_axis[r]) for r, _c in cells}
    if len(top_colors) == 1 and len(left_colors) == 1:
        color = next(iter(top_colors))
        if color == next(iter(left_colors)):
            return True, color
    return False, OBJECT_COLOR


def _all_candidates(top_axis: np.ndarray, left_axis: np.ndarray) -> dict[bool, list[dict[str, object]]]:
    h = len(left_axis)
    w = len(top_axis)
    buckets: dict[bool, list[dict[str, object]]] = {True: [], False: []}

    for shape in SHAPES:
        sh, sw = _shape_dims(shape)
        for r0 in range(1, h - sh + 1):
            for c0 in range(1, w - sw + 1):
                cells = tuple(sorted((r0 + r, c0 + c) for r, c in shape))
                matched, color = _classify_cells(cells, top_axis, left_axis)
                buckets[matched].append(
                    {
                        "cells": cells,
                        "matched": matched,
                        "color": color,
                    }
                )

    random.shuffle(buckets[True])
    random.shuffle(buckets[False])
    return buckets


def _expanded_cells(
    cells: tuple[tuple[int, int], ...],
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    expanded = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 1 <= nr < h and 1 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _sample_placements(
    candidates: dict[bool, list[dict[str, object]]],
    h: int,
    w: int,
    match_count: int,
    mismatch_count: int,
) -> list[dict[str, object]] | None:
    desired = [True] * match_count + [False] * mismatch_count

    for _ in range(80):
        random.shuffle(desired)
        blocked: set[tuple[int, int]] = set()
        placed: list[dict[str, object]] = []

        for need_match in desired:
            pool = [
                cand
                for cand in candidates[need_match]
                if not any(cell in blocked for cell in cand["cells"])
            ]
            if not pool:
                break
            choice = random.choice(pool)
            placed.append(choice)
            blocked |= _expanded_cells(choice["cells"], h, w)
        else:
            return placed

    return None


def _build_axes(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    palette_size = random.randint(*PALETTE_SIZE_RANGE)
    palette = tuple(random.sample(COLOR_POOL, palette_size))
    anchor_color = random.choice(palette)

    top_axis = np.full(w, OBJECT_COLOR, dtype=int)
    left_axis = np.full(h, OBJECT_COLOR, dtype=int)
    top_axis[1:] = _segmented_axis(w - 1, palette, anchor_color)
    left_axis[1:] = _segmented_axis(h - 1, palette, anchor_color)
    return top_axis, left_axis


def _render_pair(
    top_axis: np.ndarray,
    left_axis: np.ndarray,
    placements: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    h = len(left_axis)
    w = len(top_axis)
    inp = np.full((h, w), BACKGROUND, dtype=int)
    out = np.full((h, w), BACKGROUND, dtype=int)

    inp[0, :] = top_axis
    inp[:, 0] = left_axis
    out[0, :] = top_axis
    out[:, 0] = left_axis

    for placement in placements:
        color = int(placement["color"])
        for r, c in placement["cells"]:
            inp[r, c] = OBJECT_COLOR
            out[r, c] = color

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    top_axis, left_axis = _build_axes(h, w)
    candidates = _all_candidates(top_axis, left_axis)

    if not candidates[True] or not candidates[False]:
        return None

    match_count = random.randint(*MATCH_COUNT_RANGE)
    mismatch_count = random.randint(*MISMATCH_COUNT_RANGE)
    placements = _sample_placements(candidates, h, w, match_count, mismatch_count)
    if placements is None:
        return None

    inp, expected = _render_pair(top_axis, left_axis, placements)

    matched_objects = sum(1 for placement in placements if placement["matched"])
    mismatched_objects = len(placements) - matched_objects
    if matched_objects == 0 or mismatched_objects == 0:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None

    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
