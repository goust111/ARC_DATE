from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 80


def _overlay_visible(front: np.ndarray, back: np.ndarray) -> np.ndarray:
    out = front.copy()
    mask = out == BG
    out[mask] = back[mask]
    return out


def _find_fold_axis(g: np.ndarray) -> tuple[str, int]:
    full_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == RED)]
    full_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == RED)]
    assert len(full_rows) + len(full_cols) == 1, "expected exactly one full red fold axis"

    if full_rows:
        r = int(full_rows[0])
        axis_mask = np.zeros_like(g, dtype=bool)
        axis_mask[r, :] = True
        assert not np.any((g == RED) & ~axis_mask), "red is reserved for the fold axis"
        return "H", r

    c = int(full_cols[0])
    axis_mask = np.zeros_like(g, dtype=bool)
    axis_mask[:, c] = True
    assert not np.any((g == RED) & ~axis_mask), "red is reserved for the fold axis"
    return "V", c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the grid across the full red line and keep the visible colors after overlay.

    Input structure:
    - The background is black.
    - The grid is square and contains scattered non-background pixels forming simple motifs.
    - Exactly one full red row or one full red column acts as a fold axis and separates the grid into two slabs.

    Rule:
    - If the red axis is horizontal, remove the axis row, flip the lower slab upward, and overlay it onto the upper slab.
    - If the red axis is vertical, remove the axis column, flip the right slab leftward, and overlay it onto the left slab.
    - Black under black stays black; a colored pixel covers black; when both overlapping pixels are colored, the upper slab wins for a horizontal fold and the left slab wins for a vertical fold.

    Key insight:
    - The red line is not part of the picture. It is an instruction telling you which way to mirror one half of the grid before merging the two halves with a fixed visibility order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w, "expected a square grid"

    axis_kind, axis_pos = _find_fold_axis(g)

    if axis_kind == "H":
        top = g[:axis_pos, :]
        bottom = g[axis_pos + 1 :, :]
        half = min(int(top.shape[0]), int(bottom.shape[0]))
        assert half > 0, "fold needs cells on both sides of the axis"
        front = top[-half:, :]
        back = np.flipud(bottom[:half, :])
        return _overlay_visible(front, back)

    left = g[:, :axis_pos]
    right = g[:, axis_pos + 1 :]
    half = min(int(left.shape[1]), int(right.shape[1]))
    assert half > 0, "fold needs cells on both sides of the axis"
    front = left[:, -half:]
    back = np.fliplr(right[:, :half])
    return _overlay_visible(front, back)


def _shape_offsets(height: int, width: int) -> tuple[list[tuple[int, int]], int, int]:
    kinds = ["dot"]
    if width >= 2:
        kinds.append("hbar")
    if height >= 2:
        kinds.append("vbar")
    if height >= 2 and width >= 2:
        kinds.extend(["rect", "l"])

    kind = random.choice(kinds)
    if kind == "dot":
        return [(0, 0)], 1, 1

    if kind == "hbar":
        span = random.randint(2, min(6, width))
        return [(0, c) for c in range(span)], 1, span

    if kind == "vbar":
        span = random.randint(2, min(6, height))
        return [(r, 0) for r in range(span)], span, 1

    if kind == "rect":
        box_h = random.randint(2, min(3, height))
        box_w = random.randint(2, min(4, width))
        return [(r, c) for r in range(box_h) for c in range(box_w)], box_h, box_w

    arm_h = random.randint(2, min(4, height))
    arm_w = random.randint(2, min(4, width))
    variant = random.randrange(4)
    coords: set[tuple[int, int]] = set()
    if variant == 0:
        coords.update((r, 0) for r in range(arm_h))
        coords.update((0, c) for c in range(arm_w))
    elif variant == 1:
        coords.update((r, arm_w - 1) for r in range(arm_h))
        coords.update((0, c) for c in range(arm_w))
    elif variant == 2:
        coords.update((r, 0) for r in range(arm_h))
        coords.update((arm_h - 1, c) for c in range(arm_w))
    else:
        coords.update((r, arm_w - 1) for r in range(arm_h))
        coords.update((arm_h - 1, c) for c in range(arm_w))
    return sorted(coords), arm_h, arm_w


def _expanded_bounds(r0: int, c0: int, r1: int, c1: int, height: int, width: int) -> tuple[int, int, int, int]:
    return (
        max(0, r0 - 1),
        max(0, c0 - 1),
        min(height - 1, r1 + 1),
        min(width - 1, c1 + 1),
    )


def _place_object(blocked: np.ndarray) -> list[tuple[int, int]] | None:
    height, width = map(int, blocked.shape)
    for _ in range(PLACE_ATTEMPTS):
        rel, box_h, box_w = _shape_offsets(height, width)
        if box_h > height or box_w > width:
            continue
        r0 = random.randint(0, height - box_h)
        c0 = random.randint(0, width - box_w)
        coords = [(r0 + r, c0 + c) for r, c in rel]
        if any(blocked[r, c] for r, c in coords):
            continue
        rr = [r for r, _ in coords]
        cc = [c for _, c in coords]
        br0, bc0, br1, bc1 = _expanded_bounds(min(rr), min(cc), max(rr), max(cc), height, width)
        if blocked[br0 : br1 + 1, bc0 : bc1 + 1].any():
            continue
        blocked[br0 : br1 + 1, bc0 : bc1 + 1] = True
        return coords
    return None


def _paint(canvas: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        canvas[r, c] = int(color)


def _build_pair(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = shape
    upper = np.full(shape, BG, dtype=int)
    lower = np.full(shape, BG, dtype=int)
    blocked = np.zeros(shape, dtype=bool)

    area = height * width
    max_extra = min(4, max(0, area // 18))
    modes = ["upper", "lower", "both"]
    modes.extend(random.choice(("upper", "lower", "both")) for _ in range(random.randint(0, max_extra)))

    random.shuffle(modes)
    for mode in modes:
        coords = _place_object(blocked)
        if coords is None:
            return None
        if mode == "upper":
            _paint(upper, coords, random.choice(PALETTE))
        elif mode == "lower":
            _paint(lower, coords, random.choice(PALETTE))
        else:
            top_color, bottom_color = random.sample(PALETTE, 2)
            _paint(upper, coords, top_color)
            _paint(lower, coords, bottom_color)

    assert np.any((upper != BG) & (lower == BG))
    assert np.any((upper == BG) & (lower != BG))
    assert np.any((upper != BG) & (lower != BG))
    return upper, lower


def _assemble_horizontal(upper: np.ndarray, lower_folded: np.ndarray) -> np.ndarray:
    half, width = map(int, upper.shape)
    size = 2 * half + 1
    assert width == size
    grid = np.full((size, size), BG, dtype=int)
    grid[:half, :] = upper
    grid[half, :] = RED
    grid[half + 1 :, :] = np.flipud(lower_folded)
    return grid


def _assemble_vertical(left: np.ndarray, right_folded: np.ndarray) -> np.ndarray:
    height, half = map(int, left.shape)
    size = 2 * half + 1
    assert height == size
    grid = np.full((size, size), BG, dtype=int)
    grid[:, :half] = left
    grid[:, half] = RED
    grid[:, half + 1 :] = np.fliplr(right_folded)
    return grid


def _gen_attempt() -> np.ndarray | None:
    half = random.randint(4, 14)
    size = 2 * half + 1
    horizontal = random.choice((True, False))

    if horizontal:
        pair = _build_pair((half, size))
        if pair is None:
            return None
        upper, lower_folded = pair
        inp = _assemble_horizontal(upper, lower_folded)
        expected = _overlay_visible(upper, lower_folded)
    else:
        pair = _build_pair((size, half))
        if pair is None:
            return None
        left, right_folded = pair
        inp = _assemble_vertical(left, right_folded)
        expected = _overlay_visible(left, right_folded)

    try:
        out = main(inp)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folding input")
