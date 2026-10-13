from __future__ import annotations

import numpy as np


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _extract_ring(grid: np.ndarray, color: int) -> dict:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, "each foreground color must be present"

    r0, c0, r1, c1 = _bbox(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    assert height >= 3 and width >= 3, "each input object must be a hollow rectangular ring"

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    expected = _frame_mask(height, width)
    assert np.array_equal(sub == int(color), expected), "each color must form one complete 1px rectangular ring"

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
    }


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _rings_by_depth(grid: np.ndarray) -> list[dict]:
    colors = [int(color) for color in np.unique(grid) if int(color) != 0]
    assert colors, "expected at least one non-black ring"

    rings = [_extract_ring(grid, color) for color in colors]
    bboxes = [ring["bbox"] for ring in rings]

    depths = []
    for ring in rings:
        depth = 1 + sum(
            1
            for other_bbox in bboxes
            if other_bbox != ring["bbox"] and _strictly_contains(other_bbox, ring["bbox"])
        )
        depths.append(depth)

    assert sorted(depths) == list(range(1, len(rings) + 1)), "rings must form one strict nesting chain"
    for ring, depth in zip(rings, depths):
        ring["depth"] = int(depth)

    rings.sort(key=lambda ring: int(ring["depth"]))
    return rings


def _render_concentric(colors_outer_to_inner: list[int]) -> np.ndarray:
    size = 2 * len(colors_outer_to_inner) - 1
    out = np.zeros((size, size), dtype=int)
    for layer, color in enumerate(colors_outer_to_inner):
        end = size - layer - 1
        out[layer, layer : end + 1] = int(color)
        out[end, layer : end + 1] = int(color)
        out[layer : end + 1, layer] = int(color)
        out[layer : end + 1, end] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the nesting order of rectangular rings and redraw them as standard concentric square rings.

    Input structure:
    - The background is black.
    - Every non-black color appears exactly once as a one-pixel-thick axis-aligned rectangular ring.
    - The rings form one strict nesting chain: each ring is either fully inside another ring or fully outside it, with no partial overlaps.

    Rule:
    - Determine each ring's nesting depth by counting how many other rings strictly contain its rectangular bounding box.
    - The outermost input ring has depth 1, the next ring inside has depth 2, and so on.
    - Build a square output of side 2N-1, where N is the number of rings, and draw N one-pixel-thick concentric square rings.
    - Reverse the depth order: the deepest input ring becomes the output outer ring, and the shallowest input ring becomes the center pixel.

    Key insight:
    - The input rings' absolute sizes and positions are irrelevant noise; only the containment topology matters, and the output flips that nesting order before redrawing it in a normalized square layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    rings = _rings_by_depth(g)
    reversed_colors = [int(ring["color"]) for ring in reversed(rings)]
    return _render_concentric(reversed_colors)


# --------------- generator ---------------

_RNG = np.random.default_rng()

BG = 0
PALETTE = tuple(range(1, 10))
RING_COUNT_OPTIONS = (3, 4, 5)
OUTER_DIM_CAP = 28
EXTRA_MARGIN_CAP = 6
MAX_GENERATION_ATTEMPTS = 300


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _sample_colors(count: int) -> list[int]:
    picks = _RNG.choice(PALETTE, size=int(count), replace=False)
    return [int(color) for color in picks]


def _min_chain_dim(ring_count: int) -> int:
    return 3 + 4 * (int(ring_count) - 1)


def _build_nested_bboxes(ring_count: int) -> list[tuple[int, int, int, int]]:
    outer_h = _randint(_min_chain_dim(ring_count), OUTER_DIM_CAP)
    outer_w = _randint(_min_chain_dim(ring_count), OUTER_DIM_CAP)
    bboxes = [(0, 0, outer_h - 1, outer_w - 1)]

    for depth in range(1, ring_count):
        r0, c0, r1, c1 = bboxes[-1]
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)

        remaining_chain = ring_count - depth
        min_next_h = _min_chain_dim(remaining_chain)
        min_next_w = _min_chain_dim(remaining_chain)

        total_shrink_h = _randint(4, height - min_next_h)
        total_shrink_w = _randint(4, width - min_next_w)

        top_gap = _randint(2, total_shrink_h - 2)
        bottom_gap = total_shrink_h - top_gap
        left_gap = _randint(2, total_shrink_w - 2)
        right_gap = total_shrink_w - left_gap

        bboxes.append((r0 + top_gap, c0 + left_gap, r1 - bottom_gap, c1 - right_gap))

    return bboxes


def _draw_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _translate_bboxes(
    bboxes: list[tuple[int, int, int, int]],
    row_shift: int,
    col_shift: int,
) -> list[tuple[int, int, int, int]]:
    return [
        (int(r0 + row_shift), int(c0 + col_shift), int(r1 + row_shift), int(c1 + col_shift))
        for r0, c0, r1, c1 in bboxes
    ]


def _gen_attempt() -> np.ndarray | None:
    ring_count = int(RING_COUNT_OPTIONS[_randint(0, len(RING_COUNT_OPTIONS) - 1)])
    colors = _sample_colors(ring_count)
    bboxes = _build_nested_bboxes(ring_count)

    outer_r0, outer_c0, outer_r1, outer_c1 = bboxes[0]
    outer_h = int(outer_r1 - outer_r0 + 1)
    outer_w = int(outer_c1 - outer_c0 + 1)

    extra_h = _randint(0, min(EXTRA_MARGIN_CAP, 30 - outer_h))
    extra_w = _randint(0, min(EXTRA_MARGIN_CAP, 30 - outer_w))
    grid_h = outer_h + extra_h
    grid_w = outer_w + extra_w

    row_shift = _randint(0, grid_h - outer_h)
    col_shift = _randint(0, grid_w - outer_w)
    placed_bboxes = _translate_bboxes(bboxes, row_shift, col_shift)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for bbox, color in zip(placed_bboxes, colors):
        _draw_ring(grid, bbox, color)

    expected = _render_concentric(list(reversed(colors)))
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring input")
