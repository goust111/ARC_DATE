from __future__ import annotations

import random

import numpy as np


BLACK = 0
FRAME_COLOR = 5
LEGEND_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_rectangular_frame(coords: np.ndarray) -> bool:
    """True when the component is exactly a 1-pixel-thick hollow rectangle."""
    r0, c0, r1, c1 = _bbox(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return False

    frame = np.zeros((height, width), dtype=bool)
    frame[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros_like(frame)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(frame, expected))


def _find_frame_color(grid: np.ndarray) -> int:
    """Find the unique non-black color whose components are all hollow frames."""
    colors = [int(color) for color in np.unique(grid) if int(color) != BLACK]
    candidates: list[int] = []
    for color in colors:
        comps = _components(grid == int(color))
        if comps and all(_is_rectangular_frame(comp) for comp in comps):
            candidates.append(int(color))
    assert len(candidates) == 1, "expected one frame color"
    return candidates[0]


def _legend_colors(grid: np.ndarray, frame_color: int) -> list[int]:
    """Read the vertical 2x1 legend blocks from top to bottom."""
    mask = (grid != BLACK) & (grid != int(frame_color))
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a color legend"

    r0, c0, r1, c1 = _bbox(coords)
    cols = sorted(set(int(c) for c in coords[:, 1]))
    assert cols == [c0, c1] and c1 == c0 + 1, "legend must occupy two adjacent columns"

    colors: list[int] = []
    for r in range(r0, r1 + 1):
        row_cols = np.flatnonzero(mask[r])
        if row_cols.size == 0:
            continue
        assert row_cols.size == 2 and int(row_cols[0]) == c0 and int(row_cols[1]) == c1
        left = int(grid[r, c0])
        right = int(grid[r, c1])
        assert left == right, "each legend row must be a 2x1 monochrome block"
        colors.append(left)

    assert colors, "legend must have at least one entry"
    return colors


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each hollow frame according to how many other frames contain it.

    The legend is a vertical stack of 2x1 color blocks. The top block maps to
    depth 0, the next to depth 1, and so on.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_color = _find_frame_color(g)
    legend = _legend_colors(g, frame_color)

    frame_components = _components(g == int(frame_color))
    assert frame_components, "expected at least one frame"
    frame_boxes = [_bbox(comp) for comp in frame_components]

    out = g.copy()
    for comp, box in zip(frame_components, frame_boxes, strict=False):
        depth = sum(
            1
            for other in frame_boxes
            if other != box and _strictly_contains(other, box)
        )
        assert depth < len(legend), "legend must cover every nesting depth"
        out[comp[:, 0], comp[:, 1]] = int(legend[depth])
    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (20, 32)
LEGEND_GAP_RANGE = (2, 4)
LEGEND_EXTRA_RANGE = (0, 2)
LEGEND_SIDE_OPTIONS = ("left", "right")
ROOT_CHAIN_COUNT_RANGE = (1, 3)
CHAIN_LENGTH_RANGE = (1, 4)
OUTER_MARGIN = 1
ROOT_EXTRA_SLACK = 6
ROOT_PLACEMENT_ATTEMPTS = 150
GENERATION_ATTEMPTS = 400


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _touch_or_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _root_min_side(chain_length: int) -> int:
    return 3 + 4 * (int(chain_length) - 1)


def _sample_root(
    draw_box: tuple[int, int, int, int],
    chain_length: int,
    existing: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    dr0, dc0, dr1, dc1 = draw_box
    draw_h = int(dr1 - dr0 + 1)
    draw_w = int(dc1 - dc0 + 1)
    min_side = _root_min_side(chain_length)
    if min_side > draw_h or min_side > draw_w:
        return None

    for _ in range(ROOT_PLACEMENT_ATTEMPTS):
        height = random.randint(min_side, min(draw_h, min_side + ROOT_EXTRA_SLACK))
        width = random.randint(min_side, min(draw_w, min_side + ROOT_EXTRA_SLACK))
        r0 = random.randint(dr0, dr1 - height + 1)
        c0 = random.randint(dc0, dc1 - width + 1)
        box = (r0, c0, r0 + height - 1, c0 + width - 1)
        if all(not _touch_or_overlap(box, other) for other in existing):
            return box
    return None


def _place_roots(
    draw_box: tuple[int, int, int, int],
    chain_lengths: list[int],
) -> list[tuple[int, int, int, int]] | None:
    indexed = sorted(
        enumerate(chain_lengths),
        key=lambda item: _root_min_side(item[1]),
        reverse=True,
    )
    roots: list[tuple[int, int, int, int] | None] = [None] * len(chain_lengths)
    placed: list[tuple[int, int, int, int]] = []
    for idx, chain_length in indexed:
        root = _sample_root(draw_box, chain_length, placed)
        if root is None:
            return None
        roots[idx] = root
        placed.append(root)
    return [root for root in roots if root is not None]


def _build_chain(root: tuple[int, int, int, int], chain_length: int) -> list[tuple[int, int, int, int]] | None:
    chain = [root]
    current = root
    for depth in range(1, int(chain_length)):
        remain = int(chain_length) - depth
        min_side = _root_min_side(remain)
        r0, c0, r1, c1 = current
        max_h = int(r1 - r0 + 1) - 4
        max_w = int(c1 - c0 + 1) - 4
        if min_side > max_h or min_side > max_w:
            return None

        height = random.randint(min_side, max_h)
        width = random.randint(min_side, max_w)
        child_r0 = random.randint(r0 + 2, r1 - height - 1)
        child_c0 = random.randint(c0 + 2, c1 - width - 1)
        child = (child_r0, child_c0, child_r0 + height - 1, child_c0 + width - 1)
        chain.append(child)
        current = child
    return chain


def _expected_output(
    grid: np.ndarray,
    chains: list[list[tuple[int, int, int, int]]],
    legend: list[int],
) -> np.ndarray:
    out = grid.copy()
    for chain in chains:
        for depth, box in enumerate(chain):
            _draw_frame(out, box, legend[depth])
    return out


def _generate_case() -> np.ndarray | None:
    chain_count = random.randint(*ROOT_CHAIN_COUNT_RANGE)
    chain_lengths = [random.randint(*CHAIN_LENGTH_RANGE) for _ in range(chain_count)]
    max_depth = max(chain_lengths) - 1
    legend_len = max_depth + 1 + random.randint(*LEGEND_EXTRA_RANGE)
    if legend_len > len(LEGEND_COLOR_POOL):
        return None

    height = random.randint(max(GRID_HEIGHT_RANGE[0], legend_len + 2), GRID_HEIGHT_RANGE[1])
    width = random.randint(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1])
    side = random.choice(LEGEND_SIDE_OPTIONS)
    gap = random.randint(*LEGEND_GAP_RANGE)

    legend_top = random.randint(1, height - legend_len - 1)
    if side == "left":
        legend_col = 0
        draw_box = (OUTER_MARGIN, 2 + gap, height - 1 - OUTER_MARGIN, width - 1 - OUTER_MARGIN)
    else:
        legend_col = width - 2
        draw_box = (OUTER_MARGIN, OUTER_MARGIN, height - 1 - OUTER_MARGIN, width - 3 - gap)

    if draw_box[1] > draw_box[3] or draw_box[0] > draw_box[2]:
        return None

    roots = _place_roots(draw_box, chain_lengths)
    if roots is None:
        return None

    chains: list[list[tuple[int, int, int, int]]] = []
    for root, chain_length in zip(roots, chain_lengths, strict=False):
        chain = _build_chain(root, chain_length)
        if chain is None:
            return None
        chains.append(chain)

    legend_colors = random.sample(LEGEND_COLOR_POOL, legend_len)
    grid = np.zeros((height, width), dtype=int)
    for offset, color in enumerate(legend_colors):
        row = legend_top + offset
        grid[row, legend_col : legend_col + 2] = int(color)

    for chain in chains:
        for box in chain:
            _draw_frame(grid, box, FRAME_COLOR)

    expected = _expected_output(grid, chains, legend_colors)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nesting-frame input")
