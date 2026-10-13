from __future__ import annotations

import random

import numpy as np


BG = 0
MAROON = 9
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

ARROW_PATTERNS: dict[frozenset[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]] = {
    frozenset({(0, 0), (1, 1), (2, 0)}): ((0, 1), (1, 1)),
    frozenset({(0, 1), (1, 0), (2, 1)}): ((0, -1), (1, 0)),
    frozenset({(0, 0), (0, 2), (1, 1)}): ((1, 0), (1, 1)),
    frozenset({(0, 1), (1, 0), (1, 2)}): ((-1, 0), (0, 1)),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components_of_value(grid: np.ndarray, value: int, connectivity: int) -> list[list[tuple[int, int]]]:
    dirs = DIR4 if connectivity == 4 else DIR8
    mask = grid == int(value)
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(grid.shape, nr, nc) and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _parse_arrow(component: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0 = min(rows)
    c0 = min(cols)
    rel = frozenset((r - r0, c - c0) for r, c in component)
    assert rel in ARROW_PATTERNS, "unexpected maroon marker shape"
    direction, tip_rel = ARROW_PATTERNS[rel]
    tip = (r0 + tip_rel[0], c0 + tip_rel[1])
    return direction, tip


def _trace_broken_line(
    grid: np.ndarray, tip: tuple[int, int], direction: tuple[int, int]
) -> tuple[int, tuple[int, int], set[tuple[int, int]]]:
    dr, dc = direction
    r = tip[0] + dr
    c = tip[1] + dc
    assert _in_bounds(grid.shape, r, c), "arrow must point to a line"
    color = int(grid[r, c])
    assert color not in (BG, MAROON), "arrow must point to a colored line"

    line_cells: list[tuple[int, int]] = []
    while _in_bounds(grid.shape, r, c) and int(grid[r, c]) == color:
        line_cells.append((r, c))
        r += dr
        c += dc
    assert line_cells, "line must have cells before its gap"
    assert _in_bounds(grid.shape, r, c) and int(grid[r, c]) == BG, "line must contain a single background gap"
    gap = (r, c)

    r += dr
    c += dc
    after_gap: list[tuple[int, int]] = []
    while _in_bounds(grid.shape, r, c) and int(grid[r, c]) == color:
        after_gap.append((r, c))
        line_cells.append((r, c))
        r += dr
        c += dc
    assert after_gap, "line gap must lie strictly inside the segment"
    return color, gap, set(line_cells)


def _fragment_component(
    grid: np.ndarray, color: int, line_cells: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    fragments: list[list[tuple[int, int]]] = []
    for comp in _components_of_value(grid, color, connectivity=4):
        if not set(comp) <= line_cells:
            fragments.append(comp)
    assert len(fragments) == 1, "each line color must have exactly one separate fragment"
    return fragments[0]


def _leading_fragment_pixel(
    fragment: list[tuple[int, int]], gap: tuple[int, int], direction: tuple[int, int]
) -> tuple[int, int]:
    dr, dc = direction
    if dr == 0:
        aligned = [cell for cell in fragment if cell[0] == gap[0]]
        assert aligned, "horizontal line needs a fragment pixel on the same row as the gap"
        key = lambda cell: cell[1]
        return max(aligned, key=key) if dc > 0 else min(aligned, key=key)

    aligned = [cell for cell in fragment if cell[1] == gap[1]]
    assert aligned, "vertical line needs a fragment pixel on the same column as the gap"
    key = lambda cell: cell[0]
    return max(aligned, key=key) if dr > 0 else min(aligned, key=key)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored fragment along its arrow-marked broken line until it plugs that line's missing cell.

    Input structure:
    - The background is black.
    - Each non-maroon color forms one horizontal or vertical boundary line that has exactly one single-cell black gap inside it.
    - The same color also appears as one separate small connected fragment, and a 3-pixel maroon arrow sits just outside one end of the line, pointing along the line toward the gap.

    Rule:
    - Read every maroon arrow to determine a direction, then follow the colored cells directly in front of its tip to identify that color's broken line and its gap position.
    - Find the unique separate fragment of the same color and translate it rigidly along the arrow direction until the fragment's leading pixel on the line's row or column lands exactly on the gap.
    - Erase the fragment's old position, keep all arrows unchanged, and allow moved fragment pixels to overlap existing same-color line pixels without changing their color.

    Key insight:
    - The maroon marker is a directional instruction, not the moving object itself; the true stopping condition is when a matching fragment fills the line's gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    placements: list[tuple[int, list[tuple[int, int]], tuple[int, int]]] = []
    seen_colors: set[int] = set()

    arrow_components = _components_of_value(g, MAROON, connectivity=8)
    assert arrow_components, "expected at least one maroon arrow"

    for comp in arrow_components:
        assert len(comp) == 3, "each arrow must contain three maroon pixels"
        direction, tip = _parse_arrow(comp)
        color, gap, line_cells = _trace_broken_line(g, tip, direction)
        assert color not in seen_colors, "each line color should be paired with exactly one arrow"
        seen_colors.add(color)

        fragment = _fragment_component(g, color, line_cells)
        anchor = _leading_fragment_pixel(fragment, gap, direction)
        delta = (gap[0] - anchor[0], gap[1] - anchor[1])
        placements.append((color, fragment, delta))

    for _color, fragment, _delta in placements:
        for r, c in fragment:
            out[r, c] = BG

    for color, fragment, delta in placements:
        dr, dc = delta
        for r, c in fragment:
            nr = r + dr
            nc = c + dc
            assert _in_bounds(g.shape, nr, nc), "translated fragment must stay inside the grid"
            target = int(out[nr, nc])
            assert target in (BG, color), "moved fragments must only land on background or their own line"
            out[nr, nc] = color

    return out


def _fragment_offsets() -> list[tuple[int, int]]:
    size = random.randint(1, 3)
    side = random.choice([-1, 1])
    if size == 1:
        return [(0, 0)]
    if size == 2:
        return [(0, 0), (side, 0)]
    if random.random() < 0.5:
        return [(0, 0), (side, 0), (2 * side, 0)]
    return [(0, 0), (side, 0), (side, -1)]


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere((inp != BG) | (out != BG))
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return inp[r0:r1, c0:c1].copy(), out[r0:r1, c0:c1].copy()


def _canonical_right_patch() -> tuple[np.ndarray, np.ndarray]:
    color = 1
    line_row = 2
    line_start = 5
    prefix_len = random.randint(2, 5)
    suffix_len = random.randint(2, 5)
    gap_col = line_start + prefix_len
    line_end = gap_col + suffix_len
    input_anchor_col = random.randint(1, 2)
    offsets = _fragment_offsets()

    width = line_end + 1
    inp = np.full((5, width), BG, dtype=int)
    out = np.full((5, width), BG, dtype=int)

    for c in range(line_start, line_end + 1):
        if c != gap_col:
            inp[line_row, c] = color
        out[line_row, c] = color

    for r, c in ((1, 3), (2, 4), (3, 3)):
        inp[r, c] = MAROON
        out[r, c] = MAROON

    for dr, dc in offsets:
        inp[line_row + dr, input_anchor_col + dc] = color
        out[line_row + dr, gap_col + dc] = color

    return _crop_pair(inp, out)


def _recolor_patch(patch: np.ndarray, color: int) -> np.ndarray:
    out = patch.copy()
    out[out == 1] = int(color)
    return out


def _build_object_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    inp, out = _canonical_right_patch()
    rotation = random.randrange(4)
    inp = np.rot90(inp, rotation).copy()
    out = np.rot90(out, rotation).copy()
    return _recolor_patch(inp, color), _recolor_patch(out, color)


def _rect_clear(rects: list[tuple[int, int, int, int]], rect: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = rect
    for or0, oc0, or1, oc1 in rects:
        if not (r1 < or0 - pad or or1 + pad < r0 or c1 < oc0 - pad or oc1 + pad < c0):
            return False
    return True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    region = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    assert not np.any(region[mask] != BG), "patches must not overlap"
    region[mask] = patch[mask]


def _pack_patches(
    patches: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray] | None:
    max_h = max(inp.shape[0] for inp, _ in patches)
    max_w = max(inp.shape[1] for inp, _ in patches)
    total_area = sum(inp.shape[0] * inp.shape[1] for inp, _ in patches)
    min_side = int(np.ceil(np.sqrt(total_area))) + 2
    min_h = max(max_h, min_side, 10)
    min_w = max(max_w, min_side, 10)
    max_h_choice = min(30, max(min_h, min_h + 8))
    max_w_choice = min(30, max(min_w, min_w + 8))
    if min_h > 30 or min_w > 30:
        return None

    height = random.randint(min_h, max_h_choice)
    width = random.randint(min_w, max_w_choice)
    rects: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    order = list(range(len(patches)))
    random.shuffle(order)

    for idx in order:
        patch_in = patches[idx][0]
        ph, pw = patch_in.shape
        candidates = [
            (r0, c0)
            for r0 in range(height - ph + 1)
            for c0 in range(width - pw + 1)
            if _rect_clear(rects, (r0, c0, r0 + ph - 1, c0 + pw - 1))
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        positions[idx] = (r0, c0)
        rects.append((r0, c0, r0 + ph - 1, c0 + pw - 1))

    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    for idx, pos in enumerate(positions):
        assert pos is not None
        r0, c0 = pos
        patch_in, patch_out = patches[idx]
        _paste_patch(inp, patch_in, r0, c0)
        _paste_patch(out, patch_out, r0, c0)
    return inp, out


def _generate_attempt() -> np.ndarray | None:
    object_count = random.randint(1, 4)
    colors = random.sample(PALETTE, object_count)
    patches = [_build_object_patch(color) for color in colors]
    packed = _pack_patches(patches)
    if packed is None:
        return None

    inp, expected = packed
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
