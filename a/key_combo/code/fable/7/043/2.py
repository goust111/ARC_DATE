from __future__ import annotations

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
GEN_RETRIES = 400


def _neighbors(shape: tuple[int, int], r: int, c: int, connectivity: int) -> tuple[tuple[int, int], ...]:
    deltas = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    out: list[tuple[int, int]] = []
    h, w = shape
    for dr, dc in deltas:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return tuple(out)


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors(mask.shape, rr, cc, connectivity):
                    if mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_colors(grid: np.ndarray, coords: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted({int(grid[int(r), int(c)]) for r, c in coords}))


def _component_patch(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch


def _stamp_output(layout_patch: np.ndarray, stamp_by_color: dict[int, np.ndarray]) -> np.ndarray:
    layout_h, layout_w = layout_patch.shape
    sample_stamp = next(iter(stamp_by_color.values()))
    stamp_h, stamp_w = sample_stamp.shape
    out = np.full((layout_h * stamp_h, layout_w * stamp_w), BLACK, dtype=int)
    for r in range(layout_h):
        for c in range(layout_w):
            color = int(layout_patch[r, c])
            if color == BLACK:
                continue
            stamp = stamp_by_color[color]
            block = out[r * stamp_h : (r + 1) * stamp_h, c * stamp_w : (c + 1) * stamp_w]
            mask = stamp != BLACK
            block[mask] = stamp[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a two-color layout by stamping one of two same-sized monochrome sprites into each layout cell.

    Input structure:
    - The grid has a black background, one connected two-color layout object, and two monochrome stamp objects.
    - The two stamp objects use exactly the layout's two colors, have the same bounding-box size, and different shapes.
    - Additional isolated single-pixel distractors may appear anywhere on the background.

    Rule:
    - Ignore every isolated single-pixel component.
    - Find the two-color layout and crop its bounding box.
    - For each non-background cell in that bounding box, paste the same-colored stamp into the corresponding output block; background layout cells map to empty blocks.

    Key insight:
    - Layout colors are addressing keys rather than final appearance: each colored layout cell chooses which of the two stamp shapes to place in a larger tiled canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    foreground_components = _components(g != BLACK, connectivity=8)
    nontrivial = [coords for coords in foreground_components if len(coords) > 1]
    assert len(nontrivial) == 3, "expected exactly three non-isolated objects"

    color_sets = [_component_colors(g, coords) for coords in nontrivial]
    layout_indices = [i for i, colors in enumerate(color_sets) if len(colors) == 2]
    assert len(layout_indices) == 1, "expected one two-color layout object"
    layout_idx = int(layout_indices[0])
    layout_coords = nontrivial[layout_idx]
    layout_colors = set(color_sets[layout_idx])

    stamp_items: list[tuple[int, np.ndarray]] = []
    for i, coords in enumerate(nontrivial):
        if i == layout_idx:
            continue
        colors = color_sets[i]
        assert len(colors) == 1 and colors[0] in layout_colors, "stamps must be monochrome and match layout colors"
        stamp_items.append((int(colors[0]), coords))

    assert len(stamp_items) == 2
    assert {color for color, _coords in stamp_items} == layout_colors, "expected one stamp per layout color"

    stamp_by_color: dict[int, np.ndarray] = {}
    stamp_shape: tuple[int, int] | None = None
    stamp_masks: list[np.ndarray] = []
    for color, coords in stamp_items:
        patch = _component_patch(g, coords)
        if stamp_shape is None:
            stamp_shape = patch.shape
        assert patch.shape == stamp_shape, "stamps must share the same bbox size"
        stamp_by_color[color] = patch
        stamp_masks.append(patch != BLACK)

    assert stamp_shape is not None
    assert not np.array_equal(stamp_masks[0], stamp_masks[1]), "stamps must have different shapes"

    layout_patch = _component_patch(g, layout_coords)
    valid_layout_colors = layout_colors | {BLACK}
    assert set(map(int, np.unique(layout_patch))) <= valid_layout_colors
    return _stamp_output(layout_patch, stamp_by_color)


def _randint(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _mask_from_cells(h: int, w: int, cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_path(start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    gr, gc = goal
    cells = [(r, c)]
    while (r, c) != (gr, gc):
        choices: list[tuple[int, int]] = []
        if r < gr:
            choices.append((r + 1, c))
        elif r > gr:
            choices.append((r - 1, c))
        if c < gc:
            choices.append((r, c + 1))
        elif c > gc:
            choices.append((r, c - 1))
        r, c = choices[_randint(0, len(choices) - 1)]
        cells.append((r, c))
    return cells


def _connected_mask(h: int, w: int, min_fill: int, max_fill: int) -> np.ndarray | None:
    area = h * w
    assert 1 <= min_fill <= max_fill <= area

    anchors = [
        (0, _randint(0, w - 1)),
        (h - 1, _randint(0, w - 1)),
        (_randint(0, h - 1), 0),
        (_randint(0, h - 1), w - 1),
    ]
    order = list(np.random.permutation(len(anchors)))
    cells = {anchors[order[0]]}
    for idx in order[1:]:
        anchor = anchors[idx]
        existing = tuple(cells)[_randint(0, len(cells) - 1)]
        for cell in _random_path(anchor, existing):
            cells.add(cell)
        if len(cells) > max_fill:
            return None

    target_fill = _randint(max(min_fill, len(cells)), max_fill)
    while len(cells) < target_fill:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if not frontier:
            return None
        frontier_list = tuple(frontier)
        cells.add(frontier_list[_randint(0, len(frontier_list) - 1)])

    mask = _mask_from_cells(h, w, cells)
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    if (r0, c0, r1, c1) != (0, 0, h - 1, w - 1):
        return None
    return mask


def _make_stamp_pair(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    area = h * w
    min_fill = max(3, (area + 1) // 2)
    max_fill = area - 1 if area > 4 else area
    if min_fill > max_fill:
        return None

    for _ in range(80):
        first = _connected_mask(h, w, min_fill, max_fill)
        second = _connected_mask(h, w, min_fill, max_fill)
        if first is None or second is None:
            continue
        if np.array_equal(first, second):
            continue
        return first, second
    return None


def _make_layout_mask(h: int, w: int) -> np.ndarray | None:
    area = h * w
    min_fill = max(3, (area * 3 + 4) // 5)
    max_fill = area - 1
    if min_fill > max_fill:
        min_fill = area - 1
    if min_fill < 3 or max_fill < 3:
        return None
    return _connected_mask(h, w, min_fill, max_fill)


def _color_layout(mask: np.ndarray, colors: tuple[int, int]) -> np.ndarray | None:
    coords = np.argwhere(mask)
    if len(coords) < 2:
        return None
    chosen = np.array([colors[_randint(0, 1)] for _ in range(len(coords))], dtype=int)
    chosen[0] = colors[0]
    chosen[1] = colors[1]
    if np.all(chosen == colors[0]) or np.all(chosen == colors[1]):
        return None
    patch = np.full(mask.shape, BLACK, dtype=int)
    for (r, c), color in zip(coords, chosen):
        patch[int(r), int(c)] = int(color)
    return patch


def _render_stamp(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BLACK, dtype=int)
    patch[mask] = int(color)
    return patch


def _rect_conflicts(
    top: int,
    left: int,
    h: int,
    w: int,
    placed: list[tuple[int, int, int, int]],
    pad: int = 1,
) -> bool:
    bottom = top + h - 1
    right = left + w - 1
    for other_top, other_left, other_h, other_w in placed:
        other_bottom = other_top + other_h - 1
        other_right = other_left + other_w - 1
        separated = (
            bottom + pad < other_top
            or other_bottom + pad < top
            or right + pad < other_left
            or other_right + pad < left
        )
        if not separated:
            return True
    return False


def _place_rectangles(shapes: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]] | None:
    order = list(np.random.permutation(len(shapes)))
    positions = [(-1, -1)] * len(shapes)
    placed: list[tuple[int, int, int, int]] = []
    for idx in order:
        obj_h, obj_w = shapes[idx]
        candidates: list[tuple[int, int]] = []
        for r0 in range(h - obj_h + 1):
            for c0 in range(w - obj_w + 1):
                if not _rect_conflicts(r0, c0, obj_h, obj_w, placed, pad=1):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        positions[idx] = candidates[_randint(0, len(candidates) - 1)]
        placed.append((positions[idx][0], positions[idx][1], obj_h, obj_w))
    return positions


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != BLACK
    target = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    target[mask] = patch[mask]


def _isolated_empty_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLACK:
                continue
            r0 = max(0, r - 1)
            r1 = min(h, r + 2)
            c0 = max(0, c - 1)
            c1 = min(w, c + 2)
            if np.any(grid[r0:r1, c0:c1] != BLACK):
                continue
            cells.append((r, c))
    return cells


def _add_noise(grid: np.ndarray, layout_colors: tuple[int, int]) -> None:
    cells = _isolated_empty_cells(grid)
    if not cells:
        return
    np.random.shuffle(cells)
    count = _randint(1, min(5, len(cells)))
    palette = list(PALETTE)
    for r, c in cells[:count]:
        if np.random.random() < 0.5:
            grid[r, c] = layout_colors[_randint(0, 1)]
        else:
            grid[r, c] = palette[_randint(0, len(palette) - 1)]


def _attempt_generate() -> np.ndarray | None:
    stamp_h = _randint(2, 5)
    stamp_w = _randint(2, 5)
    layout_h = _randint(2, min(6, 30 // stamp_h))
    layout_w = _randint(2, min(6, 30 // stamp_w))

    stamp_masks = _make_stamp_pair(stamp_h, stamp_w)
    if stamp_masks is None:
        return None
    layout_mask = _make_layout_mask(layout_h, layout_w)
    if layout_mask is None:
        return None

    colors = tuple(int(x) for x in np.random.choice(PALETTE, size=2, replace=False))
    layout_patch = _color_layout(layout_mask, colors)
    if layout_patch is None:
        return None

    stamp_a = _render_stamp(stamp_masks[0], colors[0])
    stamp_b = _render_stamp(stamp_masks[1], colors[1])
    expected = _stamp_output(layout_patch, {colors[0]: stamp_a, colors[1]: stamp_b})
    if max(expected.shape) > 30:
        return None

    grid_h = _randint(max(12, max(layout_h, stamp_h) + 6), 30)
    grid_w = _randint(max(12, max(layout_w, stamp_w) + 6), 30)
    patches = [layout_patch, stamp_a, stamp_b]
    positions = _place_rectangles([patch.shape for patch in patches], grid_h, grid_w)
    if positions is None:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for patch, (top, left) in zip(patches, positions):
        _paste_patch(grid, patch, top, left)
    _add_noise(grid, colors)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 043/2")
