from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(map(int, np.asarray(grid, dtype=int).flat)))
    if exclude:
        colors = [color for color in colors if color not in exclude]
    return colors


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "each hollow frame must be at least 3x3"
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    above = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for lower, higher in edges:
        assert lower != higher
        if higher not in above[lower]:
            above[lower].add(higher)
            indegree[higher] += 1

    order: list[int] = []
    placed: set[int] = set()
    while len(order) < len(colors):
        available = [color for color in colors if color not in placed and indegree[color] == 0]
        assert len(available) == 1, "overlap constraints must determine a unique order"
        color = int(available[0])
        order.append(color)
        placed.add(color)
        for higher in above[color]:
            indegree[higher] -= 1
    return order


def _reconstruct_frames(grid: np.ndarray) -> tuple[int, list[int], dict[int, np.ndarray], dict[int, tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = _colors_present(g, exclude={bg})
    assert len(colors) >= 2, "expected multiple foreground frames"

    frame_masks: dict[int, np.ndarray] = {}
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for color in colors:
        visible = g == color
        box = _bbox(visible)
        assert box is not None, "every frame color must be visible"
        mask = _frame_mask(g.shape, box)
        assert np.all(mask[visible]), "visible pixels must lie on the reconstructed frame border"
        frame_masks[color] = mask
        boxes[color] = box
    return bg, colors, frame_masks, boxes


def _infer_layer_order(grid: np.ndarray, colors: list[int], frame_masks: dict[int, np.ndarray]) -> tuple[list[int], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    edges: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            covering = [color for color in colors if frame_masks[color][r, c]]
            if not covering:
                continue
            top = int(g[r, c])
            assert top in covering, "every reconstructed frame cell must show the top covering color"
            for color in covering:
                if color != top:
                    edges.add((int(color), top))

    assert edges, "frame overlaps must reveal at least one layer constraint"
    return _unique_topological_order(colors, edges), edges


def _drop_frames(
    order: list[int],
    frame_masks: dict[int, np.ndarray],
    bg: int,
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    shape = next(iter(frame_masks.values())).shape
    h, _w = shape
    out = np.full(shape, bg, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    info: list[dict[str, int | str]] = []

    coords_by_color = {
        color: np.argwhere(mask)
        for color, mask in frame_masks.items()
    }

    for color in order:
        coords = coords_by_color[color]
        assert not occupied[coords[:, 0], coords[:, 1]].any(), "a falling frame cannot start inside a settled frame"

        shift = 0
        reason = "ground"
        while True:
            next_rows = coords[:, 0] + shift + 1
            cols = coords[:, 1]
            if int(next_rows.max()) >= h:
                reason = "ground"
                break
            if occupied[next_rows, cols].any():
                reason = "support"
                break
            shift += 1

        final_rows = coords[:, 0] + shift
        final_cols = coords[:, 1]
        assert not occupied[final_rows, final_cols].any()
        occupied[final_rows, final_cols] = True
        out[final_rows, final_cols] = int(color)
        info.append({"color": int(color), "shift": int(shift), "reason": reason})

    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full rectangular frames, recover their bottom-to-top occlusion order, and drop them in that order.

    Input structure:
    - The grid has one dominant background color and several distinct foreground colors.
    - Each foreground color belongs to a one-cell-thick hollow rectangle frame, but higher frames may hide parts of lower ones.
    - The bottom edge of the grid is the ground.

    Rule:
    - Reconstruct every full frame as the border of the bounding box of that color's visible pixels.
    - At each cell covered by multiple reconstructed frames, the visible color is the highest layer there; use all such constraints to recover the unique bottom-to-top order.
    - Starting from the bottom layer, drop each frame straight downward until moving it one more row would overlap the ground or any already settled frame; frames that have not dropped yet are ignored.

    Key insight:
    - The overlap colors encode the vertical layer order, and that same order becomes the time order of the gravity simulation, so both reconstruction and ordered dropping are necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, colors, frame_masks, _boxes = _reconstruct_frames(g)
    order, _edges = _infer_layer_order(g, colors, frame_masks)
    out, _info = _drop_frames(order, frame_masks, bg)
    return out


PALETTE = list(range(10))
GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (16, 24)
FRAME_COUNT_RANGE = (3, 5)
FRAME_HEIGHT_RANGE = (3, 8)
FRAME_WIDTH_RANGE = (4, 11)
REQUIRED_PREV_OVERLAPS = (2, 4)
ALLOWED_PRIOR_OVERLAPS = (0, 2, 4)
PLACEMENT_TRIES = 250
GENERATION_TRIES = 500


def _frame_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _sample_palette(n_frames: int) -> tuple[int, list[int]]:
    bg = random.choice(PALETTE)
    colors = random.sample([color for color in PALETTE if color != bg], n_frames)
    random.shuffle(colors)
    return bg, colors


def _sample_frame_size(h: int, w: int) -> tuple[int, int]:
    max_height = min(FRAME_HEIGHT_RANGE[1], max(FRAME_HEIGHT_RANGE[0], h // 2 - 1))
    max_width = min(FRAME_WIDTH_RANGE[1], w - 2)
    if max_height < FRAME_HEIGHT_RANGE[0] or max_width < FRAME_WIDTH_RANGE[0]:
        raise ValueError("grid too small for frame generation")
    height = random.randint(FRAME_HEIGHT_RANGE[0], max_height)
    width = random.randint(FRAME_WIDTH_RANGE[0], max_width)
    return height, width


def _place_first_frame(h: int, w: int) -> tuple[tuple[int, int, int, int], set[tuple[int, int]]]:
    height, width = _sample_frame_size(h, w)
    bottom_cap = h // 2 - 1
    max_r0 = bottom_cap - height + 1
    assert max_r0 >= 0
    r0 = random.randint(0, max_r0)
    c0 = random.randint(0, w - width)
    box = (r0, c0, r0 + height - 1, c0 + width - 1)
    return box, _frame_cells(box)


def _place_next_frame(
    h: int,
    w: int,
    prev_box: tuple[int, int, int, int],
    prior_cell_sets: list[set[tuple[int, int]]],
) -> tuple[tuple[int, int, int, int], set[tuple[int, int]]] | None:
    prev_r0, prev_c0, prev_r1, prev_c1 = prev_box
    bottom_cap = h // 2 - 1

    for _ in range(PLACEMENT_TRIES):
        height, width = _sample_frame_size(h, w)
        r0_lo = max(0, prev_r0 - height + 1)
        r0_hi = min(bottom_cap - height + 1, prev_r1)
        c0_lo = max(0, prev_c0 - width + 1)
        c0_hi = min(w - width, prev_c1)
        if r0_lo > r0_hi or c0_lo > c0_hi:
            continue

        r0 = random.randint(r0_lo, r0_hi)
        c0 = random.randint(c0_lo, c0_hi)
        box = (r0, c0, r0 + height - 1, c0 + width - 1)
        cells = _frame_cells(box)

        if len(cells & prior_cell_sets[-1]) not in REQUIRED_PREV_OVERLAPS:
            continue
        if any(len(cells & old_cells) not in ALLOWED_PRIOR_OVERLAPS for old_cells in prior_cell_sets[:-1]):
            continue
        return box, cells
    return None


def _place_frame_stack(h: int, w: int, n_frames: int) -> tuple[list[tuple[int, int, int, int]], list[set[tuple[int, int]]]] | None:
    first_box, first_cells = _place_first_frame(h, w)
    boxes = [first_box]
    cell_sets = [first_cells]
    for _ in range(1, n_frames):
        placed = _place_next_frame(h, w, boxes[-1], cell_sets)
        if placed is None:
            return None
        box, cells = placed
        boxes.append(box)
        cell_sets.append(cells)
    return boxes, cell_sets


def _render_input(
    shape: tuple[int, int],
    bg: int,
    layer_order: list[int],
    cell_sets: list[set[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for color, cells in zip(layer_order, cell_sets, strict=False):
        for r, c in cells:
            grid[r, c] = int(color)
    return grid


def _edges_from_order(layer_order: list[int], cell_sets: list[set[tuple[int, int]]]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i in range(len(layer_order)):
        for j in range(i + 1, len(layer_order)):
            if cell_sets[i] & cell_sets[j]:
                edges.add((int(layer_order[i]), int(layer_order[j])))
    return edges


def _visible_bboxes_match(
    grid: np.ndarray,
    layer_order: list[int],
    boxes: list[tuple[int, int, int, int]],
) -> bool:
    for color, box in zip(layer_order, boxes, strict=False):
        if _bbox(grid == color) != box:
            return False
    return True


def _reconstructable_frame_masks(
    shape: tuple[int, int],
    layer_order: list[int],
    boxes: list[tuple[int, int, int, int]],
) -> dict[int, np.ndarray]:
    return {
        int(color): _frame_mask(shape, box)
        for color, box in zip(layer_order, boxes, strict=False)
    }


def _alternative_orders(order: list[int]) -> list[list[int]]:
    alts = [list(reversed(order))]
    for i in range(len(order) - 1):
        alt = order[:]
        alt[i], alt[i + 1] = alt[i + 1], alt[i]
        alts.append(alt)
    return alts


def _semantic_ok(
    inp: np.ndarray,
    expected_order: list[int],
    boxes: list[tuple[int, int, int, int]],
) -> bool:
    try:
        bg, colors, frame_masks, _ = _reconstruct_frames(inp)
        inferred_order, edges = _infer_layer_order(inp, colors, frame_masks)
        if inferred_order != expected_order:
            return False
        if _unique_topological_order(colors, edges) != expected_order:
            return False

        expected_masks = _reconstructable_frame_masks(inp.shape, expected_order, boxes)
        out, info = _drop_frames(expected_order, expected_masks, bg)
        if not np.array_equal(main(inp), out):
            return False
        if not any(int(item["shift"]) > 0 for item in info):
            return False
        if not any(item["reason"] == "ground" for item in info):
            return False
        if not any(item["reason"] == "support" for item in info):
            return False

        sensitive = False
        for alt in _alternative_orders(expected_order):
            try:
                alt_out, _alt_info = _drop_frames(alt, expected_masks, bg)
            except AssertionError:
                sensitive = True
                break
            if not np.array_equal(alt_out, out):
                sensitive = True
                break
        if not sensitive:
            return False

        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    n_frames = random.randint(*FRAME_COUNT_RANGE)

    placed = _place_frame_stack(h, w, n_frames)
    if placed is None:
        return None
    boxes, cell_sets = placed

    bg, layer_order = _sample_palette(n_frames)
    edges = _edges_from_order(layer_order, cell_sets)
    topo = _unique_topological_order(layer_order, edges)
    if topo != layer_order:
        return None

    inp = _render_input((h, w), bg, layer_order, cell_sets)
    if not _visible_bboxes_match(inp, layer_order, boxes):
        return None
    if not _semantic_ok(inp, layer_order, boxes):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layer-drop input")
