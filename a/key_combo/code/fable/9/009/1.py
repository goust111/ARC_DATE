from __future__ import annotations

from collections import Counter
import random

import numpy as np


ALL_COLORS = list(range(10))

# Generator bounds.
GRID_H_RANGE = (20, 28)
GRID_W_RANGE = (20, 28)
FRAME_COUNT_CHOICES = (4, 5)
FRAME_ROW_SPAN_RANGE = (5, 11)  # bbox height  = span + 1
FRAME_COL_SPAN_RANGE = (6, 13)  # bbox width   = span + 1
FIRST_FRAME_TOP_MIN = 2
CONSEC_OVERLAP_COUNTS = {2, 4}
EARLIER_OVERLAP_COUNTS = {0, 2, 4}
PLACEMENT_TRIES = 300
GENERATION_TRIES = 500
MIN_CONTROL_GAP = 2


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    assert r0 < r1 and c0 < c1
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _frame_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _unique_topo_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int] | None:
    above = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}

    for lower, higher in edges:
        if lower == higher:
            return None
        if higher not in above[lower]:
            above[lower].add(higher)
            indegree[higher] += 1

    order: list[int] = []
    placed: set[int] = set()
    while len(order) < len(colors):
        available = [c for c in colors if c not in placed and indegree[c] == 0]
        if len(available) != 1:
            return None
        color = available[0]
        order.append(color)
        placed.add(color)
        for higher in above[color]:
            indegree[higher] -= 1

    return order


def _recover_scene(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, int]], dict[int, np.ndarray], list[int], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3

    bg = _get_bg(g)
    control_cols = [int(c) for c in np.flatnonzero(g[0] != bg)]
    controls = [(c, int(g[0, c])) for c in control_cols]
    assert len(controls) >= 2, "expected multiple control pixels on the top row"

    body = g[1:, :]
    frame_colors = _colors_present(body, exclude={bg})
    assert frame_colors, "expected at least one visible frame below the control row"

    frame_masks: dict[int, np.ndarray] = {}
    for color in frame_colors:
        visible = np.zeros_like(g, dtype=bool)
        visible[1:, :] = body == color
        b = _bbox(visible)
        assert b is not None, "each frame color must remain visible"
        assert b[0] >= 1 and b[0] < b[2] and b[1] < b[3], "each frame must be a non-degenerate rectangle below row 0"
        mask = _frame_mask(g.shape, b)
        assert np.all(mask[visible]), "visible pixels must lie on the reconstructed frame border"
        frame_masks[int(color)] = mask

    for _col, color in controls:
        assert color in frame_masks, "every control pixel color must match some frame color"

    edges: set[tuple[int, int]] = set()
    H, W = g.shape
    for r in range(1, H):
        for c in range(W):
            cell = int(g[r, c])
            if cell == bg:
                continue
            covering = [color for color, mask in frame_masks.items() if mask[r, c]]
            assert cell in covering, "visible frame color must be one of the covering reconstructed frames"
            for color in covering:
                if color != cell:
                    edges.add((color, cell))

    order = _unique_topo_order(frame_colors, edges)
    assert order is not None, "frame overlaps must determine a unique bottom-to-top layer order"
    rank = {color: idx for idx, color in enumerate(order)}
    return bg, controls, frame_masks, order, rank


def _analyze_control(
    grid: np.ndarray,
    bg: int,
    rank: dict[int, int],
    src_color: int,
    col: int,
) -> dict:
    H = grid.shape[0]
    stop_row = H - 1
    block_row = None
    block_color = None
    hits_any_frame = False
    passed_rows_by_color: dict[int, list[int]] = {}

    for r in range(1, H):
        cell = int(grid[r, col])
        if cell == bg:
            continue
        hits_any_frame = True
        if rank[cell] > rank[src_color]:
            block_row = r
            block_color = cell
            stop_row = r - 1
            break
        passed_rows_by_color.setdefault(cell, []).append(r)

    painted_rows = [r for r in range(1, stop_row + 1) if int(grid[r, col]) == bg]
    meaningful_pass_colors = set()
    for color, rows in passed_rows_by_color.items():
        for rr in rows:
            if any(int(grid[r, col]) == bg for r in range(rr + 1, stop_row + 1)):
                meaningful_pass_colors.add(color)
                break

    meaningful_block = bool(
        block_row is not None and any(int(grid[r, col]) == bg for r in range(block_row + 1, H))
    )

    return {
        "src_color": int(src_color),
        "col": int(col),
        "blocked": block_row is not None,
        "block_row": None if block_row is None else int(block_row),
        "block_color": None if block_color is None else int(block_color),
        "hits_any_frame": bool(hits_any_frame),
        "stop_row": int(stop_row),
        "painted_rows": painted_rows,
        "meaningful_block": meaningful_block,
        "meaningful_pass_colors": meaningful_pass_colors - {int(src_color)},
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot each top-row control color downward, letting it pass only through frame edges that are not stacked above its own frame.

    Input structure:
    - The background is a single dominant color.
    - Rows below the top row contain several one-cell-thick rectangular frames of distinct colors; overlaps show only the highest frame color.
    - The top row contains multiple control pixels, and each control color matches one frame color below.

    Rule:
    - Reconstruct each frame as the border of the bounding box of its visible pixels, then recover the unique global bottom-to-top layer order from overlap cells.
    - For each control pixel, scan downward in its column: background cells are recolored to the control color; visible frame cells whose frame layer is not above the control frame are traversed without repainting; the first visible frame cell whose layer is above the control frame blocks the ray, so it stops one row above that cell.
    - Keep all frame pixels and the top-row control pixels unchanged.

    Key insight:
    - The ray length is governed by hidden global occlusion order, not by local color identity or first-contact geometry, so the frame stack must be reconstructed before any column can be painted correctly.
    """
    g = np.asarray(grid, dtype=int)
    bg, controls, _frame_masks, _order, rank = _recover_scene(g)

    out = g.copy()
    H = g.shape[0]
    for col, src_color in controls:
        src_rank = rank[src_color]
        for r in range(1, H):
            cell = int(g[r, col])
            if cell == bg:
                out[r, col] = src_color
                continue
            if rank[cell] > src_rank:
                break

    return out


def _random_bbox(H: int, W: int) -> tuple[int, int, int, int]:
    row_span = random.randint(FRAME_ROW_SPAN_RANGE[0], min(FRAME_ROW_SPAN_RANGE[1], H - 4))
    col_span = random.randint(FRAME_COL_SPAN_RANGE[0], min(FRAME_COL_SPAN_RANGE[1], W - 3))
    r0 = random.randint(FIRST_FRAME_TOP_MIN, H - 1 - row_span)
    c0 = random.randint(0, W - 1 - col_span)
    return (r0, c0, r0 + row_span, c0 + col_span)


def _place_frame_stack(H: int, W: int, n_frames: int):
    first = _random_bbox(H, W)
    bboxes = [first]
    cell_sets = [_frame_cells(first)]

    for _ in range(1, n_frames):
        prev = bboxes[-1]
        pr0, pc0, pr1, pc1 = prev
        placed = None
        for _try in range(PLACEMENT_TRIES):
            row_span = random.randint(FRAME_ROW_SPAN_RANGE[0], min(FRAME_ROW_SPAN_RANGE[1], H - 4))
            col_span = random.randint(FRAME_COL_SPAN_RANGE[0], min(FRAME_COL_SPAN_RANGE[1], W - 3))

            r0_lo = max(FIRST_FRAME_TOP_MIN, pr0 - row_span + 1)
            r0_hi = min(H - 1 - row_span, pr1)
            c0_lo = max(0, pc0 - col_span + 1)
            c0_hi = min(W - 1 - col_span, pc1)
            if r0_lo > r0_hi or c0_lo > c0_hi:
                continue

            r0 = random.randint(r0_lo, r0_hi)
            c0 = random.randint(c0_lo, c0_hi)
            bbox = (r0, c0, r0 + row_span, c0 + col_span)
            cells = _frame_cells(bbox)
            if len(cells & cell_sets[-1]) not in CONSEC_OVERLAP_COUNTS:
                continue
            if any(len(cells & older) not in EARLIER_OVERLAP_COUNTS for older in cell_sets[:-1]):
                continue
            placed = (bbox, cells)
            break
        if placed is None:
            return None
        bbox, cells = placed
        bboxes.append(bbox)
        cell_sets.append(cells)

    return bboxes, cell_sets


def _render_frames(shape: tuple[int, int], bg: int, layer_order: list[int], cell_sets: list[set[tuple[int, int]]]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for color, cells in zip(layer_order, cell_sets):
        rows, cols = zip(*cells)
        grid[rows, cols] = color
    return grid


def _visible_bboxes_match(grid: np.ndarray, layer_order: list[int], bboxes: list[tuple[int, int, int, int]]) -> bool:
    for color, bbox in zip(layer_order, bboxes):
        if _bbox(grid == color) != bbox:
            return False
    return True


def _overlap_edges(layer_order: list[int], cell_sets: list[set[tuple[int, int]]]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i in range(len(layer_order)):
        for j in range(i + 1, len(layer_order)):
            if len(cell_sets[i] & cell_sets[j]) >= 2:
                edges.add((layer_order[i], layer_order[j]))
    return edges


def _find_witness_pair(candidates: list[dict], rank: dict[int, int]) -> tuple[dict, dict] | None:
    for blocked in candidates:
        if not (blocked["blocked"] and blocked["meaningful_block"]):
            continue
        witness = blocked["block_color"]
        for passed in candidates:
            if passed["src_color"] == blocked["src_color"] or passed["col"] == blocked["col"]:
                continue
            if abs(passed["col"] - blocked["col"]) < MIN_CONTROL_GAP:
                continue
            if rank[passed["src_color"]] <= rank[witness]:
                continue
            if witness in passed["meaningful_pass_colors"]:
                return blocked, passed
    return None


def _choose_controls(body_grid: np.ndarray, layer_order: list[int]) -> list[dict] | None:
    H, W = body_grid.shape
    full = np.full((H, W), _get_bg(body_grid), dtype=int)
    full[:, :] = body_grid
    bg = _get_bg(full)
    rank = {color: idx for idx, color in enumerate(layer_order)}

    candidates = []
    for src_color in layer_order:
        for col in range(W):
            stats = _analyze_control(full, bg, rank, src_color, col)
            if stats["hits_any_frame"]:
                candidates.append(stats)

    witness_pair = _find_witness_pair(candidates, rank)
    if witness_pair is None:
        return None

    selected = [witness_pair[0], witness_pair[1]]
    used_colors = {item["src_color"] for item in selected}
    used_cols = {item["col"] for item in selected}
    target_count = random.randint(2, min(4, len(layer_order)))

    extras = candidates[:]
    random.shuffle(extras)
    for cand in extras:
        if len(selected) >= target_count:
            break
        if cand["src_color"] in used_colors or cand["col"] in used_cols:
            continue
        if any(abs(cand["col"] - prev["col"]) < MIN_CONTROL_GAP for prev in selected):
            continue
        if not (cand["meaningful_block"] or cand["meaningful_pass_colors"]):
            continue
        selected.append(cand)
        used_colors.add(cand["src_color"])
        used_cols.add(cand["col"])

    return selected


def _selected_controls_have_witness_pair(grid: np.ndarray, rank: dict[int, int], controls: list[tuple[int, int]]) -> bool:
    bg = _get_bg(grid)
    stats = [_analyze_control(grid, bg, rank, color, col) for col, color in controls]
    return _find_witness_pair(stats, rank) is not None


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        bg, controls, _frame_masks, order, rank = _recover_scene(grid)
        if _get_bg(grid) != bg:
            return False
        if not all(_analyze_control(grid, bg, rank, color, col)["hits_any_frame"] for col, color in controls):
            return False
        stats = [_analyze_control(grid, bg, rank, color, col) for col, color in controls]
        if not any(stat["meaningful_block"] for stat in stats):
            return False
        if not any(stat["meaningful_pass_colors"] for stat in stats):
            return False
        if not _selected_controls_have_witness_pair(grid, rank, controls):
            return False
        out = main(grid)
        if np.array_equal(out, grid):
            return False
        if out.shape != grid.shape:
            return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(ALL_COLORS)
    frame_colors = [color for color in ALL_COLORS if color != bg]

    n_frames = random.choice(FRAME_COUNT_CHOICES)
    layer_order = random.sample(frame_colors, n_frames)
    random.shuffle(layer_order)

    placed = _place_frame_stack(H, W, n_frames)
    if placed is None:
        return None
    bboxes, cell_sets = placed

    edges = _overlap_edges(layer_order, cell_sets)
    if _unique_topo_order(layer_order, edges) != layer_order:
        return None

    body = _render_frames((H, W), bg, layer_order, cell_sets)
    if not _visible_bboxes_match(body, layer_order, bboxes):
        return None
    if _get_bg(body) != bg:
        return None

    selected = _choose_controls(body, layer_order)
    if selected is None:
        return None

    grid = body.copy()
    grid[0, :] = bg
    for item in selected:
        grid[0, item["col"]] = item["src_color"]

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_TRIES} attempts")
