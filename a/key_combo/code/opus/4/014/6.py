import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
FRAME_COLORS = (BLUE, RED)
FRAME_SIZES = (5, 6)
LAYOUTS = ((2, 3), (3, 2))
MAX_GENERATION_ATTEMPTS = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _make_box(r0: int, c0: int, size: int, color: int) -> dict:
    return {"r0": int(r0), "c0": int(c0), "size": int(size), "color": int(color)}


def _box_bounds(box: dict) -> tuple[int, int, int, int]:
    r0 = int(box["r0"])
    c0 = int(box["c0"])
    size = int(box["size"])
    return r0, c0, r0 + size - 1, c0 + size - 1


def _interior_size(box: dict) -> int:
    return int(box["size"]) - 2


def _is_frame_candidate(grid: np.ndarray, r0: int, c0: int, size: int, color: int) -> bool:
    h, w = grid.shape
    if r0 < 0 or c0 < 0 or r0 + size > h or c0 + size > w:
        return False

    patch = grid[r0 : r0 + size, c0 : c0 + size]
    if np.any(~np.isin(patch[1:-1, 1:-1], (BLACK, color))):
        return False
    if not np.all(patch[0, :] == color):
        return False
    if not np.all(patch[-1, :] == color):
        return False
    if not np.all(patch[:, 0] == color):
        return False
    if not np.all(patch[:, -1] == color):
        return False

    r1 = r0 + size - 1
    c1 = c0 + size - 1
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(h - 1, r1 + 1)
    pc1 = min(w - 1, c1 + 1)
    padded = grid[pr0 : pr1 + 1, pc0 : pc1 + 1]
    surround = np.ones_like(padded, dtype=bool)
    surround[r0 - pr0 : r1 - pr0 + 1, c0 - pc0 : c1 - pc0 + 1] = False
    return np.all(padded[surround] == BLACK)


def _find_boxes(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    boxes: list[dict] = []
    used = np.zeros_like(g, dtype=bool)
    for size in sorted(FRAME_SIZES, reverse=True):
        for color in FRAME_COLORS:
            for r0 in range(g.shape[0] - size + 1):
                for c0 in range(g.shape[1] - size + 1):
                    if not _is_frame_candidate(g, r0, c0, size, color):
                        continue
                    mask = np.zeros_like(g, dtype=bool)
                    mask[r0 : r0 + size, c0 : c0 + size] = True
                    if np.any(used & mask):
                        continue
                    boxes.append(_make_box(r0, c0, size, color))
                    used[mask] = True
    boxes.sort(key=lambda box: (box["r0"], box["c0"]))
    return boxes


def _marker_set(grid: np.ndarray, box: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = _box_bounds(box)
    patch = grid[r0 + 1 : r1, c0 + 1 : c1]
    return {tuple(map(int, rc)) for rc in np.argwhere(patch == int(box["color"]))}


def _union_markers_per_color(grid: np.ndarray, boxes: list[dict]) -> dict[int, set[tuple[int, int]]]:
    by_color: dict[int, set[tuple[int, int]]] = {}
    for box in boxes:
        color = int(box["color"])
        by_color.setdefault(color, set()).update(_marker_set(grid, box))
    return by_color


def _span_cells(markers: set[tuple[int, int]], inner_size: int) -> set[tuple[int, int]]:
    cells = set(markers)
    for rr in range(inner_size):
        cols = sorted(cc for r, cc in markers if r == rr)
        if len(cols) >= 2:
            for cc in range(cols[0], cols[-1] + 1):
                cells.add((rr, cc))
    for cc in range(inner_size):
        rows = sorted(rr for rr, c in markers if c == cc)
        if len(rows) >= 2:
            for rr in range(rows[0], rows[-1] + 1):
                cells.add((rr, cc))
    return cells


def _render(grid: np.ndarray, boxes: list[dict], marker_sets: list[set[tuple[int, int]]], *, add_lines: bool) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for box, markers in zip(boxes, marker_sets):
        color = int(box["color"])
        r0 = int(box["r0"]) + 1
        c0 = int(box["c0"]) + 1
        for rr, cc in markers:
            out[r0 + rr, c0 + cc] = color
        if not add_lines:
            continue
        for rr, cc in _span_cells(markers, _interior_size(box)):
            out[r0 + rr, c0 + cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete shared same-color frame motifs, then connect the completed markers inside each frame.

    Input structure:
    - The grid has a black background plus several isolated square frames of size 5x5 or 6x6.
    - Each frame has a one-pixel border colored blue or red and an interior containing only black or same-color marker pixels.
    - Frames are separated by black spacer rows and columns, and frames of the same border color may have different missing marker positions.

    Rule:
    - Detect every colored frame and group frames by their border color.
    - For each color group, collect the union of all interior marker positions in frame-relative coordinates and copy that union into every frame of the group.
    - Inside each frame, using those completed marker positions only, fill every row span between the leftmost and rightmost markers in that row and every column span between the topmost and bottommost markers in that column.
    - Keep the colored borders unchanged and leave the background black.

    Key insight:
    - A frame's final line structure depends on markers contributed by other frames of the same border color, because shared completion can create new row or column pairs that were impossible locally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    boxes = _find_boxes(g)
    assert boxes, "expected at least one colored frame"

    color_union = _union_markers_per_color(g, boxes)
    marker_sets = [set(color_union[int(box["color"])]) for box in boxes]
    return _render(g, boxes, marker_sets, add_lines=True)


def _gap_pair(inner: int) -> tuple[int, int]:
    pairs = [(a, b) for a in range(inner) for b in range(a + 2, inner)]
    return random.choice(pairs)


def _make_group_subsets(inner: int, count: int) -> list[set[tuple[int, int]]]:
    mode = random.choice(("h2", "v2", "l3", "rect4"))
    subsets = [set() for _ in range(count)]

    if mode == "h2":
        row = random.randrange(inner)
        c1, c2 = _gap_pair(inner)
        a, b = random.sample(range(count), 2)
        subsets[a].add((row, c1))
        subsets[b].add((row, c2))
        return subsets

    if mode == "v2":
        col = random.randrange(inner)
        r1, r2 = _gap_pair(inner)
        a, b = random.sample(range(count), 2)
        subsets[a].add((r1, col))
        subsets[b].add((r2, col))
        return subsets

    if mode == "l3":
        r1, r2 = _gap_pair(inner)
        c1, c2 = _gap_pair(inner)
        pivot = (random.choice((r1, r2)), random.choice((c1, c2)))
        h_end = (pivot[0], c2 if pivot[1] == c1 else c1)
        v_end = (r2 if pivot[0] == r1 else r1, pivot[1])
        if count == 2:
            a, b = random.sample(range(count), 2)
            subsets[a].add(pivot)
            subsets[b].update((h_end, v_end))
        else:
            a, b, c = random.sample(range(count), 3)
            subsets[a].add(pivot)
            subsets[b].add(h_end)
            subsets[c].add(v_end)
        return subsets

    r1, r2 = _gap_pair(inner)
    c1, c2 = _gap_pair(inner)
    corners = ((r1, c1), (r1, c2), (r2, c1), (r2, c2))
    if count == 2:
        a, b = random.sample(range(count), 2)
        subsets[a].update((corners[0], corners[3]))
        subsets[b].update((corners[1], corners[2]))
    else:
        order = random.sample(range(count), 3)
        subsets[order[0]].update((corners[0], corners[3]))
        subsets[order[1]].add(corners[1])
        subsets[order[2]].add(corners[2])
    return subsets


def _draw_frame(grid: np.ndarray, box: dict) -> None:
    r0, c0, r1, c1 = _box_bounds(box)
    color = int(box["color"])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _gen_attempt() -> np.ndarray | None:
    layout_r, layout_c = random.choice(LAYOUTS)
    size = random.choice(FRAME_SIZES)
    occupied_count = random.choice((4, 5, 6))

    positions = [(r, c) for r in range(layout_r) for c in range(layout_c)]
    chosen_positions = random.sample(positions, occupied_count)

    if occupied_count == 4:
        counts = {BLUE: 2, RED: 2}
    elif occupied_count == 5:
        triple = random.choice(FRAME_COLORS)
        counts = {BLUE: 2, RED: 2}
        counts[triple] = 3
    else:
        counts = {BLUE: 3, RED: 3}

    color_list = [color for color, count in counts.items() for _ in range(count)]
    random.shuffle(color_list)

    boxes: list[dict] = []
    grouped_indices: dict[int, list[int]] = {BLUE: [], RED: []}
    for (lr, lc), color in zip(chosen_positions, color_list):
        boxes.append(_make_box(lr * (size + 1), lc * (size + 1), size, color))
        grouped_indices[color].append(len(boxes) - 1)

    inner = size - 2
    marker_sets: list[set[tuple[int, int]]] = [set() for _ in boxes]
    for color, indices in grouped_indices.items():
        if not indices:
            continue
        subsets = _make_group_subsets(inner, len(indices))
        for idx, subset in zip(indices, subsets):
            marker_sets[idx] = subset

    lattice_h = layout_r * size + (layout_r - 1)
    lattice_w = layout_c * size + (layout_c - 1)
    extra_h = random.randint(0, min(4, 30 - lattice_h))
    extra_w = random.randint(0, min(4, 30 - lattice_w))
    height = lattice_h + extra_h
    width = lattice_w + extra_w
    top = random.randint(0, height - lattice_h)
    left = random.randint(0, width - lattice_w)

    grid = np.zeros((height, width), dtype=int)
    placed_boxes: list[dict] = []
    for box in boxes:
        placed = _make_box(box["r0"] + top, box["c0"] + left, box["size"], box["color"])
        placed_boxes.append(placed)
        _draw_frame(grid, placed)

    for box, markers in zip(placed_boxes, marker_sets):
        color = int(box["color"])
        r0 = int(box["r0"]) + 1
        c0 = int(box["c0"]) + 1
        for rr, cc in markers:
            grid[r0 + rr, c0 + cc] = color

    expected = sorted((box["r0"], box["c0"], box["size"], box["color"]) for box in placed_boxes)
    observed = sorted((box["r0"], box["c0"], box["size"], box["color"]) for box in _find_boxes(grid))
    if observed != expected:
        return None

    return grid


def _is_semantic_instance(grid: np.ndarray) -> bool:
    boxes = _find_boxes(grid)
    if len(boxes) < 4:
        return False

    original_markers = [_marker_set(grid, box) for box in boxes]
    union_by_color = _union_markers_per_color(grid, boxes)
    completed_markers = [set(union_by_color[int(box["color"])]) for box in boxes]

    gained_line = False
    for box, before, after in zip(boxes, original_markers, completed_markers):
        if before == after:
            continue
        if _span_cells(after, _interior_size(box)) - _span_cells(before, _interior_size(box)):
            gained_line = True
            break
    if not gained_line:
        return False

    without_completion = _render(grid, boxes, original_markers, add_lines=True)
    completion_only = _render(grid, boxes, completed_markers, add_lines=False)
    full = _render(grid, boxes, completed_markers, add_lines=True)

    if np.array_equal(full, without_completion):
        return False
    if np.array_equal(full, completion_only):
        return False

    color_counts = {color: sum(int(box["color"]) == color for box in boxes) for color in FRAME_COLORS}
    return max(color_counts.values()) <= 3 and min(v for v in color_counts.values() if v > 0) >= 2


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None and _is_semantic_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid same-color frame sharing task")
