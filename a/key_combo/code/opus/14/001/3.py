import itertools
from collections import deque

import numpy as np


BG = 0
TEMPLATE_COLOR = 5
FRAME_COLOR_POOL = [1, 2, 4, 6, 7]

GEN_GRID_MIN = 21
GEN_GRID_MAX = 24
GEN_ATTEMPTS = 400
GEN_TEMPLATE_ATTEMPTS = 80
GEN_FRAME_ATTEMPTS = 80
GEN_CORNERS = ("tl", "tr", "bl", "br")


def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rect_perimeter_mask(shape, box):
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _draw_rect_perimeter(grid: np.ndarray, box, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _visible_frame_bboxes(grid: np.ndarray):
    colors = [int(v) for v in np.unique(grid) if int(v) not in (BG, TEMPLATE_COLOR)]
    bboxes = {}
    for color in colors:
        box = _bbox(grid == color)
        assert box is not None, f"missing visible pixels for frame color {color}"
        bboxes[color] = box
    return colors, bboxes


def _crossing_edges(grid: np.ndarray, colors, bboxes):
    perimeters = {color: _rect_perimeter_mask(grid.shape, bboxes[color]) for color in colors}
    edges = set()
    for a, b in itertools.combinations(colors, 2):
        cross = perimeters[a] & perimeters[b]
        rows, cols = np.where(cross)
        for r, c in zip(rows.tolist(), cols.tolist()):
            visible = int(grid[r, c])
            if visible == a:
                edges.add((b, a))
            elif visible == b:
                edges.add((a, b))
    return edges


def _unique_hamiltonian_path(colors, edges):
    color_list = list(colors)
    valid = []
    for perm in itertools.permutations(color_list):
        if all((perm[i], perm[i + 1]) in edges for i in range(len(perm) - 1)):
            valid.append(perm)
            if len(valid) > 1:
                break
    assert len(valid) == 1, f"expected exactly one Hamiltonian path, found {len(valid)}"
    return list(valid[0])


def _stack_template(template: np.ndarray, order):
    h, w = template.shape
    out_h = h * len(order) + (len(order) - 1)
    out = np.zeros((out_h, w), dtype=int)
    for i, color in enumerate(order):
        r0 = i * (h + 1)
        tile = np.where(template == TEMPLATE_COLOR, int(color), BG)
        out[r0 : r0 + h, :] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the occlusion order of overlapping rectangular frames and use that sequence to recolor repeated template copies.

    Input structure:
    - Black background contains 3-5 one-cell-thick axis-aligned rectangular frames in distinct colors.
    - A separate gray template component sits in a corner region away from the overlapping frame cluster.
    - At frame crossings, the visible color indicates which rectangle is in front.

    Rule:
    - Reconstruct each full frame from the bbox of its visible pixels.
    - For every pair of reconstructed frame perimeters, inspect shared crossing cells and add A -> B when color B is visible there.
    - Find the unique Hamiltonian path from the bottommost frame to the topmost frame.
    - Crop the gray template to its bbox, repeat it vertically once per frame, insert one black row between copies, and recolor copy i with the i-th color in the decoded order.

    Key insight:
    - The same hidden occlusion sequence simultaneously encodes both how many template copies to make and which color each copy receives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    template_box = _bbox(g == TEMPLATE_COLOR)
    assert template_box is not None, "expected one gray template"
    template = _crop_bbox(g, template_box)

    colors, bboxes = _visible_frame_bboxes(g)
    assert 3 <= len(colors) <= 5
    edges = _crossing_edges(g, colors, bboxes)
    order = _unique_hamiltonian_path(colors, edges)
    return _stack_template(template, order)


def _connected(coords) -> bool:
    cells = {tuple(map(int, p)) for p in coords}
    start = next(iter(cells))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(cells)


def _make_template(rng: np.random.Generator) -> np.ndarray:
    for _ in range(GEN_TEMPLATE_ATTEMPTS):
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        n_cells = int(rng.integers(3, min(6, h * w) + 1))
        start = (int(rng.integers(h)), int(rng.integers(w)))
        cells = {start}
        while len(cells) < n_cells:
            r, c = tuple(list(cells)[int(rng.integers(len(cells)))])
            candidates = []
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    candidates.append((nr, nc))
            if not candidates:
                break
            cells.add(candidates[int(rng.integers(len(candidates)))])
        if len(cells) != n_cells:
            continue
        coords = np.array(sorted(cells), dtype=int)
        if not _connected(coords):
            continue
        # Build tightly cropped template and reject rectangles/lines.
        temp = np.zeros((h, w), dtype=int)
        for r, c in cells:
            temp[r, c] = TEMPLATE_COLOR
        crop = _crop_bbox(temp, _bbox(temp == TEMPLATE_COLOR))
        if crop.shape[0] < 2 or crop.shape[1] < 2:
            continue
        if int((crop == TEMPLATE_COLOR).sum()) == crop.size:
            continue
        return crop
    raise RuntimeError("failed to build template")


def _corner_gap_ok(template_box, frame_box) -> bool:
    tr0, tc0, tr1, tc1 = template_box
    fr0, fc0, fr1, fc1 = frame_box
    if tr1 < fr0:
        row_gap = fr0 - tr1 - 1
    elif fr1 < tr0:
        row_gap = tr0 - fr1 - 1
    else:
        row_gap = -1
    if tc1 < fc0:
        col_gap = fc0 - tc1 - 1
    elif fc1 < tc0:
        col_gap = tc0 - fc1 - 1
    else:
        col_gap = -1
    return row_gap >= 2 and col_gap >= 2


def _frame_bboxes(rng: np.random.Generator, h: int, w: int, n: int, corner: str):
    for _ in range(GEN_FRAME_ATTEMPTS):
        if corner == "tl":
            base_r = int(rng.integers(6, h - 13))
            base_c = int(rng.integers(6, w - 13))
        elif corner == "tr":
            base_r = int(rng.integers(6, h - 13))
            base_c = int(rng.integers(1, w - 18))
        elif corner == "bl":
            base_r = int(rng.integers(1, h - 18))
            base_c = int(rng.integers(6, w - 13))
        else:
            base_r = int(rng.integers(1, h - 18))
            base_c = int(rng.integers(1, w - 18))

        top_offsets = sorted(rng.choice(np.arange(5), size=n, replace=False).tolist())
        left_offsets = sorted(rng.choice(np.arange(5), size=n, replace=False).tolist())

        dims = []
        bboxes = []
        for i in range(n):
            for _inner in range(40):
                height = int(rng.integers(max(5, n), 11))
                width = int(rng.integers(max(5, n), 11))
                if (height, width) in dims:
                    continue
                top = base_r + top_offsets[i]
                left = base_c + left_offsets[i]
                bottom = top + height - 1
                right = left + width - 1
                if bottom >= h or right >= w:
                    continue
                dims.append((height, width))
                bboxes.append((top, left, bottom, right))
                break
            else:
                break
        if len(bboxes) != n:
            continue

        # Explicitly validate the complete-overlap condition.
        perims = [_rect_perimeter_mask((h, w), box) for box in bboxes]
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                if not bool((perims[i] & perims[j]).any()):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return bboxes
    return None


def _place_template(grid: np.ndarray, template: np.ndarray, corner: str):
    h, w = grid.shape
    th, tw = template.shape
    if corner == "tl":
        r0 = 1
        c0 = 1
    elif corner == "tr":
        r0 = 1
        c0 = w - tw - 2
    elif corner == "bl":
        r0 = h - th - 2
        c0 = 1
    else:
        r0 = h - th - 2
        c0 = w - tw - 2
    patch = np.where(template == TEMPLATE_COLOR, TEMPLATE_COLOR, BG)
    grid[r0 : r0 + th, c0 : c0 + tw] = patch
    return (r0, c0, r0 + th - 1, c0 + tw - 1)


def _validate_instance(grid: np.ndarray, order, frame_bboxes, template_box) -> bool:
    colors, visible_bboxes = _visible_frame_bboxes(grid)
    if set(colors) != set(order):
        return False
    expected = {color: frame_bboxes[i] for i, color in enumerate(order)}
    if visible_bboxes != expected:
        return False
    edges = _crossing_edges(grid, colors, visible_bboxes)
    try:
        solved_order = _unique_hamiltonian_path(colors, edges)
    except AssertionError:
        return False
    if solved_order != list(order):
        return False
    frame_union = (
        min(box[0] for box in frame_bboxes),
        min(box[1] for box in frame_bboxes),
        max(box[2] for box in frame_bboxes),
        max(box[3] for box in frame_bboxes),
    )
    if not _corner_gap_ok(template_box, frame_union):
        return False
    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        h = int(rng.integers(GEN_GRID_MIN, GEN_GRID_MAX + 1))
        w = int(rng.integers(GEN_GRID_MIN, GEN_GRID_MAX + 1))
        n = int(rng.integers(3, 6))
        corner = GEN_CORNERS[int(rng.integers(len(GEN_CORNERS)))]
        order = rng.choice(np.array(FRAME_COLOR_POOL, dtype=int), size=n, replace=False).tolist()
        frame_bboxes = _frame_bboxes(rng, h, w, n, corner)
        if frame_bboxes is None:
            continue

        grid = np.zeros((h, w), dtype=int)
        for color, box in zip(order, frame_bboxes):
            _draw_rect_perimeter(grid, box, int(color))

        template = _make_template(rng)
        template_box = _place_template(grid, template, corner)
        if not _validate_instance(grid, order, frame_bboxes, template_box):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
