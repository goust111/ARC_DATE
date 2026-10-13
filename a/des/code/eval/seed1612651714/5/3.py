import numpy as np


SIDES = ("top", "right", "bottom", "left")
VECTORS = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}

BG_COLOR = 0
GRID_MIN = 18
GRID_MAX = 30
LAYER_RANGE = (2, 4)
SPACING_RANGE = (3, 5)
INNER_SIZE_MIN = 5
INNER_SIZE_MAX = 8
MOTIF_BBOX_MAX = 4
MOTIF_AREA_MIN = 2
MOTIF_AREA_MAX = 7
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 60


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _frame_info(grid: np.ndarray, color: int):
    mask = grid == color
    if not np.any(mask):
        return None
    comps = _components(mask)
    if len(comps) != 1:
        return None
    coords = comps[0]
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    crop = mask[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(crop & ~border):
        return None

    missing = np.argwhere(border & ~crop)
    if missing.shape[0] != 1:
        return None
    mr, mc = map(int, missing[0])
    if (mr in (0, h - 1)) and (mc in (0, w - 1)):
        return None
    if mr == 0:
        side = "top"
    elif mr == h - 1:
        side = "bottom"
    elif mc == 0:
        side = "left"
    elif mc == w - 1:
        side = "right"
    else:
        return None

    expected = int(border.sum()) - 1
    if int(crop.sum()) != expected:
        return None

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "gap_side": side,
        "gap_cell": (r0 + mr, c0 + mc),
    }


def _strictly_contains(outer_bbox, inner_bbox) -> bool:
    or0, oc0, or1, oc1 = outer_bbox
    ir0, ic0, ir1, ic1 = inner_bbox
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        info = _frame_info(grid, color)
        if info is not None:
            frames.append(info)

    assert 2 <= len(frames) <= 4, "expected 2-4 nested rectangular frames"
    frames.sort(key=lambda item: (item["bbox"][2] - item["bbox"][0] + 1) * (item["bbox"][3] - item["bbox"][1] + 1))
    for i in range(len(frames) - 1):
        assert _strictly_contains(frames[i + 1]["bbox"], frames[i]["bbox"]), "frames must form a strict nesting chain"
    return frames


def _motif(grid: np.ndarray, bg: int, frame_colors: set[int], inner_bbox):
    r0, c0, r1, c1 = inner_bbox
    non_frame = (grid != bg) & ~np.isin(grid, list(frame_colors))
    coords = np.argwhere(non_frame)
    assert coords.size > 0, "expected an inner motif"
    inside = (
        (coords[:, 0] > r0)
        & (coords[:, 0] < r1)
        & (coords[:, 1] > c0)
        & (coords[:, 1] < c1)
    )
    assert np.all(inside), "all non-frame pixels must lie inside the innermost frame"
    assert len(_components(non_frame)) == 1, "motif must be a single 4-connected region"
    values = np.array([int(grid[r, c]) for r, c in coords], dtype=int)
    return coords.astype(int), values


def _shift_distance(inner_bbox, outer_bbox, side: str) -> int:
    ir0, ic0, ir1, ic1 = inner_bbox
    or0, oc0, or1, oc1 = outer_bbox
    if side == "top":
        return ir0 - or0
    if side == "bottom":
        return or1 - ir1
    if side == "left":
        return ic0 - oc0
    if side == "right":
        return oc1 - ic1
    raise ValueError(side)


def _ring_mask(shape, inner_bbox, outer_bbox) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    or0, oc0, or1, oc1 = outer_bbox
    ir0, ic0, ir1, ic1 = inner_bbox
    mask[or0 + 1 : or1, oc0 + 1 : oc1] = True
    mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
    return mask


def _copy_sequence(motif_coords: np.ndarray, frames: list[dict], grid_shape) -> list[np.ndarray]:
    current = motif_coords.copy()
    copies = []
    h, w = grid_shape
    for i in range(len(frames) - 1):
        inner = frames[i]["bbox"]
        outer = frames[i + 1]["bbox"]
        side = frames[i]["gap_side"]
        dist = _shift_distance(inner, outer, side)
        dr, dc = VECTORS[side]
        shifted = current + np.array([[dr * dist, dc * dist]], dtype=int)
        assert np.all((0 <= shifted[:, 0]) & (shifted[:, 0] < h) & (0 <= shifted[:, 1]) & (shifted[:, 1] < w)), "copy must stay on-grid"

        ring = _ring_mask(grid_shape, inner, outer)
        ok = ring[shifted[:, 0], shifted[:, 1]]
        assert np.all(ok), "copy must land in the ring between two frames"
        copies.append(shifted)
        current = shifted
    return copies


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify nested rectangular frames, decode each gap as a translation direction,
    and copy the innermost motif outward one ring at a time.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _bg_color(g)
    frames = _detect_frames(g, bg)
    frame_colors = {frame["color"] for frame in frames}
    motif_coords, motif_values = _motif(g, bg, frame_colors, frames[0]["bbox"])

    out = g.copy()
    for shifted in _copy_sequence(motif_coords, frames, g.shape):
        assert np.all(out[shifted[:, 0], shifted[:, 1]] == bg), "target ring must start as background"
        out[shifted[:, 0], shifted[:, 1]] = motif_values
    return out


def _draw_frame(grid: np.ndarray, bbox, color: int, side: str, gap_index: int):
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if side == "top":
        grid[r0, c0 + gap_index] = BG_COLOR
    elif side == "bottom":
        grid[r1, c0 + gap_index] = BG_COLOR
    elif side == "left":
        grid[r0 + gap_index, c0] = BG_COLOR
    elif side == "right":
        grid[r0 + gap_index, c1] = BG_COLOR
    else:
        raise ValueError(side)


def _random_connected_mask(rng: np.random.Generator, h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    mask[start] = True
    frontier = {start}
    while int(mask.sum()) < area and frontier:
        cell = tuple(frontier.pop())
        neighbors = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                neighbors.append((nr, nc))
        rng.shuffle(neighbors)
        for nr, nc in neighbors:
            mask[nr, nc] = True
            frontier.add((nr, nc))
            if int(mask.sum()) >= area:
                break

    coords = np.argwhere(mask)
    if coords.size == 0:
        return mask
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _frame_bboxes_from_outer(outer_bbox, spacings: list[int]) -> list[tuple[int, int, int, int]]:
    boxes = [outer_bbox]
    current = outer_bbox
    for spacing in reversed(spacings):
        r0, c0, r1, c1 = current
        current = (r0 + spacing, c0 + spacing, r1 - spacing, c1 - spacing)
        boxes.append(current)
    boxes.reverse()
    return boxes


def _candidate_positions(shape_coords: np.ndarray, frames: list[dict], grid_shape) -> list[tuple[int, int]]:
    inner = frames[0]["bbox"]
    inner_h = inner[2] - inner[0] - 1
    inner_w = inner[3] - inner[1] - 1
    shape_h = int(shape_coords[:, 0].max()) + 1
    shape_w = int(shape_coords[:, 1].max()) + 1
    positions = []
    for top in range(inner[0] + 1, inner[0] + 1 + inner_h - shape_h + 1):
        for left in range(inner[1] + 1, inner[1] + 1 + inner_w - shape_w + 1):
            motif = shape_coords + np.array([[top, left]], dtype=int)
            try:
                shifted = _copy_sequence(motif, frames, grid_shape)
            except AssertionError:
                continue
            used = {tuple(cell) for cell in motif.tolist()}
            valid = True
            for copy_coords in shifted:
                for cell in copy_coords.tolist():
                    cell_t = tuple(cell)
                    if cell_t in used:
                        valid = False
                        break
                    used.add(cell_t)
                if not valid:
                    break
            if valid:
                positions.append((top, left))
    return positions


def _paint_motif(grid: np.ndarray, coords: np.ndarray, rng: np.random.Generator, colors: list[int]):
    palette_size = int(rng.integers(1, min(3, len(colors)) + 1))
    palette = rng.choice(colors, size=palette_size, replace=False)
    values = palette[rng.integers(0, palette_size, size=coords.shape[0])]
    for (r, c), value in zip(coords.tolist(), values.tolist()):
        grid[r, c] = int(value)


def _gen_attempt(rng: np.random.Generator):
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    layers = int(rng.integers(LAYER_RANGE[0], LAYER_RANGE[1] + 1))
    spacings = [int(rng.integers(SPACING_RANGE[0], SPACING_RANGE[1] + 1)) for _ in range(layers - 1)]
    total_margin = sum(spacings)

    max_inner_h = min(INNER_SIZE_MAX, h - 2 * total_margin - 2)
    max_inner_w = min(INNER_SIZE_MAX, w - 2 * total_margin - 2)
    if max_inner_h < INNER_SIZE_MIN or max_inner_w < INNER_SIZE_MIN:
        return None

    inner_h = int(rng.integers(INNER_SIZE_MIN, max_inner_h + 1))
    inner_w = int(rng.integers(INNER_SIZE_MIN, max_inner_w + 1))
    outer_h = inner_h + 2 * total_margin
    outer_w = inner_w + 2 * total_margin
    if outer_h > h - 2 or outer_w > w - 2:
        return None

    outer_r0 = int(rng.integers(1, h - outer_h))
    outer_c0 = int(rng.integers(1, w - outer_w))
    outer_bbox = (outer_r0, outer_c0, outer_r0 + outer_h - 1, outer_c0 + outer_w - 1)
    bboxes = _frame_bboxes_from_outer(outer_bbox, spacings)

    all_colors = list(range(1, 10))
    frame_colors = rng.choice(all_colors, size=layers, replace=False).tolist()
    motif_colors = [color for color in all_colors if color not in frame_colors]
    if not motif_colors:
        return None

    frames = []
    for bbox, color in zip(bboxes, frame_colors):
        r0, c0, r1, c1 = bbox
        side = SIDES[int(rng.integers(0, len(SIDES)))]
        if side in ("top", "bottom"):
            gap_index = int(rng.integers(1, c1 - c0))
        else:
            gap_index = int(rng.integers(1, r1 - r0))
        frames.append({"color": int(color), "bbox": bbox, "gap_side": side, "gap_index": gap_index})

    grid = np.full((h, w), BG_COLOR, dtype=int)
    for frame in frames:
        _draw_frame(grid, frame["bbox"], frame["color"], frame["gap_side"], frame["gap_index"])

    inner = frames[0]["bbox"]
    interior_h = inner[2] - inner[0] - 1
    interior_w = inner[3] - inner[1] - 1
    if interior_h < 1 or interior_w < 1:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        shape_h = int(rng.integers(1, min(MOTIF_BBOX_MAX, interior_h) + 1))
        shape_w = int(rng.integers(1, min(MOTIF_BBOX_MAX, interior_w) + 1))
        max_area = min(MOTIF_AREA_MAX, shape_h * shape_w)
        min_area = min(max_area, MOTIF_AREA_MIN)
        if max_area < min_area:
            continue
        area = int(rng.integers(min_area, max_area + 1))
        shape = _random_connected_mask(rng, shape_h, shape_w, area)
        if int(shape.sum()) < MOTIF_AREA_MIN:
            continue
        shape_coords = np.argwhere(shape)
        positions = _candidate_positions(shape_coords, frames, grid.shape)
        if not positions:
            continue

        top, left = positions[int(rng.integers(0, len(positions)))]
        motif_coords = shape_coords + np.array([[top, left]], dtype=int)
        _paint_motif(grid, motif_coords, rng, motif_colors)

        try:
            out = main(grid)
        except AssertionError:
            return None

        if np.count_nonzero(out != grid) != (layers - 1) * motif_coords.shape[0]:
            return None
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame translation puzzle")
