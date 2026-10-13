from collections import defaultdict

import numpy as np

from arc_utils import (
    Color,
    colors_present,
    get_bg,
    in_bounds,
    label_mask,
    neighbors4,
    neighbors8,
    paste,
)


def _edge_component_ids(labels: np.ndarray, n_labels: int) -> set[int]:
    height, width = labels.shape
    edge_ids: set[int] = set()
    for k in range(1, n_labels + 1):
        coords = np.argwhere(labels == k)
        if np.any(
            (coords[:, 0] == 0)
            | (coords[:, 1] == 0)
            | (coords[:, 0] == height - 1)
            | (coords[:, 1] == width - 1)
        ):
            edge_ids.add(k)
    return edge_ids


def _hole_components_by_object(
    g: np.ndarray,
    fg_labels: np.ndarray,
    bg_labels: np.ndarray,
    n_bg: int,
    edge_bg_ids: set[int],
) -> dict[int, list[np.ndarray]]:
    holes_by_object: dict[int, list[np.ndarray]] = defaultdict(list)
    for bg_id in range(1, n_bg + 1):
        if bg_id in edge_bg_ids:
            continue

        coords = np.argwhere(bg_labels == bg_id)
        neighbor_objects: set[int] = set()
        for r, c in coords:
            for nr, nc in neighbors4(int(r), int(c)):
                if in_bounds(g.shape, nr, nc) and fg_labels[nr, nc] != 0:
                    neighbor_objects.add(int(fg_labels[nr, nc]))

        assert len(neighbor_objects) == 1, "each hole must be enclosed by one blue object"
        obj_id = next(iter(neighbor_objects))
        holes_by_object[obj_id].append(coords)

    return holes_by_object


def _touches_mask8(mask: np.ndarray, r: int, c: int) -> bool:
    return any(in_bounds(mask.shape, nr, nc) and mask[nr, nc] for nr, nc in neighbors8(r, c))


def _add_top_left_diagonal_corner_caps(
    red: np.ndarray,
    solid: np.ndarray,
    bg_mask: np.ndarray,
    bg_labels: np.ndarray,
    edge_bg_ids: set[int],
) -> None:
    height, width = solid.shape
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if solid[r, c]:
                continue
            is_one_cell_top_left_gap = (
                solid[r, c + 1]
                and solid[r + 1, c]
                and solid[r + 1, c + 1]
                and not solid[r + 1, c - 1]
                and not solid[r - 1, c + 1]
            )
            if not is_one_cell_top_left_gap:
                continue

            rr, cc = r - 1, c - 1
            if bg_mask[rr, cc] and bg_labels[rr, cc] in edge_bg_ids:
                red[rr, cc] = True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Outline every closed blue rim by painting its outside boundary red and its hole-side boundary green.

    Input structure:
    - The canvas background is black and all foreground pixels are blue.
    - Blue pixels form one or more 4-connected objects; some objects enclose black holes.
    - A hole is a 4-connected black component that does not touch the grid edge and is adjacent to exactly one blue object.

    Rule:
    - Leave blue pixels unchanged.
    - For each hole-enclosing blue object, color edge-connected black cells red when they touch that object in the 8-neighborhood.
    - Color black hole cells green when they touch their enclosing blue object in the 8-neighborhood.
    - Complete the red outline at one-cell top-left diagonal closures of the filled rim.

    Key insight:
    - The task is not filling whole regions; it labels the two sides of each closed rim, while open blue objects have no enclosed hole and stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    assert colors_present(g, exclude={bg}) == [Color.BLUE], "expected only blue foreground"

    fg_mask = g == Color.BLUE
    bg_mask = g == bg
    fg_labels, n_fg = label_mask(fg_mask, connectivity=4)
    bg_labels, n_bg = label_mask(bg_mask, connectivity=4)
    assert n_fg >= 1, "expected at least one blue object"

    edge_bg_ids = _edge_component_ids(bg_labels, n_bg)
    holes_by_object = _hole_components_by_object(g, fg_labels, bg_labels, n_bg, edge_bg_ids)

    out = g.copy()
    for obj_id, hole_components in holes_by_object.items():
        obj_mask = fg_labels == obj_id
        hole_mask = np.zeros(g.shape, dtype=bool)
        for coords in hole_components:
            hole_mask[coords[:, 0], coords[:, 1]] = True

        for r, c in np.argwhere(hole_mask):
            if _touches_mask8(obj_mask, int(r), int(c)):
                out[r, c] = Color.GREEN

        red = np.zeros(g.shape, dtype=bool)
        for r, c in np.argwhere(bg_mask):
            if bg_labels[r, c] in edge_bg_ids and _touches_mask8(obj_mask, int(r), int(c)):
                red[r, c] = True

        solid = obj_mask | hole_mask
        _add_top_left_diagonal_corner_caps(red, solid, bg_mask, bg_labels, edge_bg_ids)
        out[red] = Color.RED

    return out

# --------------- generator (o5x) ---------------

from arc_utils import bbox_coords

# Generator CFG: tune these constants to alter the sampling distribution.
# Canvas bounds and object counts.
_GEN_GRID_MIN = 8
_GEN_GRID_MAX = 30
_GEN_MIN_TOTAL_OBJECTS = 1
_GEN_MAX_TOTAL_OBJECTS = 7
_GEN_MAX_RIM_OBJECTS = 4
_GEN_MAX_DISTRACTOR_OBJECTS = 4

# Local object size bounds.
_GEN_PATCH_MIN_H = 3
_GEN_PATCH_MAX_H = 8
_GEN_PATCH_MIN_W = 3
_GEN_PATCH_MAX_W = 9

# Rim construction.
_GEN_RIM_HOLE_MIN_SIZE = 1
_GEN_RIM_HOLE_MAX_SIZE = 28
_GEN_RIM_NOTCH_MIN = 0
_GEN_RIM_NOTCH_MIN_CAP = 1
_GEN_RIM_NOTCH_SCALE = 3

# Distractor construction.
_GEN_DISTRACTOR_MIN_SIZE = 1
_GEN_DISTRACTOR_MAX_SIZE = 29
_GEN_BROKEN_BOX_MIN_SIZE = 6
_GEN_BROKEN_BOX_PROB = 0.45

# Placement and validation.
_GEN_RIM_SEPARATION_RADIUS = 2
_GEN_DISTRACTOR_SEPARATION_RADIUS = 1
_GEN_MIN_BLUE_PIXELS = 1
_GEN_MAX_BLUE_DENSITY_NUMERATOR = 1
_GEN_MAX_BLUE_DENSITY_DENOMINATOR = 2
_GEN_RETRY_LIMIT = 300

_GEN_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_GEN_COUNT_PAIRS = [
    (r, d)
    for r in range(_GEN_MAX_RIM_OBJECTS + 1)
    for d in range(_GEN_MAX_DISTRACTOR_OBJECTS + 1)
    if _GEN_MIN_TOTAL_OBJECTS <= r + d <= _GEN_MAX_TOTAL_OBJECTS
]


def _gen_randint(lo, hi):
    return int(np.random.randint(int(lo), int(hi) + 1))


def _gen_choice(items):
    return items[int(np.random.randint(0, len(items)))]


def _gen_shuffled(items):
    order = np.random.permutation(len(items))
    return [items[int(i)] for i in order]


def _gen_patch_sizes(max_h, max_w):
    for height in range(_GEN_PATCH_MIN_H, min(_GEN_PATCH_MAX_H, max_h) + 1):
        for width in range(_GEN_PATCH_MIN_W, min(_GEN_PATCH_MAX_W, max_w) + 1):
            yield height, width


def _gen_object_from_patch(patch, is_rim, block_extra=None):
    blue = patch == Color.BLUE
    block = blue.copy() if block_extra is None else blue | block_extra
    return {"patch": patch, "blue": blue, "block": block, "is_rim": is_rim}


def _gen_edge_ids(labels, n_labels):
    if n_labels == 0:
        return set()
    vals = set(map(int, labels[0, :]))
    vals.update(map(int, labels[-1, :]))
    vals.update(map(int, labels[:, 0]))
    vals.update(map(int, labels[:, -1]))
    vals.discard(0)
    return vals


def _gen_nonedge_bg_components(grid):
    bg_labels, n_bg = label_mask(grid == Color.BLACK, connectivity=4)
    edge_ids = _gen_edge_ids(bg_labels, n_bg)
    return [np.argwhere(bg_labels == k) for k in range(1, n_bg + 1) if k not in edge_ids]


def _gen_nonedge_bg_mask(grid):
    out = np.zeros(grid.shape, dtype=bool)
    for coords in _gen_nonedge_bg_components(grid):
        out[coords[:, 0], coords[:, 1]] = True
    return out


def _gen_hole_counts_by_blue_object(grid):
    fg_labels, n_fg = label_mask(grid == Color.BLUE, connectivity=4)
    if n_fg == 0:
        return None
    bg_labels, n_bg = label_mask(grid == Color.BLACK, connectivity=4)
    edge_ids = _gen_edge_ids(bg_labels, n_bg)
    counts = [0 for _ in range(n_fg)]
    height, width = grid.shape
    for bg_id in range(1, n_bg + 1):
        if bg_id in edge_ids:
            continue
        owners = set()
        for r, c in np.argwhere(bg_labels == bg_id):
            rr = int(r)
            cc = int(c)
            for dr, dc in _GEN_DIRS4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < height and 0 <= nc < width and fg_labels[nr, nc]:
                    owners.add(int(fg_labels[nr, nc]))
        if len(owners) != 1:
            return None
        counts[next(iter(owners)) - 1] += 1
    return counts


def _gen_patch_is_connected(patch):
    _, n_blue = label_mask(patch == Color.BLUE, connectivity=4)
    return n_blue == 1


def _gen_rim_patch_ok(patch):
    blue = patch == Color.BLUE
    if not blue.any():
        return False
    if not (blue[0, :].any() and blue[-1, :].any() and blue[:, 0].any() and blue[:, -1].any()):
        return False
    return _gen_patch_is_connected(patch) and len(_gen_nonedge_bg_components(patch)) == 1


def _gen_distractor_patch_ok(patch):
    return (patch == Color.BLUE).any() and _gen_patch_is_connected(patch) and not _gen_nonedge_bg_components(patch)


def _gen_touches_hole4(hole, r, c):
    for dr, dc in _GEN_DIRS4:
        if (int(r) + dr, int(c) + dc) in hole:
            return True
    return False


def _gen_connected_hole(height, width, size):
    interior = {(r, c) for r in range(1, height - 1) for c in range(1, width - 1)}
    hole = {_gen_choice(list(interior))}
    while len(hole) < size:
        frontier = set()
        for r, c in hole:
            for dr, dc in _GEN_DIRS4:
                nb = (r + dr, c + dc)
                if nb in interior and nb not in hole:
                    frontier.add(nb)
        if not frontier:
            return None
        hole.add(_gen_choice(list(frontier)))
    return hole


def _gen_choose_rim_size(hole_size, max_h, max_w):
    feasible = []
    for height, width in _gen_patch_sizes(max_h, max_w):
        capacity = (height - 2) * (width - 2)
        if capacity >= hole_size:
            feasible.append((height, width, capacity))
    if not feasible:
        return None

    weights = np.array([1.0 / (1 + capacity - hole_size) for _, _, capacity in feasible], dtype=float)
    weights = weights / weights.sum()
    return feasible[int(np.random.choice(len(feasible), p=weights))][:2]


def _gen_carve_rim_notches(patch, hole):
    height, width = patch.shape
    edge_cells = []
    for r in range(height):
        for c in range(width):
            if r in (0, height - 1) or c in (0, width - 1):
                if patch[r, c] == Color.BLUE and not _gen_touches_hole4(hole, r, c):
                    edge_cells.append((r, c))

    notch_cap = min(len(edge_cells), max(_GEN_RIM_NOTCH_MIN_CAP, (height + width) // _GEN_RIM_NOTCH_SCALE))
    notch_budget = _gen_randint(_GEN_RIM_NOTCH_MIN, notch_cap)
    for r, c in _gen_shuffled(edge_cells):
        if notch_budget <= 0:
            break
        trial = patch.copy()
        trial[r, c] = Color.BLACK
        if _gen_rim_patch_ok(trial):
            patch = trial
            notch_budget -= 1
    return patch


def _gen_make_rim_object(max_h, max_w):
    hole_size = _gen_randint(_GEN_RIM_HOLE_MIN_SIZE, _GEN_RIM_HOLE_MAX_SIZE)
    size = _gen_choose_rim_size(hole_size, max_h, max_w)
    if size is None:
        return None
    height, width = size
    hole = _gen_connected_hole(height, width, hole_size)
    if hole is None:
        return None

    patch = np.full((height, width), Color.BLUE, dtype=int)
    for r, c in hole:
        patch[r, c] = Color.BLACK

    patch = _gen_carve_rim_notches(patch, hole)
    if not _gen_rim_patch_ok(patch):
        return None
    hole_mask = _gen_nonedge_bg_mask(patch)
    return _gen_object_from_patch(patch, True, hole_mask)


def _gen_bbox_dims(coords):
    r0, c0, r1, c1 = bbox_coords(np.array(list(coords), dtype=int))
    return r1 - r0 + 1, c1 - c0 + 1


def _gen_normalized_patch(coords):
    coords_arr = np.array(list(coords), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords_arr)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    shifted = coords_arr - np.array([r0, c0])
    patch[shifted[:, 0], shifted[:, 1]] = Color.BLUE
    return patch


def _gen_make_tree_patch(size, max_h, max_w):
    coords = {(0, 0)}
    while len(coords) < size:
        frontier = set()
        for r, c in coords:
            for dr, dc in _GEN_DIRS4:
                nb = (r + dr, c + dc)
                if nb in coords:
                    continue
                neighbor_count = 0
                for nr, nc in ((nb[0] + ddr, nb[1] + ddc) for ddr, ddc in _GEN_DIRS4):
                    if (nr, nc) in coords:
                        neighbor_count += 1
                if neighbor_count == 1 and _gen_bbox_dims(coords | {nb}) <= (max_h, max_w):
                    frontier.add(nb)
        if not frontier:
            return None
        coords.add(_gen_choice(list(frontier)))
    patch = _gen_normalized_patch(coords)
    return patch if _gen_distractor_patch_ok(patch) else None


def _gen_make_broken_box_patch(size, max_h, max_w):
    candidates = []
    for height, width in _gen_patch_sizes(max_h, max_w):
        perimeter = 2 * height + 2 * width - 4
        gap = perimeter - size
        if 1 <= gap <= width - 2:
            candidates.append((height, width, "top", gap))
            candidates.append((height, width, "bottom", gap))
        if 1 <= gap <= height - 2:
            candidates.append((height, width, "left", gap))
            candidates.append((height, width, "right", gap))
    if not candidates:
        return None

    height, width, side, gap = _gen_choice(candidates)
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = Color.BLUE
    patch[-1, :] = Color.BLUE
    patch[:, 0] = Color.BLUE
    patch[:, -1] = Color.BLUE
    if side in ("top", "bottom"):
        start = _gen_randint(1, width - gap - 1)
        row = 0 if side == "top" else height - 1
        patch[row, start : start + gap] = Color.BLACK
    else:
        start = _gen_randint(1, height - gap - 1)
        col = 0 if side == "left" else width - 1
        patch[start : start + gap, col] = Color.BLACK

    return patch if int((patch == Color.BLUE).sum()) == size and _gen_distractor_patch_ok(patch) else None


def _gen_make_distractor_object(max_h, max_w):
    size = _gen_randint(_GEN_DISTRACTOR_MIN_SIZE, _GEN_DISTRACTOR_MAX_SIZE)
    patch = None
    if size >= _GEN_BROKEN_BOX_MIN_SIZE and np.random.random() < _GEN_BROKEN_BOX_PROB:
        patch = _gen_make_broken_box_patch(size, max_h, max_w)
    if patch is None:
        patch = _gen_make_tree_patch(size, max_h, max_w)
    if patch is None:
        return None
    return _gen_object_from_patch(patch, False)


def _gen_cheb_dilate(mask, radius):
    out = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    for dr in range(-radius, radius + 1):
        src_r0 = max(0, -dr)
        src_r1 = min(height, height - dr)
        dst_r0 = max(0, dr)
        dst_r1 = min(height, height + dr)
        for dc in range(-radius, radius + 1):
            src_c0 = max(0, -dc)
            src_c1 = min(width, width - dc)
            dst_c0 = max(0, dc)
            dst_c1 = min(width, width + dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _gen_shift(local_mask, shape, r0, c0):
    out = np.zeros(shape, dtype=bool)
    height, width = local_mask.shape
    out[r0 : r0 + height, c0 : c0 + width] = local_mask
    return out


def _gen_valid_placement(canvas, obj, r0, c0, placed):
    patch_h, patch_w = obj["patch"].shape
    if np.any(canvas[r0 : r0 + patch_h, c0 : c0 + patch_w][obj["block"]] == Color.BLUE):
        return None

    cand_blue = _gen_shift(obj["blue"], canvas.shape, r0, c0)
    cand_block = _gen_shift(obj["block"], canvas.shape, r0, c0)
    for other in placed:
        if np.any(cand_blue & other["block"]) or np.any(cand_block & other["blue"]):
            return None
        radius = _GEN_RIM_SEPARATION_RADIUS if obj["is_rim"] or other["is_rim"] else _GEN_DISTRACTOR_SEPARATION_RADIUS
        if np.any(cand_blue & _gen_cheb_dilate(other["blue"], radius)):
            return None
    return cand_blue, cand_block


def _gen_place_objects(canvas, objects):
    placed = []
    order = list(range(len(objects)))
    order.sort(key=lambda i: (int(objects[i]["block"].sum()), float(np.random.random())), reverse=True)
    for idx in order:
        obj = objects[idx]
        patch_h, patch_w = obj["patch"].shape
        positions = [(r, c) for r in range(canvas.shape[0] - patch_h + 1) for c in range(canvas.shape[1] - patch_w + 1)]
        valid = []
        for r, c in _gen_shuffled(positions):
            shifted = _gen_valid_placement(canvas, obj, r, c, placed)
            if shifted is not None:
                valid.append((r, c, shifted[0], shifted[1]))
        if not valid:
            return None
        r, c, blue_mask, block_mask = _gen_choice(valid)
        paste(canvas, obj["patch"], r, c, transparent=Color.BLACK, overwrite=False)
        placed.append({"blue": blue_mask, "block": block_mask, "is_rim": obj["is_rim"]})
    return canvas


def _gen_scene_ok(grid, n_rim, n_dist):
    blue_count = int((grid == Color.BLUE).sum())
    if (
        blue_count < _GEN_MIN_BLUE_PIXELS
        or blue_count * _GEN_MAX_BLUE_DENSITY_DENOMINATOR >= grid.size * _GEN_MAX_BLUE_DENSITY_NUMERATOR
    ):
        return False
    counts = _gen_hole_counts_by_blue_object(grid)
    if counts is None or len(counts) != n_rim + n_dist:
        return False
    if any(count not in (0, 1) for count in counts):
        return False
    if sum(counts) != n_rim:
        return False
    try:
        main(grid)
    except Exception:
        return False
    return True


def _gen_build_objects(n_rim, n_dist, height, width):
    objects = []
    for _ in range(n_rim):
        obj = _gen_make_rim_object(height, width)
        if obj is None:
            return None
        objects.append(obj)
    for _ in range(n_dist):
        obj = _gen_make_distractor_object(height, width)
        if obj is None:
            return None
        objects.append(obj)
    return objects


def _gen_attempt():
    height = _gen_randint(_GEN_GRID_MIN, _GEN_GRID_MAX)
    width = _gen_randint(_GEN_GRID_MIN, _GEN_GRID_MAX)
    n_rim, n_dist = _gen_choice(_GEN_COUNT_PAIRS)

    objects = _gen_build_objects(n_rim, n_dist, height, width)
    if objects is None:
        return None
    canvas = np.zeros((height, width), dtype=int)
    if _gen_place_objects(canvas, objects) is None:
        return None
    if not _gen_scene_ok(canvas, n_rim, n_dist):
        return None
    return canvas


def generate_input():
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d931c21c input")
