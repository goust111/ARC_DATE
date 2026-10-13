import numpy as np


BG = 0
RED = 2
ORANGE = 7
BRIDGE_COLORS = {RED, ORANGE}
OBJECT_COLORS = (1, 3, 4, 5, 6, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GENERATE_ATTEMPTS = 400
PAIR_ATTEMPTS = 120
SHAPE_ATTEMPTS = 80


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    stack = [tuple(map(int, coords[0]))]
    seen[stack[0]] = True
    count = 0
    while stack:
        r, c = stack.pop()
        count += 1
        for nr, nc in _neighbors4(r, c, height, width):
            if mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return count == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _boundary_cell(mask: np.ndarray, r: int, c: int) -> bool:
    if not mask[r, c]:
        return False
    height, width = mask.shape
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < height and 0 <= nc < width):
            return True
        if not mask[nr, nc]:
            return True
    return False


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _find_components(grid: np.ndarray, include):
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components = []
    for r in range(height):
        for c in range(width):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            if not include(color):
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, height, width):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _bridge_axis(bridge_coords: np.ndarray, touched_dirs: dict[int, set[tuple[int, int]]]) -> str:
    rows = np.unique(bridge_coords[:, 0])
    cols = np.unique(bridge_coords[:, 1])
    if len(rows) == 1 and len(bridge_coords) > 1:
        return "horizontal"
    if len(cols) == 1 and len(bridge_coords) > 1:
        return "vertical"
    dirs = {d for obj_dirs in touched_dirs.values() for d in obj_dirs}
    if (0, -1) in dirs and (0, 1) in dirs:
        return "horizontal"
    if (-1, 0) in dirs and (1, 0) in dirs:
        return "vertical"
    raise AssertionError("bridge must be a one-cell-thick straight line")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge each bridged pair of objects by overlaying or subtracting their aligned shapes.

    Input structure:
    - The background is black.
    - Non-bridge foreground objects are separated monochrome 4-connected shapes.
    - Some object pairs are connected by a one-pixel-wide straight horizontal or vertical bridge.
    - Bridges use only red or orange, and each bridge touches exactly two different objects.
    - Any two bridged objects have the same bounding-box size, and each object belongs to at most one bridge.

    Rule:
    - For every bridge, choose A as the touched object whose bounding box is earlier in row-major order.
    - Normalize A and B into their own equally sized bounding boxes by aligning the top-left corners.
    - A red bridge replaces A with the pixelwise union of A and B, using A's color.
    - An orange bridge replaces A with the pixelwise difference A minus B, using A's color.
    - Erase B and the bridge; objects without a bridge stay unchanged.

    Key insight:
    - The bridge color is an operation code, and the two bridged objects must be compared as aligned local masks rather than as distant shapes in the global grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _find_components(g, lambda color: color not in BRIDGE_COLORS and color != BG)
    object_ids = np.full(g.shape, -1, dtype=int)
    for idx, obj in enumerate(objects):
        coords = obj["coords"]
        object_ids[coords[:, 0], coords[:, 1]] = idx

    bridges = []
    used_objects = set()
    for bridge_color in (RED, ORANGE):
        bridge_components = _find_components(g, lambda color, bridge_color=bridge_color: color == bridge_color)
        for bridge in bridge_components:
            touched_dirs: dict[int, set[tuple[int, int]]] = {}
            for r, c in bridge["coords"]:
                for dr, dc in DIR4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                        obj_id = int(object_ids[nr, nc])
                        if obj_id >= 0:
                            touched_dirs.setdefault(obj_id, set()).add((dr, dc))
            assert len(touched_dirs) == 2, "each bridge must touch exactly two objects"
            axis = _bridge_axis(bridge["coords"], touched_dirs)
            obj_pair = sorted(
                touched_dirs,
                key=lambda idx: (objects[idx]["bbox"][0], objects[idx]["bbox"][1]),
            )
            a_id, b_id = obj_pair
            assert a_id not in used_objects and b_id not in used_objects, "objects can belong to only one bridge"
            used_objects.add(a_id)
            used_objects.add(b_id)
            bridges.append({"color": bridge_color, "coords": bridge["coords"], "axis": axis, "a": a_id, "b": b_id})

    out = g.copy()
    for bridge in bridges:
        obj_a = objects[bridge["a"]]
        obj_b = objects[bridge["b"]]
        ar0, ac0, ar1, ac1 = obj_a["bbox"]
        br0, bc0, br1, bc1 = obj_b["bbox"]
        assert (ar1 - ar0, ac1 - ac0) == (br1 - br0, bc1 - bc0), "paired objects must share bbox size"

        mask_a = _component_mask(obj_a)
        mask_b = _component_mask(obj_b)
        if bridge["color"] == RED:
            result_mask = mask_a | mask_b
        else:
            result_mask = mask_a & ~mask_b

        out[obj_a["coords"][:, 0], obj_a["coords"][:, 1]] = BG
        out[obj_b["coords"][:, 0], obj_b["coords"][:, 1]] = BG
        out[bridge["coords"][:, 0], bridge["coords"][:, 1]] = BG

        target = out[ar0 : ar1 + 1, ac0 : ac1 + 1]
        target[result_mask] = int(obj_a["color"])

    return out


def _random_shape(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    cell_count = height * width
    min_area = max(height + width - 2, int(np.ceil(cell_count * 0.45)))
    max_area = max(min_area + 1, cell_count - 1)
    for _ in range(SHAPE_ATTEMPTS):
        mask = np.ones((height, width), dtype=bool)
        anchors = {
            (0, int(rng.integers(width))),
            (height - 1, int(rng.integers(width))),
            (int(rng.integers(height)), 0),
            (int(rng.integers(height)), width - 1),
        }
        target_area = int(rng.integers(min_area, max_area + 1))
        candidates = [(r, c) for r in range(height) for c in range(width) if (r, c) not in anchors]
        rng.shuffle(candidates)
        for r, c in candidates:
            if int(mask.sum()) <= target_area:
                break
            if not _boundary_cell(mask, r, c):
                continue
            mask[r, c] = False
            if not _touches_all_sides(mask) or not _mask_connected(mask):
                mask[r, c] = True
        if int(mask.sum()) == target_area and _touches_all_sides(mask) and _mask_connected(mask) and not np.all(mask):
            return mask
    raise RuntimeError("failed to generate a connected full-bbox shape")


def _pair_condition(mask_a: np.ndarray, mask_b: np.ndarray, bridge_color: int) -> bool:
    if bridge_color == RED:
        return bool(np.any(mask_b & ~mask_a))
    overlap = mask_a & mask_b
    remaining = mask_a & ~mask_b
    return bool(np.any(overlap) and np.any(remaining))


def _make_pair_unit(bridge_color: int, color_a: int, color_b: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(PAIR_ATTEMPTS):
        height = int(rng.integers(3, 7))
        width = int(rng.integers(3, 7))
        axis = "horizontal" if rng.random() < 0.5 else "vertical"
        mask_a = _random_shape(height, width, rng)
        mask_b = _random_shape(height, width, rng)
        if not _pair_condition(mask_a, mask_b, bridge_color):
            continue

        if axis == "horizontal":
            shared_rows = np.flatnonzero(mask_a[:, -1] & mask_b[:, 0])
            if len(shared_rows) == 0:
                continue
            row = int(rng.choice(shared_rows))
            bridge_len = int(rng.integers(1, 5))
            patch = np.zeros((height, width * 2 + bridge_len), dtype=int)
            left = patch[:, :width]
            right = patch[:, width + bridge_len :]
            left[mask_a] = color_a
            right[mask_b] = color_b
            patch[row, width : width + bridge_len] = bridge_color
            return patch

        shared_cols = np.flatnonzero(mask_a[-1, :] & mask_b[0, :])
        if len(shared_cols) == 0:
            continue
        col = int(rng.choice(shared_cols))
        bridge_len = int(rng.integers(1, 5))
        patch = np.zeros((height * 2 + bridge_len, width), dtype=int)
        top = patch[:height, :]
        bottom = patch[height + bridge_len :, :]
        top[mask_a] = color_a
        bottom[mask_b] = color_b
        patch[height : height + bridge_len, col] = bridge_color
        return patch
    raise RuntimeError("failed to build a valid bridged pair")


def _make_single_object(color: int, rng: np.random.Generator) -> np.ndarray:
    height = int(rng.integers(3, 7))
    width = int(rng.integers(3, 7))
    mask = _random_shape(height, width, rng)
    patch = np.zeros((height, width), dtype=int)
    patch[mask] = color
    return patch


def _place_units(units: list[np.ndarray], grid_shape: tuple[int, int], rng: np.random.Generator):
    height, width = grid_shape
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    for patch in sorted(units, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        ph, pw = patch.shape
        candidates = []
        for r in range(height - ph + 1):
            for c in range(width - pw + 1):
                r0 = max(0, r - 1)
                c0 = max(0, c - 1)
                r1 = min(height, r + ph + 1)
                c1 = min(width, c + pw + 1)
                if not blocked[r0:r1, c0:c1].any():
                    candidates.append((r, c))
        if not candidates:
            return None
        r, c = candidates[int(rng.integers(len(candidates)))]
        target = grid[r : r + ph, c : c + pw]
        mask = patch != BG
        target[mask] = patch[mask]
        blocked[max(0, r - 1) : min(height, r + ph + 1), max(0, c - 1) : min(width, c + pw + 1)] = True
    return grid


def _balanced_bridge_colors(pair_count: int, rng: np.random.Generator) -> list[int]:
    if pair_count == 1:
        return [RED if rng.random() < 0.5 else ORANGE]
    colors = [RED, ORANGE]
    while len(colors) < pair_count:
        colors.append(RED if rng.random() < 0.5 else ORANGE)
    rng.shuffle(colors)
    return colors


def _gen_attempt(rng: np.random.Generator):
    pair_count = int(rng.integers(1, 4))
    extra_limit = min(2, len(OBJECT_COLORS) - 2 * pair_count)
    extra_count = int(rng.integers(0, extra_limit + 1))
    total_objects = pair_count * 2 + extra_count
    colors = list(rng.choice(np.array(OBJECT_COLORS, dtype=int), size=total_objects, replace=False))
    bridge_colors = _balanced_bridge_colors(pair_count, rng)

    units = []
    color_idx = 0
    for bridge_color in bridge_colors:
        units.append(_make_pair_unit(bridge_color, colors[color_idx], colors[color_idx + 1], rng))
        color_idx += 2
    for _ in range(extra_count):
        units.append(_make_single_object(colors[color_idx], rng))
        color_idx += 1

    max_h = max(unit.shape[0] for unit in units)
    max_w = max(unit.shape[1] for unit in units)
    grid_h = int(rng.integers(max(12, max_h + 2), 29))
    grid_w = int(rng.integers(max(12, max_w + 2), 29))
    grid = _place_units(units, (grid_h, grid_w), rng)
    if grid is None:
        return None

    out = main(grid)
    assert out.shape == grid.shape
    assert max(grid.shape) <= 30 and max(out.shape) <= 30
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATE_ATTEMPTS} attempts")
