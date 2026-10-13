import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 300
ROOT_GAP = 2
MAX_OBJECTS = 8
MIN_GRID = 18
MAX_GRID = 30


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "set": set(cells),
                }
            )
    return comps


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and inner[2] < outer[2] and inner[3] < outer[3]


def _touching(comp_a: dict, comp_b: dict) -> bool:
    a = comp_a["set"]
    b = comp_b["set"]
    for r, c in a:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in b:
                return True
    return False


def _parent_indices(components: list[dict]) -> list[int | None]:
    parents: list[int | None] = [None] * len(components)
    for child_idx, child in enumerate(components):
        candidates: list[tuple[int, int]] = []
        for parent_idx, parent in enumerate(components):
            if parent_idx == child_idx:
                continue
            if not _strictly_contains(parent["bbox"], child["bbox"]):
                continue
            if _touching(parent, child):
                continue
            r0, c0, r1, c1 = parent["bbox"]
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            candidates.append((area, parent_idx))
        if candidates:
            candidates.sort()
            parents[child_idx] = int(candidates[0][1])
    return parents


def _depths(parents: list[int | None]) -> list[int]:
    memo: dict[int, int] = {}

    def depth(i: int) -> int:
        if i in memo:
            return memo[i]
        parent = parents[i]
        if parent is None:
            memo[i] = 0
        else:
            memo[i] = depth(parent) + 1
        return memo[i]

    return [depth(i) for i in range(len(parents))]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each contained object to the original color of its direct container.

    Objects are monochromatic 4-connected foreground components on the dominant
    background. Containment is determined by strict bounding-box nesting among
    non-touching components. Depth-0 objects keep their color; every deeper
    object inherits the original color of its nearest outer object.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _components(g, bg)
    parents = _parent_indices(components)
    _depths(parents)

    out = g.copy()
    for idx, comp in enumerate(components):
        parent_idx = parents[idx]
        if parent_idx is None:
            continue
        parent_color = int(components[parent_idx]["color"])
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = parent_color
    return out


def _crop_nonzero(arr: np.ndarray) -> np.ndarray:
    coords = np.argwhere(arr != 0)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _new_id(counter: list[int]) -> int:
    counter[0] += 1
    return counter[0]


def _solid_patch(counter: list[int]) -> np.ndarray:
    oid = _new_id(counter)
    h = random.randint(2, 4)
    w = random.randint(2, 5)
    return np.full((h, w), oid, dtype=int)


def _frame_patch(counter: list[int]) -> np.ndarray:
    oid = _new_id(counter)
    h = random.randint(4, 7)
    w = random.randint(4, 7)
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = oid
    patch[-1, :] = oid
    patch[:, 0] = oid
    patch[:, -1] = oid
    return patch


def _plus_patch(counter: list[int]) -> np.ndarray:
    oid = _new_id(counter)
    h = random.choice((3, 5))
    w = random.choice((3, 5))
    patch = np.zeros((h, w), dtype=int)
    patch[h // 2, :] = oid
    patch[:, w // 2] = oid
    return patch


def _l_patch(counter: list[int]) -> np.ndarray:
    oid = _new_id(counter)
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    patch = np.zeros((h, w), dtype=int)
    patch[:, 0] = oid
    patch[-1, :] = oid
    if random.random() < 0.5:
        patch = np.flipud(patch)
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    return patch


def _leaf_patch(counter: list[int]) -> np.ndarray:
    builders = [_solid_patch, _frame_patch, _plus_patch, _l_patch]
    return builders[random.randrange(len(builders))](counter)


def _paste_nonzero(dst: np.ndarray, src: np.ndarray, r0: int, c0: int) -> None:
    mask = src != 0
    dst_slice = dst[r0 : r0 + src.shape[0], c0 : c0 + src.shape[1]]
    dst_slice[mask] = src[mask]


def _wrap_children(children: list[np.ndarray], counter: list[int]) -> np.ndarray:
    oid = _new_id(counter)
    top_pad = 2 + random.randint(0, 2)
    bottom_pad = 2 + random.randint(0, 2)
    left_pad = 2 + random.randint(0, 2)
    right_pad = 2 + random.randint(0, 2)

    if len(children) == 1:
        child = children[0]
        inner_h, inner_w = child.shape
        canvas = np.zeros((top_pad + inner_h + bottom_pad, left_pad + inner_w + right_pad), dtype=int)
        _paste_nonzero(canvas, child, top_pad, left_pad)
    else:
        gap = random.randint(1, 3)
        horizontal = random.random() < 0.5
        if horizontal:
            inner_h = max(child.shape[0] for child in children)
            inner_w = sum(child.shape[1] for child in children) + gap * (len(children) - 1)
            canvas = np.zeros((top_pad + inner_h + bottom_pad, left_pad + inner_w + right_pad), dtype=int)
            col = left_pad
            for child in children:
                row = top_pad + random.randint(0, inner_h - child.shape[0])
                _paste_nonzero(canvas, child, row, col)
                col += child.shape[1] + gap
        else:
            inner_h = sum(child.shape[0] for child in children) + gap * (len(children) - 1)
            inner_w = max(child.shape[1] for child in children)
            canvas = np.zeros((top_pad + inner_h + bottom_pad, left_pad + inner_w + right_pad), dtype=int)
            row = top_pad
            for child in children:
                col = left_pad + random.randint(0, inner_w - child.shape[1])
                _paste_nonzero(canvas, child, row, col)
                row += child.shape[0] + gap

    canvas[0, :] = oid
    canvas[-1, :] = oid
    canvas[:, 0] = oid
    canvas[:, -1] = oid
    return canvas


def _family_patch(depth: int, counter: list[int]) -> np.ndarray:
    if depth <= 0:
        return _leaf_patch(counter)
    if depth == 1:
        child_count = 1 if random.random() < 0.45 else 2
        children = [_leaf_patch(counter) for _ in range(child_count)]
        return _wrap_children(children, counter)

    if random.random() < 0.5:
        children = [_family_patch(depth - 1, counter)]
    else:
        children = [_leaf_patch(counter), _leaf_patch(counter)]
        deep_idx = random.randrange(len(children))
        children[deep_idx] = _family_patch(depth - 1, counter)
    return _wrap_children(children, counter)


def _local_scene_ok(local_grid: np.ndarray) -> bool:
    components = _components(local_grid, 0)
    if len(components) < 3:
        return False
    parents = _parent_indices(components)
    depths = _depths(parents)
    if not any(parent is None for parent in parents):
        return False
    if not any(parent is not None for parent in parents):
        return False
    if max(depths) < 1:
        return False
    return True


def _place_roots(h: int, w: int, patches: list[np.ndarray]) -> np.ndarray | None:
    local = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    ordered = sorted(patches, key=lambda p: p.shape[0] * p.shape[1], reverse=True)

    for patch in ordered:
        ph, pw = patch.shape
        if ph > h or pw > w:
            return None
        row_lo = 1 if h - ph >= 2 else 0
        row_hi = h - ph - 1 if h - ph >= 2 else h - ph
        col_lo = 1 if w - pw >= 2 else 0
        col_hi = w - pw - 1 if w - pw >= 2 else w - pw
        if row_hi < row_lo or col_hi < col_lo:
            return None

        placed = False
        for _ in range(120):
            r0 = random.randint(row_lo, row_hi)
            c0 = random.randint(col_lo, col_hi)
            rr0 = max(0, r0 - ROOT_GAP)
            rr1 = min(h, r0 + ph + ROOT_GAP)
            cc0 = max(0, c0 - ROOT_GAP)
            cc1 = min(w, c0 + pw + ROOT_GAP)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            _paste_nonzero(local, patch, r0, c0)
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None
    return local


def _gen_attempt() -> np.ndarray | None:
    counter = [0]
    patches: list[np.ndarray] = []

    family_count = 1 if random.random() < 0.55 else 2
    for _ in range(family_count):
        depth = random.choice((1, 2, 3))
        patches.append(_family_patch(depth, counter))
        if counter[0] > MAX_OBJECTS:
            return None

    min_extra = 1 if counter[0] < 3 else 0
    max_extra = min(3, MAX_OBJECTS - counter[0])
    if max_extra < min_extra:
        return None
    for _ in range(random.randint(min_extra, max_extra)):
        patches.append(_leaf_patch(counter))

    if counter[0] > MAX_OBJECTS:
        return None

    max_ph = max(p.shape[0] for p in patches)
    max_pw = max(p.shape[1] for p in patches)
    sum_h = sum(p.shape[0] for p in patches)
    sum_w = sum(p.shape[1] for p in patches)
    h_min = max(MIN_GRID, max_ph + 3, sum_h // 2)
    w_min = max(MIN_GRID, max_pw + 3, sum_w // 2)
    if h_min > MAX_GRID or w_min > MAX_GRID:
        return None

    h = random.randint(h_min, MAX_GRID)
    w = random.randint(w_min, MAX_GRID)
    local = _place_roots(h, w, patches)
    if local is None or not _local_scene_ok(local):
        return None

    bg = random.randrange(10)
    object_ids = sorted(int(v) for v in np.unique(local) if v != 0)
    available = [c for c in range(10) if c != bg]
    if len(object_ids) > len(available):
        return None
    random.shuffle(available)
    color_map = {oid: available[i] for i, oid in enumerate(object_ids)}

    grid = np.full(local.shape, bg, dtype=int)
    for oid, color in color_map.items():
        grid[local == oid] = color

    if np.count_nonzero(grid != bg) >= (grid.size * 3) // 5:
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
