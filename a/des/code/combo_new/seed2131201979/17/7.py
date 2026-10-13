from collections import Counter, defaultdict, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400
RNG = np.random.default_rng()


def get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(g[r0, c0]) == bg:
                continue
            color = int(g[r0, c0])
            queue = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []

            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            br0, bc0, br1, bc1 = bbox_from_coords(coords)
            mask = np.zeros((br1 - br0 + 1, bc1 - bc0 + 1), dtype=bool)
            for r, c in coords:
                mask[r - br0, c - bc0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (br0, bc0, br1, bc1),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )
    return comps


def scale_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(np.asarray(mask, dtype=bool), scale, axis=0), scale, axis=1)


def erode_mask(mask: np.ndarray, steps: int) -> np.ndarray:
    cur = np.asarray(mask, dtype=bool).copy()
    for _ in range(steps):
        if cur.shape[0] < 3 or cur.shape[1] < 3:
            return np.zeros_like(cur, dtype=bool)
        nxt = np.zeros_like(cur, dtype=bool)
        nxt[1:-1, 1:-1] = (
            cur[1:-1, 1:-1]
            & cur[:-2, 1:-1]
            & cur[2:, 1:-1]
            & cur[1:-1, :-2]
            & cur[1:-1, 2:]
        )
        cur = nxt
        if not cur.any():
            break
    return cur


def canonical_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    h, w = arr.shape
    for scale in range(min(h, w), 0, -1):
        if h % scale != 0 or w % scale != 0:
            continue
        rh, rw = h // scale, w // scale
        view = arr.reshape(rh, scale, rw, scale).transpose(0, 2, 1, 3)
        coarse = view[:, :, 0, 0]
        if np.all(view == coarse[:, :, None, None]):
            return coarse.copy()
    return arr.copy()


def shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    norm = canonical_mask(mask)
    return tuple(norm.shape), norm.astype(np.uint8).tobytes()


def paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    window = grid[top : top + h, left : left + w]
    window[mask] = int(color)
    grid[top : top + h, left : left + w] = window


def clear_component(grid: np.ndarray, comp: dict, bg: int) -> None:
    rows = [r for r, _ in comp["coords"]]
    cols = [c for _, c in comp["coords"]]
    grid[rows, cols] = int(bg)


def apply_group_layers(out: np.ndarray, members: list[dict], bg: int) -> None:
    ordered = sorted(members, key=lambda comp: int(comp["area"]), reverse=True)
    for comp in ordered:
        clear_component(out, comp, bg)

    anchor = ordered[0]
    top, left, _bottom, _right = anchor["bbox"]
    anchor_mask = np.asarray(anchor["mask"], dtype=bool)
    for depth, comp in enumerate(ordered):
        layer_mask = erode_mask(anchor_mask, depth)
        if not layer_mask.any():
            break
        paint_mask(out, layer_mask, top, left, int(comp["color"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge each repeated scaled-shape family into layered colors at its largest instance.

    Input structure:
    - A single dominant background color fills the canvas.
    - Every foreground object is a monochrome 4-connected component.
    - Some objects are scaled copies of the same binary shape; other objects may be unique.

    Rule:
    - Group foreground objects by normalized shape, ignoring color and position and collapsing exact integer block scaling.
    - In each repeated-shape group, keep only the largest-area object as the anchor and erase the smaller copies.
    - Repaint the anchor location with the group colors ordered by decreasing area: the largest color fills the full anchor mask, the next fills the anchor mask eroded by one pixel, and each later color fills one more inward erosion layer.
    - Objects whose normalized shape appears only once remain unchanged.

    Key insight:
    - The non-obvious relation is scale-normalized shape equality: objects that look different in size still belong together, and their sizes determine the outside-to-inside layering order on the largest copy.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg)
    if not comps:
        return g.copy()

    groups: dict[tuple[tuple[int, int], bytes], list[dict]] = defaultdict(list)
    for comp in comps:
        groups[shape_key(comp["mask"])].append(comp)

    out = g.copy()
    for members in groups.values():
        if len(members) >= 2:
            apply_group_layers(out, members, bg)
    return out


def random_connected_mask(min_dim: int = 2, max_dim: int = 4) -> np.ndarray:
    h = int(RNG.integers(min_dim, max_dim + 1))
    w = int(RNG.integers(min_dim, max_dim + 1))
    max_cells = min(h * w, 6)
    min_cells = 1 if max_cells == 1 else 2
    target = int(RNG.integers(min_cells, max_cells + 1))

    start = (int(RNG.integers(0, h)), int(RNG.integers(0, w)))
    cells = {start}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(cells) < target and frontier:
        pick = list(frontier)[int(RNG.integers(0, len(frontier)))]
        frontier.remove(pick)
        cells.add(pick)
        add_frontier(*pick)

    coords = np.array(sorted(cells), dtype=int)
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    mask = trim_mask(mask)

    if int(RNG.integers(0, 4)):
        mask = np.rot90(mask, int(RNG.integers(0, 4)))
    if bool(RNG.integers(0, 2)):
        mask = np.flipud(mask)
    if bool(RNG.integers(0, 2)):
        mask = np.fliplr(mask)
    return np.asarray(mask, dtype=bool)


def has_visible_depth(mask: np.ndarray, group_size: int) -> bool:
    return all(erode_mask(mask, depth).any() for depth in range(group_size))


def make_group_objects(group_id: int, colors: list[int], used_keys: set) -> list[dict] | None:
    group_size = len(colors)
    for _ in range(200):
        max_scale = int(RNG.integers(group_size, 5))
        base = random_connected_mask()
        if max(base.shape) * max_scale > 12:
            continue
        anchor_mask = scale_mask(base, max_scale)
        if not has_visible_depth(anchor_mask, group_size):
            continue
        key = shape_key(anchor_mask)
        if key in used_keys:
            continue

        objects = []
        for scale, color in zip(range(max_scale - group_size + 1, max_scale + 1), colors):
            mask = scale_mask(base, scale)
            objects.append(
                {
                    "group": group_id,
                    "mask": mask,
                    "color": int(color),
                    "area": int(mask.sum()),
                    "key": key,
                }
            )
        used_keys.add(key)
        return objects
    return None


def make_singleton_object(color: int, used_keys: set) -> dict | None:
    for _ in range(200):
        scale = int(RNG.integers(1, 5))
        base = random_connected_mask(1, 4)
        if max(base.shape) * scale > 10:
            continue
        mask = scale_mask(base, scale)
        key = shape_key(mask)
        if key in used_keys:
            continue
        used_keys.add(key)
        return {
            "group": None,
            "mask": mask,
            "color": int(color),
            "area": int(mask.sum()),
            "key": key,
        }
    return None


def dilate8(mask: np.ndarray) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    h, w = src.shape
    out = np.zeros_like(src)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rs = max(0, -dr)
            re = min(h, h - dr)
            cs = max(0, -dc)
            ce = min(w, w - dc)
            out[rs + dr : re + dr, cs + dc : ce + dc] |= src[rs:re, cs:ce]
    return out


def place_objects(height: int, width: int, bg: int, objects: list[dict]) -> np.ndarray | None:
    grid = np.full((height, width), int(bg), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    for obj in sorted(objects, key=lambda item: int(item["area"]), reverse=True):
        mask = np.asarray(obj["mask"], dtype=bool)
        h, w = mask.shape
        blocked = dilate8(occupied)
        candidates: list[tuple[int, int]] = []
        for top in range(height - h + 1):
            for left in range(width - w + 1):
                if not np.any(blocked[top : top + h, left : left + w] & mask):
                    candidates.append((top, left))
        if not candidates:
            return None

        top, left = candidates[int(RNG.integers(0, len(candidates)))]
        obj["top"] = int(top)
        obj["left"] = int(left)
        paint_mask(grid, mask, top, left, int(obj["color"]))
        occupied[top : top + h, left : left + w] |= mask
    return grid


def expected_output_from_objects(inp: np.ndarray, bg: int, objects: list[dict]) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    groups: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        if obj["group"] is not None:
            groups[int(obj["group"])].append(obj)

    for members in groups.values():
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda item: int(item["area"]), reverse=True)
        for obj in ordered:
            top, left = int(obj["top"]), int(obj["left"])
            rows, cols = np.where(obj["mask"])
            out[top + rows, left + cols] = int(bg)

        anchor = ordered[0]
        top, left = int(anchor["top"]), int(anchor["left"])
        for depth, obj in enumerate(ordered):
            layer_mask = erode_mask(anchor["mask"], depth)
            if not layer_mask.any():
                break
            paint_mask(out, layer_mask, top, left, int(obj["color"]))
    return out


def choose_canvas(objects: list[dict]) -> tuple[int, int] | None:
    max_h = max(int(obj["mask"].shape[0]) for obj in objects)
    max_w = max(int(obj["mask"].shape[1]) for obj in objects)
    total_area = sum(int(obj["area"]) for obj in objects)
    max_area = max(int(obj["area"]) for obj in objects)
    min_area = total_area + max_area + 1
    min_h = max(16, max_h + 2)
    if min_h > 30:
        return None

    for _ in range(50):
        height = int(RNG.integers(min_h, 31))
        min_w = max(16, max_w + 2, (min_area + height - 1) // height)
        if min_w > 30:
            continue
        width = int(RNG.integers(min_w, 31))
        if height * width < min_area:
            continue
        return height, width
    return None


def build_object_set(bg: int) -> list[dict] | None:
    group_count = int(RNG.integers(1, 3))
    singleton_count = int(RNG.integers(0, 3))
    group_sizes = [int(RNG.integers(2, 4)) for _ in range(group_count)]
    total_objects = sum(group_sizes) + singleton_count
    if total_objects > 9:
        return None

    palette = [color for color in range(10) if color != bg]
    RNG.shuffle(palette)
    colors = palette[: total_objects]
    color_idx = 0
    used_keys: set = set()
    objects: list[dict] = []

    for group_id, group_size in enumerate(group_sizes):
        group_colors = colors[color_idx : color_idx + group_size]
        color_idx += group_size
        group_objects = make_group_objects(group_id, list(group_colors), used_keys)
        if group_objects is None:
            return None
        objects.extend(group_objects)

    for _ in range(singleton_count):
        obj = make_singleton_object(colors[color_idx], used_keys)
        color_idx += 1
        if obj is None:
            return None
        objects.append(obj)

    return objects


def gen_attempt() -> np.ndarray | None:
    bg = int(RNG.integers(0, 10))
    objects = build_object_set(bg)
    if not objects:
        return None

    canvas = choose_canvas(objects)
    if canvas is None:
        return None
    height, width = canvas

    laid_out = [dict(obj) for obj in objects]
    inp = place_objects(height, width, bg, laid_out)
    if inp is None:
        return None
    if get_bg(inp) != bg:
        return None

    comps = find_components(inp, bg)
    if len(comps) != len(laid_out):
        return None
    if Counter(shape_key(comp["mask"]) for comp in comps) != Counter(obj["key"] for obj in laid_out):
        return None

    expected = expected_output_from_objects(inp, bg, laid_out)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
