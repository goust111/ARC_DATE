from __future__ import annotations

from collections import Counter

import numpy as np


NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

RNG = np.random.default_rng()

GRID_H_MIN = 18
GRID_H_MAX = 30
GRID_W_MIN = 18
GRID_W_MAX = 30
SHAPE_SIDE_MAX = 5
AREA_MIN = 4
AREA_MAX = 10
MIN_FILL_X10 = 4
MIN_COL_GAP = 1
MIN_TOP_BOTTOM_GAP = 4
MAX_GENERATION_ATTEMPTS = 400
DUPLICATE_LAYOUTS = [(2,), (3,), (2, 2)]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int) -> None:
    h, w = patch.shape
    region = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    region[mask] = patch[mask]


def canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = []
    shape = np.asarray(mask, dtype=bool)
    for k in range(4):
        rotated = np.rot90(shape, k)
        for variant in (rotated, np.fliplr(rotated)):
            arr = np.ascontiguousarray(variant)
            variants.append((tuple(map(int, arr.shape)), arr.tobytes()))
    return min(variants)


def trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) == bg:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            pts = np.array(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            components.append(
                {
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    components.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    return components


def _extract_object(grid: np.ndarray, bg: int, comp: dict) -> dict:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    mask = np.zeros_like(patch, dtype=bool)
    for rr, cc in comp["coords"]:
        patch[rr - r0, cc - c0] = int(grid[rr, cc])
        mask[rr - r0, cc - c0] = True
    return {
        "c0": int(c0),
        "patch": patch,
        "mask": mask,
        "shape_key": canonical_shape_key(mask),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move every connected object to the top or bottom edge according to how often
    its shape appears, treating rotations and flips as the same shape.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("main expects a 2D grid")

    h, w = g.shape
    bg = get_bg(g)
    comps = find_components(g, bg)
    if not comps:
        raise ValueError("grid must contain at least one non-background object")

    objects = [_extract_object(g, bg, comp) for comp in comps]
    freq = Counter(obj["shape_key"] for obj in objects)

    out = np.full((h, w), bg, dtype=int)
    for obj in objects:
        patch = obj["patch"]
        dest_r0 = 0 if freq[obj["shape_key"]] >= 2 else h - patch.shape[0]
        paste(out, patch, dest_r0, obj["c0"], bg)
    return out


def _feasible_dims(area: int) -> list[tuple[int, int]]:
    dims = []
    for h in range(2, SHAPE_SIDE_MAX + 1):
        for w in range(2, SHAPE_SIDE_MAX + 1):
            if area > h * w:
                continue
            if area * 10 < MIN_FILL_X10 * h * w:
                continue
            if area < max(h, w):
                continue
            dims.append((h, w))
    return dims


def _line_seed(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if RNG.random() < 0.5:
        for i in range(max(h, w)):
            r = int(round(i * (h - 1) / max(max(h, w) - 1, 1)))
            c = int(round(i * (w - 1) / max(max(h, w) - 1, 1)))
            mask[r, c] = True
    else:
        for i in range(max(h, w)):
            r = int(round(i * (h - 1) / max(max(h, w) - 1, 1)))
            c = int(round((max(h, w) - 1 - i) * (w - 1) / max(max(h, w) - 1, 1)))
            mask[r, c] = True
    return mask


def _grow_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = _line_seed(h, w)
    while int(mask.sum()) > area:
        coords = np.argwhere(mask)
        rr, cc = coords[int(RNG.integers(len(coords)))]
        if int(mask.sum()) == 1:
            break
        mask[rr, cc] = False
        if _is_connected(mask):
            continue
        mask[rr, cc] = True

    while int(mask.sum()) < area:
        frontier = set()
        for rr, cc in np.argwhere(mask):
            for dr, dc in NEIGHBORS8:
                nr, nc = int(rr) + dr, int(cc) + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            raise RuntimeError("mask frontier unexpectedly empty")
        nr, nc = list(frontier)[int(RNG.integers(len(frontier)))]
        mask[nr, nc] = True

    return trim_mask(mask)


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return True
    start = tuple(map(int, pts[0]))
    stack = [start]
    seen = {start}
    while stack:
        rr, cc = stack.pop()
        for dr, dc in NEIGHBORS8:
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]):
                continue
            if not bool(mask[nr, nc]) or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _paint_mask(mask: np.ndarray, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    area = int(mask.sum())
    palette = [color for color in range(10) if color != bg]
    color_count = 2 if area <= 4 else int(RNG.integers(2, min(4, area) + 1))
    colors = RNG.choice(palette, size=color_count, replace=False).astype(int).tolist()
    coords = [tuple(map(int, pt)) for pt in np.argwhere(mask)]
    RNG.shuffle(coords)

    for index, (rr, cc) in enumerate(coords):
        if index < color_count:
            patch[rr, cc] = colors[index]
        else:
            patch[rr, cc] = colors[int(RNG.integers(color_count))]
    return patch


def _build_object(mask: np.ndarray, bg: int, group: str) -> dict:
    shape = np.asarray(mask, dtype=bool)
    return {
        "mask": shape,
        "patch": _paint_mask(shape, bg),
        "area": int(shape.sum()),
        "h": int(shape.shape[0]),
        "w": int(shape.shape[1]),
        "shape_key": canonical_shape_key(shape),
        "group": group,
    }


def _make_shape(area: int) -> np.ndarray:
    dims = _feasible_dims(area)
    if not dims:
        raise RuntimeError(f"no feasible dimensions for area {area}")
    for _ in range(200):
        h, w = dims[int(RNG.integers(len(dims)))]
        mask = _grow_mask(h, w, area)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) != area:
            continue
        if not _is_connected(mask):
            continue
        return mask
    raise RuntimeError(f"failed to build shape with area {area}")


def _make_distinct_object(
    bg: int,
    used_keys: set[tuple[tuple[int, int], bytes]],
    group: str,
    preferred_area: int | None = None,
) -> dict | None:
    area_choices = [preferred_area] if preferred_area is not None else []
    area_choices += [area for area in range(AREA_MIN, AREA_MAX + 1) if area != preferred_area]

    for area in area_choices:
        for _ in range(200):
            obj = _build_object(_make_shape(area), bg, group)
            if obj["shape_key"] in used_keys:
                continue
            return obj
    return None


def _clone_duplicate(template: dict, bg: int) -> dict:
    mask = np.asarray(template["mask"], dtype=bool)
    mask = np.rot90(mask, int(RNG.integers(4)))
    if bool(RNG.integers(2)):
        mask = np.fliplr(mask)
    mask = np.ascontiguousarray(mask)
    return _build_object(mask, bg, "top")


def _assign_columns(widths: list[int], total_width: int) -> list[int] | None:
    required = sum(widths) + MIN_COL_GAP * (len(widths) - 1)
    if required > total_width:
        return None
    slack = total_width - required
    extras = RNG.multinomial(slack, np.full(len(widths) + 1, 1.0 / (len(widths) + 1)))
    cols = []
    cursor = int(extras[0])
    for index, width in enumerate(widths):
        cols.append(cursor)
        cursor += width
        if index < len(widths) - 1:
            cursor += MIN_COL_GAP + int(extras[index + 1])
    return cols


def _render_output(objects: list[dict], h: int, w: int, bg: int) -> np.ndarray:
    out = np.full((h, w), bg, dtype=int)
    for obj in objects:
        dest_r0 = 0 if obj["group"] == "top" else h - obj["h"]
        paste(out, obj["patch"], dest_r0, obj["c0"], bg)
    return out


def _render_input(objects: list[dict], h: int, w: int, bg: int) -> np.ndarray | None:
    grid = np.full((h, w), bg, dtype=int)
    moved = False
    for obj in objects:
        out_r0 = 0 if obj["group"] == "top" else h - obj["h"]
        candidates = [r for r in range(h - obj["h"] + 1) if r != out_r0]
        if not candidates:
            return None
        r0 = int(candidates[int(RNG.integers(len(candidates)))])
        moved |= r0 != out_r0
        paste(grid, obj["patch"], r0, obj["c0"], bg)
    if not moved:
        return None
    return grid


def _has_area_collision(objects: list[dict]) -> bool:
    for i, left in enumerate(objects):
        for right in objects[i + 1 :]:
            if left["area"] == right["area"] and left["shape_key"] != right["shape_key"]:
                return True
    return False


def _build_groups(bg: int) -> list[dict] | None:
    duplicate_layout = DUPLICATE_LAYOUTS[int(RNG.integers(len(DUPLICATE_LAYOUTS)))]
    unique_count = 2 if sum(duplicate_layout) >= 4 else int(RNG.integers(2, 4))

    used_keys: set[tuple[tuple[int, int], bytes]] = set()
    objects: list[dict] = []
    duplicate_areas: list[int] = []

    for count in duplicate_layout:
        template = _make_distinct_object(bg, used_keys, "top")
        if template is None:
            return None
        used_keys.add(template["shape_key"])
        duplicate_areas.append(int(template["area"]))
        objects.append(template)
        for _ in range(count - 1):
            objects.append(_clone_duplicate(template, bg))

    forced_unique = _make_distinct_object(
        bg,
        used_keys,
        "bottom",
        preferred_area=int(duplicate_areas[int(RNG.integers(len(duplicate_areas)))]),
    )
    if forced_unique is None:
        return None
    used_keys.add(forced_unique["shape_key"])
    objects.append(forced_unique)

    while sum(obj["group"] == "bottom" for obj in objects) < unique_count:
        preferred_area = None
        if duplicate_areas and RNG.random() < 0.6:
            preferred_area = int(duplicate_areas[int(RNG.integers(len(duplicate_areas)))])
        obj = _make_distinct_object(bg, used_keys, "bottom", preferred_area=preferred_area)
        if obj is None:
            return None
        used_keys.add(obj["shape_key"])
        objects.append(obj)

    if not _has_area_collision(objects):
        return None
    return objects


def _gen_attempt() -> np.ndarray | None:
    bg = int(RNG.integers(10))
    objects = _build_groups(bg)
    if objects is None:
        return None

    RNG.shuffle(objects)
    widths = [obj["w"] for obj in objects]
    min_width = sum(widths) + MIN_COL_GAP * (len(widths) - 1)
    if min_width > GRID_W_MAX:
        return None
    width_low = max(GRID_W_MIN, min_width)
    width_high = GRID_W_MAX
    w = int(RNG.integers(width_low, width_high + 1))
    cols = _assign_columns(widths, w)
    if cols is None:
        return None
    for obj, c0 in zip(objects, cols):
        obj["c0"] = int(c0)

    top_max = max(obj["h"] for obj in objects if obj["group"] == "top")
    bottom_max = max(obj["h"] for obj in objects if obj["group"] == "bottom")
    min_height = max(GRID_H_MIN, top_max + bottom_max + MIN_TOP_BOTTOM_GAP)
    if min_height > GRID_H_MAX:
        return None
    h = int(RNG.integers(min_height, GRID_H_MAX + 1))

    inp = _render_input(objects, h, w, bg)
    if inp is None:
        return None
    if get_bg(inp) != bg:
        return None

    comps = find_components(inp, bg)
    if len(comps) != len(objects):
        return None

    expected = _render_output(objects, h, w, bg)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
