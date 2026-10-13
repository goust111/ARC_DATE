import numpy as np
from collections import Counter


_RNG = np.random.default_rng()
_NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
_PALETTE = list(range(10))
_MAX_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in _NEIGHBORS8:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            comps.append({"coords": coords, "bbox": (r0, c0, r1, c1)})
    return comps


def _mask_from_component(comp: dict) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[np.ix_(rows, cols)]


def _canonical_shape_key(mask: np.ndarray) -> tuple:
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        for cand in (rot, np.fliplr(rot)):
            trimmed = _trim_mask(cand)
            variants.append((trimmed.shape, tuple(map(tuple, trimmed.astype(int)))))
    return min(variants)


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    view = dst[top : top + h, left : left + w]
    keep = patch != bg
    view[keep] = patch[keep]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    comps = _find_components(g, bg)
    assert comps

    objects = []
    for comp in comps:
        mask, (r0, c0) = _mask_from_component(comp)
        r1, c1 = comp["bbox"][2], comp["bbox"][3]
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        for r, c in comp["coords"]:
            patch[r - r0, c - c0] = g[r, c]
        objects.append(
            {
                "left": c0,
                "patch": patch,
                "shape_key": _canonical_shape_key(mask),
            }
        )

    freq = Counter(obj["shape_key"] for obj in objects)
    out = np.full((h, w), bg, dtype=int)
    for obj in objects:
        patch = obj["patch"]
        dest_r = 0 if freq[obj["shape_key"]] >= 2 else h - patch.shape[0]
        _paste(out, patch, dest_r, obj["left"], bg)
    return out


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (_RNG.integers(height), _RNG.integers(width))
    mask[start] = True
    cells = {start}
    frontier = set()
    for dr, dc in _NEIGHBORS8:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))

    while len(cells) < area and frontier:
        idx = int(_RNG.integers(len(frontier)))
        cell = list(frontier)[idx]
        frontier.remove(cell)
        mask[cell] = True
        cells.add(cell)
        for dr, dc in _NEIGHBORS8:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))

    if len(cells) != area:
        return None
    trimmed = _trim_mask(mask)
    if trimmed.shape != (height, width):
        return None
    return trimmed


def _sample_base_mask(existing: set[tuple]) -> np.ndarray:
    for _ in range(200):
        height = int(_RNG.integers(2, 6))
        width = int(_RNG.integers(2, 6))
        area = int(_RNG.integers(3, min(9, height * width) + 1))
        mask = _random_connected_mask(height, width, area)
        if mask is None:
            continue
        key = _canonical_shape_key(mask)
        if key not in existing:
            return mask
    raise RuntimeError("failed to sample distinct shape")


def _random_transform(mask: np.ndarray) -> np.ndarray:
    rot = np.rot90(mask, int(_RNG.integers(4)))
    if bool(_RNG.integers(2)):
        rot = np.fliplr(rot)
    return _trim_mask(rot)


def _paint_patch(mask: np.ndarray, bg: int) -> np.ndarray:
    colors = [c for c in _PALETTE if c != bg]
    color_count = int(_RNG.integers(1, min(4, len(colors)) + 1))
    chosen = list(_RNG.choice(colors, size=color_count, replace=False))
    patch = np.full(mask.shape, bg, dtype=int)
    coords = np.argwhere(mask)
    for r, c in coords:
        patch[r, c] = int(chosen[int(_RNG.integers(len(chosen)))])
    return patch


def _intervals_fit(widths: list[int], total_width: int, gap: int) -> bool:
    need = sum(widths) + gap * (len(widths) - 1)
    return need <= total_width


def _sample_columns(widths: list[int], total_width: int, gap: int) -> list[int]:
    extra = total_width - (sum(widths) + gap * (len(widths) - 1))
    slots = len(widths) + 1
    slack = [0] * slots
    for _ in range(extra):
        slack[int(_RNG.integers(slots))] += 1
    cols = []
    cur = slack[0]
    for i, width in enumerate(widths):
        cols.append(cur)
        cur += width
        if i < len(widths) - 1:
            cur += gap + slack[i + 1]
    return cols


def _build_instances() -> list[dict]:
    dup_group_count = int(_RNG.integers(1, 3))
    unique_count = int(_RNG.integers(1, 4))
    group_sizes = [int(_RNG.integers(2, 4)) for _ in range(dup_group_count)]

    used_keys: set[tuple] = set()
    instances: list[dict] = []

    for _ in range(dup_group_count):
        base = _sample_base_mask(used_keys)
        key = _canonical_shape_key(base)
        used_keys.add(key)
        copies = int(_RNG.integers(2, 4))
        for _ in range(copies):
            mask = _random_transform(base)
            instances.append({"mask": mask, "shape_key": key, "duplicate": True})

    while len([x for x in instances if not x.get("unique_done", False) and not x["duplicate"]]) < unique_count:
        base = _sample_base_mask(used_keys)
        key = _canonical_shape_key(base)
        used_keys.add(key)
        instances.append({"mask": _random_transform(base), "shape_key": key, "duplicate": False})

    return instances


def _gen_attempt() -> np.ndarray | None:
    bg = int(_RNG.integers(10))
    instances = _build_instances()
    _RNG.shuffle(instances)

    widths = [inst["mask"].shape[1] for inst in instances]
    heights = [inst["mask"].shape[0] for inst in instances]
    width_needed = sum(widths) + 2 * (len(widths) - 1)
    min_w = max(18, width_needed + 2)
    if min_w > 30:
        return None
    width = int(_RNG.integers(min_w, 31))
    height = int(_RNG.integers(max(14, max(heights) + 6), 31))
    if not _intervals_fit(widths, width, gap=2):
        return None

    cols = _sample_columns(widths, width, gap=2)
    grid = np.full((height, width), bg, dtype=int)

    for inst, left in zip(instances, cols):
        patch = _paint_patch(inst["mask"], bg)
        ph, pw = patch.shape
        max_top = height - ph
        if max_top < 0:
            return None
        top = int(_RNG.integers(0, max_top + 1))
        grid[top : top + ph, left : left + pw][patch != bg] = patch[patch != bg]

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    top_rows = []
    bottom_rows = []
    for comp in _find_components(grid, bg):
        mask, _ = _mask_from_component(comp)
        key = _canonical_shape_key(mask)
        if sum(_canonical_shape_key(_mask_from_component(other)[0]) == key for other in _find_components(grid, bg)) >= 2:
            top_rows.append(comp["bbox"][0])
        else:
            bottom_rows.append(comp["bbox"][2])
    if not top_rows or not bottom_rows:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
