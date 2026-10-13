from collections import Counter, deque

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CATEGORY_ORDER = ("both", "h", "v", "none")

RNG = np.random.default_rng()

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
DOUBLE_COUNT_RANGE = (1, 4)
MASK_RETRIES = 200
PLACEMENT_RETRIES = 250
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    assert rows.any() and cols.any(), "mask must contain at least one foreground cell"
    r_idx = np.flatnonzero(rows)
    c_idx = np.flatnonzero(cols)
    return mask[int(r_idx[0]) : int(r_idx[-1]) + 1, int(c_idx[0]) : int(c_idx[-1]) + 1]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    h, w = mask.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                queue.append((rr, cc))
    return len(seen) == int(mask.sum())


def _classify_mask(mask: np.ndarray) -> str:
    horiz = np.array_equal(mask, np.flipud(mask))
    vert = np.array_equal(mask, np.fliplr(mask))
    if horiz and vert:
        return "both"
    if horiz:
        return "h"
    if vert:
        return "v"
    return "none"


def _extract_objects(grid: np.ndarray, bg: int | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)

    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

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
                for dr, dc in DIRS4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(g[rr, cc]) == color:
                        seen[rr, cc] = True
                        queue.append((rr, cc))

            rs = [r for r, _ in coords]
            cs = [c for _, c in coords]
            r_min, r_max = min(rs), max(rs)
            c_min, c_max = min(cs), max(cs)
            mask = np.zeros((r_max - r_min + 1, c_max - c_min + 1), dtype=bool)
            for r, c in coords:
                mask[r - r_min, c - c_min] = True

            objects.append(
                {
                    "index": len(objects),
                    "color": color,
                    "mask": mask,
                    "area": int(mask.sum()),
                    "category": _classify_mask(mask),
                }
            )

    return objects


def _paste_mask(canvas: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    patch = canvas[top : top + mask.shape[0], left : left + mask.shape[1]]
    patch[mask] = color


def _build_output(objects: list[dict], bg: int) -> np.ndarray:
    grouped = {category: [] for category in CATEGORY_ORDER}
    for obj in objects:
        grouped[obj["category"]].append(obj)

    grouped["both"].sort(key=lambda obj: (-int(obj["area"]), int(obj["index"])))

    row_heights = [
        max((int(obj["mask"].shape[0]) for obj in grouped[category]), default=0)
        for category in CATEGORY_ORDER
    ]
    row_widths = [
        sum(int(obj["mask"].shape[1]) for obj in grouped[category]) + max(0, len(grouped[category]) - 1)
        for category in CATEGORY_ORDER
    ]

    out_h = sum(row_heights) + (len(CATEGORY_ORDER) - 1)
    out_w = max(1, max(row_widths, default=0))
    out = np.full((out_h, out_w), bg, dtype=int)

    top = 0
    for row_idx, category in enumerate(CATEGORY_ORDER):
        left = 0
        for obj in grouped[category]:
            mask = obj["mask"]
            _paste_mask(out, mask, int(obj["color"]), top, left)
            left += int(mask.shape[1]) + 1
        top += row_heights[row_idx]
        if row_idx < len(CATEGORY_ORDER) - 1:
            top += 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify monochrome sprites by reflection symmetry and stack the categories into rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    objects = _extract_objects(g, bg)
    assert objects, "input must contain at least one foreground object"

    return _build_output(objects, bg)


def _touches_side(mask: np.ndarray, side: str) -> bool:
    if side == "top":
        return bool(mask[0, :].any())
    if side == "bottom":
        return bool(mask[-1, :].any())
    if side == "left":
        return bool(mask[:, 0].any())
    if side == "right":
        return bool(mask[:, -1].any())
    raise ValueError(f"unknown side: {side}")


def _distance_to_side(r: int, c: int, h: int, w: int, side: str) -> int:
    if side == "top":
        return r
    if side == "bottom":
        return h - 1 - r
    if side == "left":
        return c
    if side == "right":
        return w - 1 - c
    raise ValueError(f"unknown side: {side}")


def _cell_on_side(r: int, c: int, h: int, w: int, side: str) -> bool:
    return _distance_to_side(r, c, h, w, side) == 0


def _mask_area_bounds(h: int, w: int) -> tuple[int, int]:
    cells = h * w
    low = max(2, int(np.ceil(cells * 0.35)))
    high = max(low, int(np.floor(cells * 0.70)))
    if high >= cells and cells > 1:
        high = cells - 1
    return low, max(low, high)


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        rr0, cc0 = int(r), int(c)
        for dr, dc in DIRS4:
            rr, cc = rr0 + dr, cc0 + dc
            if 0 <= rr < h and 0 <= cc < w and not bool(mask[rr, cc]):
                candidates.add((rr, cc))
    return list(candidates)


def _grow_connected_mask(
    h: int,
    w: int,
    min_area: int,
    max_area: int,
    must_touch: tuple[str, ...] = (),
) -> np.ndarray | None:
    target = int(RNG.integers(min_area, max_area + 1))
    mask = np.zeros((h, w), dtype=bool)
    start_r = int(RNG.integers(0, h))
    start_c = int(RNG.integers(0, w))
    mask[start_r, start_c] = True

    while True:
        unmet = [side for side in must_touch if not _touches_side(mask, side)]
        area = int(mask.sum())
        if area >= target and not unmet:
            if area >= max_area or float(RNG.random()) < 0.55:
                break

        frontier = _frontier(mask)
        if not frontier:
            break

        if unmet:
            enriched = []
            for cell in frontier:
                r, c = cell
                gain = sum(1 for side in unmet if _cell_on_side(r, c, h, w, side))
                dist = sum(_distance_to_side(r, c, h, w, side) for side in unmet)
                enriched.append((gain, dist, cell))
            best_gain = max(item[0] for item in enriched)
            subset = [item for item in enriched if item[0] == best_gain]
            best_dist = min(item[1] for item in subset)
            frontier = [cell for gain, dist, cell in subset if dist == best_dist]

        pick = frontier[int(RNG.integers(0, len(frontier)))]
        mask[pick] = True

        if int(mask.sum()) > max_area and must_touch:
            return None

    if any(not _touches_side(mask, side) for side in must_touch):
        return None
    return mask


def _reflect_horizontal(base: np.ndarray, full_h: int) -> np.ndarray:
    full = np.zeros((full_h, base.shape[1]), dtype=bool)
    for r, c in np.argwhere(base):
        rr, cc = int(r), int(c)
        full[rr, cc] = True
        full[full_h - 1 - rr, cc] = True
    return full


def _reflect_vertical(base: np.ndarray, full_w: int) -> np.ndarray:
    full = np.zeros((base.shape[0], full_w), dtype=bool)
    for r, c in np.argwhere(base):
        rr, cc = int(r), int(c)
        full[rr, cc] = True
        full[rr, full_w - 1 - cc] = True
    return full


def _reflect_both(base: np.ndarray, full_h: int, full_w: int) -> np.ndarray:
    full = np.zeros((full_h, full_w), dtype=bool)
    for r, c in np.argwhere(base):
        rr, cc = int(r), int(c)
        full[rr, cc] = True
        full[full_h - 1 - rr, cc] = True
        full[rr, full_w - 1 - cc] = True
        full[full_h - 1 - rr, full_w - 1 - cc] = True
    return full


def _make_mask(kind: str) -> np.ndarray | None:
    for _ in range(MASK_RETRIES):
        full_h = int(RNG.integers(3, 7))
        full_w = int(RNG.integers(3, 7))

        if kind == "both":
            base_h, base_w = (full_h + 1) // 2, (full_w + 1) // 2
            low, high = _mask_area_bounds(base_h, base_w)
            base = _grow_connected_mask(base_h, base_w, low, high, must_touch=("bottom", "right"))
            if base is None:
                continue
            mask = _reflect_both(base, full_h, full_w)
        elif kind == "h":
            base_h, base_w = (full_h + 1) // 2, full_w
            low, high = _mask_area_bounds(base_h, base_w)
            base = _grow_connected_mask(base_h, base_w, low, high, must_touch=("bottom",))
            if base is None or np.array_equal(base, np.fliplr(base)):
                continue
            mask = _reflect_horizontal(base, full_h)
        elif kind == "v":
            base_h, base_w = full_h, (full_w + 1) // 2
            low, high = _mask_area_bounds(base_h, base_w)
            base = _grow_connected_mask(base_h, base_w, low, high, must_touch=("right",))
            if base is None or np.array_equal(base, np.flipud(base)):
                continue
            mask = _reflect_vertical(base, full_w)
        else:
            low, high = _mask_area_bounds(full_h, full_w)
            mask = _grow_connected_mask(full_h, full_w, low, high)
            if mask is None:
                continue

        mask = _trim_mask(mask)
        if min(mask.shape) < 2:
            continue
        if int(mask.sum()) < 4:
            continue
        if not _is_connected(mask):
            continue
        if _classify_mask(mask) != kind:
            continue
        return mask

    return None


def _place_objects(specs: list[dict], h: int, w: int, bg: int) -> np.ndarray | None:
    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    order = list(range(len(specs)))
    order.sort(key=lambda idx: specs[idx]["mask"].shape[0] * specs[idx]["mask"].shape[1], reverse=True)

    for idx in order:
        spec = specs[idx]
        mask = spec["mask"]
        mh, mw = mask.shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = int(RNG.integers(0, h - mh + 1))
            left = int(RNG.integers(0, w - mw + 1))
            if bool(blocked[top : top + mh, left : left + mw].any()):
                continue

            _paste_mask(grid, mask, int(spec["color"]), top, left)
            r0 = max(0, top - 1)
            r1 = min(h, top + mh + 1)
            c0 = max(0, left - 1)
            c1 = min(w, left + mw + 1)
            blocked[r0:r1, c0:c1] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = int(RNG.integers(0, 10))
    double_count = int(RNG.integers(*DOUBLE_COUNT_RANGE))
    categories = ["both"] * double_count + ["h", "v", "none"]

    palette = [color for color in range(10) if color != bg]
    colors = [int(color) for color in RNG.permutation(palette)[: len(categories)]]

    specs: list[dict] = []
    used_double_areas: set[int] = set()
    for color, category in zip(colors, categories, strict=False):
        mask = _make_mask(category)
        if mask is None:
            return None
        if category == "both":
            for _ in range(MASK_RETRIES):
                area = int(mask.sum())
                if area not in used_double_areas:
                    used_double_areas.add(area)
                    break
                mask = _make_mask(category)
                if mask is None:
                    return None
            else:
                return None

        specs.append({"color": color, "category": category, "mask": mask})

    max_h = max(int(spec["mask"].shape[0]) for spec in specs)
    max_w = max(int(spec["mask"].shape[1]) for spec in specs)
    reserve_area = sum((int(spec["mask"].shape[0]) + 2) * (int(spec["mask"].shape[1]) + 2) for spec in specs)
    approx_side = int(np.ceil(np.sqrt(reserve_area)))

    h_low = max(GRID_H_RANGE[0], max_h + 4, min(approx_side, GRID_H_RANGE[1]))
    w_low = max(GRID_W_RANGE[0], max_w + 4, min(approx_side, GRID_W_RANGE[1]))
    if h_low > GRID_H_RANGE[1] or w_low > GRID_W_RANGE[1]:
        return None

    intended = {int(spec["color"]): str(spec["category"]) for spec in specs}

    for _ in range(40):
        h = int(RNG.integers(h_low, GRID_H_RANGE[1] + 1))
        w = int(RNG.integers(w_low, GRID_W_RANGE[1] + 1))
        grid = _place_objects(specs, h, w, bg)
        if grid is None:
            continue
        if get_bg(grid) != bg:
            continue

        found = _extract_objects(grid, bg)
        if len(found) != len(specs):
            continue
        if any(int(obj["color"]) not in intended for obj in found):
            continue
        if any(intended[int(obj["color"])] != str(obj["category"]) for obj in found):
            continue

        expected = _build_output(found, bg)
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-sorting ARC sample")
