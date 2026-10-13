import itertools
import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords = []
        while stack:
            cr, cc2 = stack.pop()
            coords.append((cr, cc2))
            for nr, nc in ((cr - 1, cc2), (cr + 1, cc2), (cr, cc2 - 1), (cr, cc2 + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_record(coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    return {
        "coords": coords.astype(int),
        "bbox": (r0, c0, r1, c1),
        "area": int(coords.shape[0]),
        "height": int(r1 - r0 + 1),
        "width": int(c1 - c0 + 1),
    }


def _extract_scene(grid: np.ndarray) -> tuple[int, int, int, list[dict], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    colors = sorted(set(map(int, g.ravel())) - {bg})
    assert len(colors) == 2

    candidates = []
    bottom = h - 1
    for beacon_color in colors:
        beacon_coords = np.argwhere(g == beacon_color)
        if beacon_coords.size == 0:
            continue
        if not np.all(beacon_coords[:, 0] == bottom):
            continue
        beacon_cols = sorted(int(c) for c in beacon_coords[:, 1])
        if len(beacon_cols) != len(set(beacon_cols)):
            continue

        object_color = next(color for color in colors if color != beacon_color)
        if np.any(g[bottom] == object_color):
            continue
        object_mask = g == object_color
        objects = [_component_record(coords) for coords in _components(object_mask)]
        if len(objects) != len(beacon_cols):
            continue
        candidates.append((bg, object_color, beacon_color, objects, beacon_cols))

    assert len(candidates) == 1
    return candidates[0]


def _sorted_objects(objects: list[dict], pair_mode: str) -> list[dict]:
    if pair_mode == "size_desc":
        key = lambda obj: (-obj["area"], obj["bbox"][1], obj["bbox"][0])
    elif pair_mode == "size_asc":
        key = lambda obj: (obj["area"], obj["bbox"][1], obj["bbox"][0])
    elif pair_mode == "source_x":
        key = lambda obj: (
            obj["bbox"][1] + obj["bbox"][3],
            obj["bbox"][1],
            -obj["area"],
        )
    else:
        raise ValueError(f"unknown pair_mode: {pair_mode}")
    return sorted(objects, key=key)


def _aligned_coords(obj: dict, beacon_col: int, align_mode: str) -> np.ndarray:
    coords = obj["coords"].copy()
    _r0, c0, _r1, c1 = obj["bbox"]
    if align_mode == "center":
        center_col = (c0 + c1) // 2
        dc = int(beacon_col) - int(center_col)
    elif align_mode == "left":
        dc = int(beacon_col) - int(c0)
    else:
        raise ValueError(f"unknown align_mode: {align_mode}")
    coords[:, 1] += dc
    return coords


def _drop_in_order(
    shape: tuple[int, int],
    bg: int,
    object_color: int,
    beacon_color: int,
    paired: list[tuple[dict, int]],
) -> tuple[np.ndarray, list[dict]]:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    placements = []
    for _, beacon_col in paired:
        out[h - 1, int(beacon_col)] = beacon_color

    blocked = np.zeros((h, w), dtype=bool)
    blocked[h - 1, :] = True

    for obj, beacon_col in paired:
        coords = _aligned_coords(obj, beacon_col, align_mode=obj.get("align_mode", "center"))
        if coords.size == 0:
            raise AssertionError("empty object")
        if coords[:, 1].min() < 0 or coords[:, 1].max() >= w:
            raise AssertionError("aligned object leaves the grid")
        if coords[:, 0].min() < 0 or coords[:, 0].max() >= h - 1:
            raise AssertionError("object must start above the bottom row")
        if blocked[coords[:, 0], coords[:, 1]].any():
            raise AssertionError("aligned object overlaps a blocked cell")

        current = coords
        while True:
            down = current.copy()
            down[:, 0] += 1
            if down[:, 0].max() >= h or blocked[down[:, 0], down[:, 1]].any():
                break
            current = down

        out[current[:, 0], current[:, 1]] = object_color
        blocked[current[:, 0], current[:, 1]] = True
        placements.append(
            {
                "coords": current,
                "bbox": _bbox(current),
                "area": obj["area"],
                "width": obj["width"],
                "height": obj["height"],
                "beacon_col": int(beacon_col),
            }
        )
    return out, placements


def _solve(
    grid: np.ndarray,
    *,
    pair_mode: str = "size_desc",
    process_mode: str = "left_to_right",
    align_mode: str = "center",
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, object_color, beacon_color, objects, beacon_cols = _extract_scene(g)

    areas = [obj["area"] for obj in objects]
    assert len(set(areas)) == len(areas)

    ranked = _sorted_objects(objects, pair_mode=pair_mode)
    paired = []
    for obj, beacon_col in zip(ranked, sorted(beacon_cols)):
        enriched = dict(obj)
        enriched["align_mode"] = align_mode
        paired.append((enriched, int(beacon_col)))

    if process_mode == "left_to_right":
        ordered = paired
    elif process_mode == "right_to_left":
        ordered = list(reversed(paired))
    else:
        raise ValueError(f"unknown process_mode: {process_mode}")

    out, _placements = _drop_in_order((h, w), bg, object_color, beacon_color, ordered)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Hoist grey objects above bottom beacons by size rank, then let them fall and stack in beacon order.

    Input structure:
    - The background is uniform, with one foreground color used for several disconnected movable objects.
    - A second foreground color appears only as one-pixel beacons on distinct columns of the bottom row.
    - The movable objects have pairwise distinct pixel counts and do not touch each other.

    Rule:
    - Sort the objects by pixel count from largest to smallest, and sort the bottom beacons from left to right.
    - Pair the k-th largest object with the k-th beacon, shift that object horizontally so the center column of its tight bounding box aligns with the beacon column, then drop it straight down.
    - Process beacons from left to right: each shifted object falls until one more row would hit the bottom row or a previously landed object; clear all original object positions and keep the bottom beacons.

    Key insight:
    - The bottom beacons encode three roles at once: destination columns, the size-rank pairing, and the landing order, so close beacon columns force earlier landed objects to become the support for later ones.
    """
    return _solve(grid, pair_mode="size_desc", process_mode="left_to_right", align_mode="center")


BG = 0
OBJECT_COLOR = 5
BEACON_COLOR = 4
WIDTH_CHOICES = (3, 5, 7)
HEIGHT_RANGE = (2, 5)
OBJECT_COUNT_RANGE = (3, 5)
GEN_RETRIES = 500


def _interval_candidates(prev: tuple[int, int] | None, width: int, force_left: bool, force_right: bool) -> list[tuple[int, int]]:
    cands = []
    for left in range(width):
        for right in range(left, width):
            if force_left and left != 0:
                continue
            if force_right and right != width - 1:
                continue
            if prev is not None and (right < prev[0] or left > prev[1]):
                continue
            cands.append((left, right))
    return cands


def _random_shape_mask() -> np.ndarray:
    for _ in range(200):
        h = random.randint(*HEIGHT_RANGE)
        w = random.choice(WIDTH_CHOICES)
        left_row = random.randrange(h)
        right_row = random.randrange(h)
        rows = []
        prev = None
        for r in range(h):
            cands = _interval_candidates(prev, w, r == left_row, r == right_row)
            if not cands:
                break
            rows.append(random.choice(cands))
            prev = rows[-1]
        if len(rows) != h:
            continue
        if not any(left == 0 for left, _ in rows):
            continue
        if not any(right == w - 1 for _, right in rows):
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, (left, right) in enumerate(rows):
            mask[r, left : right + 1] = True
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox(coords)
        if (r0, c0, r1, c1) != (0, 0, h - 1, w - 1):
            continue
        return mask
    raise RuntimeError("failed to generate a shape mask")


def _shape_record(mask: np.ndarray) -> dict:
    coords = np.argwhere(mask)
    rec = _component_record(coords)
    rec["mask"] = mask.copy()
    return rec


def _random_shapes() -> list[dict]:
    count = random.randint(*OBJECT_COUNT_RANGE)
    chosen = []
    used_areas = set()
    for _ in range(300):
        chosen.clear()
        used_areas.clear()
        while len(chosen) < count:
            mask = _random_shape_mask()
            rec = _shape_record(mask)
            if rec["area"] in used_areas:
                continue
            used_areas.add(rec["area"])
            chosen.append(rec)
        areas = [rec["area"] for rec in chosen]
        if len(set(areas)) == count:
            return chosen
    raise RuntimeError("failed to generate distinct-area shapes")


def _beacon_candidates(ranked: list[dict], width: int) -> list[tuple[int, ...]]:
    cols = range(width)
    cands = []
    for beacon_cols in itertools.combinations(cols, len(ranked)):
        ok = True
        for obj, col in zip(ranked, beacon_cols):
            half = obj["width"] // 2
            if not (half <= col <= width - 1 - half):
                ok = False
                break
        if ok:
            cands.append(tuple(int(c) for c in beacon_cols))
    return cands


def _has_projection_overlap(ranked: list[dict], beacon_cols: tuple[int, ...]) -> bool:
    spans = []
    for obj, col in zip(ranked, beacon_cols):
        half = obj["width"] // 2
        spans.append((col - half, col + half))
    for (l0, r0), (l1, r1) in zip(spans, spans[1:]):
        if max(l0, l1) <= min(r0, r1):
            return True
    return False


def _local_object(shape: dict, row: int, col: int) -> dict:
    coords = shape["coords"].copy()
    coords[:, 0] += int(row)
    coords[:, 1] += int(col)
    return _component_record(coords)


def _output_from_shapes(shapes: list[dict], beacon_cols: tuple[int, ...], height: int, width: int) -> tuple[np.ndarray, list[dict]]:
    ranked = sorted(shapes, key=lambda obj: (-obj["area"], obj["width"], obj["height"]))
    local = [_local_object(obj, 0, 0) for obj in ranked]
    for src, mask_src in zip(local, ranked):
        src["width"] = mask_src["width"]
        src["height"] = mask_src["height"]
    paired = []
    for obj, col in zip(local, beacon_cols):
        obj["align_mode"] = "center"
        paired.append((obj, int(col)))
    return _drop_in_order((height, width), BG, OBJECT_COLOR, BEACON_COLOR, paired)


def _box_touches(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 1) -> bool:
    return not (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _place_sources(shapes: list[dict], placements: list[dict], height: int, width: int) -> list[dict] | None:
    assert len(shapes) == len(placements)
    ranked = sorted(shapes, key=lambda obj: (-obj["area"], obj["width"], obj["height"]))
    placement_by_area = {p["area"]: p for p in placements}
    source_shapes = ranked[:]
    random.shuffle(source_shapes)

    placed_boxes: list[tuple[int, int, int, int]] = []
    source_records: list[dict] = []
    upper_cap = max(1, height // 2)
    for shape in source_shapes:
        final_top = placement_by_area[shape["area"]]["bbox"][0]
        max_row = min(final_top - 1, upper_cap - shape["height"])
        if max_row < 0:
            return None
        candidates = []
        for row in range(max_row + 1):
            for col in range(width - shape["width"] + 1):
                box = (row, col, row + shape["height"] - 1, col + shape["width"] - 1)
                if box[2] >= height - 1:
                    continue
                if any(_box_touches(box, other, padding=1) for other in placed_boxes):
                    continue
                candidates.append((row, col, box))
        if not candidates:
            return None
        row, col, box = random.choice(candidates)
        placed_boxes.append(box)
        source_records.append(_local_object(shape, row, col))
    return source_records


def _render_input(source_objects: list[dict], beacon_cols: tuple[int, ...], height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for col in beacon_cols:
        grid[height - 1, int(col)] = BEACON_COLOR
    for obj in source_objects:
        coords = obj["coords"]
        grid[coords[:, 0], coords[:, 1]] = OBJECT_COLOR
    return grid


def _safe_variant(grid: np.ndarray, *, pair_mode: str, process_mode: str, align_mode: str) -> np.ndarray | None:
    try:
        return _solve(grid, pair_mode=pair_mode, process_mode=process_mode, align_mode=align_mode)
    except Exception:
        return None


def _is_same(a: np.ndarray | None, b: np.ndarray) -> bool:
    return a is not None and np.array_equal(a, b)


def _source_order_differs(objects: list[dict]) -> bool:
    by_size = [obj["area"] for obj in sorted(objects, key=lambda obj: -obj["area"])]
    by_x = [obj["area"] for obj in sorted(objects, key=lambda obj: (obj["bbox"][1] + obj["bbox"][3], obj["bbox"][1]))]
    return by_size != by_x


def _validate_sample(inp: np.ndarray, expected: np.ndarray) -> bool:
    solved = _safe_variant(inp, pair_mode="size_desc", process_mode="left_to_right", align_mode="center")
    if not _is_same(solved, expected):
        return False

    bg, _object_color, _beacon_color, objects, beacon_cols = _extract_scene(inp)
    if bg != BG or len(objects) != len(beacon_cols):
        return False
    if not _source_order_differs(objects):
        return False

    if _is_same(_safe_variant(inp, pair_mode="size_asc", process_mode="left_to_right", align_mode="center"), expected):
        return False
    if _is_same(_safe_variant(inp, pair_mode="source_x", process_mode="left_to_right", align_mode="center"), expected):
        return False
    if _is_same(_safe_variant(inp, pair_mode="size_desc", process_mode="right_to_left", align_mode="center"), expected):
        return False
    if _is_same(_safe_variant(inp, pair_mode="size_desc", process_mode="left_to_right", align_mode="left"), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shapes = _random_shapes()
    ranked = sorted(shapes, key=lambda obj: (-obj["area"], obj["width"], obj["height"]))
    total_height = sum(obj["height"] for obj in ranked)
    max_width = max(obj["width"] for obj in ranked)

    height_lo = max(14, total_height + 2)
    height_hi = min(28, total_height + 7)
    if height_lo > height_hi:
        height_lo = height_hi = min(28, max(14, total_height + 4))
    width_lo = max(14, max_width + len(ranked) * 2)
    width_hi = min(20, max_width + len(ranked) * 4)
    if width_lo > width_hi:
        width_lo = width_hi = min(20, max(width_lo, max_width + len(ranked) * 3))
    if width_lo > 20:
        return None

    height = random.randint(height_lo, height_hi)
    width = random.randint(width_lo, width_hi)

    beacon_options = _beacon_candidates(ranked, width)
    if not beacon_options:
        return None
    overlap_options = [cols for cols in beacon_options if _has_projection_overlap(ranked, cols)]
    if overlap_options:
        beacon_cols = random.choice(overlap_options if random.random() < 0.8 else beacon_options)
    else:
        beacon_cols = random.choice(beacon_options)

    try:
        expected, placements = _output_from_shapes(shapes, beacon_cols, height, width)
    except AssertionError:
        return None

    source_objects = _place_sources(shapes, placements, height, width)
    if source_objects is None:
        return None
    inp = _render_input(source_objects, beacon_cols, height, width)
    if not _validate_sample(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
