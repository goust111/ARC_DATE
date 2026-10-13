from __future__ import annotations

import random

import numpy as np


FOUR_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))
EIGHT_NEIGHBORS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

BACKGROUND = 0
PALETTE = tuple(range(1, 10))

COLOR_COUNT_CHOICES = (2, 3, 4)
COUNT_POOL = (2, 3, 4, 5)
AREA_POOL = tuple(range(2, 9))

GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 29
PLACEMENT_BORDER = 1
GENERATION_RETRY_LIMIT = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
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
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in FOUR_NEIGHBORS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0 = coords.min(axis=0)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _bbox_area(coords: np.ndarray) -> int:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _build_nested(records: list[dict], order_key: str = "area", thickness_key: str = "count") -> np.ndarray:
    keys = [int(record[order_key]) for record in records]
    assert len(keys) == len(set(keys)), f"{order_key} must uniquely determine layer order"
    ordered = sorted(records, key=lambda record: int(record[order_key]))

    inner = ordered[0]
    side = int(inner[thickness_key])
    out = np.full((side, side), int(inner["color"]), dtype=int)

    for record in ordered[1:]:
        thickness = int(record[thickness_key])
        next_side = int(out.shape[0] + 2 * thickness)
        wrapped = np.full((next_side, next_side), int(record["color"]), dtype=int)
        wrapped[thickness : thickness + out.shape[0], thickness : thickness + out.shape[1]] = out
        out = wrapped
    return out


def _analyze_grid(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    records = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components(g == color)
        assert comps, "every foreground color must appear"

        prototype = _normalize(comps[0])
        area = int(comps[0].shape[0])
        bbox_area = _bbox_area(comps[0])
        for comp in comps[1:]:
            assert int(comp.shape[0]) == area, "same-color objects must have equal area"
            assert _normalize(comp) == prototype, "same-color objects must have identical shape"

        records.append(
            {
                "color": int(color),
                "count": int(len(comps)),
                "area": area,
                "bbox_area": int(bbox_area),
                "total_area": int(area * len(comps)),
            }
        )

    areas = [record["area"] for record in records]
    assert len(areas) == len(set(areas)), "different colors must have distinct single-piece areas"
    assert records, "expected at least one foreground color"
    return bg, records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress repeated same-color pieces into concentric square layers ordered by single-piece area.

    Input structure:
    - A black background contains several monochromatic 4-connected objects.
    - Objects of the same color are repeated copies of one identical shape, while
      different colors have different single-piece areas.
    - The objects may be scattered anywhere; position and shape details are otherwise distractors.

    Rule:
    - For each foreground color, measure the area of one piece and count how many copies appear.
    - Sort the colors by single-piece area from small to large.
    - The smallest-area color becomes a solid square whose side length equals its copy count; then each
      later color wraps the current result in a square frame whose thickness equals that color's copy count.

    Key insight:
    - Each color contributes exactly two visible statistics: piece area decides layer order, and copy count
      decides geometric thickness, so the scattered source objects collapse into one nested square summary.
    """
    _bg, records = _analyze_grid(grid)
    out = _build_nested(records, order_key="area", thickness_key="count")
    assert max(out.shape) <= 30, "output must stay within ARC size limits"
    return out


def _random_blob(area: int, h: int, w: int, require_full_bbox: bool) -> np.ndarray | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in FOUR_NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    mask = _crop_mask(mask)
    if require_full_bbox and mask.shape != (h, w):
        return None
    return mask


def _monotone_path_bbox(area: int, h: int, w: int) -> np.ndarray:
    if random.random() < 0.5:
        r, c = 0, 0
        steps = [(1, 0)] * (h - 1) + [(0, 1)] * (w - 1)
    else:
        r, c = 0, w - 1
        steps = [(1, 0)] * (h - 1) + [(0, -1)] * (w - 1)
    random.shuffle(steps)

    cells = {(r, c)}
    for dr, dc in steps:
        r += dr
        c += dc
        cells.add((r, c))

    while len(cells) < area:
        frontier = []
        for cr, cc in cells:
            for dr, dc in FOUR_NEIGHBORS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for cr, cc in cells:
        mask[cr, cc] = True
    return mask


def _make_shape_mask(area: int, mode: str) -> np.ndarray:
    area = int(area)

    if mode == "stretched":
        candidates = [
            (h, w)
            for h in range(2, min(area, 6) + 1)
            for w in range(2, min(area, 6) + 1)
            if h * w > area and h + w - 1 <= area
        ]
        random.shuffle(candidates)
        for h, w in candidates:
            return _monotone_path_bbox(area, h, w)

    if mode == "compact":
        products = [
            h * w
            for h in range(1, min(area, 6) + 1)
            for w in range(1, min(area, 6) + 1)
            if h * w >= area
        ]
        min_product = min(products)
        candidates = [
            (h, w)
            for h in range(1, min(area, 6) + 1)
            for w in range(1, min(area, 6) + 1)
            if min_product <= h * w <= min_product + 1 and h * w >= area
        ]
        random.shuffle(candidates)
        for h, w in candidates:
            for _ in range(20):
                mask = _random_blob(area, h, w, require_full_bbox=False)
                if mask is not None:
                    return mask

    for _ in range(60):
        h = random.randint(1, min(area, 6))
        w = random.randint(1, min(area, 6))
        if h * w < area:
            continue
        mask = _random_blob(area, h, w, require_full_bbox=False)
        if mask is not None:
            return mask
    raise RuntimeError("failed to build a connected shape")


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _valid_slots(grid: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    ph, pw = patch.shape
    cells = np.argwhere(patch != BACKGROUND)
    slots = []

    for r0 in range(PLACEMENT_BORDER, h - ph - PLACEMENT_BORDER + 1):
        for c0 in range(PLACEMENT_BORDER, w - pw - PLACEMENT_BORDER + 1):
            ok = True
            for pr, pc in cells:
                rr, cc = r0 + int(pr), c0 + int(pc)
                if grid[rr, cc] != BACKGROUND:
                    ok = False
                    break
                for dr, dc in EIGHT_NEIGHBORS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BACKGROUND:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                slots.append((r0, c0))
    return slots


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    cells = np.argwhere(patch != BACKGROUND)
    for pr, pc in cells:
        grid[r0 + int(pr), c0 + int(pc)] = int(patch[int(pr), int(pc)])


def _pick_spec() -> tuple[list[dict], tuple[int, int]] | None:
    color_count = random.choice(COLOR_COUNT_CHOICES)
    areas = sorted(random.sample(AREA_POOL, color_count))

    adjacent = [i for i in range(color_count - 1) if 3 <= areas[i] and areas[i + 1] - areas[i] <= 3]
    if not adjacent:
        return None
    target = random.choice(adjacent)

    counts = random.sample(COUNT_POOL, color_count)
    while counts == sorted(counts):
        random.shuffle(counts)

    colors = random.sample(PALETTE, color_count)
    records = []
    for idx, (color, area, count) in enumerate(zip(colors, areas, counts)):
        if idx == target:
            mode = "stretched"
        elif idx == target + 1:
            mode = "compact"
        else:
            mode = random.choice(("compact", "random"))
        records.append({"color": int(color), "area": int(area), "count": int(count), "mode": mode})

    totals = [record["area"] * record["count"] for record in records]
    if len(set(totals)) != len(totals):
        return None
    if [record["count"] for record in records] == sorted(record["count"] for record in records):
        return None
    if sorted(range(color_count), key=lambda i: totals[i]) == list(range(color_count)):
        return None
    return records, (areas[target], areas[target + 1])


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        bg, records = _analyze_grid(grid)
        if bg != BACKGROUND:
            return False
        true_out = main(grid)
    except AssertionError:
        return False

    counts = [record["count"] for record in records]
    totals = [record["total_area"] for record in records]
    bbox_areas = [record["bbox_area"] for record in records]

    if counts == sorted(counts):
        return False
    if len(set(counts)) != len(counts):
        return False
    if len(set(totals)) != len(totals):
        return False
    if len(set(bbox_areas)) != len(bbox_areas):
        return False

    alt_count_order = _build_nested(records, order_key="count", thickness_key="count")
    alt_total_order = _build_nested(records, order_key="total_area", thickness_key="count")
    alt_bbox_order = _build_nested(records, order_key="bbox_area", thickness_key="count")
    alt_area_thickness = _build_nested(records, order_key="area", thickness_key="area")

    if np.array_equal(true_out, alt_count_order):
        return False
    if np.array_equal(true_out, alt_total_order):
        return False
    if np.array_equal(true_out, alt_bbox_order):
        return False
    if np.array_equal(true_out, alt_area_thickness):
        return False

    return max(true_out.shape) <= 30


def _gen_attempt() -> np.ndarray | None:
    picked = _pick_spec()
    if picked is None:
        return None
    spec, target_pair = picked

    enriched = []
    for record in spec:
        mask = _make_shape_mask(record["area"], record["mode"])
        coords = np.argwhere(mask)
        enriched.append(
            {
                **record,
                "mask": mask,
                "patch": _colorize(mask, record["color"]),
                "bbox_area": _bbox_area(coords),
            }
        )

    bbox_map = {record["area"]: record["bbox_area"] for record in enriched}
    if bbox_map[target_pair[0]] <= bbox_map[target_pair[1]]:
        return None

    side_h = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    side_w = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    grid = np.zeros((side_h, side_w), dtype=int)

    placements = []
    for record in sorted(enriched, key=lambda record: (-record["area"], -record["count"], -record["bbox_area"])):
        for _ in range(record["count"]):
            placements.append(record["patch"])
    random.shuffle(placements)
    placements.sort(key=lambda patch: -int(np.count_nonzero(patch)))

    for patch in placements:
        slots = _valid_slots(grid, patch)
        if not slots:
            return None
        _paste(grid, patch, *random.choice(slots))

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric counting-frame input")
