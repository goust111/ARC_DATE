from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
PANEL = 8
COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_ATTEMPTS = 600

_CLASS_COUNT_CYCLE: list[int] = []
_PANEL_SIDE_CYCLE: list[bool] = []
_CATALOG: dict[int, list[np.ndarray]] | None = None


def _mode(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        cells: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(cells, dtype=int))

    return comps


def _component_color(grid: np.ndarray, coords: np.ndarray) -> int:
    vals = set(map(int, grid[coords[:, 0], coords[:, 1]]))
    assert len(vals) == 1, "expected monochromatic component"
    return next(iter(vals))


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    shifted = coords - np.array([r0, c0])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    coords = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))
    return (int(mask.shape[0]), int(mask.shape[1])), coords


def _d4_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = [
        mask,
        np.rot90(mask, 1),
        np.rot90(mask, 2),
        np.rot90(mask, 3),
        np.fliplr(mask),
        np.flipud(mask),
        mask.T,
        np.fliplr(np.rot90(mask, 1)),
    ]
    out: list[np.ndarray] = []
    seen = set()
    for var in variants:
        arr = np.array(var, dtype=bool, copy=True)
        key = _mask_key(arr)
        if key not in seen:
            seen.add(key)
            out.append(arr)
    return out


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    return min(_mask_key(var) for var in _d4_variants(mask))


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _find_panel_box(g: np.ndarray) -> tuple[int, int, int, int]:
    panel = g == PANEL
    box = _bbox(panel)
    assert box is not None, "expected a teal legend panel"
    r0, c0, r1, c1 = box
    h, w = g.shape
    assert r0 == 0 and r1 == h - 1, "legend panel must span the full grid height"
    assert c0 == 0 or c1 == w - 1, "legend panel must touch one side of the grid"
    assert c1 - c0 + 1 >= 6, "legend panel must be wide enough"
    assert len(_components(panel)) == 1, "expected one connected teal panel"
    return box


def _panel_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _legend_entries(g: np.ndarray, bg: int, panel_mask: np.ndarray) -> list[dict]:
    entries: list[dict] = []
    for color in sorted(set(map(int, g[panel_mask])) - {bg, PANEL}):
        comps = _components((g == color) & panel_mask)
        assert len(comps) == 1, "each legend color must form one shape"
        coords = comps[0]
        mask = _mask_from_coords(coords)
        assert _touches_all_edges(mask)
        entries.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": _bbox_coords(coords),
                "area": int(coords.shape[0]),
                "key": _canonical_key(mask),
            }
        )

    areas = [entry["area"] for entry in entries]
    colors = [entry["color"] for entry in entries]
    assert len(entries) >= 3, "expected several legend shapes"
    assert len(areas) == len(set(areas)), "legend areas must be unique"
    assert len(colors) == len(set(colors)), "legend colors must be unique"
    return entries


def _scatter_objects(g: np.ndarray, bg: int, panel_mask: np.ndarray) -> tuple[int, list[np.ndarray]]:
    outside = (g != bg) & ~panel_mask
    comps = _components(outside)
    assert comps, "expected scattered objects outside the panel"
    colors = [_component_color(g, coords) for coords in comps]
    assert len(set(colors)) == 1, "scattered objects must all have one color"
    assert colors[0] not in (bg, PANEL)
    return int(colors[0]), comps


def _crop_legend_shape(entry: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = entry["bbox"]
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    coords = entry["coords"] - np.array([r0, c0])
    out[coords[:, 0], coords[:, 1]] = int(entry["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Count scattered shape classes and use that count as an area key into the legend panel.

    Input structure:
    - A black grid has one full-height teal legend panel touching the left or right side.
    - The panel contains several separated monochrome legend shapes with distinct colors and distinct pixel counts.
    - Outside the panel are separated same-color 4-connected objects; repeated objects may be rotated or reflected copies.

    Rule:
    - Ignore the scattered objects' color and raw object count.
    - Group the outside objects by equality after any rotation or mirror reflection, then count the resulting shape classes.
    - In the legend panel, find the unique shape whose pixel count equals that class count and return only that shape's bounding-box crop on the black background.

    Key insight:
    - The legend is indexed by area, but the number used as the key is the count of rotation/reflection shape classes, not a color match or the raw number of objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30 by 30"

    bg = _mode(g)
    assert bg == BACKGROUND, "expected black background"

    panel_box = _find_panel_box(g)
    in_panel = _panel_mask(g.shape, panel_box)
    legend = _legend_entries(g, bg, in_panel)
    scatter_color, objects = _scatter_objects(g, bg, in_panel)

    legend_colors = {entry["color"] for entry in legend}
    assert scatter_color in legend_colors, "scatter color must name a legend decoy"

    class_keys = {_canonical_key(_mask_from_coords(coords)) for coords in objects}
    class_count = len(class_keys)
    matches = [entry for entry in legend if int(entry["area"]) == class_count]
    assert len(matches) == 1, "shape-class count must select exactly one legend shape"
    return _crop_legend_shape(matches[0], bg)


# --------------- generator ---------------


def _shape_catalog() -> dict[int, list[np.ndarray]]:
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG

    shape_texts = (
        ("11",),
        ("111",),
        ("10", "11"),
        ("1111",),
        ("11", "11"),
        ("100", "111"),
        ("111", "010"),
        ("111", "010", "010"),
        ("110", "010", "011"),
        ("111", "100", "100"),
        ("010", "111", "010"),
        ("111", "111"),
        ("1110", "0011", "0010"),
        ("100", "100", "111", "100"),
        ("111", "101", "110"),
        ("111", "111", "010"),
        ("111", "101", "111"),
        ("1111", "1111"),
        ("111", "111", "110"),
        ("1111", "0100", "0111"),
    )
    by_area: dict[int, list[np.ndarray]] = {area: [] for area in range(2, 9)}
    seen: dict[int, set] = {area: set() for area in by_area}
    for rows in shape_texts:
        mask = np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)
        area = int(mask.sum())
        assert area in by_area
        assert _touches_all_edges(mask)
        assert len(_components(mask)) == 1
        key = _canonical_key(mask)
        if key not in seen[area]:
            seen[area].add(key)
            by_area[area].append(mask)

    assert all(by_area[area] for area in by_area)
    _CATALOG = by_area
    return by_area


def _next_class_count() -> int:
    global _CLASS_COUNT_CYCLE
    if not _CLASS_COUNT_CYCLE:
        _CLASS_COUNT_CYCLE = [2, 3, 4]
        random.shuffle(_CLASS_COUNT_CYCLE)
    return _CLASS_COUNT_CYCLE.pop()


def _next_panel_left() -> bool:
    global _PANEL_SIDE_CYCLE
    if not _PANEL_SIDE_CYCLE:
        _PANEL_SIDE_CYCLE = [False, True]
        random.shuffle(_PANEL_SIDE_CYCLE)
    return _PANEL_SIDE_CYCLE.pop()


def _random_variant(mask: np.ndarray) -> np.ndarray:
    return random.choice(_d4_variants(mask)).copy()


def _choose_scatter_classes(n: int) -> list[np.ndarray] | None:
    catalog = _shape_catalog()
    pair_areas = [area for area, shapes in catalog.items() if area >= 4 and len(shapes) >= 2]
    random.shuffle(pair_areas)

    for pair_area in pair_areas:
        pair = random.sample(catalog[pair_area], 2)
        chosen = list(pair)
        keys = {_canonical_key(mask) for mask in chosen}
        if len(chosen) == n:
            random.shuffle(chosen)
            return chosen
        all_shapes = [mask for area in range(3, 9) for mask in catalog[area]]
        random.shuffle(all_shapes)
        for mask in all_shapes:
            key = _canonical_key(mask)
            if key in keys:
                continue
            chosen.append(mask)
            keys.add(key)
            if len(chosen) == n:
                random.shuffle(chosen)
                return chosen
    return None


def _multiplicities(n: int, raw_count: int, masks: list[np.ndarray]) -> list[int] | None:
    counts = [1] * n
    extras = raw_count - n
    flexible = [i for i, mask in enumerate(masks) if len(_d4_variants(mask)) > 1]
    if extras <= 0 or not flexible:
        return None

    first = random.choice(flexible)
    counts[first] += 1
    extras -= 1
    for _ in range(extras):
        counts[random.randrange(n)] += 1
    return counts


def _object_masks(class_masks: list[np.ndarray], counts: list[int]) -> list[np.ndarray]:
    objects: list[np.ndarray] = []
    for mask, count in zip(class_masks, counts):
        variants = _d4_variants(mask)
        if count >= 2 and len(variants) > 1:
            first = random.randrange(len(variants))
            second_choices = [i for i in range(len(variants)) if _mask_key(variants[i]) != _mask_key(variants[first])]
            picked = [variants[first], variants[random.choice(second_choices)]]
            picked.extend(random.choice(variants) for _ in range(count - 2))
        else:
            picked = [random.choice(variants) for _ in range(count)]
        objects.extend(np.array(mask, dtype=bool, copy=True) for mask in picked)
    random.shuffle(objects)
    return objects


def _choose_legend_entries(n: int, raw_count: int) -> tuple[list[dict], int, np.ndarray] | None:
    catalog = _shape_catalog()
    possible_decoy_areas = [area for area in range(2, 9) if area not in (n, raw_count)]
    if not possible_decoy_areas:
        return None

    color_decoy_area = random.choice(possible_decoy_areas)
    areas = [n, raw_count, color_decoy_area]
    fillers = [area for area in range(2, 9) if area not in areas]
    random.shuffle(fillers)
    areas.extend(fillers[: random.randint(1, 2)])
    random.shuffle(areas)

    colors = random.sample(COLORS, len(areas))
    entries: list[dict] = []
    correct_output = None
    scatter_color = None

    for area, color in zip(areas, colors):
        choices = catalog[area]
        if not choices:
            return None
        mask = _random_variant(random.choice(choices))
        role = "filler"
        if area == n:
            role = "correct"
            correct_output = _paint_mask(mask, color)
        elif area == raw_count:
            role = "raw_decoy"
        elif area == color_decoy_area and scatter_color is None:
            role = "color_decoy"
            scatter_color = int(color)
        entries.append({"area": int(area), "color": int(color), "mask": mask, "role": role})

    if scatter_color is None or correct_output is None:
        return None
    return entries, int(scatter_color), correct_output


def _paint_mask(mask: np.ndarray, color: int, bg: int = BACKGROUND) -> np.ndarray:
    out = np.full(mask.shape, int(bg), dtype=int)
    out[mask] = int(color)
    return out


def _distribute_extra(total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[random.randrange(slots)] += 1
    return parts


def _place_legend(grid: np.ndarray, panel_left: bool, panel_w: int, entries: list[dict]) -> bool:
    h, w = grid.shape
    c_panel = 0 if panel_left else w - panel_w
    grid[:, c_panel : c_panel + panel_w] = PANEL

    heights = [int(entry["mask"].shape[0]) for entry in entries]
    min_total = sum(heights) + len(entries) + 1
    if min_total > h:
        return False
    gaps = [1] * (len(entries) + 1)
    extra = h - min_total
    for add, gap_i in zip(_distribute_extra(extra, len(gaps)), range(len(gaps))):
        gaps[gap_i] += add

    row = gaps[0]
    for i, entry in enumerate(entries):
        mask = entry["mask"]
        mh, mw = mask.shape
        if mw > panel_w - 2:
            return False
        c0 = c_panel + random.randint(1, panel_w - mw - 1)
        grid[row : row + mh, c0 : c0 + mw][mask] = int(entry["color"])
        row += mh + gaps[i + 1]
    return True


def _can_place_object(
    grid: np.ndarray,
    mask: np.ndarray,
    r0: int,
    c0: int,
    scatter_box: tuple[int, int, int, int],
) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    sr0, sc0, sr1, sc1 = scatter_box
    if r0 < sr0 or c0 < sc0 or r0 + mh - 1 > sr1 or c0 + mw - 1 > sc1:
        return False

    for rr, cc in np.argwhere(mask):
        r = int(r0 + rr)
        c = int(c0 + cc)
        if grid[r, c] != BACKGROUND:
            return False
        r_start = max(0, r - 1)
        r_end = min(h, r + 2)
        c_start = max(0, c - 1)
        c_end = min(w, c + 2)
        if np.any(grid[r_start:r_end, c_start:c_end] != BACKGROUND):
            return False
    return True


def _place_scatter(
    grid: np.ndarray,
    panel_left: bool,
    panel_w: int,
    objects: list[np.ndarray],
    color: int,
) -> bool:
    h, w = grid.shape
    gap = 1
    if panel_left:
        scatter_box = (1, panel_w + gap, h - 2, w - 2)
    else:
        scatter_box = (1, 1, h - 2, w - panel_w - gap - 1)

    random.shuffle(objects)
    objects.sort(key=lambda mask: int(mask.size), reverse=True)
    for mask in objects:
        mh, mw = mask.shape
        sr0, sc0, sr1, sc1 = scatter_box
        if mh > sr1 - sr0 + 1 or mw > sc1 - sc0 + 1:
            return False
        placed = False
        for _ in range(250):
            r0 = random.randint(sr0, sr1 - mh + 1)
            c0 = random.randint(sc0, sc1 - mw + 1)
            if _can_place_object(grid, mask, r0, c0, scatter_box):
                grid[r0 : r0 + mh, c0 : c0 + mw][mask] = int(color)
                placed = True
                break
        if not placed:
            return False
    return True


def _raw_crop_for_entry(entry: dict) -> np.ndarray:
    return _paint_mask(entry["mask"], int(entry["color"]))


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    n: int,
    raw_count: int,
    scatter_color: int,
    entries: list[dict],
) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    try:
        panel_box = _find_panel_box(grid)
        in_panel = _panel_mask(grid.shape, panel_box)
        legend = _legend_entries(grid, BACKGROUND, in_panel)
        actual_scatter_color, objects = _scatter_objects(grid, BACKGROUND, in_panel)
    except AssertionError:
        return False

    if actual_scatter_color != scatter_color:
        return False
    if len(objects) != raw_count:
        return False

    object_masks = [_mask_from_coords(coords) for coords in objects]
    class_keys = [_canonical_key(mask) for mask in object_masks]
    if len(set(class_keys)) != n:
        return False
    if len(set(int(mask.sum()) for mask in object_masks)) >= n:
        return False

    repeated_rotated = False
    by_class: dict[tuple, set] = {}
    for mask, key in zip(object_masks, class_keys):
        by_class.setdefault(key, set()).add(_mask_key(mask))
    repeated_rotated = any(len(orientations) > 1 for orientations in by_class.values())
    if not repeated_rotated:
        return False

    legend_by_area = {int(entry["area"]): entry for entry in legend}
    if set(legend_by_area) != {int(entry["area"]) for entry in entries}:
        return False
    if n not in legend_by_area or raw_count not in legend_by_area:
        return False

    color_decoys = [entry for entry in legend if int(entry["color"]) == scatter_color]
    if len(color_decoys) != 1 or int(color_decoys[0]["area"]) in (n, raw_count):
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    if np.array_equal(_crop_legend_shape(color_decoys[0], BACKGROUND), expected):
        return False
    if np.array_equal(_crop_legend_shape(legend_by_area[raw_count], BACKGROUND), expected):
        return False
    return True


def _gen_attempt(n: int, panel_left: bool) -> np.ndarray | None:
    raw_count = random.randint(n + 2, 8)
    class_masks = _choose_scatter_classes(n)
    if class_masks is None:
        return None
    counts = _multiplicities(n, raw_count, class_masks)
    if counts is None:
        return None
    objects = _object_masks(class_masks, counts)

    legend_pack = _choose_legend_entries(n, raw_count)
    if legend_pack is None:
        return None
    entries, scatter_color, expected = legend_pack

    panel_w = random.randint(7, 9)
    height = random.randint(max(22, sum(entry["mask"].shape[0] for entry in entries) + len(entries) + 1), 30)
    scatter_w = random.randint(15, 30 - panel_w - 1)
    width = panel_w + 1 + scatter_w
    grid = np.full((height, width), BACKGROUND, dtype=int)

    random.shuffle(entries)
    if not _place_legend(grid, panel_left, panel_w, entries):
        return None
    if not _place_scatter(grid, panel_left, panel_w, objects, scatter_color):
        return None
    if _mode(grid) != BACKGROUND:
        return None
    if not _semantic_checks(grid, expected, n, raw_count, scatter_color, entries):
        return None
    return grid


def generate_input() -> np.ndarray:
    n = _next_class_count()
    panel_left = _next_panel_left()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(n, panel_left)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid count-index dictionary input")
