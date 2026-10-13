from __future__ import annotations

import itertools
import random

import numpy as np


BACKGROUND = 0
SOURCE_COLOR = 1
MARKER_COLORS = (2, 3, 4)

GRID_HEIGHT_RANGE = (12, 16)
GRID_WIDTH_RANGE = (14, 20)
SOURCE_BBOX_HEIGHTS = (3, 4)
SOURCE_BBOX_WIDTHS = (3, 4, 5)
SOURCE_PIXEL_COUNTS = (5, 6, 7, 8)
MARKER_COUNTS = (2, 3)
MARKER_DISTANCES = (4, 5, 6, 7, 8)
GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 240


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=12)
    return int(np.argmax(counts))


def _bbox_from_coords(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_coords(coords: list[tuple[int, int]], color: int, bg: int = BACKGROUND) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        out[r - r0, c - c0] = color
    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _is_connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _reflect_coords(
    source_coords: list[tuple[int, int]],
    anchor: tuple[int, int],
    marker: tuple[int, int],
) -> list[tuple[int, int]]:
    ar, ac = anchor
    mr, mc = marker
    if ar == mr and ac != mc:
        return [(r, ac + mc - c) for r, c in source_coords]
    if ac == mc and ar != mr:
        return [(ar + mr - r, c) for r, c in source_coords]
    raise AssertionError("marker must align with anchor in exactly one axis")


def _concat_copies(copies: list[np.ndarray], bg: int = BACKGROUND) -> np.ndarray:
    assert copies
    heights = {arr.shape[0] for arr in copies}
    assert len(heights) == 1
    if len(copies) == 1:
        return copies[0].copy()
    gap = np.full((copies[0].shape[0], 1), bg, dtype=int)
    parts: list[np.ndarray] = []
    for i, arr in enumerate(copies):
        if i:
            parts.append(gap)
        parts.append(arr)
    return np.concatenate(parts, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """Reflect a source object through singleton markers, crop each colored copy, and concatenate them by marker column.

    Input structure:
    - A black background contains one multi-pixel source object and 2-3 singleton markers.
    - The source is a single blue connected shape, while each marker has a distinct non-blue color.
    - Every marker shares exactly one row or exactly one column with exactly one source pixel, never both.

    Rule:
    - Identify the unique multi-pixel color as the source object and the singleton colors as markers.
    - For each marker, find the unique aligned source pixel; same-row markers define a vertical mirror axis and same-column markers define a horizontal mirror axis.
    - Reflect the entire source across that axis, recolor the reflected copy to the marker color, crop the copy to its tight bounding box, then sort copies by the marker's input column and join them left-to-right with one black separator column.

    Key insight:
    - One marker simultaneously encodes three roles: its alignment picks the reflection axis, its color picks the output color, and its column position picks the concatenation order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    coords_by_color = {color: [tuple(map(int, rc)) for rc in np.argwhere(g == color)] for color in colors}

    source_colors = [color for color, coords in coords_by_color.items() if len(coords) > 1]
    assert len(source_colors) == 1
    source_color = source_colors[0]
    source_coords = coords_by_color[source_color]
    source_rows = {r for r, _ in source_coords}
    source_cols = {c for _, c in source_coords}

    copies_with_order: list[tuple[int, np.ndarray]] = []
    for color, coords in coords_by_color.items():
        if color == source_color:
            continue
        assert len(coords) == 1
        marker = coords[0]
        mr, mc = marker

        row_matches = [(r, c) for r, c in source_coords if r == mr and c != mc]
        col_matches = [(r, c) for r, c in source_coords if c == mc and r != mr]
        assert (len(row_matches) == 1) != (len(col_matches) == 1)
        assert (mr not in source_rows) or (len(row_matches) == 1)
        assert (mc not in source_cols) or (len(col_matches) == 1)

        anchor = row_matches[0] if len(row_matches) == 1 else col_matches[0]
        reflected = _reflect_coords(source_coords, anchor, marker)
        copy_grid = _crop_coords(reflected, color, bg=bg)
        copies_with_order.append((mc, copy_grid))

    assert copies_with_order
    ordered_copies = [arr for _col, arr in sorted(copies_with_order, key=lambda item: item[0])]
    return _concat_copies(ordered_copies, bg=bg)


def _random_connected_shape() -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.choice(SOURCE_BBOX_HEIGHTS)
        w = random.choice(SOURCE_BBOX_WIDTHS)
        count = random.choice(SOURCE_PIXEL_COUNTS)
        if count >= h * w:
            continue

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = set(_neighbors4(start[0], start[1], h, w))
        while int(mask.sum()) < count and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            mask[cell] = True
            for nbr in _neighbors4(cell[0], cell[1], h, w):
                if not mask[nbr]:
                    frontier.add(nbr)

        if int(mask.sum()) != count:
            continue
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) != (h, w):
            continue
        if not _is_connected4(mask):
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue
        if np.all(mask):
            continue
        density = float(mask.mean())
        if not (0.35 <= density <= 0.75):
            continue
        row_counts = mask.sum(axis=1)
        col_counts = mask.sum(axis=0)
        if not np.any(row_counts == 1):
            continue
        if not np.any(col_counts == 1):
            continue
        return mask
    return None


def _marker_options(source_coords: list[tuple[int, int]]) -> list[dict[str, object]]:
    row_counts: dict[int, int] = {}
    col_counts: dict[int, int] = {}
    source_cols = {c for _, c in source_coords}
    source_rows = {r for r, _ in source_coords}
    source_set = set(source_coords)
    for r, c in source_coords:
        row_counts[r] = row_counts.get(r, 0) + 1
        col_counts[c] = col_counts.get(c, 0) + 1

    options: list[dict[str, object]] = []
    for anchor in source_coords:
        ar, ac = anchor
        if row_counts[ar] == 1:
            for side, distance in itertools.product((-1, 1), MARKER_DISTANCES):
                marker = (ar, ac + side * distance)
                if marker[1] in source_cols:
                    continue
                target = _reflect_coords(source_coords, anchor, marker)
                target_set = set(target)
                if not target_set.isdisjoint(source_set):
                    continue
                options.append(
                    {
                        "axis": "H",
                        "anchor": anchor,
                        "marker": marker,
                        "target": target,
                        "target_set": target_set,
                    }
                )
        if col_counts[ac] == 1:
            for side, distance in itertools.product((-1, 1), MARKER_DISTANCES):
                marker = (ar + side * distance, ac)
                if marker[0] in source_rows:
                    continue
                target = _reflect_coords(source_coords, anchor, marker)
                target_set = set(target)
                if not target_set.isdisjoint(source_set):
                    continue
                options.append(
                    {
                        "axis": "V",
                        "anchor": anchor,
                        "marker": marker,
                        "target": target,
                        "target_set": target_set,
                    }
                )
    return options


def _combo_ok(combo: tuple[dict[str, object], ...]) -> bool:
    marker_positions = [opt["marker"] for opt in combo]
    marker_cols = [mc for _mr, mc in marker_positions]
    if len(marker_cols) != len(set(marker_cols)):
        return False
    if len(marker_positions) != len(set(marker_positions)):
        return False

    for opt_a, opt_b in itertools.combinations(combo, 2):
        if not set(opt_a["target_set"]).isdisjoint(set(opt_b["target_set"])):
            return False
    return True


def _feasible_combos(options: list[dict[str, object]]) -> list[tuple[dict[str, object], ...]]:
    combos: list[tuple[dict[str, object], ...]] = []
    for count in MARKER_COUNTS:
        if len(options) < count:
            continue
        for combo in itertools.combinations(options, count):
            if _combo_ok(combo):
                combos.append(combo)
    return combos


def _choose_combo(combos: list[tuple[dict[str, object], ...]]) -> tuple[dict[str, object], ...]:
    categories: dict[str, list[tuple[dict[str, object], ...]]] = {
        "mixed": [],
        "horizontal_only": [],
        "vertical_only": [],
    }
    for combo in combos:
        axes = {str(opt["axis"]) for opt in combo}
        if axes == {"H", "V"}:
            categories["mixed"].append(combo)
        elif axes == {"H"}:
            categories["horizontal_only"].append(combo)
        elif axes == {"V"}:
            categories["vertical_only"].append(combo)

    weighted = [
        ("mixed", 4),
        ("horizontal_only", 1),
        ("vertical_only", 1),
    ]
    available = [(name, weight) for name, weight in weighted if categories[name]]
    assert available
    choices = [name for name, weight in available for _ in range(weight)]
    chosen_category = random.choice(choices)
    return random.choice(categories[chosen_category])


def _expected_output(combo: tuple[dict[str, object], ...], marker_colors: list[int]) -> np.ndarray:
    copies: list[tuple[int, np.ndarray]] = []
    for opt, color in zip(combo, marker_colors):
        copy_grid = _crop_coords(list(opt["target"]), color, bg=BACKGROUND)
        copies.append((int(opt["marker"][1]), copy_grid))
    return _concat_copies([arr for _col, arr in sorted(copies, key=lambda item: item[0])], bg=BACKGROUND)


def _place_design(
    source_coords: list[tuple[int, int]],
    combo: tuple[dict[str, object], ...],
) -> tuple[np.ndarray, list[int], tuple[int, int]] | None:
    occupied = set(source_coords)
    for opt in combo:
        occupied.update(set(opt["target_set"]))

    rs = [r for r, _ in occupied]
    cs = [c for _, c in occupied]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    union_h = max_r - min_r + 1
    union_w = max_c - min_c + 1

    min_h = max(GRID_HEIGHT_RANGE[0], union_h + 1)
    max_h = min(GRID_HEIGHT_RANGE[1], union_h + 7)
    min_w = max(GRID_WIDTH_RANGE[0], union_w + 2)
    max_w = min(GRID_WIDTH_RANGE[1], union_w + 9)
    if min_h > max_h or min_w > max_w:
        return None

    height = random.randint(min_h, max_h)
    width = random.randint(min_w, max_w)
    r_low, r_high = -min_r, height - 1 - max_r
    c_low, c_high = -min_c, width - 1 - max_c
    if r_low > r_high or c_low > c_high:
        return None

    r_shift = random.randint(r_low, r_high)
    preferred_cols = [c for c in range(c_low, c_high + 1) if c <= width // 3]
    c_shift = random.choice(preferred_cols if preferred_cols else list(range(c_low, c_high + 1)))

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for r, c in source_coords:
        grid[r + r_shift, c + c_shift] = SOURCE_COLOR

    colors = random.sample(list(MARKER_COLORS), len(combo))
    for opt, color in zip(combo, colors):
        mr, mc = opt["marker"]
        grid[mr + r_shift, mc + c_shift] = color

    return grid, colors, (r_shift, c_shift)


def _build_input() -> np.ndarray | None:
    shape = _random_connected_shape()
    if shape is None:
        return None

    source_coords = [tuple(map(int, rc)) for rc in np.argwhere(shape)]
    options = _marker_options(source_coords)
    if not options:
        return None
    combos = _feasible_combos(options)
    if not combos:
        return None
    combo = _choose_combo(combos)

    placed = _place_design(source_coords, combo)
    if placed is None:
        return None
    grid, colors, shift = placed
    r_shift, c_shift = shift

    translated_combo: list[dict[str, object]] = []
    for opt in combo:
        translated_combo.append(
            {
                "axis": opt["axis"],
                "anchor": (opt["anchor"][0] + r_shift, opt["anchor"][1] + c_shift),
                "marker": (opt["marker"][0] + r_shift, opt["marker"][1] + c_shift),
                "target": [(r + r_shift, c + c_shift) for r, c in opt["target"]],
                "target_set": {(r + r_shift, c + c_shift) for r, c in opt["target_set"]},
            }
        )

    expected = _expected_output(tuple(translated_combo), colors)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
