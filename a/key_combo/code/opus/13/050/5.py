from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
CONTROL_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (13, 18)
PAIR_COUNT_CHOICES = (3, 4)
EXPANDED_SIZE_RANGE = (4, 7)
MAX_ATTEMPTS = 500

OVERLAP_MODE_ODD_EVEN = "odd_even"
OVERLAP_MODE_SAME_PARITY = "same_parity"


def _legend_pairs(grid: np.ndarray) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        left = int(grid[r, 0])
        right = int(grid[r, 1])
        if left == BACKGROUND or right == BACKGROUND:
            break
        pairs.append((left, right))
    assert pairs, "expected a non-empty legend"
    return pairs


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _expanded_bbox(coords: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h, w = shape
    return max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r1 + 1), min(w - 1, c1 + 1)


def _rect_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _intersection(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0 = max(box_a[0], box_b[0])
    c0 = max(box_a[1], box_b[1])
    r1 = min(box_a[2], box_b[2])
    c1 = min(box_a[3], box_b[3])
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _rect_cells(box: tuple[int, int, int, int]) -> Iterable[tuple[int, int]]:
    r0, c0, r1, c1 = box
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            yield r, c


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _priority_key(group: dict) -> tuple[int, int, int]:
    # Low priority first: even before odd, and within parity larger distance first.
    return int(group["distance"] % 2), -int(group["distance"]), -int(group["legend_index"])


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2, "expected a 2D grid with at least two columns"

    pairs = _legend_pairs(g)
    source_colors = [src for src, _fill in pairs]
    fill_colors = [fill for _src, fill in pairs]
    assert len(set(source_colors)) == len(source_colors), "legend source colors must be unique"
    assert len(set(fill_colors)) == len(fill_colors), "legend fill colors must be unique"
    assert CONTROL_COLOR not in source_colors and CONTROL_COLOR not in fill_colors, "grey is reserved for the control pixel"

    legend_rows = len(pairs)
    legend_mask = np.zeros(g.shape, dtype=bool)
    legend_mask[:legend_rows, :2] = True

    control_coords = np.argwhere((g == CONTROL_COLOR) & ~legend_mask)
    assert control_coords.shape[0] == 1, "expected exactly one control pixel"
    control = (int(control_coords[0, 0]), int(control_coords[0, 1]))

    foreground_mask = (g != BACKGROUND) & ~legend_mask
    non_control_mask = foreground_mask & (g != CONTROL_COLOR)
    allowed_colors = set(source_colors)
    assert set(map(int, g[non_control_mask])).issubset(allowed_colors), "non-legend foreground must be source colors or the control"

    groups: list[dict] = []
    dot_mask = np.zeros(g.shape, dtype=bool)
    for legend_index, (source, fill) in enumerate(pairs):
        coords = np.argwhere((g == source) & non_control_mask)
        assert coords.shape[0] >= 2, "each source color needs at least two scatter points"
        centroid = (int(np.floor(coords[:, 0].mean())), int(np.floor(coords[:, 1].mean())))
        box = _expanded_bbox(coords, g.shape)
        distance = _chebyshev(centroid, control)
        groups.append(
            {
                "legend_index": legend_index,
                "source": int(source),
                "fill": int(fill),
                "coords": coords.astype(int),
                "coord_set": {tuple(map(int, pt)) for pt in coords},
                "centroid": centroid,
                "distance": int(distance),
                "box": box,
            }
        )
        dot_mask[coords[:, 0], coords[:, 1]] = True

    distances = [int(group["distance"]) for group in groups]
    assert len(set(distances)) == len(distances), "control distances must be unique"
    return {"grid": g, "pairs": pairs, "legend_mask": legend_mask, "control": control, "groups": groups, "dot_mask": dot_mask}


def _render_scene(scene: dict, ordered_groups: list[dict]) -> np.ndarray:
    out = np.full(scene["grid"].shape, BACKGROUND, dtype=int)
    for group in ordered_groups:
        r0, c0, r1, c1 = group["box"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(group["fill"])
    for group in scene["groups"]:
        coords = group["coords"]
        out[coords[:, 0], coords[:, 1]] = int(group["source"])
    control_r, control_c = scene["control"]
    out[control_r, control_c] = CONTROL_COLOR
    return out


def _actual_order(scene: dict) -> list[dict]:
    return sorted(scene["groups"], key=_priority_key)


def _legend_order(scene: dict) -> list[dict]:
    return sorted(scene["groups"], key=lambda group: -int(group["legend_index"]))


def _distance_only_order(scene: dict) -> list[dict]:
    return sorted(scene["groups"], key=lambda group: (-int(group["distance"]), -int(group["legend_index"])))


def _parity_then_legend_order(scene: dict) -> list[dict]:
    return sorted(scene["groups"], key=lambda group: (int(group["distance"] % 2), -int(group["legend_index"])))


def main(grid: np.ndarray) -> np.ndarray:
    """Expand legend-colored scatter groups into filled rectangles, breaking overlaps by control-pixel distance priority.

    Input structure:
    - The grid has a black background, a two-column legend at the upper left, several scatter-point groups in the legend source colors, and one grey control pixel outside the legend.
    - Each legend row maps one source color in column 0 to one fill color in column 1.
    - Every non-legend foreground pixel is either a scatter point in one source color or the lone grey control pixel.

    Rule:
    - Read the legend and clear the legend area.
    - For each source color, take the bbox of its non-legend scatter points and expand that bbox by one cell in every direction, clipped to the grid.
    - Fill the expanded rectangle with the legend-paired fill color while preserving the original scatter points.
    - If expanded rectangles overlap, prioritize the group whose scatter-point centroid has odd Chebyshev distance to the grey control over any even-distance group; within the same parity, the smaller Chebyshev distance wins.
    - Keep the grey control pixel unchanged.

    Key insight:
    - The legend only supplies source-to-fill colors; overlap priority comes from the control-guided odd/even Chebyshev distance of each scatter group centroid, not from legend order.
    """
    scene = _parse_scene(grid)
    return _render_scene(scene, _actual_order(scene))


def _random_rectangle(height: int, width: int, legend_rows: int) -> tuple[int, int, int, int] | None:
    legend_box = (0, 0, legend_rows - 1, 1)
    min_size, max_size = EXPANDED_SIZE_RANGE
    for _ in range(80):
        rect_h = random.randint(min_size, max_size)
        rect_w = random.randint(min_size, max_size)
        r0 = random.randint(0, height - rect_h)
        c0 = random.randint(0, width - rect_w)
        rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if _intersection(rect, legend_box) is None:
            return rect
    return None


def _point_pattern_variants(box: tuple[int, int, int, int]) -> list[np.ndarray]:
    r0, c0, r1, c1 = box
    top = r0 + 1
    left = c0 + 1
    bottom = r1 - 1
    right = c1 - 1
    mid_r = (top + bottom) // 2
    mid_c = (left + right) // 2
    points = {
        "tl": (top, left),
        "tr": (top, right),
        "bl": (bottom, left),
        "br": (bottom, right),
        "tm": (top, mid_c),
        "bm": (bottom, mid_c),
        "lm": (mid_r, left),
        "rm": (mid_r, right),
    }

    templates = [
        ("tl", "br"),
        ("tr", "bl"),
        ("tl", "tr", "br"),
        ("tl", "bl", "br"),
        ("tl", "tr", "bl"),
        ("tr", "bl", "br"),
        ("tl", "tr", "bl", "br"),
        ("tl", "tm", "br"),
        ("tr", "tm", "bl"),
        ("tl", "rm", "bl"),
        ("tr", "lm", "br"),
        ("tl", "bm", "tr"),
        ("bl", "tm", "br"),
    ]

    out: list[np.ndarray] = []
    for template in templates:
        coords = [points[name] for name in template]
        if len(set(coords)) < len(coords):
            continue
        arr = np.asarray(coords, dtype=int)
        if _bbox_from_coords(arr) != (top, left, bottom, right):
            continue
        out.append(arr)
    assert out, "each rectangle should yield at least one valid scatter pattern"
    return out


def _choose_rectangles(height: int, width: int, legend_rows: int, pair_count: int) -> list[tuple[int, int, int, int]] | None:
    chosen: list[tuple[int, int, int, int]] | None = None
    for _ in range(120):
        primary_a = _random_rectangle(height, width, legend_rows)
        primary_b = _random_rectangle(height, width, legend_rows)
        if primary_a is None or primary_b is None or primary_a == primary_b:
            continue
        inter = _intersection(primary_a, primary_b)
        if inter is None:
            continue
        area = _rect_area(inter)
        if 2 <= area <= 9:
            chosen = [primary_a, primary_b]
            break
    if chosen is None:
        return None

    total_area = _rect_area(primary_a) + _rect_area(primary_b)

    for _ in range(pair_count - 2):
        prefer_overlap = random.random() < 0.45
        rect = None
        for _ in range(160):
            candidate = _random_rectangle(height, width, legend_rows)
            if candidate is None or candidate in chosen:
                continue
            areas = [_rect_area(inter) for other in chosen if (inter := _intersection(candidate, other)) is not None]
            unique_cells = _rect_area(candidate) - sum(areas)
            if unique_cells < 6:
                continue
            if total_area + _rect_area(candidate) > int(height * width * 0.55):
                continue
            if prefer_overlap:
                if not areas or max(areas) > 6:
                    continue
            else:
                if areas:
                    continue
            rect = candidate
            break
        if rect is None:
            for _ in range(160):
                candidate = _random_rectangle(height, width, legend_rows)
                if candidate is None or candidate in chosen:
                    continue
                areas = [_rect_area(inter) for other in chosen if (inter := _intersection(candidate, other)) is not None]
                unique_cells = _rect_area(candidate) - sum(areas)
                if unique_cells < 6:
                    continue
                if total_area + _rect_area(candidate) > int(height * width * 0.55):
                    continue
                rect = candidate
                break
        if rect is None:
            return None
        chosen.append(rect)
        total_area += _rect_area(rect)

    return chosen


def _choose_points(rectangles: list[tuple[int, int, int, int]]) -> list[np.ndarray] | None:
    variants = [_point_pattern_variants(rect) for rect in rectangles]
    order = list(range(len(rectangles)))
    order.sort(key=lambda idx: len(variants[idx]))
    chosen: list[np.ndarray | None] = [None] * len(rectangles)
    occupied: set[tuple[int, int]] = set()

    def search(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        options = variants[idx][:]
        random.shuffle(options)
        for coords in options:
            coord_set = {tuple(map(int, pt)) for pt in coords}
            if occupied & coord_set:
                continue
            chosen[idx] = coords
            occupied.update(coord_set)
            if search(pos + 1):
                return True
            occupied.difference_update(coord_set)
            chosen[idx] = None
        return False

    if not search(0):
        return None
    return [coords for coords in chosen if coords is not None]


def _visible_overlap_pairs(rectangles: list[tuple[int, int, int, int]], point_sets: list[np.ndarray]) -> list[tuple[int, int]]:
    dot_cells = set()
    for coords in point_sets:
        dot_cells.update(tuple(map(int, pt)) for pt in coords)

    visible: list[tuple[int, int]] = []
    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
            inter = _intersection(rectangles[i], rectangles[j])
            if inter is None:
                continue
            overlap_cells = set(_rect_cells(inter)) - dot_cells
            if overlap_cells:
                visible.append((i, j))
    return visible


def _control_candidates(
    height: int,
    width: int,
    legend_rows: int,
    rectangles: list[tuple[int, int, int, int]],
    point_sets: list[np.ndarray],
) -> list[tuple[int, int]]:
    dot_cells = set()
    for coords in point_sets:
        dot_cells.update(tuple(map(int, pt)) for pt in coords)

    blocked = set(dot_cells)
    for r, c in list(dot_cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    blocked.add((rr, cc))
    for r in range(legend_rows):
        blocked.add((r, 0))
        blocked.add((r, 1))
    for box in rectangles:
        blocked.update(_rect_cells(box))

    candidates = [(r, c) for r in range(height) for c in range(width) if (r, c) not in blocked]
    random.shuffle(candidates)
    return candidates


def _group_data_from_geometry(
    rectangles: list[tuple[int, int, int, int]],
    point_sets: list[np.ndarray],
    control: tuple[int, int],
) -> list[dict]:
    groups: list[dict] = []
    for idx, (box, coords) in enumerate(zip(rectangles, point_sets)):
        centroid = (int(np.floor(coords[:, 0].mean())), int(np.floor(coords[:, 1].mean())))
        groups.append(
            {
                "geom_index": idx,
                "box": box,
                "coords": coords.astype(int),
                "centroid": centroid,
                "distance": _chebyshev(centroid, control),
                "dot_count": int(coords.shape[0]),
                "area": _rect_area(box),
            }
        )
    return groups


def _priority_pair(groups: list[dict], i: int, j: int) -> tuple[int, int]:
    a = groups[i]
    b = groups[j]
    if _priority_key({"distance": a["distance"], "legend_index": 0}) > _priority_key({"distance": b["distance"], "legend_index": 0}):
        return i, j
    return j, i


def _choose_control_and_mode(
    height: int,
    width: int,
    legend_rows: int,
    rectangles: list[tuple[int, int, int, int]],
    point_sets: list[np.ndarray],
) -> tuple[tuple[int, int], str, tuple[int, int]] | None:
    visible_pairs = _visible_overlap_pairs(rectangles, point_sets)
    if not visible_pairs:
        return None

    for mode in random.sample([OVERLAP_MODE_ODD_EVEN, OVERLAP_MODE_SAME_PARITY], 2):
        for control in _control_candidates(height, width, legend_rows, rectangles, point_sets):
            groups = _group_data_from_geometry(rectangles, point_sets, control)
            distances = [group["distance"] for group in groups]
            if len(set(distances)) != len(distances):
                continue
            for i, j in visible_pairs:
                winner, loser = _priority_pair(groups, i, j)
                d_win = groups[winner]["distance"]
                d_lose = groups[loser]["distance"]
                same_parity = (d_win % 2) == (d_lose % 2)
                if mode == OVERLAP_MODE_ODD_EVEN:
                    if d_win % 2 != 1 or d_lose % 2 != 0:
                        continue
                    if d_win <= d_lose:
                        continue
                else:
                    if not same_parity:
                        continue
                    if d_win >= d_lose:
                        continue
                return control, mode, (winner, loser)
    return None


def _legend_order_indices(pair_count: int, conflict_pair: tuple[int, int]) -> list[int]:
    winner, loser = conflict_pair
    remaining = [idx for idx in range(pair_count) if idx not in conflict_pair]
    random.shuffle(remaining)
    slots = [loser] + remaining + [winner]
    return slots


def _build_input_from_scene(
    height: int,
    width: int,
    legend_order: list[int],
    point_sets: list[np.ndarray],
    control: tuple[int, int],
) -> np.ndarray:
    colors = random.sample(PALETTE, len(point_sets) * 2)
    sources = colors[: len(point_sets)]
    fills = colors[len(point_sets) :]

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for legend_row, geom_index in enumerate(legend_order):
        grid[legend_row, 0] = sources[geom_index]
        grid[legend_row, 1] = fills[geom_index]
    for geom_index, coords in enumerate(point_sets):
        source = sources[geom_index]
        grid[coords[:, 0], coords[:, 1]] = source
    grid[control] = CONTROL_COLOR
    return grid


def _validate_generated(grid: np.ndarray, mode: str) -> bool:
    try:
        scene = _parse_scene(grid)
    except Exception:
        return False

    if not any(_intersection(scene["groups"][i]["box"], scene["groups"][j]["box"]) is not None for i in range(len(scene["groups"])) for j in range(i + 1, len(scene["groups"]))):
        return False

    control = scene["control"]
    for group in scene["groups"]:
        for r, c in group["coord_set"]:
            if max(abs(r - control[0]), abs(c - control[1])) <= 1:
                return False

    actual = _render_scene(scene, _actual_order(scene))
    if not np.array_equal(main(grid), actual):
        return False

    legend_rule = _render_scene(scene, _legend_order(scene))
    if np.array_equal(actual, legend_rule):
        return False

    if mode == OVERLAP_MODE_ODD_EVEN:
        distance_only = _render_scene(scene, _distance_only_order(scene))
        if np.array_equal(actual, distance_only):
            return False
    else:
        parity_then_legend = _render_scene(scene, _parity_then_legend_order(scene))
        if np.array_equal(actual, parity_then_legend):
            return False

    distinct_outputs = set(map(int, np.unique(actual)))
    if len(distinct_outputs) < 5:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    pair_count = random.choice(PAIR_COUNT_CHOICES)

    rectangles = _choose_rectangles(height, width, pair_count, pair_count)
    if rectangles is None:
        return None

    point_sets = _choose_points(rectangles)
    if point_sets is None:
        return None

    control_choice = _choose_control_and_mode(height, width, pair_count, rectangles, point_sets)
    if control_choice is None:
        return None
    control, mode, conflict_pair = control_choice

    legend_order = _legend_order_indices(pair_count, conflict_pair)
    grid = _build_input_from_scene(height, width, legend_order, point_sets, control)
    if _validate_generated(grid, mode):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
