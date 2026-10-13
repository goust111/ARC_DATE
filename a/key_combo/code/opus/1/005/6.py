from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4)
CLUSTER_COLORS = (6, 7, 8)

SIDE_ORDER = ("top", "right", "bottom", "left")
SIDE_TO_STEPS = {side: idx for idx, side in enumerate(SIDE_ORDER)}

# Only keep rectangle sizes whose side midpoint is unambiguous and whose
# prescribed rotation stays on the integer lattice.
RECT_LIBRARY = {
    6: {"h": 4, "w": 5, "sides": ("top", "bottom")},
    9: {"h": 5, "w": 5, "sides": ("top", "right", "bottom", "left")},
    10: {"h": 4, "w": 7, "sides": ("top", "bottom")},
    12: {"h": 6, "w": 5, "sides": ("top", "bottom")},
}
MATCHABLE_AREAS = tuple(sorted(RECT_LIBRARY))
CLUSTER_SIZE_CHOICES = tuple(range(3, 13))

GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
GRID_MARGIN = 1
MIN_GAP = 1
MAX_GENERATION_ATTEMPTS = 600


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == BACKGROUND:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != BACKGROUND:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _side_midpoint(bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if side == "top":
        assert w % 2 == 1
        return r0, c0 + w // 2
    if side == "right":
        assert h % 2 == 1
        return r0 + h // 2, c1
    if side == "bottom":
        assert w % 2 == 1
        return r1, c0 + w // 2
    assert side == "left"
    assert h % 2 == 1
    return r0 + h // 2, c0


def _marker_side(
    bbox: tuple[int, int, int, int],
    marker: tuple[int, int],
) -> str | None:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    options: list[str] = []

    if w % 2 == 1:
        mid_c = c0 + w // 2
        if marker == (r0, mid_c):
            options.append("top")
        if marker == (r1, mid_c):
            options.append("bottom")
    if h % 2 == 1:
        mid_r = r0 + h // 2
        if marker == (mid_r, c1):
            options.append("right")
        if marker == (mid_r, c0):
            options.append("left")

    if len(options) != 1:
        return None
    return options[0]


def _rotated_bbox(
    bbox: tuple[int, int, int, int],
    steps: int,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    new_h, new_w = (h, w) if steps % 2 == 0 else (w, h)

    row_num = (r0 + r1) - (new_h - 1)
    col_num = (c0 + c1) - (new_w - 1)
    assert row_num % 2 == 0 and col_num % 2 == 0, "rotation leaves the integer lattice"

    new_r0 = row_num // 2
    new_c0 = col_num // 2
    return new_r0, new_c0, new_r0 + new_h - 1, new_c0 + new_w - 1


def _paint_rectangle(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    marker_color: int,
    marker_side: str,
    fill_color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill_color
    mr, mc = _side_midpoint(bbox, marker_side)
    grid[mr, mc] = marker_color


def _rectangle_from_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object] | None:
    colors = {int(grid[r, c]) for r, c in coords}
    if GREY not in colors or not colors <= {GREY, *MARKER_COLORS}:
        return None

    r0, c0, r1, c1 = bbox = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4 or len(coords) != h * w:
        return None

    marker_cells = [(r, c) for r, c in coords if int(grid[r, c]) != GREY]
    if len(marker_cells) != 1:
        return None

    marker = marker_cells[0]
    side = _marker_side(bbox, marker)
    if side is None:
        return None

    steps = SIDE_TO_STEPS[side]
    if steps % 2 == 1 and (h % 2) != (w % 2):
        return None

    return {
        "bbox": bbox,
        "h": h,
        "w": w,
        "coords": coords,
        "side": side,
        "rot_steps": steps,
        "marker_color": int(grid[marker]),
        "match_size": (h - 2) * (w - 2),
    }


def _cluster_from_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object] | None:
    colors = {int(grid[r, c]) for r, c in coords}
    if len(colors) != 1:
        return None
    color = colors.pop()
    if color not in CLUSTER_COLORS:
        return None
    return {
        "coords": coords,
        "color": color,
        "size": len(coords),
        "bbox": _bbox(coords),
    }


def _point_center(coords: list[tuple[int, int]]) -> tuple[float, float]:
    n = float(len(coords))
    return (
        sum(r for r, _ in coords) / n,
        sum(c for _, c in coords) / n,
    )


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def _dist_sq(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each grey rectangle to an equal-size scatter cluster, rotate it by its border marker side, and recolor only its interior.

    Input structure:
    - Black background contains 2-3 filled grey rectangles, each with exactly one border cell replaced by a marker color.
    - Marker colors come from {1,2,3,4}; the marker sits at the midpoint of one rectangle side.
    - Separate 4-connected monochrome scatter clusters use colors {6,7,8}; a rectangle's interior area matches exactly one cluster size.

    Rule:
    - Detect every filled rectangle-with-marker component and compute its interior area as (height-2)*(width-2).
    - Pair each rectangle with the unique scatter cluster whose pixel count equals that interior area.
    - Rotate the whole rectangle about its own center by 0/90/180/270 clockwise steps according to whether the marker starts on the top/right/bottom/left side.
    - In the rotated rectangle, keep the grey border grey, keep the marker color on its rotated border cell, recolor only the interior to the matched cluster's color, and erase all matched clusters to black.

    Key insight:
    - The rectangle area chooses the source cluster, the marker side chooses the rotation, and the border-vs-interior split chooses exactly which cells take the matched color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles: list[dict[str, object]] = []
    clusters: list[dict[str, object]] = []

    for component in _find_components(g):
        rect = _rectangle_from_component(g, component)
        if rect is not None:
            rectangles.append(rect)
            continue
        cluster = _cluster_from_component(g, component)
        if cluster is not None:
            clusters.append(cluster)
            continue
        raise AssertionError("unexpected foreground component")

    assert 2 <= len(rectangles) <= 3, "expected 2-3 rectangles"
    assert 2 <= len(clusters) <= 3, "expected 2-3 clusters"

    rect_sizes = [int(rect["match_size"]) for rect in rectangles]
    cluster_sizes = [int(cluster["size"]) for cluster in clusters]
    assert len(set(rect_sizes)) == len(rectangles), "rectangle interior areas must be unique"
    assert len(set(cluster_sizes)) == len(clusters), "cluster sizes must be unique"

    size_to_cluster = {int(cluster["size"]): cluster for cluster in clusters}
    for size in rect_sizes:
        assert size in size_to_cluster, "every rectangle must have a matching cluster"

    out = np.full_like(g, BACKGROUND)
    matched_sizes = set(rect_sizes)

    for cluster in clusters:
        if int(cluster["size"]) in matched_sizes:
            continue
        for r, c in cluster["coords"]:
            out[r, c] = int(cluster["color"])

    for rect in rectangles:
        steps = int(rect["rot_steps"])
        old_side = str(rect["side"])
        new_bbox = _rotated_bbox(rect["bbox"], steps)
        side_idx = SIDE_ORDER.index(old_side)
        new_side = SIDE_ORDER[(side_idx + steps) % 4]

        fill_color = int(size_to_cluster[int(rect["match_size"])]["color"])
        _paint_rectangle(
            out,
            new_bbox,
            marker_color=int(rect["marker_color"]),
            marker_side=new_side,
            fill_color=fill_color,
        )

    return out


def _boxes_separated(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = MIN_GAP,
) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _shape_bbox(shape: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _transform_shape(shape: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = _shape_bbox(shape)
    mask = np.zeros((h, w), dtype=int)
    for r, c in shape:
        mask[r, c] = 1
    mask = np.rot90(mask, -random.randrange(4))
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask == 1)]
    coords.sort()
    return coords


def _random_connected_shape(size: int) -> list[tuple[int, int]]:
    canvas = 7
    for _ in range(200):
        start = (random.randrange(2, 5), random.randrange(2, 5))
        cells = {start}
        frontier = {
            nbr
            for nbr in _neighbors4(*start)
            if 0 <= nbr[0] < canvas and 0 <= nbr[1] < canvas
        }

        while len(cells) < size:
            choices = list(frontier)
            weights = [
                1 + sum((adj in cells) for adj in _neighbors4(r, c))
                for r, c in choices
            ]
            cell = random.choices(choices, weights=weights, k=1)[0]
            frontier.remove(cell)
            cells.add(cell)
            for nbr in _neighbors4(*cell):
                if 0 <= nbr[0] < canvas and 0 <= nbr[1] < canvas and nbr not in cells:
                    frontier.add(nbr)

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, c0 = min(rows), min(cols)
        shape = sorted((r - r0, c - c0) for r, c in cells)
        shape = _transform_shape(shape)
        h, w = _shape_bbox(shape)
        if size >= 5 and h * w == size and random.random() < 0.75:
            continue
        return shape
    raise RuntimeError("failed to sample a cluster shape")


def _all_positions(h: int, w: int, grid_h: int, grid_w: int) -> list[tuple[int, int]]:
    r_stop = grid_h - GRID_MARGIN - h
    c_stop = grid_w - GRID_MARGIN - w
    if r_stop < GRID_MARGIN or c_stop < GRID_MARGIN:
        return []
    return [
        (r, c)
        for r in range(GRID_MARGIN, r_stop + 1)
        for c in range(GRID_MARGIN, c_stop + 1)
    ]


def _place_rectangles(
    rects: list[dict[str, object]],
    grid_h: int,
    grid_w: int,
) -> list[dict[str, object]] | None:
    for _ in range(200):
        placed: dict[int, dict[str, object]] = {}
        order = rects[:]
        random.shuffle(order)
        ok = True

        for rect in order:
            candidates = _all_positions(int(rect["h"]), int(rect["w"]), grid_h, grid_w)
            random.shuffle(candidates)
            found_bbox = None
            found_out_bbox = None

            for r0, c0 in candidates:
                bbox = (r0, c0, r0 + int(rect["h"]) - 1, c0 + int(rect["w"]) - 1)
                out_bbox = _rotated_bbox(bbox, int(rect["rot_steps"]))
                if not (0 <= out_bbox[0] and 0 <= out_bbox[1] and out_bbox[2] < grid_h and out_bbox[3] < grid_w):
                    continue
                if not all(
                    _boxes_separated(bbox, other["bbox"]) and _boxes_separated(out_bbox, other["out_bbox"])
                    for other in placed.values()
                ):
                    continue
                found_bbox = bbox
                found_out_bbox = out_bbox
                break

            if found_bbox is None or found_out_bbox is None:
                ok = False
                break

            new_rect = dict(rect)
            new_rect["bbox"] = found_bbox
            new_rect["out_bbox"] = found_out_bbox
            placed[int(rect["match_size"])] = new_rect

        if ok and len(placed) == len(rects):
            return [placed[int(rect["match_size"])] for rect in rects]
    return None


def _place_clusters(
    clusters: list[dict[str, object]],
    grid_h: int,
    grid_w: int,
    rects: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    blocked = [rect["bbox"] for rect in rects] + [rect["out_bbox"] for rect in rects]

    for _ in range(250):
        placed: dict[int, dict[str, object]] = {}
        order = clusters[:]
        random.shuffle(order)
        occupied = blocked[:]
        ok = True

        for cluster in order:
            h, w = _shape_bbox(cluster["shape"])
            candidates = _all_positions(h, w, grid_h, grid_w)
            random.shuffle(candidates)
            found = None
            found_bbox = None

            for r0, c0 in candidates:
                bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                if not all(_boxes_separated(bbox, other) for other in occupied):
                    continue
                coords = [(r0 + dr, c0 + dc) for dr, dc in cluster["shape"]]
                found = coords
                found_bbox = bbox
                break

            if found is None or found_bbox is None:
                ok = False
                break

            new_cluster = dict(cluster)
            new_cluster["coords"] = found
            new_cluster["bbox"] = found_bbox
            placed[int(cluster["size"])] = new_cluster
            occupied.append(found_bbox)

        if ok and len(placed) == len(clusters):
            return [placed[int(cluster["size"])] for cluster in clusters]
    return None


def _render_input(
    grid_h: int,
    grid_w: int,
    rects: list[dict[str, object]],
    clusters: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)

    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
        mr, mc = _side_midpoint(rect["bbox"], str(rect["side"]))
        grid[mr, mc] = int(rect["marker_color"])

    for cluster in clusters:
        for r, c in cluster["coords"]:
            grid[r, c] = int(cluster["color"])
    return grid


def _render_expected(
    grid_h: int,
    grid_w: int,
    rects: list[dict[str, object]],
    clusters: list[dict[str, object]],
) -> np.ndarray:
    out = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    size_to_cluster = {int(cluster["size"]): cluster for cluster in clusters}
    matched_sizes = {int(rect["match_size"]) for rect in rects}

    for cluster in clusters:
        if int(cluster["size"]) in matched_sizes:
            continue
        for r, c in cluster["coords"]:
            out[r, c] = int(cluster["color"])

    for rect in rects:
        old_side = str(rect["side"])
        steps = int(rect["rot_steps"])
        side_idx = SIDE_ORDER.index(old_side)
        new_side = SIDE_ORDER[(side_idx + steps) % 4]
        fill_color = int(size_to_cluster[int(rect["match_size"])]["color"])
        _paint_rectangle(
            out,
            rect["out_bbox"],
            marker_color=int(rect["marker_color"]),
            marker_side=new_side,
            fill_color=fill_color,
        )

    return out


def _choose_rectangles() -> list[dict[str, object]]:
    n_rects = random.choice((2, 3))
    while True:
        areas = random.sample(MATCHABLE_AREAS, n_rects)
        if 9 in areas or random.random() < 0.35:
            break
    random.shuffle(areas)

    marker_colors = random.sample(MARKER_COLORS, n_rects)
    rects: list[dict[str, object]] = []
    use_quarter_turn = 9 in areas and random.random() < 0.7

    for area, marker_color in zip(areas, marker_colors):
        spec = RECT_LIBRARY[area]
        if area == 9 and use_quarter_turn:
            side = random.choice(("right", "left"))
        else:
            side = random.choice(spec["sides"])
        rects.append(
            {
                "match_size": area,
                "h": spec["h"],
                "w": spec["w"],
                "side": side,
                "rot_steps": SIDE_TO_STEPS[side],
                "marker_color": marker_color,
            }
        )

    if not any(str(rect["side"]) != "top" for rect in rects):
        for rect in rects:
            if int(rect["match_size"]) == 9 and random.random() < 0.5:
                rect["side"] = random.choice(("right", "left"))
                rect["rot_steps"] = SIDE_TO_STEPS[str(rect["side"])]
                break
        else:
            chosen = random.choice(rects)
            chosen["side"] = "bottom"
            chosen["rot_steps"] = SIDE_TO_STEPS["bottom"]

    if len(rects) > 1 and len({str(rect["side"]) for rect in rects}) == 1:
        rects[0]["side"] = "bottom" if str(rects[0]["side"]) == "top" else "top"
        rects[0]["rot_steps"] = SIDE_TO_STEPS[str(rects[0]["side"])]

    return rects


def _choose_clusters(rects: list[dict[str, object]]) -> list[dict[str, object]]:
    matched_sizes = [int(rect["match_size"]) for rect in rects]
    n_clusters = len(rects)
    if len(rects) == 2 and random.random() < 0.6:
        n_clusters += 1

    if n_clusters == len(rects):
        cluster_sizes = matched_sizes[:]
    else:
        lo, hi = min(matched_sizes), max(matched_sizes)
        between = [size for size in CLUSTER_SIZE_CHOICES if lo < size < hi and size not in matched_sizes]
        extras = between or [size for size in CLUSTER_SIZE_CHOICES if size not in matched_sizes]
        extra_size = random.choice(extras)
        cluster_sizes = matched_sizes + [extra_size]
    random.shuffle(cluster_sizes)

    cluster_colors = random.sample(CLUSTER_COLORS, n_clusters)
    clusters: list[dict[str, object]] = []
    for size, color in zip(cluster_sizes, cluster_colors):
        clusters.append(
            {
                "size": size,
                "color": color,
                "shape": _random_connected_shape(size),
            }
        )
    return clusters


def _layout_is_semantic(
    rects: list[dict[str, object]],
    clusters: list[dict[str, object]],
) -> bool:
    if not any(str(rect["side"]) in {"bottom", "left", "right"} for rect in rects):
        return False

    size_to_cluster = {int(cluster["size"]): cluster for cluster in clusters}
    mismatch_exists = False
    for rect in rects:
        rect_center = _bbox_center(rect["bbox"])
        matched_center = _point_center(size_to_cluster[int(rect["match_size"])]["coords"])
        d_match = _dist_sq(rect_center, matched_center)
        other_distances = [
            _dist_sq(rect_center, _point_center(cluster["coords"]))
            for cluster in clusters
        ]
        if any(distance + 1e-9 < d_match for distance in other_distances):
            mismatch_exists = True
            break
    if not mismatch_exists:
        return False

    extra_sizes = sorted(set(int(cluster["size"]) for cluster in clusters) - {int(rect["match_size"]) for rect in rects})
    if extra_sizes:
        assert len(extra_sizes) == 1
        extra = extra_sizes[0]
        matched = sorted(int(rect["match_size"]) for rect in rects)
        if not matched[0] < extra < matched[-1]:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)

    rects = _choose_rectangles()
    rects = _place_rectangles(rects, grid_h, grid_w)
    if rects is None:
        return None

    clusters = _choose_clusters(rects)
    clusters = _place_clusters(clusters, grid_h, grid_w, rects)
    if clusters is None:
        return None

    if not _layout_is_semantic(rects, clusters):
        return None

    inp = _render_input(grid_h, grid_w, rects, clusters)
    expected = _render_expected(grid_h, grid_w, rects, clusters)
    if np.array_equal(inp, expected):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
