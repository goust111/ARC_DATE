from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
TEAL = 8

MARKER_TO_OUTLINE = {
    GREEN: PINK,
    TEAL: YELLOW,
    RED: BLUE,
}
MARKER_COLORS = tuple(MARKER_TO_OUTLINE)

GRID_MIN = 13
GRID_MAX = 18
COUNT_CHOICES = (3, 4, 5)
DISTANCE_VALUES = (2, 3, 4, 5)
MODE_WEIGHTS = (
    ("no_overlap", 3),
    ("same_distance_overlap", 4),
    ("different_distance_overlap", 4),
)
MAX_GENERATION_ATTEMPTS = 400


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _ring_cells(center: tuple[int, int], distance: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    h, w = shape
    cells: set[tuple[int, int]] = set()
    for rr in range(max(0, r - distance), min(h - 1, r + distance) + 1):
        for cc in range(max(0, c - distance), min(w - 1, c + distance) + 1):
            if max(abs(rr - r), abs(cc - c)) == distance:
                cells.add((rr, cc))
    return cells


def _outline_cells(center: tuple[int, int], nearest_anchor_distance: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    return _ring_cells(center, nearest_anchor_distance - 1, shape)


def _candidate_centers(shape: tuple[int, int], nearest_anchor_distance: int) -> list[tuple[int, int]]:
    h, w = shape
    radius = nearest_anchor_distance - 1
    return [
        (r, c)
        for r in range(radius, h - radius)
        for c in range(radius, w - radius)
    ]


def _make_marker(
    center: tuple[int, int],
    color: int,
    nearest_anchor_distance: int,
    shape: tuple[int, int],
) -> dict[str, object]:
    radius = nearest_anchor_distance - 1
    return {
        "center": center,
        "color": color,
        "distance": nearest_anchor_distance,
        "outline_color": MARKER_TO_OUTLINE[color],
        "outline": _outline_cells(center, nearest_anchor_distance, shape),
        "anchor_ring": _ring_cells(center, nearest_anchor_distance, shape),
        "top_left": (center[0] - radius, center[1] - radius),
    }


def _marker_pair_compatible(a: dict[str, object], b: dict[str, object]) -> bool:
    center_a = a["center"]
    center_b = b["center"]
    distance_between = _chebyshev(center_a, center_b)
    return (
        center_a != center_b
        and _manhattan(center_a, center_b) >= 2
        and distance_between != int(a["distance"]) - 1
        and distance_between != int(b["distance"]) - 1
    )


def _overlap_pairs(markers: list[dict[str, object]]) -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []
    for i in range(len(markers)):
        for j in range(i + 1, len(markers)):
            overlap_size = len(markers[i]["outline"] & markers[j]["outline"])
            if overlap_size > 0:
                pairs.append((i, j, overlap_size))
    return pairs


def _distance_key(marker: dict[str, object]) -> tuple[int, int, int]:
    top_left = marker["top_left"]
    return int(marker["distance"]), int(top_left[0]), int(top_left[1])


def main(grid: np.ndarray) -> np.ndarray:
    """Draw a hollow square around each colored marker, with size set by the nearest grey anchor.

    Input structure:
    - A black background contains isolated red, green, and teal marker pixels plus several grey anchor pixels.
    - Each marker is far enough from the border that its inferred outline fits fully inside the grid.
    - Markers and anchors remain single pixels in place.

    Rule:
    - For each colored marker, compute the minimum Chebyshev distance D to any grey anchor.
    - Keep the marker and every grey anchor unchanged, then draw the centered hollow square whose border lies at
      Chebyshev distance D-1 from the marker.
    - The outline color is green->pink, teal->yellow, and red->blue.
    - If multiple outlines want the same background cell, the marker with smaller D wins; ties use the earlier
      outline top-left corner in reading order.

    Key insight:
    - Grey anchors do not mark where to paint directly; they determine the outline radius, while marker color
      independently selects the outline color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    marker_coords = np.argwhere(np.isin(g, MARKER_COLORS))
    anchor_coords = np.argwhere(g == GREY)
    assert len(marker_coords) > 0
    assert len(anchor_coords) > 0

    out = g.copy()
    markers: list[dict[str, object]] = []
    for r_raw, c_raw in marker_coords:
        r, c = int(r_raw), int(c_raw)
        distances = np.maximum(np.abs(anchor_coords[:, 0] - r), np.abs(anchor_coords[:, 1] - c))
        nearest = int(distances.min())
        assert nearest >= 2
        markers.append(_make_marker((r, c), int(g[r, c]), nearest, g.shape))

    for marker in sorted(markers, key=_distance_key):
        color = int(marker["outline_color"])
        for rr, cc in marker["outline"]:
            if out[rr, cc] == BACKGROUND:
                out[rr, cc] = color

    return out


def _choose_mode() -> str:
    modes = [mode for mode, _weight in MODE_WEIGHTS]
    weights = [weight for _mode, weight in MODE_WEIGHTS]
    return random.choices(modes, weights=weights, k=1)[0]


def _sample_marker_colors(count: int, mode: str) -> list[int]:
    base = random.sample(MARKER_COLORS, len(MARKER_COLORS))
    if mode == "no_overlap":
        colors = base + [random.choice(MARKER_COLORS) for _ in range(count - len(base))]
        random.shuffle(colors)
        return colors
    extra = [random.choice(MARKER_COLORS) for _ in range(max(0, count - 3))]
    tail = [base[2], *extra]
    random.shuffle(tail)
    return [base[0], base[1], *tail]


def _sample_distances(count: int, mode: str) -> list[int]:
    def sample_extra() -> int:
        return random.choices(DISTANCE_VALUES, weights=(4, 4, 3, 2), k=1)[0]

    if mode == "same_distance_overlap":
        shared = random.choices((3, 4, 5), weights=(4, 4, 1), k=1)[0]
        distances = [shared, shared]
    elif mode == "different_distance_overlap":
        distances = list(random.choice(((2, 3), (2, 4), (3, 4), (3, 5), (4, 5))))
    else:
        distances = []
    distances.extend(sample_extra() for _ in range(count - len(distances)))
    return distances


def _place_special_pair(
    shape: tuple[int, int],
    colors: list[int],
    distances: list[int],
) -> list[dict[str, object]] | None:
    d0, d1 = int(distances[0]), int(distances[1])
    candidates: list[tuple[dict[str, object], dict[str, object]]] = []

    centers0 = _candidate_centers(shape, d0)
    centers1 = _candidate_centers(shape, d1)
    random.shuffle(centers0)
    random.shuffle(centers1)
    markers0 = [_make_marker(center, colors[0], d0, shape) for center in centers0]
    markers1 = [_make_marker(center, colors[1], d1, shape) for center in centers1]

    for marker0 in markers0:
        center0 = marker0["center"]
        for marker1 in markers1:
            center1 = marker1["center"]
            if d0 == d1 and center1 <= center0:
                continue
            if not _marker_pair_compatible(marker0, marker1):
                continue
            overlap = len(marker0["outline"] & marker1["outline"])
            if 1 <= overlap <= 6:
                candidates.append((marker0, marker1))
                if len(candidates) >= 80:
                    marker0, marker1 = random.choice(candidates)
                    return [dict(marker0), dict(marker1)]

    if not candidates:
        return None

    marker0, marker1 = random.choice(candidates)
    return [dict(marker0), dict(marker1)]


def _place_non_overlapping_marker(
    shape: tuple[int, int],
    color: int,
    nearest_anchor_distance: int,
    placed: list[dict[str, object]],
) -> dict[str, object] | None:
    centers = _candidate_centers(shape, nearest_anchor_distance)
    random.shuffle(centers)
    for center in centers:
        marker = _make_marker(center, color, nearest_anchor_distance, shape)
        if all(
            _marker_pair_compatible(marker, other) and not (marker["outline"] & other["outline"])
            for other in placed
        ):
            return marker
    return None


def _place_markers(
    shape: tuple[int, int],
    colors: list[int],
    distances: list[int],
    mode: str,
) -> list[dict[str, object]] | None:
    markers: list[dict[str, object]]
    specs: list[tuple[int, int]]
    if mode == "no_overlap":
        markers = []
        specs = list(zip(colors, distances))
    else:
        special = _place_special_pair(shape, colors, distances)
        if special is None:
            return None
        markers = special
        specs = list(zip(colors[2:], distances[2:]))

    specs.sort(key=lambda item: item[1], reverse=True)
    for color, distance in specs:
        marker = _place_non_overlapping_marker(shape, int(color), int(distance), markers)
        if marker is None:
            return None
        markers.append(marker)

    overlaps = _overlap_pairs(markers)
    if mode == "no_overlap":
        if overlaps:
            return None
    elif len(overlaps) != 1:
        return None
    else:
        i, j, _size = overlaps[0]
        same_distance = int(markers[i]["distance"]) == int(markers[j]["distance"])
        if mode == "same_distance_overlap" and not same_distance:
            return None
        if mode == "different_distance_overlap" and same_distance:
            return None
    return markers


def _anchor_candidates(
    marker: dict[str, object],
    markers: list[dict[str, object]],
    occupied: set[tuple[int, int]],
    anchors: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates = []
    for cell in marker["anchor_ring"]:
        if cell in occupied:
            continue
        if any(_manhattan(cell, other["center"]) < 2 for other in markers):
            continue
        if any(_manhattan(cell, other_anchor) < 2 for other_anchor in anchors):
            continue
        if any(_chebyshev(cell, other["center"]) < int(other["distance"]) for other in markers):
            continue
        candidates.append(cell)
    random.shuffle(candidates)
    return candidates


def _assign_anchor_rec(
    markers: list[dict[str, object]],
    remaining: list[int],
    occupied: set[tuple[int, int]],
    anchors: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    if not remaining:
        return set(anchors)

    best_idx = None
    best_options = None
    for idx in remaining:
        options = _anchor_candidates(markers[idx], markers, occupied, anchors)
        if not options:
            return None
        if best_options is None or len(options) < len(best_options):
            best_idx = idx
            best_options = options

    next_remaining = [idx for idx in remaining if idx != best_idx]
    for cell in best_options:
        anchors.add(cell)
        result = _assign_anchor_rec(markers, next_remaining, occupied | {cell}, anchors)
        anchors.remove(cell)
        if result is not None:
            return result
    return None


def _place_anchors(markers: list[dict[str, object]]) -> set[tuple[int, int]] | None:
    occupied = {marker["center"] for marker in markers}
    return _assign_anchor_rec(markers, list(range(len(markers))), occupied, set())


def _extra_anchor_candidates(
    shape: tuple[int, int],
    markers: list[dict[str, object]],
    anchors: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    occupied = {marker["center"] for marker in markers} | anchors
    candidates = []
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if cell in occupied:
                continue
            if any(_manhattan(cell, marker["center"]) < 2 for marker in markers):
                continue
            if any(_manhattan(cell, anchor) < 2 for anchor in anchors):
                continue
            if any(_chebyshev(cell, marker["center"]) <= int(marker["distance"]) for marker in markers):
                continue
            candidates.append(cell)
    random.shuffle(candidates)
    return candidates


def _add_extra_anchors(
    shape: tuple[int, int],
    markers: list[dict[str, object]],
    anchors: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    extra_limit = 5 - len(anchors)
    if extra_limit <= 0:
        return anchors

    extra_count = random.randint(0, extra_limit)
    if extra_count == 0:
        return anchors

    options = _extra_anchor_candidates(shape, markers, anchors)
    if len(options) < extra_count:
        return None
    anchors = set(anchors)
    for cell in options[:extra_count]:
        anchors.add(cell)
    return anchors


def _scene_is_valid(
    markers: list[dict[str, object]],
    anchors: set[tuple[int, int]],
    mode: str,
) -> bool:
    if not (3 <= len(markers) <= 5 and 3 <= len(anchors) <= 5):
        return False

    occupied = {marker["center"] for marker in markers} | anchors
    for marker in markers:
        center = marker["center"]
        nearest = min(_chebyshev(center, anchor) for anchor in anchors)
        if nearest != int(marker["distance"]):
            return False
        if marker["outline"] & occupied:
            return False
        if any(_manhattan(center, anchor) < 2 for anchor in anchors):
            return False

    overlaps = _overlap_pairs(markers)
    if mode == "no_overlap":
        return not overlaps
    if len(overlaps) != 1:
        return False
    i, j, _size = overlaps[0]
    same_distance = int(markers[i]["distance"]) == int(markers[j]["distance"])
    return same_distance if mode == "same_distance_overlap" else not same_distance


def _render_input(shape: tuple[int, int], markers: list[dict[str, object]], anchors: set[tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for marker in markers:
        r, c = marker["center"]
        grid[r, c] = int(marker["color"])
    for r, c in anchors:
        grid[r, c] = GREY
    return grid


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    shape = (h, w)
    count = random.choice(COUNT_CHOICES)
    mode = _choose_mode()
    colors = _sample_marker_colors(count, mode)
    distances = _sample_distances(count, mode)

    markers = _place_markers(shape, colors, distances, mode)
    if markers is None:
        return None

    anchors = _place_anchors(markers)
    if anchors is None:
        return None

    anchors = _add_extra_anchors(shape, markers, anchors)
    if anchors is None or not _scene_is_valid(markers, anchors, mode):
        return None

    inp = _render_input(shape, markers, anchors)
    out = main(inp)
    changed = int(np.count_nonzero(out != inp))
    if changed < 12:
        return None
    if changed > (h * w) // 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 3")
