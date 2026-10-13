from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import Color, in_bounds, label_mask, neighbors4
import random
from itertools import combinations_with_replacement, permutations


BASE_COLORS = {Color.BLACK, Color.BLUE}


def _best_marker_color(
    g: np.ndarray, coords: np.ndarray, markers: dict[tuple[int, int], int]
) -> int:
    contacts: dict[tuple[int, int], int] = defaultdict(int)
    for r, c in coords:
        for nr, nc in neighbors4(int(r), int(c)):
            if in_bounds(g.shape, nr, nc) and (nr, nc) in markers:
                contacts[(nr, nc)] += 1

    assert contacts, "each base-color component must touch at least one marker"
    max_contact = max(contacts.values())
    winners = [pos for pos, count in contacts.items() if count == max_contact]
    return markers[min(winners)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each black or blue connected component with the color of the marker that borders it most.

    Input structure
    - The grid contains black regions, blue line components, and isolated non-black/non-blue marker pixels.
    - Marker pixels are controls; the black and blue cells are the regions to recolor.
    Rule
    - Label black and blue cells separately using 4-connectivity.
    - For each labeled component, count side-adjacencies to every marker pixel.
    - Recolor the component with the marker color having the largest shared side boundary; equal counts use row-major marker order.
    - Leave the marker pixels unchanged.
    Key insight
    - The color assignment is local to each original-color component and is decided by shared edge contact with nearby markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = set(map(int, np.unique(g)))
    assert Color.BLACK in colors and Color.BLUE in colors, "expected black and blue base colors"

    marker_coords = np.argwhere(~np.isin(g, list(BASE_COLORS)))
    assert len(marker_coords) > 0, "expected colored marker pixels"
    markers = {(int(r), int(c)): int(g[r, c]) for r, c in marker_coords}

    out = g.copy()
    for base_color in (Color.BLACK, Color.BLUE):
        labels, n_labels = label_mask(g == base_color, connectivity=4)
        assert n_labels > 0, "expected each base color to appear"
        for label_id in range(1, n_labels + 1):
            component = np.argwhere(labels == label_id)
            out[labels == label_id] = _best_marker_color(g, component, markers)

    return out

# --------------- generator (o5x) ---------------

# Canvas and color roles.
_GEN_H = 13
_GEN_W = 13
_GEN_SHAPE = (_GEN_H, _GEN_W)
_GEN_BLACK = 0
_GEN_BLUE = 1
_GEN_TEMP_MARKER = 2
_GEN_BASE_COLORS = (_GEN_BLACK, _GEN_BLUE)
_GEN_MARKER_COLORS = tuple(range(2, 10))

# Blue wall size and topology targets.
_GEN_BLUE_TOTAL_MIN = 43
_GEN_BLUE_TOTAL_MAX = 50
_GEN_LOOP_SIZE_MIN = 3
_GEN_LOOP_SIZE_MAX = 7
_GEN_LOOP_PERIMETER_MIN = 12
_GEN_LOOP_PERIMETER_MAX = 22
_GEN_LOOP_INTERIOR_MIN = 3
_GEN_LOOP_COUNT_OPTIONS = (2, 3, 4)
_GEN_APPEND_TOTAL_MIN = 2
_GEN_APPEND_TOTAL_MAX = 10
_GEN_MIN_ENDPOINTS = 2
_GEN_MAX_ENDPOINTS = 10
_GEN_MAX_BRANCHES = 2
_GEN_DEGREE2_MIN = 0.78
_GEN_DEGREE2_MAX = 0.95
_GEN_MAX_BLUE_COMPONENTS = 5

# Placement and search limits.
_GEN_LOOP_MARGIN = 1
_GEN_MAX_TAILS = 2
_GEN_MIN_TAIL_LENGTH = 1
_GEN_MIN_OPEN_PATH_LENGTH = 2
_GEN_MAX_PATH_CANDIDATES = 40
_GEN_COVER_BLUE_RETRIES = 6
_GEN_REQUIRED_MARKER_TOP_K = 5
_GEN_BLUE_COVER_TOP_K = 5
_GEN_EXTRA_MARKER_TOP_K = 8
_GEN_GENERATION_ATTEMPTS = 300

# Component and marker validity bounds.
_GEN_BLACK_COMPONENT_MIN_COUNT = 3
_GEN_BLACK_COMPONENT_MAX_COUNT = 5
_GEN_BLACK_COMPONENT_MIN_AREA = 2
_GEN_BLACK_COMPONENT_MAX_AREA = 110
_GEN_BLUE_COMPONENT_MIN_COUNT = 1
_GEN_BLUE_COMPONENT_MAX_COUNT = 5
_GEN_MARKER_MIN_COUNT = 3
_GEN_MARKER_MAX_COUNT = 7
_GEN_MARKER_MIN_DISTINCT = 3
_GEN_MARKER_MAX_DISTINCT = 6
_GEN_DUPLICATE_MARKER_PROB = 0.25
_GEN_MARKER_NEIGHBOR_RADIUS = 1
_GEN_MAX_DUPLICATES_PER_COLOR = 2
_GEN_MAX_DUPLICATED_COLORS = 1
_GEN_MIN_OUTPUT_COLORS = 2


def _gen_perimeter(size: tuple[int, int]) -> int:
    h, w = size
    return 2 * (h + w) - 4


_GEN_SIZE_POOL = tuple(
    (h, w)
    for h in range(_GEN_LOOP_SIZE_MIN, _GEN_LOOP_SIZE_MAX + 1)
    for w in range(_GEN_LOOP_SIZE_MIN, _GEN_LOOP_SIZE_MAX + 1)
    if (h - 2) * (w - 2) >= _GEN_LOOP_INTERIOR_MIN
    and _GEN_LOOP_PERIMETER_MIN <= _gen_perimeter((h, w)) <= _GEN_LOOP_PERIMETER_MAX
)

_GEN_COMBOS_BY_TOTAL: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {}
for _gen_target in range(_GEN_BLUE_TOTAL_MIN, _GEN_BLUE_TOTAL_MAX + 1):
    _gen_options: list[tuple[tuple[int, int], ...]] = []
    for _gen_count in _GEN_LOOP_COUNT_OPTIONS:
        for _gen_combo in combinations_with_replacement(_GEN_SIZE_POOL, _gen_count):
            _gen_base = sum(_gen_perimeter(s) for s in _gen_combo)
            if _GEN_APPEND_TOTAL_MIN <= _gen_target - _gen_base <= _GEN_APPEND_TOTAL_MAX:
                _gen_options.append(_gen_combo)
    _GEN_COMBOS_BY_TOTAL[_gen_target] = tuple(_gen_options)


def _gen_top_choices(items: list, limit: int) -> list:
    return items[: max(1, min(limit, len(items)))]


def _gen_count_blue_neighbors(g: np.ndarray, r: int, c: int) -> int:
    return sum(
        1
        for nr, nc in neighbors4(r, c)
        if in_bounds(g.shape, nr, nc) and g[nr, nc] == _GEN_BLUE
    )


def _gen_marker_mask(g: np.ndarray) -> np.ndarray:
    return (g != _GEN_BLACK) & (g != _GEN_BLUE)


def _gen_rects_clear(
    box: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]
) -> bool:
    r0, c0, r1, c1 = box
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + 1 < pr0 or pr1 + 1 < r0 or c1 + 1 < pc0 or pc1 + 1 < c0):
            return False
    return True


def _gen_place_rectangles(
    sizes: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    sizes = sorted(sizes, key=lambda s: s[0] * s[1], reverse=True)
    placed: list[tuple[int, int, int, int]] = []

    def search(i: int) -> bool:
        if i == len(sizes):
            return True
        h, w = sizes[i]
        positions = [
            (r, c)
            for r in range(_GEN_LOOP_MARGIN, _GEN_H - h - _GEN_LOOP_MARGIN + 1)
            for c in range(_GEN_LOOP_MARGIN, _GEN_W - w - _GEN_LOOP_MARGIN + 1)
        ]
        random.shuffle(positions)
        for r, c in positions:
            box = (r, c, r + h - 1, c + w - 1)
            if not _gen_rects_clear(box, placed):
                continue
            placed.append(box)
            if search(i + 1):
                return True
            placed.pop()
        return False

    if search(0):
        return placed.copy()
    return None


def _gen_draw_loop(g: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    g[r0, c0 : c1 + 1] = _GEN_BLUE
    g[r1, c0 : c1 + 1] = _GEN_BLUE
    g[r0 : r1 + 1, c0] = _GEN_BLUE
    g[r0 : r1 + 1, c1] = _GEN_BLUE


def _gen_tail_candidates(
    g: np.ndarray, box: tuple[int, int, int, int], length: int
) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = box
    anchors: list[tuple[int, int, int, int]] = []
    for c in range(c0 + 1, c1):
        anchors.append((r0, c, -1, 0))
        anchors.append((r1, c, 1, 0))
    for r in range(r0 + 1, r1):
        anchors.append((r, c0, 0, -1))
        anchors.append((r, c1, 0, 1))
    random.shuffle(anchors)

    out: list[list[tuple[int, int]]] = []
    for ar, ac, dr, dc in anchors:
        path = [(ar + dr * k, ac + dc * k) for k in range(1, length + 1)]
        if any(not in_bounds(g.shape, r, c) or g[r, c] != _GEN_BLACK for r, c in path):
            continue
        path_set = set(path)
        ok = True
        for idx, (r, c) in enumerate(path):
            allowed = {(ar, ac)}
            if idx > 0:
                allowed.add(path[idx - 1])
            if idx + 1 < len(path):
                allowed.add(path[idx + 1])
            for nr, nc in neighbors4(r, c):
                if (
                    in_bounds(g.shape, nr, nc)
                    and g[nr, nc] == _GEN_BLUE
                    and (nr, nc) not in allowed
                ):
                    ok = False
                    break
                if (nr, nc) in path_set and (nr, nc) not in allowed:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            out.append(path)
    return out


def _gen_add_tail_lengths(
    g: np.ndarray, boxes: list[tuple[int, int, int, int]], tail_lengths: tuple[int, ...]
) -> bool:
    if not tail_lengths:
        return True
    configs = []
    for box_order in permutations(range(len(boxes)), len(tail_lengths)):
        for length_order in set(permutations(tail_lengths)):
            configs.append((box_order, length_order))
    random.shuffle(configs)

    for box_order, length_order in configs:
        temp = g.copy()
        ok = True
        for box_idx, length in zip(box_order, length_order):
            paths = _gen_tail_candidates(temp, boxes[box_idx], length)
            if not paths:
                ok = False
                break
            for r, c in random.choice(paths):
                temp[r, c] = _GEN_BLUE
        if ok:
            g[:, :] = temp
            return True
    return False


def _gen_integer_parts(total: int, count: int, minimum: int) -> list[tuple[int, ...]]:
    if count == 0:
        return [()] if total == 0 else []
    out = []
    max_first = total - minimum * (count - 1)
    for first in range(minimum, max_first + 1):
        for rest in _gen_integer_parts(total - first, count - 1, minimum):
            out.append((first,) + rest)
    return out


def _gen_appendage_plans(
    loop_count: int, append_total: int, target_blue: int
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    plans = []
    max_open = _GEN_MAX_BLUE_COMPONENTS - loop_count
    for tail_count in range(0, min(_GEN_MAX_TAILS, loop_count, append_total) + 1):
        min_tail_total = tail_count * _GEN_MIN_TAIL_LENGTH
        for tail_total in range(min_tail_total, append_total + 1):
            for tail_lengths in _gen_integer_parts(
                tail_total, tail_count, _GEN_MIN_TAIL_LENGTH
            ):
                remaining = append_total - sum(tail_lengths)
                for open_count in range(
                    0, min(max_open, remaining // _GEN_MIN_OPEN_PATH_LENGTH) + 1
                ):
                    for open_lengths in _gen_integer_parts(
                        remaining, open_count, _GEN_MIN_OPEN_PATH_LENGTH
                    ):
                        endpoints = tail_count + 2 * open_count
                        branches = tail_count
                        degree2 = (target_blue - endpoints - branches) / target_blue
                        if (
                            _GEN_MIN_ENDPOINTS <= endpoints <= _GEN_MAX_ENDPOINTS
                            and branches <= _GEN_MAX_BRANCHES
                            and _GEN_DEGREE2_MIN <= degree2 <= _GEN_DEGREE2_MAX
                        ):
                            plans.append((tail_lengths, open_lengths))
    return plans


def _gen_open_path_candidates(g: np.ndarray, length: int) -> list[list[tuple[int, int]]]:
    black_labels, n_labels = label_mask(g == _GEN_BLACK, connectivity=4)
    if n_labels == 0:
        return []
    exterior = max(
        range(1, n_labels + 1), key=lambda label_id: int(np.sum(black_labels == label_id))
    )
    eligible = black_labels == exterior

    def cell_ok(r: int, c: int) -> bool:
        if not (
            _GEN_LOOP_MARGIN <= r < _GEN_H - _GEN_LOOP_MARGIN
            and _GEN_LOOP_MARGIN <= c < _GEN_W - _GEN_LOOP_MARGIN
        ):
            return False
        if not eligible[r, c] or g[r, c] != _GEN_BLACK:
            return False
        return all(
            not (in_bounds(g.shape, nr, nc) and g[nr, nc] == _GEN_BLUE)
            for nr, nc in neighbors4(r, c)
        )

    starts = [(int(r), int(c)) for r, c in np.argwhere(eligible)]
    random.shuffle(starts)
    out: list[list[tuple[int, int]]] = []

    def extend(path: list[tuple[int, int]]) -> None:
        if len(out) >= _GEN_MAX_PATH_CANDIDATES:
            return
        if len(path) == length:
            out.append(path.copy())
            return
        last_r, last_c = path[-1]
        next_cells = list(neighbors4(last_r, last_c))
        random.shuffle(next_cells)
        used = set(path)
        for nr, nc in next_cells:
            if not in_bounds(g.shape, nr, nc) or not cell_ok(nr, nc) or (nr, nc) in used:
                continue
            if any(
                (ar, ac) in used and (ar, ac) != path[-1]
                for ar, ac in neighbors4(nr, nc)
            ):
                continue
            path.append((nr, nc))
            extend(path)
            path.pop()

    for start in starts:
        if len(out) >= _GEN_MAX_PATH_CANDIDATES:
            break
        if cell_ok(*start):
            extend([start])
    return out


def _gen_add_open_paths(g: np.ndarray, lengths: tuple[int, ...]) -> bool:
    lengths = tuple(sorted(lengths, reverse=True))
    for length in lengths:
        paths = _gen_open_path_candidates(g, length)
        if not paths:
            return False
        for r, c in random.choice(paths):
            g[r, c] = _GEN_BLUE
    return True


def _gen_add_appendages(
    g: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    append_total: int,
    target_blue: int,
) -> bool:
    plans = _gen_appendage_plans(len(boxes), append_total, target_blue)
    by_endpoint: dict[int, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}
    for plan in plans:
        endpoint_count = len(plan[0]) + 2 * len(plan[1])
        by_endpoint.setdefault(endpoint_count, []).append(plan)

    endpoint_order = sorted(by_endpoint)
    if endpoint_order:
        first = random.choices(endpoint_order, weights=[ep * ep for ep in endpoint_order], k=1)[0]
        endpoint_order.remove(first)
        random.shuffle(endpoint_order)
        endpoint_order.insert(0, first)

    for endpoint_count in endpoint_order:
        endpoint_plans = by_endpoint[endpoint_count]
        random.shuffle(endpoint_plans)
        for tail_lengths, open_lengths in endpoint_plans:
            temp = g.copy()
            if not _gen_add_tail_lengths(temp, boxes, tail_lengths):
                continue
            if not _gen_add_open_paths(temp, open_lengths):
                continue
            g[:, :] = temp
            return True
    return False


def _gen_marker_cells(g: np.ndarray) -> set[tuple[int, int]]:
    return {
        (int(r), int(c))
        for r, c in np.argwhere(_gen_marker_mask(g))
    }


def _gen_no_marker_neighbor(
    r: int, c: int, markers: list[tuple[int, int]]
) -> bool:
    for mr, mc in markers:
        if max(abs(int(r) - mr), abs(int(c) - mc)) <= _GEN_MARKER_NEIGHBOR_RADIUS:
            return False
    return True


def _gen_component_after_marker_ok(mask: np.ndarray, r: int, c: int) -> tuple[bool, int]:
    rest = mask.copy()
    rest[r, c] = False
    labels, n_labels = label_mask(rest, connectivity=4)
    if n_labels == 0:
        return False, 0
    for label_id in range(1, n_labels + 1):
        coords = np.argwhere(labels == label_id)
        if len(coords) < _GEN_BLACK_COMPONENT_MIN_AREA:
            return False, n_labels
        touches = False
        for nr, nc in neighbors4(r, c):
            if in_bounds(mask.shape, nr, nc) and labels[nr, nc] == label_id:
                touches = True
                break
        if not touches:
            return False, n_labels
    return True, n_labels


def _gen_required_marker_candidates(
    g: np.ndarray,
    labels: np.ndarray,
    label_id: int,
    markers: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    comp_mask = labels == label_id
    candidates = []
    for r, c in np.argwhere(comp_mask):
        r = int(r)
        c = int(c)
        if g[r, c] != _GEN_BLACK or not _gen_no_marker_neighbor(r, c, markers):
            continue
        blue_adj = _gen_count_blue_neighbors(g, r, c)
        if blue_adj == 0:
            continue
        ok, split_count = _gen_component_after_marker_ok(comp_mask, r, c)
        if ok:
            candidates.append((split_count, -blue_adj, r, c))
    random.shuffle(candidates)
    candidates.sort()
    return [(r, c) for _, _, r, c in candidates]


def _gen_coords_touch_markers(coords: np.ndarray, markers: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        for nr, nc in neighbors4(int(r), int(c)):
            if (nr, nc) in markers:
                return True
    return False


def _gen_components_touch_markers(g: np.ndarray) -> bool:
    markers = _gen_marker_cells(g)
    if not markers:
        return False
    for color in _GEN_BASE_COLORS:
        labels, n_labels = label_mask(g == color, connectivity=4)
        if n_labels == 0:
            return False
        for label_id in range(1, n_labels + 1):
            coords = np.argwhere(labels == label_id)
            if color == _GEN_BLACK and not (
                _GEN_BLACK_COMPONENT_MIN_AREA <= len(coords) <= _GEN_BLACK_COMPONENT_MAX_AREA
            ):
                return False
            if not _gen_coords_touch_markers(coords, markers):
                return False
    return True


def _gen_black_side_ok(g: np.ndarray) -> bool:
    markers = _gen_marker_cells(g)
    labels, n_labels = label_mask(g == _GEN_BLACK, connectivity=4)
    if not (_GEN_BLACK_COMPONENT_MIN_COUNT <= n_labels <= _GEN_BLACK_COMPONENT_MAX_COUNT):
        return False
    for label_id in range(1, n_labels + 1):
        coords = np.argwhere(labels == label_id)
        if not (
            _GEN_BLACK_COMPONENT_MIN_AREA <= len(coords) <= _GEN_BLACK_COMPONENT_MAX_AREA
        ):
            return False
        if not _gen_coords_touch_markers(coords, markers):
            return False
    return True


def _gen_blue_component_touched(
    g: np.ndarray, labels: np.ndarray, label_id: int
) -> bool:
    markers = _gen_marker_cells(g)
    return _gen_coords_touch_markers(np.argwhere(labels == label_id), markers)


def _gen_cover_blue_components(
    g: np.ndarray, markers: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    for _ in range(_GEN_COVER_BLUE_RETRIES):
        labels, n_labels = label_mask(g == _GEN_BLUE, connectivity=4)
        uncovered = [
            label_id
            for label_id in range(1, n_labels + 1)
            if not _gen_blue_component_touched(g, labels, label_id)
        ]
        if not uncovered:
            return markers
        label_id = random.choice(uncovered)
        candidates = set()
        for r, c in np.argwhere(labels == label_id):
            for nr, nc in neighbors4(int(r), int(c)):
                if in_bounds(g.shape, nr, nc) and g[nr, nc] == _GEN_BLACK:
                    candidates.add((nr, nc))
        scored = []
        for r, c in candidates:
            if not _gen_no_marker_neighbor(r, c, markers):
                continue
            temp = g.copy()
            temp[r, c] = _GEN_TEMP_MARKER
            if _gen_black_side_ok(temp):
                blue_adj = _gen_count_blue_neighbors(temp, r, c)
                scored.append((blue_adj, r, c))
        if not scored or len(markers) >= _GEN_MARKER_MAX_COUNT:
            return None
        random.shuffle(scored)
        scored.sort(reverse=True)
        _, r, c = random.choice(_gen_top_choices(scored, _GEN_BLUE_COVER_TOP_K))
        g[r, c] = _GEN_TEMP_MARKER
        markers.append((r, c))
    return None


def _gen_blue_stats(g: np.ndarray) -> tuple[int, int, float]:
    blue = g == _GEN_BLUE
    if not np.any(blue):
        return 0, 0, 0.0
    degrees = np.zeros(g.shape, dtype=int)
    degrees[1:, :] += blue[:-1, :]
    degrees[:-1, :] += blue[1:, :]
    degrees[:, 1:] += blue[:, :-1]
    degrees[:, :-1] += blue[:, 1:]
    blue_degrees = degrees[blue]
    endpoints = int(np.sum(blue_degrees == 1))
    branches = int(np.sum(blue_degrees >= 3))
    degree2 = float(np.sum(blue_degrees == 2) / len(blue_degrees))
    return endpoints, branches, degree2


def _gen_base_ok(g: np.ndarray) -> bool:
    if g.shape != _GEN_SHAPE:
        return False
    blue_total = int(np.sum(g == _GEN_BLUE))
    if not (_GEN_BLUE_TOTAL_MIN <= blue_total <= _GEN_BLUE_TOTAL_MAX):
        return False

    black_labels, black_count = label_mask(g == _GEN_BLACK, connectivity=4)
    _, blue_count = label_mask(g == _GEN_BLUE, connectivity=4)
    if not (
        _GEN_BLACK_COMPONENT_MIN_COUNT <= black_count <= _GEN_BLACK_COMPONENT_MAX_COUNT
        and _GEN_BLUE_COMPONENT_MIN_COUNT <= blue_count <= _GEN_BLUE_COMPONENT_MAX_COUNT
    ):
        return False

    for label_id in range(1, black_count + 1):
        area = int(np.sum(black_labels == label_id))
        if not (_GEN_BLACK_COMPONENT_MIN_AREA <= area <= _GEN_BLACK_COMPONENT_MAX_AREA):
            return False

    endpoints, branches, degree2 = _gen_blue_stats(g)
    if not (
        _GEN_MIN_ENDPOINTS <= endpoints <= _GEN_MAX_ENDPOINTS
        and 0 <= branches <= _GEN_MAX_BRANCHES
        and _GEN_DEGREE2_MIN <= degree2 <= _GEN_DEGREE2_MAX
    ):
        return False

    return _gen_components_touch_markers(g)


def _gen_place_required_markers(g: np.ndarray) -> list[tuple[int, int]] | None:
    labels, n_labels = label_mask(g == _GEN_BLACK, connectivity=4)
    component_ids = list(range(1, n_labels + 1))
    random.shuffle(component_ids)
    component_ids.sort(key=lambda i: int(np.sum(labels == i)))

    markers: list[tuple[int, int]] = []
    for label_id in component_ids:
        candidates = _gen_required_marker_candidates(g, labels, label_id, markers)
        if not candidates:
            return None
        r, c = random.choice(_gen_top_choices(candidates, _GEN_REQUIRED_MARKER_TOP_K))
        g[r, c] = _GEN_TEMP_MARKER
        markers.append((r, c))
    markers = _gen_cover_blue_components(g, markers)
    if markers is None or len(markers) > _GEN_MARKER_MAX_COUNT:
        return None
    if not _gen_base_ok(g):
        return None
    return markers


def _gen_extra_marker_candidates(
    g: np.ndarray, markers: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    candidates = []
    for r, c in np.argwhere(g == _GEN_BLACK):
        r = int(r)
        c = int(c)
        if not _gen_no_marker_neighbor(r, c, markers):
            continue
        blue_adj = _gen_count_blue_neighbors(g, r, c)
        if blue_adj == 0:
            continue
        temp = g.copy()
        temp[r, c] = _GEN_TEMP_MARKER
        if _gen_base_ok(temp):
            candidates.append((blue_adj, r, c))
    random.shuffle(candidates)
    candidates.sort(reverse=True)
    return [(r, c) for _, r, c in candidates]


def _gen_assign_marker_colors(
    g: np.ndarray, markers: list[tuple[int, int]]
) -> None:
    n_markers = len(markers)
    if n_markers == _GEN_MARKER_MAX_COUNT:
        colors = list(random.sample(_GEN_MARKER_COLORS, _GEN_MARKER_MAX_DISTINCT))
        colors.append(random.choice(colors))
    elif (
        n_markers >= _GEN_MARKER_MIN_COUNT + 1
        and random.random() < _GEN_DUPLICATE_MARKER_PROB
    ):
        colors = list(random.sample(_GEN_MARKER_COLORS, n_markers - 1))
        colors.append(random.choice(colors))
    else:
        colors = list(random.sample(_GEN_MARKER_COLORS, n_markers))
    random.shuffle(colors)
    for (r, c), color in zip(markers, colors):
        g[r, c] = color


def _gen_final_ok(g: np.ndarray) -> bool:
    if not _gen_base_ok(g):
        return False

    marker_values = [
        int(g[r, c])
        for r, c in np.argwhere(_gen_marker_mask(g))
    ]
    marker_count = len(marker_values)
    distinct = len(set(marker_values))
    counts = Counter(marker_values)
    if not (
        _GEN_MARKER_MIN_COUNT <= marker_count <= _GEN_MARKER_MAX_COUNT
        and _GEN_MARKER_MIN_DISTINCT <= distinct <= _GEN_MARKER_MAX_DISTINCT
    ):
        return False
    if (
        any(v > _GEN_MAX_DUPLICATES_PER_COLOR for v in counts.values())
        or sum(1 for v in counts.values() if v == _GEN_MAX_DUPLICATES_PER_COLOR)
        > _GEN_MAX_DUPLICATED_COLORS
    ):
        return False

    markers = _gen_marker_cells(g)
    for r, c in markers:
        for dr in range(-_GEN_MARKER_NEIGHBOR_RADIUS, _GEN_MARKER_NEIGHBOR_RADIUS + 1):
            for dc in range(-_GEN_MARKER_NEIGHBOR_RADIUS, _GEN_MARKER_NEIGHBOR_RADIUS + 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in markers:
                    return False

    try:
        out = main(g)
    except Exception:
        return False
    out_colors = set(map(int, np.unique(out)))
    if (
        _GEN_BLACK in out_colors
        or _GEN_BLUE in out_colors
        or len(out_colors) < _GEN_MIN_OUTPUT_COLORS
    ):
        return False
    return out_colors <= set(marker_values)


def _gen_select_loop_design() -> tuple[int, list[tuple[int, int]], int] | None:
    """Choose blue target, loop sizes, and appendage cell budget."""
    target_blue = random.randint(_GEN_BLUE_TOTAL_MIN, _GEN_BLUE_TOTAL_MAX)
    combo_options = list(_GEN_COMBOS_BY_TOTAL[target_blue])
    if not combo_options:
        return None
    loop_counts = sorted({len(combo) for combo in combo_options})
    loop_count = random.choice(loop_counts)
    count_options = [combo for combo in combo_options if len(combo) == loop_count]
    sizes = list(random.choice(count_options))
    random.shuffle(sizes)
    base_blue = sum(_gen_perimeter(size) for size in sizes)
    append_total = target_blue - base_blue
    return target_blue, sizes, append_total


def _gen_build_blue_walls(
    target_blue: int, sizes: list[tuple[int, int]], append_total: int
) -> np.ndarray | None:
    """Render loop walls and appendages before marker placement."""
    boxes = _gen_place_rectangles(sizes)
    if boxes is None:
        return None

    g = np.zeros(_GEN_SHAPE, dtype=int)
    for box in boxes:
        _gen_draw_loop(g, box)
    if not _gen_add_appendages(g, boxes, append_total, target_blue):
        return None
    if int(np.sum(g == _GEN_BLUE)) != target_blue:
        return None
    return g


def _gen_add_extra_markers(g: np.ndarray, markers: list[tuple[int, int]]) -> bool:
    desired_markers = random.randint(
        max(_GEN_MARKER_MIN_COUNT, len(markers)), _GEN_MARKER_MAX_COUNT
    )
    while len(markers) < desired_markers:
        candidates = _gen_extra_marker_candidates(g, markers)
        if not candidates:
            return False
        r, c = random.choice(_gen_top_choices(candidates, _GEN_EXTRA_MARKER_TOP_K))
        g[r, c] = _GEN_TEMP_MARKER
        markers.append((r, c))
    return True


def _gen_attempt() -> np.ndarray | None:
    design = _gen_select_loop_design()
    if design is None:
        return None
    target_blue, sizes, append_total = design

    g = _gen_build_blue_walls(target_blue, sizes, append_total)
    if g is None:
        return None

    markers = _gen_place_required_markers(g)
    if markers is None:
        return None

    if not _gen_add_extra_markers(g, markers):
        return None

    _gen_assign_marker_colors(g, markers)
    if _gen_final_ok(g):
        return g
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_GENERATION_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid input")
