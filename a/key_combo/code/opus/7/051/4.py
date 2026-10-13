from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
ANCHOR_COLOR = 8
TEMPLATE_COLOR = 5
MARKER_PALETTE = (1, 2, 3, 4, 7, 9)

RING_SPECS = (
    (-1, -1, -1, -1),
    (-1, 0, -1, 0),
    (-1, 1, -1, 0),
    (-1, 2, -1, 1),
    (0, -1, 0, -1),
    (1, -1, 0, -1),
    (0, 2, 0, 1),
    (1, 2, 0, 1),
    (2, -1, 1, -1),
    (2, 0, 1, 0),
    (2, 1, 1, 0),
    (2, 2, 1, 1),
)

CLUSTERS = {
    "nw": (0, 1, 2, 4, 5),
    "ne": (1, 2, 3, 6, 7),
    "sw": (4, 5, 8, 9, 10),
    "se": (6, 7, 9, 10, 11),
}

TEMPLATE_TOP_LEFT = {
    "nw": lambda h, w: (h - 3, w - 3),
    "ne": lambda h, w: (h - 3, 0),
    "sw": lambda h, w: (0, w - 3),
    "se": lambda h, w: (0, 0),
}

GRID_SIZE_RANGE = (16, 19)
ANCHOR_MARGIN = 5
GENERATION_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_anchor_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == ANCHOR_COLOR)
    assert coords.shape[0] == 4, "expected one teal 2x2 anchor block"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert r1 - r0 == 1 and c1 - c0 == 1, "anchor must be a 2x2 block"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == ANCHOR_COLOR), "anchor must be solid"
    return int(r0), int(c0), int(r1), int(c1)


def _find_template_offsets(grid: np.ndarray) -> list[tuple[int, int]]:
    comps = _components(grid == TEMPLATE_COLOR)
    assert len(comps) == 1, "expected one grey template sprite"
    coords = comps[0]
    assert 3 <= len(coords) <= 5, "template area must be 3..5"
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 - r0 <= 2 and c1 - c0 <= 2, "template bbox must fit within 3x3"
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    return sorted((r - center_r, c - center_c) for r, c in coords)


def _project_to_ring(r: int, c: int, anchor_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = anchor_bbox
    nr = r0 - 1 if r < r0 else r1 + 1 if r > r1 else r
    nc = c0 - 1 if c < c0 else c1 + 1 if c > c1 else c
    return nr, nc


def _collect_markers(
    grid: np.ndarray, anchor_bbox: tuple[int, int, int, int]
) -> list[tuple[int, int, int, int, int, int]]:
    marker_mask = (grid != BG) & (grid != ANCHOR_COLOR) & (grid != TEMPLATE_COLOR)
    marker_coords = np.argwhere(marker_mask)
    assert 4 <= marker_coords.shape[0] <= 6, "expected 4..6 markers"
    colors = [int(grid[r, c]) for r, c in marker_coords]
    assert len(colors) == len(set(colors)), "marker colors must be distinct"

    r0, c0, r1, c1 = anchor_bbox
    markers = []
    for rr, cc in marker_coords:
        r = int(rr)
        c = int(cc)
        nr, nc = _project_to_ring(r, c, anchor_bbox)
        assert (nr, nc) != (r, c), "marker must move"
        row_aligned = r0 <= r <= r1
        col_aligned = c0 <= c <= c1
        if not row_aligned and not col_aligned:
            assert abs(r - nr) == abs(c - nc), "diagonal markers must lie on 45-degree rays"
        distance = max(abs(r - nr), abs(c - nc))
        assert distance >= 1, "marker distance must be positive"
        markers.append((distance, r, c, int(grid[r, c]), nr, nc))
    return sorted(markers, key=lambda item: (-item[0], item[1], item[2]))


def _stamp(
    out: np.ndarray,
    template_offsets: list[tuple[int, int]],
    center: tuple[int, int],
    color: int,
    anchor_bbox: tuple[int, int, int, int],
) -> None:
    r0, c0, r1, c1 = anchor_bbox
    cr, cc = center
    for dr, dc in template_offsets:
        rr = cr + dr
        cc2 = cc + dc
        if not _in_bounds(out.shape, rr, cc2):
            continue
        if r0 <= rr <= r1 and c0 <= cc2 <= c1:
            continue
        out[rr, cc2] = color


def _render_output(
    grid: np.ndarray,
    markers: list[tuple[int, int, int, int, int, int]],
    template_offsets: list[tuple[int, int]],
    anchor_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[(out != BG) & (out != ANCHOR_COLOR)] = BG
    for _distance, _r, _c, color, nr, nc in markers:
        _stamp(out, template_offsets, (nr, nc), color, anchor_bbox)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each colored ray marker to the anchor ring, then stamp the grey sprite by distance order.

    Input structure:
    - A black background contains one teal 2x2 solid block acting as the anchor.
    - Four to six singleton colored markers lie on anchor-aligned rows, columns, or 45-degree diagonals.
    - One grey 4-connected sprite of area 3..5 sits elsewhere as the stamp template.

    Rule:
    - For every marker, slide it inward along its ray until it reaches the ring cell adjacent to the 2x2 anchor.
    - Record each slide distance and sort markers from farthest to nearest, breaking ties by reading order of the original marker positions.
    - Erase the original markers and the original grey template, then stamp the template at each marker's ring destination using that marker's color, with later stamps overwriting earlier ones while the teal anchor stays unchanged.

    Key insight:
    - The output depends on three coupled inferences at once: the shared template shape, each marker's ring destination, and the layer order induced by slide distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    anchor_bbox = _find_anchor_bbox(g)
    template_offsets = _find_template_offsets(g)
    markers = _collect_markers(g, anchor_bbox)
    return _render_output(g, markers, template_offsets, anchor_bbox)


def _random_template_offsets() -> list[tuple[int, int]]:
    while True:
        area = random.randint(3, 5)
        cells = {(random.randint(0, 2), random.randint(0, 2))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))
        if _bbox(list(cells)) != (0, 0, 2, 2):
            continue
        return sorted((r - 1, c - 1) for r, c in cells)


def _cluster_destinations(cluster_name: str, count: int) -> list[int]:
    cluster = list(CLUSTERS[cluster_name])
    if count == 4:
        return random.sample(cluster, 4)
    if count == 5:
        return cluster[:]
    return cluster + [random.choice(cluster)]


def _max_distance(shape: tuple[int, int], dest: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = dest
    dr, dc = direction
    limits = []
    if dr < 0:
        limits.append(r)
    elif dr > 0:
        limits.append(h - 1 - r)
    if dc < 0:
        limits.append(c)
    elif dc > 0:
        limits.append(w - 1 - c)
    return min(limits)


def _isolated(points: list[tuple[int, int]]) -> bool:
    point_set = set(points)
    for r, c in points:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) != (r, c) and (nr, nc) in point_set:
                    return False
    return True


def _choose_anchor_top_left(h: int, w: int) -> tuple[int, int]:
    rows = list(range(ANCHOR_MARGIN, h - ANCHOR_MARGIN - 1))
    cols = list(range(ANCHOR_MARGIN, w - ANCHOR_MARGIN - 1))
    candidates = []
    center_r = (h - 2) / 2.0
    center_c = (w - 2) / 2.0
    for r in rows:
        for c in cols:
            if abs(r - center_r) < 0.75 and abs(c - center_c) < 0.75:
                continue
            candidates.append((r, c))
    if not candidates:
        raise RuntimeError("no off-center anchor candidates")
    return random.choice(candidates)


def _assign_markers(
    shape: tuple[int, int],
    anchor_top_left: tuple[int, int],
    cluster_name: str,
    count: int,
    colors: list[int],
    force_tie: bool,
) -> list[tuple[int, int, int, int, int, int]] | None:
    ar, ac = anchor_top_left
    spec_ids = _cluster_destinations(cluster_name, count)
    ring_info = []
    for spec_id in spec_ids:
        ro, co, dr, dc = RING_SPECS[spec_id]
        dest = (ar + ro, ac + co)
        ring_info.append(
            {
                "spec_id": spec_id,
                "dest": dest,
                "dir": (dr, dc),
                "max_dist": _max_distance(shape, dest, (dr, dc)),
            }
        )
    if any(info["max_dist"] < 2 for info in ring_info):
        return None

    for _ in range(200):
        distances: list[int | None] = [None] * count
        used_by_spec: dict[int, set[int]] = defaultdict(set)

        if force_tie:
            tie_pairs = [
                (i, j)
                for i in range(count)
                for j in range(i + 1, count)
                if ring_info[i]["spec_id"] != ring_info[j]["spec_id"]
            ]
            if not tie_pairs:
                return None
            tie_i, tie_j = random.choice(tie_pairs)
            tie_max = min(int(ring_info[tie_i]["max_dist"]), int(ring_info[tie_j]["max_dist"]))
            if tie_max < 2:
                continue
            tie_dist = random.randint(2, tie_max)
            distances[tie_i] = tie_dist
            distances[tie_j] = tie_dist

        order = list(range(count))
        random.shuffle(order)
        ok = True
        for idx in order:
            if distances[idx] is not None:
                used_by_spec[int(ring_info[idx]["spec_id"])].add(int(distances[idx]))
                continue
            spec_id = int(ring_info[idx]["spec_id"])
            choices = [
                d
                for d in range(2, int(ring_info[idx]["max_dist"]) + 1)
                if all(abs(d - prev) >= 2 for prev in used_by_spec[spec_id])
            ]
            if not choices:
                ok = False
                break
            distances[idx] = random.choice(choices)
            used_by_spec[spec_id].add(int(distances[idx]))
        if not ok:
            continue

        markers = []
        points = []
        used_sources = set()
        for color, info, dist in zip(colors, ring_info, distances):
            assert dist is not None
            dr, dc = info["dir"]
            nr, nc = info["dest"]
            src = (nr + dr * dist, nc + dc * dist)
            if src in used_sources:
                ok = False
                break
            used_sources.add(src)
            points.append(src)
            markers.append((int(dist), src[0], src[1], color, nr, nc))
        if not ok or not _isolated(points):
            continue
        return markers
    return None


def _grid_from_parts(
    shape: tuple[int, int],
    anchor_top_left: tuple[int, int],
    template_offsets: list[tuple[int, int]],
    cluster_name: str,
    markers: list[tuple[int, int, int, int, int, int]],
) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    ar, ac = anchor_top_left
    grid[ar : ar + 2, ac : ac + 2] = ANCHOR_COLOR

    tr, tc = TEMPLATE_TOP_LEFT[cluster_name](h, w)
    for dr, dc in template_offsets:
        grid[tr + 1 + dr, tc + 1 + dc] = TEMPLATE_COLOR

    for _dist, r, c, color, _nr, _nc in markers:
        if grid[r, c] != BG:
            raise ValueError("marker source collision")
        grid[r, c] = color
    return grid


def _reverse_distance_order(
    markers: list[tuple[int, int, int, int, int, int]]
) -> list[tuple[int, int, int, int, int, int]]:
    return sorted(markers, key=lambda item: (item[0], item[1], item[2]))


def _flip_tie_order(
    markers: list[tuple[int, int, int, int, int, int]]
) -> list[tuple[int, int, int, int, int, int]]:
    groups: dict[int, list[tuple[int, int, int, int, int, int]]] = defaultdict(list)
    for marker in markers:
        groups[marker[0]].append(marker)
    out = []
    for distance in sorted(groups.keys(), reverse=True):
        group = sorted(groups[distance], key=lambda item: (item[1], item[2]))
        if len(group) > 1:
            group = list(reversed(group))
        out.extend(group)
    return out


def _valid_order_signal(
    grid: np.ndarray,
    markers: list[tuple[int, int, int, int, int, int]],
    template_offsets: list[tuple[int, int]],
    anchor_bbox: tuple[int, int, int, int],
    force_tie: bool,
) -> bool:
    correct = _render_output(
        grid,
        sorted(markers, key=lambda item: (-item[0], item[1], item[2])),
        template_offsets,
        anchor_bbox,
    )
    reversed_order = _render_output(
        grid,
        _reverse_distance_order(markers),
        template_offsets,
        anchor_bbox,
    )
    if np.array_equal(correct, reversed_order):
        return False
    if force_tie:
        tie_flip = _render_output(grid, _flip_tie_order(markers), template_offsets, anchor_bbox)
        if np.array_equal(correct, tie_flip):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    shape = (h, w)
    cluster_name = random.choice(list(CLUSTERS))
    count = random.choice((4, 5, 6))
    force_tie = random.random() < 0.55
    template_offsets = _random_template_offsets()
    colors = random.sample(list(MARKER_PALETTE), count)
    anchor_top_left = _choose_anchor_top_left(h, w)

    markers = _assign_markers(shape, anchor_top_left, cluster_name, count, colors, force_tie)
    if markers is None:
        return None

    try:
        grid = _grid_from_parts(shape, anchor_top_left, template_offsets, cluster_name, markers)
    except ValueError:
        return None

    anchor_bbox = _find_anchor_bbox(grid)
    if not _valid_order_signal(grid, markers, template_offsets, anchor_bbox, force_tie):
        return None

    try:
        expected = _render_output(
            grid,
            sorted(markers, key=lambda item: (-item[0], item[1], item[2])),
            template_offsets,
            anchor_bbox,
        )
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
