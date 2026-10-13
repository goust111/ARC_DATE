from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

GRID_ROWS_RANGE = (14, 18)
GRID_COLS_RANGE = (14, 18)
MARKER_COUNT_RANGE = (4, 6)
MAX_GENERATION_ATTEMPTS = 400
TEMPLATE_RETRIES = 120
OFFSET_RETRIES = 200
MARKER_PLACEMENT_RETRIES = 120

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_same_color(grid: np.ndarray, bg: int = BLACK) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not _in_bounds((h, w), nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return sorted((r - r0, c - c0) for r, c in coords)


def _parse_input(grid: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _components_same_color(g, bg=BLACK)
    template_candidates = [comp for comp in comps if comp["color"] == RED and comp["area"] > 1]
    assert len(template_candidates) == 1, "expected one multi-pixel red template"
    template = template_candidates[0]
    assert 4 <= template["area"] <= 6, "template must have 4-6 cells"

    tr0, tc0, tr1, tc1 = template["bbox"]
    th = tr1 - tr0 + 1
    tw = tc1 - tc0 + 1
    assert th <= 3 and tw <= 3, "template bbox must be at most 3x3"

    template_rel = _normalize(template["coords"])
    marker_coords: list[tuple[int, int]] = []
    origin: tuple[int, int] | None = None

    for comp in comps:
        if comp is template:
            continue
        assert comp["area"] == 1, "non-template objects must be singleton markers"
        (r, c) = comp["coords"][0]
        color = comp["color"]
        assert color in (RED, GREY), "markers must be red or grey"
        marker_coords.append((r, c))
        if color == RED:
            assert origin is None, "expected one red origin marker"
            origin = (r, c)

    assert origin is not None, "missing red origin marker"
    assert 4 <= len(marker_coords) <= 6, "expected 4-6 markers total"
    assert sum(int(g[r, c] == GREY) for r, c in marker_coords) >= 3, "need at least three grey markers"

    offsets: list[tuple[int, int]] = []
    for r, c in marker_coords:
        dr = r - origin[0]
        dc = c - origin[1]
        assert dr % th == 0 and dc % tw == 0, "marker lattice must match template bbox spacing"
        offsets.append((dr // th, dc // tw))

    assert (0, 0) in offsets
    assert len(set(offsets)) == len(offsets), "marker offsets must be unique"
    return template_rel, (tr0, tc0), (th, tw), offsets


def _render_counts(
    shape: tuple[int, int],
    template_rel: list[tuple[int, int]],
    base: tuple[int, int],
    offsets: list[tuple[int, int]],
) -> np.ndarray:
    counts = np.zeros(shape, dtype=int)
    br, bc = base
    for sr, sc in offsets:
        for tr, tc in template_rel:
            rr = br + sr + tr
            cc = bc + sc + tc
            assert _in_bounds(shape, rr, cc), "stamped copy leaves the grid"
            counts[rr, cc] += 1
    return counts


def _counts_to_output(counts: np.ndarray) -> np.ndarray:
    out = np.zeros_like(counts)
    out[counts == 1] = RED
    out[counts == 2] = BLUE
    out[counts == 3] = YELLOW
    assert int(counts.max()) <= 3, "overlap count must stay within 3"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp several unit-shifted copies of the template and recolor pixels by overlap count.

    Input structure:
    - Black background contains one small red template in the upper-left region.
    - Elsewhere there are 4-6 singleton markers on a lattice: one red origin marker and the
      rest grey.
    - The marker lattice spacing equals the template bbox height vertically and width
      horizontally, so marker deltas encode coarse row/column offsets.

    Rule:
    - Read the template's bbox height and width.
    - For each marker, subtract the red origin marker and divide the raw delta by the bbox
      height/width to recover a coarse offset.
    - Stamp one copy of the template at the template's own top-left shifted by that coarse
      offset in pixels; then count how many copies cover each pixel.
    - Recolor count 1 as red, count 2 as blue, count 3 as yellow, and everything else black.
      The original template display and all markers are erased automatically by this redraw.

    Key insight:
    - The markers do not mark final stamp locations directly; their lattice spacing must first
      be decoded by the template bbox, and those decoded unit offsets are what create the
      overlapping copies.
    """
    template_rel, base, _size, offsets = _parse_input(np.asarray(grid, dtype=int))
    counts = _render_counts(grid.shape, template_rel, base, offsets)
    return _counts_to_output(counts)


def _random_connected_shape(height: int, width: int, area: int) -> list[tuple[int, int]] | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    coords = list(cells)
    r0, c0, r1, c1 = _bbox(coords)
    if r0 != 0 or c0 != 0 or r1 != height - 1 or c1 != width - 1:
        return None
    return sorted(coords)


def _random_template() -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    options = [(2, 2), (2, 3), (3, 2), (3, 3)]
    for _ in range(TEMPLATE_RETRIES):
        th, tw = random.choice(options)
        max_area = min(6, th * tw)
        area = random.randint(4, max_area)
        coords = _random_connected_shape(th, tw, area)
        if coords is not None:
            return coords, (th, tw)
    return None


def _random_offsets(count: int) -> list[tuple[int, int]] | None:
    cells = {(0, 0)}
    limit = 2 if count <= 5 else 3
    for _ in range(OFFSET_RETRIES):
        while len(cells) < count:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if abs(nr) > limit or abs(nc) > limit or (nr, nc) in cells:
                        continue
                    frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) == count:
            return sorted(cells)
        cells = {(0, 0)}
    return None


def _coverage_signature(
    template_rel: list[tuple[int, int]],
    template_size: tuple[int, int],
    offsets: list[tuple[int, int]],
) -> np.ndarray:
    rows = [r for r, _ in offsets]
    cols = [c for _, c in offsets]
    th, tw = template_size
    pad_r = 3 - min(rows)
    pad_c = 3 - min(cols)
    canvas = np.zeros((th + max(rows) - min(rows) + 7, tw + max(cols) - min(cols) + 7), dtype=int)
    for sr, sc in offsets:
        for tr, tc in template_rel:
            canvas[pad_r + sr + tr, pad_c + sc + tc] += 1
    return canvas


def _valid_offset_set(
    template_rel: list[tuple[int, int]],
    template_size: tuple[int, int],
    offsets: list[tuple[int, int]],
    require_triple: bool,
    require_negative: bool,
) -> bool:
    counts = _coverage_signature(template_rel, template_size, offsets)
    max_count = int(counts.max())
    if max_count > 3:
        return False
    if require_triple and max_count != 3:
        return False
    if not require_triple and max_count != 2:
        return False
    if not np.any(counts == 2):
        return False
    if not np.any(counts == 1):
        return False
    if require_triple and not np.any(counts == 3):
        return False
    if require_negative and not any(r < 0 or c < 0 for r, c in offsets):
        return False

    covered = int(np.count_nonzero(counts))
    overlap = int(np.count_nonzero(counts >= 2))
    if not (6 <= covered <= 18):
        return False
    if overlap < 2 or overlap * 3 > covered * 2:
        return False
    return True


def _sample_offsets(
    template_rel: list[tuple[int, int]], template_size: tuple[int, int]
) -> list[tuple[int, int]] | None:
    require_triple = random.random() < 0.6
    require_negative = random.random() < 0.5
    for _ in range(OFFSET_RETRIES):
        count = random.randint(*MARKER_COUNT_RANGE)
        offsets = _random_offsets(count)
        if offsets is None:
            continue
        if _valid_offset_set(template_rel, template_size, offsets, require_triple, require_negative):
            return offsets
    return None


def _disjoint_with_gap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int
) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _place_marker_cluster(
    rows: int,
    cols: int,
    template_size: tuple[int, int],
    offsets: list[tuple[int, int]],
    stamp_bbox: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    th, tw = template_size
    min_r = min(r for r, _ in offsets)
    max_r = max(r for r, _ in offsets)
    min_c = min(c for _, c in offsets)
    max_c = max(c for _, c in offsets)

    r_lo = max(0, -min_r * th)
    r_hi = rows - 1 - max_r * th
    c_lo = max(0, -min_c * tw)
    c_hi = cols - 1 - max_c * tw
    if r_lo > r_hi or c_lo > c_hi:
        return None

    candidates: list[tuple[int, int]] = []
    for origin_r in range(r_lo, r_hi + 1):
        for origin_c in range(c_lo, c_hi + 1):
            cluster_bbox = (
                origin_r + min_r * th,
                origin_c + min_c * tw,
                origin_r + max_r * th,
                origin_c + max_c * tw,
            )
            if not _disjoint_with_gap(cluster_bbox, stamp_bbox, gap=2):
                continue
            markers = [(origin_r + dr * th, origin_c + dc * tw) for dr, dc in offsets]
            if len(set(markers)) != len(markers):
                continue
            if any(not _in_bounds((rows, cols), r, c) for r, c in markers):
                continue
            candidates.append((origin_r, origin_c))

    if not candidates:
        return None

    preferred = [
        pos for pos in candidates if pos[0] >= rows // 2 or pos[1] >= cols // 2
    ]
    return random.choice(preferred or candidates)


def _latent_output(
    shape: tuple[int, int],
    template_rel: list[tuple[int, int]],
    base: tuple[int, int],
    offsets: list[tuple[int, int]],
) -> np.ndarray:
    return _counts_to_output(_render_counts(shape, template_rel, base, offsets))


def _build_input(
    rows: int,
    cols: int,
    template_rel: list[tuple[int, int]],
    template_size: tuple[int, int],
    offsets: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray] | None:
    th, tw = template_size
    min_sr = min(r for r, _ in offsets)
    max_sr = max(r for r, _ in offsets)
    min_sc = min(c for _, c in offsets)
    max_sc = max(c for _, c in offsets)

    r0_lo = max(1, -min_sr)
    c0_lo = max(1, -min_sc)
    r0_hi = min(4, rows - th - max_sr - 1)
    c0_hi = min(4, cols - tw - max_sc - 1)
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    base = (random.randint(r0_lo, r0_hi), random.randint(c0_lo, c0_hi))
    stamp_bbox = (
        base[0] + min_sr,
        base[1] + min_sc,
        base[0] + max_sr + th - 1,
        base[1] + max_sc + tw - 1,
    )
    origin_marker = _place_marker_cluster(rows, cols, template_size, offsets, stamp_bbox)
    if origin_marker is None:
        return None

    inp = np.zeros((rows, cols), dtype=int)
    for tr, tc in template_rel:
        inp[base[0] + tr, base[1] + tc] = RED

    for dr, dc in offsets:
        rr = origin_marker[0] + dr * th
        cc = origin_marker[1] + dc * tw
        inp[rr, cc] = RED if (dr, dc) == (0, 0) else GREY

    expected = _latent_output((rows, cols), template_rel, base, offsets)
    return inp, expected


def _generate_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)

    template = _random_template()
    if template is None:
        return None
    template_rel, template_size = template

    offsets = _sample_offsets(template_rel, template_size)
    if offsets is None:
        return None

    built = _build_input(rows, cols, template_rel, template_size, offsets)
    if built is None:
        return None
    inp, expected = built

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    colors = set(np.unique(actual).tolist())
    if BLUE not in colors:
        return None
    if YELLOW in colors:
        if int((actual == YELLOW).sum()) == 0:
            return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
