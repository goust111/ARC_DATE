import itertools
import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4)
ALL_DIRECTIONS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)
PATTERNS = {
    3: (("h", "v"), ("v", "h")),
    4: (("h", "v", "h"), ("v", "h", "v")),
}
OFFSET_CHOICES = (-2, -1, 0, 1, 2)
GAP_CHOICES = (3, 4, 5)
MARGIN_CHOICES = (2, 3, 4, 5)


def _bg_color(grid: np.ndarray) -> int:
    values = grid.astype(int).ravel()
    counts = np.bincount(values, minlength=max(11, int(values.max()) + 1))
    return int(np.argmax(counts))


def _sign(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _overlap_1d(a0: int, a1: int, b0: int, b1: int) -> tuple[int, int] | None:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return (lo, hi) if lo <= hi else None


def _find_anchors(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    anchors: list[dict] = []
    seen = set()
    h, w = g.shape
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            color = int(g[r0, c0])
            if color == bg:
                continue
            if not np.all(g[r0 : r0 + 3, c0 : c0 + 3] == color):
                continue
            key = (r0, c0, color)
            if key in seen:
                continue
            seen.add(key)
            anchors.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "r1": r0 + 2,
                    "c1": c0 + 2,
                    "cr": r0 + 1,
                    "cc": c0 + 1,
                }
            )
    anchors.sort(key=lambda anchor: (anchor["r0"], anchor["c0"]))
    return anchors


def _direction_from_center(anchor: dict, r: int, c: int) -> tuple[int, int] | None:
    dr = r - anchor["cr"]
    dc = c - anchor["cc"]
    if dr == 0 and dc != 0:
        return 0, _sign(dc)
    if dc == 0 and dr != 0:
        return _sign(dr), 0
    if abs(dr) == abs(dc) and dr != 0:
        return _sign(dr), _sign(dc)
    return None


def _compress_state(grid: np.ndarray) -> tuple[np.ndarray, list[dict], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    anchors = _find_anchors(g)
    out = np.full_like(g, bg)
    densities: dict[int, int] = {}

    for anchor in anchors:
        out[anchor["r0"] : anchor["r1"] + 1, anchor["c0"] : anchor["c1"] + 1] = anchor["color"]

    for anchor in anchors:
        color = anchor["color"]
        groups: dict[tuple[int, int], list[int]] = {}
        for raw_r, raw_c in np.argwhere(g == color):
            r, c = int(raw_r), int(raw_c)
            if anchor["r0"] <= r <= anchor["r1"] and anchor["c0"] <= c <= anchor["c1"]:
                continue
            direction = _direction_from_center(anchor, r, c)
            assert direction is not None, "loose pixel must lie on a center-based 8-direction ray"
            dist = max(abs(r - anchor["cr"]), abs(c - anchor["cc"]))
            groups.setdefault(direction, []).append(dist)

        densities[color] = sum(len(dists) for dists in groups.values())
        for (dr, dc), dists in groups.items():
            for idx, _dist in enumerate(sorted(dists)):
                rr = anchor["cr"] + (idx + 2) * dr
                cc = anchor["cc"] + (idx + 2) * dc
                out[rr, cc] = color

    return out, anchors, densities


def _corridor_color(a: dict, b: dict, densities: dict[int, int]) -> int:
    da = densities[a["color"]]
    db = densities[b["color"]]
    if da > db:
        return a["color"]
    if db > da:
        return b["color"]
    if (a["r0"], a["c0"]) <= (b["r0"], b["c0"]):
        return a["color"]
    return b["color"]


def _visible_pairs(grid: np.ndarray, anchors: list[dict], densities: dict[int, int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    pairs: list[dict] = []
    for i, a in enumerate(anchors):
        for b in anchors[i + 1 :]:
            row_band = _overlap_1d(a["r0"], a["r1"], b["r0"], b["r1"])
            if row_band is not None and (a["c1"] < b["c0"] or b["c1"] < a["c0"]):
                left, right = (a, b) if a["c0"] < b["c0"] else (b, a)
                c0 = left["c1"] + 1
                c1 = right["c0"] - 1
                r0, r1 = row_band
                if c0 <= c1 and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == bg):
                    pairs.append(
                        {
                            "a": a,
                            "b": b,
                            "axis": "h",
                            "rect": (r0, c0, r1, c1),
                            "color": _corridor_color(a, b, densities),
                        }
                    )
                    continue

            col_band = _overlap_1d(a["c0"], a["c1"], b["c0"], b["c1"])
            if col_band is not None and (a["r1"] < b["r0"] or b["r1"] < a["r0"]):
                top, bottom = (a, b) if a["r0"] < b["r0"] else (b, a)
                r0 = top["r1"] + 1
                r1 = bottom["r0"] - 1
                c0, c1 = col_band
                if r0 <= r1 and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == bg):
                    pairs.append(
                        {
                            "a": a,
                            "b": b,
                            "axis": "v",
                            "rect": (r0, c0, r1, c1),
                            "color": _corridor_color(a, b, densities),
                        }
                    )
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each anchor's scattered ray pixels inward and color visible inter-anchor corridors by the denser anchor.

    Input structure:
    - The grid has a black background with 3 or 4 solid 3x3 anchor blocks in distinct colors.
    - Each anchor color also appears as loose single pixels placed on straight center-based horizontal, vertical, or diagonal rays from that anchor.
    - Anchors may overlap in row-band or column-band alignment, and loose pixels stay off the anchor blocks themselves.

    Rule:
    - For each anchor, erase its loose pixels from their scattered positions and repack them inward along the same ray directions so they form tight contiguous segments starting next to the 3x3 block.
    - Use the compressed loose-pixel count of each anchor as that anchor's density value.
    - For every pair of anchors whose 3x3 row bands or column bands overlap and whose entire gap strip is still background after compression, fill that gap strip as a corridor.
    - Color each corridor with the color of the denser anchor in that pair; if densities tie, use the anchor with the smaller row index, then smaller column index.

    Key insight:
    - The corridor color is decided only after the loose pixels are compressed, because the compressed attachment count supplies the per-anchor density that dominates each visible anchor pair.
    """
    compressed, anchors, densities = _compress_state(np.asarray(grid, dtype=int))
    out = compressed.copy()
    bg = _bg_color(out)
    for pair in _visible_pairs(compressed, anchors, densities):
        r0, c0, r1, c1 = pair["rect"]
        window = out[r0 : r1 + 1, c0 : c1 + 1]
        window[window == bg] = pair["color"]
    return out


def _build_chain_positions(anchor_count: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    pattern = random.choice(PATTERNS[anchor_count])
    positions = [(0, 0)]
    for axis in pattern:
        gap = random.choice(GAP_CHOICES)
        offset = random.choice(OFFSET_CHOICES)
        direction = random.choice((-1, 1))
        r, c = positions[-1]
        if axis == "h":
            positions.append((r + offset, c + direction * (gap + 3)))
        else:
            positions.append((r + direction * (gap + 3), c + offset))
    edges = [(idx, idx + 1) for idx in range(len(positions) - 1)]
    return positions, edges


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _anchor_bbox(pos: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = pos
    return r0, c0, r0 + 2, c0 + 2


def _layout_ok(positions: list[tuple[int, int]]) -> bool:
    bboxes = [_anchor_bbox(pos) for pos in positions]
    for i, a in enumerate(bboxes):
        for b in bboxes[i + 1 :]:
            if _rect_overlap(a, b):
                return False
    return True


def _edge_rect(a: tuple[int, int], b: tuple[int, int]) -> tuple[str, tuple[int, int, int, int]] | None:
    ar0, ac0, ar1, ac1 = _anchor_bbox(a)
    br0, bc0, br1, bc1 = _anchor_bbox(b)
    row_band = _overlap_1d(ar0, ar1, br0, br1)
    if row_band is not None and (ac1 < bc0 or bc1 < ac0):
        left, right = ((ar0, ac0, ar1, ac1), (br0, bc0, br1, bc1)) if ac0 < bc0 else ((br0, bc0, br1, bc1), (ar0, ac0, ar1, ac1))
        return "h", (row_band[0], left[3] + 1, row_band[1], right[1] - 1)
    col_band = _overlap_1d(ac0, ac1, bc0, bc1)
    if col_band is not None and (ar1 < br0 or br1 < ar0):
        top, bottom = ((ar0, ac0, ar1, ac1), (br0, bc0, br1, bc1)) if ar0 < br0 else ((br0, bc0, br1, bc1), (ar0, ac0, ar1, ac1))
        return "v", (top[2] + 1, col_band[0], bottom[0] - 1, col_band[1])
    return None


def _terminal_for_direction(pos: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = pos
    dr, dc = direction
    return r0 + 1 + 2 * dr, c0 + 1 + 2 * dc


def _forbidden_directions(
    positions: list[tuple[int, int]], edges: list[tuple[int, int]]
) -> list[set[tuple[int, int]]]:
    forbidden = [set() for _ in positions]
    for i, j in edges:
        edge = _edge_rect(positions[i], positions[j])
        if edge is None:
            continue
        _axis, rect = edge
        r0, c0, r1, c1 = rect
        for anchor_idx in (i, j):
            for direction in ALL_DIRECTIONS:
                tr, tc = _terminal_for_direction(positions[anchor_idx], direction)
                if r0 <= tr <= r1 and c0 <= tc <= c1:
                    forbidden[anchor_idx].add(direction)
    return forbidden


def _assign_counts(available_sizes: list[int]) -> list[int] | None:
    choices = []
    count_ids = list(range(len(available_sizes)))
    for count_tuple in itertools.permutations((3, 4, 5, 6), len(available_sizes)):
        if all(count_tuple[idx] <= available_sizes[idx] for idx in count_ids):
            choices.append(count_tuple)
    if not choices:
        return None
    return list(random.choice(choices))


def _choose_dirs_and_counts(
    positions: list[tuple[int, int]], edges: list[tuple[int, int]]
) -> tuple[list[list[tuple[int, int]]], list[int]] | None:
    forbidden = _forbidden_directions(positions, edges)
    available = [sorted(set(ALL_DIRECTIONS) - blocked) for blocked in forbidden]
    if any(len(options) < 3 for options in available):
        return None
    counts = _assign_counts([len(options) for options in available])
    if counts is None:
        return None
    directions = [random.sample(options, k=count) for options, count in zip(available, counts)]
    return directions, counts


def _shifted_positions(
    positions: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int, int] | None:
    min_r = min(r for r, _c in positions)
    max_r = max(r for r, _c in positions)
    min_c = min(c for _r, c in positions)
    max_c = max(c for _r, c in positions)
    span_h = max_r - min_r + 3
    span_w = max_c - min_c + 3

    max_extra_h = 30 - span_h
    max_extra_w = 30 - span_w
    if max_extra_h < 4 or max_extra_w < 4:
        return None

    top = random.choice(MARGIN_CHOICES)
    left = random.choice(MARGIN_CHOICES)
    bottom = random.choice(MARGIN_CHOICES)
    right = random.choice(MARGIN_CHOICES)
    if top + bottom > max_extra_h or left + right > max_extra_w:
        return None

    shifted = [(r - min_r + top, c - min_c + left) for r, c in positions]
    height = span_h + top + bottom
    width = span_w + left + right
    return shifted, height, width


def _ray_distance_limit(height: int, width: int, pos: tuple[int, int], direction: tuple[int, int]) -> int:
    cr = pos[0] + 1
    cc = pos[1] + 1
    dr, dc = direction
    limits = []
    if dr > 0:
        limits.append(height - 1 - cr)
    elif dr < 0:
        limits.append(cr)
    if dc > 0:
        limits.append(width - 1 - cc)
    elif dc < 0:
        limits.append(cc)
    if not limits:
        return 0
    return min(limits)


def _place_rays(
    grid: np.ndarray,
    positions: list[tuple[int, int]],
    colors: list[int],
    directions: list[list[tuple[int, int]]],
) -> bool:
    h, w = grid.shape
    for pos, color, dirs in zip(positions, colors, directions):
        cr = pos[0] + 1
        cc = pos[1] + 1
        for direction in dirs:
            dr, dc = direction
            max_dist = min(6, _ray_distance_limit(h, w, pos, direction))
            candidates = list(range(3, max_dist + 1))
            random.shuffle(candidates)
            placed = False
            for dist in candidates:
                rr = cr + dist * dr
                cc2 = cc + dist * dc
                path_clear = True
                for step in range(2, dist + 1):
                    pr = cr + step * dr
                    pc = cc + step * dc
                    if grid[pr, pc] != BG:
                        path_clear = False
                        break
                if not path_clear:
                    continue
                grid[rr, cc2] = color
                placed = True
                break
            if not placed:
                return False
    return True


def _corridors_non_overlapping(pairs: list[dict]) -> bool:
    rects = [pair["rect"] for pair in pairs]
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            if _rect_overlap(a, b):
                return False
    return True


def _intended_edge_set(colors: list[int], edges: list[tuple[int, int]]) -> set[frozenset[int]]:
    return {frozenset((colors[i], colors[j])) for i, j in edges}


def generate_input() -> np.ndarray:
    for _ in range(400):
        anchor_count = random.choice((3, 4))
        raw_positions, edges = _build_chain_positions(anchor_count)
        if not _layout_ok(raw_positions):
            continue

        picked = _choose_dirs_and_counts(raw_positions, edges)
        if picked is None:
            continue
        directions, counts = picked
        if len(set(counts)) != anchor_count:
            continue

        shifted = _shifted_positions(raw_positions)
        if shifted is None:
            continue
        positions, height, width = shifted
        if max(height, width) > 30:
            continue

        colors = random.sample(list(PALETTE), k=anchor_count)
        grid = np.zeros((height, width), dtype=int)
        for pos, color in zip(positions, colors):
            r0, c0 = pos
            grid[r0 : r0 + 3, c0 : c0 + 3] = color

        if not _place_rays(grid, positions, colors, directions):
            continue

        compressed, anchors, densities = _compress_state(grid)
        if len(anchors) != anchor_count:
            continue
        if sorted(densities.values()) != sorted(counts):
            continue

        pairs = _visible_pairs(compressed, anchors, densities)
        actual_edges = {
            frozenset((pair["a"]["color"], pair["b"]["color"]))
            for pair in pairs
        }
        if actual_edges != _intended_edge_set(colors, edges):
            continue
        if len(actual_edges) < 2:
            continue
        if not _corridors_non_overlapping(pairs):
            continue

        out = main(grid)
        if np.array_equal(out, grid):
            continue
        if np.count_nonzero(out != compressed) == 0:
            continue

        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 3")
