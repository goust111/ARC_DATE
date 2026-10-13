import random
from collections import Counter, deque
from typing import Dict, Iterable, List, Tuple

import numpy as np


BG = 0
GREY = 5
ALL_COLORS = tuple(range(10))
PALETTE_COLORS = tuple(c for c in ALL_COLORS if c not in {BG, GREY})
RETRY_LIMIT = 400


def _neighbors4(r: int, c: int) -> Iterable[Tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _label_components(mask: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: List[Dict[str, object]] = []
    next_id = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue

            next_id += 1
            queue = deque([(r, c)])
            labels[r, c] = next_id
            coords: List[Tuple[int, int]] = []
            touches_boundary = False

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_boundary = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        queue.append((nr, nc))

            components.append(
                {
                    "id": next_id,
                    "coords": np.array(coords, dtype=int),
                    "touches_boundary": touches_boundary,
                }
            )

    return labels, components


def _find_legend_bbox(grid: np.ndarray, bg: int) -> Tuple[int, int, int, int]:
    h, w = grid.shape
    assert w >= 3, "legend requires width at least 3"
    candidates: List[Tuple[int, int, int, int]] = []

    def add_candidate(r0: int, c0: int, height: int, right_clear_col: int | None, below_row: int | None) -> None:
        if height <= 0:
            return
        r1 = r0 + height
        c1 = c0 + 3
        if right_clear_col is not None and not np.all(grid[r0:r1, right_clear_col] == bg):
            return
        if below_row is not None and not np.all(grid[below_row, c0:c1] == bg):
            return
        candidates.append((r0, c0, r1, c1))

    if np.all(grid[0, 0:3] != bg):
        height = 0
        while height < h and np.all(grid[height, 0:3] != bg):
            height += 1
        add_candidate(0, 0, height, 3 if w > 3 else None, height if height < h else None)

    if np.all(grid[0, w - 3 : w] != bg):
        height = 0
        while height < h and np.all(grid[height, w - 3 : w] != bg):
            height += 1
        add_candidate(0, w - 3, height, w - 4 if w > 3 else None, height if height < h else None)

    if np.all(grid[h - 1, 0:3] != bg):
        height = 0
        while height < h and np.all(grid[h - 1 - height, 0:3] != bg):
            height += 1
        add_candidate(h - height, 0, height, 3 if w > 3 else None, h - height - 1 if h - height - 1 >= 0 else None)

    if np.all(grid[h - 1, w - 3 : w] != bg):
        height = 0
        while height < h and np.all(grid[h - 1 - height, w - 3 : w] != bg):
            height += 1
        add_candidate(
            h - height,
            w - 3,
            height,
            w - 4 if w > 3 else None,
            h - height - 1 if h - height - 1 >= 0 else None,
        )

    assert len(candidates) == 1, "expected exactly one corner legend"
    return candidates[0]


def _parse_regions(grid: np.ndarray) -> Dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    legend_bbox = _find_legend_bbox(g, bg)
    r0, c0, r1, c1 = legend_bbox
    legend = g[r0:r1, c0:c1]
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[r0:r1, c0:c1] = True

    mapping: Dict[frozenset[int], int] = {}
    for row in legend:
        a, b, fill = map(int, row)
        key = frozenset((a, b))
        assert len(key) == 2, "legend rows must contain two different fence colors"
        assert key not in mapping, "legend color pairs must be unique"
        mapping[key] = fill

    active_mask = ~legend_mask
    remaining = [int(v) for v in g[active_mask].flat if int(v) != bg]
    counts = Counter(remaining)
    assert counts, "grid must contain fence pixels outside the legend"
    grey = max(counts, key=counts.get)

    anchor_colors = sorted(color for color in counts if color != grey)
    assert anchor_colors, "expected anchor colors outside the legend"
    for color in anchor_colors:
        count = int(np.count_nonzero(active_mask & (g == color)))
        assert count == 2, "each anchor color must appear exactly twice outside the legend"

    anchor_mask = np.isin(g, np.array(anchor_colors, dtype=int))
    fence_mask = active_mask & ((g == grey) | anchor_mask)
    fence_labels, fence_components = _label_components(fence_mask)
    assert len(fence_components) == len(anchor_colors), "each fence component must have one anchor pair"

    component_color: Dict[int, int] = {}
    for comp in fence_components:
        coords = comp["coords"]
        colors = [int(g[r, c]) for r, c in coords if int(g[r, c]) != grey]
        unique = set(colors)
        assert len(unique) == 1 and len(colors) == 2, "each fence must carry exactly one anchor color pair"
        component_color[int(comp["id"])] = colors[0]

    bg_mask = active_mask & (g == bg)
    _, bg_components = _label_components(bg_mask)
    closed_regions: List[Dict[str, object]] = []

    h, w = g.shape
    for comp in bg_components:
        if bool(comp["touches_boundary"]):
            continue
        touching_ids = set()
        for r, c in comp["coords"]:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and fence_labels[nr, nc] > 0:
                    touching_ids.add(int(fence_labels[nr, nc]))
        assert len(touching_ids) == 2, "each enclosed region must touch exactly two fences"
        pair = frozenset(component_color[idx] for idx in touching_ids)
        assert pair in mapping, "legend must cover every fence-color pair used by a region"
        closed_regions.append({"coords": comp["coords"], "fill": mapping[pair]})

    return {
        "bg": bg,
        "grey": grey,
        "legend_bbox": legend_bbox,
        "mapping": mapping,
        "regions": closed_regions,
        "fence_count": len(fence_components),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fence-enclosed background band using the legend entry for the two surrounding anchor colors.

    Input structure:
    - The background is black.
    - Away from one corner legend, grey fence components each carry exactly two single-pixel anchors of one shared color.
    - The fences form nested enclosed bands of black background, and the corner legend is an N x 3 table whose first two columns show fence-anchor color pairs and whose third column shows the fill color for the region between them.

    Rule:
    - Locate the corner legend and read each row as an unordered pair-to-color mapping.
    - Treat grey pixels together with the non-legend anchors as fence components, and assign each component the color of its two anchors.
    - Find every black background component that is fully enclosed and determine which two fence components border it.
    - Fill that enclosed component with the legend color for the bordering fence-color pair, while leaving the fences, anchors, and legend unchanged.

    Key insight:
    - The region color is not attached to one fence alone; it is obtained only after identifying the two surrounding fence colors and using their combination as a legend lookup key.
    """
    g = np.asarray(grid, dtype=int)
    parsed = _parse_regions(g)
    out = g.copy()
    for region in parsed["regions"]:
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = int(region["fill"])
    return out


def _boxes_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _component_boxes(rng: random.Random, count: int) -> List[Tuple[int, int, int, int]]:
    shape = rng.choice(("hbar", "vbar", "block"))
    if shape == "hbar":
        inner_h, inner_w = 2, rng.randint(3, 6)
    elif shape == "vbar":
        inner_h, inner_w = rng.randint(3, 6), 2
    else:
        inner_h, inner_w = 2, 2

    boxes = [(0, 0, inner_h - 1, inner_w - 1)]
    for _ in range(count - 1):
        prev = boxes[-1]
        top = rng.randint(2, 4)
        bottom = rng.randint(2, 4)
        left = rng.randint(2, 4)
        right = rng.randint(2, 4)
        boxes.append((prev[0] - top, prev[1] - left, prev[2] + bottom, prev[3] + right))

    outer = boxes[-1]
    shift_r = -outer[0]
    shift_c = -outer[1]
    ordered = []
    for box in reversed(boxes):
        ordered.append((box[0] + shift_r, box[1] + shift_c, box[2] + shift_r, box[3] + shift_c))
    return ordered


def _perimeter_cells(box: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: List[Tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        cells.append((r, c1))
    if r1 > r0:
        for c in range(c1 - 1, c0 - 1, -1):
            cells.append((r1, c))
    if c1 > c0:
        for r in range(r1 - 1, r0, -1):
            cells.append((r, c0))
    return cells


def _choose_anchor_cells(rng: random.Random, cells: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    n = len(cells)
    i = rng.randrange(n)
    if n <= 4:
        j = (i + n // 2) % n
        if j == i:
            j = (i + 1) % n
        return cells[i], cells[j]

    min_gap = max(2, n // 4)
    candidates = [idx for idx in range(n) if min((idx - i) % n, (i - idx) % n) >= min_gap]
    j = rng.choice(candidates)
    return cells[i], cells[j]


def _draw_component(
    grid: np.ndarray,
    box: Tuple[int, int, int, int],
    fence_color: int,
    rng: random.Random,
) -> None:
    cells = _perimeter_cells(box)
    for r, c in cells:
        grid[r, c] = GREY
    a, b = _choose_anchor_cells(rng, cells)
    grid[a] = fence_color
    grid[b] = fence_color


def _legend_origin(corner: str, h: int, w: int, legend_h: int) -> Tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - 3
    if corner == "bl":
        return h - legend_h, 0
    return h - legend_h, w - 3


def _legend_padding_box(corner: str, h: int, w: int, legend_h: int) -> Tuple[int, int, int, int]:
    r0, c0 = _legend_origin(corner, h, w, legend_h)
    r1 = r0 + legend_h - 1
    c1 = c0 + 2
    return max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r1 + 1), min(w - 1, c1 + 1)


def _build_legend_rows(
    fence_colors: List[int],
    fill_colors: List[int],
    rng: random.Random,
) -> np.ndarray:
    rows = []
    for idx, fill in enumerate(fill_colors):
        a = fence_colors[idx]
        b = fence_colors[idx + 1]
        if rng.random() < 0.5:
            rows.append([a, b, fill])
        else:
            rows.append([b, a, fill])
    rng.shuffle(rows)
    return np.array(rows, dtype=int)


def _place_legend(grid: np.ndarray, legend: np.ndarray, corner: str) -> None:
    h, w = grid.shape
    legend_h = legend.shape[0]
    r0, c0 = _legend_origin(corner, h, w, legend_h)
    grid[r0 : r0 + legend_h, c0 : c0 + 3] = legend


def _candidate_positions(
    grid_h: int,
    grid_w: int,
    outer_h: int,
    outer_w: int,
    blocked: Tuple[int, int, int, int],
) -> List[Tuple[int, int]]:
    positions = []
    for r0 in range(1, grid_h - outer_h):
        for c0 in range(1, grid_w - outer_w):
            box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if not _boxes_overlap(box, blocked):
                positions.append((r0, c0))
    return positions


def _translate_boxes(
    boxes: List[Tuple[int, int, int, int]],
    origin: Tuple[int, int],
) -> List[Tuple[int, int, int, int]]:
    r_off, c_off = origin
    return [(r0 + r_off, c0 + c_off, r1 + r_off, c1 + c_off) for r0, c0, r1, c1 in boxes]


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    fence_count = rng.randint(2, 4)
    boxes = _component_boxes(rng, fence_count)
    outer_h = boxes[0][2] - boxes[0][0] + 1
    outer_w = boxes[0][3] - boxes[0][1] + 1
    legend_h = fence_count - 1
    corner = rng.choice(("tl", "tr", "bl", "br"))

    min_h = max(outer_h + 3, legend_h + 3, 10)
    min_w = max(outer_w + 3, 8)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)
    blocked = _legend_padding_box(corner, grid_h, grid_w, legend_h)
    positions = _candidate_positions(grid_h, grid_w, outer_h, outer_w, blocked)
    if not positions:
        return None

    fence_color_count = fence_count
    fill_color_count = fence_count - 1
    if fence_color_count + fill_color_count > len(PALETTE_COLORS):
        return None
    sampled = rng.sample(PALETTE_COLORS, fence_color_count + fill_color_count)
    fence_colors = sampled[:fence_color_count]
    fill_colors = sampled[fence_color_count:]

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    origin = rng.choice(positions)
    placed_boxes = _translate_boxes(boxes, origin)
    for box, color in zip(placed_boxes, fence_colors):
        _draw_component(grid, box, color, rng)

    legend = _build_legend_rows(fence_colors, fill_colors, rng)
    _place_legend(grid, legend, corner)

    if np.count_nonzero(grid == BG) <= grid.size // 2:
        return None

    try:
        parsed = _parse_regions(grid)
    except AssertionError:
        return None

    if int(parsed["fence_count"]) != fence_count:
        return None
    if len(parsed["regions"]) != fence_count - 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(RETRY_LIMIT):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-fence input")
