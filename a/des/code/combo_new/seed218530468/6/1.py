import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ("horizontal", "vertical", "diag_up", "diag_down")
LEGEND_WIDTHS = {
    "horizontal": 3,
    "vertical": 1,
    "diag_up": 3,
    "diag_down": 3,
}
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        raise ValueError(f"unsupported connectivity: {connectivity}")

    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(pts)
    return parts


def _decode_legend_component(points: list[tuple[int, int]]) -> str:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    if len(points) != 3:
        raise ValueError("legend components must contain exactly three cells")
    if len(set(rows)) == 1 and sorted(cols) == list(range(min(cols), max(cols) + 1)):
        return "horizontal"
    if len(set(cols)) == 1 and sorted(rows) == list(range(min(rows), max(rows) + 1)):
        return "vertical"
    if len(set(r + c for r, c in points)) == 1 and len(set(rows)) == 3 and len(set(cols)) == 3:
        return "diag_up"
    if len(set(r - c for r, c in points)) == 1 and len(set(rows)) == 3 and len(set(cols)) == 3:
        return "diag_down"
    raise ValueError(f"unrecognized legend geometry: {points}")


def _parse_legend_directions(grid: np.ndarray) -> dict[int, str]:
    legend_dirs: dict[int, str] = {}
    for color in map(int, np.unique(grid)):
        if color in (BLACK, GREY):
            continue
        comps = _components(grid == color, connectivity=8)
        for comp in comps:
            if len(comp) != 3:
                continue
            direction = _decode_legend_component(comp)
            if color in legend_dirs:
                raise ValueError(f"duplicate legend for color {color}")
            legend_dirs[color] = direction
    if not legend_dirs:
        raise ValueError("legend not found")
    return legend_dirs


def _ring_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 3 or c1 - c0 < 3:
        raise ValueError("grey component is too small to be a rectangular ring")

    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))
    actual = set(points)
    if actual != expected:
        raise ValueError("grey component is not a single-pixel rectangular ring")
    return r0, r1, c0, c1


def _stripe_mask(shape: tuple[int, int], direction: str) -> np.ndarray:
    rr, cc = np.indices(shape)
    if direction == "horizontal":
        return rr % 2 == 0
    if direction == "vertical":
        return cc % 2 == 0
    if direction == "diag_up":
        return (rr + cc) % 2 == 0
    if direction == "diag_down":
        return (rr - cc) % 2 == 0
    raise ValueError(f"unknown direction: {direction}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-bounded room with marker-colored stripes whose orientation is decoded from the matching legend segment.

    Input structure:
    - The background is black.
    - Several single-pixel-thick grey rectangular rings enclose black interiors, and each interior contains exactly one colored marker pixel.
    - Near the bottom edge, each marker color also appears as a 3-pixel same-color legend segment whose orientation is horizontal, vertical, rising diagonal, or falling diagonal.

    Rule:
    - For each marker color, find its 3-pixel legend segment and decode the stripe direction from that segment's orientation.
    - For each grey rectangle, read the unique marker color inside its interior and look up the matching legend direction.
    - Fill every other row, column, or diagonal of the interior starting from the interior's top-left cell; keep the grey border and the legend unchanged.

    Key insight:
    - The colored marker only identifies which legend entry to use; the stripe orientation itself must be decoded from the geometry of the matching legend segment.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    legend_dirs = _parse_legend_directions(g)
    for comp in _components(g == GREY, connectivity=4):
        r0, r1, c0, c1 = _ring_bbox(comp)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_coords = np.argwhere((inner != BLACK) & (inner != GREY))
        if marker_coords.shape[0] != 1:
            raise ValueError("each region interior must contain exactly one marker")
        mr, mc = map(int, marker_coords[0])
        color = int(inner[mr, mc])
        if color not in legend_dirs:
            raise ValueError(f"missing legend entry for color {color}")

        mask = _stripe_mask(inner.shape, legend_dirs[color])
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        patch[mask] = color
    return out


def _legend_cells(direction: str, start_col: int, height: int) -> list[tuple[int, int]]:
    bottom = height - 1
    if direction == "horizontal":
        return [(bottom, start_col + dc) for dc in range(3)]
    if direction == "vertical":
        return [(bottom - dr, start_col) for dr in range(3)]
    if direction == "diag_up":
        return [(bottom - step, start_col + step) for step in range(3)]
    if direction == "diag_down":
        return [(bottom - 2 + step, start_col + step) for step in range(3)]
    raise ValueError(f"unknown direction: {direction}")


def _paint_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, r1, c0, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _choose_marker(inner_shape: tuple[int, int], direction: str) -> tuple[int, int]:
    stripe = _stripe_mask(inner_shape, direction)
    on_cells = [tuple(map(int, rc)) for rc in np.argwhere(stripe)]
    off_cells = [tuple(map(int, rc)) for rc in np.argwhere(~stripe)]
    want_on = random.random() < 0.5
    if want_on and on_cells:
        return random.choice(on_cells)
    if (not want_on) and off_cells:
        return random.choice(off_cells)
    return random.choice(on_cells if on_cells else off_cells)


def _region_layout() -> tuple[np.ndarray, list[dict]]:
    n_regions = random.randint(4, 5)
    directions = list(DIRS)
    random.shuffle(directions)
    if n_regions == 5:
        directions.append(random.choice(DIRS))
    random.shuffle(directions)
    colors = random.sample(COLOR_POOL, n_regions)

    row_gap = random.randint(1, 2)
    col_gap = 1
    slot_heights = [random.randint(6, 9) for _ in range(2)]
    slot_widths = [random.randint(6, 8) for _ in range(3)]

    height = row_gap + slot_heights[0] + row_gap + slot_heights[1] + row_gap + 1 + 3
    width = col_gap + slot_widths[0] + col_gap + slot_widths[1] + col_gap + slot_widths[2] + col_gap
    grid = np.full((height, width), BLACK, dtype=int)

    row_starts = [
        row_gap,
        row_gap + slot_heights[0] + row_gap,
    ]
    col_starts = [
        col_gap,
        col_gap + slot_widths[0] + col_gap,
        col_gap + slot_widths[0] + col_gap + slot_widths[1] + col_gap,
    ]

    slot_positions = [(r, c) for r in range(2) for c in range(3)]
    chosen_slots = random.sample(slot_positions, n_regions)
    random.shuffle(chosen_slots)

    regions: list[dict] = []
    for (slot_r, slot_c), color, direction in zip(chosen_slots, colors, directions):
        slot_h = slot_heights[slot_r]
        slot_w = slot_widths[slot_c]
        outer_h = random.randint(4, slot_h - 2)
        outer_w = random.randint(4, slot_w - 2)
        off_r = random.randint(1, slot_h - outer_h - 1)
        off_c = random.randint(1, slot_w - outer_w - 1)

        r0 = row_starts[slot_r] + off_r
        c0 = col_starts[slot_c] + off_c
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        bbox = (r0, r1, c0, c1)
        _paint_ring(grid, bbox, GREY)

        inner_shape = (outer_h - 2, outer_w - 2)
        mr, mc = _choose_marker(inner_shape, direction)
        grid[r0 + 1 + mr, c0 + 1 + mc] = color
        regions.append(
            {
                "bbox": bbox,
                "color": color,
                "direction": direction,
            }
        )

    legend_order = regions[:]
    random.shuffle(legend_order)
    widths = [LEGEND_WIDTHS[region["direction"]] for region in legend_order]
    gaps = [random.randint(1, 2) for _ in range(max(0, len(legend_order) - 1))]
    used = sum(widths) + sum(gaps)
    extra = width - used
    if extra < 2:
        raise ValueError("legend does not fit")
    left_margin = random.randint(1, extra - 1)

    col = left_margin
    for idx, region in enumerate(legend_order):
        for r, c in _legend_cells(region["direction"], col, height):
            grid[r, c] = region["color"]
        col += LEGEND_WIDTHS[region["direction"]]
        if idx < len(gaps):
            col += gaps[idx]

    return grid, regions


def _expected_output(grid: np.ndarray, regions: list[dict]) -> np.ndarray:
    out = grid.copy()
    for region in regions:
        r0, r1, c0, c1 = region["bbox"]
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        patch[_stripe_mask(patch.shape, region["direction"])] = region["color"]
    return out


def _valid_input(grid: np.ndarray, regions: list[dict]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    try:
        legend_dirs = _parse_legend_directions(grid)
        grey_parts = _components(grid == GREY, connectivity=4)
        if len(grey_parts) != len(regions):
            return False
        if set(legend_dirs) != {region["color"] for region in regions}:
            return False
        bboxes = set()
        for comp in grey_parts:
            bboxes.add(_ring_bbox([(int(r), int(c)) for r, c in comp]))
        for region in regions:
            if legend_dirs.get(region["color"]) != region["direction"]:
                return False
            if region["bbox"] not in bboxes:
                return False
            r0, r1, c0, c1 = region["bbox"]
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            if np.count_nonzero((inner != BLACK) & (inner != GREY)) != 1:
                return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    try:
        grid, regions = _region_layout()
    except Exception:
        return None
    if not _valid_input(grid, regions):
        return None
    expected = _expected_output(grid, regions)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
