from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    for _parent in Path(__file__).resolve().parents:
        _candidate = _parent / "new" / "solvers"
        if _candidate.is_dir():
            if str(_candidate) not in sys.path:
                sys.path.append(str(_candidate))
            break
    from arc_utils import bbox_coords, colors_present, get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


    def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
        colors = set(map(int, np.unique(grid)))
        if exclude is not None:
            colors -= set(map(int, exclude))
        return sorted(colors)


    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG_COLOR = 0
GEN_ATTEMPTS = 400
PAIR_COUNT_OPTIONS = (3, 4)
SLOT_SIDE_MIN = 7
SLOT_SIDE_MAX = 11
RECT_SIDE_MIN = 5
RECT_SIDE_MAX = 7
OUTER_MARGIN_OPTIONS = (1, 2)
GAP_OPTIONS = (2, 3)
PALETTE = tuple(range(1, 10))
BASE_STATUSES = ("open_interior", "blocked", "open_onecorridor")
EXTRA_STATUS_OPTIONS = ("open_clean", "open_interior", "blocked", "open_onecorridor")


def _axis_segment(a: Coord, b: Coord) -> list[Coord]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br >= ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise AssertionError("axis segment endpoints must share a row or a column")


def _l_corridors(a: Coord, b: Coord) -> tuple[list[Coord], list[Coord]]:
    ar, ac = a
    br, bc = b
    assert ar != br and ac != bc, "pair markers must be diagonal corners"
    via_hv = (ar, bc)
    via_vh = (br, ac)
    first = _axis_segment(a, via_hv) + _axis_segment(via_hv, b)[1:]
    second = _axis_segment(a, via_vh) + _axis_segment(via_vh, b)[1:]
    return first, second


def _corridor_clear(grid: np.ndarray, obstacle_color: int, path: list[Coord]) -> bool:
    return all(int(grid[r, c]) != obstacle_color for r, c in path)


def _collect_pairs(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    bg = int(get_bg(grid))
    pair_entries: list[tuple[int, np.ndarray]] = []
    obstacle_colors: list[int] = []

    for color in colors_present(grid, exclude={bg}):
        coords = np.argwhere(grid == color)
        rows = set(map(int, coords[:, 0]))
        cols = set(map(int, coords[:, 1]))
        if coords.shape[0] == 2 and len(rows) == 2 and len(cols) == 2:
            pair_entries.append((int(color), coords))
        else:
            obstacle_colors.append(int(color))

    assert len(obstacle_colors) == 1, "expected exactly one obstacle color"
    obstacle_color = obstacle_colors[0]
    assert pair_entries, "expected at least one marker pair"

    pairs = []
    for color, coords in pair_entries:
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        bbox = bbox_coords(coords)
        corridor_a, corridor_b = _l_corridors(a, b)
        clear_a = _corridor_clear(grid, obstacle_color, corridor_a)
        clear_b = _corridor_clear(grid, obstacle_color, corridor_b)
        pairs.append(
            {
                "color": int(color),
                "coords": (a, b),
                "bbox": bbox,
                "open": bool(clear_a or clear_b),
                "corridors": (bool(clear_a), bool(clear_b)),
            }
        )

    pairs.sort(key=lambda item: (item["bbox"], item["color"]))
    return bg, obstacle_color, pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each diagonal marker pair's bounding box only when an unobstructed border corridor connects the two corners.

    Input structure:
    - The grid has one dominant background color, one obstacle color, and several
      other colors that each appear exactly twice as diagonal corner markers.
    - The obstacle pixels form straight barrier segments.
    - Each marker pair defines an axis-aligned bounding box whose interior may or
      may not contain obstacle pixels.

    Rule:
    - For each same-color marker pair, inspect the two L-shaped border corridors
      between the opposite corners: horizontal-then-vertical and vertical-then-horizontal.
    - If at least one corridor avoids the obstacle color entirely, recolor every
      background cell in that whole bounding box to the pair color while keeping
      obstacle pixels unchanged.
    - If both corridors are blocked by obstacle pixels, erase the two corner
      markers back to the background; obstacles never change.

    Key insight:
    - Visibility is tested only on the bounding-box border routes, not through the
      box interior, so interior barriers can remain visible inside a successful fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, _obstacle_color, pairs = _collect_pairs(g)
    input_bg = g == bg
    out = g.copy()

    for pair in pairs:
        if pair["open"]:
            r0, c0, r1, c1 = pair["bbox"]
            fill_mask = input_bg[r0 : r1 + 1, c0 : c1 + 1]
            out_window = out[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(fill_mask & (out_window != bg)), "overlapping fills are ambiguous"
            out_window[fill_mask] = pair["color"]
            continue

        for r, c in pair["coords"]:
            out[r, c] = bg

    return out


def _choose_run(cells: list[Coord], *, min_len: int = 1, max_len: int = 3) -> list[Coord]:
    assert cells, "expected at least one candidate cell"
    max_len = min(max_len, len(cells))
    min_len = min(min_len, max_len)
    length = random.randint(min_len, max_len)
    start = random.randint(0, len(cells) - length)
    return cells[start : start + length]


def _edge_cells(bbox: BBox, edge: str) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    if edge == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if edge == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if edge == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    if edge == "right":
        return [(r, c1) for r in range(r0 + 1, r1)]
    raise ValueError(f"unknown edge {edge}")


def _interior_segment(bbox: BBox) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    interior_h = r1 - r0 - 1
    interior_w = c1 - c0 - 1
    modes = []
    if interior_w >= 2:
        modes.append("horizontal")
    if interior_h >= 2:
        modes.append("vertical")
    assert modes, "expected enough interior room for an obstacle segment"

    mode = random.choice(modes)
    if mode == "horizontal":
        row = random.randint(r0 + 1, r1 - 1)
        length = random.randint(2, interior_w)
        start = random.randint(c0 + 1, c1 - length)
        return [(row, c) for c in range(start, start + length)]

    col = random.randint(c0 + 1, c1 - 1)
    length = random.randint(2, interior_h)
    start = random.randint(r0 + 1, r1 - length)
    return [(r, col) for r in range(start, start + length)]


def _paint_cells(grid: np.ndarray, cells: list[Coord], color: int) -> bool:
    for r, c in cells:
        if int(grid[r, c]) != BG_COLOR:
            return False
    for r, c in cells:
        grid[r, c] = color
    return True


def _marker_coords(bbox: BBox, diagonal: str) -> tuple[Coord, Coord]:
    r0, c0, r1, c1 = bbox
    if diagonal == "nwse":
        return (r0, c0), (r1, c1)
    if diagonal == "nesw":
        return (r0, c1), (r1, c0)
    raise ValueError(f"unknown diagonal {diagonal}")


def _slot_layout() -> tuple[int, int, list[tuple[int, int, int, int]]]:
    top_h = random.randint(SLOT_SIDE_MIN, SLOT_SIDE_MAX)
    bottom_h = random.randint(SLOT_SIDE_MIN, SLOT_SIDE_MAX)
    left_w = random.randint(SLOT_SIDE_MIN, SLOT_SIDE_MAX)
    right_w = random.randint(SLOT_SIDE_MIN, SLOT_SIDE_MAX)
    top_margin = random.choice(OUTER_MARGIN_OPTIONS)
    bottom_margin = random.choice(OUTER_MARGIN_OPTIONS)
    left_margin = random.choice(OUTER_MARGIN_OPTIONS)
    right_margin = random.choice(OUTER_MARGIN_OPTIONS)
    row_gap = random.choice(GAP_OPTIONS)
    col_gap = random.choice(GAP_OPTIONS)

    height = top_margin + top_h + row_gap + bottom_h + bottom_margin
    width = left_margin + left_w + col_gap + right_w + right_margin
    assert height <= 30 and width <= 30

    top_row = top_margin
    bottom_row = top_margin + top_h + row_gap
    left_col = left_margin
    right_col = left_margin + left_w + col_gap
    slots = [
        (top_row, left_col, top_h, left_w),
        (top_row, right_col, top_h, right_w),
        (bottom_row, left_col, bottom_h, left_w),
        (bottom_row, right_col, bottom_h, right_w),
    ]
    return height, width, slots


def _choose_bbox(slot: tuple[int, int, int, int]) -> BBox | None:
    sr, sc, sh, sw = slot
    max_h = min(RECT_SIDE_MAX, sh - 2)
    max_w = min(RECT_SIDE_MAX, sw - 2)
    if max_h < RECT_SIDE_MIN or max_w < RECT_SIDE_MIN:
        return None

    rect_h = random.randint(RECT_SIDE_MIN, max_h)
    rect_w = random.randint(RECT_SIDE_MIN, max_w)
    r0 = random.randint(sr + 1, sr + sh - rect_h - 1)
    c0 = random.randint(sc + 1, sc + sw - rect_w - 1)
    return r0, c0, r0 + rect_h - 1, c0 + rect_w - 1


def _place_pair(
    grid: np.ndarray,
    bbox: BBox,
    color: int,
    obstacle_color: int,
    status: str,
) -> dict | None:
    diagonal = random.choice(("nwse", "nesw"))
    a, b = _marker_coords(bbox, diagonal)
    if int(grid[a]) != BG_COLOR or int(grid[b]) != BG_COLOR:
        return None
    grid[a] = color
    grid[b] = color

    if status == "blocked":
        segments = [
            _choose_run(_edge_cells(bbox, "top"), min_len=1, max_len=2),
            _choose_run(_edge_cells(bbox, "bottom"), min_len=1, max_len=2),
        ]
    elif status == "open_onecorridor":
        edge = random.choice(("top", "bottom"))
        segments = [_choose_run(_edge_cells(bbox, edge), min_len=1, max_len=2)]
    elif status == "open_interior":
        segments = [_interior_segment(bbox)]
    elif status == "open_clean":
        segments = []
    else:
        raise ValueError(f"unknown status {status}")

    for cells in segments:
        if not _paint_cells(grid, cells, obstacle_color):
            return None

    return {"bbox": bbox, "coords": (a, b), "color": color, "status": status}


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        if spec["status"] == "blocked":
            for r, c in spec["coords"]:
                out[r, c] = BG_COLOR
            continue

        window = inp[r0 : r1 + 1, c0 : c1 + 1]
        out_window = out[r0 : r1 + 1, c0 : c1 + 1]
        out_window[window == BG_COLOR] = spec["color"]
    return out


def _verify_semantics(grid: np.ndarray, obstacle_color: int, specs: list[dict]) -> None:
    counts = Counter(spec["status"] for spec in specs)
    assert counts["open_interior"] >= 1
    assert counts["blocked"] >= 1
    assert counts["open_onecorridor"] >= 1

    for spec in specs:
        a, b = spec["coords"]
        path_a, path_b = _l_corridors(a, b)
        clear_a = _corridor_clear(grid, obstacle_color, path_a)
        clear_b = _corridor_clear(grid, obstacle_color, path_b)
        r0, c0, r1, c1 = spec["bbox"]
        interior_has_obstacle = bool(np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == obstacle_color))

        if spec["status"] == "blocked":
            assert not clear_a and not clear_b
        elif spec["status"] == "open_onecorridor":
            assert clear_a ^ clear_b
        elif spec["status"] == "open_clean":
            assert clear_a and clear_b
        elif spec["status"] == "open_interior":
            assert clear_a or clear_b
            assert interior_has_obstacle
        else:
            raise ValueError(f"unknown status {spec['status']}")


def _gen_attempt() -> np.ndarray | None:
    height, width, slots = _slot_layout()
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    used_slots = random.sample(slots, pair_count)

    statuses = list(BASE_STATUSES)
    while len(statuses) < pair_count:
        statuses.append(random.choice(EXTRA_STATUS_OPTIONS))
    random.shuffle(statuses)

    colors = random.sample(PALETTE, pair_count + 1)
    obstacle_color = colors[0]
    pair_colors = colors[1:]

    grid = np.full((height, width), BG_COLOR, dtype=int)
    specs = []
    for slot, status, color in zip(used_slots, statuses, pair_colors):
        bbox = _choose_bbox(slot)
        if bbox is None:
            return None
        spec = _place_pair(grid, bbox, color, obstacle_color, status)
        if spec is None:
            return None
        specs.append(spec)

    _verify_semantics(grid, obstacle_color, specs)
    expected = _expected_output(grid, specs)

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid border-corridor visibility puzzle")
