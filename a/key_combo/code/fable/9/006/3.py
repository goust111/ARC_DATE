from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
PALETTE = tuple(range(1, 10))
ARROW_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _parse_arrow(coords: np.ndarray) -> tuple[Coord, Coord] | None:
    if int(coords.shape[0]) != 6:
        return None

    pts = [tuple(map(int, p)) for p in coords]
    row_counts = Counter(r for r, _ in pts)
    col_counts = Counter(c for _, c in pts)

    long_rows = [r for r, n in row_counts.items() if n == 4]
    long_cols = [c for c, n in col_counts.items() if n == 4]
    cross_rows = [r for r, n in row_counts.items() if n == 3]
    cross_cols = [c for c, n in col_counts.items() if n == 3]

    is_horizontal = len(long_rows) == 1 and len(cross_cols) == 1 and not long_cols and not cross_rows
    is_vertical = len(long_cols) == 1 and len(cross_rows) == 1 and not long_rows and not cross_cols
    if not (is_horizontal ^ is_vertical):
        return None

    if is_horizontal:
        row = int(long_rows[0])
        cross_col = int(cross_cols[0])
        line_cols = sorted(c for r, c in pts if r == row)
        cross_rows_here = sorted(r for r, c in pts if c == cross_col)
        if line_cols != list(range(line_cols[0], line_cols[0] + 4)):
            return None
        if cross_rows_here != [row - 1, row, row + 1]:
            return None
        left_dist = cross_col - line_cols[0]
        right_dist = line_cols[-1] - cross_col
        if {left_dist, right_dist} != {1, 2}:
            return None
        if left_dist < right_dist:
            return (row, line_cols[0]), (0, -1)
        return (row, line_cols[-1]), (0, 1)

    col = int(long_cols[0])
    cross_row = int(cross_rows[0])
    line_rows = sorted(r for r, c in pts if c == col)
    cross_cols_here = sorted(c for r, c in pts if r == cross_row)
    if line_rows != list(range(line_rows[0], line_rows[0] + 4)):
        return None
    if cross_cols_here != [col - 1, col, col + 1]:
        return None
    up_dist = cross_row - line_rows[0]
    down_dist = line_rows[-1] - cross_row
    if {up_dist, down_dist} != {1, 2}:
        return None
    if up_dist < down_dist:
        return (line_rows[0], col), (-1, 0)
    return (line_rows[-1], col), (1, 0)


def _parse_box(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    frame_color = int(comp["color"])
    expected_area = 2 * (h + w) - 4
    if int(comp["area"]) != expected_area:
        return None

    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    if not np.all(region[border_mask] == frame_color):
        return None

    interior = region[1:-1, 1:-1]
    if interior.size == 0:
        return None
    liquid_colors = np.unique(interior)
    if liquid_colors.size != 1:
        return None
    liquid_color = int(liquid_colors[0])
    if liquid_color in {bg, frame_color}:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "frame_color": frame_color,
        "liquid_color": liquid_color,
        "interior": (slice(r0 + 1, r1), slice(c0 + 1, c1)),
    }


def _line_to_first_hit(grid: np.ndarray, start: Coord, direction: Coord, bg: int) -> Coord | None:
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    while _in_bounds(grid.shape, r, c):
        if int(grid[r, c]) != bg:
            return (r, c)
        r += dr
        c += dc
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Break the wall of each pointed liquid box, empty the box, and draw a draining line back to the arrow tip.

    Input structure:
    - The grid has a dominant background plus several one-cell-thick rectangular frames.
    - Every frame surrounds a solid interior filled with one liquid color distinct from both the frame and the background.
    - Outside the frames are monochrome six-pixel asymmetric T arrows: a length-4 bar with a length-3 crossbar one cell closer to the pointed end. Some arrows are distractors whose bar axis points to no frame wall.

    Rule:
    - Detect each asymmetric T arrow and decode its pointing direction from the off-center crossbar.
    - From the arrow tip, cast a ray forward along the bar axis; if the first non-background cell is a rectangular frame wall, that wall cell becomes background.
    - For every hit frame, clear the entire liquid interior to background, then paint the cells between the opening and the arrow tip with that box's liquid color, stopping just before the tip.
    - Leave distractor arrows, untouched frames, and arrow bodies unchanged.

    Key insight:
    - The crossbar is intentionally biased toward the tip, so the shorter bar end is the only valid drilling direction; only arrows whose forward ray first meets a frame wall actually drain a box.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg)

    arrows: list[dict] = []
    boxes: list[dict] = []
    for comp in comps:
        arrow = _parse_arrow(comp["coords"])
        if arrow is not None:
            tip, direction = arrow
            arrows.append({"tip": tip, "direction": direction, "color": int(comp["color"])})
            continue
        box = _parse_box(g, comp, bg)
        if box is not None:
            boxes.append(box)

    wall_to_box: dict[Coord, int] = {}
    for idx, box in enumerate(boxes):
        r0, c0, r1, c1 = box["bbox"]
        for c in range(c0, c1 + 1):
            wall_to_box[(r0, c)] = idx
            wall_to_box[(r1, c)] = idx
        for r in range(r0, r1 + 1):
            wall_to_box[(r, c0)] = idx
            wall_to_box[(r, c1)] = idx

    hits: list[tuple[int, Coord, Coord]] = []
    for arrow in arrows:
        hit = _line_to_first_hit(g, arrow["tip"], arrow["direction"], bg)
        if hit is None or hit not in wall_to_box:
            continue
        hits.append((wall_to_box[hit], hit, arrow["direction"]))

    out = g.copy()
    drained_boxes = {box_idx for box_idx, _opening, _direction in hits}
    for box_idx in drained_boxes:
        interior = boxes[box_idx]["interior"]
        out[interior] = bg

    for box_idx, opening, direction in hits:
        out[opening] = bg
        liquid_color = int(boxes[box_idx]["liquid_color"])
        dr, dc = direction
        r, c = opening[0] - dr, opening[1] - dc
        while (r, c) != next(arrow["tip"] for arrow in arrows if _line_to_first_hit(g, arrow["tip"], arrow["direction"], bg) == opening and arrow["direction"] == direction):
            out[r, c] = liquid_color
            r -= dr
            c -= dc
        # The loop stops on the tip itself; the tip pixel stays part of the arrow.

    return out


def _arrow_cells(tip: Coord, direction: Coord) -> list[Coord]:
    if direction == (0, 1):
        rels = [(0, 0), (0, -1), (0, -2), (0, -3), (-1, -1), (1, -1)]
    elif direction == (0, -1):
        rels = [(0, 0), (0, 1), (0, 2), (0, 3), (-1, 1), (1, 1)]
    elif direction == (1, 0):
        rels = [(0, 0), (-1, 0), (-2, 0), (-3, 0), (-1, -1), (-1, 1)]
    else:
        rels = [(0, 0), (1, 0), (2, 0), (3, 0), (1, -1), (1, 1)]
    return [(tip[0] + dr, tip[1] + dc) for dr, dc in rels]


def _paint_cells(grid: np.ndarray, cells: list[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _rect_border_cells(box: dict) -> list[Coord]:
    r0, c0, r1, c1 = box["bbox"]
    cells: list[Coord] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _place_box(grid: np.ndarray, bbox: BBox, frame_color: int, liquid_color: int) -> dict:
    r0, c0, r1, c1 = bbox
    grid[r0:r1 + 1, c0:c1 + 1] = liquid_color
    grid[r0, c0:c1 + 1] = frame_color
    grid[r1, c0:c1 + 1] = frame_color
    grid[r0:r1 + 1, c0] = frame_color
    grid[r0:r1 + 1, c1] = frame_color
    return {
        "bbox": bbox,
        "frame_color": frame_color,
        "liquid_color": liquid_color,
        "interior": (slice(r0 + 1, r1), slice(c0 + 1, c1)),
    }


def _bbox_with_margin(box: BBox, margin: int, shape: tuple[int, int]) -> BBox:
    r0, c0, r1, c1 = box
    return (
        max(0, r0 - margin),
        max(0, c0 - margin),
        min(shape[0] - 1, r1 + margin),
        min(shape[1] - 1, c1 + margin),
    )


def _bboxes_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_boxes(grid: np.ndarray, count: int) -> list[dict] | None:
    boxes: list[dict] = []
    reserved: list[BBox] = []
    frame_colors = random.sample(PALETTE, count)
    liquid_choices = PALETTE * 2

    for i in range(count):
        frame_color = frame_colors[i]
        liquid_pool = [c for c in liquid_choices if c != frame_color]
        liquid_color = random.choice(liquid_pool)
        placed = False
        for _ in range(150):
            h = random.randint(5, 8)
            w = random.randint(5, 8)
            r0 = random.randint(4, grid.shape[0] - h - 5)
            c0 = random.randint(4, grid.shape[1] - w - 5)
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            halo = _bbox_with_margin(bbox, 3, grid.shape)
            if any(_bboxes_overlap(halo, other) for other in reserved):
                continue
            boxes.append(_place_box(grid, bbox, frame_color, liquid_color))
            reserved.append(halo)
            placed = True
            break
        if not placed:
            return None
    return boxes


def _side_openings(box: dict, side: str) -> list[Coord]:
    r0, c0, r1, c1 = box["bbox"]
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _side_direction(side: str) -> Coord:
    if side == "top":
        return (1, 0)
    if side == "bottom":
        return (-1, 0)
    if side == "left":
        return (0, 1)
    return (0, -1)


def _free_cells(grid: np.ndarray, cells: list[Coord]) -> bool:
    return all(_in_bounds(grid.shape, r, c) and int(grid[r, c]) == BG for r, c in cells)


def _path_cells_between_tip_and_wall(tip: Coord, wall: Coord, direction: Coord) -> list[Coord]:
    dr, dc = direction
    cells: list[Coord] = []
    r, c = tip[0] + dr, tip[1] + dc
    while (r, c) != wall:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _try_place_valid_arrow(grid: np.ndarray, box: dict, color: int) -> dict | None:
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    for side in sides:
        direction = _side_direction(side)
        openings = _side_openings(box, side)
        random.shuffle(openings)
        gaps = [1, 2, 3, 4]
        random.shuffle(gaps)
        for opening in openings:
            for gap in gaps:
                tip = (opening[0] - direction[0] * (gap + 1), opening[1] - direction[1] * (gap + 1))
                arrow_cells = _arrow_cells(tip, direction)
                corridor = _path_cells_between_tip_and_wall(tip, opening, direction)
                if not _free_cells(grid, arrow_cells):
                    continue
                if not _free_cells(grid, corridor):
                    continue
                _paint_cells(grid, arrow_cells, color)
                return {"tip": tip, "direction": direction, "opening": opening, "box": box}
    return None


def _ray_is_empty(grid: np.ndarray, tip: Coord, direction: Coord) -> bool:
    dr, dc = direction
    r, c = tip[0] + dr, tip[1] + dc
    saw_bg = False
    while _in_bounds(grid.shape, r, c):
        if int(grid[r, c]) != BG:
            return False
        saw_bg = True
        r += dr
        c += dc
    return saw_bg


def _try_place_distractor_arrow(grid: np.ndarray, color: int) -> bool:
    directions = list(ARROW_DIRECTIONS)
    random.shuffle(directions)
    for direction in directions:
        for _ in range(80):
            tip = (random.randint(1, grid.shape[0] - 2), random.randint(1, grid.shape[1] - 2))
            arrow_cells = _arrow_cells(tip, direction)
            if not _free_cells(grid, arrow_cells):
                continue
            if not _ray_is_empty(grid, tip, direction):
                continue
            _paint_cells(grid, arrow_cells, color)
            return True
    return False


def _expected_output(grid: np.ndarray, valid_arrows: list[dict]) -> np.ndarray:
    out = grid.copy()
    drained_boxes = {id(item["box"]): item["box"] for item in valid_arrows}
    for box in drained_boxes.values():
        out[box["interior"]] = BG
    for arrow in valid_arrows:
        opening = arrow["opening"]
        direction = arrow["direction"]
        out[opening] = BG
        liquid = int(arrow["box"]["liquid_color"])
        dr, dc = direction
        r, c = opening[0] - dr, opening[1] - dc
        while (r, c) != arrow["tip"]:
            out[r, c] = liquid
            r -= dr
            c -= dc
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 28)
    w = random.randint(20, 28)
    grid = np.full((h, w), BG, dtype=int)

    box_count = random.randint(2, 3)
    boxes = _place_boxes(grid, box_count)
    if boxes is None:
        return None

    if box_count > 1 and random.random() < 0.7:
        valid_count = random.randint(1, box_count - 1)
    else:
        valid_count = random.randint(1, box_count)
    valid_boxes = random.sample(boxes, valid_count)

    color_pool = list(PALETTE)
    random.shuffle(color_pool)
    valid_arrows: list[dict] = []
    for box in valid_boxes:
        color = color_pool.pop() if color_pool else random.choice(PALETTE)
        placed = _try_place_valid_arrow(grid, box, color)
        if placed is None:
            return None
        valid_arrows.append(placed)

    distract_count = random.choice([0, 1, 1, 2])
    for _ in range(distract_count):
        color = color_pool.pop() if color_pool else random.choice(PALETTE)
        if not _try_place_distractor_arrow(grid, color):
            return None

    expected = _expected_output(grid, valid_arrows)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-drain puzzle input")
