import random

import numpy as np


BG = 0
WALL = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300
PLACEMENT_TRIES = 200


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each 2-cell block away from its same-color tail marker until the maze stops it.

    Input structure:
    - Background is black and grey pixels form 1-cell-wide horizontal or vertical corridors.
    - Every movable object uses a unique non-grey, non-black color and appears as three collinear same-color pixels.
    - For each such triple, two pixels are the domino block inside a corridor and the remaining endpoint pixel is a tail marker placed just outside that corridor end on the block axis.

    Rule:
    - Determine which endpoint is the tail marker by checking which two same-color pixels sit between the corridor side walls.
    - Process the blocks in top-to-bottom, then left-to-right order using the block's 2-cell position in the input.
    - Remove all tail markers, then for each block slide its two cells away from its marker along the corridor until the next step would hit a grey wall, another settled block, or the grid boundary.

    Key insight:
    - The same-color triple is ambiguous unless you use the maze geometry: the block is the pair of cells that lie inside the walled corridor, and processing order matters because earlier blocks become obstacles for later ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    blocks = _extract_blocks(g)
    out = g.copy()
    for block in blocks:
        ar, ac = block["arrow"]
        out[ar, ac] = BG

    for block in blocks:
        color = block["color"]
        cells = [tuple(cell) for cell in block["block_cells"]]
        for r, c in cells:
            out[r, c] = BG

        shifted = _slide_block(out, cells, block["orientation"], block["direction"])
        for r, c in shifted:
            out[r, c] = color

    return out


def _extract_blocks(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BG, WALL)]
    blocks = []

    for color in colors:
        pts = np.argwhere(grid == color)
        assert pts.shape[0] == 3, f"color {color} must occupy exactly three cells"

        if len({int(r) for r, _ in pts}) == 1:
            orientation = "H"
            ordered = sorted((int(r), int(c)) for r, c in pts)
            ordered.sort(key=lambda rc: rc[1])
        elif len({int(c) for _, c in pts}) == 1:
            orientation = "V"
            ordered = sorted((int(r), int(c)) for r, c in pts)
            ordered.sort(key=lambda rc: rc[0])
        else:
            raise AssertionError(f"color {color} must form a straight 3-cell line")

        corridor_flags = [_is_corridor_cell(grid, cell, orientation) for cell in ordered]
        low_block = corridor_flags == [True, True, False]
        high_block = corridor_flags == [False, True, True]
        assert low_block ^ high_block, f"color {color} must have exactly one marker endpoint"

        if low_block:
            block_cells = ordered[:2]
            arrow = ordered[2]
            direction = -1
        else:
            block_cells = ordered[1:]
            arrow = ordered[0]
            direction = 1

        top_left = (min(r for r, _ in block_cells), min(c for _, c in block_cells))
        blocks.append(
            {
                "color": color,
                "orientation": orientation,
                "block_cells": block_cells,
                "arrow": arrow,
                "direction": direction,
                "order_key": top_left,
            }
        )

    blocks.sort(key=lambda item: item["order_key"])
    return blocks


def _is_corridor_cell(grid: np.ndarray, cell: tuple[int, int], orientation: str) -> bool:
    r, c = cell
    h, w = grid.shape

    if orientation == "H":
        return 0 < r < h - 1 and int(grid[r - 1, c]) == WALL and int(grid[r + 1, c]) == WALL
    return 0 < c < w - 1 and int(grid[r, c - 1]) == WALL and int(grid[r, c + 1]) == WALL


def _slide_block(
    grid: np.ndarray, cells: list[tuple[int, int]], orientation: str, direction: int
) -> list[tuple[int, int]]:
    shifted = [tuple(cell) for cell in cells]

    if orientation == "H":
        shifted.sort(key=lambda rc: rc[1])
        step = (0, direction)
        front = shifted[-1] if direction > 0 else shifted[0]
    else:
        shifted.sort(key=lambda rc: rc[0])
        step = (direction, 0)
        front = shifted[-1] if direction > 0 else shifted[0]

    dr, dc = step
    h, w = grid.shape

    while True:
        nr = front[0] + dr
        nc = front[1] + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if int(grid[nr, nc]) != BG:
            break

        shifted = [(r + dr, c + dc) for r, c in shifted]
        front = (front[0] + dr, front[1] + dc)

    return shifted


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if _has_moving_block(grid, out):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    mode = rng.choice(["mixed", "boundary"])
    h = rng.randint(14, 24)
    w = rng.randint(14, 24)
    grid = np.full((h, w), BG, dtype=int)
    used = np.zeros((h, w), dtype=bool)
    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    color_idx = 0

    if mode == "mixed":
        lane_specs = []
        lane_count = rng.randint(3, 4)
        orientations = ["H", "V"]
        while len(orientations) < lane_count:
            orientations.append(rng.choice(["H", "V"]))
        rng.shuffle(orientations)

        kinds = ["wall", "double"]
        while len(kinds) < lane_count:
            kinds.append(rng.choice(["wall", "double"]))
        rng.shuffle(kinds)
        lane_specs = list(zip(orientations, kinds))
    else:
        orientation = rng.choice(["H", "V"])
        lane_count = rng.randint(2, 3)
        kinds = ["boundary"]
        while len(kinds) < lane_count:
            kinds.append(rng.choice(["wall", "double", "boundary"]))
        rng.shuffle(kinds)
        lane_specs = [(orientation, kind) for kind in kinds]

    for orientation, kind in lane_specs:
        needed = 2 if kind == "double" else 1
        if color_idx + needed > len(colors):
            return None

        lane_colors = colors[color_idx : color_idx + needed]
        color_idx += needed
        ok = _place_lane(grid, used, orientation, kind, lane_colors, rng)
        if not ok:
            return None

    blocks = _extract_blocks(grid)
    if len(blocks) < 2:
        return None
    return grid


def _place_lane(
    grid: np.ndarray,
    used: np.ndarray,
    orientation: str,
    kind: str,
    colors: list[int],
    rng: random.Random,
) -> bool:
    h, w = grid.shape

    for _ in range(PLACEMENT_TRIES):
        if orientation == "H":
            spec = _sample_horizontal_spec(h, w, kind, rng)
            if spec is None:
                return False
            r0, c0, r1, c1 = spec["box"]
            if not _box_is_free(used, r0, c0, r1, c1):
                continue
            _draw_horizontal_lane(grid, spec, kind, colors)
        else:
            spec = _sample_vertical_spec(h, w, kind, rng)
            if spec is None:
                return False
            r0, c0, r1, c1 = spec["box"]
            if not _box_is_free(used, r0, c0, r1, c1):
                continue
            _draw_vertical_lane(grid, spec, kind, colors)

        _reserve_box(used, r0, c0, r1, c1)
        return True

    return False


def _sample_horizontal_spec(h: int, w: int, kind: str, rng: random.Random) -> dict | None:
    if h < 3 or w < 6:
        return None

    r = rng.randint(1, h - 2)

    if kind == "wall":
        max_len = min(10, w - 2)
        if max_len < 4:
            return None
        length = rng.randint(4, max_len)
        start = rng.randint(1, w - length - 1)
        end = start + length - 1
        direction = rng.choice([-1, 1])
        arrow = end + 1 if direction < 0 else start - 1
        cap = start - 1 if direction < 0 else end + 1
        return {
            "row": r,
            "start": start,
            "end": end,
            "direction": direction,
            "arrow": arrow,
            "cap": cap,
            "box": (r - 1, min(arrow, cap), r + 1, max(arrow, cap)),
        }

    if kind == "boundary":
        direction = rng.choice([-1, 1])
        if direction > 0:
            start = rng.randint(1, w - 4)
            end = w - 1
            arrow = start - 1
        else:
            start = 0
            end = rng.randint(3, w - 2)
            arrow = end + 1
        return {
            "row": r,
            "start": start,
            "end": end,
            "direction": direction,
            "arrow": arrow,
            "box": (r - 1, min(start, arrow), r + 1, max(end, arrow)),
        }

    max_len = min(11, w - 2)
    if max_len < 5:
        return None
    length = rng.randint(5, max_len)
    start = rng.randint(1, w - length - 1)
    end = start + length - 1
    return {
        "row": r,
        "start": start,
        "end": end,
        "box": (r - 1, start - 1, r + 1, end + 1),
    }


def _sample_vertical_spec(h: int, w: int, kind: str, rng: random.Random) -> dict | None:
    if w < 3 or h < 6:
        return None

    c = rng.randint(1, w - 2)

    if kind == "wall":
        max_len = min(10, h - 2)
        if max_len < 4:
            return None
        length = rng.randint(4, max_len)
        start = rng.randint(1, h - length - 1)
        end = start + length - 1
        direction = rng.choice([-1, 1])
        arrow = end + 1 if direction < 0 else start - 1
        cap = start - 1 if direction < 0 else end + 1
        return {
            "col": c,
            "start": start,
            "end": end,
            "direction": direction,
            "arrow": arrow,
            "cap": cap,
            "box": (min(arrow, cap), c - 1, max(arrow, cap), c + 1),
        }

    if kind == "boundary":
        direction = rng.choice([-1, 1])
        if direction > 0:
            start = rng.randint(1, h - 4)
            end = h - 1
            arrow = start - 1
        else:
            start = 0
            end = rng.randint(3, h - 2)
            arrow = end + 1
        return {
            "col": c,
            "start": start,
            "end": end,
            "direction": direction,
            "arrow": arrow,
            "box": (min(start, arrow), c - 1, max(end, arrow), c + 1),
        }

    max_len = min(11, h - 2)
    if max_len < 5:
        return None
    length = rng.randint(5, max_len)
    start = rng.randint(1, h - length - 1)
    end = start + length - 1
    return {
        "col": c,
        "start": start,
        "end": end,
        "box": (start - 1, c - 1, end + 1, c + 1),
    }


def _draw_horizontal_lane(grid: np.ndarray, spec: dict, kind: str, colors: list[int]) -> None:
    r = spec["row"]
    start = spec["start"]
    end = spec["end"]

    for c in range(start, end + 1):
        grid[r - 1, c] = WALL
        grid[r + 1, c] = WALL

    if kind == "double":
        left_color, right_color = colors
        grid[r, start - 1] = left_color
        grid[r, start] = left_color
        grid[r, start + 1] = left_color
        grid[r, end - 1] = right_color
        grid[r, end] = right_color
        grid[r, end + 1] = right_color
        return

    color = colors[0]
    direction = spec["direction"]
    arrow = spec["arrow"]
    grid[r, arrow] = color

    if direction > 0:
        grid[r, start] = color
        grid[r, start + 1] = color
    else:
        grid[r, end - 1] = color
        grid[r, end] = color

    if kind == "wall":
        grid[r, spec["cap"]] = WALL


def _draw_vertical_lane(grid: np.ndarray, spec: dict, kind: str, colors: list[int]) -> None:
    c = spec["col"]
    start = spec["start"]
    end = spec["end"]

    for r in range(start, end + 1):
        grid[r, c - 1] = WALL
        grid[r, c + 1] = WALL

    if kind == "double":
        top_color, bottom_color = colors
        grid[start - 1, c] = top_color
        grid[start, c] = top_color
        grid[start + 1, c] = top_color
        grid[end - 1, c] = bottom_color
        grid[end, c] = bottom_color
        grid[end + 1, c] = bottom_color
        return

    color = colors[0]
    direction = spec["direction"]
    arrow = spec["arrow"]
    grid[arrow, c] = color

    if direction > 0:
        grid[start, c] = color
        grid[start + 1, c] = color
    else:
        grid[end - 1, c] = color
        grid[end, c] = color

    if kind == "wall":
        grid[spec["cap"], c] = WALL


def _box_is_free(used: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = used.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(used[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve_box(used: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = used.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    used[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _has_moving_block(inp: np.ndarray, out: np.ndarray) -> bool:
    for block in _extract_blocks(inp):
        color = block["color"]
        before = sorted(block["block_cells"])
        after = sorted((int(r), int(c)) for r, c in np.argwhere(out == color))
        if before != after:
            return True
    return False
