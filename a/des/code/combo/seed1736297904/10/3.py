import random

import numpy as np


BG = 0
FRAME_SIZE = 5
PATTERN_SIZE = 3
MAX_GENERATION_ATTEMPTS = 400
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

DIR_DATA = {
    "up": {"delta": (-1, 0), "gap": (0, 2)},
    "right": {"delta": (0, 1), "gap": (2, 4)},
    "down": {"delta": (1, 0), "gap": (4, 2)},
    "left": {"delta": (0, -1), "gap": (2, 0)},
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _rotate_pattern(pattern: np.ndarray, direction: str) -> np.ndarray:
    if direction == "left":
        return np.rot90(pattern, 1)
    if direction == "right":
        return np.rot90(pattern, -1)
    return pattern.copy()


def _find_instruction_blocks(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    blocks: list[dict] = []

    for r0 in range(h - FRAME_SIZE + 1):
        for c0 in range(w - FRAME_SIZE + 1):
            window = g[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE]
            gaps = [
                direction
                for direction, data in DIR_DATA.items()
                if int(window[data["gap"]]) == BG
            ]
            if len(gaps) != 1:
                continue

            direction = gaps[0]
            frame_color = None
            valid = True
            for rr in range(FRAME_SIZE):
                for cc in range(FRAME_SIZE):
                    on_border = rr in (0, FRAME_SIZE - 1) or cc in (0, FRAME_SIZE - 1)
                    if not on_border:
                        continue
                    val = int(window[rr, cc])
                    if (rr, cc) == DIR_DATA[direction]["gap"]:
                        if val != BG:
                            valid = False
                        continue
                    if val == BG:
                        valid = False
                        continue
                    if frame_color is None:
                        frame_color = val
                    elif val != frame_color:
                        valid = False
            if not valid or frame_color is None:
                continue

            pattern = window[1:4, 1:4].copy()
            inner_colors = set(map(int, np.unique(pattern))) - {BG}
            if len(inner_colors) != 1:
                continue
            inner_color = inner_colors.pop()
            if inner_color == frame_color:
                continue

            blocks.append(
                {
                    "top": r0,
                    "left": c0,
                    "direction": direction,
                    "frame_color": frame_color,
                    "inner_color": inner_color,
                    "pattern": pattern,
                }
            )

    blocks.sort(key=lambda item: (item["top"], item["left"]))
    return blocks


def _marker_candidates(end_r: int, end_c: int, direction: str) -> list[tuple[int, int]]:
    if direction == "down":
        return [(end_r - 1, end_c), (end_r - 1, end_c - 1)]
    if direction == "up":
        return [(end_r, end_c), (end_r, end_c - 1)]
    if direction == "right":
        return [(end_r, end_c - 1), (end_r - 1, end_c - 1)]
    return [(end_r, end_c), (end_r - 1, end_c)]


def _find_marker_anchor(grid: np.ndarray, block: dict) -> tuple[int, int] | None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    direction = block["direction"]
    frame_color = block["frame_color"]
    gap_r, gap_c = DIR_DATA[direction]["gap"]
    dr, dc = DIR_DATA[direction]["delta"]

    r = block["top"] + gap_r + dr
    c = block["left"] + gap_c + dc
    end = None
    while _in_bounds((h, w), r, c) and int(g[r, c]) == frame_color:
        end = (r, c)
        r += dr
        c += dc

    if end is None:
        return None

    anchors: list[tuple[int, int]] = []
    for ar, ac in _marker_candidates(end[0], end[1], direction):
        if not (0 <= ar <= h - 2 and 0 <= ac <= w - 2):
            continue
        if np.all(g[ar : ar + 2, ac : ac + 2] == frame_color):
            anchors.append((ar, ac))

    unique_anchors = list(dict.fromkeys(anchors))
    if not unique_anchors:
        return None
    return min(unique_anchors)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for block in _find_instruction_blocks(g):
        anchor = _find_marker_anchor(g, block)
        if anchor is None:
            continue
        ar, ac = anchor
        if ar < 0 or ac < 0 or ar + PATTERN_SIZE > g.shape[0] or ac + PATTERN_SIZE > g.shape[1]:
            continue

        rotated = _rotate_pattern(block["pattern"], block["direction"])
        mask = rotated != BG
        patch = out[ar : ar + PATTERN_SIZE, ac : ac + PATTERN_SIZE]
        patch[mask] = rotated[mask]
        out[ar : ar + PATTERN_SIZE, ac : ac + PATTERN_SIZE] = patch

    return out


def _random_pattern(color: int) -> np.ndarray:
    coords = [(r, c) for r in range(PATTERN_SIZE) for c in range(PATTERN_SIZE)]
    for _ in range(200):
        target = random.randint(2, 6)
        start = random.choice(coords)
        filled = {start}
        frontier = {
            (start[0] + dr, start[1] + dc)
            for dr, dc in NEIGHBORS4
            if 0 <= start[0] + dr < PATTERN_SIZE and 0 <= start[1] + dc < PATTERN_SIZE
        }
        while frontier and len(filled) < target:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            filled.add(cell)
            for dr, dc in NEIGHBORS4:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < PATTERN_SIZE and 0 <= nc < PATTERN_SIZE and (nr, nc) not in filled:
                    frontier.add((nr, nc))

        pattern = np.zeros((PATTERN_SIZE, PATTERN_SIZE), dtype=int)
        for rr, cc in filled:
            pattern[rr, cc] = color

        mask = pattern != BG
        if np.array_equal(mask, np.rot90(mask, 1)):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        if not (np.any(mask[2, :]) or np.any(mask[:, 2])):
            continue
        return pattern

    fallback = np.zeros((PATTERN_SIZE, PATTERN_SIZE), dtype=int)
    fallback[0, 0] = color
    fallback[1, 0] = color
    fallback[1, 1] = color
    fallback[2, 1] = color
    fallback[2, 2] = color
    return fallback


def _marker_anchor_from_spec(spec: dict) -> tuple[int, int]:
    r0, c0 = spec["top"], spec["left"]
    length = spec["rail_len"]
    align = spec["align"]
    direction = spec["direction"]

    if direction == "down":
        return r0 + 5 + length, c0 + 2 - align
    if direction == "up":
        return r0 - length - 2, c0 + 2 - align
    if direction == "right":
        return r0 + 2 - align, c0 + 5 + length
    return r0 + 2 - align, c0 - length - 2


def _object_pixels(spec: dict) -> list[tuple[int, int, int]]:
    r0, c0 = spec["top"], spec["left"]
    frame_color = spec["frame_color"]
    pattern = spec["pattern"]
    direction = spec["direction"]
    rail_len = spec["rail_len"]

    pixels: list[tuple[int, int, int]] = []
    gap = DIR_DATA[direction]["gap"]
    for rr in range(FRAME_SIZE):
        for cc in range(FRAME_SIZE):
            on_border = rr in (0, FRAME_SIZE - 1) or cc in (0, FRAME_SIZE - 1)
            if not on_border or (rr, cc) == gap:
                continue
            pixels.append((r0 + rr, c0 + cc, frame_color))

    for rr in range(PATTERN_SIZE):
        for cc in range(PATTERN_SIZE):
            val = int(pattern[rr, cc])
            if val != BG:
                pixels.append((r0 + 1 + rr, c0 + 1 + cc, val))

    if direction == "down":
        for rr in range(r0 + 5, r0 + 5 + rail_len):
            pixels.append((rr, c0 + 2, frame_color))
    elif direction == "up":
        for rr in range(r0 - 1, r0 - 1 - rail_len, -1):
            pixels.append((rr, c0 + 2, frame_color))
    elif direction == "right":
        for cc in range(c0 + 5, c0 + 5 + rail_len):
            pixels.append((r0 + 2, cc, frame_color))
    else:
        for cc in range(c0 - 1, c0 - 1 - rail_len, -1):
            pixels.append((r0 + 2, cc, frame_color))

    anchor_r, anchor_c = _marker_anchor_from_spec(spec)
    for rr in range(anchor_r, anchor_r + 2):
        for cc in range(anchor_c, anchor_c + 2):
            pixels.append((rr, cc, frame_color))

    return pixels


def _protected_cells(spec: dict) -> set[tuple[int, int]]:
    cells = {(r, c) for r, c, _ in _object_pixels(spec)}
    anchor_r, anchor_c = _marker_anchor_from_spec(spec)
    for rr in range(anchor_r, anchor_r + PATTERN_SIZE):
        for cc in range(anchor_c, anchor_c + PATTERN_SIZE):
            cells.add((rr, cc))
    return cells


def _expanded_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if _in_bounds(shape, nr, nc):
                    expanded.add((nr, nc))
    return expanded


def _sample_top_left(height: int, width: int, direction: str, rail_len: int) -> tuple[int, int] | None:
    if direction in ("up", "down"):
        row_lo = 1
        row_hi = height - FRAME_SIZE - 1
        col_lo = 1
        col_hi = width - FRAME_SIZE - 1
        if direction == "up":
            row_lo = max(row_lo, rail_len + 3)
        else:
            row_hi = min(row_hi, height - rail_len - 9)
    else:
        row_lo = 1
        row_hi = height - FRAME_SIZE - 1
        col_lo = 1
        col_hi = width - FRAME_SIZE - 1
        if direction == "left":
            col_lo = max(col_lo, rail_len + 3)
        else:
            col_hi = min(col_hi, width - rail_len - 9)

    if row_lo > row_hi or col_lo > col_hi:
        return None
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        ar, ac = _marker_anchor_from_spec(spec)
        rotated = _rotate_pattern(spec["pattern"], spec["direction"])
        mask = rotated != BG
        patch = out[ar : ar + PATTERN_SIZE, ac : ac + PATTERN_SIZE]
        patch[mask] = rotated[mask]
        out[ar : ar + PATTERN_SIZE, ac : ac + PATTERN_SIZE] = patch
    return out


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(1, 4)
    height = random.randint(16 + 2 * n_objects, 30)
    width = random.randint(16 + 2 * n_objects, 30)
    grid = np.zeros((height, width), dtype=int)
    forbidden = np.zeros((height, width), dtype=bool)

    colors = random.sample(COLOR_POOL, 2 * n_objects)
    specs: list[dict] = []

    for idx in range(n_objects):
        frame_color = colors[2 * idx]
        inner_color = colors[2 * idx + 1]
        placed = False

        for _ in range(200):
            direction = random.choice(tuple(DIR_DATA))
            rail_len = random.randint(1, 6)
            top_left = _sample_top_left(height, width, direction, rail_len)
            if top_left is None:
                continue

            spec = {
                "top": top_left[0],
                "left": top_left[1],
                "direction": direction,
                "rail_len": rail_len,
                "align": random.randint(0, 1),
                "frame_color": frame_color,
                "pattern": _random_pattern(inner_color),
            }

            protected = _protected_cells(spec)
            if not all(1 <= r < height - 1 and 1 <= c < width - 1 for r, c in protected):
                continue

            input_pixels = _object_pixels(spec)
            if not all(_in_bounds((height, width), r, c) for r, c, _ in input_pixels):
                continue

            expanded = _expanded_cells(protected, (height, width))
            if any(forbidden[r, c] for r, c in expanded):
                continue

            for r, c, color in input_pixels:
                if grid[r, c] != BG and grid[r, c] != color:
                    break
            else:
                for r, c, color in input_pixels:
                    grid[r, c] = color
                for r, c in expanded:
                    forbidden[r, c] = True
                specs.append(spec)
                placed = True
                break

        if not placed:
            return None

    detected = _find_instruction_blocks(grid)
    if len(detected) != len(specs):
        return None

    expected = _expected_output(grid, specs)
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
    raise RuntimeError("failed to generate a valid input for task 3")
