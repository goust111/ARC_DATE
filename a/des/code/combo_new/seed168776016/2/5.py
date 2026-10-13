import random
from collections import deque

import numpy as np


DIR_INFO = {
    "up": {"delta": (-1, 0), "hole": (2, 1)},
    "down": {"delta": (1, 0), "hole": (0, 1)},
    "left": {"delta": (0, -1), "hole": (1, 2)},
    "right": {"delta": (0, 1), "hole": (1, 0)},
}

COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
BLOCK_COUNT_RANGE = (2, 5)
GRID_MIN = 10
GRID_MAX = 30
MAX_PAIR_GAP = 6
GEN_ATTEMPTS = 400
PAIR_TRIES = 120
BOUNDARY_TRIES = 120


def _same_color_components(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        coords.append((nr, nc))
            comps.append((color, coords))
    return comps


def _detect_blocks(grid: np.ndarray) -> list[dict]:
    blocks = []
    for color, coords in _same_color_components(grid):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        top, left = min(rows), min(cols)
        bottom, right = max(rows), max(cols)
        if bottom - top != 2 or right - left != 2 or len(coords) != 8:
            continue
        patch = grid[top : top + 3, left : left + 3]
        if not np.all((patch == 0) | (patch == color)):
            continue
        if int((patch == 0).sum()) != 1:
            continue
        direction = None
        for name, info in DIR_INFO.items():
            hr, hc = info["hole"]
            if int(patch[hr, hc]) != 0:
                continue
            ok = True
            for rr in range(3):
                for cc in range(3):
                    expected = 0 if (rr, cc) == (hr, hc) else color
                    if int(patch[rr, cc]) != expected:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                direction = name
                break
        if direction is None:
            continue
        blocks.append(
            {
                "top": top,
                "left": left,
                "color": int(color),
                "direction": direction,
            }
        )
    blocks.sort(key=lambda b: (int(b["top"]), int(b["left"]), int(b["color"])))
    return blocks


def _project_band(block: dict, obstacle_mask: np.ndarray) -> tuple[list[tuple[int, int]], str]:
    h, w = obstacle_mask.shape
    top = int(block["top"])
    left = int(block["left"])
    direction = block["direction"]
    band: list[tuple[int, int]] = []

    if direction == "up":
        row = top - 1
        while row >= 0:
            cells = [(row, c) for c in range(left, left + 3)]
            if any(bool(obstacle_mask[r, c]) for r, c in cells):
                return band, "block"
            band.extend(cells)
            row -= 1
        return band, "boundary"

    if direction == "down":
        row = top + 3
        while row < h:
            cells = [(row, c) for c in range(left, left + 3)]
            if any(bool(obstacle_mask[r, c]) for r, c in cells):
                return band, "block"
            band.extend(cells)
            row += 1
        return band, "boundary"

    if direction == "left":
        col = left - 1
        while col >= 0:
            cells = [(r, col) for r in range(top, top + 3)]
            if any(bool(obstacle_mask[r, c]) for r, c in cells):
                return band, "block"
            band.extend(cells)
            col -= 1
        return band, "boundary"

    col = left + 3
    while col < w:
        cells = [(r, col) for r in range(top, top + 3)]
        if any(bool(obstacle_mask[r, c]) for r, c in cells):
            return band, "block"
        band.extend(cells)
        col += 1
    return band, "boundary"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each holed 3x3 color block into a same-color strip in the hole-opposite direction.

    Input structure:
    - Black (0) background fills the canvas.
    - Several monochrome 3x3 blocks are scattered on the grid; each block has exactly one black hole
      in one of the four cells adjacent to its center.
    - Other colored cells, when present, behave as obstacles but only valid holed 3x3 blocks emit strips.

    Rule:
    - Detect every 3x3 monochrome block whose unique hole is above, below, left, or right of the center.
    - The hole encodes the opposite projection direction: top hole -> project down, bottom hole -> project up,
      left hole -> project right, right hole -> project left.
    - From the block, paint a 3-pixel-wide strip of the same color one row or column at a time until the next
      step would hit the grid boundary or any existing non-black obstacle. The original block, including its hole,
      stays unchanged.

    Key insight:
    - The missing cell is a directional marker rather than damage: its offset from the center specifies the reverse
      projection direction, while the block's fixed 3x3 size locks the strip width to exactly three cells.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    obstacle_mask = g != 0
    for block in _detect_blocks(g):
        band, _ = _project_band(block, obstacle_mask)
        for r, c in band:
            out[r, c] = int(block["color"])
    return out


def _footprint_cells(top: int, left: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(top, top + 3) for c in range(left, left + 3)}


def _color_cells(spec: dict) -> set[tuple[int, int]]:
    hole_r, hole_c = DIR_INFO[spec["direction"]]["hole"]
    cells = _footprint_cells(int(spec["top"]), int(spec["left"]))
    cells.remove((int(spec["top"]) + hole_r, int(spec["left"]) + hole_c))
    return cells


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _band_to_boundary(top: int, left: int, direction: str, h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    if direction == "up":
        for row in range(top - 1, -1, -1):
            for col in range(left, left + 3):
                cells.add((row, col))
    elif direction == "down":
        for row in range(top + 3, h):
            for col in range(left, left + 3):
                cells.add((row, col))
    elif direction == "left":
        for col in range(left - 1, -1, -1):
            for row in range(top, top + 3):
                cells.add((row, col))
    else:
        for col in range(left + 3, w):
            for row in range(top, top + 3):
                cells.add((row, col))
    return cells


def _band_between_blocks(src_top: int, src_left: int, direction: str, gap: int) -> set[tuple[int, int]]:
    cells = set()
    if direction == "up":
        for row in range(src_top - gap, src_top):
            for col in range(src_left, src_left + 3):
                cells.add((row, col))
    elif direction == "down":
        for row in range(src_top + 3, src_top + 3 + gap):
            for col in range(src_left, src_left + 3):
                cells.add((row, col))
    elif direction == "left":
        for col in range(src_left - gap, src_left):
            for row in range(src_top, src_top + 3):
                cells.add((row, col))
    else:
        for col in range(src_left + 3, src_left + 3 + gap):
            for row in range(src_top, src_top + 3):
                cells.add((row, col))
    return cells


def _paint_block(grid: np.ndarray, spec: dict) -> None:
    top = int(spec["top"])
    left = int(spec["left"])
    color = int(spec["color"])
    grid[top : top + 3, left : left + 3] = color
    hole_r, hole_c = DIR_INFO[spec["direction"]]["hole"]
    grid[top + hole_r, left + hole_c] = 0


def _try_place_pair(
    h: int,
    w: int,
    direction: str,
    colors: tuple[int, int],
    occupied_space: set[tuple[int, int]],
    block_clearance: set[tuple[int, int]],
) -> list[dict] | None:
    span_max = (w - 7) if direction in ("left", "right") else (h - 7)
    if span_max < 1:
        return None

    for _ in range(PAIR_TRIES):
        gap = random.randint(1, min(MAX_PAIR_GAP, span_max))
        if direction == "right":
            top = random.randint(0, h - 3)
            src_left = random.randint(0, w - (7 + gap))
            blocker_left = src_left + 3 + gap
            src_top = blocker_top = top
        elif direction == "left":
            top = random.randint(0, h - 3)
            src_left = random.randint(3 + gap, w - 4)
            blocker_left = src_left - 3 - gap
            src_top = blocker_top = top
        elif direction == "down":
            left = random.randint(0, w - 3)
            src_top = random.randint(0, h - (7 + gap))
            blocker_top = src_top + 3 + gap
            src_left = blocker_left = left
        else:
            left = random.randint(0, w - 3)
            src_top = random.randint(3 + gap, h - 4)
            blocker_top = src_top - 3 - gap
            src_left = blocker_left = left

        source = {
            "top": int(src_top),
            "left": int(src_left),
            "color": int(colors[0]),
            "direction": direction,
            "stop_mode": "block",
        }
        blocker = {
            "top": int(blocker_top),
            "left": int(blocker_left),
            "color": int(colors[1]),
            "direction": direction,
            "stop_mode": "boundary",
        }

        source_footprint = _footprint_cells(int(source["top"]), int(source["left"]))
        blocker_footprint = _footprint_cells(int(blocker["top"]), int(blocker["left"]))
        if source_footprint & block_clearance or blocker_footprint & block_clearance:
            continue

        source_band = _band_between_blocks(int(source["top"]), int(source["left"]), direction, gap)
        blocker_band = _band_to_boundary(int(blocker["top"]), int(blocker["left"]), direction, h, w)
        if not source_band or not blocker_band:
            continue

        occupied = source_footprint | blocker_footprint | source_band | blocker_band
        if occupied & occupied_space:
            continue

        source["band"] = source_band
        blocker["band"] = blocker_band
        return [source, blocker]

    return None


def _try_place_boundary_block(
    h: int,
    w: int,
    direction: str,
    color: int,
    occupied_space: set[tuple[int, int]],
    block_clearance: set[tuple[int, int]],
) -> dict | None:
    for _ in range(BOUNDARY_TRIES):
        if direction == "right":
            top = random.randint(0, h - 3)
            left = random.randint(0, w - 4)
        elif direction == "left":
            top = random.randint(0, h - 3)
            left = random.randint(1, w - 3)
        elif direction == "down":
            top = random.randint(0, h - 4)
            left = random.randint(0, w - 3)
        else:
            top = random.randint(1, h - 3)
            left = random.randint(0, w - 3)

        footprint = _footprint_cells(top, left)
        if footprint & block_clearance:
            continue
        band = _band_to_boundary(top, left, direction, h, w)
        if not band:
            continue
        occupied = footprint | band
        if occupied & occupied_space:
            continue
        return {
            "top": int(top),
            "left": int(left),
            "color": int(color),
            "direction": direction,
            "band": band,
            "stop_mode": "boundary",
        }
    return None


def _render_expected(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        for r, c in spec["band"]:
            out[r, c] = int(spec["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    block_count = random.randint(*BLOCK_COUNT_RANGE)
    pair_count = random.randint(1, block_count // 2)
    single_count = block_count - 2 * pair_count

    min_dim = max(GRID_MIN, 8 + 3 * block_count)
    max_dim = min(GRID_MAX, min_dim + 10)
    h = random.randint(min_dim, max_dim)
    w = random.randint(min_dim, max_dim)

    if block_count > len(COLOR_POOL):
        return None
    colors = random.sample(COLOR_POOL, block_count)

    specs: list[dict] = []
    occupied_space: set[tuple[int, int]] = set()
    block_clearance: set[tuple[int, int]] = set()

    directions = list(DIR_INFO.keys())
    for pair_index in range(pair_count):
        direction = random.choice(directions)
        pair = _try_place_pair(
            h,
            w,
            direction,
            (colors[2 * pair_index], colors[2 * pair_index + 1]),
            occupied_space,
            block_clearance,
        )
        if pair is None:
            return None
        for spec in pair:
            footprint = _footprint_cells(int(spec["top"]), int(spec["left"]))
            occupied_space |= footprint | set(spec["band"])
            block_clearance |= _expand_cells(footprint, h, w, radius=1)
            specs.append(spec)

    for idx in range(single_count):
        color = colors[2 * pair_count + idx]
        direction = random.choice(directions)
        spec = _try_place_boundary_block(h, w, direction, color, occupied_space, block_clearance)
        if spec is None:
            return None
        footprint = _footprint_cells(int(spec["top"]), int(spec["left"]))
        occupied_space |= footprint | set(spec["band"])
        block_clearance |= _expand_cells(footprint, h, w, radius=1)
        specs.append(spec)

    inp = np.zeros((h, w), dtype=int)
    for spec in specs:
        _paint_block(inp, spec)

    expected = _render_expected(inp, specs)

    detected = _detect_blocks(inp)
    if len(detected) != block_count:
        return None

    by_key = {(int(s["top"]), int(s["left"])): s for s in specs}
    stop_modes = []
    obstacle_mask = inp != 0
    for block in detected:
        key = (int(block["top"]), int(block["left"]))
        if key not in by_key:
            return None
        band, stop_mode = _project_band(block, obstacle_mask)
        if set(band) != set(by_key[key]["band"]):
            return None
        stop_modes.append(stop_mode)

    if "block" not in stop_modes or "boundary" not in stop_modes:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
