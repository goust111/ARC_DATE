import random
from collections import Counter

import numpy as np


ORIENTATIONS = ("tl", "tr", "bl", "br")
FILLED_OFFSETS = {
    "tl": ((0, 1), (1, 0), (1, 1)),
    "tr": ((0, 0), (1, 0), (1, 1)),
    "bl": ((0, 0), (0, 1), (1, 1)),
    "br": ((0, 0), (0, 1), (1, 0)),
}
PALETTE_SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 300
FRAME_COUNT_CHOICES = (2, 3, 4, 5)
FRAME_SIZE_CHOICES = (6, 8, 10)
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 26
PALETTE_GAP = 1
PALETTE_SPAN = 4 * 2 + 3 * PALETTE_GAP


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _l_block_info(block: np.ndarray, bg: int) -> tuple[int, str] | None:
    if block.shape != (2, 2):
        return None
    bg_cells = np.argwhere(block == bg)
    if len(bg_cells) != 1:
        return None
    colors = [int(v) for v in block.ravel() if int(v) != bg]
    if len(colors) != 3 or len(set(colors)) != 1:
        return None
    missing = tuple(int(x) for x in bg_cells[0])
    orientation = {(0, 0): "tl", (0, 1): "tr", (1, 0): "bl", (1, 1): "br"}[missing]
    return colors[0], orientation


def _palette_units_on_side(grid: np.ndarray, bg: int, side: str) -> list[tuple[int, int, int, str]]:
    h, w = grid.shape
    units: list[tuple[int, int, int, str]] = []
    if side == "top":
        r = 0
        for c in range(w - 1):
            info = _l_block_info(grid[r : r + 2, c : c + 2], bg)
            if info is not None:
                color, ori = info
                units.append((r, c, color, ori))
    elif side == "bottom":
        r = h - 2
        for c in range(w - 1):
            info = _l_block_info(grid[r : r + 2, c : c + 2], bg)
            if info is not None:
                color, ori = info
                units.append((r, c, color, ori))
    elif side == "left":
        c = 0
        for r in range(h - 1):
            info = _l_block_info(grid[r : r + 2, c : c + 2], bg)
            if info is not None:
                color, ori = info
                units.append((r, c, color, ori))
    elif side == "right":
        c = w - 2
        for r in range(h - 1):
            info = _l_block_info(grid[r : r + 2, c : c + 2], bg)
            if info is not None:
                color, ori = info
                units.append((r, c, color, ori))
    else:
        raise ValueError(f"unknown side {side}")
    return units


def _find_palette_mapping(grid: np.ndarray, bg: int) -> dict[str, int]:
    candidates: list[tuple[str, dict[str, int]]] = []
    for side in PALETTE_SIDES:
        units = _palette_units_on_side(grid, bg, side)
        if len(units) != 4:
            continue
        orientation_to_color: dict[str, int] = {}
        valid = True
        for _, _, color, orientation in units:
            if orientation in orientation_to_color and orientation_to_color[orientation] != color:
                valid = False
                break
            orientation_to_color[orientation] = color
        if valid and set(orientation_to_color) == set(ORIENTATIONS) and len(set(orientation_to_color.values())) == 4:
            candidates.append((side, orientation_to_color))
    assert len(candidates) == 1, f"expected one palette side, got {len(candidates)}"
    return candidates[0][1]


def _is_rectangular_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    if len(points) != 2 * (h + w) - 4:
        return False
    rel = {(r - r0, c - c0) for r, c in points}
    for rr in range(h):
        for cc in range(w):
            on_border = rr in (0, h - 1) or cc in (0, w - 1)
            present = (rr, cc) in rel
            if on_border != present:
                return False
    return True


def _interior_l_orientation(grid: np.ndarray, bg: int, frame_points: list[tuple[int, int]]) -> str:
    r0, c0, r1, c1 = _bbox(frame_points)
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    pts = np.argwhere(interior != bg)
    assert len(pts) == 3, f"expected one 3-cell marker, got {len(pts)}"
    rows = [int(p[0]) for p in pts]
    cols = [int(p[1]) for p in pts]
    rr0, cc0 = min(rows), min(cols)
    rr1, cc1 = max(rows), max(cols)
    assert rr1 - rr0 == 1 and cc1 - cc0 == 1, "marker must occupy a 2x2 block"
    rel = {(int(r - rr0), int(c - cc0)) for r, c in pts}
    missing = ({(0, 0), (0, 1), (1, 0), (1, 1)} - rel).pop()
    return {(0, 0): "tl", (0, 1): "tr", (1, 0): "bl", (1, 1): "br"}[missing]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each rectangular frame border using the palette color keyed by the
    orientation of the 3-cell L marker inside that frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    palette = _find_palette_mapping(g, bg)

    out = g.copy()
    for points in _components(g != bg):
        if not _is_rectangular_frame(points):
            continue
        orientation = _interior_l_orientation(g, bg, points)
        new_color = palette[orientation]
        for r, c in points:
            out[r, c] = new_color
    return out


def _draw_l(grid: np.ndarray, top: int, left: int, orientation: str, color: int) -> None:
    for dr, dc in FILLED_OFFSETS[orientation]:
        grid[top + dr, left + dc] = color


def _body_bounds(h: int, w: int, side: str) -> tuple[int, int, int, int]:
    r0, r1 = 1, h - 1
    c0, c1 = 1, w - 1
    if side == "top":
        r0 = 3
    elif side == "bottom":
        r1 = h - 3
    elif side == "left":
        c0 = 3
    elif side == "right":
        c1 = w - 3
    return r0, r1, c0, c1


def _place_palette(
    grid: np.ndarray,
    side: str,
    orientation_to_color: dict[str, int],
    rng: random.Random,
) -> None:
    h, w = grid.shape
    order = list(ORIENTATIONS)
    rng.shuffle(order)
    if side in ("top", "bottom"):
        start = rng.randint(1, w - PALETTE_SPAN - 1)
        row = 0 if side == "top" else h - 2
        for idx, orientation in enumerate(order):
            col = start + idx * (2 + PALETTE_GAP)
            _draw_l(grid, row, col, orientation, orientation_to_color[orientation])
    else:
        start = rng.randint(1, h - PALETTE_SPAN - 1)
        col = 0 if side == "left" else w - 2
        for idx, orientation in enumerate(order):
            row = start + idx * (2 + PALETTE_GAP)
            _draw_l(grid, row, col, orientation, orientation_to_color[orientation])


def _frame_marker_top_left(top: int, left: int, height: int, width: int) -> tuple[int, int]:
    inner_h = height - 2
    inner_w = width - 2
    return top + 1 + inner_h // 2 - 1, left + 1 + inner_w // 2 - 1


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + height + 1)
    c1 = min(w, left + width + 1)
    return not occupied[r0:r1, c0:c1].any()


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, border_color: int) -> None:
    grid[top, left : left + width] = border_color
    grid[top + height - 1, left : left + width] = border_color
    grid[top : top + height, left] = border_color
    grid[top : top + height, left + width - 1] = border_color


def _choose_frame_orientations(count: int, rng: random.Random) -> list[str]:
    pool = list(ORIENTATIONS)
    rng.shuffle(pool)
    result = pool[: min(count, 4)]
    while len(result) < count:
        result.append(rng.choice(ORIENTATIONS))
    rng.shuffle(result)
    return result


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    w = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    if h < PALETTE_SPAN + 2:
        h = PALETTE_SPAN + 2
    if w < PALETTE_SPAN + 2:
        w = PALETTE_SPAN + 2

    side = rng.choice(PALETTE_SIDES)
    body_r0, body_r1, body_c0, body_c1 = _body_bounds(h, w, side)
    body_h = body_r1 - body_r0
    body_w = body_c1 - body_c0
    if body_h < 10 or body_w < 10:
        return None

    colors = list(range(10))
    rng.shuffle(colors)
    bg = colors.pop()
    border_color = colors.pop()
    palette_colors = [colors.pop() for _ in range(4)]
    orientation_to_color = dict(zip(ORIENTATIONS, palette_colors))
    inner_colors = colors
    if not inner_colors:
        return None

    grid = np.full((h, w), bg, dtype=int)
    _place_palette(grid, side, orientation_to_color, rng)

    max_frames = min(max(2, body_h * body_w // 70), 5)
    possible_counts = [n for n in FRAME_COUNT_CHOICES if n <= max_frames]
    if not possible_counts:
        return None
    frame_count = rng.choice(possible_counts)
    frame_orientations = _choose_frame_orientations(frame_count, rng)

    sizes: list[tuple[int, int]] = []
    for _ in range(frame_count):
        fh = rng.choice(FRAME_SIZE_CHOICES)
        fw = rng.choice(FRAME_SIZE_CHOICES)
        if fh >= body_h or fw >= body_w:
            return None
        sizes.append((fh, fw))
    sizes.sort(key=lambda item: item[0] * item[1], reverse=True)

    occupied = np.zeros((h, w), dtype=bool)
    placed: list[tuple[int, int, int, int]] = []
    for fh, fw in sizes:
        choices: list[tuple[int, int]] = []
        for top in range(body_r0, body_r1 - fh + 1):
            for left in range(body_c0, body_c1 - fw + 1):
                if _can_place(occupied, top, left, fh, fw):
                    choices.append((top, left))
        if not choices:
            return None
        top, left = rng.choice(choices)
        occupied[top : top + fh, left : left + fw] = True
        placed.append((top, left, fh, fw))

    rng.shuffle(placed)
    for (top, left, fh, fw), orientation in zip(placed, frame_orientations):
        _draw_frame(grid, top, left, fh, fw, border_color)
        marker_top, marker_left = _frame_marker_top_left(top, left, fh, fw)
        marker_color = rng.choice(inner_colors)
        _draw_l(grid, marker_top, marker_left, orientation, marker_color)

    counts = Counter(int(v) for v in grid.ravel())
    if counts[bg] != max(counts.values()):
        return None
    if sum(1 for value in counts.values() if value == counts[bg]) != 1:
        return None

    try:
        palette = _find_palette_mapping(grid, bg)
        if set(palette) != set(ORIENTATIONS):
            return None
        frame_components = [pts for pts in _components(grid != bg) if _is_rectangular_frame(pts)]
        if len(frame_components) != frame_count:
            return None
        for pts in frame_components:
            _ = _interior_l_orientation(grid, bg, pts)
        _ = main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
