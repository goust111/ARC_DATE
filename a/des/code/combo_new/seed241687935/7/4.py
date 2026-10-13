import random

import numpy as np


BG = 0
ARROW_COLOR = 9
GEN_ATTEMPTS = 400
GEN_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)

ARROW_PATTERNS = {
    "right": {(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)},
    "left": {(1, 0), (1, 1), (1, 2), (0, 0), (2, 0)},
    "up": {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    "down": {(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)},
}

HOLE_PLANS = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 1, 1),
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 2),
    (0, 2, 3),
    (0, 1, 2, 3),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_connected(mask: np.ndarray) -> bool:
    return bool(mask.size) and len(_mask_components(mask)) == 1


def _count_holes_in_mask(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_edge = False
            while stack:
                cr, cc = stack.pop()
                if cr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _arrow_array(direction: str) -> np.ndarray:
    arrow = np.zeros((3, 3), dtype=int)
    for r, c in ARROW_PATTERNS[direction]:
        arrow[r, c] = ARROW_COLOR
    return arrow


def _find_arrow(grid: np.ndarray) -> tuple[str, list[tuple[int, int]]]:
    comps = _mask_components(grid == ARROW_COLOR)
    assert len(comps) == 1, "expected exactly one maroon arrow"
    coords = comps[0]
    assert len(coords) == 5, "arrow must contain exactly five pixels"
    r0, c0, r1, c1 = _bbox(coords)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "arrow must fit a 3x3 box"
    normalized = {(r - r0, c - c0) for r, c in coords}
    for direction, pattern in ARROW_PATTERNS.items():
        if normalized == pattern:
            return direction, coords
    raise AssertionError("unrecognized arrow geometry")


def _object_sprite(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> tuple[np.ndarray, int, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    holes = _count_holes_in_mask(mask)
    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = color
    return sprite, holes, r0, c0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each colored object in the arrow direction as many times as its enclosed black-hole count.

    Input structure:
    - The background is black.
    - One maroon 5-pixel arrow appears as a 3-cell shaft plus a 2-cell arrowhead, pointing up, down, left, or right.
    - Every other non-black object is a single-color 4-connected component, and different objects use different colors.
    - An object may contain zero to three internal black holes.

    Rule:
    - Detect the arrow direction and erase the arrow to black in the output.
    - For each non-arrow object, count its internal black holes inside its tight bounding box.
    - Keep the original object in place. If the hole count is n >= 1, place n full translated copies of that object, preserving its holes, in the arrow direction.
    - Consecutive copies are separated from each other, and from the original, by exactly one black row or one black column.

    Key insight:
    - The maroon symbol gives a single global translation direction, while each object's copy count is determined independently by a structural property: the number of enclosed black cavities.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    direction, arrow_coords = _find_arrow(g)
    out = g.copy()
    for r, c in arrow_coords:
        out[r, c] = BG

    for color in sorted(int(v) for v in np.unique(g) if v not in (BG, ARROW_COLOR)):
        comps = _mask_components(g == color)
        assert len(comps) == 1, "each object color must form exactly one component"
        sprite, hole_count, top, left = _object_sprite(g, comps[0], color)
        sh, sw = sprite.shape

        if direction == "right":
            delta_r, delta_c = 0, sw + 1
        elif direction == "left":
            delta_r, delta_c = 0, -(sw + 1)
        elif direction == "down":
            delta_r, delta_c = sh + 1, 0
        else:
            delta_r, delta_c = -(sh + 1), 0

        sprite_cells = np.argwhere(sprite == color)
        for k in range(1, hole_count + 1):
            base_r = top + k * delta_r
            base_c = left + k * delta_c
            for sr, sc in sprite_cells:
                rr = int(base_r + sr)
                cc = int(base_c + sc)
                assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "copy exceeded grid bounds"
                out[rr, cc] = color

    return out


def _random_split(total: int, slots: int, rng: random.Random) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[rng.randrange(slots)] += 1
    return parts


def _sample_hole_cells(h: int, w: int, hole_count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    rng.shuffle(interior)
    holes: list[tuple[int, int]] = []
    for r, c in interior:
        if all(abs(r - pr) + abs(c - pc) > 1 for pr, pc in holes):
            holes.append((r, c))
        if len(holes) == hole_count:
            return holes
    return None


def _make_polyomino_mask(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(3, 5)
    w = rng.randint(3, 5)
    target = rng.randint(max(5, (h * w) // 2), h * w - 1)
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    for _ in range(target - 1):
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(sorted(frontier)))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    mask = _trim_mask(mask)
    if mask.size == 0 or np.count_nonzero(mask) < 5:
        return None
    if not _mask_connected(mask) or _count_holes_in_mask(mask) != 0:
        return None
    return mask


def _valid_mask(mask: np.ndarray, hole_count: int) -> bool:
    if mask.size == 0 or np.count_nonzero(mask) < max(5, hole_count + 4):
        return False
    return _mask_connected(mask) and _count_holes_in_mask(mask) == hole_count


def _make_sprite(color: int, hole_count: int, rng: random.Random) -> np.ndarray:
    for _ in range(200):
        if hole_count == 0 and rng.random() < 0.45:
            mask = _make_polyomino_mask(rng)
            if mask is None:
                continue
        else:
            if hole_count == 0:
                h = rng.randint(3, 5)
                w = rng.randint(3, 5)
            elif hole_count == 1:
                h = rng.randint(4, 5)
                w = rng.randint(4, 5)
            elif hole_count == 2:
                h = rng.randint(4, 6)
                w = rng.randint(4, 6)
            else:
                h = rng.randint(5, 6)
                w = rng.randint(5, 6)

            mask = np.ones((h, w), dtype=bool)
            hole_cells = _sample_hole_cells(h, w, hole_count, rng)
            if hole_cells is None:
                continue
            for r, c in hole_cells:
                mask[r, c] = False

            if hole_count:
                for r, c in hole_cells:
                    if rng.random() >= 0.35:
                        continue
                    dirs = list(_neighbors4(0, 0))
                    rng.shuffle(dirs)
                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        if 1 <= nr < h - 1 and 1 <= nc < w - 1 and mask[nr, nc]:
                            mask[nr, nc] = False
                            trial = _trim_mask(mask)
                            if _valid_mask(trial, hole_count):
                                mask = trial
                                h, w = mask.shape
                            else:
                                mask[nr, nc] = True
                            break

            for _ in range(rng.randint(0, 3 if hole_count else 4)):
                h, w = mask.shape
                boundary = [
                    (r, c)
                    for r in range(h)
                    for c in range(w)
                    if mask[r, c] and (r in (0, h - 1) or c in (0, w - 1))
                ]
                rng.shuffle(boundary)
                removed = False
                for r, c in boundary:
                    trial = mask.copy()
                    trial[r, c] = False
                    trial = _trim_mask(trial)
                    if _valid_mask(trial, hole_count):
                        mask = trial
                        removed = True
                        break
                if not removed:
                    break

        k = rng.randrange(4)
        mask = np.rot90(mask, k)
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        mask = _trim_mask(mask)
        if not _valid_mask(mask, hole_count):
            continue

        sprite = np.zeros(mask.shape, dtype=int)
        sprite[mask] = color
        return sprite

    raise RuntimeError(f"failed to generate object with {hole_count} holes")


def _place_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int) -> bool:
    h, w = sprite.shape
    if top < 0 or left < 0 or top + h > grid.shape[0] or left + w > grid.shape[1]:
        return False
    region = grid[top : top + h, left : left + w]
    occupied = sprite != BG
    if np.any(region[occupied] != BG):
        return False
    region[occupied] = sprite[occupied]
    return True


def _layout_horizontal(
    direction: str,
    hole_counts: list[int],
    sprites: list[np.ndarray],
    rng: random.Random,
) -> np.ndarray | None:
    heights = [sprite.shape[0] for sprite in sprites]
    widths = [sprite.shape[1] for sprite in sprites]
    reserved_widths = [(holes + 1) * width + holes for holes, width in zip(hole_counts, widths)]

    arrow_side = rng.choice(("top", "bottom"))
    arrow_band = 4
    min_h = arrow_band + sum(heights) + (len(sprites) - 1)
    min_w = max(reserved_widths) + 2
    if min_h > 30 or min_w > 30:
        return None

    grid_h = rng.randint(min_h, min(30, min_h + 5))
    grid_w = rng.randint(min_w, min(30, min_w + 5))
    vertical_slack = grid_h - min_h
    gap_extras = _random_split(vertical_slack, len(sprites) + 1, rng)

    row_positions: list[int] = []
    cursor = (arrow_band if arrow_side == "top" else 0) + gap_extras[0]
    for index, height in enumerate(heights):
        row_positions.append(cursor)
        cursor += height
        if index < len(sprites) - 1:
            cursor += 1 + gap_extras[index + 1]

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for row, sprite, holes, width in zip(row_positions, sprites, hole_counts, widths):
        reserved = (holes + 1) * width + holes
        start = rng.randint(1, grid_w - reserved - 1)
        left = start if direction == "right" else start + holes * (width + 1)
        if not _place_sprite(grid, sprite, row, left):
            return None

    arrow = _arrow_array(direction)
    arrow_row = 0 if arrow_side == "top" else grid_h - 3
    arrow_col = rng.randint(0, grid_w - 3)
    if not _place_sprite(grid, arrow, arrow_row, arrow_col):
        return None
    return grid


def _layout_vertical(
    direction: str,
    hole_counts: list[int],
    sprites: list[np.ndarray],
    rng: random.Random,
) -> np.ndarray | None:
    heights = [sprite.shape[0] for sprite in sprites]
    widths = [sprite.shape[1] for sprite in sprites]
    reserved_heights = [(holes + 1) * height + holes for holes, height in zip(hole_counts, heights)]

    arrow_side = rng.choice(("left", "right"))
    arrow_band = 4
    min_h = max(reserved_heights) + 2
    min_w = arrow_band + sum(widths) + (len(sprites) - 1)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = rng.randint(min_h, min(30, min_h + 5))
    grid_w = rng.randint(min_w, min(30, min_w + 5))
    horizontal_slack = grid_w - min_w
    gap_extras = _random_split(horizontal_slack, len(sprites) + 1, rng)

    col_positions: list[int] = []
    cursor = (arrow_band if arrow_side == "left" else 0) + gap_extras[0]
    for index, width in enumerate(widths):
        col_positions.append(cursor)
        cursor += width
        if index < len(sprites) - 1:
            cursor += 1 + gap_extras[index + 1]

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for col, sprite, holes, height in zip(col_positions, sprites, hole_counts, heights):
        reserved = (holes + 1) * height + holes
        start = rng.randint(1, grid_h - reserved - 1)
        top = start if direction == "down" else start + holes * (height + 1)
        if not _place_sprite(grid, sprite, top, col):
            return None

    arrow = _arrow_array(direction)
    arrow_col = 0 if arrow_side == "left" else grid_w - 3
    arrow_row = rng.randint(0, grid_h - 3)
    if not _place_sprite(grid, arrow, arrow_row, arrow_col):
        return None
    return grid


def _validate_generated_grid(grid: np.ndarray, hole_counts: list[int]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    try:
        direction, _ = _find_arrow(grid)
    except AssertionError:
        return False
    if direction not in ARROW_PATTERNS:
        return False

    seen_zero = False
    seen_positive = False
    for color in (int(v) for v in np.unique(grid) if v not in (BG, ARROW_COLOR)):
        comps = _mask_components(grid == color)
        if len(comps) != 1:
            return False
        sprite, holes, _, _ = _object_sprite(grid, comps[0], color)
        if holes == 0:
            seen_zero = True
        else:
            seen_positive = True
        if holes < 0 or holes > 3 or sprite.size == 0:
            return False
    if not seen_zero or not seen_positive:
        return False

    out = main(grid)
    return out.ndim == 2 and out.shape == grid.shape and max(out.shape) <= 30


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    direction = rng.choice(("right", "left", "up", "down"))
    hole_counts = list(rng.choice(HOLE_PLANS))
    rng.shuffle(hole_counts)

    colors = rng.sample(GEN_COLORS, len(hole_counts))
    sprites = [_make_sprite(color, holes, rng) for color, holes in zip(colors, hole_counts)]

    if direction in ("right", "left"):
        grid = _layout_horizontal(direction, hole_counts, sprites, rng)
    else:
        grid = _layout_vertical(direction, hole_counts, sprites, rng)
    if grid is None:
        return None

    if not _validate_generated_grid(grid, hole_counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
