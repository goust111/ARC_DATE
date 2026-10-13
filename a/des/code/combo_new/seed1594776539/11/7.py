import random

import numpy as np


BG = 0
GREY = 5
BORDER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "right", "bottom", "left")
ROTATIONS = {"top": 0, "right": 3, "bottom": 2, "left": 1}
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _gap_side(r: int, c: int, height: int, width: int) -> str:
    if r == 0 and 0 < c < width - 1:
        return "top"
    if c == width - 1 and 0 < r < height - 1:
        return "right"
    if r == height - 1 and 0 < c < width - 1:
        return "bottom"
    if c == 0 and 0 < r < height - 1:
        return "left"
    raise AssertionError("gap must lie on exactly one non-corner border side")


def _parse_region(grid: np.ndarray, color: int, points: list[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = _bbox(points)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    height, width = patch.shape
    assert height >= 4 and width >= 4, "frame must have a non-empty interior with margin"

    border = _frame_mask(height, width)
    color_mask = patch == color
    assert not np.any(color_mask & ~border), "border color must only appear on the rectangle outline"

    missing = np.argwhere(border & ~color_mask)
    interior = patch[1:-1, 1:-1]
    assert interior.shape[0] == interior.shape[1], "interior must be square to support quarter-turn rotation"

    if len(missing) == 0:
        assert np.any(interior == GREY), "source region must contain a grey pattern"
        assert np.all((interior == BG) | (interior == GREY)), "source interior may only contain black and grey"
        return {
            "kind": "source",
            "bbox": (r0, c0, r1, c1),
            "pattern": interior == GREY,
            "inner_size": interior.shape[0],
        }

    assert len(missing) == 1, "target frame must have exactly one missing border pixel"
    mr, mc = map(int, missing[0])
    assert patch[mr, mc] == BG, "gap cell must be black"
    assert np.all(interior == BG), "target interior must start empty"
    return {
        "kind": "target",
        "bbox": (r0, c0, r1, c1),
        "gap_side": _gap_side(mr, mc, height, width),
        "inner_size": interior.shape[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each grey source pattern into its same-colored target frame after rotating it by the target frame's border-gap direction.

    Input structure:
    - Black background with several separated one-pixel-thick colored rectangular frames.
    - Each non-grey border color appears exactly twice, forming one source/target pair with equal square interiors.
    - The source frame has a complete border and contains a grey pattern inset from the border, while the target frame has an empty black interior and exactly one black gap on one border side.

    Rule:
    - Group frames by border color to form pairs.
    - Within each pair, treat the complete frame as the source and the single-gap frame as the target.
    - Read the target gap as a rotation instruction for the source interior pattern: top = 0 degrees, right = 90 degrees clockwise, bottom = 180 degrees, left = 270 degrees clockwise.
    - Copy the rotated grey pattern into the target interior, keeping every border pixel, source pattern pixel, and target gap unchanged.

    Key insight:
    - The visible border color solves the pairing problem, while the lone gap on the empty mate is not a missing wall to repair but a directional control symbol for how the source pattern should turn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    colors = [int(color) for color in np.unique(g) if int(color) not in (BG, GREY)]
    assert colors, "expected at least one colored frame pair"

    for color in colors:
        comps = _components(g == color)
        assert len(comps) == 2, "each border color must form exactly two regions"
        infos = [_parse_region(g, color, comp) for comp in comps]

        sources = [info for info in infos if info["kind"] == "source"]
        targets = [info for info in infos if info["kind"] == "target"]
        assert len(sources) == 1 and len(targets) == 1, "each color must have one source and one target"

        source = sources[0]
        target = targets[0]
        assert source["inner_size"] == target["inner_size"], "paired interiors must match"

        pattern = source["pattern"]
        rotated = np.rot90(pattern, ROTATIONS[target["gap_side"]])
        r0, c0, r1, c1 = target["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[rotated] = GREY

    return out


def _grow_connected(core_size: int, target_area: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(core_size), random.randrange(core_size))
    cells = {start}
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < core_size and 0 <= nc < core_size and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _make_pattern(inner_size: int) -> np.ndarray:
    core_size = inner_size - 2
    assert core_size >= 2
    max_area = core_size * core_size - 1
    for _ in range(200):
        area = random.randint(2, max_area)
        cells = _grow_connected(core_size, area)
        if cells is None:
            continue
        pattern = np.zeros((inner_size, inner_size), dtype=bool)
        for r, c in cells:
            pattern[r + 1, c + 1] = True
        if all(not np.array_equal(pattern, np.rot90(pattern, k)) for k in (1, 2, 3)):
            return pattern
    raise RuntimeError("failed to create a non-symmetric pattern")


def _balanced_sides(count: int) -> list[str]:
    full_cycles, extra = divmod(count, len(SIDES))
    sides = list(SIDES) * full_cycles + random.sample(list(SIDES), extra)
    random.shuffle(sides)
    return sides


def _distribute(extra: int, parts: int) -> list[int]:
    bins = [1] * parts
    for _ in range(extra):
        bins[random.randrange(parts)] += 1
    return bins


def _slot_starts(count: int, outer: int, total: int) -> list[int]:
    min_total = count * outer + (count + 1)
    assert min_total <= total
    gaps = _distribute(total - min_total, count + 1)
    starts: list[int] = []
    pos = gaps[0]
    for idx in range(count):
        starts.append(pos)
        pos += outer + gaps[idx + 1]
    return starts


def _feasible_layouts(obj_count: int, outer: int) -> list[tuple[int, int]]:
    layouts = []
    for rows in range(1, obj_count + 1):
        for cols in range(1, obj_count + 1):
            if rows * cols < obj_count:
                continue
            if rows * outer + (rows + 1) > 30:
                continue
            if cols * outer + (cols + 1) > 30:
                continue
            layouts.append((rows, cols))
    return layouts


def _paint_region(
    grid: np.ndarray,
    top: int,
    left: int,
    outer: int,
    color: int,
    pattern: np.ndarray | None,
    gap_side: str | None,
    gap_index: int | None,
) -> None:
    bottom = top + outer - 1
    right = left + outer - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color

    if gap_side is not None:
        assert gap_index is not None
        if gap_side == "top":
            grid[top, left + gap_index] = BG
        elif gap_side == "right":
            grid[top + gap_index, right] = BG
        elif gap_side == "bottom":
            grid[bottom, left + gap_index] = BG
        else:
            grid[top + gap_index, left] = BG

    if pattern is not None:
        interior = grid[top + 1 : bottom, left + 1 : right]
        interior[pattern] = GREY


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(1, 4)
    inner_size = random.randint(4, 6)
    outer = inner_size + 2
    obj_count = 2 * n_pairs

    layouts = _feasible_layouts(obj_count, outer)
    if not layouts:
        return None
    slot_rows, slot_cols = random.choice(layouts)

    min_height = slot_rows * outer + (slot_rows + 1)
    min_width = slot_cols * outer + (slot_cols + 1)
    height = random.randint(min_height, 30)
    width = random.randint(min_width, 30)

    row_starts = _slot_starts(slot_rows, outer, height)
    col_starts = _slot_starts(slot_cols, outer, width)
    slots = [(r, c) for r in row_starts for c in col_starts]
    chosen_slots = random.sample(slots, obj_count)
    random.shuffle(chosen_slots)

    colors = random.sample(BORDER_COLORS, n_pairs)
    sides = _balanced_sides(n_pairs)

    inp = np.full((height, width), BG, dtype=int)
    expected = inp.copy()

    for idx, color in enumerate(colors):
        pattern = _make_pattern(inner_size)
        source_top, source_left = chosen_slots[2 * idx]
        target_top, target_left = chosen_slots[2 * idx + 1]
        gap_side = sides[idx]
        gap_index = random.randint(1, inner_size)

        _paint_region(inp, source_top, source_left, outer, color, pattern, None, None)
        _paint_region(inp, target_top, target_left, outer, color, None, gap_side, gap_index)

        _paint_region(expected, source_top, source_left, outer, color, pattern, None, None)
        rotated = np.rot90(pattern, ROTATIONS[gap_side])
        _paint_region(expected, target_top, target_left, outer, color, rotated, gap_side, gap_index)

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
