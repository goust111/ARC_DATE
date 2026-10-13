import random

import numpy as np


BG = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GRID_SIZES = tuple(range(17, 30, 2))
QUADRANTS = ("ul", "ur", "lr", "ll")
ROTATION_BY_QUADRANT = {
    "ul": 0,  # 0 degrees
    "ur": 3,  # 90 degrees clockwise
    "lr": 2,  # 180 degrees
    "ll": 1,  # 270 degrees clockwise
}
SHAPE_DIMS = ((3, 3), (3, 5), (5, 3), (5, 5))
GEN_RETRIES = 400
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(cells)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _cells_to_patch(cells: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        patch[r - r0, c - c0] = True
    return patch


def _quadrant_for_anchor(r: int, c: int, cr: int, cc: int) -> str:
    assert r != cr and c != cc, "anchor must be strictly inside one quadrant"
    if r < cr and c < cc:
        return "ul"
    if r < cr and c > cc:
        return "ur"
    if r > cr and c > cc:
        return "lr"
    return "ll"


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, color: int, bg: int) -> None:
    rr, cc = np.where(patch)
    for dr, dc in zip(rr.tolist(), cc.tolist()):
        tr, tc = top + dr, left + dc
        assert grid[tr, tc] in (bg, color)
        grid[tr, tc] = color


def _paint_centered(grid: np.ndarray, patch: np.ndarray, center_r: int, center_c: int, color: int, bg: int) -> None:
    ph, pw = patch.shape
    top = center_r - ph // 2
    left = center_c - pw // 2
    assert 0 <= top and top + ph <= grid.shape[0]
    assert 0 <= left and left + pw <= grid.shape[1]
    _paint_patch(grid, patch, top, left, color, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each monochrome stamp according to its same-colored anchor quadrant and re-stamp it on the anchor.

    Input structure:
    - The background is black and the grid has a unique center row and center column that divide it into four quadrants.
    - Each non-background color appears in exactly two 4-connected groups: one singleton anchor pixel and one larger monochrome stamp.
    - Every stamp is spatially separated from its anchor and occupies a compact 3x3 to 5x5 bounding box.

    Rule:
    - Pair each stamp with the singleton anchor of the same color.
    - Read the anchor quadrant relative to the grid center: upper-left -> 0 degrees, upper-right -> 90 degrees clockwise, lower-right -> 180 degrees, lower-left -> 270 degrees clockwise.
    - Erase the original stamp positions, rotate the stamp by that angle, and stamp it onto a blank background so the rotated bounding-box center lands on the anchor.
    - The anchor pixel is overwritten by the stamped result.

    Key insight:
    - Color tells you which anchor belongs to which stamp, while the anchor's quadrant encodes the rotation instruction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1

    bg = _get_bg(g)
    cr, cc = h // 2, w // 2
    out = np.full((h, w), bg, dtype=int)

    for color in [int(x) for x in np.unique(g) if int(x) != bg]:
        comps = _components(g == color)
        assert len(comps) == 2, "each color must have one stamp and one anchor"

        anchor_cells = [comp for comp in comps if len(comp) == 1]
        stamp_cells = [comp for comp in comps if len(comp) > 1]
        assert len(anchor_cells) == 1 and len(stamp_cells) == 1

        anchor_r, anchor_c = anchor_cells[0][0]
        stamp_patch = _cells_to_patch(stamp_cells[0])
        quadrant = _quadrant_for_anchor(anchor_r, anchor_c, cr, cc)
        rotated_patch = np.rot90(stamp_patch, ROTATION_BY_QUADRANT[quadrant])
        _paint_centered(out, rotated_patch, anchor_r, anchor_c, color, bg)

    return out


# --------------- generator ---------------


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    path = [(r, c)]
    while (r, c) != (er, ec):
        steps = []
        if r < er:
            steps.append((1, 0))
        elif r > er:
            steps.append((-1, 0))
        if c < ec:
            steps.append((0, 1))
        elif c > ec:
            steps.append((0, -1))
        dr, dc = random.choice(steps)
        r += dr
        c += dc
        path.append((r, c))
    return path


def _shape_is_valid(mask: np.ndarray) -> bool:
    h, w = mask.shape
    center = (h // 2, w // 2)
    if not mask[center]:
        return False
    if int(mask.sum()) < 4:
        return False
    if len(_components(mask)) != 1:
        return False
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) != h or len(cols) != w:
        return False
    for k in (1, 2, 3):
        if np.array_equal(mask, np.rot90(mask, k)):
            return False
    return True


def _grow_shape(h: int, w: int) -> np.ndarray | None:
    center = (h // 2, w // 2)
    edge_targets = [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]

    cells = {center}
    for target in edge_targets:
        cells.update(_random_path(center, target))

    area = h * w
    min_fill = max(4, len(cells))
    max_fill = min(area - 1, len(cells) + area // 3 + 2)
    if min_fill > max_fill:
        return None
    target_fill = random.randint(min_fill, max_fill)

    frontier = set()
    for r, c in cells:
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    while len(cells) < target_fill and frontier:
        nr, nc = random.choice(tuple(frontier))
        frontier.remove((nr, nc))
        cells.add((nr, nc))
        for dr, dc in DIRS:
            ar, ac = nr + dr, nc + dc
            if 0 <= ar < h and 0 <= ac < w and (ar, ac) not in cells:
                frontier.add((ar, ac))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask if _shape_is_valid(mask) else None


def _generate_shape() -> np.ndarray:
    for _ in range(200):
        h, w = random.choice(SHAPE_DIMS)
        mask = _grow_shape(h, w)
        if mask is not None:
            return mask
    raise RuntimeError("failed to generate a valid asymmetric stamp")


def _quadrant_bounds(name: str, h: int, w: int) -> tuple[int, int, int, int]:
    cr, cc = h // 2, w // 2
    if name == "ul":
        return 0, cr - 1, 0, cc - 1
    if name == "ur":
        return 0, cr - 1, cc + 1, w - 1
    if name == "lr":
        return cr + 1, h - 1, cc + 1, w - 1
    return cr + 1, h - 1, 0, cc - 1


def _anchor_candidates(h: int, w: int, quadrant: str, patch_shape: tuple[int, int]) -> list[tuple[int, int]]:
    q_r0, q_r1, q_c0, q_c1 = _quadrant_bounds(quadrant, h, w)
    ph, pw = patch_shape
    row_half = ph // 2
    col_half = pw // 2
    row_lo = q_r0 + 1 + row_half
    row_hi = q_r1 - 1 - row_half
    col_lo = q_c0 + 1 + col_half
    col_hi = q_c1 - 1 - col_half
    if row_lo > row_hi or col_lo > col_hi:
        return []
    return [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]


def _reserve(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + height + pad)
    c1 = min(occupied.shape[1], left + width + pad)
    occupied[r0:r1, c0:c1] = True


def _box_free(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + height + pad)
    c1 = min(occupied.shape[1], left + width + pad)
    return not occupied[r0:r1, c0:c1].any()


def _source_candidates(
    h: int,
    w: int,
    quadrant: str,
    patch_shape: tuple[int, int],
    occupied: np.ndarray,
) -> list[tuple[int, int]]:
    q_r0, q_r1, q_c0, q_c1 = _quadrant_bounds(quadrant, h, w)
    ph, pw = patch_shape
    row_lo = q_r0 + 1
    row_hi = q_r1 - ph
    col_lo = q_c0 + 1
    col_hi = q_c1 - pw
    if row_lo > row_hi or col_lo > col_hi:
        return []
    candidates = []
    for top in range(row_lo, row_hi + 1):
        for left in range(col_lo, col_hi + 1):
            if _box_free(occupied, top, left, ph, pw, pad=1):
                candidates.append((top, left))
    return candidates


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(GRID_SIZES)
    w = random.choice(GRID_SIZES)
    n_colors = random.randint(2, 4)
    colors = random.sample(COLOR_POOL, n_colors)
    anchor_quadrants = random.sample(QUADRANTS, n_colors)

    specs = []
    for color, anchor_quadrant in zip(colors, anchor_quadrants):
        stamp = _generate_shape()
        rotated = np.rot90(stamp, ROTATION_BY_QUADRANT[anchor_quadrant])
        anchor_options = _anchor_candidates(h, w, anchor_quadrant, rotated.shape)
        if not anchor_options:
            return None
        anchor = random.choice(anchor_options)
        specs.append(
            {
                "color": color,
                "stamp": stamp,
                "rotated": rotated,
                "anchor_quadrant": anchor_quadrant,
                "anchor": anchor,
            }
        )

    occupied = np.zeros((h, w), dtype=bool)
    for spec in specs:
        ar, ac = spec["anchor"]
        _reserve(occupied, ar, ac, 1, 1, pad=1)

    for spec in sorted(specs, key=lambda item: int(item["stamp"].sum()), reverse=True):
        source_quadrants = [q for q in QUADRANTS if q != spec["anchor_quadrant"]]
        random.shuffle(source_quadrants)
        source_top_left = None
        source_quadrant = None
        for quadrant in source_quadrants:
            options = _source_candidates(h, w, quadrant, spec["stamp"].shape, occupied)
            if options:
                source_top_left = random.choice(options)
                source_quadrant = quadrant
                break
        if source_top_left is None:
            return None
        spec["source_quadrant"] = source_quadrant
        spec["source_top_left"] = source_top_left
        _reserve(
            occupied,
            source_top_left[0],
            source_top_left[1],
            spec["stamp"].shape[0],
            spec["stamp"].shape[1],
            pad=1,
        )

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        sr, sc = spec["source_top_left"]
        ar, ac = spec["anchor"]
        _paint_patch(inp, spec["stamp"], sr, sc, color, BG)
        assert inp[ar, ac] == BG
        inp[ar, ac] = color
        _paint_centered(out, spec["rotated"], ar, ac, color, BG)

    try:
        solved = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(solved, out) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
