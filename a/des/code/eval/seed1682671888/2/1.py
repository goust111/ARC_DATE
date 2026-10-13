import random
from typing import Iterable

import numpy as np


GRAY = 5
MAROON = 9
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(pts, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_separators(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    row_candidates = [
        r
        for r in range(grid.shape[0])
        if np.all(grid[r, :] == grid[r, 0]) and int(grid[r, 0]) != bg
    ]
    col_candidates = [
        c
        for c in range(grid.shape[1])
        if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != bg
    ]
    for r in row_candidates:
        for c in col_candidates:
            color = int(grid[r, 0])
            if int(grid[0, c]) == color and int(grid[r, c]) == color:
                return r, c, color
    raise AssertionError("failed to find matching separator row/column")


def _find_template_mask(grid: np.ndarray) -> np.ndarray:
    maroon = np.argwhere(grid == MAROON)
    assert len(maroon) > 0, "template must contain maroon pixels"
    r0, c0, r1, c1 = _bbox(maroon)
    assert r1 - r0 == c1 - c0, "template must be square"
    mask = grid[r0 : r1 + 1, c0 : c1 + 1] == MAROON
    assert 3 <= mask.shape[0] <= 5, "template side must be 3..5"
    assert np.any(mask) and np.any(~mask), "template must have keep cells and holes"
    return mask


def _quadrant_bounds(grid: np.ndarray, split_r: int, split_c: int) -> dict[str, tuple[int, int, int, int]]:
    h, w = grid.shape
    return {
        "TL": (0, split_r, 0, split_c),
        "TR": (0, split_r, split_c + 1, w),
        "BL": (split_r + 1, h, 0, split_c),
        "BR": (split_r + 1, h, split_c + 1, w),
    }


def _border_cells(name: str, r0: int, r1: int, c0: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if name[0] == "T":
        cells.extend((r0, c) for c in range(c0, c1))
    else:
        cells.extend((r1 - 1, c) for c in range(c0, c1))
    if name[1] == "L":
        cells.extend((r, c0) for r in range(r0, r1))
    else:
        cells.extend((r, c1 - 1) for r in range(r0, r1))
    out: list[tuple[int, int]] = []
    seen = set()
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            out.append(cell)
    return out


def _find_anchor_color(
    grid: np.ndarray,
    name: str,
    bounds: tuple[int, int, int, int],
    bg: int,
    sep_color: int,
) -> int:
    r0, r1, c0, c1 = bounds
    colors = {
        int(grid[r, c])
        for r, c in _border_cells(name, r0, r1, c0, c1)
        if int(grid[r, c]) not in {bg, sep_color, MAROON}
    }
    assert len(colors) == 1, f"{name}: anchor color must be unique"
    return next(iter(colors))


def _component_mask(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[coords[:, 0], coords[:, 1]] = True
    return out


def _extract_quadrant_objects(
    subgrid: np.ndarray,
    color: int,
    side: int,
) -> tuple[np.ndarray, np.ndarray]:
    parts = _components(subgrid == color)
    anchor_idx = None
    block_idx = None
    sprite_idx = None
    for i, coords in enumerate(parts):
        r0, c0, r1, c1 = _bbox(coords)
        area = len(coords)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if area == 1:
            rr, cc = coords[0]
            if rr in {0, subgrid.shape[0] - 1} or cc in {0, subgrid.shape[1] - 1}:
                anchor_idx = i
        elif hh == side and ww == side and area == side * side:
            block_idx = i
        else:
            sprite_idx = i

    assert anchor_idx is not None, "missing anchor"
    assert block_idx is not None, "missing solid block"
    assert sprite_idx is not None, "missing sprite"

    sprite_coords = parts[sprite_idx]
    sr0, sc0, sr1, sc1 = _bbox(sprite_coords)
    sprite = np.zeros((side, side), dtype=bool)
    local = sprite_coords - np.array([sr0, sc0])
    sh, sw = sr1 - sr0 + 1, sc1 - sc0 + 1
    assert sh <= side and sw <= side, "sprite bbox must fit block side"
    sprite_view = sprite[:sh, :sw]
    sprite_view[local[:, 0], local[:, 1]] = True
    return _component_mask(parts[block_idx], subgrid.shape), sprite


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each corner sprite to the quadrant with the same anchor color, then
    apply the central maroon template as a hole mask over the quadrant's solid
    block. The matching sprite fills only the template's holes.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    split_r, split_c, sep_color = _find_separators(g, bg)
    quad_bounds = _quadrant_bounds(g, split_r, split_c)
    template = _find_template_mask(g)
    side = int(template.shape[0])

    out = np.full((side * 2 + 1, side * 2 + 1), bg, dtype=int)
    out[side, :] = sep_color
    out[:, side] = sep_color

    placements = {
        "TL": (0, 0),
        "TR": (0, side + 1),
        "BL": (side + 1, 0),
        "BR": (side + 1, side + 1),
    }

    for name, bounds in quad_bounds.items():
        color = _find_anchor_color(g, name, bounds, bg, sep_color)
        r0, r1, c0, c1 = bounds
        subgrid = g[r0:r1, c0:c1]
        _block_mask, sprite = _extract_quadrant_objects(subgrid, color, side)
        filled = template | sprite
        orow, ocol = placements[name]
        patch = np.full((side, side), bg, dtype=int)
        patch[filled] = color
        out[orow : orow + side, ocol : ocol + side] = patch

    return out


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    return len(_components(mask)) == 1


def _random_connected_mask(rng: random.Random, side: int, cells: int | None = None) -> np.ndarray:
    total = side * side
    if cells is None:
        cells = rng.randint(max(2, total // 4), max(2, total - 2))
    cells = max(1, min(total, cells))
    start = (rng.randrange(side), rng.randrange(side))
    mask = np.zeros((side, side), dtype=bool)
    mask[start] = True
    frontier = {start}
    while int(mask.sum()) < cells and frontier:
        cr, cc = rng.choice(tuple(frontier))
        nbrs = []
        for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
            if 0 <= nr < side and 0 <= nc < side and not mask[nr, nc]:
                nbrs.append((nr, nc))
        if not nbrs:
            frontier.remove((cr, cc))
            continue
        nr, nc = rng.choice(nbrs)
        mask[nr, nc] = True
        frontier.add((nr, nc))
    return mask


def _has_full_bbox(mask: np.ndarray) -> bool:
    return (
        np.any(mask[0, :])
        and np.any(mask[-1, :])
        and np.any(mask[:, 0])
        and np.any(mask[:, -1])
    )


def _random_template_mask(rng: random.Random, side: int) -> np.ndarray:
    for _ in range(200):
        cells = rng.randint(max(2, side), side * side - 1)
        mask = _random_connected_mask(rng, side, cells)
        if not _has_full_bbox(mask):
            continue
        holes = ~mask
        if holes.sum() < max(2, side - 1):
            continue
        if holes.sum() > side * side - 2:
            continue
        return mask
    raise RuntimeError("failed to build template mask")


def _random_sprite_masks(rng: random.Random, side: int, template: np.ndarray) -> list[np.ndarray]:
    holes = ~template
    solids = template
    masks: list[np.ndarray] = []
    keys = set()
    for _ in range(1000):
        if len(masks) == 4:
            return masks
        cells = rng.randint(max(2, side - 1), side * side - 2)
        mask = _random_connected_mask(rng, side, cells)
        if not _has_full_bbox(mask):
            continue
        if not np.any(mask & holes):
            continue
        if not np.any(mask & solids):
            continue
        if not np.any(holes & ~mask):
            continue
        key = mask.tobytes()
        if key in keys:
            continue
        keys.add(key)
        masks.append(mask)
    raise RuntimeError("failed to build sprite masks")


def _can_place_bbox(existing: Iterable[tuple[int, int, int, int]], box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    for er0, ec0, er1, ec1 in existing:
        if not (r1 < er0 - 1 or er1 < r0 - 1 or c1 < ec0 - 1 or ec1 < c0 - 1):
            return False
    return True


def _quadrant_box_candidates(
    name: str,
    bounds: tuple[int, int, int, int],
    side: int,
    near_center: bool,
) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = bounds
    pts: list[tuple[int, int]] = []
    if near_center:
        row_range = range(max(r0 + 1, r1 - side - 2), r1 - side + 1)
        col_range = range(max(c0 + 1, c1 - side - 2), c1 - side + 1)
        if name == "TR":
            col_range = range(c0, min(c1 - side + 1, c0 + 3))
            row_range = range(max(r0 + 1, r1 - side - 2), r1 - side + 1)
        elif name == "BL":
            row_range = range(r0, min(r1 - side + 1, r0 + 3))
            col_range = range(max(c0 + 1, c1 - side - 2), c1 - side + 1)
        elif name == "BR":
            row_range = range(r0, min(r1 - side + 1, r0 + 3))
            col_range = range(c0, min(c1 - side + 1, c0 + 3))
    else:
        if name == "TL":
            row_range = range(r0 + 1, min(r1 - side + 1, r0 + 3))
            col_range = range(c0 + 1, min(c1 - side + 1, c0 + 3))
        elif name == "TR":
            row_range = range(r0 + 1, min(r1 - side + 1, r0 + 3))
            col_range = range(max(c0, c1 - side - 2), c1 - side + 1)
        elif name == "BL":
            row_range = range(max(r0, r1 - side - 2), r1 - side + 1)
            col_range = range(c0 + 1, min(c1 - side + 1, c0 + 3))
        else:
            row_range = range(max(r0, r1 - side - 2), r1 - side + 1)
            col_range = range(max(c0, c1 - side - 2), c1 - side + 1)
    for r in row_range:
        for c in col_range:
            pts.append((r, c))
    return pts


def _random_free_box(
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    side: int,
    blocked: list[tuple[int, int, int, int]],
    preferred: list[tuple[int, int]] | None = None,
) -> tuple[int, int] | None:
    r0, r1, c0, c1 = bounds
    candidates = []
    if preferred:
        candidates.extend(preferred)
    for r in range(r0 + 1, r1 - side):
        for c in range(c0 + 1, c1 - side):
            candidates.append((r, c))
    rng.shuffle(candidates)
    seen = set()
    for r, c in candidates:
        if (r, c) in seen:
            continue
        seen.add((r, c))
        box = (r, c, r + side - 1, c + side - 1)
        if r < r0 or c < c0 or r + side > r1 or c + side > c1:
            continue
        if _can_place_bbox(blocked, box):
            return r, c
    return None


def _place_mask(grid: np.ndarray, top: int, left: int, color: int, mask: np.ndarray) -> None:
    patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    patch[mask] = color


def _anchor_options(name: str, bounds: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = bounds
    opts: list[tuple[int, int]] = []
    if name[0] == "T":
        opts.extend((r0, c) for c in range(c0, c1))
    else:
        opts.extend((r1 - 1, c) for c in range(c0, c1))
    if name[1] == "L":
        opts.extend((r, c0) for r in range(r0, r1))
    else:
        opts.extend((r, c1 - 1) for r in range(r0, r1))
    return list(dict.fromkeys(opts))


def _expected_output(
    side: int,
    bg: int,
    sep_color: int,
    template: np.ndarray,
    sprite_masks: dict[str, np.ndarray],
    colors: dict[str, int],
) -> np.ndarray:
    out = np.full((side * 2 + 1, side * 2 + 1), bg, dtype=int)
    out[side, :] = sep_color
    out[:, side] = sep_color
    placements = {
        "TL": (0, 0),
        "TR": (0, side + 1),
        "BL": (side + 1, 0),
        "BR": (side + 1, side + 1),
    }
    for name, (r, c) in placements.items():
        patch = np.full((side, side), bg, dtype=int)
        patch[template | sprite_masks[name]] = colors[name]
        out[r : r + side, c : c + side] = patch
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.choice([3, 4, 5])
    pool = [0, 1, 2, 3, 4, 6, 7, 8]
    bg = rng.choice(pool)
    color_pool = [c for c in pool if c != bg]
    quad_colors = rng.sample(color_pool, 4)
    colors = dict(zip(("TL", "TR", "BL", "BR"), quad_colors))

    top_h = rng.randint(2 * side + 4, 3 * side + 7)
    bottom_h = rng.randint(2 * side + 4, 3 * side + 7)
    left_w = rng.randint(2 * side + 4, 3 * side + 7)
    right_w = rng.randint(2 * side + 4, 3 * side + 7)
    h = top_h + 1 + bottom_h
    w = left_w + 1 + right_w
    split_r = top_h
    split_c = left_w

    grid = np.full((h, w), bg, dtype=int)
    grid[split_r, :] = GRAY
    grid[:, split_c] = GRAY

    bounds = _quadrant_bounds(grid, split_r, split_c)
    template = _random_template_mask(rng, side)
    sprite_masks = dict(zip(("TL", "TR", "BL", "BR"), _random_sprite_masks(rng, side, template)))
    template_quad = rng.choice(["TL", "TR", "BL", "BR"])

    boxes: dict[str, list[tuple[int, int, int, int]]] = {name: [] for name in bounds}
    sprite_pos: dict[str, tuple[int, int]] = {}
    block_pos: dict[str, tuple[int, int]] = {}

    for name, quad in bounds.items():
        candidates = _quadrant_box_candidates(name, quad, side, near_center=False)
        rng.shuffle(candidates)
        placed = None
        for r, c in candidates:
            box = (r, c, r + side - 1, c + side - 1)
            if _can_place_bbox(boxes[name], box):
                placed = (r, c)
                boxes[name].append(box)
                break
        if placed is None:
            return None
        sprite_pos[name] = placed

    near_candidates = _quadrant_box_candidates(template_quad, bounds[template_quad], side, near_center=True)
    rng.shuffle(near_candidates)
    template_pos = None
    for r, c in near_candidates:
        box = (r, c, r + side - 1, c + side - 1)
        if _can_place_bbox(boxes[template_quad], box):
            template_pos = (r, c)
            boxes[template_quad].append(box)
            break
    if template_pos is None:
        return None

    for name, quad in bounds.items():
        preferred = []
        r0, r1, c0, c1 = quad
        for _ in range(12):
            preferred.append(
                (
                    rng.randint(r0 + 1, r1 - side - 1),
                    rng.randint(c0 + 1, c1 - side - 1),
                )
            )
        pos = _random_free_box(rng, quad, side, boxes[name], preferred)
        if pos is None:
            return None
        block_pos[name] = pos
        boxes[name].append((pos[0], pos[1], pos[0] + side - 1, pos[1] + side - 1))

    for name in ("TL", "TR", "BL", "BR"):
        _place_mask(grid, sprite_pos[name][0], sprite_pos[name][1], colors[name], sprite_masks[name])
    _place_mask(grid, template_pos[0], template_pos[1], MAROON, template)

    for name in ("TL", "TR", "BL", "BR"):
        r, c = block_pos[name]
        grid[r : r + side, c : c + side] = colors[name]

    for name, quad in bounds.items():
        opts = _anchor_options(name, quad)
        rng.shuffle(opts)
        chosen = None
        for r, c in opts:
            if grid[r, c] == bg:
                chosen = (r, c)
                break
        if chosen is None:
            return None
        grid[chosen] = colors[name]

    expected = _expected_output(side, bg, GRAY, template, sprite_masks, colors)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
