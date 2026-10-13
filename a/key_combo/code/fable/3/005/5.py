from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
SIDES = ("top", "bottom", "left", "right")
MAX_ATTEMPTS = 400


def _bg_color(g: np.ndarray) -> int:
    return int(Counter(map(int, g.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _perimeter(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    assert h >= 3 and w >= 3, "perimeter needs a true interior"
    cells: list[tuple[int, int]] = []
    cells.extend((0, c) for c in range(w))
    cells.extend((r, w - 1) for r in range(1, h))
    cells.extend((h - 1, c) for c in range(w - 2, -1, -1))
    cells.extend((r, 0) for r in range(h - 2, 0, -1))
    assert len(cells) == 2 * h + 2 * w - 4
    return cells


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[h - 1, :] = True
    mask[:, 0] = True
    mask[:, w - 1] = True
    return mask


def _projection_cells(
    shape: tuple[int, int],
    side: str,
    bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(0, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        return [(h - 1, c) for c in range(c1, c0 - 1, -1)]
    if side == "right":
        return [(r, w - 1) for r in range(r0, r1 + 1)]
    assert side == "left"
    return [(r, 0) for r in range(r1, r0 - 1, -1)]


def _analyze_grid(g: np.ndarray) -> dict:
    bg = _bg_color(g)
    h, w = g.shape
    border = _border_mask(g.shape)
    interior_coords = np.argwhere((g != bg) & ~border)
    assert interior_coords.size > 0, "expected one interior block"

    interior_colors = set(map(int, g[interior_coords[:, 0], interior_coords[:, 1]]))
    assert len(interior_colors) == 1, "interior block must be one color"
    block_color = interior_colors.pop()

    block_bbox = _bbox(interior_coords)
    r0, c0, r1, c1 = block_bbox
    block_region = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(block_region == block_color), "interior block must be a solid rectangle"
    assert int(interior_coords.shape[0]) == block_region.size, "expected no extra interior foreground"

    touches = []
    if r0 == 1:
        touches.append("top")
    if r1 == h - 2:
        touches.append("bottom")
    if c0 == 1:
        touches.append("left")
    if c1 == w - 2:
        touches.append("right")
    assert len(touches) == 1, "block must touch exactly one edge through the border layer"
    side = touches[0]

    perim = _perimeter(g.shape)
    p_index = {cell: i for i, cell in enumerate(perim)}
    shadow_cells = _projection_cells(g.shape, side, block_bbox)
    shadow_indices = sorted(p_index[cell] for cell in shadow_cells)
    assert shadow_indices == list(range(shadow_indices[0], shadow_indices[-1] + 1))
    assert all(int(g[r, c]) == bg for r, c in shadow_cells), "shadow arc must be hidden in the input"

    return {
        "bg": bg,
        "block_color": block_color,
        "block_bbox": block_bbox,
        "side": side,
        "perim": perim,
        "shadow_indices": shadow_indices,
        "wall_start": shadow_indices[0],
    }


def _corner_indices(shape: tuple[int, int]) -> set[int]:
    h, w = shape
    perim = _perimeter(shape)
    p_index = {cell: i for i, cell in enumerate(perim)}
    return {p_index[(0, 0)], p_index[(0, w - 1)], p_index[(h - 1, w - 1)], p_index[(h - 1, 0)]}


def _distance_to_wall(idx: int, wall_start: int, n: int) -> int:
    return int((wall_start - idx) % n)


def _crosses_corner(idx: int, wall_start: int, n: int, corners: set[int]) -> bool:
    dist = _distance_to_wall(idx, wall_start, n)
    return any(((idx + step) % n) in corners for step in range(1, dist + 1))


def _solve(g: np.ndarray) -> tuple[np.ndarray, dict]:
    info = _analyze_grid(g)
    bg = int(info["bg"])
    block_color = int(info["block_color"])
    perim = info["perim"]
    shadow_indices = list(info["shadow_indices"])
    shadow_set = set(shadow_indices)
    wall_start = int(info["wall_start"])
    n = len(perim)

    moving = []
    for idx, (r, c) in enumerate(perim):
        color = int(g[r, c])
        if color == bg:
            continue
        assert idx not in shadow_set, "moving pixels cannot start on the hidden shadow"
        moving.append(
            {
                "idx": idx,
                "color": color,
                "distance": _distance_to_wall(idx, wall_start, n),
            }
        )

    assert moving, "expected border scatter pixels"
    assert all(item["distance"] > 0 for item in moving)

    out = g.copy()
    for r, c in perim:
        out[r, c] = bg

    occupied = np.zeros(n, dtype=bool)
    for idx in shadow_indices:
        occupied[idx] = True
        r, c = perim[idx]
        out[r, c] = block_color

    settled = []
    for item in sorted(moving, key=lambda x: (x["distance"], x["idx"])):
        pos = int(item["idx"])
        assert not occupied[pos], "moving pixel starts in an occupied perimeter cell"
        while not occupied[(pos + 1) % n]:
            pos = (pos + 1) % n
        assert not occupied[pos], "settled pixel collision"
        occupied[pos] = True
        r, c = perim[pos]
        out[r, c] = int(item["color"])
        settled.append({**item, "final_idx": pos})

    info["moving"] = moving
    info["settled"] = settled
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the border pixels clockwise and stack them behind the hidden projection of the interior block.

    Input structure:
    - The background canvas has isolated colored single pixels only on the outer border.
    - Inside the canvas there is exactly one solid monochrome rectangle, one cell in from exactly one edge.
    - The border arc directly adjacent to that rectangle is still background, so the anchor wall is invisible.

    Rule:
    - Treat the outer border as a clockwise cyclic path around the grid.
    - Project the interior rectangle orthogonally onto the adjacent border arc and paint that shadow with the rectangle color.
    - Clear the original border scatter pixels, then settle them nearest-shadow-first: each pixel slides clockwise until the next perimeter cell is the shadow wall or an already settled pixel.
    - Preserve each moving pixel's color and cyclic order, and leave the interior rectangle unchanged.

    Key insight:
    - The anchor is not drawn until it is projected from the interior block, and the gravity direction is tangential along the one-dimensional border cycle rather than straight toward the block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out, _info = _solve(g)
    return out


def _block_bbox_for(side: str, h: int, w: int) -> tuple[int, int, int, int]:
    if side in ("top", "bottom"):
        max_span = min(6, w - 4)
        max_depth = min(5, h - 4)
        if max_span < 2 or max_depth < 2:
            raise ValueError("grid too small")
        span = random.randint(2, max_span)
        depth = random.randint(2, max_depth)
        c0 = random.randint(2, w - span - 2)
        c1 = c0 + span - 1
        if side == "top":
            return 1, c0, depth, c1
        return h - depth - 1, c0, h - 2, c1

    max_span = min(6, h - 4)
    max_depth = min(5, w - 4)
    if max_span < 2 or max_depth < 2:
        raise ValueError("grid too small")
    span = random.randint(2, max_span)
    depth = random.randint(2, max_depth)
    r0 = random.randint(2, h - span - 2)
    r1 = r0 + span - 1
    if side == "left":
        return r0, 1, r1, depth
    return r0, w - depth - 1, r1, w - 2


def _indices_are_isolated(indices: list[int], n: int) -> bool:
    used = set(indices)
    return all(((idx - 1) % n) not in used and ((idx + 1) % n) not in used for idx in used)


def _choose_scatter_indices(
    shape: tuple[int, int],
    shadow_indices: list[int],
    count: int,
) -> list[int] | None:
    n = len(_perimeter(shape))
    wall_start = shadow_indices[0]
    shadow_set = set(shadow_indices)
    corners = _corner_indices(shape)
    final_stack = {(wall_start - d) % n for d in range(1, count + 1)}
    forbidden = shadow_set | final_stack

    candidates = [
        idx
        for idx in range(n)
        if idx not in forbidden
        and _distance_to_wall(idx, wall_start, n) > count
        and (idx - 1) % n not in shadow_set
        and (idx + 1) % n not in shadow_set
    ]
    random.shuffle(candidates)

    chosen: list[int] = []
    for idx in candidates:
        if any((idx - other) % n in (1, n - 1) for other in chosen):
            continue
        chosen.append(idx)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        return None
    if not _indices_are_isolated(chosen, n):
        return None
    if not any(_crosses_corner(idx, wall_start, n, corners) for idx in chosen):
        return None
    return chosen


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.ndim != 2 or out.ndim != 2 or inp.shape != out.shape:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if np.array_equal(inp, out):
        return False

    try:
        info = _analyze_grid(inp)
    except AssertionError:
        return False

    bg = int(info["bg"])
    block_color = int(info["block_color"])
    perim = info["perim"]
    n = len(perim)
    shadow = list(info["shadow_indices"])
    wall_start = int(info["wall_start"])
    corners = _corner_indices(inp.shape)

    r0, c0, r1, c1 = info["block_bbox"]
    if not np.array_equal(inp[r0 : r1 + 1, c0 : c1 + 1], out[r0 : r1 + 1, c0 : c1 + 1]):
        return False
    if not all(int(inp[r, c]) == bg and int(out[r, c]) == block_color for r, c in (perim[i] for i in shadow)):
        return False

    moving = []
    for idx, (r, c) in enumerate(perim):
        if int(inp[r, c]) != bg:
            moving.append((idx, int(inp[r, c]), _distance_to_wall(idx, wall_start, n)))
    if len(moving) < 4:
        return False
    if not _indices_are_isolated([idx for idx, _color, _dist in moving], n):
        return False
    if not any(_crosses_corner(idx, wall_start, n, corners) for idx, _color, _dist in moving):
        return False

    for rank, (idx, color, _dist) in enumerate(sorted(moving, key=lambda x: (x[2], x[0])), start=1):
        final_idx = (wall_start - rank) % n
        if idx == final_idx:
            return False
        r, c = perim[final_idx]
        if int(out[r, c]) != color:
            return False

    expected_border_colors = sorted(color for _idx, color, _dist in moving)
    actual_border_colors = []
    shadow_set = set(shadow)
    for idx, (r, c) in enumerate(perim):
        if idx not in shadow_set and int(out[r, c]) != bg:
            actual_border_colors.append(int(out[r, c]))
    return sorted(actual_border_colors) == expected_border_colors


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(9, 18)
    w = random.randint(9, 18)
    side = random.choice(SIDES)
    block_color = random.choice(COLORS)
    block_bbox = _block_bbox_for(side, h, w)

    grid = np.full((h, w), BG, dtype=int)
    r0, c0, r1, c1 = block_bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = block_color

    perim = _perimeter(grid.shape)
    p_index = {cell: i for i, cell in enumerate(perim)}
    shadow_indices = sorted(p_index[cell] for cell in _projection_cells(grid.shape, side, block_bbox))
    max_count = min(7, len(COLORS) - 1, (len(perim) - len(shadow_indices)) // 3)
    if max_count < 4:
        return None
    count = random.randint(4, max_count)
    indices = _choose_scatter_indices(grid.shape, shadow_indices, count)
    if indices is None:
        return None

    colors = random.sample([c for c in COLORS if c != block_color], count)
    for idx, color in zip(indices, colors):
        r, c = perim[idx]
        grid[r, c] = color

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not _semantic_checks(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid perimeter-shadow input after {MAX_ATTEMPTS} attempts")
