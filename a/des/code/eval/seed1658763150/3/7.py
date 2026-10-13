import random
from typing import Dict, List, Optional, Tuple

import numpy as np


CROSS_COLOR = 5
MARKER_COLOR = 9
GRID_SIDES = (15, 17, 19)
MAX_GENERATION_ATTEMPTS = 400
ARMS = ("left", "right", "top", "bottom")
QUADRANT_TO_ARMS = {
    "tl": ("left", "top"),
    "tr": ("right", "top"),
    "bl": ("left", "bottom"),
    "br": ("right", "bottom"),
}


def _dominant_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[np.argmax(counts)])


def _ordered_arm(grid: np.ndarray, side: str, mid: int) -> np.ndarray:
    if side == "left":
        return grid[mid, :mid]
    if side == "right":
        return grid[mid, mid + 1 :][::-1]
    if side == "top":
        return grid[:mid, mid]
    if side == "bottom":
        return grid[mid + 1 :, mid][::-1]
    raise ValueError(f"unknown side: {side}")


def _extract_arm_pairs(grid: np.ndarray, side: str, mid: int, cross_color: int) -> List[Tuple[int, int]]:
    arm = _ordered_arm(grid, side, mid)
    pairs: List[Tuple[int, int]] = []
    i = 0
    while i < len(arm):
        if int(arm[i]) == cross_color:
            i += 1
            continue
        j = i
        while j < len(arm) and int(arm[j]) != cross_color:
            j += 1
        run = arm[i:j]
        assert len(run) > 0 and len(run) % 2 == 0
        for k in range(0, len(run), 2):
            src = int(run[k])
            dst = int(run[k + 1])
            assert src != dst
            pairs.append((src, dst))
        i = j
    return pairs


def _quadrant_slices(n: int) -> Dict[str, Tuple[slice, slice]]:
    mid = n // 2
    return {
        "tl": (slice(0, mid), slice(0, mid)),
        "tr": (slice(0, mid), slice(mid + 1, n)),
        "bl": (slice(mid + 1, n), slice(0, mid)),
        "br": (slice(mid + 1, n), slice(mid + 1, n)),
    }


def _find_marker_block(quad: np.ndarray, bg: int, cross_color: int) -> Optional[Tuple[int, int, int]]:
    q = quad.shape[0]
    assert quad.shape == (q, q) and q >= 2
    hits: List[Tuple[int, int, int]] = []
    for r0, c0 in ((0, 0), (0, q - 2), (q - 2, 0), (q - 2, q - 2)):
        block = quad[r0 : r0 + 2, c0 : c0 + 2]
        color = int(block[0, 0])
        if np.all(block == color) and color not in {bg, cross_color}:
            hits.append((r0, c0, color))
    if not hits:
        return None
    assert len(hits) == 1
    return hits[0]


def _local_to_global(name: str, q: int, r: int, c: int) -> Tuple[int, int]:
    if name == "tl":
        return r, c
    if name == "tr":
        return r, q - 1 - c
    if name == "bl":
        return q - 1 - r, c
    if name == "br":
        return q - 1 - r, q - 1 - c
    raise ValueError(f"unknown quadrant: {name}")


def _neighbors(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _grow_connected_mask(h: int, w: int, target_cells: int) -> Optional[np.ndarray]:
    for _ in range(120):
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = set(_neighbors(start[0], start[1], h, w))
        while int(mask.sum()) < target_cells and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if mask[cell]:
                continue
            mask[cell] = True
            for nbr in _neighbors(cell[0], cell[1], h, w):
                if not mask[nbr]:
                    frontier.add(nbr)
        if int(mask.sum()) != target_cells:
            continue
        if not np.any(mask[0]) or not np.any(mask[-1]) or not np.any(mask[:, 0]) or not np.any(mask[:, -1]):
            continue
        if np.array_equal(mask, np.flipud(mask)) and np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    return None


def _paint_arm_pair(grid: np.ndarray, side: str, start: int, src: int, dst: int) -> None:
    n = grid.shape[0]
    mid = n // 2
    if side == "left":
        grid[mid, start] = src
        grid[mid, start + 1] = dst
        return
    if side == "right":
        grid[mid, n - 1 - start] = src
        grid[mid, n - 2 - start] = dst
        return
    if side == "top":
        grid[start, mid] = src
        grid[start + 1, mid] = dst
        return
    if side == "bottom":
        grid[n - 1 - start, mid] = src
        grid[n - 2 - start, mid] = dst
        return
    raise ValueError(f"unknown side: {side}")


def _choose_marker_corner(q: int) -> Tuple[int, int]:
    return random.choice(((0, 0), (0, q - 2), (q - 2, 0), (q - 2, q - 2)))


def _assign_colors(
    coords: List[Tuple[int, int]],
    palette: List[int],
    required: List[Tuple[int, int]],
) -> List[int]:
    assert coords
    assert sum(count for _, count in required) <= len(coords)
    idx = list(range(len(coords)))
    random.shuffle(idx)
    colors: List[Optional[int]] = [None] * len(coords)
    used = 0
    for color, count in required:
        for _ in range(count):
            colors[idx[used]] = color
            used += 1
    for i in range(len(coords)):
        if colors[i] is None:
            colors[i] = random.choice(palette)
    return [int(c) for c in colors]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the marked quadrants of a mirror-board layout using directional pairs on the center cross.

    Marked quadrants are identified by a 2x2 corner marker. For each marked quadrant, the
    source-to-target mappings come from the two cross arms adjacent to that quadrant. Along
    an arm, each colored pair is read from the outer cell toward the center cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    n, m = g.shape
    assert n == m and n % 2 == 1
    mid = n // 2

    cross_color = int(g[mid, mid])
    off_cross = np.ones_like(g, dtype=bool)
    off_cross[mid, :] = False
    off_cross[:, mid] = False
    bg = _dominant_color(g[off_cross])

    arm_pairs = {side: _extract_arm_pairs(g, side, mid, cross_color) for side in ARMS}
    assert any(arm_pairs.values())

    quad_slices = _quadrant_slices(n)
    markers: Dict[str, Tuple[int, int]] = {}
    marker_colors = set()
    for name, (rows, cols) in quad_slices.items():
        hit = _find_marker_block(g[rows, cols], bg, cross_color)
        if hit is None:
            continue
        lr, lc, color = hit
        markers[name] = (rows.start + lr, cols.start + lc)
        marker_colors.add(color)
    if marker_colors:
        assert len(marker_colors) == 1

    quadrant_maps: Dict[str, Dict[int, int]] = {}
    for name, sides in QUADRANT_TO_ARMS.items():
        mapping: Dict[int, int] = {}
        for side in sides:
            for src, dst in arm_pairs[side]:
                prev = mapping.get(src)
                assert prev is None or prev == dst
                mapping[src] = dst
        quadrant_maps[name] = mapping

    out = g.copy()
    for r0, c0 in markers.values():
        out[r0 : r0 + 2, c0 : c0 + 2] = bg

    for name in markers:
        rows, cols = quad_slices[name]
        src_quad = g[rows, cols]
        dst_quad = out[rows, cols].copy()
        for src, dst in quadrant_maps[name].items():
            dst_quad[src_quad == src] = dst
        out[rows, cols] = dst_quad

    return out


def _gen_attempt() -> Optional[np.ndarray]:
    side = random.choice(GRID_SIDES)
    q = side // 2

    bg = random.choice([c for c in range(10) if c not in {CROSS_COLOR, MARKER_COLOR}])
    safe_colors = [c for c in range(10) if c not in {bg, CROSS_COLOR, MARKER_COLOR}]
    src_colors = random.sample(safe_colors, 4)
    target_candidates = [c for c in safe_colors if c not in src_colors]
    target_pool = random.sample(target_candidates, random.randint(2, len(target_candidates)))
    src_by_arm = {arm: color for arm, color in zip(ARMS, src_colors, strict=False)}
    dst_by_arm = {arm: random.choice(target_pool) for arm in ARMS}

    grid = np.full((side, side), bg, dtype=int)
    mid = q
    grid[mid, :] = CROSS_COLOR
    grid[:, mid] = CROSS_COLOR

    for arm in ARMS:
        start = random.randint(0, q - 2)
        _paint_arm_pair(grid, arm, start, src_by_arm[arm], dst_by_arm[arm])

    box_h = random.randint(3, q - 4)
    box_w = random.randint(3, q - 4)
    box_r0 = random.randint(2, q - 2 - box_h)
    box_c0 = random.randint(2, q - 2 - box_w)
    area = box_h * box_w
    min_cells = max(6, (area + 1) // 2)
    max_cells = area - 1
    if max_cells < min_cells:
        return None
    mask = _grow_connected_mask(box_h, box_w, random.randint(min_cells, max_cells))
    if mask is None:
        return None

    base_coords = [(box_r0 + r, box_c0 + c) for r, c in np.argwhere(mask)]
    marked_count = random.randint(1, 4)
    marked_quadrants = set(random.sample(tuple(QUADRANT_TO_ARMS), marked_count))

    quad_slices = _quadrant_slices(side)
    marker_starts: Dict[str, Tuple[int, int]] = {}
    for name in QUADRANT_TO_ARMS:
        if name not in marked_quadrants:
            continue
        marker_starts[name] = _choose_marker_corner(q)

    for name in QUADRANT_TO_ARMS:
        coords = [_local_to_global(name, q, r, c) for r, c in base_coords]
        if name in marked_quadrants:
            needed_sources = [src_by_arm[side_name] for side_name in QUADRANT_TO_ARMS[name]]
            forbidden = {dst_by_arm[side_name] for side_name in QUADRANT_TO_ARMS[name]}
            fillers = [c for c in safe_colors if c not in set(needed_sources) | forbidden]
            local_palette = needed_sources + random.sample(fillers, random.randint(1, min(2, len(fillers))))
            required = [(needed_sources[0], 2), (needed_sources[1], 2)]
            colors = _assign_colors(coords, local_palette, required)
        else:
            hinted = [src_by_arm[side_name] for side_name in QUADRANT_TO_ARMS[name] if random.random() < 0.5]
            base_palette = list(dict.fromkeys(hinted))
            filler_candidates = [c for c in safe_colors if c not in base_palette]
            palette_size = random.randint(max(2, len(base_palette)), min(4, len(safe_colors)))
            while len(base_palette) < palette_size:
                choice = random.choice(filler_candidates)
                base_palette.append(choice)
                filler_candidates.remove(choice)
            required = [(color, 1) for color in base_palette[: min(2, len(base_palette))]]
            colors = _assign_colors(coords, base_palette, required)

        row_slice, col_slice = quad_slices[name]
        for (lr, lc), color in zip(coords, colors, strict=False):
            grid[row_slice.start + lr, col_slice.start + lc] = color

    for name, (mr, mc) in marker_starts.items():
        rows, cols = quad_slices[name]
        grid[rows.start + mr : rows.start + mr + 2, cols.start + mc : cols.start + mc + 2] = MARKER_COLOR

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None

    for name in marked_quadrants:
        rows, cols = quad_slices[name]
        changed = int(np.count_nonzero(grid[rows, cols] != out[rows, cols]))
        if changed < 8:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
