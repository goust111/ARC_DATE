from __future__ import annotations

from collections import Counter
import random

import numpy as np


RED = 2
PALETTE = list(range(10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- solver ---------------


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _find_upper_objects(grid: np.ndarray, bg: int) -> list[dict[str, np.ndarray | int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, np.ndarray | int]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            objects.append({"color": color, "coords": np.asarray(coords, dtype=int), "area": len(coords)})

    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each upper object by its own area, then keep only the cells shared with the lower mask.

    Input structure:
    - A single solid horizontal divider row splits the grid into two equal-height halves.
    - The upper half contains several separated 4-connected monochromatic objects on one background color.
    - The lower half contains one single-color foreground mask on the same background.

    Rule:
    - For every upper object, compute its area in pixels and shift the whole object straight downward by that many rows inside the upper half.
    - Clip any shifted pixels that move below the bottom of the upper half, then ignore the divider row and align the fallen upper half with the lower half.
    - Output one half-sized grid where cells occupied by both the fallen upper objects and the lower mask are colored red, and all other cells stay background.

    Key insight:
    - The visible result is the intersection with the lower mask after the upper objects have fallen by an object-specific distance equal to their own area.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    assert height % 2 == 1 and height >= 5

    divider_r = height // 2
    divider_color = int(g[divider_r, 0])
    assert np.all(g[divider_r] == divider_color)

    upper = g[:divider_r]
    lower = g[divider_r + 1 :]
    assert upper.shape == lower.shape

    bg = _most_common_color(np.concatenate([upper.ravel(), lower.ravel()]))
    assert bg != RED

    lower_colors = set(map(int, np.unique(lower))) - {bg}
    assert len(lower_colors) == 1
    lower_mask = lower != bg

    fallen = np.zeros_like(lower_mask, dtype=bool)
    objects = _find_upper_objects(upper, bg)
    assert objects

    for obj in objects:
        coords = np.asarray(obj["coords"], dtype=int)
        area = int(obj["area"])
        shifted = coords.copy()
        shifted[:, 0] += area
        visible = shifted[shifted[:, 0] < upper.shape[0]]
        if visible.size == 0:
            continue
        fallen[visible[:, 0], visible[:, 1]] = True

    out = np.full(upper.shape, bg, dtype=int)
    out[fallen & lower_mask] = RED
    return out


# --------------- generator ---------------

HALF_H_RANGE = (7, 12)
WIDTH_RANGE = (9, 16)
OBJECT_COUNT_RANGE = (2, 5)
AREA_RANGE = (1, 5)
MASK_RECT_SIDE_RANGE = (2, 6)
MAX_GEN_ATTEMPTS = 500


def _normalize_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(sorted(coords), dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return arr


def _random_polyomino(area: int) -> np.ndarray:
    cells = {(0, 0)}
    frontier = {(0, 1), (1, 0), (0, -1), (-1, 0)}
    while len(cells) < area:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        r, c = cell
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt not in cells:
                frontier.add(nxt)

    arr = np.asarray(sorted(cells), dtype=int)
    for _ in range(random.randrange(4)):
        arr = np.column_stack((arr[:, 1], -arr[:, 0]))
    if random.random() < 0.5:
        arr[:, 1] *= -1
    return _normalize_coords(set(map(tuple, arr)))


def _reserve_bbox(mask: np.ndarray, coords: np.ndarray, pad: int) -> bool:
    r0 = max(0, int(coords[:, 0].min()) - pad)
    c0 = max(0, int(coords[:, 1].min()) - pad)
    r1 = min(mask.shape[0] - 1, int(coords[:, 0].max()) + pad)
    c1 = min(mask.shape[1] - 1, int(coords[:, 1].max()) + pad)
    if mask[r0 : r1 + 1, c0 : c1 + 1].any():
        return False
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return True


def _candidate_positions(
    shape: np.ndarray,
    area: int,
    half_h: int,
    width: int,
    blocked_input: np.ndarray,
    blocked_fall: np.ndarray,
    want_clip: bool | None,
) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    shape_h = int(shape[:, 0].max()) + 1
    shape_w = int(shape[:, 1].max()) + 1
    positions = [(r, c) for r in range(half_h - shape_h + 1) for c in range(width - shape_w + 1)]
    random.shuffle(positions)

    candidates: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for r, c in positions:
        placed = shape + np.array([r, c])
        shifted = placed.copy()
        shifted[:, 0] += area
        visible = shifted[shifted[:, 0] < half_h]
        if visible.size == 0:
            continue
        clipped = len(visible) < len(placed)
        if want_clip is not None and clipped != want_clip:
            continue
        if blocked_fall[visible[:, 0], visible[:, 1]].any():
            continue
        if not _reserve_bbox(blocked_input.copy(), placed, pad=1):
            continue
        candidates.append((placed, visible, clipped))
    return candidates


def _place_upper_objects(
    half_h: int,
    width: int,
    bg: int,
    areas: list[int],
    colors: list[int],
    shapes: list[np.ndarray],
    require_clip: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    upper = np.full((half_h, width), bg, dtype=int)
    blocked_input = np.zeros((half_h, width), dtype=bool)
    blocked_fall = np.zeros((half_h, width), dtype=bool)
    upper_occ = np.zeros((half_h, width), dtype=bool)
    fallen_occ = np.zeros((half_h, width), dtype=bool)

    clip_index = max(range(len(areas)), key=lambda i: (areas[i], random.random())) if require_clip else None
    order = list(range(len(areas)))
    if clip_index is not None:
        others = [i for i in order if i != clip_index]
        others.sort(key=lambda i: areas[i], reverse=True)
        order = [clip_index, *others]
    else:
        order.sort(key=lambda i: areas[i], reverse=True)

    clipped_count = 0
    for idx in order:
        want_clip = True if idx == clip_index else None
        candidates = _candidate_positions(
            shapes[idx], areas[idx], half_h, width, blocked_input, blocked_fall, want_clip
        )
        if not candidates:
            return None
        placed, visible, clipped = random.choice(candidates)
        if not _reserve_bbox(blocked_input, placed, pad=1):
            return None
        upper[placed[:, 0], placed[:, 1]] = colors[idx]
        upper_occ[placed[:, 0], placed[:, 1]] = True
        blocked_fall[visible[:, 0], visible[:, 1]] = True
        fallen_occ[visible[:, 0], visible[:, 1]] = True
        clipped_count += int(clipped)

    if require_clip and clipped_count == 0:
        return None
    return upper, upper_occ, fallen_occ


def _add_anchor_rect(mask: np.ndarray, anchor: tuple[int, int]) -> None:
    half_h, width = mask.shape
    rect_h = random.randint(MASK_RECT_SIDE_RANGE[0], min(MASK_RECT_SIDE_RANGE[1], half_h))
    rect_w = random.randint(MASK_RECT_SIDE_RANGE[0], min(MASK_RECT_SIDE_RANGE[1], width))
    ar, ac = anchor

    r0_min = max(0, ar - rect_h + 1)
    r0_max = min(ar, half_h - rect_h)
    c0_min = max(0, ac - rect_w + 1)
    c0_max = min(ac, width - rect_w)
    if r0_min > r0_max or c0_min > c0_max:
        return

    r0 = random.randint(r0_min, r0_max)
    c0 = random.randint(c0_min, c0_max)
    mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True


def _draw_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    sr, sc = start
    er, ec = end
    if random.random() < 0.5:
        mask[sr, min(sc, ec) : max(sc, ec) + 1] = True
        mask[min(sr, er) : max(sr, er) + 1, ec] = True
    else:
        mask[min(sr, er) : max(sr, er) + 1, sc] = True
        mask[er, min(sc, ec) : max(sc, ec) + 1] = True


def _make_lower_mask(fallen_occ: np.ndarray, upper_occ: np.ndarray) -> np.ndarray | None:
    moved_only = np.argwhere(fallen_occ & ~upper_occ)
    if moved_only.size == 0:
        return None

    mask = np.zeros_like(fallen_occ, dtype=bool)
    anchor = tuple(map(int, moved_only[random.randrange(len(moved_only))]))
    _add_anchor_rect(mask, anchor)

    fallen_points = [tuple(map(int, pt)) for pt in np.argwhere(fallen_occ)]
    random.shuffle(fallen_points)
    target_count = random.randint(1, min(3, len(fallen_points)))
    used_targets = {anchor}

    for target in fallen_points:
        if len(used_targets) >= target_count:
            break
        if target in used_targets:
            continue
        existing = [tuple(map(int, pt)) for pt in np.argwhere(mask)]
        if not existing:
            return None
        _draw_path(mask, random.choice(existing), target)
        _add_anchor_rect(mask, target)
        used_targets.add(target)

    for _ in range(random.randint(0, 1)):
        existing = [tuple(map(int, pt)) for pt in np.argwhere(mask)]
        if not existing:
            return None
        _add_anchor_rect(mask, random.choice(existing))

    return mask


def _gen_attempt() -> np.ndarray | None:
    half_h = random.randint(*HALF_H_RANGE)
    width = random.randint(*WIDTH_RANGE)
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    require_clip = random.random() < 0.6

    bg = random.choice([c for c in PALETTE if c != RED])
    divider_color = random.choice([c for c in PALETTE if c not in {bg, RED}])
    mask_color = random.choice([c for c in PALETTE if c not in {bg, divider_color, RED}])
    object_palette = [c for c in PALETTE if c not in {bg, divider_color, mask_color, RED}]
    if len(object_palette) < object_count:
        return None
    object_colors = random.sample(object_palette, object_count)

    areas = [random.randint(*AREA_RANGE) for _ in range(object_count)]
    if max(areas) == 1:
        areas[random.randrange(object_count)] = random.randint(2, AREA_RANGE[1])
    shapes = [_random_polyomino(area) for area in areas]

    placed = _place_upper_objects(half_h, width, bg, areas, object_colors, shapes, require_clip=require_clip)
    if placed is None:
        return None
    upper, upper_occ, fallen_occ = placed

    if fallen_occ.sum() < 3:
        return None

    lower_mask = _make_lower_mask(fallen_occ, upper_occ)
    if lower_mask is None:
        return None

    lower_area = int(lower_mask.sum())
    if not (5 <= lower_area <= (half_h * width) // 2):
        return None

    lower = np.full((half_h, width), bg, dtype=int)
    lower[lower_mask] = mask_color

    divider = np.full((1, width), divider_color, dtype=int)
    inp = np.vstack([upper, divider, lower])

    out = main(inp)
    red_mask = out == RED
    naive_mask = upper_occ & lower_mask

    red_count = int(red_mask.sum())
    if red_count < 2:
        return None
    if red_count >= int(fallen_occ.sum()):
        return None
    if red_count >= lower_area:
        return None
    if np.array_equal(red_mask, naive_mask):
        return None
    if not np.any(red_mask & ~naive_mask):
        return None
    if _most_common_color(np.concatenate([upper.ravel(), lower.ravel()])) != bg:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
