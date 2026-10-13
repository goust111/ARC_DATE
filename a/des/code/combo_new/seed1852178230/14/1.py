import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 300
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(values) -> int:
    counts = Counter(int(v) for v in values)
    best = None
    best_count = -1
    seen = set()
    for value in values:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        count = int(counts[value])
        if count > best_count:
            best = value
            best_count = count
    assert best is not None
    return int(best)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_contiguous(indices: list[int]) -> bool:
    return bool(indices) and all(b == a + 1 for a, b in zip(indices, indices[1:]))


def _find_frame(
    grid: np.ndarray, bg: int
) -> tuple[int, tuple[int, int, int, int], dict[str, tuple[int, int]]]:
    candidates = []
    for color in (int(v) for v in np.unique(grid) if int(v) != bg):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        if not np.all(
            (coords[:, 0] == r0)
            | (coords[:, 0] == r1)
            | (coords[:, 1] == c0)
            | (coords[:, 1] == c1)
        ):
            continue
        if not (
            int(grid[r0, c0]) == color
            and int(grid[r0, c1]) == color
            and int(grid[r1, c0]) == color
            and int(grid[r1, c1]) == color
        ):
            continue

        top_gap = [c for c in range(c0 + 1, c1) if int(grid[r0, c]) == bg]
        bottom_gap = [c for c in range(c0 + 1, c1) if int(grid[r1, c]) == bg]
        left_gap = [r for r in range(r0 + 1, r1) if int(grid[r, c0]) == bg]
        right_gap = [r for r in range(r0 + 1, r1) if int(grid[r, c1]) == bg]

        side_gaps = [top_gap, bottom_gap, left_gap, right_gap]
        if any(not (1 <= len(gap) <= 3 and _is_contiguous(gap)) for gap in side_gaps):
            continue

        ok = True
        for c in range(c0 + 1, c1):
            if c in top_gap:
                ok = ok and int(grid[r0, c]) == bg
            else:
                ok = ok and int(grid[r0, c]) == color
            if c in bottom_gap:
                ok = ok and int(grid[r1, c]) == bg
            else:
                ok = ok and int(grid[r1, c]) == color
        for r in range(r0 + 1, r1):
            if r in left_gap:
                ok = ok and int(grid[r, c0]) == bg
            else:
                ok = ok and int(grid[r, c0]) == color
            if r in right_gap:
                ok = ok and int(grid[r, c1]) == bg
            else:
                ok = ok and int(grid[r, c1]) == color
        if not ok:
            continue

        gaps = {
            "top": (top_gap[0], top_gap[-1]),
            "bottom": (bottom_gap[0], bottom_gap[-1]),
            "left": (left_gap[0], left_gap[-1]),
            "right": (right_gap[0], right_gap[-1]),
        }
        candidates.append((color, (r0, c0, r1, c1), gaps))

    assert len(candidates) == 1, "expected exactly one gap-bearing frame"
    return candidates[0]


def _region_markers(
    markers: list[tuple[int, int, int]],
    side: str,
    frame_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = frame_bbox
    if side == "top":
        return [m for m in markers if m[0] < r0 and c0 <= m[1] <= c1]
    if side == "bottom":
        return [m for m in markers if m[0] > r1 and c0 <= m[1] <= c1]
    if side == "left":
        return [m for m in markers if m[1] < c0 and r0 <= m[0] <= r1]
    if side == "right":
        return [m for m in markers if m[1] > c1 and r0 <= m[0] <= r1]
    raise ValueError(side)


def _nearest_marker_color(
    markers: list[tuple[int, int, int]],
    copy_pixels: list[tuple[int, int, int]],
) -> int:
    best_color = None
    best_dist = None
    for mr, mc, color in markers:
        dist = min(abs(mr - r) + abs(mc - c) for r, c, _ in copy_pixels)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_color = int(color)
    assert best_color is not None
    return int(best_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project the inner sprite outward through each frame gap and recolor each projected copy by its outer marker.

    Input structure:
    - The background is black.
    - A single grey rectangular border sits near the center, with one continuous gap on each of its four sides.
    - A small multicolor sprite lies inside the border, while isolated colored marker pixels lie outside the border in the four projection regions.

    Rule:
    - Detect the rectangular border and the sprite inside it.
    - For each side that contains a gap, translate one copy of the sprite straight outward until the copy is flush with the outer side of that border.
    - Draw each copy only onto black cells, then find the nearest marker in that same outer region and replace the copy's most frequent non-black color with the marker color.
    - Keep the original border and the original interior sprite unchanged.

    Key insight:
    - The gap only determines which outward projection to make; the external marker in that side region supplies the recoloring instruction for the projected copy's dominant color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g.ravel())
    frame_color, frame_bbox, gaps = _find_frame(g, bg)
    r0, c0, r1, c1 = frame_bbox

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    sprite_mask = (interior != bg) & (interior != frame_color)
    sprite_rel = np.argwhere(sprite_mask)
    assert sprite_rel.size > 0, "frame interior must contain a sprite"

    sprite_coords = sprite_rel + np.array([[r0 + 1, c0 + 1]])
    sprite_vals = [int(g[r, c]) for r, c in sprite_coords]
    dominant = _most_common_color(sprite_vals)
    sr0, sc0, sr1, sc1 = _bbox_from_coords(sprite_coords)

    markers = []
    for r, c in np.argwhere((g != bg) & (g != frame_color)):
        r = int(r)
        c = int(c)
        if r0 <= r <= r1 and c0 <= c <= c1:
            continue
        markers.append((r, c, int(g[r, c])))

    out = g.copy()
    translations = {}
    if "top" in gaps:
        translations["top"] = ((r0 - 1) - sr1, 0)
    if "bottom" in gaps:
        translations["bottom"] = ((r1 + 1) - sr0, 0)
    if "left" in gaps:
        translations["left"] = (0, (c0 - 1) - sc1)
    if "right" in gaps:
        translations["right"] = (0, (c1 + 1) - sc0)

    h, w = g.shape
    for side in ("top", "bottom", "left", "right"):
        if side not in translations:
            continue
        region = _region_markers(markers, side, frame_bbox)
        assert region, f"missing marker for {side} projection"

        dr, dc = translations[side]
        placed = []
        for (r, c), value in zip(sprite_coords, sprite_vals):
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) == bg:
                out[rr, cc] = int(value)
                placed.append((rr, cc, int(value)))

        if not placed:
            continue

        marker_color = _nearest_marker_color(region, placed)
        for rr, cc, value in placed:
            if value == dominant:
                out[rr, cc] = marker_color

    return out


def _partition_positive(rng: random.Random, total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    numbers = []
    prev = 0
    for cut in cuts + [total]:
        numbers.append(cut - prev)
        prev = cut
    return numbers


def _grow_connected_cells(
    rng: random.Random, height: int, width: int, area: int
) -> set[tuple[int, int]]:
    start = (rng.randrange(height), rng.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(tuple(frontier)))
    return cells


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    r1 = max(rows)
    c1 = max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _build_sprite(rng: random.Random) -> np.ndarray:
    for _ in range(100):
        seed_h = rng.randint(2, 4)
        seed_w = rng.randint(2, 4)
        area = rng.randint(max(3, (seed_h * seed_w) // 2), seed_h * seed_w)
        mask = _cells_to_mask(_grow_connected_cells(rng, seed_h, seed_w, area))
        sh, sw = mask.shape
        count = int(mask.sum())
        if sh < 2 or sw < 2 or sh > 4 or sw > 4 or count < 3:
            continue

        extra_total = rng.randint(1, count // 2)
        secondary_count = rng.randint(1, min(3, extra_total))
        colors = rng.sample(PALETTE, secondary_count + 1)
        dominant = int(colors[0])
        secondary_parts = _partition_positive(rng, extra_total, secondary_count)

        sprite = np.zeros(mask.shape, dtype=int)
        cells = [tuple(cell) for cell in np.argwhere(mask)]
        rng.shuffle(cells)
        dominant_cells = count - extra_total
        for r, c in cells[:dominant_cells]:
            sprite[r, c] = dominant

        cursor = dominant_cells
        for color, part in zip(colors[1:], secondary_parts):
            for r, c in cells[cursor : cursor + part]:
                sprite[r, c] = int(color)
            cursor += part
        return sprite
    raise RuntimeError("failed to build sprite")


def _draw_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    top_gap: tuple[int, int],
    bottom_gap: tuple[int, int],
    left_gap: tuple[int, int],
    right_gap: tuple[int, int],
) -> None:
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME

    tc, tw = top_gap
    bc, bw = bottom_gap
    lr, lh = left_gap
    rr, rh = right_gap

    grid[r0, tc : tc + tw] = BG
    grid[r1, bc : bc + bw] = BG
    grid[lr : lr + lh, c0] = BG
    grid[rr : rr + rh, c1] = BG


def _place_markers(
    grid: np.ndarray,
    rng: random.Random,
    frame_bbox: tuple[int, int, int, int],
    sprite_bbox: tuple[int, int, int, int],
    gap_meta: dict[str, tuple[int, int]],
    marker_colors: list[int],
) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = frame_bbox
    sr0, sc0, sr1, sc1 = sprite_bbox
    sh = sr1 - sr0 + 1
    sw = sc1 - sc0 + 1

    top_copy_top = r0 - sh
    bottom_copy_bottom = r1 + sh
    left_copy_left = c0 - sw
    right_copy_right = c1 + sw

    if top_copy_top - 2 < 0 or bottom_copy_bottom + 2 >= h:
        return False
    if left_copy_left - 2 < 0 or right_copy_right + 2 >= w:
        return False

    markers = [
        (
            rng.randint(0, top_copy_top - 2),
            rng.randint(gap_meta["top"][0], gap_meta["top"][1]),
            marker_colors[0],
        ),
        (
            rng.randint(bottom_copy_bottom + 2, h - 1),
            rng.randint(gap_meta["bottom"][0], gap_meta["bottom"][1]),
            marker_colors[1],
        ),
        (
            rng.randint(gap_meta["left"][0], gap_meta["left"][1]),
            rng.randint(0, left_copy_left - 2),
            marker_colors[2],
        ),
        (
            rng.randint(gap_meta["right"][0], gap_meta["right"][1]),
            rng.randint(right_copy_right + 2, w - 1),
            marker_colors[3],
        ),
    ]

    seen = set()
    for r, c, color in markers:
        if (r, c) in seen or int(grid[r, c]) != BG:
            return False
        seen.add((r, c))
        grid[r, c] = int(color)

    for r, c, _ in markers:
        neighbors = 0
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BG:
                neighbors += 1
        if neighbors != 0:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    sprite = _build_sprite(rng)
    sh, sw = sprite.shape

    interior_h = rng.randint(sh + 2, min(8, sh + 5))
    interior_w = rng.randint(sw + 2, min(8, sw + 5))
    frame_h = interior_h + 2
    frame_w = interior_w + 2

    top_pad = sh + rng.randint(2, 5)
    bottom_pad = sh + rng.randint(2, 5)
    left_pad = sw + rng.randint(2, 5)
    right_pad = sw + rng.randint(2, 5)

    height = top_pad + frame_h + bottom_pad
    width = left_pad + frame_w + right_pad
    if height > 30 or width > 30:
        return None

    r0 = top_pad
    c0 = left_pad
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1

    sprite_top = r0 + 1 + rng.randint(0, interior_h - sh)
    sprite_left = c0 + 1 + rng.randint(0, interior_w - sw)
    sprite_bottom = sprite_top + sh - 1
    sprite_right = sprite_left + sw - 1

    top_w = rng.randint(1, min(3, sw))
    bottom_w = rng.randint(1, min(3, sw))
    left_h = rng.randint(1, min(3, sh))
    right_h = rng.randint(1, min(3, sh))

    top_start = rng.randint(sprite_left, sprite_right - top_w + 1)
    bottom_start = rng.randint(sprite_left, sprite_right - bottom_w + 1)
    left_start = rng.randint(sprite_top, sprite_bottom - left_h + 1)
    right_start = rng.randint(sprite_top, sprite_bottom - right_h + 1)

    grid = np.zeros((height, width), dtype=int)
    _draw_frame(
        grid,
        r0,
        c0,
        r1,
        c1,
        (top_start, top_w),
        (bottom_start, bottom_w),
        (left_start, left_h),
        (right_start, right_h),
    )

    sprite_mask = sprite != BG
    patch = grid[sprite_top : sprite_top + sh, sprite_left : sprite_left + sw]
    patch[sprite_mask] = sprite[sprite_mask]

    sprite_colors = {int(v) for v in np.unique(sprite) if int(v) != BG}
    choices = [color for color in PALETTE if color not in sprite_colors]
    if len(choices) < 4:
        return None
    marker_colors = rng.sample(choices, 4)

    gap_meta = {
        "top": (top_start, top_start + top_w - 1),
        "bottom": (bottom_start, bottom_start + bottom_w - 1),
        "left": (left_start, left_start + left_h - 1),
        "right": (right_start, right_start + right_h - 1),
    }
    ok = _place_markers(
        grid,
        rng,
        (r0, c0, r1, c1),
        (sprite_top, sprite_left, sprite_bottom, sprite_right),
        gap_meta,
        marker_colors,
    )
    if not ok:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
