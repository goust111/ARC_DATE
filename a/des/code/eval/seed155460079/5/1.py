import random
from typing import Iterable

import numpy as np


SIDES = ("top", "bottom", "left", "right")
DIR_FROM_SIDE = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

BACKGROUND = 0
GRID_MIN = 22
GRID_MAX = 30
OBJECT_COUNT_CHOICES = (2, 3, 4)
OUTER_SIZE_CHOICES = (5, 7, 9)
PLACEMENT_RETRIES = 200
GENERATION_RETRIES = 300
COLORS = tuple(range(1, 10))


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    starts = np.argwhere(mask)
    for r, c in starts:
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        pts: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            pts.append((cr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.asarray(pts, dtype=int))
    return comps


def _edge_marker_map(grid: np.ndarray, bg: int) -> dict[int, str]:
    h, w = grid.shape
    markers: dict[int, str] = {}
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            color = int(grid[r, c])
            if color == bg:
                continue
            if r == 0 and c not in (0, w - 1):
                side = "top"
            elif r == h - 1 and c not in (0, w - 1):
                side = "bottom"
            elif c == 0 and r not in (0, h - 1):
                side = "left"
            elif c == w - 1 and r not in (0, h - 1):
                side = "right"
            else:
                continue
            markers[color] = side
    return markers


def _frame_info(
    grid: np.ndarray, coords: np.ndarray, color: int, bg: int
) -> dict[str, object] | None:
    h, w = grid.shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    if r0 <= 0 or c0 <= 0 or r1 >= h - 1 or c1 >= w - 1:
        return None

    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3 or height % 2 == 0 or width % 2 == 0:
        return None

    cells = {tuple(map(int, pt)) for pt in coords}
    perimeter: list[tuple[int, int]] = []
    for rr in range(r0, r1 + 1):
        perimeter.append((rr, c0))
        perimeter.append((rr, c1))
    for cc in range(c0 + 1, c1):
        perimeter.append((r0, cc))
        perimeter.append((r1, cc))
    perimeter_set = set(perimeter)
    if not cells <= perimeter_set:
        return None
    missing = sorted(perimeter_set - cells)
    if len(missing) != 1:
        return None
    gap_r, gap_c = missing[0]
    if gap_r in (r0, r1) and gap_c in (c0, c1):
        return None

    if gap_r == r0:
        gap_side = "top"
    elif gap_r == r1:
        gap_side = "bottom"
    elif gap_c == c0:
        gap_side = "left"
    else:
        gap_side = "right"

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    inner = patch[1:-1, 1:-1]
    seed_points = np.argwhere(inner != bg)
    if seed_points.shape[0] != 1:
        return None

    center_r = height // 2
    center_c = width // 2
    sr, sc = map(int, seed_points[0])
    if (sr, sc) != (center_r - 1, center_c - 1):
        return None

    seed_r = r0 + center_r
    seed_c = c0 + center_c
    seed_color = int(grid[seed_r, seed_c])
    if seed_color in (bg, color):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "border_color": color,
        "seed_color": seed_color,
        "gap_side": gap_side,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each gapped rectangular frame according to the matching edge marker,
    then grow the mirrored copy outward through the gap direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background(g)
    h, w = g.shape

    marker_side_by_color = _edge_marker_map(g, bg)
    frames: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for comp in _components_of_color(g, color):
            info = _frame_info(g, comp, color, bg)
            if info is not None:
                frames.append(info)

    frames.sort(key=lambda item: tuple(item["bbox"]))  # deterministic order

    out = g.copy()
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                if out[r, c] != bg:
                    out[r, c] = bg

    for info in frames:
        r0, c0, r1, c1 = info["bbox"]
        border_color = int(info["border_color"])
        seed_color = int(info["seed_color"])
        gap_side = str(info["gap_side"])
        marker_side = marker_side_by_color[seed_color]

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if marker_side in ("top", "bottom"):
            patch = np.flipud(patch)
        else:
            patch = np.fliplr(patch)

        transformed = patch.copy()
        border_mask = patch == border_color
        seed_mask = patch == seed_color
        transformed[border_mask] = seed_color
        transformed[seed_mask] = border_color

        ph, pw = transformed.shape
        if gap_side == "top":
            dr, dc = -ph, 0
        elif gap_side == "bottom":
            dr, dc = ph, 0
        elif gap_side == "left":
            dr, dc = 0, -pw
        else:
            dr, dc = 0, pw

        rr0 = r0 + dr
        cc0 = c0 + dc
        rr1 = rr0 + ph
        cc1 = cc0 + pw
        assert 0 <= rr0 and rr1 <= h and 0 <= cc0 and cc1 <= w
        out[rr0:rr1, cc0:cc1] = transformed

    return out


def _balanced_sides(rng: random.Random, count: int) -> list[str]:
    sides: list[str] = []
    while len(sides) < count:
        batch = list(SIDES)
        rng.shuffle(batch)
        sides.extend(batch)
    return sides[:count]


def _build_frame_patch(
    height: int, width: int, border_color: int, seed_color: int, gap_side: str, gap_index: int
) -> np.ndarray:
    patch = np.full((height, width), BACKGROUND, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    if gap_side == "top":
        patch[0, gap_index] = BACKGROUND
    elif gap_side == "bottom":
        patch[-1, gap_index] = BACKGROUND
    elif gap_side == "left":
        patch[gap_index, 0] = BACKGROUND
    else:
        patch[gap_index, -1] = BACKGROUND
    patch[height // 2, width // 2] = seed_color
    return patch


def _copy_bbox_for_gap(r0: int, c0: int, height: int, width: int, gap_side: str) -> tuple[int, int, int, int]:
    if gap_side == "top":
        return (r0 - height, c0, r0 - 1, c0 + width - 1)
    if gap_side == "bottom":
        return (r0 + height, c0, r0 + 2 * height - 1, c0 + width - 1)
    if gap_side == "left":
        return (r0, c0 - width, r0 + height - 1, c0 - 1)
    return (r0, c0 + width, r0 + height - 1, c0 + 2 * width - 1)


def _union_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _padded_bbox(
    bbox: tuple[int, int, int, int], pad: int, h: int, w: int
) -> tuple[int, int, int, int]:
    return (
        max(0, bbox[0] - pad),
        max(0, bbox[1] - pad),
        min(h - 1, bbox[2] + pad),
        min(w - 1, bbox[3] + pad),
    )


def _fits(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return not bool(blocked[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _edge_positions(side: str, h: int, w: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(1, w - 1)]
    if side == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    if side == "left":
        return [(r, 0) for r in range(1, h - 1)]
    return [(r, w - 1) for r in range(1, h - 1)]


def _allowed_top_left_ranges(
    grid_h: int, grid_w: int, height: int, width: int, gap_side: str
) -> tuple[range, range] | None:
    r_min = 1
    c_min = 1
    r_max = grid_h - height - 1
    c_max = grid_w - width - 1

    if gap_side == "top":
        r_min = max(r_min, height + 1)
    elif gap_side == "bottom":
        r_max = min(r_max, grid_h - 2 * height - 1)
    elif gap_side == "left":
        c_min = max(c_min, width + 1)
    else:
        c_max = min(c_max, grid_w - 2 * width - 1)

    if r_min > r_max or c_min > c_max:
        return None
    return range(r_min, r_max + 1), range(c_min, c_max + 1)


def _validate_generated_grid(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape and not np.array_equal(out, grid)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(GRID_MIN, GRID_MAX)
    grid_w = rng.randint(GRID_MIN, GRID_MAX)
    n_objects = rng.choice(OBJECT_COUNT_CHOICES)

    usable_sizes = [s for s in OUTER_SIZE_CHOICES if 2 * s + 2 <= max(grid_h, grid_w)]
    if not usable_sizes:
        return None

    palette = list(COLORS)
    border_color = rng.choice(palette)
    palette.remove(border_color)
    rng.shuffle(palette)
    seed_colors = palette[:n_objects]
    if len(seed_colors) < n_objects:
        return None

    gap_sides = _balanced_sides(rng, n_objects)
    marker_sides = _balanced_sides(rng, n_objects)

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    specs: list[dict[str, object]] = []

    for idx in range(n_objects):
        seed_color = seed_colors[idx]
        gap_side = gap_sides[idx]
        marker_side = marker_sides[idx]
        placed = False

        for _ in range(PLACEMENT_RETRIES):
            height = rng.choice(usable_sizes)
            width = rng.choice(usable_sizes)
            ranges = _allowed_top_left_ranges(grid_h, grid_w, height, width, gap_side)
            if ranges is None:
                continue
            row_range, col_range = ranges
            r0 = rng.choice(list(row_range))
            c0 = rng.choice(list(col_range))
            bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
            copy_bbox = _copy_bbox_for_gap(r0, c0, height, width, gap_side)
            union_bbox = _union_bbox(bbox, copy_bbox)
            padded = _padded_bbox(union_bbox, pad=1, h=grid_h, w=grid_w)
            if not _fits(blocked, padded):
                continue

            if gap_side in ("top", "bottom"):
                gap_index = rng.randint(1, width - 2)
            else:
                gap_index = rng.randint(1, height - 2)

            patch = _build_frame_patch(height, width, border_color, seed_color, gap_side, gap_index)
            grid[r0 : r0 + height, c0 : c0 + width] = patch
            _mark(blocked, padded)
            specs.append(
                {
                    "seed_color": seed_color,
                    "marker_side": marker_side,
                }
            )
            placed = True
            break

        if not placed:
            return None

    used_markers: set[tuple[int, int]] = set()
    for spec in specs:
        candidates = _edge_positions(str(spec["marker_side"]), grid_h, grid_w)
        rng.shuffle(candidates)
        seed_color = int(spec["seed_color"])
        marker_pos = None
        for pos in candidates:
            if pos not in used_markers:
                marker_pos = pos
                break
        if marker_pos is None:
            return None
        used_markers.add(marker_pos)
        grid[marker_pos] = seed_color

    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
