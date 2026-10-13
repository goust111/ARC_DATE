from __future__ import annotations

import random
import sys
from collections import Counter

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

try:
    from arc_utils import bbox_coords, colors_present, crop_bbox, get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])

    def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
        cols = set(map(int, np.unique(grid)))
        if exclude is not None:
            cols -= set(map(int, exclude))
        return sorted(cols)

    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = bbox
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


BG = 0
MIN_GRID = 16
MAX_GRID = 30
MIN_OBJECTS = 2
MAX_OBJECTS = 4
MAX_ATTEMPTS = 600
OBJECT_ATTEMPTS = 120
MASK_ATTEMPTS = 120

QUADRANTS = {
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SW": (1, -1),
    "SE": (1, 1),
}


def _singleton_marker_info(grid: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int]]]:
    info = []
    for color in colors_present(grid, exclude={bg}):
        coords = np.argwhere(grid == color)
        if coords.shape[0] == 1:
            info.append((int(color), tuple(map(int, coords[0]))))
    info.sort()
    assert len(info) == 2
    return info


def _bbox_center_deltas(bbox: tuple[int, int, int, int], ref: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    rr = int(r0 + r1 - 2 * ref[0])
    cc = int(c0 + c1 - 2 * ref[1])
    return rr, cc


def _axis_from_bbox(bbox: tuple[int, int, int, int], ref: tuple[int, int]) -> str:
    dr, dc = _bbox_center_deltas(bbox, ref)
    assert dr != 0 and dc != 0
    return "lr" if dr * dc > 0 else "ud"


def _transform_with_reference(grid: np.ndarray, ref: tuple[int, int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    marker_info = _singleton_marker_info(g, bg)
    marker_colors = {color for color, _pos in marker_info}
    out = np.full_like(g, bg)

    for color, (r, c) in marker_info:
        out[r, c] = int(color)

    for color in colors_present(g, exclude={bg}):
        color = int(color)
        if color in marker_colors:
            continue
        coords = np.argwhere(g == color)
        assert coords.shape[0] >= 2
        r0, c0, r1, c1 = bbox_coords(coords)
        if _axis_from_bbox((r0, c0, r1, c1), ref) == "lr":
            rr = coords[:, 0]
            cc = c0 + c1 - coords[:, 1]
        else:
            rr = r0 + r1 - coords[:, 0]
            cc = coords[:, 1]
        out[rr, cc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each monochrome mask inside its own bbox using the hidden midpoint as the axis selector.

    Input structure:
    - Black background with exactly two different non-black singleton marker pixels.
    - Every other non-black color is a distinct monochrome mask object.
    - The two markers differ by an even row offset and an even column offset, so their midpoint is an unpainted grid cell.

    Rule:
    - Compute the hidden reference point as the midpoint of the two marker pixels.
    - For each mask color, find the center of that color's own bounding box and compare it with the hidden midpoint.
    - If the bbox center is in the northwest or southeast diagonal quadrant, mirror that mask left-right inside its own bbox; if it is in the northeast or southwest quadrant, mirror it up-down inside its own bbox.
    - Keep the two marker pixels unchanged and return a grid with the same shape as the input.

    Key insight:
    - The deciding reference point is never drawn directly: it is the geometric midpoint between the two singleton markers, and using either marker itself as the origin gives the wrong flip axis on valid examples.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    marker_info = _singleton_marker_info(g, int(get_bg(g)))
    (_c1, p1), (_c2, p2) = marker_info
    assert (p1[0] + p2[0]) % 2 == 0 and (p1[1] + p2[1]) % 2 == 0
    midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    return _transform_with_reference(g, midpoint)


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _touches_all_sides(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    rows = {r for r, _c in cells}
    cols = {c for _r, c in cells}
    return 0 in rows and h - 1 in rows and 0 in cols and w - 1 in cols


def _random_mask(h: int, w: int) -> np.ndarray | None:
    max_area = h * w - 1
    min_area = max(5, (h * w) // 3)
    if min_area > max_area:
        return None

    for _ in range(MASK_ATTEMPTS):
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = set(_neighbors4(start[0], start[1], h, w))
        target = random.randint(min_area, max_area)

        while frontier and (len(cells) < target or not _touches_all_sides(cells, h, w)):
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nr, nc in _neighbors4(cell[0], cell[1], h, w):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
            if len(cells) >= h * w:
                break

        if len(cells) == h * w or not _touches_all_sides(cells, h, w):
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(np.fliplr(mask), np.flipud(mask)):
            continue
        return mask
    return None


def _ranges_for_quadrant(name: str, grid_shape: tuple[int, int], midpoint: tuple[int, int], patch_shape: tuple[int, int]):
    h, w = grid_shape
    ph, pw = patch_shape
    mr, mc = midpoint
    border = 1
    gap = 2

    if name == "NW":
        return range(border, mr - ph - gap + 1), range(border, mc - pw - gap + 1)
    if name == "NE":
        return range(border, mr - ph - gap + 1), range(mc + gap, w - pw - border + 1)
    if name == "SW":
        return range(mr + gap, h - ph - border + 1), range(border, mc - pw - gap + 1)
    if name == "SE":
        return range(mr + gap, h - ph - border + 1), range(mc + gap, w - pw - border + 1)
    raise ValueError(name)


def _pick_quadrants(object_count: int) -> list[str]:
    axis_a = ["NW", "SE"]
    axis_b = ["NE", "SW"]
    mode = random.random()
    if mode < 0.45:
        quads = ["NW", "NE", "SW", "SE"]
    elif mode < 0.8:
        quads = random.sample(axis_a, 1) + random.sample(axis_b, 1) + [random.choice(list(QUADRANTS))]
    else:
        quads = random.sample(axis_a, 1) + random.sample(axis_b, 1)
    random.shuffle(quads)
    quads = list(dict.fromkeys(quads))
    if len(quads) < object_count:
        for quad in random.sample(list(QUADRANTS), 4):
            if quad not in quads:
                quads.append(quad)
            if len(quads) == object_count:
                break
    return quads[:object_count]


def _place_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    midpoint: tuple[int, int],
    quadrant: str,
) -> dict | None:
    h, w = grid.shape
    for _ in range(OBJECT_ATTEMPTS):
        ph = random.randint(3, min(6, h // 2))
        pw = random.randint(3, min(6, w // 2))
        mask = _random_mask(ph, pw)
        if mask is None:
            continue

        rows, cols = _ranges_for_quadrant(quadrant, (h, w), midpoint, (ph, pw))
        rows = list(rows)
        cols = list(cols)
        if not rows or not cols:
            continue
        random.shuffle(rows)
        random.shuffle(cols)

        for r0 in rows:
            for c0 in cols:
                r1 = r0 + ph - 1
                c1 = c0 + pw - 1
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(h - 1, r1 + 1)
                cc1 = min(w - 1, c1 + 1)
                if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    continue

                bbox = (r0, c0, r1, c1)
                axis = _axis_from_bbox(bbox, midpoint)
                expected_axis = "lr" if quadrant in {"NW", "SE"} else "ud"
                if axis != expected_axis:
                    continue

                patch = np.full((ph, pw), BG, dtype=int)
                patch[mask] = int(color)
                grid[r0 : r1 + 1, c0 : c1 + 1][mask] = int(color)
                blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
                return {"color": int(color), "bbox": bbox, "quadrant": quadrant, "axis": axis}
    return None


def _alternative_axis_differs(bbox: tuple[int, int, int, int], midpoint: tuple[int, int], marker: tuple[int, int]) -> bool:
    mid_dr, mid_dc = _bbox_center_deltas(bbox, midpoint)
    alt_dr, alt_dc = _bbox_center_deltas(bbox, marker)
    if alt_dr == 0 or alt_dc == 0:
        return False
    return (mid_dr * mid_dc > 0) != (alt_dr * alt_dc > 0)


def _validate_scene(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        if g.ndim != 2 or max(g.shape) > 30:
            return False
        bg = int(get_bg(g))
        if bg != BG:
            return False

        marker_info = _singleton_marker_info(g, bg)
        (_mc1, p1), (_mc2, p2) = marker_info
        if (p1[0] + p2[0]) % 2 or (p1[1] + p2[1]) % 2:
            return False
        midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        if g[midpoint] != bg:
            return False

        mask_colors = []
        axes = set()
        marker_based_diffs = [False, False]
        for color in colors_present(g, exclude={bg}):
            coords = np.argwhere(g == color)
            if coords.shape[0] == 1:
                continue
            if coords.shape[0] < 2:
                return False
            bbox = bbox_coords(coords)
            dr, dc = _bbox_center_deltas(bbox, midpoint)
            if dr == 0 or dc == 0:
                return False
            axis = _axis_from_bbox(bbox, midpoint)
            axes.add(axis)
            mask_colors.append(int(color))

            patch = crop_bbox(g, bbox)
            color_mask = patch == int(color)
            if np.array_equal(color_mask, np.fliplr(color_mask)) or np.array_equal(color_mask, np.flipud(color_mask)):
                return False
            if np.array_equal(np.fliplr(color_mask), np.flipud(color_mask)):
                return False

            if _alternative_axis_differs(bbox, midpoint, p1):
                marker_based_diffs[0] = True
            if _alternative_axis_differs(bbox, midpoint, p2):
                marker_based_diffs[1] = True

        if not (MIN_OBJECTS <= len(mask_colors) <= MAX_OBJECTS):
            return False
        if axes != {"lr", "ud"}:
            return False
        if not all(marker_based_diffs):
            return False

        out = main(g)
        if out.shape != g.shape:
            return False
        if np.array_equal(out, g):
            return False

        try:
            alt1 = _transform_with_reference(g, p1)
        except Exception:
            alt1 = None
        try:
            alt2 = _transform_with_reference(g, p2)
        except Exception:
            alt2 = None
        if alt1 is not None and np.array_equal(out, alt1):
            return False
        if alt2 is not None and np.array_equal(out, alt2):
            return False

        for color, pos in marker_info:
            if int(out[pos]) != int(color):
                return False

        for color in mask_colors:
            in_coords = np.argwhere(g == color)
            bbox = bbox_coords(in_coords)
            in_patch = crop_bbox(g, bbox)
            out_patch = crop_bbox(out, bbox)
            axis = _axis_from_bbox(bbox, midpoint)
            expected = np.fliplr(in_patch) if axis == "lr" else np.flipud(in_patch)
            if not np.array_equal(out_patch, expected):
                return False

        return True
    except Exception:
        return False


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID, MAX_GRID)
    w = random.randint(MIN_GRID, MAX_GRID)
    grid = np.zeros((h, w), dtype=int)

    if h < 16 or w < 16:
        return None

    mr = random.randint(6, h - 7)
    mc = random.randint(6, w - 7)
    midpoint = (mr, mc)

    max_dr = min(mr - 2, h - mr - 3, 6)
    max_dc = min(mc - 2, w - mc - 3, 6)
    if max_dr < 2 or max_dc < 2:
        return None

    dr = random.randint(2, max_dr)
    dc = random.randint(2, max_dc)
    dc *= random.choice((-1, 1))

    marker_positions = [(mr - dr, mc - dc), (mr + dr, mc + dc)]
    object_count = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    colors = random.sample(list(range(1, 10)), 2 + object_count)
    marker_colors = colors[:2]
    object_colors = colors[2:]

    blocked = np.zeros_like(grid, dtype=bool)
    for color, (r, c) in zip(marker_colors, marker_positions):
        grid[r, c] = int(color)
        rr0 = max(0, r - 1)
        cc0 = max(0, c - 1)
        rr1 = min(h - 1, r + 1)
        cc1 = min(w - 1, c + 1)
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

    quadrants = _pick_quadrants(object_count)

    objects = []
    for color, quadrant in zip(object_colors, quadrants):
        placed = _place_object(grid, blocked, int(color), midpoint, quadrant)
        if placed is None:
            return None
        objects.append(placed)

    if len(objects) < MIN_OBJECTS:
        return None
    if not _validate_scene(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-midpoint mirror puzzle")
