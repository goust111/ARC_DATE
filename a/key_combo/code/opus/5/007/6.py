import math
import random
from collections import Counter

import numpy as np


BG = 0
REGION_COLORS = (1, 2, 3, 6)
DIVIDER_MAIN = 4
MARKER_COLOR = 5
DIVIDER_CENTER = 7
TARGET_K = 3
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude=()) -> list[int]:
    cols = set(map(int, grid.ravel()))
    cols -= set(map(int, exclude))
    return sorted(cols)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(
    grid: np.ndarray,
    *,
    target_colors,
    connectivity: int = 4,
) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    targets = set(map(int, target_colors))
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) not in targets:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if not in_bounds(grid.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def normalize_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    mins = coords.min(axis=0)
    shifted = coords - mins
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def centroid(coords: np.ndarray) -> tuple[float, float]:
    return float(coords[:, 0].mean()), float(coords[:, 1].mean())


def angle_clockwise_from_up(center: tuple[float, float], point: tuple[float, float]) -> float:
    dr = point[0] - center[0]
    dc = point[1] - center[1]
    return float((math.atan2(dc, -dr) + 2.0 * math.pi) % (2.0 * math.pi))


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    divider_mask = np.isin(g, [DIVIDER_MAIN, DIVIDER_CENTER])
    assert divider_mask.any(), "missing divider"

    orange = np.argwhere(g == DIVIDER_CENTER)
    if orange.size:
        center = (float(orange[:, 0].mean()), float(orange[:, 1].mean()))
    else:
        divider_coords = np.argwhere(divider_mask)
        center = centroid(divider_coords)

    marker_comps = find_components(g, target_colors=[MARKER_COLOR], connectivity=4)
    markers = [comp for comp in marker_comps if comp["area"] <= 3]
    region_colors = colors_present(g, exclude={BG, MARKER_COLOR, DIVIDER_MAIN, DIVIDER_CENTER})
    regions = find_components(g, target_colors=region_colors, connectivity=4)
    regions = [comp for comp in regions if comp["area"] >= 4]

    region_infos = []
    for comp in regions:
        ctr = centroid(comp["coords"])
        region_infos.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "bbox": comp["bbox"],
                "centroid": ctr,
                "angle": angle_clockwise_from_up(center, ctr),
            }
        )
    region_infos.sort(key=lambda item: item["angle"])

    shape_classes = {normalize_shape(comp["coords"]) for comp in markers}
    return {
        "grid": g,
        "center": center,
        "markers": markers,
        "regions": region_infos,
        "k": len(shape_classes),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count translation-normalized grey marker shapes and rotate the quadrant region colors by that many steps.

    Input structure:
    - Black background with a central yellow/orange cross divider.
    - Four larger single-color 4-connected regions sit around the divider in clockwise order.
    - Several small grey 4-connected marker objects of area at most three are scattered in background gaps.

    Rule:
    - Treat grey components of area at most three as markers and group them by translation-normalized shape.
    - Let K be the number of distinct marker shape classes.
    - Sort the large regions by centroid angle around the divider center, starting from 12 o'clock and moving clockwise.
    - Simultaneously recolor region R_i with the original color of region R_(i-K mod N), keep the divider unchanged, and erase the grey markers to background.

    Key insight:
    - The shift amount is encoded by marker shape diversity, not by the number or positions of the markers, so the solver must normalize shapes before rotating the region colors.
    """
    info = _parse_grid(np.asarray(grid, dtype=int))
    out = np.array(grid, dtype=int, copy=True)
    regions = info["regions"]
    k = info["k"]
    n = len(regions)
    assert n >= 2, "need at least two large regions"

    src_colors = [int(region["color"]) for region in regions]
    for i, region in enumerate(regions):
        new_color = src_colors[(i - k) % n]
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = new_color

    for marker in info["markers"]:
        coords = marker["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    return out


REGION_TEMPLATES = (
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0]], dtype=int),
)

MARKER_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (1, 1)),
)

QUADRANT_NAMES = ("TL", "TR", "BR", "BL")


def _paint_cross(grid: np.ndarray, center_r: int, center_c: int, half_h: int, half_w: int) -> None:
    grid[center_r - half_h : center_r + half_h + 1, center_c] = DIVIDER_MAIN
    grid[center_r, center_c - half_w : center_c + half_w + 1] = DIVIDER_MAIN
    grid[center_r, center_c] = DIVIDER_CENTER


def _transform_region_mask(mask: np.ndarray, quadrant: str) -> np.ndarray:
    out = mask
    if quadrant in ("TR", "BR"):
        out = np.fliplr(out)
    if quadrant in ("BL", "BR"):
        out = np.flipud(out)
    return out


def _region_anchor_bounds(
    grid_shape: tuple[int, int],
    center_r: int,
    center_c: int,
    mask_shape: tuple[int, int],
    quadrant: str,
) -> list[tuple[int, int]]:
    h, w = map(int, grid_shape)
    mh, mw = map(int, mask_shape)
    if quadrant == "TL":
        row_max = center_r - 2 - (mh - 1)
        col_max = center_c - 2 - (mw - 1)
        rows = range(0, row_max + 1)
        cols = range(0, col_max + 1)
    elif quadrant == "TR":
        row_max = center_r - 2 - (mh - 1)
        col_min = center_c + 2
        rows = range(0, row_max + 1)
        cols = range(col_min, w - mw + 1)
    elif quadrant == "BL":
        row_min = center_r + 2
        col_max = center_c - 2 - (mw - 1)
        rows = range(row_min, h - mh + 1)
        cols = range(0, col_max + 1)
    else:
        row_min = center_r + 2
        col_min = center_c + 2
        rows = range(row_min, h - mh + 1)
        cols = range(col_min, w - mw + 1)
    return [(r, c) for r in rows for c in cols]


def _place_regions(
    grid: np.ndarray,
    center_r: int,
    center_c: int,
    rng: random.Random,
) -> bool:
    colors = list(REGION_COLORS)
    rng.shuffle(colors)
    for quadrant, color in zip(QUADRANT_NAMES, colors):
        masks = list(REGION_TEMPLATES)
        rng.shuffle(masks)
        placed = False
        for base_mask in masks:
            mask = _transform_region_mask(base_mask, quadrant)
            anchors = _region_anchor_bounds(grid.shape, center_r, center_c, mask.shape, quadrant)
            rng.shuffle(anchors)
            for r0, c0 in anchors:
                coords = np.argwhere(mask)
                rr = coords[:, 0] + r0
                cc = coords[:, 1] + c0
                if np.any(grid[rr, cc] != BG):
                    continue
                grid[rr, cc] = int(color)
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _quadrant_of_point(center_r: int, center_c: int, r: int, c: int) -> str:
    if r < center_r and c < center_c:
        return "TL"
    if r < center_r and c > center_c:
        return "TR"
    if r > center_r and c > center_c:
        return "BR"
    if r > center_r and c < center_c:
        return "BL"
    return "AXIS"


def _can_place_marker(grid: np.ndarray, shape, origin_r: int, origin_c: int) -> bool:
    cells = {(origin_r + dr, origin_c + dc) for dr, dc in shape}
    for r, c in cells:
        if not in_bounds(grid.shape, r, c) or int(grid[r, c]) != BG:
            return False
    for r, c in cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cells or not in_bounds(grid.shape, nr, nc):
                continue
            if int(grid[nr, nc]) != BG:
                return False
    return True


def _place_markers(
    grid: np.ndarray,
    center_r: int,
    center_c: int,
    rng: random.Random,
) -> bool:
    total = rng.randint(4, 8)
    counts = [1, 1, 1]
    for _ in range(total - 3):
        counts[rng.randrange(TARGET_K)] += 1

    assignments = ["TL", "TR", "BR", "BL"]
    remaining = total - 4
    assignments.extend(rng.choice(QUADRANT_NAMES) for _ in range(remaining))
    rng.shuffle(assignments)

    shape_instances = []
    for shape_idx, count in enumerate(counts):
        shape_instances.extend([shape_idx] * count)
    rng.shuffle(shape_instances)

    if len(shape_instances) != len(assignments):
        return False

    for shape_idx, quadrant in zip(shape_instances, assignments):
        shape = MARKER_SHAPES[shape_idx]
        max_dr = max(dr for dr, _ in shape)
        max_dc = max(dc for _, dc in shape)
        candidates = []
        for r in range(grid.shape[0] - max_dr):
            for c in range(grid.shape[1] - max_dc):
                if _quadrant_of_point(center_r, center_c, r, c) != quadrant:
                    continue
                cells = [(r + dr, c + dc) for dr, dc in shape]
                if any(_quadrant_of_point(center_r, center_c, rr, cc) != quadrant for rr, cc in cells):
                    continue
                candidates.append((r, c))
        rng.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            if not _can_place_marker(grid, shape, r0, c0):
                continue
            for dr, dc in shape:
                grid[r0 + dr, c0 + dc] = MARKER_COLOR
            placed = True
            break
        if not placed:
            return False
    return True


def _validate_candidate(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    info = _parse_grid(grid)
    if len(info["regions"]) != 4:
        return False
    if len(info["markers"]) < 4 or len(info["markers"]) > 8:
        return False
    if info["k"] != TARGET_K:
        return False

    if len({region["color"] for region in info["regions"]}) != 4:
        return False

    marker_total = len(info["markers"])
    if marker_total == TARGET_K:
        return False

    quadrants = set()
    for marker in info["markers"]:
        q = _quadrant_of_point(int(round(info["center"][0])), int(round(info["center"][1])), *marker["coords"][0])
        if q == "AXIS":
            return False
        quadrants.add(q)
        coords_set = {tuple(map(int, rc)) for rc in marker["coords"]}
        for r, c in coords_set:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in coords_set or not in_bounds(grid.shape, nr, nc):
                    continue
                if int(grid[nr, nc]) != BG:
                    return False
    if quadrants != set(QUADRANT_NAMES):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.any(out == MARKER_COLOR):
        return False
    if np.any(grid[np.isin(grid, REGION_COLORS)] == BG):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.choice([15, 16, 17, 18, 19])
    w = rng.choice([15, 16, 17, 18, 19])
    center_r = h // 2
    center_c = w // 2
    grid = np.zeros((h, w), dtype=int)

    half_h = rng.choice([2, 3])
    half_w = rng.choice([2, 3])
    _paint_cross(grid, center_r, center_c, half_h, half_w)

    if not _place_regions(grid, center_r, center_c, rng):
        return None
    if not _place_markers(grid, center_r, center_c, rng):
        return None
    if not _validate_candidate(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
