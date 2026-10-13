import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = list(range(1, 10))
MAX_ATTEMPTS = 500
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FEASIBLE_BBOXES = [
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 5),
    (4, 2),
    (4, 3),
    (4, 4),
    (5, 2),
]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": len(coords),
                }
            )
    return comps


def _component_crop(grid: np.ndarray, comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_clusters(grid: np.ndarray) -> list[dict]:
    comps = _find_components(grid)
    singleton_by_color: dict[int, set[tuple[int, int]]] = {}
    comp_ids = {id(comp): idx for idx, comp in enumerate(comps)}
    for comp in comps:
        if comp["area"] == 1:
            singleton_by_color.setdefault(int(comp["color"]), set()).add(comp["coords"][0])

    clusters: list[dict] = []
    used_centers: set[int] = set()
    used_corner_sets: set[frozenset[tuple[int, int]]] = set()

    for marker_color, points in singleton_by_color.items():
        rows = sorted({r for r, _ in points})
        cols = sorted({c for _, c in points})
        for r0 in rows:
            for r1 in rows:
                if r1 <= r0:
                    continue
                for c0 in cols:
                    for c1 in cols:
                        if c1 <= c0:
                            continue
                        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
                        if not corners.issubset(points):
                            continue
                        key = frozenset(corners)
                        if key in used_corner_sets:
                            continue

                        boundary_ok = True
                        for cc in range(c0, c1 + 1):
                            if (r0, cc) not in corners and int(grid[r0, cc]) != BG:
                                boundary_ok = False
                            if (r1, cc) not in corners and int(grid[r1, cc]) != BG:
                                boundary_ok = False
                        for rr in range(r0, r1 + 1):
                            if (rr, c0) not in corners and int(grid[rr, c0]) != BG:
                                boundary_ok = False
                            if (rr, c1) not in corners and int(grid[rr, c1]) != BG:
                                boundary_ok = False
                        if not boundary_ok:
                            continue

                        inside = []
                        for comp in comps:
                            comp_color = int(comp["color"])
                            if comp_color == marker_color:
                                continue
                            cr0, cc0, cr1, cc1 = comp["bbox"]
                            if cr0 >= r0 + 2 and cr1 <= r1 - 2 and cc0 >= c0 + 2 and cc1 <= c1 - 2:
                                if all(r0 < rr < r1 and c0 < cc < c1 for rr, cc in comp["coords"]):
                                    inside.append(comp)
                        if len(inside) != 1:
                            continue

                        center = inside[0]
                        center_id = comp_ids[id(center)]
                        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
                        if int(np.count_nonzero(interior)) != center["area"]:
                            continue
                        if center_id in used_centers:
                            continue

                        used_centers.add(center_id)
                        used_corner_sets.add(key)
                        clusters.append(
                            {
                                "marker_color": marker_color,
                                "marker_bbox": (r0, c0, r1, c1),
                                "center": center,
                            }
                        )
    clusters.sort(key=lambda item: (item["marker_bbox"][0], item["marker_bbox"][1]))
    return clusters


def main(grid: np.ndarray) -> np.ndarray:
    """Extract each corner-marked center object, then rotate the ordered object colors one slot right.

    Input structure:
    - Black background surrounds 3 to 5 separated clusters.
    - Each cluster has four same-color single-pixel corner markers at the corners of an axis-aligned rectangle.
    - Inside each rectangle sits one monochrome 4-connected center object, separated from the markers by at least one background row and column.

    Rule:
    - Detect every valid four-corner marker rectangle and its unique enclosed center object.
    - Crop each center object to its own bounding box, then order the clusters by the marker rectangle's top-left coordinate.
    - Recolor slot i with the original center color from slot i-1 cyclically, keep each cropped shape unchanged, and place the recolored crops left-to-right with one black spacer column and top alignment.

    Key insight:
    - The marker pixels only define which objects to extract and how to order them; the cyclic change is applied to the enclosed object colors, not to the marker colors or to the marker geometry.
    """
    g = np.asarray(grid, dtype=int)
    clusters = _extract_clusters(g)
    if not clusters:
        raise ValueError("no valid corner-marker clusters found")

    crops = [_component_crop(g, cluster["center"]) for cluster in clusters]
    colors = [int(cluster["center"]["color"]) for cluster in clusters]
    rotated = [colors[(i - 1) % len(colors)] for i in range(len(colors))]

    out_h = max(crop.shape[0] for crop in crops)
    out_w = sum(crop.shape[1] for crop in crops) + len(crops) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    c0 = 0
    for crop, new_color in zip(crops, rotated):
        patch = crop.copy()
        patch[patch != BG] = new_color
        h, w = patch.shape
        out[:h, c0 : c0 + w] = patch
        c0 += w + 1
    return out


def _random_walk_path(h: int, w: int) -> list[tuple[int, int]]:
    corners = [((0, 0), (h - 1, w - 1)), ((0, w - 1), (h - 1, 0))]
    (sr, sc), (tr, tc) = random.choice(corners)
    r, c = sr, sc
    path = [(r, c)]
    while (r, c) != (tr, tc):
        moves = []
        if r < tr:
            moves.append((1, 0))
        elif r > tr:
            moves.append((-1, 0))
        if c < tc:
            moves.append((0, 1))
        elif c > tc:
            moves.append((0, -1))
        dr, dc = random.choice(moves)
        r += dr
        c += dc
        path.append((r, c))
    return path


def _shape_mask_with_bbox(h: int, w: int) -> np.ndarray:
    min_area = max(3, h + w - 1)
    max_area = min(7, h * w)
    area = random.randint(min_area, max_area)
    cells = set(_random_walk_path(h, w))
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.argmax() != 0 or len(rows) - 1 - rows[::-1].argmax() != h - 1:
        raise RuntimeError("shape bbox height shrank")
    if cols.argmax() != 0 or len(cols) - 1 - cols[::-1].argmax() != w - 1:
        raise RuntimeError("shape bbox width shrank")
    return mask


def _choose_center_colors(marker_colors: list[int]) -> list[int]:
    n = len(marker_colors)
    for _ in range(200):
        colors = random.sample(PALETTE, n)
        if len(set(colors)) == n and all(colors[i] != marker_colors[i] for i in range(n)):
            return colors
    raise RuntimeError("failed to sample center colors")


def _render_cluster(marker_color: int, center_color: int, mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    top_gap = random.randint(1, 2)
    bottom_gap = random.randint(1, 2)
    left_gap = random.randint(1, 2)
    right_gap = random.randint(1, 2)
    outer_h = h + top_gap + bottom_gap + 2
    outer_w = w + left_gap + right_gap + 2
    patch = np.full((outer_h, outer_w), BG, dtype=int)
    patch[0, 0] = marker_color
    patch[0, outer_w - 1] = marker_color
    patch[outer_h - 1, 0] = marker_color
    patch[outer_h - 1, outer_w - 1] = marker_color
    r0 = 1 + top_gap
    c0 = 1 + left_gap
    patch[r0 : r0 + h, c0 : c0 + w][mask] = center_color
    return patch


def _random_partition(n: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, n), parts - 1))
    sizes = []
    prev = 0
    for cut in cuts + [n]:
        sizes.append(cut - prev)
        prev = cut
    return sizes


def _layout_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    n = len(patches)
    order = list(range(n))
    random.shuffle(order)
    ordered = [patches[i] for i in order]

    row_options = [1, 2] if n <= 4 else [2, 3]
    random.shuffle(row_options)
    for rows in row_options:
        for _ in range(30):
            row_sizes = _random_partition(n, rows)
            x_gaps = [[random.randint(1, 3) for _ in range(size - 1)] for size in row_sizes]
            y_gaps = [random.randint(1, 3) for _ in range(rows - 1)]
            row_patches = []
            idx = 0
            row_widths = []
            row_heights = []
            for size, gaps in zip(row_sizes, x_gaps):
                current = ordered[idx : idx + size]
                idx += size
                row_patches.append(current)
                row_width = sum(p.shape[1] for p in current) + sum(gaps)
                row_height = max(p.shape[0] for p in current)
                row_widths.append(row_width)
                row_heights.append(row_height)
            left = random.randint(1, 3)
            right = random.randint(1, 3)
            top = random.randint(1, 3)
            bottom = random.randint(1, 3)
            width = left + max(row_widths) + right
            height = top + sum(row_heights) + sum(y_gaps) + bottom
            if width > 30 or height > 30:
                continue
            grid = np.full((height, width), BG, dtype=int)
            rr = top
            row_start_choices = [random.randint(left, width - right - rw) for rw in row_widths]
            for row_idx, current in enumerate(row_patches):
                cc = row_start_choices[row_idx]
                for j, patch in enumerate(current):
                    ph, pw = patch.shape
                    grid[rr : rr + ph, cc : cc + pw] = np.maximum(grid[rr : rr + ph, cc : cc + pw], patch)
                    if j + 1 < len(current):
                        cc += pw + x_gaps[row_idx][j]
                rr += row_heights[row_idx]
                if row_idx + 1 < rows:
                    rr += y_gaps[row_idx]
            return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(3, 5)
    center_h, center_w = random.choice(FEASIBLE_BBOXES)
    marker_colors = random.sample(PALETTE, n)
    center_colors = _choose_center_colors(marker_colors)

    patches = []
    for marker_color, center_color in zip(marker_colors, center_colors):
        mask = _shape_mask_with_bbox(center_h, center_w)
        patches.append(_render_cluster(marker_color, center_color, mask))

    grid = _layout_patches(patches)
    if grid is None:
        return None

    clusters = _extract_clusters(grid)
    if len(clusters) != n:
        return None
    center_boxes = {tuple(cluster["center"]["bbox"]) for cluster in clusters}
    if len({(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in center_boxes}) != 1:
        return None
    if len({cluster["center"]["color"] for cluster in clusters}) != n:
        return None

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    if len({tuple(row) for row in out}) == 1 and out.shape[0] > 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
