import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int):
    return ((r + dr, c + dc) for dr, dc in DIR4)


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and mask[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(comp, dtype=int))
    return comps


def _touches_border(comp: np.ndarray, h: int, w: int) -> bool:
    return bool(np.any((comp[:, 0] == 0) | (comp[:, 1] == 0) | (comp[:, 0] == h - 1) | (comp[:, 1] == w - 1)))


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    b = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(h, w, nr, nc) or not mask[nr, nc]:
                    b[r, c] = True
                    break
    return b


def _interior_cells(mask: np.ndarray) -> np.ndarray:
    return mask & ~_boundary_mask(mask)


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    comps = _components(mask)
    return len(comps) == 1


def _has_internal_hole(mask: np.ndarray) -> bool:
    inv = ~mask
    h, w = inv.shape
    for comp in _components(inv):
        if not _touches_border(comp, h, w):
            return True
    return False


def _grow_blob_mask(
    rows: int,
    cols: int,
    required_interior: int,
    rng: random.Random,
) -> np.ndarray | None:
    min_area = max(required_interior + 4, rows * cols // 2)
    target = rng.randint(min_area, rows * cols)
    mask = np.zeros((rows, cols), dtype=bool)
    sr, sc = rng.randint(0, rows - 1), rng.randint(0, cols - 1)
    mask[sr, sc] = True
    frontier: list[tuple[int, int]] = []
    for nr, nc in _neighbors4(sr, sc):
        if 0 <= nr < rows and 0 <= nc < cols:
            frontier.append((nr, nc))
    while int(mask.sum()) < target:
        if not frontier:
            return None
        ri = rng.randrange(len(frontier))
        r, c = frontier.pop(ri)
        if mask[r, c]:
            continue
        mask[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < rows and 0 <= nc < cols and not mask[nr, nc]:
                frontier.append((nr, nc))
    interior = _interior_cells(mask)
    if int(interior.sum()) < required_interior:
        return None
    return mask


def _rough_rect_mask(
    rows: int,
    cols: int,
    required_interior: int,
    rng: random.Random,
) -> np.ndarray | None:
    target = int(rng.randint(max(required_interior + 4, rows * cols // 2), rows * cols))
    mask = np.ones((rows, cols), dtype=bool)
    for _ in range(rows * cols * 2):
        if int(mask.sum()) <= target:
            break
        boundary = _boundary_mask(mask)
        candidates = list(zip(*np.where(boundary)))
        rng.shuffle(candidates)
        removed = False
        for rr, cc in candidates:
            if int(mask.sum()) - 1 < required_interior:
                break
            cand = mask.copy()
            cand[rr, cc] = False
            if not _is_connected(cand):
                continue
            if _has_internal_hole(cand):
                continue
            interior = _interior_cells(cand)
            if int(interior.sum()) < required_interior:
                continue
            mask = cand
            removed = True
            break
        if not removed:
            break
    if int(mask.sum()) < required_interior + 4:
        return None
    return mask if not _has_internal_hole(mask) else None


def _region_mask(
    rows: int,
    cols: int,
    required_interior: int,
    rng: random.Random,
) -> np.ndarray | None:
    for _ in range(30):
        if rng.random() < 0.5:
            mask = _grow_blob_mask(rows, cols, required_interior, rng)
        else:
            mask = _rough_rect_mask(rows, cols, required_interior, rng)
        if mask is None:
            continue
        interior = int(_interior_cells(mask).sum())
        if interior >= required_interior:
            return mask
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill concentric black rings inside each closed grey contour based on the number of seed pixels.

    Input structure:
    - Background is black. Several disjoint thin grey closed contours (4-connected, width 1) enclose interior regions.
    - Inside each region there are several seed pixels (1-4) of a single color, different from black and grey, and the same color within that region.

    Rule:
    - For each contour, count the interior seed pixels N and then flood one ring at a time from the boundary inward.
    - Repaint the first N layers of black pixels inside that region with the region's seed color (ring 1 touches the grey contour).
    - Keep grey boundaries unchanged; seed pixels stay as their original color.
    - Apply this independently for every enclosed region.

    Key insight:
    - The only non-black seed color identifies the fill color, and the number of seeds gives the exact ring thickness to grow inward.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    if h == 0 or w == 0:
        return out

    bg = int(BACKGROUND if BACKGROUND in g else Counter(g.flatten()).most_common(1)[0][0])
    vals, counts = np.unique(g, return_counts=True)
    non_bg_colors = [int(v) for v in vals if int(v) != bg]
    if not non_bg_colors:
        return out

    # Use the color with the largest connected component as the contour color.
    # In valid inputs, only the grey line satisfies this.
    color_maxcomp = {}
    for color in non_bg_colors:
        comps = _components(g == color)
        if not comps:
            continue
        color_maxcomp[int(color)] = max(int(len(c)) for c in comps)
    if not color_maxcomp:
        return out
    contour_color = max(color_maxcomp.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    if contour_color == bg:
        return out

    contour_comps = _components(g == contour_color)
    if not contour_comps:
        return out

    contour_id = -np.ones((h, w), dtype=int)
    for i, comp in enumerate(contour_comps):
        for r, c in comp:
            contour_id[r, c] = int(i)

    # Non-contour cells split into components; interior regions are components not touching border.
    regions = _components(g != contour_color)

    # Fill each enclosed region.
    region_mask_cache: dict[int, np.ndarray] = {}
    for comp in regions:
        if _touches_border(comp, h, w):
            continue

        comp_set = {(int(r), int(c)) for r, c in comp}
        if not comp_set:
            continue

        adjacent = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    cid = int(contour_id[nr, nc])
                    if cid >= 0:
                        adjacent.add(cid)
        if not adjacent or len(adjacent) != 1:
            # Skip ambiguous layouts to avoid cross-region overwrite.
            continue

        comp_coords = np.array(list(comp_set), dtype=int)
        seed_mask = np.array([g[r, c] != bg and g[r, c] != contour_color for r, c in comp_coords])
        if not bool(seed_mask.any()):
            continue

        seed_cols = [int(g[r, c]) for (r, c), is_seed in zip(comp_coords, seed_mask) if bool(is_seed)]
        if not seed_cols:
            continue
        # Expected to be one color per region, but tolerate noise by choosing the dominant one.
        seed_color, n = Counter(seed_cols).most_common(1)[0]
        if int(n) <= 0:
            continue

        black_coords = [tuple(p) for p in comp if g[int(p[0]), int(p[1])] == bg]
        if not black_coords:
            continue

        if int(contour_id[int(comp_coords[0, 0]), int(comp_coords[0, 1])]) not in range(len(contour_comps)):
            continue
        adj_id = next(iter(adjacent))
        mask_key = adj_id
        if mask_key in region_mask_cache:
            boundary_adj = region_mask_cache[mask_key]
        else:
            # The first layer: black cells touching the contour from inside.
            boundary_adj = np.zeros((h, w), dtype=bool)
            for rr, cc in black_coords:
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and contour_id[nr, nc] == adj_id:
                        boundary_adj[rr, cc] = True
                        break
            region_mask_cache[mask_key] = boundary_adj

        layer = np.zeros((h, w), dtype=int)
        for r, c in black_coords:
            layer[r, c] = 1 if boundary_adj[r, c] else 0

        if layer.sum() == 0:
            continue

        q = deque()
        dist = -np.ones((h, w), dtype=int)
        for r, c in black_coords:
            if layer[r, c]:
                dist[r, c] = 1
                q.append((r, c))

        if int(n) == 1:
            # Fast path: only immediate boundary ring.
            for r, c in black_coords:
                if dist[r, c] == 1:
                    out[r, c] = int(seed_color)
            continue

        region_set = {(int(r), int(c)) for r, c in black_coords}
        while q:
            r, c = q.popleft()
            d = int(dist[r, c])
            if d >= n:
                continue
            nd = d + 1
            for nr, nc in _neighbors4(r, c):
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and (int(nr), int(nc)) in region_set
                    and int(dist[nr, nc]) == -1
                ):
                    dist[nr, nc] = nd
                    q.append((nr, nc))

        for r, c in black_coords:
            if 1 <= dist[r, c] <= int(n):
                out[r, c] = int(seed_color)

    return out


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(12, 30)
    cols = rng.randint(12, 30)
    grid = np.full((rows, cols), BACKGROUND, dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)

    region_count = rng.choice([1, 1, 2, 2, 3, 4])
    for _ in range(region_count):
        placed = False
        for _ in range(180):
            seed_count = rng.randint(1, 4)
            region_rows = rng.randint(6, min(16, rows - 2))
            region_cols = rng.randint(6, min(16, cols - 2))
            if region_rows + 2 >= rows or region_cols + 2 >= cols:
                continue

            mask = _region_mask(region_rows, region_cols, seed_count + 1, rng)
            if mask is None:
                continue
            boundary = _boundary_mask(mask)
            interior_cells = np.where(_interior_cells(mask))
            interior_count = len(interior_cells[0])
            if interior_count <= seed_count:
                continue

            boundary_count = int(boundary.sum())
            if not (boundary_count > 0):
                continue

            max_r = rows - region_rows - 1
            max_c = cols - region_cols - 1
            if max_r < 1 or max_c < 1:
                continue

            for _ in range(80):
                r0 = rng.randint(1, max_r)
                c0 = rng.randint(1, max_c)
                rr0, rr1 = max(0, r0 - 1), min(rows, r0 + region_rows + 1)
                cc0, cc1 = max(0, c0 - 1), min(cols, c0 + region_cols + 1)
                if np.any(occupied[rr0:rr1, cc0:cc1]):
                    continue

                # Apply boundary and seeds.
                sub = mask
                local_b = np.argwhere(sub).copy()
                sub_r = local_b[:, 0] + r0
                sub_c = local_b[:, 1] + c0

                grid[sub_r, sub_c] = BACKGROUND
                grid[boundary.astype(bool), :] if False else None
                grid[np.ix_(sub_r, sub_c)]  # no-op, keeps type-checkers quiet

                global_boundary = np.zeros((rows, cols), dtype=bool)
                b_coords = np.argwhere(boundary)
                br = b_coords[:, 0] + r0
                bc = b_coords[:, 1] + c0
                global_boundary[br, bc] = True

                interior_indices = list(zip(interior_cells[0], interior_cells[1]))
                if len(interior_indices) < seed_count:
                    continue
                seed_color = rng.choice(SEED_COLORS)
                seed_pos = rng.sample(interior_indices, seed_count)

                grid[global_boundary] = GREY
                for dr, dc in seed_pos:
                    grid[r0 + dr, c0 + dc] = seed_color

                occ_mask = np.zeros_like(mask, dtype=bool)
                occ_mask[boundary] = True
                for dr, dc in seed_pos:
                    occ_mask[dr, dc] = True
                occupied[r0 : r0 + region_rows, c0 : c0 + region_cols] |= occ_mask
                placed = True
                break

            if placed:
                break
        if not placed:
            return None

    # Validate generated instance.
    try:
        _ = main(grid)
    except Exception:
        return None

    # Basic structural sanity: each contour must have at least one enclosed non-background color set.
    if not _is_valid_instance(grid):
        return None
    return grid


def _is_valid_instance(grid: np.ndarray) -> bool:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    if h < 5 or w < 5:
        return False

    non_bg = [int(v) for v in np.unique(g) if int(v) != BACKGROUND]
    if not non_bg:
        return False

    if GREY not in non_bg:
        return False

    # Must have at least one enclosed non-grey region with seeds.
    contour_id = g == GREY
    comps = _components(contour_id)
    if not comps:
        return False
    non_contour = g != GREY
    for region in _components(non_contour):
        if _touches_border(region, h, w):
            continue
        seed_count = 0
        for r, c in region:
            if g[r, c] != BACKGROUND and g[r, c] != GREY:
                seed_count += 1
        if seed_count > 0:
            return True
    return False


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = int(rng.integers(5, 31))
    w = int(rng.integers(5, 31))
    grid = np.zeros((h, w), dtype=int)
    for color in range(1, int(rng.integers(2, 6))):
        for _ in range(int(rng.integers(1, 4))):
            grid[int(rng.integers(0, h)), int(rng.integers(0, w))] = color
    return grid
