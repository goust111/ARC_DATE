from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
OUTLINE_COLOR = 7
MARKER_COLOR = 6
OBSTACLE_COLOR = 2

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = DIRS4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

GENERATE_ATTEMPTS = 700


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    dirs = DIRS4 if int(connectivity) == 4 else DIRS8
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def flood_from_border(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((int(r), int(c)))
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if in_bounds(passable.shape, nr, nc) and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def direction_from_centroid(cr: float, cc: float, r: int, c: int) -> tuple[int, int]:
    dr = float(r) - float(cr)
    dc = float(c) - float(cc)
    if abs(dc) >= abs(dr):
        return (0, 1) if dc > 0 else (0, -1)
    return (1, 0) if dr > 0 else (-1, 0)


def outer_contour(bg_mask: np.ndarray, outline_mask: np.ndarray, exterior_mask: np.ndarray) -> np.ndarray:
    src = np.zeros_like(outline_mask, dtype=bool)
    for r, c in np.argwhere(exterior_mask & bg_mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRS4:
            nr, nc = rr + dr, cc + dc
            if in_bounds(outline_mask.shape, nr, nc) and outline_mask[nr, nc]:
                src[rr, cc] = True
                break
    return src


def largest_non_bg_component(grid: np.ndarray, bg: int) -> np.ndarray:
    comps = components(grid != int(bg), connectivity=8)
    if not comps:
        raise AssertionError("expected a foreground outline")
    coords = max(comps, key=len)
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def extract_layout(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    bg = int(Counter(map(int, grid.flat)).most_common(1)[0][0])
    outline_mask = largest_non_bg_component(grid, bg)
    exterior_mask = flood_from_border(~outline_mask)
    interior_mask = (~exterior_mask) & (~outline_mask)

    interior_colors = Counter(int(grid[r, c]) for r, c in np.argwhere(interior_mask & (grid != bg)))
    if not interior_colors:
        raise AssertionError("expected at least one interior marker pixel")
    marker_color = min(interior_colors, key=lambda color: (interior_colors[color], color))

    marker_source_mask = interior_mask & (grid == int(marker_color))
    sources = outer_contour(grid == bg, outline_mask, exterior_mask)
    return bg, outline_mask, exterior_mask, interior_mask, int(marker_color), marker_source_mask


def ray_paths(outline_mask: np.ndarray, source_mask: np.ndarray) -> list[dict[str, object]]:
    coords = np.argwhere(outline_mask)
    cr, cc = coords.mean(axis=0)
    rays: list[dict[str, object]] = []
    for r, c in np.argwhere(source_mask):
        rr = int(r)
        cc2 = int(c)
        dr, dc = direction_from_centroid(float(cr), float(cc), rr, cc2)
        path: list[tuple[int, int]] = []
        x, y = rr, cc2
        while in_bounds(outline_mask.shape, x, y):
            path.append((x, y))
            x += dr
            y += dc
        rays.append({"source": (rr, cc2), "dir": (dr, dc), "path": path})
    return rays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace a closed outline with outward marker-colored rays clipped by exterior obstacles.

    Input structure:
    - The grid has a dominant black background, one largest 8-connected outline component,
      a few marker pixels inside that closed outline, and several small obstacle components outside.
    - The outline is only a thin border, so its enclosed region remains mostly background.
    - Exterior obstacle components are separated from the outline and from each other.

    Rule:
    - Identify the largest non-background 8-connected component as the outline and compute its centroid.
    - Find the outside contour: background cells that are 4-adjacent to the outline and reachable from the grid border.
    - Infer the paint color from the least frequent non-background color inside the enclosed interior.
    - From each outside-contour cell, fire a ray in the dominant cardinal direction away from the centroid;
      ties between vertical and horizontal go to the horizontal direction.
    - Paint each ray until the next step would leave the grid or hit an exterior obstacle, then erase the original
      outline and the original interior marker pixels.

    Key insight:
    - The visible output keeps only the rays: the outline and its interior markers are temporary geometry used to
      determine ray sources, directions, and paint color.
    """
    g = np.asarray(grid, dtype=int)
    bg, outline_mask, _exterior_mask, _interior_mask, marker_color, marker_source_mask = extract_layout(g)
    obstacle_mask = (g != int(bg)) & (~outline_mask) & (~marker_source_mask)

    out = g.copy()
    for ray in ray_paths(outline_mask, outer_contour(g == bg, outline_mask, flood_from_border(~outline_mask))):
        dr, dc = ray["dir"]
        for x, y in ray["path"]:
            out[int(x), int(y)] = int(marker_color)
            nx, ny = int(x) + int(dr), int(y) + int(dc)
            if not in_bounds(g.shape, nx, ny) or obstacle_mask[nx, ny]:
                break

    out[outline_mask] = int(bg)
    out[marker_source_mask] = int(bg)
    return out


def build_outline_local() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    for _ in range(600):
        h = random.randint(5, 7)
        w = random.randint(5, 7)
        left = [random.randint(0, 1)]
        right = [left[0] + random.randint(2, min(4, w - 1 - left[0]))]
        for _row in range(1, h):
            new_left = max(0, min(w - 3, left[-1] + random.choice([-1, 0, 1])))
            new_right = max(new_left + 2, min(w - 1, right[-1] + random.choice([-1, 0, 1])))
            left.append(new_left)
            right.append(new_right)

        fill = np.zeros((h, w), dtype=bool)
        for r, (c0, c1) in enumerate(zip(left, right)):
            fill[r, c0 : c1 + 1] = True

        outline = fill.copy()
        for r in range(h):
            for c in range(w):
                if not fill[r, c]:
                    continue
                if all(in_bounds(fill.shape, r + dr, c + dc) and fill[r + dr, c + dc] for dr, dc in DIRS4):
                    outline[r, c] = False

        outline = np.pad(outline, 1)
        exterior = flood_from_border(~outline)
        interior = (~exterior) & (~outline)
        source_mask = outer_contour(np.ones_like(outline, dtype=bool), outline, exterior)

        if len(components(outline, connectivity=8)) != 1:
            continue
        if not 12 <= int(outline.sum()) <= 20:
            continue
        if int(interior.sum()) < 3:
            continue

        coords = np.argwhere(outline)
        cr, cc = coords.mean(axis=0)
        if not interior[int(round(float(cr))), int(round(float(cc)))]:
            continue

        ambiguous = False
        for r, c in np.argwhere(source_mask):
            if abs(abs(float(r) - float(cr)) - abs(float(c) - float(cc))) < 1e-9:
                ambiguous = True
                break
        if ambiguous:
            continue
        return outline, interior, source_mask
    return None


def touches_mask8(cells: set[tuple[int, int]], mask: np.ndarray) -> bool:
    for r, c in cells:
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and (nr, nc) not in cells:
                return True
    return False


def touches_cells8(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for dr, dc in DIRS8:
            p = (r + dr, c + dc)
            if p in occupied and p not in cells:
                return True
    return False


def can_place_component(
    cells: set[tuple[int, int]],
    allowed_mask: np.ndarray,
    outline_mask: np.ndarray,
    occupied: set[tuple[int, int]],
) -> bool:
    for r, c in cells:
        if not in_bounds(allowed_mask.shape, r, c) or not allowed_mask[r, c] or (r, c) in occupied:
            return False
    return not touches_mask8(cells, outline_mask) and not touches_cells8(cells, occupied)


def grow_component(
    anchor: tuple[int, int],
    target_size: int,
    allowed_mask: np.ndarray,
    outline_mask: np.ndarray,
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    comp = {anchor}
    if not can_place_component(comp, allowed_mask, outline_mask, occupied):
        return None
    while len(comp) < int(target_size):
        frontier: list[tuple[int, int]] = []
        for r, c in comp:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if in_bounds(allowed_mask.shape, nr, nc) and allowed_mask[nr, nc] and (nr, nc) not in comp:
                    frontier.append((nr, nc))
        random.shuffle(frontier)
        for cand in frontier:
            new_comp = set(comp)
            new_comp.add(cand)
            if can_place_component(new_comp, allowed_mask, outline_mask, occupied):
                comp = new_comp
                break
        else:
            break
    return comp if can_place_component(comp, allowed_mask, outline_mask, occupied) else None


def place_obstacles(
    outline_mask: np.ndarray,
    interior_mask: np.ndarray,
    source_mask: np.ndarray,
    exterior_mask: np.ndarray,
) -> list[set[tuple[int, int]]]:
    rays = ray_paths(outline_mask, source_mask)
    free_exterior = exterior_mask & (~source_mask) & (~interior_mask)
    occupied: set[tuple[int, int]] = set()
    obstacle_components: list[set[tuple[int, int]]] = []

    n_obstacles = random.randint(2, 4)
    n_blocking = random.randint(1, min(2, n_obstacles))
    blocking_rays = [ray for ray in rays if len(ray["path"]) >= 4]
    random.shuffle(blocking_rays)

    for ray in blocking_rays:
        if len(obstacle_components) >= n_blocking:
            break
        candidate_cells = [p for p in ray["path"][2:-1] if free_exterior[p]]
        random.shuffle(candidate_cells)
        for anchor in candidate_cells:
            comp = grow_component(anchor, random.randint(1, 3), free_exterior, outline_mask, occupied)
            if comp is None:
                continue
            obstacle_components.append(comp)
            occupied |= comp
            break

    loose_candidates = [tuple(map(int, p)) for p in np.argwhere(free_exterior)]
    random.shuffle(loose_candidates)
    while len(obstacle_components) < n_obstacles:
        for anchor in loose_candidates:
            if anchor in occupied:
                continue
            comp = grow_component(anchor, random.randint(1, 3), free_exterior, outline_mask, occupied)
            if comp is None:
                continue
            obstacle_components.append(comp)
            occupied |= comp
            break
        else:
            break
    return obstacle_components


def marker_candidates(interior_mask: np.ndarray, outline_mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r, c in np.argwhere(interior_mask):
        cell = {(int(r), int(c))}
        if not touches_mask8(cell, outline_mask):
            out.append((int(r), int(c)))
    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        local = build_outline_local()
        if local is None:
            continue
        outline_local, _interior_local, _source_local = local

        h = random.randint(15, 19)
        w = random.randint(15, 19)
        if outline_local.shape[0] + 4 > h or outline_local.shape[1] + 4 > w:
            continue

        off_r = random.randint(2, h - outline_local.shape[0] - 2)
        off_c = random.randint(2, w - outline_local.shape[1] - 2)

        outline_mask = np.zeros((h, w), dtype=bool)
        for r, c in np.argwhere(outline_local):
            outline_mask[off_r + int(r), off_c + int(c)] = True

        exterior_mask = flood_from_border(~outline_mask)
        interior_mask = (~exterior_mask) & (~outline_mask)
        source_mask = outer_contour(np.ones((h, w), dtype=bool), outline_mask, exterior_mask)

        markers = marker_candidates(interior_mask, outline_mask)
        if len(markers) < 2:
            continue

        obstacle_components = place_obstacles(outline_mask, interior_mask, source_mask, exterior_mask)
        if len(obstacle_components) < 2:
            continue

        grid = np.full((h, w), BACKGROUND, dtype=int)
        grid[outline_mask] = OUTLINE_COLOR

        marker_count = 1 if random.random() < 0.5 else 2
        for r, c in random.sample(markers, marker_count):
            grid[r, c] = MARKER_COLOR

        for comp in obstacle_components:
            for r, c in comp:
                grid[r, c] = OBSTACLE_COLOR

        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
