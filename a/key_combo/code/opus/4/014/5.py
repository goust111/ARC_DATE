import random
from collections import deque

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    ORANGE = 7
    TEAL = 8

    MARKERS = (BLUE, RED, GREEN, YELLOW, ORANGE)
    ALLOWED = (BLACK, BLUE, RED, GREEN, YELLOW, GREY, ORANGE, TEAL)


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 400


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.asarray(coords, dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("expected a 2D mask")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    if connectivity == 4:
        dirs = DIRS4
    else:
        dirs = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        )

    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def object_interior(grid: np.ndarray, background: int = Color.BLACK) -> np.ndarray:
    """
    Mirror the `arc_utils.object_interior()` semantics:
    return True for object pixels plus enclosed background holes.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(background)
    h, w = g.shape
    outside = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def enqueue(rr: int, cc: int) -> None:
        if 0 <= rr < h and 0 <= cc < w and g[rr, cc] == bg and not outside[rr, cc]:
            outside[rr, cc] = True
            queue.append((rr, cc))

    for rr in range(h):
        enqueue(rr, 0)
        enqueue(rr, w - 1)
    for cc in range(w):
        enqueue(0, cc)
        enqueue(h - 1, cc)

    while queue:
        rr, cc = queue.popleft()
        for dr, dc in DIRS4:
            enqueue(rr + dr, cc + dc)

    return (g != bg) | ((g == bg) & ~outside)


def _expanded_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r1 + 1), min(w - 1, c1 + 1)


def _marker_direction(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = shape
    if r == 0:
        return 1, 0
    if r == h - 1:
        return -1, 0
    if c == 0:
        return 0, 1
    if c == w - 1:
        return 0, -1
    raise ValueError("marker must lie on the boundary")


def _extract_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    markers = []
    seen = set()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            color = int(g[r, c])
            if color in Color.MARKERS:
                markers.append((r, c, color))
                seen.add((r, c))
    return sorted(markers)


def _unique_teal_object(grid: np.ndarray) -> np.ndarray:
    comps = components_mask(np.asarray(grid, dtype=int) == Color.TEAL, connectivity=4)
    assert len(comps) == 1, "expected exactly one teal 4-connected object"
    return comps[0]


def _paint_halo(grid: np.ndarray, teal_coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    out = np.asarray(grid, dtype=int).copy()
    bbox = bbox_coords(teal_coords)
    pr0, pc0, pr1, pc1 = _expanded_bbox(bbox, out.shape)
    patch = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=int)
    rel = teal_coords - np.array([[pr0, pc0]], dtype=int)
    patch[rel[:, 0], rel[:, 1]] = Color.TEAL
    enclosed = object_interior(patch, background=Color.BLACK)
    fill_mask = (patch == Color.BLACK) & ~enclosed
    region = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
    region[fill_mask] = Color.GREY
    return out, (pr0, pc0, pr1, pc1)


def _project_marker(out: np.ndarray, r: int, c: int, color: int) -> tuple[tuple[int, int] | None, str, list[tuple[int, int]]]:
    h, w = out.shape
    dr, dc = _marker_direction((h, w), r, c)
    rr = r + dr
    cc = c + dc
    path: list[tuple[int, int]] = []
    while 0 <= rr < h and 0 <= cc < w:
        path.append((rr, cc))
        value = int(out[rr, cc])
        if value == Color.GREY:
            out[rr, cc] = int(color)
            return (rr, cc), "hit", path
        if value == Color.TEAL:
            return None, "blocked", path
        rr += dr
        cc += dc
    return None, "miss", path


def _hole_count(mask: np.ndarray) -> int:
    obj = np.where(mask, Color.TEAL, Color.BLACK)
    interior = object_interior(obj, background=Color.BLACK)
    holes = interior & (obj == Color.BLACK)
    return len(components_mask(holes, connectivity=4))


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    if random.random() < 0.5:
        out = np.fliplr(out)
    out = np.rot90(out, k=random.randrange(4))
    return _trim_mask(out)


def _random_polyomino(area: int) -> np.ndarray | None:
    canvas = np.zeros((7, 7), dtype=bool)
    cells = {(3, 3)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < 7 and 0 <= nc < 7 and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        # Bias toward candidates with multiple neighbors to avoid long skinny paths.
        weights = []
        for nr, nc in frontier:
            degree = sum((nr + dr, nc + dc) in cells for dr, dc in DIRS4)
            center_bias = 6 - abs(nr - 3) - abs(nc - 3)
            weights.append(max(1, 2 * degree + center_bias))
        cells.add(random.choices(frontier, weights=weights, k=1)[0])

    for r, c in cells:
        canvas[r, c] = True
    mask = _trim_mask(canvas)
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return None
    if mask.all():
        return None
    if _hole_count(mask) != 0:
        return None
    return _transform_mask(mask)


def _random_holey_polyomino(area: int) -> np.ndarray | None:
    canvas = np.zeros((7, 7), dtype=bool)
    hole = (3, 3)
    cells = {
        (hole[0] + dr, hole[1] + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0)
    }
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < 7 and 0 <= nc < 7 and (nr, nc) not in cells and (nr, nc) != hole:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        frontier = sorted(set(frontier))
        weights = []
        for nr, nc in frontier:
            degree = sum((nr + dr, nc + dc) in cells for dr, dc in DIRS4)
            weights.append(max(1, 2 * degree))
        cells.add(random.choices(frontier, weights=weights, k=1)[0])

    for r, c in cells:
        canvas[r, c] = True
    mask = _trim_mask(canvas)
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return None
    if _hole_count(mask) != 1:
        return None
    return _transform_mask(mask)


def _random_object_mask() -> np.ndarray | None:
    if random.random() < 0.45:
        area = random.randint(8, 15)
        return _random_holey_polyomino(area)
    area = random.randint(8, 15)
    return _random_polyomino(area)


def _centered_positions(limit: int, size: int) -> list[int]:
    positions = [p for p in range(2, limit - size - 1)]
    if not positions:
        return []
    grid_center = (limit - 1) / 2.0
    central = [
        p
        for p in positions
        if abs((p + (size - 1) / 2.0) - grid_center) <= max(2.0, limit / 5.0)
    ]
    return central or positions


def _candidate_markers(shape: tuple[int, int], halo_bbox: tuple[int, int, int, int]) -> list[dict]:
    h, w = shape
    pr0, pc0, pr1, pc1 = halo_bbox
    candidates = []

    for c in range(pc0, pc1 + 1):
        path = [(rr, c) for rr in range(1, pr0 + 1)]
        candidates.append(
            {"pos": (0, c), "landing": (pr0, c), "path": path, "side": "top"}
        )
    for c in range(pc0, pc1 + 1):
        path = [(rr, c) for rr in range(h - 2, pr1 - 1, -1)]
        candidates.append(
            {"pos": (h - 1, c), "landing": (pr1, c), "path": path, "side": "bottom"}
        )
    for r in range(pr0, pr1 + 1):
        path = [(r, cc) for cc in range(1, pc0 + 1)]
        candidates.append(
            {"pos": (r, 0), "landing": (r, pc0), "path": path, "side": "left"}
        )
    for r in range(pr0, pr1 + 1):
        path = [(r, cc) for cc in range(w - 2, pc1 - 1, -1)]
        candidates.append(
            {"pos": (r, w - 1), "landing": (r, pc1), "path": path, "side": "right"}
        )
    return candidates


def _select_markers(shape: tuple[int, int], halo_bbox: tuple[int, int, int, int]) -> list[dict] | None:
    candidates = _candidate_markers(shape, halo_bbox)
    random.shuffle(candidates)
    target_count = random.choices((1, 2, 3), weights=(2, 4, 4), k=1)[0]
    chosen: list[dict] = []
    used_cells: set[tuple[int, int]] = set()
    used_landings: set[tuple[int, int]] = set()

    for cand in candidates:
        cand_path = set(cand["path"])
        if cand["landing"] in used_landings:
            continue
        if used_cells & cand_path:
            continue
        chosen.append(cand)
        used_cells |= cand_path
        used_landings.add(cand["landing"])
        if len(chosen) == target_count:
            break

    if len(chosen) != target_count:
        return None
    return chosen


def _simulate_paths(grid: np.ndarray) -> list[dict]:
    teal_coords = _unique_teal_object(grid)
    halo_grid, _ = _paint_halo(grid, teal_coords)
    markers = _extract_markers(grid)
    for r, c, _color in markers:
        halo_grid[r, c] = Color.BLACK

    results = []
    for r, c, color in markers:
        landing, status, path = _project_marker(halo_grid, r, c, color)
        results.append(
            {
                "marker": (r, c, color),
                "landing": landing,
                "status": status,
                "path": path,
            }
        )
    return results


def _valid_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not set(map(int, np.unique(g))) <= set(Color.ALLOWED):
        return False

    teal_comps = components_mask(g == Color.TEAL, connectivity=4)
    if len(teal_comps) != 1:
        return False
    teal = teal_comps[0]
    if not 8 <= len(teal) <= 15:
        return False

    bbox = bbox_coords(teal)
    halo_bbox = _expanded_bbox(bbox, g.shape)
    pr0, pc0, pr1, pc1 = halo_bbox
    if pr0 <= 0 or pc0 <= 0 or pr1 >= g.shape[0] - 1 or pc1 >= g.shape[1] - 1:
        return False

    patch = crop_bbox(np.where(g == Color.TEAL, Color.TEAL, Color.BLACK), bbox)
    holes = _hole_count(patch == Color.TEAL)
    if holes not in (0, 1):
        return False

    markers = _extract_markers(g)
    if not 1 <= len(markers) <= 3:
        return False
    if len({color for _, _, color in markers}) != len(markers):
        return False
    if any((r in (0, g.shape[0] - 1)) and (c in (0, g.shape[1] - 1)) for r, c, _ in markers):
        return False

    nonzero = np.argwhere(g != Color.BLACK)
    marker_cells = {(r, c) for r, c, _ in markers}
    teal_cells = {tuple(coord) for coord in teal}
    if any(tuple(coord) not in marker_cells and tuple(coord) not in teal_cells for coord in nonzero):
        return False

    sims = _simulate_paths(g)
    all_path_cells = [set(item["path"]) for item in sims]
    for item in sims:
        if item["status"] not in {"hit", "blocked"}:
            return False
    for i in range(len(all_path_cells)):
        for j in range(i + 1, len(all_path_cells)):
            if all_path_cells[i] & all_path_cells[j]:
                return False
    landings = [item["landing"] for item in sims if item["landing"] is not None]
    if len(landings) != len(set(landings)):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project boundary marker colors onto the grey halo around a central teal object.

    Input structure:
    - The grid is a black background containing one central teal 4-connected object.
    - Single-pixel colored markers sit on the outer boundary and point inward along their row or column.
    - The teal object may contain one enclosed black hole, and the padded bbox around it does not touch the canvas edge.

    Rule:
    - Find the teal object's tight bbox and expand it by one cell in every direction, clipped to the canvas.
    - Recolor black pixels inside that expanded bbox grey, but preserve any enclosed black holes and keep the teal object unchanged.
    - Remove each boundary marker from the canvas, then march inward from that edge cell until the first grey or teal cell is reached.
    - If the first hit is grey, replace it with the marker's color; if teal is reached first, the marker has no effect.

    Key insight:
    - The output is defined by a two-stage interaction: the teal object's geometry first creates a halo surface, and the boundary markers then recolor the first visible halo cells along straight inward projections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    teal_coords = _unique_teal_object(g)
    markers = _extract_markers(g)

    out, _ = _paint_halo(g, teal_coords)
    for r, c, _color in markers:
        out[r, c] = Color.BLACK
    for r, c, color in markers:
        _project_marker(out, r, c, color)
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(13, 18)
        w = random.randint(13, 18)

        mask = _random_object_mask()
        if mask is None:
            continue
        mh, mw = mask.shape

        row_choices = _centered_positions(h, mh)
        col_choices = _centered_positions(w, mw)
        if not row_choices or not col_choices:
            continue

        r0 = random.choice(row_choices)
        c0 = random.choice(col_choices)

        grid = np.zeros((h, w), dtype=int)
        region = grid[r0 : r0 + mh, c0 : c0 + mw]
        region[mask] = Color.TEAL

        teal_coords = np.argwhere(grid == Color.TEAL)
        bbox = bbox_coords(teal_coords)
        halo_bbox = _expanded_bbox(bbox, grid.shape)
        markers = _select_markers(grid.shape, halo_bbox)
        if markers is None:
            continue

        colors = random.sample(list(Color.MARKERS), k=len(markers))
        for cand, color in zip(markers, colors):
            rr, cc = cand["pos"]
            grid[rr, cc] = color

        if _valid_input(grid):
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
