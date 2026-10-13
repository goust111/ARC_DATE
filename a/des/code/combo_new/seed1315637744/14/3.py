import numpy as np
from collections import deque


BG = 0
GREY = 5
_RNG = np.random.default_rng()
_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_QUADRANTS = ("ur", "ul", "ll", "lr")
_CW_TURNS = {"ur": 0, "ul": 1, "ll": 2, "lr": 3}
_ALL_DIMS = [(3, 3), (3, 5), (5, 3), (5, 5)]
_SMALL_DIMS = [(3, 3), (3, 5), (5, 3)]
_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
_MAX_GENERATION_ATTEMPTS = 400


def _find_cross(grid: np.ndarray) -> tuple[int, int, int]:
    H, W = grid.shape
    row_candidates: list[tuple[int, int]] = []
    col_candidates: list[tuple[int, int]] = []

    for r in range(H):
        color = int(grid[r, 0])
        if color != BG and bool(np.all(grid[r, :] == color)):
            row_candidates.append((r, color))
    for c in range(W):
        color = int(grid[0, c])
        if color != BG and bool(np.all(grid[:, c] == color)):
            col_candidates.append((c, color))

    for r, row_color in row_candidates:
        for c, col_color in col_candidates:
            if row_color == col_color:
                return row_color, int(r), int(c)
    raise AssertionError("failed to locate the grey cross")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _quadrant_of(coords: np.ndarray, center_r: int, center_c: int) -> str:
    r = int(coords[0, 0])
    c = int(coords[0, 1])
    if r < center_r and c > center_c:
        return "ur"
    if r < center_r and c < center_c:
        return "ul"
    if r > center_r and c < center_c:
        return "ll"
    if r > center_r and c > center_c:
        return "lr"
    raise AssertionError("object lies on the cross")


def _rotate_clockwise(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, -int(turns))


def _paste_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    H, W = grid.shape
    for r, c in np.argwhere(mask):
        rr = int(top + r)
        cc = int(left + c)
        if 0 <= rr < H and 0 <= cc < W:
            grid[rr, cc] = int(color)


def _render_centered(objects: list[dict], shape: tuple[int, int], center_r: int, center_c: int) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for obj in sorted(objects, key=lambda item: (-int(item["area"]), int(item["order"]))):
        rotated = _rotate_clockwise(obj["mask"], _CW_TURNS[obj["quadrant"]])
        top = int(center_r - rotated.shape[0] // 2)
        left = int(center_c - rotated.shape[1] // 2)
        _paste_mask(out, rotated, int(obj["color"]), top, left)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each quadrant object by a quadrant-dependent angle, then stack the rotated objects at the cross center.

    Input structure:
    - The background is black.
    - One complete grey row and one complete grey column intersect to form a cross that splits the grid into four quadrants.
    - Each quadrant contains one or more solid monochrome 4-connected objects, and each color appears only once.

    Rule:
    - Use the grey cross intersection as the center anchor.
    - Rotate every object according to its quadrant: upper-right = 0 degrees, upper-left = 90 degrees clockwise, lower-left = 180 degrees, lower-right = 270 degrees clockwise.
    - Sort the rotated objects by area from largest to smallest.
    - Draw them one by one on a black output grid of the same size, aligning each rotated bounding-box center to the cross center, so smaller objects are painted later and stay on top.
    - Remove the grey cross from the output.

    Key insight:
    - The puzzle combines two independent cues: quadrant location determines rotation, while visible object area determines the stacking order.
    """
    g = np.asarray(grid, dtype=int)
    cross_color, center_r, center_c = _find_cross(g)
    fg_mask = (g != BG) & (g != cross_color)
    comps = _components(fg_mask)

    objects: list[dict] = []
    for order, coords in enumerate(comps):
        colors = np.unique(g[coords[:, 0], coords[:, 1]])
        assert len(colors) == 1
        objects.append(
            {
                "mask": _coords_to_mask(coords),
                "color": int(colors[0]),
                "quadrant": _quadrant_of(coords, center_r, center_c),
                "area": int(len(coords)),
                "order": int(order),
            }
        )

    return _render_centered(objects, g.shape, center_r, center_c)


def _has_hole(mask: np.ndarray) -> bool:
    H, W = mask.shape
    bg_mask = ~mask
    if not bool(bg_mask.any()):
        return False

    seen = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(H):
        for c in (0, W - 1):
            if bool(bg_mask[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                queue.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if bool(bg_mask[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in _NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < H and 0 <= nc < W and bool(bg_mask[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                queue.append((nr, nc))
    return bool((bg_mask & ~seen).any())


def _grow_shape(used_areas: set[int], small_only: bool) -> np.ndarray | None:
    dims_pool = _SMALL_DIMS if small_only else _ALL_DIMS
    for _ in range(200):
        h, w = dims_pool[int(_RNG.integers(0, len(dims_pool)))]
        mask = np.zeros((h, w), dtype=bool)
        seeds = {
            (h // 2, w // 2),
            (0, w // 2),
            (h - 1, w // 2),
            (h // 2, 0),
            (h // 2, w - 1),
        }
        for r, c in seeds:
            mask[r, c] = True

        area_options = [area for area in range(len(seeds), h * w + 1) if area not in used_areas]
        if not area_options:
            continue
        target_area = int(area_options[int(_RNG.integers(0, len(area_options)))])

        while int(mask.sum()) < target_area:
            frontier: list[tuple[int, int]] = []
            frontier_seen: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                for dr, dc in _NEIGHBORS4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]) and (nr, nc) not in frontier_seen:
                        frontier_seen.add((nr, nc))
                        frontier.append((nr, nc))
            if not frontier:
                break
            pick_r, pick_c = frontier[int(_RNG.integers(0, len(frontier)))]
            mask[pick_r, pick_c] = True

        if int(mask.sum()) != target_area:
            continue
        if _has_hole(mask):
            continue

        mask = np.rot90(mask, int(_RNG.integers(0, 4)))
        if bool(_RNG.integers(0, 2)):
            mask = np.fliplr(mask)

        if not bool(mask[mask.shape[0] // 2, mask.shape[1] // 2]):
            continue
        if not bool(mask[0, :].any()) or not bool(mask[-1, :].any()):
            continue
        if not bool(mask[:, 0].any()) or not bool(mask[:, -1].any()):
            continue
        return mask
    return None


def _quadrant_bounds(quadrant: str, H: int, W: int, center_r: int, center_c: int) -> tuple[int, int, int, int]:
    if quadrant == "ul":
        return 0, center_r - 1, 0, center_c - 1
    if quadrant == "ur":
        return 0, center_r - 1, center_c + 1, W - 1
    if quadrant == "ll":
        return center_r + 1, H - 1, 0, center_c - 1
    return center_r + 1, H - 1, center_c + 1, W - 1


def _separated(rects: list[tuple[int, int, int, int]], cand: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = cand
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _place_object(
    mask: np.ndarray,
    quadrant: str,
    H: int,
    W: int,
    center_r: int,
    center_c: int,
    rects: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    h, w = mask.shape
    r_lo, r_hi, c_lo, c_hi = _quadrant_bounds(quadrant, H, W, center_r, center_c)
    if h > r_hi - r_lo + 1 or w > c_hi - c_lo + 1:
        return None

    row_values = np.arange(r_lo, r_hi - h + 2)
    col_values = np.arange(c_lo, c_hi - w + 2)
    candidates = [(int(r0), int(c0)) for r0 in row_values for c0 in col_values]
    _RNG.shuffle(candidates)

    for r0, c0 in candidates:
        cand = (r0, c0, r0 + h - 1, c0 + w - 1)
        if _separated(rects, cand):
            rects.append(cand)
            return r0, c0
    return None


def _build_input(objects: list[dict], H: int, W: int, center_r: int, center_c: int) -> np.ndarray:
    grid = np.full((H, W), BG, dtype=int)
    grid[center_r, :] = GREY
    grid[:, center_c] = GREY
    for obj in objects:
        _paste_mask(grid, obj["mask"], int(obj["color"]), int(obj["top"]), int(obj["left"]))
    return grid


def _gen_attempt() -> np.ndarray | None:
    H = int(_RNG.choice(np.arange(19, 30, 2)))
    W = int(_RNG.choice(np.arange(19, 30, 2)))
    center_r = H // 2
    center_c = W // 2

    counts = {quadrant: 1 + int(_RNG.integers(0, 2)) for quadrant in _QUADRANTS}
    total_objects = sum(counts.values())
    colors = list(_RNG.choice(_COLOR_POOL, size=total_objects, replace=False))

    used_areas: set[int] = set()
    rects_by_quadrant = {quadrant: [] for quadrant in _QUADRANTS}
    objects: list[dict] = []
    color_idx = 0

    for quadrant in _QUADRANTS:
        small_only = counts[quadrant] == 2
        for _ in range(counts[quadrant]):
            mask = _grow_shape(used_areas, small_only=small_only)
            if mask is None:
                return None
            pos = _place_object(mask, quadrant, H, W, center_r, center_c, rects_by_quadrant[quadrant])
            if pos is None:
                return None
            area = int(mask.sum())
            used_areas.add(area)
            top, left = pos
            objects.append(
                {
                    "mask": mask,
                    "color": int(colors[color_idx]),
                    "quadrant": quadrant,
                    "area": area,
                    "order": len(objects),
                    "top": int(top),
                    "left": int(left),
                }
            )
            color_idx += 1

    inp = _build_input(objects, H, W, center_r, center_c)
    expected = _render_centered(objects, inp.shape, center_r, center_c)

    fg_mask = (inp != BG) & (inp != GREY)
    if len(_components(fg_mask)) != total_objects:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
