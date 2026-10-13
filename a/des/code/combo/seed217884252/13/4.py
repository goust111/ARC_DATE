from __future__ import annotations

import random

import numpy as np


BG_COLOR = 7
FRAME_COLOR = 5
COLOR_POOL = tuple(color for color in range(10) if color not in (BG_COLOR, FRAME_COLOR))
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- generator cfg ---------------

GRID_H_RANGE = (24, 34)
GRID_W_RANGE = (24, 34)
PAIR_COUNT_RANGE = (2, 5)
PLACEMENT_MARGIN = 1
PLACEMENT_CLEARANCE = 1
MAX_CLUSTER_BOX_SIDE = 7
CLUSTER_BOX_SLACK = 6
GEN_ATTEMPTS = 400

AREA_DIM_OPTIONS: dict[int, list[tuple[int, int]]] = {
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
    9: [(3, 3)],
    10: [(2, 5), (5, 2)],
    12: [(2, 6), (6, 2), (3, 4), (4, 3)],
    15: [(3, 5), (5, 3)],
    16: [(4, 4)],
    18: [(3, 6), (6, 3)],
    20: [(4, 5), (5, 4)],
    24: [(4, 6), (6, 4)],
    25: [(5, 5)],
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components4_by_color(grid: np.ndarray, exclude: set[int] | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    blocked = set() if exclude is None else {int(v) for v in exclude}
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) in blocked:
                continue

            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(pts, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(coords.shape[0]),
                }
            )

    return comps


def _is_rect_frame(comp: dict, grid: np.ndarray, bg: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False

    ring_area = 2 * h + 2 * w - 4
    if int(comp["area"]) != ring_area:
        return False

    color = int(comp["color"])
    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    if not (
        np.all(patch[0, :] == color)
        and np.all(patch[-1, :] == color)
        and np.all(patch[:, 0] == color)
        and np.all(patch[:, -1] == color)
    ):
        return False
    if not np.all(patch[1:-1, 1:-1] == int(bg)):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each scattered monochrome cluster into the grey frame whose interior area matches it.

    Input structure:
    - orange background
    - several disjoint 1-pixel-thick grey rectangular frames with empty interiors
    - several disjoint monochrome 4-connected clusters in distinct colors

    Rule:
    - count the pixels in each cluster
    - find the frame whose interior area equals that count
    - fill that frame interior with the cluster color
    - erase the original cluster back to the background
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    components = _components4_by_color(g, exclude={bg})
    frames = [comp for comp in components if _is_rect_frame(comp, g, bg)]
    clusters = [comp for comp in components if not _is_rect_frame(comp, g, bg)]

    area_to_frame: dict[int, dict] = {}
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior_area = int((r1 - r0 - 1) * (c1 - c0 - 1))
        area_to_frame[interior_area] = frame

    for cluster in clusters:
        frame = area_to_frame.get(int(cluster["area"]))
        if frame is None:
            continue

        coords = cluster["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)

        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(cluster["color"])

    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _has_hole(mask: np.ndarray) -> bool:
    empty = ~mask
    if not empty.any():
        return False

    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((int(r), int(c)))

    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    return bool(np.any(empty & ~seen))


def _random_cluster_mask(area: int, rng: random.Random) -> np.ndarray | None:
    dims: list[tuple[int, int]] = []
    for h in range(2, min(MAX_CLUSTER_BOX_SIDE, area) + 1):
        for w in range(2, min(MAX_CLUSTER_BOX_SIDE, area) + 1):
            box_area = h * w
            if area <= box_area <= area + CLUSTER_BOX_SLACK:
                dims.append((int(h), int(w)))

    if not dims:
        return None

    for _ in range(250):
        h, w = rng.choice(dims)
        mask = np.zeros((h, w), dtype=bool)
        seed = (rng.randrange(h), rng.randrange(w))
        mask[seed] = True
        frontier: set[tuple[int, int]] = set()
        for dr, dc in DIR4:
            nr = int(seed[0] + dr)
            nc = int(seed[1] + dc)
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))

        while int(mask.sum()) < area and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            r, c = cell
            if mask[r, c]:
                continue
            mask[r, c] = True
            for dr, dc in DIR4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))

        if int(mask.sum()) != area:
            continue

        mask = _trim_mask(mask)
        if _has_hole(mask):
            continue
        if min(mask.shape) < 2:
            continue
        return mask

    return None


def _paint_mask(mask: np.ndarray, color: int, bg: int = BG_COLOR) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _frame_patch(inner_h: int, inner_w: int, frame_color: int = FRAME_COLOR, bg: int = BG_COLOR) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), int(bg), dtype=int)
    patch[0, :] = int(frame_color)
    patch[-1, :] = int(frame_color)
    patch[:, 0] = int(frame_color)
    patch[:, -1] = int(frame_color)
    return patch


def _bbox_clear(a: tuple[int, int, int, int], b: tuple[int, int, int, int], clearance: int = PLACEMENT_CLEARANCE) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + clearance < br0
        or br1 + clearance < ar0
        or ac1 + clearance < bc0
        or bc1 + clearance < ac0
    )


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    used_bboxes: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape

    min_r = PLACEMENT_MARGIN
    min_c = PLACEMENT_MARGIN
    max_r = h - ph - PLACEMENT_MARGIN
    max_c = w - pw - PLACEMENT_MARGIN
    if max_r < min_r or max_c < min_c:
        return None

    candidates = [(r0, c0) for r0 in range(min_r, max_r + 1) for c0 in range(min_c, max_c + 1)]
    rng.shuffle(candidates)

    for r0, c0 in candidates:
        bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        if all(_bbox_clear(bbox, other) for other in used_bboxes):
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            used_bboxes.append(bbox)
            return bbox

    return None


def _sample_pairs(rng: random.Random, n_pairs: int) -> list[dict] | None:
    if n_pairs > len(AREA_DIM_OPTIONS) or n_pairs > len(COLOR_POOL):
        return None

    areas = rng.sample(list(AREA_DIM_OPTIONS), k=n_pairs)
    colors = rng.sample(list(COLOR_POOL), k=n_pairs)

    pairs: list[dict] = []
    for area, color in zip(areas, colors):
        inner_h, inner_w = rng.choice(AREA_DIM_OPTIONS[int(area)])
        cluster_mask = _random_cluster_mask(int(area), rng)
        if cluster_mask is None:
            return None
        pairs.append(
            {
                "area": int(area),
                "color": int(color),
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "frame_patch": _frame_patch(int(inner_h), int(inner_w)),
                "cluster_patch": _paint_mask(cluster_mask, int(color)),
            }
        )

    return pairs


def _build_expected(inp: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for pair in pairs:
        cr0, cc0 = pair["cluster_origin"]
        cluster_patch = pair["cluster_patch"]
        cluster_view = out[cr0 : cr0 + cluster_patch.shape[0], cc0 : cc0 + cluster_patch.shape[1]]
        cluster_view[cluster_patch != BG_COLOR] = BG_COLOR

        fr0, fc0 = pair["frame_origin"]
        out[fr0 + 1 : fr0 + 1 + pair["inner_h"], fc0 + 1 : fc0 + 1 + pair["inner_w"]] = int(pair["color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    n_pairs = rng.randint(*PAIR_COUNT_RANGE)
    pairs = _sample_pairs(rng, n_pairs)
    if pairs is None:
        return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    used_bboxes: list[tuple[int, int, int, int]] = []

    for pair in sorted(pairs, key=lambda item: item["frame_patch"].size, reverse=True):
        bbox = _place_patch(grid, pair["frame_patch"], used_bboxes, rng)
        if bbox is None:
            return None
        pair["frame_origin"] = (int(bbox[0]), int(bbox[1]))

    for pair in sorted(pairs, key=lambda item: item["cluster_patch"].size, reverse=True):
        bbox = _place_patch(grid, pair["cluster_patch"], used_bboxes, rng)
        if bbox is None:
            return None
        pair["cluster_origin"] = (int(bbox[0]), int(bbox[1]))

    if _get_bg(grid) != BG_COLOR:
        return None

    expected = _build_expected(grid, pairs)
    if np.array_equal(expected, grid):
        return None

    try:
        predicted = main(grid)
    except Exception:
        return None
    if not np.array_equal(predicted, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
