import random
from collections import deque

import numpy as np


BG = 0
ORANGE = 7
SKY = 8
PAIR_COLORS = (1, 2, 3, 4, 5, 6, 9)
MAX_GEN_ATTEMPTS = 400
PLACEMENT_GAP = 1


def _legend_cells(shape: tuple[int, int]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = shape
    assert h >= 2 and w >= 3, "grid must be large enough for the corner legend"
    orange = {(h - 2, w - 1), (h - 1, w - 1)}
    sky = {(h - 1, w - 3), (h - 1, w - 2)}
    return orange, sky


def _legend_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    orange, sky = _legend_cells(shape)
    for r, c in orange | sky:
        mask[r, c] = True
    return mask


def _coords_array(coords: list[tuple[int, int]] | set[tuple[int, int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(coords), dtype=int)
    assert arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0
    return arr


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _normalize_coords(coords: list[tuple[int, int]] | set[tuple[int, int]] | np.ndarray) -> tuple[tuple[tuple[int, int], ...], int, int]:
    arr = _coords_array(coords)
    r0, c0, _r1, _c1 = _bbox(arr)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in arr))
    return norm, int(r0), int(c0)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _scan_order_colors(grid: np.ndarray, legend_mask: np.ndarray) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if legend_mask[r, c]:
                continue
            color = int(grid[r, c])
            if color == BG:
                continue
            assert color not in {ORANGE, SKY}, "orange/sky must be reserved for the legend"
            if color not in seen:
                seen.add(color)
                colors.append(color)
    return colors


def _candidate_rectangles(
    stationary: np.ndarray,
    mover: np.ndarray,
    grid_shape: tuple[int, int],
    blocked: np.ndarray,
) -> list[dict[str, int]]:
    h, w = grid_shape
    total_area = int(stationary.shape[0] + mover.shape[0])
    stat_norm, stat_r0, stat_c0 = _normalize_coords(stationary)
    move_norm, move_r0, move_c0 = _normalize_coords(mover)
    stat_bbox = _bbox(stationary)
    stat_h = stat_bbox[2] - stat_bbox[0] + 1
    stat_w = stat_bbox[3] - stat_bbox[1] + 1
    move_bbox = _bbox(mover)
    move_h = move_bbox[2] - move_bbox[0] + 1
    move_w = move_bbox[3] - move_bbox[1] + 1

    candidates: list[dict[str, int]] = []
    for rect_h in range(1, total_area + 1):
        if total_area % rect_h != 0:
            continue
        rect_w = total_area // rect_h
        if rect_h < stat_h or rect_w < stat_w or rect_h < move_h or rect_w < move_w:
            continue
        for off_r in range(rect_h - stat_h + 1):
            for off_c in range(rect_w - stat_w + 1):
                placed_stationary = {(r + off_r, c + off_c) for r, c in stat_norm}
                complement = [
                    (rr, cc)
                    for rr in range(rect_h)
                    for cc in range(rect_w)
                    if (rr, cc) not in placed_stationary
                ]
                if len(complement) != int(mover.shape[0]):
                    continue
                comp_norm, comp_r0, comp_c0 = _normalize_coords(complement)
                if comp_norm != move_norm:
                    continue
                top_r = stat_r0 - off_r
                top_c = stat_c0 - off_c
                if top_r < 0 or top_c < 0 or top_r + rect_h > h or top_c + rect_w > w:
                    continue
                if bool(blocked[top_r : top_r + rect_h, top_c : top_c + rect_w].any()):
                    continue
                mover_dst_r0 = top_r + comp_r0
                mover_dst_c0 = top_c + comp_c0
                candidates.append(
                    {
                        "top_r": int(top_r),
                        "top_c": int(top_c),
                        "rect_h": int(rect_h),
                        "rect_w": int(rect_w),
                        "stationary_bbox_area": int(_bbox_area(stat_bbox)),
                        "move_l1": int(abs(mover_dst_r0 - move_r0) + abs(mover_dst_c0 - move_c0)),
                    }
                )
    return candidates


def _choose_rect(
    comp_a: np.ndarray,
    comp_b: np.ndarray,
    grid_shape: tuple[int, int],
    blocked: np.ndarray,
) -> tuple[int, int, int, int]:
    raw = _candidate_rectangles(comp_a, comp_b, grid_shape, blocked) + _candidate_rectangles(comp_b, comp_a, grid_shape, blocked)
    unique: dict[tuple[int, int, int, int], dict[str, int]] = {}
    for cand in raw:
        key = (cand["top_r"], cand["top_c"], cand["rect_h"], cand["rect_w"])
        prev = unique.get(key)
        if prev is None or (
            cand["stationary_bbox_area"],
            -cand["move_l1"],
        ) > (
            prev["stationary_bbox_area"],
            -prev["move_l1"],
        ):
            unique[key] = cand
    candidates = list(unique.values())
    assert candidates, "each color pair must assemble into at least one rectangle"

    anchor_like = [c for c in candidates if c["stationary_bbox_area"] == c["rect_h"] * c["rect_w"]]
    if anchor_like:
        candidates = anchor_like

    candidates.sort(
        key=lambda c: (
            -c["stationary_bbox_area"],
            c["move_l1"],
            c["top_r"],
            c["top_c"],
            c["rect_h"],
            c["rect_w"],
        )
    )
    best = candidates[0]
    return best["top_r"], best["top_c"], best["rect_h"], best["rect_w"]


def _paint_striped_rect(out: np.ndarray, top_r: int, top_c: int, rect_h: int, rect_w: int, color: int) -> None:
    patch = np.full((rect_h, rect_w), int(color), dtype=int)
    if rect_h > rect_w:
        patch[:, 1::2] = ORANGE
    else:
        patch[1::2, :] = SKY
    out[top_r : top_r + rect_h, top_c : top_c + rect_w] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each same-color fragment set into a completed rectangle, then stripe the rectangle according to its aspect ratio.

    Input structure:
    - The background is black.
    - Every non-black color appears in exactly two 4-connected monochrome fragments, and each pair can be completed into one solid rectangle by translation only.
    - The lower-right corner contains a fixed legend: a 2x1 orange vertical bar on the right edge and a 1x2 sky-blue horizontal bar on the bottom edge.

    Rule:
    - For each fragment color, find the rectangle obtained when the two same-color fragments are translated into a perfect, gap-free rectangle.
    - If the completed rectangle is taller than wide, redraw that rectangle in place and alternate its columns from left to right between the original color and orange.
    - If the completed rectangle is wider than tall or square, redraw that rectangle in place and alternate its rows from top to bottom between the original color and sky blue.
    - Clear any original fragment cells not occupied by the reassembled rectangle, and keep the orange/sky legend unchanged.

    Key insight:
    - The hard step is not the stripe coloring itself but recovering which two fragments belong to one hidden rectangle and where that rectangle sits; the corner legend then tells you whether to reason in vertical-column stripes or horizontal-row stripes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    orange_cells, sky_cells = _legend_cells((h, w))
    expected_orange = np.zeros_like(g, dtype=bool)
    expected_sky = np.zeros_like(g, dtype=bool)
    for r, c in orange_cells:
        expected_orange[r, c] = True
    for r, c in sky_cells:
        expected_sky[r, c] = True
    assert np.array_equal(g == ORANGE, expected_orange), "orange legend must be the unique lower-right 2x1 vertical bar"
    assert np.array_equal(g == SKY, expected_sky), "sky legend must be the unique lower-right 1x2 horizontal bar"

    legend_mask = expected_orange | expected_sky
    colors = _scan_order_colors(g, legend_mask)
    assert colors, "expected at least one colored fragment pair"

    out = np.full_like(g, BG)
    out[expected_orange] = ORANGE
    out[expected_sky] = SKY

    non_pair_fg = (g != BG) & ~legend_mask
    for color in colors:
        comps = _components((g == color) & ~legend_mask)
        assert len(comps) == 2, "each fragment color must appear in exactly two components"
        pair_mask = np.zeros_like(g, dtype=bool)
        for comp in comps:
            pair_mask[comp[:, 0], comp[:, 1]] = True
        blocked = non_pair_fg & ~pair_mask
        top_r, top_c, rect_h, rect_w = _choose_rect(comps[0], comps[1], g.shape, blocked)
        _paint_striped_rect(out, top_r, top_c, rect_h, rect_w, color)

    return out


def _grow_connected_mask(height: int, width: int, target_area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    mask[start] = True
    frontier = set()
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))
    while len(cells) < target_area:
        assert frontier
        chosen = random.choice(tuple(frontier))
        frontier.remove(chosen)
        cells.add(chosen)
        mask[chosen] = True
        cr, cc = chosen
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return mask


def _sample_rect_dims(branch: str) -> tuple[int, int]:
    if branch == "vert":
        rect_w = random.randint(3, 5)
        rect_h = random.randint(rect_w + 1, 8)
        return rect_h, rect_w
    if random.random() < 0.3:
        side = random.randint(4, 6)
        return side, side
    rect_h = random.randint(3, 5)
    rect_w = random.randint(max(4, rect_h), 8)
    return rect_h, rect_w


def _make_pair_patches(color: int, rect_h: int, rect_w: int) -> tuple[np.ndarray, np.ndarray]:
    inner_h, inner_w = rect_h - 2, rect_w - 2
    assert inner_h >= 1 and inner_w >= 1
    inner_area = inner_h * inner_w
    min_hole = 2 if inner_area >= 2 else 1
    max_hole = inner_area
    hole_area = random.randint(min_hole, max_hole)
    hole_mask = _grow_connected_mask(inner_h, inner_w, hole_area)

    anchor = np.full((rect_h, rect_w), int(color), dtype=int)
    inner = anchor[1:-1, 1:-1]
    inner[hole_mask] = BG

    hole_coords = np.argwhere(hole_mask)
    r0, c0 = map(int, hole_coords.min(axis=0))
    r1, c1 = map(int, hole_coords.max(axis=0))
    filler = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    filler_coords = hole_coords - np.array([r0, c0], dtype=int)
    filler[filler_coords[:, 0], filler_coords[:, 1]] = int(color)
    return anchor, filler


def _place_patch(grid: np.ndarray, patch: np.ndarray, top_r: int, top_c: int) -> None:
    h, w = patch.shape
    region = grid[top_r : top_r + h, top_c : top_c + w]
    mask = patch != BG
    assert not np.any(region[mask] != BG), "patch placement must not overlap existing colors"
    region[mask] = patch[mask]


def _find_slot(reserved: np.ndarray, patch_h: int, patch_w: int, gap: int) -> tuple[int, int] | None:
    h, w = reserved.shape
    positions = [(r, c) for r in range(h - patch_h + 1) for c in range(w - patch_w + 1)]
    random.shuffle(positions)
    for top_r, top_c in positions:
        rr0 = max(0, top_r - gap)
        cc0 = max(0, top_c - gap)
        rr1 = min(h, top_r + patch_h + gap)
        cc1 = min(w, top_c + patch_w + gap)
        if not bool(reserved[rr0:rr1, cc0:cc1].any()):
            return top_r, top_c
    return None


def _reserve_bbox(reserved: np.ndarray, top_r: int, top_c: int, patch_h: int, patch_w: int, gap: int) -> None:
    h, w = reserved.shape
    rr0 = max(0, top_r - gap)
    cc0 = max(0, top_c - gap)
    rr1 = min(h, top_r + patch_h + gap)
    cc1 = min(w, top_c + patch_w + gap)
    reserved[rr0:rr1, cc0:cc1] = True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    orange_cells, sky_cells = _legend_cells((h, w))
    for r, c in orange_cells:
        grid[r, c] = ORANGE
    for r, c in sky_cells:
        grid[r, c] = SKY
    _reserve_bbox(reserved, h - 2, w - 3, 2, 3, gap=1)

    pair_count = random.choice((2, 3))
    branches = ["vert", "horiz"]
    if pair_count == 3:
        branches.append(random.choice(("vert", "horiz")))
    random.shuffle(branches)
    colors = random.sample(PAIR_COLORS, pair_count)

    for color, branch in zip(colors, branches):
        rect_h, rect_w = _sample_rect_dims(branch)
        anchor, filler = _make_pair_patches(color, rect_h, rect_w)

        anchor_slot = _find_slot(reserved, anchor.shape[0], anchor.shape[1], PLACEMENT_GAP)
        if anchor_slot is None:
            return None
        _place_patch(grid, anchor, anchor_slot[0], anchor_slot[1])
        _reserve_bbox(reserved, anchor_slot[0], anchor_slot[1], anchor.shape[0], anchor.shape[1], PLACEMENT_GAP)

        filler_slot = _find_slot(reserved, filler.shape[0], filler.shape[1], PLACEMENT_GAP)
        if filler_slot is None:
            return None
        _place_patch(grid, filler, filler_slot[0], filler_slot[1])
        _reserve_bbox(reserved, filler_slot[0], filler_slot[1], filler.shape[0], filler.shape[1], PLACEMENT_GAP)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rectangle-pair assembly input")
