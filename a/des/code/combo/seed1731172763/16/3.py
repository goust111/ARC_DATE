import random
from collections import deque

import numpy as np


BG = 8
FG_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_RETRIES = 300
GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (12, 20)
OBJECT_COUNT_OPTIONS = (1, 2, 3)
EDGE_MARGIN = 2
BOX_GAP = 1
MODE_OPTIONS = ("none", "col_overlap", "row_overlap")


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each interior object to the four grid edges using its bounding-box size.

    Every object disappears from the center. Its bounding-box width becomes a
    horizontal stripe on the top and bottom edges, and its bounding-box height
    becomes a vertical stripe on the left and right edges. Components are
    processed in row-major discovery order, so later objects overwrite earlier
    ones on overlapping edge cells.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((h, w), BG, dtype=int)

    for comp in _find_components(g):
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        out[0, c0 : c1 + 1] = color
        out[h - 1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, 0] = color
        out[r0 : r1 + 1, w - 1] = color

    return out


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))

    return len(seen) == int(mask.sum())


def _spans_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _make_frame(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _make_plus(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 2, :] = True
    mask[:, w // 2] = True
    return mask


def _make_tee(h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    orient = rng.choice(("up", "down", "left", "right"))
    if orient == "up":
        mask[0, :] = True
        mask[:, w // 2] = True
    elif orient == "down":
        mask[-1, :] = True
        mask[:, w // 2] = True
    elif orient == "left":
        mask[:, 0] = True
        mask[h // 2, :] = True
    else:
        mask[:, -1] = True
        mask[h // 2, :] = True
    return mask


def _make_ell(h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    corner = rng.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        mask[0, :] = True
        mask[:, 0] = True
    elif corner == "tr":
        mask[0, :] = True
        mask[:, -1] = True
    elif corner == "bl":
        mask[-1, :] = True
        mask[:, 0] = True
    else:
        mask[-1, :] = True
        mask[:, -1] = True
    return mask


def _make_stair(h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    r = 0
    c = 0
    mask[r, c] = True
    while r < h - 1 or c < w - 1:
        if r == h - 1:
            c += 1
        elif c == w - 1:
            r += 1
        else:
            remaining_down = h - 1 - r
            remaining_right = w - 1 - c
            if rng.random() < remaining_right / (remaining_down + remaining_right):
                c += 1
            else:
                r += 1
        mask[r, c] = True

    if h >= 3 and w >= 3:
        for rr, cc in list(map(tuple, np.argwhere(mask))):
            if rng.random() < 0.35:
                if rr + 1 < h:
                    mask[rr + 1, cc] = True
                if cc + 1 < w:
                    mask[rr, cc + 1] = True
    return mask


def _make_carved(h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    min_area = max(h + w - 1, (h * w + 1) // 2)
    target = rng.randint(min_area, h * w)
    attempts = 0
    while int(mask.sum()) > target and attempts < 200:
        attempts += 1
        r = rng.randrange(h)
        c = rng.randrange(w)
        if not mask[r, c]:
            continue
        trial = mask.copy()
        trial[r, c] = False
        if _spans_bbox(trial) and _is_connected(trial):
            mask = trial
    return mask


def _generate_shape(h: int, w: int, rng: random.Random) -> np.ndarray:
    modes = ["rect", "carved", "stair"]
    if h >= 3 and w >= 3:
        modes.extend(["frame", "plus", "tee", "ell"])
    mode = rng.choice(modes)

    if mode == "rect":
        mask = np.ones((h, w), dtype=bool)
    elif mode == "frame":
        mask = _make_frame(h, w)
    elif mode == "plus":
        mask = _make_plus(h, w)
    elif mode == "tee":
        mask = _make_tee(h, w, rng)
    elif mode == "ell":
        mask = _make_ell(h, w, rng)
    elif mode == "stair":
        mask = _make_stair(h, w, rng)
    else:
        mask = _make_carved(h, w, rng)

    assert _spans_bbox(mask)
    assert _is_connected(mask)
    return mask


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _boxes_separated(r0: int, c0: int, h: int, w: int, placed: list[tuple[int, int, int, int]]) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 < pr0 - BOX_GAP or r0 > pr1 + BOX_GAP or c1 < pc0 - BOX_GAP or c0 > pc1 + BOX_GAP):
            return False
    return True


def _candidate_positions(
    h: int,
    w: int,
    grid_h: int,
    grid_w: int,
    placed: list[tuple[int, int, int, int]],
    mode: str,
) -> list[tuple[int, int]]:
    positions = []
    row_stop = grid_h - EDGE_MARGIN - h
    col_stop = grid_w - EDGE_MARGIN - w
    for r0 in range(EDGE_MARGIN, row_stop + 1):
        for c0 in range(EDGE_MARGIN, col_stop + 1):
            if not _boxes_separated(r0, c0, h, w, placed):
                continue

            r1 = r0 + h - 1
            c1 = c0 + w - 1
            row_overlap = any(_intervals_overlap(r0, r1, pr0, pr1) for pr0, _pc0, pr1, _pc1 in placed)
            col_overlap = any(_intervals_overlap(c0, c1, pc0, pc1) for _pr0, pc0, _pr1, pc1 in placed)

            if mode == "none" and (row_overlap or col_overlap):
                continue
            if mode == "col_overlap" and (not col_overlap or row_overlap):
                continue
            if mode == "row_overlap" and (not row_overlap or col_overlap):
                continue

            positions.append((r0, c0))
    return positions


def _place_objects(
    masks: list[np.ndarray],
    grid_h: int,
    grid_w: int,
    mode: str,
    rng: random.Random,
) -> list[tuple[int, int]]:
    placed_boxes: list[tuple[int, int, int, int]] = []
    placements: list[tuple[int, int]] = []

    order = list(range(len(masks)))
    rng.shuffle(order)
    ordered_masks = [masks[idx] for idx in order]

    for mask in ordered_masks:
        h, w = mask.shape
        local_mode = "none" if not placed_boxes else mode
        candidates = _candidate_positions(h, w, grid_h, grid_w, placed_boxes, local_mode)
        if not candidates:
            raise ValueError("no valid placement")
        r0, c0 = rng.choice(candidates)
        placed_boxes.append((r0, c0, r0 + h - 1, c0 + w - 1))
        placements.append((r0, c0))

    result = [None] * len(masks)
    for src_idx, obj_idx in enumerate(order):
        result[obj_idx] = placements[src_idx]
    return result


def _build_grid(
    grid_h: int,
    grid_w: int,
    colors: list[int],
    masks: list[np.ndarray],
    placements: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for color, mask, (r0, c0) in zip(colors, masks, placements):
        h, w = mask.shape
        sub = grid[r0 : r0 + h, c0 : c0 + w]
        sub[mask] = int(color)
    return grid


def _validate_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if int(np.count_nonzero(grid == BG)) <= int(grid.size // 2):
        return False

    comps = _find_components(grid)
    if len(comps) not in OBJECT_COUNT_OPTIONS:
        return False
    if len({int(comp["color"]) for comp in comps}) != len(comps):
        return False

    h, w = grid.shape
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if r0 <= 0 or c0 <= 0 or r1 >= h - 1 or c1 >= w - 1:
            return False
        if not _is_connected(_component_mask(comp)):
            return False

    out = main(grid)
    if not np.all(out[1:-1, 1:-1] == BG):
        return False
    if int(np.count_nonzero(out != BG)) == 0:
        return False
    return True


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    count = rng.choice(OBJECT_COUNT_OPTIONS)
    mode = "none" if count == 1 else rng.choice(MODE_OPTIONS)

    size_cap = {1: 7, 2: 6, 3: 5}[count]
    max_h = min(size_cap, grid_h - 2 * EDGE_MARGIN - 1)
    max_w = min(size_cap, grid_w - 2 * EDGE_MARGIN - 1)
    if max_h < 2 or max_w < 2:
        return None

    colors = rng.sample(FG_COLORS, count)
    masks = []
    for _ in range(count):
        shape_h = rng.randint(2, max_h)
        shape_w = rng.randint(2, max_w)
        masks.append(_generate_shape(shape_h, shape_w, rng))

    try:
        placements = _place_objects(masks, grid_h, grid_w, mode, rng)
    except ValueError:
        return None

    grid = _build_grid(grid_h, grid_w, colors, masks, placements)
    if not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
