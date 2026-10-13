import random

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
GREY = 5

GRID_MIN = 12
GRID_MAX = 24
SHAPE_MIN = 2
SHAPE_MAX = 6
MAX_OBJECTS_PER_CLASS = 2
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _touch_count(cells: list[tuple[int, int]], h: int, w: int) -> int:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return int(min(rows) == 0) + int(max(rows) == h - 1) + int(min(cols) == 0) + int(max(cols) == w - 1)


def _local_mask_is_valid(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    if len(_components(mask)) != 1:
        return False
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _solid_mask(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    if h >= 3 and w >= 3:
        mask[1:-1, 1:-1] = False
    return mask


def _l_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    arm_h = random.randint(1, min(2, h))
    arm_w = random.randint(1, min(2, w))
    orient = random.choice(("tl", "tr", "bl", "br"))
    if orient == "tl":
        mask[:arm_h, :] = True
        mask[:, :arm_w] = True
    elif orient == "tr":
        mask[:arm_h, :] = True
        mask[:, w - arm_w :] = True
    elif orient == "bl":
        mask[h - arm_h :, :] = True
        mask[:, :arm_w] = True
    else:
        mask[h - arm_h :, :] = True
        mask[:, w - arm_w :] = True
    return mask


def _t_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    bar = random.randint(1, min(2, max(h, w)))
    stem = random.randint(1, min(2, min(h, w)))
    orient = random.choice(("top", "bottom", "left", "right"))
    if orient == "top":
        mask[: min(bar, h), :] = True
        c0 = random.randint(0, w - stem)
        mask[:, c0 : c0 + stem] = True
    elif orient == "bottom":
        mask[h - min(bar, h) :, :] = True
        c0 = random.randint(0, w - stem)
        mask[:, c0 : c0 + stem] = True
    elif orient == "left":
        mask[:, : min(bar, w)] = True
        r0 = random.randint(0, h - stem)
        mask[r0 : r0 + stem, :] = True
    else:
        mask[:, w - min(bar, w) :] = True
        r0 = random.randint(0, h - stem)
        mask[r0 : r0 + stem, :] = True
    return mask


def _plus_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[random.randrange(h), :] = True
    mask[:, random.randrange(w)] = True
    return mask


def _h_mask(h: int, w: int) -> np.ndarray:
    if w < 3:
        return _solid_mask(h, w)
    mask = np.zeros((h, w), dtype=bool)
    left_w = random.randint(1, min(2, w - 1))
    right_w = random.randint(1, min(2, w - left_w))
    mask[:, :left_w] = True
    mask[:, w - right_w :] = True
    bridge_h = random.randint(1, min(2, h))
    r0 = random.randint(0, h - bridge_h)
    mask[r0 : r0 + bridge_h, :] = True
    return mask


def _prune_mask(mask: np.ndarray) -> np.ndarray:
    pruned = mask.copy()
    coords = [(r, c) for r in range(1, mask.shape[0] - 1) for c in range(1, mask.shape[1] - 1) if pruned[r, c]]
    random.shuffle(coords)
    target = random.randint(0, len(coords) // 2)
    removed = 0
    for r, c in coords:
        if removed >= target:
            break
        trial = pruned.copy()
        trial[r, c] = False
        if _local_mask_is_valid(trial):
            pruned = trial
            removed += 1
    return pruned


def _make_mask(h: int, w: int) -> np.ndarray:
    builders = [_solid_mask, _frame_mask, _l_mask, _t_mask, _plus_mask, _h_mask]
    random.shuffle(builders)
    for build in builders:
        mask = _prune_mask(build(h, w))
        if _local_mask_is_valid(mask):
            return mask
    return np.ones((h, w), dtype=bool)


def _candidate_positions(h: int, w: int, box_h: int, box_w: int, edge_count: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []

    if edge_count == 0:
        if box_h > h - 2 or box_w > w - 2:
            return positions
        for top in range(1, h - box_h):
            for left in range(1, w - box_w):
                positions.append((top, left))
        return positions

    if edge_count == 1:
        if box_h <= h - 1 and box_w <= w - 2:
            for left in range(1, w - box_w):
                positions.append((0, left))
                positions.append((h - box_h, left))
        if box_w <= w - 1 and box_h <= h - 2:
            for top in range(1, h - box_h):
                positions.append((top, 0))
                positions.append((top, w - box_w))
        return positions

    if box_h <= h - 1 and box_w <= w - 1:
        positions.extend(
            [
                (0, 0),
                (0, w - box_w),
                (h - box_h, 0),
                (h - box_h, w - box_w),
            ]
        )
    return positions


def _place_object(grid: np.ndarray, blocked: np.ndarray, edge_count: int) -> bool:
    h, w = grid.shape
    max_h = min(SHAPE_MAX, h - (2 if edge_count == 0 else 1))
    max_w = min(SHAPE_MAX, w - (2 if edge_count == 0 else 1))
    if max_h < SHAPE_MIN or max_w < SHAPE_MIN:
        return False

    for _ in range(120):
        box_h = random.randint(SHAPE_MIN, max_h)
        box_w = random.randint(SHAPE_MIN, max_w)
        mask = _make_mask(box_h, box_w)
        positions = _candidate_positions(h, w, box_h, box_w, edge_count)
        random.shuffle(positions)
        for top, left in positions:
            if blocked[top : top + box_h, left : left + box_w].any():
                continue
            window = grid[top : top + box_h, left : left + box_w]
            window[mask] = GREY
            blocked[max(0, top - 1) : min(h, top + box_h + 1), max(0, left - 1) : min(w, left + box_w + 1)] = True
            return True
    return False


def _component_touch_histogram(grid: np.ndarray) -> dict[int, int]:
    h, w = grid.shape
    hist = {0: 0, 1: 0, 2: 0}
    for cells in _components(grid == GREY):
        hist[min(_touch_count(cells, h, w), 2)] += 1
    return hist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each object by how many grid edges its bounding box touches.

    Input structure:
    - Background is Black.
    - The foreground consists of one or more 4-connected single-color objects.
    - In the intended puzzle all foreground objects start as Grey, with some fully
      inside the canvas and others aligned to one edge or a corner.

    Rule:
    - Find every 4-connected object of a single color and compute how many grid
      edges are touched by that object's bounding box (top, bottom, left, right).
    - If the bbox touches 0 edges, keep the object in its original color.
    - If the bbox touches exactly 1 edge, recolor the whole object Green.
    - If the bbox touches 2 or more edges, recolor the whole object Yellow.
    - Leave the Black background unchanged.

    Key insight:
    - The objects are distinguished only by their spatial relationship to the
      outer border; the bbox edge-contact count is the decisive feature.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    out = g.copy()

    for color in np.unique(g):
        if int(color) == BLACK:
            continue
        for cells in _components(g == int(color)):
            touches = _touch_count(cells, h, w)
            new_color = int(color) if touches == 0 else GREEN if touches == 1 else YELLOW
            for r, c in cells:
                out[r, c] = new_color
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)
        grid = np.full((h, w), BLACK, dtype=int)
        blocked = np.zeros((h, w), dtype=bool)

        counts = {kind: random.randint(1, MAX_OBJECTS_PER_CLASS) for kind in (0, 1, 2)}
        order = [2] * counts[2] + [1] * counts[1] + [0] * counts[0]
        random.shuffle(order)
        order.sort(reverse=True)

        ok = True
        for edge_count in order:
            if not _place_object(grid, blocked, edge_count):
                ok = False
                break
        if not ok:
            continue

        hist = _component_touch_histogram(grid)
        if any(hist[k] < 1 for k in (0, 1, 2)):
            continue

        out = main(grid)
        if not ((out == GREY).any() and (out == GREEN).any() and (out == YELLOW).any()):
            continue
        if len(_components(grid == GREY)) != sum(hist.values()):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
