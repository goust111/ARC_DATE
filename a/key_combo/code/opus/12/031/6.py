import numpy as np
from collections import deque


BLACK = 0
BLUE = 1
RED = 2

_RNG = np.random.default_rng()
_GEN_ATTEMPTS = 400
_TEMP_SIZE = 15
_MAX_INPUT_SIDE = 13


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _count_enclosed_black_regions(grid: np.ndarray) -> int:
    h, w = grid.shape
    black = grid == BLACK
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not black[r, c] or seen[r, c]:
                continue
            touches_edge = False
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_to_grid(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, RED, BLACK).astype(int)


def _hole_count_from_mask(mask: np.ndarray) -> int:
    return _count_enclosed_black_regions(_mask_to_grid(mask))


def _rotate_cw(grid: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(grid, -int(turns % 4))


def _stack_with_blue_frame(tile: np.ndarray) -> np.ndarray:
    h, w = tile.shape
    out = np.full((2 * h + 3, w + 2), BLUE, dtype=int)
    out[1 : 1 + h, 1 : 1 + w] = tile
    out[h + 2 : h + 2 + h, 1 : 1 + w] = tile
    for row in (0, h + 1, 2 * h + 2):
        out[row, 0] = BLACK
        out[row, -1] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the enclosed black regions inside the red shape, rotate the whole grid by that amount, then stack two framed copies vertically.

    Input structure:
    - The input is a small square grid with black background and no blue pixels.
    - A single 4-connected red shape occupies the central area.
    - The red shape may contain 0 to 3 enclosed black 4-connected regions that do not touch the grid boundary.

    Rule:
    - Count all black 4-connected components that are not connected to any edge cell; call that count N.
    - Rotate the entire input grid clockwise by N quarter-turns.
    - Copy the rotated grid twice and stack the two copies vertically.
    - Surround the stack with a blue one-pixel frame and insert a blue separator row between the copies, but keep the two end pixels of each horizontal frame/separator row black.

    Key insight:
    - The red shape encodes its own rotation angle through topology: the number of enclosed black holes determines how many 90-degree clockwise turns are applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))).issubset({BLACK, RED})
    assert BLUE not in set(map(int, g.ravel()))
    red_mask = g == RED
    assert red_mask.any()
    assert _component_count(red_mask) == 1

    turns = _count_enclosed_black_regions(g)
    assert 0 <= turns <= 3
    return _stack_with_blue_frame(_rotate_cw(g, turns))


def _ring_module(hole_h: int, hole_w: int) -> np.ndarray:
    module = np.ones((hole_h + 2, hole_w + 2), dtype=bool)
    module[1:-1, 1:-1] = False
    return module


def _open_ring_module(hole_h: int, hole_w: int, side: str) -> np.ndarray:
    module = _ring_module(hole_h, hole_w)
    if side == "top":
        module[0, 1 + hole_w // 2] = False
    elif side == "bottom":
        module[-1, 1 + hole_w // 2] = False
    elif side == "left":
        module[1 + hole_h // 2, 0] = False
    elif side == "right":
        module[1 + hole_h // 2, -1] = False
    else:
        raise ValueError("invalid side")
    return module


def _connect_boxes(mask: np.ndarray, a, b) -> None:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    if ar0 == br0 and ar1 == br1:
        row = (max(ar0, br0) + min(ar1, br1)) // 2
        c0 = min(ac1, bc1)
        c1 = max(ac0, bc0)
        mask[row, c0 : c1 + 1] = True
        return
    if ac0 == bc0 and ac1 == bc1:
        col = (max(ac0, bc0) + min(ac1, bc1)) // 2
        r0 = min(ar1, br1)
        r1 = max(ar0, br0)
        mask[r0 : r1 + 1, col] = True
        return

    row = (max(ar0, br0) + min(ar1, br1)) // 2
    col = (max(ac0, bc0) + min(ac1, bc1)) // 2
    mid_r = (ar0 + ar1) // 2
    mid_c = (bc0 + bc1) // 2
    mask[min(mid_r, row) : max(mid_r, row) + 1, (ac0 + ac1) // 2] = True
    mask[row, min((ac0 + ac1) // 2, mid_c) : max((ac0 + ac1) // 2, mid_c) + 1] = True
    mask[min(row, (br0 + br1) // 2) : max(row, (br0 + br1) // 2) + 1, mid_c] = True


def _slot_templates(hole_count: int):
    if hole_count == 1:
        return [[(0, 0)]]
    if hole_count == 2:
        return [[(0, 0), (0, 1)], [(0, 0), (1, 0)]]
    return [
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
    ]


def _build_shape(hole_count: int) -> np.ndarray | None:
    mask = np.zeros((_TEMP_SIZE, _TEMP_SIZE), dtype=bool)

    if hole_count == 0:
        hole_h = int(_RNG.integers(1, 3))
        hole_w = int(_RNG.integers(1, 3))
        module = _open_ring_module(hole_h, hole_w, _RNG.choice(["top", "bottom", "left", "right"]).item())
        r0 = (_TEMP_SIZE - module.shape[0]) // 2
        c0 = (_TEMP_SIZE - module.shape[1]) // 2
        mask[r0 : r0 + module.shape[0], c0 : c0 + module.shape[1]] = module
        return mask

    templates = _slot_templates(hole_count)
    slots = templates[int(_RNG.integers(0, len(templates)))]
    boxes = []
    for sr, sc in slots:
        hole_h = int(_RNG.integers(1, 3))
        hole_w = int(_RNG.integers(1, 3))
        module = _ring_module(hole_h, hole_w)
        r0 = 2 + 5 * sr
        c0 = 2 + 5 * sc
        h, w = module.shape
        if mask[r0 : r0 + h, c0 : c0 + w].any():
            return None
        mask[r0 : r0 + h, c0 : c0 + w] = module
        boxes.append((r0, c0, r0 + h - 1, c0 + w - 1))

    for i, (sr, sc) in enumerate(slots):
        for j in range(i + 1, len(slots)):
            tr, tc = slots[j]
            if abs(sr - tr) + abs(sc - tc) == 1:
                _connect_boxes(mask, boxes[i], boxes[j])
    return mask


def _add_irregularity(mask: np.ndarray, hole_count: int) -> np.ndarray:
    ranges = {0: (15, 26), 1: (16, 27), 2: (20, 29), 3: (24, 30)}
    target_area = int(_RNG.integers(ranges[hole_count][0], ranges[hole_count][1] + 1))
    current = mask.copy()

    for _ in range(300):
        area = int(current.sum())
        if area >= target_area:
            break
        candidates = []
        for r, c in np.argwhere(current):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < _TEMP_SIZE and 0 <= nc < _TEMP_SIZE and not current[nr, nc]:
                    candidates.append((r, c, dr, dc))
        if not candidates:
            break
        _RNG.shuffle(candidates)
        accepted = False
        for r, c, dr, dc in candidates:
            proposal = current.copy()
            steps = int(_RNG.integers(1, 3))
            rr, cc = r, c
            changed = 0
            for _ in range(steps):
                rr += dr
                cc += dc
                if not (0 <= rr < _TEMP_SIZE and 0 <= cc < _TEMP_SIZE) or proposal[rr, cc]:
                    break
                proposal[rr, cc] = True
                changed += 1
            if changed == 0:
                continue
            if _component_count(proposal) != 1:
                continue
            if _hole_count_from_mask(proposal) != hole_count:
                continue
            current = proposal
            accepted = True
            break
        if not accepted:
            break

    for _ in range(60):
        coords = np.argwhere(current)
        _RNG.shuffle(coords)
        changed = False
        for r, c in coords:
            if int(current.sum()) <= ranges[hole_count][0]:
                break
            proposal = current.copy()
            proposal[r, c] = False
            if not proposal.any():
                continue
            if _component_count(proposal) != 1:
                continue
            if _hole_count_from_mask(proposal) != hole_count:
                continue
            current = proposal
            changed = True
            break
        if not changed:
            break

    return current


def _embed(mask: np.ndarray) -> np.ndarray | None:
    patch = _crop(mask)
    for _ in range(int(_RNG.integers(1, 5))):
        patch = np.rot90(patch, int(_RNG.integers(0, 4)))
        if bool(_RNG.integers(0, 2)):
            patch = np.fliplr(patch)
    ph, pw = patch.shape
    side_min = max(ph, pw) + 2
    side_max = min(_MAX_INPUT_SIDE, max(ph, pw) + 4)
    if side_min > side_max:
        return None
    side = int(_RNG.integers(side_min, side_max + 1))
    grid = np.zeros((side, side), dtype=int)
    top = int(_RNG.integers(1, side - ph))
    left = int(_RNG.integers(1, side - pw))
    grid[top : top + ph, left : left + pw][patch] = RED
    return grid


def _valid_input(grid: np.ndarray, hole_count: int) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or max(g.shape) > _MAX_INPUT_SIDE:
        return False
    if BLUE in set(map(int, g.ravel())):
        return False
    if not set(map(int, np.unique(g))).issubset({BLACK, RED}):
        return False
    red_mask = g == RED
    if not red_mask.any():
        return False
    if _component_count(red_mask) != 1:
        return False
    if np.any(red_mask[0, :]) or np.any(red_mask[-1, :]) or np.any(red_mask[:, 0]) or np.any(red_mask[:, -1]):
        return False
    if _count_enclosed_black_regions(g) != hole_count:
        return False
    red_area = int(red_mask.sum())
    if not (15 <= red_area <= 30):
        return False
    for turns in (1, 2, 3):
        if np.array_equal(g, _rotate_cw(g, turns)):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        hole_count = int(_RNG.integers(0, 4))
        mask = _build_shape(hole_count)
        if mask is None:
            continue
        mask = _add_irregularity(mask, hole_count)
        grid = _embed(mask)
        if grid is None:
            continue
        if _valid_input(grid, hole_count):
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
