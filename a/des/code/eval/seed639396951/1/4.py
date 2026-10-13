import random
from collections import deque

import numpy as np


BG = 0
GEN_ATTEMPTS = 300
GRID_H_RANGE = (16, 28)
GRID_W_RANGE = (16, 28)
PAIR_COUNT_OPTIONS = (1, 2, 3)
PAIR_PLACE_ATTEMPTS = 120
SMALL_H_RANGE = (2, 5)
SMALL_W_RANGE = (2, 5)
BIG_EXTRA_H_RANGE = (2, 5)
BIG_EXTRA_W_RANGE = (2, 5)
GAP_RANGE = (2, 5)
DIRECTIONS = ("up", "down", "left", "right")
OUTER_MARGIN = 1
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = DIRS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = DIRS4 if connectivity == 4 else DIRS8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _patch_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return patch


def _center_of_bbox(box: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = box
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _infer_direction(
    small_box: tuple[int, int, int, int],
    big_box: tuple[int, int, int, int],
) -> str:
    sr0, sc0, sr1, sc1 = small_box
    br0, bc0, br1, bc1 = big_box
    if sr1 < br0:
        return "up"
    if sr0 > br1:
        return "down"
    if sc1 < bc0:
        return "left"
    if sc0 > bc1:
        return "right"

    (sr, sc) = _center_of_bbox(small_box)
    (br, bc) = _center_of_bbox(big_box)
    if abs(sr - br) >= abs(sc - bc):
        return "up" if sr < br else "down"
    return "left" if sc < bc else "right"


def _project_patch(
    patch: np.ndarray,
    big_box: tuple[int, int, int, int],
    direction: str,
) -> tuple[np.ndarray, int, int]:
    br0, bc0, br1, bc1 = big_box
    big_h = br1 - br0 + 1
    big_w = bc1 - bc0 + 1
    h, w = patch.shape
    assert big_h >= h and big_w >= w

    if direction == "up":
        return np.flipud(patch), br0, bc0 + (big_w - w) // 2
    if direction == "down":
        return np.flipud(patch), br1 - h + 1, bc0 + (big_w - w) // 2
    if direction == "left":
        return np.fliplr(patch), br0 + (big_h - h) // 2, bc0
    if direction == "right":
        return np.fliplr(patch), br0 + (big_h - h) // 2, bc1 - w + 1
    raise ValueError(direction)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color shapes, identify the smaller member of each pair, and stamp
    its silhouette into the larger member as a black cutout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    colors = [int(v) for v in np.unique(g) if int(v) != BG]
    for color in colors:
        comps = _components(g == color, connectivity=4)
        assert len(comps) == 2, "each non-background color must appear in exactly two components"
        comps.sort(key=lambda pts: int(pts.shape[0]))
        small, big = comps
        assert small.shape[0] < big.shape[0], "same-color pair must have different sizes"

        small_box = _bbox(small)
        big_box = _bbox(big)
        direction = _infer_direction(small_box, big_box)
        patch = _patch_from_coords(small)
        projected, dest_r, dest_c = _project_patch(patch, big_box, direction)

        out[small[:, 0], small[:, 1]] = BG
        rows, cols = np.nonzero(projected)
        for rr, cc in zip(rows, cols, strict=False):
            gr = dest_r + int(rr)
            gc = dest_c + int(cc)
            if out[gr, gc] == color:
                out[gr, gc] = BG

    return out


def _is_connected(mask: np.ndarray) -> bool:
    return np.count_nonzero(mask) > 0 and len(_components(mask, connectivity=4)) == 1


def _corner_slice(h: int, w: int, name: str, cut_h: int, cut_w: int) -> tuple[slice, slice]:
    if name == "tl":
        return slice(0, cut_h), slice(0, cut_w)
    if name == "tr":
        return slice(0, cut_h), slice(w - cut_w, w)
    if name == "bl":
        return slice(h - cut_h, h), slice(0, cut_w)
    if name == "br":
        return slice(h - cut_h, h), slice(w - cut_w, w)
    raise ValueError(name)


def _notched_rectangle(
    h: int,
    w: int,
    rng: random.Random,
    protected: np.ndarray | None = None,
    min_area: int = 1,
) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    protected = np.zeros((h, w), dtype=bool) if protected is None else protected.copy()

    corners = ["tl", "tr", "bl", "br"]
    rng.shuffle(corners)
    for name in corners:
        if rng.random() < 0.35:
            continue
        max_cut_h = max(1, min(h - 1, h // 2))
        max_cut_w = max(1, min(w - 1, w // 2))
        for _ in range(8):
            cut_h = rng.randint(1, max_cut_h)
            cut_w = rng.randint(1, max_cut_w)
            rr, cc = _corner_slice(h, w, name, cut_h, cut_w)
            if protected[rr, cc].any():
                continue
            trial = mask.copy()
            trial[rr, cc] = False
            if np.count_nonzero(trial) < min_area:
                continue
            if _is_connected(trial):
                mask = trial
                break

    return mask


def _sample_small_shape(rng: random.Random) -> np.ndarray:
    for _ in range(60):
        h = rng.randint(*SMALL_H_RANGE)
        w = rng.randint(*SMALL_W_RANGE)
        mask = _notched_rectangle(h, w, rng, min_area=max(3, (h * w) // 2))
        area = int(mask.sum())
        if area >= 3 and _is_connected(mask):
            return mask
    raise RuntimeError("failed to sample small shape")


def _sample_big_shape(
    small_mask: np.ndarray,
    direction: str,
    rng: random.Random,
) -> np.ndarray:
    sh, sw = small_mask.shape
    flip = np.flipud(small_mask) if direction in ("up", "down") else np.fliplr(small_mask)

    for _ in range(80):
        bh = sh + rng.randint(*BIG_EXTRA_H_RANGE)
        bw = sw + rng.randint(*BIG_EXTRA_W_RANGE)
        protected = np.zeros((bh, bw), dtype=bool)

        if direction == "up":
            r0 = 0
            c0 = (bw - sw) // 2
        elif direction == "down":
            r0 = bh - sh
            c0 = (bw - sw) // 2
        elif direction == "left":
            r0 = (bh - sh) // 2
            c0 = 0
        else:
            r0 = (bh - sh) // 2
            c0 = bw - sw

        protected[r0 : r0 + sh, c0 : c0 + sw] = flip
        min_area = int(small_mask.sum()) + max(4, int(small_mask.sum()) // 2)
        mask = _notched_rectangle(bh, bw, rng, protected=protected, min_area=min_area)
        if mask.sum() > small_mask.sum() and protected[mask == 0].sum() == 0 and _is_connected(mask):
            return mask

    raise RuntimeError("failed to sample big shape")


def _can_place(mask: np.ndarray, top: int, left: int, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    rows, cols = np.nonzero(mask)
    for rr, cc in zip(rows, cols, strict=False):
        gr = top + int(rr)
        gc = left + int(cc)
        if not (0 <= gr < h and 0 <= gc < w):
            return False
        if occupied[gr, gc]:
            return False
        for dr, dc in DIRS8:
            nr, nc = gr + dr, gc + dc
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return False
    return True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.nonzero(mask)
    grid[top + rows, left + cols] = color


def _centered_offset(outer: int, inner: int, rng: random.Random) -> int:
    slack = outer - inner
    mid = slack // 2
    if slack <= 1:
        return mid
    jitter = rng.choice([-1, 0, 1])
    return max(0, min(slack, mid + jitter))


def _validate_generated(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or not np.any(grid == BG):
        return False

    colors = [int(v) for v in np.unique(grid) if int(v) != BG]
    if not colors:
        return False

    for color in colors:
        comps = _components(grid == color, connectivity=4)
        if len(comps) != 2:
            return False
        comps.sort(key=lambda pts: int(pts.shape[0]))
        small, big = comps
        if small.shape[0] >= big.shape[0]:
            return False
        direction = _infer_direction(_bbox(small), _bbox(big))
        projected, dest_r, dest_c = _project_patch(_patch_from_coords(small), _bbox(big), direction)
        rows, cols = np.nonzero(projected)
        for rr, cc in zip(rows, cols, strict=False):
            if grid[dest_r + int(rr), dest_c + int(cc)] != color:
                return False

    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape and np.any(out == BG)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    pair_count = rng.choice(PAIR_COUNT_OPTIONS)

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    colors = rng.sample(range(1, 10), pair_count)

    for color in colors:
        placed = False
        for _ in range(PAIR_PLACE_ATTEMPTS):
            direction = rng.choice(DIRECTIONS)
            try:
                small_mask = _sample_small_shape(rng)
                big_mask = _sample_big_shape(small_mask, direction, rng)
            except RuntimeError:
                continue
            sh, sw = small_mask.shape
            bh, bw = big_mask.shape
            gap = rng.randint(*GAP_RANGE)

            if direction in ("up", "down"):
                offset_c = _centered_offset(bw, sw, rng)
                total_h = sh + gap + bh
                if total_h + 2 * OUTER_MARGIN > h or bw + 2 * OUTER_MARGIN > w:
                    continue
                pair_top = rng.randint(OUTER_MARGIN, h - OUTER_MARGIN - total_h)
                pair_left = rng.randint(OUTER_MARGIN, w - OUTER_MARGIN - bw)
                if direction == "up":
                    small_top = pair_top
                    big_top = pair_top + sh + gap
                else:
                    big_top = pair_top
                    small_top = pair_top + bh + gap
                small_left = pair_left + offset_c
                big_left = pair_left
            else:
                offset_r = _centered_offset(bh, sh, rng)
                total_w = sw + gap + bw
                if bh + 2 * OUTER_MARGIN > h or total_w + 2 * OUTER_MARGIN > w:
                    continue
                pair_top = rng.randint(OUTER_MARGIN, h - OUTER_MARGIN - bh)
                pair_left = rng.randint(OUTER_MARGIN, w - OUTER_MARGIN - total_w)
                if direction == "left":
                    small_left = pair_left
                    big_left = pair_left + sw + gap
                else:
                    big_left = pair_left
                    small_left = pair_left + bw + gap
                small_top = pair_top + offset_r
                big_top = pair_top

            if not _can_place(small_mask, small_top, small_left, occupied):
                continue
            if not _can_place(big_mask, big_top, big_left, occupied):
                continue

            _paint_mask(grid, small_mask, small_top, small_left, color)
            _paint_mask(grid, big_mask, big_top, big_left, color)
            rows, cols = np.nonzero(grid == color)
            occupied[rows, cols] = True
            placed = True
            break

        if not placed:
            return None

    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
