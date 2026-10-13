import random
from collections import Counter

import numpy as np


BG = 7
SIDE_NAMES = ("left", "right", "above", "below")
SHAPE_CHOICES = ((2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3))
RETRY_LIMIT = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            rs = [r0 for r0, _ in coords]
            cs = [c0 for _, c0 in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                }
            )
    return comps


def _is_solid_2x2(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    return len(component["coords"]) == 4 and (r1 - r0 + 1, c1 - c0 + 1) == (2, 2)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape

    comps = _find_components(g, bg)
    assert len(comps) == 2

    anchors = [comp for comp in comps if _is_solid_2x2(comp)]
    assert len(anchors) == 1
    anchor = anchors[0]
    sprite = next(comp for comp in comps if comp is not anchor)
    assert int(sprite["color"]) != int(anchor["color"])

    sr0, sc0, sr1, sc1 = map(int, sprite["bbox"])
    ar0, ac0, ar1, ac1 = map(int, anchor["bbox"])

    candidates: list[tuple[str, int, int]] = []
    if sc1 < ac0 and _intervals_overlap(sr0, sr1, ar0, ar1):
        candidates.append(("left", 0, ac0 - sc1 - 1))
    if sc0 > ac1 and _intervals_overlap(sr0, sr1, ar0, ar1):
        candidates.append(("right", 0, ac1 + 1 - sc0))
    if sr1 < ar0 and _intervals_overlap(sc0, sc1, ac0, ac1):
        candidates.append(("above", ar0 - sr1 - 1, 0))
    if sr0 > ar1 and _intervals_overlap(sc0, sc1, ac0, ac1):
        candidates.append(("below", ar1 + 1 - sr0, 0))
    assert len(candidates) == 1

    side, dr, dc = candidates[0]
    moved: list[tuple[int, int]] = []
    mirrored: list[tuple[int, int]] = []
    for r, c in sprite["coords"]:
        rr = int(r) + int(dr)
        cc = int(c) + int(dc)
        moved.append((rr, cc))
        if side == "left":
            mirrored.append((rr, 2 * ac0 - 1 - cc))
        elif side == "right":
            mirrored.append((rr, 2 * ac1 + 1 - cc))
        elif side == "above":
            mirrored.append((2 * ar0 - 1 - rr, cc))
        else:
            mirrored.append((2 * ar1 + 1 - rr, cc))

    assert all(0 <= r < h and 0 <= c < w for r, c in moved)
    assert all(0 <= r < h and 0 <= c < w for r, c in mirrored)
    assert not (set(moved) & set(mirrored))

    out = np.full_like(g, bg)
    for r, c in moved:
        out[r, c] = int(sprite["color"])
    for r, c in mirrored:
        out[r, c] = int(anchor["color"])
    return out


def _mask_is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _random_sprite_mask(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h, w = rng.choice(SHAPE_CHOICES)
        max_area = min(h * w - 1, 7)
        target = rng.randint(4, max_area)
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                            frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _mask_is_connected(mask):
            continue
        mask = _crop_mask(mask)
        h, w = mask.shape
        if h == 2 and w == 2 and mask.all():
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    raise RuntimeError("failed to generate a non-symmetric sprite")


def _pick_overlap_offset(rng: random.Random, limit: int, anchor_pos: int, sprite_span: int) -> int:
    low = max(0, anchor_pos - sprite_span + 1)
    high = min(limit - sprite_span, anchor_pos + 1)
    assert low <= high
    return rng.randint(low, high)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(10, 18)
    w = rng.randint(10, 18)
    grid = np.full((h, w), BG, dtype=int)

    sprite_color, anchor_color = rng.sample([c for c in range(10) if c != BG], 2)
    side = rng.choice(SIDE_NAMES)
    gap = rng.randint(1, 4)
    mask = _random_sprite_mask(rng)
    sh, sw = map(int, mask.shape)

    if side in ("left", "right"):
        if w < 2 * sw + gap + 2 or h < max(sh, 2):
            return None
        ar0_low = sh - 2 if sh >= 2 else 0
        ar0_high = h - sh
        if ar0_low > ar0_high:
            return None
        ar0 = rng.randint(ar0_low, ar0_high)
        moved_r = _pick_overlap_offset(rng, h, ar0, sh)
        if side == "left":
            ac0_low = sw + gap
            ac0_high = w - sw
            if ac0_low > ac0_high:
                return None
            ac0 = rng.randint(ac0_low, ac0_high)
            moved_c = ac0 - sw
            input_r, input_c = moved_r, moved_c - gap
        else:
            ac0_low = sw
            ac0_high = w - sw - gap - 2
            if ac0_low > ac0_high:
                return None
            ac0 = rng.randint(ac0_low, ac0_high)
            moved_c = ac0 + 2
            input_r, input_c = moved_r, moved_c + gap
    else:
        if h < 2 * sh + gap + 2 or w < max(sw, 2):
            return None
        ac0_low = sw - 2 if sw >= 2 else 0
        ac0_high = w - sw
        if ac0_low > ac0_high:
            return None
        ac0 = rng.randint(ac0_low, ac0_high)
        moved_c = _pick_overlap_offset(rng, w, ac0, sw)
        if side == "above":
            ar0_low = sh + gap
            ar0_high = h - sh
            if ar0_low > ar0_high:
                return None
            ar0 = rng.randint(ar0_low, ar0_high)
            moved_r = ar0 - sh
            input_r, input_c = moved_r - gap, moved_c
        else:
            ar0_low = sh
            ar0_high = h - sh - gap - 2
            if ar0_low > ar0_high:
                return None
            ar0 = rng.randint(ar0_low, ar0_high)
            moved_r = ar0 + 2
            input_r, input_c = moved_r + gap, moved_c

    if not (0 <= ar0 <= h - 2 and 0 <= ac0 <= w - 2):
        return None
    if not (0 <= input_r <= h - sh and 0 <= input_c <= w - sw):
        return None

    anchor_cells = {(ar0 + dr, ac0 + dc) for dr in range(2) for dc in range(2)}
    sprite_cells = {(input_r + r, input_c + c) for r, c in zip(*np.where(mask), strict=False)}
    if anchor_cells & sprite_cells:
        return None

    for r, c in anchor_cells:
        grid[r, c] = anchor_color
    for r, c in sprite_cells:
        grid[r, c] = sprite_color

    comps = _find_components(grid, BG)
    if len(comps) != 2:
        return None
    if sum(_is_solid_2x2(comp) for comp in comps) != 1:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    seed = random.SystemRandom().randrange(1 << 62)
    rng = random.Random(seed)
    side_counts: Counter[str] = Counter()
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        bg = _get_bg(grid)
        comps = _find_components(grid, bg)
        anchor = next(comp for comp in comps if _is_solid_2x2(comp))
        sprite = next(comp for comp in comps if comp is not anchor)
        sr0, sc0, sr1, sc1 = sprite["bbox"]
        ar0, ac0, ar1, ac1 = anchor["bbox"]
        if sc1 < ac0:
            side_counts["left"] += 1
        elif sc0 > ac1:
            side_counts["right"] += 1
        elif sr1 < ar0:
            side_counts["above"] += 1
        else:
            side_counts["below"] += 1
        return grid
    raise RuntimeError("failed to generate a valid input")
