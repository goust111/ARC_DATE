import itertools
import random

import numpy as np


AZURE = 8
ANCHOR_SIZE = 2
SIDES = ("left", "right", "above", "below")
BRANCHES = ("vertical", "horizontal", "asymmetric")
NUM_COLORS = 10
RETRY_LIMIT = 400

_NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _bbox(coords):
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(grid: np.ndarray, bg: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) == bg:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append({"coords": coords, "bbox": _bbox(coords)})
    return comps


def _is_solid_2x2(comp, grid: np.ndarray) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return False
    if len(comp["coords"]) != 4:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(np.all(patch == int(patch[0, 0])))


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _mask_from_component(comp) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _symmetry_branch(mask: np.ndarray) -> str:
    has_vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    has_horizontal = bool(np.array_equal(mask, np.flipud(mask)))
    if has_vertical:
        return "vertical"
    if has_horizontal:
        return "horizontal"
    return "asymmetric"


def _copy_mask(mask: np.ndarray) -> np.ndarray:
    branch = _symmetry_branch(mask)
    if branch == "vertical":
        return np.fliplr(mask).copy()
    if branch == "horizontal":
        return np.flipud(mask).copy()
    return np.rot90(mask, 2).copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a sprite to a square anchor and draw a symmetry-selected mirror copy beyond the anchor edge.

    Input structure:
    - The background is Azure.
    - There is one 8-connected non-background sprite that is not a solid 2x2 square.
    - There is one solid 2x2 square anchor in a different non-background color.
    - The sprite and anchor are separated along exactly one cardinal direction.

    Rule:
    - Move the sprite straight toward the anchor until its bounding box touches the anchor's near edge.
    - If the sprite has vertical symmetry, draw a horizontally flipped copy across that edge in the anchor color.
    - If the sprite has horizontal symmetry, draw a vertically flipped copy across that edge in the anchor color.
    - If the sprite is asymmetric, draw a 180-degree rotated copy across that edge in the anchor color.
    - Clear the original sprite and anchor positions; keep only the moved sprite and its anchor-colored copy.

    Key insight:
    - The square anchor supplies the contact edge, but the sprite's own symmetry decides which mirror operation is used for the copied shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    bg = AZURE

    comps = _components(g, bg)
    assert len(comps) == 2

    anchors = [comp for comp in comps if _is_solid_2x2(comp, g)]
    assert len(anchors) == 1
    anchor = anchors[0]
    sprite = comps[0] if comps[1] is anchor else comps[1]

    sr0, sc0, sr1, sc1 = map(int, sprite["bbox"])
    ar0, ac0, ar1, ac1 = map(int, anchor["bbox"])
    anchor_color = int(g[ar0, ac0])

    candidates = []
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
    mask = _mask_from_component(sprite)
    sh, sw = map(int, mask.shape)
    moved_r0, moved_c0 = sr0 + dr, sc0 + dc

    if side == "left":
        copy_r0, copy_c0 = moved_r0, ac0
    elif side == "right":
        copy_r0, copy_c0 = moved_r0, ac1 - sw + 1
    elif side == "above":
        copy_r0, copy_c0 = ar0, moved_c0
    else:
        copy_r0, copy_c0 = ar1 - sh + 1, moved_c0

    out = np.full_like(g, bg)
    moved_cells = set()
    for r, c in sprite["coords"]:
        rr, cc = int(r) + dr, int(c) + dc
        assert 0 <= rr < h and 0 <= cc < w
        out[rr, cc] = int(g[r, c])
        moved_cells.add((rr, cc))

    mirrored = _copy_mask(mask)
    for rr, cc in np.argwhere(mirrored):
        r, c = copy_r0 + int(rr), copy_c0 + int(cc)
        assert 0 <= r < h and 0 <= c < w
        assert (r, c) not in moved_cells
        out[r, c] = anchor_color

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not mask[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _strict_category(mask: np.ndarray):
    has_vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    has_horizontal = bool(np.array_equal(mask, np.flipud(mask)))
    if has_vertical and not has_horizontal:
        return "vertical"
    if has_horizontal and not has_vertical:
        return "horizontal"
    if not has_vertical and not has_horizontal:
        return "asymmetric"
    return None


def _build_mask_bank():
    bank = {branch: [] for branch in BRANCHES}
    dims = ((2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4))
    for mh, mw in dims:
        cells_count = mh * mw
        for area in range(4, min(9, cells_count) + 1):
            for cells in itertools.combinations(range(cells_count), area):
                mask = np.zeros((mh, mw), dtype=bool)
                for cell in cells:
                    mask[cell // mw, cell % mw] = True
                if not _full_bbox(mask):
                    continue
                if not _mask_connected(mask):
                    continue
                category = _strict_category(mask)
                if category is not None:
                    bank[category].append(mask.copy())
    assert all(bank[branch] for branch in BRANCHES)
    return bank


_MASK_BANK = _build_mask_bank()


def _random_mask(branch: str) -> np.ndarray:
    mask = random.choice(_MASK_BANK[branch]).copy()
    if branch == "vertical":
        if random.randrange(2):
            mask = np.flipud(mask)
    elif branch == "horizontal":
        if random.randrange(2):
            mask = np.fliplr(mask)
    else:
        for _ in range(20):
            cand = np.rot90(mask, random.randrange(4))
            if random.randrange(2):
                cand = np.fliplr(cand)
            cand = cand.copy()
            if _strict_category(cand) == "asymmetric":
                mask = cand
                break
    assert _strict_category(mask) == branch
    return mask.copy()


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    for rr, cc in np.argwhere(mask):
        grid[r0 + int(rr), c0 + int(cc)] = int(color)


def _layout(mask: np.ndarray, side: str, gap: int, h: int, w: int):
    sh, sw = map(int, mask.shape)
    for _ in range(120):
        ar = random.randint(0, h - ANCHOR_SIZE)
        ac = random.randint(0, w - ANCHOR_SIZE)

        if side == "left":
            face_rows = np.where(mask[:, -1])[0]
            moved_r = ar + random.randrange(ANCHOR_SIZE) - int(random.choice(face_rows))
            moved_c = ac - sw
            input_r, input_c = moved_r, moved_c - gap
            copy_r, copy_c = moved_r, ac
        elif side == "right":
            face_rows = np.where(mask[:, 0])[0]
            moved_r = ar + random.randrange(ANCHOR_SIZE) - int(random.choice(face_rows))
            moved_c = ac + ANCHOR_SIZE
            input_r, input_c = moved_r, moved_c + gap
            copy_r, copy_c = moved_r, ac + ANCHOR_SIZE - sw
        elif side == "above":
            face_cols = np.where(mask[-1, :])[0]
            moved_r = ar - sh
            moved_c = ac + random.randrange(ANCHOR_SIZE) - int(random.choice(face_cols))
            input_r, input_c = moved_r - gap, moved_c
            copy_r, copy_c = ar, moved_c
        else:
            face_cols = np.where(mask[0, :])[0]
            moved_r = ar + ANCHOR_SIZE
            moved_c = ac + random.randrange(ANCHOR_SIZE) - int(random.choice(face_cols))
            input_r, input_c = moved_r + gap, moved_c
            copy_r, copy_c = ar + ANCHOR_SIZE - sh, moved_c

        boxes = (
            (input_r, input_c, sh, sw),
            (moved_r, moved_c, sh, sw),
            (copy_r, copy_c, sh, sw),
            (ar, ac, ANCHOR_SIZE, ANCHOR_SIZE),
        )
        if all(0 <= r and 0 <= c and r + bh <= h and c + bw <= w for r, c, bh, bw in boxes):
            return ar, ac, input_r, input_c
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(10, 24)
    w = random.randint(10, 24)
    branch = random.choice(BRANCHES)
    side = random.choice(SIDES)
    mask = _random_mask(branch)
    gap = random.randint(2, 5)

    layout = _layout(mask, side, gap, h, w)
    if layout is None:
        return None
    ar, ac, input_r, input_c = layout

    colors = [color for color in range(NUM_COLORS) if color != AZURE]
    sprite_color, anchor_color = random.sample(colors, 2)

    grid = np.full((h, w), AZURE, dtype=int)
    grid[ar : ar + ANCHOR_SIZE, ac : ac + ANCHOR_SIZE] = anchor_color
    _paint_mask(grid, mask, input_r, input_c, sprite_color)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if len(_components(grid, AZURE)) != 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for seed2047415201/4/1")
