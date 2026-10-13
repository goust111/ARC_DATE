from collections import Counter, deque
import random

import numpy as np


RED = 2
BLUE = 1
GREY = 5

MAX_GENERATE_ATTEMPTS = 400
OBJECT_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 300

LOCAL_CANVAS = 17
BASE_SIZE_RANGE = (5, 9)
ARM_COUNT_RANGE = (0, 3)
ARM_EXTENSION_RANGE = (1, 4)
GRID_SIZE_RANGE = (16, 30)
OBJECT_COUNT_RANGE = (1, 3)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _find_components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    neighbor_fn = _neighbors4 if connectivity == 4 else _neighbors8

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in neighbor_fn(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _flood_outside(blocked: np.ndarray) -> np.ndarray:
    h, w = blocked.shape
    seen = np.zeros_like(blocked, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _distance_layers(region_mask: np.ndarray, boundary_mask: np.ndarray) -> np.ndarray:
    h, w = region_mask.shape
    dist = np.zeros((h, w), dtype=int)
    q: deque[tuple[int, int]] = deque()

    for r, c in np.argwhere(region_mask):
        rr = int(r)
        cc = int(c)
        if any(
            0 <= nr < h and 0 <= nc < w and boundary_mask[nr, nc]
            for nr, nc in _neighbors4(rr, cc)
        ):
            dist[rr, cc] = 1
            q.append((rr, cc))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and region_mask[nr, nc] and dist[nr, nc] == 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))

    assert np.all(dist[region_mask] > 0), "region must be reachable from its boundary"
    return dist


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _outline_from_region(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    outline = np.zeros_like(region, dtype=bool)
    for r, c in np.argwhere(region):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not region[nr, nc]:
                outline[nr, nc] = True
    return outline


def _seed_depth(color: int) -> int:
    if color == RED:
        return 1
    if color == BLUE:
        return 2
    raise AssertionError(f"unexpected seed color: {color}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a seed-colored contracted copy of each closed outline inside its own region.

    Input structure:
    - The background is the dominant color.
    - One non-seed color forms one-pixel closed outlines around background interiors.
    - Each enclosed region contains exactly one single-pixel seed, colored red or blue.

    Rule:
    - Keep the original outline unchanged.
    - For every enclosed region, erase its seed back to background.
    - Compute the interior cells' 4-neighbor distance from the surrounding outline.
    - If the seed is red, paint the cells at depth 1; if the seed is blue, paint the
      cells at depth 2, leaving narrow parts blank where that depth does not exist.

    Key insight:
    - The seed color does not mark a destination point; it selects how many inward
      boundary layers to skip before tracing the next inner contour.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    non_bg = {int(v) for v in np.unique(g) if int(v) != bg}
    outline_candidates = [color for color in non_bg if color not in {RED, BLUE}]
    assert len(outline_candidates) == 1, "expected a single outline color"
    outline_color = outline_candidates[0]

    outline_mask = g == outline_color
    outside = _flood_outside(outline_mask)
    inside_mask = (~outline_mask) & (~outside)
    regions = _find_components(inside_mask, connectivity=4)
    assert regions, "expected at least one enclosed region"

    out = g.copy()
    for region in regions:
        seed_cells = [(r, c) for r, c in region if int(g[r, c]) in (RED, BLUE)]
        assert len(seed_cells) == 1, "each enclosed region must contain exactly one seed"
        sr, sc = seed_cells[0]
        seed_color = int(g[sr, sc])
        depth = _seed_depth(seed_color)

        region_mask = np.zeros_like(outline_mask, dtype=bool)
        for r, c in region:
            region_mask[r, c] = True
        dist = _distance_layers(region_mask, outline_mask)

        paint_mask = region_mask & (dist == depth)
        out[paint_mask] = seed_color
        out[sr, sc] = bg

    return out


def _random_bg() -> int:
    return random.choice([0, 3, 4, 6, 7, 8, 9])


def _attach_arm(region: np.ndarray) -> None:
    r0, c0, r1, c1 = _bbox(region)
    h, w = region.shape
    sides = []
    if r0 > 0:
        sides.append("top")
    if r1 < h - 1:
        sides.append("bottom")
    if c0 > 0:
        sides.append("left")
    if c1 < w - 1:
        sides.append("right")
    if not sides:
        return

    side = random.choice(sides)
    if side in ("top", "bottom"):
        width = c1 - c0 + 1
        span = random.randint(2, width)
        cs = random.randint(c0, c1 - span + 1)
        ext = random.randint(*ARM_EXTENSION_RANGE)
        overlap = random.randint(1, min(2, r1 - r0 + 1))
        if side == "top":
            rs = max(0, r0 - ext)
            re = r0 + overlap - 1
        else:
            rs = r1 - overlap + 1
            re = min(h - 1, r1 + ext)
        region[rs : re + 1, cs : cs + span] = True
    else:
        height = r1 - r0 + 1
        span = random.randint(2, height)
        rs = random.randint(r0, r1 - span + 1)
        ext = random.randint(*ARM_EXTENSION_RANGE)
        overlap = random.randint(1, min(2, c1 - c0 + 1))
        if side == "left":
            cs = max(0, c0 - ext)
            ce = c0 + overlap - 1
        else:
            cs = c1 - overlap + 1
            ce = min(w - 1, c1 + ext)
        region[rs : rs + span, cs : ce + 1] = True


def _build_region() -> np.ndarray:
    region = np.zeros((LOCAL_CANVAS, LOCAL_CANVAS), dtype=bool)
    bh = random.randint(*BASE_SIZE_RANGE)
    bw = random.randint(*BASE_SIZE_RANGE)
    r0 = random.randint(3, LOCAL_CANVAS - bh - 3)
    c0 = random.randint(3, LOCAL_CANVAS - bw - 3)
    region[r0 : r0 + bh, c0 : c0 + bw] = True

    for _ in range(random.randint(*ARM_COUNT_RANGE)):
        _attach_arm(region)

    region = _crop_mask(region)
    region = np.rot90(region, k=random.randrange(4))
    if random.random() < 0.5:
        region = region.T
    region = np.pad(region, 2, constant_values=False)
    return region


def _make_object_patch(seed_color: int) -> np.ndarray | None:
    depth = _seed_depth(seed_color)
    for _ in range(OBJECT_ATTEMPTS):
        region = _build_region()
        outline = _outline_from_region(region)
        if not np.any(outline):
            continue
        if len(_find_components(outline, connectivity=8)) != 1:
            continue

        dist = _distance_layers(region, outline)
        if not np.any(dist == depth):
            continue
        seed_candidates = np.argwhere(dist >= depth + 1)
        if seed_candidates.size == 0:
            continue

        seed_r, seed_c = map(int, seed_candidates[random.randrange(len(seed_candidates))])
        patch = np.full(region.shape, -1, dtype=int)
        patch[outline] = GREY
        patch[seed_r, seed_c] = seed_color

        crop_mask = patch >= 0
        r0, c0, r1, c1 = _bbox(crop_mask)
        patch = patch[r0 : r1 + 1, c0 : c1 + 1].copy()
        return patch
    return None


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph + 2 > h or pw + 2 > w:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        top = random.randint(1, h - ph - 1)
        left = random.randint(1, w - pw - 1)
        r0 = max(0, top - 1)
        c0 = max(0, left - 1)
        r1 = min(h, top + ph + 1)
        c1 = min(w, left + pw + 1)
        if occupied[r0:r1, c0:c1].any():
            continue

        mask = patch >= 0
        view = grid[top : top + ph, left : left + pw]
        view[mask] = patch[mask]
        occupied[r0:r1, c0:c1] = True
        return True
    return False


def _color_plan(count: int) -> list[int]:
    if count == 1:
        return [random.choice([RED, BLUE])]
    colors = [RED, BLUE]
    while len(colors) < count:
        colors.append(random.choice([RED, BLUE]))
    random.shuffle(colors)
    return colors


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    seed_colors = _color_plan(object_count)
    patches: list[np.ndarray] = []

    for color in seed_colors:
        patch = _make_object_patch(color)
        if patch is None:
            return None
        patches.append(patch)

    total_reserved = sum((p.shape[0] + 2) * (p.shape[1] + 2) for p in patches)
    side_hint = max(int(np.ceil(np.sqrt(total_reserved * 1.45))), max(max(p.shape) for p in patches) + 2)
    min_side = max(GRID_SIZE_RANGE[0], min(30, side_hint))
    if min_side > GRID_SIZE_RANGE[1]:
        return None

    h = random.randint(min_side, GRID_SIZE_RANGE[1])
    w = random.randint(min_side, GRID_SIZE_RANGE[1])
    bg = _random_bg()
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    order = list(range(object_count))
    random.shuffle(order)
    for idx in order:
        if not _place_patch(grid, occupied, patches[idx]):
            return None

    if _most_common_color(grid) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    present_seed_colors = {int(v) for v in np.unique(grid) if int(v) in (RED, BLUE)}
    for color in present_seed_colors:
        if not np.any(out == color):
            return None
    if np.any(out[grid == RED] == RED) or np.any(out[grid == BLUE] == BLUE):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shrinking-outline ARC input")
