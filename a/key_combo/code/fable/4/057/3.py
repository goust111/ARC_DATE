import random
from collections import deque

import numpy as np


MAX_GEN_ATTEMPTS = 500
GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
PATTERN_H_RANGE = (6, 11)
PATTERN_W_RANGE = (7, 14)
PALETTE_SIZE_RANGE = (3, 5)
DISTRACTOR_COUNT_CHOICES = (1, 1, 2)
ALL_COLORS = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_pattern_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    comps = _component_coords(grid != bg)
    assert comps, "expected a foreground pattern rectangle"
    largest = max(comps, key=len)
    box = _bbox(largest)
    r0, c0, r1, c1 = box
    assert len(largest) == (r1 - r0 + 1) * (c1 - c0 + 1), "pattern region must fill its bbox"
    return box


def _find_sticker(region: np.ndarray) -> tuple[int, np.ndarray]:
    h, w = region.shape
    mismatch = region != region[:, ::-1]
    candidates: list[tuple[int, np.ndarray]] = []
    for color in np.unique(region):
        mask = region == int(color)
        for coords in _component_coords(mask):
            rows = coords[:, 0]
            cols = coords[:, 1]
            if not np.all(mismatch[rows, cols]):
                continue
            if w % 2 == 1 and np.any(cols == w // 2):
                continue
            mirrored_colors = region[rows, w - 1 - cols]
            if len(set(map(int, mirrored_colors.tolist()))) < 2:
                continue
            restored = region.copy()
            restored[rows, cols] = mirrored_colors
            if np.array_equal(restored, restored[:, ::-1]):
                candidates.append((int(color), coords))
    assert len(candidates) == 1, "expected exactly one sticker component"
    return candidates[0]


def _move_sticker(
    grid: np.ndarray,
    pattern_bbox: tuple[int, int, int, int],
    sticker_color: int,
    sticker_local: np.ndarray,
) -> np.ndarray:
    r0, c0, r1, c1 = pattern_bbox
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    out = grid.copy()

    _sr0, sc0, _sr1, sc1 = _bbox(sticker_local)
    delta_c = (region.shape[1] - 1 - sc1) - sc0

    rows = sticker_local[:, 0]
    cols = sticker_local[:, 1]
    out[r0 + rows, c0 + cols] = region[rows, region.shape[1] - 1 - cols]
    out[r0 + rows, c0 + cols + delta_c] = int(sticker_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the monochrome sticker across the symmetric pattern rectangle without flipping it.

    Input structure:
    - The grid has a dominant background plus one filled multicolor subrectangle.
    - That subrectangle is vertically mirror-symmetric before a monochrome sticker covers one side.
    - One or two same-colored distractor blobs may sit outside the rectangle and do not touch it.

    Rule:
    - Find the largest non-background connected component and treat its filled bbox as the pattern rectangle.
    - Inside that rectangle, compare each cell with its horizontal mirror; the sticker is the unique monochrome connected mismatch component.
    - Restore the cells vacated by the sticker from their mirrored partners, then translate the whole sticker to the mirrored bbox position while preserving its internal orientation.

    Key insight:
    - The transformation mirrors the sticker's location, not the sticker's shape: symmetry identifies which pixels belong to the overlaid component, while same-colored distractors outside the symmetric rectangle are irrelevant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    pattern_bbox = _find_pattern_bbox(g, bg)
    r0, c0, r1, c1 = pattern_bbox
    region = g[r0 : r1 + 1, c0 : c1 + 1]
    sticker_color, sticker_local = _find_sticker(region)
    return _move_sticker(g, pattern_bbox, sticker_color, sticker_local)


def _random_polyomino(height: int, width: int, min_area: int, max_area: int) -> np.ndarray:
    target = random.randint(min_area, min(max_area, height * width))
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_symmetric_pattern(height: int, width: int, palette: list[int]) -> np.ndarray:
    half = width // 2
    left = np.array(
        [[random.choice(palette) for _ in range(half)] for _ in range(height)],
        dtype=int,
    )
    for _ in range(random.randint(2, 5)):
        color = random.choice(palette)
        r0 = random.randrange(height)
        r1 = random.randrange(r0 + 1, height + 1)
        if half > 0:
            c0 = random.randrange(half)
            c1 = random.randrange(c0 + 1, half + 1)
            left[r0:r1, c0:c1] = color
    if len(set(map(int, left.flatten().tolist()))) == 1 and len(palette) > 1 and half > 0:
        left[random.randrange(height), random.randrange(half)] = random.choice([c for c in palette if c != left[0, 0]])

    region = np.empty((height, width), dtype=int)
    region[:, :half] = left
    if width % 2 == 1:
        center = np.array([random.choice(palette) for _ in range(height)], dtype=int)
        if random.random() < 0.7:
            color = random.choice(palette)
            r0 = random.randrange(height)
            r1 = random.randrange(r0 + 1, height + 1)
            center[r0:r1] = color
        region[:, half] = center
    region[:, width - half :] = left[:, ::-1]
    return region


def _sticker_candidates(
    region: np.ndarray,
    shape_mask: np.ndarray,
    sticker_color: int,
    side: str,
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    h, w = region.shape
    sh, sw = shape_mask.shape
    side_width = w // 2
    offsets = list(range(side_width - sw + 1))
    random.shuffle(offsets)
    tops = list(range(h - sh + 1))
    random.shuffle(tops)
    local_shape = set(map(tuple, np.argwhere(shape_mask).tolist()))
    choices = []
    for top in tops:
        for offset in offsets:
            left = offset if side == "left" else w - sw - offset
            cells = np.array([(top + r, left + c) for r, c in local_shape], dtype=int)
            colors = region[cells[:, 0], cells[:, 1]]
            if sticker_color in set(map(int, colors.tolist())):
                continue
            if len(set(map(int, colors.tolist()))) < 2:
                continue
            cell_set = set(map(tuple, cells.tolist()))
            ok = True
            for r, c in cell_set:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cell_set and region[nr, nc] == sticker_color:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            choices.append((cells, (top, left, top + sh - 1, left + sw - 1)))
    return choices


def _ensure_pattern_has_sticker_color(
    region: np.ndarray,
    sticker_color: int,
    forbidden: set[tuple[int, int]],
) -> bool:
    h, w = region.shape
    pairs = []
    for r in range(h):
        for c in range(w // 2):
            mc = w - 1 - c
            if (r, c) in forbidden or (r, mc) in forbidden:
                continue
            pairs.append((r, c, mc))
    random.shuffle(pairs)
    for r, c, mc in pairs[: max(3, len(pairs))]:
        region[r, c] = sticker_color
        region[r, mc] = sticker_color
        return True
    return False


def _place_distractor(
    grid: np.ndarray,
    bg: int,
    color: int,
    forbidden: np.ndarray,
    shape_mask: np.ndarray,
) -> bool:
    h, w = grid.shape
    sh, sw = shape_mask.shape
    shape_cells = np.argwhere(shape_mask)
    starts = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
    random.shuffle(starts)
    for top, left in starts:
        cells = [(top + int(r), left + int(c)) for r, c in shape_cells]
        if any(forbidden[r, c] for r, c in cells):
            continue
        ok = True
        for r, c in cells:
            if grid[r, c] != bg:
                ok = False
                break
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells and grid[nr, nc] != bg:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for r, c in cells:
            grid[r, c] = color
            forbidden[r, c] = True
        return True
    return False


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.choice(ALL_COLORS)
    fg_pool = [c for c in ALL_COLORS if c != bg]
    pattern_h = random.randint(*PATTERN_H_RANGE)
    pattern_w = random.randint(*PATTERN_W_RANGE)
    if pattern_w // 2 < 2:
        return None

    grid_h = random.randint(max(pattern_h + 4, GRID_H_RANGE[0]), GRID_H_RANGE[1])
    grid_w = random.randint(max(pattern_w + 4, GRID_W_RANGE[0]), GRID_W_RANGE[1])
    if grid_h > 30 or grid_w > 30:
        return None

    top = random.randint(1, grid_h - pattern_h - 1)
    left = random.randint(1, grid_w - pattern_w - 1)
    palette_size = random.randint(*PALETTE_SIZE_RANGE)
    palette = random.sample(fg_pool, palette_size)
    sticker_color = random.choice(palette)
    region = _make_symmetric_pattern(pattern_h, pattern_w, palette)

    shape_h = random.randint(2, min(5, pattern_h))
    shape_w = random.randint(2, min(4, pattern_w // 2))
    shape_mask = _random_polyomino(shape_h, shape_w, 2, min(shape_h * shape_w - 1, 7))
    if not np.any(shape_mask):
        return None

    side = random.choice(["left", "right"])
    candidates = _sticker_candidates(region, shape_mask, sticker_color, side)
    if not candidates:
        return None
    sticker_local, sticker_bbox = random.choice(candidates)
    if sticker_color not in set(map(int, region.flatten().tolist())):
        if not _ensure_pattern_has_sticker_color(region, sticker_color, set(map(tuple, sticker_local.tolist()))):
            return None

    base = np.full((grid_h, grid_w), bg, dtype=int)
    base[top : top + pattern_h, left : left + pattern_w] = region

    forbidden = np.zeros_like(base, dtype=bool)
    forbidden[top - 1 : top + pattern_h + 1, left - 1 : left + pattern_w + 1] = True
    distractor_count = random.choice(DISTRACTOR_COUNT_CHOICES)
    for _ in range(distractor_count):
        dh = random.randint(1, 3)
        dw = random.randint(1, 3)
        dmask = _random_polyomino(dh, dw, 1, min(3, dh * dw))
        if not _place_distractor(base, bg, sticker_color, forbidden, dmask):
            return None

    inp = base.copy()
    inp[top + sticker_local[:, 0], left + sticker_local[:, 1]] = sticker_color

    out = base.copy()
    _r0, sc0, _r1, sc1 = sticker_bbox
    delta_c = (pattern_w - 1 - sc1) - sc0
    out[top + sticker_local[:, 0], left + sticker_local[:, 1] + delta_c] = sticker_color
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _build_scene()
        if built is None:
            continue
        inp, expected = built
        try:
            if np.array_equal(main(inp), expected):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input for puzzle 3")
