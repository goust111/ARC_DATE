from __future__ import annotations

import random

import numpy as np


RED = 2
GREY = 5
NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size, "expected a nonempty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in NEIGHBORS_4:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        comp = np.zeros_like(mask, dtype=bool)
        for cr, cc in coords:
            comp[cr, cc] = True
        comps.append(comp)
    return comps


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    tight = _tight_mask(mask)
    return tuple(map(int, tight.shape)), np.ascontiguousarray(tight.astype(np.uint8)).tobytes()


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    base = _tight_mask(mask)
    variants = []
    for k in range(4):
        rotated = np.rot90(base, k)
        variants.append(_mask_key(rotated))
        variants.append(_mask_key(np.fliplr(rotated)))
    return min(variants)


def _legend_entries(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    rows = np.flatnonzero(grid[:, -1] != bg)
    assert rows.size > 0, "expected a right-edge legend"
    entries = [(int(r), int(grid[int(r), -1])) for r in rows]
    colors = [color for _row, color in entries]
    assert len(colors) == len(set(colors)), "legend colors must be unique"
    fg = {int(color) for color in np.unique(grid) if int(color) != bg}
    assert set(colors) == fg, "legend must list every foreground color"
    return entries


def _main_object_mask(grid: np.ndarray, color: int, legend_row: int) -> np.ndarray:
    mask = np.asarray(grid == color, dtype=bool)
    mask[legend_row, -1] = False
    comps = _component_masks(mask)
    assert comps, "expected a main object"
    areas = [int(comp.sum()) for comp in comps]
    main = comps[int(np.argmax(areas))]
    return _tight_mask(main)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a color-labeled shape-equivalence table from the objects listed on the right edge.

    Input structure:
    - The background contains one main 4-connected object for each foreground color.
    - The rightmost column has one singleton pixel of each foreground color, listed top-to-bottom.
    - Those singleton pixels define the row and column order of the output table.

    Rule:
    - Normalize each main object to its tight mask and compare masks up to rotation or reflection.
    - Draw 2x2 color labels across the top and down the left, separated by one background row/column.
    - For each off-diagonal table cell, paint a 2x2 red block when the two labeled objects have the same shape under rotation/reflection, otherwise paint a 2x2 grey block.
    - Leave diagonal cells and all separators as background.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    entries = _legend_entries(g, bg)
    order = [color for _row, color in entries]
    signatures = {
        color: _shape_signature(_main_object_mask(g, color, legend_row))
        for legend_row, color in entries
    }

    n = len(order)
    out = np.full((3 * n + 2, 3 * n + 2), bg, dtype=int)

    for i, color in enumerate(order):
        band = 3 + 3 * i
        out[0:2, band : band + 2] = color
        out[band : band + 2, 0:2] = color

    for i, row_color in enumerate(order):
        for j, col_color in enumerate(order):
            if i == j:
                continue
            cell_color = RED if signatures[row_color] == signatures[col_color] else GREY
            r = 3 + 3 * i
            c = 3 + 3 * j
            out[r : r + 2, c : c + 2] = cell_color

    return out


# --------------- generator ---------------

OBJECT_COUNT_RANGE = (3, 6)
GRID_ROW_RANGE = (18, 27)
GRID_COL_RANGE = (18, 27)
RESERVED_RIGHT_COLS = 2
PLACEMENT_PADDING = 1
PLACEMENT_ATTEMPTS = 200
GEN_ATTEMPTS = 400

OBJECT_AREA_RANGE = (5, 13)
OBJECT_BBOX_RANGE = (2, 6)
OBJECT_MAX_EXTRA_CELLS = 4
UNIQUE_MASK_RETRIES = 80

BG_POOL = [0, 1, 3, 4, 6, 7, 8, 9]
OBJECT_COLOR_POOL = [1, 3, 4, 6, 7, 8, 9]


def _has_hole(mask: np.ndarray) -> bool:
    bg = ~np.asarray(mask, dtype=bool)
    if not bool(bg.any()):
        return False

    h, w = bg.shape
    seen = np.zeros_like(bg, dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS_4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    return bool((bg & ~seen).any())


def _random_corner_path(h: int, w: int) -> np.ndarray:
    starts = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    sr, sc = random.choice(starts)
    er, ec = h - 1 - sr, w - 1 - sc
    moves = [(1 if er > sr else -1, 0)] * (h - 1)
    moves += [(0, 1 if ec > sc else -1)] * (w - 1)
    random.shuffle(moves)

    mask = np.zeros((h, w), dtype=bool)
    r, c = sr, sc
    mask[r, c] = True
    for dr, dc in moves:
        r += dr
        c += dc
        mask[r, c] = True
    return mask


def _growth_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    choices: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            touches = 0
            for dr, dc in NEIGHBORS_4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    touches += 1
            if touches != 1:
                continue
            trial = mask.copy()
            trial[r, c] = True
            if not _has_hole(trial):
                choices.append((r, c))
    return choices


def _random_mask() -> np.ndarray | None:
    area = random.randint(*OBJECT_AREA_RANGE)
    pairs = [
        (h, w)
        for h in range(OBJECT_BBOX_RANGE[0], OBJECT_BBOX_RANGE[1] + 1)
        for w in range(OBJECT_BBOX_RANGE[0], OBJECT_BBOX_RANGE[1] + 1)
        if h + w - 1 <= area <= h * w and area - (h + w - 1) <= OBJECT_MAX_EXTRA_CELLS
    ]
    if not pairs:
        return None
    h, w = random.choice(pairs)
    mask = _random_corner_path(h, w)

    while int(mask.sum()) < area:
        choices = _growth_candidates(mask)
        if not choices:
            return None
        r, c = random.choice(choices)
        mask[r, c] = True
    return mask


def _class_sizes(count: int) -> list[int]:
    max_classes = min(4, count - 1)
    class_count = random.randint(2, max_classes)
    sizes = [1] * class_count
    for _ in range(count - class_count):
        sizes[random.randrange(class_count)] += 1
    random.shuffle(sizes)
    return sizes


def _unique_canonical_masks(class_count: int) -> list[np.ndarray] | None:
    masks: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(class_count):
        for _ in range(UNIQUE_MASK_RETRIES):
            mask = _random_mask()
            if mask is None:
                continue
            sig = _shape_signature(mask)
            if sig in seen:
                continue
            seen.add(sig)
            masks.append(mask)
            break
        else:
            return None
    return masks


def _dihedral(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.astype(bool).copy()


def _build_expected(bg: int, colors: list[int], class_ids: list[int]) -> np.ndarray:
    n = len(colors)
    out = np.full((3 * n + 2, 3 * n + 2), bg, dtype=int)

    for i, color in enumerate(colors):
        band = 3 + 3 * i
        out[0:2, band : band + 2] = color
        out[band : band + 2, 0:2] = color

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = 3 + 3 * i
            c = 3 + 3 * j
            out[r : r + 2, c : c + 2] = RED if class_ids[i] == class_ids[j] else GREY
    return out


def _can_place(occupied: np.ndarray, shape: tuple[int, int], top: int, left: int, padding: int) -> bool:
    h, w = occupied.shape
    ph, pw = shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(h, top + ph + padding)
    c1 = min(w, left + pw + padding)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_objects(bg: int, colors: list[int], masks: list[np.ndarray]) -> np.ndarray | None:
    h = random.randint(*GRID_ROW_RANGE)
    w = random.randint(*GRID_COL_RANGE)
    if h < len(colors) or w <= RESERVED_RIGHT_COLS + 2:
        return None

    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w - RESERVED_RIGHT_COLS), dtype=bool)

    order = sorted(range(len(colors)), key=lambda i: int(masks[i].sum()), reverse=True)
    for idx in order:
        mask = masks[idx]
        ph, pw = map(int, mask.shape)
        if ph > h or pw > w - RESERVED_RIGHT_COLS:
            return None
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(0, h - ph)
            left = random.randint(0, (w - RESERVED_RIGHT_COLS) - pw)
            if not _can_place(occupied, (ph, pw), top, left, PLACEMENT_PADDING):
                continue
            occupied[top : top + ph, left : left + pw] |= mask
            grid[top : top + ph, left : left + pw][mask] = colors[idx]
            break
        else:
            return None

    marker_rows = sorted(random.sample(range(h), len(colors)))
    for row, color in zip(marker_rows, colors):
        grid[row, -1] = color

    return grid


def _sample_design() -> tuple[int, list[int], list[int], list[np.ndarray], np.ndarray] | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    class_sizes = _class_sizes(count)
    class_ids = [class_id for class_id, size in enumerate(class_sizes) for _ in range(size)]
    random.shuffle(class_ids)

    canonical_masks = _unique_canonical_masks(len(class_sizes))
    if canonical_masks is None:
        return None

    colors = random.sample(OBJECT_COLOR_POOL, count)
    bg_choices = [color for color in BG_POOL if color not in colors and color not in (RED, GREY)]
    if not bg_choices:
        return None
    bg = random.choice(bg_choices)

    oriented_masks = [_dihedral(canonical_masks[class_id]) for class_id in class_ids]
    expected = _build_expected(bg, colors, class_ids)
    return bg, colors, class_ids, oriented_masks, expected


def _gen_attempt() -> np.ndarray | None:
    design = _sample_design()
    if design is None:
        return None
    bg, colors, _class_ids, masks, expected = design

    grid = _place_objects(bg, colors, masks)
    if grid is None:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
