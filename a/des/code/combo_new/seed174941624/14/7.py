import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]
PATTERN_INDEX_SETS = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
]
MAX_GENERATION_ATTEMPTS = 400

BASE_RING_COORDS = {
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),
}
AREA13_CHOICES = [(1, 4), (2, 4)]
AREA15_CHOICES = [(4, 1), (4, 2), (0, 0), (0, 3), (3, 0), (3, 3)]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    mask = np.zeros_like(patch, dtype=bool)
    for r, c in coords:
        rr = r - r0
        cc = c - c0
        patch[rr, cc] = int(grid[r, c])
        mask[rr, cc] = True
    return patch, mask, (r0, c0, r1, c1)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int):
    h, w = patch.shape
    for rr in range(h):
        tr = r0 + rr
        if not (0 <= tr < dst.shape[0]):
            continue
        for cc in range(w):
            tc = c0 + cc
            if not (0 <= tc < dst.shape[1]):
                continue
            val = int(patch[rr, cc])
            if val != BLACK:
                dst[tr, tc] = val


def _find_palette_entries(grid: np.ndarray) -> dict[tuple[int, int, int, int], int]:
    h, w = grid.shape
    entries: dict[tuple[int, int, int, int], int] = {}
    for r, c in np.argwhere(grid == MAROON):
        candidates = []
        for left in (int(c), int(c) - 1):
            top = int(r) - 2
            if top < 0 or left < 0 or left + 1 >= w:
                continue
            block = grid[top : top + 2, left : left + 2]
            vals = [int(x) for x in block.reshape(-1)]
            cnt = Counter(vals)
            if BLACK in cnt or GREY in cnt or MAROON in cnt:
                continue
            if len(cnt) == 2 and sorted(cnt.values()) == [2, 2]:
                candidates.append(tuple(vals))
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) == 1:
            entries[candidates[0]] = int(c)
    assert entries, "palette entries not found"
    return entries


def _find_reference_bottom(grid: np.ndarray) -> int:
    grey_pts = np.argwhere(grid == GREY)
    assert grey_pts.size > 0, "grey reference line missing"
    col_counts = Counter(int(c) for _, c in grey_pts)
    ref_col = max(col_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return max(int(r) for r, c in grey_pts if int(c) == ref_col)


def _extract_identity_pattern(patch: np.ndarray, mask: np.ndarray) -> tuple[int, int, int, int]:
    values = [int(patch[r, c]) for r, c in np.argwhere(mask)]
    dominant = Counter(values).most_common(1)[0][0]
    h, w = patch.shape
    candidates = []
    for r in range(1, h - 2):
        for c in range(1, w - 2):
            if not mask[r : r + 2, c : c + 2].all():
                continue
            vals = [int(x) for x in patch[r : r + 2, c : c + 2].reshape(-1)]
            cnt = Counter(vals)
            if len(cnt) != 2 or sorted(cnt.values()) != [2, 2]:
                continue
            ring = [
                (r - 1, c),
                (r - 1, c + 1),
                (r + 2, c),
                (r + 2, c + 1),
                (r, c - 1),
                (r + 1, c - 1),
                (r, c + 2),
                (r + 1, c + 2),
            ]
            if not all(mask[rr, cc] for rr, cc in ring):
                continue
            if any(int(patch[rr, cc]) != dominant for rr, cc in ring):
                continue
            candidates.append(tuple(vals))
    assert len(candidates) == 1, "shape must contain exactly one enclosed 2x2 identity pattern"
    return candidates[0]


def _extract_shapes(grid: np.ndarray) -> list[dict]:
    mask = (grid != BLACK) & (grid != GREY) & (grid != MAROON)
    shapes = []
    for comp in _components(mask):
        if len(comp) <= 4:
            continue
        patch, comp_mask, bbox = _crop_component(grid, comp)
        pattern = _extract_identity_pattern(patch, comp_mask)
        shapes.append(
            {
                "area": len(comp),
                "patch": patch,
                "pattern": pattern,
                "source_bbox": bbox,
            }
        )
    assert shapes, "no shapes found"
    return shapes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each shape's enclosed 2x2 color code to the palette, then stack the shapes by area at palette-selected columns.

    Input structure:
    - Background is black.
    - The upper area contains several 2x2 two-color palette blocks laid out horizontally, each with a single maroon marker pixel directly below one of the block columns.
    - The lower area contains several larger 4-connected shapes; each is mostly one color but has exactly one enclosed 2x2 two-color pattern embedded inside it.
    - A grey vertical reference line appears on the right side.

    Rule:
    - For each large shape, extract the unique enclosed 2x2 two-color pattern.
    - Find the palette block with the exact same 2x2 pattern, without rotating or flipping it, and read the matched maroon marker column.
    - Sort the large shapes by area from largest to smallest, place the largest with its bbox bottom aligned to the grey line's bottom cell, then stack the rest directly above it.
    - During placement, horizontally align each shape's bbox center to the matched maroon marker column, and erase the palette and grey line from the output.

    Key insight:
    - The identity of each shape is encoded by the hidden interior 2x2 pattern rather than by the outer color or silhouette, while the maroon marker under the matching palette tile gives the destination column.
    """
    g = np.asarray(grid, dtype=int)
    anchors = _find_palette_entries(g)
    ref_bottom = _find_reference_bottom(g)
    shapes = _extract_shapes(g)
    shapes.sort(key=lambda item: (-item["area"], item["source_bbox"][0], item["source_bbox"][1]))

    out = np.full_like(g, BLACK)
    current_bottom = int(ref_bottom)
    for shape in shapes:
        pattern = shape["pattern"]
        assert pattern in anchors, "shape pattern missing from palette"
        patch = shape["patch"]
        h, w = patch.shape
        dest_c0 = int(anchors[pattern] - (w // 2))
        dest_r0 = int(current_bottom - h + 1)
        _paste(out, patch, dest_r0, dest_c0)
        current_bottom = dest_r0 - 1
    return out


ALL_PATTERNS = []
for a in COLOR_POOL:
    for b in COLOR_POOL:
        if a == b:
            continue
        for idxs in PATTERN_INDEX_SETS:
            cells = [b, b, b, b]
            for idx in idxs:
                cells[idx] = a
            ALL_PATTERNS.append(tuple(cells))
ALL_PATTERNS = sorted(set(ALL_PATTERNS))


def _shape_patch_for_area(base_color: int, pattern: tuple[int, int, int, int], area: int, rng: random.Random) -> np.ndarray:
    coords = set(BASE_RING_COORDS)
    if area == 13:
        coords.add(rng.choice(AREA13_CHOICES))
    elif area == 14:
        coords.update(AREA13_CHOICES)
    elif area == 15:
        coords.update(AREA13_CHOICES)
        coords.add(rng.choice(AREA15_CHOICES))
    else:
        raise ValueError(f"unsupported area: {area}")

    canvas = np.full((5, 5), BLACK, dtype=int)
    for r, c in coords:
        canvas[r, c] = int(base_color)
    canvas[1:3, 1:3] = np.array(pattern, dtype=int).reshape(2, 2)

    if rng.random() < 0.5:
        canvas = np.fliplr(canvas)
    if rng.random() < 0.5:
        canvas = np.flipud(canvas)

    pts = np.argwhere(canvas != BLACK)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return canvas[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _bbox_is_clear(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occ.shape[0], r0 + h + pad)
    cc1 = min(occ.shape[1], c0 + w + pad)
    return not occ[rr0:rr1, cc0:cc1].any()


def _mark_bbox(occ: np.ndarray, r0: int, c0: int, h: int, w: int):
    occ[r0 : r0 + h, c0 : c0 + w] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_shapes = rng.choice([2, 3])
    areas = rng.sample([13, 14, 15], num_shapes)
    h = rng.randint(20, 30)
    palette_width = 3 * num_shapes - 1
    min_w = max(18, palette_width + 6)
    w = rng.randint(min_w, 30)

    palette_row = rng.randint(1, 2)
    max_palette_start = w - palette_width - 4
    if max_palette_start < 2:
        return None
    palette_start = rng.randint(2, max_palette_start)
    line_top = rng.randint(max(h // 2 - 1, palette_row + 4), h - 6)

    shape_patches = []
    raw_patterns = rng.sample(ALL_PATTERNS, num_shapes)
    rng.shuffle(raw_patterns)
    for area, pattern in zip(areas, raw_patterns):
        pattern_colors = set(pattern)
        base_choices = [c for c in COLOR_POOL if c not in pattern_colors]
        if not base_choices:
            return None
        base_color = rng.choice(base_choices)
        patch = _shape_patch_for_area(base_color, pattern, area, rng)
        actual_pattern = _extract_identity_pattern(patch, patch != BLACK)
        shape_patches.append(
            {
                "area": area,
                "pattern": actual_pattern,
                "patch": patch,
            }
        )
    patterns = [item["pattern"] for item in shape_patches]
    if len(set(patterns)) != num_shapes:
        return None
    palette_order = patterns[:]
    rng.shuffle(palette_order)

    total_output_height = sum(int(item["patch"].shape[0]) for item in shape_patches)
    if total_output_height > h - (palette_row + 4):
        return None

    grid = np.full((h, w), BLACK, dtype=int)

    marker_cols: dict[tuple[int, int, int, int], int] = {}
    col = palette_start
    for pattern in palette_order:
        block = np.array(pattern, dtype=int).reshape(2, 2)
        grid[palette_row : palette_row + 2, col : col + 2] = block
        marker_col = col + rng.choice([0, 1])
        grid[palette_row + 2, marker_col] = MAROON
        marker_cols[pattern] = int(marker_col)
        col += 3

    grid[line_top:, w - 1] = GREY

    occ = np.zeros((h, w), dtype=bool)
    _mark_bbox(occ, palette_row, palette_start, 3, palette_width)
    _mark_bbox(occ, line_top, w - 1, h - line_top, 1)

    lower_min_row = max(h // 2, palette_row + 5)
    for item in shape_patches:
        patch = item["patch"]
        ph, pw = patch.shape
        placed = False
        row_hi = h - ph
        col_hi = w - 1 - pw
        if row_hi < lower_min_row or col_hi < 0:
            return None
        for _ in range(80):
            r0 = rng.randint(lower_min_row, row_hi)
            c0 = rng.randint(0, col_hi)
            if not _bbox_is_clear(occ, r0, c0, ph, pw, pad=1):
                continue
            patch_mask = patch != BLACK
            if (grid[r0 : r0 + ph, c0 : c0 + pw][patch_mask] != BLACK).any():
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = patch[patch_mask]
            _mark_bbox(occ, r0, c0, ph, pw)
            placed = True
            break
        if not placed:
            return None

    if len(marker_cols) != num_shapes:
        return None

    found_entries = _find_palette_entries(grid)
    found_shapes = _extract_shapes(grid)
    if len(found_entries) != num_shapes or len(found_shapes) != num_shapes:
        return None
    if set(found_entries) != set(patterns):
        return None
    if sorted(item["area"] for item in found_shapes) != sorted(areas):
        return None

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
