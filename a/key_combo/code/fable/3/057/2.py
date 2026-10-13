import random
from collections import Counter

import numpy as np


BACKGROUND = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_MIN_SIDE = 18
GRID_MAX_SIDE = 28
PAIR_COUNT_CHOICES = (1, 2, 2, 3)
PAIR_PADDING = 1
VECTOR_COMPONENTS = range(-6, 7)
MIN_VECTOR_MANHATTAN = 2
MAX_VECTOR_MANHATTAN = 7
MAX_TOTAL_MANHATTAN = 10
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
        raise ValueError("patch out of bounds")
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"coords": coords, "bbox": _bbox_coords(coords), "area": int(coords.shape[0])})
    return comps


def _shape_signature(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    anchored = np.asarray(coords, dtype=int) - np.min(coords, axis=0, keepdims=True)
    order = np.lexsort((anchored[:, 1], anchored[:, 0]))
    anchored = anchored[order]
    return tuple((int(r), int(c)) for r, c in anchored.tolist())


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _shape_signature(np.argwhere(mask))


def _mask_is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _build_shape_catalog() -> tuple[np.ndarray, ...]:
    catalog: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for h in range(2, 5):
        for w in range(2, 5):
            cells = h * w
            max_area = min(8, cells)
            for bits in range(1 << cells):
                area = bits.bit_count()
                if area < 3 or area > max_area:
                    continue
                mask = np.zeros((h, w), dtype=bool)
                for idx in range(cells):
                    if bits & (1 << idx):
                        mask[idx // w, idx % w] = True
                if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
                    continue
                if not _mask_is_connected(mask):
                    continue
                sig = _mask_signature(mask)
                if sig in seen:
                    continue
                seen.add(sig)
                catalog.append(mask)
    return tuple(catalog)


SHAPE_CATALOG = _build_shape_catalog()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove every shadow-paired shape and translate the lone unpaired colored sprite by the sum of all shadow vectors.

    Input structure:
    - A black-background grid contains several separated foreground sprites.
    - Some sprites are pure grey silhouettes; every grey silhouette has exactly one colored sprite with the same occupied shape.
    - Exactly one colored sprite has no matching grey silhouette.

    Rule:
    - For each grey silhouette / colored-shape pair, compute the translation vector from the grey silhouette to the colored sprite.
    - Sum all of those vectors.
    - Erase every paired grey and colored sprite, then translate the unique unpaired colored sprite by the summed vector.

    Key insight:
    - The grey silhouettes are displacement instructions rather than missing-color targets; their shape matches identify vectors whose sum moves the only colored sprite without a shadow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert len(comps) >= 3

    shadows: dict[tuple[tuple[int, int], ...], dict] = {}
    colored: dict[tuple[tuple[int, int], ...], dict] = {}
    for comp in comps:
        coords = comp["coords"]
        values = {int(v) for v in g[coords[:, 0], coords[:, 1]]}
        sig = _shape_signature(coords)
        if values == {GREY}:
            assert sig not in shadows
            shadows[sig] = comp
        else:
            assert sig not in colored
            colored[sig] = comp

    assert shadows
    assert len(colored) == len(shadows) + 1
    assert set(shadows).issubset(colored)

    total_dr = 0
    total_dc = 0
    for sig, shadow in shadows.items():
        sprite = colored[sig]
        total_dr += int(sprite["bbox"][0] - shadow["bbox"][0])
        total_dc += int(sprite["bbox"][1] - shadow["bbox"][1])

    unpaired_sigs = [sig for sig in colored if sig not in shadows]
    assert len(unpaired_sigs) == 1
    unpaired = colored[unpaired_sigs[0]]

    out = np.full(g.shape, bg, dtype=int)
    patch = _crop_bbox(g, unpaired["bbox"])
    _paste(out, patch, unpaired["bbox"][0] + total_dr, unpaired["bbox"][1] + total_dc, transparent=bg)
    return out


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _padded_rect(top_left: tuple[int, int], shape: tuple[int, int], pad: int = PAIR_PADDING) -> tuple[int, int, int, int]:
    r, c = top_left
    h, w = shape
    return (r - pad, c - pad, r + h - 1 + pad, c + w - 1 + pad)


def _valid_interval(limit: int, size: int, delta: int) -> tuple[int, int]:
    lo = max(0, -delta)
    hi = min(limit - size, limit - size - delta)
    return lo, hi


def _rects_clear(rects: list[tuple[int, int, int, int]], reserved: list[tuple[int, int, int, int]]) -> bool:
    for i, rect in enumerate(rects):
        for other in reserved:
            if _rects_overlap(rect, other):
                return False
        for j in range(i):
            if _rects_overlap(rect, rects[j]):
                return False
    return True


def _vector_candidates(shape: tuple[int, int]) -> list[tuple[int, int]]:
    rect0 = _padded_rect((0, 0), shape)
    out: list[tuple[int, int]] = []
    for dr in VECTOR_COMPONENTS:
        for dc in VECTOR_COMPONENTS:
            manhattan = abs(dr) + abs(dc)
            if manhattan < MIN_VECTOR_MANHATTAN or manhattan > MAX_VECTOR_MANHATTAN:
                continue
            if _rects_overlap(rect0, _padded_rect((dr, dc), shape)):
                continue
            out.append((dr, dc))
    return out


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    area = len(coords)
    n_colors = 2 if area <= 4 else random.choice((2, 3))
    palette = random.sample(COLOR_POOL, n_colors)
    colors = list(palette)
    while len(colors) < area:
        colors.append(random.choice(palette))
    random.shuffle(colors)

    patch = np.full(mask.shape, BACKGROUND, dtype=int)
    for (r, c), color in zip(coords, colors):
        patch[r, c] = int(color)
    return patch


def _shadow_patch(mask: np.ndarray) -> np.ndarray:
    patch = np.full(mask.shape, BACKGROUND, dtype=int)
    patch[mask] = GREY
    return patch


def _place_pair(
    shape: tuple[int, int],
    vector: tuple[int, int],
    grid_shape: tuple[int, int],
    reserved: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    gh, gw = grid_shape
    dr, dc = vector
    r_lo, r_hi = _valid_interval(gh, h, dr)
    c_lo, c_hi = _valid_interval(gw, w, dc)
    if r_lo > r_hi or c_lo > c_hi:
        return None

    for _ in range(160):
        r = random.randint(r_lo, r_hi)
        c = random.randint(c_lo, c_hi)
        shadow_pos = (r, c)
        color_pos = (r + dr, c + dc)
        rects = [_padded_rect(shadow_pos, shape), _padded_rect(color_pos, shape)]
        if not _rects_clear(rects, reserved):
            continue
        reserved.extend(rects)
        return shadow_pos, color_pos
    return None


def _place_unpaired(
    shape: tuple[int, int],
    total_vector: tuple[int, int],
    grid_shape: tuple[int, int],
    reserved: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    gh, gw = grid_shape
    dr, dc = total_vector
    r_lo, r_hi = _valid_interval(gh, h, dr)
    c_lo, c_hi = _valid_interval(gw, w, dc)
    if r_lo > r_hi or c_lo > c_hi:
        return None

    for _ in range(160):
        r = random.randint(r_lo, r_hi)
        c = random.randint(c_lo, c_hi)
        rect = _padded_rect((r, c), shape)
        if not _rects_clear([rect], reserved):
            continue
        return (r, c), (r + dr, c + dc)
    return None


def _build_expected(grid_shape: tuple[int, int], patch: np.ndarray, final_pos: tuple[int, int]) -> np.ndarray:
    out = np.full(grid_shape, BACKGROUND, dtype=int)
    _paste(out, patch, final_pos[0], final_pos[1], transparent=BACKGROUND)
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    n_shapes = pair_count + 1
    shape_masks = random.sample(SHAPE_CATALOG, n_shapes)
    pair_masks = shape_masks[:pair_count]
    unpaired_mask = shape_masks[-1]

    vectors: list[tuple[int, int]] = []
    for _ in range(80):
        vectors = [random.choice(_vector_candidates(mask.shape)) for mask in pair_masks]
        total_dr = sum(dr for dr, _ in vectors)
        total_dc = sum(dc for _, dc in vectors)
        total_manhattan = abs(total_dr) + abs(total_dc)
        if MIN_VECTOR_MANHATTAN <= total_manhattan <= MAX_TOTAL_MANHATTAN:
            break
    else:
        return None

    total_vector = (total_dr, total_dc)
    side_lo = GRID_MIN_SIDE + pair_count
    gh = random.randint(side_lo, GRID_MAX_SIDE)
    gw = random.randint(side_lo, GRID_MAX_SIDE)
    grid = np.full((gh, gw), BACKGROUND, dtype=int)
    reserved: list[tuple[int, int, int, int]] = []

    placements: list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]] = []
    for mask, vector in zip(pair_masks, vectors):
        placed = _place_pair(mask.shape, vector, grid.shape, reserved)
        if placed is None:
            return None
        placements.append((mask, placed[0], placed[1]))

    unpaired_placed = _place_unpaired(unpaired_mask.shape, total_vector, grid.shape, reserved)
    if unpaired_placed is None:
        return None
    unpaired_patch = _colorize_mask(unpaired_mask)

    for mask, shadow_pos, color_pos in placements:
        _paste(grid, _shadow_patch(mask), shadow_pos[0], shadow_pos[1], transparent=BACKGROUND)
        _paste(grid, _colorize_mask(mask), color_pos[0], color_pos[1], transparent=BACKGROUND)
    _paste(grid, unpaired_patch, unpaired_placed[0][0], unpaired_placed[0][1], transparent=BACKGROUND)

    expected = _build_expected(grid.shape, unpaired_patch, unpaired_placed[1])
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
