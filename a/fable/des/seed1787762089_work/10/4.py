import random

import numpy as np


BLACK = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 400
GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _corner_score(bbox: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    corner_dists = [
        r0 + c0,
        r0 + (w - 1 - c1),
        (h - 1 - r1) + c0,
        (h - 1 - r1) + (w - 1 - c1),
    ]
    edge_touches = [
        int(r0 == 0) + int(c0 == 0),
        int(r0 == 0) + int(c1 == w - 1),
        int(r1 == h - 1) + int(c0 == 0),
        int(r1 == h - 1) + int(c1 == w - 1),
    ]
    return min(corner_dists), -max(edge_touches), r0, c0


def _find_reference_tile(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = grid.shape
    candidates = []
    for coords in _components(grid != BLACK):
        r0, c0, r1, c1 = _bbox(coords)
        side_r = r1 - r0 + 1
        side_c = c1 - c0 + 1
        if side_r != side_c or side_r not in (2, 3):
            continue
        if len(coords) != side_r * side_r:
            continue
        tile = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = {int(v) for v in np.unique(tile)}
        if BLACK in colors or GREY in colors:
            continue
        if not (2 <= len(colors) <= 3):
            continue
        score = _corner_score((r0, c0, r1, c1), h, w)
        candidates.append((score, tile.copy(), (r0, c0, r1, c1)))

    assert candidates, "missing reference tile"
    candidates.sort(key=lambda item: item[0])
    _, tile, bbox = candidates[0]
    return tile, bbox


def _extract_main_region(grid: np.ndarray, ref_bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = ref_bbox
    remaining = np.array(grid, copy=True)
    remaining[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    mask = remaining != BLACK
    assert mask.any(), "missing transform target"
    mr0, mc0, mr1, mc1 = _bbox_from_mask(mask)
    return remaining[mr0 : mr1 + 1, mc0 : mc1 + 1]


def _expand(mask_region: np.ndarray, tile: np.ndarray) -> np.ndarray:
    k = int(tile.shape[0])
    h, w = mask_region.shape
    out = np.zeros((h * k, w * k), dtype=int)
    for r in range(h):
        for c in range(w):
            value = int(mask_region[r, c])
            assert value in (BLACK, GREY), "mask state must use only black and grey"
            if value == GREY:
                out[r * k : (r + 1) * k, c * k : (c + 1) * k] = tile
    return out


def _shrink(pattern: np.ndarray, tile: np.ndarray) -> np.ndarray:
    k = int(tile.shape[0])
    h, w = pattern.shape
    assert h % k == 0 and w % k == 0, "expanded pattern must align to the reference tile"
    out = np.zeros((h // k, w // k), dtype=int)
    for r in range(0, h, k):
        for c in range(0, w, k):
            block = pattern[r : r + k, c : c + k]
            if np.all(block == BLACK):
                continue
            assert np.array_equal(block, tile), "every non-empty macro-cell must equal the reference tile"
            out[r // k, c // k] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    tile, ref_bbox = _find_reference_tile(g)
    main_region = _extract_main_region(g, ref_bbox)
    main_colors = {int(v) for v in np.unique(main_region) if int(v) != BLACK}
    assert main_colors, "target region must be non-empty"

    if main_colors == {GREY}:
        return _expand(main_region, tile)
    return _shrink(main_region, tile)


def _neighbors(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    result = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            result.append((nr, nc))
    return result


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _make_tile(k: int) -> np.ndarray:
    for _ in range(200):
        color_count = random.choice([2, 3])
        colors = random.sample(GEN_COLORS, color_count)
        values = [random.choice(colors) for _ in range(k * k)]
        tile = np.array(values, dtype=int).reshape(k, k)
        if len(np.unique(tile)) != color_count:
            continue
        return tile
    raise RuntimeError("failed to build a valid reference tile")


def _make_mask(h: int, w: int) -> np.ndarray:
    area = h * w
    min_filled = max(3, area // 2)
    for _ in range(300):
        target = random.randint(min_filled, area - 1)
        start = (random.randrange(h), random.randrange(w))
        mask = np.zeros((h, w), dtype=bool)
        mask[start] = True
        filled = [start]
        while len(filled) < target:
            frontier = set()
            for r, c in filled:
                for nr, nc in _neighbors(r, c, h, w):
                    if not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True
            filled.append((nr, nc))
        if len(filled) != target:
            continue
        if not _touches_all_sides(mask):
            continue
        if not (~mask).any():
            continue
        return mask
    raise RuntimeError("failed to build a valid grey mask")


def _ref_origin(h: int, w: int, k: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - k
    if corner == "bl":
        return h - k, 0
    return h - k, w - k


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    k = random.choice([2, 3])
    tile = _make_tile(k)

    dim_hi = 5 if k == 2 else 4
    mask_h = random.randint(2, dim_hi)
    mask_w = random.randint(2, dim_hi)
    coarse_mask = _make_mask(mask_h, mask_w)
    coarse = np.where(coarse_mask, GREY, BLACK).astype(int)

    branch = random.choice(["expand", "shrink"])
    if branch == "expand":
        main_object = coarse
        expected = _expand(coarse, tile)
    else:
        main_object = _expand(coarse, tile)
        expected = coarse

    main_h, main_w = main_object.shape
    h = main_h + random.randint(8, 12)
    w = main_w + random.randint(8, 12)
    grid = np.zeros((h, w), dtype=int)

    ref_r, ref_c = _ref_origin(h, w, k, random.choice(["tl", "tr", "bl", "br"]))
    grid[ref_r : ref_r + k, ref_c : ref_c + k] = tile

    r0 = random.randint(4, h - main_h - 4)
    c0 = random.randint(4, w - main_w - 4)
    grid[r0 : r0 + main_h, c0 : c0 + main_w] = main_object

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input")
