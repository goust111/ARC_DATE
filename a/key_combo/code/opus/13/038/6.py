from __future__ import annotations

import random
from collections import Counter

import numpy as np


PLACEHOLDER = 5
TILE_COLORS = (1, 2, 3, 4)
OBJECT_COLORS = (7, 8, 9)
STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

GRID_H_RANGE = (11, 15)
GRID_W_RANGE = (11, 15)
VISIBLE_CUT_RANGE = (6, 9)
OBJECT_COUNT_RANGE = (2, 4)
OBJECT_AREA_CHOICES = (3, 4, 5, 6)
GEN_RETRIES = 500
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    H, W = mask.shape
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _find_objects(grid: np.ndarray) -> list[dict]:
    out = []
    for color in OBJECT_COLORS:
        for coords in _components(grid == int(color)):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _visible_bounds(grid: np.ndarray) -> tuple[int, int]:
    tile_mask = np.isin(grid, TILE_COLORS)
    rows = np.flatnonzero(tile_mask.any(axis=1))
    cols = np.flatnonzero(tile_mask.any(axis=0))
    assert rows.size and cols.size, "expected visible tile samples"
    r_cut = int(rows[-1]) + 1
    c_cut = int(cols[-1]) + 1
    assert 0 < r_cut < grid.shape[0] and 0 < c_cut < grid.shape[1]
    return r_cut, c_cut


def _infer_tile_3x3(grid: np.ndarray, r_cut: int, c_cut: int) -> np.ndarray:
    tile = np.full((3, 3), -1, dtype=int)
    block = grid[:r_cut, :c_cut]
    for tr in range(3):
        for tc in range(3):
            vals = [
                int(block[r, c])
                for r in range(tr, r_cut, 3)
                for c in range(tc, c_cut, 3)
                if int(block[r, c]) in TILE_COLORS
            ]
            assert vals, f"missing evidence for tile cell {(tr, tc)}"
            counts = Counter(vals)
            assert len(counts) == 1, "occluded tile evidence must agree"
            tile[tr, tc] = vals[0]
    return tile


def _full_background(shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    H, W = shape
    rr = np.arange(H)[:, None] % 3
    cc = np.arange(W)[None, :] % 3
    return tile[rr, cc]


def _candidate_periods(block: np.ndarray, known_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = block.shape
    candidates = []
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            ok = True
            for r in range(h):
                for c in range(w):
                    if not known_mask[r, c]:
                        continue
                    if int(block[r, c]) != int(block[r % ph, c % pw]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                candidates.append((ph, pw))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the hidden repeating tile, then drop each overlaid object downward by its own area.

    Input structure:
    - A top-left visible rectangle shows part of a 3x3 repeating tile using colors 1-4.
    - The right and bottom suffix outside that rectangle are placeholder color 5.
    - Small 4-connected monochrome objects in colors 7-9 may sit on both the visible tile and the placeholder area.

    Rule:
    - Treat colors 7-9 as overlaid objects and record each 4-connected component separately.
    - Infer the 3x3 tile from the visible top-left rectangle, then extend it periodically over the whole grid to replace every placeholder cell and the removed object cells.
    - For each object, compute its pixel area A and redraw the whole object A rows lower, clipping pixels below the bottom edge.

    Key insight:
    - Object pixels can hide both visible and hidden parts of the periodic background, so the tile must be reconstructed first; only then can the true background be restored and the area-driven downward slides applied correctly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r_cut, c_cut = _visible_bounds(g)
    tile = _infer_tile_3x3(g, r_cut, c_cut)
    bg = _full_background(g.shape, tile)
    objects = _find_objects(g)
    assert objects, "expected overlaid objects"

    out = bg.copy()
    painted = np.zeros(g.shape, dtype=bool)
    H, W = g.shape
    for obj in objects:
        coords = np.asarray(obj["coords"], dtype=int)
        area = int(obj["area"])
        color = int(obj["color"])
        target = coords.copy()
        target[:, 0] += area
        keep = (0 <= target[:, 0]) & (target[:, 0] < H) & (0 <= target[:, 1]) & (target[:, 1] < W)
        target = target[keep]
        if target.size == 0:
            continue
        rr = target[:, 0]
        cc = target[:, 1]
        assert not painted[rr, cc].any(), "shifted objects must not overlap"
        out[rr, cc] = color
        painted[rr, cc] = True
    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(sorted(cells), dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return arr


def _random_shape(area: int) -> np.ndarray:
    cells = {(0, 0)}
    frontier = {(0, 1), (1, 0), (0, -1), (-1, 0)}
    while len(cells) < area:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        r, c = cell
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt not in cells:
                frontier.add(nxt)
    arr = np.asarray(sorted(cells), dtype=int)
    for _ in range(random.randrange(4)):
        arr = np.column_stack((arr[:, 1], -arr[:, 0]))
    if random.random() < 0.5:
        arr[:, 1] *= -1
    if random.random() < 0.5:
        arr[:, 0] *= -1
    return _normalize_shape(set(map(tuple, arr)))


def _tile_exact_period(tile: np.ndarray) -> bool:
    for ph in (1, 2):
        ok = True
        for r in range(3):
            for c in range(3):
                if int(tile[r, c]) != int(tile[r % ph, c]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return False
    for pw in (1, 2):
        ok = True
        for r in range(3):
            for c in range(3):
                if int(tile[r, c]) != int(tile[r, c % pw]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return False
    return True


def _random_tile() -> np.ndarray:
    for _ in range(200):
        color_count = random.choice((2, 3))
        colors = random.sample(TILE_COLORS, color_count)
        tile = np.asarray([[random.choice(colors) for _ in range(3)] for _ in range(3)], dtype=int)
        if len(set(map(int, tile.flat))) < color_count:
            continue
        if not _tile_exact_period(tile):
            continue
        return tile
    raise RuntimeError("failed to build non-degenerate 3x3 tile")


def _touches_occupied(cells: np.ndarray, occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        rc = (int(r), int(c))
        if rc in occupied:
            return True
        for dr, dc in NEIGHBORS4:
            if (int(r) + dr, int(c) + dc) in occupied:
                return True
    return False


def _touches_occupied8(cells: np.ndarray, occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        rc = (int(r), int(c))
        if rc in occupied:
            return True
        for dr, dc in NEIGHBORS8:
            if (int(r) + dr, int(c) + dc) in occupied:
                return True
    return False


def _place_crossing_shape(
    shape: np.ndarray,
    H: int,
    W: int,
    r_cut: int,
    c_cut: int,
    occupied: set[tuple[int, int]],
) -> np.ndarray | None:
    shape_h = int(shape[:, 0].max()) + 1
    shape_w = int(shape[:, 1].max()) + 1
    candidates = []
    for top in range(0, H - shape_h + 1):
        for left in range(0, W - shape_w + 1):
            pts = shape + np.array([top, left])
            in_visible = pts[:, 0] < r_cut
            in_hidden_rows = pts[:, 0] >= r_cut
            in_visible_cols = pts[:, 1] < c_cut
            in_hidden_cols = pts[:, 1] >= c_cut
            crosses_row = bool(in_visible.any() and in_hidden_rows.any() and np.any(pts[:, 1] < c_cut))
            crosses_col = bool(in_visible_cols.any() and in_hidden_cols.any() and np.any(pts[:, 0] < r_cut))
            if not (crosses_row or crosses_col):
                continue
            if _touches_occupied(pts, occupied):
                continue
            candidates.append(pts)
    return random.choice(candidates) if candidates else None


def _place_shape(
    shape: np.ndarray,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
) -> np.ndarray | None:
    shape_h = int(shape[:, 0].max()) + 1
    shape_w = int(shape[:, 1].max()) + 1
    candidates = []
    for top in range(0, H - shape_h + 1):
        for left in range(0, W - shape_w + 1):
            pts = shape + np.array([top, left])
            if _touches_occupied(pts, occupied):
                continue
            candidates.append(pts)
    return random.choice(candidates) if candidates else None


def _object_visible_after_shift(coords: np.ndarray, area: int, H: int) -> bool:
    return bool(np.any(coords[:, 0] + area < H))


def _final_overlap(objs: list[dict], shape: tuple[int, int]) -> bool:
    H, W = shape
    seen = set()
    for obj in objs:
        area = int(obj["coords"].shape[0])
        for r, c in obj["coords"]:
            rr = int(r) + area
            cc = int(c)
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if (rr, cc) in seen:
                return True
            seen.add((rr, cc))
    return False


def _known_tile_masks(
    grid: np.ndarray, r_cut: int, c_cut: int
) -> tuple[np.ndarray, np.ndarray]:
    block = grid[:r_cut, :c_cut]
    known = np.isin(block, TILE_COLORS)
    return block, known


def _validate_case(inp: np.ndarray, expected: np.ndarray) -> bool:
    H, W = inp.shape
    if inp.ndim != 2 or expected.ndim != 2 or max(H, W) > 30 or max(expected.shape) > 30:
        return False

    r_cut, c_cut = _visible_bounds(inp)
    if r_cut < 6 or c_cut < 6:
        return False

    block, known = _known_tile_masks(inp, r_cut, c_cut)
    tile = _infer_tile_3x3(inp, r_cut, c_cut)
    if not _tile_exact_period(tile):
        return False

    periods = _candidate_periods(block, known)
    if not periods or periods[0] != (3, 3):
        return False
    if any(ph < 3 or pw < 3 for ph, pw in periods if (ph, pw) != (3, 3)):
        return False

    for tr in range(3):
        for tc in range(3):
            if not known[tr:r_cut:3, tc:c_cut:3].any():
                return False

    objects = _find_objects(inp)
    if not (2 <= len(objects) <= 4):
        return False
    areas = [int(obj["area"]) for obj in objects]
    if len(set(areas)) != len(areas):
        return False
    if min(areas) < 3 or max(areas) > 6:
        return False
    if any(not _object_visible_after_shift(obj["coords"], obj["area"], H) for obj in objects):
        return False
    if _final_overlap(objects, inp.shape):
        return False

    crossing = False
    for obj in objects:
        coords = obj["coords"]
        row_side = coords[:, 0] < r_cut
        col_side = coords[:, 1] < c_cut
        if bool(row_side.any() and (~row_side).any() and np.any(coords[:, 1] < c_cut)):
            crossing = True
        if bool(col_side.any() and (~col_side).any() and np.any(coords[:, 0] < r_cut)):
            crossing = True
    if not crossing:
        return False

    if np.array_equal(inp, expected):
        return False
    return np.array_equal(main(inp), expected)


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    r_cut_hi = min(VISIBLE_CUT_RANGE[1], H - 2)
    c_cut_hi = min(VISIBLE_CUT_RANGE[1], W - 2)
    r_cut = random.randint(VISIBLE_CUT_RANGE[0], r_cut_hi)
    c_cut = random.randint(VISIBLE_CUT_RANGE[0], c_cut_hi)

    tile = _random_tile()
    base = _full_background((H, W), tile)
    inp = np.full((H, W), PLACEHOLDER, dtype=int)
    inp[:r_cut, :c_cut] = base[:r_cut, :c_cut]

    obj_count = random.randint(*OBJECT_COUNT_RANGE)
    areas = random.sample(OBJECT_AREA_CHOICES, obj_count)
    random.shuffle(areas)
    colors = [random.choice(OBJECT_COLORS) for _ in range(obj_count)]
    shapes = [_random_shape(area) for area in areas]

    objects = []
    occupied: set[tuple[int, int]] = set()
    crossing_index = random.randrange(obj_count)

    for i, (area, color, shape) in enumerate(zip(areas, colors, shapes)):
        if i == crossing_index:
            coords = _place_crossing_shape(shape, H, W, r_cut, c_cut, occupied)
        else:
            coords = _place_shape(shape, H, W, occupied)
        if coords is None:
            return None
        if not _object_visible_after_shift(coords, area, H):
            return None
        objects.append({"color": int(color), "coords": coords})
        occupied.update(map(lambda p: (int(p[0]), int(p[1])), coords))

    if _final_overlap(objects, (H, W)):
        return None

    clip_count = 0
    for obj in objects:
        area = int(obj["coords"].shape[0])
        if np.any(obj["coords"][:, 0] + area >= H):
            clip_count += 1
    if obj_count >= 3 and random.random() < 0.45 and clip_count == 0:
        return None

    for obj in objects:
        inp[obj["coords"][:, 0], obj["coords"][:, 1]] = int(obj["color"])

    expected = main(inp)
    if not _validate_case(inp, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _generate_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate valid input for puzzle 6")
