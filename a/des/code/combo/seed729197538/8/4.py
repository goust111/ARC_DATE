import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


MIN_GRID = 12
MAX_GRID = 18
MIN_COLORS = 2
MAX_COLORS = 4
PAIR_LENGTHS = (3, 4, 5, 6)
GENERATION_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    r0 = int(arr[:, 0].min())
    c0 = int(arr[:, 1].min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in arr))


def _shape_dims(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int]:
    arr = np.asarray(coords, dtype=int)
    return int(arr[:, 0].max()) + 1, int(arr[:, 1].max()) + 1


def _mask_from_coords(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_dims(coords)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _all_transforms(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    mask = _mask_from_coords(_normalize(coords))
    out: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for flip_lr in (False, True):
        base = np.fliplr(mask) if flip_lr else mask
        for k in range(4):
            transformed = np.rot90(base, k)
            cells = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(transformed)))
            if cells not in seen:
                seen.add(cells)
                out.append(cells)
    return out


def _is_l_shape(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> bool:
    if not 3 <= len(coords) <= 6:
        return False
    for transformed in _all_transforms(coords):
        h, w = _shape_dims(transformed)
        if h < 2 or w < 2:
            continue
        if h + w - 1 != len(coords):
            continue
        canonical = {(0, c) for c in range(w)} | {(r, 0) for r in range(h)}
        if set(transformed) == canonical:
            return True
    return False


def _placed(coords: tuple[tuple[int, int], ...], top: int, left: int) -> set[tuple[int, int]]:
    return {(int(r + top), int(c + left)) for r, c in coords}


def _rectangle_candidates(
    shape_a: tuple[tuple[int, int], ...],
    shape_b: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]]:
    total_area = len(shape_a) + len(shape_b)
    candidates: set[tuple[int, int]] = set()
    transforms_a = _all_transforms(shape_a)
    transforms_b = _all_transforms(shape_b)

    for h in range(1, total_area + 1):
        if total_area % h:
            continue
        w = total_area // h
        found = False
        for ta in transforms_a:
            ah, aw = _shape_dims(ta)
            if ah > h or aw > w:
                continue
            for tb in transforms_b:
                bh, bw = _shape_dims(tb)
                if bh > h or bw > w:
                    continue
                for ra in range(h - ah + 1):
                    for ca in range(w - aw + 1):
                        placed_a = _placed(ta, ra, ca)
                        for rb in range(h - bh + 1):
                            for cb in range(w - bw + 1):
                                placed_b = _placed(tb, rb, cb)
                                if placed_a.isdisjoint(placed_b):
                                    candidates.add((int(h), int(w)))
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break

    return sorted(candidates)


def _clamped_origin(anchor: tuple[int, int], rect_shape: tuple[int, int], grid_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = rect_shape
    gh, gw = grid_shape
    top = min(anchor[0], max(0, gh - h))
    left = min(anchor[1], max(0, gw - w))
    return int(top), int(left)


def _anchor_shift(anchor: tuple[int, int], rect_shape: tuple[int, int], grid_shape: tuple[int, int]) -> int:
    top, left = _clamped_origin(anchor, rect_shape, grid_shape)
    return int(anchor[0] - top + anchor[1] - left)


def _choose_rectangle_shape(
    candidates: list[tuple[int, int]],
    anchor: tuple[int, int],
    grid_shape: tuple[int, int],
) -> tuple[int, int]:
    assert candidates
    keyed = {}
    for shape in sorted(set(candidates)):
        keyed.setdefault((_anchor_shift(anchor, shape, grid_shape), shape[0], shape[1]), []).append(shape)
    best_key = min(keyed)
    best_shapes = keyed[best_key]
    assert len(best_shapes) == 1, "ambiguous rectangle orientation"
    return best_shapes[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored L fragments, assemble each pair into a rectangle, then place the rectangle at the same-colored anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _background_color(g)
    out = np.full((h, w), bg, dtype=int)

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components(g == color)
        assert len(comps) == 3, "each color must have two L pieces and one anchor"
        anchors = [comp for comp in comps if len(comp) == 1]
        pieces = [comp for comp in comps if len(comp) > 1]
        assert len(anchors) == 1 and len(pieces) == 2
        anchor = anchors[0][0]

        shapes = [_normalize(piece) for piece in pieces]
        assert all(_is_l_shape(shape) for shape in shapes), "non-anchor components must be L shapes"

        candidates = _rectangle_candidates(shapes[0], shapes[1])
        rect_h, rect_w = _choose_rectangle_shape(candidates, anchor, g.shape)
        top, left = _clamped_origin(anchor, (rect_h, rect_w), g.shape)

        block = out[top : top + rect_h, left : left + rect_w]
        assert np.all(block == bg), "assembled rectangles must not overlap"
        block[:] = color

    return out


def _canonical_pair(rect_shape: tuple[int, int]) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    h, w = rect_shape
    if h == 2:
        a = {(0, c) for c in range(w - 1)} | {(1, 0)}
        b = {(1, c) for c in range(1, w)} | {(0, w - 1)}
        return tuple(sorted(a)), tuple(sorted(b))
    assert w == 2
    a = {(r, 0) for r in range(h - 1)} | {(0, 1)}
    b = {(r, 1) for r in range(1, h)} | {(h - 1, 0)}
    return tuple(sorted(a)), tuple(sorted(b))


def _cells_fit(cells: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    gh, gw = occupied.shape
    for r, c in cells:
        if not (0 <= r < gh and 0 <= c < gw):
            return False
        if occupied[r, c]:
            return False
    for r, c in cells:
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < gh and 0 <= nc < gw and occupied[nr, nc]:
                return False
    return True


def _occupy(cells: list[tuple[int, int]], occupied: np.ndarray) -> None:
    for r, c in cells:
        occupied[r, c] = True


def _rect_cells(top: int, left: int, rect_shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = rect_shape
    return [(top + r, left + c) for r in range(h) for c in range(w)]


def _anchor_candidates(
    grid_shape: tuple[int, int],
    rect_shape: tuple[int, int],
    want_clipped: bool,
) -> list[tuple[int, int]]:
    gh, gw = grid_shape
    h, w = rect_shape
    swapped = (w, h)
    out: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []
    for r in range(gh):
        for c in range(gw):
            anchor = (r, c)
            desired_shift = _anchor_shift(anchor, rect_shape, grid_shape)
            swapped_shift = _anchor_shift(anchor, swapped, grid_shape)
            if desired_shift >= swapped_shift:
                continue
            fallback.append(anchor)
            if want_clipped and desired_shift > 0:
                out.append(anchor)
            if not want_clipped and desired_shift == 0:
                out.append(anchor)
    return out if out else fallback


def _place_transformed_shape(
    shape: tuple[tuple[int, int], ...],
    occupied: np.ndarray,
) -> list[tuple[int, int]] | None:
    gh, gw = occupied.shape
    sh, sw = _shape_dims(shape)
    positions = [(r, c) for r in range(gh - sh + 1) for c in range(gw - sw + 1)]
    random.shuffle(positions)
    for top, left in positions:
        cells = [(top + r, left + c) for r, c in shape]
        if _cells_fit(cells, occupied):
            return cells
    return None


def _gen_attempt() -> np.ndarray | None:
    gh = random.randint(MIN_GRID, MAX_GRID)
    gw = random.randint(MIN_GRID, MAX_GRID)
    n_colors = random.randint(MIN_COLORS, MAX_COLORS)
    palette = random.sample(range(10), n_colors + 1)
    bg = palette[0]
    colors = palette[1:]

    input_grid = np.full((gh, gw), bg, dtype=int)
    expected = np.full((gh, gw), bg, dtype=int)
    input_occ = np.zeros((gh, gw), dtype=bool)
    output_occ = np.zeros((gh, gw), dtype=bool)
    specs: list[dict] = []

    for color in colors:
        length = random.choice(PAIR_LENGTHS)
        rect_shape = (2, length) if random.random() < 0.5 else (length, 2)
        want_clipped = random.random() < 0.5
        anchors = _anchor_candidates((gh, gw), rect_shape, want_clipped)
        random.shuffle(anchors)

        placed = False
        for anchor in anchors:
            if not _cells_fit([anchor], input_occ):
                continue
            top, left = _clamped_origin(anchor, rect_shape, (gh, gw))
            rect_cells = _rect_cells(top, left, rect_shape)
            if not _cells_fit(rect_cells, output_occ):
                continue
            _occupy([anchor], input_occ)
            _occupy(rect_cells, output_occ)
            input_grid[anchor] = color
            expected[top : top + rect_shape[0], left : left + rect_shape[1]] = color
            specs.append(
                {
                    "color": color,
                    "rect_shape": rect_shape,
                    "anchor": anchor,
                }
            )
            placed = True
            break
        if not placed:
            return None

    random.shuffle(specs)
    for spec in specs:
        piece_a, piece_b = _canonical_pair(spec["rect_shape"])
        variants_a = _all_transforms(piece_a)
        variants_b = _all_transforms(piece_b)
        random.shuffle(variants_a)
        random.shuffle(variants_b)

        choice_a = random.choice(variants_a)
        choice_b = random.choice(variants_b)
        cells_a = _place_transformed_shape(choice_a, input_occ)
        if cells_a is None:
            return None
        _occupy(cells_a, input_occ)
        for r, c in cells_a:
            input_grid[r, c] = spec["color"]

        cells_b = _place_transformed_shape(choice_b, input_occ)
        if cells_b is None:
            return None
        _occupy(cells_b, input_occ)
        for r, c in cells_b:
            input_grid[r, c] = spec["color"]

    if _background_color(input_grid) != bg:
        return None

    try:
        got = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
