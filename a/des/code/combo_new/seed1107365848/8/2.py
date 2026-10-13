import random
from collections import defaultdict, deque

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _norm_shape(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0, _, _ = _bbox(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _holes_in_mother(
    grid: np.ndarray, color: int, cells: list[tuple[int, int]], bg: int
) -> list[list[tuple[int, int]]] | None:
    r0, c0, r1, c1 = _bbox(cells)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]

    if np.any((sub != bg) & (sub != color)):
        return None
    if np.any(sub[0, :] != color) or np.any(sub[-1, :] != color):
        return None
    if np.any(sub[:, 0] != color) or np.any(sub[:, -1] != color):
        return None
    if not np.any(sub == bg):
        return None

    holes: list[list[tuple[int, int]]] = []
    for comp in _components_mask(sub == bg):
        if any(rr == 0 or cc == 0 or rr == sub.shape[0] - 1 or cc == sub.shape[1] - 1 for rr, cc in comp):
            return None
        holes.append([(r0 + rr, c0 + cc) for rr, cc in comp])
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered same-color fragments back into the rectangular bodies that contain matching holes.

    Input structure:
    - The background is the dominant color.
    - Several colors each form one large rectangular body whose bbox border is intact, but whose interior contains one or more background holes.
    - Additional small 4-connected fragments of those same colors are scattered away from the bodies and are separated from them by background.

    Rule:
    - For each color, find the large component whose bbox is a rectangle of that color with background hole regions inside it.
    - Normalize each hole shape by translation and match it with a scattered fragment of the same color having the same normalized shape.
    - Fill every matched hole with the body color so the body becomes a complete rectangle.
    - Erase the original matched fragment positions back to background.

    Key insight:
    - Fragment coordinates themselves are irrelevant; the only reliable link is shape equality between a body's internal hole and a scattered same-color component.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components_mask(g == color)
        fragment_by_shape: defaultdict[frozenset[tuple[int, int]], list[list[tuple[int, int]]]] = defaultdict(list)
        mothers: list[list[list[tuple[int, int]]]] = []

        for comp in comps:
            holes = _holes_in_mother(g, color, comp, bg)
            if holes is None:
                fragment_by_shape[_norm_shape(comp)].append(comp)
            else:
                mothers.append(holes)

        for holes in mothers:
            for hole in holes:
                shape = _norm_shape(hole)
                matches = fragment_by_shape.get(shape)
                assert matches, "each hole must have a same-color fragment with the same shape"
                fragment = matches.pop()
                for r, c in fragment:
                    out[r, c] = bg
                for r, c in hole:
                    out[r, c] = color

    return out


# --------------- generator ---------------

_CANVAS = 30
_ATTEMPTS = 400
_BODY_COUNT_CHOICES = [1, 2, 2, 3, 3, 4]
_RECT_H_RANGE = (5, 10)
_RECT_W_RANGE = (5, 10)
_HOLE_COUNT_CHOICES = [1, 1, 2, 2, 3]
_HOLE_AREA_CHOICES = [1, 2, 2, 3, 3, 4, 4, 5, 6]
_CROP_MARGIN_RANGE = (1, 6)


def _random_shape(area: int, rng: random.Random) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(rng.choice(frontier))

    if rng.random() < 0.5:
        cells = {(-r, c) for r, c in cells}
    if rng.random() < 0.5:
        cells = {(r, -c) for r, c in cells}
    if rng.random() < 0.5:
        cells = {(c, r) for r, c in cells}
    return _norm_shape(cells)


def _shape_dims(shape: frozenset[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _place_holes_in_rect(
    rect_h: int, rect_w: int, hole_count: int, rng: random.Random
) -> list[frozenset[tuple[int, int]]] | None:
    used_shapes: set[frozenset[tuple[int, int]]] = set()
    occupied: set[tuple[int, int]] = set()
    holes: list[frozenset[tuple[int, int]]] = []

    for _ in range(hole_count):
        placed = False
        for _ in range(80):
            area = rng.choice(_HOLE_AREA_CHOICES)
            shape = _random_shape(area, rng)
            if shape in used_shapes:
                continue
            sh, sw = _shape_dims(shape)
            if sh > rect_h - 2 or sw > rect_w - 2:
                continue

            positions = [
                (r0, c0)
                for r0 in range(1, rect_h - sh)
                for c0 in range(1, rect_w - sw)
            ]
            rng.shuffle(positions)
            for r0, c0 in positions:
                cells = frozenset((r0 + r, c0 + c) for r, c in shape)
                ok = True
                for r, c in cells:
                    if (r, c) in occupied:
                        ok = False
                        break
                    for nr, nc in _neighbors4(r, c):
                        if (nr, nc) in occupied:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                holes.append(cells)
                used_shapes.add(shape)
                occupied |= set(cells)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return holes


def _patch_from_cells(
    cells: frozenset[tuple[int, int]] | list[tuple[int, int]], color: int, bg: int
) -> np.ndarray:
    shape = _norm_shape(cells)
    h, w = _shape_dims(shape)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in shape:
        patch[r, c] = color
    return patch


def _can_place_bbox(reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    if r0 < 0 or c0 < 0 or r0 + h > reserved.shape[0] or c0 + w > reserved.shape[1]:
        return False
    return not reserved[r0 : r0 + h, c0 : c0 + w].any()


def _mark_bbox(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0], r0 + h + pad)
    cc1 = min(reserved.shape[1], c0 + w + pad)
    reserved[rr0:rr1, cc0:cc1] = True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    view = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    mask = patch != bg
    view[mask] = patch[mask]


def _non_bg_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != bg)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _crop_pair(inp: np.ndarray, out: np.ndarray, bg: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = _non_bg_bbox(inp, bg)
    top = min(r0, rng.randint(*_CROP_MARGIN_RANGE))
    left = min(c0, rng.randint(*_CROP_MARGIN_RANGE))
    bottom = min(inp.shape[0] - 1 - r1, rng.randint(*_CROP_MARGIN_RANGE))
    right = min(inp.shape[1] - 1 - c1, rng.randint(*_CROP_MARGIN_RANGE))
    rs = slice(r0 - top, r1 + bottom + 1)
    cs = slice(c0 - left, c1 + right + 1)
    return inp[rs, cs].copy(), out[rs, cs].copy()


def _analyze_shapes(grid: np.ndarray) -> tuple[int, dict[int, list[frozenset[tuple[int, int]]]], dict[int, list[frozenset[tuple[int, int]]]]]:
    bg = _bg_color(grid)
    hole_shapes: dict[int, list[frozenset[tuple[int, int]]]] = defaultdict(list)
    fragment_shapes: dict[int, list[frozenset[tuple[int, int]]]] = defaultdict(list)
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        comps = _components_mask(grid == color)
        for comp in comps:
            holes = _holes_in_mother(grid, color, comp, bg)
            if holes is None:
                fragment_shapes[color].append(_norm_shape(comp))
            else:
                for hole in holes:
                    hole_shapes[color].append(_norm_shape(hole))
    return bg, hole_shapes, fragment_shapes


def _valid_input(inp: np.ndarray, out: np.ndarray) -> bool:
    bg, hole_shapes, fragment_shapes = _analyze_shapes(inp)
    colors = sorted(hole_shapes)
    if not colors:
        return False
    for color in colors:
        if sorted(hole_shapes[color]) != sorted(fragment_shapes[color]):
            return False
    values, counts = np.unique(inp, return_counts=True)
    if int(values[np.argmax(counts)]) != bg:
        return False
    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, out))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randint(0, 9)
    colors = [c for c in range(10) if c != bg]
    rng.shuffle(colors)
    body_count = rng.choice(_BODY_COUNT_CHOICES)
    body_colors = colors[:body_count]

    bodies = []
    fragment_specs = []
    for color in body_colors:
        rect_h = rng.randint(*_RECT_H_RANGE)
        rect_w = rng.randint(*_RECT_W_RANGE)
        hole_count = rng.choice(_HOLE_COUNT_CHOICES)
        holes = _place_holes_in_rect(rect_h, rect_w, hole_count, rng)
        if holes is None:
            return None

        out_patch = np.full((rect_h, rect_w), color, dtype=int)
        inp_patch = out_patch.copy()
        inp_patch[:, :] = color
        for hole in holes:
            for r, c in hole:
                inp_patch[r, c] = bg
            fragment_specs.append((color, _patch_from_cells(hole, color, bg)))
        bodies.append((color, out_patch, inp_patch))

    inp = np.full((_CANVAS, _CANVAS), bg, dtype=int)
    out = np.full((_CANVAS, _CANVAS), bg, dtype=int)
    reserved = np.zeros((_CANVAS, _CANVAS), dtype=bool)

    body_order = list(range(len(bodies)))
    body_order.sort(key=lambda idx: bodies[idx][1].shape[0] * bodies[idx][1].shape[1], reverse=True)
    body_positions: dict[int, tuple[int, int]] = {}
    for idx in body_order:
        _color, out_patch, inp_patch = bodies[idx]
        h, w = out_patch.shape
        positions = [(r0, c0) for r0 in range(_CANVAS - h + 1) for c0 in range(_CANVAS - w + 1)]
        rng.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if not _can_place_bbox(reserved, r0, c0, h, w):
                continue
            body_positions[idx] = (r0, c0)
            _mark_bbox(reserved, r0, c0, h, w, pad=1)
            _paste_patch(inp, inp_patch, r0, c0, bg)
            _paste_patch(out, out_patch, r0, c0, bg)
            placed = True
            break
        if not placed:
            return None

    fragment_specs.sort(key=lambda item: int(np.count_nonzero(item[1] != bg)), reverse=True)
    for color, patch in fragment_specs:
        h, w = patch.shape
        positions = [(r0, c0) for r0 in range(_CANVAS - h + 1) for c0 in range(_CANVAS - w + 1)]
        rng.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if not _can_place_bbox(reserved, r0, c0, h, w):
                continue
            _mark_bbox(reserved, r0, c0, h, w, pad=1)
            _paste_patch(inp, patch, r0, c0, bg)
            placed = True
            break
        if not placed:
            return None

    inp, out = _crop_pair(inp, out, bg, rng)
    if not _valid_input(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
