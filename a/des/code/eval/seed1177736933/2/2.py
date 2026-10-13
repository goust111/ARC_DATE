import random
from typing import Iterable

import numpy as np


GREY = 5
NEIGH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


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
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGH4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows, cols = zip(*coords)
    return min(rows), min(cols), max(rows), max(cols)


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _mask_to_shape(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_to_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    out = np.zeros((max(rows) + 1, max(cols) + 1), dtype=bool)
    for r, c in shape:
        out[r, c] = True
    return out


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for cand in (rot, np.fliplr(rot)):
            trimmed = _trim(cand)
            key = (trimmed.tobytes(), trimmed.shape)
            if key in seen:
                continue
            seen.add(key)
            out.append(trimmed)
    return out


def _candidate_placements(
    cavity: frozenset[tuple[int, int]], shape: tuple[tuple[int, int], ...]
) -> list[frozenset[tuple[int, int]]]:
    placements: list[frozenset[tuple[int, int]]] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    cavity_cells = sorted(cavity)
    for cr, cc in cavity_cells:
        for sr, sc in shape:
            dr, dc = cr - sr, cc - sc
            shifted = frozenset((r + dr, c + dc) for r, c in shape)
            if shifted in seen:
                continue
            if shifted <= cavity:
                seen.add(shifted)
                placements.append(shifted)
    placements.sort(key=lambda cells: (min(cells), len(cells)))
    return placements


def _count_cover_solutions(
    cavity_coords: Iterable[tuple[int, int]],
    shapes: list[tuple[tuple[int, int], ...]],
    limit: int = 2,
) -> int:
    cavity = frozenset(_mask_to_shape(_coords_to_mask(cavity_coords)))
    if sum(len(shape) for shape in shapes) != len(cavity):
        return 0
    candidates = [_candidate_placements(cavity, shape) for shape in shapes]
    if any(not opts for opts in candidates):
        return 0
    memo: dict[tuple[frozenset[tuple[int, int]], tuple[int, ...]], int] = {}

    def rec(remaining: frozenset[tuple[int, int]], unused: tuple[int, ...]) -> int:
        if not unused:
            return 1 if not remaining else 0
        key = (remaining, unused)
        if key in memo:
            return memo[key]
        best_idx = -1
        best_valid: list[frozenset[tuple[int, int]]] | None = None
        for idx in unused:
            valid = [cells for cells in candidates[idx] if cells <= remaining]
            if not valid:
                memo[key] = 0
                return 0
            if best_valid is None or len(valid) < len(best_valid):
                best_idx = idx
                best_valid = valid
        assert best_valid is not None
        total = 0
        next_unused = tuple(i for i in unused if i != best_idx)
        for cells in best_valid:
            total += rec(frozenset(remaining.difference(cells)), next_unused)
            if total >= limit:
                memo[key] = limit
                return limit
        memo[key] = total
        return total

    return rec(cavity, tuple(range(len(shapes))))


def _can_cover_cavity(
    cavity_coords: Iterable[tuple[int, int]],
    shapes: list[tuple[tuple[int, int], ...]],
) -> bool:
    return _count_cover_solutions(cavity_coords, shapes, limit=1) >= 1


def _coords_to_mask(coords: Iterable[tuple[int, int]]) -> np.ndarray:
    coords = list(coords)
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _holes_inside_component(
    grid: np.ndarray, coords: list[tuple[int, int]], bg: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    wall = np.zeros(patch.shape, dtype=bool)
    for r, c in coords:
        wall[r - r0, c - c0] = True
    passable = patch == bg
    seen = np.zeros_like(passable, dtype=bool)
    stack: list[tuple[int, int]] = []
    ph, pw = passable.shape
    for r in range(ph):
        for c in (0, pw - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(pw):
        for r in (0, ph - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    while stack:
        cr, cc = stack.pop()
        for dr, dc in NEIGH4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < ph and 0 <= nc < pw and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    holes = np.argwhere(passable & ~seen)
    return [(int(r0 + r), int(c0 + c)) for r, c in holes]


def _find_molds(grid: np.ndarray, bg: int) -> tuple[list[dict], set[tuple[int, int]]]:
    molds: list[dict] = []
    mold_cells: set[tuple[int, int]] = set()
    for coords in _components(grid != bg):
        colors = {int(grid[r, c]) for r, c in coords}
        if GREY not in colors:
            continue
        holes = _holes_inside_component(grid, coords, bg)
        if not holes:
            continue
        markers = [(r, c) for r, c in coords if int(grid[r, c]) != GREY]
        assert len(markers) == 1, "each mold should contain exactly one colored border marker"
        mr, mc = markers[0]
        color = int(grid[mr, mc])
        assert color != bg and color != GREY
        molds.append({"coords": coords, "holes": holes, "color": color})
        mold_cells.update(coords)
    assert molds, "no molds found"
    colors = [mold["color"] for mold in molds]
    assert len(colors) == len(set(colors)), "marker colors must identify molds uniquely"
    return molds, mold_cells


def _fragments_for_color(
    grid: np.ndarray, color: int, excluded: set[tuple[int, int]]
) -> list[dict]:
    mask = grid == color
    for r, c in excluded:
        mask[r, c] = False
    frags: list[dict] = []
    for coords in _components(mask):
        local = _coords_to_mask(coords)
        frags.append({"coords": coords, "shape": _mask_to_shape(local)})
    assert frags, f"no fragments found for color {color}"
    return frags


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    molds, mold_cells = _find_molds(g, bg)

    out = g.copy()
    for mold in molds:
        color = int(mold["color"])
        fragments = _fragments_for_color(g, color, mold_cells)
        shapes = [frag["shape"] for frag in fragments]
        cavity = mold["holes"]
        assert _can_cover_cavity(cavity, shapes), "fragments must translate to cover the cavity"
        for frag in fragments:
            for r, c in frag["coords"]:
                out[r, c] = bg
        for r, c in cavity:
            out[r, c] = color
    return out


BASE_SHAPES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
]

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
MOLD_COUNT_CHOICES = [1, 2, 3]
FRAGMENT_COUNT_CHOICES = [1, 2, 3]
INTERIOR_H_RANGE = (4, 7)
INTERIOR_W_RANGE = (4, 7)
COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 8, 9]


def _reserve(blocked: np.ndarray, r: int, c: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(blocked.shape[0], r + h + pad)
    cc1 = min(blocked.shape[1], c + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _random_box_position(
    rng: random.Random, blocked: np.ndarray, h: int, w: int, pad: int
) -> tuple[int, int] | None:
    H, W = blocked.shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, H - h):
        for c in range(1, W - w):
            rr0 = max(0, r - pad)
            cc0 = max(0, c - pad)
            rr1 = min(H, r + h + pad)
            cc1 = min(W, c + w + pad)
            if not blocked[rr0:rr1, cc0:cc1].any():
                candidates.append((r, c))
    if not candidates:
        return None
    return rng.choice(candidates)


def _sample_distinct_shapes(rng: random.Random, count: int) -> list[np.ndarray] | None:
    for _ in range(80):
        shapes: list[np.ndarray] = []
        seen: set[tuple[bytes, tuple[int, int]]] = set()
        while len(shapes) < count:
            base = rng.choice(BASE_SHAPES)
            cand = rng.choice(_mask_variants(base))
            key = (cand.tobytes(), cand.shape)
            if key in seen:
                continue
            seen.add(key)
            shapes.append(cand)
        area = sum(int(shape.sum()) for shape in shapes)
        if 2 <= area <= 10:
            return shapes
    return None


def _union_component_count(coords: set[tuple[int, int]]) -> int:
    if not coords:
        return 0
    mask = _coords_to_mask(coords)
    return len(_components(mask))


def _place_shapes_in_interior(
    rng: random.Random,
    shapes: list[np.ndarray],
    interior_h: int,
    interior_w: int,
    want_connected: bool,
) -> list[set[tuple[int, int]]] | None:
    indexed = list(enumerate(shapes))
    indexed.sort(key=lambda item: -int(item[1].sum()))
    for _ in range(200):
        occupied = np.zeros((interior_h, interior_w), dtype=bool)
        placements_by_idx: list[set[tuple[int, int]] | None] = [None] * len(shapes)
        for idx, shape in indexed:
            sh, sw = shape.shape
            candidates: list[set[tuple[int, int]]] = []
            for r in range(interior_h - sh + 1):
                for c in range(interior_w - sw + 1):
                    cells = {
                        (r + int(rr), c + int(cc))
                        for rr, cc in np.argwhere(shape)
                    }
                    if any(occupied[rr, cc] for rr, cc in cells):
                        continue
                    candidates.append(cells)
            if not candidates:
                break
            rng.shuffle(candidates)
            placements_by_idx[idx] = candidates[0]
            for rr, cc in candidates[0]:
                occupied[rr, cc] = True
        if any(p is None for p in placements_by_idx):
            continue
        placements = [p for p in placements_by_idx if p is not None]
        union = set().union(*placements)
        comp_count = _union_component_count(union)
        if want_connected and comp_count != 1:
            continue
        if not want_connected and comp_count == 1:
            continue
        shape_keys = [_mask_to_shape(shape) for shape in shapes]
        if _count_cover_solutions(union, shape_keys, limit=2) != 1:
            continue
        return placements
    return None


def _marker_position(
    rng: random.Random, patch_h: int, patch_w: int
) -> tuple[int, int]:
    side = rng.choice(["top", "bottom", "left", "right"])
    if side == "top":
        return 0, rng.randrange(patch_w)
    if side == "bottom":
        return patch_h - 1, rng.randrange(patch_w)
    if side == "left":
        return rng.randrange(patch_h), 0
    return rng.randrange(patch_h), patch_w - 1


def _build_group(
    rng: random.Random, bg: int, color: int
) -> dict | None:
    for _ in range(200):
        fragment_count = rng.choice(FRAGMENT_COUNT_CHOICES)
        shapes = _sample_distinct_shapes(rng, fragment_count)
        if shapes is None:
            continue
        interior_h = rng.randint(*INTERIOR_H_RANGE)
        interior_w = rng.randint(*INTERIOR_W_RANGE)
        if any(shape.shape[0] > interior_h or shape.shape[1] > interior_w for shape in shapes):
            continue
        want_connected = True if fragment_count == 1 else rng.choice([True, False])
        placements = _place_shapes_in_interior(
            rng, shapes, interior_h, interior_w, want_connected
        )
        if placements is None:
            continue
        patch = np.full((interior_h + 2, interior_w + 2), GREY, dtype=int)
        cavity_coords: list[tuple[int, int]] = []
        for cells in placements:
            for r, c in cells:
                patch[r + 1, c + 1] = bg
                cavity_coords.append((r + 1, c + 1))
        mr, mc = _marker_position(rng, patch.shape[0], patch.shape[1])
        patch[mr, mc] = color
        return {
            "color": color,
            "patch": patch,
            "cavity_coords": cavity_coords,
            "shapes": shapes,
        }
    return None


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _paint_shape(
    grid: np.ndarray, shape: np.ndarray, color: int, top: int, left: int
) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for rr, cc in np.argwhere(shape):
        r = top + int(rr)
        c = left + int(cc)
        grid[r, c] = color
        coords.append((r, c))
    return coords


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(COLOR_POOL)
    available = [c for c in COLOR_POOL if c != bg]
    mold_count = rng.choice(MOLD_COUNT_CHOICES)
    if len(available) < mold_count:
        return None
    colors = rng.sample(available, mold_count)
    groups: list[dict] = []
    for color in colors:
        group = _build_group(rng, bg, color)
        if group is None:
            return None
        groups.append(group)

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    for group in sorted(groups, key=lambda g: -(g["patch"].shape[0] * g["patch"].shape[1])):
        patch = group["patch"]
        pos = _random_box_position(rng, blocked, patch.shape[0], patch.shape[1], pad=2)
        if pos is None:
            return None
        top, left = pos
        _paint_patch(grid, patch, top, left)
        _reserve(blocked, top, left, patch.shape[0], patch.shape[1], pad=2)
        group["mold_top_left"] = pos
        group["fragment_cells"] = []

    for group in groups:
        shapes = list(group["shapes"])
        rng.shuffle(shapes)
        for shape in sorted(shapes, key=lambda s: -int(s.sum())):
            pos = _random_box_position(rng, blocked, shape.shape[0], shape.shape[1], pad=1)
            if pos is None:
                return None
            top, left = pos
            group["fragment_cells"].extend(
                _paint_shape(grid, shape, group["color"], top, left)
            )
            _reserve(blocked, top, left, shape.shape[0], shape.shape[1], pad=1)

    expected = grid.copy()
    for group in groups:
        top, left = group["mold_top_left"]
        for r, c in group["cavity_coords"]:
            expected[top + r, left + c] = group["color"]
    for group in groups:
        for r, c in group["fragment_cells"]:
            expected[r, c] = bg

    if _get_bg(grid) != bg:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
