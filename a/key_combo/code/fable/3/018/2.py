from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
COLORS = tuple(range(1, 10))
AXIS_DIFF = "diff"  # r - c = k
AXIS_SUM = "sum"    # r + c = k


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(grid: np.ndarray) -> list[dict]:
    H, W = map(int, grid.shape)
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []
    for sr, sc in np.argwhere(grid != BACKGROUND):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        pixels: list[tuple[int, int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            pixels.append((r, c, int(grid[r, c])))
            for rr, cc in _neighbors8(r, c):
                if 0 <= rr < H and 0 <= cc < W and not seen[rr, cc] and int(grid[rr, cc]) != BACKGROUND:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        r0, c0, r1, c1 = _bbox(coords)
        comps.append(
            {
                "coords": coords,
                "pixels": pixels,
                "colors": {col for _r, _c, col in pixels},
                "bbox": (r0, c0, r1, c1),
                "area": len(coords),
            }
        )
    return comps


def _swap_for(colors: set[int] | tuple[int, int]) -> dict[int, int]:
    a, b = tuple(colors)
    return {int(a): int(b), int(b): int(a)}


def _reflect_coord(r: int, c: int, axis: tuple[str, int]) -> tuple[int, int]:
    kind, k = axis
    if kind == AXIS_DIFF:
        return c + k, r - k
    return k - c, k - r


def _side_value(r: int, c: int, axis: tuple[str, int]) -> int:
    kind, k = axis
    if kind == AXIS_DIFF:
        return r - c - k
    return r + c - k


def _translation_signature(comp: dict) -> tuple[tuple[int, int, int], ...]:
    r0, c0, _r1, _c1 = comp["bbox"]
    return tuple(sorted((r - r0, c - c0, col) for r, c, col in comp["pixels"]))


def _pair_matches_axis(comp_a: dict, comp_b: dict, axis: tuple[str, int], swap: dict[int, int]) -> bool:
    b_pixels = {(r, c): col for r, c, col in comp_b["pixels"]}
    seen: set[tuple[int, int]] = set()
    for r, c, col in comp_a["pixels"]:
        rr, cc = _reflect_coord(r, c, axis)
        if b_pixels.get((rr, cc)) != swap[int(col)]:
            return False
        seen.add((rr, cc))
    return len(seen) == len(b_pixels)


def _mirror_axes(comp_a: dict, comp_b: dict) -> list[tuple[str, int]]:
    if comp_a["area"] != comp_b["area"]:
        return []
    if len(comp_a["colors"]) != 2 or comp_a["colors"] != comp_b["colors"]:
        return []

    swap = _swap_for(comp_a["colors"])
    seed_r, seed_c, seed_col = comp_a["pixels"][0]
    target_pixels = [(r, c) for r, c, col in comp_b["pixels"] if int(col) == swap[int(seed_col)]]
    axes: set[tuple[str, int]] = set()
    for tr, tc in target_pixels:
        k = int(tr - seed_c)
        if int(tc) == int(seed_r - k) and _pair_matches_axis(comp_a, comp_b, (AXIS_DIFF, k), swap):
            axes.add((AXIS_DIFF, k))

        k = int(tr + seed_c)
        if int(tc) == int(k - seed_r) and _pair_matches_axis(comp_a, comp_b, (AXIS_SUM, k), swap):
            axes.add((AXIS_SUM, k))
    return list(axes)


def _find_rule_parts(grid: np.ndarray):
    comps = _components(grid)
    assert len(comps) >= 5, "expected true twins, translation decoy, and at least one lone object"
    for comp in comps:
        assert len(comp["colors"]) == 2, "every visible object must be bichrome"

    mirror_candidates: list[tuple[int, int, tuple[str, int]]] = []
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            for axis in _mirror_axes(comps[i], comps[j]):
                mirror_candidates.append((i, j, axis))
    assert len(mirror_candidates) == 1, "expected one swapped-color diagonal mirror twin pair"

    true_i, true_j, axis = mirror_candidates[0]
    true_pair = {true_i, true_j}

    decoys: list[tuple[int, int]] = []
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if i in true_pair or j in true_pair:
                continue
            if _translation_signature(comps[i]) == _translation_signature(comps[j]):
                decoys.append((i, j))
    assert len(decoys) == 1, "expected one unchanged-color translated decoy pair"

    decoy_pair = set(decoys[0])
    lone_indices = [i for i in range(len(comps)) if i not in true_pair and i not in decoy_pair]
    assert lone_indices, "expected at least one lone object"
    return comps, axis, true_pair, decoy_pair, lone_indices


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden diagonal mirror from the true twin pair and reflect every lone bichrome object with its own colors swapped.

    Input structure:
    - A black background contains no drawn separator line.
    - Exactly one pair of bichrome objects is related by a 45-degree diagonal reflection and a swap of that pair's two colors.
    - Exactly one other pair is a pure translated copy with unchanged colors, and the remaining bichrome objects are unpaired lones.

    Rule:
    - Find the swapped-color mirror twin pair and derive its unique hidden diagonal axis, either r-c=k or r+c=k in (row, col) coordinates.
    - Preserve all input pixels, including the true twins and the translated decoy pair.
    - For each lone object, reflect its pixels across that same axis and draw the copy with that lone object's own two colors exchanged.

    Key insight:
    - The translated pair is a decoy; only the mirror pair defines the invisible axis, and the color operation is an abstract within-object swap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = map(int, g.shape)

    comps, axis, _true_pair, _decoy_pair, lone_indices = _find_rule_parts(g)
    out = g.copy()
    for idx in lone_indices:
        comp = comps[idx]
        swap = _swap_for(comp["colors"])
        for r, c, col in comp["pixels"]:
            rr, cc = _reflect_coord(r, c, axis)
            assert 0 <= rr < H and 0 <= cc < W, "lone reflection must stay inside the grid"
            assert int(g[rr, cc]) == BACKGROUND, "lone reflection target must be background in the input"
            assert int(out[rr, cc]) == BACKGROUND, "lone reflection copies must not collide"
            out[rr, cc] = swap[int(col)]
    return out


# ----------------------------- generator -----------------------------

_GENERATION_COUNTER = 0


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    out: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if any(0 <= rr < h and 0 <= cc < w and mask[rr, cc] for rr, cc in _neighbors8(r, c)):
                out.append((r, c))
    return out


def _grow_mask(h: int, w: int, count: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[random.randrange(h), random.randrange(w)] = True
    while int(mask.sum()) < count:
        cells = _frontier(mask)
        if not cells:
            break
        r, c = random.choice(cells)
        mask[r, c] = True
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _sample_patch(colors: tuple[int, int]) -> np.ndarray:
    for _ in range(200):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        count = random.randint(5, min(h * w - 1, 10))
        mask = _grow_mask(h, w, count)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        if len(coords) < 4:
            continue
        patch = np.zeros(mask.shape, dtype=int)
        if random.random() < 0.55:
            axis_i = random.randrange(4)
            keys = [
                lambda p: p[0],
                lambda p: p[1],
                lambda p: p[0] + p[1],
                lambda p: p[0] - p[1],
            ]
            coords.sort(key=keys[axis_i])
        else:
            random.shuffle(coords)
        cut = random.randint(1, len(coords) - 1)
        for r, c in coords[:cut]:
            patch[r, c] = colors[0]
        for r, c in coords[cut:]:
            patch[r, c] = colors[1]
        if len(set(map(int, patch[patch != 0]))) == 2:
            return patch
    raise RuntimeError("failed to sample a bichrome patch")


def _patch_pixels(patch: np.ndarray, r0: int, c0: int) -> list[tuple[int, int, int]]:
    return [
        (int(r0 + r), int(c0 + c), int(patch[r, c]))
        for r, c in np.argwhere(patch != BACKGROUND)
    ]


def _coords_of(pixels: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    return [(r, c) for r, c, _col in pixels]


def _reflect_pixels(
    pixels: list[tuple[int, int, int]], axis: tuple[str, int], colors: tuple[int, int]
) -> list[tuple[int, int, int]]:
    swap = {colors[0]: colors[1], colors[1]: colors[0]}
    out = []
    for r, c, col in pixels:
        rr, cc = _reflect_coord(r, c, axis)
        out.append((rr, cc, swap[int(col)]))
    return out


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in coords)


def _one_side(coords: list[tuple[int, int]], axis: tuple[str, int], min_abs: int = 2) -> bool:
    vals = [_side_value(r, c, axis) for r, c in coords]
    return all(v >= min_abs for v in vals) or all(v <= -min_abs for v in vals)


def _touches_occupied(coords: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    H, W = map(int, occupied.shape)
    for r, c in coords:
        if not (0 <= r < H and 0 <= c < W):
            return True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and occupied[rr, cc]:
                    return True
    return False


def _sets_are_separate(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    bset = set(b)
    for r, c in a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in bset:
                    return False
    return True


def _mark(occupied: np.ndarray, coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        occupied[int(r), int(c)] = True


def _paint(grid: np.ndarray, pixels: list[tuple[int, int, int]]) -> None:
    for r, c, col in pixels:
        grid[int(r), int(c)] = int(col)


def _sample_axis(H: int, W: int, kind: str) -> tuple[str, int] | None:
    if kind == AXIS_DIFF:
        lo, hi = 8 - W, H - 9
    else:
        lo, hi = 8, H + W - 10
    if lo > hi:
        return None
    return kind, random.randint(lo, hi)


def _place_reflected_patch(
    H: int,
    W: int,
    patch: np.ndarray,
    colors: tuple[int, int],
    axis: tuple[str, int],
    occupied_out: np.ndarray,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]] | None:
    ph, pw = map(int, patch.shape)
    for _ in range(900):
        r0 = random.randint(0, H - ph)
        c0 = random.randint(0, W - pw)
        src = _patch_pixels(patch, r0, c0)
        src_coords = _coords_of(src)
        if not _one_side(src_coords, axis):
            continue
        ref = _reflect_pixels(src, axis, colors)
        ref_coords = _coords_of(ref)
        if not _in_bounds(ref_coords, H, W):
            continue
        if not _sets_are_separate(src_coords, ref_coords):
            continue
        if _touches_occupied(src_coords, occupied_out):
            continue
        if _touches_occupied(ref_coords, occupied_out):
            continue
        return src, ref
    return None


def _place_plain_patch(
    H: int,
    W: int,
    patch: np.ndarray,
    occupied_out: np.ndarray,
) -> list[tuple[int, int, int]] | None:
    ph, pw = map(int, patch.shape)
    for _ in range(600):
        r0 = random.randint(0, H - ph)
        c0 = random.randint(0, W - pw)
        pixels = _patch_pixels(patch, r0, c0)
        coords = _coords_of(pixels)
        if not _touches_occupied(coords, occupied_out):
            return pixels
    return None


def _semantic_ok(inp: np.ndarray, expected: np.ndarray, axis: tuple[str, int], lone_count: int) -> bool:
    try:
        comps, found_axis, _true_pair, _decoy_pair, lone_indices = _find_rule_parts(inp)
        if found_axis != axis:
            return False
        if len(lone_indices) != lone_count:
            return False
        if len(comps) != lone_count + 4:
            return False
        return np.array_equal(main(inp), expected)
    except Exception:
        return False


def _gen_attempt(axis_kind: str, lone_count: int) -> np.ndarray | None:
    H = random.randint(22, 30)
    W = random.randint(22, 30)
    axis = _sample_axis(H, W, axis_kind)
    if axis is None:
        return None

    palette = list(random.sample(COLORS, len(COLORS)))
    true_colors = tuple(palette[0:2])
    decoy_colors = tuple(palette[2:4])
    lone_pairs = [tuple(palette[4:6]), tuple(palette[6:8])]
    if lone_count == 3:
        lone_pairs.append((palette[8], palette[4]))
    lone_pairs = [pair if random.random() < 0.5 else (pair[1], pair[0]) for pair in lone_pairs]

    grid = np.full((H, W), BACKGROUND, dtype=int)
    occupied_in = np.zeros((H, W), dtype=bool)
    occupied_out = np.zeros((H, W), dtype=bool)

    true_patch = _sample_patch(true_colors)
    true_pair = _place_reflected_patch(H, W, true_patch, true_colors, axis, occupied_out)
    if true_pair is None:
        return None
    true_a, true_b = true_pair
    _paint(grid, true_a)
    _paint(grid, true_b)
    true_coords = _coords_of(true_a) + _coords_of(true_b)
    _mark(occupied_in, true_coords)
    _mark(occupied_out, true_coords)

    decoy_patch = _sample_patch(decoy_colors)
    decoy_a = _place_plain_patch(H, W, decoy_patch, occupied_out)
    if decoy_a is None:
        return None
    temp_occupied = occupied_out.copy()
    _mark(temp_occupied, _coords_of(decoy_a))
    decoy_b = _place_plain_patch(H, W, decoy_patch, temp_occupied)
    if decoy_b is None:
        return None
    _paint(grid, decoy_a)
    _paint(grid, decoy_b)
    decoy_coords = _coords_of(decoy_a) + _coords_of(decoy_b)
    _mark(occupied_in, decoy_coords)
    _mark(occupied_out, decoy_coords)

    lone_reflections: list[list[tuple[int, int, int]]] = []
    for colors in lone_pairs:
        patch = _sample_patch(colors)
        placed = _place_reflected_patch(H, W, patch, colors, axis, occupied_out)
        if placed is None:
            return None
        src, ref = placed
        _paint(grid, src)
        src_coords = _coords_of(src)
        ref_coords = _coords_of(ref)
        _mark(occupied_in, src_coords)
        _mark(occupied_out, src_coords)
        _mark(occupied_out, ref_coords)
        lone_reflections.append(ref)

    expected = grid.copy()
    for ref in lone_reflections:
        _paint(expected, ref)

    if not _semantic_ok(grid, expected, axis, lone_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GENERATION_COUNTER

    forced_kind = AXIS_DIFF if _GENERATION_COUNTER % 2 == 0 else AXIS_SUM
    forced_lones = 2 if (_GENERATION_COUNTER // 2) % 2 == 0 else 3
    for _ in range(500):
        grid = _gen_attempt(forced_kind, forced_lones)
        if grid is not None:
            _GENERATION_COUNTER += 1
            return grid

    for _ in range(500):
        grid = _gen_attempt(random.choice([AXIS_DIFF, AXIS_SUM]), random.choice([2, 3]))
        if grid is not None:
            _GENERATION_COUNTER += 1
            return grid
    raise RuntimeError("failed to generate a valid hidden diagonal mirror task input")
