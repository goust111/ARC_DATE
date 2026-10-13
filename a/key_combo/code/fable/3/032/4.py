from __future__ import annotations

import random

import numpy as np


HOLE = 0
NON_BLACK_COLORS = list(range(1, 10))
UNDO_ROTATIONS = (0, 3, 2, 1)  # TL, TR, BR, BL -> base orientation
REDO_ROTATIONS = (0, 1, 2, 3)
GENERATION_RETRIES = 300
QUADRANT_SIDE_RANGE = (6, 9)
PALETTE_SIZE_RANGE = (3, 5)
PLACEMENT_RETRIES = 60
BASE_PATTERN_RETRIES = 120


def _split_quadrants(grid: np.ndarray) -> list[np.ndarray]:
    n = grid.shape[0] // 2
    return [
        grid[:n, :n],
        grid[:n, n:],
        grid[n:, n:],
        grid[n:, :n],
    ]


def _normalize_quadrants(grid: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(quad, k) for quad, k in zip(_split_quadrants(grid), UNDO_ROTATIONS)]


def _assemble_quadrants(quadrants: list[np.ndarray]) -> np.ndarray:
    restored = [np.rot90(quad, k) for quad, k in zip(quadrants, REDO_ROTATIONS)]
    top = np.hstack([restored[0], restored[1]])
    bottom = np.hstack([restored[3], restored[2]])
    return np.vstack([top, bottom])


def _apply_color_map(arr: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = arr.copy()
    for src, dst in mapping.items():
        out[arr == src] = dst
    return out


def _assign_mapping(mapping: dict[int, int], src: int, dst: int) -> bool:
    current = mapping.get(src)
    if current is not None:
        if current != dst:
            raise AssertionError(f"inconsistent mapping for {src}: {current} vs {dst}")
        return False
    if dst in mapping.values():
        raise AssertionError(f"target color {dst} reused in one quadrant map")
    mapping[src] = dst
    return True


def _propagate_bijection_completion(palette: list[int], maps: list[dict[int, int]]) -> None:
    changed = True
    while changed:
        changed = False
        for qi in range(1, 4):
            remaining_src = [color for color in palette if color not in maps[qi]]
            remaining_dst = [color for color in palette if color not in maps[qi].values()]
            if len(remaining_src) == 1 and len(remaining_dst) == 1:
                changed |= _assign_mapping(maps[qi], remaining_src[0], remaining_dst[0])


def _orbit_is_feasible(observed: tuple[int, int, int, int], palette: list[int], maps: list[dict[int, int]]) -> bool:
    if all(value == HOLE for value in observed):
        return False

    source_candidates = [observed[0]] if observed[0] != HOLE else palette
    for source in source_candidates:
        ok = True
        for qi in range(1, 4):
            value = observed[qi]
            if value == HOLE:
                continue
            mapped = maps[qi].get(source)
            if mapped is not None:
                if mapped != value:
                    ok = False
                    break
            elif value in maps[qi].values():
                ok = False
                break
        if ok:
            return True
    return False


def _state_is_feasible(norms: list[np.ndarray], palette: list[int], maps: list[dict[int, int]]) -> bool:
    n = norms[0].shape[0]
    for r in range(n):
        for c in range(n):
            observed = tuple(int(norms[qi][r, c]) for qi in range(4))
            if not _orbit_is_feasible(observed, palette, maps):
                return False
    return True


def _search_maps(norms: list[np.ndarray], palette: list[int], maps: list[dict[int, int]]) -> list[dict[int, int]] | None:
    _propagate_bijection_completion(palette, maps)
    if not _state_is_feasible(norms, palette, maps):
        return None

    best_var: tuple[int, int] | None = None
    best_candidates: list[int] | None = None
    for qi in range(1, 4):
        remaining_dst = [color for color in palette if color not in maps[qi].values()]
        for src in palette:
            if src in maps[qi]:
                continue
            candidates = remaining_dst
            if best_candidates is None or len(candidates) < len(best_candidates):
                best_var = (qi, src)
                best_candidates = candidates

    if best_var is None:
        return maps

    qi, src = best_var
    assert best_candidates is not None
    for dst in best_candidates:
        next_maps = [mapping.copy() for mapping in maps]
        try:
            _assign_mapping(next_maps[qi], src, dst)
        except AssertionError:
            continue
        solved = _search_maps(norms, palette, next_maps)
        if solved is not None:
            return solved
    return None


def _recover_maps(norms: list[np.ndarray], palette: list[int]) -> list[dict[int, int]]:
    maps: list[dict[int, int]] = [{color: color for color in palette}, {}, {}, {}]
    n = norms[0].shape[0]

    for qi in range(1, 4):
        for r in range(n):
            for c in range(n):
                src = int(norms[0][r, c])
                dst = int(norms[qi][r, c])
                if src != HOLE and dst != HOLE:
                    _assign_mapping(maps[qi], src, dst)

    solved = _search_maps(norms, palette, maps)
    if solved is None:
        raise AssertionError("failed to recover the quadrant color bijections")
    return solved


def _repair_normalized(norms: list[np.ndarray], maps: list[dict[int, int]]) -> list[np.ndarray]:
    repaired = [arr.copy() for arr in norms]
    inverse_maps = [{value: key for key, value in mapping.items()} for mapping in maps]
    n = norms[0].shape[0]

    for r in range(n):
        for c in range(n):
            observed = [int(norms[qi][r, c]) for qi in range(4)]
            if observed[0] != HOLE:
                base_color = observed[0]
            else:
                inferred = {
                    inverse_maps[qi][observed[qi]]
                    for qi in range(1, 4)
                    if observed[qi] != HOLE
                }
                if len(inferred) != 1:
                    raise AssertionError("hole orbit does not determine a unique base color")
                base_color = next(iter(inferred))

            for qi in range(4):
                expected = base_color if qi == 0 else maps[qi][base_color]
                if observed[qi] == HOLE:
                    repaired[qi][r, c] = expected
                elif observed[qi] != expected:
                    raise AssertionError("observed cell conflicts with recovered color maps")

    return repaired


def main(grid: np.ndarray) -> np.ndarray:
    """Repair a four-quadrant rotational mosaic whose rotated copies were recolored independently and partly erased.

    Input structure:
    - The grid is a square of even side 2n, split into four n×n quadrants.
    - The top-left quadrant is the base pattern; the other quadrants are its 90°, 180°, and 270° counterclockwise rotations.
    - Each non-top-left quadrant applies one fixed bijective recoloring to its rotated copy, and black cells mark holes rather than genuine pattern pixels.

    Rule:
    - Rotate every quadrant back to the top-left orientation so aligned cells form one rotation orbit.
    - Recover the three recoloring bijections from intact aligned cells and the fact that each quadrant uses one global permutation.
    - For every hole, infer the orbit's base color from any intact mate, reapply the correct quadrant recoloring, and fill the missing cell.

    Key insight:
    - The geometry is rotationally shared but the palettes are scrambled per quadrant, so the hidden symmetry appears only after factoring out one color permutation per quadrant.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or g.shape[0] % 2 != 0:
        raise AssertionError("expected an even square grid")

    palette = sorted(int(color) for color in np.unique(g) if int(color) != HOLE)
    if not palette:
        raise AssertionError("expected at least one visible non-black color")

    norms = _normalize_quadrants(g)
    maps = _recover_maps(norms, palette)
    repaired = _repair_normalized(norms, maps)
    return _assemble_quadrants(repaired)


def _random_feature_mask(side: int) -> np.ndarray:
    max_dim = min(4, side - 1)
    kind = random.choice(["rect", "line", "el", "plus", "corner"])

    if kind == "rect":
        h = random.randint(1, min(3, max_dim))
        w = random.randint(1, min(3, max_dim))
        if h * w == 1:
            if h == 1 and max_dim >= 2:
                w = 2
            else:
                h = 2
        return np.ones((h, w), dtype=bool)

    if kind == "line":
        length = random.randint(2, max_dim)
        if random.random() < 0.5:
            return np.ones((1, length), dtype=bool)
        return np.ones((length, 1), dtype=bool)

    if kind == "el":
        h = random.randint(2, max_dim)
        w = random.randint(2, max_dim)
        mask = np.zeros((h, w), dtype=bool)
        mask[:, 0] = True
        mask[-1, :] = True
        return np.rot90(mask, random.randrange(4))

    if kind == "plus" and side >= 3:
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, :] = True
        mask[:, 1] = True
        return mask

    mask = np.ones((2, 2), dtype=bool)
    mask[random.randrange(2), random.randrange(2)] = False
    return np.rot90(mask, random.randrange(4))


def _place_feature(canvas: np.ndarray, occupied: np.ndarray, color: int, mask: np.ndarray) -> bool:
    h, w = mask.shape
    side = canvas.shape[0]
    candidates = [(r, c) for r in range(side - h + 1) for c in range(side - w + 1)]
    random.shuffle(candidates)

    for top, left in candidates[:PLACEMENT_RETRIES]:
        r0 = max(0, top - 1)
        c0 = max(0, left - 1)
        r1 = min(side, top + h + 1)
        c1 = min(side, left + w + 1)
        if occupied[r0:r1, c0:c1].any():
            continue
        window = canvas[top : top + h, left : left + w]
        window[mask] = color
        occupied[top : top + h, left : left + w][mask] = True
        return True
    return False


def _base_pattern_is_good(base: np.ndarray, palette: list[int], base_color: int) -> bool:
    if any(not np.any(base == color) for color in palette):
        return False

    accent_mask = base != base_color
    accent_cells = int(accent_mask.sum())
    if accent_cells < base.shape[0] or accent_cells > int(base.size * 0.55):
        return False

    for k in (1, 2, 3):
        if np.array_equal(accent_mask, np.rot90(accent_mask, k)):
            return False
    return True


def _generate_base_quadrant(side: int, palette: list[int]) -> np.ndarray | None:
    base_color = palette[0]
    accents = palette[1:]

    for _ in range(BASE_PATTERN_RETRIES):
        base = np.full((side, side), base_color, dtype=int)
        occupied = np.zeros((side, side), dtype=bool)
        feature_colors = accents + random.choices(accents, k=random.randint(2, len(accents) + 3))
        random.shuffle(feature_colors)

        ok = True
        for color in feature_colors:
            mask = _random_feature_mask(side)
            if not _place_feature(base, occupied, color, mask):
                ok = False
                break
        if ok and _base_pattern_is_good(base, palette, base_color):
            return base
    return None


def _random_permutation(palette: list[int], *, avoid: tuple[tuple[int, ...], ...]) -> dict[int, int] | None:
    for _ in range(40):
        perm = palette[:]
        random.shuffle(perm)
        perm_tuple = tuple(perm)
        if perm_tuple in avoid:
            continue
        return dict(zip(palette, perm))
    return None


def _choose_witnesses(base: np.ndarray, palette: list[int]) -> set[tuple[int, int]]:
    witnesses: set[tuple[int, int]] = set()
    for color in palette:
        coords = [tuple(map(int, coord)) for coord in np.argwhere(base == color)]
        witnesses.add(random.choice(coords))
    return witnesses


def _make_hole_masks(side: int, witnesses: set[tuple[int, int]]) -> list[np.ndarray]:
    holes = [np.zeros((side, side), dtype=bool) for _ in range(4)]

    for r in range(side):
        for c in range(side):
            if (r, c) in witnesses:
                continue
            roll = random.random()
            if roll < 0.38:
                hole_count = 1
            elif roll < 0.58:
                hole_count = 2
            elif roll < 0.64:
                hole_count = 3
            else:
                hole_count = 0
            if hole_count == 0:
                continue
            for qi in random.sample(range(4), hole_count):
                holes[qi][r, c] = True

    available = [(r, c) for r in range(side) for c in range(side) if (r, c) not in witnesses]
    if not any(mask.any() for mask in holes):
        r, c = random.choice(available)
        holes[random.randrange(4)][r, c] = True

    for qi in range(4):
        if holes[qi].any():
            continue
        random.shuffle(available)
        for r, c in available:
            if sum(int(mask[r, c]) for mask in holes) < 3:
                holes[qi][r, c] = True
                break
    return holes


def _build_input_and_output() -> tuple[np.ndarray, np.ndarray] | None:
    side = random.randint(*QUADRANT_SIDE_RANGE)
    palette_size = random.randint(*PALETTE_SIZE_RANGE)
    palette = random.sample(NON_BLACK_COLORS, palette_size)

    base = _generate_base_quadrant(side, palette)
    if base is None:
        return None

    perms: list[dict[int, int]] = [{color: color for color in palette}]
    used = {tuple(palette)}
    for _ in range(3):
        perm = _random_permutation(palette, avoid=tuple(used))
        if perm is None:
            return None
        perm_tuple = tuple(perm[color] for color in palette)
        used.add(perm_tuple)
        perms.append(perm)

    full_norms = [_apply_color_map(base, perm) for perm in perms]
    expected = _assemble_quadrants(full_norms)

    witnesses = _choose_witnesses(base, palette)
    holes = _make_hole_masks(side, witnesses)
    input_norms = [arr.copy() for arr in full_norms]
    for qi in range(4):
        input_norms[qi][holes[qi]] = HOLE
    inp = _assemble_quadrants(input_norms)

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        sample = _build_input_and_output()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid rotational recoloring repair instance")
