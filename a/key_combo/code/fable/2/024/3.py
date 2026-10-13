import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _split_quadrants(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w and h % 2 == 0, "expected an even square grid"
    n = h // 2
    return (
        g[:n, :n].copy(),
        g[:n, n:].copy(),
        g[n:, :n].copy(),
        g[n:, n:].copy(),
    )


def _align_quadrants(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ul, ur, ll, lr = _split_quadrants(grid)
    return (
        ul,
        np.rot90(ur, 1),
        np.rot90(ll, -1),
        np.rot90(lr, 2),
    )


def _assemble_from_aligned(aligned: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    ul, ur, ll, lr = aligned
    top = np.concatenate([ul, np.rot90(ur, -1)], axis=1)
    bottom = np.concatenate([np.rot90(ll, 1), np.rot90(lr, 2)], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _visible_palette(aligned: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> list[int]:
    colors = sorted({int(v) for block in aligned for v in block.flat if int(v) != BLACK})
    assert colors, "expected at least one visible non-black color"
    return colors


def _build_pair_maps(
    aligned: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], palette: list[int]
) -> list[list[dict[int, int]]]:
    n = int(aligned[0].shape[0])
    pair_maps = [[{} for _ in range(4)] for _ in range(4)]
    for q in range(4):
        pair_maps[q][q] = {color: color for color in palette}

    for r in range(n):
        for c in range(n):
            vals = [int(block[r, c]) for block in aligned]
            for qa in range(4):
                ca = vals[qa]
                if ca == BLACK:
                    continue
                for qb in range(qa + 1, 4):
                    cb = vals[qb]
                    if cb == BLACK:
                        continue
                    prev = pair_maps[qa][qb].get(ca)
                    assert prev in (None, cb), "conflicting visible color correspondence"
                    inv_prev = pair_maps[qb][qa].get(cb)
                    assert inv_prev in (None, ca), "visible correspondences break bijectivity"
                    pair_maps[qa][qb][ca] = cb
                    pair_maps[qb][qa][cb] = ca

    return pair_maps


def _propagate_maps(
    partials: list[dict[int, int]], pair_maps: list[list[dict[int, int]]], palette: list[int]
) -> list[dict[int, int]] | None:
    work = [mapping.copy() for mapping in partials]
    changed = True
    while changed:
        changed = False
        for qa in range(4):
            for qb in range(4):
                if qa == qb:
                    continue
                edge = pair_maps[qa][qb]
                if not edge:
                    continue
                for source_color, color_a in list(work[qa].items()):
                    if color_a not in edge:
                        continue
                    color_b = edge[color_a]
                    prev = work[qb].get(source_color)
                    if prev is None:
                        if color_b in work[qb].values():
                            return None
                        work[qb][source_color] = color_b
                        changed = True
                    elif prev != color_b:
                        return None

        for q in range(1, 4):
            if len(work[q]) == len(palette):
                continue
            missing_src = [color for color in palette if color not in work[q]]
            missing_dst = [color for color in palette if color not in work[q].values()]
            if len(missing_src) == 1 and len(missing_dst) == 1:
                work[q][missing_src[0]] = missing_dst[0]
                changed = True

    for q in range(4):
        if len(set(work[q].values())) != len(work[q]):
            return None
    return work


def _candidate_colors(
    q: int,
    source_color: int,
    partials: list[dict[int, int]],
    pair_maps: list[list[dict[int, int]]],
    palette: list[int],
) -> list[int]:
    used = set(partials[q].values())
    candidates = [color for color in palette if color not in used]
    for qa in range(4):
        if source_color not in partials[qa]:
            continue
        color_a = partials[qa][source_color]
        edge = pair_maps[qa][q]
        if color_a in edge:
            needed = edge[color_a]
            candidates = [color for color in candidates if color == needed]
        reverse_edge = pair_maps[q][qa]
        candidates = [color for color in candidates if color not in reverse_edge or reverse_edge[color] == color_a]
    return candidates


def _permutations_fit(
    aligned: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], perms: list[dict[int, int]]
) -> bool:
    inverses = [{dst: src for src, dst in mapping.items()} for mapping in perms]
    n = int(aligned[0].shape[0])
    for r in range(n):
        for c in range(n):
            base_color = None
            for q in range(4):
                val = int(aligned[q][r, c])
                if val == BLACK:
                    continue
                if val not in inverses[q]:
                    return False
                src = inverses[q][val]
                if base_color is None:
                    base_color = src
                elif base_color != src:
                    return False
            if base_color is None:
                return False
    return True


def _search_permutations(
    aligned: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    partials: list[dict[int, int]],
    pair_maps: list[list[dict[int, int]]],
    palette: list[int],
    solutions: list[list[dict[int, int]]],
) -> None:
    if len(solutions) > 1:
        return
    current = _propagate_maps(partials, pair_maps, palette)
    if current is None:
        return
    if all(len(mapping) == len(palette) for mapping in current):
        if _permutations_fit(aligned, current):
            solutions.append(current)
        return

    best_q = None
    best_source = None
    best_candidates = None
    for q in range(1, 4):
        for source_color in palette:
            if source_color in current[q]:
                continue
            candidates = _candidate_colors(q, source_color, current, pair_maps, palette)
            if not candidates:
                return
            if best_candidates is None or len(candidates) < len(best_candidates):
                best_q = q
                best_source = source_color
                best_candidates = candidates

    assert best_q is not None and best_source is not None and best_candidates is not None
    for candidate in best_candidates:
        nxt = [mapping.copy() for mapping in current]
        nxt[best_q][best_source] = candidate
        _search_permutations(aligned, nxt, pair_maps, palette, solutions)
        if len(solutions) > 1:
            return


def _solve_permutations(aligned: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> list[dict[int, int]]:
    palette = _visible_palette(aligned)
    pair_maps = _build_pair_maps(aligned, palette)
    partials = [{color: color for color in palette}]
    partials.extend(pair_maps[0][q].copy() for q in range(1, 4))
    solutions: list[list[dict[int, int]]] = []
    _search_permutations(aligned, partials, pair_maps, palette, solutions)
    assert len(solutions) == 1, "visible clues must determine a unique set of color bijections"
    return solutions[0]


def _complete_aligned(
    aligned: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], list[dict[int, int]]]:
    perms = _solve_permutations(aligned)
    inverses = [{dst: src for src, dst in mapping.items()} for mapping in perms]
    n = int(aligned[0].shape[0])
    completed = [np.zeros_like(aligned[0]) for _ in range(4)]

    for r in range(n):
        for c in range(n):
            candidates = []
            for q in range(4):
                val = int(aligned[q][r, c])
                if val != BLACK:
                    candidates.append(inverses[q][val])
            assert candidates, "each source position must stay visible in at least one quadrant"
            base_color = candidates[0]
            assert all(color == base_color for color in candidates), "visible quadrants disagree on the source color"
            for q in range(4):
                completed[q][r, c] = perms[q][base_color]

    completed_tuple = tuple(completed)
    for q in range(4):
        mask = aligned[q] != BLACK
        assert np.array_equal(completed_tuple[q][mask], aligned[q][mask]), "completion must preserve visible cells"
    return completed_tuple, perms


def _apply_map(tile: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    lut = np.arange(10, dtype=int)
    for src, dst in mapping.items():
        lut[int(src)] = int(dst)
    return lut[np.asarray(tile, dtype=int)]


def _neighbors4(r: int, c: int, n: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < n:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < n:
        out.append((r, c + 1))
    return out


def _boundary_mask(tile: np.ndarray) -> np.ndarray:
    n = int(tile.shape[0])
    mask = np.zeros_like(tile, dtype=bool)
    for r in range(n):
        for c in range(n):
            color = int(tile[r, c])
            for nr, nc in _neighbors4(r, c, n):
                if int(tile[nr, nc]) != color:
                    mask[r, c] = True
                    break
    return mask


def _random_derangement(
    rng: np.random.Generator, palette: list[int], forbidden: list[dict[int, int]]
) -> dict[int, int]:
    values = list(palette)
    for _ in range(500):
        perm = values.copy()
        rng.shuffle(perm)
        if any(src == dst for src, dst in zip(values, perm)):
            continue
        mapping = {src: dst for src, dst in zip(values, perm)}
        if all(mapping != old for old in forbidden):
            return mapping
    raise RuntimeError("failed to sample a nontrivial color derangement")


def _grow_tile(rng: np.random.Generator, n: int, palette: list[int]) -> np.ndarray | None:
    k = len(palette)
    labels = np.full((n, n), -1, dtype=int)
    seeds: list[tuple[int, int]] = []
    for _ in range(200):
        seeds = []
        coords = [(r, c) for r in range(n) for c in range(n)]
        rng.shuffle(coords)
        for r, c in coords:
            if all(abs(r - sr) + abs(c - sc) >= 2 for sr, sc in seeds):
                seeds.append((r, c))
                if len(seeds) == k:
                    break
        if len(seeds) == k:
            break
    if len(seeds) != k:
        return None

    frontier: list[list[tuple[int, int]]] = []
    for idx, (r, c) in enumerate(seeds):
        labels[r, c] = idx
        frontier.append([(r, c)])

    while np.any(labels < 0):
        active: list[tuple[int, list[tuple[int, int]]]] = []
        weights = []
        for idx in range(k):
            candidates = []
            for r, c in frontier[idx]:
                for nr, nc in _neighbors4(r, c, n):
                    if labels[nr, nc] < 0:
                        candidates.append((nr, nc))
            if candidates:
                active.append((idx, candidates))
                weights.append(1.0 / (1.0 + float(np.sum(labels == idx))))
        if not active:
            return None
        weights_arr = np.asarray(weights, dtype=float)
        weights_arr /= weights_arr.sum()
        pick = int(rng.choice(len(active), p=weights_arr))
        idx, candidates = active[pick]
        nr, nc = candidates[int(rng.integers(len(candidates)))]
        labels[nr, nc] = idx
        frontier[idx].append((nr, nc))

    tile = np.asarray(palette, dtype=int)[labels]
    counts = [int(np.sum(tile == color)) for color in palette]
    if min(counts) < 4:
        return None
    if any(np.array_equal(tile, np.rot90(tile, k)) for k in (1, 2, 3)):
        return None
    if int(_boundary_mask(tile).sum()) < n + 2:
        return None
    return tile


def _force_state(forced: np.ndarray, q: int, r: int, c: int, state: int) -> bool:
    prev = int(forced[q, r, c])
    if prev != -1 and prev != state:
        return False
    forced[q, r, c] = state
    return True


def _choose_witnesses(
    rng: np.random.Generator, tile: np.ndarray, palette: list[int]
) -> dict[tuple[int, int], tuple[int, int]]:
    witnesses: dict[tuple[int, int], tuple[int, int]] = {}
    for color in palette:
        coords = [tuple(coord) for coord in np.argwhere(tile == color)]
        rng.shuffle(coords)
        assert len(coords) >= 4
        for q, coord in zip((1, 2, 3), coords[:3]):
            witnesses[(q, color)] = (int(coord[0]), int(coord[1]))
    return witnesses


def _choose_composition_cases(
    rng: np.random.Generator,
    tile: np.ndarray,
    forced: np.ndarray,
    witnesses: dict[tuple[int, int], tuple[int, int]],
) -> list[tuple[int, int, int, tuple[int, ...]]]:
    n = int(tile.shape[0])
    boundary = _boundary_mask(tile)
    reserved = set(witnesses.values())
    candidates = [tuple(coord) for coord in np.argwhere(boundary)]
    rng.shuffle(candidates)

    triple_target = int(rng.choice([1, 2, 3]))
    cases = []
    used = set(reserved)
    for target in (1, 2, 3):
        chosen = None
        for r, c in candidates:
            rc = (int(r), int(c))
            if rc not in used:
                chosen = rc
                used.add(rc)
                break
        if chosen is None:
            return []
        if target == triple_target:
            hidden = tuple(sorted((0, target, ((target % 3) + 1))))
        else:
            hidden = (0, target)
        cases.append((target, chosen[0], chosen[1], hidden))

    for target, r, c, hidden in cases:
        for q in range(4):
            state = 1 if q in hidden else 0
            if not _force_state(forced, q, r, c, state):
                return []
    return cases


def _randomize_holes(
    rng: np.random.Generator, tile: np.ndarray, forced: np.ndarray
) -> np.ndarray | None:
    n = int(tile.shape[0])
    boundary = _boundary_mask(tile)
    for r in range(n):
        for c in range(n):
            free = [q for q in range(4) if int(forced[q, r, c]) == -1]
            current_holes = int(np.sum(forced[:, r, c] == 1))
            max_extra = min(len(free), 3 - current_holes)
            if max_extra < 0:
                return None
            if not free:
                continue

            weights = np.array([0.52, 0.30, 0.14, 0.04], dtype=float)
            if boundary[r, c]:
                weights = np.array([0.20, 0.38, 0.28, 0.14], dtype=float)
            use_weights = weights[: max_extra + 1].copy()
            use_weights /= use_weights.sum()
            extra = int(rng.choice(np.arange(max_extra + 1), p=use_weights))
            chosen = set()
            if extra > 0:
                chosen = set(int(x) for x in rng.choice(free, size=extra, replace=False))
            for q in free:
                forced[q, r, c] = 1 if q in chosen else 0

    hole_masks = forced == 1
    if np.any(np.sum(hole_masks, axis=0) >= 4):
        return None
    hole_counts = hole_masks.reshape(4, -1).sum(axis=1)
    if np.any(hole_counts < 3):
        return None
    if np.any(hole_counts > int(0.58 * n * n)):
        return None
    return hole_masks


def _build_instance(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None:
    n = int(rng.integers(5, 8))
    palette_size = int(rng.choice([4, 5]))
    palette = sorted(int(x) for x in rng.choice(PALETTE, size=palette_size, replace=False))
    tile = _grow_tile(rng, n, palette)
    if tile is None:
        return None

    perms = [{color: color for color in palette}]
    perms.append(_random_derangement(rng, palette, perms))
    perms.append(_random_derangement(rng, palette, perms))
    perms.append(_random_derangement(rng, palette, perms))

    aligned_full = (
        tile.copy(),
        _apply_map(tile, perms[1]),
        _apply_map(tile, perms[2]),
        _apply_map(tile, perms[3]),
    )

    witnesses = _choose_witnesses(rng, tile, palette)
    forced = np.full((4, n, n), -1, dtype=int)
    for (q, _color), (r, c) in witnesses.items():
        if not _force_state(forced, 0, r, c, 0):
            return None
        if not _force_state(forced, q, r, c, 0):
            return None

    composition_cases = _choose_composition_cases(rng, tile, forced, witnesses)
    if len(composition_cases) != 3:
        return None

    hole_masks = _randomize_holes(rng, tile, forced)
    if hole_masks is None:
        return None

    aligned_input = []
    for q in range(4):
        block = aligned_full[q].copy()
        block[hole_masks[q]] = BLACK
        aligned_input.append(block)
    aligned_input_tuple = tuple(aligned_input)

    inp = _assemble_from_aligned(aligned_input_tuple)
    out = _assemble_from_aligned(aligned_full)

    if np.array_equal(inp, out):
        return None
    if np.array_equal(_wrong_rotate_only(inp), out):
        return None

    completed, solved_perms = _complete_aligned(aligned_input_tuple)
    if not np.array_equal(_assemble_from_aligned(completed), out):
        return None
    for q in range(4):
        if solved_perms[q] != perms[q]:
            return None

    if not _has_required_composition_case(aligned_input_tuple, solved_perms):
        return None
    if np.any(inp == BLACK) and not np.any(out == BLACK):
        return inp, out, aligned_input_tuple
    return None


def _wrong_rotate_only(grid: np.ndarray) -> np.ndarray:
    aligned = _align_quadrants(grid)
    n = int(aligned[0].shape[0])
    filled = [np.zeros_like(aligned[0]) for _ in range(4)]
    for r in range(n):
        for c in range(n):
            visible = [int(block[r, c]) for block in aligned if int(block[r, c]) != BLACK]
            assert visible, "generator should never hide all four aligned copies"
            for q in range(4):
                filled[q][r, c] = visible[0]
    return _assemble_from_aligned(tuple(filled))


def _has_required_composition_case(
    aligned: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], perms: list[dict[int, int]]
) -> bool:
    inverses = [{dst: src for src, dst in mapping.items()} for mapping in perms]
    n = int(aligned[0].shape[0])
    found_two_hole = False
    found_three_hole = False
    for r in range(n):
        for c in range(n):
            visible = [q for q in range(4) if int(aligned[q][r, c]) != BLACK]
            hidden = [q for q in range(4) if int(aligned[q][r, c]) == BLACK]
            if 0 in hidden and len(hidden) >= 2 and visible:
                base_color = inverses[visible[0]][int(aligned[visible[0]][r, c])]
                for donor in visible[1:]:
                    if inverses[donor][int(aligned[donor][r, c])] != base_color:
                        return False
                found_two_hole |= len(hidden) == 2
                found_three_hole |= len(hidden) == 3
    return found_two_hole and found_three_hole


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a four-quadrant rotational collage by inferring each rotated copy's color bijection and repairing its black holes.

    Input structure
    - The grid is an even square split into four equal quadrants.
    - The upper-left quadrant is a source tile with some black missing cells.
    - The upper-right, lower-left, and lower-right quadrants are the source tile
      rotated clockwise, counterclockwise, and by 180 degrees, then recolored by
      their own fixed color bijections; black marks missing cells only.

    Rule
    - Inverse-rotate the three copied quadrants so all four tiles share the same
      source coordinates.
    - Use positions where two aligned quadrants are both visible to recover the
      three bijections from source colors to copied-quadrant colors.
    - For each aligned position, infer the underlying source color from any
      visible quadrant, then repaint all black holes with that quadrant's mapped
      color and rotate the repaired tiles back into the original collage layout.

    Key insight
    - Matching only by rotation is insufficient because each copied quadrant also
      applies a different color permutation; once those bijections are inferred
      from jointly visible aligned cells, any remaining hole can be restored by
      translating a visible counterpart through the appropriate inverse and
      forward color maps.
    """
    aligned = _align_quadrants(np.asarray(grid, dtype=int))
    completed, _perms = _complete_aligned(aligned)
    return _assemble_from_aligned(completed)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        instance = _build_instance(rng)
        if instance is not None:
            return instance[0]
    raise RuntimeError("failed to generate a valid rotational recoloring collage")
