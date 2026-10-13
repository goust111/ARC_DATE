from __future__ import annotations

import itertools
import random

import numpy as np


_TRANSFORMS = tuple((k, flip) for k in range(4) for flip in (False, True))
_SEPARATOR_COLOR = 9
_QUADRANT_COLOR_POOL = tuple(range(1, 9))
_GEN_Q_RANGE = range(5, 9)
_GEN_RETRIES = 400


def _apply_transform(arr: np.ndarray, transform_id: int) -> np.ndarray:
    k, flip = _TRANSFORMS[int(transform_id)]
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _freeze(arr: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in np.asarray(arr, dtype=int))


def _thaw(frozen: tuple[tuple[int, ...], ...]) -> np.ndarray:
    return np.array(frozen, dtype=int)


def _recolor(arr: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.empty_like(arr)
    for src, dst in mapping.items():
        out[arr == int(src)] = int(dst)
    return out


def _canonical_partition(arr: np.ndarray) -> tuple[int, ...]:
    mapping: dict[int, int] = {}
    next_id = 0
    flat = []
    for value in np.asarray(arr, dtype=int).ravel():
        key = int(value)
        if key not in mapping:
            mapping[key] = next_id
            next_id += 1
        flat.append(mapping[key])
    return tuple(flat)


def _best_candidates(src: np.ndarray, dst: np.ndarray) -> dict:
    src = np.asarray(src, dtype=int)
    dst = np.asarray(dst, dtype=int)
    assert src.shape == dst.shape

    best_score: int | None = None
    best_frozen: set[tuple[tuple[int, ...], ...]] = set()
    best_transform_ids: set[int] = set()

    for transform_id in range(len(_TRANSFORMS)):
        transformed = _apply_transform(src, transform_id)
        info = _best_recolor_only(transformed, dst)
        if best_score is None or info["score"] < best_score:
            best_score = int(info["score"])
            best_frozen = set(info["candidates"])
            best_transform_ids = {transform_id}
        elif info["score"] == best_score:
            best_frozen.update(info["candidates"])
            best_transform_ids.add(transform_id)

    assert best_score is not None, "no transform+bijection candidate found"
    return {
        "score": int(best_score),
        "candidates": tuple(sorted(best_frozen)),
        "transform_ids": tuple(sorted(best_transform_ids)),
    }


def _best_recolor_only(src: np.ndarray, dst: np.ndarray) -> dict:
    src = np.asarray(src, dtype=int)
    dst = np.asarray(dst, dtype=int)
    assert src.shape == dst.shape

    dst_colors = sorted(int(v) for v in np.unique(dst))
    best_score: int | None = None
    best_frozen: set[tuple[tuple[int, ...], ...]] = set()
    src_colors = sorted(int(v) for v in np.unique(src))
    if len(src_colors) != len(dst_colors):
        raise AssertionError("source and target must have equally many colors")
    for perm in itertools.permutations(dst_colors):
        mapped = _recolor(src, dict(zip(src_colors, perm)))
        score = int(np.count_nonzero(mapped != dst))
        frozen = _freeze(mapped)
        if best_score is None or score < best_score:
            best_score = score
            best_frozen = {frozen}
        elif score == best_score:
            best_frozen.add(frozen)

    assert best_score is not None, "no transform+bijection candidate found"
    return {
        "score": int(best_score),
        "candidates": tuple(sorted(best_frozen)),
    }


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    candidates = []

    for r in range(height):
        row_color = int(g[r, 0])
        if not np.all(g[r, :] == row_color):
            continue
        for c in range(width):
            col_color = int(g[0, c])
            if col_color != row_color:
                continue
            if not np.all(g[:, c] == col_color):
                continue
            top_h = r
            bottom_h = height - r - 1
            left_w = c
            right_w = width - c - 1
            if top_h <= 0 or left_w <= 0:
                continue
            if top_h != bottom_h or left_w != right_w or top_h != left_w:
                continue
            candidates.append((r, c, row_color))

    assert len(candidates) == 1, "expected exactly one separator cross"
    return candidates[0]


def _extract_quadrants(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    cross_r, cross_c, sep = _find_separator_cross(g)
    quads = [
        g[:cross_r, :cross_c].copy(),
        g[:cross_r, cross_c + 1 :].copy(),
        g[cross_r + 1 :, :cross_c].copy(),
        g[cross_r + 1 :, cross_c + 1 :].copy(),
    ]
    side = int(cross_r)
    assert all(q.shape == (side, side) for q in quads)
    return quads, int(sep)


def _analyze_quadrants(quads: list[np.ndarray]) -> dict:
    pair_cache: dict[tuple[int, int], dict] = {}

    def pair_info(i: int, j: int) -> dict:
        key = (int(i), int(j))
        if key not in pair_cache:
            pair_cache[key] = _best_candidates(quads[i], quads[j])
        return pair_cache[key]

    trio_scores = []
    for trio in itertools.combinations(range(4), 3):
        score = sum(pair_info(a, b)["score"] for a, b in itertools.combinations(trio, 2))
        trio_scores.append((int(score), trio))
    trio_scores.sort(key=lambda item: (item[0], item[1]))

    assert len(trio_scores) == 4
    assert trio_scores[0][0] == 0, "truthful trio must be perfectly consistent"
    assert trio_scores[0][0] < trio_scores[1][0], "truthful trio must be unique"

    honest = trio_scores[0][1]
    liar = next(idx for idx in range(4) if idx not in honest)

    candidate_sets = []
    liar_scores = []
    for h in honest:
        info = pair_info(h, liar)
        liar_scores.append(int(info["score"]))
        candidate_sets.append(set(info["candidates"]))
    assert all(score > 0 for score in liar_scores), "liar quadrant must contain visible errors"

    intersection = set.intersection(*candidate_sets)
    assert len(intersection) == 1, "corrected liar quadrant must be uniquely determined"
    correction = _thaw(next(iter(intersection)))

    return {
        "honest": honest,
        "liar": int(liar),
        "correction": correction,
        "trio_scores": trio_scores,
        "pair_cache": pair_cache,
        "liar_scores": tuple(liar_scores),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the uniquely inconsistent quadrant in its own orientation and palette.

    Input structure:
    - One solid-color row and one solid-color column form a separator cross that splits the grid into four equal square quadrants.
    - Each quadrant is the same hidden two-region base picture after an independent dihedral transform and an independent quadrant-wide color bijection.
    - The four quadrant palettes are pairwise disjoint, so colors cannot be matched directly across quadrants.

    Rule:
    - Remove the separator cross and compare quadrants only through all eight rotations/reflections plus a whole-quadrant color bijection.
    - The unique trio whose pairwise comparisons fit perfectly are the truthful quadrants; the remaining quadrant is the liar.
    - Use the truthful quadrants to predict the liar under its best transform+bijection explanation, then fix the mismatching cells and output only that corrected quadrant.

    Key insight:
    - Because palettes never overlap, consistency lives in the same-color partition structure under dihedral alignment, and the final answer must be recolored back into the liar quadrant's local palette instead of any shared canonical view.
    """
    quads, _sep = _extract_quadrants(np.asarray(grid, dtype=int))
    analysis = _analyze_quadrants(quads)
    return analysis["correction"]


def _stamp_rect(mask: np.ndarray, value: int) -> None:
    h, w = mask.shape
    rh = random.randint(1, max(1, h // 2 + 1))
    rw = random.randint(1, max(1, w // 2 + 1))
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    mask[r0 : r0 + rh, c0 : c0 + rw] = int(value)


def _stamp_line(mask: np.ndarray, value: int) -> None:
    h, w = mask.shape
    if random.random() < 0.5:
        r = random.randrange(h)
        length = random.randint(2, w)
        c0 = random.randint(0, w - length)
        mask[r, c0 : c0 + length] = int(value)
    else:
        c = random.randrange(w)
        length = random.randint(2, h)
        r0 = random.randint(0, h - length)
        mask[r0 : r0 + length, c] = int(value)


def _stamp_l(mask: np.ndarray, value: int) -> None:
    h, w = mask.shape
    arm_h = random.randint(2, h)
    arm_w = random.randint(2, w)
    r0 = random.randint(0, h - arm_h)
    c0 = random.randint(0, w - arm_w)
    corner = random.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        mask[r0 : r0 + arm_h, c0] = int(value)
        mask[r0, c0 : c0 + arm_w] = int(value)
    elif corner == "tr":
        mask[r0 : r0 + arm_h, c0 + arm_w - 1] = int(value)
        mask[r0, c0 : c0 + arm_w] = int(value)
    elif corner == "bl":
        mask[r0 : r0 + arm_h, c0] = int(value)
        mask[r0 + arm_h - 1, c0 : c0 + arm_w] = int(value)
    else:
        mask[r0 : r0 + arm_h, c0 + arm_w - 1] = int(value)
        mask[r0 + arm_h - 1, c0 : c0 + arm_w] = int(value)


def _sample_base_pattern(side: int) -> np.ndarray | None:
    p = random.uniform(0.35, 0.65)
    mask = np.array(
        [[1 if random.random() < p else 0 for _ in range(side)] for _ in range(side)],
        dtype=int,
    )

    for _ in range(random.randint(1, 3)):
        random.choice((_stamp_rect, _stamp_line, _stamp_l))(mask, random.randint(0, 1))
    for _ in range(random.randint(1, 3)):
        r = random.randrange(side)
        c = random.randrange(side)
        mask[r, c] = 1 - mask[r, c]

    ones = int(np.count_nonzero(mask))
    area = side * side
    if ones < area // 4 or ones > (3 * area) // 4:
        return None

    signatures = {_canonical_partition(_apply_transform(mask, t)) for t in range(8)}
    if len(signatures) != 8:
        return None

    variants = [_apply_transform(mask, t) for t in range(8)]
    for src_id, dst_id in itertools.permutations(range(8), 2):
        scores = [
            _best_recolor_only(_apply_transform(variants[src_id], extra_t), variants[dst_id])["score"]
            for extra_t in range(8)
        ]
        best = min(scores)
        if best != 0 or scores.count(0) != 1:
            return None
        next_best = min(score for score in scores if score > best)
        if next_best < 4:
            return None
    return mask


def _recolor_binary(mask: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    out = np.empty_like(mask)
    out[mask == 0] = int(colors[0])
    out[mask == 1] = int(colors[1])
    return out


def _choose_corruption_cells(side: int, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(side) for c in range(side)]
    random.shuffle(cells)
    chosen = []
    for cell in cells:
        if all(abs(cell[0] - rr) + abs(cell[1] - cc) >= 2 for rr, cc in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return cells[:count]


def _assemble_grid(quads: list[np.ndarray], sep: int) -> np.ndarray:
    side = int(quads[0].shape[0])
    grid = np.full((2 * side + 1, 2 * side + 1), int(sep), dtype=int)
    grid[:side, :side] = quads[0]
    grid[:side, side + 1 :] = quads[1]
    grid[side + 1 :, :side] = quads[2]
    grid[side + 1 :, side + 1 :] = quads[3]
    return grid


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    side = random.choice(tuple(_GEN_Q_RANGE))
    base = _sample_base_pattern(side)
    if base is None:
        return None

    transform_ids = random.sample(range(8), 4)
    colors = list(_QUADRANT_COLOR_POOL)
    random.shuffle(colors)
    palettes = [tuple(colors[2 * i : 2 * i + 2]) for i in range(4)]
    for i in range(4):
        if random.random() < 0.5:
            palettes[i] = (palettes[i][1], palettes[i][0])

    clean_quads = []
    for idx in range(4):
        transformed = _apply_transform(base, transform_ids[idx])
        clean_quads.append(_recolor_binary(transformed, palettes[idx]))

    liar = random.randrange(4)
    margin = None
    for honest in range(4):
        if honest == liar:
            continue
        scores = [
            _best_recolor_only(_apply_transform(clean_quads[honest], extra_t), clean_quads[liar])["score"]
            for extra_t in range(8)
        ]
        if min(scores) != 0 or scores.count(0) != 1:
            return None
        second_best = min(score for score in scores if score > 0)
        margin = second_best if margin is None else min(margin, second_best)
    assert margin is not None
    max_corrupt = min(3, (int(margin) - 1) // 2)
    if max_corrupt < 1:
        return None

    corrupt_count = random.randint(1, max_corrupt)
    observed_quads = [quad.copy() for quad in clean_quads]
    for r, c in _choose_corruption_cells(side, corrupt_count):
        a, b = palettes[liar]
        observed_quads[liar][r, c] = int(b if observed_quads[liar][r, c] == a else a)

    grid = _assemble_grid(observed_quads, _SEPARATOR_COLOR)
    expected = clean_quads[liar]

    analysis = _analyze_quadrants(observed_quads)
    if analysis["liar"] != liar:
        return None
    if not np.array_equal(analysis["correction"], expected):
        return None
    if analysis["trio_scores"][1][0] == analysis["trio_scores"][0][0]:
        return None
    if not all(score <= corrupt_count for score in analysis["liar_scores"]):
        return None
    if np.array_equal(observed_quads[liar], expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        generated = _generate_attempt()
        if generated is None:
            continue
        grid, expected = generated
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid liar-quadrant puzzle")
