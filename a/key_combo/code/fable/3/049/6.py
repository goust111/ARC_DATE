from __future__ import annotations

import random

import numpy as np


ARC_COLORS = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 800


def _assert_arc_grid(g: np.ndarray) -> None:
    assert g.ndim == 2, "expected a 2D grid"
    assert g.size > 0, "expected a non-empty grid"
    assert int(g.min()) >= 0 and int(g.max()) <= 9, "expected ARC colors 0..9"


def _split_quadrants(g: np.ndarray) -> list[np.ndarray]:
    _assert_arc_grid(g)
    h, w = map(int, g.shape)
    assert h == w and h % 2 == 1, "expected a square grid with centered separators"
    n = h // 2
    assert n >= 2, "expected non-trivial square quadrants"
    assert np.all(g[n, :] == int(g[n, 0])), "expected one solid separator row"
    assert np.all(g[:, n] == int(g[0, n])), "expected one solid separator column"
    assert int(g[n, 0]) == int(g[0, n]), "separator row and column must share one color"
    return [
        g[:n, :n],
        g[:n, n + 1 :],
        g[n + 1 :, :n],
        g[n + 1 :, n + 1 :],
    ]


def _best_bijection(src: np.ndarray, dst: np.ndarray) -> tuple[int, np.ndarray]:
    """Return the max-match color permutation mapping src colors onto dst colors."""
    assert src.shape == dst.shape
    _assert_arc_grid(src)
    _assert_arc_grid(dst)

    scores = np.zeros((10, 10), dtype=int)
    np.add.at(scores, (src.ravel(), dst.ravel()), 1)

    dp: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for src_color in ARC_COLORS:
        next_dp: dict[int, tuple[int, tuple[int, ...]]] = {}
        for used_mask, (score, path) in dp.items():
            for dst_color in ARC_COLORS:
                bit = 1 << dst_color
                if used_mask & bit:
                    continue
                new_mask = used_mask | bit
                new_path = path + (dst_color,)
                new_score = score + int(scores[src_color, dst_color])
                old = next_dp.get(new_mask)
                if old is None or new_score > old[0] or (new_score == old[0] and new_path < old[1]):
                    next_dp[new_mask] = (new_score, new_path)
        dp = next_dp

    best_score, best_path = dp[(1 << 10) - 1]
    return int(best_score), np.array(best_path, dtype=int)


def _alignment_candidates(q: np.ndarray, target: np.ndarray) -> list[dict]:
    candidates = []
    for k in range(4):
        aligned = np.rot90(q, -k)
        score, lut = _best_bijection(aligned, target)
        candidates.append({"k": k, "score": score, "lut": lut, "aligned": aligned})
    return candidates


def _infer_alignment(q: np.ndarray, target: np.ndarray) -> dict:
    candidates = _alignment_candidates(q, target)
    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
    cells = int(target.size)
    assert ranked[0]["score"] > ranked[1]["score"], "rotation must be uniquely determined"
    assert ranked[0]["score"] * 4 >= cells * 3, "best alignment must explain a large majority"
    return ranked[0]


def _apply_lut(g: np.ndarray, lut: np.ndarray) -> np.ndarray:
    _assert_arc_grid(g)
    return lut[g].astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the clean quadrant by undoing rotations and color bijections, then majority-voting.

    Input structure:
    - A square grid is divided by one solid center row and one solid center column into four equal square quadrants.
    - The upper-left quadrant is the reference orientation and reference coloring of one pattern.
    - The other quadrants are the same pattern rotated by 90, 180, and 270 degrees in unknown quadrant order, each with its own one-to-one recoloring.
    - Each quadrant has a few wrong-color pixels, and no underlying pattern cell is corrupted in more than one quadrant.

    Rule:
    - For each non-reference quadrant, choose the unique rotation and color bijection that best align it to the upper-left quadrant.
    - Rotate those quadrants back to upper-left orientation and map their colors back to upper-left colors.
    - At each cell, take the unique majority color among the four aligned candidates.

    Key insight:
    - The copies only agree after both disguises are removed; mirror flips and raw color comparison are decoys, while the four-way vote exposes the isolated noisy pixels.
    """
    g = np.asarray(grid, dtype=int)
    quadrants = _split_quadrants(g)
    target = quadrants[0]

    aligned = [target.copy()]
    rotations = []
    for q in quadrants[1:]:
        best = _infer_alignment(q, target)
        rotations.append(int(best["k"]))
        aligned.append(_apply_lut(best["aligned"], best["lut"]))

    assert sorted(rotations) == [1, 2, 3], "non-reference quadrants must be the three non-zero rotations"

    stack = np.stack(aligned, axis=0)
    out = np.empty_like(target)
    for r in range(target.shape[0]):
        for c in range(target.shape[1]):
            values, counts = np.unique(stack[:, r, c], return_counts=True)
            best = int(counts.max())
            assert best >= 3, "expected at most one noisy candidate per aligned cell"
            assert int(np.sum(counts == best)) == 1, "expected a unique majority color"
            out[r, c] = int(values[np.argmax(counts)])
    return out


# --------------- generator ---------------


def _random_shape_mask(n: int) -> np.ndarray:
    max_side = min(5, n)
    kind = random.choice(["rect", "line", "ell", "tee", "stair"])

    if kind == "rect":
        h = random.randint(2, max_side)
        w = random.randint(2, max_side)
        return np.ones((h, w), dtype=bool)

    if kind == "line":
        if random.random() < 0.5:
            return np.ones((1, random.randint(3, max_side + 1)), dtype=bool)
        return np.ones((random.randint(3, max_side + 1), 1), dtype=bool)

    if kind == "ell":
        h = random.randint(3, max_side)
        w = random.randint(3, max_side)
        mask = np.zeros((h, w), dtype=bool)
        mask[:, 0] = True
        mask[-1, :] = True
        return np.rot90(mask, random.randrange(4))

    if kind == "tee":
        h = random.randint(3, max_side)
        w = random.randint(3, max_side)
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[:, w // 2] = True
        return np.rot90(mask, random.randrange(4))

    length = random.randint(4, max_side + 1)
    mask = np.zeros((length, length), dtype=bool)
    for i in range(length):
        mask[i, i] = True
        if i + 1 < length:
            mask[i, i + 1] = True
    return np.rot90(mask, random.randrange(4))


def _place_mask(base: np.ndarray, mask: np.ndarray, color: int, bg: int) -> bool:
    h, w = mask.shape
    n = base.shape[0]
    positions = []
    for r in range(n - h + 1):
        for c in range(n - w + 1):
            patch = base[r : r + h, c : c + w]
            if np.all(patch[mask] == bg):
                positions.append((r, c))
    if not positions:
        return False
    r, c = random.choice(positions)
    base[r : r + h, c : c + w][mask] = int(color)
    return True


def _best_transform_score(src: np.ndarray, dst: np.ndarray) -> int:
    return _best_bijection(src, dst)[0]


def _clean_pattern_is_distinct(base: np.ndarray) -> bool:
    cells = int(base.size)
    n = int(base.shape[0])
    margin = max(8, n)
    wrong_rotation = max(_best_transform_score(np.rot90(base, k), base) for k in (1, 2, 3))
    mirror = max(_best_transform_score(np.rot90(np.fliplr(base), k), base) for k in range(4))
    return cells - max(wrong_rotation, mirror) >= margin


def _make_base(n: int, palette: list[int]) -> np.ndarray | None:
    bg = int(palette[0])
    fg = [int(c) for c in palette[1:]]
    target_min = int(round(0.34 * n * n))
    target_max = int(round(0.56 * n * n))
    min_color_cells = max(5, n // 2)

    for _ in range(120):
        base = np.full((n, n), bg, dtype=int)
        for color in fg:
            placed = False
            for _try in range(80):
                if _place_mask(base, _random_shape_mask(n), color, bg):
                    placed = True
                    break
            if not placed:
                break
        else:
            attempts = 0
            while int(np.count_nonzero(base != bg)) < target_min and attempts < 250:
                attempts += 1
                _place_mask(base, _random_shape_mask(n), random.choice(fg), bg)

            fg_count = int(np.count_nonzero(base != bg))
            color_counts = {c: int(np.count_nonzero(base == c)) for c in palette}
            if not (target_min <= fg_count <= target_max):
                continue
            if any(color_counts[c] < min_color_cells for c in palette):
                continue
            if not np.all(np.any(base != bg, axis=0)):
                continue
            if not np.all(np.any(base != bg, axis=1)):
                continue
            if not _clean_pattern_is_distinct(base):
                continue
            return base
    return None


def _derangement(palette: list[int], used: set[tuple[int, ...]]) -> dict[int, int]:
    for _ in range(300):
        perm = palette.copy()
        random.shuffle(perm)
        key = tuple(perm)
        if key in used:
            continue
        if all(src != dst for src, dst in zip(palette, perm)):
            used.add(key)
            return {int(src): int(dst) for src, dst in zip(palette, perm)}
    raise RuntimeError("failed to choose a color derangement")


def _map_palette(g: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    lut = np.arange(10, dtype=int)
    for src, dst in mapping.items():
        lut[int(src)] = int(dst)
    return lut[g].astype(int)


def _noise_mask(n: int, count: int, unavailable: set[tuple[int, int]]) -> list[tuple[int, int]]:
    candidates = [(r, c) for r in range(n) for c in range(n) if (r, c) not in unavailable]
    random.shuffle(candidates)
    chosen = candidates[:count]
    unavailable.update(chosen)
    return chosen


def _add_noise(
    quadrant: np.ndarray,
    aligned_cells: list[tuple[int, int]],
    k: int,
    visible_palette: list[int],
) -> None:
    mask = np.zeros_like(quadrant, dtype=bool)
    for r, c in aligned_cells:
        mask[int(r), int(c)] = True
    actual_mask = np.rot90(mask, int(k))
    for r, c in np.argwhere(actual_mask):
        old = int(quadrant[r, c])
        choices = [color for color in visible_palette if color != old]
        quadrant[r, c] = int(random.choice(choices))


def _mirror_score(q: np.ndarray, target: np.ndarray) -> int:
    return max(_best_transform_score(np.rot90(np.fliplr(q), k), target) for k in range(4))


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    rotations: list[int],
    mappings: list[dict[int, int]],
    noise_cells: list[list[tuple[int, int]]],
) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30 or expected.ndim != 2 or max(expected.shape) > 30:
        return False
    if sorted(rotations) != [1, 2, 3]:
        return False
    if len({tuple(mapping[c] for c in sorted(mapping)) for mapping in mappings}) != 3:
        return False
    if any(any(src == dst for src, dst in mapping.items()) for mapping in mappings):
        return False
    if any(len(cells) == 0 for cells in noise_cells):
        return False
    all_noise = [cell for cells in noise_cells for cell in cells]
    if len(all_noise) != len(set(all_noise)):
        return False

    try:
        quadrants = _split_quadrants(inp)
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(quadrants[0], expected):
        return False

    margin = max(4, expected.shape[0] // 2)
    for q, true_k in zip(quadrants[1:], rotations):
        candidates = sorted(_alignment_candidates(q, quadrants[0]), key=lambda item: item["score"], reverse=True)
        if int(candidates[0]["k"]) != int(true_k):
            return False
        if candidates[0]["score"] - candidates[1]["score"] < margin:
            return False
        no_rotation_score = next(item["score"] for item in candidates if int(item["k"]) == 0)
        if candidates[0]["score"] - no_rotation_score < margin:
            return False
        raw_aligned = np.rot90(q, -int(true_k))
        identity_color_score = int(np.sum(raw_aligned == quadrants[0]))
        if candidates[0]["score"] - identity_color_score < margin:
            return False
        if candidates[0]["score"] - _mirror_score(q, quadrants[0]) < margin:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(8, 13)
    palette = random.sample(list(ARC_COLORS), 5)
    remaining = [c for c in ARC_COLORS if c not in palette]
    sep = random.choice(remaining)
    base = _make_base(n, palette)
    if base is None:
        return None

    used_perms: set[tuple[int, ...]] = set()
    color_maps = [_derangement(palette, used_perms) for _ in range(3)]
    rotations = [1, 2, 3]
    random.shuffle(rotations)

    unavailable: set[tuple[int, int]] = set()
    noise_cells: list[list[tuple[int, int]]] = []
    noise_max = max(1, min(4, n // 3))
    for _ in range(4):
        count = random.randint(1, noise_max)
        noise_cells.append(_noise_mask(n, count, unavailable))

    quadrants: list[np.ndarray] = [base.copy()]
    _add_noise(quadrants[0], noise_cells[0], 0, palette)

    for k, mapping, cells in zip(rotations, color_maps, noise_cells[1:]):
        recolored = _map_palette(base, mapping)
        q = np.rot90(recolored, int(k)).copy()
        visible_palette = [mapping[c] for c in palette]
        _add_noise(q, cells, int(k), visible_palette)
        quadrants.append(q)

    side = 2 * n + 1
    inp = np.full((side, side), int(sep), dtype=int)
    inp[:n, :n] = quadrants[0]
    inp[:n, n + 1 :] = quadrants[1]
    inp[n + 1 :, :n] = quadrants[2]
    inp[n + 1 :, n + 1 :] = quadrants[3]

    if _semantic_checks(inp, base, rotations, color_maps, noise_cells):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid fable/3/049/6 input")
