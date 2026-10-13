from __future__ import annotations

import random
from collections import Counter

import numpy as np


TRANSFORM_COUNT = 4
ALL_COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 400


def _apply_transform(tile: np.ndarray, transform_id: int) -> np.ndarray:
    if transform_id == 0:
        return tile.copy()
    if transform_id == 1:
        return np.fliplr(tile).copy()
    if transform_id == 2:
        return np.flipud(tile).copy()
    if transform_id == 3:
        return np.flipud(np.fliplr(tile)).copy()
    raise ValueError(f"unknown transform id: {transform_id}")


def _valid_block_widths(total_width: int) -> list[int]:
    return [w for w in range(1, total_width + 1) if total_width % w == 0 and total_width // w >= 4]


def _split_tiles(grid: np.ndarray, block_width: int) -> list[np.ndarray]:
    return [grid[:, c0 : c0 + block_width].copy() for c0 in range(0, grid.shape[1], block_width)]


def _consensus_output(tiles: list[np.ndarray]) -> tuple[np.ndarray, bool, int]:
    h, w = tiles[0].shape
    k = len(tiles)
    out = np.empty((h, w), dtype=int)
    strict = True
    majority_total = 0
    for r in range(h):
        for c in range(w):
            counts = Counter(int(tile[r, c]) for tile in tiles)
            color, count = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
            out[r, c] = color
            majority_total += count
            strict &= count > k // 2
    return out, strict, majority_total


def _pairwise_score(tiles: list[np.ndarray]) -> int:
    score = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            score += int(np.count_nonzero(tiles[i] == tiles[j]))
    return score


def _best_assignment_for_tiles(tiles: list[np.ndarray]) -> dict:
    variants = [[_apply_transform(tile, t) for t in range(TRANSFORM_COUNT)] for tile in tiles]
    anchor = tiles[0]
    assignment = [0]
    ambiguous_tiles = 0
    tile_score_sum = 0
    tile_margins = []
    per_tile_scores = {0: [0]}
    for j in range(1, len(tiles)):
        scores = [int(np.count_nonzero(anchor == variants[j][tj])) for tj in range(TRANSFORM_COUNT)]
        per_tile_scores[j] = scores
        best_score = max(scores)
        best_states = [state for state, score in enumerate(scores) if score == best_score]
        ambiguous_tiles += int(len(best_states) != 1)
        assignment.append(best_states[0])
        tile_score_sum += best_score
        second_best = max(score for state, score in enumerate(scores) if state != best_states[0])
        tile_margins.append(best_score - second_best)

    normalized = [variants[i][assignment[i]] for i in range(len(tiles))]
    out, strict, majority_total = _consensus_output(normalized)
    area = tiles[0].size
    pair_denom = max(1, (len(tiles) * (len(tiles) - 1) // 2) * area)
    majority_denom = len(tiles) * area
    anchor_denom = max(1, (len(tiles) - 1) * area)
    return {
        "assignment": tuple(assignment),
        "score": _pairwise_score(normalized),
        "score_frac": _pairwise_score(normalized) / pair_denom,
        "strict": strict,
        "majority_total": majority_total,
        "majority_frac": majority_total / majority_denom,
        "output": out,
        "ambiguous_tiles": ambiguous_tiles,
        "tile_score_sum": tile_score_sum,
        "tile_score_frac": tile_score_sum / anchor_denom,
        "tile_margins": tile_margins,
        "per_tile_scores": per_tile_scores,
    }


def _best_period_solution(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 4, "expected horizontally concatenated tiles"

    candidates = []
    for block_width in _valid_block_widths(g.shape[1]):
        tiles = _split_tiles(g, block_width)
        result = _best_assignment_for_tiles(tiles)
        result = dict(result)
        result["block_width"] = block_width
        result["tile_count"] = len(tiles)
        candidates.append(result)

    assert candidates, "grid width must admit at least four equal blocks"
    best_key = max(
        (
            cand["score_frac"],
            int(cand["strict"]),
            -cand["ambiguous_tiles"],
            cand["majority_frac"],
            cand["tile_score_frac"],
        )
        for cand in candidates
    )
    winners = [
        cand
        for cand in candidates
        if (
            cand["score_frac"],
            int(cand["strict"]),
            -cand["ambiguous_tiles"],
            cand["majority_frac"],
            cand["tile_score_frac"],
        )
        == best_key
    ]
    assert len(winners) == 1, "block width and orientations must be uniquely determined"
    return winners[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the clean leftmost-oriented base tile from noisy reflected copies.

    Input structure:
    - The grid is a horizontal concatenation of k equal-width tiles with k >= 4.
    - Every tile is a noisy copy of one hidden base tile, using exactly one of: identity, left-right mirror, up-down mirror, or 180-degree rotation.
    - No tile is clean, and the output must use the leftmost tile's orientation as the reference frame.

    Rule:
    - Try each feasible tile width that splits the grid into at least four blocks.
    - For each split, choose one orientation per tile (keeping the leftmost tile fixed) so the normalized tiles agree with one another as much as possible overall.
    - After normalizing all tiles into the leftmost orientation, take the strict cellwise majority color and return that restored base tile.

    Key insight:
    - Majority voting only works after undoing each tile's own flip or rotation, so the task is a joint alignment-and-denoising problem rather than a direct cross-block vote.
    """
    return _best_period_solution(np.asarray(grid, dtype=int))["output"].copy()


def _has_smaller_reflective_period(tile: np.ndarray) -> bool:
    h, w = tile.shape
    for period in range(1, w):
        if w % period != 0:
            continue
        blocks = _split_tiles(tile, period)
        anchor = blocks[0]
        ok = True
        for block in blocks[1:]:
            if not any(np.array_equal(block, _apply_transform(anchor, t)) for t in range(TRANSFORM_COUNT)):
                ok = False
                break
        if ok:
            return True
    return False


def _random_stamp(tile: np.ndarray, palette: list[int]) -> None:
    h, w = tile.shape
    color = random.choice(palette[1:])
    kind = random.choice(("rect", "hline", "vline", "l"))
    r0 = random.randrange(h)
    c0 = random.randrange(w)
    r1 = random.randrange(r0, h)
    c1 = random.randrange(c0, w)
    if kind == "rect":
        tile[r0 : r1 + 1, c0 : c1 + 1] = color
    elif kind == "hline":
        tile[r0, c0 : c1 + 1] = color
    elif kind == "vline":
        tile[r0 : r1 + 1, c0] = color
    else:
        tile[r0 : r1 + 1, c0] = color
        tile[r1, c0 : c1 + 1] = color


def _generate_base_tile(h: int, w: int) -> np.ndarray | None:
    palette = random.sample(list(ALL_COLORS), random.randint(3, 5))
    tile = np.full((h, w), palette[0], dtype=int)
    for _ in range(random.randint(5, 8)):
        _random_stamp(tile, palette)
    if len(set(map(int, np.unique(tile)))) < 3:
        return None
    if _has_smaller_reflective_period(tile):
        return None
    area = h * w
    for transform_id in range(1, TRANSFORM_COUNT):
        diff = int(np.count_nonzero(tile != _apply_transform(tile, transform_id)))
        if diff < max(4, area // 4):
            return None
    return tile


def _sample_transforms(tile_count: int) -> list[int]:
    transforms = [0]
    others = [1, 2, 3]
    while len(others) < tile_count - 1:
        others.append(random.randrange(TRANSFORM_COUNT))
    random.shuffle(others)
    transforms.extend(others[: tile_count - 1])
    if len(set(transforms)) < 2:
        transforms[-1] = 1
    return transforms


def _choose_noise_positions(h: int, w: int, tile_count: int) -> list[list[tuple[int, int]]]:
    area = h * w
    max_noise = 2 if area < 25 else 3
    counts = [random.randint(1, max_noise) for _ in range(tile_count)]
    total_needed = sum(counts)
    if total_needed > area:
        return []
    pool = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(pool)
    chosen = []
    cursor = 0
    for count in counts:
        chosen.append(sorted(pool[cursor : cursor + count]))
        cursor += count
    if not chosen[0]:
        return []
    if len(chosen[0]) > 1:
        return chosen
    if cursor >= len(pool):
        return []
    chosen[0] = sorted(chosen[0] + [pool[cursor]])
    return chosen


def _normalized_to_displayed(h: int, w: int, transform_id: int, r: int, c: int) -> tuple[int, int]:
    if transform_id == 0:
        return r, c
    if transform_id == 1:
        return r, w - 1 - c
    if transform_id == 2:
        return h - 1 - r, c
    if transform_id == 3:
        return h - 1 - r, w - 1 - c
    raise ValueError(f"unknown transform id: {transform_id}")


def _corrupt_tiles(base_tile: np.ndarray, transforms: list[int]) -> tuple[list[np.ndarray], list[list[tuple[int, int]]]] | tuple[None, None]:
    h, w = base_tile.shape
    noise_positions = _choose_noise_positions(h, w, len(transforms))
    if not noise_positions:
        return None, None

    noisy_tiles = []
    for tile_index, transform_id in enumerate(transforms):
        clean_tile = _apply_transform(base_tile, transform_id)
        noisy_tile = clean_tile.copy()
        for nr, nc in noise_positions[tile_index]:
            rr, cc = _normalized_to_displayed(h, w, transform_id, nr, nc)
            current = int(clean_tile[rr, cc])
            choices = [color for color in ALL_COLORS if color != current]
            noisy_tile[rr, cc] = random.choice(choices)
        if np.array_equal(noisy_tile, clean_tile):
            return None, None
        noisy_tiles.append(noisy_tile)
    return noisy_tiles, noise_positions


def _raw_majority(tiles: list[np.ndarray]) -> np.ndarray:
    out, _strict, _majority_total = _consensus_output(tiles)
    return out


def _clean_raw_majority_fails(base_tile: np.ndarray, transforms: list[int]) -> bool:
    clean_tiles = [_apply_transform(base_tile, transform_id) for transform_id in transforms]
    return not np.array_equal(_raw_majority(clean_tiles), base_tile)


def _semantic_checks(
    base_tile: np.ndarray,
    transforms: list[int],
    noisy_tiles: list[np.ndarray],
    noise_positions: list[list[tuple[int, int]]],
) -> bool:
    h, w = base_tile.shape
    tile_count = len(noisy_tiles)

    if any(np.array_equal(noisy_tile, _apply_transform(base_tile, transforms[i])) for i, noisy_tile in enumerate(noisy_tiles)):
        return False
    if any(len(pos) == 0 for pos in noise_positions):
        return False
    if len(set(transforms)) < 2 or all(t == 0 for t in transforms[1:]):
        return False

    aligned = [_apply_transform(noisy_tiles[i], transforms[i]) for i in range(tile_count)]
    restored, strict, _majority_total = _consensus_output(aligned)
    if not strict or not np.array_equal(restored, base_tile):
        return False

    if np.array_equal(_raw_majority(noisy_tiles), base_tile):
        return False

    witness_ok = False
    for nr, nc in noise_positions[0]:
        if int(aligned[0][nr, nc]) != int(base_tile[nr, nc]):
            witness_ok = True
            break
    if not witness_ok:
        return False

    grid = np.concatenate(noisy_tiles, axis=1)
    if max(grid.shape) > 30 or max(base_tile.shape) > 30:
        return False

    expected_assignment = tuple(transforms)
    period_candidates = []
    for block_width in _valid_block_widths(grid.shape[1]):
        tiles = _split_tiles(grid, block_width)
        result = _best_assignment_for_tiles(tiles)
        period_candidates.append((block_width, result))

    true_width = w
    true_hits = [item for item in period_candidates if item[0] == true_width]
    if len(true_hits) != 1:
        return False
    true_result = true_hits[0][1]
    if true_result["assignment"] != expected_assignment:
        return False
    if true_result["ambiguous_tiles"] != 0:
        return False
    true_key = (
        true_result["score_frac"],
        int(true_result["strict"]),
        -true_result["ambiguous_tiles"],
        true_result["majority_frac"],
        true_result["tile_score_frac"],
    )
    if sum(
        (
            result["score_frac"],
            int(result["strict"]),
            -result["ambiguous_tiles"],
            result["majority_frac"],
            result["tile_score_frac"],
        )
        == true_key
        for _width, result in period_candidates
    ) != 1:
        return False

    solved = main(grid)
    return np.array_equal(solved, base_tile)


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(5, 7)
    w = random.randint(5, 7)
    max_tiles = min(5, 30 // w)
    if max_tiles < 4:
        return None
    tile_count = random.randint(4, max_tiles)

    base_tile = _generate_base_tile(h, w)
    if base_tile is None:
        return None

    transforms = _sample_transforms(tile_count)
    if not _clean_raw_majority_fails(base_tile, transforms):
        return None
    noisy_tiles, noise_positions = _corrupt_tiles(base_tile, transforms)
    if noisy_tiles is None or noise_positions is None:
        return None

    if not _semantic_checks(base_tile, transforms, noisy_tiles, noise_positions):
        return None
    return np.concatenate(noisy_tiles, axis=1)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 众声成像 input")
