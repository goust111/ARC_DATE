from __future__ import annotations

import random

import numpy as np


BG = 0
BARRIER = 7
PAIR_COLORS = (1, 2, 3, 4)
GEN_ATTEMPTS = 400


def _find_barrier_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == BARRIER)]
    assert len(cols) == 1, f"expected exactly one full-height barrier column, got {cols}"
    return int(cols[0])


def _row_markers(row: np.ndarray, sep: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    left = [(int(c), int(row[c])) for c in range(sep) if int(row[c]) not in (BG, BARRIER)]
    right = [(int(c), int(row[c])) for c in range(sep + 1, row.shape[0]) if int(row[c]) not in (BG, BARRIER)]
    return left, right


def _valid_pairs(grid: np.ndarray) -> list[dict]:
    sep = _find_barrier_col(grid)
    pairs: list[dict] = []
    for r in range(grid.shape[0]):
        left, right = _row_markers(grid[r], sep)
        if len(left) == 1 and len(right) == 1 and left[0][1] == right[0][1]:
            lc, color = left[0]
            rc, _ = right[0]
            pairs.append(
                {
                    "row": int(r),
                    "color": int(color),
                    "left_c": int(lc),
                    "right_c": int(rc),
                    "d_left": int(sep - lc),
                    "d_right": int(rc - sep),
                }
            )
    return pairs


def _paint_left_ray(out: np.ndarray, r: int, start_c: int, length: int, color: int) -> None:
    for dc in range(max(0, int(length))):
        c = start_c - dc
        if c < 0:
            break
        out[r, c] = int(color)


def _paint_right_ray(out: np.ndarray, r: int, start_c: int, length: int, color: int) -> None:
    width = out.shape[1]
    for dc in range(max(0, int(length))):
        c = start_c + dc
        if c >= width:
            break
        out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each same-color cross-barrier pair onto the barrier edge, then emit outward rays whose lengths are inherited from the opposite-side partner.

    Input structure:
    - The grid has black background and one full-height vertical orange barrier column.
    - Some rows contain exactly one single-pixel marker on each side of the barrier; a valid pair uses the same non-black, non-orange color on both sides.
    - Other rows may contain unmatched side markers, which are not part of any transform.

    Rule:
    - For every row with exactly one left marker and one right marker of the same color, measure the left marker's distance to the barrier as D_L and the right marker's distance as D_R.
    - Erase both original markers, move the left marker to the cell immediately left of the barrier and the right marker to the cell immediately right of the barrier, keeping their row and color.
    - From the new left marker position, paint a horizontal ray leftward using the pair color for D_R cells total, including the marker cell; from the new right marker position, paint a horizontal ray rightward for D_L cells total, including the marker cell.
    - The orange barrier stays fixed, rays clip at the grid boundary, and rows without a valid same-color pair stay unchanged.

    Key insight:
    - The movement and the ray lengths are coupled: each marker moves to the barrier-adjacent cell, but the length of its outgoing ray comes from the opposite-side partner's original distance rather than its own.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sep = _find_barrier_col(g)

    out = g.copy()
    for info in _valid_pairs(g):
        r = int(info["row"])
        color = int(info["color"])
        lc = int(info["left_c"])
        rc = int(info["right_c"])
        d_left = int(info["d_left"])
        d_right = int(info["d_right"])

        out[r, lc] = BG
        out[r, rc] = BG

        left_adj = sep - 1
        right_adj = sep + 1
        _paint_left_ray(out, r, left_adj, d_right, color)
        _paint_right_ray(out, r, right_adj, d_left, color)

    return out


def _wrong_same_distance(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep = _find_barrier_col(g)
    out = g.copy()
    for info in _valid_pairs(g):
        r = int(info["row"])
        color = int(info["color"])
        out[r, int(info["left_c"])] = BG
        out[r, int(info["right_c"])] = BG
        _paint_left_ray(out, r, sep - 1, int(info["d_left"]), color)
        _paint_right_ray(out, r, sep + 1, int(info["d_right"]), color)
    return out


def _wrong_swap_only(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep = _find_barrier_col(g)
    out = g.copy()
    for info in _valid_pairs(g):
        r = int(info["row"])
        color = int(info["color"])
        out[r, int(info["left_c"])] = BG
        out[r, int(info["right_c"])] = BG
        out[r, sep - 1] = color
        out[r, sep + 1] = color
    return out


def _sample_rows(rng: random.Random, height: int, count: int) -> list[int]:
    choices = list(range(height))
    for _ in range(200):
        rows = sorted(rng.sample(choices, count))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            return rows
    return sorted(rng.sample(choices, count))


def _choose_pair_distances(
    rng: random.Random,
    sep: int,
    right_room: int,
    pair_count: int,
) -> tuple[list[tuple[int, int]], bool]:
    left_max = min(6, sep)
    right_max = min(6, right_room)
    assert left_max >= 2 and right_max >= 2

    distances: list[tuple[int, int]] = []
    truncation = False
    need_trunc = (sep == 5 or right_room == 5) and rng.random() < 0.75

    if need_trunc and sep == 5 and right_max == 6:
        d_left = rng.randint(2, 5)
        distances.append((d_left, 6))
        truncation = True
    elif need_trunc and right_room == 5 and left_max == 6:
        d_right = rng.randint(2, 5)
        distances.append((6, d_right))
        truncation = True

    directions = [1, -1] * max(1, pair_count)
    rng.shuffle(directions)
    while len(distances) < pair_count:
        sign = directions[len(distances)]
        if sign > 0 and left_max >= 5 and right_max >= 2:
            d_left = rng.randint(max(2, min(4, left_max - 1)), left_max)
            d_right = rng.randint(2, min(right_max, max(2, d_left - 1)))
        elif sign < 0 and right_max >= 5 and left_max >= 2:
            d_right = rng.randint(max(2, min(4, right_max - 1)), right_max)
            d_left = rng.randint(2, min(left_max, max(2, d_right - 1)))
        else:
            d_left = rng.randint(2, left_max)
            d_right = rng.randint(2, right_max)
            if d_left == d_right:
                continue
        if d_left == d_right:
            continue
        distances.append((int(d_left), int(d_right)))

    rng.shuffle(distances)
    return distances, truncation


def _analyze_generated_grid(grid: np.ndarray) -> dict:
    sep = _find_barrier_col(grid)
    pairs = _valid_pairs(grid)
    other_rows = []
    for r in range(grid.shape[0]):
        left, right = _row_markers(grid[r], sep)
        if left or right:
            if not (len(left) == 1 and len(right) == 1 and left[0][1] == right[0][1]):
                other_rows.append(int(r))
        assert len(left) <= 1 and len(right) <= 1, "generator should keep rows unambiguous"
    return {"sep": sep, "pairs": pairs, "other_rows": other_rows}


def _validate_semantics(grid: np.ndarray, expect_distractors: bool, expect_truncation: bool) -> bool:
    info = _analyze_generated_grid(grid)
    pairs = info["pairs"]
    sep = int(info["sep"])
    out = main(grid)

    if len(pairs) not in (3, 4):
        return False
    if len({int(p["row"]) for p in pairs}) != len(pairs):
        return False
    if len({int(p["color"]) for p in pairs}) != len(pairs):
        return False
    if any(not (2 <= int(p["d_left"]) <= 6 and 2 <= int(p["d_right"]) <= 6) for p in pairs):
        return False
    if any(int(p["d_left"]) == int(p["d_right"]) for p in pairs):
        return False
    if not any(int(p["d_left"]) < int(p["d_right"]) for p in pairs):
        return False
    if not any(int(p["d_left"]) > int(p["d_right"]) for p in pairs):
        return False

    truncation_seen = False
    for p in pairs:
        r = int(p["row"])
        color = int(p["color"])
        d_left = int(p["d_left"])
        d_right = int(p["d_right"])
        if out[r, sep] != BARRIER:
            return False
        if out[r, sep - 1] != color or out[r, sep + 1] != color:
            return False

        left_expected = set(range(max(0, sep - d_right), sep))
        right_expected = set(range(sep + 1, min(out.shape[1], sep + 1 + d_left)))

        for c in range(sep):
            should = c in left_expected
            if should != (int(out[r, c]) == color):
                return False
        for c in range(sep + 1, out.shape[1]):
            should = c in right_expected
            if should != (int(out[r, c]) == color):
                return False

        if d_right > sep or (sep + d_left) >= out.shape[1]:
            truncation_seen = True

    if expect_distractors and not info["other_rows"]:
        return False
    if expect_distractors:
        for r in info["other_rows"]:
            if not np.array_equal(out[r], grid[r]):
                return False

    if expect_truncation and not truncation_seen:
        return False

    if np.array_equal(out, _wrong_same_distance(grid)):
        return False
    if np.array_equal(out, _wrong_swap_only(grid)):
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(13, 19)
    width = rng.randint(13, 18)
    if width < 13:
        return None

    sep = rng.randint(5, width - 6)
    right_room = width - sep - 1
    pair_count = rng.randint(3, 4)
    distractor_count = rng.choice([0, 1, 1, 2])
    total_rows = pair_count + distractor_count
    if total_rows > height:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[:, sep] = BARRIER

    active_rows = _sample_rows(rng, height, total_rows)
    pair_rows = active_rows[:pair_count]
    distractor_rows = active_rows[pair_count:]

    colors = list(rng.sample(PAIR_COLORS, pair_count))
    distances, truncation = _choose_pair_distances(rng, sep, right_room, pair_count)

    for row, color, (d_left, d_right) in zip(pair_rows, colors, distances):
        left_c = sep - d_left
        right_c = sep + d_right
        if left_c < 0 or right_c >= width:
            return None
        grid[row, left_c] = int(color)
        grid[row, right_c] = int(color)

    for row in distractor_rows:
        side = rng.choice(["L", "R"])
        color = rng.choice(PAIR_COLORS)
        if side == "L":
            max_d = min(6, sep)
            if max_d < 2:
                return None
            d = rng.randint(2, max_d)
            grid[row, sep - d] = int(color)
        else:
            max_d = min(6, right_room)
            if max_d < 2:
                return None
            d = rng.randint(2, max_d)
            grid[row, sep + d] = int(color)

    expect_distractors = distractor_count > 0
    if not _validate_semantics(grid, expect_distractors=expect_distractors, expect_truncation=truncation):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
