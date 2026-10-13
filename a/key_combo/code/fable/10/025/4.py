from __future__ import annotations

import numpy as np


SEPARATOR_COLOR = 5
DAMAGE_COLOR = 6
CONTENT_PALETTE = (1, 2, 3, 4, 7, 8, 9)
GEN_RETRIES = 400


def _infer_candidate(
    left: np.ndarray, right_aligned: np.ndarray, damage_color: int
) -> dict[str, object] | None:
    mapping: dict[int, int] = {}
    inverse: dict[int, int] = {}
    left_damage = 0
    right_damage = 0

    for left_color, right_color in zip(left.flat, right_aligned.flat):
        l = int(left_color)
        r = int(right_color)
        if l == damage_color and r == damage_color:
            return None
        if l != damage_color and r != damage_color:
            if l in mapping and mapping[l] != r:
                return None
            if r in inverse and inverse[r] != l:
                return None
            mapping[l] = r
            inverse[r] = l
        elif l == damage_color:
            left_damage += 1
        else:
            right_damage += 1

    damage_total = left_damage + right_damage
    if left_damage == 0 or right_damage == 0:
        return None
    if damage_total == 0 or damage_total > left.size // 4:
        return None
    if not mapping or len(mapping) != len(inverse):
        return None
    if damage_color in mapping or damage_color in inverse:
        return None
    if all(src == dst for src, dst in mapping.items()):
        return None

    repaired_left = left.copy()
    repaired_right = right_aligned.copy()
    for r in range(left.shape[0]):
        for c in range(left.shape[1]):
            if repaired_left[r, c] == damage_color:
                partner = int(repaired_right[r, c])
                if partner not in inverse:
                    return None
                repaired_left[r, c] = inverse[partner]
            elif repaired_right[r, c] == damage_color:
                partner = int(repaired_left[r, c])
                if partner not in mapping:
                    return None
                repaired_right[r, c] = mapping[partner]

    for left_color, right_color in zip(repaired_left.flat, repaired_right.flat):
        if mapping.get(int(left_color)) != int(right_color):
            return None

    return {
        "damage_color": int(damage_color),
        "mapping": mapping,
        "inverse": inverse,
        "left_damage": left_damage,
        "right_damage": right_damage,
        "repaired_left": repaired_left,
        "repaired_right_aligned": repaired_right,
    }


def _separator_and_halves(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    assert grid.ndim == 2, "expected a 2D grid"
    h, w = map(int, grid.shape)
    assert h > 0 and w >= 3 and w % 2 == 1, "expected two equal halves separated by one column"
    separator = w // 2
    separator_color = int(grid[0, separator])
    assert np.all(grid[:, separator] == separator_color), "middle separator column must be uniform"
    left = np.asarray(grid[:, :separator], dtype=int)
    right_aligned = np.fliplr(np.asarray(grid[:, separator + 1 :], dtype=int))
    assert left.shape == right_aligned.shape
    return separator, separator_color, left, right_aligned


def _solve_candidates(grid: np.ndarray) -> list[dict[str, object]]:
    separator, separator_color, left, right_aligned = _separator_and_halves(grid)
    del separator
    colors = sorted(set(map(int, np.unique(grid))) - {separator_color})
    solutions: list[dict[str, object]] = []
    for color in colors:
        candidate = _infer_candidate(left, right_aligned, color)
        if candidate is not None:
            candidate["occurrences"] = int(np.count_nonzero(grid == color))
            solutions.append(candidate)
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair both damaged mirror copies by inferring the hidden color permutation.

    Input structure:
    - A single full-height separator column splits the grid into two equal-width halves.
    - The right half is the left half reflected across the separator and then recolored by one fixed bijection.
    - A special damage color overwrites a few cells on both halves, and no mirror pair is damaged on both sides.

    Rule:
    - Pair each left cell with the horizontally mirrored cell on the right half.
    - Use all undamaged pairs to infer the color bijection from left colors to right colors.
    - Repair a damaged left cell by inverse-mapping its right partner, and repair a damaged right cell by forward-mapping its left partner.
    - Return the fully repaired grid, keeping the separator column unchanged.

    Key insight:
    - The halves are not plain mirrors: they are mirror images after a hidden color permutation, so the repair must reconstruct that permutation and use its direction correctly on each side.
    """
    g = np.asarray(grid, dtype=int)
    separator, separator_color, _left, _right_aligned = _separator_and_halves(g)
    solutions = _solve_candidates(g)
    assert solutions, "no valid damage color candidate found"

    solutions.sort(key=lambda item: (int(item["occurrences"]), int(item["damage_color"])))
    best = solutions[0]
    tied = [item for item in solutions if int(item["occurrences"]) == int(best["occurrences"])]
    assert len(tied) == 1, "damage color must be uniquely identifiable"

    repaired_left = np.asarray(best["repaired_left"], dtype=int)
    repaired_right = np.fliplr(np.asarray(best["repaired_right_aligned"], dtype=int))
    out = np.empty_like(g)
    out[:, :separator] = repaired_left
    out[:, separator] = separator_color
    out[:, separator + 1 :] = repaired_right
    return out


def _nonoverlapping_rectangle(
    rng: np.random.Generator, used: np.ndarray, height: int, width: int
) -> tuple[int, int, int, int] | None:
    for _ in range(80):
        rect_h = int(rng.integers(2, min(5, height) + 1))
        rect_w = int(rng.integers(2, min(5, width) + 1))
        r0 = int(rng.integers(0, height - rect_h + 1))
        c0 = int(rng.integers(0, width - rect_w + 1))
        if used[r0 : r0 + rect_h, c0 : c0 + rect_w].any():
            continue
        return r0, c0, rect_h, rect_w
    return None


def _generate_left_half(
    rng: np.random.Generator, height: int, width: int, colors: list[int]
) -> np.ndarray | None:
    base = colors[0]
    left = np.full((height, width), base, dtype=int)
    used = np.zeros((height, width), dtype=bool)

    for color in colors[1:]:
        rect = _nonoverlapping_rectangle(rng, used, height, width)
        if rect is None:
            return None
        r0, c0, rect_h, rect_w = rect
        used[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
        left[r0 : r0 + rect_h, c0 : c0 + rect_w] = color

    for _ in range(int(rng.integers(2, 7))):
        color = int(rng.choice(colors))
        mode = int(rng.integers(0, 4))
        if mode == 0:
            rect_h = int(rng.integers(1, min(4, height) + 1))
            rect_w = int(rng.integers(1, min(4, width) + 1))
            r0 = int(rng.integers(0, height - rect_h + 1))
            c0 = int(rng.integers(0, width - rect_w + 1))
            patch = left[r0 : r0 + rect_h, c0 : c0 + rect_w]
            mask = (patch == base) | (patch == color)
            if np.any(mask):
                patch[mask] = color
        elif mode == 1:
            row = int(rng.integers(0, height))
            c0 = int(rng.integers(0, width))
            c1 = int(rng.integers(c0, width))
            segment = left[row, c0 : c1 + 1]
            mask = (segment == base) | (segment == color)
            if np.any(mask):
                segment[mask] = color
        elif mode == 2:
            col = int(rng.integers(0, width))
            r0 = int(rng.integers(0, height))
            r1 = int(rng.integers(r0, height))
            segment = left[r0 : r1 + 1, col]
            mask = (segment == base) | (segment == color)
            if np.any(mask):
                segment[mask] = color
        else:
            row = int(rng.integers(0, height))
            col = int(rng.integers(0, width))
            if left[row, col] in (base, color):
                left[row, col] = color

    counts = {color: int(np.count_nonzero(left == color)) for color in colors}
    if min(counts.values()) < 4:
        return None
    return left


def _cyclic_permutation(colors: list[int]) -> dict[int, int]:
    return {colors[i]: colors[(i + 1) % len(colors)] for i in range(len(colors))}


def _pick_damage_positions(
    rng: np.random.Generator, left: np.ndarray, total_damage: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    coords = [(int(r), int(c)) for r in range(left.shape[0]) for c in range(left.shape[1])]
    rng.shuffle(coords)
    remaining = {int(color): int(np.count_nonzero(left == color)) for color in np.unique(left)}
    left_damage: list[tuple[int, int]] = []
    right_damage: list[tuple[int, int]] = []

    for side in ("left", "right"):
        placed = False
        for r, c in coords:
            color = int(left[r, c])
            if remaining[color] <= 1:
                continue
            if (r, c) in left_damage or (r, c) in right_damage:
                continue
            if side == "left":
                left_damage.append((r, c))
            else:
                right_damage.append((r, c))
            remaining[color] -= 1
            placed = True
            break
        if not placed:
            return None

    for r, c in coords:
        if len(left_damage) + len(right_damage) >= total_damage:
            break
        color = int(left[r, c])
        if remaining[color] <= 1:
            continue
        if (r, c) in left_damage or (r, c) in right_damage:
            continue
        if rng.random() < 0.5:
            left_damage.append((r, c))
        else:
            right_damage.append((r, c))
        remaining[color] -= 1

    if len(left_damage) + len(right_damage) != total_damage:
        return None
    return left_damage, right_damage


def _build_input(
    left: np.ndarray,
    sigma: dict[int, int],
    left_damage: list[tuple[int, int]],
    right_damage: list[tuple[int, int]],
) -> np.ndarray:
    right = np.vectorize(lambda x: sigma[int(x)])(np.fliplr(left)).astype(int)
    height, width = left.shape
    grid = np.full((height, 2 * width + 1), SEPARATOR_COLOR, dtype=int)
    grid[:, :width] = left
    grid[:, width + 1 :] = right
    for r, c in left_damage:
        grid[r, c] = DAMAGE_COLOR
    for r, c in right_damage:
        grid[r, 2 * width - c] = DAMAGE_COLOR
    return grid


def _expected_output(left: np.ndarray, sigma: dict[int, int]) -> np.ndarray:
    right = np.vectorize(lambda x: sigma[int(x)])(np.fliplr(left)).astype(int)
    height, width = left.shape
    out = np.full((height, 2 * width + 1), SEPARATOR_COLOR, dtype=int)
    out[:, :width] = left
    out[:, width + 1 :] = right
    return out


def _shortcut_checks(
    left: np.ndarray,
    sigma: dict[int, int],
    left_damage: list[tuple[int, int]],
    right_damage: list[tuple[int, int]],
) -> bool:
    for r, c in left_damage:
        true_left = int(left[r, c])
        mirrored_right = int(sigma[true_left])
        if mirrored_right == true_left:
            return False
        if int(sigma[mirrored_right]) == true_left:
            return False
    inverse = {dst: src for src, dst in sigma.items()}
    for r, c in right_damage:
        true_left = int(left[r, c])
        true_right = int(sigma[true_left])
        if true_left == true_right:
            return False
        if inverse[true_left] == true_right:
            return False
    return True


def _semantic_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    sigma: dict[int, int],
    left_damage: list[tuple[int, int]],
    right_damage: list[tuple[int, int]],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False
    if np.count_nonzero(inp == DAMAGE_COLOR) != len(left_damage) + len(right_damage):
        return False
    if not left_damage or not right_damage:
        return False
    if not _shortcut_checks(expected[:, : expected.shape[1] // 2], sigma, left_damage, right_damage):
        return False

    solutions = _solve_candidates(inp)
    if not solutions:
        return False
    solutions.sort(key=lambda item: (int(item["occurrences"]), int(item["damage_color"])))
    if int(solutions[0]["damage_color"]) != DAMAGE_COLOR:
        return False
    if len(solutions) > 1 and int(solutions[1]["occurrences"]) == int(solutions[0]["occurrences"]):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        height = int(rng.integers(7, 13))
        width = int(rng.integers(7, 13))
        color_count = int(rng.integers(3, 6))
        colors = list(map(int, rng.choice(CONTENT_PALETTE, size=color_count, replace=False)))
        left = _generate_left_half(rng, height, width, colors)
        if left is None:
            continue

        sigma = _cyclic_permutation(colors)
        total_damage = int(rng.integers(2, 4))
        damage_positions = _pick_damage_positions(rng, left, total_damage)
        if damage_positions is None:
            continue
        left_damage, right_damage = damage_positions

        inp = _build_input(left, sigma, left_damage, right_damage)
        expected = _expected_output(left, sigma)
        if inp.shape[0] > 30 or inp.shape[1] > 30:
            continue
        if _semantic_ok(inp, expected, sigma, left_damage, right_damage):
            return inp

    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
