from __future__ import annotations

import random

import numpy as np


SEP = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
TRANSFORMS = ("rot180", "flip_lr", "flip_ud")

ROWCOL_RANGE = (3, 5)
BLOCK_SIZE_CHOICES = (2, 3, 4, 5, 6)
GEN_ATTEMPTS = 1200


def _non_separator_spans(is_separator: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    n = int(is_separator.size)
    while i < n:
        if bool(is_separator[i]):
            i += 1
            continue
        j = i
        while j < n and not bool(is_separator[j]):
            j += 1
        spans.append((i, j))
        i = j
    return spans


def _compress_blocks(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    row_spans = _non_separator_spans(np.all(g == SEP, axis=1))
    col_spans = _non_separator_spans(np.all(g == SEP, axis=0))
    assert len(row_spans) >= 2 and len(col_spans) >= 2, "expected a separator-defined block matrix"

    matrix = np.empty((len(row_spans), len(col_spans)), dtype=int)
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            block = g[r0:r1, c0:c1]
            color = int(block[0, 0])
            assert color != SEP, "block interiors cannot be separator-colored"
            assert np.all(block == color), "each block must be monochrome"
            matrix[i, j] = color
    return matrix, row_spans, col_spans


def _paint_blocks(
    template: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    matrix: np.ndarray,
) -> np.ndarray:
    out = np.array(template, copy=True)
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            out[r0:r1, c0:c1] = int(matrix[i, j])
    return out


def _partner(shape: tuple[int, int], transform: str, r: int, c: int) -> tuple[int, int]:
    h, w = map(int, shape)
    if transform == "rot180":
        return h - 1 - int(r), w - 1 - int(c)
    if transform == "flip_lr":
        return int(r), w - 1 - int(c)
    if transform == "flip_ud":
        return h - 1 - int(r), int(c)
    raise ValueError(f"unknown transform: {transform}")


def _orbits(shape: tuple[int, int], transform: str) -> list[list[tuple[int, int]]]:
    h, w = map(int, shape)
    seen: set[tuple[int, int]] = set()
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            mate = _partner(shape, transform, r, c)
            orbit = sorted({(r, c), mate})
            seen.update(orbit)
            out.append(orbit)
    return out


def _record_pair(mapping: dict[int, int], left: int, right: int) -> bool:
    left = int(left)
    right = int(right)
    prev_left = mapping.get(left)
    prev_right = mapping.get(right)
    if prev_left is not None and prev_left != right:
        return False
    if prev_right is not None and prev_right != left:
        return False
    mapping[left] = right
    mapping[right] = left
    return True


def _analyze_transform(matrix: np.ndarray, transform: str) -> dict:
    mapping: dict[int, int] = {}
    contradictions = 0
    observed_pairs = 0
    for orbit in _orbits(matrix.shape, transform):
        colors = [int(matrix[r, c]) for r, c in orbit]
        if any(color == GREY for color in colors):
            continue
        left = colors[0]
        right = colors[0] if len(colors) == 1 else colors[1]
        observed_pairs += 1
        if not _record_pair(mapping, left, right):
            contradictions += 1
    return {
        "transform": transform,
        "mapping": mapping,
        "observed_pairs": observed_pairs,
        "contradictions": contradictions,
        "consistent": contradictions == 0 and observed_pairs > 0,
    }


def _reconstruct_matrix(matrix: np.ndarray, transform: str, mapping: dict[int, int]) -> np.ndarray:
    out = np.array(matrix, copy=True)
    for orbit in _orbits(out.shape, transform):
        if len(orbit) == 1:
            r, c = orbit[0]
            color = int(out[r, c])
            assert color != GREY, "a fixed-point block cannot be grey"
            assert mapping.get(color) == color, "fixed-point blocks must use self-paired colors"
            continue

        (r0, c0), (r1, c1) = orbit
        a = int(out[r0, c0])
        b = int(out[r1, c1])
        if a == GREY and b == GREY:
            raise AssertionError("a grey block must have a visible partner")
        if a == GREY:
            fill = mapping.get(b)
            assert fill is not None and fill != GREY
            out[r0, c0] = int(fill)
        elif b == GREY:
            fill = mapping.get(a)
            assert fill is not None and fill != GREY
            out[r1, c1] = int(fill)
        else:
            assert mapping.get(a) == b and mapping.get(b) == a
    assert not np.any(out == GREY), "all grey blocks must be recoverable"
    return out


def _solve_compressed(matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    winners: list[dict] = []
    for transform in TRANSFORMS:
        info = _analyze_transform(matrix, transform)
        if not info["consistent"]:
            continue
        try:
            solved = _reconstruct_matrix(matrix, transform, info["mapping"])
        except AssertionError:
            continue
        info["solved"] = solved
        winners.append(info)
    assert len(winners) == 1, "expected exactly one self-consistent hidden transform"
    return winners[0]["solved"], winners[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a separator-split block matrix, identify the only color-pairing transform, and unmask the grey blocks.

    Input structure:
    - Full black separator rows and columns split the canvas into a matrix of rectangular monochrome blocks.
    - Block heights and widths vary, so any useful relation lives on the compressed block coordinates rather than raw pixels.
    - Grey appears only as an occlusion color on some blocks; every grey block has a non-grey partner under the hidden transform.

    Rule:
    - Ignore block sizes and read the scene as a compressed color matrix.
    - Test the three candidate transforms on that matrix: 180-degree rotation, left-right mirror, and up-down mirror.
    - From every non-grey block pair under one candidate, build a bidirectional color-pair table; keep the unique candidate with no contradictions.
    - Recolor each grey block from its visible partner through that inferred table, while leaving separators and non-grey blocks unchanged.

    Key insight:
    - The hidden relation is not equality under a visible symmetry; it is consistency of repeated color pairings after coordinate compression, so the transform must be discovered by hypothesis testing.
    """
    g = np.asarray(grid, dtype=int)
    compressed, row_spans, col_spans = _compress_blocks(g)
    solved, _info = _solve_compressed(compressed)
    return _paint_blocks(g, row_spans, col_spans, solved)


def _assign_orbit(hidden: np.ndarray, orbit: list[tuple[int, int]], color: int, mapping: dict[int, int]) -> None:
    first = int(color)
    second = int(mapping[first])
    r0, c0 = orbit[0]
    hidden[r0, c0] = first
    if len(orbit) == 1:
        assert first == second
        return
    r1, c1 = orbit[1]
    hidden[r1, c1] = second


def _sample_sizes(count: int) -> list[int]:
    sizes = random.sample(BLOCK_SIZE_CHOICES, int(count))
    random.shuffle(sizes)
    return sizes


def _render_matrix(matrix: np.ndarray, row_sizes: list[int], col_sizes: list[int]) -> np.ndarray:
    rows = len(row_sizes)
    cols = len(col_sizes)
    h = sum(row_sizes) + rows - 1
    w = sum(col_sizes) + cols - 1
    grid = np.full((h, w), SEP, dtype=int)

    r = 0
    for i, block_h in enumerate(row_sizes):
        c = 0
        for j, block_w in enumerate(col_sizes):
            grid[r : r + block_h, c : c + block_w] = int(matrix[i, j])
            c += block_w
            if j != cols - 1:
                c += 1
        r += block_h
        if i != rows - 1:
            r += 1
    return grid


def _build_case() -> tuple[np.ndarray, np.ndarray, str] | None:
    rows = random.randint(*ROWCOL_RANGE)
    cols = random.randint(*ROWCOL_RANGE)
    transform = random.choice(TRANSFORMS)
    orbits = _orbits((rows, cols), transform)
    pair_orbits = [orbit for orbit in orbits if len(orbit) == 2]
    fixed_orbits = [orbit for orbit in orbits if len(orbit) == 1]
    if len(pair_orbits) < 4:
        return None

    random.shuffle(pair_orbits)
    random.shuffle(fixed_orbits)
    colors = list(PALETTE)
    random.shuffle(colors)

    n_swaps = 2 if len(pair_orbits) >= 7 and random.random() < 0.65 else 1
    swap_pairs: list[tuple[int, int]] = []
    idx = 0
    for _ in range(n_swaps):
        swap_pairs.append((colors[idx], colors[idx + 1]))
        idx += 2
    self_color = int(colors[idx])

    mapping: dict[int, int] = {self_color: self_color}
    for left, right in swap_pairs:
        mapping[int(left)] = int(right)
        mapping[int(right)] = int(left)

    self_uses_pair = 0 if fixed_orbits else 1
    max_gray = len(pair_orbits) - n_swaps - self_uses_pair
    if max_gray < 1:
        return None
    n_gray = 1 + int(max_gray >= 2 and random.random() < 0.5)
    n_gray = min(n_gray, max_gray)

    hidden = np.full((rows, cols), -1, dtype=int)
    grey_cells: list[tuple[int, int]] = []

    for left, right in swap_pairs:
        orbit = pair_orbits.pop()
        _assign_orbit(hidden, orbit, random.choice((left, right)), mapping)

    self_orbit = fixed_orbits.pop() if fixed_orbits else pair_orbits.pop()
    _assign_orbit(hidden, self_orbit, self_color, mapping)

    for _ in range(n_gray):
        left, right = random.choice(swap_pairs)
        orbit = pair_orbits.pop()
        _assign_orbit(hidden, orbit, random.choice((left, right)), mapping)
        grey_cells.append(random.choice(orbit))

    choice_pool = [self_color] + [color for pair in swap_pairs for color in pair] * 2
    for orbit in pair_orbits:
        _assign_orbit(hidden, orbit, random.choice(choice_pool), mapping)
    for orbit in fixed_orbits:
        _assign_orbit(hidden, orbit, self_color, mapping)

    if np.any(hidden < 0):
        return None

    masked = np.array(hidden, copy=True)
    for r, c in grey_cells:
        masked[r, c] = GREY
    return masked, hidden, transform


def _validate_semantics(masked: np.ndarray, hidden: np.ndarray, transform: str) -> bool:
    winners: list[str] = []
    analyses: dict[str, dict] = {}
    for candidate in TRANSFORMS:
        info = _analyze_transform(masked, candidate)
        analyses[candidate] = info
        if not info["consistent"]:
            continue
        try:
            solved = _reconstruct_matrix(masked, candidate, info["mapping"])
        except AssertionError:
            continue
        info["solved"] = solved
        winners.append(candidate)

    if winners != [transform]:
        return False
    if not np.array_equal(analyses[transform]["solved"], hidden):
        return False
    if not any(left == right for left, right in analyses[transform]["mapping"].items()):
        return False

    noncopy_gray = False
    for orbit in _orbits(masked.shape, transform):
        if len(orbit) != 2:
            continue
        (r0, c0), (r1, c1) = orbit
        a = int(masked[r0, c0])
        b = int(masked[r1, c1])
        if a == GREY and b != GREY:
            if analyses[transform]["mapping"].get(b) != int(hidden[r0, c0]):
                return False
            if int(hidden[r0, c0]) != b:
                noncopy_gray = True
        elif b == GREY and a != GREY:
            if analyses[transform]["mapping"].get(a) != int(hidden[r1, c1]):
                return False
            if int(hidden[r1, c1]) != a:
                noncopy_gray = True
        elif a == GREY and b == GREY:
            return False
    return noncopy_gray


def _gen_attempt() -> np.ndarray | None:
    built = _build_case()
    if built is None:
        return None
    masked, hidden, transform = built
    if not _validate_semantics(masked, hidden, transform):
        return None

    row_sizes = _sample_sizes(masked.shape[0])
    col_sizes = _sample_sizes(masked.shape[1])
    inp = _render_matrix(masked, row_sizes, col_sizes)
    expected = _render_matrix(hidden, row_sizes, col_sizes)

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if np.array_equal(inp, np.fliplr(inp)) or np.array_equal(inp, np.flipud(inp)) or np.array_equal(inp, np.flipud(np.fliplr(inp))):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid color-pairing block matrix")
