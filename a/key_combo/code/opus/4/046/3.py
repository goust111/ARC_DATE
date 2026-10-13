from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GRAY = 5
MACRO = 3
GRID_MACRO_OPTIONS = (5, 7, 9)
GEN_RETRIES = 400


def _rotate_pos(pos: tuple[int, int], size: int, turns: int = 1) -> tuple[int, int]:
    """Rotate one macro-cell position clockwise around the macro-grid center."""
    r, c = map(int, pos)
    n = int(size)
    for _ in range(turns % 4):
        r, c = c, n - 1 - r
    return r, c


def _orbit(pos: tuple[int, int], size: int) -> tuple[tuple[int, int], ...]:
    return tuple(_rotate_pos(pos, size, turns) for turns in range(4))


def _paint_block(grid: np.ndarray, pos: tuple[int, int], color: int) -> None:
    mr, mc = map(int, pos)
    r0 = mr * MACRO
    c0 = mc * MACRO
    grid[r0 : r0 + MACRO, c0 : c0 + MACRO] = int(color)


def _paint_marker(grid: np.ndarray, pos: tuple[int, int]) -> None:
    mr, mc = map(int, pos)
    grid[mr * MACRO + 1, mc * MACRO + 1] = GRAY


def _macro_patch(grid: np.ndarray, pos: tuple[int, int]) -> np.ndarray:
    mr, mc = map(int, pos)
    r0 = mr * MACRO
    c0 = mc * MACRO
    return grid[r0 : r0 + MACRO, c0 : c0 + MACRO]


def _is_original_blue_block(grid: np.ndarray, pos: tuple[int, int]) -> bool:
    patch = _macro_patch(grid, pos)
    center = int(patch[1, 1])
    if center not in (BLUE, GRAY):
        return False
    mask = np.ones((MACRO, MACRO), dtype=bool)
    mask[1, 1] = False
    return bool(np.all(patch[mask] == BLUE))


def _has_gray_marker(grid: np.ndarray, pos: tuple[int, int]) -> bool:
    patch = _macro_patch(grid, pos)
    if int(patch[1, 1]) != GRAY:
        return False
    mask = np.ones((MACRO, MACRO), dtype=bool)
    mask[1, 1] = False
    return bool(np.all(patch[mask] == BG))


def _detect_sets(grid: np.ndarray) -> tuple[int, set[tuple[int, int]], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w and h % MACRO == 0, "expected a square grid tiled by 3x3 macro blocks"

    macro_n = h // MACRO
    assert macro_n % 2 == 1, "macro grid must have odd side length for a unique rotation center"

    originals: set[tuple[int, int]] = set()
    enabled: set[tuple[int, int]] = set()
    allowed = {BG, BLUE, GRAY}
    assert set(map(int, np.unique(g))).issubset(allowed), "input must use only black, blue, and gray"

    for mr in range(macro_n):
        for mc in range(macro_n):
            pos = (mr, mc)
            patch = _macro_patch(g, pos)
            if _is_original_blue_block(g, pos):
                originals.add(pos)
            elif _has_gray_marker(g, pos):
                enabled.add(pos)
            else:
                assert np.all(patch == BG), "every macro block must be empty, marker-only, or blue-filled"

    assert originals, "expected at least one original blue macro block"
    return macro_n, originals, enabled


def _symmetric_closure(points: Iterable[tuple[int, int]], size: int) -> set[tuple[int, int]]:
    closure: set[tuple[int, int]] = set()
    for pos in points:
        closure.update(_orbit(pos, size))
    return closure


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a fourfold rotational macro pattern, but only fill inferred blocks when their marker gate is active.

    Input structure:
    - The grid is a square whose side length is a multiple of 3, interpreted as an odd-by-odd array of 3x3 macro blocks on black background.
    - Some macro blocks are visible blue corner clues: every non-center cell in that 3x3 block is blue, and the center is blue or gray.
    - Independently, some otherwise empty macro blocks contain a single gray center marker.

    Rule:
    - Read the visible blue macro-block positions in macro coordinates and close them under 0/90/180/270 degree rotation about the macro-grid center, producing the ideal symmetric set S.
    - Read the gray-center marker positions as the enabled set E.
    - Keep every original blue clue block blue.
    - For every other position in S, fill the whole 3x3 macro block blue iff that position is also in E; leave it black otherwise.
    - For every position in E but not in S, fill the whole 3x3 macro block red, and remove all gray markers from the output.

    Key insight:
    - The final color of a macro block is determined by the set relation between two independently recovered patterns on the same macro lattice: rotational symmetry predicts where blocks could exist, and gray markers decide which predicted blocks are activated.
    """
    macro_n, originals, enabled = _detect_sets(grid)
    symmetric = _symmetric_closure(originals, macro_n)

    out = np.zeros((macro_n * MACRO, macro_n * MACRO), dtype=int)
    for pos in symmetric:
        if pos in originals or pos in enabled:
            _paint_block(out, pos, BLUE)
    for pos in enabled - symmetric:
        _paint_block(out, pos, RED)
    return out


def _corner_band(size: int) -> int:
    return 1 if size == 5 else random.randint(1, min(2, size // 2 - 1))


def _connected_corner_shape(size: int) -> set[tuple[int, int]]:
    band = _corner_band(size)
    candidates = {(r, c) for r in range(band + 1) for c in range(band + 1)}
    target = random.randint(1, min(4, len(candidates)))
    cluster = {random.choice(tuple(candidates))}

    while len(cluster) < target:
        frontier = []
        for r, c in cluster:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (r + dr, c + dc)
                if nxt in candidates and nxt not in cluster:
                    frontier.append(nxt)
        if not frontier:
            break
        cluster.add(random.choice(frontier))

    return cluster


def _visible_corner_turns() -> tuple[int, ...]:
    if random.random() < 0.5:
        return (random.randrange(4),)
    choices = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
    return random.choice(choices)


def _pick_enabled_sets(
    size: int,
    originals: set[tuple[int, int]],
    symmetric: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    inferred = list(symmetric - originals)
    if len(inferred) < 2:
        return None

    random.shuffle(inferred)
    blue_gate = {inferred[0]}
    blocked = inferred[1]
    extras_in_s = inferred[2:]

    outside = [(r, c) for r in range(size) for c in range(size) if (r, c) not in symmetric]
    if not outside:
        return None
    random.shuffle(outside)
    distractors = {outside[0]}

    for pos in extras_in_s:
        if random.random() < 0.45:
            blue_gate.add(pos)
    for pos in outside[1:]:
        if len(distractors) >= 4:
            break
        if random.random() < 0.28:
            distractors.add(pos)

    enabled = blue_gate | distractors
    assert blue_gate
    assert blocked not in enabled
    assert distractors - symmetric
    return enabled


def _render_input(size: int, originals: set[tuple[int, int]], enabled: set[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((size * MACRO, size * MACRO), dtype=int)
    for pos in originals:
        _paint_block(grid, pos, BLUE)
    for pos in enabled:
        if pos in originals:
            continue
        _paint_marker(grid, pos)
    return grid


def _expected_output(size: int, originals: set[tuple[int, int]], enabled: set[tuple[int, int]]) -> np.ndarray:
    symmetric = _symmetric_closure(originals, size)
    out = np.zeros((size * MACRO, size * MACRO), dtype=int)
    for pos in symmetric:
        if pos in originals or pos in enabled:
            _paint_block(out, pos, BLUE)
    for pos in enabled - symmetric:
        _paint_block(out, pos, RED)
    return out


def _validate_instance(inp: np.ndarray, originals: set[tuple[int, int]], enabled: set[tuple[int, int]]) -> bool:
    size = inp.shape[0] // MACRO
    symmetric = _symmetric_closure(originals, size)
    inferred = symmetric - originals

    if not inferred:
        return False
    if not (enabled & inferred):
        return False
    if not (inferred - enabled):
        return False
    if not (enabled - symmetric):
        return False

    try:
        parsed_size, parsed_originals, parsed_enabled = _detect_sets(inp)
    except AssertionError:
        return False
    if parsed_size != size or parsed_originals != originals or parsed_enabled != enabled:
        return False

    out = main(inp)
    expected = _expected_output(size, originals, enabled)
    if not np.array_equal(out, expected):
        return False

    for pos in originals:
        if not np.all(_macro_patch(out, pos) == BLUE):
            return False
    for pos in enabled & inferred:
        if not np.all(_macro_patch(out, pos) == BLUE):
            return False
    for pos in inferred - enabled:
        if not np.all(_macro_patch(out, pos) == BG):
            return False
    for pos in enabled - symmetric:
        if not np.all(_macro_patch(out, pos) == RED):
            return False
    return True


def _generate_case() -> np.ndarray | None:
    size = random.choice(GRID_MACRO_OPTIONS)
    base = _connected_corner_shape(size)
    visible_turns = _visible_corner_turns()

    originals: set[tuple[int, int]] = set()
    for turn in visible_turns:
        originals.update(_rotate_pos(pos, size, turn) for pos in base)
    symmetric = _symmetric_closure(base, size)

    if len(symmetric - originals) < 2:
        return None

    enabled = _pick_enabled_sets(size, originals, symmetric)
    if enabled is None:
        return None

    inp = _render_input(size, originals, enabled)
    if not _validate_instance(inp, originals, enabled):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _generate_case()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid gated rotational-symmetry instance")
