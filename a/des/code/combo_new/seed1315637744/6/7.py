from __future__ import annotations

import random

import numpy as np


BLACK = 0
GRAY = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = (
    ("up", -1, 0),
    ("down", 1, 0),
    ("left", 0, -1),
    ("right", 0, 1),
)

HEIGHT_RANGE = (8, 16)
WIDTH_RANGE = (8, 16)
PAIR_COUNT_RANGE = (2, 6)
GEN_RETRIES = 500
KEEPER_RETRIES = 300
TWIN_RETRIES = 300


def _checker_value(r: int, c: int, gray_phase: int) -> int:
    return GRAY if (r + c + gray_phase) % 2 == 0 else BLACK


def _make_checkerboard(shape: tuple[int, int], gray_phase: int) -> np.ndarray:
    h, w = shape
    grid = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            grid[r, c] = _checker_value(r, c, gray_phase)
    return grid


def _infer_gray_phase(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    candidates: list[int] = []
    for gray_phase in (0, 1):
        ok = True
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                value = int(g[r, c])
                if value in (BLACK, GRAY) and value != _checker_value(r, c, gray_phase):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            candidates.append(gray_phase)

    assert len(candidates) == 1, "expected a unique black/gray checkerboard phase"
    return candidates[0]


def _gray_cells(shape: tuple[int, int], gray_phase: int) -> list[tuple[int, int]]:
    h, w = shape
    return [
        (r, c)
        for r in range(h)
        for c in range(w)
        if _checker_value(r, c, gray_phase) == GRAY
    ]


def _black_cells(shape: tuple[int, int], gray_phase: int) -> list[tuple[int, int]]:
    h, w = shape
    return [
        (r, c)
        for r in range(h)
        for c in range(w)
        if _checker_value(r, c, gray_phase) == BLACK
    ]


def _pair_data(grid: np.ndarray, gray_phase: int) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) not in (BLACK, GRAY)]
    assert colors, "expected colored singleton pairs on the checkerboard"

    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each non-background color must appear exactly twice"
        black_pos: tuple[int, int] | None = None
        gray_pos: tuple[int, int] | None = None
        for r, c in coords:
            pos = (int(r), int(c))
            if _checker_value(pos[0], pos[1], gray_phase) == GRAY:
                assert gray_pos is None, "each color must occupy exactly one gray cell"
                gray_pos = pos
            else:
                assert black_pos is None, "each color must occupy exactly one black cell"
                black_pos = pos
        assert black_pos is not None and gray_pos is not None
        pairs.append((color, black_pos, gray_pos))

    return pairs


def _targets_for_keepers(
    keepers: list[tuple[int, int]], shape: tuple[int, int]
) -> tuple[dict[tuple[int, int], dict[str, tuple[int, int]]], set[str], bool]:
    h, w = shape
    keeper_set = set(keepers)
    targets: dict[tuple[int, int], dict[str, tuple[int, int]]] = {}
    used_dirs: set[str] = set()
    has_blocked_jump = False

    for keeper in keepers:
        kr, kc = keeper
        local: dict[str, tuple[int, int]] = {}
        for name, dr, dc in DIRS:
            step = 1
            skipped = False
            while True:
                nr = kr + 2 * step * dr
                nc = kc + 2 * step * dc
                if not (0 <= nr < h and 0 <= nc < w):
                    break
                if (nr, nc) in keeper_set:
                    skipped = True
                    step += 1
                    continue
                local[name] = (nr, nc)
                used_dirs.add(name)
                if skipped:
                    has_blocked_jump = True
                break
        targets[keeper] = local

    return targets, used_dirs, has_blocked_jump


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    gray_phase = _infer_gray_phase(g)
    pairs = _pair_data(g, gray_phase)

    base = g.copy()
    keepers: list[tuple[int, int]] = []
    keeper_colors: dict[tuple[int, int], int] = {}
    for color, black_pos, gray_pos in pairs:
        assert int(base[black_pos]) == color and int(base[gray_pos]) == color
        base[black_pos] = BLACK
        keepers.append(gray_pos)
        keeper_colors[gray_pos] = color

    targets, _, _ = _targets_for_keepers(keepers, g.shape)
    destinations: list[tuple[int, int]] = []
    for gray_pos in keepers:
        destinations.extend(targets[gray_pos].values())
    assert len(destinations) == len(set(destinations)), "copy targets must be unique"

    out = base.copy()
    for gray_pos in keepers:
        for target in targets[gray_pos].values():
            tr, tc = target
            assert int(base[tr, tc]) == GRAY, "copies must land on empty gray cells"
            out[tr, tc] = keeper_colors[gray_pos]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each black-square twin and copy each gray-square twin outward along the gray checkerboard lanes.

    Input structure:
    - The background is a two-color checkerboard made from black and gray cells.
    - Colored pieces are singleton pixels placed on top of that checkerboard, with each non-background color appearing exactly twice.
    - For every color pair, one pixel sits on a black checkerboard square and the other sits on a gray checkerboard square.

    Rule:
    - Infer which cells belong to the gray parity of the checkerboard.
    - For each color pair, keep the pixel on the gray square, but erase the pixel on the black square back to black.
    - From each kept gray-square pixel, copy that color to the nearest empty gray-square cell in each cardinal direction, skipping over occupied gray squares if needed.

    Key insight:
    - The colored pairs are distinguished by the hidden checkerboard parity underneath them, and the new copies must follow two-step gray-square lanes rather than ordinary adjacent cells.
    """
    return _transform(grid)


def _all_far_enough(points: list[tuple[int, int]], min_manhattan: int) -> bool:
    for i, (r0, c0) in enumerate(points):
        for r1, c1 in points[i + 1 :]:
            if abs(r0 - r1) + abs(c0 - c1) < min_manhattan:
                return False
    return True


def _sample_keepers(
    shape: tuple[int, int], gray_phase: int, pair_count: int, blocked_mode: bool
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], dict[str, tuple[int, int]]]] | None:
    gray_cells = _gray_cells(shape, gray_phase)
    h, w = shape

    for _ in range(KEEPER_RETRIES):
        keepers: list[tuple[int, int]] = []
        if blocked_mode:
            options: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for r, c in gray_cells:
                for _, dr, dc in DIRS:
                    mid = (r + 2 * dr, c + 2 * dc)
                    end = (r + 4 * dr, c + 4 * dc)
                    if 0 <= mid[0] < h and 0 <= mid[1] < w and 0 <= end[0] < h and 0 <= end[1] < w:
                        if _checker_value(mid[0], mid[1], gray_phase) == GRAY and _checker_value(
                            end[0], end[1], gray_phase
                        ) == GRAY:
                            options.append(((r, c), mid))
            if not options:
                return None
            first, second = random.choice(options)
            keepers.extend([first, second])

        remaining = [cell for cell in gray_cells if cell not in keepers]
        random.shuffle(remaining)
        for cell in remaining:
            if len(keepers) == pair_count:
                break
            if all(abs(cell[0] - r) + abs(cell[1] - c) >= 2 for r, c in keepers):
                keepers.append(cell)

        if len(keepers) != pair_count:
            continue

        random.shuffle(keepers)
        targets, used_dirs, has_blocked_jump = _targets_for_keepers(keepers, shape)
        if any(len(local) == 0 for local in targets.values()):
            continue
        if used_dirs != {"up", "down", "left", "right"}:
            continue
        if blocked_mode and not has_blocked_jump:
            continue

        destinations = [dst for local in targets.values() for dst in local.values()]
        if len(destinations) != len(set(destinations)):
            continue
        if not _all_far_enough(keepers + destinations, 2):
            continue
        return keepers, targets

    return None


def _sample_twins(
    shape: tuple[int, int], gray_phase: int, keepers: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    black_cells = _black_cells(shape, gray_phase)
    occupied = set(keepers)

    for _ in range(TWIN_RETRIES):
        twins = random.sample(black_cells, len(keepers))
        all_points = keepers + twins
        if len(set(twins)) != len(twins):
            continue
        if any(twin in occupied for twin in twins):
            continue
        if not _all_far_enough(all_points, 2):
            continue
        return twins

    return None


def _build_expected(
    shape: tuple[int, int],
    gray_phase: int,
    keepers: list[tuple[int, int]],
    twins: list[tuple[int, int]],
    colors: list[int],
    targets: dict[tuple[int, int], dict[str, tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = _make_checkerboard(shape, gray_phase)
    out = _make_checkerboard(shape, gray_phase)

    for color, keeper, twin in zip(colors, keepers, twins, strict=False):
        inp[keeper] = color
        inp[twin] = color
        out[keeper] = color
        for target in targets[keeper].values():
            out[target] = color

    return inp, out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*HEIGHT_RANGE)
    w = random.randint(*WIDTH_RANGE)
    gray_phase = random.randint(0, 1)
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    blocked_mode = random.randrange(2) == 0

    max_pairs = min(len(PIECE_COLORS), (h * w) // 12)
    if pair_count > max_pairs:
        pair_count = max(2, max_pairs)
    if pair_count > len(PIECE_COLORS):
        return None

    keeper_bundle = _sample_keepers((h, w), gray_phase, pair_count, blocked_mode)
    if keeper_bundle is None:
        return None
    keepers, targets = keeper_bundle

    twins = _sample_twins((h, w), gray_phase, keepers)
    if twins is None:
        return None

    colors = random.sample(list(PIECE_COLORS), pair_count)
    inp, out = _build_expected((h, w), gray_phase, keepers, twins, colors, targets)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _generate_case()
        if candidate is not None:
            inp, _ = candidate
            return inp
    raise RuntimeError("failed to generate a valid checkerboard parity puzzle")
