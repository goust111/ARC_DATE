import random
from collections import defaultdict

import numpy as np


GROUP_ORDER = ("horizontal", "vertical", "main_diag", "anti_diag")
GENERATION_TRIES = 600
MIN_SIZE = 9
MAX_SIZE = 15
MAX_EXTRA_PAIRS = 2
PALETTE = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _line_cells(shape: tuple[int, int], direction: str, anchor: int) -> list[tuple[int, int]]:
    h, w = shape
    cells: list[tuple[int, int]] = []
    if direction == "horizontal":
        return [(anchor, c) for c in range(w)]
    if direction == "vertical":
        return [(r, anchor) for r in range(h)]
    if direction == "main_diag":
        for r in range(h):
            c = r - anchor
            if 0 <= c < w:
                cells.append((r, c))
        return cells
    if direction == "anti_diag":
        for r in range(h):
            c = anchor - r
            if 0 <= c < w:
                cells.append((r, c))
        return cells
    raise AssertionError(f"unknown direction {direction}")


def _classify_pair(a: tuple[int, int], b: tuple[int, int]) -> tuple[str, int]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        return "horizontal", r0
    if c0 == c1:
        return "vertical", c0
    if r0 - c0 == r1 - c1:
        return "main_diag", r0 - c0
    if r0 + c0 == r1 + c1:
        return "anti_diag", r0 + c0
    raise AssertionError("same-color pixels must align horizontally, vertically, or diagonally")


def _extract_pairs(grid: np.ndarray, bg: int) -> dict[str, list[tuple[int, int]]]:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        coords = np.argwhere(grid == color)
        assert len(coords) == 2, "each foreground color must appear exactly twice"
        pair = tuple(map(tuple, coords.tolist()))
        direction, anchor = _classify_pair(pair[0], pair[1])
        groups[direction].append((anchor, color))

    for direction, items in groups.items():
        anchors = [anchor for anchor, _color in items]
        assert len(set(anchors)) == len(anchors), f"duplicate {direction} stripe anchors are ambiguous"
    return groups


def _render(shape: tuple[int, int], bg: int, groups: dict[str, list[tuple[int, int]]]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for direction in GROUP_ORDER:
        for anchor, color in sorted(groups.get(direction, []), key=lambda item: item[0]):
            for r, c in _line_cells(shape, direction, anchor):
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Turn same-color pixel pairs into full-span parallel bands."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    groups = _extract_pairs(g, bg)
    assert any(groups.values()), "expected at least one foreground pair"
    return _render(g.shape, bg, groups)


def _sample_spaced(values: list[int], count: int, gap: int, rng: random.Random) -> list[int] | None:
    values = list(values)
    for _ in range(200):
        rng.shuffle(values)
        chosen: list[int] = []
        for value in values:
            if all(abs(value - prev) >= gap for prev in chosen):
                chosen.append(value)
                if len(chosen) == count:
                    return chosen
    return None


def _diagonal_candidates(
    shape: tuple[int, int],
    direction: str,
    min_len: int,
    rows: set[int] | None = None,
    cols: set[int] | None = None,
) -> list[int]:
    h, w = shape
    if direction == "main_diag":
        anchors = range(-(w - 1), h)
    else:
        anchors = range(h + w - 1)

    result: list[int] = []
    for anchor in anchors:
        cells = _line_cells(shape, direction, anchor)
        if len(cells) < min_len:
            continue
        if rows is not None or cols is not None:
            if not any((rows is not None and r in rows) or (cols is not None and c in cols) for r, c in cells):
                continue
        result.append(anchor)
    return result


def _diag_intersects(shape: tuple[int, int], main_anchor: int, anti_anchor: int) -> bool:
    r_num = main_anchor + anti_anchor
    c_num = anti_anchor - main_anchor
    if (r_num & 1) or (c_num & 1):
        return False
    r = r_num // 2
    c = c_num // 2
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _mark_reserved(reserved: np.ndarray, r: int, c: int) -> None:
    h, w = reserved.shape
    r0 = max(0, r - 1)
    r1 = min(h, r + 2)
    c0 = max(0, c - 1)
    c1 = min(w, c + 2)
    reserved[r0:r1, c0:c1] = True


def _step_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _pick_pair(
    cells: list[tuple[int, int]],
    reserved: np.ndarray,
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    free_cells = [(r, c) for r, c in cells if not reserved[r, c]]
    if len(free_cells) < 2:
        return None

    target_gap = max(2, len(cells) // 3)
    pairs = [
        (a, b)
        for i, a in enumerate(free_cells)
        for b in free_cells[i + 1 :]
        if _step_distance(a, b) >= target_gap
    ]
    if not pairs:
        pairs = [
            (a, b)
            for i, a in enumerate(free_cells)
            for b in free_cells[i + 1 :]
            if _step_distance(a, b) >= 2
        ]
    if not pairs:
        return None
    return rng.choice(pairs)


def _build_descriptors(shape: tuple[int, int], rng: random.Random) -> list[tuple[str, int]] | None:
    h, w = shape
    counts = {direction: 1 for direction in GROUP_ORDER}
    for _ in range(rng.randint(0, MAX_EXTRA_PAIRS)):
        counts[rng.choice(GROUP_ORDER)] += 1

    rows = _sample_spaced(list(range(h)), counts["horizontal"], gap=2, rng=rng)
    cols = _sample_spaced(list(range(w)), counts["vertical"], gap=2, rng=rng)
    if rows is None or cols is None:
        return None

    row_set = set(rows)
    col_set = set(cols)
    main_candidates = _diagonal_candidates(shape, "main_diag", min_len=4, rows=row_set, cols=col_set)
    main_anchors = _sample_spaced(main_candidates, counts["main_diag"], gap=2, rng=rng)
    if main_anchors is None:
        return None

    anti_candidates = [
        anchor
        for anchor in _diagonal_candidates(shape, "anti_diag", min_len=4, rows=row_set, cols=col_set)
        if all(not _diag_intersects(shape, main_anchor, anchor) for main_anchor in main_anchors)
    ]
    anti_anchors = _sample_spaced(anti_candidates, counts["anti_diag"], gap=2, rng=rng)
    if anti_anchors is None:
        return None

    descriptors: list[tuple[str, int]] = []
    descriptors.extend(("horizontal", row) for row in rows)
    descriptors.extend(("vertical", col) for col in cols)
    descriptors.extend(("main_diag", anchor) for anchor in main_anchors)
    descriptors.extend(("anti_diag", anchor) for anchor in anti_anchors)
    rng.shuffle(descriptors)
    return descriptors


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    shape = (rng.randint(MIN_SIZE, MAX_SIZE), rng.randint(MIN_SIZE, MAX_SIZE))
    descriptors = _build_descriptors(shape, rng)
    if descriptors is None:
        return None

    pair_count = len(descriptors)
    bg, *colors = rng.sample(PALETTE, pair_count + 1)
    grid = np.full(shape, bg, dtype=int)
    reserved = np.zeros(shape, dtype=bool)
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)

    placement_order = sorted(descriptors, key=lambda item: len(_line_cells(shape, item[0], item[1])))
    for color, (direction, anchor) in zip(colors, placement_order):
        pair = _pick_pair(_line_cells(shape, direction, anchor), reserved, rng)
        if pair is None:
            return None
        for r, c in pair:
            grid[r, c] = color
            _mark_reserved(reserved, r, c)
        groups[direction].append((anchor, color))

    expected = _render(shape, bg, groups)
    if len(np.unique(grid)) != pair_count + 1:
        return None
    if np.count_nonzero(grid == bg) <= 2 * pair_count:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_TRIES} attempts")
