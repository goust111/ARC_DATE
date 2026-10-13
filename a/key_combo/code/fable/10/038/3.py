from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GRID_MIN = 9
GRID_MAX = 15
MIN_GAP = 2
COUNT_CHOICES = (
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 3),
    (4, 4),
)
MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values = Counter(map(int, np.ravel(grid)))
    return int(values.most_common(1)[0][0])


def _extract_hidden_lines(grid: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    boundary_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            color = int(g[r, c])
            if color != bg:
                boundary_by_color[color].append((r, c))

    row_lines: dict[int, int] = {}
    col_lines: dict[int, int] = {}
    for color, cells in boundary_by_color.items():
        assert len(cells) == 2, f"color {color} must have exactly 2 boundary endpoints"
        (r0, c0), (r1, c1) = cells
        if r0 == r1 and {c0, c1} == {0, w - 1}:
            assert r0 not in row_lines, "duplicate horizontal line"
            row_lines[int(r0)] = int(color)
        elif c0 == c1 and {r0, r1} == {0, h - 1}:
            assert c0 not in col_lines, "duplicate vertical line"
            col_lines[int(c0)] = int(color)
        else:
            raise AssertionError(f"color {color} endpoints do not define a full row/col line: {cells}")

    assert row_lines and col_lines, "need at least one hidden row and one hidden column"
    return row_lines, col_lines


def _build_order_from_samples(
    grid: np.ndarray, row_lines: dict[int, int], col_lines: dict[int, int]
) -> tuple[list[int], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    nodes = list(row_lines.values()) + list(col_lines.values())
    graph = {color: set() for color in nodes}
    indeg = {color: 0 for color in nodes}
    samples: dict[tuple[int, int], int] = {}

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(g[r, c])
            if color == bg:
                continue
            assert r in row_lines and c in col_lines, "interior mark must sit on a row/col intersection"
            row_color = int(row_lines[r])
            col_color = int(col_lines[c])
            assert color in (row_color, col_color), "sample color must match one of the two crossing lines"
            samples[(r, c)] = color
            lower, upper = (col_color, row_color) if color == row_color else (row_color, col_color)
            if upper not in graph[lower]:
                graph[lower].add(upper)
                indeg[upper] += 1

    assert samples, "need at least one interior crossing sample"

    order: list[int] = []
    remaining = set(nodes)
    while remaining:
        ready = sorted(color for color in remaining if indeg[color] == 0)
        assert len(ready) == 1, f"samples do not determine a unique total order: {ready}"
        color = int(ready[0])
        order.append(color)
        remaining.remove(color)
        for nxt in graph[color]:
            indeg[nxt] -= 1

    assert len(order) == len(nodes)
    return order, samples


def _render_output(
    shape: tuple[int, int], row_lines: dict[int, int], col_lines: dict[int, int], order: list[int]
) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), BG, dtype=int)
    line_by_color = {color: ("row", row) for row, color in row_lines.items()}
    line_by_color.update({color: ("col", col) for col, color in col_lines.items()})

    for color in order:
        kind, idx = line_by_color[int(color)]
        if kind == "row":
            out[int(idx), :] = int(color)
        else:
            out[:, int(idx)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct full hidden row/column lines from boundary endpoints, then use sampled crossings to recover the unique stacking order.

    Input structure:
    - The background contains paired same-color boundary endpoints; each pair marks one hidden full row line or full column line.
    - The interior contains a few single-pixel crossing samples, each lying exactly at the intersection of one hidden row line and one hidden column line.
    - A sample pixel is colored as the line that sits on top at that crossing; the lines themselves are otherwise not drawn.

    Rule:
    - Recover every hidden line from its boundary endpoint pair and extend it across its entire row or column in its own color.
    - Interpret each interior sample as a comparison between the two crossing lines: the visible sample color is the higher line in the stack.
    - Take the unique total order consistent with all such comparisons, and at every row/column intersection keep only the color of the higher line.

    Key insight:
    - The visible interior pixels are not partial lines; they are pairwise "who covers whom" witnesses. The full output only emerges after chaining those local comparisons into one global layer order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    row_lines, col_lines = _extract_hidden_lines(g)
    order, _samples = _build_order_from_samples(g, row_lines, col_lines)
    return _render_output(g.shape, row_lines, col_lines, order)


def _spaced_positions(length: int, count: int, gap: int = MIN_GAP) -> list[int] | None:
    interior = length - 2
    free = interior - (count - 1) * (gap - 1)
    if free < count:
        return None
    base = sorted(random.sample(range(free), count))
    return [1 + b + i * (gap - 1) for i, b in enumerate(base)]


def _choose_shape(row_count: int, col_count: int) -> tuple[int, int]:
    min_h = max(GRID_MIN, 2 * row_count + 1)
    min_w = max(GRID_MIN, 2 * col_count + 1)
    return random.randint(min_h, GRID_MAX), random.randint(min_w, GRID_MAX)


def _alternating_order(rows: list[int], cols: list[int]) -> list[tuple[str, int]]:
    row_slots = rows[:]
    col_slots = cols[:]
    random.shuffle(row_slots)
    random.shuffle(col_slots)

    if len(row_slots) > len(col_slots):
        start = "row"
    elif len(col_slots) > len(row_slots):
        start = "col"
    else:
        start = random.choice(("row", "col"))

    order: list[tuple[str, int]] = []
    current = start
    while row_slots or col_slots:
        if current == "row":
            assert row_slots
            order.append(("row", row_slots.pop()))
            current = "col"
        else:
            assert col_slots
            order.append(("col", col_slots.pop()))
            current = "row"
    return order


def _is_non_geometric(order: list[tuple[str, int]]) -> bool:
    for kind in ("row", "col"):
        seq = [idx for orient, idx in order if orient == kind]
        if len(seq) >= 3:
            spatial = sorted(seq)
            if seq != spatial and seq != spatial[::-1]:
                return True
    return all(len([idx for orient, idx in order if orient == kind]) < 3 for kind in ("row", "col"))


def _build_scene(
    shape: tuple[int, int],
    row_lines: dict[int, int],
    col_lines: dict[int, int],
    order: list[int],
    samples: dict[tuple[int, int], int],
) -> np.ndarray:
    h, w = map(int, shape)
    inp = np.full((h, w), BG, dtype=int)
    for row, color in row_lines.items():
        inp[int(row), 0] = int(color)
        inp[int(row), w - 1] = int(color)
    for col, color in col_lines.items():
        inp[0, int(col)] = int(color)
        inp[h - 1, int(col)] = int(color)
    for (r, c), color in samples.items():
        inp[int(r), int(c)] = int(color)
    return inp


def _validate_generated_case(
    inp: np.ndarray,
    row_lines: dict[int, int],
    col_lines: dict[int, int],
    intended_order: list[int],
    samples: dict[tuple[int, int], int],
) -> bool:
    try:
        parsed_rows, parsed_cols = _extract_hidden_lines(inp)
        if parsed_rows != row_lines or parsed_cols != col_lines:
            return False

        recovered_order, recovered_samples = _build_order_from_samples(inp, parsed_rows, parsed_cols)
        if recovered_order != intended_order:
            return False
        if recovered_samples != samples:
            return False

        if len(samples) >= len(row_lines) * len(col_lines):
            return False

        has_row_on_top = any(color == row_lines[r] for (r, _c), color in samples.items())
        has_col_on_top = any(color == col_lines[c] for (_r, c), color in samples.items())
        if not (has_row_on_top and has_col_on_top):
            return False

        expected = _render_output(inp.shape, row_lines, col_lines, intended_order)
        actual = main(inp)
        if not np.array_equal(actual, expected):
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    row_count, col_count = random.choice(COUNT_CHOICES)
    if abs(row_count - col_count) > 1:
        return None

    h, w = _choose_shape(row_count, col_count)
    rows = _spaced_positions(h, row_count)
    cols = _spaced_positions(w, col_count)
    if rows is None or cols is None:
        return None

    orientation_order = _alternating_order(rows, cols)
    if not _is_non_geometric(orientation_order):
        return None

    line_colors = random.sample(PALETTE, row_count + col_count)
    row_lines: dict[int, int] = {}
    col_lines: dict[int, int] = {}
    line_by_slot: list[tuple[str, int, int]] = []
    for (kind, idx), color in zip(orientation_order, line_colors):
        if kind == "row":
            row_lines[int(idx)] = int(color)
        else:
            col_lines[int(idx)] = int(color)
        line_by_slot.append((kind, int(idx), int(color)))

    intended_order = [color for _kind, _idx, color in line_by_slot]
    rank = {color: i for i, color in enumerate(intended_order)}

    samples: dict[tuple[int, int], int] = {}
    for lower, upper in zip(line_by_slot, line_by_slot[1:]):
        if lower[0] == "row":
            r, c = int(lower[1]), int(upper[1])
        else:
            r, c = int(upper[1]), int(lower[1])
        samples[(r, c)] = int(upper[2])

    remaining_pairs = [
        (r, c)
        for r in row_lines
        for c in col_lines
        if (r, c) not in samples
    ]
    max_extra = min(3, max(0, len(remaining_pairs) - 1))
    extra_count = random.randint(0, max_extra)
    for r, c in random.sample(remaining_pairs, extra_count):
        row_color = int(row_lines[r])
        col_color = int(col_lines[c])
        top_color = row_color if rank[row_color] > rank[col_color] else col_color
        samples[(int(r), int(c))] = int(top_color)

    inp = _build_scene((h, w), row_lines, col_lines, intended_order, samples)
    if not _validate_generated_case(inp, row_lines, col_lines, intended_order, samples):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layered-line reconstruction input")
