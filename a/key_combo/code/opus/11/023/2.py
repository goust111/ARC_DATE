import random
from collections import defaultdict

import numpy as np


BG = 0
MARKER = 7
SYMBOL_COLORS = (1, 2, 3)
BORDER_DELTAS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
]
COUNT_PATTERNS = {
    8: [()],
    7: [(i,) for i in range(8)],
    6: [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 0),
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 0),
        (1, 3),
        (3, 5),
        (5, 7),
        (7, 1),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ],
}
GEN_ATTEMPTS = 400


def _find_symbols(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            patch = g[r0 : r0 + 3, c0 : c0 + 3]
            if int(patch[1, 1]) != BG:
                continue
            border_values = [int(patch[r, c]) for r, c in [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]]
            colors = {value for value in border_values if value != BG}
            if len(colors) != 1:
                continue
            color = next(iter(colors))
            if color not in SYMBOL_COLORS:
                continue
            count = sum(value == color for value in border_values)
            if count < 6:
                continue
            coords = []
            for idx, (dr, dc) in enumerate(BORDER_DELTAS):
                if int(patch[dr + 1, dc + 1]) == color:
                    coords.append((r0 + dr + 1, c0 + dc + 1))
            candidates.append(
                {
                    "center": (r0 + 1, c0 + 1),
                    "row": r0 + 1,
                    "col": c0 + 1,
                    "color": color,
                    "count": count,
                    "coords": tuple(coords),
                }
            )

    foreground = {
        (int(r), int(c))
        for r, c in np.argwhere((g != BG) & (g != MARKER))
    }
    claimed = defaultdict(int)
    for symbol in candidates:
        for cell in symbol["coords"]:
            claimed[cell] += 1

    assert foreground, "expected hollow symbols below the marker line"
    assert set(claimed) == foreground, "foreground must decompose into 3x3 hollow symbols"
    assert all(v == 1 for v in claimed.values()), "symbol windows must not overlap"
    centers = [symbol["center"] for symbol in candidates]
    assert len(centers) == len(set(centers)), "duplicate symbol centers detected"
    return candidates


def _stack_symbols(shape: tuple[int, int], symbols: list[dict], *, mode: str) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[0, :] = MARKER
    by_col: dict[int, list[dict]] = defaultdict(list)
    for symbol in symbols:
        by_col[int(symbol["col"])].append(symbol)

    for col, items in by_col.items():
        if mode == "rule":
            ordered = sorted(items, key=lambda item: (-int(item["count"]), int(item["row"])))
        elif mode == "row":
            ordered = sorted(items, key=lambda item: int(item["row"]))
        elif mode == "reverse_tie":
            ordered = sorted(items, key=lambda item: (-int(item["count"]), -int(item["row"])))
        else:
            raise ValueError(f"unknown stacking mode: {mode}")

        for depth, symbol in enumerate(ordered, start=1):
            out[depth, int(col)] = int(symbol["color"])
    return out


def _compressed_only(shape: tuple[int, int], symbols: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[0, :] = MARKER
    for symbol in symbols:
        out[int(symbol["row"]), int(symbol["col"])] = int(symbol["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress hollow 3x3 symbols to center pixels and stack each column upward by original frame size.

    Input structure:
    - The grid has a black background, a full orange marker line on the top edge, and 5 to 7 colored hollow 3x3 symbols below it.
    - Each symbol uses one of three foreground colors, has a black center, and keeps 6 to 8 border pixels so different symbols can have different original areas.
    - Symbols are separated from one another, and multiple symbols may share the same center column after compression.

    Rule:
    - Replace every hollow 3x3 symbol by a single pixel at its center, keeping the symbol color.
    - The top-edge marker means gravity points upward, so each compressed pixel moves within its own column toward row 1.
    - When several compressed pixels share a column, sort them by the original symbol's border-pixel count from large to small; equal counts keep the smaller original row first.
    - Keep the top marker line unchanged and clear all other non-stacked cells to background.

    Key insight:
    - The visible holes in the hollow symbols are only an intermediate encoding: after compression, the missing-frame count determines collision priority in the upward stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all(g[0, :] == MARKER), "expected a full top-edge marker line"
    symbols = _find_symbols(g)
    return _stack_symbols(g.shape, symbols, mode="rule")


def _sample_column_counts(total_symbols: int, active_cols: int) -> list[int]:
    if active_cols == 3:
        counts = [2, 2, 1]
    else:
        counts = [2, 2, 1, 1]
    extra = total_symbols - sum(counts)
    while extra > 0:
        choices = [idx for idx, value in enumerate(counts) if value < 3]
        pick = random.choice(choices)
        counts[pick] += 1
        extra -= 1
    return counts


def _priority_sizes(rows: list[int]) -> list[int]:
    while True:
        sizes = [random.choice((6, 7, 8)) for _ in rows]
        order = sorted(range(len(rows)), key=lambda idx: (-sizes[idx], rows[idx]))
        if order != list(range(len(rows))):
            return sizes


def _tie_sizes(rows: list[int]) -> list[int]:
    shared = random.choice((6, 7, 8))
    if len(rows) == 2:
        return [shared, shared]
    other = random.choice([value for value in (6, 7, 8) if value != shared])
    sizes = [shared, shared, other]
    random.shuffle(sizes)
    return sizes


def _assign_colors(specs: list[dict], tie_column: int) -> bool:
    for _ in range(80):
        for spec in specs:
            spec["color"] = random.choice(SYMBOL_COLORS)

        tie_specs = [spec for spec in specs if spec["col"] == tie_column]
        by_size: dict[int, list[dict]] = defaultdict(list)
        for spec in tie_specs:
            by_size[int(spec["count"])].append(spec)
        tied_group = max(by_size.values(), key=len)
        if len(tied_group) >= 2:
            chosen = random.sample(SYMBOL_COLORS, 2)
            tied_group[0]["color"] = chosen[0]
            tied_group[1]["color"] = chosen[1]

        sizes_by_color: dict[int, set[int]] = defaultdict(set)
        colors_by_size: dict[int, set[int]] = defaultdict(set)
        for spec in specs:
            sizes_by_color[int(spec["color"])].add(int(spec["count"]))
            colors_by_size[int(spec["count"])].add(int(spec["color"]))
        if not any(len(values) >= 2 for values in sizes_by_color.values()):
            continue
        if not any(len(values) >= 2 for values in colors_by_size.values()):
            continue
        return True
    return False


def _paint_symbol(grid: np.ndarray, center: tuple[int, int], color: int, count: int) -> None:
    r, c = center
    missing = set(random.choice(COUNT_PATTERNS[count]))
    for idx, (dr, dc) in enumerate(BORDER_DELTAS):
        if idx not in missing:
            grid[r + dr, c + dc] = color


def _build_candidate() -> np.ndarray:
    height = random.randint(13, 19)
    width = random.randint(15, 19)
    row_slots = list(range(3, height - 1, 4))
    col_slots = list(range(3, width - 1, 4))
    if len(row_slots) < 3 or len(col_slots) < 3:
        raise AssertionError("not enough lattice slots")

    total_symbols = random.randint(5, 7)
    active_cols = 4 if len(col_slots) >= 4 and total_symbols >= 6 and random.random() < 0.5 else 3
    chosen_cols = random.sample(col_slots, active_cols)
    col_counts = _sample_column_counts(total_symbols, active_cols)

    priority_col = chosen_cols[0]
    tie_col = chosen_cols[1]
    specs = []
    for col, n_in_col in zip(chosen_cols, col_counts):
        rows = sorted(random.sample(row_slots, n_in_col))
        if col == priority_col:
            sizes = _priority_sizes(rows)
        elif col == tie_col:
            sizes = _tie_sizes(rows)
        else:
            sizes = [random.choice((6, 7, 8)) for _ in rows]
        for row, size in zip(rows, sizes):
            specs.append({"row": row, "col": col, "count": size})

    all_counts = {int(spec["count"]) for spec in specs}
    if all_counts != {6, 7, 8}:
        return None
    if not _assign_colors(specs, tie_col):
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[0, :] = MARKER
    for spec in specs:
        _paint_symbol(grid, (int(spec["row"]), int(spec["col"])), int(spec["color"]), int(spec["count"]))
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    symbols = _find_symbols(grid)
    assert 5 <= len(symbols) <= 7
    assert np.all(grid[0, :] == MARKER)

    by_col: dict[int, list[dict]] = defaultdict(list)
    for symbol in symbols:
        by_col[int(symbol["col"])].append(symbol)
    assert max(len(items) for items in by_col.values()) <= 3
    assert any(len(items) >= 2 for items in by_col.values())
    assert any(
        len(items) >= 2 and any(items[i]["count"] == items[j]["count"] for i in range(len(items)) for j in range(i + 1, len(items)))
        for items in by_col.values()
    ), "need a visible tie case"

    out = main(grid)
    row_only = _stack_symbols(grid.shape, symbols, mode="row")
    reverse_tie = _stack_symbols(grid.shape, symbols, mode="reverse_tie")
    compressed = _compressed_only(grid.shape, symbols)

    assert not np.array_equal(out, compressed), "gravity must change the compressed layout"
    assert not np.array_equal(out, row_only), "pixel counts must affect stacking order"
    assert not np.array_equal(out, reverse_tie), "row tie-break direction must be observable"

    all_counts = {int(symbol["count"]) for symbol in symbols}
    assert all_counts == {6, 7, 8}

    sizes_by_color: dict[int, set[int]] = defaultdict(set)
    colors_by_size: dict[int, set[int]] = defaultdict(set)
    for symbol in symbols:
        sizes_by_color[int(symbol["color"])].add(int(symbol["count"]))
        colors_by_size[int(symbol["count"])].add(int(symbol["color"]))
    assert any(len(values) >= 2 for values in sizes_by_color.values())
    assert any(len(values) >= 2 for values in colors_by_size.values())

    assert np.all(out[0, :] == MARKER)
    assert np.count_nonzero(out[1:, :]) == len(symbols)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is None:
            continue
        try:
            _validate_generated(grid)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a non-degenerate input")
