from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations, permutations
import random

import numpy as np


BG = 0
GRAY = 5
ORANGE = 7
LINE_COLORS = (1, 2, 3, 4, 6, 8, 9)

SIZE_RANGE = (13, 18)
LINE_COUNT_CHOICES = (3, 4)
MIN_SEG_LEN = 5
MAX_GEN_ATTEMPTS = 4000

KIND_FAMILIES = {
    3: [
        ("H", "V", "H"),
        ("V", "H", "V"),
        ("H", "D", "H"),
        ("V", "U", "V"),
        ("D", "H", "D"),
        ("U", "V", "U"),
    ],
    4: [
        ("H", "V", "H", "V"),
        ("V", "H", "V", "H"),
        ("H", "D", "H", "D"),
        ("V", "U", "V", "U"),
        ("D", "H", "D", "H"),
        ("U", "V", "U", "V"),
    ],
}


def _coord_key(rc: tuple[int, int]) -> tuple[int, int]:
    return int(rc[0]), int(rc[1])


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    dr = r1 - r0
    dc = c1 - c0
    assert (dr == 0) or (dc == 0) or (abs(dr) == abs(dc)), "markers must be queen-aligned"
    step_r = 0 if dr == 0 else dr // abs(dr)
    step_c = 0 if dc == 0 else dc // abs(dc)
    steps = max(abs(dr), abs(dc))
    assert steps > 0, "markers must be distinct"
    return tuple((r0 + i * step_r, c0 + i * step_c) for i in range(steps + 1))


def _extract_lines(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    counts = Counter(map(int, g.reshape(-1)))
    colors = [color for color, n in counts.items() if color not in {BG, GRAY} and n > 0]
    assert 3 <= len(colors) <= 4, "expected 3-4 line colors"

    records = []
    for color in sorted(colors):
        assert counts[color] == 2, "each line color must appear exactly twice as markers"
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        coords.sort(key=_coord_key)
        a, b = coords
        path = _segment_cells(a, b)
        records.append(
            {
                "color": int(color),
                "markers": (a, b),
                "first_key": min(a, b),
                "draw_key": max(a, b),
                "path": path,
                "path_set": frozenset(path),
            }
        )
    assert len(records) == len(colors)
    return records


def _simulate_visibility(grid: np.ndarray, records: list[dict]) -> tuple[np.ndarray, set[tuple[int, int]], dict[int, set[int]]]:
    g = np.asarray(grid, dtype=int)
    cell_to_covering: dict[tuple[int, int], list[int]] = defaultdict(list)
    ordered = sorted(records, key=lambda rec: _coord_key(rec["draw_key"]))
    for rec in ordered:
        for cell in rec["path"]:
            cell_to_covering[cell].append(int(rec["color"]))

    out = g.copy()
    below_edges: set[tuple[int, int]] = set()
    intersects: dict[int, set[int]] = {int(rec["color"]): set() for rec in records}

    for cell, covering in cell_to_covering.items():
        r, c = cell
        if len(covering) == 1:
            out[r, c] = ORANGE if int(g[r, c]) == GRAY else int(covering[0])
            continue

        top = int(covering[-1])
        out[r, c] = top
        for a, b in combinations(covering, 2):
            intersects[int(a)].add(int(b))
            intersects[int(b)].add(int(a))
        for lower in covering[:-1]:
            below_edges.add((int(lower), top))

    return out, below_edges, intersects


def _unique_chain(records: list[dict], below_edges: set[tuple[int, int]], intersects: dict[int, set[int]]) -> tuple[int, ...]:
    colors = [int(rec["color"]) for rec in records]

    def valid(order: tuple[int, ...]) -> bool:
        pos = {color: i for i, color in enumerate(order)}
        if any(pos[lower] >= pos[upper] for lower, upper in below_edges):
            return False
        return all(order[i + 1] in intersects[order[i]] for i in range(len(order) - 1))

    orders = [order for order in permutations(colors) if valid(order)]
    assert len(orders) == 1, "expected a unique bottom-to-top chain"
    return tuple(map(int, orders[0]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored markers, simulate the scanned line overlays, and output the unique bottom-to-top line-color chain.

    Input structure:
    - The grid is mostly black background with 3-4 colors that each appear exactly twice as marker pixels.
    - Each same-colored marker pair is horizontally, vertically, or 45-degree diagonally aligned.
    - Extra gray pixels are scattered as original foreground cells that would turn orange when a single line crosses them.

    Rule:
    - Recover one straight inclusive segment for each same-colored marker pair.
    - Raster-scan the input from top-left to bottom-right; when the second marker of a color is reached, regard that line as drawn, recoloring its path from gray to orange and from black/marker cells to the line color.
    - At cells used by multiple lines, the later-drawn line stays visible and is above every earlier line through that crossing.
    - Return the only bottom-to-top ordering of line colors that satisfies all above/below crossing constraints and whose consecutive colors correspond to geometrically crossing lines.

    Key insight:
    - The marker pairs define hidden line objects, and the visible color at each simulated crossing encodes which line is above which; the answer is the unique crossing-respecting layer chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    records = _extract_lines(g)
    _visible, below_edges, intersects = _simulate_visibility(g, records)
    order = _unique_chain(records, below_edges, intersects)
    return np.array([order], dtype=int)


def _random_segment_of_kind(size: int, kind: str) -> tuple[tuple[int, int], tuple[int, int]]:
    length = random.randint(MIN_SEG_LEN, size - 2)
    if kind == "H":
        r = random.randint(1, size - 2)
        c0 = random.randint(0, size - length)
        return (r, c0), (r, c0 + length - 1)
    if kind == "V":
        c = random.randint(1, size - 2)
        r0 = random.randint(0, size - length)
        return (r0, c), (r0 + length - 1, c)
    if kind == "D":
        r0 = random.randint(0, size - length)
        c0 = random.randint(0, size - length)
        return (r0, c0), (r0 + length - 1, c0 + length - 1)
    if kind == "U":
        r0 = random.randint(length - 1, size - 1)
        c0 = random.randint(0, size - length)
        return (r0, c0), (r0 - length + 1, c0 + length - 1)
    raise ValueError(f"unknown segment kind: {kind}")


def _records_from_segments(colors: list[int], segments: list[tuple[tuple[int, int], tuple[int, int]]]) -> list[dict]:
    records = []
    for color, (a, b) in zip(colors, segments):
        path = _segment_cells(a, b)
        records.append(
            {
                "color": int(color),
                "markers": (a, b),
                "first_key": min(a, b),
                "draw_key": max(a, b),
                "path": path,
                "path_set": frozenset(path),
            }
        )
    return records


def _path_graph_ok(records: list[dict]) -> bool:
    endpoint_sets = [set(rec["markers"]) for rec in records]
    degrees = [0] * len(records)
    cell_use = Counter()

    for rec in records:
        for cell in rec["path"]:
            cell_use[cell] += 1
    if any(n > 2 for n in cell_use.values()):
        return False

    edge_count = 0
    for i, j in combinations(range(len(records)), 2):
        inter = records[i]["path_set"] & records[j]["path_set"]
        if not inter:
            continue
        if len(inter) != 1:
            return False
        cell = next(iter(inter))
        if cell in endpoint_sets[i] or cell in endpoint_sets[j]:
            return False
        if cell in endpoint_sets[i] | endpoint_sets[j]:
            return False
        edge_count += 1
        degrees[i] += 1
        degrees[j] += 1

    k = len(records)
    if edge_count != k - 1:
        return False
    if sorted(degrees) != ([1, 1, 2] if k == 3 else [1, 1, 2, 2]):
        return False

    marker_cells = set().union(*(set(rec["markers"]) for rec in records))
    if len(marker_cells) != 2 * k:
        return False

    # Reject lines whose path runs through some other marker even without crossing there.
    for i, rec in enumerate(records):
        for j, other in enumerate(records):
            if i == j:
                continue
            if rec["path_set"] & set(other["markers"]):
                return False
    return True


def _choose_gray_cells(size: int, records: list[dict]) -> set[tuple[int, int]] | None:
    crossings = set()
    for i, j in combinations(range(len(records)), 2):
        crossings |= records[i]["path_set"] & records[j]["path_set"]

    marker_cells = set().union(*(set(rec["markers"]) for rec in records))
    path_union = set().union(*(set(rec["path"]) for rec in records))
    chosen: set[tuple[int, int]] = set()

    for rec in records:
        core = [cell for cell in rec["path"] if cell not in marker_cells and cell not in crossings]
        if len(core) < 2:
            return None
        chosen.add(random.choice(core))

    off_path = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if (r, c) not in path_union and (r, c) not in marker_cells
    ]
    if len(off_path) < 2:
        return None
    extra_count = random.randint(2, min(6, len(off_path)))
    chosen.update(random.sample(off_path, extra_count))
    return chosen


def _build_input_grid(size: int, records: list[dict], gray_cells: set[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    for r, c in gray_cells:
        grid[r, c] = GRAY
    for rec in records:
        for r, c in rec["markers"]:
            grid[r, c] = int(rec["color"])
    return grid


def _order_by(records: list[dict], key_name: str) -> tuple[int, ...]:
    return tuple(int(rec["color"]) for rec in sorted(records, key=lambda rec: _coord_key(rec[key_name])))


def _semantic_checks(grid: np.ndarray, records: list[dict]) -> bool:
    try:
        visible, below_edges, intersects = _simulate_visibility(grid, records)
        order = _unique_chain(records, below_edges, intersects)
        actual = tuple(map(int, main(grid)[0]))
    except Exception:
        return False

    if order != actual:
        return False
    if order == tuple(sorted(order)):
        return False
    if order == _order_by(records, "first_key"):
        return False

    crossings = set()
    for i, j in combinations(range(len(records)), 2):
        inter = records[i]["path_set"] & records[j]["path_set"]
        crossings |= inter
        if inter:
            top = max((rec for rec in records if next(iter(inter)) in rec["path_set"]), key=lambda rec: _coord_key(rec["draw_key"]))
            if int(visible[next(iter(inter))]) != int(top["color"]):
                return False

    marker_cells = set().union(*(set(rec["markers"]) for rec in records))
    for rec in records:
        solo = [cell for cell in rec["path"] if cell not in crossings and cell not in marker_cells]
        if not any(int(grid[cell]) == GRAY for cell in solo):
            return False
        if not any(int(grid[cell]) == BG for cell in solo):
            return False
    return True


def _gen_attempt(k: int) -> np.ndarray | None:
    size = random.randint(*SIZE_RANGE)
    kinds = random.choice(KIND_FAMILIES[k])
    colors = random.sample(LINE_COLORS, k)
    segments = [_random_segment_of_kind(size, kind) for kind in kinds]
    records = _records_from_segments(colors, segments)

    if not _path_graph_ok(records):
        return None

    gray_cells = _choose_gray_cells(size, records)
    if gray_cells is None:
        return None

    grid = _build_input_grid(size, records, gray_cells)
    if not _semantic_checks(grid, records):
        return None
    return grid


def generate_input() -> np.ndarray:
    target_k = random.choice(LINE_COUNT_CHOICES)
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(target_k)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid {target_k}-line input for puzzle 11")
