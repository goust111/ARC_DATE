from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
LEFT_COLOR = 1
RIGHT_COLOR = 2
BAR_COLOR = 5

GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (21, 30)
BAR_COUNT_CHOICES = (3, 4, 4, 5, 5)
BAR_LENGTHS = (3, 5, 7)
MAX_GENERATION_ATTEMPTS = 400


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of (r, c) coordinates."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _extract_bars(grid: np.ndarray) -> list[dict]:
    bars: list[dict] = []
    for coords in _components_mask(grid == BAR_COLOR):
        rows = sorted({int(r) for r, _c in coords})
        cols = sorted({int(c) for _r, c in coords})
        area = int(coords.shape[0])
        if len(rows) == 1:
            assert len(cols) == area and cols == list(range(cols[0], cols[-1] + 1))
            bars.append(
                {
                    "orientation": "H",
                    "length": area,
                    "row": rows[0],
                    "c0": cols[0],
                    "c1": cols[-1],
                    "center_c": (cols[0] + cols[-1]) // 2,
                }
            )
        else:
            assert len(cols) == 1 and len(rows) == area and rows == list(range(rows[0], rows[-1] + 1))
            bars.append(
                {
                    "orientation": "V",
                    "length": area,
                    "col": cols[0],
                    "r0": rows[0],
                    "r1": rows[-1],
                    "center_c": cols[0],
                }
            )
    return bars


def _envelope_cells_from_bar(bar: dict) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if bar["orientation"] == "H":
        row = int(bar["row"])
        c0 = int(bar["c0"])
        c1 = int(bar["c1"])
        radius = (int(bar["length"]) - 1) // 2
        for offset in range(1, radius + 1):
            left = c0 + offset
            right = c1 - offset
            if left > right:
                break
            for rr in (row - offset, row + offset):
                for cc in range(left, right + 1):
                    cells.append((rr, cc))
    else:
        col = int(bar["col"])
        r0 = int(bar["r0"])
        r1 = int(bar["r1"])
        radius = (int(bar["length"]) - 1) // 2
        for offset in range(1, radius + 1):
            top = r0 + offset
            bottom = r1 - offset
            if top > bottom:
                break
            for cc in (col - offset, col + offset):
                for rr in range(top, bottom + 1):
                    cells.append((rr, cc))
    return cells


def _bar_color_from_position(width: int, center_c: int) -> int:
    d_left = int(center_c)
    d_right = int(width - 1 - center_c)
    assert d_left != d_right
    return LEFT_COLOR if d_left < d_right else RIGHT_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint a blue or red diamond envelope around each gray bar based on the nearer side wall.

    Input structure:
    - The background is black, with a full blue line in the leftmost column and a full red line in the rightmost column.
    - The interior contains several disconnected gray one-pixel-thick bars, each horizontal or vertical.
    - Every gray bar is isolated from the side walls and from the other bars.

    Rule:
    - Keep the blue left wall, the red right wall, and every gray bar unchanged.
    - Around each horizontal or vertical gray bar, generate the centered shrinking diamond envelope obtained by shortening the span by one cell at each end on every step away from the bar.
    - Color that envelope from the bar's horizontal center position: use blue when the center is closer to the left wall, otherwise use red when it is closer to the right wall.
    - Process bars by decreasing length, then by increasing center column, and paint only onto black cells so later envelopes never overwrite earlier colored pixels.

    Key insight:
    - The bar itself determines the envelope geometry, but the bar's global left-right position determines the envelope color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert np.all(g[:, 0] == LEFT_COLOR)
    assert np.all(g[:, w - 1] == RIGHT_COLOR)

    out = g.copy()
    bars = _extract_bars(g)
    for bar in sorted(bars, key=lambda item: (-int(item["length"]), int(item["center_c"]))):
        color = _bar_color_from_position(w, int(bar["center_c"]))
        for r, c in _envelope_cells_from_bar(bar):
            if 0 <= r < h and 0 <= c < w and out[r, c] == BG:
                out[r, c] = color
    return out


def _full_shape_cells(spec: dict) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if spec["orientation"] == "H":
        row = int(spec["row"])
        c0 = int(spec["c0"])
        length = int(spec["length"])
        for cc in range(c0, c0 + length):
            cells.add((row, cc))
        cells.update(_envelope_cells_from_bar(spec))
    else:
        r0 = int(spec["r0"])
        col = int(spec["col"])
        length = int(spec["length"])
        for rr in range(r0, r0 + length):
            cells.add((rr, col))
        cells.update(_envelope_cells_from_bar(spec))
    return cells


def _dilate8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _candidate_specs(height: int, width: int, orientation: str, side: str, length: int) -> list[dict]:
    specs: list[dict] = []
    radius = (length - 1) // 2
    midpoint = (width - 1) / 2.0
    if orientation == "H":
        for row in range(radius, height - radius):
            for c0 in range(3, width - 3 - length + 1):
                center_c = c0 + length // 2
                if side == "L" and not (center_c < midpoint):
                    continue
                if side == "R" and not (center_c > midpoint):
                    continue
                specs.append(
                    {
                        "orientation": "H",
                        "length": length,
                        "row": row,
                        "c0": c0,
                        "c1": c0 + length - 1,
                        "center_c": center_c,
                    }
                )
    else:
        col_min = max(4, radius + 1)
        col_max = min(width - 5, width - radius - 2)
        for r0 in range(1, height - length):
            for col in range(col_min, col_max + 1):
                center_c = col
                if side == "L" and not (center_c < midpoint):
                    continue
                if side == "R" and not (center_c > midpoint):
                    continue
                specs.append(
                    {
                        "orientation": "V",
                        "length": length,
                        "col": col,
                        "r0": r0,
                        "r1": r0 + length - 1,
                        "center_c": center_c,
                    }
                )
    random.shuffle(specs)
    return specs


def _combo_plan(count: int) -> list[tuple[str, str]]:
    base = [("H", "L"), ("H", "R"), ("V", "L"), ("V", "R")]
    if count == 3:
        keep = random.choice(
            [
                [("H", "L"), ("H", "R"), ("V", "L")],
                [("H", "L"), ("V", "L"), ("V", "R")],
                [("H", "R"), ("V", "L"), ("V", "R")],
                [("H", "L"), ("H", "R"), ("V", "R")],
            ]
        )
        random.shuffle(keep)
        return keep
    combos = base[:]
    if count == 5:
        combos.append(random.choice(base))
    random.shuffle(combos)
    return combos


def _sample_lengths(count: int) -> list[int]:
    lengths = [random.choice(BAR_LENGTHS) for _ in range(count)]
    if max(lengths) == 3:
        lengths[random.randrange(count)] = random.choice((5, 7))
    if len(set(lengths)) == 1:
        lengths[random.randrange(count)] = random.choice([x for x in BAR_LENGTHS if x != lengths[0]])
    return lengths


def _build_input(height: int, width: int, specs: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[:, 0] = LEFT_COLOR
    grid[:, width - 1] = RIGHT_COLOR
    for spec in specs:
        if spec["orientation"] == "H":
            grid[int(spec["row"]), int(spec["c0"]) : int(spec["c1"]) + 1] = BAR_COLOR
        else:
            grid[int(spec["r0"]) : int(spec["r1"]) + 1, int(spec["col"])] = BAR_COLOR
    return grid


def _oracle_output(height: int, width: int, specs: list[dict]) -> np.ndarray:
    out = _build_input(height, width, specs)
    for spec in sorted(specs, key=lambda item: (-int(item["length"]), int(item["center_c"]))):
        color = _bar_color_from_position(width, int(spec["center_c"]))
        for r, c in _envelope_cells_from_bar(spec):
            if out[r, c] == BG:
                out[r, c] = color
    return out


def _semantic_checks(inp: np.ndarray, out: np.ndarray, specs: list[dict]) -> None:
    h, w = inp.shape
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    assert 3 <= len(specs) <= 5
    assert np.all(inp[:, 0] == LEFT_COLOR) and np.all(inp[:, w - 1] == RIGHT_COLOR)
    assert not np.any(inp[:, 1 : w - 1] == LEFT_COLOR)
    assert not np.any(inp[:, 1 : w - 1] == RIGHT_COLOR)

    seen_orientations = {spec["orientation"] for spec in specs}
    seen_sides = {_bar_color_from_position(w, int(spec["center_c"])) for spec in specs}
    assert seen_orientations == {"H", "V"}
    assert seen_sides == {LEFT_COLOR, RIGHT_COLOR}

    union_cells: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()
    for spec in specs:
        center_c = int(spec["center_c"])
        assert center_c != (w - 1) / 2.0
        shape = _full_shape_cells(spec)
        envelope = set(_envelope_cells_from_bar(spec))
        assert envelope
        assert union_cells.isdisjoint(envelope)
        assert all(cell not in forbidden for cell in shape)
        union_cells.update(shape)
        forbidden.update(_dilate8(shape, (h, w)))
        if spec["orientation"] == "H":
            assert int(spec["length"]) == int(spec["c1"]) - int(spec["c0"]) + 1
        else:
            assert int(spec["length"]) == int(spec["r1"]) - int(spec["r0"]) + 1

    bars = _extract_bars(inp)
    decoded = sorted((b["orientation"], int(b["length"]), int(b["center_c"])) for b in bars)
    original = sorted((s["orientation"], int(s["length"]), int(s["center_c"])) for s in specs)
    assert decoded == original

    recomputed = main(inp)
    assert np.array_equal(recomputed, out)
    assert np.any(out == LEFT_COLOR) and np.any(out == RIGHT_COLOR)
    assert np.count_nonzero(out == BAR_COLOR) == sum(int(spec["length"]) for spec in specs)


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    count = random.choice(BAR_COUNT_CHOICES)

    combos = _combo_plan(count)
    lengths = _sample_lengths(count)
    order = list(range(count))
    order.sort(key=lambda idx: -lengths[idx])

    specs: list[dict] = []
    forbidden: set[tuple[int, int]] = set()
    for idx in order:
        orientation, side = combos[idx]
        length = lengths[idx]
        placed = False
        for candidate in _candidate_specs(height, width, orientation, side, length):
            shape = _full_shape_cells(candidate)
            if any(cell in forbidden for cell in shape):
                continue
            specs.append(candidate)
            forbidden.update(_dilate8(shape, (height, width)))
            placed = True
            break
        if not placed:
            return None

    random.shuffle(specs)
    inp = _build_input(height, width, specs)
    out = _oracle_output(height, width, specs)

    try:
        _semantic_checks(inp, out, specs)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
