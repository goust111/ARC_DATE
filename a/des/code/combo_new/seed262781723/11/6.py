import random

import numpy as np


GEN_BACKGROUND = 0
GEN_MOLD_COLOR = 5
GEN_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_GRID_MIN = 20
GEN_GRID_MAX = 28
GEN_MOLD_MIN = 8
GEN_MOLD_MAX = 14
GEN_GROOVE_MIN = 2
GEN_GROOVE_MAX = 5
GEN_GROOVE_COUNT_MIN = 2
GEN_GROOVE_COUNT_MAX = 5
GEN_BAR_EXTRA_MAX = 4
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _runs_of_target(values: np.ndarray, target: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(values.tolist()):
        if int(value) == int(target):
            if start is None:
                start = idx
            continue
        if start is not None:
            runs.append((int(start), int(idx - 1)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(values) - 1)))
    return runs


def _extract_grooves(
    grid: np.ndarray, bg: int, mold_color: int, bbox: tuple[int, int, int, int]
) -> list[dict]:
    r0, c0, r1, c1 = bbox
    grooves: list[dict] = []

    def add_horizontal(side: str, row: int, marker_row: int) -> None:
        edge = grid[row, c0 + 1 : c1]
        for start, end in _runs_of_target(edge, bg):
            abs_c0 = c0 + 1 + start
            abs_c1 = c0 + 1 + end
            marker = grid[marker_row, abs_c0 : abs_c1 + 1]
            colors = set(map(int, marker.tolist()))
            assert len(colors) == 1
            color = colors.pop()
            assert color not in {bg, mold_color}
            groove_coords = [(int(row), int(c)) for c in range(abs_c0, abs_c1 + 1)]
            marker_coords = [(int(marker_row), int(c)) for c in range(abs_c0, abs_c1 + 1)]
            grooves.append(
                {
                    "side": side,
                    "color": int(color),
                    "length": int(abs_c1 - abs_c0 + 1),
                    "groove_coords": groove_coords,
                    "marker_coords": marker_coords,
                }
            )

    def add_vertical(side: str, col: int, marker_col: int) -> None:
        edge = grid[r0 + 1 : r1, col]
        for start, end in _runs_of_target(edge, bg):
            abs_r0 = r0 + 1 + start
            abs_r1 = r0 + 1 + end
            marker = grid[abs_r0 : abs_r1 + 1, marker_col]
            colors = set(map(int, marker.tolist()))
            assert len(colors) == 1
            color = colors.pop()
            assert color not in {bg, mold_color}
            groove_coords = [(int(r), int(col)) for r in range(abs_r0, abs_r1 + 1)]
            marker_coords = [(int(r), int(marker_col)) for r in range(abs_r0, abs_r1 + 1)]
            grooves.append(
                {
                    "side": side,
                    "color": int(color),
                    "length": int(abs_r1 - abs_r0 + 1),
                    "groove_coords": groove_coords,
                    "marker_coords": marker_coords,
                }
            )

    assert r0 > 0 and c0 > 0 and r1 + 1 < grid.shape[0] and c1 + 1 < grid.shape[1]
    add_horizontal("top", r0, r0 - 1)
    add_horizontal("bottom", r1, r1 + 1)
    add_vertical("left", c0, c0 - 1)
    add_vertical("right", c1, c1 + 1)
    assert grooves
    assert len({int(item["color"]) for item in grooves}) == len(grooves)
    return grooves


def _find_bar(
    grid: np.ndarray, color: int, marker_coords: list[tuple[int, int]]
) -> dict:
    mask = grid == int(color)
    for r, c in marker_coords:
        mask[int(r), int(c)] = False
    coords = np.argwhere(mask)
    assert coords.size > 0
    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.all(rows == rows[0]):
        row = int(rows[0])
        c0 = int(cols.min())
        c1 = int(cols.max())
        assert int(coords.shape[0]) == c1 - c0 + 1
        return {"orientation": "h", "row": row, "start": c0, "length": int(coords.shape[0])}
    if np.all(cols == cols[0]):
        col = int(cols[0])
        r0 = int(rows.min())
        r1 = int(rows.max())
        assert int(coords.shape[0]) == r1 - r0 + 1
        return {"orientation": "v", "col": col, "start": r0, "length": int(coords.shape[0])}
    raise AssertionError("raw-material bar must be a single straight line")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each mold-edge groove by cutting a same-colored segment from its raw-material bar.

    Input structure:
    - The background is one dominant color, and one much larger structural color forms a solid rectangle whose border may contain black groove runs.
    - Every groove is a consecutive missing segment on one rectangle edge, with a same-length colored marker immediately outside that edge.
    - Elsewhere, each marker color appears again as one isolated horizontal or vertical monochrome bar that is at least as long as its groove.

    Rule:
    - Find the large rectangle, read every border groove, and use the adjacent outside marker to recover that groove's color and length.
    - Match each marker color to the unique raw-material bar of the same color elsewhere in the grid.
    - Clear the marker, fill the groove with that color, and delete an equally long prefix from the bar: horizontal bars lose their leftmost cells, vertical bars lose their topmost cells.
    - All other pixels remain unchanged.

    Key insight:
    - The outside colored markers jointly encode the groove location, orientation, and required cut length, while color acts as the lookup key that selects the distant source bar.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    fg = g[g != bg]
    assert fg.size > 0
    colors, counts = np.unique(fg, return_counts=True)
    mold_color = int(colors[int(np.argmax(counts))])
    mold_bbox = _bbox(g == mold_color)
    assert mold_bbox is not None
    r0, c0, r1, c1 = mold_bbox
    assert r1 - r0 >= 2 and c1 - c0 >= 2
    assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == mold_color)

    grooves = _extract_grooves(g, bg, mold_color, mold_bbox)
    bars = {
        int(groove["color"]): _find_bar(g, int(groove["color"]), groove["marker_coords"])
        for groove in grooves
    }

    out = g.copy()
    for groove in grooves:
        color = int(groove["color"])
        length = int(groove["length"])
        bar = bars[color]
        assert int(bar["length"]) >= length

        for r, c in groove["marker_coords"]:
            out[int(r), int(c)] = bg
        for r, c in groove["groove_coords"]:
            out[int(r), int(c)] = color

        if bar["orientation"] == "h":
            row = int(bar["row"])
            start = int(bar["start"])
            out[row, start : start + length] = bg
        else:
            col = int(bar["col"])
            start = int(bar["start"])
            out[start : start + length, col] = bg

    return out


def _side_capacities(mold_h: int, mold_w: int) -> dict[str, int]:
    return {
        "top": max(1, (mold_w - 1) // 3),
        "bottom": max(1, (mold_w - 1) // 3),
        "left": max(1, (mold_h - 1) // 3),
        "right": max(1, (mold_h - 1) // 3),
    }


def _sample_sides(mold_h: int, mold_w: int, n_grooves: int) -> list[str]:
    capacities = _side_capacities(mold_h, mold_w)
    counts = {side: 0 for side in capacities}
    top_side = random.choice(["top", "bottom"])
    left_side = random.choice(["left", "right"])
    counts[top_side] += 1
    counts[left_side] += 1
    sides = [top_side, left_side]

    while len(sides) < n_grooves:
        candidates = [side for side, cap in capacities.items() if counts[side] < cap]
        if not candidates:
            break
        side = random.choice(candidates)
        counts[side] += 1
        sides.append(side)

    if len(sides) != n_grooves:
        raise RuntimeError("insufficient side capacity")
    random.shuffle(sides)
    return sides


def _interval_clear(existing: list[tuple[int, int]], start: int, end: int) -> bool:
    return all(end < old_start - 1 or start > old_end + 1 for old_start, old_end in existing)


def _sample_grooves(
    grid: np.ndarray, r0: int, c0: int, mold_h: int, mold_w: int
) -> tuple[list[dict], dict[int, dict]] | None:
    total_capacity = sum(_side_capacities(mold_h, mold_w).values())
    max_grooves = min(GEN_GROOVE_COUNT_MAX, total_capacity, len(GEN_COLOR_POOL))
    if max_grooves < GEN_GROOVE_COUNT_MIN:
        return None

    n_grooves = random.randint(GEN_GROOVE_COUNT_MIN, max_grooves)
    try:
        sides = _sample_sides(mold_h, mold_w, n_grooves)
    except RuntimeError:
        return None

    intervals: dict[str, list[tuple[int, int]]] = {side: [] for side in ("top", "bottom", "left", "right")}
    colors = random.sample(GEN_COLOR_POOL, n_grooves)
    grooves: list[dict] = []

    for side, color in zip(sides, colors):
        span = mold_w if side in {"top", "bottom"} else mold_h
        for _ in range(80):
            length = random.randint(GEN_GROOVE_MIN, min(GEN_GROOVE_MAX, span - 2))
            start = random.randint(1, span - length - 1)
            end = start + length - 1
            if _interval_clear(intervals[side], start, end):
                intervals[side].append((start, end))
                break
        else:
            return None

        if side == "top":
            groove_coords = [(r0, c0 + k) for k in range(start, end + 1)]
            marker_coords = [(r0 - 1, c0 + k) for k in range(start, end + 1)]
        elif side == "bottom":
            groove_coords = [(r0 + mold_h - 1, c0 + k) for k in range(start, end + 1)]
            marker_coords = [(r0 + mold_h, c0 + k) for k in range(start, end + 1)]
        elif side == "left":
            groove_coords = [(r0 + k, c0) for k in range(start, end + 1)]
            marker_coords = [(r0 + k, c0 - 1) for k in range(start, end + 1)]
        else:
            groove_coords = [(r0 + k, c0 + mold_w - 1) for k in range(start, end + 1)]
            marker_coords = [(r0 + k, c0 + mold_w) for k in range(start, end + 1)]

        for r, c in groove_coords:
            grid[int(r), int(c)] = GEN_BACKGROUND
        for r, c in marker_coords:
            grid[int(r), int(c)] = int(color)

        grooves.append(
            {
                "side": side,
                "color": int(color),
                "length": int(length),
                "groove_coords": [(int(r), int(c)) for r, c in groove_coords],
                "marker_coords": [(int(r), int(c)) for r, c in marker_coords],
            }
        )

    bars: dict[int, dict] = {}
    exact_flags = [random.choice([False, True]) for _ in range(n_grooves)]
    if n_grooves >= 2 and (all(exact_flags) or not any(exact_flags)):
        exact_flags[0] = not exact_flags[0]

    orientations = [random.choice(["h", "v"]) for _ in range(n_grooves)]
    if n_grooves >= 2 and (all(ori == "h" for ori in orientations) or all(ori == "v" for ori in orientations)):
        orientations[0] = "v" if orientations[0] == "h" else "h"

    occupied = grid != GEN_BACKGROUND
    for groove, exact, orientation in zip(grooves, exact_flags, orientations):
        extra = 0 if exact else random.randint(1, GEN_BAR_EXTRA_MAX)
        bar_len = int(groove["length"]) + int(extra)
        placed = False

        for _ in range(200):
            if orientation == "h":
                row = random.randint(0, grid.shape[0] - 1)
                col = random.randint(0, grid.shape[1] - bar_len)
                rr0 = max(0, row - 1)
                rr1 = min(grid.shape[0] - 1, row + 1)
                cc0 = max(0, col - 1)
                cc1 = min(grid.shape[1] - 1, col + bar_len)
                if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    continue
                grid[row, col : col + bar_len] = int(groove["color"])
                occupied[row, col : col + bar_len] = True
                bars[int(groove["color"])] = {
                    "orientation": "h",
                    "row": int(row),
                    "start": int(col),
                    "length": int(bar_len),
                }
                placed = True
                break

            row = random.randint(0, grid.shape[0] - bar_len)
            col = random.randint(0, grid.shape[1] - 1)
            rr0 = max(0, row - 1)
            rr1 = min(grid.shape[0] - 1, row + bar_len)
            cc0 = max(0, col - 1)
            cc1 = min(grid.shape[1] - 1, col + 1)
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            grid[row : row + bar_len, col] = int(groove["color"])
            occupied[row : row + bar_len, col] = True
            bars[int(groove["color"])] = {
                "orientation": "v",
                "col": int(col),
                "start": int(row),
                "length": int(bar_len),
            }
            placed = True
            break

        if not placed:
            return None

    return grooves, bars


def _expected_output(grid: np.ndarray, grooves: list[dict], bars: dict[int, dict]) -> np.ndarray:
    out = grid.copy()
    for groove in grooves:
        color = int(groove["color"])
        length = int(groove["length"])
        for r, c in groove["marker_coords"]:
            out[int(r), int(c)] = GEN_BACKGROUND
        for r, c in groove["groove_coords"]:
            out[int(r), int(c)] = color

        bar = bars[color]
        if bar["orientation"] == "h":
            row = int(bar["row"])
            start = int(bar["start"])
            out[row, start : start + length] = GEN_BACKGROUND
        else:
            col = int(bar["col"])
            start = int(bar["start"])
            out[start : start + length, col] = GEN_BACKGROUND
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    mold_h = random.randint(GEN_MOLD_MIN, min(GEN_MOLD_MAX, h - 4))
    mold_w = random.randint(GEN_MOLD_MIN, min(GEN_MOLD_MAX, w - 4))
    r0 = random.randint(2, h - mold_h - 2)
    c0 = random.randint(2, w - mold_w - 2)

    grid = np.full((h, w), GEN_BACKGROUND, dtype=int)
    grid[r0 : r0 + mold_h, c0 : c0 + mold_w] = GEN_MOLD_COLOR

    sampled = _sample_grooves(grid, r0, c0, mold_h, mold_w)
    if sampled is None:
        return None
    grooves, bars = sampled

    expected = _expected_output(grid, grooves, bars)
    produced = main(grid)
    assert np.array_equal(produced, expected)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mold-and-bar input")
