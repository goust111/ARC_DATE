from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_pairs(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    pairs: list[dict[str, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        coords = np.argwhere(grid == int(color))
        if int(coords.shape[0]) != 2:
            continue

        (r0, c0), (r1, c1) = map(tuple, coords.tolist())
        if r0 == r1 and c0 != c1:
            c0, c1 = sorted((int(c0), int(c1)))
            pairs.append(
                {
                    "color": int(color),
                    "orientation": 0,  # horizontal
                    "fixed": int(r0),
                    "start": int(c0),
                    "end": int(c1),
                }
            )
        elif c0 == c1 and r0 != r1:
            r0, r1 = sorted((int(r0), int(r1)))
            pairs.append(
                {
                    "color": int(color),
                    "orientation": 1,  # vertical
                    "fixed": int(c0),
                    "start": int(r0),
                    "end": int(r1),
                }
            )

    pairs.sort(key=lambda item: (item["orientation"], item["fixed"], item["start"], item["color"]))
    return pairs


def _paint_line_and_expansions(
    out: np.ndarray,
    source: np.ndarray,
    pair: dict[str, int],
    bg: int,
) -> None:
    color = int(pair["color"])
    fixed = int(pair["fixed"])
    start = int(pair["start"])
    end = int(pair["end"])
    height, width = out.shape

    if int(pair["orientation"]) == 0:
        row = fixed
        out[row, start : end + 1] = color
        for col in range(start + 1, end):
            cell_color = int(source[row, col])
            if cell_color in (int(bg), color):
                continue
            dist = min(col - start, end - col)
            for step in range(1, dist + 1):
                up = row - step
                down = row + step
                if 0 <= up < height:
                    out[up, col] = cell_color
                if 0 <= down < height:
                    out[down, col] = cell_color
        return

    col = fixed
    out[start : end + 1, col] = color
    for row in range(start + 1, end):
        cell_color = int(source[row, col])
        if cell_color in (int(bg), color):
            continue
        dist = min(row - start, end - row)
        for step in range(1, dist + 1):
            left = col - step
            right = col + step
            if 0 <= left < width:
                out[row, left] = cell_color
            if 0 <= right < width:
                out[row, right] = cell_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color endpoint pair with a straight line and let crossed singleton pixels expand perpendicularly.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    pairs = _find_pairs(g, bg)

    out = g.copy()
    for pair in pairs:
        _paint_line_and_expansions(out, g, pair, bg)
    return out


# --------------- generator ---------------

MIN_GRID_SIZE = 11
MAX_GRID_SIZE = 19
MIN_PAIR_COUNT = 2
MAX_PAIR_COUNT = 4
MAX_SPAN = 8
GEN_ATTEMPTS = 500
PAIR_ATTEMPTS = 250


def _dilate8(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    out.add((rr, cc))
    return out


def _pair_footprint(
    height: int,
    width: int,
    orientation: str,
    fixed: int,
    start: int,
    end: int,
    activators: list[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    cells: set[tuple[int, int]] = set()

    if orientation == "h":
        row = fixed
        for col in range(start, end + 1):
            cells.add((row, col))
        for col, _color in activators:
            dist = min(col - start, end - col)
            for step in range(1, dist + 1):
                up = row - step
                down = row + step
                if up < 0 or down >= height:
                    return None
                cells.add((up, col))
                cells.add((down, col))
        return cells

    col = fixed
    for row in range(start, end + 1):
        cells.add((row, col))
    for row, _color in activators:
        dist = min(row - start, end - row)
        for step in range(1, dist + 1):
            left = col - step
            right = col + step
            if left < 0 or right >= width:
                return None
            cells.add((row, left))
            cells.add((row, right))
    return cells


def _render_expected(
    inp: np.ndarray,
    records: list[dict[str, object]],
) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    for record in records:
        if record["orientation"] == "h":
            row = int(record["fixed"])
            start = int(record["start"])
            end = int(record["end"])
            out[row, start : end + 1] = int(record["pair_color"])
            for col, color in record["activators"]:
                dist = min(int(col) - start, end - int(col))
                for step in range(1, dist + 1):
                    out[row - step, int(col)] = int(color)
                    out[row + step, int(col)] = int(color)
        else:
            col = int(record["fixed"])
            start = int(record["start"])
            end = int(record["end"])
            out[start : end + 1, col] = int(record["pair_color"])
            for row, color in record["activators"]:
                dist = min(int(row) - start, end - int(row))
                for step in range(1, dist + 1):
                    out[int(row), col - step] = int(color)
                    out[int(row), col + step] = int(color)
    return out


def _orientation_plan(pair_count: int) -> list[str]:
    if pair_count == 2 or random.random() < 0.6:
        h_count = random.randint(1, pair_count - 1)
        plan = ["h"] * h_count + ["v"] * (pair_count - h_count)
        random.shuffle(plan)
        return plan
    orientation = random.choice(["h", "v"])
    return [orientation] * pair_count


def _activator_counts(pair_count: int) -> list[int] | None:
    max_extra = 9 - pair_count
    if max_extra < 1:
        return None

    counts = [0] * pair_count
    triggered = random.randint(1, min(pair_count - 1, max_extra))
    triggered_ids = random.sample(range(pair_count), triggered)
    for idx in triggered_ids:
        counts[idx] = 1

    remaining = max_extra - triggered
    extra_ids = triggered_ids[:]
    random.shuffle(extra_ids)
    for idx in extra_ids:
        if remaining <= 0:
            break
        if random.random() < 0.5:
            counts[idx] += 1
            remaining -= 1
    return counts


def _choose_colors(pair_count: int, activator_total: int) -> tuple[int, list[int], list[int]]:
    palette = list(range(10))
    bg = random.choice(palette)
    non_bg = [color for color in palette if color != bg]
    random.shuffle(non_bg)
    need = pair_count + activator_total
    chosen = non_bg[:need]
    return bg, chosen[:pair_count], chosen[pair_count:]


def _try_place_pair(
    height: int,
    width: int,
    orientation: str,
    pair_color: int,
    activator_colors: list[int],
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
) -> dict[str, object] | None:
    act_count = len(activator_colors)
    min_span = max(2, act_count + 1)
    max_span = min(MAX_SPAN, width - 1 if orientation == "h" else height - 1)
    if max_span < min_span:
        return None

    for _ in range(PAIR_ATTEMPTS):
        span = random.randint(min_span, max_span)
        if orientation == "h":
            row_lo = 1 if act_count > 0 else 0
            row_hi = height - 2 if act_count > 0 else height - 1
            if row_lo > row_hi:
                return None
            row = random.randint(row_lo, row_hi)
            start = random.randint(0, width - span - 1)
            end = start + span

            valid_cols = []
            vertical_clear = min(row, height - 1 - row)
            for col in range(start + 1, end):
                dist = min(col - start, end - col)
                if dist <= vertical_clear:
                    valid_cols.append(col)
            if act_count > len(valid_cols):
                continue
            chosen_cols = sorted(random.sample(valid_cols, act_count))
            activators = list(zip(chosen_cols, activator_colors))
            footprint = _pair_footprint(height, width, orientation, row, start, end, activators)
            if footprint is None or footprint & reserved:
                continue

            record = {
                "orientation": orientation,
                "fixed": row,
                "start": start,
                "end": end,
                "pair_color": int(pair_color),
                "activators": [(int(col), int(color)) for col, color in activators],
                "footprint": footprint,
            }
            return record

        col_lo = 1 if act_count > 0 else 0
        col_hi = width - 2 if act_count > 0 else width - 1
        if col_lo > col_hi:
            return None
        col = random.randint(col_lo, col_hi)
        start = random.randint(0, height - span - 1)
        end = start + span

        valid_rows = []
        horizontal_clear = min(col, width - 1 - col)
        for row in range(start + 1, end):
            dist = min(row - start, end - row)
            if dist <= horizontal_clear:
                valid_rows.append(row)
        if act_count > len(valid_rows):
            continue
        chosen_rows = sorted(random.sample(valid_rows, act_count))
        activators = list(zip(chosen_rows, activator_colors))
        footprint = _pair_footprint(height, width, orientation, col, start, end, activators)
        if footprint is None or footprint & reserved:
            continue

        record = {
            "orientation": orientation,
            "fixed": col,
            "start": start,
            "end": end,
            "pair_color": int(pair_color),
            "activators": [(int(row), int(color)) for row, color in activators],
            "footprint": footprint,
        }
        return record

    return None


def _apply_record(grid: np.ndarray, record: dict[str, object]) -> None:
    color = int(record["pair_color"])
    start = int(record["start"])
    end = int(record["end"])

    if record["orientation"] == "h":
        row = int(record["fixed"])
        grid[row, start] = color
        grid[row, end] = color
        for col, act_color in record["activators"]:
            grid[row, int(col)] = int(act_color)
        return

    col = int(record["fixed"])
    grid[start, col] = color
    grid[end, col] = color
    for row, act_color in record["activators"]:
        grid[int(row), col] = int(act_color)


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    width = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    pair_count = random.randint(MIN_PAIR_COUNT, MAX_PAIR_COUNT)

    act_counts = _activator_counts(pair_count)
    if act_counts is None:
        return None

    orientations = _orientation_plan(pair_count)
    activator_total = sum(int(v) for v in act_counts)
    bg, pair_colors, extra_colors = _choose_colors(pair_count, activator_total)

    specs: list[dict[str, object]] = []
    cursor = 0
    for orientation, pair_color, act_count in zip(orientations, pair_colors, act_counts):
        specs.append(
            {
                "orientation": orientation,
                "pair_color": int(pair_color),
                "activator_colors": extra_colors[cursor : cursor + int(act_count)],
            }
        )
        cursor += int(act_count)

    random.shuffle(specs)
    specs.sort(key=lambda item: len(item["activator_colors"]), reverse=True)

    grid = np.full((height, width), int(bg), dtype=int)
    reserved: set[tuple[int, int]] = set()
    records: list[dict[str, object]] = []

    for spec in specs:
        record = _try_place_pair(
            height=height,
            width=width,
            orientation=str(spec["orientation"]),
            pair_color=int(spec["pair_color"]),
            activator_colors=[int(v) for v in spec["activator_colors"]],
            grid=grid,
            reserved=reserved,
        )
        if record is None:
            return None
        _apply_record(grid, record)
        records.append(record)
        reserved |= _dilate8(set(record["footprint"]), height, width)

    if _most_common_color(grid) != int(bg):
        return None

    non_bg_counts = Counter(int(v) for v in grid.ravel() if int(v) != int(bg))
    pair_like = 0
    singleton_count = 0
    for color, count in non_bg_counts.items():
        if count == 2:
            coords = np.argwhere(grid == int(color))
            if int(coords.shape[0]) == 2 and (
                int(coords[0, 0]) == int(coords[1, 0]) or int(coords[0, 1]) == int(coords[1, 1])
            ):
                pair_like += 1
        elif count == 1:
            singleton_count += 1
        else:
            return None
    if pair_like != pair_count or singleton_count != activator_total:
        return None

    expected = _render_expected(grid, records)
    produced = main(grid)
    if not np.array_equal(expected, produced):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
