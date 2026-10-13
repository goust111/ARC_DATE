import itertools
import random

import numpy as np


BACKGROUND = 0
HEAD_MARKER = 9
LINE_COLORS = tuple(range(1, 9))

GENERATION_ATTEMPTS = 500
GRID_SIZE_RANGE = (16, 26)
LINE_LENGTH_RANGE = (5, 12)
LINE_COUNT_RANGE = (2, 3)
POSITION_MARGIN = 2
POSITION_GAP = 3
MAX_CROSSINGS = 6


def _neighbors(pos: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _line_cells(line: dict) -> list[tuple[int, int]]:
    if line["orientation"] == "h":
        return [(line["row"], c) for c in range(line["start"], line["end"] + 1)]
    return [(r, line["col"]) for r in range(line["start"], line["end"] + 1)]


def _head_position(line: dict, at_start: bool) -> tuple[int, int]:
    if line["orientation"] == "h":
        return (line["row"], line["start"] if at_start else line["end"])
    return (line["start"] if at_start else line["end"], line["col"])


def _head_distance(line: dict, cell: tuple[int, int]) -> int:
    hr, hc = line["head"]
    r, c = cell
    return abs(r - hr) + abs(c - hc)


def _extract_lines(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    heads = [tuple(map(int, coord)) for coord in np.argwhere(g == HEAD_MARKER)]
    assert heads, "expected at least one maroon head marker"

    lines: list[dict] = []
    seen_colors: set[int] = set()
    for head_r, head_c in heads:
        neighbors = []
        for nr, nc in _neighbors((head_r, head_c)):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                color = int(g[nr, nc])
                if color not in {BACKGROUND, HEAD_MARKER}:
                    neighbors.append((nr - head_r, nc - head_c, color))

        assert len(neighbors) == 1, "each maroon head must touch exactly one colored line pixel"
        dr, dc, color = neighbors[0]
        assert color not in seen_colors, "line colors must be unique"
        seen_colors.add(color)

        coords = np.argwhere(g == color)
        assert len(coords) > 0, "every line color must still have visible pixels"
        if dc != 0:
            assert np.all(coords[:, 0] == head_r), "horizontal line colors must stay on one row"
            step = 1 if dc > 0 else -1
            deltas = coords[:, 1] - head_c
            assert np.all(deltas * step > 0), "horizontal line must extend away from its head"
            tail_c = int(coords[:, 1].max() if step > 0 else coords[:, 1].min())
            for c in range(min(head_c, tail_c), max(head_c, tail_c) + 1):
                assert int(g[head_r, c]) in {BACKGROUND, color, HEAD_MARKER}
            lines.append(
                {
                    "orientation": "h",
                    "color": color,
                    "head": (head_r, head_c),
                    "tail": (head_r, tail_c),
                    "row": head_r,
                    "start": min(head_c, tail_c),
                    "end": max(head_c, tail_c),
                }
            )
        else:
            assert np.all(coords[:, 1] == head_c), "vertical line colors must stay on one column"
            step = 1 if dr > 0 else -1
            deltas = coords[:, 0] - head_r
            assert np.all(deltas * step > 0), "vertical line must extend away from its head"
            tail_r = int(coords[:, 0].max() if step > 0 else coords[:, 0].min())
            for r in range(min(head_r, tail_r), max(head_r, tail_r) + 1):
                assert int(g[r, head_c]) in {BACKGROUND, color, HEAD_MARKER}
            lines.append(
                {
                    "orientation": "v",
                    "color": color,
                    "head": (head_r, head_c),
                    "tail": (tail_r, head_c),
                    "col": head_c,
                    "start": min(head_r, tail_r),
                    "end": max(head_r, tail_r),
                }
            )

    visible_colors = set(map(int, np.unique(g))) - {BACKGROUND, HEAD_MARKER}
    assert visible_colors == seen_colors, "every visible color must belong to exactly one marked line"
    return lines


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each marked line and resolve every broken crossing by the closer head.

    Input structure:
    - The background is black.
    - Several one-pixel-wide horizontal or vertical colored line segments appear.
    - Each segment has exactly one maroon endpoint marker replacing the segment's own color, and the opposite endpoint keeps the segment color.
    - Whenever a horizontal and a vertical segment cross, their shared pixel is black, so both segments are visibly broken by one cell there.

    Rule:
    - Identify each line's orientation, original color, full straight extent, and which endpoint is the maroon-marked head.
    - Recolor every maroon head back to that line's original color.
    - For each black crossing cell covered by one horizontal line and one vertical line, compare the step distance from the crossing to each line's head along that line.
    - Fill the crossing with the color of the line whose head is closer, and leave all other cells unchanged.

    Key insight:
    - The maroon endpoint is not a final color; it encodes the head direction of its line, and that head distance decides which broken line should appear on top at each crossing.
    """
    g = np.asarray(grid, dtype=int)
    lines = _extract_lines(g)

    out = g.copy()
    horizontals = [line for line in lines if line["orientation"] == "h"]
    verticals = [line for line in lines if line["orientation"] == "v"]

    for line in lines:
        out[line["head"]] = line["color"]

    for h_line in horizontals:
        for v_line in verticals:
            cross = (h_line["row"], v_line["col"])
            if not (h_line["start"] <= cross[1] <= h_line["end"]):
                continue
            if not (v_line["start"] <= cross[0] <= v_line["end"]):
                continue
            if int(g[cross]) != BACKGROUND:
                continue

            h_dist = _head_distance(h_line, cross)
            v_dist = _head_distance(v_line, cross)
            assert h_dist != v_dist, "crossing distances must be unambiguous"
            out[cross] = h_line["color"] if h_dist < v_dist else v_line["color"]

    return out


def _sample_spaced_positions(
    rng: random.Random, low: int, high: int, count: int, min_gap: int
) -> list[int] | None:
    values = list(range(low, high + 1))
    if len(values) < count:
        return None

    for _ in range(200):
        picks = sorted(rng.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _build_crossings(horizontals: list[dict], verticals: list[dict]) -> list[tuple[int, int, int, int]] | None:
    crossings: list[tuple[int, int, int, int]] = []
    for hi, h_line in enumerate(horizontals):
        for vi, v_line in enumerate(verticals):
            overlaps = h_line["start"] <= v_line["col"] <= h_line["end"]
            overlaps &= v_line["start"] <= h_line["row"] <= v_line["end"]
            if not overlaps:
                continue

            interior = h_line["start"] < v_line["col"] < h_line["end"]
            interior &= v_line["start"] < h_line["row"] < v_line["end"]
            if not interior:
                return None

            crossings.append((hi, vi, h_line["row"], v_line["col"]))
    return crossings


def _occupied_cells(horizontals: list[dict], verticals: list[dict], crossings: list[tuple[int, int, int, int]]) -> set[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    for line in horizontals + verticals:
        occupied.update(_line_cells(line))
    for _hi, _vi, r, c in crossings:
        occupied.discard((r, c))
    return occupied


def _choose_heads(
    rng: random.Random,
    horizontals: list[dict],
    verticals: list[dict],
    crossings: list[tuple[int, int, int, int]],
) -> tuple[list[bool], list[bool]] | None:
    occupied = _occupied_cells(horizontals, verticals, crossings)

    h_options: list[list[bool]] = []
    for line in horizontals:
        options = []
        for at_start in (True, False):
            head = _head_position(line, at_start)
            count = sum((neighbor in occupied) for neighbor in _neighbors(head))
            if count == 1:
                options.append(at_start)
        if not options:
            return None
        h_options.append(options)

    v_options: list[list[bool]] = []
    for line in verticals:
        options = []
        for at_start in (True, False):
            head = _head_position(line, at_start)
            count = sum((neighbor in occupied) for neighbor in _neighbors(head))
            if count == 1:
                options.append(at_start)
        if not options:
            return None
        v_options.append(options)

    valid_assignments: list[tuple[list[bool], list[bool]]] = []
    for h_choice in itertools.product(*h_options):
        h_heads = [_head_position(line, at_start) for line, at_start in zip(horizontals, h_choice)]
        for v_choice in itertools.product(*v_options):
            v_heads = [_head_position(line, at_start) for line, at_start in zip(verticals, v_choice)]
            winners: set[str] = set()
            tie = False
            for hi, vi, r, c in crossings:
                h_dist = abs(c - h_heads[hi][1])
                v_dist = abs(r - v_heads[vi][0])
                if h_dist == v_dist:
                    tie = True
                    break
                winners.add("h" if h_dist < v_dist else "v")
            if not tie and winners == {"h", "v"}:
                valid_assignments.append((list(h_choice), list(v_choice)))

    if not valid_assignments:
        return None
    return rng.choice(valid_assignments)


def _render_case(
    shape: tuple[int, int],
    horizontals: list[dict],
    verticals: list[dict],
    crossings: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BACKGROUND, dtype=int)
    out = np.full(shape, BACKGROUND, dtype=int)

    for line in horizontals + verticals:
        if line["orientation"] == "h":
            inp[line["row"], line["start"] : line["end"] + 1] = line["color"]
            out[line["row"], line["start"] : line["end"] + 1] = line["color"]
        else:
            inp[line["start"] : line["end"] + 1, line["col"]] = line["color"]
            out[line["start"] : line["end"] + 1, line["col"]] = line["color"]

    for hi, vi, r, c in crossings:
        inp[r, c] = BACKGROUND
        h_line = horizontals[hi]
        v_line = verticals[vi]
        h_dist = _head_distance(h_line, (r, c))
        v_dist = _head_distance(v_line, (r, c))
        out[r, c] = h_line["color"] if h_dist < v_dist else v_line["color"]

    for line in horizontals + verticals:
        inp[line["head"]] = HEAD_MARKER

    return inp, out


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    height = rng.randint(*GRID_SIZE_RANGE)
    width = rng.randint(*GRID_SIZE_RANGE)

    h_count = rng.randint(*LINE_COUNT_RANGE)
    v_count = rng.randint(*LINE_COUNT_RANGE)
    if h_count + v_count > len(LINE_COLORS):
        return None

    rows = _sample_spaced_positions(
        rng,
        POSITION_MARGIN,
        height - POSITION_MARGIN - 1,
        h_count,
        POSITION_GAP,
    )
    cols = _sample_spaced_positions(
        rng,
        POSITION_MARGIN,
        width - POSITION_MARGIN - 1,
        v_count,
        POSITION_GAP,
    )
    if rows is None or cols is None:
        return None

    min_h_len = LINE_LENGTH_RANGE[0]
    max_h_len = min(LINE_LENGTH_RANGE[1], width - 2)
    min_v_len = LINE_LENGTH_RANGE[0]
    max_v_len = min(LINE_LENGTH_RANGE[1], height - 2)
    if min_h_len > max_h_len or min_v_len > max_v_len:
        return None

    horizontals = []
    for row in rows:
        length = rng.randint(min_h_len, max_h_len)
        start = rng.randint(1, width - length - 1)
        horizontals.append({"orientation": "h", "row": row, "start": start, "end": start + length - 1})

    verticals = []
    for col in cols:
        length = rng.randint(min_v_len, max_v_len)
        start = rng.randint(1, height - length - 1)
        verticals.append({"orientation": "v", "col": col, "start": start, "end": start + length - 1})

    crossings = _build_crossings(horizontals, verticals)
    if crossings is None or not (2 <= len(crossings) <= MAX_CROSSINGS):
        return None

    touched_h = {hi for hi, _vi, _r, _c in crossings}
    touched_v = {vi for _hi, vi, _r, _c in crossings}
    if len(touched_h) < 2 or len(touched_v) < 2:
        return None

    head_assignment = _choose_heads(rng, horizontals, verticals, crossings)
    if head_assignment is None:
        return None
    h_choice, v_choice = head_assignment

    colors = rng.sample(LINE_COLORS, h_count + v_count)
    for line, color, at_start in zip(horizontals, colors[:h_count], h_choice):
        line["color"] = color
        line["head"] = _head_position(line, at_start)
    for line, color, at_start in zip(verticals, colors[h_count:], v_choice):
        line["color"] = color
        line["head"] = _head_position(line, at_start)

    inp, out = _render_case((height, width), horizontals, verticals, crossings)
    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input for the line-over-under puzzle")
