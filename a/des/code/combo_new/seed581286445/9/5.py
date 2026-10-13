import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

SOURCE_COLORS = (RED, BLUE, YELLOW)
MIX_COLORS = (MAGENTA, ORANGE, GREEN)
MIX_MAP = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
}

GENERATION_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (12, 24)
GRID_WIDTH_RANGE = (12, 24)
LINE_COUNT_RANGE = (2, 4)
LINE_GAP = 2


def _horizontal_runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    height, width = mask.shape
    for r in range(height):
        c = 0
        while c < width:
            if not bool(mask[r, c]):
                c += 1
                continue
            c0 = c
            while c + 1 < width and bool(mask[r, c + 1]):
                c += 1
            c1 = c
            if c1 - c0 + 1 >= 2:
                runs.append((r, c0, c1))
            c += 1
    return runs


def _vertical_runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    height, width = mask.shape
    for c in range(width):
        r = 0
        while r < height:
            if not bool(mask[r, c]):
                r += 1
                continue
            r0 = r
            while r + 1 < height and bool(mask[r + 1, c]):
                r += 1
            r1 = r
            if r1 - r0 + 1 >= 2:
                runs.append((c, r0, r1))
            r += 1
    return runs


def _source_color(grid: np.ndarray, r: int, c: int) -> int | None:
    height, width = grid.shape
    if 0 <= r < height and 0 <= c < width:
        color = int(grid[r, c])
        if color in SOURCE_COLORS:
            return color
    return None


def _line_signal(grid: np.ndarray, line: tuple[int, int, int], orientation: str) -> int | None:
    sources: list[int] = []
    if orientation == "H":
        r, c0, c1 = line
        left = _source_color(grid, r, c0 - 1)
        right = _source_color(grid, r, c1 + 1)
    else:
        c, r0, r1 = line
        left = _source_color(grid, r0 - 1, c)
        right = _source_color(grid, r1 + 1, c)
    for color in (left, right):
        if color is not None:
            sources.append(color)
    assert len(sources) <= 1, "each pipe may have at most one adjacent source"
    return sources[0] if sources else None


def _mix_signal(horizontal: int | None, vertical: int | None) -> int | None:
    if horizontal is None:
        return vertical
    if vertical is None:
        return horizontal
    if horizontal == vertical:
        return horizontal
    return MIX_MAP[frozenset((horizontal, vertical))]


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate source colors through straight grey pipes and mix the two directions at crossings.

    Input structure:
    - The background is black.
    - Grey pixels form one-cell-thick horizontal and vertical straight pipe segments, and a crossing is a single grey cell shared by one horizontal and one vertical segment.
    - Some pipe endpoints have one adjacent source pixel in red, blue, or yellow; endpoints without a source are open to black or the grid boundary.

    Rule:
    - Each sourced pipe carries its source color across every grey cell on that straight segment.
    - At a crossing, combine the horizontal and vertical signals: red+blue becomes magenta, red+yellow becomes orange, and blue+yellow becomes green.
    - If only one direction carries a signal, use that signal color at the crossing; if neither direction carries a signal, leave that pipe cell grey.
    - Source pixels themselves stay unchanged.

    Key insight:
    - Recover the horizontal and vertical pipe segments separately, then treat each crossing as the overlap of one horizontal signal and one vertical signal instead of a simple overwrite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pipe_mask = g == GREY
    out = g.copy()
    hsig = np.full(g.shape, -1, dtype=int)
    vsig = np.full(g.shape, -1, dtype=int)

    for line in _horizontal_runs(pipe_mask):
        color = _line_signal(g, line, "H")
        if color is None:
            continue
        r, c0, c1 = line
        hsig[r, c0 : c1 + 1] = color

    for line in _vertical_runs(pipe_mask):
        color = _line_signal(g, line, "V")
        if color is None:
            continue
        c, r0, r1 = line
        vsig[r0 : r1 + 1, c] = color

    for r, c in np.argwhere(pipe_mask):
        horizontal = None if hsig[r, c] < 0 else int(hsig[r, c])
        vertical = None if vsig[r, c] < 0 else int(vsig[r, c])
        mixed = _mix_signal(horizontal, vertical)
        if mixed is not None:
            out[r, c] = mixed

    return out


def _max_line_count(size: int) -> int:
    available = size - 4
    return 0 if available <= 0 else (available + LINE_GAP - 1) // LINE_GAP


def _sample_spaced_positions(count: int, lo: int, hi: int, gap: int) -> list[int] | None:
    available = hi - lo + 1
    compressed = available - (gap - 1) * (count - 1)
    if compressed < count:
        return None
    base = sorted(random.sample(range(compressed), count))
    return [lo + value + i * (gap - 1) for i, value in enumerate(base)]


def _sample_activity(count: int) -> list[bool]:
    active_count = random.randint(1, count - 1)
    flags = [True] * active_count + [False] * (count - active_count)
    random.shuffle(flags)
    return flags


def _assign_colors(h_active: list[bool], v_active: list[bool]) -> tuple[list[int | None], list[int | None]]:
    h_colors: list[int | None] = [None] * len(h_active)
    v_colors: list[int | None] = [None] * len(v_active)

    h_indices = [i for i, active in enumerate(h_active) if active]
    v_indices = [i for i, active in enumerate(v_active) if active]
    assert h_indices and v_indices

    first_h = random.choice(h_indices)
    first_v = random.choice(v_indices)
    h_colors[first_h] = random.choice(SOURCE_COLORS)
    v_colors[first_v] = random.choice([color for color in SOURCE_COLORS if color != h_colors[first_h]])

    for i in h_indices:
        if h_colors[i] is None:
            h_colors[i] = random.choice(SOURCE_COLORS)
    for i in v_indices:
        if v_colors[i] is None:
            v_colors[i] = random.choice(SOURCE_COLORS)
    return h_colors, v_colors


def _random_span(lo: int, hi: int) -> tuple[int, int]:
    start = random.randint(*lo)
    stop = random.randint(*hi)
    return start, stop


def _build_lines(height: int, width: int) -> tuple[list[dict], list[dict]] | None:
    max_h = min(LINE_COUNT_RANGE[1], _max_line_count(height))
    max_v = min(LINE_COUNT_RANGE[1], _max_line_count(width))
    if max_h < LINE_COUNT_RANGE[0] or max_v < LINE_COUNT_RANGE[0]:
        return None

    h_count = random.randint(LINE_COUNT_RANGE[0], max_h)
    v_count = random.randint(LINE_COUNT_RANGE[0], max_v)
    rows = _sample_spaced_positions(h_count, 2, height - 3, LINE_GAP)
    cols = _sample_spaced_positions(v_count, 2, width - 3, LINE_GAP)
    if rows is None or cols is None:
        return None

    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    h_active = _sample_activity(h_count)
    v_active = _sample_activity(v_count)
    h_colors, v_colors = _assign_colors(h_active, v_active)

    horizontals: list[dict] = []
    verticals: list[dict] = []

    for i, row in enumerate(rows):
        active = h_active[i]
        side = random.choice(("left", "right")) if active else None

        c0_lo = 1 if side == "left" else 0
        c0_hi = min_col - 1
        c1_lo = max_col + 1
        c1_hi = width - 2 if side == "right" else width - 1
        if c0_lo > c0_hi or c1_lo > c1_hi:
            return None
        c0 = random.randint(c0_lo, c0_hi)
        c1 = random.randint(c1_lo, c1_hi)
        horizontals.append(
            {
                "row": row,
                "c0": c0,
                "c1": c1,
                "active": active,
                "side": side,
                "color": h_colors[i],
            }
        )

    for i, col in enumerate(cols):
        active = v_active[i]
        side = random.choice(("top", "bottom")) if active else None

        r0_lo = 1 if side == "top" else 0
        r0_hi = min_row - 1
        r1_lo = max_row + 1
        r1_hi = height - 2 if side == "bottom" else height - 1
        if r0_lo > r0_hi or r1_lo > r1_hi:
            return None
        r0 = random.randint(r0_lo, r0_hi)
        r1 = random.randint(r1_lo, r1_hi)
        verticals.append(
            {
                "col": col,
                "r0": r0,
                "r1": r1,
                "active": active,
                "side": side,
                "color": v_colors[i],
            }
        )

    return horizontals, verticals


def _draw_input(height: int, width: int, horizontals: list[dict], verticals: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)

    for line in horizontals:
        grid[line["row"], line["c0"] : line["c1"] + 1] = GREY
    for line in verticals:
        grid[line["r0"] : line["r1"] + 1, line["col"]] = GREY

    for line in horizontals:
        if not line["active"]:
            continue
        if line["side"] == "left":
            sr, sc = line["row"], line["c0"] - 1
        else:
            sr, sc = line["row"], line["c1"] + 1
        assert grid[sr, sc] == BLACK
        grid[sr, sc] = int(line["color"])

    for line in verticals:
        if not line["active"]:
            continue
        if line["side"] == "top":
            sr, sc = line["r0"] - 1, line["col"]
        else:
            sr, sc = line["r1"] + 1, line["col"]
        assert grid[sr, sc] == BLACK
        grid[sr, sc] = int(line["color"])

    return grid


def _line_overlap(horizontal: dict, vertical: dict) -> bool:
    return (
        horizontal["c0"] <= vertical["col"] <= horizontal["c1"]
        and vertical["r0"] <= horizontal["row"] <= vertical["r1"]
    )


def _render_expected(inp: np.ndarray, horizontals: list[dict], verticals: list[dict]) -> np.ndarray:
    out = inp.copy()

    for line in horizontals:
        if line["active"]:
            out[line["row"], line["c0"] : line["c1"] + 1] = int(line["color"])
    for line in verticals:
        if line["active"]:
            out[line["r0"] : line["r1"] + 1, line["col"]] = int(line["color"])

    for horizontal in horizontals:
        for vertical in verticals:
            if not _line_overlap(horizontal, vertical):
                continue
            r, c = horizontal["row"], vertical["col"]
            h_color = int(horizontal["color"]) if horizontal["active"] else None
            v_color = int(vertical["color"]) if vertical["active"] else None
            mixed = _mix_signal(h_color, v_color)
            if mixed is None:
                out[r, c] = GREY
            else:
                out[r, c] = mixed

    return out


def _branch_counts(horizontals: list[dict], verticals: list[dict]) -> tuple[int, int, int]:
    mixed = 0
    single = 0
    idle = 0
    for horizontal in horizontals:
        for vertical in verticals:
            if not _line_overlap(horizontal, vertical):
                continue
            h_active = bool(horizontal["active"])
            v_active = bool(vertical["active"])
            if h_active and v_active:
                if horizontal["color"] != vertical["color"]:
                    mixed += 1
            elif h_active or v_active:
                single += 1
            else:
                idle += 1
    return mixed, single, idle


def _validate_structure(inp: np.ndarray, horizontals: list[dict], verticals: list[dict], expected: np.ndarray) -> bool:
    if not np.array_equal(main(inp), expected):
        return False
    if np.array_equal(inp, expected):
        return False

    pipe_mask = inp == GREY
    if len(_horizontal_runs(pipe_mask)) != len(horizontals):
        return False
    if len(_vertical_runs(pipe_mask)) != len(verticals):
        return False

    mixed, single, idle = _branch_counts(horizontals, verticals)
    if mixed <= 0 or single <= 0 or idle <= 0:
        return False

    if not np.any(np.isin(expected, MIX_COLORS)):
        return False
    if not np.any(expected == GREY):
        return False

    source_mask = np.isin(inp, SOURCE_COLORS)
    if not np.array_equal(inp[source_mask], expected[source_mask]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    built = _build_lines(height, width)
    if built is None:
        return None
    horizontals, verticals = built
    inp = _draw_input(height, width, horizontals, verticals)
    expected = _render_expected(inp, horizontals, verticals)
    if not _validate_structure(inp, horizontals, verticals, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid pipe-mixing puzzle instance")
