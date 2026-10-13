from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np


_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _UTILS_DIR not in sys.path:
    sys.path.append(_UTILS_DIR)

try:
    from arc_utils import get_bg as _arc_get_bg
except Exception:  # pragma: no cover - keeps SourceFileLoader use standalone.
    _arc_get_bg = None


Bar = dict[str, int]
Marble = tuple[int, int]
ARC_COLORS = tuple(range(10))

GEN_ATTEMPTS = 700
ROW_GAP = 3
MIN_BAR_LEN = 5
MAX_BAR_LEN = 13


def _get_bg(g: np.ndarray) -> int:
    if _arc_get_bg is not None:
        return int(_arc_get_bg(g))
    return int(Counter(map(int, g.reshape(-1))).most_common(1)[0][0])


def _unique_dominant_bg(g: np.ndarray) -> int:
    bg = _get_bg(g)
    counts = Counter(map(int, g.reshape(-1)))
    max_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == max_count]
    assert winners == [bg], "background color must be uniquely dominant"
    return int(bg)


def _runs(cols: list[int]) -> list[tuple[int, int]]:
    if not cols:
        return []
    spans: list[tuple[int, int]] = []
    start = end = int(cols[0])
    for col in map(int, cols[1:]):
        if col == end + 1:
            end = col
            continue
        spans.append((start, end))
        start = end = col
    spans.append((start, end))
    return spans


def _find_top_marbles(g: np.ndarray, bg: int) -> list[Marble]:
    cols = [int(c) for c in np.flatnonzero(g[0] != int(bg))]
    assert len(cols) >= 2, "expected several top-row marbles"
    assert all(b - a > 1 for a, b in zip(cols, cols[1:])), "top marbles must be nonadjacent"

    marbles = [(int(c), int(g[0, c])) for c in cols]
    colors = [color for _col, color in marbles]
    assert len(set(colors)) == len(colors), "top marble colors must be distinct"
    return marbles


def _extract_bars(g: np.ndarray, bg: int) -> list[Bar]:
    H, _W = g.shape
    bars: list[Bar] = []
    for r in range(1, H):
        cols = [int(c) for c in np.flatnonzero(g[r] != int(bg))]
        if not cols:
            continue
        assert r < H - 1, "bar rows must leave space below them"
        spans = _runs(cols)
        assert len(spans) == 1, "each non-top foreground row must contain one horizontal bar"
        c0, c1 = spans[0]
        assert c1 - c0 + 1 >= MIN_BAR_LEN, "bar must be long enough to show a gate"

        run = g[r, c0 : c1 + 1]
        assert not np.any(run == int(bg)), "bar body and gate must be a contiguous run"
        count_by_color = Counter(map(int, run))
        assert len(count_by_color) == 2, "bar row must contain one body color and one gate color"

        singletons = [color for color, count in count_by_color.items() if count == 1]
        bodies = [color for color, count in count_by_color.items() if count == len(run) - 1]
        assert len(singletons) == 1 and len(bodies) == 1, "bar must have exactly one gate pixel"
        gate_color = int(singletons[0])
        body_color = int(bodies[0])
        gate_rel = int(np.flatnonzero(run == gate_color)[0])
        gate_col = int(c0 + gate_rel)
        assert c0 < gate_col < c1, "gate pixel must be an interior opening"
        assert body_color != gate_color and body_color != int(bg) and gate_color != int(bg)

        bars.append(
            {
                "row": int(r),
                "c0": int(c0),
                "c1": int(c1),
                "gate_col": int(gate_col),
                "body": int(body_color),
                "gate": int(gate_color),
            }
        )

    assert len(bars) >= 2, "expected multiple gate bars"
    bars.sort(key=lambda b: int(b["row"]))
    assert len({int(b["row"]) for b in bars}) == len(bars), "bar heights must be unique"
    return bars


def _trace_marble(col: int, color: int, bars: list[Bar], height: int) -> tuple[tuple[int, int, int], list[tuple]]:
    col = int(col)
    color = int(color)
    events: list[tuple] = []
    for idx, bar in enumerate(bars):
        if int(bar["c0"]) <= col <= int(bar["c1"]):
            before_col = col
            before_color = color
            if color == int(bar["gate"]):
                col = int(bar["gate_col"])
                color = int(bar["body"])
                events.append(("pass", int(idx), before_col, int(bar["gate_col"]), before_color, color))
            else:
                events.append(("block", int(idx), before_col, int(bar["gate_col"]), before_color, int(bar["gate"])))
                return (int(bar["row"]) - 1, int(bar["gate_col"]), int(color)), events
    events.append(("bottom", int(col), int(color)))
    return (int(height) - 1, int(col), int(color)), events


def _settle_all(g: np.ndarray, bg: int, marbles: list[Marble], bars: list[Bar]) -> np.ndarray:
    out = g.copy()
    for col, _color in marbles:
        out[0, int(col)] = int(bg)

    occupied: set[tuple[int, int]] = set()
    for col, color in marbles:
        r, c, final_color = _trace_marble(int(col), int(color), bars, g.shape[0])[0]
        assert (r, c) not in occupied, "marbles must have distinct final cells"
        assert int(out[r, c]) == int(bg), "final marble cell must be empty background"
        occupied.add((int(r), int(c)))
        out[int(r), int(c)] = int(final_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Drop top-row marbles through color-keyed gapped bars, recoloring each marble after every successful gate.

    Input structure:
    - The dominant background surrounds several nonadjacent single-pixel marbles
      in the top row; their colors are all distinct.
    - Below them are horizontal bars on distinct rows.  Each bar is one
      contiguous run: all cells share a body color except for one interior gate
      pixel of a different color.

    Rule:
    - Erase the top-row marbles.  Process each marble independently from top to
      bottom in its current column.
    - When the marble's column intersects a lower bar span, slide it along the
      bar top to that bar's gate column.
    - If the marble's current color equals the gate pixel color, it passes
      through the opening, moves to the gate column, and immediately becomes
      the bar body color.  Otherwise it stops on the cell directly above the
      gate, keeping its current color.
    - A marble that never meets another bar falls to the bottom row.  All bars
      and gate pixels remain unchanged.

    Key insight:
    - The visible key color is mutable: passing one gate rewrites the marble
      color to the bar body, so later gates must be tested against the current
      color rather than the initial top-row color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 4 <= min(g.shape) and max(g.shape) <= 30

    bg = _unique_dominant_bg(g)
    marbles = _find_top_marbles(g, bg)
    bars = _extract_bars(g, bg)
    return _settle_all(g, bg, marbles, bars)


# ---------------- generator ----------------


def _nonadjacent(cols: list[int]) -> bool:
    ordered = sorted(map(int, cols))
    return all(b - a > 1 for a, b in zip(ordered, ordered[1:]))


def _span_options(
    width: int,
    required: list[int],
    gate: int,
    forbidden: set[int] | None = None,
    *,
    max_len: int = MAX_BAR_LEN,
) -> list[tuple[int, int]]:
    forbidden = set() if forbidden is None else set(map(int, forbidden))
    required = [int(x) for x in required]
    gate = int(gate)
    options: list[tuple[int, int]] = []
    hi_limit = int(width) - 2
    for c0 in range(1, hi_limit + 1):
        for c1 in range(c0 + MIN_BAR_LEN - 1, min(hi_limit, c0 + max_len - 1) + 1):
            if not (c0 < gate < c1):
                continue
            if not all(c0 <= x <= c1 for x in required):
                continue
            if any(c0 <= x <= c1 for x in forbidden):
                continue
            options.append((int(c0), int(c1)))
    random.shuffle(options)
    return options


def _pick_span(width: int, required: list[int], gate: int, forbidden: set[int] | None = None) -> tuple[int, int] | None:
    opts = _span_options(width, required, gate, forbidden)
    return opts[0] if opts else None


def _sample_rows(height: int, count: int) -> list[int] | None:
    candidates = list(range(3, int(height) - 1))
    for _ in range(300):
        rows = sorted(random.sample(candidates, int(count)))
        if all(b - a >= ROW_GAP for a, b in zip(rows, rows[1:])):
            return [int(r) for r in rows]
    return None


def _sample_core_geometry(width: int) -> dict | None:
    inner = list(range(1, int(width) - 1))
    gate_pool = list(range(2, int(width) - 2))

    for _ in range(800):
        bypass_col = random.choice([0, int(width) - 1])
        top_pool = [c for c in inner if abs(c - bypass_col) > 1]
        chain_col, block_col, pass_col = random.sample(top_pool, 3)
        if not _nonadjacent([bypass_col, chain_col, block_col, pass_col]):
            continue

        gate0, gate1, gate2 = random.sample(gate_pool, 3)
        if gate0 in (chain_col, block_col, pass_col):
            continue
        if gate1 == pass_col or gate2 == pass_col:
            continue

        span0 = _pick_span(width, [chain_col, block_col, gate0], gate0, {pass_col})
        if span0 is None:
            continue
        span1 = _pick_span(width, [gate0, gate1], gate1, {pass_col})
        if span1 is None:
            continue
        span2 = _pick_span(width, [gate1, pass_col, gate2], gate2)
        if span2 is None:
            continue

        return {
            "bypass_col": int(bypass_col),
            "chain_col": int(chain_col),
            "block_col": int(block_col),
            "pass_col": int(pass_col),
            "gates": [int(gate0), int(gate1), int(gate2)],
            "spans": [span0, span1, span2],
        }
    return None


def _sample_extra_span(width: int, avoid_col: int) -> tuple[int, int, int] | None:
    for _ in range(200):
        length = random.randint(MIN_BAR_LEN, min(MAX_BAR_LEN, int(width) - 2))
        c0 = random.randint(1, int(width) - length - 1)
        c1 = c0 + length - 1
        if c0 <= int(avoid_col) <= c1:
            continue
        if c1 - c0 + 1 < 3:
            continue
        gate = random.randint(c0 + 1, c1 - 1)
        return int(c0), int(c1), int(gate)
    return None


def _paint_bar(grid: np.ndarray, row: int, c0: int, c1: int, gate_col: int, body: int, gate: int) -> None:
    grid[int(row), int(c0) : int(c1) + 1] = int(body)
    grid[int(row), int(gate_col)] = int(gate)


def _bar_from_parts(row: int, c0: int, c1: int, gate_col: int, body: int, gate: int) -> Bar:
    return {
        "row": int(row),
        "c0": int(c0),
        "c1": int(c1),
        "gate_col": int(gate_col),
        "body": int(body),
        "gate": int(gate),
    }


def _semantic_accepts(
    grid: np.ndarray,
    expected: np.ndarray,
    bg: int,
    meta: dict,
    core_bars: list[Bar],
    top_colors: dict[str, int],
) -> bool:
    try:
        bars = _extract_bars(grid, int(bg))
        marbles = dict(_find_top_marbles(grid, int(bg)))
        produced = main(grid)
    except Exception:
        return False

    if not np.array_equal(produced, expected):
        return False
    if len(bars) < 3:
        return False
    for actual, planned in zip(bars[:3], core_bars):
        for key in ("row", "c0", "c1", "gate_col", "body", "gate"):
            if int(actual[key]) != int(planned[key]):
                return False

    chain_final, chain_events = _trace_marble(
        meta["chain_col"], top_colors["chain"], bars, grid.shape[0]
    )
    block_final, block_events = _trace_marble(
        meta["block_col"], top_colors["block"], bars, grid.shape[0]
    )
    pass_final, pass_events = _trace_marble(
        meta["pass_col"], top_colors["pass"], bars, grid.shape[0]
    )
    bypass_final, bypass_events = _trace_marble(
        meta["bypass_col"], top_colors["bypass"], bars, grid.shape[0]
    )

    if [e[0] for e in chain_events[:3]] != ["pass", "pass", "block"]:
        return False
    if chain_events[1][4] == chain_events[1][5]:
        return False
    if top_colors["chain"] == int(core_bars[1]["gate"]):
        return False
    if int(core_bars[0]["body"]) != int(core_bars[1]["gate"]):
        return False
    if chain_final != (int(core_bars[2]["row"]) - 1, int(core_bars[2]["gate_col"]), int(core_bars[1]["body"])):
        return False
    if [e[0] for e in block_events[:1]] != ["block"]:
        return False
    if block_final != (int(core_bars[0]["row"]) - 1, int(core_bars[0]["gate_col"]), top_colors["block"]):
        return False
    if not (pass_events[0][0] == "pass" and pass_events[0][1] == 2):
        return False
    if pass_final != (grid.shape[0] - 1, int(core_bars[2]["gate_col"]), int(core_bars[2]["body"])):
        return False
    if bypass_events != [("bottom", int(meta["bypass_col"]), int(top_colors["bypass"]))]:
        return False
    if len({chain_final[:2], block_final[:2], pass_final[:2], bypass_final[:2]}) != 4:
        return False

    for col, color in marbles.items():
        if int(grid[0, col]) != int(color):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 29)
    width = random.randint(16, 29)
    bar_count = random.randint(3, 5)
    rows = _sample_rows(height, bar_count)
    if rows is None:
        return None

    bg = random.choice(ARC_COLORS)
    palette = [c for c in ARC_COLORS if c != bg]
    if len(palette) < 7:
        return None
    chain_color, block_color, pass_color, bypass_color, body0, body1, body2 = random.sample(palette, 7)
    top_colors = {
        "chain": int(chain_color),
        "block": int(block_color),
        "pass": int(pass_color),
        "bypass": int(bypass_color),
    }

    geom = _sample_core_geometry(width)
    if geom is None:
        return None

    grid = np.full((height, width), int(bg), dtype=int)
    grid[0, int(geom["chain_col"])] = int(chain_color)
    grid[0, int(geom["block_col"])] = int(block_color)
    grid[0, int(geom["pass_col"])] = int(pass_color)
    grid[0, int(geom["bypass_col"])] = int(bypass_color)

    gate0, gate1, gate2 = geom["gates"]
    span0, span1, span2 = geom["spans"]
    core_specs = [
        (rows[0], span0[0], span0[1], gate0, body0, chain_color),
        (rows[1], span1[0], span1[1], gate1, body1, body0),
        (rows[2], span2[0], span2[1], gate2, body2, pass_color),
    ]
    core_bars = [_bar_from_parts(*spec) for spec in core_specs]
    for spec in core_specs:
        _paint_bar(grid, *spec)

    for row in rows[3:]:
        extra = _sample_extra_span(width, avoid_col=gate2)
        if extra is None:
            return None
        c0, c1, gate = extra
        body, gate_color = random.sample(palette, 2)
        _paint_bar(grid, row, c0, c1, gate, body, gate_color)

    if int(_get_bg(grid)) != int(bg):
        return None

    try:
        marbles = _find_top_marbles(grid, int(bg))
        bars = _extract_bars(grid, int(bg))
        expected = _settle_all(grid, int(bg), marbles, bars)
    except Exception:
        return None

    if _semantic_accepts(grid, expected, int(bg), geom, core_bars, top_colors):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid color-key gate input after {GEN_ATTEMPTS} attempts")
