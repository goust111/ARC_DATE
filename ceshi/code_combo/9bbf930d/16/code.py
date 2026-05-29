# Combo variant of 9bbf930d (index 16)
# combo_spec: D4/2+D20/2
# Name: 仅转弯处涂色+螺旋走廊

from __future__ import annotations

import random

import numpy as np


# Design analysis for the combo variant
# a) Concretization and behavior trace
#    The base task already has the right local motion rule: go straight through background, and
#    when a colored blocker is directly ahead, turn into the unique perpendicular same-color
#    corridor. The combo changes two parts together.
#    - Output side (D4/2): do not paint straight motion at all; only the turn-entry cells and the
#      final resting cell become magenta.
#    - Input side (D20/2): instead of at most one L/U detour, each real corridor is a 2-channel
#      same-color band with two inward connectors, which yields a stable 3-turn spiral-like zig-zag.
#    Concrete trace for one band:
#      upper row moving right -> blocker at x0+1 -> turn down through gap x0 -> forced left on the
#      lower row -> blocker at x1-1 -> turn up through gap x1 -> forced right on the upper row ->
#      stop because both horizontal exits at that re-entry point are sealed.
#    The output therefore shows a sparse dotted inward snake: magenta only at the corner cells
#    plus the endpoint, never along the straight runs between them.
# b) Complexity calibration
#    A plain single-turn L/U detour would be too close to the base task. The generator therefore
#    makes every active band execute exactly three turns. That still satisfies the "3 turns or more"
#    requirement from the combo description while keeping the geometry stable and readable.
# c) Hidden assumptions in main()
#    - _trace() in the base code already tolerates repeated turns; that still holds, but the
#      fixed 500-step cap is replaced by an area-derived cap because longer snakes would make the
#      old constant shape-dependent.
#    - The entrance scan assumes the first real corridor begins at col 2 or 3 and that fake /
#      partner rows can be rejected by a local mismatch near the entrance. The generator keeps
#      that contract by recoloring the blocked partner row's lower bar at the bar start.
#    - Marking turn cells is safe only if tracing happens on the untouched input grid. main()
#      therefore computes traces from g and writes to a separate out copy.
#    - The ambiguity assert still relies on the generator suppressing the unwanted horizontal
#      exit after every connector. The builder explicitly closes only the flank needed to force
#      the next turn and closes both flanks at the final re-entry point.
# d) Placement legality
#    Each real path occupies a 3-bar-row / 2-channel-row band. Bands are spaced every three
#    channels, so no two active bands overlap vertically. Connector columns are chosen by a
#    right-then-left inward process, keeping every blocker, gap, and breaker in-bounds.
# e) Visual observability
#    From a few examples a human can see that moved particles do not leave full trails, only
#    corner dots. Those dots are not random: they line up with the zig-zag ribbon of same-color
#    corridors in the input. The rule depends only on visibly matching flank colors, not on hidden
#    numeric color identities.
# f) Visual cleanliness
#    The combo keeps the original striped layout and uses three isolated real bands. Fake rows only
#    get sparse decorative blockers. Because the output paints only sparse turn cells instead of
#    whole paths, the grid stays readable.
# g) Change inferability
#    No new palette mapping is introduced. The only color write is the existing magenta particle
#    color, reused for turn marks and final endpoints. Repeated same-structure bands in a single
#    grid give the solver multiple examples that "same-color ribbon => hidden multi-turn path =>
#    only the corners are marked".

BG = 7
PARTICLE = 6
TURN_MARK = PARTICLE
_DELTAS = {
    "right": (0, 1),
    "left": (0, -1),
    "up": (-1, 0),
    "down": (1, 0),
}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is traversable and flanked by same-color bars."""
    h, w = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == h - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == w - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and a not in {PARTICLE, BG}


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the left-side corridor entrance for row r, if it exists at col 2 or 3."""
    h, w = grid.shape
    if not (0 < r < h - 1) or int(grid[r, 0]) != PARTICLE:
        return None
    for c in range(2, min(w, 4)):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in {PARTICLE, BG}:
            return c
    return None


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    """Trace one particle and return the final cell plus all turn-entry cells."""
    h, w = grid.shape
    direction = "right"
    turns: list[tuple[int, int]] = []
    max_steps = h * w * 4
    for _ in range(max_steps):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            continue

        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < h and 0 <= tc < w and _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
        turns.append((r, c))
    return r, c, tuple(turns)


def _collect_traces(grid: np.ndarray) -> list[dict[str, object]]:
    """Collect trace summaries for all valid particle rows in the input."""
    g = np.array(grid, dtype=int)
    h, _ = g.shape
    traces: list[dict[str, object]] = []
    for r in range(1, h - 1):
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c, turns = _trace(g, r, 0)
        traces.append(
            {
                "row": r,
                "end": (end_r, end_c),
                "turns": turns,
                "moved": (end_r, end_c) != (r, 0),
            }
        )
    return traces


def main(grid: np.ndarray) -> np.ndarray:
    """Move particles through multi-turn same-color corridors and mark only the corners.

    Input structure:
    - bg=7, particle=6. Column 0 is filled with particles, column 1 is background.
    - Even rows are colored bar rows; odd rows are corridor rows or empty rows.
    - Real paths are same-color 2-channel bands whose connector gaps create 3-turn zig-zag paths.

    Rule:
    - A valid particle row is one whose first corridor entrance is a background cell flanked by
      the same non-background color starting at column 2 or 3.
    - The particle moves straight through background cells. If a colored cell blocks the current
      direction, it turns 90 degrees into the unique perpendicular same-color corridor.
    - In the output, the start particle disappears, the endpoint is magenta, and only the cells
      entered on turns are additionally marked magenta. Straight segments stay unchanged.

    Key insight:
    - The path itself is hidden in the striped same-color ribbon. The output reveals only the
      sequence of corner cells, so solving requires reconstructing the inward multi-turn snake
      from the corridor geometry rather than copying a visible trail.
    """
    g = np.array(grid, dtype=int)
    h, _ = g.shape
    out = g.copy()

    for r in range(1, h - 1):
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c, turns = _trace(g, r, 0)
        if (end_r, end_c) == (r, 0):
            continue
        out[r, 0] = BG
        for tr, tc in turns:
            out[tr, tc] = TURN_MARK
        out[end_r, end_c] = PARTICLE

    return out


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (17, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4
_FAKE_DECOR_RANGE = (0, 2)
_MAX_RETRIES = 500
_MIN_DISTINCT_COLORS = 6
_MIN_MOVED = 3
_MIN_TURNS_PER_PATH = 3
_MIN_MAX_TURNS = 3


def _pick_layout(n_channels: int) -> list[int] | None:
    """Pick three separated 2-channel bands."""
    options: list[list[int]] = []
    if n_channels >= 8:
        options.append([0, 3, 6])
    if n_channels >= 9:
        options.append([1, 4, 7])
    if n_channels >= 10:
        options.append([2, 5, 8])
    return random.choice(options) if options else None


def _fill_bar_colors(n_bar_rows: int, reserved: dict[int, int], extras: list[int]) -> list[int]:
    """Fill bar-row colors while keeping adjacent non-reserved rows mismatched."""
    colors: list[int | None] = [None] * n_bar_rows
    for idx, color in reserved.items():
        colors[idx] = color

    missing = set(extras) - {c for c in colors if c is not None}
    for i in range(n_bar_rows):
        if colors[i] is not None:
            continue
        forbidden = set()
        if i > 0 and colors[i - 1] is not None:
            forbidden.add(colors[i - 1])
        if i + 1 < n_bar_rows and colors[i + 1] is not None:
            forbidden.add(colors[i + 1])
        choices = [c for c in extras if c not in forbidden]
        if not choices:
            choices = extras[:]
        preferred = [c for c in choices if c in missing]
        chosen = random.choice(preferred or choices)
        colors[i] = chosen
        missing.discard(chosen)
    return [int(c) for c in colors]


def _choose_simple_columns(bar_start: int, w: int) -> tuple[int, int] | None:
    """Pick right/left connector columns for a stable 3-turn band."""
    if bar_start + 8 > w - 4:
        return None
    for _ in range(100):
        x0 = random.randint(bar_start + 8, w - 4)
        x1 = random.randint(bar_start + 4, x0 - 4)
        if x1 + 1 < x0 and x0 + 1 < w and x1 - 1 > 1:
            return x0, x1
    return None


def _build_simple_band(
    g: np.ndarray,
    bar_rows: list[int],
    start: int,
    color: int,
    breaker: int,
    bar_start: int,
) -> dict[str, object] | None:
    """Build one stable 3-turn band."""
    _, w = g.shape
    cols = _choose_simple_columns(bar_start, w)
    if cols is None:
        return None
    x0, x1 = cols

    br0, br1, br2 = bar_rows[start : start + 3]
    ch0, ch1 = br0 + 1, br1 + 1

    g[br2, bar_start:x1] = breaker
    g[br2, x1 : x0 + 1] = color

    g[br1, x0] = BG
    g[ch0, x0 + 1] = color
    g[br2, x0 + 1] = breaker

    g[br1, x1] = BG
    g[ch1, x1 - 1] = color
    g[br0, x1 - 1] = breaker
    g[br0, x1 + 1] = breaker

    return {
        "entry_row": ch0,
        "blocked_rows": (ch1,),
        "turns": 3,
        "connectors": cols,
    }


def _decorate_fake_rows(
    g: np.ndarray,
    channel_kind: list[str],
    bar_rows: list[int],
    bar_colors: list[int],
    bar_start: int,
) -> None:
    """Add sparse distractor blockers on fake rows."""
    _, w = g.shape
    for i, kind in enumerate(channel_kind):
        if kind != "fake":
            continue
        r = bar_rows[i] + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, w), k=min(n_decor, w - bar_start - 1))
        for c in cols:
            if g[r, c] == BG:
                g[r, c] = random.choice([bar_colors[i], bar_colors[i + 1]])


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the combo variant."""
    h = random.choice(_HEIGHT_OPTIONS)
    w = random.randint(*_WIDTH_RANGE)
    bar_start = random.choice(_BAR_START_OPTIONS)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((h, w), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    bar_rows = list(range(0, h, 2))
    n_channels = len(bar_rows) - 1
    starts = _pick_layout(n_channels)
    if starts is None:
        return None

    band_colors = random.sample(extras, len(starts))

    reserved: dict[int, int] = {}
    for band_idx, start in enumerate(starts):
        for idx in range(start, start + 3):
            reserved[idx] = band_colors[band_idx]

    bar_colors = _fill_bar_colors(len(bar_rows), reserved, extras)
    for idx, br in enumerate(bar_rows):
        g[br, bar_start:w] = bar_colors[idx]

    channel_kind = ["fake"] * n_channels
    band_infos = []
    for band_idx, start in enumerate(starts):
        forbidden = {band_colors[band_idx]}
        if start > 0:
            forbidden.add(bar_colors[start - 1])
        if start + 3 < len(bar_rows):
            forbidden.add(bar_colors[start + 3])
        choices = [c for c in extras if c not in forbidden]
        breaker = random.choice(choices or [c for c in extras if c != band_colors[band_idx]])
        info = _build_simple_band(g, bar_rows, start, band_colors[band_idx], breaker, bar_start)
        if info is None:
            return None
        channel_kind[start] = "real_entry"
        channel_kind[start + 1] = "real_blocked"
        band_infos.append(info)

    _decorate_fake_rows(g, channel_kind, bar_rows, bar_colors, bar_start)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
    except Exception:
        return None

    traces = [trace for trace in _collect_traces(g) if trace["moved"]]
    turn_counts = [len(trace["turns"]) for trace in traces]
    if len(traces) != _MIN_MOVED:
        return None
    if any(int(out[int(r), 0]) != PARTICLE for info in band_infos for r in info["blocked_rows"]):
        return None
    if not turn_counts or min(turn_counts) < _MIN_TURNS_PER_PATH:
        return None
    if max(turn_counts) < _MIN_MAX_TURNS:
        return None
    if sorted(turn_counts) != [3, 3, 3]:
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    if np.array_equal(out, g):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
