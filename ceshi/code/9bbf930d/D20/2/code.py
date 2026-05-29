# Variant of 9bbf930d
# Dimension: D20 — L/U/S形走廊的拓扑变化 (path)
# Change: 走廊支持0-3次转弯：直通（0转）、L形（1转）、U形（2转，同色通道由垂直条连接）。... → 螺旋走廊（3次以上转弯）

from __future__ import annotations

import random

import numpy as np


# Design analysis for this D20 variant
# ------------------------------------
# 1. Concrete behavior:
#    - The original solver already supports repeated 90-degree turns; the real limit was the
#      generator, which only rendered straight / L / U paths.
#    - This variant keeps the striped "bar rows + channel rows" visual language, but upgrades
#      each real path into a 3-turn inward snake: the particle enters on the upper row of a
#      same-color 2-channel band, turns down at a right-side connector, comes back left on the
#      lower row, turns up through a second connector, then stops at an internal dead end.
#    - The pixel mechanism stays local and visible:
#      * a forward blocker is a colored cell placed in the current channel row;
#      * a valid vertical connector is a single BG gap in the middle bar row;
#      * the forbidden branch at the re-entry corner is suppressed by recoloring one flank bar,
#        not by blocking the traversed cell itself.
#    - Small trace example:
#      original U-turn: top row -> down connector -> lower row -> stop/edge.
#      variant snake: top row -> down at x1 -> lower row left -> up at x2 -> internal stop.
#
# 2. Complexity calibration:
#    - Each moved particle now has exactly 3 turns, so the topology is still outside the original
#      0-2 turn regime while remaining acyclic and easy to validate.
#    - The band width is still only 2 channels, which keeps the pattern legible and avoids
#      over-parameterizing the topology.
#
# 3. Main() assumption audit:
#    - _trace() assumed turns may repeat; that remains valid.
#    - The old hardcoded 500-step cap was only harmless because original paths were short.
#      In this variant, a longer snake can cross much more of the grid, so the cap becomes
#      shape-dependent. The fix is to derive it from grid area.
#    - Entrance detection still assumes the first real corridor starts at col 2 or 3. The new
#      generator preserves that for active rows and intentionally breaks it for partner rows.
#
# 4. Placement legality:
#    - Real bands consume 3 even bar rows / 2 odd channel rows and are spaced every 3 channels,
#      so paths never overlap vertically.
#    - Connector columns are chosen as an inward alternating high/low sequence, which guarantees
#      each horizontal run has visible length and every blocker/gap stays in-bounds.
#
# 5. Visual observability:
#    - The new behavior is inferable by eye: moved particles now bounce multiple times inside the
#      same-color ribbon instead of making at most one U-shaped detour.
#    - The rule never depends on color ids; only visible same-color vs mismatched flanks matter.
#
# 6. Visual cleanliness:
#    - Each active path lives in its own band, fake rows remain separate, and crossing logic is
#      avoided entirely. This keeps the grid readable despite the longer trajectories.
#
# 7. Generator/solver contract:
#    - main() still asserts away ambiguous turns.
#    - generate_input() guarantees that every vertical connector has exactly one legal exit, and
#      that the lower partner row in the band has a broken entrance so only one particle uses the
#      long snake.

BG = 7
PARTICLE = 6
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
    """True if (r,c) is a traversable cell flanked by same-color bars."""
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


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace a particle through any number of unambiguous 90-degree turns."""
    h, w = grid.shape
    direction = "right"
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
    return r, c


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles along same-color corridors, including long snake paths."""
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for r in range(1, h - 1):
        if int(g[r, 0]) != PARTICLE:
            continue

        entrance_col = None
        for c in range(2, w):
            if int(g[r, c]) != BG:
                continue
            above, below = int(g[r - 1, c]), int(g[r + 1, c])
            if above == below and above not in {PARTICLE, BG}:
                entrance_col = c
                break
        if entrance_col is None:
            continue

        bar_above = next(
            (c for c in range(w) if int(g[r - 1, c]) == int(g[r - 1, entrance_col])),
            w,
        )
        bar_below = next(
            (c for c in range(w) if int(g[r + 1, c]) == int(g[r + 1, entrance_col])),
            w,
        )
        if bar_above != bar_below or bar_above > 3:
            continue

        end_r, end_c = _trace(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (16, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4
_FAKE_DECOR_RANGE = (0, 2)
_MAX_RETRIES = 400
_MIN_DISTINCT_COLORS = 6


def _pick_band_starts(n_channels: int) -> list[int]:
    """Pick non-overlapping 2-channel bands, separated by one fake channel."""
    feasible = []
    for offset in range(3):
        starts = [s for s in range(offset, n_channels - 1, 3) if s <= n_channels - 2]
        if len(starts) >= 3:
            feasible.append(starts)
    if not feasible:
        return []
    starts = random.choice(feasible)
    if len(starts) == 4 and random.random() < 0.45:
        starts = starts[:-1]
    return starts


def _assign_bar_colors(n_bar_rows: int, bands: dict[int, int], extras: list[int]) -> list[int]:
    """Fill even-row bar colors, keeping non-band adjacent pairs mismatched."""
    colors: list[int | None] = [None] * n_bar_rows
    for start, color in bands.items():
        colors[start] = color
        colors[start + 1] = color
        colors[start + 2] = color

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


def _choose_snake_columns(bar_start: int, w: int, n_connectors: int) -> list[int] | None:
    """Choose inward alternating right/left connector columns."""
    left0 = bar_start + 4
    right0 = w - 3
    if left0 >= right0:
        return None

    for _ in range(60):
        left = left0
        right = right0
        cols = []
        ok = True
        for idx in range(n_connectors):
            if idx % 2 == 0:
                lo, hi = left + 1, right
            else:
                lo, hi = left, right - 1
            if lo > hi:
                ok = False
                break
            x = random.randint(lo, hi)
            cols.append(x)
            if idx % 2 == 0:
                right = x - 1
            else:
                left = x + 1
        if ok and all(abs(a - b) >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _build_snake_band(
    g: np.ndarray,
    bar_rows: list[int],
    start: int,
    color: int,
    breaker: int,
    bar_start: int,
) -> dict[str, int | list[int]] | None:
    """Render one acyclic 3-turn snake in a 2-channel same-color band."""
    _, w = g.shape
    top = bar_rows[start]
    mid = bar_rows[start + 1]
    bot = bar_rows[start + 2]
    row_top = top + 1
    row_bot = mid + 1

    cols = _choose_snake_columns(bar_start, w, 2)
    if cols is None:
        return None

    # Break the lower row's entrance without disturbing the real upper entrance.
    g[bot, bar_start : min(w, bar_start + 2)] = breaker

    x1, x2 = cols
    if x1 + 1 >= w or x2 - 1 <= 1 or x2 + 1 >= w:
        return None

    g[mid, x1] = BG
    g[row_top, x1 + 1] = color              # forward blocker: top row turns down
    g[bot, x1 + 1] = breaker                # suppress right exit on lower row

    g[mid, x2] = BG
    g[row_bot, x2 - 1] = color              # forward blocker: lower row turns up
    g[top, x2 - 1] = breaker                # suppress left exit on upper row
    g[top, x2 + 1] = breaker                # suppress right exit after re-entry; forces stop

    return {
        "entry_row": row_top,
        "blocked_row": row_bot,
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
    h, w = g.shape
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
    h = random.choice(_HEIGHT_OPTIONS)
    w = random.randint(*_WIDTH_RANGE)
    bar_start = random.choice(_BAR_START_OPTIONS)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((h, w), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    bar_rows = list(range(0, h, 2))
    n_channels = len(bar_rows) - 1
    starts = _pick_band_starts(n_channels)
    if len(starts) < 3:
        return None

    band_palette = random.sample(extras, k=min(len(starts), len(extras)))
    bands = {start: band_palette[i % len(band_palette)] for i, start in enumerate(starts)}
    bar_colors = _assign_bar_colors(len(bar_rows), bands, extras)

    for idx, br in enumerate(bar_rows):
        g[br, bar_start:w] = bar_colors[idx]

    channel_kind = ["fake"] * n_channels
    band_infos = []
    for start in starts:
        color = bands[start]
        breaker_choices = [c for c in extras if c != color]
        breaker = random.choice(breaker_choices)
        info = _build_snake_band(g, bar_rows, start, color, breaker, bar_start)
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

    moved_rows = [r for r in range(h) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE]
    if len(moved_rows) != len(starts):
        return None
    if any(int(out[int(info["blocked_row"]), 0]) != PARTICLE for info in band_infos):
        return None
    if any(int(info["turns"]) < 3 for info in band_infos):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
