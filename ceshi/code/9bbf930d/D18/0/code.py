# Variant of 9bbf930d
# Dimension: D18 — 装饰/干扰项的角色 (obstacle)
# Change: 装饰条是放在假走廊通道中的单个彩色像素，作为纯视觉干扰项，不影响粒子移动。... → 装饰条作为方向提示

from __future__ import annotations

import random

import numpy as np

"""
Design analysis for this D18 variant
====================================

a) Concrete behavior
--------------------
- The striped scaffold stays the same: even rows are colored bars, odd rows are
  horizontal channels, col 0 is filled with particles, and col 1 is background.
- Real source channels are still identified by same-color bar pairs before the
  turn. The change happens only at the turning site.
- At selected blockers, both the upward and downward turn cells are made locally
  valid. They are given different local corridor colors by recoloring the tiny
  left/right flanks around the two turn-entry gaps.
- A single colored decoration pixel is placed in the nearest fake channel on the
  intended side, in the same blocker column. Its color matches the intended turn
  corridor color. The opposite side has no matching decoration.
- Solver change: when a blocker yields two perpendicular options, it no longer
  treats this as an unrecoverable ambiguity. It checks the adjacent fake-channel
  decoration on each side and chooses the option whose corridor color is echoed
  by the hint pixel.

Small example at one blocker:
- Old logic: particle moves right to a blocker; both up and down turn cells are
  valid -> assertion / ambiguity.
- New logic: the fake channel two rows above contains a single pixel colored the
  same as the upper turn-entry flanks -> choose the upper turn and continue.
- Difference: the decorative single pixel is now semantically active because it
  resolves a turn that local geometry alone leaves ambiguous.

b) Complexity calibration
-------------------------
- The rule uses only two hint sides (up/down) and a single visible cue type
  (one decoration pixel), so the branching factor stays small.
- Every generated instance contains several repeated examples of the same cue
  pattern, making the new behavior visible from a few samples.
- The generator does not add extra symbolic encodings or hidden ordering by
  color id; only visible local color matching matters.

c) Old solver assumptions that break here
-----------------------------------------
- `_trace(...)` originally assumed that at most one perpendicular turn option can
  be valid at a blocker. That assumption is intentionally violated here.
- The original decoration pixels were harmless because the solver ignored them.
  In this variant, ignoring them makes the turn rule incomplete.
- `main(...)` can keep the original source-channel detection because the channel
  entrance is still a same-color pair near the left edge; only the later turn
  site is changed.

d) Placement legality
---------------------
- Source channels are spaced with one fake channel between them so each blocker
  always has a legal fake-channel host row for the hint on either side.
- Blocker columns are chosen from alternating columns with spacing >= 2, keeping
  vertical paths and local recolor patches from overlapping.
- Local recoloring is restricted to the 2-cell flanks around the turn entries,
  so the source corridor remains intact on the left side and the hint row stays
  fake at the hint column.

e) Visual observability
-----------------------
- The new cue is visible directly next to the decision point: a lone colored
  pixel appears in a nearby fake corridor above or below the blocker column.
- Multiple particles in the same grid use the same mechanism, so a human can
  notice that the decoration color predicts the chosen turn side.

f) Clean visualization
----------------------
- Hints are only one pixel per moving source row and always live in fake rows,
  so they do not stack on the particle path itself.
- Each ambiguous turn uses a single blocker column and two small local flank
  patches; this keeps the layout readable instead of densely cross-linked.

g) Color inference
------------------
- No invisible color ordering is used. The only color reasoning is direct local
  equality: hint pixel color == intended turn-entry corridor color.

h) Generator guarantee for new solver assumptions
-------------------------------------------------
- Every ambiguous turn generated here has exactly two local options.
- Exactly one of those two options has a matching hint in the adjacent fake row.
- The chosen side is extended to the edge; the unchosen side is left as a short
  stub, so the hint-controlled choice changes the final particle position.
"""


BG = 7
PARTICLE = 6
_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a traversable turn-entry cell for the given direction."""
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


def _corridor_color(grid: np.ndarray, r: int, c: int, direction: str) -> int | None:
    """Color of the bar pair that makes this turn-entry cell valid."""
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        return None
    if direction in {"left", "right"}:
        if r == 0 or r == h - 1:
            return None
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == w - 1:
            return None
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    if a == b and a not in {PARTICLE, BG}:
        return a
    return None


def _is_fake_channel_at(grid: np.ndarray, r: int, c: int) -> bool:
    """True if row r is locally a fake corridor at column c."""
    h, w = grid.shape
    if not (1 <= r < h - 1 and 0 <= c < w):
        return False
    above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
    return above not in {PARTICLE, BG} and below not in {PARTICLE, BG} and above != below


def _resolve_ambiguous_turn(
    grid: np.ndarray,
    r: int,
    blocked_c: int,
    direction: str,
    options: list[tuple[int, int, str]],
) -> tuple[int, int, str] | None:
    """Use a nearby fake-row hint pixel to choose among multiple turn options."""
    if direction not in {"left", "right"}:
        return None

    matched: list[tuple[int, int, str]] = []
    for tr, tc, turn_dir in options:
        color = _corridor_color(grid, tr, tc, turn_dir)
        if color is None:
            continue
        hint_r = r - 2 if turn_dir == "up" else r + 2
        if not _is_fake_channel_at(grid, hint_r, blocked_c):
            continue
        if int(grid[hint_r, blocked_c]) == color:
            matched.append((tr, tc, turn_dir))

    assert len(matched) <= 1, "multiple matching hints for one blocker"
    return matched[0] if matched else None


def _trace_with_stats(grid: np.ndarray, r: int, c: int) -> tuple[int, int, int]:
    """Trace one particle and count how many times a hint resolves ambiguity."""
    h, w = grid.shape
    direction = "right"
    hint_turns = 0
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            continue

        options: list[tuple[int, int, str]] = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < h and 0 <= tc < w and _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))

        if len(options) > 1:
            chosen = _resolve_ambiguous_turn(grid, r, nc, direction, options)
            assert chosen is not None, "ambiguous turn without a unique hint"
            hint_turns += 1
            options = [chosen]

        if not options:
            break
        r, c, direction = options[0]
    return r, c, hint_turns


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace one particle from (r, c) with hint-resolved ambiguous turns."""
    end_r, end_c, _ = _trace_with_stats(grid, r, c)
    return end_r, end_c


def main(grid: np.ndarray) -> np.ndarray:
    """Move particles; decoration pixels now resolve ambiguous up/down turns."""
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
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4
_MAX_RETRIES = 300
_MIN_DISTINCT_COLORS = 6


def _source_indices(n_channels: int) -> list[int]:
    return list(range(1, n_channels - 1, 2))


def _assign_bar_colors(bar_rows: list[int], source_set: set[int], extras: list[int]) -> dict[int, int]:
    """Real source channels keep their top color; all other channels are fake."""
    bar_color = {bar_rows[0]: extras[0]}
    cursor = 1
    for i in range(len(bar_rows) - 1):
        top = bar_rows[i]
        bot = bar_rows[i + 1]
        if i in source_set:
            bar_color[bot] = bar_color[top]
            continue
        while extras[cursor % len(extras)] == bar_color[top]:
            cursor += 1
        bar_color[bot] = extras[cursor % len(extras)]
        cursor += 1
    return bar_color


def _blocker_columns(bar_start: int, w: int, n_needed: int) -> list[int] | None:
    """Pick spaced blocker columns; path columns are blocker-1."""
    candidates = list(range(bar_start + 4, w - 1, 2))
    if len(candidates) < n_needed:
        return None
    return sorted(random.sample(candidates, n_needed))


def _extend_vertical_to_edge(
    g: np.ndarray,
    top: int,
    bot: int,
    path_c: int,
    direction: str,
) -> None:
    """Carve the chosen branch all the way to the relevant edge."""
    h, _ = g.shape
    if direction == "up":
        for r_br in range(0, top, 2):
            g[r_br, path_c] = BG
    else:
        for r_br in range(bot + 2, h, 2):
            g[r_br, path_c] = BG


def _pick_turn_colors(
    extras: list[int],
    upper_outer: int,
    lower_outer: int,
    source_color: int,
) -> tuple[int, int] | None:
    """Choose distinct local colors for the up/down turn entries."""
    up_candidates = [c for c in extras if c not in {upper_outer, source_color}]
    down_candidates = [c for c in extras if c not in {lower_outer, source_color}]
    random.shuffle(up_candidates)
    random.shuffle(down_candidates)
    for up_color in up_candidates:
        for down_color in down_candidates:
            if up_color != down_color:
                return up_color, down_color
    return None


def _build_ambiguous_turn(
    g: np.ndarray,
    bar_rows: list[int],
    bar_color: dict[int, int],
    chan_idx: int,
    blocker_c: int,
    chosen_dir: str,
    extras: list[int],
) -> bool:
    """
    Create an ambiguous up/down turn and place exactly one matching fake-row hint.

    The source row stays a real corridor on the left. At the blocker column:
    - both up/down entries become locally valid,
    - their local corridor colors differ,
    - only the chosen side gets a matching fake-row hint.
    """
    h, w = g.shape
    top, bot = bar_rows[chan_idx], bar_rows[chan_idx + 1]
    row = top + 1
    path_c = blocker_c - 1
    if not (2 <= path_c < w - 1):
        return False

    upper_outer = int(g[top - 2, blocker_c])
    lower_outer = int(g[bot + 2, blocker_c])
    source_color = bar_color[top]
    picked = _pick_turn_colors(extras, upper_outer, lower_outer, source_color)
    if picked is None:
        return False
    up_color, down_color = picked

    g[row, blocker_c] = random.choice(extras)

    g[top, path_c] = BG
    g[bot, path_c] = BG
    g[top, path_c - 1] = up_color
    g[top, blocker_c] = up_color
    g[bot, path_c - 1] = down_color
    g[bot, blocker_c] = down_color

    _extend_vertical_to_edge(g, top, bot, path_c, chosen_dir)

    if chosen_dir == "up":
        hint_r = row - 2
        hint_color = up_color
    else:
        hint_r = row + 2
        hint_color = down_color

    if not _is_fake_channel_at(g, hint_r, blocker_c):
        return False
    g[hint_r, blocker_c] = hint_color
    return True


def _count_hint_resolved_sites(grid: np.ndarray) -> int:
    """Internal validation: count hint-based turn resolutions on source paths."""
    g = np.array(grid, dtype=int)
    h, w = g.shape
    count = 0
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
        _, _, hint_turns = _trace_with_stats(g, r, 0)
        count += hint_turns
    return count


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(_HEIGHT_OPTIONS)
    w = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((h, w), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    bar_rows = list(range(0, h, 2))
    n_channels = len(bar_rows) - 1
    sources = _source_indices(n_channels)
    if not (3 <= len(sources) <= 6):
        return None
    source_set = set(sources)

    bar_color = _assign_bar_colors(bar_rows, source_set, extras)
    if len(set(bar_color.values())) < 4:
        return None

    bar_start = random.choice(_BAR_START_OPTIONS)
    for br in bar_rows:
        g[br, bar_start:w] = bar_color[br]

    blocker_cols = _blocker_columns(bar_start, w, len(sources))
    if blocker_cols is None:
        return None

    random_dirs = [random.choice(["up", "down"]) for _ in sources]
    for chan_idx, blocker_c, chosen_dir in zip(sources, blocker_cols, random_dirs):
        ok = _build_ambiguous_turn(g, bar_rows, bar_color, chan_idx, blocker_c, chosen_dir, extras)
        if not ok:
            return None

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
    except Exception:
        return None

    n_moved = sum(1 for r in range(h) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE)
    if n_moved != len(sources):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    if _count_hint_resolved_sites(g) < len(sources):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
