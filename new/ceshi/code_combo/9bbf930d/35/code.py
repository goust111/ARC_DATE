# Combo variant of 9bbf930d (index 35)
# combo_spec: D18/0+D14/2
# Name: 装饰条作方向提示+控制像素标记可转弯位置

from __future__ import annotations

import random

import numpy as np

"""
Design analysis for combo variant D18/0 + D14/2
================================================

a) Concrete behavior
--------------------
- The striped scaffold stays recognizable: even rows are colored bar rows, odd
  rows are horizontal channels, column 0 holds the magenta particles, and
  column 1 is background.
- Each moving source channel is engineered so that, at one blocker column, both
  the upward and downward turn entries are locally valid. The two candidate
  turn entries use different local corridor colors.
- A single decoration pixel is placed in the fake channel two rows above or two
  rows below the blocker. Its color matches exactly one of the two candidate
  turn-corridor colors, so the decoration acts as a direction hint.
- A red control pixel is placed on the decision cell just before the blocker
  only on rows where the turn is allowed. If the control pixel is absent, the
  particle stops even though the same hint and the same geometric turn options
  are visible.
- Small example:
  old rule: the particle reaches a blocker and turns as soon as one geometric
  perpendicular corridor is valid.
  D18-only rule: if both up and down are valid, the hint color in the adjacent
  fake row chooses the correct side.
  fused rule: the particle first needs to stand on a red control pixel to be
  allowed to turn at all; once turning is allowed, the fake-row decoration
  chooses which of the two locally valid turn entrances to use.
- Difference: the decoration no longer merely decorates, and the control pixel
  no longer merely gates an obvious turn. Both must be read together at the
  same blocker.

b) Complexity calibration
-------------------------
- The combination still uses only two visible cue families:
  1. one red control pixel on the path centerline,
  2. one colored decoration pixel in a neighboring fake corridor.
- The generator repeats the same fused motif across 4 source rows in one grid,
  which is enough for a human to notice the pattern from a few examples.
- To avoid collapsing into a trivial one-mechanism task, every source row is
  made geometrically ambiguous, so the decoration is always needed, and at
  least one source row omits the control pixel, so the control gate is also
  visibly needed.

c) Old solver assumptions that break here
-----------------------------------------
- `_has_corridor()` can no longer treat only `{BG, PARTICLE}` as traversable,
  because the red control pixel must be walkable.
- The original `_trace()` assumed that "blocked + valid side corridor" implies
  "must turn". That is false now: without a control pixel on the current cell,
  the particle must stop.
- The original `_trace()` also assumed that at most one perpendicular option is
  valid. This combo intentionally violates that by making both up and down turn
  entries locally valid, then resolving the ambiguity via the decoration hint.
- `main()` originally scanned source entrances using only background cells. In
  the combo, a control pixel may sit on the source corridor, so entrance
  detection must treat control as a path cell as well.

d) Placement legality
---------------------
- Source channels are every other interior channel, so each source has one fake
  channel above and one fake channel below to host hints.
- Blocker columns are sampled from spaced columns. Each source gets its own
  blocker, so local recolor patches and hint pixels do not overlap.
- The control pixel is placed only on an existing path cell, never on a bar.
- Hint pixels are written only into fake channels. Their column is the blocker
  column itself, so the cue is spatially tied to the nearby turn site.

e) Visual observability
-----------------------
- A human sees rows with near-identical geometry but different outcomes:
  rows with red-at-decision turn, rows without red stop.
- Among the rows that do turn, both up and down outcomes appear, and the only
  local difference between them is which neighboring fake corridor carries the
  decoration color matching the turn-entry color.
- This makes the fusion discoverable from 2-4 training examples without hidden
  ordering by numeric color ids.

f) Clean visualization
----------------------
- The fake-row hint is only one pixel, and the red control pixel is only one
  pixel, so the grid does not become cluttered.
- The source rows, fake rows, and local turn flanks remain visually separated:
  red appears only on the path centerline, hint colors appear only in fake
  channels, and the colored bars remain the structural frame.

g) Color inference
------------------
- No invisible color ranking is used. The only color relation is visible local
  equality: hint pixel color equals the locally recolored flank color of the
  intended turn entrance.
- The palette is kept to exactly 6 colors: 3 corridor colors plus background,
  particle, and red control. That keeps the cue mapping compact and learnable.

h) Fusion metrics used by the generator
---------------------------------------
- `potential_hint_rows`: rows that would execute a hint-resolved turn if
  control gating were ignored. This must equal the number of source rows, so
  the hint mechanic is actually present everywhere.
- `executed_hint_rows`: rows that execute such a hint-resolved turn under the
  real combo rule. This must equal the number of active control pixels.
- `control_suppressed_rows`: rows whose final position differs when control
  gating is ignored. This must be at least 1, proving the red pixel matters.

i) New solver assumptions and how the generator guarantees them
---------------------------------------------------------------
- Assumption 1: whenever two perpendicular turn options exist, at most one
  matching hint pixel exists. The generator writes exactly one matching hint.
- Assumption 2: every actual turn under the combo rule happens from a control
  cell. The generator places control on all active source rows.
- Assumption 3: the hint host row is genuinely fake at the blocker column. The
  generator uses an alternating real/fake scaffold and chooses turn-entry
  recolors so the adjacent bar colors differ there.
- Assumption 4: final particle positions are not themselves control pixels. The
  generator places control only on pre-turn decision cells, and active rows
  continue beyond that cell.
"""


BG = 7
PARTICLE = 6
CONTROL = 2

_PATH_VALUES = {BG, PARTICLE, CONTROL}
_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


def _is_path(value: int) -> bool:
    return int(value) in _PATH_VALUES


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is traversable and flanked by same-color bars for that direction."""
    h, w = grid.shape
    if not _is_path(int(grid[r, c])):
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == h - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == w - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and a not in _PATH_VALUES


def _corridor_color(grid: np.ndarray, r: int, c: int, direction: str) -> int | None:
    """Color of the bar pair that makes (r,c) a valid corridor cell."""
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
    if a == b and a not in _PATH_VALUES:
        return a
    return None


def _is_fake_channel_at(grid: np.ndarray, r: int, c: int) -> bool:
    """True if row r is locally a fake corridor at column c."""
    h, w = grid.shape
    if not (1 <= r < h - 1 and 0 <= c < w):
        return False
    above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
    return above not in _PATH_VALUES and below not in _PATH_VALUES and above != below


def _resolve_ambiguous_turn(
    grid: np.ndarray,
    r: int,
    blocked_c: int,
    direction: str,
    options: list[tuple[int, int, str]],
) -> tuple[int, int, str] | None:
    """Use the neighboring fake-row decoration to choose one turn direction."""
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


def _trace_state(
    grid: np.ndarray,
    r: int,
    c: int,
    allow_unmarked_turns: bool = False,
) -> tuple[int, int, int, int]:
    """Trace one particle and return `(final_r, final_c, turns, hint_turns)`."""
    h, w = grid.shape
    direction = "right"
    turns = 0
    hint_turns = 0

    for _ in range(h * w * 4):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            break
        if _is_path(int(grid[nr, nc])):
            r, c = nr, nc
            continue

        if (not allow_unmarked_turns) and int(grid[r, c]) != CONTROL:
            break

        options: list[tuple[int, int, str]] = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < h and 0 <= tc < w and _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))

        if len(options) > 1:
            chosen = _resolve_ambiguous_turn(grid, r, nc, direction, options)
            assert chosen is not None, "ambiguous turn without a unique hint"
            options = [chosen]
            hint_turns += 1

        if not options:
            break
        r, c, direction = options[0]
        turns += 1

    return r, c, turns, hint_turns


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    end_r, end_c, _, _ = _trace_state(grid, r, c, allow_unmarked_turns=False)
    return end_r, end_c


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    h, w = grid.shape
    if r <= 0 or r >= h - 1:
        return None
    for c in range(2, w):
        if not _is_path(int(grid[r, c])):
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in _PATH_VALUES:
            return c
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles using control-gated, hint-resolved corridor turns.

    Input structure:
    - bg=7, particle=6, control=2. Column 0 holds particles; column 1 is background.
    - Even rows are colored bars. Odd rows are horizontal channels or fake channels.
    - Fake channels may contain single decoration pixels whose color matches one nearby
      true turn entrance.

    Rule:
    - A particle moves straight through background and red control cells.
    - When the next cell is a colored blocker, the particle may try to turn only if its
      current cell is red.
    - If both perpendicular turn entries are locally valid, compare each entry's local
      corridor color with the decoration pixel in the adjacent fake channel on that side.
      Turn into the uniquely matching side. If there is no control or no unique match,
      stop.

    Key insight:
    - The red pixel answers "may I turn here?" and the fake-row decoration answers
      "which side is the real turn entrance?" Both cues are needed at the same blocker.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for r in range(1, h - 1):
        if int(g[r, 0]) != PARTICLE:
            continue

        entrance_col = _find_entrance_col(g, r)
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

_HEIGHT_OPTIONS = [19, 21]
_WIDTH_RANGE = (18, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 3
_FAKE_EXTRA_DECOR_RANGE = (0, 1)
_MAX_RETRIES = 400


def _source_indices(n_channels: int) -> list[int]:
    """Use every other interior channel as a real source row."""
    return list(range(1, n_channels - 1, 2))


def _bar_row_colors(bar_rows: list[int], extras: list[int], n_sources: int) -> dict[int, int]:
    """Create fake-leading, real-middle, fake-trailing bar-row colors explicitly."""
    sequence = [extras[0]]
    for j in range(n_sources):
        source_color = extras[(j + 1) % len(extras)]
        sequence.extend([source_color, source_color])
    tail_cursor = n_sources + 1
    while len(sequence) < len(bar_rows):
        color = extras[tail_cursor % len(extras)]
        if color == sequence[-1]:
            tail_cursor += 1
            color = extras[tail_cursor % len(extras)]
        sequence.append(color)
        tail_cursor += 1
    return {br: sequence[i] for i, br in enumerate(bar_rows)}


def _blocker_columns(bar_start: int, w: int, n_needed: int) -> list[int] | None:
    """Pick spaced blocker columns; the decision cell is one column to the left."""
    candidates = list(range(bar_start + 4, w - 1, 3))
    if len(candidates) < n_needed:
        return None
    if len(candidates) == n_needed:
        return candidates
    start = random.randint(0, len(candidates) - n_needed)
    return candidates[start : start + n_needed]


def _extend_vertical_to_edge(
    g: np.ndarray,
    top: int,
    bot: int,
    path_c: int,
    direction: str,
) -> None:
    """Carve the chosen vertical branch all the way to the relevant edge."""
    h, _ = g.shape
    if direction == "up":
        for r_br in range(0, top, 2):
            g[r_br, path_c] = BG
    else:
        for r_br in range(bot + 2, h, 2):
            g[r_br, path_c] = BG


def _build_ambiguous_turn(
    g: np.ndarray,
    bar_rows: list[int],
    bar_color: dict[int, int],
    chan_idx: int,
    blocker_c: int,
    chosen_dir: str,
    add_control: bool,
) -> None:
    """Make both up/down turn entries locally valid; hint selects the chosen side."""
    top, bot = bar_rows[chan_idx], bar_rows[chan_idx + 1]
    row = top + 1
    path_c = blocker_c - 1
    source_color = bar_color[top]
    upper_outer = int(g[top - 2, blocker_c])
    lower_outer = int(g[bot + 2, blocker_c])

    assert upper_outer != source_color and lower_outer != source_color
    assert upper_outer != lower_outer

    up_color = lower_outer
    down_color = upper_outer

    g[row, blocker_c] = source_color

    g[top, path_c] = BG
    g[top, path_c - 1] = up_color
    g[top, blocker_c] = up_color

    g[bot, path_c] = BG
    g[bot, path_c - 1] = down_color
    g[bot, blocker_c] = down_color

    _extend_vertical_to_edge(g, top, bot, path_c, chosen_dir)

    if add_control:
        assert int(g[row, path_c]) == BG
        g[row, path_c] = CONTROL

    if chosen_dir == "up":
        hint_r = row - 2
        hint_color = up_color
    else:
        hint_r = row + 2
        hint_color = down_color

    assert _is_fake_channel_at(g, hint_r, blocker_c), "hint host must stay fake"
    g[hint_r, blocker_c] = hint_color


def _add_extra_fake_decor(
    g: np.ndarray,
    bar_rows: list[int],
    source_set: set[int],
    bar_color: dict[int, int],
    bar_start: int,
    blocker_cols: set[int],
) -> None:
    """Add a few non-critical decoration pixels to unused fake channels."""
    h, w = g.shape
    for i in range(len(bar_rows) - 1):
        if i in source_set:
            continue
        r = bar_rows[i] + 1
        n_decor = random.randint(*_FAKE_EXTRA_DECOR_RANGE)
        if n_decor == 0:
            continue
        pool = [c for c in range(bar_start + 1, w) if c not in blocker_cols]
        if not pool:
            continue
        cols = random.sample(pool, min(n_decor, len(pool)))
        for c in cols:
            if int(g[r, c]) != BG:
                continue
            if not _is_fake_channel_at(g, r, c):
                continue
            g[r, c] = random.choice([bar_color[bar_rows[i]], bar_color[bar_rows[i + 1]]])


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    """Measure whether both the hint and the control pixel are behaviorally active."""
    g = np.array(grid, dtype=int)
    h, _ = g.shape
    moved_rows = 0
    executed_hint_rows = 0
    potential_hint_rows = 0
    control_suppressed_rows = 0

    for r in range(1, h - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r) is None:
            continue

        end_r, end_c, _, hint_turns = _trace_state(g, r, 0, allow_unmarked_turns=False)
        free_r, free_c, _, free_hint_turns = _trace_state(g, r, 0, allow_unmarked_turns=True)

        if (end_r, end_c) != (r, 0):
            moved_rows += 1
        if hint_turns > 0:
            executed_hint_rows += 1
        if free_hint_turns > 0:
            potential_hint_rows += 1
        if (end_r, end_c) != (free_r, free_c):
            control_suppressed_rows += 1

    return {
        "control_pixels": int(np.count_nonzero(g == CONTROL)),
        "moved_rows": moved_rows,
        "executed_hint_rows": executed_hint_rows,
        "potential_hint_rows": potential_hint_rows,
        "control_suppressed_rows": control_suppressed_rows,
    }


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(_HEIGHT_OPTIONS)
    w = random.randint(*_WIDTH_RANGE)
    extras = random.sample(
        [c for c in range(10) if c not in (BG, PARTICLE, CONTROL)],
        _N_EXTRA_COLORS,
    )

    g = np.full((h, w), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    bar_rows = list(range(0, h, 2))
    n_channels = len(bar_rows) - 1
    sources = _source_indices(n_channels)
    if len(sources) < 4:
        return None
    source_set = set(sources)

    bar_color = _bar_row_colors(bar_rows, extras, len(sources))
    bar_start = random.choice(_BAR_START_OPTIONS)
    for br in bar_rows:
        g[br, bar_start:w] = bar_color[br]

    blocker_cols = _blocker_columns(bar_start, w, len(sources))
    if blocker_cols is None:
        return None

    chosen_dirs = [random.choice(["up", "down"]) for _ in sources]
    if len(set(chosen_dirs)) == 1:
        chosen_dirs[0] = "down" if chosen_dirs[0] == "up" else "up"

    blocked_row = random.choice(sources)
    for chan_idx, blocker_c, chosen_dir in zip(sources, blocker_cols, chosen_dirs):
        _build_ambiguous_turn(
            g,
            bar_rows,
            bar_color,
            chan_idx,
            blocker_c,
            chosen_dir,
            add_control=(chan_idx != blocked_row),
        )

    _add_extra_fake_decor(g, bar_rows, source_set, bar_color, bar_start, set(blocker_cols))

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
    except Exception:
        return None

    metrics = _variant_metrics(g)
    distinct = set(g.flatten().tolist())
    n_moved = sum(1 for r in range(h) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE)

    if distinct != {BG, PARTICLE, CONTROL, *extras}:
        return None
    if n_moved != len(sources):
        return None
    if metrics["moved_rows"] != len(sources):
        return None
    if metrics["executed_hint_rows"] != len(sources) - 1:
        return None
    if metrics["potential_hint_rows"] != len(sources):
        return None
    if metrics["control_suppressed_rows"] != 1:
        return None
    if metrics["control_pixels"] != len(sources) - 1:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
