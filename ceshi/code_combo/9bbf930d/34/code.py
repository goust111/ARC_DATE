# Combo variant of 9bbf930d (index 34)
# combo_spec: D1/0+D12/1
# Name: 直线射线+粒子互相阻挡

from __future__ import annotations

import random
from typing import Any

import numpy as np


BG = 7
PARTICLE = 6
_DELTAS = {"up": (-1, 0), "down": (1, 0)}


"""
Design analysis for the combo variant
=====================================

1. Concrete behavior and traced example
   - I preserve the original striped look: even rows are colored bars, odd rows are
     potential lanes, col 0 is filled with magenta, and col 1 is background.
   - A row is still "real" exactly when some background cell on that odd row is
     flanked by equal non-bg colors above and below whose bars begin together at
     col 2 or 3. Fake rows still use mismatched bar colors and therefore do not move.
   - D1/0 removes corridor following after the entrance. Each movable particle now
     shoots a purely horizontal ray and stops immediately before the first colored
     pixel on its own row, or at the right edge if there is no blocker.
   - D12/1 then has to be reinterpreted because literal same-cell collisions vanish
     once turning is removed. The fused rule uses the only remaining visible ordering:
     top-to-bottom settlement. After a particle finishes at column c, later movable
     rows cannot pass column c and therefore stop at or before c-1 on their own rows.
   - Tiny trace:
       top real row has a turn-bait blocker at col 12, so its static ray ends at 11.
       the next real row has no blocker until the edge, so independently it would end
       at W-1, but under the fused rule it is capped by the earlier particle and ends
       at 10. A third real row with a far blocker at col 16 is then capped again and
       ends at 9.
     Original task: the top row would have turned into the bait shaft instead of
     stopping; lower rows would have been solved independently and would not queue.
     Fused task: the top row stays horizontal, and its endpoint creates the first
     blocking frontier for all later real rows.

2. Complexity calibration
   - I keep real-vs-fake lane recognition from the original task so the input is not
     reduced to an unstructured list of row blockers.
   - I inject 1-3 old-style turn openings as decoys. At least one of them is placed on
     the first movable row, making the "straight ray, do not turn" change easy to see.
   - I also force 3-6 movable rows and require a visible blocking staircase on later
     rows, so the collision component is not inert.
   - The collision rule only uses one ordering axis, top to bottom. That keeps the
     freedom small enough that humans can infer it from 2-4 examples.

3. Implicit assumption audit against the original solver
   - Original `_trace()` assumed a colored blocker could trigger a perpendicular search.
     That is no longer valid; the fused solver never turns during actual motion.
   - Original `main()` processed each row independently and wrote its endpoint directly
     to the output. That breaks under D12/1 because earlier settled particles must alter
     later rays.
   - Original output writes were safe because endpoints were independent. In the fused
     solver I first compute all final positions in row order, then repaint once.
   - The original entrance test remains useful and is deliberately preserved. The input
     still advertises which rows are real via same-color bar pairs.
   - The new top-to-bottom shadowing assumption is not present in the original code, so
     the generator must deliberately create examples where later static endpoints lie to
     the right of earlier ones; otherwise D12/1 would visually disappear.

4. Placement legality
   - Bars still start at col 2 or 3 and stay in bounds.
   - Turn-bait shafts use reserved 3-column windows so local gap carving does not clash
     with other bait gadgets.
   - The lead blocker is placed far enough from the left edge that even the last queued
     particle still moves a visible distance.
   - Later blockers, when present, are always placed strictly to the right of the lead
     blocker so the collision frontier, not the local blocker, determines their result.

5. Visual observability
   - A human can still identify the movable rows from same-color bar pairs.
   - On at least one row, the input visibly contains an old turn opening immediately
     before a blocker, yet the output particle remains on the same row. That makes D1/0
     observable without hidden bookkeeping.
   - Across the movable rows, output particles form a descending leftward staircase.
     Since lower rows often have farther blockers or even no blocker, the only simple
     explanation is that earlier particles are constraining later ones.

6. Visual cleanliness
   - The layout remains sparse and striped.
   - Only a few bait shafts are carved, and they stay in reserved columns.
   - Collision is shown through one monotone staircase rather than multiple unrelated
     queues, so the output remains easy to parse by eye.

7. Color-change inference
   - No recoloring rule is introduced. Colors only mark real/fake bar pairs, blockers,
     and distractors. The transformation is positional rather than palette-based.

8. Quantified fusion checks used by the generator
   - `changed_vs_straight`: how many movable rows differ from independent straight rays.
     This must be at least 2, otherwise the collision dimension is too weak.
   - `bait_rows`: how many movable rows contain a valid old-style turn opportunity at
     their straight-ray stop. This must be at least 1, otherwise the straight-ray
     dimension is visually inert.
   - `changed_vs_original`: how many movable rows differ from the original corridor
     follower. This must be at least 1, otherwise the sample may accidentally behave
     like the source task.

9. New solver assumptions and how the generator guarantees them
   - Assumption: only rows with valid real entrances participate in the blocking chain.
     Guarantee: fake rows keep mismatched bar pairs, so they remain at col 0.
   - Assumption: blocking order is top-to-bottom.
     Guarantee: the generator creates several real rows whose static endpoints all lie
     to the right of the first row's endpoint, so the shadowing order is visible.
   - Assumption: at least one old-style turn would have been available under the source
     task.
     Guarantee: the first movable row always receives a deliberate turn-bait gadget.
"""


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a traversable cell flanked by equal non-bg walls."""
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if int(grid[r, c]) not in {BG, PARTICLE}:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == h - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == w - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and a not in {BG, PARTICLE}


def _find_entrance(grid: np.ndarray, r: int) -> int | None:
    """Return the first valid entrance column for a real lane, else None."""
    h, w = grid.shape
    if not (0 < r < h - 1) or int(grid[r, 0]) != PARTICLE:
        return None

    entrance_col = None
    for c in range(2, w):
        if int(grid[r, c]) != BG:
            continue
        above = int(grid[r - 1, c])
        below = int(grid[r + 1, c])
        if above == below and above not in {BG, PARTICLE}:
            entrance_col = c
            break
    if entrance_col is None:
        return None

    color = int(grid[r - 1, entrance_col])
    bar_above = next((c for c in range(w) if int(grid[r - 1, c]) == color), w)
    bar_below = next((c for c in range(w) if int(grid[r + 1, c]) == color), w)
    if bar_above != bar_below or bar_above > 3:
        return None
    return entrance_col


def _ray_endpoint(grid: np.ndarray, r: int, c0: int = 0) -> tuple[int, int]:
    """Move horizontally until the first colored pixel or the right boundary."""
    _, w = grid.shape
    c = c0
    while True:
        nc = c + 1
        if nc >= w:
            break
        if int(grid[r, nc]) in {BG, PARTICLE}:
            c = nc
            continue
        break
    return r, c


def _trace_original(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Source-task trace, used only for generator-side validation."""
    direction = "right"
    for _ in range(grid.shape[0] * grid.shape[1] * 4):
        if direction == "right":
            nr, nc = r, c + 1
        elif direction == "left":
            nr, nc = r, c - 1
        elif direction == "up":
            nr, nc = r - 1, c
        else:
            nr, nc = r + 1, c

        if nr < 0 or nr >= grid.shape[0] or nc < 0 or nc >= grid.shape[1]:
            break
        if int(grid[nr, nc]) in {BG, PARTICLE}:
            r, c = nr, nc
            continue

        if direction in {"left", "right"}:
            turn_dirs = ["up", "down"]
        else:
            turn_dirs = ["left", "right"]
        options = []
        for turn_dir in turn_dirs:
            dr, dc = {
                "up": (-1, 0),
                "down": (1, 0),
                "left": (0, -1),
                "right": (0, 1),
            }[turn_dir]
            tr, tc = r + dr, c + dc
            if _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        if len(options) != 1:
            break
        r, c, direction = options[0]
    return r, c


def _collect_particles(grid: np.ndarray) -> list[dict[str, Any]]:
    """Collect visible particles; only rows with real entrances are movable."""
    h, _ = grid.shape
    particles: list[dict[str, Any]] = []
    for r in range(h):
        if int(grid[r, 0]) != PARTICLE:
            continue
        movable = _find_entrance(grid, r) is not None
        particles.append(
            {
                "start": (r, 0),
                "movable": movable,
            }
        )
    particles.sort(key=lambda p: p["start"][0])
    return particles


def _simulate_particles(grid: np.ndarray, with_blocking: bool) -> list[tuple[int, int]]:
    """Return final particle positions with or without top-to-bottom shadowing."""
    particles = _collect_particles(grid)
    finals: list[tuple[int, int]] = []
    frontier_col: int | None = None

    for p in particles:
        r, c0 = p["start"]
        if not p["movable"]:
            finals.append((r, c0))
            continue

        _, static_c = _ray_endpoint(grid, r, c0)
        end_c = static_c
        if with_blocking and frontier_col is not None:
            end_c = min(end_c, frontier_col - 1)
            end_c = max(end_c, c0)

        finals.append((r, end_c))
        if with_blocking:
            frontier_col = end_c if frontier_col is None else min(frontier_col, end_c)

    return finals


def _simulate_original(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return source-task endpoints for comparison during generation."""
    finals: list[tuple[int, int]] = []
    for p in _collect_particles(grid):
        r, c = p["start"]
        finals.append(_trace_original(grid, r, c) if p["movable"] else (r, c))
    return finals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move magenta particles as straight rays, with earlier settled particles shortening later rays.

    Input structure:
    - bg=7 and particle=6. Col 0 contains visible particles, col 1 is clear background.
    - Even rows are colored bars. Odd rows are candidate lanes; only rows flanked by equal
      non-bg bar colors are real movers.

    Rule:
    - Each real-row particle shoots horizontally to the right and would normally stop just
      before the first colored pixel on its own row, or at the right edge.
    - Resolve movable rows from top to bottom. Once a particle settles at column c, every
      later movable row is capped at column c-1 even if its own row would allow a longer ray.
    - Fake-row particles never move.

    Key insight:
    - Same-color bar pairs still identify which rows are active, but old turn openings are
      now decoys. The output is determined by horizontal ray endpoints plus a top-to-bottom
      blocking frontier formed by earlier settled particles.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    particles = _collect_particles(g)
    finals = _simulate_particles(g, with_blocking=True)

    for p in particles:
        sr, sc = p["start"]
        out[sr, sc] = BG
    for r, c in finals:
        out[r, c] = PARTICLE
    return out


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (18, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

_N_REAL_OPTIONS = [3, 4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2, 1]
_MIN_PALETTE_DIVERSITY = 3

_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)
_FAKE_DECOR_RANGE = (0, 2)

_MIN_DISTINCT_COLORS = 6
_MAX_RETRIES = 400


def _allocate_vcol(min_col: int, bar_start: int, w: int, used_vcols: set[int]) -> int | None:
    """Reserve a turn-bait column window and return the blocker column."""
    c_min = max(bar_start + _VCOL_LEFT_PAD, min_col)
    c_max = w - _VCOL_RIGHT_PAD
    candidates = [
        c
        for c in range(c_min, c_max + 1)
        if all((c + d) not in used_vcols for d in _VCOL_WINDOW)
    ]
    if not candidates:
        return None
    c_v = random.choice(candidates)
    used_vcols.update({c_v + d for d in _VCOL_WINDOW})
    return c_v


def _choose_bait_direction(channel_idx: int, n_channels: int) -> str:
    """Prefer a direction with visible room for the shaft."""
    if channel_idx == 0:
        return "down"
    if channel_idx == n_channels - 1:
        return "up"
    return random.choice(["up", "down"])


def _build_turn_bait(
    g: np.ndarray,
    top: int,
    bot: int,
    row: int,
    color: int,
    c_v: int,
    direction: str,
    bar_colors: list[int],
) -> None:
    """Build an old-style L-turn opening that the fused solver deliberately ignores."""
    h, _ = g.shape
    g[row, c_v] = color

    if direction == "up":
        turn_bar, opp_bar = top, bot
    else:
        turn_bar, opp_bar = bot, top

    g[turn_bar, c_v - 1] = BG
    g[turn_bar, c_v - 2] = bar_colors[turn_bar // 2]
    g[turn_bar, c_v] = bar_colors[turn_bar // 2]
    g[opp_bar, c_v - 1] = bar_colors[opp_bar // 2]

    if direction == "up":
        for r_br in range(0, top, 2):
            g[r_br, c_v - 1] = BG
    else:
        for r_br in range(bot + 2, h, 2):
            g[r_br, c_v - 1] = BG


def _assign_bar_colors(channel_spec: list[str], extras: list[int]) -> list[int]:
    """Create even-row colors satisfying the requested real/fake pattern."""
    first = random.choice(extras)
    bar_colors = [first]
    unused = [c for c in extras if c != first]
    random.shuffle(unused)

    for ctype in channel_spec:
        prev = bar_colors[-1]
        if ctype == "real":
            bar_colors.append(prev)
            continue
        choices = [c for c in extras if c != prev]
        preferred = [c for c in unused if c in choices]
        nxt = random.choice(preferred or choices)
        if nxt in unused:
            unused.remove(nxt)
        bar_colors.append(nxt)
    return bar_colors


def _paint_bars(g: np.ndarray, bar_colors: list[int], bar_start: int) -> None:
    """Fill each even row with its bar color from bar_start to the right edge."""
    _, w = g.shape
    for i, color in enumerate(bar_colors):
        g[2 * i, bar_start:w] = color


def _place_row_blocker(g: np.ndarray, row: int, col: int, palette: list[int]) -> None:
    """Place a colored blocker on one odd row."""
    g[row, col] = random.choice(palette)


def _seed_fake_decor(
    g: np.ndarray,
    fake_indices: list[int],
    bar_start: int,
    extras: list[int],
    used_colors: set[int],
) -> None:
    """Add small distractions on fake rows and force any missing extra colors to appear."""
    _, w = g.shape
    pending = [c for c in extras if c not in used_colors]
    random.shuffle(fake_indices)

    for color, idx in zip(pending, fake_indices):
        row = 2 * idx + 1
        col = random.randint(bar_start + 1, w - 2)
        if g[row, col] == BG:
            g[row, col] = color
            used_colors.add(color)

    for idx in fake_indices:
        row = 2 * idx + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, w - 1), min(n_decor, w - bar_start - 2))
        for c in cols:
            if g[row, c] == BG:
                g[row, c] = random.choice(extras)


def _behavior_stats(grid: np.ndarray) -> dict[str, Any]:
    """Quantify whether both combo dimensions are visibly active."""
    particles = _collect_particles(grid)
    movable_mask = [bool(p["movable"]) for p in particles]

    straight = _simulate_particles(grid, with_blocking=False)
    fused = _simulate_particles(grid, with_blocking=True)
    original = _simulate_original(grid)

    changed_vs_straight = sum(
        1 for movable, a, b in zip(movable_mask, straight, fused) if movable and a != b
    )
    changed_vs_original = sum(
        1 for movable, a, b in zip(movable_mask, original, fused) if movable and a != b
    )

    movable_cols = [c for movable, (_, c) in zip(movable_mask, fused) if movable]
    staircase_run = 0
    run = 0
    prev = None
    for c in movable_cols:
        if prev is None or c == prev - 1:
            run = 1 if prev is None else run + 1
        else:
            run = 1
        staircase_run = max(staircase_run, run)
        prev = c

    bait_rows = 0
    for movable, p in zip(movable_mask, particles):
        if not movable:
            continue
        r, _ = p["start"]
        _, end_c = _ray_endpoint(grid, r, 0)
        blocker_col = end_c + 1
        if blocker_col >= grid.shape[1]:
            continue
        if int(grid[r, blocker_col]) in {BG, PARTICLE}:
            continue
        options = 0
        for turn_dir in ("up", "down"):
            dr, dc = _DELTAS[turn_dir]
            if _has_corridor(grid, r + dr, end_c + dc, turn_dir):
                options += 1
        if options == 1:
            bait_rows += 1

    return {
        "n_particles": len(particles),
        "n_movable": sum(movable_mask),
        "changed_vs_straight": changed_vs_straight,
        "changed_vs_original": changed_vs_original,
        "staircase_run": staircase_run,
        "bait_rows": bait_rows,
    }


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(_HEIGHT_OPTIONS)
    w = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in {BG, PARTICLE}], _N_EXTRA_COLORS)
    bar_start = random.choice(_BAR_START_OPTIONS)

    n_channels = (h - 1) // 2
    n_real = min(random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0], n_channels)
    real_indices = sorted(random.sample(range(n_channels), n_real))
    real_set = set(real_indices)
    fake_indices = [i for i in range(n_channels) if i not in real_set]
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]

    bar_colors = _assign_bar_colors(channel_spec, extras)
    if len(set(bar_colors)) < _MIN_PALETTE_DIVERSITY:
        return None

    g = np.full((h, w), BG, dtype=int)
    _paint_bars(g, bar_colors, bar_start)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    lead_min = bar_start + 6 + n_real
    lead_max = w - 4
    if lead_min > lead_max:
        return None
    lead_blocker = random.randint(lead_min, lead_max)
    used_vcols: set[int] = set()

    lead_idx = real_indices[0]
    lead_top = 2 * lead_idx
    lead_row = lead_top + 1
    lead_bot = lead_top + 2
    lead_color = bar_colors[lead_idx]
    used_vcols.update({lead_blocker + d for d in _VCOL_WINDOW})
    _build_turn_bait(
        g,
        lead_top,
        lead_bot,
        lead_row,
        lead_color,
        lead_blocker,
        _choose_bait_direction(lead_idx, n_channels),
        bar_colors,
    )

    bait_budget = random.randint(0, min(2, max(0, n_real - 1)))
    later_bait_rows = set(random.sample(real_indices[1:], bait_budget)) if n_real > 1 else set()

    for idx in real_indices[1:]:
        row = 2 * idx + 1
        top = 2 * idx
        bot = top + 2
        min_col = lead_blocker + 2
        plan = random.choice(["edge", "far_blocker", "far_blocker", "far_bait"])

        if min_col > w - 2:
            continue

        if idx in later_bait_rows or plan == "far_bait":
            c_v = _allocate_vcol(min_col, bar_start, w, used_vcols)
            if c_v is not None:
                _build_turn_bait(
                    g,
                    top,
                    bot,
                    row,
                    bar_colors[idx],
                    c_v,
                    _choose_bait_direction(idx, n_channels),
                    bar_colors,
                )
                continue

        if plan != "edge":
            _place_row_blocker(g, row, random.randint(min_col, w - 2), extras)

    used_colors = set(int(x) for x in np.unique(g))
    _seed_fake_decor(g, fake_indices, bar_start, extras, used_colors)

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
        stats = _behavior_stats(g)
    except Exception:
        return None

    if stats["n_movable"] != n_real:
        return None
    if stats["changed_vs_straight"] < 2:
        return None
    if stats["staircase_run"] < 2:
        return None
    if stats["bait_rows"] < 1:
        return None
    if stats["changed_vs_original"] < 1:
        return None
    if len(set(int(x) for x in np.unique(g))) < _MIN_DISTINCT_COLORS:
        return None
    if np.array_equal(g, out):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
