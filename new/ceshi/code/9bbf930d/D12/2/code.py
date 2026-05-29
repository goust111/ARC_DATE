# Variant of 9bbf930d
# Dimension: D12 — 粒子数量与路径多样性 (meta)
# Change: 3-6个粒子同时存在，每个粒子有0-3次转弯，路径长度14-46格。... → 粒子按颜色分批次处理

from __future__ import annotations

import random
import numpy as np

"""
Design analysis for this D12 variant
====================================

a) Concrete behavior and trace
   - The striped corridor geometry stays the same: even rows are colored bars, odd rows are
     horizontal channels, and vertical turns are carved by opening single-column gaps.
   - The visible change is that the left column now contains two particle colors:
     `PARTICLE_A` and `PARTICLE_B`.
   - Solver semantics:
     1. Trace every A particle on the current grid snapshot.
     2. Commit all A endpoints at once; those settled A pixels stay in the grid.
     3. Trace every B particle on the modified grid, so settled A pixels behave like colored
        obstacles.
   - Generator guarantee:
     at least one A particle takes an L-turn into a vertical shaft and stops exactly on a B
     corridor crossing cell. In the original one-phase logic that crossing is plain BG, so the
     B particle would pass through it and continue. In this variant the crossing is occupied by
     the settled A particle before phase B starts, so the B particle is forced to stop much
     earlier. That gives an observable two-stage cascade.

b) Complexity calibration
   - Only two particle colors are used, with a fixed visible order A then B. There is no hidden
     sorting by color id.
   - Each grid contains 4-6 particles total and at least one explicit A→B interaction, while the
     remaining particles still use the original straight / one-turn corridor vocabulary so the
     task does not collapse into a single special case.

c) Original main() assumptions that break under the variant
   - Original `_has_corridor()` and `_trace()` treat `PARTICLE` as passable everywhere.
     That is no longer valid: settled A must be impassable during phase B.
   - Original `main()` iterates once over a static input and writes endpoints directly into `out`.
     That implicitly assumes particles do not affect each other. Here phase order matters, so each
     color phase is traced from a snapshot and only then committed.
   - Original start detection checks only one particle color. The variant must detect either A or B.

d) Placement legality
   - Generator reserves vertical turn columns with a 3-column exclusion window, matching the
     original geometry, so independent vertical shafts do not overlap.
   - The staged cascade uses a controlled internal stop row that is exactly one of the bar rows
     adjacent to a B channel; this guarantees the A endpoint lands on a valid odd-row crossing cell
     without going out of bounds.
   - The cascade target channel itself is kept straight so the B particle has a long unobstructed
     route in the pre-cascade grid, making the phase effect measurable.

e) Visual observability
   - Humans can infer the rule from input/output pairs because:
     1. there are always two visibly different particle colors,
     2. A-colored endpoints remain in the output,
     3. at least one B particle is truncated by an already-settled A particle.
   - The processing order is constant across samples rather than inferred from numeric ids.

f) Visual cleanliness
   - The generator uses only one forced cross-phase blockage per grid and keeps other paths mostly
     straight or edge-turning. That avoids dense stacks of overlapping decorations.

g) Color-change inference
   - There is no color remapping. Particle colors are preserved from input to output; only positions
     change. That keeps the variant readable from a few examples.

Post-change solver assumptions audited against the generator
------------------------------------------------------------
   - `main()` now assumes particle colors are exactly `PARTICLE_A` and `PARTICLE_B`; generator uses
     only those two colors for starts and excludes them from bar colors.
   - The phase solver assumes starts remain in column 0 and corridor entrances still begin at
     columns 2-3; generator preserves the original striped layout and `col_1 = BG`.
   - The phase solver assumes same-phase particles do not need to block each other while moving;
     generator avoids overlapping destinations, and phase moves are committed simultaneously.
"""

BG = 7
PARTICLE_A = 6
PARTICLE_B = 8
PARTICLE_COLORS = (PARTICLE_A, PARTICLE_B)
_PARTICLE_SET = set(PARTICLE_COLORS)

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


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str, passable: set[int]) -> bool:
    """True if (r,c) is passable and flanked by same-color bars perpendicular to direction."""
    H, W = grid.shape
    if int(grid[r, c]) not in passable:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == W - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and a not in passable


def _trace(grid: np.ndarray, r: int, c: int, particle_color: int) -> tuple[int, int]:
    """Trace one particle through the corridor system using phase-local passability."""
    H, W = grid.shape
    direction = "right"
    passable = {BG, int(particle_color)}
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) in passable:
            r, c = nr, nc
            continue
        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, turn_dir, passable):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
    return r, c


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the first valid horizontal entrance cell for row r."""
    H, W = grid.shape
    if not (1 <= r < H - 1):
        return None
    for c in range(2, W):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in _PARTICLE_SET and above != BG:
            return c
    return None


def _is_valid_start(grid: np.ndarray, r: int, particle_color: int) -> bool:
    """Start rows must have a particle in col 0 and an original-style corridor entrance."""
    if int(grid[r, 0]) != int(particle_color):
        return False
    entrance_col = _find_entrance_col(grid, r)
    if entrance_col is None:
        return False
    H, W = grid.shape
    if not (1 <= r < H - 1):
        return False
    bar_color = int(grid[r - 1, entrance_col])
    bar_above = next((c for c in range(W) if int(grid[r - 1, c]) == bar_color), W)
    bar_below = next((c for c in range(W) if int(grid[r + 1, c]) == bar_color), W)
    return bar_above == bar_below and bar_above <= 3


def _collect_moves(grid: np.ndarray, particle_color: int) -> dict[int, tuple[int, int]]:
    """Trace all movable starts of one phase from a fixed snapshot."""
    H, _W = grid.shape
    moves: dict[int, tuple[int, int]] = {}
    for r in range(1, H - 1):
        if not _is_valid_start(grid, r, particle_color):
            continue
        end_r, end_c = _trace(grid, r, 0, particle_color)
        if (end_r, end_c) != (r, 0):
            moves[int(r)] = (int(end_r), int(end_c))
    return moves


def _apply_moves(grid: np.ndarray, particle_color: int, moves: dict[int, tuple[int, int]]) -> np.ndarray:
    """Apply one phase after all destinations have been computed."""
    out = grid.copy()
    for r in moves:
        out[int(r), 0] = BG
    seen: set[tuple[int, int]] = set()
    for end_r, end_c in moves.values():
        assert (end_r, end_c) not in seen, "phase destinations must be unique"
        seen.add((end_r, end_c))
        out[int(end_r), int(end_c)] = int(particle_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move A particles first, then move B particles with settled A endpoints as obstacles."""
    out = np.array(grid, dtype=int)
    for particle_color in PARTICLE_COLORS:
        moves = _collect_moves(out, particle_color)
        out = _apply_moves(out, particle_color, moves)
    return out


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (16, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

_N_REAL_OPTIONS = [4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2]
_MIN_PALETTE_DIVERSITY = 3

_STRAIGHT_PROB = 0.45
_FAKE_DECOR_RANGE = (0, 2)

_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

_MIN_TOTAL_PARTICLES = 4
_MAX_TOTAL_PARTICLES = 6
_MIN_DISTINCT_COLORS = 6
_MIN_PHASE_EFFECT = 3
_MAX_RETRIES = 400


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for a vertical corridor and return the wall column."""
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = W - _VCOL_RIGHT_PAD
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


def _build_l_turn(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    color: int,
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
    stop_at: int | None,
) -> None:
    """Build an L-turn exactly as in the base task, optionally stopping internally."""
    g[row, c_v] = color
    if direction == "up":
        turn_bar, opp_bar = top, bot
    else:
        turn_bar, opp_bar = bot, top

    g[turn_bar, c_v - 1] = BG
    g[turn_bar, c_v - 2] = bar_color[turn_bar]
    g[turn_bar, c_v] = bar_color[turn_bar]
    g[opp_bar, c_v - 1] = bar_color[opp_bar]

    if direction == "up":
        for r_br in range(0, top, 2):
            if stop_at is not None and r_br <= stop_at:
                continue
            g[r_br, c_v - 1] = BG
    else:
        for r_br in range(bot + 2, H, 2):
            if stop_at is not None and r_br >= stop_at:
                continue
            g[r_br, c_v - 1] = BG


def _assign_particle_colors(real_indices: list[int], source_idx: int, target_idx: int) -> dict[int, int]:
    """Keep both particle colors repeated several times while fixing the cascade roles."""
    n_real = len(real_indices)
    want_a = n_real // 2
    want_b = n_real - want_a
    assignment = {int(source_idx): PARTICLE_A, int(target_idx): PARTICLE_B}
    count_a, count_b = 1, 1
    remaining = [i for i in real_indices if i not in {source_idx, target_idx}]
    random.shuffle(remaining)
    for idx in remaining:
        if count_a < want_a and count_b < want_b:
            color = random.choice(PARTICLE_COLORS)
        elif count_a < want_a:
            color = PARTICLE_A
        else:
            color = PARTICLE_B
        assignment[int(idx)] = int(color)
        if color == PARTICLE_A:
            count_a += 1
        else:
            count_b += 1
    return assignment


def _pick_cascade_pair(real_indices: list[int]) -> tuple[int, int] | None:
    """Prefer a source/target pair separated by at least one full channel."""
    far = [(a, b) for a in real_indices for b in real_indices if a != b and abs(a - b) >= 2]
    near = [(a, b) for a in real_indices for b in real_indices if a != b and abs(a - b) >= 1]
    if far:
        return random.choice(far)
    if near:
        return random.choice(near)
    return None


def _seal_cascade_stop(g: np.ndarray, target_top: int, target_bot: int, c_v: int) -> None:
    """
    Prevent the stopped A particle from turning left/right when its vertical shaft
    terminates on a B crossing row.

    This keeps the crossing visually open for B in the input, but makes the planned
    stop cell a true dead end for the incoming vertical motion.
    """
    g[target_top, c_v - 2] = BG
    g[target_bot, c_v] = BG


def _plan_remaining_turns(
    real_indices: list[int],
    excluded: set[int],
    bar_rows: list[int],
    bar_start: int,
    H: int,
    W: int,
    used_vcols: set[int],
) -> dict[int, dict[str, int | str]]:
    """Give the non-cascade channels mostly straight or outward edge-turn paths."""
    plans: dict[int, dict[str, int | str]] = {}
    min_real = min(real_indices)
    max_real = max(real_indices)
    for idx in real_indices:
        if idx in excluded:
            continue
        if random.random() < _STRAIGHT_PROB:
            plans[int(idx)] = {"kind": "straight"}
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            plans[int(idx)] = {"kind": "straight"}
            continue
        directions = []
        if idx == min_real and bar_rows[idx] > 0:
            directions.append("up")
        if idx == max_real and bar_rows[idx + 1] < H - 1:
            directions.append("down")
        if not directions:
            plans[int(idx)] = {"kind": "straight"}
            continue
        plans[int(idx)] = {
            "kind": "l_edge",
            "direction": random.choice(directions),
            "c_v": int(c_v),
        }
    return plans


def _simulate_independent_b(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    """Baseline for degeneration checks: B traced before any A endpoints exist."""
    return _collect_moves(np.array(grid, dtype=int), PARTICLE_B)


def _measure_phase_effect(grid: np.ndarray) -> tuple[int, int]:
    """
    Count B particles whose endpoint changes under staged execution and the max
    leftward shift in columns relative to the independent baseline.
    """
    independent_b = _simulate_independent_b(grid)
    after_a = _apply_moves(np.array(grid, dtype=int), PARTICLE_A, _collect_moves(grid, PARTICLE_A))
    staged_b = _collect_moves(after_a, PARTICLE_B)
    changed = 0
    max_delta = 0
    for r, independent_end in independent_b.items():
        staged_end = staged_b.get(r, (r, 0))
        if staged_end != independent_end:
            changed += 1
            max_delta = max(max_delta, int(independent_end[1]) - int(staged_end[1]))
    return changed, max_delta


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in _PARTICLE_SET and c != BG], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1

    n_real = min(random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0], n_channels)
    real_indices = sorted(random.sample(range(n_channels), n_real))
    real_set = set(real_indices)
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]

    bar_color = {bar_rows[0]: random.choice(extras)}
    for i, ctype in enumerate(channel_spec):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if ctype == "real":
            bar_color[bot] = bar_color[top]
        else:
            bar_color[bot] = random.choice([c for c in extras if c != bar_color[top]])

    if len(set(bar_color.values())) < _MIN_PALETTE_DIVERSITY:
        return None

    bar_start = random.choice(_BAR_START_OPTIONS)
    for br in bar_rows:
        g[br, bar_start:W] = bar_color[br]

    pair = _pick_cascade_pair(real_indices)
    if pair is None:
        return None
    source_idx, target_idx = pair

    used_vcols: set[int] = set()
    c_v = _allocate_vcol(bar_start, W, used_vcols)
    if c_v is None:
        return None

    source_top, source_bot = bar_rows[source_idx], bar_rows[source_idx + 1]
    source_row = source_top + 1
    target_top, target_bot = bar_rows[target_idx], bar_rows[target_idx + 1]

    if source_idx > target_idx:
        direction = "up"
        stop_at = target_top
    else:
        direction = "down"
        stop_at = target_bot

    _build_l_turn(
        g,
        H,
        source_top,
        source_bot,
        source_row,
        bar_color[source_top],
        c_v,
        direction,
        bar_color,
        stop_at,
    )
    _seal_cascade_stop(g, target_top, target_bot, c_v)

    cascade_cell = (target_top + 1, c_v - 1)

    remaining_plans = _plan_remaining_turns(
        real_indices,
        {source_idx, target_idx},
        bar_rows,
        bar_start,
        H,
        W,
        used_vcols,
    )
    for idx, plan in remaining_plans.items():
        if plan["kind"] == "straight":
            continue
        top, bot = bar_rows[idx], bar_rows[idx + 1]
        row = top + 1
        _build_l_turn(
            g,
            H,
            top,
            bot,
            row,
            bar_color[top],
            int(plan["c_v"]),
            str(plan["direction"]),
            bar_color,
            None,
        )

    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        r = bar_rows[i] + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, W), min(n_decor, W - bar_start - 1))
        for c_decor in cols:
            if g[r, c_decor] == BG:
                g[r, c_decor] = random.choice([bar_color[bar_rows[i]], bar_color[bar_rows[i + 1]]])

    particle_by_channel = _assign_particle_colors(real_indices, source_idx, target_idx)
    g[:, 0] = BG
    for idx, particle_color in particle_by_channel.items():
        g[bar_rows[idx] + 1, 0] = particle_color
    g[:, 1] = BG

    # The forced cascade endpoint must stay open in the input.
    if int(g[cascade_cell]) != BG:
        return None

    try:
        out = main(g)
    except Exception:
        return None

    moved = sum(1 for idx in real_indices if int(out[bar_rows[idx] + 1, 0]) == BG)
    if not (_MIN_TOTAL_PARTICLES <= moved <= _MAX_TOTAL_PARTICLES):
        return None
    if len(set(map(int, g.flatten()))) < _MIN_DISTINCT_COLORS:
        return None

    changed_b, max_delta = _measure_phase_effect(g)
    if changed_b < 1 or max_delta < _MIN_PHASE_EFFECT:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
