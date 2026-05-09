# Variant of 9bbf930d
# Dimension: D12 — 粒子数量与路径多样性 (meta)
# Change: 3-6个粒子同时存在，每个粒子有0-3次转弯，路径长度14-46格。... → 粒子间互相阻挡

from __future__ import annotations

import random
from typing import Any

import numpy as np

BG = 7
PARTICLE = 6
_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


# Design analysis for the D12 collision variant
# ----------------------------------------------
# reference_task:
#   requested_source_task: e12f9a14
#   availability: solver code not present in the provided workspace/resources
#   usable_signal: only the task id and "collision_detection" tag from the prompt
#   consequence: this implementation keeps the collision mechanism simple and explicit:
#     particles move in lockstep; if a particle tries to enter an occupied next cell,
#     it stops permanently at its current cell.
#
# concrete_behavior:
#   preserved_visual_language:
#     - even rows are colored bars
#     - odd rows are horizontal corridors
#     - turns are still 90-degree turns triggered by a colored blocker ahead
#   new_visual_behavior:
#     - only 3-6 magenta particles are placed in col 0, all on odd rows
#     - 2-4 of them are routed into the same target corridor tail
#     - they reach that tail at different times, so the first one reaches the stop cell,
#       later ones queue immediately behind it
#   tiny_trace:
#     - original: A and B independently enter the same tail and both "finish" at the same
#       terminal cell, with the later write overwriting the earlier one
#     - variant: A reaches the terminal cell first and stays there; when B later tries to
#       step into A's cell, B freezes one cell earlier; C then freezes behind B
#     - observable difference: output shows a packed queue of magenta cells instead of a
#       single survivor at the shared endpoint
#
# complexity_calibration:
#   - the collision group size is 2-4, not unbounded
#   - all colliding particles share the same target row, so humans can see the queue
#   - arrival order is geometric, not color-id-based: lower source rows have shorter paths
#     to the shared tail and therefore arrive first
#   - some non-colliding particles remain straight to preserve contrast with the old rule
#
# implicit_assumption_audit_against_original_main:
#   - original loop "for each row, trace independently" assumed particles never affect each
#     other; that is invalid now, so the solver must keep dynamic particle state
#   - original output write "out[end_r, end_c] = PARTICLE" was harmless because each trace
#     was isolated; with collisions it would silently overwrite earlier results, so the new
#     solver computes all final positions first and paints them afterwards
#   - original _trace depended only on static grid occupancy; this remains valid for turn
#     discovery, but movement legality now also depends on dynamic particle occupancy
#
# placement_legality:
#   - the generator uses one shared-color band of consecutive channels
#   - several source channels connect downward into one target channel through distinct,
#     well-spaced shafts; shafts are at least 3 columns apart so their local turn gadgets
#     do not overlap
#   - every downward re-entry blocks the left option on the target row, making the turn
#     uniquely rightward and avoiding ambiguous assertions
#
# visual_observability_and_tidiness:
#   - the collision always occurs on one clean horizontal tail near the right side
#   - the queue is contiguous, so it is easy to see that later particles stop behind earlier
#     ones instead of vanishing or teleporting
#   - no hidden ordering uses color ids or random tie-breaks; ties are avoided by geometry
#
# new_solver_assumptions_checked_by_generator:
#   - movable particles start at col 0 and have a valid same-color entrance
#   - collision shafts are acyclic; no head-on cases are generated
#   - simultaneous same-cell ties are avoided by giving colliding particles distinct path
#     lengths to the shared tail


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a traversable cell flanked by same-color bars."""
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
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


def _find_entrance(grid: np.ndarray, r: int) -> int | None:
    """Return the first real-corridor entrance column for row r, or None."""
    h, w = grid.shape
    if r <= 0 or r >= h - 1 or int(grid[r, 0]) != PARTICLE:
        return None
    entrance_col = None
    for c in range(2, w):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in {PARTICLE, BG}:
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


def _step_candidate(grid: np.ndarray, r: int, c: int, direction: str) -> tuple[int, int, str] | None:
    """Single-step movement proposal, ignoring other particles."""
    h, w = grid.shape
    dr, dc = _DELTAS[direction]
    nr, nc = r + dr, c + dc
    if nr < 0 or nr >= h or nc < 0 or nc >= w:
        return None
    if int(grid[nr, nc]) in {PARTICLE, BG}:
        return nr, nc, direction
    options = []
    for turn_dir in _PERPENDICULAR[direction]:
        tdr, tdc = _DELTAS[turn_dir]
        tr, tc = r + tdr, c + tdc
        if _has_corridor(grid, tr, tc, turn_dir):
            options.append((tr, tc, turn_dir))
    assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
    return options[0] if options else None


def _collect_particles(grid: np.ndarray) -> list[dict[str, Any]]:
    """Collect visible particles; only those with real entrances are active movers."""
    h, _ = grid.shape
    particles: list[dict[str, Any]] = []
    for r in range(h):
        if int(grid[r, 0]) != PARTICLE:
            continue
        movable = _find_entrance(grid, r) is not None
        particles.append(
            {
                "start": (r, 0),
                "r": r,
                "c": 0,
                "dir": "right",
                "active": movable,
                "movable": movable,
            }
        )
    particles.sort(key=lambda p: p["start"][0])
    return particles


def _simulate_particles(grid: np.ndarray, with_collisions: bool) -> list[tuple[int, int]]:
    """Return final particle positions, with or without dynamic blocking."""
    particles = _collect_particles(grid)
    if not with_collisions:
        max_steps = grid.shape[0] * grid.shape[1] * 4
        finals = []
        for p in particles:
            r, c, direction = p["r"], p["c"], p["dir"]
            if p["movable"]:
                for _ in range(max_steps):
                    nxt = _step_candidate(grid, r, c, direction)
                    if nxt is None:
                        break
                    r, c, direction = nxt
            finals.append((r, c))
        return finals

    max_steps = grid.shape[0] * grid.shape[1] * 4
    for _ in range(max_steps):
        active_indices = [i for i, p in enumerate(particles) if p["active"]]
        if not active_indices:
            break

        current_pos = {(p["r"], p["c"]): i for i, p in enumerate(particles)}
        proposals: list[tuple[int, int, str] | None] = [None] * len(particles)
        for i in active_indices:
            p = particles[i]
            nxt = _step_candidate(grid, p["r"], p["c"], p["dir"])
            if nxt is None:
                p["active"] = False
            else:
                proposals[i] = nxt

        if not any(proposals):
            break

        target_to_ids: dict[tuple[int, int], list[int]] = {}
        for i, nxt in enumerate(proposals):
            if nxt is None:
                continue
            target_to_ids.setdefault((nxt[0], nxt[1]), []).append(i)

        winners = set()
        blocked = set()
        for ids in target_to_ids.values():
            if len(ids) == 1:
                winners.add(ids[0])
            else:
                winner = min(ids)
                winners.add(winner)
                blocked.update(i for i in ids if i != winner)

        moves: dict[int, tuple[int, int, str]] = {}
        for i in sorted(winners):
            if i in blocked:
                continue
            nxt = proposals[i]
            if nxt is None:
                continue
            tr, tc, ndir = nxt
            occupant = current_pos.get((tr, tc))
            if occupant is None or occupant == i:
                moves[i] = nxt
                continue

            occ_prop = proposals[occupant]
            occ_leaves = occ_prop is not None and (occ_prop[0], occ_prop[1]) != (
                particles[occupant]["r"],
                particles[occupant]["c"],
            )
            head_on_swap = occ_prop is not None and (occ_prop[0], occ_prop[1]) == (
                particles[i]["r"],
                particles[i]["c"],
            )
            if occ_leaves and not head_on_swap:
                moves[i] = nxt
            else:
                blocked.add(i)

        moved = False
        for i, nxt in moves.items():
            if particles[i]["active"]:
                nr, nc, ndir = nxt
                moved = moved or (nr, nc) != (particles[i]["r"], particles[i]["c"])
                particles[i]["r"], particles[i]["c"], particles[i]["dir"] = nr, nc, ndir

        for i, p in enumerate(particles):
            if not p["active"]:
                continue
            if proposals[i] is None or i in blocked or i not in moves:
                p["active"] = False

        if not moved:
            break

    return [(p["r"], p["c"]) for p in particles]


def main(grid: np.ndarray) -> np.ndarray:
    """Move all visible particles simultaneously; occupied cells block followers."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    particles = _collect_particles(g)
    finals = _simulate_particles(g, with_collisions=True)

    for p in particles:
        sr, sc = p["start"]
        out[sr, sc] = BG
    for r, c in finals:
        out[r, c] = PARTICLE
    return out


_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (17, 21)
_BAR_START_OPTIONS = [2, 3]
_MAX_RETRIES = 400


def _sample_spaced_columns(candidates: list[int], count: int, min_gap: int) -> list[int] | None:
    """Pick count columns with at least min_gap spacing."""
    random.shuffle(candidates)
    chosen: list[int] = []
    for c in sorted(candidates):
        if all(abs(c - prev) >= min_gap for prev in chosen):
            chosen.append(c)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _build_shared_tail_source(
    g: np.ndarray,
    source_idx: int,
    target_idx: int,
    wall_col: int,
    color: int,
) -> None:
    """Source channel turns down into the target row, then must turn right."""
    top = 2 * source_idx
    bot = top + 2
    source_row = top + 1
    target_top = 2 * target_idx
    target_row = 2 * target_idx + 1
    shaft = wall_col - 1

    g[source_row, wall_col] = color
    for rr in range(bot, target_row + 1):
        g[rr, shaft] = BG
    # Break the left option structurally: the cell stays BG, but its bars mismatch.
    g[target_top, shaft - 1] = BG


def _paint_bars(g: np.ndarray, bar_colors: list[int], bar_start: int) -> None:
    """Fill each even row with its bar color from bar_start to the right edge."""
    h, w = g.shape
    for i, color in enumerate(bar_colors):
        r = 2 * i
        if r >= h:
            break
        g[r, bar_start:w] = color


def _behavior_stats(grid: np.ndarray) -> dict[str, Any]:
    """Quantify whether collisions changed the result in a visible way."""
    particles = _collect_particles(grid)
    starts = [p["start"] for p in particles]
    independent = _simulate_particles(grid, with_collisions=False)
    collided = _simulate_particles(grid, with_collisions=True)
    changed = sum(1 for a, b in zip(independent, collided) if a != b)

    row_counts: dict[int, list[int]] = {}
    for r, c in collided:
        row_counts.setdefault(r, []).append(c)
    max_queue = 1
    for cols in row_counts.values():
        cols = sorted(cols)
        run = 1
        best = 1
        for i in range(1, len(cols)):
            if cols[i] == cols[i - 1] + 1:
                run += 1
                best = max(best, run)
            else:
                run = 1
        max_queue = max(max_queue, best)

    return {
        "n_particles": len(starts),
        "changed_particles": changed,
        "max_queue": max_queue,
        "starts": starts,
        "independent": independent,
        "collided": collided,
    }


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(_HEIGHT_OPTIONS)
    w = random.randint(*_WIDTH_RANGE)
    n_channels = (h - 1) // 2
    bar_start = random.choice(_BAR_START_OPTIONS)
    extras = random.sample([c for c in range(10) if c not in {BG, PARTICLE}], 4)

    n_particles = random.randint(3, min(6, n_channels - 1))
    group_size = random.randint(2, min(4, n_particles))
    if n_channels < group_size + 2:
        return None

    band_start = random.randint(0, n_channels - group_size - 1)
    target_idx = band_start + group_size
    source_indices = list(range(band_start, target_idx))

    protected = set(range(max(0, band_start - 1), min(n_channels, target_idx + 2)))
    extra_count = n_particles - group_size
    extra_pool = [i for i in range(n_channels) if i not in protected]
    if len(extra_pool) < extra_count:
        return None
    extra_indices = random.sample(extra_pool, extra_count)

    bar_colors = [random.choice(extras) for _ in range((h + 1) // 2)]
    group_color = extras[0]
    for i in range(band_start, target_idx + 2):
        bar_colors[i] = group_color

    extra_color_cycle = extras[1:] or extras[:1]
    for idx, ch in enumerate(sorted(extra_indices)):
        color = extra_color_cycle[idx % len(extra_color_cycle)]
        bar_colors[ch] = color
        bar_colors[ch + 1] = color

    if len(set(bar_colors)) < 3:
        return None

    g = np.full((h, w), BG, dtype=int)
    _paint_bars(g, bar_colors, bar_start)

    turn_candidates = list(range(bar_start + 5, w - 4))
    wall_cols = _sample_spaced_columns(turn_candidates, group_size, min_gap=3)
    if wall_cols is None:
        return None

    stop_col = w - 1
    target_row = 2 * target_idx + 1
    g[target_row, stop_col] = group_color

    for source_idx, wall_col in zip(source_indices, wall_cols):
        _build_shared_tail_source(g, source_idx, target_idx, wall_col, group_color)

    active_channels = source_indices + extra_indices
    for ch in active_channels:
        g[2 * ch + 1, 0] = PARTICLE

    # Keep the staging column clear and add a small amount of harmless fake decoration.
    g[:, 1] = BG
    fake_rows = [2 * i + 1 for i in range(n_channels) if i not in active_channels and i != target_idx]
    random.shuffle(fake_rows)
    for r in fake_rows[: random.randint(0, min(2, len(fake_rows)))]:
        c = random.randint(bar_start + 1, w - 3)
        if g[r, c] == BG:
            g[r, c] = random.choice(extras[1:])

    try:
        out = main(g)
    except Exception:
        return None

    stats = _behavior_stats(g)
    if stats["n_particles"] != n_particles:
        return None
    if stats["changed_particles"] < 1:
        return None
    if stats["max_queue"] < 2:
        return None
    if len(set(g.flatten().tolist())) < 6:
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
