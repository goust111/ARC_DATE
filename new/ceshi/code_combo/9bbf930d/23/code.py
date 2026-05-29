# Combo variant of 9bbf930d (index 23)
# combo_spec: D4/0+D12/1
# Name: 连续轨迹+粒子互相阻挡

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

"""
Design analysis for combo D4/0 + D12/1
=======================================

a) Concrete behavior and traced execution
-----------------------------------------
Base task:
- A magenta particle starts in column 0, follows same-color corridors through orange
  background, turns 90 degrees only when a colored blocker ahead exposes exactly one
  perpendicular same-color corridor, and the output keeps only the terminal magenta
  cell.

Fusion behavior:
- Several visible particles start together.
- Movement is simulated in lockstep.
- Whenever a particle first reaches a background cell, that cell is claimed forever by
  that particle's magenta trail.
- Any later particle that tries to enter an already-claimed cell stops immediately at
  its current cell.
- Output paints every claimed path cell, so blocking is visible as a shorter trail.

Concrete mini-trace:
- Particle A reaches a shared tail first and claims cells ...,(r,10),(r,11),(r,12).
- Particle B comes from a longer shaft and later reaches (r,9); when it next tries to
  step into (r,10), that cell is already claimed by A, so B freezes at (r,9).
- Original output would show only A's and B's independent endpoints.
- Fusion output shows A's full tail plus B's shorter branch ending one cell earlier.

b) Complexity calibration
-------------------------
- The collision group is limited to 2-4 particles so the shared-tail interaction is
  visible without flooding the grid.
- Extra particles stay on separate straight corridors, giving repeated evidence that
  long trails are normal while only the converging group gets truncated.
- The geometry, not the color ids, determines arrival order: lower source channels have
  shorter vertical shafts and farther-right re-entry, so they claim shared tail cells
  earlier.

c) Implicit assumption audit against the original main()
--------------------------------------------------------
- Original `_trace()` assumed static occupancy and independent particles. That fails
  here because earlier arrivals alter future reachability. The replacement solver keeps
  dynamic particle state plus a permanent claimed-cell map.
- Original `out[end_r, end_c] = PARTICLE` assumed one visible output cell per moved
  particle. That fails because every visited background cell must now remain visible.
- Original entrance filtering still matters: only rows whose first real corridor starts
  near columns 2-3 are treated as movable.
- Turn legality still depends only on the static wall pattern. The new dynamic effect
  is applied only when entering background cells, not when deciding whether a turn
  corridor exists.

d) Placement legality
---------------------
- The generator uses striped even-row bars exactly as in the base task.
- A collision group of consecutive real channels shares one color band and turns
  downward through well-spaced shafts into the same target row; shaft windows are at
  least 3 columns apart so their turn gadgets do not overlap.
- The target row has a right-edge stopper, guaranteeing the first arrival leaves a long
  visible tail for later particles to collide with.
- Extra particles are placed on disjoint real corridors outside the protected band, so
  they do not interfere with the shared-tail construction.

e) Visual observability
-----------------------
- In 2-4 examples, a human sees multiple magenta trails, with one long trail on the
  shared tail and shorter converging trails that stop behind it.
- The difference from the base task is visible twice over: trails remain drawn, and
  converging particles no longer all finish at the same far-right cell.
- No hidden palette ordering is used; the only special color is the already-visible
  magenta particle color.

f) Visual cleanliness
---------------------
- All trails stay inside 1-cell corridors and never repaint the colored bars.
- The collision is localized to one shared tail, so the output has one readable focal
  interaction instead of many overlapping crossings.
- Distinct shaft columns keep the bent trails separated until they merge.

g) Change inferability
----------------------
- Every changed cell is magenta because it was visited by a particle.
- The shortened converging branches provide structural evidence that "earlier trail
  blocks later motion"; this is inferable from shape and timing, not from arbitrary
  recoloring.

New solver-side assumptions guaranteed by the generator
-------------------------------------------------------
- Active particles only appear on rows with valid real-corridor entrances.
- The shared-tail band is acyclic, so no particle needs to revisit its own claimed
  trail; permanent claim blocking therefore causes no self-conflict.
- Simultaneous same-cell proposals are avoided by geometry in normal samples, though the
  solver still resolves them deterministically for robustness.
"""


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
    """Single-step proposal from the static corridor geometry."""
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
    """Collect visible particles and mark which ones are movable."""
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
                "movable": movable,
                "active": movable,
                "path": [],
            }
        )
    particles.sort(key=lambda p: p["start"][0])
    return particles


def _trace_independent_path(grid: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    """Trace one particle independently and return every visited non-start cell."""
    direction = "right"
    path: list[tuple[int, int]] = []
    for _ in range(grid.shape[0] * grid.shape[1] * 4):
        nxt = _step_candidate(grid, r, c, direction)
        if nxt is None:
            break
        r, c, direction = nxt
        path.append((r, c))
    return path


def _simulate_particles(grid: np.ndarray, with_claim_blocking: bool) -> list[dict[str, Any]]:
    """Simulate all visible particles and return their final state plus path."""
    particles = _collect_particles(grid)
    if not with_claim_blocking:
        for p in particles:
            if p["movable"]:
                p["path"] = _trace_independent_path(grid, p["r"], p["c"])
                if p["path"]:
                    p["r"], p["c"] = p["path"][-1]
        return particles

    claimed: dict[tuple[int, int], int] = {}
    max_steps = grid.shape[0] * grid.shape[1] * 4
    for _ in range(max_steps):
        active = [i for i, p in enumerate(particles) if p["active"]]
        if not active:
            break

        proposals: list[tuple[int, int, str] | None] = [None] * len(particles)
        blocked = set()
        for i in active:
            p = particles[i]
            nxt = _step_candidate(grid, p["r"], p["c"], p["dir"])
            if nxt is None:
                p["active"] = False
                continue
            tr, tc, _ = nxt
            owner = claimed.get((tr, tc))
            if owner is not None and owner != i:
                blocked.add(i)
                continue
            proposals[i] = nxt

        if not any(proposals):
            for i in blocked:
                particles[i]["active"] = False
            continue

        target_to_ids: dict[tuple[int, int], list[int]] = {}
        for i, nxt in enumerate(proposals):
            if nxt is None:
                continue
            target_to_ids.setdefault((nxt[0], nxt[1]), []).append(i)

        movers = set()
        for ids in target_to_ids.values():
            if len(ids) == 1:
                movers.add(ids[0])
            else:
                winner = min(ids)
                movers.add(winner)
                blocked.update(i for i in ids if i != winner)

        for i in sorted(movers):
            if i in blocked:
                continue
            nr, nc, ndir = proposals[i]
            particles[i]["r"], particles[i]["c"], particles[i]["dir"] = nr, nc, ndir
            particles[i]["path"].append((nr, nc))
            claimed[(nr, nc)] = i

        for i in active:
            if proposals[i] is None or i in blocked or i not in movers:
                particles[i]["active"] = False

    return particles


def _render_output(grid: np.ndarray, particles: list[dict[str, Any]]) -> np.ndarray:
    """Render trails on a copy of the grid."""
    out = np.array(grid, dtype=int)
    for p in particles:
        if p["path"]:
            sr, sc = p["start"]
            out[sr, sc] = BG
            for r, c in p["path"]:
                if int(grid[r, c]) == BG:
                    out[r, c] = PARTICLE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move multiple particles, leave full trails, and let earlier trails block later ones.

    Input structure:
    - bg=7 and particle=6. Only some odd rows in column 0 contain visible particles.
    - Even rows are colored bars; odd rows are horizontal corridors.
    - Real corridors are same-color bar pairs that may connect through vertical shafts.

    Rule:
    - All visible particles begin moving to the right at the same time.
    - At each step, a particle goes straight through bg cells; if blocked by a colored
      cell, it may turn 90 degrees into the unique perpendicular same-color corridor.
    - Every background cell first visited by a particle becomes part of a permanent
      magenta trail. Later particles cannot enter already-claimed trail cells and stop
      immediately before them. Cells visited by a particle remain magenta in the output.

    Key insight:
    - The output is not just independent path painting: trails are also the dynamic
      blocking state, so earlier arrivals shorten later trajectories where paths merge.
    """
    g = np.array(grid, dtype=int)
    particles = _simulate_particles(g, with_claim_blocking=True)
    return _render_output(g, particles)


_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (17, 21)
_BAR_START_OPTIONS = [2, 3]
_MAX_RETRIES = 500
_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MIN_TRAIL_CELLS = 18
_MIN_CHANGED_PARTICLES = 1
_MIN_VISIBLE_DELTA = 4
_MIN_BLOCKED = 1


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


def _sample_spaced_channels(candidates: list[int], count: int, min_gap: int) -> list[int] | None:
    """Pick channel indices whose bar pairs do not overlap."""
    candidates = sorted(candidates)
    random.shuffle(candidates)
    chosen: list[int] = []
    for idx in candidates:
        if all(abs(idx - prev) >= min_gap for prev in chosen):
            chosen.append(idx)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _paint_bars(g: np.ndarray, bar_colors: list[int], bar_start: int) -> None:
    """Fill each even row with its bar color from bar_start to the right edge."""
    h, w = g.shape
    for i, color in enumerate(bar_colors):
        r = 2 * i
        if r >= h:
            break
        g[r, bar_start:w] = color


def _build_shared_tail_source(
    g: np.ndarray,
    source_idx: int,
    target_idx: int,
    wall_col: int,
    color: int,
) -> None:
    """Turn one source channel downward into the shared target tail."""
    top = 2 * source_idx
    bot = top + 2
    source_row = top + 1
    target_top = 2 * target_idx
    target_row = 2 * target_idx + 1
    shaft = wall_col - 1

    g[source_row, wall_col] = color
    for rr in range(bot, target_row + 1):
        g[rr, shaft] = BG
    g[target_top, shaft - 1] = BG


def _decorate_fake_rows(
    g: np.ndarray,
    extras: list[int],
    active_channels: set[int],
    target_idx: int,
    bar_start: int,
) -> None:
    """Add a small amount of harmless noise so the palette stays diverse."""
    n_channels = (g.shape[0] - 1) // 2
    fake_rows = [2 * i + 1 for i in range(n_channels) if i not in active_channels and i != target_idx]
    random.shuffle(fake_rows)
    n_decor = random.randint(1, min(3, len(fake_rows))) if fake_rows else 0
    for r in fake_rows[:n_decor]:
        c = random.randint(bar_start + 2, g.shape[1] - 3)
        if g[r, c] == BG:
            g[r, c] = random.choice(extras[1:])


def _behavior_stats(grid: np.ndarray) -> dict[str, Any]:
    """Quantify whether the fusion behavior is visible and non-degenerate."""
    independent = _simulate_particles(grid, with_claim_blocking=False)
    collided = _simulate_particles(grid, with_claim_blocking=True)
    out_independent = _render_output(grid, independent)
    out_collided = _render_output(grid, collided)

    moved = sum(1 for p in collided if p["path"])
    trail_cells = int(np.sum((grid == BG) & (out_collided == PARTICLE)))
    changed_particles = sum(
        1
        for p0, p1 in zip(independent, collided)
        if p0["path"] != p1["path"]
    )
    blocked_particles = sum(
        1
        for p0, p1 in zip(independent, collided)
        if len(p1["path"]) < len(p0["path"])
    )
    return {
        "n_particles": len(collided),
        "moved": moved,
        "trail_cells": trail_cells,
        "changed_particles": changed_particles,
        "blocked_particles": blocked_particles,
        "visible_delta": int(np.sum(out_independent != out_collided)),
    }


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input with one guaranteed shared-tail collision group."""
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
    extra_indices = _sample_spaced_channels(extra_pool, extra_count, min_gap=2)
    if extra_indices is None:
        return None

    bar_colors = [extras[i % len(extras)] for i in range((h + 1) // 2)]
    group_color = extras[0]
    for i in range(band_start, target_idx + 2):
        bar_colors[i] = group_color

    extra_cycle = extras[1:] or extras[:1]
    for idx, ch in enumerate(sorted(extra_indices)):
        color = extra_cycle[idx % len(extra_cycle)]
        bar_colors[ch] = color
        bar_colors[ch + 1] = color

    g = np.full((h, w), BG, dtype=int)
    _paint_bars(g, bar_colors, bar_start)

    turn_candidates = list(range(bar_start + 5, w - 4))
    wall_cols = _sample_spaced_columns(turn_candidates, group_size, min_gap=3)
    if wall_cols is None:
        return None

    target_row = 2 * target_idx + 1
    g[target_row, w - 1] = group_color

    for source_idx, wall_col in zip(source_indices, wall_cols):
        _build_shared_tail_source(g, source_idx, target_idx, wall_col, group_color)

    active_channels = set(source_indices + extra_indices)
    for ch in active_channels:
        g[2 * ch + 1, 0] = PARTICLE

    g[:, 1] = BG
    _decorate_fake_rows(g, extras, active_channels, target_idx, bar_start)
    if any(_find_entrance(g, 2 * ch + 1) is None for ch in active_channels):
        return None

    try:
        out = main(g)
    except Exception:
        return None

    stats = _behavior_stats(g)
    if stats["n_particles"] != n_particles:
        return None
    if not (_MIN_MOVED <= stats["moved"] <= _MAX_MOVED):
        return None
    if stats["trail_cells"] < _MIN_TRAIL_CELLS:
        return None
    if stats["changed_particles"] < _MIN_CHANGED_PARTICLES:
        return None
    if stats["blocked_particles"] < _MIN_BLOCKED:
        return None
    if stats["visible_delta"] < _MIN_VISIBLE_DELTA:
        return None
    if stats["moved"] != n_particles:
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
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
