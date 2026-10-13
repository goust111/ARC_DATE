from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
ORANGE = 7
WHITE = 8
MIRROR = 9

LIGHT_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE)
GEN_RETRIES = 500


@dataclass
class Beam:
    beam_id: int
    color: int
    source: tuple[int, int]
    pos: tuple[int, int]
    direction: tuple[int, int]
    reflected: bool = False
    active: bool = True
    stop_reason: str | None = None
    trail: list[tuple[int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class StepProposal:
    target: tuple[int, int]
    new_direction: tuple[int, int]
    reflected_now: bool = False


@dataclass
class SimResult:
    output: np.ndarray
    beams: list[Beam]
    overlay_cells: list[tuple[int, int]]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _detect_mirror_kind(grid: np.ndarray) -> int:
    coords = np.argwhere(grid == MIRROR)
    assert coords.size > 0, "expected a mirror diagonal"
    assert coords.shape[0] >= 2, "mirror should contain at least two pixels"

    deltas = {int(r - c) for r, c in coords}
    sums = {int(r + c) for r, c in coords}
    is_backslash = len(deltas) == 1
    is_slash = len(sums) == 1
    assert is_backslash ^ is_slash, "mirror must be a single 45-degree diagonal"
    return 1 if is_backslash else -1


def _reflect(direction: tuple[int, int], mirror_kind: int) -> tuple[int, int]:
    dr, dc = direction
    if mirror_kind == 1:
        return (dc, dr)
    return (-dc, -dr)


def _edge_direction(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = shape
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    assert (on_top or on_bottom or on_left or on_right), "source must touch the grid edge"
    assert sum((on_top, on_bottom, on_left, on_right)) == 1, "corner sources are ambiguous"
    if on_top:
        return (1, 0)
    if on_bottom:
        return (-1, 0)
    if on_left:
        return (0, 1)
    return (0, -1)


def _find_sources(grid: np.ndarray) -> list[Beam]:
    h, w = grid.shape
    sources: list[Beam] = []
    beam_id = 0
    seen: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if not (r == 0 or r == h - 1 or c == 0 or c == w - 1):
                continue
            color = int(grid[r, c])
            if color in (BG, GRAY, WHITE, MIRROR):
                continue
            if (r, c) in seen:
                continue
            seen.add((r, c))
            sources.append(
                Beam(
                    beam_id=beam_id,
                    color=color,
                    source=(r, c),
                    pos=(r, c),
                    direction=_edge_direction(grid.shape, r, c),
                )
            )
            beam_id += 1
    return sources


def _propose_step(grid: np.ndarray, beam: Beam, mirror_kind: int) -> StepProposal | None:
    r, c = beam.pos
    dr, dc = beam.direction
    nr, nc = r + dr, c + dc
    if not _in_bounds(grid.shape, nr, nc):
        beam.active = False
        beam.stop_reason = "boundary"
        return None

    cell = int(grid[nr, nc])
    if cell == MIRROR:
        new_dir = _reflect(beam.direction, mirror_kind)
        beam.reflected = True
        rr, cc = nr + new_dir[0], nc + new_dir[1]
        if not _in_bounds(grid.shape, rr, cc):
            beam.active = False
            beam.stop_reason = "boundary"
            return None
        bounced = int(grid[rr, cc])
        if bounced == GRAY:
            beam.active = False
            beam.stop_reason = "blocker"
            return None
        if bounced != BG:
            beam.active = False
            beam.stop_reason = "solid"
            return None
        return StepProposal((rr, cc), new_dir, reflected_now=True)

    if cell == GRAY:
        beam.active = False
        beam.stop_reason = "blocker"
        return None
    if cell != BG:
        beam.active = False
        beam.stop_reason = "solid"
        return None
    return StepProposal((nr, nc), beam.direction, reflected_now=False)


def _simulate(grid: np.ndarray) -> SimResult:
    g = np.asarray(grid, dtype=int)
    mirror_kind = _detect_mirror_kind(g)
    beams = _find_sources(g)
    out = g.copy()
    trail_sets: dict[tuple[int, int], set[int]] = {}

    def add_color(cell: tuple[int, int], color: int) -> None:
        colors = trail_sets.setdefault(cell, set())
        colors.add(color)
        out[cell] = WHITE if len(colors) >= 2 else next(iter(colors))

    while any(beam.active for beam in beams):
        painted_before_tick = set(trail_sets)
        proposals: dict[tuple[int, int], list[tuple[Beam, StepProposal]]] = {}

        for beam in beams:
            if not beam.active:
                continue
            step = _propose_step(g, beam, mirror_kind)
            if step is None:
                continue
            proposals.setdefault(step.target, []).append((beam, step))

        if not proposals:
            break

        for cell, entries in proposals.items():
            if cell not in painted_before_tick and len(entries) == 1:
                continue
            for beam, step in entries:
                add_color(cell, beam.color)
                beam.trail.append(cell)
                beam.pos = cell
                beam.direction = step.new_direction
                beam.reflected = beam.reflected or step.reflected_now
                beam.active = False
                beam.stop_reason = "trail"

        for cell, entries in proposals.items():
            if cell in painted_before_tick or len(entries) != 1:
                continue
            beam, step = entries[0]
            add_color(cell, beam.color)
            beam.trail.append(cell)
            beam.pos = cell
            beam.direction = step.new_direction
            beam.reflected = beam.reflected or step.reflected_now

    overlay_cells = [cell for cell, colors in trail_sets.items() if len(colors) >= 2]
    overlay_cells.sort()
    return SimResult(output=out, beams=beams, overlay_cells=overlay_cells)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw light trails emitted from colored edge sources, reflecting off a maroon 45-degree mirror.

    Input structure:
    - Black background with a single maroon diagonal mirror line.
    - Colored source pixels sit on the grid edge and emit inward normally.
    - Optional gray blockers stop a beam.

    Rule:
    - Each beam paints black cells in front of it with the source color.
    - Hitting the mirror turns a vertical beam horizontal or a horizontal beam vertical.
    - Mirror pixels themselves are preserved.
    - A beam stops at the border, a gray blocker, a source pixel, or a pre-existing light trail.
    - If beams of different colors share a trail cell, that cell becomes white.
    """
    return _simulate(np.asarray(grid, dtype=int)).output


def _mirror_coords(
    start: tuple[int, int],
    mirror_kind: int,
    length: int,
) -> list[tuple[int, int]]:
    dr, dc = (1, 1) if mirror_kind == 1 else (1, -1)
    r0, c0 = start
    return [(r0 + i * dr, c0 + i * dc) for i in range(length)]


def _build_base_grid(
    rng: random.Random,
    want_truncated: bool,
    want_unreflected: bool,
) -> np.ndarray | None:
    h = rng.randint(10, 18)
    w = rng.randint(10, 18)
    mirror_kind = rng.choice((1, -1))
    min_len = 5
    starts: list[tuple[tuple[int, int], int]] = []

    if mirror_kind == 1:
        for c0 in range(w):
            starts.append(((0, c0), min(h, w - c0)))
        for r0 in range(1, h):
            starts.append(((r0, 0), min(h - r0, w)))
    else:
        for c0 in range(w):
            starts.append(((0, c0), min(h, c0 + 1)))
        for r0 in range(1, h):
            starts.append(((r0, w - 1), min(h - r0, w)))

    eligible = [(start, max_len) for start, max_len in starts if max_len >= min_len]
    rng.shuffle(eligible)
    for start, max_len in eligible:
        if want_truncated:
            if max_len < min_len + 1:
                continue
            length = rng.randint(min_len, max_len - 1)
        else:
            length = max_len
            if want_unreflected and h == w:
                coords = _mirror_coords(start, mirror_kind, length)
                rows = {r for r, _ in coords}
                cols = {c for _, c in coords}
                if len(rows) == h and len(cols) == w:
                    continue

        grid = np.full((h, w), BG, dtype=int)
        coords = _mirror_coords(start, mirror_kind, length)
        for r, c in coords:
            grid[r, c] = MIRROR

        if want_truncated:
            full = _mirror_coords(start, mirror_kind, max_len)
            br, bc = full[length]
            grid[br, bc] = GRAY
        return grid
    return None


def _candidate_sources(base: np.ndarray) -> list[tuple[int, int]]:
    h, w = base.shape
    candidates: list[tuple[int, int]] = []

    for c in range(1, w - 1):
        if base[0, c] == BG and base[1, c] == BG:
            candidates.append((0, c))
        if base[h - 1, c] == BG and base[h - 2, c] == BG:
            candidates.append((h - 1, c))
    for r in range(1, h - 1):
        if base[r, 0] == BG and base[r, 1] == BG:
            candidates.append((r, 0))
        if base[r, w - 1] == BG and base[r, w - 2] == BG:
            candidates.append((r, w - 1))

    return candidates


def _spaced_enough(chosen: list[tuple[int, int]], candidate: tuple[int, int]) -> bool:
    cr, cc = candidate
    for orow, ocol in chosen:
        if abs(cr - orow) + abs(cc - ocol) <= 1:
            return False
    return True


def _render_sources(
    base: np.ndarray,
    positions: list[tuple[int, int]],
    colors: tuple[int, ...],
) -> np.ndarray:
    grid = base.copy()
    for (r, c), color in zip(positions, colors):
        grid[r, c] = color
    return grid


def _positions_with_features(
    base: np.ndarray,
    positions: list[tuple[int, int]],
    want_cross: bool,
    want_unreflected: bool,
) -> bool:
    colors = tuple(LIGHT_COLORS[: len(positions)])
    sim = _simulate(_render_sources(base, positions, colors))
    if not any(beam.reflected for beam in sim.beams):
        return False
    if want_cross and not sim.overlay_cells:
        return False
    if not want_cross and sim.overlay_cells:
        return False
    if want_unreflected and not any((not beam.reflected) and beam.trail for beam in sim.beams):
        return False
    if sum(bool(beam.trail) for beam in sim.beams) < 2:
        return False
    if sum(len(beam.trail) for beam in sim.beams) < 4:
        return False
    return True


def _select_source_positions(
    base: np.ndarray,
    want_cross: bool,
    want_unreflected: bool,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    candidates = _candidate_sources(base)
    max_sources = min(len(candidates), len(LIGHT_COLORS), 5)
    min_sources = 3 if want_unreflected else 2
    if max_sources < min_sources:
        return None
    target_count = rng.randint(min_sources, max_sources)

    pairs = list(itertools.combinations(candidates, 2))
    rng.shuffle(pairs)
    for first, second in pairs:
        seed = [first, second]
        if not _spaced_enough([first], second):
            continue
        if not _positions_with_features(base, seed, want_cross, False):
            continue

        chosen = seed[:]
        remaining = [pos for pos in candidates if pos not in chosen]
        rng.shuffle(remaining)

        for pos in remaining:
            if len(chosen) >= target_count:
                break
            if not _spaced_enough(chosen, pos):
                continue
            trial = chosen + [pos]
            colors = tuple(LIGHT_COLORS[: len(trial)])
            sim = _simulate(_render_sources(base, trial, colors))
            if want_cross and len(sim.overlay_cells) > 3:
                continue
            if not want_cross and sim.overlay_cells:
                continue
            chosen = trial

        if len(chosen) < min_sources:
            continue
        if _positions_with_features(base, chosen, want_cross, want_unreflected):
            return chosen
    return None


def _insert_blocker_if_needed(
    grid: np.ndarray,
    sim: SimResult,
    want_cross: bool,
    want_blocked: bool,
    rng: random.Random,
) -> np.ndarray | None:
    if not want_blocked:
        return grid
    if any(beam.stop_reason == "blocker" for beam in sim.beams):
        return grid

    usage: dict[tuple[int, int], int] = {}
    for beam in sim.beams:
        for cell in beam.trail:
            usage[cell] = usage.get(cell, 0) + 1

    candidates: list[tuple[int, int]] = []
    h, w = grid.shape
    for beam in sim.beams:
        for cell in beam.trail[1:-1]:
            if usage.get(cell, 0) != 1:
                continue
            r, c = cell
            if not (0 < r < h - 1 and 0 < c < w - 1):
                continue
            candidates.append(cell)

    rng.shuffle(candidates)
    for cell in candidates:
        trial = grid.copy()
        trial[cell] = GRAY
        trial_sim = _simulate(trial)
        if not any(beam.stop_reason == "blocker" for beam in trial_sim.beams):
            continue
        if want_cross and not trial_sim.overlay_cells:
            continue
        if not want_cross and trial_sim.overlay_cells:
            continue
        if not any(beam.reflected for beam in trial_sim.beams):
            continue
        return trial
    return None


def _accept_generated(
    grid: np.ndarray,
    want_cross: bool,
    want_blocked: bool,
    want_unreflected: bool,
) -> bool:
    try:
        sim = _simulate(grid)
        out = main(grid)
    except Exception:
        return False

    if not np.array_equal(out, sim.output):
        return False
    if np.array_equal(out, grid):
        return False
    if len(_find_sources(grid)) < 2:
        return False
    if not any(beam.reflected for beam in sim.beams):
        return False
    if want_cross and not sim.overlay_cells:
        return False
    if want_blocked and not any(beam.stop_reason == "blocker" for beam in sim.beams):
        return False
    if want_unreflected and not any((not beam.reflected) and beam.trail for beam in sim.beams):
        return False

    changed = int(np.count_nonzero(out != grid))
    if changed < 4 or changed > (grid.shape[0] * grid.shape[1]) // 2:
        return False
    if len(sim.overlay_cells) > 3:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    want_truncated = rng.random() < 0.5
    want_cross = rng.random() < 0.5
    want_blocked = rng.random() < 0.5
    want_unreflected = rng.random() < 0.5

    base = _build_base_grid(rng, want_truncated=want_truncated, want_unreflected=want_unreflected)
    if base is None:
        return None

    positions = _select_source_positions(
        base,
        want_cross=want_cross,
        want_unreflected=want_unreflected,
        rng=rng,
    )
    if positions is None:
        return None

    colors = tuple(rng.sample(LIGHT_COLORS, k=len(positions)))
    grid = _render_sources(base, positions, colors)
    sim = _simulate(grid)

    maybe_blocked = _insert_blocker_if_needed(
        grid,
        sim,
        want_cross=want_cross,
        want_blocked=want_blocked,
        rng=rng,
    )
    if maybe_blocked is None:
        return None
    grid = maybe_blocked

    if _accept_generated(
        grid,
        want_cross=want_cross,
        want_blocked=want_blocked,
        want_unreflected=want_unreflected,
    ):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-beam ARC input")
