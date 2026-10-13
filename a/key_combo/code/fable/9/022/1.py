from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GRAY = 5
SPIRAL_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
LEGEND_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_HEIGHT_RANGE = (12, 23)
GRID_WIDTH_RANGE = (12, 24)
LEGEND_LENGTH_RANGE = (4, 7)
MAX_GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _extract_legend(grid: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    top = np.asarray(grid[0], dtype=int)
    nz = np.flatnonzero(top != BLACK)
    assert len(nz) >= 2, "expected one multi-color top legend"
    c0, c1 = int(nz[0]), int(nz[-1])
    assert len(nz) == c1 - c0 + 1, "top legend must be a single contiguous run"
    legend = tuple(int(v) for v in top[c0 : c1 + 1])
    assert len(set(legend)) >= 2, "legend must contain multiple colors"
    assert np.all(grid[1] == BLACK), "legend must be separated from the interior by a black row"
    return c0, c1, legend


def _seed_pos(grid: np.ndarray) -> tuple[int, int]:
    seeds = np.argwhere(np.asarray(grid, dtype=int) == GRAY)
    assert seeds.shape == (1, 2), "expected exactly one gray seed pixel"
    r, c = map(int, seeds[0])
    assert r >= 2, "seed must be below the separator row"
    return r, c


def _blocker_positions(grid: np.ndarray, legend_colors: Iterable[int], seed: tuple[int, int]) -> set[tuple[int, int]]:
    legend_colors = set(int(v) for v in legend_colors)
    blockers: set[tuple[int, int]] = set()
    h, w = grid.shape
    for r in range(2, h):
        for c in range(w):
            if (r, c) == seed:
                continue
            value = int(grid[r, c])
            if value == BLACK:
                continue
            assert value != GRAY, "gray is reserved for the single seed"
            assert value not in legend_colors, "legend colors must not be reused as blockers"
            blockers.add((r, c))
    return blockers


def _spiral_events(
    shape: tuple[int, int],
    seed: tuple[int, int],
    blockers: set[tuple[int, int]],
    legend: tuple[int, ...],
    *,
    consume_offgrid: bool = True,
    ignore_blockers: bool = False,
    return_on_block: bool = False,
) -> list[dict]:
    h, w = shape
    r, c = seed
    band_idx = 0
    leg_length = 2
    dir_idx = 0
    max_leg_length = 2 * (max(h, w) + 2)
    events: list[dict] = []

    while leg_length <= max_leg_length:
        dr, dc = SPIRAL_DIRS[dir_idx]
        for _ in range(leg_length):
            nr, nc = r + dr, c + dc
            blocked = _in_bounds(shape, nr, nc) and (nr, nc) in blockers
            if blocked and not ignore_blockers:
                if return_on_block:
                    return events
                return events

            r, c = nr, nc
            color = int(legend[band_idx % len(legend)])
            in_grid = _in_bounds(shape, r, c)
            on_legend_row = in_grid and r == 0
            paintable = in_grid and r != 0
            consumed = in_grid or consume_offgrid
            if consumed:
                band_idx += 1

            events.append(
                {
                    "step": len(events),
                    "pos": (r, c),
                    "color": color,
                    "in_grid": in_grid,
                    "on_legend_row": on_legend_row,
                    "paintable": paintable,
                    "blocked": blocked,
                }
            )

        dir_idx = (dir_idx + 1) % 4
        if dir_idx in (0, 2):
            leg_length += 2

    return events


def _simulate(
    grid: np.ndarray,
    *,
    consume_offgrid: bool = True,
    ignore_blockers: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    _, _, legend = _extract_legend(g)
    seed = _seed_pos(g)
    blockers = _blocker_positions(g, legend, seed)
    out = g.copy()

    for event in _spiral_events(
        g.shape,
        seed,
        blockers,
        legend,
        consume_offgrid=consume_offgrid,
        ignore_blockers=ignore_blockers,
    ):
        if not event["in_grid"] or event["on_legend_row"]:
            continue
        r, c = event["pos"]
        if int(out[r, c]) == BLACK:
            out[r, c] = int(event["color"])

    return out, _spiral_events(
        g.shape,
        seed,
        blockers,
        legend,
        consume_offgrid=consume_offgrid,
        ignore_blockers=ignore_blockers,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a color-cycling square spiral from the gray seed while off-grid steps still consume the top legend band.

    Input structure:
    - The top row contains exactly one contiguous multi-color legend string, with a full black separator row beneath it.
    - Below the separator there is exactly one gray seed pixel on a black background plus zero or more fixed blocker pixels of non-legend colors.
    - The legend row is reserved reference data and must stay unchanged.

    Rule:
    - Starting at the seed, walk an unbounded square spiral using directions up, right, down, left and leg lengths 2, 2, 4, 4, 6, 6, ...
    - Before each actual step, if the next in-grid landing cell is a blocker, stop immediately; otherwise take the step and consume the next legend color, cycling when needed.
    - A stepped cell is painted only when it lies inside the grid, is not on the legend row, and is still black; off-grid steps consume colors but paint nothing.

    Key insight:
    - The color pointer follows the full spiral trajectory in the plane, not just the visible painted cells, so re-entering segments are phase-shifted after hidden off-grid travel.
    """
    out, _ = _simulate(grid, consume_offgrid=True, ignore_blockers=False)
    return out


def _has_offgrid_phase_shift(events: list[dict], legend_len: int) -> bool:
    painted_steps = [event["step"] for event in events if event["paintable"] and not event["blocked"]]
    for prev_step, next_step in zip(painted_steps, painted_steps[1:]):
        offgrid_between = sum(
            1
            for event in events[prev_step + 1 : next_step]
            if not event["in_grid"]
        )
        if offgrid_between > 0 and offgrid_between % legend_len != 0:
            return True
    return False


def _touches_reserved_top_row(events: list[dict]) -> bool:
    return any(event["on_legend_row"] for event in events)


def _paint_count(events: list[dict]) -> int:
    return sum(1 for event in events if event["paintable"] and not event["blocked"])


def _available_offpath_cells(
    shape: tuple[int, int],
    seed: tuple[int, int],
    path_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    cells = []
    for r in range(2, h):
        for c in range(w):
            if (r, c) == seed or (r, c) in path_cells:
                continue
            cells.append((r, c))
    random.shuffle(cells)
    return cells


def _pick_spaced(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    chosen: list[tuple[int, int]] = []
    for r, c in cells:
        if all(max(abs(r - rr), abs(c - cc)) >= 2 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
    return None


def _make_candidate() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    legend_len = random.randint(*LEGEND_LENGTH_RANGE)
    if legend_len + 2 > w:
        return None

    legend_colors = tuple(random.sample(LEGEND_POOL, legend_len))
    remaining_colors = [c for c in LEGEND_POOL if c not in legend_colors]
    if not remaining_colors:
        return None
    blocker_color = random.choice(remaining_colors)

    branch = random.choices(
        ["terminate", "offpath_only", "none"],
        weights=[5, 3, 2],
        k=1,
    )[0]

    grid = np.zeros((h, w), dtype=int)
    legend_start = random.randint(1, w - legend_len - 1)
    grid[0, legend_start : legend_start + legend_len] = np.asarray(legend_colors, dtype=int)

    seed_mode = random.choice(["upper", "left", "right", "upper_left", "upper_right"])
    if seed_mode == "upper":
        seed_r = random.randint(2, min(h - 4, 5))
        seed_c = random.randint(3, w - 4)
    elif seed_mode == "left":
        seed_r = random.randint(4, h - 4)
        seed_c = random.randint(2, min(w - 4, 4))
    elif seed_mode == "right":
        seed_r = random.randint(4, h - 4)
        seed_c = random.randint(max(2, w - 5), w - 3)
    elif seed_mode == "upper_left":
        seed_r = random.randint(2, min(h - 4, 5))
        seed_c = random.randint(2, min(w - 4, 4))
    else:
        seed_r = random.randint(2, min(h - 4, 5))
        seed_c = random.randint(max(2, w - 5), w - 3)
    grid[seed_r, seed_c] = GRAY

    base_events = _spiral_events(grid.shape, (seed_r, seed_c), set(), legend_colors)
    if not _touches_reserved_top_row(base_events):
        return None
    if not _has_offgrid_phase_shift(base_events, len(legend_colors)):
        return None
    if _paint_count(base_events) < 10:
        return None

    painted = [event for event in base_events if event["paintable"]]
    path_cells = {tuple(event["pos"]) for event in painted}
    evidence_step = min(
        event["step"]
        for event in base_events
        if event["paintable"]
        and any(not later["in_grid"] for later in base_events[: event["step"] + 1])
    )

    if branch == "terminate":
        candidates = []
        for event in painted:
            future_paints = sum(1 for later in painted if later["step"] > event["step"])
            if event["step"] <= evidence_step or future_paints < 3:
                continue
            candidates.append(tuple(event["pos"]))
        if not candidates:
            return None
        stop_r, stop_c = random.choice(candidates)
        grid[stop_r, stop_c] = blocker_color
        extra_count = random.randint(0, 2)
    elif branch == "offpath_only":
        extra_count = random.randint(1, 3)
    else:
        extra_count = 0

    if extra_count:
        offpath = _available_offpath_cells(grid.shape, (seed_r, seed_c), path_cells)
        picks = _pick_spaced(offpath, extra_count)
        if picks is None:
            return None
        for r, c in picks:
            grid[r, c] = blocker_color

    return grid


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        correct, events = _simulate(inp, consume_offgrid=True, ignore_blockers=False)
        direct = main(inp)
        refund_offgrid, _ = _simulate(inp, consume_offgrid=False, ignore_blockers=False)
        ignore_blockers, _ = _simulate(inp, consume_offgrid=True, ignore_blockers=True)
        seed = _seed_pos(inp)
        _, _, legend = _extract_legend(inp)
        blockers = _blocker_positions(inp, legend, seed)
    except Exception:
        return False

    if not np.array_equal(correct, direct):
        return False
    if np.array_equal(correct, refund_offgrid):
        return False
    if not _has_offgrid_phase_shift(events, len(legend)):
        return False
    if not _touches_reserved_top_row(events):
        return False
    if _paint_count(events) < 10:
        return False
    if blockers and np.array_equal(correct, ignore_blockers):
        return False
    if any(int(correct[r, c]) != int(inp[r, c]) for r, c in blockers):
        return False
    if not np.array_equal(correct[0], inp[0]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        candidate = _make_candidate()
        if candidate is not None and _semantic_checks(candidate):
            return candidate
    raise RuntimeError("failed to generate a valid spiral-band input")
