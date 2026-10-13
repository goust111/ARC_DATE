from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
CORNER_TO_DIR = {
    "tl": (1, 1),
    "tr": (1, -1),
    "bl": (-1, 1),
    "br": (-1, -1),
}

ARENA_H_RANGE = (8, 19)
ARENA_W_RANGE = (8, 19)
LEGEND_LEN_RANGE = (2, 5)
STOP_BOUNCES_MIN = 2
CYCLE_BRANCH_PROB = 0.45
DISTRACTOR_BLOCK_COUNT = (0, 3)
BLOCK_SIDE_RANGE = (1, 4)
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _separator_row(grid: np.ndarray) -> int:
    grey_rows = [
        r
        for r in range(grid.shape[0] - 1)
        if np.all(grid[r] == GREY) and np.any(grid[r + 1] != GREY)
    ]
    assert len(grey_rows) == 1, "expected exactly one full grey separator row"
    return int(grey_rows[0])


def _legend_from_row(row: np.ndarray) -> list[int]:
    seq = []
    for value in row:
        color = int(value)
        if color == BLACK:
            break
        assert color != GREY, "legend row may not contain grey cells"
        seq.append(color)
    assert len(seq) >= 2, "expected at least two legend colors"
    tail = row[len(seq) :]
    assert np.all(tail == BLACK), "legend colors must occupy the left prefix only"
    return seq


def _source_info(arena: np.ndarray, legend: list[int]) -> tuple[int, int, int, int]:
    h, w = arena.shape
    corners = {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "bl": (h - 1, 0),
        "br": (h - 1, w - 1),
    }
    found = []
    for name, (r, c) in corners.items():
        color = int(arena[r, c])
        if color not in (BLACK, GREY):
            found.append((name, r, c, color))
    assert len(found) == 1, "expected exactly one corner source pixel"
    name, r, c, color = found[0]
    assert color == int(legend[0]), "source color must match the first legend color"
    dr, dc = CORNER_TO_DIR[name]
    return r, c, dr, dc


def _trace_beam(
    arena: np.ndarray,
    legend: list[int],
) -> tuple[np.ndarray, list[dict[str, int]], tuple[int, int] | None]:
    h, w = arena.shape
    out = arena.copy()
    r, c, dr, dc = _source_info(arena, legend)
    color_idx = 0
    steps_since_bounce = 0
    visited_states = {(r, c, dr, dc, color_idx)}
    events: list[dict[str, int]] = []

    while True:
        nr = r + dr
        nc = c + dc
        bounced = False
        if nr < 0 or nr >= h:
            dr *= -1
            bounced = True
        if nc < 0 or nc >= w:
            dc *= -1
            bounced = True
        if bounced:
            color_idx = (color_idx + 1) % len(legend)
            steps_since_bounce = 0
            nr = r + dr
            nc = c + dc

        assert 0 <= nr < h and 0 <= nc < w, "beam reflection must stay in bounds"
        if int(arena[nr, nc]) == GREY:
            return out, events, (int(nr), int(nc))

        if int(out[nr, nc]) == BLACK:
            out[nr, nc] = int(legend[color_idx])
        steps_since_bounce += 1
        events.append(
            {
                "r": int(nr),
                "c": int(nc),
                "color_idx": int(color_idx),
                "steps_since_bounce": int(steps_since_bounce),
            }
        )
        r, c = nr, nc
        state = (r, c, dr, dc, color_idx)
        if state in visited_states:
            raise AssertionError("beam repeats forever before hitting an absorber")
        visited_states.add(state)

    return out, events, None


def _trace_empty_path(
    h: int,
    w: int,
    source_corner: str,
    legend_len: int,
) -> list[dict[str, int]]:
    arena = np.full((h, w), BLACK, dtype=int)
    corners = {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "bl": (h - 1, 0),
        "br": (h - 1, w - 1),
    }
    sr, sc = corners[source_corner]
    arena[sr, sc] = 1
    r, c = sr, sc
    dr, dc = CORNER_TO_DIR[source_corner]
    color_idx = 0
    bounce_count = 0
    steps_since_bounce = 0
    seen = {(r, c, dr, dc)}
    path = []

    while True:
        nr = r + dr
        nc = c + dc
        bounced = False
        if nr < 0 or nr >= h:
            dr *= -1
            bounced = True
        if nc < 0 or nc >= w:
            dc *= -1
            bounced = True
        if bounced:
            bounce_count += 1
            color_idx = (color_idx + 1) % legend_len
            steps_since_bounce = 0
            nr = r + dr
            nc = c + dc

        if (nr in (0, h - 1)) and (nc in (0, w - 1)) and (nr, nc) != (sr, sc):
            # Avoid corner impacts in generated tasks; they complicate bounce counting.
            break

        steps_since_bounce += 1
        path.append(
            {
                "r": int(nr),
                "c": int(nc),
                "bounce_count": int(bounce_count),
                "color_idx": int(color_idx),
                "segment_step": int(steps_since_bounce),
            }
        )
        r, c = nr, nc
        state = (r, c, dr, dc)
        if state in seen:
            break
        seen.add(state)

    return path


def _candidate_stop_cells(
    h: int,
    w: int,
    source_corner: str,
    legend_len: int,
) -> tuple[list[dict[str, int]], list[dict[str, int]], list[dict[str, int]]]:
    path = _trace_empty_path(h, w, source_corner, legend_len)
    counts: dict[tuple[int, int], int] = {}
    for item in path:
        key = (int(item["r"]), int(item["c"]))
        counts[key] = counts.get(key, 0) + 1

    safe = [
        item
        for item in path
        if counts[(int(item["r"]), int(item["c"]))] == 1
        and int(item["bounce_count"]) >= STOP_BOUNCES_MIN
        and int(item["segment_step"]) >= 2
    ]
    cycle = [
        item
        for item in safe
        if int(item["bounce_count"]) >= legend_len and int(item["segment_step"]) >= 2
    ]
    return path, safe, cycle


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a 45-degree beam through the arena, changing color after each wall bounce.

    Input structure:
    - The bottom of the grid has a full grey separator row, with one legend row beneath it.
    - The legend row starts with a left-aligned sequence of colored cells and is black elsewhere.
    - Above the separator is a black arena containing grey absorber blocks and one colored corner source pixel.

    Rule:
    - Read the legend colors from left to right; the source color equals the first legend color.
    - Launch the beam diagonally from that corner into the arena. Whenever the next move would leave the arena,
      reflect off the touched wall and advance to the next legend color, cycling when needed.
    - Paint only black arena cells visited by each beam segment, and stop immediately before entering a grey absorber cell.
      Leave absorber blocks, the separator, the legend, and the source pixel unchanged.

    Key insight:
    - The legend is a bounce-driven color schedule: beam color depends on the number of wall reflections, not on step count or position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"

    sep = _separator_row(g)
    legend = _legend_from_row(g[sep + 1])
    arena = g[:sep, :]
    assert arena.size > 0, "arena must be non-empty"

    out_arena, _, _ = _trace_beam(arena, legend)
    out = g.copy()
    out[:sep, :] = out_arena
    return out


def _place_distractor_blocks(
    arena: np.ndarray,
    forbidden: set[tuple[int, int]],
    count: int,
) -> None:
    h, w = arena.shape
    for _ in range(count):
        for _ in range(60):
            bh = random.randint(*BLOCK_SIDE_RANGE)
            bw = random.randint(*BLOCK_SIDE_RANGE)
            if bh > h or bw > w:
                continue
            r0 = random.randint(0, h - bh)
            c0 = random.randint(0, w - bw)
            cells = {(r, c) for r in range(r0, r0 + bh) for c in range(c0, c0 + bw)}
            if cells & forbidden:
                continue
            if any(arena[r, c] != BLACK for r, c in cells):
                continue
            for r, c in cells:
                arena[r, c] = GREY
            forbidden.update(cells)
            break


def _assemble_grid(
    arena_h: int,
    arena_w: int,
    legend: list[int],
    source_corner: str,
    stopper: tuple[int, int],
    path_cells: set[tuple[int, int]],
    distractor_count: int,
) -> np.ndarray:
    arena = np.full((arena_h, arena_w), BLACK, dtype=int)
    corners = {
        "tl": (0, 0),
        "tr": (0, arena_w - 1),
        "bl": (arena_h - 1, 0),
        "br": (arena_h - 1, arena_w - 1),
    }
    sr, sc = corners[source_corner]
    arena[sr, sc] = int(legend[0])
    arena[stopper] = GREY

    forbidden = set(path_cells)
    forbidden.add((sr, sc))
    _place_distractor_blocks(arena, forbidden, distractor_count)

    grid = np.full((arena_h + 2, arena_w), BLACK, dtype=int)
    grid[:arena_h, :] = arena
    grid[arena_h, :] = GREY
    grid[arena_h + 1, : len(legend)] = np.array(legend, dtype=int)
    return grid


def _semantic_checks(inp: np.ndarray, target_stop: dict[str, int], expect_cycle: bool) -> None:
    sep = _separator_row(inp)
    legend = _legend_from_row(inp[sep + 1])
    arena = inp[:sep, :]
    out = main(inp)

    changed = (out[:sep, :] != arena)
    assert np.count_nonzero(changed) >= 3, "beam must paint a visible path"
    painted_colors = set(map(int, np.unique(out[:sep, :][changed])))
    assert len(painted_colors) >= 2, "examples must require at least two segment colors"

    _path, safe, cycle = _candidate_stop_cells(arena.shape[0], arena.shape[1], _corner_name(arena), len(legend))
    assert safe, "generator must produce reachable stop cells after multiple bounces"
    assert any(
        int(item["r"]) == int(target_stop["r"]) and int(item["c"]) == int(target_stop["c"])
        for item in safe
    ), "chosen stopper must lie on the valid beam path"
    if expect_cycle:
        assert any(
            int(item["r"]) == int(target_stop["r"]) and int(item["c"]) == int(target_stop["c"])
            for item in cycle
        ), "cycle branch must visibly reuse the legend sequence"

    predicted, _, hit = _trace_beam(arena, legend)
    assert hit == (int(target_stop["r"]), int(target_stop["c"])), "beam must first hit the chosen absorber"
    assert np.array_equal(predicted, out[:sep, :]), "beam simulator must agree with main"


def _corner_name(arena: np.ndarray) -> str:
    h, w = arena.shape
    corners = {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "bl": (h - 1, 0),
        "br": (h - 1, w - 1),
    }
    found = [name for name, (r, c) in corners.items() if int(arena[r, c]) not in (BLACK, GREY)]
    assert len(found) == 1
    return found[0]


def _gen_attempt() -> np.ndarray | None:
    arena_h = random.randint(*ARENA_H_RANGE)
    arena_w = random.randint(*ARENA_W_RANGE)
    legend_len = random.randint(*LEGEND_LEN_RANGE)
    source_corner = random.choice(list(CORNER_TO_DIR))
    legend = random.sample(PALETTE, legend_len)

    path, safe, cycle = _candidate_stop_cells(arena_h, arena_w, source_corner, legend_len)
    if not safe:
        return None

    use_cycle = bool(cycle) and random.random() < CYCLE_BRANCH_PROB
    pool = cycle if use_cycle else safe
    stopper_info = random.choice(pool)
    stopper = (int(stopper_info["r"]), int(stopper_info["c"]))
    path_cells = {
        (int(item["r"]), int(item["c"]))
        for item in path
    }

    distractor_count = random.randint(*DISTRACTOR_BLOCK_COUNT)
    inp = _assemble_grid(
        arena_h,
        arena_w,
        legend,
        source_corner,
        stopper,
        path_cells,
        distractor_count,
    )

    if max(inp.shape) > 30:
        return None

    try:
        _semantic_checks(inp, stopper_info, use_cycle)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid beam-billiards ARC input")
