from __future__ import annotations

import random

import numpy as np


_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

_GEN_BG = 0
_GEN_PIPE = 5
_GEN_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
_GEN_MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _pipe_color(grid: np.ndarray, bg: int) -> int:
    values, counts = np.unique(grid, return_counts=True)
    candidates = [(int(cnt), int(value)) for value, cnt in zip(values, counts) if int(value) != int(bg)]
    assert candidates, "expected at least one foreground color"
    return max(candidates)[1]


def _pipe_directions(pipe_mask: np.ndarray) -> dict[tuple[int, int], list[int]]:
    directions: dict[tuple[int, int], list[int]] = {}
    for r0, c0 in np.argwhere(pipe_mask):
        r = int(r0)
        c = int(c0)
        cell_dirs: list[int] = []
        for d, (dr, dc) in enumerate(_DIRS):
            nr = r + dr
            nc = c + dc
            if _in_bounds(pipe_mask.shape, nr, nc) and bool(pipe_mask[nr, nc]):
                cell_dirs.append(d)
        directions[(r, c)] = cell_dirs
    return directions


def _find_sources(
    grid: np.ndarray,
    bg: int,
    pipe_color: int,
    cell_dirs: dict[tuple[int, int], list[int]],
) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    for (r, c), dirs in cell_dirs.items():
        if len(dirs) != 1:
            continue

        heading = int(dirs[0])
        dr, dc = _DIRS[heading]
        sr = r - dr
        sc = c - dc
        if not _in_bounds(grid.shape, sr, sc):
            continue

        source_color = int(grid[sr, sc])
        if source_color in {int(bg), int(pipe_color)}:
            continue

        sources.append(
            {
                "injector": (int(sr), int(sc)),
                "start": (int(r), int(c)),
                "heading": int(heading),
                "color": int(source_color),
            }
        )

    sources.sort(key=lambda src: (src["injector"], src["start"], src["color"]))
    return sources


def _advance(
    cell: tuple[int, int],
    heading: int,
    cell_dirs: dict[tuple[int, int], list[int]],
) -> tuple[tuple[int, int], int, bool, bool] | None:
    dirs = cell_dirs[cell]
    back = (int(heading) + 2) % 4
    options = [d for d in dirs if d != back]
    if not options:
        return None

    degree = len(dirs)
    turned_at_t = False
    used_cross = False

    if len(options) == 1:
        next_dir = int(options[0])
    elif len(options) == 2:
        right = (int(heading) + 1) % 4
        left = (int(heading) - 1) % 4
        if int(heading) in options:
            next_dir = int(heading)
        elif right in options:
            next_dir = int(right)
            turned_at_t = degree == 3
        else:
            assert left in options
            next_dir = int(left)
            turned_at_t = degree == 3
    else:
        assert len(options) == 3 and degree == 4 and int(heading) in options
        next_dir = int(heading)
        used_cross = True

    dr, dc = _DIRS[next_dir]
    return (cell[0] + dr, cell[1] + dc), int(next_dir), bool(turned_at_t), bool(used_cross)


def _simulate(
    grid: np.ndarray,
    bg: int,
    pipe_color: int,
) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    pipe_mask = g == int(pipe_color)
    cell_dirs = _pipe_directions(pipe_mask)
    sources = _find_sources(g, bg, pipe_color, cell_dirs)

    out = g.copy()
    owner = np.full(g.shape, -1, dtype=int)

    source_cross = [False] * len(sources)
    source_t_turn = [False] * len(sources)
    blocked_by_other = 0
    tie_count = 0

    active: list[tuple[int, tuple[int, int], int]] = []
    for idx, src in enumerate(sources):
        r, c = src["start"]
        if owner[r, c] != -1:
            tie_count += 1
            continue
        owner[r, c] = int(idx)
        out[r, c] = int(src["color"])
        active.append((int(idx), (int(r), int(c)), int(src["heading"])))

    while active:
        proposals: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for idx, cell, heading in active:
            step = _advance(cell, heading, cell_dirs)
            if step is None:
                continue

            target, next_heading, turned_at_t, used_cross = step
            source_cross[idx] = bool(source_cross[idx] or used_cross)
            source_t_turn[idx] = bool(source_t_turn[idx] or turned_at_t)

            tr, tc = target
            if owner[tr, tc] != -1:
                if owner[tr, tc] != int(idx):
                    blocked_by_other += 1
                continue

            proposals.setdefault((int(tr), int(tc)), []).append((int(idx), int(next_heading)))

        next_active: list[tuple[int, tuple[int, int], int]] = []
        for target, claims in proposals.items():
            if len(claims) > 1:
                tie_count += 1
                continue

            idx, next_heading = claims[0]
            tr, tc = target
            owner[tr, tc] = int(idx)
            out[tr, tc] = int(sources[idx]["color"])
            next_active.append((int(idx), (int(tr), int(tc)), int(next_heading)))

        active = next_active

    junction_counts = {3: 0, 4: 0}
    for dirs in cell_dirs.values():
        if len(dirs) in junction_counts:
            junction_counts[len(dirs)] += 1

    meta = {
        "sources": sources,
        "source_cross": source_cross,
        "source_t_turn": source_t_turn,
        "blocked_by_other": int(blocked_by_other),
        "tie_count": int(tie_count),
        "junction_counts": junction_counts,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inject colored fluids into a grey pipe network and let each flow follow the junction geometry.

    Input structure:
    - The background is black, with a one-pixel-wide grey orthogonal pipe network.
    - Some pipe endpoints are dead ends, and a single colored injector pixel sits
      just outside each such endpoint.
    - Junction cells have either three pipe neighbors (T junctions) or four pipe
      neighbors (cross junctions).

    Rule:
    - Each injector sends its color into the adjacent pipe endpoint and paints the
      visited grey pipe cells with that color.
    - On ordinary pipe cells, the flow follows the only forward continuation.
    - At a T junction, the flow keeps going straight if possible; otherwise it
      turns right relative to its incoming direction, and only turns left if the
      right turn is absent.
    - At a four-way cross, the flow always goes straight.
    - If a flow tries to enter a pipe cell that was already reached earlier by a
      different color, it stops there; injector pixels themselves stay unchanged.

    Key insight:
    - The pipe color is only the neutral wiring. To solve the puzzle, one must
      simulate directional motion through the network: local junction shape fixes
      the route, and conflicts are resolved by first arrival along the pipe.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    pipe_color = _pipe_color(g, bg)
    out, _meta = _simulate(g, bg, pipe_color)
    return out


def _draw_segment(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        grid[r0, lo : hi + 1] = int(color)
    else:
        assert c0 == c1
        lo, hi = sorted((r0, r1))
        grid[lo : hi + 1, c0] = int(color)


def _sample_lengths() -> tuple[int, int, int, int, int, int] | None:
    stem_len = random.randint(2, 5)
    between_junctions = random.randint(2, 4)

    east_t_options = [
        length
        for length in range(stem_len + between_junctions + 2, 10)
        if length % 2 != (stem_len + between_junctions) % 2
    ]
    if not east_t_options:
        return None

    east_t = random.choice(east_t_options)
    west_t = random.randint(stem_len + between_junctions + 1, 9)
    west_cross = random.randint(stem_len + 1, 8)
    east_cross = random.randint(stem_len + 1, 8)
    return stem_len, between_junctions, west_cross, east_cross, west_t, east_t


def _source_pattern() -> list[str]:
    patterns = [
        ["south", "t_east", "t_west", "cross_west"],
        ["south", "t_east", "t_west", "cross_east"],
        ["south", "t_east", "t_west", "cross_west", "cross_east"],
        ["south", "t_east", "cross_west", "cross_east"],
    ]
    return list(random.choice(patterns))


def _build_grid() -> np.ndarray | None:
    sampled = _sample_lengths()
    if sampled is None:
        return None

    stem_len, between_junctions, west_cross, east_cross, west_t, east_t = sampled
    source_names = _source_pattern()
    source_colors = random.sample(_GEN_COLOR_POOL, len(source_names))
    color_by_source = {name: color for name, color in zip(source_names, source_colors)}

    top = random.randint(1, 4)
    bottom = random.randint(1, 5)
    left = random.randint(1, 4)
    right = random.randint(1, 5)

    anchor_r = top + between_junctions
    anchor_c = left + max(west_cross, west_t) + 1
    height = top + between_junctions + stem_len + bottom + 2
    width = left + max(west_cross, west_t) + max(east_cross, east_t) + right + 3
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), _GEN_BG, dtype=int)

    cross = (anchor_r, anchor_c)
    tee = (anchor_r - between_junctions, anchor_c)
    endpoints = {
        "south": (anchor_r + stem_len, anchor_c),
        "cross_west": (anchor_r, anchor_c - west_cross),
        "cross_east": (anchor_r, anchor_c + east_cross),
        "t_west": (anchor_r - between_junctions, anchor_c - west_t),
        "t_east": (anchor_r - between_junctions, anchor_c + east_t),
    }
    injectors = {
        "south": (endpoints["south"][0] + 1, endpoints["south"][1]),
        "cross_west": (endpoints["cross_west"][0], endpoints["cross_west"][1] - 1),
        "cross_east": (endpoints["cross_east"][0], endpoints["cross_east"][1] + 1),
        "t_west": (endpoints["t_west"][0], endpoints["t_west"][1] - 1),
        "t_east": (endpoints["t_east"][0], endpoints["t_east"][1] + 1),
    }

    _draw_segment(grid, cross, tee, _GEN_PIPE)
    _draw_segment(grid, cross, endpoints["south"], _GEN_PIPE)
    _draw_segment(grid, cross, endpoints["cross_west"], _GEN_PIPE)
    _draw_segment(grid, cross, endpoints["cross_east"], _GEN_PIPE)
    _draw_segment(grid, tee, endpoints["t_west"], _GEN_PIPE)
    _draw_segment(grid, tee, endpoints["t_east"], _GEN_PIPE)

    for name, color in color_by_source.items():
        rr, cc = injectors[name]
        if not _in_bounds(grid.shape, rr, cc):
            return None
        grid[rr, cc] = int(color)

    rotation_k = random.randrange(4)
    if rotation_k:
        grid = np.rot90(grid, rotation_k).copy()
    return grid


def _valid_generated(grid: np.ndarray) -> bool:
    out, meta = _simulate(grid, _GEN_BG, _GEN_PIPE)
    pipe_mask = grid == _GEN_PIPE
    colored_pipe = out[pipe_mask]
    colored_values = {int(v) for v in np.unique(colored_pipe) if int(v) != _GEN_PIPE}

    if meta["tie_count"] != 0:
        return False
    if len(meta["sources"]) < 3:
        return False
    if meta["junction_counts"][3] < 1 or meta["junction_counts"][4] < 1:
        return False
    if sum(bool(x) for x in meta["source_t_turn"]) < 1:
        return False
    if sum(bool(x) for x in meta["source_cross"]) < 1:
        return False
    if not any(cross and turn for cross, turn in zip(meta["source_cross"], meta["source_t_turn"])):
        return False
    if meta["blocked_by_other"] < 1:
        return False
    if len(colored_values) < 2:
        return False
    if np.count_nonzero(pipe_mask & (out != _GEN_PIPE)) < 8:
        return False
    if np.array_equal(grid, out):
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if not np.array_equal(main(grid), out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _build_grid()
        if grid is not None and _valid_generated(grid):
            return grid
    raise RuntimeError(f"failed to generate valid input after {_GEN_MAX_ATTEMPTS} attempts")
