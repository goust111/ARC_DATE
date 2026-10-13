from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PATH_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_MIN = 12
GRID_MAX = 22
PATH_LEN_MIN = 12
PATH_LEN_MAX = 24
PERIOD_MIN = 2
PERIOD_MAX = 6
GEN_RETRIES = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _mask_component(mask: np.ndarray) -> set[tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert coords, "expected a non-empty path"
    seen = {coords[0]}
    stack = [coords[0]]
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors8(r, c):
            if (nr, nc) in seen or not _in_bounds(mask.shape, nr, nc):
                continue
            if not bool(mask[nr, nc]):
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    assert len(seen) == len(coords), "path must be a single 8-connected component"
    return seen


def _extract_path_order(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = _mask_component(mask)
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in coords:
        r, c = cell
        adjacency[cell] = [nbr for nbr in neighbors8(r, c) if nbr in coords]

    degrees = {cell: len(nbrs) for cell, nbrs in adjacency.items()}
    assert all(deg <= 2 for deg in degrees.values()), "path cannot branch"
    endpoints = sorted([cell for cell, deg in degrees.items() if deg == 1])
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    order = [endpoints[0]]
    prev = None
    cur = endpoints[0]
    while True:
        nxt = [nbr for nbr in adjacency[cur] if nbr != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path traversal must be unambiguous"
        prev, cur = cur, nxt[0]
        order.append(cur)
        assert len(order) <= len(coords), "path traversal cannot revisit cells"

    assert len(order) == len(coords), "path traversal must cover the whole curve"
    return order


def _samples_from_grid(grid: np.ndarray, order: list[tuple[int, int]], bg: int) -> dict[int, int]:
    samples: dict[int, int] = {}
    for idx, (r, c) in enumerate(order):
        color = int(grid[r, c])
        if color not in (bg, GREY):
            samples[idx] = color
    assert samples, "expected at least one colored residue"
    return samples


def _period_color_consistent(samples: dict[int, int], period: int) -> bool:
    by_residue: dict[int, int] = {}
    for idx, color in samples.items():
        residue = idx % period
        prev = by_residue.get(residue)
        if prev is not None and prev != color:
            return False
        by_residue[residue] = color
    return True


def _covers_all_residues(samples: dict[int, int], period: int) -> bool:
    return len({idx % period for idx in samples}) == period


def _candidate_cycles(samples: dict[int, int], path_length: int) -> list[tuple[int, tuple[int, ...]]]:
    candidates: list[tuple[int, tuple[int, ...]]] = []
    for period in range(1, path_length + 1):
        if not _period_color_consistent(samples, period):
            continue
        if not _covers_all_residues(samples, period):
            continue
        by_residue: dict[int, int] = {}
        for idx, color in samples.items():
            by_residue[idx % period] = color
        cycle = tuple(by_residue[i] for i in range(period))
        candidates.append((period, cycle))
    return candidates


def _infer_shortest_cycle(samples: dict[int, int], path_length: int) -> tuple[int, tuple[int, ...]]:
    candidates = _candidate_cycles(samples, path_length)
    assert candidates, "samples must determine at least one full periodic coloring"
    return candidates[0]


def _render_colored_path(
    shape: tuple[int, int],
    order: list[tuple[int, int]],
    cycle: tuple[int, ...],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    period = len(cycle)
    for idx, (r, c) in enumerate(order):
        out[r, c] = int(cycle[idx % period])
    return out


def _sequence_has_period(seq: tuple[int, ...], period: int) -> bool:
    size = len(seq)
    return all(seq[i] == seq[(i + period) % size] for i in range(size))


def _minimal_cycle_period(seq: tuple[int, ...]) -> int:
    for period in range(1, len(seq)):
        if _sequence_has_period(seq, period):
            return period
    return len(seq)


def _turn_count(order: list[tuple[int, int]]) -> int:
    steps = [
        (order[i + 1][0] - order[i][0], order[i + 1][1] - order[i][1])
        for i in range(len(order) - 1)
    ]
    return sum(steps[i] != steps[i - 1] for i in range(1, len(steps)))


def _bbox_span(order: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _c in order]
    cols = [c for _r, c in order]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _extension_candidates(
    shape: tuple[int, int],
    used: set[tuple[int, int]],
    endpoint: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for nr, nc in neighbors8(*endpoint):
        if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
            continue
        if (nr, nc) in used:
            continue
        if any((ar, ac) in used and (ar, ac) != endpoint for ar, ac in neighbors8(nr, nc)):
            continue
        out.append((nr, nc))
    return out


def _future_freedom(
    shape: tuple[int, int],
    used: set[tuple[int, int]],
    endpoint: tuple[int, int],
    candidate: tuple[int, int],
) -> int:
    new_used = set(used)
    new_used.add(candidate)
    return len(_extension_candidates(shape, new_used, candidate))


def _generate_path(shape: tuple[int, int], target_len: int) -> list[tuple[int, int]] | None:
    h, w = shape
    for _ in range(300):
        start = (random.randint(1, h - 2), random.randint(1, w - 2))
        order = [start]
        used = {start}
        while len(order) < target_len:
            options = []
            sides = [("left", order[0]), ("right", order[-1])]
            for side, endpoint in sides:
                for cand in _extension_candidates(shape, used, endpoint):
                    score = _future_freedom(shape, used, endpoint, cand)
                    center_margin = min(cand[0], h - 1 - cand[0], cand[1], w - 1 - cand[1])
                    options.append((score, center_margin, random.random(), side, cand))
            if not options:
                break
            options.sort(reverse=True)
            best_score = options[0][0]
            candidates = [item for item in options if item[0] >= best_score - 1]
            _score, _margin, _noise, side, cell = random.choice(candidates)
            if side == "left":
                order.insert(0, cell)
            else:
                order.append(cell)
            used.add(cell)
        if len(order) != target_len:
            continue
        if order[0] > order[-1]:
            order = list(reversed(order))
        span_r, span_c = _bbox_span(order)
        if span_r < 4 or span_c < 4:
            continue
        if _turn_count(order) < 2:
            continue
        return order
    return None


def _make_primitive_cycle(period: int) -> tuple[int, ...]:
    for _ in range(300):
        palette_size = random.randint(2, min(4, period))
        palette = random.sample(PATH_COLORS, palette_size)
        cycle = tuple(random.choice(palette) for _ in range(period))
        if len(set(cycle)) < 2:
            continue
        if _minimal_cycle_period(cycle) != period:
            continue
        return cycle
    raise RuntimeError("failed to construct a primitive color cycle")


def _max_run(indices: set[int]) -> int:
    best = 0
    run = 0
    prev = None
    for idx in sorted(indices):
        if prev is not None and idx == prev + 1:
            run += 1
        else:
            run = 1
        best = max(best, run)
        prev = idx
    return best


def _choose_sample_indices(full_colors: list[int], period: int) -> list[int] | None:
    path_len = len(full_colors)
    residue_positions = {r: list(range(r, path_len, period)) for r in range(period)}
    target_cycle = tuple(full_colors[r] for r in range(period))

    for _ in range(400):
        chosen = {random.choice(residue_positions[r]) for r in range(period)}
        remaining = [idx for idx in range(path_len) if idx not in chosen]
        random.shuffle(remaining)
        samples = {idx: full_colors[idx] for idx in chosen}

        failed = False
        while True:
            shorter = [q for q in range(1, period) if _period_color_consistent(samples, q)]
            if not shorter:
                break
            if len(chosen) >= path_len - 2:
                failed = True
                break

            scored = []
            for idx in remaining:
                test_samples = dict(samples)
                test_samples[idx] = full_colors[idx]
                broken = sum(
                    not _period_color_consistent(test_samples, q)
                    for q in shorter
                )
                if broken == 0:
                    continue
                run = _max_run(set(test_samples))
                scored.append((broken, -run, random.random(), idx))
            if not scored:
                failed = True
                break

            scored.sort(reverse=True)
            best = [item for item in scored if item[0] == scored[0][0]]
            _broken, _run_penalty, _noise, idx = random.choice(best)
            chosen.add(idx)
            remaining.remove(idx)
            samples[idx] = full_colors[idx]

        if failed:
            continue
        if _max_run(chosen) > 2:
            continue
        candidates = _candidate_cycles(samples, path_len)
        if not candidates or candidates[0][0] != period:
            continue
        if candidates[0][1] != target_cycle:
            continue
        if len(chosen) <= period:
            continue
        return sorted(chosen)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor an entire one-stroke curve by recovering the shortest periodic residue pattern along arc length.

    Input structure:
    - A black background contains one width-1, 8-connected, non-branching curve.
    - Uncolored curve cells are grey, while a sparse subset of curve cells already carry non-grey residue colors.
    - The residue colors look irregular in 2D, but every non-background cell belongs to the same path.

    Rule:
    - Serialize the full curve into a one-dimensional sequence by walking from either endpoint to the other.
    - Among all periodic colorings consistent with the observed residue cells, find the shortest fully determined cycle.
    - Recolor every path cell by repeating that cycle along the serialized curve, replacing all remaining grey cells.

    Key insight:
    - The pattern exists only in arc-length order on the path, so the solver must recover the path sequence before inferring the minimal repeating color schedule.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    order = _extract_path_order(g != bg)
    samples = _samples_from_grid(g, order, bg)
    _period, cycle = _infer_shortest_cycle(samples, len(order))
    return _render_colored_path(g.shape, order, cycle, bg)


def _validate_candidate(
    inp: np.ndarray,
    expected: np.ndarray,
    period: int,
    cycle: tuple[int, ...],
) -> None:
    order = _extract_path_order(inp != BG)
    span_r, span_c = _bbox_span(order)
    assert span_r >= 4 and span_c >= 4
    assert _turn_count(order) >= 2
    assert np.any(inp == GREY), "input should keep some grey residue-free cells"
    assert np.any((inp != BG) & (inp != GREY)), "input should contain colored residues"
    assert not np.array_equal(inp, expected), "output should change the grid"
    assert len(set(cycle)) >= 2
    assert _minimal_cycle_period(cycle) == period

    samples = _samples_from_grid(inp, order, BG)
    assert all(not _period_color_consistent(samples, q) for q in range(1, period))
    candidates = _candidate_cycles(samples, len(order))
    assert candidates and candidates[0][0] == period
    assert candidates[0][1] == cycle

    produced = main(inp)
    assert np.array_equal(produced, expected), "main() must reconstruct the hidden full coloring"
    assert np.array_equal(_extract_path_mask(inp), _extract_path_mask(expected))


def _extract_path_mask(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid != BG, dtype=bool)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    max_len = min(PATH_LEN_MAX, max(PATH_LEN_MIN, ((h - 2) * (w - 2)) // 5))
    if max_len < PATH_LEN_MIN:
        return None
    path_len = random.randint(PATH_LEN_MIN, max_len)
    order = _generate_path((h, w), path_len)
    if order is None:
        return None

    max_period = min(PERIOD_MAX, (path_len - 3) // 2)
    if max_period < PERIOD_MIN:
        return None
    period = random.randint(PERIOD_MIN, max_period)
    base_cycle = _make_primitive_cycle(period)
    phase = random.randrange(period)
    cycle = tuple(base_cycle[(phase + i) % period] for i in range(period))
    full_colors = [cycle[i % period] for i in range(path_len)]

    sample_indices = _choose_sample_indices(full_colors, period)
    if sample_indices is None:
        return None

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    sample_set = set(sample_indices)
    for idx, (r, c) in enumerate(order):
        out[r, c] = full_colors[idx]
        inp[r, c] = full_colors[idx] if idx in sample_set else GREY

    try:
        _validate_candidate(inp, out, period, cycle)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic residue path instance")
