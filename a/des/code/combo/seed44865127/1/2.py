from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
DIRS4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8: tuple[Coord, ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        comp: list[Coord] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _path_degrees(cells: list[Coord]) -> dict[Coord, int]:
    cell_set = set(cells)
    return {
        cell: sum((nb in cell_set) for nb in _neighbors4(*cell))
        for cell in cell_set
    }


def _fragment_endpoints(cells: list[Coord]) -> list[Coord]:
    if len(cells) == 1:
        return [cells[0]]
    degrees = _path_degrees(cells)
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    assert endpoints, "fragment must have at least one endpoint"
    assert all(degree in (1, 2) for degree in degrees.values()), "fragment must not branch"
    assert len(endpoints) == 2, "non-singleton fragment must have exactly two endpoints"
    return endpoints


def _ordered_path(cells: list[Coord]) -> list[Coord]:
    if len(cells) == 1:
        return [cells[0]]

    cell_set = set(cells)
    degrees = _path_degrees(cells)
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must not branch"

    ordered: list[Coord] = []
    prev: Coord | None = None
    cur = endpoints[0]
    while True:
        ordered.append(cur)
        next_cells = [nb for nb in _neighbors4(*cur) if nb in cell_set and nb != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path traversal must be unique"
        prev, cur = cur, next_cells[0]

    assert len(ordered) == len(cell_set), "path traversal must visit every cell"
    return ordered


def _gap_cells(a: Coord, b: Coord) -> list[Coord] | None:
    if a[0] == b[0]:
        c0, c1 = sorted((a[1], b[1]))
        gap = c1 - c0 - 1
        if 1 <= gap <= 3:
            return [(a[0], c) for c in range(c0 + 1, c1)]
        return None
    if a[1] == b[1]:
        r0, r1 = sorted((a[0], b[0]))
        gap = r1 - r0 - 1
        if 1 <= gap <= 3:
            return [(r, a[1]) for r in range(r0 + 1, r1)]
        return None
    return None


def _collect_connections(grid: np.ndarray, bg: int) -> list[tuple[Coord, Coord, list[Coord]]]:
    comps = _components(grid != bg)
    endpoints: list[tuple[int, Coord]] = []
    for comp_id, comp in enumerate(comps):
        for cell in _fragment_endpoints(comp):
            endpoints.append((comp_id, cell))

    candidates: list[tuple[int, Coord, Coord, int, int, list[Coord]]] = []
    for i in range(len(endpoints)):
        comp_i, a = endpoints[i]
        color = int(grid[a[0], a[1]])
        for j in range(i + 1, len(endpoints)):
            comp_j, b = endpoints[j]
            if comp_i == comp_j:
                continue
            if int(grid[b[0], b[1]]) != color:
                continue
            gap = _gap_cells(a, b)
            if gap is None:
                continue
            if any(int(grid[r, c]) != bg for r, c in gap):
                continue
            candidates.append((len(gap), min(a, b), max(a, b), i, j, gap))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used: set[int] = set()
    selected: list[tuple[Coord, Coord, list[Coord]]] = []
    for _gap_len, _lo, _hi, i, j, gap in candidates:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        selected.append((endpoints[i][1], endpoints[j][1], gap))
    return selected


def _fill_connections(grid: np.ndarray, bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for a, _b, gap in _collect_connections(grid, bg):
        color = int(grid[a[0], a[1]])
        for r, c in gap:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-colored aligned path fragments across short background gaps,
    then reverse the color sequence along each completed path.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    connected = _fill_connections(g, bg)
    out = connected.copy()
    for comp in _components(connected != bg):
        path = _ordered_path(comp)
        values = [int(connected[r, c]) for r, c in path]
        for (r, c), value in zip(path, reversed(values), strict=False):
            out[r, c] = value
    return out


# --------------- generator ---------------

GRID_SIZE_MIN = 15
GRID_SIZE_MAX = 22
PATH_COUNT_MIN = 1
PATH_COUNT_MAX = 3
PATH_LEN_MIN = 11
PATH_LEN_MAX = 24
SEGMENT_COUNT_MIN = 3
SEGMENT_COUNT_MAX = 6
SEGMENT_STEPS_MIN = 2
SEGMENT_STEPS_MAX = 6
MAX_GENERATION_ATTEMPTS = 400
PATH_BUILD_ATTEMPTS = 80


def _turn_dirs(direction: Coord | None) -> list[Coord]:
    if direction is None:
        dirs = list(DIRS4)
    else:
        dr, dc = direction
        dirs = [(-dc, dr), (dc, -dr)]
    random.shuffle(dirs)
    return dirs


def _sample_run_lengths(n_segments: int, total_steps: int) -> list[int]:
    runs: list[int] = []
    remaining = total_steps
    for i in range(n_segments):
        slots = n_segments - i - 1
        lo = max(SEGMENT_STEPS_MIN, remaining - SEGMENT_STEPS_MAX * slots)
        hi = min(SEGMENT_STEPS_MAX, remaining - SEGMENT_STEPS_MIN * slots)
        if lo > hi:
            raise AssertionError("invalid run allocation")
        runs.append(random.randint(lo, hi))
        remaining -= runs[-1]
    random.shuffle(runs)
    return runs


def _sample_run_plan() -> list[int]:
    total_steps = random.randint(PATH_LEN_MIN - 1, PATH_LEN_MAX - 1)
    valid_counts = [
        n_segments
        for n_segments in range(SEGMENT_COUNT_MIN, SEGMENT_COUNT_MAX + 1)
        if SEGMENT_STEPS_MIN * n_segments <= total_steps <= SEGMENT_STEPS_MAX * n_segments
    ]
    n_segments = random.choice(valid_counts)
    return _sample_run_lengths(n_segments, total_steps)


def _extend_segment(
    path: list[Coord],
    used: set[Coord],
    direction: Coord,
    steps: int,
    blocked: set[Coord],
    height: int,
    width: int,
) -> tuple[list[Coord], set[Coord]] | None:
    dr, dc = direction
    new_path = list(path)
    new_used = set(used)
    r, c = new_path[-1]

    for _ in range(steps):
        nr, nc = r + dr, c + dc
        if not (1 <= nr < height - 1 and 1 <= nc < width - 1):
            return None
        if (nr, nc) in blocked or (nr, nc) in new_used:
            return None
        if any((ar, ac) in new_used and (ar, ac) != (r, c) for ar, ac in _neighbors4(nr, nc)):
            return None
        new_path.append((nr, nc))
        new_used.add((nr, nc))
        r, c = nr, nc

    return new_path, new_used


def _build_path(height: int, width: int, blocked: set[Coord], run_lengths: list[int]) -> list[Coord] | None:
    starts = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if (r, c) not in blocked
    ]
    random.shuffle(starts)

    def search(
        seg_i: int,
        prev_dir: Coord | None,
        path: list[Coord],
        used: set[Coord],
    ) -> list[Coord] | None:
        if seg_i == len(run_lengths):
            return path
        for direction in _turn_dirs(prev_dir):
            extended = _extend_segment(path, used, direction, run_lengths[seg_i], blocked, height, width)
            if extended is None:
                continue
            found = search(seg_i + 1, direction, extended[0], extended[1])
            if found is not None:
                return found
        return None

    for start in starts:
        found = search(0, None, [start], {start})
        if found is not None:
            return found
    return None


def _path_runs(path: list[Coord]) -> list[tuple[int, int, Coord]]:
    dirs = [(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1)]
    runs: list[tuple[int, int, Coord]] = []
    run_start = 0
    cur_dir = dirs[0]
    for step_i in range(1, len(dirs)):
        if dirs[step_i] != cur_dir:
            runs.append((run_start, step_i, cur_dir))
            run_start = step_i
            cur_dir = dirs[step_i]
    runs.append((run_start, len(dirs), cur_dir))
    return runs


def _choose_gaps(path: list[Coord]) -> list[tuple[int, int]] | None:
    eligible: list[tuple[int, int, int]] = []
    for run_start, run_end, _direction in _path_runs(path):
        run_len = run_end - run_start + 1
        max_gap = min(3, run_len - 4)
        if max_gap >= 1:
            eligible.append((run_start, run_end, max_gap))
    if not eligible:
        return None

    gap_count = 1 if len(eligible) == 1 else random.choice((1, 2))
    chosen_runs = random.sample(eligible, gap_count)
    gaps: list[tuple[int, int]] = []
    for run_start, run_end, max_gap in chosen_runs:
        gap_len = random.randint(1, max_gap)
        gap_start = random.randint(run_start + 2, run_end - gap_len - 1)
        gaps.append((gap_start, gap_len))
    gaps.sort()
    return gaps


def _gap_metadata(path: list[Coord], gaps: list[tuple[int, int]]) -> tuple[set[int], set[int], set[frozenset[Coord]]]:
    removed: set[int] = set()
    protected = {0, len(path) - 1}
    pairs: set[frozenset[Coord]] = set()
    for gap_start, gap_len in gaps:
        removed.update(range(gap_start, gap_start + gap_len))
        protected.add(gap_start - 1)
        protected.add(gap_start + gap_len)
        pairs.add(frozenset((path[gap_start - 1], path[gap_start + gap_len])))
    return removed, protected, pairs


def _sample_sequence(
    path_len: int,
    base: int,
    bg: int,
    removed: set[int],
    protected: set[int],
) -> list[int] | None:
    visible = [i for i in range(path_len) if i not in removed]
    flexible = [i for i in visible if i not in protected]
    if not flexible:
        return None

    non_bg = [color for color in range(10) if color != bg]
    extras = [color for color in non_bg if color != base]
    for _ in range(60):
        extra_count = random.randint(1, min(3, len(extras)))
        palette = [base] + random.sample(extras, extra_count)
        seq = [base if i in removed else random.choice(palette) for i in range(path_len)]
        for i in protected:
            seq[i] = base
        if all(seq[i] == base for i in visible):
            seq[random.choice(flexible)] = random.choice(palette[1:])
        if len({seq[i] for i in visible}) < 2:
            continue
        if seq == seq[::-1]:
            continue
        return seq
    return None


def _halo(cells: list[Coord], height: int, width: int) -> set[Coord]:
    out: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    out.add((nr, nc))
    return out


def _render_input(height: int, width: int, bg: int, paths: list[dict]) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for item in paths:
        path = item["path"]
        seq = item["sequence"]
        removed = item["removed"]
        for i, (r, c) in enumerate(path):
            if i in removed:
                continue
            grid[r, c] = seq[i]
    return grid


def _expected_mask(height: int, width: int, paths: list[dict]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for item in paths:
        for r, c in item["path"]:
            mask[r, c] = True
    return mask


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    width = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    bg = random.randint(0, 9)

    path_count_max = PATH_COUNT_MAX if height * width >= 300 else 2
    path_count = random.randint(PATH_COUNT_MIN, path_count_max)
    base_colors = random.sample([color for color in range(10) if color != bg], path_count)

    blocked: set[Coord] = set()
    paths: list[dict] = []
    for base in base_colors:
        built = False
        for _ in range(PATH_BUILD_ATTEMPTS):
            run_lengths = _sample_run_plan()
            path = _build_path(height, width, blocked, run_lengths)
            if path is None:
                continue
            gaps = _choose_gaps(path)
            if gaps is None:
                continue
            removed, protected, pairs = _gap_metadata(path, gaps)
            sequence = _sample_sequence(len(path), base, bg, removed, protected)
            if sequence is None:
                continue
            item = {
                "path": path,
                "base": base,
                "gaps": gaps,
                "removed": removed,
                "pairs": pairs,
                "sequence": sequence,
            }
            paths.append(item)
            blocked |= _halo(path, height, width)
            built = True
            break
        if not built:
            return None

    grid = _render_input(height, width, bg, paths)
    if _bg_color(grid) != bg:
        return None

    expected_pairs: set[frozenset[Coord]] = set()
    for item in paths:
        expected_pairs |= item["pairs"]
    found_pairs = {frozenset((a, b)) for a, b, _gap in _collect_connections(grid, bg)}
    if found_pairs != expected_pairs:
        return None

    connected = _fill_connections(grid, bg)
    if not np.array_equal(connected != bg, _expected_mask(height, width, paths)):
        return None

    try:
        output = main(grid)
    except AssertionError:
        return None
    if np.array_equal(output, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
