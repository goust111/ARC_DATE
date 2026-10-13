from __future__ import annotations

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = tuple(
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
)

_GEN_RNG = np.random.default_rng()

_GRID_MIN = 11
_GRID_MAX = 18
_BBOX_CHOICES = np.array([1, 3, 5], dtype=int)
_BBOX_WEIGHTS = np.array([0.18, 0.44, 0.38], dtype=float)
_GEN_ATTEMPTS = 400
_PLACEMENT_TRIES = 220


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, np.ndarray]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(sorted(cells), dtype=int)
            comps.append((color, coords))

    comps.sort(key=lambda item: (int(item[1][:, 0].min()), int(item[1][:, 1].min()), int(item[0])))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Explode each 4-connected foreground object away from its bbox center.

    Each pixel moves by its signed row/column offset from the object's bbox
    midpoint, so the new position is `old + (old - center)`. Pixels pushed
    outside the grid are dropped.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    comps = _components(g, bg)
    assert comps, "expected at least one foreground object"

    out = np.full_like(g, bg)
    placed: dict[tuple[int, int], int] = {}

    for color, coords in comps:
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        bbox_h = int(maxs[0] - mins[0] + 1)
        bbox_w = int(maxs[1] - mins[1] + 1)
        assert bbox_h <= 5 and bbox_w <= 5, "foreground bbox exceeds puzzle limits"

        center_r = (int(mins[0]) + int(maxs[0])) // 2
        center_c = (int(mins[1]) + int(maxs[1])) // 2

        for r, c in coords.tolist():
            nr = 2 * int(r) - center_r
            nc = 2 * int(c) - center_c
            if 0 <= nr < h and 0 <= nc < w:
                key = (nr, nc)
                prev = placed.get(key)
                assert prev is None or prev == color, "exploded objects must not overlap"
                placed[key] = color

    for (r, c), color in placed.items():
        out[r, c] = color
    return out


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _trace_path(
    start: tuple[int, int], end: tuple[int, int], horizontal_first: bool
) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    path = [(r, c)]

    if horizontal_first:
        while c != ec:
            c += 1 if ec > c else -1
            path.append((r, c))
        while r != er:
            r += 1 if er > r else -1
            path.append((r, c))
    else:
        while r != er:
            r += 1 if er > r else -1
            path.append((r, c))
        while c != ec:
            c += 1 if ec > c else -1
            path.append((r, c))

    return path


def _shape_cells(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    center = (h // 2, w // 2)
    cells = {center}
    anchors: list[tuple[int, int]] = []

    if h > 1:
        anchors.extend([(0, int(rng.integers(w))), (h - 1, int(rng.integers(w)))])
    if w > 1:
        anchors.extend([(int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)])

    rng.shuffle(anchors)
    for anchor in anchors:
        for cell in _trace_path(anchor, center, bool(rng.integers(2))):
            cells.add(cell)

    max_extra = h * w - len(cells)
    if max_extra > 0:
        extra = int(rng.integers(0, max_extra + 1))
        if extra == max_extra and h * w > 5 and rng.random() < 0.7:
            extra -= 1
        for _ in range(extra):
            frontier: list[tuple[int, int]] = []
            frontier_seen: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells and (nr, nc) not in frontier_seen:
                        frontier.append((nr, nc))
                        frontier_seen.add((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])

    return np.array(sorted(cells), dtype=int)


def _explode_relative(coords: np.ndarray) -> np.ndarray:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center_r = (int(mins[0]) + int(maxs[0])) // 2
    center_c = (int(mins[1]) + int(maxs[1])) // 2
    out = coords.copy()
    out[:, 0] = 2 * out[:, 0] - center_r
    out[:, 1] = 2 * out[:, 1] - center_c
    return out


def _sample_interval(interval: tuple[int, int] | None, rng: np.random.Generator) -> int | None:
    if interval is None:
        return None
    lo, hi = interval
    if lo > hi:
        return None
    return int(rng.integers(lo, hi + 1))


def _axis_intervals(
    limit: int, input_max: int, out_min: int, out_max: int
) -> tuple[tuple[int, int], tuple[int, int] | None, tuple[int, int] | None, tuple[int, int] | None]:
    input_range = (0, limit - 1 - input_max)
    if input_range[1] < input_range[0]:
        return input_range, None, None, None

    safe_low = max(0, -out_min)
    safe_high = min(input_range[1], limit - 1 - out_max)
    safe = (safe_low, safe_high) if safe_low <= safe_high else None

    low_clip_high = min(input_range[1], -out_min - 1)
    low_clip = (0, low_clip_high) if low_clip_high >= 0 else None

    high_clip_low = max(0, limit - out_max)
    high_clip = (high_clip_low, input_range[1]) if high_clip_low <= input_range[1] else None

    return input_range, safe, low_clip, high_clip


def _sample_base(
    rel_in: np.ndarray, rel_out: np.ndarray, h: int, w: int, must_clip: bool, rng: np.random.Generator
) -> tuple[int, int] | None:
    in_rmax = int(rel_in[:, 0].max())
    in_cmax = int(rel_in[:, 1].max())
    out_rmin = int(rel_out[:, 0].min())
    out_rmax = int(rel_out[:, 0].max())
    out_cmin = int(rel_out[:, 1].min())
    out_cmax = int(rel_out[:, 1].max())

    r_input, r_safe, r_low, r_high = _axis_intervals(h, in_rmax, out_rmin, out_rmax)
    c_input, c_safe, c_low, c_high = _axis_intervals(w, in_cmax, out_cmin, out_cmax)

    if not must_clip:
        base_r = _sample_interval(r_safe, rng)
        base_c = _sample_interval(c_safe, rng)
        if base_r is None or base_c is None:
            return None
        return base_r, base_c

    modes: list[tuple[str, tuple[int, int]]] = []
    if r_low is not None:
        modes.append(("r", r_low))
    if r_high is not None:
        modes.append(("r", r_high))
    if c_low is not None:
        modes.append(("c", c_low))
    if c_high is not None:
        modes.append(("c", c_high))
    if not modes:
        return None

    rng.shuffle(modes)
    axis, clip_interval = modes[0]
    if axis == "r":
        base_r = _sample_interval(clip_interval, rng)
        base_c = _sample_interval(c_safe if c_safe is not None else c_input, rng)
    else:
        base_r = _sample_interval(r_safe if r_safe is not None else r_input, rng)
        base_c = _sample_interval(clip_interval, rng)

    if base_r is None or base_c is None:
        return None
    return base_r, base_c


def _expanded_cells(
    cells: set[tuple[int, int]], shape: tuple[int, int], dirs: tuple[tuple[int, int], ...]
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        out.add((r, c))
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if _in_bounds(shape, nr, nc):
                out.add((nr, nc))
    return out


def _to_absolute(coords: np.ndarray, base_r: int, base_c: int) -> np.ndarray:
    out = coords.copy()
    out[:, 0] += base_r
    out[:, 1] += base_c
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(_GRID_MIN, _GRID_MAX + 1))
    w = int(rng.integers(_GRID_MIN, _GRID_MAX + 1))
    min_dim = min(h, w)

    if min_dim <= 12:
        max_objects = 3
    elif min_dim <= 15:
        max_objects = 4
    else:
        max_objects = 5
    object_count = int(rng.integers(2, max_objects + 1))

    need_clipped = bool(rng.integers(2))
    clip_flags = [False] * object_count
    if need_clipped:
        clip_flags[0] = True
        rng.shuffle(clip_flags)

    bg = int(rng.integers(10))
    palette = [color for color in range(10) if color != bg]
    rng.shuffle(palette)
    colors = palette[:object_count]

    grid = np.full((h, w), bg, dtype=int)
    max_foreground = max(8, (h * w) // 3)
    used_area = 0
    input_forbidden: set[tuple[int, int]] = set()
    output_forbidden: set[tuple[int, int]] = set()
    clipped_seen = 0

    for color, must_clip in zip(colors, clip_flags):
        placed = False
        for _ in range(_PLACEMENT_TRIES):
            box_h = int(rng.choice(_BBOX_CHOICES, p=_BBOX_WEIGHTS))
            box_w = int(rng.choice(_BBOX_CHOICES, p=_BBOX_WEIGHTS))
            if box_h == 1 and box_w == 1:
                continue

            rel_in = _shape_cells(box_h, box_w, rng)
            if used_area + int(len(rel_in)) > max_foreground:
                continue

            rel_out = _explode_relative(rel_in)
            base = _sample_base(rel_in, rel_out, h, w, must_clip, rng)
            if base is None:
                continue
            base_r, base_c = base

            abs_in = _to_absolute(rel_in, base_r, base_c)
            input_cells = {tuple(map(int, cell)) for cell in abs_in.tolist()}
            if any(cell in input_forbidden for cell in input_cells):
                continue

            abs_out = _to_absolute(rel_out, base_r, base_c)
            visible_out = {
                (int(r), int(c))
                for r, c in abs_out.tolist()
                if _in_bounds((h, w), int(r), int(c))
            }
            clipped = len(visible_out) < len(abs_out)
            if clipped != must_clip:
                continue
            if not visible_out:
                continue
            if all((int(r), int(c)) in input_cells for r, c in visible_out):
                continue
            if any(cell in output_forbidden for cell in visible_out):
                continue

            for r, c in input_cells:
                grid[r, c] = color

            used_area += len(input_cells)
            input_forbidden.update(_expanded_cells(input_cells, (h, w), DIRS8))
            output_forbidden.update(_expanded_cells(visible_out, (h, w), DIRS4))
            clipped_seen += int(clipped)
            placed = True
            break

        if not placed:
            return None

    if used_area == 0 or clipped_seen != int(need_clipped):
        return None
    if _background_color(grid) != bg:
        return None

    comps = _components(grid, bg)
    if len(comps) != object_count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out != grid):
        return None
    input_fg = int(np.count_nonzero(grid != bg))
    output_fg = int(np.count_nonzero(out != bg))
    if need_clipped != (output_fg < input_fg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(_GEN_RNG)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {_GEN_ATTEMPTS} attempts")
