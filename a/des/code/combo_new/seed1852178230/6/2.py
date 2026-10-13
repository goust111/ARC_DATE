import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _ordered_path(grid: np.ndarray, comp: list[tuple[int, int]]) -> tuple[str, tuple[int, int], list[tuple[int, int]]]:
    rows = {r for r, _ in comp}
    cols = {c for _, c in comp}
    if len(rows) == 1:
        ordered = sorted(comp, key=lambda p: p[1])
        for i in range(1, len(ordered)):
            assert ordered[i][1] == ordered[i - 1][1] + 1
        return "H", ordered[0], ordered
    if len(cols) == 1:
        ordered = sorted(comp, key=lambda p: p[0])
        for i in range(1, len(ordered)):
            assert ordered[i][0] == ordered[i - 1][0] + 1
        return "V", ordered[0], ordered
    raise AssertionError("each foreground component must be a straight line")


def _extract_runs(grid: np.ndarray, ordered: list[tuple[int, int]]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    cur_color = int(grid[ordered[0]])
    cur_len = 1
    for cell in ordered[1:]:
        color = int(grid[cell])
        if color == cur_color:
            cur_len += 1
        else:
            runs.append((cur_color, cur_len))
            cur_color = color
            cur_len = 1
    runs.append((cur_color, cur_len))
    return runs


def _paint_rotated_stair(
    out: np.ndarray,
    origin: tuple[int, int],
    orientation: str,
    runs: list[tuple[int, int]],
) -> None:
    h, w = out.shape
    cur_r, cur_c = origin
    if orientation == "H":
        for color, length in runs:
            for step in range(length):
                rr, cc = cur_r + step, cur_c
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = int(color)
            cur_r += length - 1
            cur_c += 1
    else:
        for color, length in runs:
            for step in range(length):
                rr, cc = cur_r, cur_c + step
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = int(color)
            cur_r += 1
            cur_c += length - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each straight multicolor path into a staircase by rotating every color run 90 degrees.

    Input structure:
    - The dominant background is black.
    - Foreground pixels form several disjoint 1-cell-wide 4-connected paths.
    - Each path is a straight horizontal or vertical line, and color changes along that line define consecutive runs.

    Rule:
    - For each path, read its pixels left-to-right if horizontal or top-to-bottom if vertical, then split that order at every color change.
    - Rotate each run around its first pixel: a horizontal run becomes a downward vertical leg, and a vertical run becomes a rightward horizontal leg.
    - Keep the first run's first pixel fixed; place each later run one step farther along the original line from the previous rotated leg's end, and clip any cells that fall outside the grid.

    Key insight:
    - The color boundaries are hidden hinges: once the straight path is segmented at those hinges, each segment turns 90 degrees and the ordered segments redraw as a staircase.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = np.full_like(g, bg)

    for comp in _components4(g != bg):
        orientation, origin, ordered = _ordered_path(g, comp)
        runs = _extract_runs(g, ordered)
        _paint_rotated_stair(out, origin, orientation, runs)

    return out


_RNG = np.random.default_rng()
_BG = 0
_PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_GEN_RETRIES = 400
_NUM_PATHS_CHOICES = (1, 2, 3, 4)
_GRID_MIN_BY_PATHS = {1: 12, 2: 16, 3: 20, 4: 24}
_GRID_MAX = 30
_MAX_SEGMENTS = 5
_MAX_TOTAL_LEN = 11


def _sample_composition(total: int, parts: int) -> list[int]:
    cuts = sorted(int(x) for x in _RNG.choice(np.arange(1, total), size=parts - 1, replace=False))
    points = [0] + cuts + [total]
    return [points[i + 1] - points[i] for i in range(parts)]


def _choose_orientations(num_paths: int) -> list[str]:
    if num_paths == 1:
        return ["H" if int(_RNG.integers(2)) == 0 else "V"]

    mode = int(_RNG.integers(3))
    if mode == 0:
        return ["H"] * num_paths
    if mode == 1:
        return ["V"] * num_paths

    start = "H" if int(_RNG.integers(2)) == 0 else "V"
    other = "V" if start == "H" else "H"
    return [start if i % 2 == 0 else other for i in range(num_paths)]


def _input_cells(origin: tuple[int, int], orientation: str, total_len: int) -> list[tuple[int, int]]:
    r0, c0 = origin
    if orientation == "H":
        return [(r0, c0 + i) for i in range(total_len)]
    return [(r0 + i, c0) for i in range(total_len)]


def _output_segments(
    shape: tuple[int, int],
    origin: tuple[int, int],
    orientation: str,
    lengths: list[int],
) -> list[list[tuple[int, int]]]:
    h, w = shape
    cur_r, cur_c = origin
    segments: list[list[tuple[int, int]]] = []
    if orientation == "H":
        for length in lengths:
            cells = []
            for step in range(length):
                rr, cc = cur_r + step, cur_c
                if 0 <= rr < h and 0 <= cc < w:
                    cells.append((rr, cc))
            segments.append(cells)
            cur_r += length - 1
            cur_c += 1
    else:
        for length in lengths:
            cells = []
            for step in range(length):
                rr, cc = cur_r, cur_c + step
                if 0 <= rr < h and 0 <= cc < w:
                    cells.append((rr, cc))
            segments.append(cells)
            cur_r += 1
            cur_c += length - 1
    return segments


def _expand(cells: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                expanded.add((nr, nc))
    return expanded


def _sample_path_spec(h: int, w: int, orientation: str, require_truncation: bool) -> dict[str, object] | None:
    axis_cap = w - 2 if orientation == "H" else h - 2
    if axis_cap < 4:
        return None

    seg_max = min(_MAX_SEGMENTS, axis_cap - 1, len(_PALETTE))
    if seg_max < 2:
        return None

    for _ in range(60):
        seg_count = int(_RNG.integers(2, seg_max + 1))
        total_min = seg_count + (2 if require_truncation else 1)
        total_max = min(axis_cap, _MAX_TOTAL_LEN)
        if total_min > total_max:
            continue
        total_len = int(_RNG.integers(total_min, total_max + 1))
        lengths = _sample_composition(total_len, seg_count)
        if max(lengths) == 1:
            continue
        colors = [int(v) for v in _RNG.choice(_PALETTE, size=seg_count, replace=False)]
        return {
            "orientation": orientation,
            "lengths": lengths,
            "colors": colors,
        }
    return None


def _candidate_origins(
    shape: tuple[int, int],
    spec: dict[str, object],
    require_truncation: bool,
    input_forbidden: set[tuple[int, int]],
    output_forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    orientation = str(spec["orientation"])
    lengths = list(spec["lengths"])
    total_len = int(sum(lengths))

    candidates: list[tuple[int, int]] = []
    if orientation == "H":
        row_range = range(1, h - 1)
        col_range = range(1, w - total_len)
    else:
        row_range = range(1, h - total_len)
        col_range = range(1, w - 1)

    for r0 in row_range:
        for c0 in col_range:
            origin = (r0, c0)
            input_cells = _input_cells(origin, orientation, total_len)
            if any(cell in input_forbidden for cell in input_cells):
                continue

            visible_segments = _output_segments(shape, origin, orientation, lengths)
            is_truncated = any(len(seg) < length for seg, length in zip(visible_segments, lengths))
            if is_truncated != require_truncation:
                continue

            if sum(bool(seg) for seg in visible_segments) < 2:
                continue

            output_cells = [cell for seg in visible_segments for cell in seg]
            if not output_cells:
                continue
            if any(cell in output_forbidden for cell in output_cells):
                continue

            candidates.append(origin)

    return candidates


def _paint_input_path(grid: np.ndarray, spec: dict[str, object]) -> None:
    r0, c0 = spec["origin"]
    orientation = str(spec["orientation"])
    lengths = list(spec["lengths"])
    colors = list(spec["colors"])

    if orientation == "H":
        c = c0
        for color, length in zip(colors, lengths):
            grid[r0, c : c + length] = int(color)
            c += length
    else:
        r = r0
        for color, length in zip(colors, lengths):
            grid[r : r + length, c0] = int(color)
            r += length


def _build_expected(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, _BG, dtype=int)
    for spec in specs:
        runs = list(zip(spec["colors"], spec["lengths"]))
        _paint_rotated_stair(out, spec["origin"], str(spec["orientation"]), runs)
    return out


def _gen_attempt() -> np.ndarray | None:
    num_paths = int(_RNG.choice(_NUM_PATHS_CHOICES))
    min_side = _GRID_MIN_BY_PATHS[num_paths]
    h = int(_RNG.integers(min_side, _GRID_MAX + 1))
    w = int(_RNG.integers(min_side, _GRID_MAX + 1))

    orientations = _choose_orientations(num_paths)
    trunc_flags = [bool(int(_RNG.integers(2))) for _ in range(num_paths)]
    if num_paths > 1 and all(flag == trunc_flags[0] for flag in trunc_flags):
        flip_idx = int(_RNG.integers(num_paths))
        trunc_flags[flip_idx] = not trunc_flags[flip_idx]

    order = list(range(num_paths))
    order.sort(key=lambda idx: (not trunc_flags[idx], idx))

    input_forbidden: set[tuple[int, int]] = set()
    output_forbidden: set[tuple[int, int]] = set()
    specs: list[dict[str, object] | None] = [None] * num_paths

    for idx in order:
        require_truncation = trunc_flags[idx]
        placed = False
        for _ in range(100):
            spec = _sample_path_spec(h, w, orientations[idx], require_truncation)
            if spec is None:
                break
            candidates = _candidate_origins((h, w), spec, require_truncation, input_forbidden, output_forbidden)
            if not candidates:
                continue

            origin = candidates[int(_RNG.integers(len(candidates)))]
            spec["origin"] = origin
            specs[idx] = spec

            input_cells = _input_cells(origin, str(spec["orientation"]), int(sum(spec["lengths"])))
            output_cells = [cell for seg in _output_segments((h, w), origin, str(spec["orientation"]), list(spec["lengths"])) for cell in seg]
            input_forbidden.update(_expand(input_cells, (h, w)))
            output_forbidden.update(_expand(output_cells, (h, w)))
            placed = True
            break

        if not placed:
            return None

    final_specs = [spec for spec in specs if spec is not None]
    if len(final_specs) != num_paths:
        return None

    grid = np.full((h, w), _BG, dtype=int)
    for spec in final_specs:
        _paint_input_path(grid, spec)

    expected = _build_expected((h, w), final_specs)
    if not np.array_equal(main(grid), expected):
        return None

    if len(_components4(grid != _BG)) != num_paths:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
