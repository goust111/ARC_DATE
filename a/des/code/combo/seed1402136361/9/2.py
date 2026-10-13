import random
from collections import Counter

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPPOSITE = {(-1, 0): (1, 0), (1, 0): (-1, 0), (0, -1): (0, 1), (0, 1): (0, -1)}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _ordered_path(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert coords, "expected a non-empty foreground path"
    coord_set = set(coords)

    nbrs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    endpoints = []
    for pt in coords:
        local = [nb for nb in _neighbors4(pt[0], pt[1], h, w) if nb in coord_set]
        assert 1 <= len(local) <= 2, "foreground must be a simple path"
        nbrs[pt] = local
        if len(local) == 1:
            endpoints.append(pt)

    assert len(endpoints) == 2, "path must have exactly two endpoints"
    start = min(endpoints)
    ordered = [start]
    prev = None
    cur = start
    while len(ordered) < len(coords):
        nxts = [nb for nb in nbrs[cur] if nb != prev]
        assert len(nxts) == 1, "path traversal must be unambiguous"
        nxt = nxts[0]
        ordered.append(nxt)
        prev, cur = cur, nxt

    assert len(set(ordered)) == len(coords), "foreground must be one connected path"
    return ordered


def _segment_spans(path: list[tuple[int, int]], grid: np.ndarray) -> list[tuple[int, int]]:
    spans = []
    start = 0
    cur_color = int(grid[path[0]])
    for i in range(1, len(path)):
        color = int(grid[path[i]])
        if color != cur_color:
            spans.append((start, i))
            start = i
            cur_color = color
    spans.append((start, len(path)))
    return spans


def _segment_direction(
    segment: list[tuple[int, int]],
    prev_pt: tuple[int, int] | None,
    next_pt: tuple[int, int] | None,
) -> tuple[int, int]:
    start = segment[0]
    end = segment[-1]
    dr = end[0] - start[0]
    dc = end[1] - start[1]
    if abs(dr) > abs(dc):
        return (1 if dr > 0 else -1, 0)
    if abs(dc) > abs(dr):
        return (0, 1 if dc > 0 else -1)

    coords = np.asarray(segment, dtype=int)
    if len(coords) > 1:
        steps = coords[1:] - coords[:-1]
        vmag = int(np.abs(steps[:, 0]).sum())
        hmag = int(np.abs(steps[:, 1]).sum())
        if vmag > hmag:
            sign = int(np.sign(steps[:, 0].sum()))
            if sign == 0:
                sign = int(np.sign(steps[steps[:, 0] != 0][0, 0]))
            return (sign, 0)
        if hmag > vmag:
            sign = int(np.sign(steps[:, 1].sum()))
            if sign == 0:
                sign = int(np.sign(steps[steps[:, 1] != 0][0, 1]))
            return (0, sign)
        first_step = steps[0]
        if first_step[0] != 0:
            return (int(np.sign(first_step[0])), 0)
        if first_step[1] != 0:
            return (0, int(np.sign(first_step[1])))

    context_steps = []
    if prev_pt is not None:
        context_steps.append((segment[0][0] - prev_pt[0], segment[0][1] - prev_pt[1]))
    if next_pt is not None:
        context_steps.append((next_pt[0] - segment[-1][0], next_pt[1] - segment[-1][1]))
    total_dr = sum(step[0] for step in context_steps)
    total_dc = sum(step[1] for step in context_steps)
    if abs(total_dr) > abs(total_dc) and total_dr != 0:
        return (int(np.sign(total_dr)), 0)
    if total_dc != 0:
        return (0, int(np.sign(total_dc)))
    if total_dr != 0:
        return (int(np.sign(total_dr)), 0)
    return (0, 1)


def _shift_to_edge(
    coords: np.ndarray, direction: tuple[int, int], shape: tuple[int, int]
) -> np.ndarray:
    h, w = shape
    dr, dc = direction
    shifted = coords.copy()
    if dr < 0:
        shifted[:, 0] -= int(coords[:, 0].min())
    elif dr > 0:
        shifted[:, 0] += h - 1 - int(coords[:, 0].max())
    elif dc < 0:
        shifted[:, 1] -= int(coords[:, 1].min())
    else:
        shifted[:, 1] += w - 1 - int(coords[:, 1].max())
    return shifted


def _transformed_segments(grid: np.ndarray, bg: int | None = None) -> list[tuple[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = _most_common_color(g)
    path = _ordered_path(g != bg)
    spans = _segment_spans(path, g)

    transformed = []
    for lo, hi in spans:
        segment = path[lo:hi]
        prev_pt = path[lo - 1] if lo > 0 else None
        next_pt = path[hi] if hi < len(path) else None
        direction = _segment_direction(segment, prev_pt, next_pt)
        shifted = _shift_to_edge(np.asarray(segment, dtype=int), direction, g.shape)
        transformed.append((int(g[segment[0]]), shifted))
    return transformed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split a multicolor path into color runs and send each run to the edge
    indicated by its own local start-to-end direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = np.full_like(g, bg)

    for color, shifted in _transformed_segments(g, bg):
        rr = shifted[:, 0]
        cc = shifted[:, 1]
        free = out[rr, cc] == bg
        out[rr[free], cc[free]] = color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (10, 16)
GRID_W_RANGE = (10, 16)
SEGMENT_COUNT_RANGE = (3, 7)
SEGMENT_LEN_RANGE = (2, 6)
FG_COLOR_COUNT_RANGE = (2, 3)
GEN_RETRIES = 400
PATH_BUILD_RETRIES = 300


def _inside_inner(r: int, c: int, h: int, w: int) -> bool:
    return 0 < r < h - 1 and 0 < c < w - 1


def _flatten_segments(segments: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    path = list(segments[0])
    for seg in segments[1:]:
        path.extend(seg)
    return path


def _path_is_valid(segments: list[list[tuple[int, int]]], h: int, w: int) -> bool:
    if not segments or any(len(seg) < 2 for seg in segments):
        return False
    for seg in segments:
        rows = {pt[0] for pt in seg}
        cols = {pt[1] for pt in seg}
        if len(rows) != 1 and len(cols) != 1:
            return False

    path = _flatten_segments(segments)
    if len(set(path)) != len(path):
        return False
    if not all(_inside_inner(r, c, h, w) for r, c in path):
        return False

    coord_set = set(path)
    for idx, pt in enumerate(path):
        if idx > 0 and abs(pt[0] - path[idx - 1][0]) + abs(pt[1] - path[idx - 1][1]) != 1:
            return False
        nbrs = [nb for nb in _neighbors4(pt[0], pt[1], h, w) if nb in coord_set]
        expected = 1 if idx in (0, len(path) - 1) else 2
        if len(nbrs) != expected:
            return False
        if idx > 0 and path[idx - 1] not in nbrs:
            return False
        if idx + 1 < len(path) and path[idx + 1] not in nbrs:
            return False
    return True


def _max_straight_extension(
    cur: tuple[int, int], direction: tuple[int, int], occupied: set[tuple[int, int]], h: int, w: int
) -> list[tuple[int, int]]:
    dr, dc = direction
    prev = cur
    cells = []
    while True:
        nxt = (prev[0] + dr, prev[1] + dc)
        if not _inside_inner(nxt[0], nxt[1], h, w) or nxt in occupied:
            break
        if any(nb in occupied and nb != prev for nb in _neighbors4(nxt[0], nxt[1], h, w)):
            break
        cells.append(nxt)
        prev = nxt
    return cells


def _build_random_walk_segments(
    h: int, w: int, num_segments: int, require_both_axes: bool
) -> list[list[tuple[int, int]]] | None:
    for _ in range(PATH_BUILD_RETRIES):
        start = (random.randint(1, h - 2), random.randint(1, w - 2))
        occupied = {start}
        current = start
        prev_dir = None
        segments = []
        axes = []
        ok = True

        for seg_idx in range(num_segments):
            options = []
            for direction in DIRS4:
                if prev_dir is not None and direction == OPPOSITE[prev_dir]:
                    continue
                ext = _max_straight_extension(current, direction, occupied, h, w)
                max_len = 1 + len(ext) if seg_idx == 0 else len(ext)
                if max_len >= SEGMENT_LEN_RANGE[0]:
                    options.append((direction, ext, max_len))
            if not options:
                ok = False
                break

            if prev_dir is not None:
                turn_options = [item for item in options if item[0] != prev_dir]
                same_options = [item for item in options if item[0] == prev_dir]
                if turn_options and (not same_options or random.random() < 0.72):
                    options = turn_options
                elif same_options and random.random() < 0.35:
                    options = same_options

            direction, ext, max_len = random.choice(options)
            upper = min(max_len, SEGMENT_LEN_RANGE[1])
            length = random.randint(SEGMENT_LEN_RANGE[0], upper)
            if seg_idx == 0:
                new_cells = ext[: length - 1]
                segment = [start, *new_cells]
            else:
                new_cells = ext[:length]
                segment = list(new_cells)
            for cell in new_cells:
                occupied.add(cell)
            current = segment[-1]
            prev_dir = direction
            axes.append(0 if direction[0] != 0 else 1)
            segments.append(segment)

        if not ok or not _path_is_valid(segments, h, w):
            continue
        if require_both_axes and len(set(axes)) < 2:
            continue
        return segments
    return None


def _base_horizontal_overlap(h: int, w: int) -> list[list[tuple[int, int]]] | None:
    for _ in range(PATH_BUILD_RETRIES):
        la = random.randint(2, min(4, w - 4))
        ld = random.randint(2, min(4, w - 4))
        gap = random.randint(1, 2)
        lb = random.randint(2, min(4, h - 3))
        c0_min = 1 + gap + ld
        c0_max = w - la - 1
        if c0_min > c0_max:
            continue
        c0 = random.randint(c0_min, c0_max)
        direction = random.choice([-1, 1])
        if direction > 0:
            r_min, r_max = 1, h - 2 - lb
        else:
            r_min, r_max = 1 + lb, h - 2
        if r_min > r_max:
            continue
        r0 = random.randint(r_min, r_max)
        c1 = c0 - gap - ld
        lc = la - 1 + gap + ld

        seg1 = [(r0, c0 + i) for i in range(la)]
        seg2 = [(r0 + direction * i, c0 + la - 1) for i in range(1, lb + 1)]
        seg3 = [(r0 + direction * lb, c0 + la - 1 - i) for i in range(1, lc + 1)]
        seg4 = [(r0 + direction * (lb - i), c1) for i in range(1, lb + 1)]
        seg5 = [(r0, c1 + i) for i in range(1, ld + 1)]
        segments = [seg1, seg2, seg3, seg4, seg5]
        if _path_is_valid(segments, h, w):
            return segments
    return None


def _overlap_template_segments(h: int, w: int) -> list[list[tuple[int, int]]] | None:
    use_vertical = random.random() < 0.5
    if use_vertical:
        base = _base_horizontal_overlap(w, h)
        if base is None:
            return None
        segments = [[(c, r) for r, c in seg] for seg in base]
        if random.random() < 0.5:
            segments = [[(h - 1 - r, c) for r, c in seg] for seg in segments]
    else:
        segments = _base_horizontal_overlap(h, w)
        if segments is None:
            return None
        if random.random() < 0.5:
            segments = [[(r, w - 1 - c) for r, c in seg] for seg in segments]
    return segments if _path_is_valid(segments, h, w) else None


def _pick_segment_colors(bg: int, num_segments: int) -> list[int]:
    palette_size = random.randint(FG_COLOR_COUNT_RANGE[0], min(FG_COLOR_COUNT_RANGE[1], 9))
    palette = random.sample([c for c in range(10) if c != bg], palette_size)
    colors = []
    for _ in range(num_segments):
        choices = [color for color in palette if not colors or color != colors[-1]]
        colors.append(random.choice(choices))
    assert len(set(colors)) >= 2
    return colors


def _render_input(
    h: int, w: int, bg: int, segments: list[list[tuple[int, int]]], colors: list[int]
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for seg, color in zip(segments, colors):
        for r, c in seg:
            grid[r, c] = color
    return grid


def _overlap_count(transformed: list[tuple[int, np.ndarray]]) -> int:
    counts: Counter[tuple[int, int]] = Counter()
    for _color, coords in transformed:
        for r, c in coords:
            counts[(int(r), int(c))] += 1
    return sum(v - 1 for v in counts.values() if v > 1)


def _segment_motion_axes(grid: np.ndarray) -> set[int]:
    bg = _most_common_color(grid)
    axes = set()
    path = _ordered_path(grid != bg)
    for lo, hi in _segment_spans(path, grid):
        segment = path[lo:hi]
        prev_pt = path[lo - 1] if lo > 0 else None
        next_pt = path[hi] if hi < len(path) else None
        dr, dc = _segment_direction(segment, prev_pt, next_pt)
        axes.add(0 if dr != 0 else 1)
    return axes


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    want_overlap = random.random() < 0.5
    require_both_axes = random.random() < 0.75

    if want_overlap:
        segments = _overlap_template_segments(h, w)
    else:
        num_segments = random.randint(*SEGMENT_COUNT_RANGE)
        segments = _build_random_walk_segments(h, w, num_segments, require_both_axes)
    if segments is None:
        return None

    bg = random.randint(0, 9)
    colors = _pick_segment_colors(bg, len(segments))
    grid = _render_input(h, w, bg, segments, colors)

    if np.count_nonzero(grid != bg) >= (h * w) // 2:
        return None

    try:
        transformed = _transformed_segments(grid, bg)
    except AssertionError:
        return None

    if require_both_axes and len(_segment_motion_axes(grid)) < 2:
        return None

    overlap = _overlap_count(transformed)
    if want_overlap and overlap == 0:
        return None
    if (not want_overlap) and overlap != 0:
        return None

    if len(np.unique(grid)) < 3:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
