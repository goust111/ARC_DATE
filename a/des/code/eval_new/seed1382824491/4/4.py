import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
RNG = np.random.default_rng()
GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _run_from_col(row: np.ndarray, start_col: int, bg: int) -> tuple[int, int] | None:
    if not (0 <= start_col < row.shape[0]) or int(row[start_col]) == bg:
        return None
    color = int(row[start_col])
    end = start_col
    while end < row.shape[0] and int(row[end]) == color:
        end += 1
    return end - start_col, color


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray, int]:
    coords = np.argwhere(grid != bg)
    assert len(coords) > 0, "grid must contain non-background pixels"
    legend_col = int(coords[:, 1].min())

    best_entries: list[tuple[int, int, int]] = []
    h = grid.shape[0]
    for start_row in range(h):
        info = _run_from_col(grid[start_row], legend_col, bg)
        if info is None or info[0] != 1:
            continue
        entries: list[tuple[int, int, int]] = []
        used_colors: set[int] = set()
        length = 1
        row = start_row
        while row < h:
            here = _run_from_col(grid[row], legend_col, bg)
            if here is None or here[0] != length or here[1] in used_colors:
                break
            entries.append((row, length, here[1]))
            used_colors.add(here[1])
            length += 1
            row += 2
        if len(entries) > len(best_entries):
            best_entries = entries

    assert len(best_entries) >= 2, "failed to detect left-side length legend"
    mapping = {length: color for _, length, color in best_entries}
    legend_mask = np.zeros_like(grid, dtype=bool)
    for row, length, _color in best_entries:
        legend_mask[row, legend_col : legend_col + length] = True
    return mapping, legend_mask, legend_col


def _path_neighbor_count(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    count = 0
    if r > 0 and mask[r - 1, c]:
        count += 1
    if r + 1 < h and mask[r + 1, c]:
        count += 1
    if c > 0 and mask[r, c - 1]:
        count += 1
    if c + 1 < w and mask[r, c + 1]:
        count += 1
    return count


def _intersection_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if mask[r, c] and _path_neighbor_count(mask, r, c) >= 3:
                out[r, c] = True
    return out


def _segment_orientation(mask: np.ndarray, r: int, c: int) -> str | None:
    h, w = mask.shape
    left = c > 0 and mask[r, c - 1]
    right = c + 1 < w and mask[r, c + 1]
    up = r > 0 and mask[r - 1, c]
    down = r + 1 < h and mask[r + 1, c]
    horiz = left or right
    vert = up or down
    if horiz and vert:
        return None
    if horiz:
        return "h"
    if vert:
        return "v"
    return "s"


def _extract_segments(path_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    intersections = _intersection_mask(path_mask)
    body = path_mask & (~intersections)
    visited = np.zeros_like(path_mask, dtype=bool)
    segments: list[list[tuple[int, int]]] = []
    h, w = path_mask.shape

    for r in range(h):
        for c in range(w):
            if not body[r, c] or visited[r, c]:
                continue
            orient = _segment_orientation(path_mask, r, c)
            if orient is None or orient == "s":
                visited[r, c] = True
                segments.append([(r, c)])
                continue
            if orient == "h":
                c0 = c
                while c0 > 0 and body[r, c0 - 1] and _segment_orientation(path_mask, r, c0 - 1) == "h":
                    c0 -= 1
                c1 = c
                while c1 + 1 < w and body[r, c1 + 1] and _segment_orientation(path_mask, r, c1 + 1) == "h":
                    c1 += 1
                segment = []
                for cc in range(c0, c1 + 1):
                    assert _segment_orientation(path_mask, r, cc) == "h"
                    visited[r, cc] = True
                    segment.append((r, cc))
                segments.append(segment)
                continue
            r0 = r
            while r0 > 0 and body[r0 - 1, c] and _segment_orientation(path_mask, r0 - 1, c) == "v":
                r0 -= 1
            r1 = r
            while r1 + 1 < h and body[r1 + 1, c] and _segment_orientation(path_mask, r1 + 1, c) == "v":
                r1 += 1
            segment = []
            for rr in range(r0, r1 + 1):
                assert _segment_orientation(path_mask, rr, c) == "v"
                visited[rr, c] = True
                segment.append((rr, c))
            segments.append(segment)

    assert bool(np.all(visited[body])), "every non-intersection path pixel must belong to a segment"
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor orthogonal path segments by their lengths after splitting at crossings.

    Input structure:
    - The background is black, and the left side contains a vertical legend of horizontal bars.
    - The legend bars have lengths 1, 2, 3, ... in order, with one background row between entries.
    - The main drawing contains one-pixel-wide horizontal and vertical path pieces, all in one shared color.
    - Paths may meet at crossings where a path pixel has three or four four-neighbors.

    Rule:
    - Detect the left-side legend and read it as a mapping from segment length to color.
    - Find every crossing pixel in the path drawing; those pixels split the path into straight segments.
    - For each remaining maximal horizontal or vertical segment, measure its length and recolor it with the legend color for that length.
    - Recolor each crossing pixel to grey and leave the legend unchanged.

    Key insight:
    The path color itself is irrelevant: the useful structure is that crossings break the network
    into independent straight segments, and the legend converts each segment length into a new color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    length_to_color, legend_mask, _legend_col = _find_legend(g, bg)

    legend_colors = set(length_to_color.values())
    colors = [int(v) for v in np.unique(g) if int(v) != bg and int(v) not in legend_colors]
    assert colors, "expected a path color outside the legend palette"
    path_color = max(colors, key=lambda value: int(np.count_nonzero(g == value)))

    path_mask = (g == path_color) & (~legend_mask)
    intersections = _intersection_mask(path_mask)
    segments = _extract_segments(path_mask)

    out = g.copy()
    out[path_mask] = bg
    out[intersections] = GREY
    for segment in segments:
        color = length_to_color[len(segment)]
        for r, c in segment:
            out[r, c] = color
    return out


def _paint_legend(grid: np.ndarray, start_row: int, start_col: int, length_to_color: dict[int, int]) -> None:
    for length, color in length_to_color.items():
        row = start_row + 2 * (length - 1)
        grid[row, start_col : start_col + length] = color


def _primitive_cells(primitive: tuple[str, int, int, int]) -> list[tuple[int, int]]:
    orient, fixed, lo, hi = primitive
    if orient == "h":
        return [(fixed, c) for c in range(lo, hi + 1)]
    return [(r, fixed) for r in range(lo, hi + 1)]


def _paint_primitive(mask: np.ndarray, primitive: tuple[str, int, int, int]) -> None:
    for r, c in _primitive_cells(primitive):
        mask[r, c] = True


def _valid_mask(mask: np.ndarray, max_len: int) -> bool:
    try:
        segments = _extract_segments(mask)
    except AssertionError:
        return False
    if not segments:
        return False
    return all(1 <= len(segment) <= max_len for segment in segments)


def _count_cross_types(mask: np.ndarray) -> tuple[int, int]:
    intersections = _intersection_mask(mask)
    t_count = 0
    plus_count = 0
    for r, c in np.argwhere(intersections):
        degree = _path_neighbor_count(mask, int(r), int(c))
        if degree == 3:
            t_count += 1
        elif degree == 4:
            plus_count += 1
    return t_count, plus_count


def _can_place_isolated(mask: np.ndarray, primitive: tuple[str, int, int, int]) -> bool:
    cells = _primitive_cells(primitive)
    h, w = mask.shape
    for r, c in cells:
        if mask[r, c]:
            return False
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                return False
    return True


def _random_isolated(mask: np.ndarray, row_lo: int, row_hi: int, col_lo: int, col_hi: int, max_len: int) -> bool:
    for _ in range(80):
        orient = str(RNG.choice(["h", "v"]))
        length = int(RNG.integers(1, max_len + 1))
        if orient == "h":
            r = int(RNG.integers(row_lo, row_hi + 1))
            if col_hi - col_lo + 1 < length:
                return False
            c0 = int(RNG.integers(col_lo, col_hi - length + 2))
            primitive = ("h", r, c0, c0 + length - 1)
        else:
            c = int(RNG.integers(col_lo, col_hi + 1))
            if row_hi - row_lo + 1 < length:
                return False
            r0 = int(RNG.integers(row_lo, row_hi - length + 2))
            primitive = ("v", c, r0, r0 + length - 1)
        if _can_place_isolated(mask, primitive):
            _paint_primitive(mask, primitive)
            return True
    return False


def _add_plus(mask: np.ndarray, row_lo: int, row_hi: int, col_lo: int, col_hi: int, max_len: int) -> bool:
    for _ in range(80):
        r = int(RNG.integers(row_lo, row_hi + 1))
        c = int(RNG.integers(col_lo, col_hi + 1))
        left = int(RNG.integers(1, max_len + 1))
        right = int(RNG.integers(1, max_len + 1))
        up = int(RNG.integers(1, max_len + 1))
        down = int(RNG.integers(1, max_len + 1))
        if c - left < col_lo or c + right > col_hi or r - up < row_lo or r + down > row_hi:
            continue
        h_prim = ("h", r, c - left, c + right)
        v_prim = ("v", c, r - up, r + down)
        trial = mask.copy()
        _paint_primitive(trial, h_prim)
        _paint_primitive(trial, v_prim)
        if _valid_mask(trial, max_len):
            mask[:, :] = trial
            return True
    return False


def _add_t(
    mask: np.ndarray,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    max_len: int,
    expected_plus_count: int = 0,
) -> bool:
    for _ in range(120):
        r = int(RNG.integers(row_lo, row_hi + 1))
        c = int(RNG.integers(col_lo, col_hi + 1))
        missing = str(RNG.choice(["up", "down", "left", "right"]))
        left = int(RNG.integers(1, max_len + 1))
        right = int(RNG.integers(1, max_len + 1))
        up = int(RNG.integers(1, max_len + 1))
        down = int(RNG.integers(1, max_len + 1))
        if missing == "up":
            up = 0
        elif missing == "down":
            down = 0
        elif missing == "left":
            left = 0
        else:
            right = 0
        if left + right == 0 or up + down == 0:
            continue
        if c - left < col_lo or c + right > col_hi or r - up < row_lo or r + down > row_hi:
            continue
        trial = mask.copy()
        _paint_primitive(trial, ("h", r, c - left, c + right))
        _paint_primitive(trial, ("v", c, r - up, r + down))
        t_count, plus_count = _count_cross_types(trial)
        if plus_count != expected_plus_count:
            continue
        if t_count < 1 or not _valid_mask(trial, max_len):
            continue
        mask[:, :] = trial
        return True
    return False


def _build_path_mask(shape: tuple[int, int], max_len: int, path_left: int) -> np.ndarray | None:
    h, w = shape
    row_lo, row_hi = 1, h - 2
    col_lo, col_hi = path_left, w - 2
    if row_hi < row_lo or col_hi < col_lo:
        return None

    scenario = str(RNG.choice(["none", "t_only", "plus_only", "mixed"]))
    mask = np.zeros(shape, dtype=bool)

    if scenario == "none":
        count = int(RNG.integers(3, 6))
        for _ in range(count):
            if not _random_isolated(mask, row_lo, row_hi, col_lo, col_hi, max_len):
                return None
    elif scenario == "t_only":
        if not _add_t(mask, row_lo, row_hi, col_lo, col_hi, max_len):
            return None
        for _ in range(int(RNG.integers(1, 4))):
            _random_isolated(mask, row_lo, row_hi, col_lo, col_hi, max_len)
    elif scenario == "plus_only":
        if not _add_plus(mask, row_lo, row_hi, col_lo, col_hi, max_len):
            return None
        for _ in range(int(RNG.integers(1, 4))):
            _random_isolated(mask, row_lo, row_hi, col_lo, col_hi, max_len)
    else:
        if not _add_plus(mask, row_lo, row_hi, col_lo, col_hi, max_len):
            return None
        if not _add_t(mask, row_lo, row_hi, col_lo, col_hi, max_len, expected_plus_count=1):
            return None
        for _ in range(int(RNG.integers(0, 3))):
            _random_isolated(mask, row_lo, row_hi, col_lo, col_hi, max_len)

    if not _valid_mask(mask, max_len):
        return None
    t_count, plus_count = _count_cross_types(mask)
    if scenario == "none" and (t_count != 0 or plus_count != 0):
        return None
    if scenario == "t_only" and (t_count < 1 or plus_count != 0):
        return None
    if scenario == "plus_only" and plus_count < 1:
        return None
    if scenario == "mixed" and (t_count < 1 or plus_count < 1):
        return None
    return mask


def _gen_attempt() -> np.ndarray | None:
    max_len = int(RNG.integers(3, 7))
    legend_col = int(RNG.integers(0, 2))
    legend_row = int(RNG.integers(0, 3))
    gap = int(RNG.integers(2, 4))
    path_left = legend_col + max_len + gap

    min_h = max(legend_row + 2 * max_len, 2 * max_len + 3, 12)
    min_w = path_left + 2 * max_len + 2
    if min_h > 30 or min_w > 30:
        return None
    h = int(RNG.integers(min_h, 31))
    w = int(RNG.integers(min_w, 31))

    colors = list(RNG.permutation(PALETTE))
    legend_colors = colors[:max_len]
    path_color = colors[max_len]
    length_to_color = {length: legend_colors[length - 1] for length in range(1, max_len + 1)}

    grid = np.full((h, w), BG, dtype=int)
    _paint_legend(grid, legend_row, legend_col, length_to_color)

    path_mask = _build_path_mask(grid.shape, max_len, path_left)
    if path_mask is None:
        return None
    grid[path_mask] = path_color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
