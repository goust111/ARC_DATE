import numpy as np


MAX_GENERATION_ATTEMPTS = 600
_BG = 0
_FG_COLORS = tuple(range(1, 10))


def _extract_strip_segments(strip: np.ndarray, bg: int) -> tuple[list[tuple[int, int, int]], dict[int, tuple[int, int]]]:
    """Parse one projection strip into contiguous same-color segments."""
    segments: list[tuple[int, int, int]] = []
    spans: dict[int, tuple[int, int]] = {}
    i = 0
    n = int(len(strip))
    while i < n:
        color = int(strip[i])
        if color == bg:
            i += 1
            continue
        j = i + 1
        while j < n and int(strip[j]) == color:
            j += 1
        assert color not in spans, "each color may appear in at most one segment per strip"
        span = (i, j - 1)
        spans[color] = span
        segments.append((color, span[0], span[1]))
        i = j
    return segments, spans


def _common_rectangles(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int]], list[tuple[int, int, int]], list[dict[str, int]]]:
    bg = int(grid[0, 0])
    top_segments, top_spans = _extract_strip_segments(grid[0, 1:], bg)
    left_segments, left_spans = _extract_strip_segments(grid[1:, 0], bg)

    rectangles: list[dict[str, int]] = []
    for color in sorted(set(top_spans) & set(left_spans)):
        c0, c1 = top_spans[color]
        r0, r1 = left_spans[color]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        rectangles.append(
            {
                "color": int(color),
                "r0": int(r0 + 1),
                "c0": int(c0 + 1),
                "r1": int(r1 + 1),
                "c1": int(c1 + 1),
                "area": int(height * width),
                "height": int(height),
                "width": int(width),
            }
        )
    return bg, top_segments, left_segments, rectangles


def _paint_rectangles(base: np.ndarray, bg: int, rectangles: list[dict[str, int]]) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    out[1:, 1:] = int(bg)
    for rect in sorted(rectangles, key=lambda item: item["area"], reverse=True):
        out[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    return out


def _solve_pair_by_shared_order(grid: np.ndarray) -> np.ndarray:
    bg = int(grid[0, 0])
    top_segments, top_spans = _extract_strip_segments(grid[0, 1:], bg)
    left_segments, left_spans = _extract_strip_segments(grid[1:, 0], bg)
    common = set(top_spans) & set(left_spans)
    top_common = [seg for seg in top_segments if seg[0] in common]
    left_common = [seg for seg in left_segments if seg[0] in common]
    rectangles: list[dict[str, int]] = []
    for (color, c0, c1), (_other, r0, r1) in zip(top_common, left_common):
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        rectangles.append(
            {
                "color": int(color),
                "r0": int(r0 + 1),
                "c0": int(c0 + 1),
                "r1": int(r1 + 1),
                "c1": int(c1 + 1),
                "area": int(height * width),
            }
        )
    return _paint_rectangles(grid, bg, rectangles)


def _solve_pair_by_raw_order(grid: np.ndarray) -> np.ndarray:
    bg = int(grid[0, 0])
    top_segments, _top_spans = _extract_strip_segments(grid[0, 1:], bg)
    left_segments, _left_spans = _extract_strip_segments(grid[1:, 0], bg)
    rectangles: list[dict[str, int]] = []
    for (color, c0, c1), (_other, r0, r1) in zip(top_segments, left_segments):
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        rectangles.append(
            {
                "color": int(color),
                "r0": int(r0 + 1),
                "c0": int(c0 + 1),
                "r1": int(r1 + 1),
                "c1": int(c1 + 1),
                "area": int(height * width),
            }
        )
    return _paint_rectangles(grid, bg, rectangles)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the hidden solid rectangles from the top and left projection strips.

    Input structure:
    - The first row and first column are projection strips, with the top-left cell
      equal to the background; every other cell starts as background.
    - Each strip contains several solid same-color segments separated by background.
    - Most colors appear once on both strips, while some colors appear on only one
      strip and act as distractor remnants.

    Rule:
    - For each color that appears on both strips, use the top-strip segment as the
      column span and the left-strip segment as the row span, then fill the implied
      solid rectangle in the field.
    - If rectangles overlap, keep the color of the smaller-area rectangle on top;
      the puzzle instances guarantee distinct rectangle areas.
    - Colors that appear on only one strip generate nothing, and both strips stay
      unchanged in the output.

    Key insight:
    - The true objects are invisible and must be reconstructed by pairing matching
      colors across two orthogonal one-dimensional projections; single-sided
      segments are decoys, and any overlap priority is determined by area, not by
      position or strip order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(g.shape) >= 2, "expected a projection row, a projection column, and a field"

    bg, top_segments, left_segments, rectangles = _common_rectangles(g)
    assert top_segments, "expected at least one top-strip segment"
    assert left_segments, "expected at least one left-strip segment"
    assert rectangles, "expected at least one color present on both strips"

    areas = [rect["area"] for rect in rectangles]
    assert len(areas) == len(set(areas)), "rectangle areas must be distinct"
    return _paint_rectangles(g, bg, rectangles)


def _shared_suborder(order: list[int], shared: set[int]) -> list[int]:
    return [color for color in order if color in shared]


def _pack_projection_segments(
    ordered_colors: list[int],
    shared_colors: set[int],
    field_len: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    n = len(ordered_colors)
    min_lengths = [2 if color in shared_colors else 1 for color in ordered_colors]
    min_total = sum(min_lengths) + max(0, n - 1)
    assert field_len >= min_total

    lengths = list(min_lengths)
    slack = field_len - min_total
    gap_extras = [0] * (n + 1)
    max_len = min(6, field_len)

    for _ in range(slack):
        choices: list[tuple[str, int]] = [("gap", idx) for idx in range(n + 1)]
        for idx in range(n):
            color = ordered_colors[idx]
            limit = max_len if color in shared_colors else 4
            if lengths[idx] < limit:
                choices.append(("seg", idx))
        kind, idx = choices[int(rng.integers(0, len(choices)))]
        if kind == "seg":
            lengths[idx] += 1
        else:
            gap_extras[idx] += 1

    segments: list[tuple[int, int, int]] = []
    pos = gap_extras[0]
    for idx, color in enumerate(ordered_colors):
        start = pos
        end = start + lengths[idx] - 1
        segments.append((int(color), int(start), int(end)))
        pos = end + 1
        if idx < n - 1:
            pos += 1 + gap_extras[idx + 1]

    assert pos + gap_extras[-1] == field_len
    return segments


def _render_input(
    field_h: int,
    field_w: int,
    top_segments: list[tuple[int, int, int]],
    left_segments: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.full((field_h + 1, field_w + 1), _BG, dtype=int)
    for color, c0, c1 in top_segments:
        grid[0, c0 + 1 : c1 + 2] = int(color)
    for color, r0, r1 in left_segments:
        grid[r0 + 1 : r1 + 2, 0] = int(color)
    return grid


def _valid_sample(grid: np.ndarray) -> bool:
    bg, top_segments, left_segments, rectangles = _common_rectangles(grid)
    if not rectangles:
        return False

    common = {rect["color"] for rect in rectangles}
    top_colors = {color for color, _s, _e in top_segments}
    left_colors = {color for color, _s, _e in left_segments}
    if len(common) < 2:
        return False
    if len((top_colors - left_colors) | (left_colors - top_colors)) < 1:
        return False

    areas = [rect["area"] for rect in rectangles]
    if len(areas) != len(set(areas)):
        return False

    top_shared_order = [color for color, _s, _e in top_segments if color in common]
    left_shared_order = [color for color, _s, _e in left_segments if color in common]
    if top_shared_order == left_shared_order:
        return False

    total_area = sum(areas)
    if total_area < 12:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.array_equal(_solve_pair_by_shared_order(grid), out):
        return False
    if np.array_equal(_solve_pair_by_raw_order(grid), out):
        return False

    if not np.all(out[0, :] == grid[0, :]):
        return False
    if not np.all(out[:, 0] == grid[:, 0]):
        return False
    if not np.all(out[1:, 1:][grid[1:, 1:] != bg] == grid[1:, 1:][grid[1:, 1:] != bg]):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_real = int(rng.integers(2, 5))
    n_top_only = int(rng.integers(0, 3))
    n_left_only = int(rng.integers(0, 3))
    if n_top_only + n_left_only == 0:
        if rng.random() < 0.5:
            n_top_only = 1
        else:
            n_left_only = 1

    total_colors = n_real + n_top_only + n_left_only
    if total_colors > len(_FG_COLORS):
        return None

    colors = list(map(int, rng.choice(_FG_COLORS, size=total_colors, replace=False)))
    shared_colors = colors[:n_real]
    top_only_colors = colors[n_real : n_real + n_top_only]
    left_only_colors = colors[n_real + n_top_only :]
    shared_set = set(shared_colors)

    top_palette = shared_colors + top_only_colors
    left_palette = shared_colors + left_only_colors

    for _ in range(60):
        top_order = list(map(int, rng.permutation(top_palette)))
        left_order = list(map(int, rng.permutation(left_palette)))
        if _shared_suborder(top_order, shared_set) != _shared_suborder(left_order, shared_set):
            break
    else:
        return None

    row_min_total = sum(2 if color in shared_set else 1 for color in left_order) + len(left_order) - 1
    col_min_total = sum(2 if color in shared_set else 1 for color in top_order) + len(top_order) - 1

    field_h_lo = max(9, row_min_total)
    field_w_lo = max(9, col_min_total)
    field_h_hi = min(29, max(field_h_lo, row_min_total + 10))
    field_w_hi = min(29, max(field_w_lo, col_min_total + 10))
    if field_h_lo > field_h_hi or field_w_lo > field_w_hi:
        return None

    field_h = int(rng.integers(field_h_lo, field_h_hi + 1))
    field_w = int(rng.integers(field_w_lo, field_w_hi + 1))

    top_segments = _pack_projection_segments(top_order, shared_set, field_w, rng)
    left_segments = _pack_projection_segments(left_order, shared_set, field_h, rng)
    grid = _render_input(field_h, field_w, top_segments, left_segments)

    return grid if _valid_sample(grid) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-projection reconstruction input")
