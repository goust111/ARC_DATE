import random
from typing import Dict, List, Tuple

import numpy as np


Rect = Tuple[int, int, int, int]

BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRY_LIMIT = 400

LAYER_COUNT_OPTIONS = (3, 3, 4)
RECT_HEIGHT_RANGE = (3, 6)
RECT_WIDTH_RANGE = (4, 9)
OUTPUT_GAP_RANGE = (1, 3)
OUTPUT_MARGIN_RANGE = (0, 2)
STAGE_WIDTH_RANGE = (14, 24)
MIN_VISIBLE_CELLS = 4

ROW_BIASES = ("top", "middle", "bottom")
COL_BIASES = ("left", "middle", "right")


def _mode_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _prefix(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.int16).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(ps: np.ndarray, rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    total = int(ps[r1, c1])
    if r0:
        total -= int(ps[r0 - 1, c1])
    if c0:
        total -= int(ps[r1, c0 - 1])
    if r0 and c0:
        total += int(ps[r0 - 1, c0 - 1])
    return total


def _bbox_from_mask(mask: np.ndarray) -> Rect:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_area(rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _rect_dims(rect: Rect) -> Tuple[int, int]:
    r0, c0, r1, c1 = rect
    return r1 - r0 + 1, c1 - c0 + 1


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _find_marker_column(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    for c in range(w - 1, -1, -1):
        rows = np.where(grid[:, c] != bg)[0]
        if len(rows) < 2:
            continue
        colors = [int(grid[r, c]) for r in rows]
        if len(set(colors)) != len(colors):
            continue
        isolated = True
        for r in rows:
            r0 = max(0, int(r) - 1)
            r1 = min(h, int(r) + 2)
            c0 = max(0, c - 1)
            c1 = min(w, c + 2)
            if int(np.count_nonzero(grid[r0:r1, c0:c1] != bg)) != 1:
                isolated = False
                break
        if isolated:
            return int(c)
    raise AssertionError("marker column not found")


def _parse_scene(grid: np.ndarray, bg: int) -> Tuple[int, List[int], List[int], np.ndarray]:
    marker_col = _find_marker_column(grid, bg)
    marker_rows = [int(r) for r in np.where(grid[:, marker_col] != bg)[0]]
    marker_colors = [int(grid[r, marker_col]) for r in marker_rows]
    assert len(marker_rows) == len(marker_colors) >= 2
    assert len(set(marker_colors)) == len(marker_colors)
    stage = grid[:, :marker_col].copy()
    return marker_col, marker_rows, marker_colors, stage


def _enumerate_candidates(
    vis_mask: np.ndarray,
    vis_ps: np.ndarray,
    vis_total: int,
    cover_mask: np.ndarray,
    blocked_mask: np.ndarray,
) -> List[Rect]:
    if bool((cover_mask & vis_mask).any()):
        return []

    h, w = vis_mask.shape
    coords = np.argwhere(vis_mask)
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)

    cover_ps = _prefix(cover_mask)
    blocked_ps = _prefix(blocked_mask)
    candidates: List[Rect] = []

    for r0 in range(int(r_min) + 1):
        for r1 in range(int(r_max), h):
            for c0 in range(int(c_min) + 1):
                for c1 in range(int(c_max), w):
                    rect = (r0, c0, r1, c1)
                    if _rect_sum(blocked_ps, rect):
                        continue
                    if _rect_sum(vis_ps, rect) != vis_total:
                        continue
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area != vis_total + _rect_sum(cover_ps, rect):
                        continue
                    candidates.append(rect)

    candidates.sort(key=lambda rect: (_rect_area(rect), rect))
    return candidates


def _compose_stage(
    shape: Tuple[int, int], rects: Dict[int, Rect], top_to_bottom: List[int], bg: int
) -> np.ndarray:
    stage = np.full(shape, bg, dtype=int)
    for color in reversed(top_to_bottom):
        r0, c0, r1, c1 = rects[color]
        stage[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return stage


def _recover_solutions(
    stage: np.ndarray, colors: List[int], bg: int, marker_rank: Dict[int, int], limit: int
) -> List[Tuple[Tuple[int, ...], Dict[int, Rect]]]:
    h, w = stage.shape
    visible = {int(color): stage == int(color) for color in colors}
    visible_ps = {int(color): _prefix(mask) for color, mask in visible.items()}
    visible_totals = {int(color): int(mask.sum()) for color, mask in visible.items()}
    if any(total == 0 for total in visible_totals.values()):
        return []

    solutions: List[Tuple[Tuple[int, ...], Dict[int, Rect]]] = []

    def rec(remaining: List[int], cover_mask: np.ndarray, order: List[int], rects: Dict[int, Rect]) -> None:
        if len(solutions) >= limit:
            return
        if not remaining:
            rebuilt = _compose_stage((h, w), rects, order, bg)
            if np.array_equal(rebuilt, stage):
                solutions.append((tuple(order), dict(rects)))
            return

        for color in remaining:
            if bool((cover_mask & visible[color]).any()):
                return

        candidate_sets = []
        for color in remaining:
            blocked_mask = np.zeros((h, w), dtype=bool)
            for other in remaining:
                if other != color:
                    blocked_mask |= visible[other]
            candidates = _enumerate_candidates(
                visible[color],
                visible_ps[color],
                visible_totals[color],
                cover_mask,
                blocked_mask,
            )
            if candidates:
                candidate_sets.append((len(candidates), visible_totals[color], marker_rank[color], color, candidates))

        if not candidate_sets:
            return

        candidate_sets.sort()
        for _count, _vis_total, _rank, chosen, candidates in candidate_sets:
            next_remaining = [color for color in remaining if color != chosen]
            for rect in candidates:
                new_cover = cover_mask.copy()
                r0, c0, r1, c1 = rect
                new_cover[r0 : r1 + 1, c0 : c1 + 1] = True
                rects[chosen] = rect
                order.append(chosen)
                rec(next_remaining, new_cover, order, rects)
                order.pop()
                rects.pop(chosen, None)
                if len(solutions) >= limit:
                    return

    rec(list(colors), np.zeros((h, w), dtype=bool), [], {})
    return solutions


def _render_output(
    shape: Tuple[int, int],
    marker_col: int,
    marker_rows: List[int],
    marker_colors: List[int],
    rects: Dict[int, Rect],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for row, color in zip(marker_rows, marker_colors):
        out[row, marker_col] = int(color)

    marker_rank = {color: idx for idx, color in enumerate(marker_colors)}
    sorted_colors = sorted(marker_colors, key=lambda color: (_rect_area(rects[color]), marker_rank[color]))
    for row, color in zip(marker_rows, sorted_colors):
        rect = rects[color]
        height, width = _rect_dims(rect)
        left = (marker_col - width) // 2
        assert left >= 0
        assert row + height <= shape[0]
        out[row : row + height, left : left + width] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Separate the overlapping colored rectangles and restack the full layers by increasing area.

    Input structure:
    - Black is the background.
    - The left part of the grid contains several axis-aligned colored rectangles drawn with occlusion:
      when rectangles overlap, only the frontmost color stays visible.
    - The right edge contains one isolated single-pixel marker for each rectangle color, arranged from
      top to bottom.

    Rule:
    - Recover each full rectangle by treating every visible color patch as the uncovered part of one
      solid rectangle hidden underneath higher layers.
    - Determine the complete area of every recovered rectangle, including the hidden cells.
    - Clear the overlapping stack, keep the right-edge markers, then place the recovered rectangles
      without overlap: the smallest area goes to the top marker row, the largest to the bottom marker
      row, and every rectangle is horizontally centered.

    Key insight:
    - The visible pixels are only partial evidence of deeper rectangles; the hidden parts can be inferred
      because each layer must still be one intact axis-aligned rectangle underneath the already recovered
      higher layers, and the marker column supplies the target ordering slots.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _mode_color(g)
    marker_col, marker_rows, marker_colors, stage = _parse_scene(g, bg)
    marker_rank = {color: idx for idx, color in enumerate(marker_colors)}
    solutions = _recover_solutions(stage, marker_colors, bg, marker_rank, limit=1)
    assert solutions, "failed to reconstruct the occluded rectangles"
    _order, rects = solutions[0]
    return _render_output(g.shape, marker_col, marker_rows, marker_colors, rects, bg)


def _sample_dimensions(colors: List[int]) -> Dict[int, Tuple[int, int]] | None:
    dims: Dict[int, Tuple[int, int]] = {}
    used_areas = set()
    for color in colors:
        for _ in range(80):
            height = random.randint(*RECT_HEIGHT_RANGE)
            width = random.randint(*RECT_WIDTH_RANGE)
            area = height * width
            if area in used_areas:
                continue
            dims[color] = (height, width)
            used_areas.add(area)
            break
        else:
            return None
    return dims


def _sample_output_rows(sorted_colors: List[int], dims: Dict[int, Tuple[int, int]]) -> Tuple[int, List[int]] | None:
    rows: List[int] = []
    cur = random.randint(*OUTPUT_MARGIN_RANGE)
    for idx, color in enumerate(sorted_colors):
        height, _width = dims[color]
        rows.append(cur)
        cur += height
        if idx + 1 < len(sorted_colors):
            cur += random.randint(*OUTPUT_GAP_RANGE)
    height = cur + random.randint(*OUTPUT_MARGIN_RANGE)
    if height > 30:
        return None
    if height < 14:
        extra = min(30 - height, random.randint(0, 6))
        height += extra
    return height, rows


def _bias_offsets(length: int, bias: str) -> List[int]:
    if bias == "top" or bias == "left":
        return list(range(0, min(length, 2)))
    if bias == "bottom" or bias == "right":
        start = max(0, length - 2)
        return list(range(start, length))
    center = (length - 1) // 2
    return list(range(max(0, center - 1), min(length, center + 2)))


def _place_rectangles(
    colors: List[int], dims: Dict[int, Tuple[int, int]], height: int, stage_width: int
) -> Dict[int, Rect] | None:
    max_h = max(h for h, _w in dims.values())
    max_w = max(w for _h, w in dims.values())
    if height < 2 * max_h or stage_width < 2 * max_w:
        return None

    pivot_r = random.randint(max_h - 1, height - max_h)
    pivot_c = random.randint(max_w - 1, stage_width - max_w)

    rects: Dict[int, Rect] = {}
    modes = [(rb, cb) for rb in ROW_BIASES for cb in COL_BIASES]
    random.shuffle(modes)

    for color in colors:
        rect = None
        height_i, width_i = dims[color]
        for _ in range(40):
            row_bias, col_bias = random.choice(modes)
            off_r = random.choice(_bias_offsets(height_i, row_bias))
            off_c = random.choice(_bias_offsets(width_i, col_bias))
            r0 = pivot_r - off_r
            c0 = pivot_c - off_c
            rect = (r0, c0, r0 + height_i - 1, c0 + width_i - 1)
            if rect not in rects.values():
                break
        else:
            return None
        rects[color] = rect
    return rects


def _render_stage(
    height: int, stage_width: int, rects: Dict[int, Rect], bottom_to_top: List[int]
) -> np.ndarray:
    stage = np.full((height, stage_width), BG, dtype=int)
    for color in bottom_to_top:
        r0, c0, r1, c1 = rects[color]
        stage[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return stage


def _nontrivial_occlusion(stage: np.ndarray, rects: Dict[int, Rect], colors: List[int]) -> bool:
    occluded_count = 0
    fragmented = False
    for color in colors:
        visible = stage == int(color)
        if int(visible.sum()) < MIN_VISIBLE_CELLS:
            return False
        if _bbox_from_mask(visible) != rects[color]:
            return False
        if int(visible.sum()) < _rect_area(rects[color]):
            occluded_count += 1
            if _count_components(visible) >= 2:
                fragmented = True
    return occluded_count >= 1 and fragmented


def _grid_with_markers(
    stage: np.ndarray, marker_rows: List[int], marker_colors: List[int]
) -> np.ndarray:
    height, stage_width = stage.shape
    grid = np.full((height, stage_width + 1), BG, dtype=int)
    grid[:, :stage_width] = stage
    for row, color in zip(marker_rows, marker_colors):
        grid[row, stage_width] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    layer_count = random.choice(LAYER_COUNT_OPTIONS)
    colors = random.sample(PALETTE, layer_count)
    dims = _sample_dimensions(colors)
    if dims is None:
        return None

    sorted_colors = sorted(colors, key=lambda color: dims[color][0] * dims[color][1])
    row_data = _sample_output_rows(sorted_colors, dims)
    if row_data is None:
        return None
    height, marker_rows = row_data

    max_width = max(width for _height, width in dims.values())
    min_stage_width = max(STAGE_WIDTH_RANGE[0], max_width + 5)
    max_stage_width = min(STAGE_WIDTH_RANGE[1], 29)
    if min_stage_width > max_stage_width:
        return None
    stage_width = random.randint(min_stage_width, max_stage_width)

    rects = _place_rectangles(colors, dims, height, stage_width)
    if rects is None:
        return None

    bottom_to_top = colors[:]
    random.shuffle(bottom_to_top)
    stage = _render_stage(height, stage_width, rects, bottom_to_top)

    if not _nontrivial_occlusion(stage, rects, colors):
        return None

    marker_colors = sorted_colors[:]
    grid = _grid_with_markers(stage, marker_rows, marker_colors)

    marker_rank = {color: idx for idx, color in enumerate(marker_colors)}
    recovered = _recover_solutions(stage, marker_colors, BG, marker_rank, limit=2)
    if len(recovered) != 1:
        return None

    expected = _render_output(grid.shape, stage_width, marker_rows, marker_colors, rects, BG)
    try:
        actual = main(grid)
    except Exception:
        return None

    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-rectangles ARC input")
