from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
ORANGE = 7
GROUP_COLORS = (1, 2, 3, 6, 8, 9)

LEGEND_BAND_HEIGHT = 3
LEGEND_COLOR_BLOCK_WIDTH = 2
GRID_WIDTH_RANGE = (28, 42)
SCENE_HEIGHT_RANGE = (18, 30)
GROUP_COUNT_RANGE = (2, 4)
GENERATION_RETRIES = 400
PLACEMENT_ATTEMPTS = 400
GROUP_MARGIN = 1

PATTERN_POOL: tuple[np.ndarray, ...] = (
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [0, 1]], dtype=bool),
    np.array([[0, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _column_intervals(active: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(active.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            intervals.append((start, idx - 1))
            start = None
    if start is not None:
        intervals.append((start, len(active) - 1))
    return intervals


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _fill_gap(grid: np.ndarray, outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = outer
    ir0, ic0, ir1, ic1 = inner
    gap = np.zeros_like(grid, dtype=bool)
    gap[r0 + 1 : r1, c0 + 1 : c1] = True
    gap[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
    grid[gap] = ORANGE


def _frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    expected_area = 2 * (h + w) - 4
    if len(coords) != expected_area:
        return None
    cells = set(coords)
    for r, c in cells:
        if r not in (r0, r1) and c not in (c0, c1):
            return None
    for r in range(r0, r1 + 1):
        if (r, c0) not in cells or (r, c1) not in cells:
            return None
    for c in range(c0, c1 + 1):
        if (r0, c) not in cells or (r1, c) not in cells:
            return None
    return r0, c0, r1, c1


def _parse_legend(grid: np.ndarray) -> tuple[int, dict[int, np.ndarray]]:
    grey_rows = np.where(grid == GREY)[0]
    assert grey_rows.size > 0, "legend must contain grey symbol pixels"
    legend_top = int(grey_rows.min())
    legend = grid[legend_top:, :]
    intervals = _column_intervals(np.any(legend != BLACK, axis=0))
    mapping: dict[int, np.ndarray] = {}
    for c0, c1 in intervals:
        sub = legend[:, c0 : c1 + 1]
        colors = set(map(int, np.unique(sub))) - {BLACK, GREY}
        assert len(colors) == 1, "each legend entry must expose exactly one source color"
        color = colors.pop()
        grey = sub == GREY
        assert grey.any(), "each legend entry must contain a grey symbol"
        pts = list(zip(*np.where(grey), strict=False))
        gr0, gc0, gr1, gc1 = _bbox([(int(r), int(c)) for r, c in pts])
        mapping[color] = grey[gr0 : gr1 + 1, gc0 : gc1 + 1].copy()
    assert mapping, "expected at least one legend entry"
    return legend_top, mapping


def _parse_groups(scene: np.ndarray) -> list[tuple[int, list[tuple[int, int, int, int]]]]:
    colors = [int(c) for c in np.unique(scene) if int(c) != BLACK]
    groups: list[tuple[int, list[tuple[int, int, int, int]]]] = []
    accounted = 0
    for color in colors:
        comps = _components(scene == color)
        bboxes = []
        area_sum = 0
        for comp in comps:
            area_sum += len(comp)
            frame = _frame_bbox(comp)
            assert frame is not None, "scene objects must be hollow rectangular frames"
            bboxes.append(frame)
        assert 2 <= len(bboxes) <= 4, "each color group must have 2 to 4 nested frames"
        bboxes.sort(key=lambda b: ((b[2] - b[0] + 1) * (b[3] - b[1] + 1),), reverse=True)
        for outer, inner in zip(bboxes, bboxes[1:]):
            assert outer[0] + 1 < inner[0] and outer[1] + 1 < inner[1], "frames must be nested with a gap"
            assert inner[2] < outer[2] - 1 and inner[3] < outer[3] - 1, "frames must be nested with a gap"
        groups.append((color, bboxes))
        accounted += area_sum
    assert accounted == int(np.count_nonzero(scene != BLACK)), "unexpected non-frame scene pixels"
    return groups


def _render_output(
    shape: tuple[int, int],
    legend_top: int,
    groups: list[tuple[int, list[tuple[int, int, int, int]]]],
    legend_map: dict[int, np.ndarray],
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for color, bboxes in groups:
        assert color in legend_map, "missing legend symbol for group color"
        for idx, bbox in enumerate(bboxes, start=1):
            frame_color = color if idx % 2 == 1 else YELLOW
            _draw_frame(out, bbox, frame_color)
            if idx % 2 == 0 and idx < len(bboxes):
                _fill_gap(out, bbox, bboxes[idx])
        inner = bboxes[-1]
        pattern = legend_map[color]
        ph, pw = pattern.shape
        ih = inner[2] - inner[0] - 1
        iw = inner[3] - inner[1] - 1
        assert ih >= ph and iw >= pw, "inner frame must fit its symbol"
        start_r = inner[0] + 1 + (ih - ph) // 2
        start_c = inner[1] + 1 + (iw - pw) // 2
        patch = out[start_r : start_r + ph, start_c : start_c + pw]
        patch[pattern] = GREY
    out[legend_top:, :] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor even nested frames and place the legend symbol inside each innermost frame.
    """
    g = np.asarray(grid, dtype=int)
    legend_top, legend_map = _parse_legend(g)
    scene = g[:legend_top, :]
    groups = _parse_groups(scene)
    return _render_output(g.shape, legend_top, groups, legend_map)


def _shift_bboxes(
    bboxes: list[tuple[int, int, int, int]], top: int, left: int
) -> list[tuple[int, int, int, int]]:
    return [(r0 + top, c0 + left, r1 + top, c1 + left) for r0, c0, r1, c1 in bboxes]


def _sample_group_pattern() -> np.ndarray:
    return random.choice(PATTERN_POOL).copy()


def _sample_group_bboxes(pattern: np.ndarray) -> list[tuple[int, int, int, int]] | None:
    ph, pw = pattern.shape
    layers = random.randint(2, 4)
    for _ in range(40):
        row_insets = [random.choice((2, 2, 3)) for _layer in range(layers - 1)]
        col_insets = [random.choice((2, 2, 3)) for _layer in range(layers - 1)]
        max_inner_h = 17 - 2 - 2 * sum(row_insets)
        max_inner_w = 19 - 2 - 2 * sum(col_insets)
        min_inner_h = ph
        min_inner_w = pw
        if max_inner_h < min_inner_h or max_inner_w < min_inner_w:
            continue
        inner_h = random.randint(min_inner_h, min(max_inner_h, min_inner_h + 4))
        inner_w = random.randint(min_inner_w, min(max_inner_w, min_inner_w + 5))
        bboxes = [(0, 0, inner_h + 1, inner_w + 1)]
        for inset_r, inset_c in zip(row_insets, col_insets, strict=True):
            r0, c0, r1, c1 = bboxes[0]
            bboxes.insert(0, (r0 - inset_r, c0 - inset_c, r1 + inset_r, c1 + inset_c))
        shift_r = -bboxes[0][0]
        shift_c = -bboxes[0][1]
        shifted = _shift_bboxes(bboxes, shift_r, shift_c)
        outer = shifted[0]
        oh = outer[2] - outer[0] + 1
        ow = outer[3] - outer[1] + 1
        if oh <= 17 and ow <= 19:
            return shifted
    return None


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _place_groups(
    specs: list[dict[str, object]], scene_h: int, width: int
) -> list[dict[str, object]] | None:
    placed: list[dict[str, object]] = []
    for spec in sorted(specs, key=lambda item: item["area"], reverse=True):
        local_bboxes = spec["local_bboxes"]
        outer = local_bboxes[0]
        oh = outer[2] - outer[0] + 1
        ow = outer[3] - outer[1] + 1
        if oh > scene_h or ow > width:
            return None
        placed_here = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(0, scene_h - oh)
            left = random.randint(0, width - ow)
            abs_outer = (top, left, top + oh - 1, left + ow - 1)
            if any(_rectangles_overlap(abs_outer, other["outer_bbox"], GROUP_MARGIN) for other in placed):
                continue
            item = dict(spec)
            item["bboxes"] = _shift_bboxes(local_bboxes, top, left)
            item["outer_bbox"] = abs_outer
            placed.append(item)
            placed_here = True
            break
        if not placed_here:
            return None
    return placed


def _entry_total_width(pattern: np.ndarray) -> int:
    return LEGEND_COLOR_BLOCK_WIDTH + pattern.shape[1]


def _sample_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    assert sum(out) == total
    return out


def _legend_positions(width: int, patterns: list[np.ndarray]) -> list[int] | None:
    widths = [_entry_total_width(pattern) for pattern in patterns]
    minimum = sum(widths) + max(0, len(widths) - 1)
    if minimum > width:
        return None
    extra = width - minimum
    slack = _sample_composition(extra, len(widths) + 1)
    starts: list[int] = []
    pos = slack[0]
    for idx, entry_w in enumerate(widths):
        starts.append(pos)
        pos += entry_w
        if idx + 1 < len(widths):
            pos += 1 + slack[idx + 1]
    return starts


def _build_input(
    shape: tuple[int, int],
    legend_top: int,
    groups: list[dict[str, object]],
    legend_order: list[int],
    starts: list[int],
) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BLACK, dtype=int)
    for group in groups:
        color = int(group["color"])
        for bbox in group["bboxes"]:
            _draw_frame(grid, bbox, color)
    ordered_groups = {int(group["color"]): group for group in groups}
    for color, start in zip(legend_order, starts, strict=True):
        pattern = ordered_groups[color]["pattern"]
        grid[legend_top : legend_top + LEGEND_BAND_HEIGHT, start : start + LEGEND_COLOR_BLOCK_WIDTH] = color
        sym_left = start + LEGEND_COLOR_BLOCK_WIDTH
        grid[legend_top : legend_top + pattern.shape[0], sym_left : sym_left + pattern.shape[1]][pattern] = GREY
    return grid


def _spec_output(
    shape: tuple[int, int], legend_top: int, groups: list[dict[str, object]]
) -> np.ndarray:
    parsed_groups = [
        (int(group["color"]), list(group["bboxes"]))
        for group in groups
    ]
    legend_map = {int(group["color"]): np.asarray(group["pattern"], dtype=bool) for group in groups}
    return _render_output(shape, legend_top, parsed_groups, legend_map)


def generate_input() -> np.ndarray:
    """
    Generate nested rectangular frame groups with a bottom legend of color→symbol entries.
    """
    for _ in range(GENERATION_RETRIES):
        group_count = random.randint(*GROUP_COUNT_RANGE)
        colors = random.sample(GROUP_COLORS, group_count)
        patterns = random.sample(list(PATTERN_POOL), group_count)
        specs: list[dict[str, object]] = []
        ok = True
        for color, pattern in zip(colors, patterns, strict=True):
            bboxes = _sample_group_bboxes(pattern)
            if bboxes is None:
                ok = False
                break
            outer = bboxes[0]
            specs.append(
                {
                    "color": color,
                    "pattern": pattern,
                    "local_bboxes": bboxes,
                    "area": (outer[2] - outer[0] + 1) * (outer[3] - outer[1] + 1),
                }
            )
        if not ok:
            continue

        width = random.randint(*GRID_WIDTH_RANGE)
        scene_h = random.randint(*SCENE_HEIGHT_RANGE)
        placed = _place_groups(specs, scene_h, width)
        if placed is None:
            continue

        legend_patterns = [np.asarray(group["pattern"], dtype=bool) for group in placed]
        legend_order = random.sample([int(group["color"]) for group in placed], len(placed))
        ordered_patterns = [next(np.asarray(group["pattern"], dtype=bool) for group in placed if int(group["color"]) == color) for color in legend_order]
        starts = _legend_positions(width, ordered_patterns)
        if starts is None:
            continue

        height = scene_h + LEGEND_BAND_HEIGHT
        legend_top = scene_h
        inp = _build_input((height, width), legend_top, placed, legend_order, starts)
        out = _spec_output((height, width), legend_top, placed)
        try:
            solved = main(inp)
        except Exception:
            continue
        if not np.array_equal(solved, out):
            continue
        if np.array_equal(inp, out):
            continue
        if int(np.count_nonzero(out == GREY)) < group_count:
            continue
        return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
