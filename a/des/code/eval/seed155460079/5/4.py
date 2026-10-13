import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
AZURE = 8
MAROON = 9

SEPARATOR_COLORS = (BLUE, RED, GREEN, YELLOW, PINK, ORANGE, AZURE)
MAX_GENERATION_ATTEMPTS = 400
MIN_REGION_SPAN = 7

BASE_PATTERNS = [
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1], [1]], dtype=bool),
    np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0, 1], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=bool),
]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        pts: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            pts.append((cr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_output_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for coords in _components(grid == MAROON):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        expected = np.zeros((h, w), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        if not np.array_equal(patch == MAROON, expected):
            continue
        candidates.append((r0 + 1, c0 + 1, r1 - 1, c1 - 1))
    assert len(candidates) == 1, "expected a unique maroon output frame"
    return candidates[0]


def _separator_infos(grid: np.ndarray) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {BLACK, GREY, MAROON}):
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h == 1 and len(coords) == w and w >= 5:
                infos.append(
                    {
                        "color": color,
                        "bbox": (r0, c0, r1, c1),
                        "orientation": "H",
                        "coords": coords,
                    }
                )
            elif w == 1 and len(coords) == h and h >= 5:
                infos.append(
                    {
                        "color": color,
                        "bbox": (r0, c0, r1, c1),
                        "orientation": "V",
                        "coords": coords,
                    }
                )
    assert infos, "expected separator lines"
    return infos


def _region_labels(grid: np.ndarray, separators: list[dict[str, object]]) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]]]:
    blocked = np.zeros_like(grid, dtype=bool)
    for info in separators:
        rows = info["coords"][:, 0]
        cols = info["coords"][:, 1]
        blocked[rows, cols] = True

    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    next_label = 0
    for r in range(h):
        for c in range(w):
            if blocked[r, c] or labels[r, c] != -1:
                continue
            stack = [(r, c)]
            labels[r, c] = next_label
            r0 = r1 = r
            c0 = c1 = c
            while stack:
                cr, cc = stack.pop()
                r0 = min(r0, cr)
                r1 = max(r1, cr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = next_label
                        stack.append((nr, nc))
            bboxes[next_label] = (r0, c0, r1, c1)
            next_label += 1
    return labels, bboxes


def _adjacent_region_labels(labels: np.ndarray, line_info: dict[str, object]) -> tuple[int, int]:
    r0, c0, r1, c1 = line_info["bbox"]
    if line_info["orientation"] == "H":
        assert r0 > 0 and r0 < labels.shape[0] - 1
        above = {int(labels[r0 - 1, c]) for c in range(c0, c1 + 1) if labels[r0 - 1, c] >= 0}
        below = {int(labels[r0 + 1, c]) for c in range(c0, c1 + 1) if labels[r0 + 1, c] >= 0}
        assert len(above) == 1 and len(below) == 1
        return next(iter(above)), next(iter(below))
    assert c0 > 0 and c0 < labels.shape[1] - 1
    left = {int(labels[r, c0 - 1]) for r in range(r0, r1 + 1) if labels[r, c0 - 1] >= 0}
    right = {int(labels[r, c0 + 1]) for r in range(r0, r1 + 1) if labels[r, c0 + 1] >= 0}
    assert len(left) == 1 and len(right) == 1
    return next(iter(left)), next(iter(right))


def _grey_offsets(grid: np.ndarray, labels: np.ndarray, label: int, bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    coords = np.argwhere((labels == label) & (grid == GREY))
    assert len(coords) > 0, "each target region must contain a grey pattern"
    return {(int(r - center_r), int(c - center_c)) for r, c in coords}


def _paint_offsets(grid: np.ndarray, interior_bbox: tuple[int, int, int, int], offsets: set[tuple[int, int]]) -> None:
    r0, c0, r1, c1 = interior_bbox
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    for dr, dc in offsets:
        rr = center_r + dr
        cc = center_c + dc
        assert r0 <= rr <= r1 and c0 <= cc <= c1
        grid[rr, cc] = GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the uniquely colored separator line, take the two regions it directly separates,
    overlay their centered grey motifs by union, and write the result into the maroon frame.
    """
    g = np.asarray(grid, dtype=int)
    interior_bbox = _find_output_frame(g)
    separators = _separator_infos(g)

    counts: dict[int, int] = {}
    for info in separators:
        color = int(info["color"])
        counts[color] = counts.get(color, 0) + 1
    unique_colors = [color for color, n in counts.items() if n == 1]
    assert len(unique_colors) == 1, "expected exactly one uniquely colored separator line"
    unique_color = unique_colors[0]
    unique_lines = [info for info in separators if int(info["color"]) == unique_color]
    assert len(unique_lines) == 1
    unique_line = unique_lines[0]

    labels, bboxes = _region_labels(g, separators)
    label_a, label_b = _adjacent_region_labels(labels, unique_line)
    offsets = _grey_offsets(g, labels, label_a, bboxes[label_a]) | _grey_offsets(g, labels, label_b, bboxes[label_b])

    out = g.copy()
    _paint_offsets(out, interior_bbox, offsets)
    return out


def _area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _odd_range(lo: int, hi: int) -> list[int]:
    return [v for v in range(lo, hi + 1) if v % 2 == 1]


def _sample_pattern() -> np.ndarray:
    pattern = random.choice(BASE_PATTERNS).copy()
    if random.choice((False, True)):
        pattern = np.fliplr(pattern)
    pattern = np.rot90(pattern, random.randrange(4))
    return pattern.astype(bool)


def _draw_hline(grid: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    grid[row, c0 : c1 + 1] = color


def _draw_vline(grid: np.ndarray, col: int, r0: int, r1: int, color: int) -> None:
    grid[r0 : r1 + 1, col] = color


def _draw_frame(grid: np.ndarray, outer_bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = outer_bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _place_pattern(grid: np.ndarray, rect: tuple[int, int, int, int], pattern: np.ndarray) -> None:
    r0, c0, r1, c1 = rect
    ph, pw = pattern.shape
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    top = center_r - ph // 2
    left = center_c - pw // 2
    assert r0 <= top and top + ph - 1 <= r1
    assert c0 <= left and left + pw - 1 <= c1
    patch = grid[top : top + ph, left : left + pw]
    patch[pattern] = GREY


def _pattern_offsets_from_shape(pattern: np.ndarray) -> set[tuple[int, int]]:
    ph, pw = pattern.shape
    center_r = ph // 2
    center_c = pw // 2
    return {(int(r - center_r), int(c - center_c)) for r, c in np.argwhere(pattern)}


def _build_expected(inp: np.ndarray, interior_bbox: tuple[int, int, int, int], target_patterns: list[np.ndarray]) -> np.ndarray:
    out = inp.copy()
    offsets: set[tuple[int, int]] = set()
    for pattern in target_patterns:
        offsets |= _pattern_offsets_from_shape(pattern)
    _paint_offsets(out, interior_bbox, offsets)
    return out


def _sample_layout(unique_orientation: str) -> dict[str, object] | None:
    for _ in range(200):
        top_h = random.choice(_odd_range(15, 19))
        bottom_h = random.choice(_odd_range(11, 15))
        H = top_h + 1 + bottom_h

        W = random.choice(_odd_range(23, 31))
        output_w_choices = [w for w in _odd_range(11, 15) if W - 1 - w >= MIN_REGION_SPAN and (W - 1 - w) % 2 == 1]
        if not output_w_choices:
            continue
        output_w = random.choice(output_w_choices)
        bottom_left_w = W - 1 - output_w

        left_min = 15 if unique_orientation == "V" else MIN_REGION_SPAN
        left_choices = [w for w in _odd_range(left_min, W - 1 - MIN_REGION_SPAN) if W - 1 - w >= MIN_REGION_SPAN]
        if not left_choices:
            continue
        left_top_w = random.choice(left_choices)
        right_top_w = W - 1 - left_top_w

        rt_upper_choices = [h for h in _odd_range(MIN_REGION_SPAN, top_h - 1 - MIN_REGION_SPAN) if top_h - 1 - h >= MIN_REGION_SPAN]
        if not rt_upper_choices:
            continue
        rt_upper_h = random.choice(rt_upper_choices)

        rects: dict[str, tuple[int, int, int, int]] = {
            "bottom_left": (top_h + 1, 0, H - 1, bottom_left_w - 1),
            "output": (top_h + 1, bottom_left_w + 1, H - 1, W - 1),
            "rt_upper": (0, left_top_w + 1, rt_upper_h - 1, W - 1),
            "rt_lower": (rt_upper_h + 1, left_top_w + 1, top_h - 1, W - 1),
        }

        if unique_orientation == "V":
            left_choices_u = [w for w in _odd_range(MIN_REGION_SPAN, left_top_w - 1 - MIN_REGION_SPAN) if left_top_w - 1 - w >= MIN_REGION_SPAN]
            if not left_choices_u:
                continue
            left_target_w = random.choice(left_choices_u)
            rects["target_a"] = (0, 0, top_h - 1, left_target_w - 1)
            rects["target_b"] = (0, left_target_w + 1, top_h - 1, left_top_w - 1)
            unique_line = {"orientation": "V", "bbox": (0, left_target_w, top_h - 1, left_target_w)}
        else:
            top_choices_u = [h for h in _odd_range(MIN_REGION_SPAN, top_h - 1 - MIN_REGION_SPAN) if top_h - 1 - h >= MIN_REGION_SPAN]
            if not top_choices_u:
                continue
            top_target_h = random.choice(top_choices_u)
            rects["target_a"] = (0, 0, top_target_h - 1, left_top_w - 1)
            rects["target_b"] = (top_target_h + 1, 0, top_h - 1, left_top_w - 1)
            unique_line = {"orientation": "H", "bbox": (top_target_h, 0, top_target_h, left_top_w - 1)}

        if any(min(rect[2] - rect[0] + 1, rect[3] - rect[1] + 1) < MIN_REGION_SPAN for rect in rects.values()):
            continue
        output_area = _area(rects["output"])
        if output_area <= max(_area(rect) for key, rect in rects.items() if key != "output"):
            continue

        return {
            "shape": (H, W),
            "top_h": top_h,
            "bottom_left_w": bottom_left_w,
            "left_top_w": left_top_w,
            "rt_upper_h": rt_upper_h,
            "unique_line": unique_line,
            "rects": rects,
        }
    return None


def _place_output_box(grid: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    interior_h = random.choice((5, 7))
    interior_w = random.choice((5, 7))
    outer_h = interior_h + 2
    outer_w = interior_w + 2
    top_choices = list(range(r0 + 1, r1 - outer_h + 2))
    left_choices = list(range(c0 + 1, c1 - outer_w + 2))
    assert top_choices and left_choices
    top = random.choice(top_choices)
    left = random.choice(left_choices)
    outer_bbox = (top, left, top + outer_h - 1, left + outer_w - 1)
    _draw_frame(grid, outer_bbox, MAROON)
    return (top + 1, left + 1, top + interior_h, left + interior_w)


def _gen_attempt() -> np.ndarray | None:
    unique_orientation = random.choice(("H", "V"))
    layout = _sample_layout(unique_orientation)
    if layout is None:
        return None

    H, W = layout["shape"]
    top_h = int(layout["top_h"])
    bottom_left_w = int(layout["bottom_left_w"])
    left_top_w = int(layout["left_top_w"])
    rt_upper_h = int(layout["rt_upper_h"])
    rects = layout["rects"]
    unique_line = layout["unique_line"]

    color_a, color_b, unique_color = random.sample(SEPARATOR_COLORS, 3)
    grid = np.full((H, W), BLACK, dtype=int)

    _draw_hline(grid, top_h, 0, W - 1, color_a)
    _draw_vline(grid, bottom_left_w, top_h + 1, H - 1, color_b)
    _draw_vline(grid, left_top_w, 0, top_h - 1, color_b)
    _draw_hline(grid, rt_upper_h, left_top_w + 1, W - 1, color_a)

    ur0, uc0, ur1, uc1 = unique_line["bbox"]
    if unique_line["orientation"] == "H":
        _draw_hline(grid, ur0, uc0, uc1, unique_color)
    else:
        _draw_vline(grid, uc0, ur0, ur1, unique_color)

    target_patterns = [_sample_pattern(), _sample_pattern()]
    for key in ("target_a", "target_b"):
        _place_pattern(grid, rects[key], target_patterns[0 if key == "target_a" else 1])
    for key in ("rt_upper", "rt_lower", "bottom_left"):
        _place_pattern(grid, rects[key], _sample_pattern())

    interior_bbox = _place_output_box(grid, rects["output"])
    expected = _build_expected(grid, interior_bbox, target_patterns)

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
