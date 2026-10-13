import random

import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500

PATTERN_LIBRARY = (
    np.array([[1]], dtype=int),
    np.array([[2, 0], [0, 2]], dtype=int),
    np.array([[3, 4], [4, 3]], dtype=int),
    np.array([[7, 7, 7], [0, 7, 0]], dtype=int),
    np.array([[8, 0, 8], [8, 8, 8]], dtype=int),
    np.array([[6, 0, 6], [0, 6, 0], [6, 0, 6]], dtype=int),
    np.array([[9, 0, 0], [0, 9, 0], [0, 0, 9]], dtype=int),
)


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_edge_frame(mask: np.ndarray, coords: np.ndarray) -> bool:
    h, w = mask.shape
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
        return False
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if int(sub.sum()) != expected:
        return False
    if not (np.all(sub[0, :]) and np.all(sub[-1, :]) and np.all(sub[:, 0]) and np.all(sub[:, -1])):
        return False
    return not bool(np.any(sub[1:-1, 1:-1]))


def _find_dictionary_frame(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        mask = grid == color
        for coords in _components(mask):
            if _is_edge_frame(mask, coords):
                candidates.append((int(coords.shape[0]), color, _bbox(coords)))
    assert candidates, "expected an edge-touching framed dictionary"
    candidates.sort(reverse=True)
    _area, frame_color, bbox = candidates[0]
    return bbox, frame_color


def _parse_dictionary(
    grid: np.ndarray, dict_bbox: tuple[int, int, int, int], bg: int
) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = dict_bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    row_has_fg = np.any(inner != bg, axis=1)
    mapping: dict[int, np.ndarray] = {}
    row = 0
    while row < inner.shape[0]:
        if not row_has_fg[row]:
            row += 1
            continue
        end = row
        while end + 1 < inner.shape[0] and row_has_fg[end + 1]:
            end += 1
        band = inner[row : end + 1]
        nz = np.argwhere(band != bg)
        assert nz.size > 0, "dictionary band must contain a marker and a pattern"
        left = int(nz[:, 1].min())
        right = int(nz[:, 1].max())
        marker_rows = np.where(band[:, left] != bg)[0]
        assert marker_rows.size == 1, "dictionary marker must be a single pixel"
        key = int(band[int(marker_rows[0]), left])
        assert key not in mapping, "dictionary keys must be unique"
        assert left + 1 <= right, "dictionary pattern must sit to the right of the marker"
        pattern = band[:, left + 1 : right + 1].copy()
        assert np.any(pattern != bg), "dictionary pattern must contain foreground cells"
        mapping[key] = pattern
        row = end + 1
    assert mapping, "dictionary parsing must produce at least one texture"
    return mapping


def _segment_endpoints(seg: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    if seg["orientation"] == "h":
        return (seg["row"], seg["c0"]), (seg["row"], seg["c1"])
    return (seg["r0"], seg["col"]), (seg["r1"], seg["col"])


def _segment_bbox(seg: dict) -> tuple[int, int, int, int]:
    if seg["orientation"] == "h":
        return seg["row"], seg["c0"], seg["row"], seg["c1"]
    return seg["r0"], seg["col"], seg["r1"], seg["col"]


def _draw_segment(grid: np.ndarray, seg: dict) -> None:
    if seg["orientation"] == "h":
        grid[seg["row"], seg["c0"] : seg["c1"] + 1] = seg["color"]
    else:
        grid[seg["r0"] : seg["r1"] + 1, seg["col"]] = seg["color"]


def _collect_segments(
    grid: np.ndarray, dict_bbox: tuple[int, int, int, int], bg: int
) -> list[dict]:
    scene_mask = np.ones(grid.shape, dtype=bool)
    r0, c0, r1, c1 = dict_bbox
    scene_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    segments: list[dict] = []
    colors = sorted(int(v) for v in np.unique(grid[scene_mask]) if int(v) != bg)
    for color in colors:
        coords = np.argwhere((grid == color) & scene_mask)
        assert coords.shape[0] == 2, "each non-dictionary color must form one unique pair"
        (r0, c0), (r1, c1) = coords.tolist()
        if r0 == r1:
            c0, c1 = sorted((int(c0), int(c1)))
            assert c1 > c0, "pixel pairs must be distinct"
            segments.append({"color": color, "orientation": "h", "row": int(r0), "c0": c0, "c1": c1})
        else:
            assert c0 == c1, "pixel pairs must align horizontally or vertically"
            r0, r1 = sorted((int(r0), int(r1)))
            assert r1 > r0, "pixel pairs must be distinct"
            segments.append({"color": color, "orientation": "v", "col": int(c0), "r0": r0, "r1": r1})
    return segments


def _detect_rectangles(segments: list[dict]) -> list[dict]:
    horizontals = [seg for seg in segments if seg["orientation"] == "h"]
    vertical_lookup = {
        (seg["col"], seg["r0"], seg["r1"]): int(seg["color"])
        for seg in segments
        if seg["orientation"] == "v"
    }
    found: dict[tuple[int, int, int, int], dict] = {}
    for top in horizontals:
        rt = int(top["row"])
        for bottom in horizontals:
            rb = int(bottom["row"])
            if rb <= rt:
                continue

            if int(bottom["c0"]) == int(top["c0"]) + 1 and int(bottom["c1"]) == int(top["c1"]) + 1:
                cl, cr = int(top["c0"]), int(bottom["c1"])
                if cr - cl >= 2 and rb - rt >= 2:
                    left_key = (cl, rt + 1, rb)
                    right_key = (cr, rt, rb - 1)
                    if left_key in vertical_lookup and right_key in vertical_lookup:
                        found[(rt, cl, rb, cr)] = {
                            "bbox": (rt, cl, rb, cr),
                            "mode": "cw",
                            "top": int(top["color"]),
                            "bottom": int(bottom["color"]),
                            "left": int(vertical_lookup[left_key]),
                            "right": int(vertical_lookup[right_key]),
                        }

            if int(bottom["c0"]) == int(top["c0"]) - 1 and int(bottom["c1"]) == int(top["c1"]) - 1:
                cl, cr = int(bottom["c0"]), int(top["c1"])
                if cr - cl >= 2 and rb - rt >= 2:
                    left_key = (cl, rt, rb - 1)
                    right_key = (cr, rt + 1, rb)
                    if left_key in vertical_lookup and right_key in vertical_lookup:
                        found[(rt, cl, rb, cr)] = {
                            "bbox": (rt, cl, rb, cr),
                            "mode": "ccw",
                            "top": int(top["color"]),
                            "bottom": int(bottom["color"]),
                            "left": int(vertical_lookup[left_key]),
                            "right": int(vertical_lookup[right_key]),
                        }
    return [found[key] for key in sorted(found)]


def _winner_color(rect: dict) -> int:
    r0, c0, r1, c1 = rect["bbox"]
    if (c1 - c0) >= (r1 - r0):
        return int(rect["top"])
    return int(rect["left"])


def _tile_fill(grid: np.ndarray, bbox: tuple[int, int, int, int], pattern: np.ndarray) -> None:
    r0, c0, r1, c1 = bbox
    ph, pw = pattern.shape
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            grid[r, c] = int(pattern[(r - r0 - 1) % ph, (c - c0 - 1) % pw])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color pixel pairs into line segments, detect rectangular rings, and tile-fill each enclosed interior from an edge dictionary.

    Input structure:
    - The background is the most common color.
    - Away from the dictionary, every foreground color appears exactly twice, and each pair lies on one row or one column.
    - One edge contains a grey rectangular frame whose interior is a vertical dictionary: each entry has a 1x1 color key immediately left of a small texture sample.

    Rule:
    - First connect every same-color pair by a 1-cell-wide horizontal or vertical segment of that color.
    - Then find sets of four segments that form a closed rectangular ring by meeting around the four corners without overlapping the dictionary.
    - For each detected rectangle, choose the color of the longest side; when horizontal and vertical side lengths tie, use a horizontal side color.
    - Look up that color in the dictionary and tile the corresponding sample over the rectangle interior, leaving the border segments in place.

    Key insight:
    - The puzzle composes three hidden layers: infer the pairwise line drawing, recover which drawn segments really close into rectangles, then use a side-color lookup table to choose the fill texture.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    dict_bbox, _frame_color = _find_dictionary_frame(g, bg)
    texture_by_color = _parse_dictionary(g, dict_bbox, bg)
    segments = _collect_segments(g, dict_bbox, bg)

    out = g.copy()
    for seg in segments:
        _draw_segment(out, seg)

    for rect in _detect_rectangles(segments):
        winner = _winner_color(rect)
        assert winner in texture_by_color, "every chosen edge color must have a dictionary entry"
        _tile_fill(out, rect["bbox"], texture_by_color[winner])
    return out


def _pattern_choices(entry_count: int) -> list[np.ndarray]:
    if entry_count >= 7:
        return [pat.copy() for pat in PATTERN_LIBRARY if pat.shape[0] <= 2]
    return [pat.copy() for pat in PATTERN_LIBRARY]


def _rectangle_segments_from_bbox(
    bbox: tuple[int, int, int, int], mode: str, colors: tuple[int, int, int, int]
) -> list[dict]:
    r0, c0, r1, c1 = bbox
    top, bottom, left, right = colors
    if mode == "cw":
        return [
            {"color": top, "orientation": "h", "row": r0, "c0": c0, "c1": c1 - 1},
            {"color": right, "orientation": "v", "col": c1, "r0": r0, "r1": r1 - 1},
            {"color": bottom, "orientation": "h", "row": r1, "c0": c0 + 1, "c1": c1},
            {"color": left, "orientation": "v", "col": c0, "r0": r0 + 1, "r1": r1},
        ]
    assert mode == "ccw"
    return [
        {"color": top, "orientation": "h", "row": r0, "c0": c0 + 1, "c1": c1},
        {"color": right, "orientation": "v", "col": c1, "r0": r0 + 1, "r1": r1},
        {"color": bottom, "orientation": "h", "row": r1, "c0": c0, "c1": c1 - 1},
        {"color": left, "orientation": "v", "col": c0, "r0": r0, "r1": r1 - 1},
    ]


def _expanded_bbox(
    bbox: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _box_is_free(
    occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1
) -> bool:
    er0, ec0, er1, ec1 = _expanded_bbox(bbox, occupied.shape, pad=pad)
    return not bool(occupied[er0 : er1 + 1, ec0 : ec1 + 1].any())


def _reserve_box(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    er0, ec0, er1, ec1 = _expanded_bbox(bbox, occupied.shape, pad=pad)
    occupied[er0 : er1 + 1, ec0 : ec1 + 1] = True


def _place_box(
    occupied: np.ndarray, box_h: int, box_w: int, pad: int = 1
) -> tuple[int, int] | None:
    h, w = occupied.shape
    options: list[tuple[int, int]] = []
    for r0 in range(h - box_h + 1):
        for c0 in range(w - box_w + 1):
            bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if _box_is_free(occupied, bbox, pad=pad):
                options.append((r0, c0))
    if not options:
        return None
    return random.choice(options)


def _render_dictionary(
    grid: np.ndarray,
    dict_bbox: tuple[int, int, int, int],
    entries: list[tuple[int, np.ndarray]],
) -> None:
    r0, c0, r1, c1 = dict_bbox
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    row = 0
    for key, pattern in entries:
        ph, pw = pattern.shape
        inner[row, 0] = key
        inner[row : row + ph, 1 : 1 + pw] = pattern
        row += ph + 1


def _render_input(
    shape: tuple[int, int],
    dict_bbox: tuple[int, int, int, int],
    entries: list[tuple[int, np.ndarray]],
    segments: list[dict],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    _render_dictionary(grid, dict_bbox, entries)
    for seg in segments:
        (r0, c0), (r1, c1) = _segment_endpoints(seg)
        grid[r0, c0] = seg["color"]
        grid[r1, c1] = seg["color"]
    return grid


def _expected_output(inp: np.ndarray, segments: list[dict], rect_specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for seg in segments:
        _draw_segment(out, seg)
    for spec in sorted(rect_specs, key=lambda item: item["bbox"]):
        _tile_fill(out, spec["bbox"], spec["fill_pattern"])
    return out


def _choose_rect_dims(branch: str) -> tuple[int, int]:
    if branch == "horizontal":
        outer_h = random.randint(4, 7)
        outer_w = random.randint(outer_h + 1, min(10, outer_h + 4))
        return outer_h, outer_w
    if branch == "vertical":
        outer_w = random.randint(4, 7)
        outer_h = random.randint(outer_w + 1, min(10, outer_w + 4))
        return outer_h, outer_w
    side = random.randint(4, 8)
    return side, side


def _dictionary_size(entries: list[tuple[int, np.ndarray]]) -> tuple[int, int]:
    inner_h = sum(pattern.shape[0] for _key, pattern in entries) + max(0, len(entries) - 1)
    inner_w = 1 + max(pattern.shape[1] for _key, pattern in entries)
    return inner_h + 2, inner_w + 2


def _place_dictionary_bbox(
    shape: tuple[int, int], dict_h: int, dict_w: int, side: str
) -> tuple[int, int, int, int]:
    h, w = shape
    if side == "top":
        r0, c0 = 0, random.randint(0, w - dict_w)
    elif side == "bottom":
        r0, c0 = h - dict_h, random.randint(0, w - dict_w)
    elif side == "left":
        r0, c0 = random.randint(0, h - dict_h), 0
    else:
        r0, c0 = random.randint(0, h - dict_h), w - dict_w
    return r0, c0, r0 + dict_h - 1, c0 + dict_w - 1


def _validate_generated_case(
    inp: np.ndarray, expected: np.ndarray, n_rects: int
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if _most_common_color(inp) != BG:
        return False
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    try:
        dict_bbox, _frame_color = _find_dictionary_frame(inp, BG)
        rects = _detect_rectangles(_collect_segments(inp, dict_bbox, BG))
    except Exception:
        return False
    return len(rects) == n_rects


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.choice([1, 2])
    n_decoys = random.randint(0, 2) if n_rects == 1 else 0
    total_colors = 4 * n_rects + n_decoys
    colors = list(COLOR_POOL)
    random.shuffle(colors)
    used_colors = colors[:total_colors]
    pattern_options = _pattern_choices(total_colors)

    color_to_pattern: dict[int, np.ndarray] = {}
    rect_specs: list[dict] = []
    color_idx = 0
    for _ in range(n_rects):
        branch = random.choice(["horizontal", "vertical", "square"])
        outer_h, outer_w = _choose_rect_dims(branch)
        corner_mode = random.choice(["cw", "ccw"])
        top, bottom, left, right = used_colors[color_idx : color_idx + 4]
        color_idx += 4
        fill_pattern = random.choice(pattern_options).copy()
        if branch in ("horizontal", "square"):
            color_to_pattern[top] = fill_pattern.copy()
            color_to_pattern[bottom] = fill_pattern.copy()
            color_to_pattern[left] = random.choice(pattern_options).copy()
            color_to_pattern[right] = random.choice(pattern_options).copy()
        else:
            color_to_pattern[top] = random.choice(pattern_options).copy()
            color_to_pattern[bottom] = random.choice(pattern_options).copy()
            color_to_pattern[left] = fill_pattern.copy()
            color_to_pattern[right] = fill_pattern.copy()
        rect_specs.append(
            {
                "outer_h": outer_h,
                "outer_w": outer_w,
                "corner_mode": corner_mode,
                "colors": (top, bottom, left, right),
                "fill_pattern": fill_pattern,
            }
        )

    decoy_colors = used_colors[color_idx:]
    for color in decoy_colors:
        color_to_pattern[color] = random.choice(pattern_options).copy()

    entries = [(color, color_to_pattern[color].copy()) for color in used_colors]
    random.shuffle(entries)
    dict_h, dict_w = _dictionary_size(entries)

    min_h = max(dict_h, 14)
    min_w = max(dict_w, 14)
    if min_h > 30 or min_w > 30:
        return None
    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    side = random.choice(["top", "bottom", "left", "right"])
    dict_bbox = _place_dictionary_bbox((grid_h, grid_w), dict_h, dict_w, side)

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    _reserve_box(occupied, dict_bbox, pad=1)

    rect_specs.sort(key=lambda spec: spec["outer_h"] * spec["outer_w"], reverse=True)
    segments: list[dict] = []
    for spec in rect_specs:
        pos = _place_box(occupied, spec["outer_h"], spec["outer_w"], pad=1)
        if pos is None:
            return None
        r0, c0 = pos
        bbox = (r0, c0, r0 + spec["outer_h"] - 1, c0 + spec["outer_w"] - 1)
        spec["bbox"] = bbox
        spec["segments"] = _rectangle_segments_from_bbox(bbox, spec["corner_mode"], spec["colors"])
        segments.extend(spec["segments"])
        _reserve_box(occupied, bbox, pad=1)

    for color in decoy_colors:
        placed = False
        for _ in range(80):
            orientation = random.choice(["h", "v"])
            cell_count = random.randint(3, 8)
            if orientation == "h":
                if grid_w < cell_count:
                    continue
                row = random.randint(0, grid_h - 1)
                c0 = random.randint(0, grid_w - cell_count)
                seg = {"color": color, "orientation": "h", "row": row, "c0": c0, "c1": c0 + cell_count - 1}
            else:
                if grid_h < cell_count:
                    continue
                col = random.randint(0, grid_w - 1)
                r0 = random.randint(0, grid_h - cell_count)
                seg = {"color": color, "orientation": "v", "col": col, "r0": r0, "r1": r0 + cell_count - 1}
            bbox = _segment_bbox(seg)
            if not _box_is_free(occupied, bbox, pad=1):
                continue
            segments.append(seg)
            if len(_detect_rectangles(segments)) != n_rects:
                segments.pop()
                continue
            _reserve_box(occupied, bbox, pad=1)
            placed = True
            break
        if not placed:
            return None

    if len(_detect_rectangles(segments)) != n_rects:
        return None

    inp = _render_input((grid_h, grid_w), dict_bbox, entries, segments)
    expected = _expected_output(inp, segments, rect_specs)
    if not _validate_generated_case(inp, expected, n_rects):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid input for task 4")
