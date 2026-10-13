import numpy as np


SIDES = ("top", "bottom", "left", "right")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
FRAME_COLOR = 5
LINE_LEN = 2
GENERATION_RETRIES = 300
GRID_MIN_SIZE = 28
GRID_MAX_SIZE = 40
PAIR_COUNT_RANGE = (2, 5)
FRAME_HEIGHT_RANGE = (8, 12)
FRAME_WIDTH_RANGE = (8, 12)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _monochromatic_components(grid: np.ndarray, bg: int) -> list[dict]:
    items: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            items.append({"color": color, "coords": coords, "bbox": _bbox(coords), "area": int(len(coords))})
    return items


def _is_rect_frame(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    if comp["area"] != 2 * (h + w) - 4:
        return False
    color = int(comp["color"])
    box = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(box[border] == color))


def _is_solid_multicolor_rectangle(mask: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return bool(mask[r0 : r1 + 1, c0 : c1 + 1].all())


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _pair_side(stamp_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]) -> str | None:
    sr0, sc0, sr1, sc1 = stamp_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    if sr1 + 1 == fr0 and _overlap_len(sc0, sc1, fc0, fc1) > 0:
        return "top"
    if sr0 - 1 == fr1 and _overlap_len(sc0, sc1, fc0, fc1) > 0:
        return "bottom"
    if sc1 + 1 == fc0 and _overlap_len(sr0, sr1, fr0, fr1) > 0:
        return "left"
    if sc0 - 1 == fc1 and _overlap_len(sr0, sr1, fr0, fr1) > 0:
        return "right"
    return None


def _parse_stamp(patch: np.ndarray) -> tuple[str, int, int]:
    row_colors = [int(row[0]) for row in patch]
    col_colors = [int(col[0]) for col in patch.T]
    rows_uniform = all(np.all(row == row[0]) for row in patch)
    cols_uniform = all(np.all(col == col[0]) for col in patch.T)
    assert rows_uniform != cols_uniform, "stamp stripes must have a unique orientation"

    axis_colors = row_colors if rows_uniform else col_colors
    values, counts = np.unique(axis_colors, return_counts=True)
    base_color = int(values[np.argmax(counts)])

    stripes: list[tuple[int, int, int]] = []
    i = 0
    while i < len(axis_colors):
        color = int(axis_colors[i])
        j = i + 1
        while j < len(axis_colors) and int(axis_colors[j]) == color:
            j += 1
        if color != base_color:
            stripes.append((i, j - 1, color))
        i = j

    assert 1 <= len(stripes) <= 3, "expected 1..3 stripes"
    orientation = "horizontal" if rows_uniform else "vertical"
    return orientation, len(stripes), int(stripes[-1][2])


def _segment_starts(length: int, count: int, seg_len: int) -> list[int]:
    assert count >= 1
    assert length >= count * seg_len
    free = length - count * seg_len
    gap_base = free // (count + 1)
    extra = free % (count + 1)
    gaps = [gap_base] * (count + 1)
    for i in range(extra):
        gaps[i + 1] += 1
    starts: list[int] = []
    cursor = gaps[0]
    for i in range(count):
        starts.append(cursor)
        cursor += seg_len + gaps[i + 1]
    return starts


def _decoration_coords(
    frame_bbox: tuple[int, int, int, int], side: str, orientation: str, count: int
) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    coords: list[tuple[int, int]] = []
    if side in ("top", "bottom"):
        span_start = c0 + 1
        span_len = c1 - c0 - 1
        along = LINE_LEN if orientation == "horizontal" else 1
        starts = _segment_starts(span_len, count, along)
        if side == "top":
            row_base = r0 - LINE_LEN
            for off in starts:
                if orientation == "horizontal":
                    rr = r0 - 1
                    cc0 = span_start + off
                    for cc in range(cc0, cc0 + LINE_LEN):
                        coords.append((rr, cc))
                else:
                    cc = span_start + off
                    for rr in range(row_base, r0):
                        coords.append((rr, cc))
        else:
            row_base = r1 + 1
            for off in starts:
                if orientation == "horizontal":
                    rr = r1 + 1
                    cc0 = span_start + off
                    for cc in range(cc0, cc0 + LINE_LEN):
                        coords.append((rr, cc))
                else:
                    cc = span_start + off
                    for rr in range(row_base, row_base + LINE_LEN):
                        coords.append((rr, cc))
    else:
        span_start = r0 + 1
        span_len = r1 - r0 - 1
        along = LINE_LEN if orientation == "vertical" else 1
        starts = _segment_starts(span_len, count, along)
        if side == "left":
            col_base = c0 - LINE_LEN
            for off in starts:
                if orientation == "vertical":
                    cc = c0 - 1
                    rr0 = span_start + off
                    for rr in range(rr0, rr0 + LINE_LEN):
                        coords.append((rr, cc))
                else:
                    rr = span_start + off
                    for cc in range(col_base, c0):
                        coords.append((rr, cc))
        else:
            col_base = c1 + 1
            for off in starts:
                if orientation == "vertical":
                    cc = c1 + 1
                    rr0 = span_start + off
                    for rr in range(rr0, rr0 + LINE_LEN):
                        coords.append((rr, cc))
                else:
                    rr = span_start + off
                    for cc in range(col_base, col_base + LINE_LEN):
                        coords.append((rr, cc))
    return np.array(coords, dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    For each stamp attached to a rectangular frame, copy stripe instructions to the opposite side.

    The stamp encodes:
    - which side it touches,
    - stripe orientation,
    - stripe count,
    - the last stripe color.

    The output keeps the stamp and frame unchanged, then draws the corresponding
    decoration lines outside the opposite side of the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    mono = _monochromatic_components(g, bg)
    frames = [comp for comp in mono if _is_rect_frame(g, comp)]
    assert frames, "expected at least one envelope frame"

    frame_mask = np.zeros_like(g, dtype=bool)
    for frame in frames:
        frame_mask[frame["coords"][:, 0], frame["coords"][:, 1]] = True

    stamp_mask = (g != bg) & (~frame_mask)
    stamps = []
    for coords in _components(stamp_mask):
        if not _is_solid_multicolor_rectangle(stamp_mask, coords):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if len(np.unique(patch)) < 2:
            continue
        stamps.append({"coords": coords, "bbox": (r0, c0, r1, c1), "patch": patch})

    assert stamps, "expected at least one stamp"

    out = g.copy()
    for stamp in stamps:
        matches = []
        for idx, frame in enumerate(frames):
            side = _pair_side(stamp["bbox"], frame["bbox"])
            if side is not None:
                matches.append((idx, side))
        assert len(matches) == 1, "each stamp must match exactly one frame"

        frame = frames[matches[0][0]]
        stamp_side = matches[0][1]
        deco_side = OPPOSITE[stamp_side]
        orientation, count, last_color = _parse_stamp(stamp["patch"])
        deco = _decoration_coords(frame["bbox"], deco_side, orientation, count)
        out[deco[:, 0], deco[:, 1]] = last_color

    return out


def _sample_colors(bg: int, needed: int) -> list[int]:
    palette = [c for c in range(10) if c not in (bg, FRAME_COLOR)]
    chosen = list(np.random.choice(palette, size=needed, replace=False))
    return [int(c) for c in chosen]


def _make_stamp_patch(orientation: str, count: int, stamp_h: int, stamp_w: int, colors: list[int]) -> np.ndarray:
    patch = np.full((stamp_h, stamp_w), int(colors[0]), dtype=int)
    axis_len = stamp_h if orientation == "horizontal" else stamp_w
    stripe_slots = np.sort(np.random.choice(np.arange(1, axis_len - 1), size=count, replace=False))
    for idx, pos in enumerate(stripe_slots, start=1):
        if orientation == "horizontal":
            patch[int(pos), :] = int(colors[idx])
        else:
            patch[:, int(pos)] = int(colors[idx])
    return patch


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([dr, dc], dtype=int)


def _build_pair(bg: int) -> dict:
    stamp_side = str(np.random.choice(SIDES))
    orientation = str(np.random.choice(("horizontal", "vertical")))
    count = int(np.random.randint(1, 4))

    if orientation == "horizontal":
        stamp_h = int(np.random.randint(count + 2, count + 5))
        stamp_w = int(np.random.randint(4, 7))
    else:
        stamp_h = int(np.random.randint(4, 7))
        stamp_w = int(np.random.randint(count + 2, count + 5))

    min_frame_h = max(FRAME_HEIGHT_RANGE[0], stamp_h + 2)
    min_frame_w = max(FRAME_WIDTH_RANGE[0], stamp_w + 2)
    frame_h = int(np.random.randint(min_frame_h, FRAME_HEIGHT_RANGE[1] + 1))
    frame_w = int(np.random.randint(min_frame_w, FRAME_WIDTH_RANGE[1] + 1))

    frame = np.full((frame_h, frame_w), bg, dtype=int)
    frame[0, :] = FRAME_COLOR
    frame[-1, :] = FRAME_COLOR
    frame[:, 0] = FRAME_COLOR
    frame[:, -1] = FRAME_COLOR

    colors = _sample_colors(bg, count + 1)
    stamp = _make_stamp_patch(orientation, count, stamp_h, stamp_w, colors)

    frame_bbox = (0, 0, frame_h - 1, frame_w - 1)
    if stamp_side == "top":
        sc0 = int(np.random.randint(1, frame_w - stamp_w))
        sr0 = -stamp_h
    elif stamp_side == "bottom":
        sc0 = int(np.random.randint(1, frame_w - stamp_w))
        sr0 = frame_h
    elif stamp_side == "left":
        sr0 = int(np.random.randint(1, frame_h - stamp_h))
        sc0 = -stamp_w
    else:
        sr0 = int(np.random.randint(1, frame_h - stamp_h))
        sc0 = frame_w

    stamp_coords = np.array([(r, c) for r in range(stamp_h) for c in range(stamp_w)], dtype=int)
    stamp_coords = _translate(stamp_coords, sr0, sc0)

    deco_side = OPPOSITE[stamp_side]
    deco_coords = _decoration_coords(frame_bbox, deco_side, orientation, count)

    frame_coords = np.array([(r, c) for r in range(frame_h) for c in range(frame_w) if frame[r, c] == FRAME_COLOR], dtype=int)
    all_coords = np.vstack([frame_coords, stamp_coords, deco_coords])
    r0, c0, r1, c1 = _bbox(all_coords)

    return {
        "frame": frame,
        "frame_bbox": frame_bbox,
        "stamp": stamp,
        "stamp_offset": (sr0, sc0),
        "stamp_side": stamp_side,
        "orientation": orientation,
        "count": count,
        "deco_coords": deco_coords,
        "local_bbox": (r0, c0, r1, c1),
    }


def _place_pair(grid: np.ndarray, pair: dict, top: int, left: int) -> None:
    frame = pair["frame"]
    fh, fw = frame.shape
    grid[top : top + fh, left : left + fw][frame == FRAME_COLOR] = FRAME_COLOR

    stamp = pair["stamp"]
    sr0, sc0 = pair["stamp_offset"]
    abs_sr0 = top + sr0
    abs_sc0 = left + sc0
    grid[abs_sr0 : abs_sr0 + stamp.shape[0], abs_sc0 : abs_sc0 + stamp.shape[1]] = stamp


def _gen_attempt() -> np.ndarray | None:
    bg_choices = [c for c in range(10) if c != FRAME_COLOR]
    bg = int(np.random.choice(bg_choices))
    h = int(np.random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE + 1))
    w = int(np.random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE + 1))
    grid = np.full((h, w), bg, dtype=int)

    n_pairs = int(np.random.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1]))
    reserved_boxes: list[tuple[int, int, int, int]] = []

    for _ in range(n_pairs):
        placed = False
        for _ in range(200):
            pair = _build_pair(bg)
            lr0, lc0, lr1, lc1 = pair["local_bbox"]
            top_min = -lr0
            top_max = h - 1 - lr1
            left_min = -lc0
            left_max = w - 1 - lc1
            if top_min > top_max or left_min > left_max:
                continue

            top = int(np.random.randint(top_min, top_max + 1))
            left = int(np.random.randint(left_min, left_max + 1))
            box = (
                top + lr0 - 1,
                left + lc0 - 1,
                top + lr1 + 1,
                left + lc1 + 1,
            )

            overlaps = False
            for or0, oc0, or1, oc1 in reserved_boxes:
                if not (box[2] < or0 or or1 < box[0] or box[3] < oc0 or oc1 < box[1]):
                    overlaps = True
                    break
            if overlaps:
                continue

            _place_pair(grid, pair, top, left)
            reserved_boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
