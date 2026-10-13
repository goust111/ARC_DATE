import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 160


def _bbox_from_coords(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return (min(rows), min(cols), max(rows), max(cols))


def _bbox_hw(bbox):
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _bbox_gap_distance(a, b):
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    if ar1 < br0:
        row_gap = br0 - ar1 - 1
    elif br1 < ar0:
        row_gap = ar0 - br1 - 1
    else:
        row_gap = 0
    if ac1 < bc0:
        col_gap = bc0 - ac1 - 1
    elif bc1 < ac0:
        col_gap = ac0 - bc1 - 1
    else:
        col_gap = 0
    return row_gap + col_gap


def _bbox_center_distance(a, b):
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    ac_r = ar0 + ar1
    ac_c = ac0 + ac1
    bc_r = br0 + br1
    bc_c = bc0 + bc1
    return abs(ac_r - bc_r) + abs(ac_c - bc_c)


def _components_by_color(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": _bbox_from_coords(coords)})
    return comps


def _detect_palette(grid):
    h, w = grid.shape
    candidates = []

    def add_candidate(rows, col, name):
        if not rows:
            return
        colors = [int(grid[r, col]) for r in rows]
        if any(color == BG for color in colors):
            return
        if len(colors) < 2 or len(set(colors)) != len(colors):
            return
        if w > 1:
            adj_col = 1 if col == 0 else w - 2
            if any(int(grid[r, adj_col]) != BG for r in rows):
                return
        candidates.append(
            {
                "name": name,
                "colors": colors,
                "coords": {(r, col) for r in rows},
                "bbox": (rows[0], col, rows[-1], col),
            }
        )

    if int(grid[0, 0]) != BG:
        end = 0
        while end < h and int(grid[end, 0]) != BG:
            end += 1
        add_candidate(list(range(end)), 0, "top_left")
    if int(grid[0, w - 1]) != BG:
        end = 0
        while end < h and int(grid[end, w - 1]) != BG:
            end += 1
        add_candidate(list(range(end)), w - 1, "top_right")
    if int(grid[h - 1, 0]) != BG:
        start = h - 1
        while start >= 0 and int(grid[start, 0]) != BG:
            start -= 1
        add_candidate(list(range(start + 1, h)), 0, "bottom_left")
    if int(grid[h - 1, w - 1]) != BG:
        start = h - 1
        while start >= 0 and int(grid[start, w - 1]) != BG:
            start -= 1
        add_candidate(list(range(start + 1, h)), w - 1, "bottom_right")

    if not candidates:
        raise ValueError("palette not found")
    candidates.sort(key=lambda item: (-len(item["colors"]), item["bbox"]))
    return candidates[0]


def _frame_candidate(grid, comp, palette_cells, palette_len):
    if any(cell in palette_cells for cell in comp["coords"]):
        return None
    color = int(comp["color"])
    if color in (BG, GREY):
        return None
    r0, c0, r1, c1 = comp["bbox"]
    h, w = _bbox_hw(comp["bbox"])
    if h < 5 or w < 5:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all((patch == color) | (patch == BG)):
        return None
    if not np.all(patch[0, :] == color):
        return None
    if not np.all(patch[-1, :] == color):
        return None
    if not np.all(patch[:, 0] == color):
        return None
    if not np.all(patch[:, -1] == color):
        return None
    inner = patch[1:-1, 1:-1]
    missing = np.argwhere(inner == BG)
    if missing.shape[0] != 1:
        return None
    if not np.all((inner == color) | (inner == BG)):
        return None
    inner_h, inner_w = inner.shape
    if inner_h != palette_len:
        return None
    mr, mc = map(int, missing[0])
    return {
        "bbox": comp["bbox"],
        "color": color,
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "missing_row": mr,
        "missing_col": mc,
    }


def _outline_candidate(grid, comp, palette_cells):
    if any(cell in palette_cells for cell in comp["coords"]):
        return None
    if int(comp["color"]) != GREY:
        return None
    r0, c0, r1, c1 = comp["bbox"]
    h, w = _bbox_hw(comp["bbox"])
    if h < 3 or w < 3:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all((patch == GREY) | (patch == BG)):
        return None
    if not np.all(patch[0, :] == GREY):
        return None
    if not np.all(patch[-1, :] == GREY):
        return None
    if not np.all(patch[:, 0] == GREY):
        return None
    if not np.all(patch[:, -1] == GREY):
        return None
    if not np.all(patch[1:-1, 1:-1] == BG):
        return None
    return {
        "bbox": comp["bbox"],
        "bbox_h": h,
        "bbox_w": w,
        "inner_h": h - 2,
        "inner_w": w - 2,
    }


def _select_frames_and_outlines(grid, palette):
    comps = _components_by_color(grid)
    palette_cells = palette["coords"]
    palette_len = len(palette["colors"])

    frame_candidates = []
    outline_candidates = []
    for comp in comps:
        frame = _frame_candidate(grid, comp, palette_cells, palette_len)
        if frame is not None:
            frame_candidates.append(frame)
        outline = _outline_candidate(grid, comp, palette_cells)
        if outline is not None:
            outline_candidates.append(outline)

    if not frame_candidates:
        raise ValueError("no valid source frames found")

    shape_counts = Counter((item["inner_h"], item["inner_w"]) for item in frame_candidates)
    shape = max(shape_counts.items(), key=lambda kv: (kv[1], kv[0][0] * kv[0][1], kv[0]))[0]
    frames = [item for item in frame_candidates if (item["inner_h"], item["inner_w"]) == shape]
    outlines = [
        item
        for item in outline_candidates
        if (item["bbox_h"], item["bbox_w"]) == shape or (item["inner_h"], item["inner_w"]) == shape
    ]
    if not outlines:
        outlines = outline_candidates
    if not outlines:
        raise ValueError("no valid grey outlines found")

    frames.sort(key=lambda item: item["bbox"])
    outlines.sort(key=lambda item: item["bbox"])
    return frames, outlines


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each missing interior pixel as a palette index, then repaint the nearest grey outline with that palette color.

    Input structure:
    - The background is black.
    - Several same-sized colored rectangles have a one-cell border and a same-colored interior with exactly one black missing cell.
    - A vertical one-pixel-wide palette sits in one corner, with distinct colors ordered from top to bottom.
    - Several grey hollow rectangular outlines elsewhere in the grid match the frame-interior shape.

    Rule:
    - For each colored rectangle, find the row index of its missing interior cell, counted from the top starting at 0.
    - Use that row index to select a color from the corner palette.
    - Find the spatially nearest matching grey outline and recolor both its border and its interior to the selected palette color.
    - Leave the original colored rectangles and the palette unchanged.

    Key insight:
    - The black gap inside each rectangle is not damage to repair; it is a row-position pointer into the palette, and the grey outline is the object that receives the chosen color.
    """
    g = np.asarray(grid, dtype=int)
    palette = _detect_palette(g)
    frames, outlines = _select_frames_and_outlines(g, palette)
    palette_colors = palette["colors"]

    out = g.copy()
    for frame in frames:
        candidates = [
            outline
            for outline in outlines
            if (outline["bbox_h"], outline["bbox_w"]) == (frame["inner_h"], frame["inner_w"])
            or (outline["inner_h"], outline["inner_w"]) == (frame["inner_h"], frame["inner_w"])
        ]
        if not candidates:
            candidates = outlines
        target = min(
            candidates,
            key=lambda item: (
                _bbox_gap_distance(frame["bbox"], item["bbox"]),
                _bbox_center_distance(frame["bbox"], item["bbox"]),
                item["bbox"],
            ),
        )
        color = int(palette_colors[frame["missing_row"]])
        r0, c0, r1, c1 = target["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _mark_blocked(mask, bbox, pad=1):
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _pair_layout(inner_h, inner_w, direction, gap):
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    outline_h = inner_h
    outline_w = inner_w

    if direction == "right":
        frame = (0, 0, frame_h - 1, frame_w - 1)
        outline = (1, frame_w + gap, inner_h, frame_w + gap + outline_w - 1)
    elif direction == "left":
        outline = (1, 0, inner_h, outline_w - 1)
        frame = (0, outline_w + gap, frame_h - 1, outline_w + gap + frame_w - 1)
    elif direction == "down":
        frame = (0, 0, frame_h - 1, frame_w - 1)
        outline = (frame_h + gap, 1, frame_h + gap + outline_h - 1, inner_w)
    elif direction == "up":
        outline = (0, 1, outline_h - 1, inner_w)
        frame = (outline_h + gap, 0, outline_h + gap + frame_h - 1, frame_w - 1)
    else:
        raise ValueError(f"unknown direction: {direction}")

    pair_h = max(frame[2], outline[2]) + 1
    pair_w = max(frame[3], outline[3]) + 1
    return frame, outline, pair_h, pair_w


def _translate_bbox(bbox, dr, dc):
    r0, c0, r1, c1 = bbox
    return (r0 + dr, c0 + dc, r1 + dr, c1 + dc)


def _draw_source_frame(grid, bbox, color, missing_row, missing_col):
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color
    grid[r0 + 1 + missing_row, c0 + 1 + missing_col] = BG


def _draw_outline(grid, bbox):
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = BG
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _draw_palette(grid, corner, colors):
    n = len(colors)
    if corner == "top_left":
        for r, color in enumerate(colors):
            grid[r, 0] = color
    elif corner == "top_right":
        for r, color in enumerate(colors):
            grid[r, grid.shape[1] - 1] = color
    elif corner == "bottom_left":
        start = grid.shape[0] - n
        for i, color in enumerate(colors):
            grid[start + i, 0] = color
    elif corner == "bottom_right":
        start = grid.shape[0] - n
        col = grid.shape[1] - 1
        for i, color in enumerate(colors):
            grid[start + i, col] = color
    else:
        raise ValueError(f"unknown corner: {corner}")


def _reserve_palette_zone(mask, corner, palette_len):
    h, w = mask.shape
    if corner == "top_left":
        bbox = (0, 0, palette_len - 1, 0)
    elif corner == "top_right":
        bbox = (0, w - 1, palette_len - 1, w - 1)
    elif corner == "bottom_left":
        bbox = (h - palette_len, 0, h - 1, 0)
    elif corner == "bottom_right":
        bbox = (h - palette_len, w - 1, h - 1, w - 1)
    else:
        raise ValueError(f"unknown corner: {corner}")
    _mark_blocked(mask, bbox, pad=2)


def _place_pairs(rng, height, width, inner_h, inner_w, count, corner):
    blocked = np.zeros((height, width), dtype=bool)
    _reserve_palette_zone(blocked, corner, inner_h)

    placed = []
    directions = ["right", "left", "down", "up"]
    rng.shuffle(directions)

    for index in range(count):
        success = False
        choices = directions[:]
        rng.shuffle(choices)
        for _ in range(PLACEMENT_ATTEMPTS):
            direction = choices[_ % len(choices)]
            gap = rng.randint(1, 4)
            rel_frame, rel_outline, pair_h, pair_w = _pair_layout(inner_h, inner_w, direction, gap)
            if pair_h > height - 2 or pair_w > width - 2:
                continue
            top = rng.randint(1, height - pair_h - 1)
            left = rng.randint(1, width - pair_w - 1)
            pair_bbox = (top, left, top + pair_h - 1, left + pair_w - 1)
            rr0, cc0, rr1, cc1 = pair_bbox
            if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            frame_bbox = _translate_bbox(rel_frame, top, left)
            outline_bbox = _translate_bbox(rel_outline, top, left)
            _mark_blocked(blocked, pair_bbox, pad=2)
            placed.append({"frame_bbox": frame_bbox, "outline_bbox": outline_bbox, "index": index})
            success = True
            break
        if not success:
            return None
    return placed


def _nearest_is_unique(pairs):
    frames = [item["frame_bbox"] for item in pairs]
    outlines = [item["outline_bbox"] for item in pairs]
    for idx, frame_bbox in enumerate(frames):
        scores = []
        for jdx, outline_bbox in enumerate(outlines):
            scores.append(
                (
                    _bbox_gap_distance(frame_bbox, outline_bbox),
                    _bbox_center_distance(frame_bbox, outline_bbox),
                    outline_bbox,
                    jdx,
                )
            )
        scores.sort()
        if scores[0][3] != idx:
            return False
        if len(scores) > 1 and scores[0][:3] == scores[1][:3]:
            return False
    return True


def _generate_attempt(rng):
    inner_h = rng.randint(3, 6)
    inner_w = rng.randint(3, 6)
    count = rng.randint(2, 4)
    corner = rng.choice(["top_left", "top_right", "bottom_left", "bottom_right"])

    min_side = min(30, max(14, 12 + count * 3, inner_h + 6, inner_w + 6))
    height = rng.randint(min_side, 30)
    width = rng.randint(min_side, 30)

    pairs = _place_pairs(rng, height, width, inner_h, inner_w, count, corner)
    if pairs is None:
        return None
    if not _nearest_is_unique(pairs):
        return None

    grid = np.zeros((height, width), dtype=int)
    palette_colors = rng.sample(COLOR_POOL, inner_h)
    _draw_palette(grid, corner, palette_colors)

    frame_colors = [rng.choice(COLOR_POOL) for _ in range(count)]
    for pair, frame_color in zip(pairs, frame_colors):
        missing_row = rng.randrange(inner_h)
        missing_col = rng.randrange(inner_w)
        _draw_source_frame(grid, pair["frame_bbox"], frame_color, missing_row, missing_col)
        _draw_outline(grid, pair["outline_bbox"])

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
