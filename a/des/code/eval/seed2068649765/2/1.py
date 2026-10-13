from __future__ import annotations

import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CENTER_PRIORITY = ("top", "bottom", "left", "right")
GEN_RETRIES = 400


def _shape_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


RAW_SHAPES = [
    _shape_mask(["11"]),
    _shape_mask(["111"]),
    _shape_mask(["10", "11"]),
    _shape_mask(["11", "11"]),
    _shape_mask(["111", "010"]),
    _shape_mask(["110", "011"]),
    _shape_mask(["10", "10", "11"]),
    _shape_mask(["111", "110"]),
]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _touches_edge(coords: list[tuple[int, int]], height: int, width: int) -> bool:
    return any(r == 0 or c == 0 or r == height - 1 or c == width - 1 for r, c in coords)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _rot_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append((int(rot.shape[0]), int(rot.shape[1]), rot.astype(np.uint8).tobytes()))
    return min(variants)


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _mono_components(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    comps: list[dict] = []
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                    "touches_edge": _touches_edge(coords, height, width),
                }
            )
    return comps


def _internal_bg_components(
    grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int
) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1] == bg
    holes: list[list[tuple[int, int]]] = []
    for comp in _mask_components(sub):
        if any(rr == 0 or cc == 0 or rr == sub.shape[0] - 1 or cc == sub.shape[1] - 1 for rr, cc in comp):
            continue
        holes.append([(r0 + rr, c0 + cc) for rr, cc in comp])
    return holes


def _adjacent_colors(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> set[int]:
    height, width = grid.shape
    colors: set[int] = set()
    for r, c in coords:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                color = int(grid[nr, nc])
                if color != bg:
                    colors.add(color)
    return colors


def _longest_run(line: np.ndarray) -> int:
    best = 0
    cur = 0
    for value in line:
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _bar_spans(frame_mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    row_runs = np.array([_longest_run(frame_mask[r, :]) for r in range(frame_mask.shape[0])], dtype=int)
    col_runs = np.array([_longest_run(frame_mask[:, c]) for c in range(frame_mask.shape[1])], dtype=int)

    pos_rows = row_runs[row_runs > 0]
    pos_cols = col_runs[col_runs > 0]
    assert pos_rows.size and pos_cols.size

    h_rows = np.where(row_runs > np.median(pos_rows))[0]
    v_cols = np.where(col_runs > np.median(pos_cols))[0]

    if h_rows.size == 0:
        row_counts = frame_mask.sum(axis=1)
        h_rows = np.where(row_counts > np.median(row_counts[row_counts > 0]))[0]
    if v_cols.size == 0:
        col_counts = frame_mask.sum(axis=0)
        v_cols = np.where(col_counts > np.median(col_counts[col_counts > 0]))[0]

    assert h_rows.size and v_cols.size
    return (int(h_rows.min()), int(h_rows.max())), (int(v_cols.min()), int(v_cols.max()))


def _classify_hole(
    hole: list[tuple[int, int]], frame_bbox: tuple[int, int, int, int], h_span: tuple[int, int], v_span: tuple[int, int]
) -> str:
    fr0, fc0, _fr1, _fc1 = frame_bbox
    hr0, hr1 = h_span
    vc0, vc1 = v_span
    rows = [r - fr0 for r, _c in hole]
    cols = [c - fc0 for _r, c in hole]
    if max(rows) < hr0:
        return "top"
    if min(rows) > hr1:
        return "bottom"
    if max(cols) < vc0:
        return "left"
    if min(cols) > vc1:
        return "right"

    center_r = 0.5 * (hr0 + hr1)
    center_c = 0.5 * (vc0 + vc1)
    hole_r = float(sum(rows)) / len(rows)
    hole_c = float(sum(cols)) / len(cols)
    if abs(hole_r - center_r) >= abs(hole_c - center_c):
        return "top" if hole_r < center_r else "bottom"
    return "left" if hole_c < center_c else "right"


def _arm_masks_from_bbox(
    frame_mask: np.ndarray,
    h_span: tuple[int, int],
    v_span: tuple[int, int],
) -> dict[str, np.ndarray]:
    hr0, hr1 = h_span
    vc0, vc1 = v_span
    masks: dict[str, np.ndarray] = {}

    top = np.zeros_like(frame_mask, dtype=bool)
    top[:hr0, vc0 : vc1 + 1] = True
    masks["top"] = top & frame_mask

    bottom = np.zeros_like(frame_mask, dtype=bool)
    bottom[hr1 + 1 :, vc0 : vc1 + 1] = True
    masks["bottom"] = bottom & frame_mask

    left = np.zeros_like(frame_mask, dtype=bool)
    left[hr0 : hr1 + 1, :vc0] = True
    masks["left"] = left & frame_mask

    right = np.zeros_like(frame_mask, dtype=bool)
    right[hr0 : hr1 + 1, vc1 + 1 :] = True
    masks["right"] = right & frame_mask

    center = np.zeros_like(frame_mask, dtype=bool)
    center[hr0 : hr1 + 1, vc0 : vc1 + 1] = True
    masks["center"] = center & frame_mask
    return masks


def _extract_segment_shape(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    holes = []
    for hole in _internal_bg_components(grid, comp["bbox"], bg):
        if _adjacent_colors(grid, hole, bg) == {int(comp["color"])}:
            holes.append(hole)
    assert len(holes) == 1
    return _tight_mask(holes[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each arm-hole shape to the equally shaped cutout in the edge strip, then recolor the arm.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _mono_components(g, bg)

    frame_candidates = [comp for comp in comps if not comp["touches_edge"]]
    assert frame_candidates
    frame = max(frame_candidates, key=lambda comp: int(comp["area"]))
    frame_color = int(frame["color"])
    fr0, fc0, fr1, fc1 = frame["bbox"]

    frame_mask = g[fr0 : fr1 + 1, fc0 : fc1 + 1] == frame_color
    h_span, v_span = _bar_spans(frame_mask)
    local_arm_masks = _arm_masks_from_bbox(frame_mask, h_span, v_span)

    hole_map: dict[str, np.ndarray] = {}
    for hole in _internal_bg_components(g, frame["bbox"], bg):
        if _adjacent_colors(g, hole, bg) != {frame_color}:
            continue
        direction = _classify_hole(hole, frame["bbox"], h_span, v_span)
        hole_map[direction] = _tight_mask(hole)
    assert hole_map

    shape_to_color: dict[tuple[int, int, bytes], int] = {}
    for comp in comps:
        if comp is frame or not comp["touches_edge"]:
            continue
        segment_shape = _extract_segment_shape(g, comp, bg)
        key = _rot_key(segment_shape)
        assert key not in shape_to_color
        shape_to_color[key] = int(comp["color"])

    out = g.copy()
    local = out[fr0 : fr1 + 1, fc0 : fc1 + 1]
    direction_colors: dict[str, int] = {}
    for direction, mask in hole_map.items():
        key = _rot_key(mask)
        assert key in shape_to_color
        color = int(shape_to_color[key])
        direction_colors[direction] = color
        local[local_arm_masks[direction]] = color

    for direction in CENTER_PRIORITY:
        if direction in direction_colors:
            local[local_arm_masks["center"]] = direction_colors[direction]
            break
    return out


def _unique_shapes() -> list[np.ndarray]:
    seen: set[tuple[int, int, bytes]] = set()
    unique: list[np.ndarray] = []
    for mask in RAW_SHAPES:
        key = _rot_key(mask)
        if key not in seen:
            seen.add(key)
            unique.append(mask)
    return unique


SHAPES = _unique_shapes()


def _choose_present_dirs() -> list[str]:
    dirs = ["top", "bottom", "left", "right"]
    if random.random() < 0.5:
        return dirs
    missing = random.choice(dirs)
    return [direction for direction in dirs if direction != missing]


def _centered_offset(total: int, size: int) -> int:
    slack = total - size
    assert slack >= 0
    base = slack // 2
    jitter = min(2, slack)
    return max(0, min(slack, base + random.randint(-jitter, jitter)))


def _build_frame(
    arm_shapes: dict[str, np.ndarray], bar_w: int, lengths: dict[str, int]
) -> tuple[np.ndarray, dict[str, tuple[int, int]], tuple[int, int]]:
    top_len = lengths.get("top", 0)
    bottom_len = lengths.get("bottom", 0)
    left_len = lengths.get("left", 0)
    right_len = lengths.get("right", 0)

    height = top_len + bar_w + bottom_len
    width = left_len + bar_w + right_len
    frame = np.zeros((height, width), dtype=bool)

    if top_len:
        frame[:top_len, left_len : left_len + bar_w] = True
    if bottom_len:
        frame[top_len + bar_w :, left_len : left_len + bar_w] = True
    if left_len:
        frame[top_len : top_len + bar_w, :left_len] = True
    if right_len:
        frame[top_len : top_len + bar_w, left_len + bar_w :] = True
    frame[top_len : top_len + bar_w, left_len : left_len + bar_w] = True

    hole_positions: dict[str, tuple[int, int]] = {}
    for direction, hole in arm_shapes.items():
        hh, ww = hole.shape
        if direction == "top":
            r0 = random.randint(1, top_len - hh - 1)
            c0 = left_len + random.randint(1, bar_w - ww - 1)
        elif direction == "bottom":
            r0 = top_len + bar_w + random.randint(1, bottom_len - hh - 1)
            c0 = left_len + random.randint(1, bar_w - ww - 1)
        elif direction == "left":
            r0 = top_len + random.randint(1, bar_w - hh - 1)
            c0 = random.randint(1, left_len - ww - 1)
        else:
            r0 = top_len + random.randint(1, bar_w - hh - 1)
            c0 = left_len + bar_w + random.randint(1, right_len - ww - 1)
        frame[r0 : r0 + hh, c0 : c0 + ww][hole] = False
        hole_positions[direction] = (r0, c0)

    return frame, hole_positions, (top_len, left_len)


def _build_strip(side: str, direction_colors: dict[str, int], base_shapes: dict[str, np.ndarray]) -> np.ndarray:
    order = list(direction_colors)
    random.shuffle(order)
    strip_holes = {direction: np.rot90(base_shapes[direction], random.randint(0, 3)) for direction in order}

    if side in ("top", "bottom"):
        strip_h = max(int(hole.shape[0]) for hole in strip_holes.values()) + 2 + random.randint(0, 1)
        patches = []
        for direction in order:
            hole = strip_holes[direction]
            hh, ww = hole.shape
            seg_w = ww + 2 + random.randint(0, 2)
            patch = np.full((strip_h, seg_w), int(direction_colors[direction]), dtype=int)
            r0 = random.randint(1, strip_h - hh - 1)
            c0 = random.randint(1, seg_w - ww - 1)
            patch[r0 : r0 + hh, c0 : c0 + ww][hole] = 0
            patches.append(patch)
        return np.concatenate(patches, axis=1)

    strip_w = max(int(hole.shape[1]) for hole in strip_holes.values()) + 2 + random.randint(0, 1)
    patches = []
    for direction in order:
        hole = strip_holes[direction]
        hh, ww = hole.shape
        seg_h = hh + 2 + random.randint(0, 2)
        patch = np.full((seg_h, strip_w), int(direction_colors[direction]), dtype=int)
        r0 = random.randint(1, seg_h - hh - 1)
        c0 = random.randint(1, strip_w - ww - 1)
        patch[r0 : r0 + hh, c0 : c0 + ww][hole] = 0
        patches.append(patch)
    return np.concatenate(patches, axis=0)


def _render_expected(
    grid: np.ndarray,
    frame_top: int,
    frame_left: int,
    frame_mask: np.ndarray,
    bar_w: int,
    lengths: dict[str, int],
    direction_colors: dict[str, int],
) -> np.ndarray:
    top_len = lengths.get("top", 0)
    bottom_len = lengths.get("bottom", 0)
    left_len = lengths.get("left", 0)
    right_len = lengths.get("right", 0)

    h_span = (top_len, top_len + bar_w - 1)
    v_span = (left_len, left_len + bar_w - 1)
    arm_masks = _arm_masks_from_bbox(frame_mask, h_span, v_span)

    out = grid.copy()
    local = out[frame_top : frame_top + frame_mask.shape[0], frame_left : frame_left + frame_mask.shape[1]]
    for direction, color in direction_colors.items():
        local[arm_masks[direction]] = int(color)
    for direction in CENTER_PRIORITY:
        if direction in direction_colors:
            local[arm_masks["center"]] = int(direction_colors[direction])
            break
    return out


def _gen_attempt() -> np.ndarray | None:
    present_dirs = _choose_present_dirs()
    base_shapes = {direction: shape.copy() for direction, shape in zip(present_dirs, random.sample(SHAPES, len(present_dirs)), strict=False)}
    arm_shapes = {direction: np.rot90(base_shapes[direction], random.randint(0, 3)) for direction in present_dirs}

    bar_w = random.choice([5, 6])
    lengths: dict[str, int] = {}
    for direction in ("top", "bottom", "left", "right"):
        if direction not in present_dirs:
            lengths[direction] = 0
            continue
        hole = arm_shapes[direction]
        need = int(hole.shape[0] + 2) if direction in ("top", "bottom") else int(hole.shape[1] + 2)
        lengths[direction] = random.randint(need, need + 4)

    frame_mask, _hole_positions, (top_len, left_len) = _build_frame(arm_shapes, bar_w, lengths)
    frame_h, frame_w = frame_mask.shape

    frame_color = random.choice([color for color in range(1, 10)])
    palette_colors = random.sample([color for color in range(1, 10) if color != frame_color], len(present_dirs))
    direction_colors = {direction: color for direction, color in zip(present_dirs, palette_colors, strict=False)}

    side = random.choice(["top", "bottom", "left", "right"])
    strip = _build_strip(side, direction_colors, base_shapes)
    strip_h, strip_w = strip.shape
    gap = random.randint(2, 4)

    if side in ("top", "bottom"):
        width = max(frame_w + random.randint(4, 8), strip_w + random.randint(0, 4))
        height = frame_h + strip_h + gap + random.randint(3, 7)
        frame_left = _centered_offset(width, frame_w)
        strip_left = random.randint(0, width - strip_w)
        if side == "top":
            strip_top = 0
            frame_top = strip_h + gap + random.randint(1, 3)
        else:
            frame_top = random.randint(1, 3)
            strip_top = height - strip_h
    else:
        width = frame_w + strip_w + gap + random.randint(3, 7)
        height = max(frame_h + random.randint(4, 8), strip_h + random.randint(0, 4))
        frame_top = _centered_offset(height, frame_h)
        strip_top = random.randint(0, height - strip_h)
        if side == "left":
            strip_left = 0
            frame_left = strip_w + gap + random.randint(1, 3)
        else:
            frame_left = random.randint(1, 3)
            strip_left = width - strip_w

    grid = np.zeros((height, width), dtype=int)
    local = grid[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w]
    local[frame_mask] = frame_color
    grid[strip_top : strip_top + strip_h, strip_left : strip_left + strip_w] = strip

    if _get_bg(grid) != 0:
        return None

    expected = _render_expected(grid, frame_top, frame_left, frame_mask, bar_w, lengths, direction_colors)
    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
