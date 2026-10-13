from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GAP_TO_ROT_K = {
    "top": 0,
    "right": 3,   # clockwise 90
    "bottom": 2,
    "left": 1,    # clockwise 270
}
SIDE_ORDER = ("top", "right", "bottom", "left")

MAX_GENERATION_ATTEMPTS = 500
LAYER_RANGE = (2, 4)
OUTER_MARGIN_RANGE = (1, 4)
INNER_EXTRA_MARGIN_RANGE = (1, 4)
PATTERN_SIDE_RANGE = (2, 4)
PATTERN_PIXELS_RANGE = (3, 8)
MAX_SHAPE_ATTEMPTS = 200
MAX_TOTAL_SIDE = 30


def _components_of_color(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _crop_non_bg(arr: np.ndarray, bg: int = BG) -> np.ndarray:
    pts = np.argwhere(arr != bg)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _frame_from_component(coords: np.ndarray, color: int) -> dict[str, object] | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    local = np.zeros((h, w), dtype=bool)
    local[rows - r0, cols - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(local & ~border):
        return None

    missing = np.argwhere(border & ~local)
    if len(missing) != 1:
        return None

    mr, mc = map(int, missing[0])
    if mr == 0 and 0 < mc < w - 1:
        gap = "top"
    elif mr == h - 1 and 0 < mc < w - 1:
        gap = "bottom"
    elif mc == 0 and 0 < mr < h - 1:
        gap = "left"
    elif mc == w - 1 and 0 < mr < h - 1:
        gap = "right"
    else:
        return None

    if int(local.sum()) != int(border.sum()) - 1:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "color": int(color),
    }


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _detect_frames(grid: np.ndarray, bg: int = BG) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components_of_color(grid == color):
            frame = _frame_from_component(comp, color)
            if frame is not None:
                frames.append(frame)

    assert len(frames) >= 2
    frames.sort(key=lambda item: _bbox_area(item["bbox"]))
    for idx in range(1, len(frames)):
        assert _contains(frames[idx]["bbox"], frames[idx - 1]["bbox"])
    return frames


def _annulus_rects(
    inner_bbox: tuple[int, int, int, int],
    outer_bbox: tuple[int, int, int, int],
) -> dict[str, tuple[int, int, int, int]]:
    ir0, ic0, ir1, ic1 = inner_bbox
    or0, oc0, or1, oc1 = outer_bbox
    rects = {
        "top": (or0 + 1, oc0 + 1, ir0 - or0 - 1, oc1 - oc0 - 1),
        "bottom": (ir1 + 1, oc0 + 1, or1 - ir1 - 1, oc1 - oc0 - 1),
        "left": (ir0, oc0 + 1, ir1 - ir0 + 1, ic0 - oc0 - 1),
        "right": (ir0, ic1 + 1, ir1 - ir0 + 1, oc1 - ic1 - 1),
    }
    return rects


def _choose_rect_for_pattern(
    rects: dict[str, tuple[int, int, int, int]],
    patch_shape: tuple[int, int],
    preferred_side: str,
) -> tuple[str, tuple[int, int, int, int]]:
    ph, pw = patch_shape
    candidates: list[str] = []
    for side in SIDE_ORDER:
        _, _, rh, rw = rects[side]
        if rh >= ph and rw >= pw:
            candidates.append(side)

    assert candidates
    if preferred_side in candidates:
        return preferred_side, rects[preferred_side]
    return candidates[0], rects[candidates[0]]


def _paste_patch(
    out: np.ndarray,
    patch: np.ndarray,
    rect: tuple[int, int, int, int],
    bg: int = BG,
) -> None:
    top, left, height, width = rect
    ph, pw = patch.shape
    pr = top + (height - ph) // 2
    pc = left + (width - pw) // 2
    pts = np.argwhere(patch != bg)
    for dr, dc in pts:
        rr = pr + int(dr)
        cc = pc + int(dc)
        assert out[rr, cc] == bg
        out[rr, cc] = int(patch[dr, dc])


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames = _detect_frames(g, bg=BG)

    inner_bbox = frames[0]["bbox"]
    r0, c0, r1, c1 = inner_bbox
    inner_view = g[r0 + 1 : r1, c0 + 1 : c1]
    current = _crop_non_bg(inner_view, bg=BG)

    out = g.copy()
    for idx in range(1, len(frames)):
        rotated = np.rot90(current, GAP_TO_ROT_K[frames[idx]["gap"]])
        rects = _annulus_rects(frames[idx - 1]["bbox"], frames[idx]["bbox"])
        _, rect = _choose_rect_for_pattern(rects, rotated.shape, frames[idx]["gap"])
        _paste_patch(out, rotated, rect, bg=BG)
        current = rotated

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


def _grow_shape(height: int, width: int, cells: int) -> np.ndarray:
    start = (random.randrange(height), random.randrange(width))
    occupied = {start}
    frontier = {start}
    while len(occupied) < cells:
        origin = random.choice(tuple(frontier))
        neighbors = []
        for dr, dc in DIRS4:
            nr, nc = origin[0] + dr, origin[1] + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in occupied:
                neighbors.append((nr, nc))
        if neighbors:
            nxt = random.choice(neighbors)
            occupied.add(nxt)
            frontier.add(nxt)
        else:
            frontier.remove(origin)
            if not frontier:
                break
    arr = np.zeros((height, width), dtype=int)
    for r, c in occupied:
        arr[r, c] = 1
    return _crop_non_bg(arr, bg=0)


def _make_pattern(colors: list[int]) -> np.ndarray:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        h = random.randint(*PATTERN_SIDE_RANGE)
        w = random.randint(*PATTERN_SIDE_RANGE)
        cells = random.randint(PATTERN_PIXELS_RANGE[0], min(PATTERN_PIXELS_RANGE[1], h * w - 1))
        mask = _grow_shape(h, w, cells)
        mh, mw = mask.shape
        if mh < 2 or mw < 2:
            continue

        coords = [tuple(map(int, pt)) for pt in np.argwhere(mask == 1)]
        random.shuffle(coords)

        pattern = np.zeros_like(mask, dtype=int)
        use_colors = colors[: random.randint(2, min(3, len(colors)))]
        for idx, (r, c) in enumerate(coords):
            if idx < len(use_colors):
                pattern[r, c] = use_colors[idx]
            else:
                pattern[r, c] = random.choice(use_colors)

        if len(set(int(v) for v in pattern.flat if int(v) != 0)) < 2:
            continue
        if any(np.array_equal(pattern, np.rot90(pattern, k)) for k in (1, 2, 3)):
            continue
        return pattern
    raise RuntimeError("failed to build asymmetric pattern")


def _draw_gapped_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    gap_side: str,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if gap_side == "top":
        gc = random.randint(c0 + 1, c1 - 1)
        grid[r0, gc] = BG
    elif gap_side == "bottom":
        gc = random.randint(c0 + 1, c1 - 1)
        grid[r1, gc] = BG
    elif gap_side == "left":
        gr = random.randint(r0 + 1, r1 - 1)
        grid[gr, c0] = BG
    else:
        gr = random.randint(r0 + 1, r1 - 1)
        grid[gr, c1] = BG


def _expected_output(
    grid: np.ndarray,
    frames: list[dict[str, object]],
    seed_pattern: np.ndarray,
) -> np.ndarray:
    out = grid.copy()
    current = seed_pattern.copy()
    for idx in range(1, len(frames)):
        rotated = np.rot90(current, GAP_TO_ROT_K[frames[idx]["gap"]])
        rects = _annulus_rects(frames[idx - 1]["bbox"], frames[idx]["bbox"])
        _, rect = _choose_rect_for_pattern(rects, rotated.shape, frames[idx]["gap"])
        _paste_patch(out, rotated, rect, bg=BG)
        current = rotated
    return out


def _gen_attempt() -> np.ndarray | None:
    layer_count = random.randint(*LAYER_RANGE)
    color_count = layer_count + 3
    palette = random.sample(range(1, 10), color_count)
    frame_colors = palette[:layer_count]
    pattern = _make_pattern(palette[layer_count:])
    ph, pw = pattern.shape

    inner_h = ph + random.randint(*INNER_EXTRA_MARGIN_RANGE)
    inner_w = pw + random.randint(*INNER_EXTRA_MARGIN_RANGE)
    frames: list[dict[str, object]] = [{
        "bbox": (0, 0, inner_h + 1, inner_w + 1),
        "gap": random.choice(SIDE_ORDER),
        "color": frame_colors[0],
    }]

    current = pattern.copy()
    for idx in range(1, layer_count):
        pair = random.choice(("tb", "lr"))
        if pair == "tb":
            gap = random.choice(("top", "bottom"))
            rotated = np.rot90(current, GAP_TO_ROT_K[gap])
            rh, rw = rotated.shape
            pad_r = random.randint(rh + 1, rh + 3)
            pad_c = random.randint(2, rw)
        else:
            gap = random.choice(("left", "right"))
            rotated = np.rot90(current, GAP_TO_ROT_K[gap])
            rh, rw = rotated.shape
            pad_r = random.randint(2, rh)
            pad_c = random.randint(rw + 1, rw + 3)

        pr0, pc0, pr1, pc1 = frames[-1]["bbox"]
        frames.append({
            "bbox": (pr0 - pad_r, pc0 - pad_c, pr1 + pad_r, pc1 + pad_c),
            "gap": gap,
            "color": frame_colors[idx],
        })
        current = rotated

    outer_bbox = frames[-1]["bbox"]
    min_r = min(int(frame["bbox"][0]) for frame in frames)
    min_c = min(int(frame["bbox"][1]) for frame in frames)
    max_r = max(int(frame["bbox"][2]) for frame in frames)
    max_c = max(int(frame["bbox"][3]) for frame in frames)
    struct_h = max_r - min_r + 1
    struct_w = max_c - min_c + 1
    if struct_h > MAX_TOTAL_SIDE or struct_w > MAX_TOTAL_SIDE:
        return None

    margin_top = random.randint(*OUTER_MARGIN_RANGE)
    margin_bottom = random.randint(*OUTER_MARGIN_RANGE)
    margin_left = random.randint(*OUTER_MARGIN_RANGE)
    margin_right = random.randint(*OUTER_MARGIN_RANGE)
    grid = np.zeros((struct_h + margin_top + margin_bottom, struct_w + margin_left + margin_right), dtype=int)

    frame_abs: list[dict[str, object]] = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        abs_bbox = (
            int(r0 - min_r + margin_top),
            int(c0 - min_c + margin_left),
            int(r1 - min_r + margin_top),
            int(c1 - min_c + margin_left),
        )
        frame_abs.append({
            "bbox": abs_bbox,
            "gap": frame["gap"],
            "color": frame["color"],
        })

    for frame in reversed(frame_abs):
        _draw_gapped_frame(grid, frame["bbox"], int(frame["color"]), str(frame["gap"]))

    ir0, ic0, ir1, ic1 = frame_abs[0]["bbox"]
    inner_h = ir1 - ir0 - 1
    inner_w = ic1 - ic0 - 1
    max_pr = inner_h - ph
    max_pc = inner_w - pw
    if max_pr < 0 or max_pc < 0:
        return None

    place_r = random.randint(0, max_pr)
    place_c = random.randint(0, max_pc)
    for dr, dc in np.argwhere(pattern != BG):
        grid[ir0 + 1 + place_r + int(dr), ic0 + 1 + place_c + int(dc)] = int(pattern[dr, dc])

    expected = _expected_output(grid, frame_abs, pattern)
    try:
        actual = _solve(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
