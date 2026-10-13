from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
DEPTH_COLORS = [1, 2, 3, 4]  # blue, red, green, yellow
DIRECTIONS = ("up", "right", "down", "left")
FRAME_SOURCE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GRID = 30
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _monochromatic_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == bg:
                continue
            color = int(grid[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            points: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                points.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "pts": points,
                    "set": set(points),
                    "bbox": _bbox(points),
                }
            )
    return comps


def _component_to_frame(comp: dict) -> dict | None:
    points = comp["set"]
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0 + 1, r1):
        border.add((r, c0))
        border.add((r, c1))

    if not points.issubset(border):
        return None

    missing = border - points
    if len(missing) != 1 or len(points) != len(border) - 1:
        return None

    gap_r, gap_c = next(iter(missing))
    direction = None
    if width % 2 == 1 and gap_r == r0 and gap_c == c0 + width // 2 and c0 < gap_c < c1:
        direction = "up"
    elif width % 2 == 1 and gap_r == r1 and gap_c == c0 + width // 2 and c0 < gap_c < c1:
        direction = "down"
    elif height % 2 == 1 and gap_c == c0 and gap_r == r0 + height // 2 and r0 < gap_r < r1:
        direction = "left"
    elif height % 2 == 1 and gap_c == c1 and gap_r == r0 + height // 2 and r0 < gap_r < r1:
        direction = "right"
    if direction is None:
        return None

    return {
        "color": comp["color"],
        "pts": comp["pts"],
        "set": points,
        "bbox": comp["bbox"],
        "gap": (gap_r, gap_c),
        "direction": direction,
        "height": height,
        "width": width,
    }


def _contains_frame(outer: dict, inner: dict) -> bool:
    or0, oc0, or1, oc1 = outer["bbox"]
    ir0, ic0, ir1, ic1 = inner["bbox"]
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _component_inside_frame(comp: dict, frame: dict) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    return all(r0 < r < r1 and c0 < c < c1 for r, c in comp["pts"])


def _eject_points(
    points: list[tuple[int, int]], direction: str, shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = _bbox(points)
    if direction == "up":
        dr, dc = -r0, 0
    elif direction == "down":
        dr, dc = h - 1 - r1, 0
    elif direction == "left":
        dr, dc = 0, -c0
    elif direction == "right":
        dr, dc = 0, w - 1 - c1
    else:
        raise AssertionError("unknown direction")
    return [(r + dr, c + dc) for r, c in points]


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _monochromatic_components(g, bg)

    frames: list[dict] = []
    others: list[dict] = []
    for comp in components:
        frame = _component_to_frame(comp)
        if frame is None:
            others.append(comp)
        else:
            frames.append(frame)

    assert frames, "expected at least one frame"

    for frame in frames:
        depth = 1 + sum(_contains_frame(other, frame) for other in frames if other is not frame)
        frame["depth"] = int(depth)
        frame["new_color"] = int(DEPTH_COLORS[(depth - 1) % len(DEPTH_COLORS)])

    candidates: list[tuple[int, int, dict, dict]] = []
    for comp in others:
        if int(comp["color"]) != GREY:
            continue
        containing = [frame for frame in frames if _component_inside_frame(comp, frame)]
        if containing:
            best_frame = max(containing, key=lambda frame: int(frame["depth"]))
            candidates.append((int(best_frame["depth"]), -len(comp["pts"]), best_frame, comp))

    if not candidates:
        deepest = max(frames, key=lambda frame: int(frame["depth"]))
        for comp in others:
            if _component_inside_frame(comp, deepest):
                candidates.append((int(deepest["depth"]), -len(comp["pts"]), deepest, comp))

    assert candidates, "expected one movable inner object"
    _, _, mover_frame, mover_comp = max(candidates, key=lambda item: (item[0], item[1]))

    return {
        "bg": bg,
        "frames": frames,
        "mover_frame": mover_frame,
        "mover_comp": mover_comp,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested gap-frames by depth, then eject the innermost grey object
    toward the shared opening side and repaint it with the innermost frame color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    scene = _extract_scene(g)

    for frame in scene["frames"]:
        for r, c in frame["pts"]:
            out[r, c] = int(frame["new_color"])

    bg = int(scene["bg"])
    mover_color = int(scene["mover_frame"]["new_color"])
    source_points = list(scene["mover_comp"]["pts"])
    for r, c in source_points:
        out[r, c] = bg

    for r, c in _eject_points(source_points, scene["mover_frame"]["direction"], g.shape):
        out[r, c] = mover_color

    return out


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, direction: str) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if direction == "up":
        grid[r0, (c0 + c1) // 2] = BG
    elif direction == "down":
        grid[r1, (c0 + c1) // 2] = BG
    elif direction == "left":
        grid[(r0 + r1) // 2, c0] = BG
    elif direction == "right":
        grid[(r0 + r1) // 2, c1] = BG
    else:
        raise AssertionError("unknown direction")


def _odd_candidates(lo: int, hi: int) -> list[int]:
    return [v for v in range(lo, hi + 1) if v % 2 == 1]


def _sample_margins(outer_h: int, outer_w: int, direction: str, obj_len: int) -> tuple[int, int, int, int] | None:
    min_top = obj_len if direction == "up" else 1
    min_bottom = obj_len if direction == "down" else 1
    min_left = obj_len if direction == "left" else 1
    min_right = obj_len if direction == "right" else 1

    if outer_h + min_top + min_bottom > MAX_GRID or outer_w + min_left + min_right > MAX_GRID:
        return None

    extra_h = random.randint(0, min(8, MAX_GRID - outer_h - min_top - min_bottom))
    extra_w = random.randint(0, min(8, MAX_GRID - outer_w - min_left - min_right))
    extra_top = random.randint(0, extra_h)
    extra_left = random.randint(0, extra_w)
    top = min_top + extra_top
    bottom = min_bottom + (extra_h - extra_top)
    left = min_left + extra_left
    right = min_right + (extra_w - extra_left)
    return top, bottom, left, right


def _generate_attempt() -> np.ndarray | None:
    direction = random.choice(DIRECTIONS)
    layers = random.randint(1, 5)
    obj_len = random.randint(1, 3)
    insets = [random.randint(2, 3) for _ in range(max(0, layers - 1))]
    total_inset = sum(insets)

    max_inner_h = min(11, MAX_GRID - 2 - 2 * total_inset)
    max_inner_w = min(11, MAX_GRID - 2 - 2 * total_inset)
    inner_h_choices = _odd_candidates(5, max_inner_h)
    inner_w_choices = _odd_candidates(5, max_inner_w)
    if not inner_h_choices or not inner_w_choices:
        return None

    inner_h = random.choice(inner_h_choices)
    inner_w = random.choice(inner_w_choices)
    if inner_h - 2 < (obj_len if direction in ("up", "down") else 1):
        return None
    if inner_w - 2 < (obj_len if direction in ("left", "right") else 1):
        return None

    outer_h = inner_h + 2 * total_inset
    outer_w = inner_w + 2 * total_inset
    margins = _sample_margins(outer_h, outer_w, direction, obj_len)
    if margins is None:
        return None
    top, bottom, left, right = margins

    grid_h = top + outer_h + bottom
    grid_w = left + outer_w + right
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    bboxes: list[tuple[int, int, int, int]] = []
    r0, c0 = top, left
    r1, c1 = r0 + outer_h - 1, c0 + outer_w - 1
    bboxes.append((r0, c0, r1, c1))
    for inset in insets:
        r0 += inset
        c0 += inset
        r1 -= inset
        c1 -= inset
        bboxes.append((r0, c0, r1, c1))

    colors = random.sample(FRAME_SOURCE_COLORS, layers)
    for bbox, color in zip(bboxes, colors):
        _paint_frame(grid, bbox, color, direction)

    inner = bboxes[-1]
    ir0, ic0, ir1, ic1 = inner
    object_points: list[tuple[int, int]]
    if direction in ("up", "down"):
        col = (ic0 + ic1) // 2
        start_min = ir0 + 1
        start_max = ir1 - obj_len
        if start_max < start_min:
            return None
        start = random.randint(start_min, start_max)
        object_points = [(start + k, col) for k in range(obj_len)]
    else:
        row = (ir0 + ir1) // 2
        start_min = ic0 + 1
        start_max = ic1 - obj_len
        if start_max < start_min:
            return None
        start = random.randint(start_min, start_max)
        object_points = [(row, start + k) for k in range(obj_len)]

    for r, c in object_points:
        grid[r, c] = GREY

    frame_pixels = set()
    for bbox in bboxes:
        br0, bc0, br1, bc1 = bbox
        for c in range(bc0, bc1 + 1):
            frame_pixels.add((br0, c))
            frame_pixels.add((br1, c))
        for r in range(br0, br1 + 1):
            frame_pixels.add((r, bc0))
            frame_pixels.add((r, bc1))
    if any(point in frame_pixels for point in _eject_points(object_points, direction, grid.shape)):
        return None

    try:
        scene = _extract_scene(grid)
    except AssertionError:
        return None

    depths = sorted(int(frame["depth"]) for frame in scene["frames"])
    if depths != list(range(1, layers + 1)):
        return None
    if len(scene["frames"]) != layers:
        return None
    if any(frame["direction"] != direction for frame in scene["frames"]):
        return None
    if int(scene["mover_frame"]["depth"]) != layers:
        return None
    if len(scene["mover_comp"]["pts"]) != obj_len:
        return None

    expected_dest = set(_eject_points(scene["mover_comp"]["pts"], direction, grid.shape))
    if any(r < 0 or c < 0 or r >= grid_h or c >= grid_w for r, c in expected_dest):
        return None

    out = main(grid)
    if np.any(out == GREY):
        return None
    mover_color = int(scene["mover_frame"]["new_color"])
    if any(int(out[r, c]) != mover_color for r, c in expected_dest):
        return None
    if any(int(out[r, c]) != BG for r, c in object_points):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
