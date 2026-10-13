import random
from collections import Counter

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(10))


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def bbox_span(box: tuple[int, int, int, int], axis: int) -> int:
    if axis == 0:
        return int(box[2] - box[0] + 1)
    return int(box[3] - box[1] + 1)


def center2(box: tuple[int, int, int, int], axis: int) -> int:
    if axis == 0:
        return int(box[0] + box[2])
    return int(box[1] + box[3])


def axis_low(box: tuple[int, int, int, int], axis: int) -> int:
    return int(box[0] if axis == 0 else box[1])


def axis_high(box: tuple[int, int, int, int], axis: int) -> int:
    return int(box[2] if axis == 0 else box[3])


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append({"color": color, "coords": coords, "bbox": bbox_from_coords(coords), "area": len(coords)})
    return out


def is_solid_rectangle(obj: dict) -> bool:
    box = obj["bbox"]
    return int(obj["area"]) == bbox_span(box, 0) * bbox_span(box, 1)


def rect_slice(box: tuple[int, int, int, int]) -> tuple[slice, slice]:
    r0, c0, r1, c1 = box
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
    bg = get_bg(g)
    objects = find_components(g, bg)
    assert len(objects) >= 4, "expected pusher, mover, and multiple walls"
    assert all(is_solid_rectangle(obj) for obj in objects), "all objects must be solid rectangles"

    candidates = []
    for axis in (0, 1):
        perp = 1 - axis
        center_keys = {center2(obj["bbox"], perp) for obj in objects}
        if len(center_keys) != 1:
            continue
        ordered = sorted(objects, key=lambda obj: axis_low(obj["bbox"], axis))
        for sign, seq in ((1, ordered), (-1, list(reversed(ordered)))):
            pusher = seq[0]
            mover = seq[1]
            walls = seq[2:]
            if len(walls) < 2:
                continue
            if bbox_span(mover["bbox"], axis) < bbox_span(mover["bbox"], perp):
                continue
            if bbox_span(mover["bbox"], perp) % 2 != 1:
                continue
            if bbox_span(mover["bbox"], perp) <= 2 * len(walls):
                continue
            if sign == 1:
                gaps = [axis_low(seq[i + 1]["bbox"], axis) - axis_high(seq[i]["bbox"], axis) for i in range(len(seq) - 1)]
            else:
                gaps = [axis_low(seq[i]["bbox"], axis) - axis_high(seq[i + 1]["bbox"], axis) for i in range(len(seq) - 1)]
            if any(gap <= 0 for gap in gaps):
                continue
            if any(bbox_span(wall["bbox"], perp) <= bbox_span(wall["bbox"], axis) for wall in walls):
                continue

            cur_lo = axis_low(mover["bbox"], perp)
            cur_hi = axis_high(mover["bbox"], perp)
            ok = True
            for wall in walls:
                if not (axis_low(wall["bbox"], perp) < cur_lo and cur_hi < axis_high(wall["bbox"], perp)):
                    ok = False
                    break
                cur_lo += 1
                cur_hi -= 1
            if not ok:
                continue

            final_box = list(mover["bbox"])
            if axis == 1:
                width = bbox_span(mover["bbox"], 1)
                if sign == 1:
                    final_box = [cur_lo, g.shape[1] - width, cur_hi, g.shape[1] - 1]
                else:
                    final_box = [cur_lo, 0, cur_hi, width - 1]
            else:
                height = bbox_span(mover["bbox"], 0)
                if sign == 1:
                    final_box = [g.shape[0] - height, cur_lo, g.shape[0] - 1, cur_hi]
                else:
                    final_box = [0, cur_lo, height - 1, cur_hi]

            fr0, fc0, fr1, fc1 = map(int, final_box)
            if not (0 <= fr0 <= fr1 < g.shape[0] and 0 <= fc0 <= fc1 < g.shape[1]):
                continue

            candidates.append(
                {
                    "bg": bg,
                    "axis": axis,
                    "direction": sign,
                    "pusher": pusher,
                    "mover": mover,
                    "walls": walls,
                    "final_perp_lo": cur_lo,
                    "final_perp_hi": cur_hi,
                    "final_box": tuple(final_box),
                }
            )

    assert len(candidates) == 1, "scene must admit a unique movement interpretation"
    return candidates[0]


def _simulate(scene: dict, grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = int(scene["bg"])
    mover = scene["mover"]
    axis = int(scene["axis"])
    color = int(mover["color"])

    out[rect_slice(mover["bbox"])] = bg

    cur_lo = axis_low(mover["bbox"], 1 - axis)
    cur_hi = axis_high(mover["bbox"], 1 - axis)
    for wall in scene["walls"]:
        r0, c0, r1, c1 = map(int, wall["bbox"])
        if axis == 1:
            out[cur_lo : cur_hi + 1, c0 : c1 + 1] = bg
        else:
            out[r0 : r1 + 1, cur_lo : cur_hi + 1] = bg
        cur_lo += 1
        cur_hi -= 1

    fr0, fc0, fr1, fc1 = map(int, scene["final_box"])
    assert np.all(out[fr0 : fr1 + 1, fc0 : fc1 + 1] == bg), "final resting place must be clear"
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push the solid moving rectangle through each perpendicular wall, shrinking it by one layer per wall before it parks at the edge.

    Input structure:
    - One dominant background contains several separated monochrome solid rectangles.
    - All foreground rectangles share one movement line: a pusher block at one end, a solid moving rectangle next to it, and multiple perpendicular wall bars farther along the line.
    - The moving rectangle has odd thickness across the movement axis, and each wall fully spans that thickness with visible material left above and below it.

    Rule:
    - Move the solid rectangle straight away from the pusher until its front edge is flush with the grid border.
    - Whenever it reaches a wall, erase the wall cells covered by the rectangle's current cross-section to form a gap.
    - After each wall crossing, remove one outer layer from both sides of the rectangle across the movement axis, so later gaps are narrower by two cells each time.
    - Keep the pusher unchanged, preserve the untouched wall material, erase the rectangle's start position, and draw only its final eroded position at the edge.

    Key insight:
    - The rectangle carries state while moving: every earlier wall determines the thickness used for every later wall, so the final object and the chain of wall gaps must be computed in order.
    """
    scene = _analyze_scene(np.asarray(grid, dtype=int))
    return _simulate(scene, np.asarray(grid, dtype=int))


def _make_base_scene() -> tuple[np.ndarray, dict]:
    bg = 0
    k = random.choice((2, 3))
    if k == 2:
        thickness = random.choice((5, 7, 9))
        mover_len = random.randint(thickness, min(thickness + 3, 10))
        pusher_w = random.randint(1, 3)
        first_gap = random.randint(1, 3)
        wall_gap = [random.randint(1, 3) for _ in range(k)]
        tail_gap = random.randint(1, 3)
        prefix = random.randint(1, 3)
        wall_extras = [random.randint(1, 3) for _ in range(k)]
        top_margin = random.randint(1, 3)
        bottom_margin = random.randint(1, 3)
    else:
        thickness = random.choice((7, 9))
        mover_len = random.randint(thickness, min(thickness + 1, 9))
        pusher_w = random.randint(1, 2)
        first_gap = random.randint(1, 2)
        wall_gap = [random.randint(1, 2) for _ in range(k)]
        tail_gap = random.randint(1, 2)
        prefix = random.randint(1, 2)
        wall_extras = [random.randint(1, 2) for _ in range(k)]
        top_margin = random.randint(1, 2)
        bottom_margin = random.randint(1, 2)
    pusher_h = random.choice((1, 3))
    wall_widths = [random.randint(1, 2) for _ in range(k)]
    wall_heights = [thickness + 2 * extra for extra in wall_extras]
    max_wall_h = max(wall_heights)
    h = top_margin + max_wall_h + bottom_margin
    w = (
        prefix
        + pusher_w
        + first_gap
        + mover_len
        + wall_gap[0]
        + sum(wall_widths)
        + sum(wall_gap[1:])
        + tail_gap
        + mover_len
    )
    if h > 30 or w > 30:
        raise ValueError("scene too large")

    grid = np.full((h, w), bg, dtype=int)
    colors = random.sample([c for c in PALETTE if c != bg], 2 + k)
    pusher_color = colors[0]
    mover_color = colors[1]
    wall_colors = colors[2:]

    center_r = top_margin + max_wall_h // 2
    pusher_r0 = center_r - pusher_h // 2
    mover_r0 = center_r - thickness // 2

    pusher_c0 = prefix
    mover_c0 = pusher_c0 + pusher_w + first_gap
    c = mover_c0 + mover_len + wall_gap[0]

    grid[pusher_r0 : pusher_r0 + pusher_h, pusher_c0 : pusher_c0 + pusher_w] = pusher_color
    grid[mover_r0 : mover_r0 + thickness, mover_c0 : mover_c0 + mover_len] = mover_color

    wall_boxes = []
    for idx in range(k):
        wall_h = wall_heights[idx]
        wall_r0 = center_r - wall_h // 2
        wall_c0 = c
        grid[wall_r0 : wall_r0 + wall_h, wall_c0 : wall_c0 + wall_widths[idx]] = wall_colors[idx]
        wall_boxes.append((wall_r0, wall_c0, wall_r0 + wall_h - 1, wall_c0 + wall_widths[idx] - 1))
        c += wall_widths[idx]
        if idx + 1 < k:
            c += wall_gap[idx + 1]

    spec = {
        "k": k,
        "thickness": thickness,
        "final_thickness": thickness - 2 * k,
        "mover_len": mover_len,
        "pusher_box": (pusher_r0, pusher_c0, pusher_r0 + pusher_h - 1, pusher_c0 + pusher_w - 1),
        "mover_box": (mover_r0, mover_c0, mover_r0 + thickness - 1, mover_c0 + mover_len - 1),
        "wall_boxes": wall_boxes,
    }
    return grid, spec


def _orient(grid: np.ndarray) -> np.ndarray:
    mode = random.choice(("right", "left", "down", "up"))
    if mode == "right":
        return grid
    if mode == "left":
        return np.fliplr(grid)
    if mode == "down":
        return grid.T
    return np.flipud(grid.T)


def _wall_gap_widths(inp: np.ndarray, out: np.ndarray, scene: dict) -> list[int]:
    axis = scene["axis"]
    widths = []
    for wall in scene["walls"]:
        r0, c0, r1, c1 = map(int, wall["bbox"])
        color = int(wall["color"])
        if axis == 1:
            cut_rows = [r for r in range(r0, r1 + 1) if np.all(out[r, c0 : c1 + 1] != color)]
            widths.append(len(cut_rows))
        else:
            cut_cols = [c for c in range(c0, c1 + 1) if np.all(out[r0 : r1 + 1, c] != color)]
            widths.append(len(cut_cols))
    return widths


def _semantic_check(inp: np.ndarray, spec: dict) -> bool:
    scene = _analyze_scene(inp)
    out = main(inp)

    widths = _wall_gap_widths(inp, out, scene)
    expected = [bbox_span(scene["mover"]["bbox"], 1 - scene["axis"]) - 2 * i for i in range(len(scene["walls"]))]
    if widths != expected:
        return False
    if any(widths[i] - widths[i + 1] != 2 for i in range(len(widths) - 1)):
        return False

    final_box = scene["final_box"]
    if bbox_span(final_box, 1 - scene["axis"]) != spec["final_thickness"]:
        return False

    mover_color = int(scene["mover"]["color"])
    fr0, fc0, fr1, fc1 = final_box
    if not np.all(out[fr0 : fr1 + 1, fc0 : fc1 + 1] == mover_color):
        return False

    sr0, sc0, sr1, sc1 = scene["mover"]["bbox"]
    if np.any(out[sr0 : sr1 + 1, sc0 : sc1 + 1] == mover_color):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            base, spec = _make_base_scene()
        except ValueError:
            continue
        grid = _orient(base)
        try:
            if _semantic_check(grid, spec):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid worn-through-walls input")
