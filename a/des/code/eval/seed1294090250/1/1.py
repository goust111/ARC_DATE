import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "right", "bottom", "left")
DIR_BY_SIDE = {
    "top": (1, 0),
    "right": (0, -1),
    "bottom": (-1, 0),
    "left": (0, 1),
}
GEN_ATTEMPTS = 400
SPRITE_PLACEMENT_ATTEMPTS = 400

VERTICAL_TEMPLATES = (
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2)),
)
HORIZONTAL_TEMPLATES = tuple(tuple((c, r) for r, c in shape) for shape in VERTICAL_TEMPLATES)


def _get_bg(grid: np.ndarray) -> int:
    vals = np.bincount(np.asarray(grid, dtype=int).reshape(-1))
    return int(np.argmax(vals))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _sort_markers_clockwise(markers):
    top = sorted((m for m in markers if m["side"] == "top"), key=lambda m: m["pos"][1])
    right = sorted((m for m in markers if m["side"] == "right"), key=lambda m: m["pos"][0])
    bottom = sorted((m for m in markers if m["side"] == "bottom"), key=lambda m: -m["pos"][1])
    left = sorted((m for m in markers if m["side"] == "left"), key=lambda m: -m["pos"][0])
    return top + right + bottom + left


def _parse_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert colors, "expected non-background colors"
    frame_color = max(colors, key=lambda c: int(np.sum(g == c)))

    frame_pts = [(int(r), int(c)) for r, c in np.argwhere(g == frame_color)]
    assert frame_pts, "frame color must exist"
    frame_bbox = _bbox(frame_pts)
    top, left, bottom, right = frame_bbox

    markers = []
    for c in range(left, right + 1):
        if int(g[top, c]) not in (bg, frame_color):
            markers.append({"pos": (top, c), "side": "top", "color": int(g[top, c])})
    for r in range(top + 1, bottom):
        if int(g[r, right]) not in (bg, frame_color):
            markers.append({"pos": (r, right), "side": "right", "color": int(g[r, right])})
    for c in range(right, left - 1, -1):
        if int(g[bottom, c]) not in (bg, frame_color):
            markers.append({"pos": (bottom, c), "side": "bottom", "color": int(g[bottom, c])})
    for r in range(bottom - 1, top, -1):
        if int(g[r, left]) not in (bg, frame_color):
            markers.append({"pos": (r, left), "side": "left", "color": int(g[r, left])})

    assert markers, "expected at least one gap marker"
    marker_positions = {m["pos"] for m in markers}
    marker_colors = [m["color"] for m in markers]
    assert len(set(marker_colors)) == len(marker_colors), "marker colors must be unique"

    object_mask = (g != bg) & (g != frame_color)
    for r, c in marker_positions:
        object_mask[r, c] = False

    sprites = []
    for pts in _components(object_mask):
        colors_here = {int(g[r, c]) for r, c in pts}
        assert len(colors_here) == 1, "sprites must be monochrome"
        color = colors_here.pop()
        assert color not in {bg, frame_color}
        sprites.append({"color": color, "coords": pts, "bbox": _bbox(pts), "area": len(pts)})

    assert len(sprites) == len(markers), "expected one sprite per marker"
    sprite_colors = {sp["color"] for sp in sprites}
    assert sprite_colors == set(marker_colors), "sprite colors must match marker colors exactly"

    return bg, frame_color, frame_bbox, _sort_markers_clockwise(markers), sprites


def _normalize_shape(coords, side: str):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    if side == "top":
        edge = min(rows)
        center = (min(cols) + max(cols)) // 2
        return [(r - edge, c - center) for r, c in coords]
    if side == "bottom":
        edge = max(rows)
        center = (min(cols) + max(cols)) // 2
        return [(r - edge, c - center) for r, c in coords]
    if side == "left":
        edge = min(cols)
        center = (min(rows) + max(rows)) // 2
        return [(r - center, c - edge) for r, c in coords]
    if side == "right":
        edge = max(cols)
        center = (min(rows) + max(rows)) // 2
        return [(r - center, c - edge) for r, c in coords]
    raise AssertionError(side)


def _motion_plan(coords, side: str, marker_pos, frame_bbox):
    top, left, bottom, right = frame_bbox
    rel = _normalize_shape(coords, side)

    if side == "top":
        anchor = (top + 1, marker_pos[1])
        dr, dc = 1, 0
    elif side == "bottom":
        anchor = (bottom - 1, marker_pos[1])
        dr, dc = -1, 0
    elif side == "left":
        anchor = (marker_pos[0], left + 1)
        dr, dc = 0, 1
    elif side == "right":
        anchor = (marker_pos[0], right - 1)
        dr, dc = 0, -1
    else:
        raise AssertionError(side)

    initial = [(anchor[0] + rr, anchor[1] + cc) for rr, cc in rel]
    for r, c in initial:
        assert top + 1 <= r <= bottom - 1
        assert left + 1 <= c <= right - 1

    rows = [r for r, _ in initial]
    cols = [c for _, c in initial]
    if side == "top":
        travel = (bottom - 1) - max(rows)
    elif side == "bottom":
        travel = min(rows) - (top + 1)
    elif side == "left":
        travel = (right - 1) - max(cols)
    else:
        travel = min(cols) - (left + 1)

    assert travel >= 0

    trail = set()
    for step in range(travel):
        for r, c in initial:
            trail.add((r + step * dr, c + step * dc))

    final = {(r + travel * dr, c + travel * dc) for r, c in initial}
    for r, c in trail | final:
        assert top + 1 <= r <= bottom - 1
        assert left + 1 <= c <= right - 1

    return {
        "trail": trail,
        "final": final,
        "travel": int(travel),
        "dir": (dr, dc),
        "initial": set(initial),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored outer sprite to the same-colored gap marker on the hollow frame,
    slide the sprite into the frame from that side, leave a grey trail, and park it
    against the opposite inner wall.
    """
    g = np.asarray(grid, dtype=int)
    bg, frame_color, frame_bbox, markers, sprites = _parse_scene(g)

    out = np.full_like(g, bg)
    out[g == frame_color] = frame_color
    for marker in markers:
        r, c = marker["pos"]
        out[r, c] = marker["color"]

    sprite_by_color = {sp["color"]: sp for sp in sprites}
    for marker in markers:
        sprite = sprite_by_color[marker["color"]]
        plan = _motion_plan(sprite["coords"], marker["side"], marker["pos"], frame_bbox)
        for r, c in plan["trail"]:
            out[r, c] = frame_color
        for r, c in plan["final"]:
            out[r, c] = marker["color"]

    return out


def _draw_frame(grid: np.ndarray, frame_bbox):
    top, left, bottom, right = frame_bbox
    grid[top, left : right + 1] = FRAME
    grid[bottom, left : right + 1] = FRAME
    grid[top : bottom + 1, left] = FRAME
    grid[top : bottom + 1, right] = FRAME


def _gap_cells(side: str, center: int, frame_bbox):
    top, left, bottom, right = frame_bbox
    if side == "top":
        return [(top, center - 1), (top, center), (top, center + 1)]
    if side == "bottom":
        return [(bottom, center - 1), (bottom, center), (bottom, center + 1)]
    if side == "left":
        return [(center - 1, left), (center, left), (center + 1, left)]
    if side == "right":
        return [(center - 1, right), (center, right), (center + 1, right)]
    raise AssertionError(side)


def _pick_centers_for_side(rng: random.Random, side: str, frame_bbox, count: int, preset=None):
    preset = [] if preset is None else list(preset)
    top, left, bottom, right = frame_bbox
    if side in ("top", "bottom"):
        candidates = list(range(left + 2, right - 1))
    else:
        candidates = list(range(top + 2, bottom - 1))

    chosen = []
    for val in preset:
        if val not in candidates:
            return None
        chosen.append(val)

    rng.shuffle(candidates)
    for cand in candidates:
        if len(chosen) >= count:
            break
        if any(abs(cand - prev) < 4 for prev in chosen):
            continue
        chosen.append(cand)

    if len(chosen) != count:
        return None
    return chosen


def _choose_side_allocation(rng: random.Random, total: int, require_cross: bool):
    counts = {side: 0 for side in SIDES}
    mandatory = []
    if require_cross:
        vside = rng.choice(("top", "bottom"))
        hside = rng.choice(("left", "right"))
        mandatory = [vside, hside]
    else:
        mandatory = list(rng.sample(SIDES, 2))

    for side in mandatory:
        counts[side] += 1

    while sum(counts.values()) < total:
        choices = [side for side in SIDES if counts[side] < 2]
        side = rng.choice(choices)
        counts[side] += 1

    if len([side for side, n in counts.items() if n > 0]) < 2:
        return None
    return counts, mandatory


def _shape_templates_for_side(side: str):
    return VERTICAL_TEMPLATES if side in ("top", "bottom") else HORIZONTAL_TEMPLATES


def _choose_shape(rng: random.Random, side: str, marker_pos, frame_bbox):
    candidates = []
    for shape in _shape_templates_for_side(side):
        try:
            plan = _motion_plan(shape, side, marker_pos, frame_bbox)
        except AssertionError:
            continue
        if plan["travel"] >= 2:
            candidates.append(shape)
    if not candidates:
        return None
    return list(rng.choice(candidates))


def _final_and_trail_masks(items, frame_bbox):
    plans = []
    for item in items:
        plan = _motion_plan(item["shape"], item["side"], item["marker_pos"], frame_bbox)
        plans.append(plan)
    return plans


def _valid_output_interactions(plans, require_cross: bool):
    crossed = False
    for i in range(len(plans)):
        for j in range(i + 1, len(plans)):
            if plans[i]["final"] & plans[j]["final"]:
                return False
            if plans[i]["final"] & plans[j]["trail"]:
                return False
            if plans[j]["final"] & plans[i]["trail"]:
                return False
            if plans[i]["trail"] & plans[j]["trail"]:
                crossed = True
    return crossed if require_cross else True


def _place_shape(grid: np.ndarray, coords, color: int):
    for r, c in coords:
        grid[r, c] = color


def _can_place_input_sprite(grid: np.ndarray, abs_coords, frame_bbox):
    h, w = grid.shape
    top, left, bottom, right = frame_bbox
    pts = set(abs_coords)
    for r, c in abs_coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if top <= r <= bottom and left <= c <= right:
            return False
        if int(grid[r, c]) != BG:
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pts:
                    continue
                if int(grid[nr, nc]) != BG:
                    return False
    return True


def _place_input_sprite_randomly(rng: random.Random, grid: np.ndarray, shape, color: int, frame_bbox):
    h, w = grid.shape
    shape_rows = [r for r, _ in shape]
    shape_cols = [c for _, c in shape]
    sh = max(shape_rows) + 1
    sw = max(shape_cols) + 1
    for _ in range(SPRITE_PLACEMENT_ATTEMPTS):
        r0 = rng.randint(0, h - sh)
        c0 = rng.randint(0, w - sw)
        abs_coords = [(r0 + r, c0 + c) for r, c in shape]
        if _can_place_input_sprite(grid, abs_coords, frame_bbox):
            _place_shape(grid, abs_coords, color)
            return True
    return False


def _build_gap_plan(rng: random.Random, frame_bbox, total_markers: int, require_cross: bool):
    alloc = _choose_side_allocation(rng, total_markers, require_cross)
    if alloc is None:
        return None
    counts, mandatory = alloc

    top, left, bottom, right = frame_bbox
    cross_col = None
    cross_row = None
    if require_cross:
        cross_col = rng.randint(left + 2, right - 2)
        cross_row = rng.randint(top + 2, bottom - 2)

    centers = {}
    for side, count in counts.items():
        preset = []
        if require_cross and side == mandatory[0]:
            if side in ("top", "bottom"):
                preset = [cross_col]
        if require_cross and side == mandatory[1]:
            if side in ("left", "right"):
                preset = [cross_row]
        picked = _pick_centers_for_side(rng, side, frame_bbox, count, preset=preset)
        if picked is None:
            return None
        centers[side] = picked

    return centers


def _build_items(rng: random.Random, frame_bbox, centers, colors):
    items = []
    color_iter = iter(colors)
    for side in SIDES:
        for center in centers.get(side, []):
            marker_pos = (frame_bbox[0], center) if side == "top" else None
            if side == "bottom":
                marker_pos = (frame_bbox[2], center)
            elif side == "left":
                marker_pos = (center, frame_bbox[1])
            elif side == "right":
                marker_pos = (center, frame_bbox[3])
            shape = _choose_shape(rng, side, marker_pos, frame_bbox)
            if shape is None:
                return None
            items.append(
                {
                    "side": side,
                    "marker_pos": marker_pos,
                    "color": next(color_iter),
                    "shape": shape,
                }
            )
    return items


def _draw_input_grid(frame_bbox, items):
    top, left, bottom, right = frame_bbox
    h = bottom + 5 + random.randint(0, 3)
    w = right + 5 + random.randint(0, 3)
    grid = np.full((h, w), BG, dtype=int)
    _draw_frame(grid, frame_bbox)

    for item in items:
        side = item["side"]
        marker_r, marker_c = item["marker_pos"]
        for r, c in _gap_cells(side, item["marker_pos"][1] if side in ("top", "bottom") else item["marker_pos"][0], frame_bbox):
            grid[r, c] = BG
        grid[marker_r, marker_c] = item["color"]

    for item in items:
        if not _place_input_sprite_randomly(random, grid, item["shape"], item["color"], frame_bbox):
            return None
    return grid


def _gen_attempt(rng: random.Random):
    inner_h = rng.randint(8, 14)
    inner_w = rng.randint(8, 14)
    top_margin = rng.randint(4, 7)
    left_margin = rng.randint(4, 7)
    bottom_margin = rng.randint(5, 8)
    right_margin = rng.randint(5, 8)

    top = top_margin
    left = left_margin
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    h = bottom + bottom_margin + 1
    w = right + right_margin + 1
    frame_bbox = (top, left, bottom, right)

    total_markers = rng.randint(2, min(5, len(PALETTE)))
    require_cross = rng.random() < 0.5

    centers = _build_gap_plan(rng, frame_bbox, total_markers, require_cross)
    if centers is None:
        return None

    colors = rng.sample(PALETTE, total_markers)
    items = _build_items(rng, frame_bbox, centers, colors)
    if items is None:
        return None

    plans = _final_and_trail_masks(items, frame_bbox)
    if not _valid_output_interactions(plans, require_cross):
        return None

    grid = np.full((h, w), BG, dtype=int)
    _draw_frame(grid, frame_bbox)
    for item in items:
        side = item["side"]
        if side in ("top", "bottom"):
            center = item["marker_pos"][1]
        else:
            center = item["marker_pos"][0]
        for r, c in _gap_cells(side, center, frame_bbox):
            grid[r, c] = BG
        mr, mc = item["marker_pos"]
        grid[mr, mc] = item["color"]

    for item in items:
        if not _place_input_sprite_randomly(rng, grid, item["shape"], item["color"], frame_bbox):
            return None

    try:
        _bg, _frame, _bbox, markers, sprites = _parse_scene(grid)
    except AssertionError:
        return None
    if len(markers) != total_markers or len(sprites) != total_markers:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
