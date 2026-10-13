import random
from collections import Counter

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

MARKER_COLORS = (BLUE, RED, GREEN)
OBJECT_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE)
TOP_H = 2
GEN_RETRIES = 400

_GEN_COUNTER = 0


def _mode_color(grid: np.ndarray) -> int:
    flat = [int(v) for v in grid.ravel()]
    return Counter(flat).most_common(1)[0][0]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords):
    coords = list(coords)
    r0, c0, _, _ = _bbox(coords)
    return sorted((r - r0, c - c0) for r, c in coords)


def _shape_dims(shape):
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _shift(shape, dr: int, dc: int):
    return [(r + dr, c + dc) for r, c in shape]


def _overlap_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _overlap_bbox(a, b) -> bool:
    return _overlap_1d(a[0], a[2], b[0], b[2]) and _overlap_1d(a[1], a[3], b[1], b[3])


def _connected_components_by_color(panel: np.ndarray, bg: int):
    h, w = panel.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(1, w - 1):
            if seen[r, c]:
                continue
            color = int(panel[r, c])
            if color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 1 <= nc < w - 1 and not seen[nr, nc] and int(panel[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            r0, c0, r1, c1 = _bbox(coords)
            objects.append(
                {
                    "color": color,
                    "coords": sorted(coords),
                    "bbox": (r0, c0, r1, c1),
                    "shape": _normalize(coords),
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                    "top": r0,
                    "left": c0,
                }
            )
    return objects


def _find_panel_start(grid: np.ndarray) -> int:
    h, _ = grid.shape
    for r in range(1, h):
        if np.all(grid[r:, 0] == BLUE) and np.all(grid[r:, -1] == RED):
            return r
    raise ValueError("could not locate the scene panel")


def _marker_rows(grid: np.ndarray, panel_start: int):
    rows = []
    for r in range(panel_start):
        color = int(grid[r, 0])
        if color in MARKER_COLORS:
            rows.append((r, color))
    return rows


def _paint_shape(panel: np.ndarray, shape, top: int, left: int, color: int):
    for r, c in shape:
        panel[top + r, left + c] = color


def _render_scene(panel_shape, bg: int, objects, selected_colors):
    h, w = panel_shape
    out = np.full(panel_shape, bg, dtype=int)
    out[:, 0] = BLUE
    out[:, -1] = RED

    kept = [obj for obj in objects if obj["color"] in selected_colors]
    counts = {color: sum(1 for obj in kept if obj["color"] == color) for color in MARKER_COLORS}

    stationary = [obj for obj in kept if obj["color"] not in (BLUE, RED)]
    for obj in stationary:
        _paint_shape(out, obj["shape"], obj["top"], obj["left"], obj["color"])

    for color, side in ((BLUE, "left"), (RED, "right")):
        movers = [obj for obj in kept if obj["color"] == color]
        if not movers:
            continue
        if side == "left":
            movers.sort(key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))
        else:
            movers.sort(key=lambda obj: (-obj["bbox"][3], obj["bbox"][0]))

        placed = []
        for obj in movers:
            r0, _, r1, _ = obj["bbox"]
            width = obj["width"]
            if side == "left":
                dest_left = 1
                while any(_overlap_bbox((r0, dest_left, r1, dest_left + width - 1), prev) for prev in placed):
                    dest_left += width
            else:
                dest_left = w - width - 1
                while any(_overlap_bbox((r0, dest_left, r1, dest_left + width - 1), prev) for prev in placed):
                    dest_left -= width
            _paint_shape(out, obj["shape"], obj["top"], dest_left, color)
            placed.append((r0, dest_left, r1, dest_left + width - 1))

    return out, counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the indicator-selected scene objects, slide surviving blue/red objects to their edge bars, and rewrite the indicator as count bars.

    Input structure:
    A black indicator strip sits above a grey scene panel. The panel has a blue vertical border on the left edge, a red vertical border on the right edge, and several separated 4-connected monochrome objects in the interior. Indicator rows contain single colored markers in column 0.

    Rule:
    Read the marker colors from the top strip. In the scene panel, keep every object whose color appears in the marker set and recolor every other object to the grey panel background. Then slide surviving blue objects horizontally toward the left so their leftmost pixel reaches column 1, and slide surviving red objects horizontally toward the right so their rightmost pixel reaches the penultimate column. Process each moving color from nearest boundary to farthest; if a proposed placement would overlap an already placed same-side object in row range, shift it inward by one object-width step. Objects of other surviving colors do not move. Finally, replace each top marker with a horizontal bar from column 0 whose length equals the number of surviving objects of that marker color; erase rows whose surviving count is zero.

    Key insight:
    The top strip is both an input filter and an output summary: it decides which colors survive, while the surviving object counts are written back into the same strip after the blue/red objects have been edge-stacked.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    panel_start = _find_panel_start(g)
    markers = _marker_rows(g, panel_start)
    selected = {color for _, color in markers}

    out = np.zeros_like(g)
    panel = g[panel_start:]
    bg = _mode_color(panel[:, 1:-1])
    objects = _connected_components_by_color(panel, bg)
    scene, counts = _render_scene(panel.shape, bg, objects, selected)
    out[panel_start:] = scene

    for row, color in markers:
        length = counts.get(color, 0)
        if length > 0:
            out[row, :length] = color
    return out


def _random_polyomino(rng: random.Random, area: int, min_h: int, max_h: int, min_w: int, max_w: int):
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(rng.choice(frontier))
        shape = _normalize(cells)
        if rng.random() < 0.5:
            shape = _normalize((r, -c) for r, c in shape)
        if rng.random() < 0.5:
            shape = _normalize((-r, c) for r, c in shape)
        if rng.random() < 0.35:
            shape = _normalize((c, r) for r, c in shape)
        h, w = _shape_dims(shape)
        if min_h <= h <= max_h and min_w <= w <= max_w:
            return shape
    return None


def _can_place(panel: np.ndarray, shape, top: int, left: int, color: int, bg: int, allow_touch_edge: bool = False):
    h, w = panel.shape
    cells = _shift(shape, top, left)
    for r, c in cells:
        if not (0 <= r < h and 1 <= c < w - 1):
            return False
        if not allow_touch_edge and (c <= 1 or c >= w - 2):
            return False
        if int(panel[r, c]) != bg:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(panel[nr, nc])
                if v != bg and (nr, nc) not in cells:
                    return False
    return True


def _place_object(panel: np.ndarray, shape, top: int, left: int, color: int):
    for r, c in shape:
        panel[top + r, left + c] = color


def _sample_rows(rng: random.Random, panel_h: int, shapes, overlap: bool):
    heights = [_shape_dims(shape)[0] for shape in shapes]
    if len(shapes) == 1:
        return [rng.randint(0, panel_h - heights[0])]

    if overlap:
        for _ in range(200):
            first = rng.randint(0, panel_h - heights[0])
            delta = rng.choice([-1, 1])
            second = first + delta
            if 0 <= second <= panel_h - heights[1]:
                span0 = (first, first + heights[0] - 1)
                span1 = (second, second + heights[1] - 1)
                if _overlap_1d(span0[0], span0[1], span1[0], span1[1]) and first != second:
                    return [first, second]
    else:
        for _ in range(200):
            rows = []
            ok = True
            for height in heights:
                row = rng.randint(0, panel_h - height)
                if any(_overlap_1d(row, row + height - 1, prev, prev + ph - 1) for prev, ph in rows):
                    ok = False
                    break
                rows.append((row, height))
            if ok:
                return [row for row, _ in rows]
    raise RuntimeError("failed to sample rows")


def _mover_destinations(panel_w: int, objects, side: str):
    placed = []
    dests = []
    for obj in objects:
        r0 = obj["top"]
        r1 = r0 + obj["height"] - 1
        width = obj["width"]
        if side == "left":
            left = 1
            while any(_overlap_bbox((r0, left, r1, left + width - 1), prev) for prev in placed):
                left += width
        else:
            left = panel_w - width - 1
            while any(_overlap_bbox((r0, left, r1, left + width - 1), prev) for prev in placed):
                left -= width
        placed.append((r0, left, r1, left + width - 1))
        dests.append(left)
    return dests


def _scenario(mode: int, rng: random.Random):
    total_objects = rng.randint(6, 10)
    if mode == 0:
        markers = [BLUE, RED]
        selected_counts = {BLUE: 2, RED: 1}
        overlap_color = BLUE
        green_present = False
    elif mode == 1:
        markers = [RED, GREEN]
        selected_counts = {RED: 2, GREEN: 0}
        overlap_color = RED
        green_present = False
    elif mode == 2:
        markers = [BLUE, GREEN]
        selected_counts = {BLUE: 2, GREEN: 1}
        overlap_color = BLUE if rng.random() < 0.5 else None
        green_present = True
    elif mode == 3:
        markers = [BLUE]
        selected_counts = {BLUE: 2}
        overlap_color = BLUE if rng.random() < 0.7 else None
        green_present = False
    elif mode == 4:
        markers = [RED]
        selected_counts = {RED: 2}
        overlap_color = RED if rng.random() < 0.7 else None
        green_present = False
    else:
        markers = [BLUE, RED]
        selected_counts = {BLUE: 1, RED: 2}
        overlap_color = RED if rng.random() < 0.5 else None
        green_present = False

    selected_total = sum(selected_counts.values())
    distractor_total = total_objects - selected_total
    distractor_total = max(distractor_total, 2)

    distractors = []
    if BLUE not in markers:
        distractors.append(BLUE)
    if RED not in markers:
        distractors.append(RED)
    if GREEN in markers and not green_present:
        pass
    else:
        if any(color not in markers for color in (YELLOW, ORANGE)) and not any(color not in markers for color in distractors):
            pass
    extra_palette = [color for color in OBJECT_COLORS if color not in markers]
    if not extra_palette:
        extra_palette = [YELLOW, ORANGE]
    if not any(color not in markers for color in distractors):
        distractors.append(rng.choice([color for color in extra_palette if color not in (BLUE, RED)]))
    while len(distractors) < distractor_total:
        distractors.append(rng.choice(extra_palette))

    return markers, selected_counts, overlap_color, distractors


def _build_objects(rng: random.Random, panel_h: int, panel_w: int, markers, selected_counts, overlap_color):
    survivors = []

    for color in markers:
        count = selected_counts.get(color, 0)
        if count == 0:
            continue
        shapes = []
        for idx in range(count):
            if color in (BLUE, RED):
                area = rng.randint(3, 7)
                min_h = 2 if count >= 2 and idx < 2 else 1
                shape = _random_polyomino(rng, area, min_h=min_h, max_h=3, min_w=2, max_w=4)
            else:
                area = rng.randint(3, 7)
                shape = _random_polyomino(rng, area, min_h=1, max_h=3, min_w=2, max_w=4)
            if shape is None:
                raise RuntimeError("failed to build a survivor shape")
            shapes.append(shape)

        overlap = color == overlap_color and count >= 2
        rows = _sample_rows(rng, panel_h, shapes[:2], overlap) if count >= 2 else _sample_rows(rng, panel_h, shapes, False)
        if count > 2:
            used = rows[:]
            for shape in shapes[2:]:
                h, _ = _shape_dims(shape)
                for _ in range(200):
                    row = rng.randint(0, panel_h - h)
                    if all(not _overlap_1d(row, row + h - 1, prev, prev + _shape_dims(prev_shape)[0] - 1) for prev, prev_shape in used):
                        used.append((row, shape))
                        break
            rows = [row for row in rows]

        objs = []
        for idx, shape in enumerate(shapes):
            h, w = _shape_dims(shape)
            top = rows[idx]
            objs.append({"color": color, "shape": shape, "height": h, "width": w, "top": top})

        if color == BLUE:
            objs.sort(key=lambda obj: (obj["top"], obj["height"]))
            dests = _mover_destinations(panel_w, objs, "left")
            for obj, left in zip(objs, dests):
                obj["left"] = left
                obj["bbox"] = (obj["top"], left, obj["top"] + obj["height"] - 1, left + obj["width"] - 1)
            survivors.extend(objs)
        elif color == RED:
            objs.sort(key=lambda obj: (obj["top"], obj["height"]))
            dests = _mover_destinations(panel_w, objs, "right")
            for obj, left in zip(objs, dests):
                obj["left"] = left
                obj["bbox"] = (obj["top"], left, obj["top"] + obj["height"] - 1, left + obj["width"] - 1)
            survivors.extend(objs)
        else:
            survivors.extend(objs)

    green_objects = [obj for obj in survivors if obj["color"] == GREEN]
    left_zone = 1 + sum(obj["width"] for obj in survivors if obj["color"] == BLUE) + 1
    right_zone = panel_w - 1 - sum(obj["width"] for obj in survivors if obj["color"] == RED) - 1
    for obj in green_objects:
        lo = max(left_zone, 3)
        hi = min(right_zone - obj["width"], panel_w - obj["width"] - 3)
        if lo > hi:
            raise RuntimeError("no room for green survivor")
        for _ in range(200):
            left = rng.randint(lo, hi)
            obj["left"] = left
            obj["bbox"] = (obj["top"], left, obj["top"] + obj["height"] - 1, left + obj["width"] - 1)
            if not any(
                other is not obj and _overlap_bbox(obj["bbox"], other["bbox"])
                for other in survivors
                if "bbox" in other
            ):
                break
        else:
            raise RuntimeError("failed to place green survivor")

    return survivors


def _render_expected(h: int, w: int, marker_rows, selected_counts, survivors):
    grid = np.zeros((h, w), dtype=int)
    panel = np.full((h - TOP_H, w), GREY, dtype=int)
    panel[:, 0] = BLUE
    panel[:, -1] = RED
    for obj in survivors:
        _place_object(panel, obj["shape"], obj["top"], obj["left"], obj["color"])
    grid[TOP_H:] = panel
    for row, color in marker_rows:
        count = selected_counts.get(color, 0)
        if count > 0:
            grid[row, :count] = color
    return grid


def _place_survivors_in_input(rng: random.Random, panel: np.ndarray, survivors):
    bg = GREY
    blues = [obj for obj in survivors if obj["color"] == BLUE]
    reds = [obj for obj in survivors if obj["color"] == RED]
    others = [obj for obj in survivors if obj["color"] not in (BLUE, RED)]

    for obj in others:
        _place_object(panel, obj["shape"], obj["top"], obj["left"], obj["color"])

    prev_left = 1
    for obj in sorted(blues, key=lambda item: (item["left"], item["top"])):
        width = obj["width"]
        candidates = []
        for left in range(max(3, obj["left"] + 1, prev_left + 1), panel.shape[1] - width - 2):
            if _can_place(panel, obj["shape"], obj["top"], left, obj["color"], bg):
                candidates.append(left)
        if not candidates:
            return False
        left = rng.choice(candidates)
        obj["input_left"] = left
        prev_left = left + width
        _place_object(panel, obj["shape"], obj["top"], left, obj["color"])

    prev_right = panel.shape[1] - 2
    for obj in sorted(reds, key=lambda item: (-item["left"], item["top"])):
        width = obj["width"]
        candidates = []
        for left in range(2, min(obj["left"] - 1, prev_right - width) + 1):
            if _can_place(panel, obj["shape"], obj["top"], left, obj["color"], bg):
                candidates.append(left)
        if not candidates:
            return False
        left = rng.choice(candidates)
        obj["input_left"] = left
        prev_right = left - 1
        _place_object(panel, obj["shape"], obj["top"], left, obj["color"])

    return True


def _place_distractors(rng: random.Random, panel: np.ndarray, colors):
    bg = GREY
    metadata = []
    for color in colors:
        for _ in range(200):
            area = rng.randint(3, 8)
            shape = _random_polyomino(rng, area, min_h=1, max_h=3, min_w=2, max_w=4)
            if shape is None:
                continue
            height, width = _shape_dims(shape)
            top = rng.randint(0, panel.shape[0] - height)
            left = rng.randint(2, panel.shape[1] - width - 3)
            if _can_place(panel, shape, top, left, color, bg):
                _place_object(panel, shape, top, left, color)
                metadata.append((color, top, left))
                break
        else:
            return False, metadata
    return True, metadata


def _make_attempt(mode: int, rng: random.Random):
    h = rng.randint(18, 24)
    w = rng.randint(18, 24)
    markers, selected_counts, overlap_color, distractor_colors = _scenario(mode, rng)
    panel_h = h - TOP_H
    rows = list(range(TOP_H))
    rng.shuffle(rows)
    marker_rows = [(rows[idx], color) for idx, color in enumerate(markers)]

    survivors = _build_objects(rng, panel_h, w, markers, selected_counts, overlap_color)
    expected = _render_expected(h, w, marker_rows, selected_counts, survivors)

    panel = np.full((panel_h, w), GREY, dtype=int)
    panel[:, 0] = BLUE
    panel[:, -1] = RED
    survivors_for_input = [dict(obj) for obj in survivors]
    if not _place_survivors_in_input(rng, panel, survivors_for_input):
        return None
    ok, _ = _place_distractors(rng, panel, distractor_colors)
    if not ok:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[TOP_H:] = panel
    for row, color in marker_rows:
        grid[row, 0] = color

    out = main(grid)
    if not np.array_equal(out, expected):
        return None

    objects = _connected_components_by_color(grid[TOP_H:], GREY)
    colors_present = {obj["color"] for obj in objects}
    if BLUE not in colors_present or RED not in colors_present:
        return None
    if not any(color not in markers for color in colors_present):
        return None
    if not any(selected_counts.get(color, 0) > 0 for color in markers):
        return None
    if GREEN in markers and selected_counts.get(GREEN, 0) == 0 and GREEN in colors_present:
        return None

    branch_ok = True
    if overlap_color is not None:
        movers = [obj for obj in survivors if obj["color"] == overlap_color]
        branch_ok = len(movers) >= 2 and any(
            _overlap_1d(a["bbox"][0], a["bbox"][2], b["bbox"][0], b["bbox"][2]) and a["left"] != b["left"]
            for i, a in enumerate(movers)
            for b in movers[i + 1 :]
        )
    if not branch_ok:
        return None

    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for _ in range(GEN_RETRIES):
        mode = _GEN_COUNTER % 6
        rng = random.Random(f"{random.randrange(1 << 30)}:{_GEN_COUNTER}")
        _GEN_COUNTER += 1
        grid = _make_attempt(mode, rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
