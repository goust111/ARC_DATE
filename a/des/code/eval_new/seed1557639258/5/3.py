import numpy as np


BG = 0
GREY = 5
MAROON = 9
OBJECT_COLORS = np.array([1, 2, 3, 4, 6, 7, 8], dtype=int)

ARROW_PATTERNS = {
    "right": {(0, 0), (1, 0), (2, 0), (1, 1)},
    "left": {(0, 1), (1, 1), (2, 1), (1, 0)},
    "down": {(0, 0), (0, 1), (0, 2), (1, 1)},
    "up": {(1, 0), (1, 1), (1, 2), (0, 1)},
}

DIRS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}

GEN_H_RANGE = (12, 25)
GEN_W_RANGE = (12, 25)
GEN_OBJECT_COUNT_RANGE = (1, 4)
GEN_EXTRA_OBS_RANGE = (0, 3)
GEN_ATTEMPTS = 400
GEN_OBJECT_ATTEMPTS = 120
GEN_EXTRA_OBS_ATTEMPTS = 60


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), max(rows), min(cols), max(cols)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _normalize(coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, _, c0, _ = _bbox(coords)
    return {(r - r0, c - c0) for r, c in coords}


def _match_arrow(coords: list[tuple[int, int]]) -> str | None:
    norm = _normalize(coords)
    for direction, pattern in ARROW_PATTERNS.items():
        if norm == pattern:
            return direction
    return None


def _rectangles_from_mask(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects = []
    for comp in _components(mask):
        box = _bbox(comp)
        r0, r1, c0, c1 = box
        if len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1):
            rects.append(box)
    return rects


def _row_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _col_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _axis_slices(patch: np.ndarray, axis: int) -> list[np.ndarray]:
    if axis == 1:
        return [patch[:, i].copy() for i in range(patch.shape[1])]
    return [patch[i, :].copy() for i in range(patch.shape[0])]


def _slice_period(slices: list[np.ndarray]) -> int:
    for period in range(1, len(slices) + 1):
        if all(np.array_equal(slices[i], slices[i % period]) for i in range(len(slices))):
            return period
    return len(slices)


def _stop_distance(
    obj_box: tuple[int, int, int, int],
    direction: str,
    obstacles: list[tuple[int, int, int, int]],
    h: int,
    w: int,
) -> int:
    r0, r1, c0, c1 = obj_box
    if direction == "right":
        dist = w - 1 - c1
        for or0, or1, oc0, _oc1 in obstacles:
            if _row_overlap(r0, r1, or0, or1) and oc0 > c1:
                dist = min(dist, oc0 - 1 - c1)
        return dist
    if direction == "left":
        dist = c0
        for or0, or1, _oc0, oc1 in obstacles:
            if _row_overlap(r0, r1, or0, or1) and oc1 < c0:
                dist = min(dist, c0 - (oc1 + 1))
        return dist
    if direction == "down":
        dist = h - 1 - r1
        for or0, _or1, oc0, oc1 in obstacles:
            if _col_overlap(c0, c1, oc0, oc1) and or0 > r1:
                dist = min(dist, or0 - 1 - r1)
        return dist
    dist = r0
    for _or0, or1, oc0, oc1 in obstacles:
        if _col_overlap(c0, c1, oc0, oc1) and or1 < r0:
            dist = min(dist, r0 - (or1 + 1))
    return dist


def _paint_sweep(
    out: np.ndarray,
    patch: np.ndarray,
    obj_box: tuple[int, int, int, int],
    direction: str,
    distance: int,
) -> None:
    r0, r1, c0, c1 = obj_box
    axis = 1 if direction in ("left", "right") else 0
    slices = _axis_slices(patch, axis)
    period = _slice_period(slices)

    if axis == 1:
        start_c = c0 if direction == "right" else c0 - distance
        total = patch.shape[1] + distance
        for offset in range(total):
            out[r0 : r1 + 1, start_c + offset] = slices[offset % period]
        return

    start_r = r0 if direction == "down" else r0 - distance
    total = patch.shape[0] + distance
    for offset in range(total):
        out[start_r + offset, c0 : c1 + 1] = slices[offset % period]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each striped rectangle along its adjacent maroon arrow and extend its stripe pattern across the swept path.

    Input structure:
    - The background is the most common color.
    - Foreground rectangles are solid-filled bounding boxes whose cells use exactly two alternating colors as horizontal or vertical stripes.
    - Each moving rectangle has a touching maroon arrow marker on one side; grey solid rectangles are static obstacles.

    Rule:
    - Identify every maroon arrow by its 4-pixel shape and use its pointing direction as the rectangle's motion direction.
    - Move the adjacent striped rectangle straight along that axis until its front edge would hit the grid boundary or a grey obstacle.
    - Remove the arrow, and fill the whole swept span from the rectangle's first occupied slice to its final occupied slice by repeating the rectangle's own row/column slice period along the motion axis.

    Key insight:
    - The trail is not a single-color drag mark: it is a periodic continuation of the rectangle's internal stripe slices, anchored to the rectangle's top-to-bottom or left-to-right ordering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = get_bg(g)

    arrows = []
    for comp in _components(g == MAROON):
        direction = _match_arrow(comp)
        assert direction is not None, "invalid maroon component"
        arrows.append({"direction": direction, "coords": comp, "bbox": _bbox(comp)})

    object_mask = (g != bg) & (g != GREY) & (g != MAROON)
    objects = []
    for comp in _components(object_mask):
        box = _bbox(comp)
        r0, r1, c0, c1 = box
        assert len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1), "objects must be solid rectangles"
        patch = _crop(g, box)
        objects.append({"bbox": box, "patch": patch})

    obstacles = _rectangles_from_mask(g == GREY)
    out = g.copy()
    out[g == MAROON] = bg

    for arrow in arrows:
        ar0, ar1, ac0, ac1 = arrow["bbox"]
        direction = arrow["direction"]
        candidates = []
        for obj in objects:
            r0, r1, c0, c1 = obj["bbox"]
            if direction == "right" and c0 == ac1 + 1 and _row_overlap(r0, r1, ar0, ar1):
                candidates.append(obj)
            elif direction == "left" and c1 + 1 == ac0 and _row_overlap(r0, r1, ar0, ar1):
                candidates.append(obj)
            elif direction == "down" and r0 == ar1 + 1 and _col_overlap(c0, c1, ac0, ac1):
                candidates.append(obj)
            elif direction == "up" and r1 + 1 == ar0 and _col_overlap(c0, c1, ac0, ac1):
                candidates.append(obj)
        assert len(candidates) == 1, "each arrow must match exactly one rectangle"
        obj = candidates[0]
        dist = _stop_distance(obj["bbox"], direction, obstacles, h, w)
        _paint_sweep(out, obj["patch"], obj["bbox"], direction, dist)

    return out


def _stripe_patch(height: int, width: int, direction: str, colors: tuple[int, int]) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    if direction in ("left", "right"):
        for c in range(width):
            patch[:, c] = colors[c % 2]
    else:
        for r in range(height):
            patch[r, :] = colors[r % 2]
    return patch


def _arrow_patch(direction: str) -> np.ndarray:
    coords = ARROW_PATTERNS[direction]
    h = 1 + max(r for r, _ in coords)
    w = 1 + max(c for _, c in coords)
    patch = np.full((h, w), BG, dtype=int)
    for r, c in coords:
        patch[r, c] = MAROON
    return patch


def _rect_free(mask: np.ndarray, rect: tuple[int, int, int, int], pad: int = 0) -> bool:
    h, w = mask.shape
    r0, r1, c0, c1 = rect
    rr0 = max(0, r0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(w - 1, c1 + pad)
    return not mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark_rect(mask: np.ndarray, rect: tuple[int, int, int, int], pad: int = 0) -> None:
    h, w = mask.shape
    r0, r1, c0, c1 = rect
    rr0 = max(0, r0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int, transparent: int | None = None) -> None:
    h, w = patch.shape
    target = grid[top : top + h, left : left + w]
    if transparent is None:
        target[:] = patch
    else:
        keep = patch != transparent
        target[keep] = patch[keep]


def _plan_object(
    h: int,
    w: int,
    reserved: np.ndarray,
    rng: np.random.Generator,
) -> dict | None:
    for _ in range(GEN_OBJECT_ATTEMPTS):
        direction = str(rng.choice(np.array(["right", "left", "down", "up"], dtype=object)))
        horizontal = direction in ("left", "right")
        obj_h = int(rng.integers(3, 5)) if horizontal else int(rng.integers(2, 5))
        obj_w = int(rng.integers(2, 5)) if horizontal else int(rng.integers(3, 5))
        colors = tuple(int(v) for v in rng.choice(OBJECT_COLORS, size=2, replace=False))
        obstacle_stop = bool(rng.integers(0, 2))

        if direction == "right":
            if w < obj_w + 5:
                continue
            r0 = int(rng.integers(0, h - obj_h + 1))
            c0_max = w - obj_w - (3 if not obstacle_stop else 6)
            if c0_max < 2:
                continue
            c0 = int(rng.integers(2, c0_max + 1))
            arrow_top = r0 + int(rng.integers(0, obj_h - 2))
            arrow_rect = (arrow_top, arrow_top + 2, c0 - 2, c0 - 1)
            obstacle_rect = None
            span_end = w - 1
            if obstacle_stop:
                obs_w = int(rng.integers(1, 4))
                obs_left_min = c0 + obj_w + 2
                obs_left_max = w - obs_w - 1
                if obs_left_min > obs_left_max:
                    continue
                obs_left = int(rng.integers(obs_left_min, obs_left_max + 1))
                obstacle_rect = (r0, r0 + obj_h - 1, obs_left, obs_left + obs_w - 1)
                span_end = obs_left - 1
            sweep_rect = (r0, r0 + obj_h - 1, c0, span_end)
            object_rect = (r0, r0 + obj_h - 1, c0, c0 + obj_w - 1)
        elif direction == "left":
            if w < obj_w + 5:
                continue
            r0 = int(rng.integers(0, h - obj_h + 1))
            c0_min = 0 if not obstacle_stop else 3
            c0_max = w - obj_w - 2
            if c0_min > c0_max:
                continue
            c0 = int(rng.integers(c0_min, c0_max + 1))
            arrow_top = r0 + int(rng.integers(0, obj_h - 2))
            arrow_rect = (arrow_top, arrow_top + 2, c0 + obj_w, c0 + obj_w + 1)
            obstacle_rect = None
            span_start = 0
            if obstacle_stop:
                obs_w = int(rng.integers(1, 4))
                obs_left_max = c0 - 3
                if obs_left_max < 0:
                    continue
                obs_left = int(rng.integers(0, obs_left_max + 1))
                obstacle_rect = (r0, r0 + obj_h - 1, obs_left, obs_left + obs_w - 1)
                span_start = obs_left + obs_w
            sweep_rect = (r0, r0 + obj_h - 1, span_start, c0 + obj_w - 1)
            object_rect = (r0, r0 + obj_h - 1, c0, c0 + obj_w - 1)
        elif direction == "down":
            if h < obj_h + 5:
                continue
            r0_max = h - obj_h - (3 if not obstacle_stop else 6)
            if r0_max < 2:
                continue
            r0 = int(rng.integers(2, r0_max + 1))
            c0 = int(rng.integers(0, w - obj_w + 1))
            arrow_left = c0 + int(rng.integers(0, obj_w - 2))
            arrow_rect = (r0 - 2, r0 - 1, arrow_left, arrow_left + 2)
            obstacle_rect = None
            span_end = h - 1
            if obstacle_stop:
                obs_h = int(rng.integers(1, 4))
                obs_top_min = r0 + obj_h + 2
                obs_top_max = h - obs_h - 1
                if obs_top_min > obs_top_max:
                    continue
                obs_top = int(rng.integers(obs_top_min, obs_top_max + 1))
                obstacle_rect = (obs_top, obs_top + obs_h - 1, c0, c0 + obj_w - 1)
                span_end = obs_top - 1
            sweep_rect = (r0, span_end, c0, c0 + obj_w - 1)
            object_rect = (r0, r0 + obj_h - 1, c0, c0 + obj_w - 1)
        else:
            if h < obj_h + 5:
                continue
            r0_min = 0 if not obstacle_stop else 3
            r0_max = h - obj_h - 2
            if r0_min > r0_max:
                continue
            r0 = int(rng.integers(r0_min, r0_max + 1))
            c0 = int(rng.integers(0, w - obj_w + 1))
            arrow_left = c0 + int(rng.integers(0, obj_w - 2))
            arrow_rect = (r0 + obj_h, r0 + obj_h + 1, arrow_left, arrow_left + 2)
            obstacle_rect = None
            span_start = 0
            if obstacle_stop:
                obs_h = int(rng.integers(1, 4))
                obs_top_max = r0 - 3
                if obs_top_max < 0:
                    continue
                obs_top = int(rng.integers(0, obs_top_max + 1))
                obstacle_rect = (obs_top, obs_top + obs_h - 1, c0, c0 + obj_w - 1)
                span_start = obs_top + obs_h
            sweep_rect = (span_start, r0 + obj_h - 1, c0, c0 + obj_w - 1)
            object_rect = (r0, r0 + obj_h - 1, c0, c0 + obj_w - 1)

        rects = [arrow_rect, sweep_rect]
        if obstacle_rect is not None:
            rects.append(obstacle_rect)
        if not all(_rect_free(reserved, rect, pad=1) for rect in rects):
            continue

        patch = _stripe_patch(obj_h, obj_w, direction, colors)
        arrow = _arrow_patch(direction)
        return {
            "direction": direction,
            "object_rect": object_rect,
            "sweep_rect": sweep_rect,
            "obstacle_rect": obstacle_rect,
            "patch": patch,
            "arrow": arrow,
            "arrow_rect": arrow_rect,
        }
    return None


def _render_plan(
    h: int,
    w: int,
    plans: list[dict],
    extra_obstacles: list[tuple[int, int, int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for plan in plans:
        r0, _r1, c0, _c1 = plan["object_rect"]
        _paste(grid, plan["patch"], r0, c0)
        ar0, _ar1, ac0, _ac1 = plan["arrow_rect"]
        _paste(grid, plan["arrow"], ar0, ac0, transparent=BG)
        if plan["obstacle_rect"] is not None:
            or0, or1, oc0, oc1 = plan["obstacle_rect"]
            grid[or0 : or1 + 1, oc0 : oc1 + 1] = GREY
    for or0, or1, oc0, oc1 in extra_obstacles:
        grid[or0 : or1 + 1, oc0 : oc1 + 1] = GREY
    return grid


def _add_extra_obstacles(
    h: int,
    w: int,
    reserved: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    count = int(rng.integers(*GEN_EXTRA_OBS_RANGE))
    for _ in range(count):
        placed = False
        for _ in range(GEN_EXTRA_OBS_ATTEMPTS):
            rh = int(rng.integers(1, min(5, h) + 1))
            rw = int(rng.integers(1, min(5, w) + 1))
            r0 = int(rng.integers(0, h - rh + 1))
            c0 = int(rng.integers(0, w - rw + 1))
            rect = (r0, r0 + rh - 1, c0, c0 + rw - 1)
            if not _rect_free(reserved, rect, pad=1):
                continue
            _mark_rect(reserved, rect, pad=1)
            rects.append(rect)
            placed = True
            break
        if not placed:
            break
    return rects


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GEN_H_RANGE))
    w = int(rng.integers(*GEN_W_RANGE))
    reserved = np.zeros((h, w), dtype=bool)
    plans: list[dict] = []
    target_count = int(rng.integers(*GEN_OBJECT_COUNT_RANGE))

    for _ in range(target_count):
        plan = _plan_object(h, w, reserved, rng)
        if plan is None:
            if not plans:
                return None
            break
        plans.append(plan)
        _mark_rect(reserved, plan["sweep_rect"], pad=1)
        _mark_rect(reserved, plan["arrow_rect"], pad=1)
        if plan["obstacle_rect"] is not None:
            _mark_rect(reserved, plan["obstacle_rect"], pad=1)

    if not plans:
        return None

    extra_obstacles = _add_extra_obstacles(h, w, reserved, rng)
    grid = _render_plan(h, w, plans, extra_obstacles)
    out = main(grid)

    if get_bg(grid) != BG or get_bg(out) != BG:
        return None
    if grid.shape != out.shape or max(grid.shape) > 30:
        return None
    if len(_components(grid == MAROON)) != len(plans):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid striped-push input")
