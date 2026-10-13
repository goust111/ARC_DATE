from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["top", "bottom", "left", "right"]
OUTLINE_CHOICES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_SHAPE_ATTEMPTS = 200
MAX_PLACEMENT_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _mask_bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_of_same_color(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append((color, pts))
    comps.sort(key=lambda item: _bbox(item[1])[:2])
    return comps


def _enclosed_mask(component_mask: np.ndarray) -> np.ndarray:
    h, w = component_mask.shape
    open_mask = ~component_mask
    reached = np.zeros_like(component_mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if open_mask[r, c] and not reached[r, c]:
                reached[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if open_mask[r, c] and not reached[r, c]:
                reached[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and not reached[nr, nc]:
                reached[nr, nc] = True
                queue.append((nr, nc))

    return open_mask & ~reached


def _outline_from_solid(solid: np.ndarray) -> np.ndarray:
    h, w = solid.shape
    outline = np.zeros_like(solid, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not solid[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not solid[nr, nc]:
                    outline[r, c] = True
                    break
    return outline


def _adjacent_pairs(mask: np.ndarray, orientation: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = mask.shape
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if orientation == "horizontal" and c + 1 < w and mask[r, c + 1]:
                pairs.append(((r, c), (r, c + 1)))
            if orientation == "vertical" and r + 1 < h and mask[r + 1, c]:
                pairs.append(((r, c), (r + 1, c)))
    return pairs


def _reflect_mask(mask: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "horizontal":
        return np.flipud(mask)
    return np.fliplr(mask)


def _side_corner_candidates(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    if side == "top":
        row = int(coords[:, 0].max())
        cols = np.flatnonzero(mask[row])
        return list(dict.fromkeys([(row, int(cols.min())), (row, int(cols.max()))]))
    if side == "bottom":
        row = int(coords[:, 0].min())
        cols = np.flatnonzero(mask[row])
        return list(dict.fromkeys([(row, int(cols.min())), (row, int(cols.max()))]))
    if side == "left":
        col = int(coords[:, 1].max())
        rows = np.flatnonzero(mask[:, col])
        return list(dict.fromkeys([(int(rows.min()), col), (int(rows.max()), col)]))
    col = int(coords[:, 1].min())
    rows = np.flatnonzero(mask[:, col])
    return list(dict.fromkeys([(int(rows.min()), col), (int(rows.max()), col)]))


def _detect_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    objects: list[dict] = []
    for color, pts in _components_of_same_color(grid, BG):
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            local_mask[r - r0, c - c0] = True

        enclosed = _enclosed_mask(local_mask)
        if not enclosed.any():
            continue

        local_grid = grid[r0 : r1 + 1, c0 : c1 + 1]
        helpers = [
            (r0 + lr, c0 + lc)
            for lr, lc in np.argwhere(enclosed & (local_grid != BG) & (local_grid != color))
        ]
        if len(helpers) != 2:
            continue

        helper_color = int(grid[helpers[0]])
        if any(int(grid[r, c]) != helper_color for r, c in helpers):
            continue

        (hr0, hc0), (hr1, hc1) = helpers
        if abs(hr0 - hr1) + abs(hc0 - hc1) != 1:
            continue
        if hr0 == hr1:
            orientation = "horizontal"
        elif hc0 == hc1:
            orientation = "vertical"
        else:
            continue

        marker_candidates: list[tuple[str, tuple[int, int]]] = []
        if r0 - 1 >= 0:
            for c in range(c0, c1 + 1):
                if int(grid[r0 - 1, c]) == helper_color:
                    marker_candidates.append(("top", (r0 - 1, c)))
        if r1 + 1 < h:
            for c in range(c0, c1 + 1):
                if int(grid[r1 + 1, c]) == helper_color:
                    marker_candidates.append(("bottom", (r1 + 1, c)))
        if c0 - 1 >= 0:
            for r in range(r0, r1 + 1):
                if int(grid[r, c0 - 1]) == helper_color:
                    marker_candidates.append(("left", (r, c0 - 1)))
        if c1 + 1 < w:
            for r in range(r0, r1 + 1):
                if int(grid[r, c1 + 1]) == helper_color:
                    marker_candidates.append(("right", (r, c1 + 1)))
        if len(marker_candidates) != 1:
            continue

        side, marker = marker_candidates[0]
        objects.append(
            {
                "color": color,
                "points": pts,
                "bbox": (r0, c0, r1, c1),
                "helpers": helpers,
                "helper_color": helper_color,
                "orientation": orientation,
                "marker": marker,
                "side": side,
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each closed outline onto the side marked by its external dot, choosing the flip axis from the two internal guide pixels.

    Input structure:
    - The background is black.
    - The grid contains one or more single-color closed outline shapes drawn as 4-connected one-pixel-wide borders, with black interior cells.
    - Inside each closed outline there are exactly two adjacent guide pixels of one non-background, non-border color; the pair is either horizontal or vertical.
    - Outside each outline, immediately next to one side, there is one dot of the same color as the guide pair.

    Rule:
    - For each outline, inspect the guide pair: a horizontal pair means flip the outline across a horizontal axis, and a vertical pair means flip it across a vertical axis.
    - Draw that flipped outline as a separate copy on the side indicated by the external dot, using the original outline color and keeping the original outline unchanged.
    - The two guide pixels and the external dot are removed from the output and become background.

    Key insight:
    - The inner pair encodes which reflection to apply, while the outer same-color dot encodes where the reflected copy must be placed.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    objects = _detect_objects(g)
    for obj in objects:
        for r, c in obj["helpers"]:
            out[r, c] = BG
        mr, mc = obj["marker"]
        out[mr, mc] = BG

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        local = np.zeros((box_h, box_w), dtype=bool)
        for r, c in obj["points"]:
            local[r - r0, c - c0] = True
        reflected = _reflect_mask(local, obj["orientation"])

        if obj["side"] == "top":
            base_r = r0 - box_h - 1
            base_c = c0
        elif obj["side"] == "bottom":
            base_r = r1 + 2
            base_c = c0
        elif obj["side"] == "left":
            base_r = r0
            base_c = c0 - box_w - 1
        else:
            base_r = r0
            base_c = c1 + 2

        for lr, lc in np.argwhere(reflected):
            rr = base_r + int(lr)
            cc = base_c + int(lc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = obj["color"]

    return out


def _add_bump(
    solid: np.ndarray,
    top: int,
    left: int,
    base_h: int,
    base_w: int,
    side: str,
    rng: random.Random,
) -> None:
    depth = rng.randint(1, 2)
    if side in {"top", "bottom"}:
        length = rng.randint(1, base_w)
        start = rng.randint(0, base_w - length)
    else:
        length = rng.randint(1, base_h)
        start = rng.randint(0, base_h - length)

    if side == "top":
        solid[top - depth : top, left + start : left + start + length] = True
    elif side == "bottom":
        solid[top + base_h : top + base_h + depth, left + start : left + start + length] = True
    elif side == "left":
        solid[top + start : top + start + length, left - depth : left] = True
    else:
        solid[top + start : top + start + length, left + base_w : left + base_w + depth] = True


def _make_shape(
    orientation: str, rng: random.Random
) -> tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int]], np.ndarray]:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        base_h = rng.randint(4, 6)
        base_w = rng.randint(4, 6)
        solid = np.zeros((12, 12), dtype=bool)
        top = 3
        left = 3
        solid[top : top + base_h, left : left + base_w] = True

        if orientation == "horizontal":
            _add_bump(solid, top, left, base_h, base_w, rng.choice(["top", "bottom"]), rng)
        else:
            _add_bump(solid, top, left, base_h, base_w, rng.choice(["left", "right"]), rng)

        for _extra in range(rng.randint(0, 2)):
            _add_bump(solid, top, left, base_h, base_w, rng.choice(SIDES), rng)

        solid = _crop_mask(solid)
        outline = _outline_from_solid(solid)
        interior = solid & ~outline
        pairs = _adjacent_pairs(interior, orientation)
        if not pairs:
            continue

        reflected = _reflect_mask(outline, orientation)
        if np.array_equal(outline, reflected):
            continue

        return outline, rng.choice(pairs), reflected

    raise RuntimeError("failed to generate a non-degenerate outline shape")


def _valid_marker_choices(
    outline: np.ndarray, reflected: np.ndarray
) -> list[tuple[str, tuple[int, int]]]:
    h, w = outline.shape
    choices: list[tuple[str, tuple[int, int]]] = []
    for side in SIDES:
        for rr, cc in _side_corner_candidates(reflected, side):
            if side == "top" and outline[0, cc]:
                choices.append((side, (0, cc)))
            elif side == "bottom" and outline[h - 1, cc]:
                choices.append((side, (h - 1, cc)))
            elif side == "left" and outline[rr, 0]:
                choices.append((side, (rr, 0)))
            elif side == "right" and outline[rr, w - 1]:
                choices.append((side, (rr, w - 1)))
    return choices


def _placement_box(
    side: str, top: int, left: int, height: int, width: int
) -> tuple[int, int, int, int]:
    if side == "top":
        return top - height - 1, left, top + height - 1, left + width - 1
    if side == "bottom":
        return top, left, top + 2 * height, left + width - 1
    if side == "left":
        return top, left - width - 1, top + height - 1, left + width - 1
    return top, left, top + height - 1, left + 2 * width


def _can_place(blocked: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = blocked.shape
    r0, c0, r1, c1 = box
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    er0 = max(0, r0 - pad)
    ec0 = max(0, c0 - pad)
    er1 = min(h - 1, r1 + pad)
    ec1 = min(w - 1, c1 + pad)
    return not blocked[er0 : er1 + 1, ec0 : ec1 + 1].any()


def _mark_blocked(blocked: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = box
    er0 = max(0, r0 - pad)
    ec0 = max(0, c0 - pad)
    er1 = min(h - 1, r1 + pad)
    ec1 = min(w - 1, c1 + pad)
    blocked[er0 : er1 + 1, ec0 : ec1 + 1] = True


def _paint_object(
    grid: np.ndarray,
    expected: np.ndarray,
    outline: np.ndarray,
    pair: tuple[tuple[int, int], tuple[int, int]],
    reflected: np.ndarray,
    side: str,
    marker_local: tuple[int, int],
    top: int,
    left: int,
    outline_color: int,
    helper_color: int,
) -> None:
    h, w = outline.shape
    for lr, lc in np.argwhere(outline):
        rr = top + int(lr)
        cc = left + int(lc)
        grid[rr, cc] = outline_color
        expected[rr, cc] = outline_color

    for lr, lc in pair:
        rr = top + lr
        cc = left + lc
        grid[rr, cc] = helper_color
        expected[rr, cc] = BG

    mr = top + marker_local[0]
    mc = left + marker_local[1]
    if side == "top":
        mr -= 1
        copy_top = top - h - 1
        copy_left = left
    elif side == "bottom":
        mr += 1
        copy_top = top + h + 1
        copy_left = left
    elif side == "left":
        mc -= 1
        copy_top = top
        copy_left = left - w - 1
    else:
        mc += 1
        copy_top = top
        copy_left = left + w + 1
    grid[mr, mc] = helper_color
    expected[mr, mc] = BG

    for lr, lc in np.argwhere(reflected):
        rr = copy_top + int(lr)
        cc = copy_left + int(lc)
        expected[rr, cc] = outline_color


def _orientation_plan(count: int, rng: random.Random) -> list[str]:
    if count == 1:
        return [rng.choice(["horizontal", "vertical"])]
    plan = ["horizontal", "vertical"]
    while len(plan) < count:
        plan.append(rng.choice(["horizontal", "vertical"]))
    rng.shuffle(plan)
    return plan


def _generate_candidate(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    height = rng.randint(18, 30)
    width = rng.randint(18, 30)
    object_count = rng.choice([1, 2, 2, 3])
    outline_color = rng.choice(OUTLINE_CHOICES)
    helper_pool = [c for c in OUTLINE_CHOICES if c != outline_color]
    if object_count > len(helper_pool):
        return None

    grid = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    helper_colors = rng.sample(helper_pool, object_count)
    for orientation, helper_color in zip(_orientation_plan(object_count, rng), helper_colors):
        outline, pair, reflected = _make_shape(orientation, rng)
        h, w = outline.shape
        marker_choices = _valid_marker_choices(outline, reflected)
        if not marker_choices:
            return None

        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            side, marker_local = rng.choice(marker_choices)
            if side == "top":
                if h + 1 > height - h:
                    continue
                top = rng.randint(h + 1, height - h)
                left = rng.randint(0, width - w)
            elif side == "bottom":
                if height - 2 * h - 1 < 0:
                    continue
                top = rng.randint(0, height - 2 * h - 1)
                left = rng.randint(0, width - w)
            elif side == "left":
                if w + 1 > width - w:
                    continue
                top = rng.randint(0, height - h)
                left = rng.randint(w + 1, width - w)
            else:
                if width - 2 * w - 1 < 0:
                    continue
                top = rng.randint(0, height - h)
                left = rng.randint(0, width - 2 * w - 1)
            box = _placement_box(side, top, left, h, w)
            if not _can_place(blocked, box, pad=1):
                continue
            _mark_blocked(blocked, box, pad=1)
            _paint_object(
                grid,
                expected,
                outline,
                pair,
                reflected,
                side,
                marker_local,
                top,
                left,
                outline_color,
                helper_color,
            )
            placed = True
            break
        if not placed:
            return None

    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _generate_candidate(rng)
        if candidate is None:
            continue
        grid, expected = candidate
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
