from __future__ import annotations

from collections import deque
import random

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _points_to_mask(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in points:
        mask[r, c] = True
    return mask


def _touches_boundary(points: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _single_side_marker(region: np.ndarray, obj_mask: np.ndarray) -> int | None:
    touched = 0
    side_values: list[int] = []

    if obj_mask[0, :].any():
        touched += 1
        side_values.extend(int(v) for v in region[0, obj_mask[0, :]])
    if obj_mask[-1, :].any():
        touched += 1
        side_values.extend(int(v) for v in region[-1, obj_mask[-1, :]])
    if obj_mask[:, 0].any():
        touched += 1
        side_values.extend(int(v) for v in region[obj_mask[:, 0], 0])
    if obj_mask[:, -1].any():
        touched += 1
        side_values.extend(int(v) for v in region[obj_mask[:, -1], -1])

    if touched != 1 or not side_values:
        return None

    marker = side_values[0]
    if all(value == marker for value in side_values):
        return int(marker)
    return None


def _looks_like_panel(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    area = len(points)
    if h < 5 or w < 5 or area < 20:
        return False
    if area / float(h * w) < 0.35:
        return False

    local = np.zeros((h, w), dtype=bool)
    for r, c in points:
        local[r - r0, c - c0] = True
    return bool(local[0, :].any() and local[-1, :].any() and local[:, 0].any() and local[:, -1].any())


def _object_fill_points(region: np.ndarray, panel_color: int, obj_points: list[tuple[int, int]]) -> set[tuple[int, int]]:
    obj_set = set(obj_points)
    fill_points = set(obj_points)
    panel_mask = region == panel_color

    for hole in _components(panel_mask):
        if _touches_boundary(hole, region.shape):
            continue
        boundary: set[tuple[int, int]] = set()
        for r, c in hole:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < region.shape[0] and 0 <= nc < region.shape[1] and not panel_mask[nr, nc]:
                    boundary.add((nr, nc))
        if boundary and boundary.issubset(obj_set):
            fill_points.update(hole)

    return fill_points


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each side-marked object inside a monochrome panel to its boundary marker color.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = _background_color(g)
    out = g.copy()
    planned: dict[tuple[int, int], int] = {}

    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for panel_points in _components(g == color):
            if not _looks_like_panel(panel_points):
                continue

            r0, c0, r1, c1 = _bbox(panel_points)
            region = g[r0 : r1 + 1, c0 : c1 + 1]

            for obj_points in _components(region != color):
                obj_colors = {int(region[r, c]) for r, c in obj_points}
                if len(obj_colors) < 2:
                    continue

                obj_mask = _points_to_mask(region.shape, obj_points)
                marker = _single_side_marker(region, obj_mask)
                if marker is None:
                    continue

                for rr, cc in _object_fill_points(region, color, obj_points):
                    planned[(r0 + rr, c0 + cc)] = marker

    for (r, c), marker in planned.items():
        out[r, c] = marker
    return out


def _component_count(mask: np.ndarray) -> int:
    return len(_components(mask))


def _hole_count(mask: np.ndarray) -> int:
    holes = 0
    empty = ~mask
    for comp in _components(empty):
        if not _touches_boundary(comp, mask.shape):
            holes += 1
    return holes


def _rand_color(rng: random.Random, exclude: set[int]) -> int:
    choices = [color for color in range(10) if color not in exclude]
    return rng.choice(choices)


def _make_target_mask(h: int, w: int, side: str, want_hole: bool, rng: random.Random) -> np.ndarray | None:
    for _ in range(80):
        mask = np.ones((h, w), dtype=bool)

        if side in ("top", "bottom"):
            left_trim = rng.randint(0, max(0, w - 2))
            right_trim = rng.randint(0, max(0, w - left_trim - 2))
            if side == "top":
                if left_trim:
                    mask[0, :left_trim] = False
                if right_trim:
                    mask[0, w - right_trim :] = False
            else:
                if left_trim:
                    mask[-1, :left_trim] = False
                if right_trim:
                    mask[-1, w - right_trim :] = False
        else:
            top_trim = rng.randint(0, max(0, h - 2))
            bottom_trim = rng.randint(0, max(0, h - top_trim - 2))
            if side == "left":
                if top_trim:
                    mask[:top_trim, 0] = False
                if bottom_trim:
                    mask[h - bottom_trim :, 0] = False
            else:
                if top_trim:
                    mask[:top_trim, -1] = False
                if bottom_trim:
                    mask[h - bottom_trim :, -1] = False

        edges = [edge for edge in SIDES if edge != side]
        notch_count = rng.choices([0, 1, 2], weights=[1, 3, 2])[0]
        for _ in range(notch_count):
            edge = rng.choice(edges)
            if edge in ("top", "bottom") and h >= 3 and w >= 3:
                depth = rng.randint(1, min(2, h - 2))
                width = rng.randint(1, w - 2)
                start = rng.randint(1, w - width)
                if edge == "top":
                    mask[:depth, start : start + width] = False
                else:
                    mask[h - depth :, start : start + width] = False
            elif edge in ("left", "right") and h >= 3 and w >= 3:
                depth = rng.randint(1, min(2, w - 2))
                height = rng.randint(1, h - 2)
                start = rng.randint(1, h - height)
                if edge == "left":
                    mask[start : start + height, :depth] = False
                else:
                    mask[start : start + height, w - depth :] = False

        wanted_holes = 0
        if want_hole and h >= 5 and w >= 5:
            wanted_holes = 1
            if h >= 7 and w >= 7 and rng.random() < 0.35:
                wanted_holes = 2

        hole_guard = np.zeros((h, w), dtype=bool)
        carved = 0
        for _ in range(wanted_holes * 20):
            if carved >= wanted_holes:
                break
            hh = rng.randint(1, min(2, h - 4))
            ww = rng.randint(1, min(2, w - 4))
            hr = rng.randint(1, h - hh - 1)
            hc = rng.randint(1, w - ww - 1)
            if hole_guard[max(0, hr - 1) : min(h, hr + hh + 1), max(0, hc - 1) : min(w, hc + ww + 1)].any():
                continue
            mask[hr : hr + hh, hc : hc + ww] = False
            hole_guard[hr : hr + hh, hc : hc + ww] = True
            carved += 1

        if _component_count(mask) != 1:
            continue
        if want_hole and _hole_count(mask) == 0:
            continue
        if not want_hole and mask.sum() < 5:
            continue

        if side == "top" and not mask[0, :].any():
            continue
        if side == "bottom" and not mask[-1, :].any():
            continue
        if side == "left" and not mask[:, 0].any():
            continue
        if side == "right" and not mask[:, -1].any():
            continue
        return mask
    return None


def _mark_contact_side(local: np.ndarray, mask: np.ndarray, side: str, marker_color: int) -> None:
    if side == "top":
        local[0, mask[0, :]] = marker_color
    elif side == "bottom":
        local[-1, mask[-1, :]] = marker_color
    elif side == "left":
        local[mask[:, 0], 0] = marker_color
    else:
        local[mask[:, -1], -1] = marker_color


def _place_bbox(used: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    ph, pw = used.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(ph, r0 + h + 1)
    cc1 = min(pw, c0 + w + 1)
    used[rr0:rr1, cc0:cc1] = True


def _can_place_bbox(used: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    ph, pw = used.shape
    if r0 < 0 or c0 < 0 or r0 + h > ph or c0 + w > pw:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(ph, r0 + h + 1)
    cc1 = min(pw, c0 + w + 1)
    return not used[rr0:rr1, cc0:cc1].any()


def _add_target_object(
    panel: np.ndarray,
    used: np.ndarray,
    side: str,
    rng: random.Random,
) -> bool:
    ph, pw = panel.shape
    want_hole = rng.random() < 0.65

    if side in ("top", "bottom"):
        min_h = 5 if want_hole else 3
        min_w = 5 if want_hole else 3
        max_h = min(ph - 2, 8)
        max_w = min(pw - 2, 9)
    else:
        min_h = 5 if want_hole else 3
        min_w = 5 if want_hole else 3
        max_h = min(ph - 2, 9)
        max_w = min(pw - 2, 8)

    if max_h < min_h or max_w < min_w:
        want_hole = False
        min_h = min_w = 3
        max_h = min(ph - 2, 8)
        max_w = min(pw - 2, 8)
    if max_h < min_h or max_w < min_w:
        return False

    panel_color = int(panel[0, 0])

    for _ in range(120):
        h = rng.randint(min_h, max_h)
        w = rng.randint(min_w, max_w)
        if side == "top":
            r0 = 0
            c0 = rng.randint(1, pw - w - 1)
        elif side == "bottom":
            r0 = ph - h
            c0 = rng.randint(1, pw - w - 1)
        elif side == "left":
            r0 = rng.randint(1, ph - h - 1)
            c0 = 0
        else:
            r0 = rng.randint(1, ph - h - 1)
            c0 = pw - w

        if not _can_place_bbox(used, r0, c0, h, w):
            continue

        mask = _make_target_mask(h, w, side, want_hole, rng)
        if mask is None:
            continue

        body_color = _rand_color(rng, {panel_color})
        marker_color = _rand_color(rng, {panel_color, body_color})

        local = panel[r0 : r0 + h, c0 : c0 + w]
        local[mask] = body_color
        _mark_contact_side(local, mask, side, marker_color)
        _place_bbox(used, r0, c0, h, w)
        return True
    return False


def _add_distractor(panel: np.ndarray, used: np.ndarray, rng: random.Random) -> bool:
    ph, pw = panel.shape
    if ph < 7 or pw < 7:
        return False
    panel_color = int(panel[0, 0])

    for _ in range(80):
        h = rng.randint(3, min(5, ph - 2))
        w = rng.randint(3, min(5, pw - 2))
        r0 = rng.randint(1, ph - h - 1)
        c0 = rng.randint(1, pw - w - 1)
        if not _can_place_bbox(used, r0, c0, h, w):
            continue

        mask = np.ones((h, w), dtype=bool)
        if h >= 4 and w >= 4 and rng.random() < 0.5:
            mask[1:-1, 1:-1] = False
            mask[1, 1] = True
        accent_side = rng.choice(SIDES)
        body_color = _rand_color(rng, {panel_color})
        accent_color = _rand_color(rng, {panel_color, body_color})

        local = panel[r0 : r0 + h, c0 : c0 + w]
        local[mask] = body_color
        if accent_side == "top":
            local[0, mask[0, :]] = accent_color
        elif accent_side == "bottom":
            local[-1, mask[-1, :]] = accent_color
        elif accent_side == "left":
            local[mask[:, 0], 0] = accent_color
        else:
            local[mask[:, -1], -1] = accent_color

        _place_bbox(used, r0, c0, h, w)
        return True
    return False


def _paint_panel(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    panel_color: int,
    rng: random.Random,
) -> bool:
    panel = grid[top : top + height, left : left + width]
    panel[:, :] = panel_color
    used = np.zeros((height, width), dtype=bool)

    target_count = rng.choices([1, 2, 3], weights=[3, 4, 2])[0]
    if min(height, width) < 9:
        target_count = min(target_count, 2)
    if min(height, width) < 7:
        target_count = 1

    sides = rng.sample(list(SIDES), k=target_count)
    for side in sides:
        if not _add_target_object(panel, used, side, rng):
            return False

    distractor_count = rng.choices([0, 1, 2], weights=[3, 3, 1])[0]
    for _ in range(distractor_count):
        _add_distractor(panel, used, rng)

    return True


def _panel_slots(h: int, w: int, rng: random.Random) -> list[tuple[int, int, int, int]]:
    slots: list[tuple[int, int, int, int]] = []
    used = np.zeros((h, w), dtype=bool)
    panel_count = rng.choices([1, 2, 3], weights=[3, 4, 2])[0]

    for _ in range(panel_count):
        placed = False
        for _ in range(120):
            ph = rng.randint(7, min(13, h - 2))
            pw = rng.randint(7, min(13, w - 2))
            if ph >= h or pw >= w:
                continue
            r0 = rng.randint(1, h - ph - 1)
            c0 = rng.randint(1, w - pw - 1)
            rr0 = max(0, r0 - 2)
            cc0 = max(0, c0 - 2)
            rr1 = min(h, r0 + ph + 2)
            cc1 = min(w, c0 + pw + 2)
            if used[rr0:rr1, cc0:cc1].any():
                continue
            used[rr0:rr1, cc0:cc1] = True
            slots.append((r0, c0, ph, pw))
            placed = True
            break
        if not placed and not slots:
            return []
    return slots


def _valid_background_mode(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)]) == bg


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 31)
    w = rng.randint(18, 31)
    bg = rng.randrange(10)
    grid = np.full((h, w), bg, dtype=int)

    slots = _panel_slots(h, w, rng)
    if not slots:
        return None

    panel_colors = rng.sample([color for color in range(10) if color != bg], k=len(slots))
    for (r0, c0, ph, pw), panel_color in zip(slots, panel_colors):
        if not _paint_panel(grid, r0, c0, ph, pw, panel_color, rng):
            return None

    if not _valid_background_mode(grid, bg):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
