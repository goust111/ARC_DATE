from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
MARKER = 9
OBJECT_COLORS = (1, 2, 3)
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _marker_side(grid: np.ndarray) -> str:
    sides: list[str] = []
    if np.all(grid[0, :] == MARKER):
        sides.append("top")
    if np.all(grid[-1, :] == MARKER):
        sides.append("bottom")
    if np.all(grid[:, 0] == MARKER):
        sides.append("left")
    if np.all(grid[:, -1] == MARKER):
        sides.append("right")
    assert len(sides) == 1, f"expected exactly one full-edge marker, got {sides}"
    return sides[0]


def _find_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, MARKER) or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(pts, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
    return objects


def _sort_key(obj: dict, side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = obj["bbox"]
    if side == "top":
        return (r0, c0)
    if side == "bottom":
        return (-r1, c0)
    if side == "left":
        return (c0, r0)
    return (-c1, r0)


def _transverse_size(obj: dict, side: str) -> int:
    return _bbox_w(obj["bbox"]) if side in ("top", "bottom") else _bbox_h(obj["bbox"])


def _render_stacked(
    grid: np.ndarray,
    side: str,
    *,
    gap_mode: str = "current",
    reverse_order: bool = False,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    objects = _find_objects(g)
    assert 4 <= len(objects) <= 6, f"expected 4-6 objects, got {len(objects)}"

    order = sorted(objects, key=lambda obj: _sort_key(obj, side))
    if reverse_order:
        order = list(reversed(order))

    out = np.full_like(g, BG)
    if side == "top":
        out[0, :] = MARKER
    elif side == "bottom":
        out[-1, :] = MARKER
    elif side == "left":
        out[:, 0] = MARKER
    else:
        out[:, -1] = MARKER

    prev_obj = None
    prev_far = None
    for idx, obj in enumerate(order):
        coords = obj["coords"]
        r0, c0, r1, c1 = obj["bbox"]
        height = _bbox_h(obj["bbox"])
        width = _bbox_w(obj["bbox"])

        if idx == 0:
            gap = 0
        elif gap_mode == "current":
            gap = _transverse_size(obj, side)
        elif gap_mode == "previous":
            assert prev_obj is not None
            gap = _transverse_size(prev_obj, side)
        elif gap_mode == "zero":
            gap = 0
        elif gap_mode == "one":
            gap = 1
        else:
            raise ValueError(f"unknown gap mode: {gap_mode}")

        if side == "top":
            new_r0 = 1 if idx == 0 else int(prev_far) + gap + 1
            dr = new_r0 - r0
            dc = 0
            prev_far = new_r0 + height - 1
        elif side == "bottom":
            new_r1 = h - 2 if idx == 0 else int(prev_far) - gap - 1
            dr = new_r1 - r1
            dc = 0
            prev_far = new_r1 - height + 1
        elif side == "left":
            new_c0 = 1 if idx == 0 else int(prev_far) + gap + 1
            dr = 0
            dc = new_c0 - c0
            prev_far = new_c0 + width - 1
        else:
            new_c1 = w - 2 if idx == 0 else int(prev_far) - gap - 1
            dr = 0
            dc = new_c1 - c1
            prev_far = new_c1 - width + 1

        moved = coords + np.array([dr, dc], dtype=int)
        assert moved[:, 0].min() >= 0 and moved[:, 0].max() < h
        assert moved[:, 1].min() >= 0 and moved[:, 1].max() < w
        assert np.all(out[moved[:, 0], moved[:, 1]] == BG), "stacked objects must remain disjoint"
        out[moved[:, 0], moved[:, 1]] = obj["color"]
        prev_obj = obj

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Stack the foreground components toward the marker edge using self-width-dependent gaps.

    Input structure:
    - A black background contains exactly one full-edge maroon marker line on the top, bottom, left, or right border.
    - Away from that border are 4-6 pairwise separated 8-connected single-color objects in blue, red, or green.
    - Each object is small, irregular, and keeps its cross-axis position when moved.

    Rule:
    - The full-edge maroon marker defines gravity toward that same border: top->up, bottom->down, left->left, right->right.
    - Sort the foreground connected components by their original position along that gravity axis, with the component already closest to the target border handled first.
    - Move the first component so it sits immediately next to the marker line, then place each later component behind the previous one with a blank gap equal to that later component's own bounding-box size measured perpendicular to gravity.
    - Preserve each component's shape, color, and cross-axis coordinate, and keep the marker line in the output.

    Key insight:
    - The edge marker chooses the global fall direction, but the spacing is not tight or constant: every later component's offset depends on its own transverse bounding-box width/height.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    side = _marker_side(g)
    return _render_stacked(g, side)


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    if area < 1 or area > h * w:
        return None
    for _ in range(80):
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS8:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    return None


def _make_object(side: str, transverse: int) -> dict | None:
    for _ in range(120):
        if side in ("top", "bottom"):
            w = transverse
            h = random.randint(1, 3)
        else:
            h = transverse
            w = random.randint(1, 3)
        max_area = min(8, h * w)
        if max_area < 3:
            continue
        area = random.randint(3, max_area)
        mask = _random_connected_mask(h, w, area)
        if mask is None:
            continue
        coords = np.argwhere(mask)
        box = _bbox(coords)
        if _transverse_size({"bbox": box}, side) != transverse:
            continue
        return {
            "mask": mask,
            "coords": coords,
            "bbox": box,
            "color": random.choice(OBJECT_COLORS),
            "height": h,
            "width": w,
            "transverse": transverse,
            "axial": h if side in ("top", "bottom") else w,
        }
    return None


def _axis_positions(
    objs: list[dict],
    side: str,
    lead_gap: int,
    inner_gaps: list[int],
    axis_len: int,
) -> list[int]:
    pos: list[int] = []
    if side == "top":
        cur = 1 + lead_gap
        for idx, obj in enumerate(objs):
            pos.append(cur)
            cur += obj["height"]
            if idx < len(objs) - 1:
                cur += inner_gaps[idx]
    elif side == "bottom":
        cur = axis_len - 2 - lead_gap
        for idx, obj in enumerate(objs):
            pos.append(cur)
            cur -= obj["height"]
            if idx < len(objs) - 1:
                cur -= inner_gaps[idx]
    elif side == "left":
        cur = 1 + lead_gap
        for idx, obj in enumerate(objs):
            pos.append(cur)
            cur += obj["width"]
            if idx < len(objs) - 1:
                cur += inner_gaps[idx]
    else:
        cur = axis_len - 2 - lead_gap
        for idx, obj in enumerate(objs):
            pos.append(cur)
            cur -= obj["width"]
            if idx < len(objs) - 1:
                cur -= inner_gaps[idx]
    return pos


def _place_input_grid(
    objs: list[dict],
    side: str,
    axis_positions: list[int],
    cross_positions: list[int],
    shape: tuple[int, int],
) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    if side == "top":
        grid[0, :] = MARKER
    elif side == "bottom":
        grid[-1, :] = MARKER
    elif side == "left":
        grid[:, 0] = MARKER
    else:
        grid[:, -1] = MARKER

    for obj, axis_pos, cross_pos in zip(objs, axis_positions, cross_positions):
        if side in ("top", "bottom"):
            r0 = axis_pos if side == "top" else axis_pos - obj["height"] + 1
            c0 = cross_pos
        else:
            r0 = cross_pos
            c0 = axis_pos if side == "left" else axis_pos - obj["width"] + 1
        coords = obj["coords"] + np.array([r0, c0], dtype=int)
        assert coords[:, 0].min() >= 0 and coords[:, 0].max() < h
        assert coords[:, 1].min() >= 0 and coords[:, 1].max() < w
        assert np.all(grid[coords[:, 0], coords[:, 1]] == BG)
        grid[coords[:, 0], coords[:, 1]] = obj["color"]
    return grid


def _same_array(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(a.shape == b.shape and np.array_equal(a, b))


def _wrong_rule_collision_checks(inp: np.ndarray, expected: np.ndarray, side: str) -> bool:
    if _same_array(_render_stacked(inp, side, gap_mode="zero"), expected):
        return False
    if _same_array(_render_stacked(inp, side, gap_mode="one"), expected):
        return False
    if _same_array(_render_stacked(inp, side, gap_mode="previous"), expected):
        return False
    if _same_array(_render_stacked(inp, side, reverse_order=True), expected):
        return False
    return True


def _build_candidate() -> np.ndarray | None:
    side = random.choice(("top", "bottom", "left", "right"))
    count = random.randint(4, 6)

    transverse_values = [random.choice((1, 2, 3)) for _ in range(count)]
    if len(set(transverse_values)) < 2 or max(transverse_values) == 1:
        return None
    if all(transverse_values[i] == transverse_values[i - 1] for i in range(1, count)):
        return None

    objs: list[dict] = []
    for transverse in transverse_values:
        obj = _make_object(side, transverse)
        if obj is None:
            return None
        objs.append(obj)

    output_span = sum(obj["axial"] for obj in objs) + sum(obj["transverse"] for obj in objs[1:])
    lead_gap = random.randint(1, 4)
    inner_gaps = [random.randint(1, 4) for _ in range(count - 1)]
    input_span = lead_gap + sum(obj["axial"] for obj in objs) + sum(inner_gaps)

    if side in ("top", "bottom"):
        h_min = max(output_span + 2, input_span + 2)
        h = h_min + random.randint(1, 4)
        if h > 30:
            return None
        max_width = max(obj["width"] for obj in objs)
        w_low = max(12, max_width + 4)
        w_high = min(22, 30)
        if w_low > w_high:
            return None
        w = random.randint(w_low, w_high)
        cross_positions = [random.randint(1, w - obj["width"] - 1) for obj in objs]
        axis_positions = _axis_positions(objs, side, lead_gap, inner_gaps, h)
        grid = _place_input_grid(objs, side, axis_positions, cross_positions, (h, w))
    else:
        w_min = max(output_span + 2, input_span + 2)
        w = w_min + random.randint(1, 4)
        if w > 30:
            return None
        max_height = max(obj["height"] for obj in objs)
        h_low = max(12, max_height + 4)
        h_high = min(22, 30)
        if h_low > h_high:
            return None
        h = random.randint(h_low, h_high)
        cross_positions = [random.randint(1, h - obj["height"] - 1) for obj in objs]
        axis_positions = _axis_positions(objs, side, lead_gap, inner_gaps, w)
        grid = _place_input_grid(objs, side, axis_positions, cross_positions, (h, w))

    objects = _find_objects(grid)
    if len(objects) != count:
        return None

    primary_keys = [_sort_key(obj, side)[0] for obj in sorted(objects, key=lambda obj: _sort_key(obj, side))]
    if len(set(primary_keys)) != len(primary_keys):
        return None

    expected = main(grid)
    if _same_array(grid, expected):
        return None
    if not _wrong_rule_collision_checks(grid, expected, side):
        return None

    order = sorted(objects, key=lambda obj: _sort_key(obj, side))
    later_sizes = [_transverse_size(obj, side) for obj in order[1:]]
    if len(set(later_sizes)) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_candidate()
        if grid is None:
            continue
        out = main(grid)
        if grid.ndim != 2 or out.ndim != 2:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
