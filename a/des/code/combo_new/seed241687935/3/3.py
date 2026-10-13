from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_DIR_TO_VEC = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}
_DIRS = tuple(_DIR_TO_VEC)
_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_ARROW_LENGTHS = range(1, 6)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in _N4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))

            rows = [coord[0] for coord in coords]
            cols = [coord[1] for coord in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return comps


def _arrow_template(direction: str, length: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    assert direction in _DIR_TO_VEC
    assert 1 <= length <= 5

    if direction == "right":
        coords = {(1, c) for c in range(length)}
        coords |= {(0, length - 1), (2, length - 1), (1, length)}
        return coords, (1, length)
    if direction == "left":
        coords = {(1, c) for c in range(1, length + 1)}
        coords |= {(0, 1), (2, 1), (1, 0)}
        return coords, (1, 0)
    if direction == "down":
        coords = {(r, 1) for r in range(length)}
        coords |= {(length - 1, 0), (length - 1, 2), (length, 1)}
        return coords, (length, 1)

    coords = {(r, 1) for r in range(1, length + 1)}
    coords |= {(1, 0), (1, 2), (0, 1)}
    return coords, (0, 1)


def _is_solid_rectangle(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    return len(comp["coords"]) == area


def _match_arrow(comp: dict) -> dict | None:
    r0, c0, _r1, _c1 = comp["bbox"]
    rel = {(r - r0, c - c0) for r, c in comp["coords"]}
    for direction in _DIRS:
        for length in _ARROW_LENGTHS:
            expected, tip_rel = _arrow_template(direction, length)
            if rel == expected:
                return {
                    "color": int(comp["color"]),
                    "coords": list(comp["coords"]),
                    "bbox": comp["bbox"],
                    "direction": direction,
                    "length": int(length),
                    "tip": (int(r0 + tip_rel[0]), int(c0 + tip_rel[1])),
                }
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each solid panel from the side indicated by its adjacent arrow, then erase the arrows.

    Input structure:
    - The background is the most common color.
    - Several monochrome solid rectangles act as target panels, and they do not touch.
    - Each panel has one adjacent monochrome arrow on one side. The arrow is a straight
      segment plus a 3-pixel triangular head, points up/down/left/right toward the panel,
      and has shaft length between 1 and 5.

    Rule:
    - Identify every arrow's target panel, direction, color, and shaft length L.
    - The side touched by the arrow is the preserved side of the panel.
    - Keep the first L rows or columns of the panel unchanged when moving inward from that side.
    - Recolor the remaining rows or columns of the panel to the arrow's color, then erase the arrow to background.

    Key insight:
    - Each arrow encodes three independent instructions at once: which panel side it acts from,
      which replacement color to use, and how many rows or columns should stay in the panel's original color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    bg = _get_bg(g)

    panels: list[dict] = []
    arrows: list[dict] = []
    for comp in _components(g, bg):
        if _is_solid_rectangle(comp):
            panels.append(comp)
            continue
        arrow = _match_arrow(comp)
        assert arrow is not None, "every non-panel component must be a valid arrow"
        arrows.append(arrow)

    assert panels, "expected at least one panel"
    assert len(panels) == len(arrows), "expected one arrow per panel"

    panel_id = -np.ones((h, w), dtype=int)
    for idx, panel in enumerate(panels):
        r0, c0, r1, c1 = panel["bbox"]
        panel_id[r0 : r1 + 1, c0 : c1 + 1] = idx

    assigned: dict[int, dict] = {}
    for arrow in arrows:
        dr, dc = _DIR_TO_VEC[arrow["direction"]]
        tr = int(arrow["tip"][0] + dr)
        tc = int(arrow["tip"][1] + dc)
        assert 0 <= tr < h and 0 <= tc < w, "arrow must point inside the grid"
        idx = int(panel_id[tr, tc])
        assert idx >= 0, "arrow tip must point to a panel"
        assert int(panels[idx]["color"]) != int(arrow["color"]), "arrow color must differ from its panel color"
        assert idx not in assigned, "each panel must have exactly one arrow"
        assigned[idx] = arrow

    assert len(assigned) == len(panels), "every panel must receive an arrow"

    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["coords"]:
            out[r, c] = bg

    for idx, panel in enumerate(panels):
        arrow = assigned[idx]
        r0, c0, r1, c1 = panel["bbox"]
        repaint_color = int(arrow["color"])
        keep = int(arrow["length"])

        if arrow["direction"] == "right":
            start = c0 + keep
            if start <= c1:
                out[r0 : r1 + 1, start : c1 + 1] = repaint_color
        elif arrow["direction"] == "left":
            end = c1 - keep
            if c0 <= end:
                out[r0 : r1 + 1, c0 : end + 1] = repaint_color
        elif arrow["direction"] == "down":
            start = r0 + keep
            if start <= r1:
                out[start : r1 + 1, c0 : c1 + 1] = repaint_color
        else:
            end = r1 - keep
            if r0 <= end:
                out[r0 : end + 1, c0 : c1 + 1] = repaint_color

    return out


# --------------- generator ---------------

_BG = 0
_GRID_MIN = 18
_GRID_MAX = 30
_PANEL_COUNT_OPTIONS = (1, 2, 2, 3, 3, 4)
_PANEL_SIDE_MIN = 3
_PANEL_SIDE_MAX = 7
_GEN_ATTEMPTS = 400


def _sample_direction_list(count: int) -> list[str]:
    dirs = random.sample(list(_DIRS), k=min(count, len(_DIRS)))
    while len(dirs) < count:
        dirs.append(random.choice(_DIRS))
    random.shuffle(dirs)
    return dirs


def _sample_panel_shape(direction: str, length: int) -> tuple[int, int]:
    if direction in ("right", "left"):
        panel_h = random.randint(_PANEL_SIDE_MIN, _PANEL_SIDE_MAX)
        panel_w = random.randint(max(_PANEL_SIDE_MIN, length + 1), _PANEL_SIDE_MAX)
    else:
        panel_h = random.randint(max(_PANEL_SIDE_MIN, length + 1), _PANEL_SIDE_MAX)
        panel_w = random.randint(_PANEL_SIDE_MIN, _PANEL_SIDE_MAX)
    return int(panel_h), int(panel_w)


def _build_group(
    direction: str,
    length: int,
    panel_h: int,
    panel_w: int,
    panel_color: int,
    arrow_color: int,
) -> dict:
    arrow_rel, _tip_rel = _arrow_template(direction, length)

    if direction in ("right", "left"):
        anchor = random.randint(1, panel_h - 2)
    else:
        anchor = random.randint(1, panel_w - 2)

    if direction == "right":
        arrow_shift_r, arrow_shift_c = anchor - 1, 0
        panel_bbox = (0, length + 1, panel_h - 1, length + panel_w)
        group_shape = (panel_h, length + 1 + panel_w)
    elif direction == "left":
        arrow_shift_r, arrow_shift_c = anchor - 1, panel_w
        panel_bbox = (0, 0, panel_h - 1, panel_w - 1)
        group_shape = (panel_h, panel_w + length + 1)
    elif direction == "down":
        arrow_shift_r, arrow_shift_c = 0, anchor - 1
        panel_bbox = (length + 1, 0, length + panel_h, panel_w - 1)
        group_shape = (length + 1 + panel_h, panel_w)
    else:
        arrow_shift_r, arrow_shift_c = panel_h, anchor - 1
        panel_bbox = (0, 0, panel_h - 1, panel_w - 1)
        group_shape = (panel_h + length + 1, panel_w)

    arrow_coords = [(int(r + arrow_shift_r), int(c + arrow_shift_c)) for r, c in sorted(arrow_rel)]
    panel_coords = [
        (int(r), int(c))
        for r in range(panel_bbox[0], panel_bbox[2] + 1)
        for c in range(panel_bbox[1], panel_bbox[3] + 1)
    ]
    return {
        "direction": direction,
        "length": int(length),
        "panel_color": int(panel_color),
        "arrow_color": int(arrow_color),
        "group_shape": tuple(int(v) for v in group_shape),
        "panel_bbox": tuple(int(v) for v in panel_bbox),
        "panel_coords": panel_coords,
        "arrow_coords": arrow_coords,
        "all_coords": panel_coords + arrow_coords,
    }


def _can_place(group: dict, top: int, left: int, occupied: np.ndarray) -> bool:
    h, w = map(int, occupied.shape)
    for r, c in group["all_coords"]:
        rr = int(top + r)
        cc = int(left + c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        r0 = max(0, rr - 1)
        r1 = min(h, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(w, cc + 2)
        if bool(np.any(occupied[r0:r1, c0:c1])):
            return False
    return True


def _place_group(group: dict, top: int, left: int, grid: np.ndarray, occupied: np.ndarray) -> dict:
    pr0, pc0, pr1, pc1 = group["panel_bbox"]
    abs_panel_bbox = (top + pr0, left + pc0, top + pr1, left + pc1)

    for r, c in group["panel_coords"]:
        rr = int(top + r)
        cc = int(left + c)
        grid[rr, cc] = int(group["panel_color"])
        occupied[rr, cc] = True

    abs_arrow_coords: list[tuple[int, int]] = []
    for r, c in group["arrow_coords"]:
        rr = int(top + r)
        cc = int(left + c)
        grid[rr, cc] = int(group["arrow_color"])
        occupied[rr, cc] = True
        abs_arrow_coords.append((rr, cc))

    return {
        "direction": group["direction"],
        "length": int(group["length"]),
        "panel_color": int(group["panel_color"]),
        "arrow_color": int(group["arrow_color"]),
        "panel_bbox": tuple(int(v) for v in abs_panel_bbox),
        "arrow_coords": abs_arrow_coords,
    }


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        for r, c in spec["arrow_coords"]:
            out[r, c] = _BG

    for spec in specs:
        r0, c0, r1, c1 = spec["panel_bbox"]
        keep = int(spec["length"])
        color = int(spec["arrow_color"])
        if spec["direction"] == "right":
            start = c0 + keep
            if start <= c1:
                out[r0 : r1 + 1, start : c1 + 1] = color
        elif spec["direction"] == "left":
            end = c1 - keep
            if c0 <= end:
                out[r0 : r1 + 1, c0 : end + 1] = color
        elif spec["direction"] == "down":
            start = r0 + keep
            if start <= r1:
                out[start : r1 + 1, c0 : c1 + 1] = color
        else:
            end = r1 - keep
            if r0 <= end:
                out[r0 : end + 1, c0 : c1 + 1] = color
    return out


def _background_is_unique_mode(grid: np.ndarray) -> bool:
    counts = Counter(int(v) for v in grid.ravel())
    return counts[_BG] > max(count for color, count in counts.items() if color != _BG)


def _generate_case() -> np.ndarray | None:
    h = random.randint(_GRID_MIN, _GRID_MAX)
    w = random.randint(_GRID_MIN, _GRID_MAX)
    grid = np.full((h, w), _BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    panel_count = random.choice(_PANEL_COUNT_OPTIONS)
    if panel_count >= 4 and min(h, w) < 22:
        panel_count = 3

    colors = random.sample(range(1, 10), 2 * panel_count)
    panel_colors = colors[:panel_count]
    arrow_colors = colors[panel_count:]
    directions = _sample_direction_list(panel_count)

    groups: list[dict] = []
    for idx in range(panel_count):
        direction = directions[idx]
        length = random.randint(1, 5)
        panel_h, panel_w = _sample_panel_shape(direction, length)
        groups.append(
            _build_group(
                direction=direction,
                length=length,
                panel_h=panel_h,
                panel_w=panel_w,
                panel_color=panel_colors[idx],
                arrow_color=arrow_colors[idx],
            )
        )

    random.shuffle(groups)
    specs: list[dict] = []
    for group in groups:
        gh, gw = group["group_shape"]
        placed = False
        for _ in range(200):
            top = random.randint(0, h - gh)
            left = random.randint(0, w - gw)
            if not _can_place(group, top, left, occupied):
                continue
            specs.append(_place_group(group, top, left, grid, occupied))
            placed = True
            break
        if not placed:
            return None

    if not _background_is_unique_mode(grid):
        return None

    expected = _expected_output(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-panel puzzle")
