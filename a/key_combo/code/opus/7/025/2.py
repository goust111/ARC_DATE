from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _settle_water(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape
    for c in range(w):
        rows = sorted((int(r) for r in np.flatnonzero(g[:, c] == BLUE)), reverse=True)
        if not rows:
            continue
        g[rows, c] = BLACK
        for start_r in rows:
            r = int(start_r)
            while r + 1 < h and int(g[r + 1, c]) == BLACK:
                r += 1
            g[r, c] = BLUE
    return g


def _is_closed_group(grid: np.ndarray, coords: np.ndarray) -> bool:
    h, w = grid.shape
    cells = {tuple(map(int, pt)) for pt in coords}
    if any(r == 0 or c == 0 or c == w - 1 for r, c in cells):
        return False

    q = deque()
    seen: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if nr == h:
                continue
            if not (0 <= nr < h and 0 <= nc < w):
                return False
            if int(grid[nr, nc]) == BLACK and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))

    while q:
        r, c = q.popleft()
        if r == 0 or c == 0 or c == w - 1:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == BLACK and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return True


def _fill_color_for_area(area: int) -> int:
    if area <= 2:
        return RED
    if area <= 4:
        return GREEN
    return YELLOW


def _group_info(grid: np.ndarray) -> list[dict]:
    info = []
    for coords in _components_of_color(grid, BLUE):
        area = int(coords.shape[0])
        info.append(
            {
                "coords": coords,
                "area": area,
                "closed": _is_closed_group(grid, coords),
                "bbox": (
                    int(coords[:, 0].min()),
                    int(coords[:, 1].min()),
                    int(coords[:, 0].max()),
                    int(coords[:, 1].max()),
                ),
            }
        )
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let blue water pixels fall straight down into gray basin structures, then recolor each enclosed settled group by its area.

    Input structure:
    - The grid is a black background with gray obstacle structures and blue water pixels.
    - Gray pixels are static barriers; blue pixels start above empty space and move only within their own columns.
    - After falling, blue pixels may form several 4-connected groups inside closed or side-open basins.

    Rule:
    - Drop every blue pixel vertically through black cells until it reaches the grid bottom, a gray obstacle, or a previously settled blue pixel in the same column.
    - Partition the settled blue pixels into 4-connected groups.
    - A group is enclosed iff every adjacent black background pocket is trapped away from the top edge and the left/right edges; bottom-edge support is allowed.
    - Recolor enclosed groups by area: 1-2 cells -> red, 3-4 cells -> green, 5+ cells -> yellow. Leave non-enclosed groups blue and keep gray obstacles and black background unchanged.

    Key insight:
    - The final color depends on post-gravity grouping and enclosure, not on the water pixels' starting positions: gravity determines the groups, and each group's settled area determines its recolor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    settled = _settle_water(g)
    out = settled.copy()
    for comp in _group_info(settled):
        if not comp["closed"]:
            continue
        color = _fill_color_for_area(comp["area"])
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _random_connected_heights(area: int, width: int, max_height: int) -> list[int] | None:
    candidates: list[list[int]] = []

    def rec(idx: int, left: int, cur: list[int]) -> None:
        if idx == width:
            if left == 0 and any(h > 0 for h in cur):
                intervals = []
                for h in cur:
                    if h <= 0:
                        return
                    intervals.append((0, h - 1))
                for a, b in zip(intervals, intervals[1:]):
                    if max(a[0], b[0]) > min(a[1], b[1]):
                        return
                candidates.append(cur.copy())
            return
        remain = width - idx - 1
        lo = 1
        hi = min(max_height, left - remain)
        for h in range(lo, hi + 1):
            if idx > 0 and abs(h - cur[-1]) > 2:
                continue
            cur.append(h)
            rec(idx + 1, left - h, cur)
            cur.pop()

    rec(0, int(area), [])
    if not candidates:
        return None
    return random.choice(candidates)


def _sample_basin_shape(area: int, force_irregular: bool) -> tuple[list[int], list[int]] | None:
    for _ in range(120):
        width = random.randint(1 if area <= 2 else 2, min(4, area))
        max_height = min(5, area - width + 1)
        heights = _random_connected_heights(area, width, max_height)
        if heights is None:
            continue
        cap = max(heights) + random.randint(1, 2)
        steps = [0] * width
        if width >= 2 and random.random() < 0.75:
            idx = random.randrange(width)
            steps[idx] = random.randint(0, 1)
            if width >= 3 and random.random() < 0.4:
                idx2 = random.randrange(width)
                steps[idx2] = random.randint(0, 1)
        if force_irregular and len(set(heights)) == 1 and max(steps) == 0:
            continue
        if max(h + s for h, s in zip(heights, steps)) > cap:
            continue
        bands = []
        ok = True
        for h, s in zip(heights, steps):
            bands.append((s, s + h - 1))
        for a, b in zip(bands, bands[1:]):
            if max(a[0], b[0]) > min(a[1], b[1]):
                ok = False
                break
        if ok:
            return heights, steps
    return None


def _draw_closed_box(grid: np.ndarray, top: int, left: int, cavity_h: int, width: int) -> None:
    grid[top, left : left + width + 2] = GREY
    grid[top + cavity_h + 1, left : left + width + 2] = GREY
    grid[top : top + cavity_h + 2, left] = GREY
    grid[top : top + cavity_h + 2, left + width + 1] = GREY


def _place_basin(
    grid: np.ndarray,
    *,
    left: int,
    top: int,
    heights: list[int],
    steps: list[int],
    open_to_edge: bool,
    open_left: bool,
) -> dict | None:
    h, w = grid.shape
    width = len(heights)
    cavity_h = max(hh + ss for hh, ss in zip(heights, steps)) + random.randint(1, 2)
    bottom = top + cavity_h + 1
    right = left + width + 1
    if top < 1 or left < 1 or bottom >= h - 1 or right >= w - 1:
        return None

    region = grid[top : bottom + 1, left : right + 1]
    if np.any(region != BLACK):
        return None

    _draw_closed_box(grid, top, left, cavity_h, width)

    for idx, step in enumerate(steps):
        if step <= 0:
            continue
        c = left + 1 + idx
        r0 = bottom - step
        grid[r0:bottom, c] = GREY

    if open_to_edge:
        row_interior = bottom - random.randint(1, max(1, cavity_h - 1))
        if open_left:
            grid[row_interior, left] = BLACK
            grid[row_interior, :left] = BLACK
        else:
            grid[row_interior, right] = BLACK
            grid[row_interior, right + 1 :] = BLACK

    water_cells: list[tuple[int, int]] = []
    moved = False
    for idx, count in enumerate(heights):
        c = left + 1 + idx
        support = bottom - 1 - steps[idx]
        final_rows = list(range(support - count + 1, support + 1))
        available_rows = list(range(top + 1, support + 1))
        if len(available_rows) < count:
            return None
        slack = len(available_rows) - count
        if slack > 0:
            moved = True
        if random.random() < 0.5:
            chosen_rows = available_rows[:count]
        else:
            start = random.randint(0, slack)
            chosen_rows = available_rows[start : start + count]
        for r in chosen_rows:
            water_cells.append((r, c))
        if chosen_rows != final_rows:
            moved = True

    if not moved:
        return None

    for r, c in water_cells:
        if int(grid[r, c]) != BLACK:
            return None
        grid[r, c] = BLUE

    return {
        "kind": "open" if open_to_edge else "closed",
        "water_area": sum(heights),
        "bbox": (top, left, bottom, right),
    }


def _class_for_area(area: int) -> str:
    if area <= 2:
        return "small"
    if area <= 4:
        return "medium"
    return "large"


def _generate_layout() -> np.ndarray | None:
    basin_count = 3 if random.random() < 0.65 else 4
    closed_classes = random.sample(["small", "medium", "large"], 2)
    closed_areas = []
    for cls in closed_classes:
        if cls == "small":
            closed_areas.append(random.randint(1, 2))
        elif cls == "medium":
            closed_areas.append(random.randint(3, 4))
        else:
            closed_areas.append(random.randint(5, 8))

    open_area = random.randint(2, 7)
    basin_plan = [{"kind": "closed", "area": a} for a in closed_areas]
    basin_plan.append({"kind": "open", "area": open_area})
    if basin_count == 4:
        extra_kind = "closed" if random.random() < 0.55 else "open"
        extra_area = random.randint(1, 7)
        basin_plan.append({"kind": extra_kind, "area": extra_area})
    random.shuffle(basin_plan)

    basin_shapes = []
    total_slot = 0
    for idx, spec in enumerate(basin_plan):
        shape = _sample_basin_shape(spec["area"], force_irregular=idx == 0 or random.random() < 0.6)
        if shape is None:
            return None
        heights, steps = shape
        width = len(heights)
        top_pad = random.randint(0, 2)
        cavity_h = max(h + s for h, s in zip(heights, steps)) + 2 + top_pad
        slot_w = width + 4
        basin_shapes.append(
            {
                "kind": spec["kind"],
                "area": spec["area"],
                "heights": heights,
                "steps": steps,
                "slot_w": slot_w,
                "cavity_h": cavity_h,
            }
        )
        total_slot += slot_w

    gap_count = len(basin_shapes) + 1
    min_w = total_slot + gap_count
    if min_w > 28:
        return None
    w = random.randint(max(14, min_w), min(30, min_w + 4))
    h = random.randint(max(12, max(s["cavity_h"] for s in basin_shapes) + 5), 19)
    grid = np.zeros((h, w), dtype=int)

    remaining = w - total_slot
    gaps = [1] * gap_count
    remaining -= gap_count
    for _ in range(remaining):
        gaps[random.randrange(gap_count)] += 1

    x = gaps[0]
    open_specs = [i for i, spec in enumerate(basin_shapes) if spec["kind"] == "open"]
    if not open_specs:
        return None
    side_pref = {}
    if open_specs:
        first_open = random.choice(open_specs)
        side_pref[first_open] = random.choice([True, False])

    metadata = []
    for idx, spec in enumerate(basin_shapes):
        width = len(spec["heights"])
        left = x + 1
        usable_top_max = h - (max(hh + ss for hh, ss in zip(spec["heights"], spec["steps"])) + 4)
        top = random.randint(2, max(2, min(usable_top_max, h // 2)))
        open_to_edge = spec["kind"] == "open"
        if open_to_edge:
            open_left = side_pref.get(idx, left < (w - (left + width + 2)))
        else:
            open_left = False
        placed = _place_basin(
            grid,
            left=left,
            top=top,
            heights=spec["heights"],
            steps=spec["steps"],
            open_to_edge=open_to_edge,
            open_left=open_left,
        )
        if placed is None:
            return None
        metadata.append(placed)
        x += spec["slot_w"] + gaps[idx + 1]

    return grid


def _semantic_ok(grid: np.ndarray) -> bool:
    inp = np.asarray(grid, dtype=int)
    settled = _settle_water(inp)
    groups = _group_info(settled)
    if len(groups) < 3:
        return False

    closed = [g for g in groups if g["closed"]]
    open_groups = [g for g in groups if not g["closed"]]
    if len(closed) < 2 or len(open_groups) < 1:
        return False

    closed_classes = {_class_for_area(g["area"]) for g in closed}
    if len(closed_classes) < 2:
        return False

    out = main(inp)
    if np.array_equal(out, inp) or np.array_equal(out, settled):
        return False

    out_colors = set(map(int, np.unique(out)))
    if BLUE not in out_colors:
        return False
    if not ({RED, GREEN, YELLOW} & out_colors):
        return False

    moved = False
    input_blues = set(map(tuple, np.argwhere(inp == BLUE)))
    settled_blues = set(map(tuple, np.argwhere(settled == BLUE)))
    if input_blues != settled_blues:
        moved = True
    if not moved:
        return False

    if np.any(inp[0] == BLUE):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_layout()
        if grid is not None and _semantic_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
