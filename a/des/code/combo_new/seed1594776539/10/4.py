from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SHADOW = 5
DIRS = ("up", "down", "left", "right")
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

# Generator tuning.
GRID_SIZE_RANGE = (19, 25)
OBJECT_COUNT_RANGE = (2, 4)
GENERATION_ATTEMPTS = 500
EDGE_BAND = 5
LARGE_OBJECT_PROB = 0.35


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _hole_direction(bbox: tuple[int, int, int, int], hole_r: int, hole_c: int) -> str:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    if hole_r < center_r:
        return "up"
    if hole_r > center_r:
        return "down"
    if hole_c < center_c:
        return "left"
    return "right"


def _shadow_bounds(
    bbox: tuple[int, int, int, int], area: int, direction: str, h: int, w: int
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = bbox
    if direction == "up":
        sr0, sr1 = max(0, r0 - area), r0
        sc0, sc1 = c0, c1 + 1
    elif direction == "down":
        sr0, sr1 = r1 + 1, min(h, r1 + 1 + area)
        sc0, sc1 = c0, c1 + 1
    elif direction == "left":
        sr0, sr1 = r0, r1 + 1
        sc0, sc1 = max(0, c0 - area), c0
    else:
        sr0, sr1 = r0, r1 + 1
        sc0, sc1 = c1 + 1, min(w, c1 + 1 + area)

    if sr0 >= sr1 or sc0 >= sc1:
        return None
    return sr0, sr1, sc0, sc1


def _rect_cells(r0: int, r1: int, c0: int, c1: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def _shadow_cells(
    bbox: tuple[int, int, int, int], area: int, direction: str, h: int, w: int
) -> set[tuple[int, int]]:
    bounds = _shadow_bounds(bbox, area, direction, h, w)
    if bounds is None:
        return set()
    r0, r1, c0, c1 = bounds
    return _rect_cells(r0, r1, c0, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project a grey shadow from each perforated object in the direction encoded by its unique hole.

    Input structure:
    - The background is black.
    - Several non-overlapping monochrome 4-connected foreground objects of distinct colors appear on the grid.
    - Each object's bounding box is filled except for exactly one black hole cell inside that box.

    Rule:
    - For each object, count its foreground pixels to get N.
    - Determine the hole's direction relative to the bounding-box center: row offset decides up/down first, otherwise column offset decides left/right.
    - From the object's outer bbox edge on that side, paint a grey rectangular strip of length N and width equal to the bbox span perpendicular to the direction, truncating at the grid boundary.
    - Keep the original objects unchanged.

    Key insight:
    - The hole is not an absence to fill; it is a directional arrow, while the object's area determines how far the shadow extends.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for comp in _find_components(g):
        bbox = comp["bbox"]
        r0, c0, r1, c1 = bbox
        crop = g[r0 : r1 + 1, c0 : c1 + 1]
        holes = np.argwhere(crop == BG)
        assert holes.shape[0] == 1, "each object bbox must contain exactly one black hole"
        hole_r = r0 + int(holes[0, 0])
        hole_c = c0 + int(holes[0, 1])
        direction = _hole_direction(bbox, hole_r, hole_c)
        bounds = _shadow_bounds(bbox, int(comp["area"]), direction, h, w)
        if bounds is None:
            continue
        sr0, sr1, sc0, sc1 = bounds
        out[sr0:sr1, sc0:sc1] = SHADOW

    return out


def _sample_size(direction: str, large: bool) -> tuple[int, int]:
    if large:
        return 5, 5
    if direction in ("up", "down"):
        return 5, 3
    return 3, 5


def _hole_choices(height: int, width: int, direction: str) -> list[tuple[int, int]]:
    center_r = height // 2
    center_c = width // 2
    choices: list[tuple[int, int]] = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if direction == "up" and r < center_r:
                choices.append((r, c))
            elif direction == "down" and r > center_r:
                choices.append((r, c))
            elif direction == "left" and r == center_r and c < center_c:
                choices.append((r, c))
            elif direction == "right" and r == center_r and c > center_c:
                choices.append((r, c))
    return choices


def _bbox_buffer_cells(top: int, left: int, height: int, width: int, h: int, w: int) -> set[tuple[int, int]]:
    r0 = max(0, top - 1)
    r1 = min(h, top + height + 1)
    c0 = max(0, left - 1)
    c1 = min(w, left + width + 1)
    return _rect_cells(r0, r1, c0, c1)


def _visible_shadow_length(top: int, left: int, height: int, width: int, direction: str, h: int, w: int) -> int:
    if direction == "up":
        return min(height * width - 1, top)
    if direction == "down":
        return min(height * width - 1, h - (top + height))
    if direction == "left":
        return min(height * width - 1, left)
    return min(height * width - 1, w - (left + width))


def _preferred_position(top: int, left: int, height: int, width: int, direction: str, h: int, w: int) -> bool:
    if direction == "up":
        return top >= max(0, h - height - EDGE_BAND)
    if direction == "down":
        return top <= EDGE_BAND
    if direction == "left":
        return left >= max(0, w - width - EDGE_BAND)
    return left <= EDGE_BAND


def _candidate_positions(
    height: int,
    width: int,
    direction: str,
    h: int,
    w: int,
    blocked_bbox_cells: set[tuple[int, int]],
    occupied_bbox_cells: set[tuple[int, int]],
) -> list[tuple[int, int, set[tuple[int, int]], set[tuple[int, int]]]]:
    area = height * width - 1
    min_visible = min(area, max(6, area // 2))
    preferred: list[tuple[int, int, set[tuple[int, int]], set[tuple[int, int]]]] = []
    fallback: list[tuple[int, int, set[tuple[int, int]], set[tuple[int, int]]]] = []

    for top in range(h - height + 1):
        for left in range(w - width + 1):
            if _visible_shadow_length(top, left, height, width, direction, h, w) < min_visible:
                continue
            bbox_cells = _rect_cells(top, top + height, left, left + width)
            if bbox_cells & blocked_bbox_cells:
                continue
            bbox = (top, left, top + height - 1, left + width - 1)
            shadow = _shadow_cells(bbox, area, direction, h, w)
            if shadow & occupied_bbox_cells:
                continue
            record = (top, left, bbox_cells, shadow)
            if _preferred_position(top, left, height, width, direction, h, w):
                preferred.append(record)
            else:
                fallback.append(record)

    return preferred if preferred else fallback


def _build_spec_list(grid_h: int, grid_w: int, object_count: int) -> list[dict]:
    directions = random.sample(DIRS, object_count)
    large_index = None
    if object_count <= 3 and min(grid_h, grid_w) >= 21 and random.random() < LARGE_OBJECT_PROB:
        large_index = random.randrange(object_count)

    specs = []
    for idx, direction in enumerate(directions):
        height, width = _sample_size(direction, large=idx == large_index)
        hole = random.choice(_hole_choices(height, width, direction))
        specs.append(
            {
                "direction": direction,
                "height": height,
                "width": width,
                "hole": hole,
                "area": height * width - 1,
            }
        )
    specs.sort(key=lambda spec: (-int(spec["area"]), spec["direction"]))
    return specs


def _render_grid(specs: list[dict], grid_h: int, grid_w: int) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    colors = random.sample(COLOR_POOL, len(specs))
    for color, spec in zip(colors, specs):
        top = int(spec["top"])
        left = int(spec["left"])
        height = int(spec["height"])
        width = int(spec["width"])
        patch = np.full((height, width), int(color), dtype=int)
        hole_r, hole_c = spec["hole"]
        patch[int(hole_r), int(hole_c)] = BG
        grid[top : top + height, left : left + width] = patch
    return grid


def _valid_generated_instance(grid: np.ndarray) -> bool:
    try:
        comps = _find_components(grid)
        if not (OBJECT_COUNT_RANGE[0] <= len(comps) <= OBJECT_COUNT_RANGE[1]):
            return False
        if len({int(comp["color"]) for comp in comps}) != len(comps):
            return False

        out = main(grid)
        if np.any(out[grid != BG] != grid[grid != BG]):
            return False
        if not np.any(out == SHADOW):
            return False

        for comp in comps:
            r0, c0, r1, c1 = comp["bbox"]
            crop = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.count_nonzero(crop == BG) != 1:
                return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_SIZE_RANGE)
    grid_w = random.randint(*GRID_SIZE_RANGE)
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    specs = _build_spec_list(grid_h, grid_w, object_count)

    blocked_bbox_cells: set[tuple[int, int]] = set()
    occupied_bbox_cells: set[tuple[int, int]] = set()
    placed_specs: list[dict] = []

    for spec in specs:
        positions = _candidate_positions(
            int(spec["height"]),
            int(spec["width"]),
            str(spec["direction"]),
            grid_h,
            grid_w,
            blocked_bbox_cells,
            occupied_bbox_cells,
        )
        if not positions:
            return None

        top, left, bbox_cells, shadow_cells = random.choice(positions)
        placed = dict(spec)
        placed["top"] = int(top)
        placed["left"] = int(left)
        placed_specs.append(placed)

        blocked_bbox_cells |= _bbox_buffer_cells(top, left, int(spec["height"]), int(spec["width"]), grid_h, grid_w)
        blocked_bbox_cells |= shadow_cells
        occupied_bbox_cells |= bbox_cells

    grid = _render_grid(placed_specs, grid_h, grid_w)
    if not _valid_generated_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-projection ARC input")
