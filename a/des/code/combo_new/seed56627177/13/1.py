from __future__ import annotations

from itertools import product
import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_MIN = 18
GRID_MAX = 30
BOX_MIN = 4
BOX_MAX = 8
PLACEMENT_RETRIES = 200
GEN_RETRIES = 500


def _mode_color(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _edge_color_map(grid: np.ndarray) -> dict[str, int]:
    h, w = grid.shape
    top = grid[0, 1:-1] if w > 2 else grid[0]
    bottom = grid[h - 1, 1:-1] if w > 2 else grid[h - 1]
    left = grid[1:-1, 0] if h > 2 else grid[:, 0]
    right = grid[1:-1, w - 1] if h > 2 else grid[:, w - 1]
    return {
        "top": _mode_color(top),
        "bottom": _mode_color(bottom),
        "left": _mode_color(left),
        "right": _mode_color(right),
    }


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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords.extend((r1, c) for c in range(c0, c1 + 1))
    coords.extend((r, c0) for r in range(r0 + 1, r1))
    coords.extend((r, c1) for r in range(r0 + 1, r1))
    return coords


def _side_coords(r0: int, c0: int, r1: int, c1: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _find_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    rects: list[dict[str, object]] = []
    for pts in _components(grid == FRAME):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r0 <= 0 or c0 <= 0 or r1 >= h - 1 or c1 >= w - 1:
            continue
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue

        border = set(_border_coords(r0, c0, r1, c1))
        comp = set(pts)
        if not comp <= border:
            continue

        missing = list(border - comp)
        if len(missing) != 1:
            continue
        gap = missing[0]
        if gap in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            continue
        if int(grid[gap]) != BG:
            continue

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or np.any(inner == FRAME):
            continue
        marker_locs = np.argwhere((inner != BG) & (inner != FRAME))
        if marker_locs.shape[0] != 1:
            continue
        mr, mc = map(int, marker_locs[0])
        marker = (r0 + 1 + mr, c0 + 1 + mc)
        color = int(grid[marker])

        rects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "marker": marker,
                "color": color,
            }
        )

    rects.sort(key=lambda rect: tuple(rect["bbox"]))  # type: ignore[arg-type]
    return rects


def _trace_ray(grid: np.ndarray, gap: tuple[int, int], direction: tuple[int, int]) -> tuple[list[tuple[int, int]], bool]:
    h, w = grid.shape
    dr, dc = direction
    rr, cc = gap
    cells: list[tuple[int, int]] = []
    blocked = False
    while True:
        rr += dr
        cc += dc
        if rr <= 0 or rr >= h - 1 or cc <= 0 or cc >= w - 1:
            break
        value = int(grid[rr, cc])
        if value == FRAME:
            blocked = True
            break
        if value != BG:
            break
        cells.append((rr, cc))
    return cells, blocked


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    side_colors = _edge_color_map(g)
    if len(set(side_colors.values())) != 4:
        raise AssertionError("expected four distinct outer boundary colors")

    rects = _find_rectangles(g)
    marker_colors = {int(rect["color"]) for rect in rects}
    if marker_colors != set(side_colors.values()):
        raise AssertionError("marker colors must match the four boundary-strip colors")

    color_to_side = {color: side for side, color in side_colors.items()}
    out = g.copy()
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]  # type: ignore[misc]
        color = int(rect["color"])
        gap = rect["gap"]  # type: ignore[assignment]
        out[r0 + 1 : r1, c0 + 1 : c1] = color
        ray_cells, _ = _trace_ray(g, gap, DIRS[color_to_side[color]])
        for rr, cc in ray_cells:
            out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each framed marker color to the same-colored outer edge, fill the frame interior, and emit an outward ray from the frame gap.

    Input structure:
    - The background is black.
    - Several grey one-pixel-thick rectangular frames appear inside the grid; each frame has exactly one one-cell gap on its border.
    - Each frame encloses exactly one non-grey, non-black marker pixel, and the marker colors are all distinct.
    - The four outer grid edges carry four distinct one-pixel-thick color strips whose color set matches the marker-color set.

    Rule:
    - For each grey frame, read its unique interior marker color and find the outer grid edge with the same color.
    - Use that matched edge to determine a cardinal direction relative to the frame: top, bottom, left, or right.
    - Fill the entire frame interior with the marker color.
    - Starting from the frame's gap, extend a same-colored straight ray toward the matched outer edge, stopping before the outer edge strip or before any other grey frame.

    Key insight:
    - The marker color is a legend key into the four outer boundary strips, while the gap provides only the ray origin; solving needs both cues together.
    """
    return _solve(grid)


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _draw_edge_strips(grid: np.ndarray, side_colors: dict[str, int]) -> None:
    h, w = grid.shape
    grid[0, :] = side_colors["top"]
    grid[h - 1, :] = side_colors["bottom"]
    grid[1:-1, 0] = side_colors["left"]
    grid[1:-1, w - 1] = side_colors["right"]


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _ray_avoids_boxes(
    ray_cells: list[tuple[int, int]], rects: list[dict[str, object]], owner_idx: int
) -> bool:
    for rr, cc in ray_cells:
        for idx, rect in enumerate(rects):
            if idx == owner_idx:
                continue
            r0, c0, r1, c1 = rect["bbox"]  # type: ignore[misc]
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                return False
    return True


def _choose_gaps(
    base_grid: np.ndarray, rects: list[dict[str, object]], need_blocked: bool, rng: random.Random
) -> np.ndarray | None:
    gap_lists = []
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]  # type: ignore[misc]
        gap_lists.append(_side_coords(r0, c0, r1, c1, rect["side"]))  # type: ignore[arg-type]

    valid: list[tuple[tuple[tuple[int, int], ...], list[bool]]] = []
    products = list(product(*gap_lists))
    rng.shuffle(products)
    for gaps in products:
        grid = base_grid.copy()
        for gap in gaps:
            grid[gap] = BG

        blocked_flags: list[bool] = []
        ok = True
        for idx, (rect, gap) in enumerate(zip(rects, gaps)):
            ray_cells, blocked = _trace_ray(grid, gap, DIRS[rect["side"]])  # type: ignore[index]
            if not ray_cells or not _ray_avoids_boxes(ray_cells, rects, idx):
                ok = False
                break
            blocked_flags.append(blocked)

        if ok and ((need_blocked and any(blocked_flags)) or ((not need_blocked) and not any(blocked_flags))):
            valid.append((gaps, blocked_flags))

    if not valid:
        return None

    gaps, _ = rng.choice(valid)
    out = base_grid.copy()
    for rect, gap in zip(rects, gaps):
        rect["gap"] = gap
        out[gap] = BG
    return out


def _place_rectangles(
    h: int, w: int, side_colors: dict[str, int], rng: random.Random
) -> list[dict[str, object]] | None:
    rects: list[dict[str, object]] = []
    order = list(SIDES)
    rng.shuffle(order)
    for side in order:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            box_h = rng.randint(BOX_MIN, min(BOX_MAX, h - 4))
            box_w = rng.randint(BOX_MIN, min(BOX_MAX, w - 4))
            r0 = rng.randint(2, h - box_h - 2)
            c0 = rng.randint(2, w - box_w - 2)
            bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if any(_boxes_too_close(bbox, other["bbox"], pad=1) for other in rects):
                continue
            marker = (rng.randint(r0 + 1, r0 + box_h - 2), rng.randint(c0 + 1, c0 + box_w - 2))
            rects.append(
                {
                    "bbox": bbox,
                    "side": side,
                    "color": side_colors[side],
                    "marker": marker,
                }
            )
            placed = True
            break
        if not placed:
            return None
    return rects


def _blocked_count(grid: np.ndarray, rects: list[dict[str, object]], side_colors: dict[str, int]) -> int:
    color_to_side = {color: side for side, color in side_colors.items()}
    count = 0
    for rect in _find_rectangles(grid):
        _, blocked = _trace_ray(grid, rect["gap"], DIRS[color_to_side[int(rect["color"])]])  # type: ignore[index]
        count += int(blocked)
    return count


def _validate_input(grid: np.ndarray, need_blocked: bool) -> bool:
    rects = _find_rectangles(grid)
    if len(rects) != 4:
        return False
    side_colors = _edge_color_map(grid)
    if set(side_colors.values()) != {int(rect["color"]) for rect in rects}:
        return False
    blocked = _blocked_count(grid, rects, side_colors)
    if need_blocked and blocked == 0:
        return False
    if (not need_blocked) and blocked != 0:
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        h = rng.randint(GRID_MIN, GRID_MAX)
        w = rng.randint(GRID_MIN, GRID_MAX)
        need_blocked = rng.random() < 0.5
        side_colors = dict(zip(SIDES, rng.sample(PALETTE, len(SIDES))))

        rects = _place_rectangles(h, w, side_colors, rng)
        if rects is None:
            continue

        grid = np.full((h, w), BG, dtype=int)
        _draw_edge_strips(grid, side_colors)
        for rect in rects:
            _draw_frame(grid, rect["bbox"])  # type: ignore[arg-type]
            mr, mc = rect["marker"]  # type: ignore[misc]
            grid[mr, mc] = int(rect["color"])

        grid = _choose_gaps(grid, rects, need_blocked, rng)
        if grid is None:
            continue
        if _validate_input(grid, need_blocked):
            return grid
    raise RuntimeError("generate_input: exhausted retries")
