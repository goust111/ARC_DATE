import itertools
import random
from collections import Counter

import numpy as np


PALETTE = list(range(10))
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude=None) -> list[int]:
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _extract_controls(grid: np.ndarray, bg: int) -> list[tuple[str, int, int, int]]:
    h, w = grid.shape
    controls = []
    for c in range(1, w - 1):
        if int(grid[0, c]) != bg:
            controls.append(("top", 0, c, int(grid[0, c])))
        if int(grid[h - 1, c]) != bg:
            controls.append(("bottom", h - 1, c, int(grid[h - 1, c])))
    for r in range(1, h - 1):
        if int(grid[r, 0]) != bg:
            controls.append(("left", r, 0, int(grid[r, 0])))
        if int(grid[r, w - 1]) != bg:
            controls.append(("right", r, w - 1, int(grid[r, w - 1])))
    return controls


def _obstacle_mask(grid: np.ndarray, bg: int, controls: list[tuple[str, int, int, int]]) -> np.ndarray:
    border_colors = {color for _side, _r, _c, color in controls}
    interior = np.asarray(grid)[1:-1, 1:-1]
    obstacle_colors = [color for color in _colors_present(interior, exclude={bg}) if color not in border_colors]
    assert len(obstacle_colors) <= 1, f"expected at most one obstacle color, got {obstacle_colors}"
    if not obstacle_colors:
        return np.zeros_like(grid, dtype=bool)
    return np.asarray(grid) == obstacle_colors[0]


def _trace_ray(
    shape: tuple[int, int],
    obstacle: np.ndarray,
    control: tuple[str, int, int, int],
) -> tuple[list[tuple[int, int]], bool]:
    h, w = shape
    side, r, c, _color = control
    cells = []
    if side == "left":
        col = 1
        while col < w and not obstacle[r, col]:
            cells.append((r, col))
            col += 1
        return cells, col < w
    if side == "right":
        col = w - 2
        while col >= 0 and not obstacle[r, col]:
            cells.append((r, col))
            col -= 1
        return cells, col >= 0
    if side == "top":
        row = 1
        while row < h and not obstacle[row, c]:
            cells.append((row, c))
            row += 1
        return cells, row < h
    if side == "bottom":
        row = h - 2
        while row >= 0 and not obstacle[row, c]:
            cells.append((row, c))
            row -= 1
        return cells, row >= 0
    raise AssertionError(f"unknown control side: {side}")


def _band_maps(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    controls = _extract_controls(g, bg)
    obstacle = _obstacle_mask(g, bg, controls)
    hmap = np.full(g.shape, -1, dtype=int)
    vmap = np.full(g.shape, -1, dtype=int)
    blocked = []
    open_rays = []
    for control in controls:
        cells, hit_obstacle = _trace_ray(g.shape, obstacle, control)
        side, _r, _c, color = control
        target = hmap if side in ("left", "right") else vmap
        for rr, cc in cells:
            prev = int(target[rr, cc])
            assert prev in (-1, color), "same-direction bands must not overlap"
            target[rr, cc] = color
        (blocked if hit_obstacle else open_rays).append((control, cells))
    return bg, controls, obstacle, hmap, vmap, blocked, open_rays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend border control pixels into inward color bands and resolve crossings by global color rarity.

    Input structure:
    - The background is the dominant color.
    - Non-background single control pixels lie on the four outer edges, excluding corners.
    - One interior obstacle color forms one or more solid blocker rectangles that never touch the border.

    Rule:
    - Each control pixel emits a 1-cell-wide band in the inward direction until the cell before the first obstacle; if no obstacle appears, the band reaches the opposite edge.
    - Horizontal and vertical bands are computed independently, while the original control pixels and obstacle cells stay unchanged.
    - At a horizontal/vertical crossing, paint the color whose total input-pixel count is smaller; the task guarantees no tie.

    Key insight:
    - Crossing outcomes do not depend on draw order: they are determined entirely by the full-grid input color frequencies of the competing band colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    counts = Counter(map(int, g.ravel()))
    bg, controls, obstacle, hmap, vmap, _blocked, _open_rays = _band_maps(g)
    control_mask = np.zeros_like(g, dtype=bool)
    for _side, r, c, _color in controls:
        control_mask[r, c] = True
    fixed = control_mask | obstacle

    out = g.copy()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if fixed[r, c]:
                continue
            h_color = int(hmap[r, c])
            v_color = int(vmap[r, c])
            if h_color == -1 and v_color == -1:
                continue
            if h_color == -1:
                out[r, c] = v_color
            elif v_color == -1 or h_color == v_color:
                out[r, c] = h_color
            else:
                assert counts[h_color] != counts[v_color], "crossing colors must have distinct input frequencies"
                out[r, c] = h_color if counts[h_color] < counts[v_color] else v_color
    return out


def _distinct_partitions(total: int, parts: int) -> list[tuple[int, ...]]:
    return [combo for combo in itertools.combinations(range(1, total + 1), parts) if sum(combo) == total]


def _place_obstacles(grid: np.ndarray, obstacle_color: int) -> list[tuple[int, int, int, int]] | None:
    h, w = grid.shape
    rects = []
    target_count = random.randint(1, 3)
    for _ in range(120):
        if len(rects) == target_count:
            break
        rect_h = random.randint(2, min(5, h - 5))
        rect_w = random.randint(2, min(6, w - 5))
        r0 = random.randint(2, h - rect_h - 3)
        c0 = random.randint(2, w - rect_w - 3)
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1
        clear = True
        for or0, oc0, or1, oc1 in rects:
            if not (r1 + 1 < or0 or or1 + 1 < r0 or c1 + 1 < oc0 or oc1 + 1 < c0):
                clear = False
                break
        if not clear:
            continue
        rects.append((r0, c0, r1, c1))
    if len(rects) != target_count:
        return None
    for r0, c0, r1, c1 in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = obstacle_color
    return rects


def _blocked_positions(rects, size: int, axis: str) -> tuple[list[int], list[int]]:
    blocked = set()
    for r0, c0, r1, c1 in rects:
        if axis == "row":
            blocked.update(range(r0, r1 + 1))
        else:
            blocked.update(range(c0, c1 + 1))
    all_positions = set(range(1, size - 1))
    open_positions = sorted(all_positions - blocked)
    return sorted(blocked), open_positions


def _choose_lines(blocked_positions: list[int], open_positions: list[int], total: int):
    if len(blocked_positions) < 1 or len(open_positions) < total - 1:
        return None
    blocked_count = 1
    if len(blocked_positions) >= 2 and total >= 4 and random.random() < 0.5:
        blocked_count = 2
    blocked_count = min(blocked_count, len(blocked_positions), total - 1)
    open_count = total - blocked_count
    if len(open_positions) < open_count:
        return None
    chosen = [(pos, True) for pos in random.sample(blocked_positions, blocked_count)]
    chosen.extend((pos, False) for pos in random.sample(open_positions, open_count))
    random.shuffle(chosen)
    return chosen


def _assign_colors(slots: list[dict], colors: list[int], counts: list[int]) -> bool:
    assignments = [None] * len(slots)
    color_pool = []
    for color, count in zip(colors, counts):
        color_pool.extend([color] * count)

    open_h = next((i for i, slot in enumerate(slots) if slot["axis"] == "h" and not slot["blocked"]), None)
    open_v = next((i for i, slot in enumerate(slots) if slot["axis"] == "v" and not slot["blocked"]), None)
    if open_h is None or open_v is None:
        return False

    c1, c2 = random.sample(colors, 2)
    assignments[open_h] = c1
    assignments[open_v] = c2
    color_pool.remove(c1)
    color_pool.remove(c2)
    random.shuffle(color_pool)
    for idx, value in enumerate(assignments):
        if value is None:
            assignments[idx] = color_pool.pop()
    for slot, color in zip(slots, assignments):
        slot["color"] = color
    return True


def _build_candidate() -> np.ndarray | None:
    h = random.randint(12, 22)
    w = random.randint(12, 22)
    palette = random.sample(PALETTE, len(PALETTE))
    bg = palette[0]
    obstacle_color = palette[1]
    band_colors = palette[2:5]

    grid = np.full((h, w), bg, dtype=int)
    rects = _place_obstacles(grid, obstacle_color)
    if rects is None:
        return None

    blocked_rows, open_rows = _blocked_positions(rects, h, "row")
    blocked_cols, open_cols = _blocked_positions(rects, w, "col")
    if len(open_rows) < 2 or len(open_cols) < 2:
        return None

    horizontal_total = random.randint(3, 4)
    vertical_total = random.randint(3, 4)
    rows = _choose_lines(blocked_rows, open_rows, horizontal_total)
    cols = _choose_lines(blocked_cols, open_cols, vertical_total)
    if rows is None or cols is None:
        return None

    total_controls = horizontal_total + vertical_total
    partitions = _distinct_partitions(total_controls, 3)
    if not partitions:
        return None
    counts = list(random.choice(partitions))
    random.shuffle(counts)

    slots = []
    for row, blocked in rows:
        slots.append(
            {
                "axis": "h",
                "pos": row,
                "blocked": blocked,
                "side": random.choice(["left", "right"]),
            }
        )
    for col, blocked in cols:
        slots.append(
            {
                "axis": "v",
                "pos": col,
                "blocked": blocked,
                "side": random.choice(["top", "bottom"]),
            }
        )
    if not _assign_colors(slots, band_colors, counts):
        return None

    for slot in slots:
        color = slot["color"]
        if slot["axis"] == "h":
            row = slot["pos"]
            if slot["side"] == "left":
                grid[row, 0] = color
            else:
                grid[row, w - 1] = color
        else:
            col = slot["pos"]
            if slot["side"] == "top":
                grid[0, col] = color
            else:
                grid[h - 1, col] = color
    return grid


def _generated_grid_is_valid(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    bg, controls, obstacle, hmap, vmap, blocked, open_rays = _band_maps(g)
    if np.count_nonzero(obstacle) == 0:
        return False
    if not blocked or not open_rays:
        return False
    if len(controls) < 6:
        return False

    counts = Counter(map(int, g.ravel()))
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return False

    crossing_found = False
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            hc = int(hmap[r, c])
            vc = int(vmap[r, c])
            if hc != -1 and vc != -1 and hc != vc:
                if counts[hc] == counts[vc]:
                    return False
                crossing_found = True
    if not crossing_found:
        return False

    out = main(g)
    return bool(np.any(out != g))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None and _generated_grid_is_valid(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
