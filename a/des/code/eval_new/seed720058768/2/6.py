from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


ARC_MAX = 30
COLORS = tuple(range(10))
DIRECTIONS = ("up", "down", "left", "right")
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue = deque([(r, c)])
        pts: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            pts.append((cr, cc))
            for nr, nc in neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        components.append(np.array(pts, dtype=int))

    return components


def _u_shape(direction: str, inner_rows: int, inner_cols: int) -> set[tuple[int, int]]:
    if direction == "up":
        height = inner_rows + 1
        width = inner_cols + 2
        pts = {(r, 0) for r in range(height)} | {(r, width - 1) for r in range(height)}
        pts |= {(height - 1, c) for c in range(width)}
        return pts
    if direction == "down":
        height = inner_rows + 1
        width = inner_cols + 2
        pts = {(r, 0) for r in range(height)} | {(r, width - 1) for r in range(height)}
        pts |= {(0, c) for c in range(width)}
        return pts
    if direction == "left":
        height = inner_rows + 2
        width = inner_cols + 1
        pts = {(0, c) for c in range(width)} | {(height - 1, c) for c in range(width)}
        pts |= {(r, width - 1) for r in range(height)}
        return pts
    height = inner_rows + 2
    width = inner_cols + 1
    pts = {(0, c) for c in range(width)} | {(height - 1, c) for c in range(width)}
    pts |= {(r, 0) for r in range(height)}
    return pts


def _container_info(coords: np.ndarray) -> dict[str, int | str] | None:
    r0, c0, r1, c1 = bbox_coords(coords)
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    matches: list[dict[str, int | str]] = []

    if 4 <= height <= 6 and 5 <= width <= 7:
        inner_rows = height - 1
        inner_cols = width - 2
        if rel == _u_shape("up", inner_rows, inner_cols):
            matches.append(
                {
                    "direction": "up",
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "ir0": r0,
                    "ic0": c0 + 1,
                    "ir1": r1 - 1,
                    "ic1": c1 - 1,
                }
            )
        if rel == _u_shape("down", inner_rows, inner_cols):
            matches.append(
                {
                    "direction": "down",
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "ir0": r0 + 1,
                    "ic0": c0 + 1,
                    "ir1": r1,
                    "ic1": c1 - 1,
                }
            )

    if 5 <= height <= 7 and 4 <= width <= 6:
        inner_rows = height - 2
        inner_cols = width - 1
        if rel == _u_shape("left", inner_rows, inner_cols):
            matches.append(
                {
                    "direction": "left",
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "ir0": r0 + 1,
                    "ic0": c0,
                    "ir1": r1 - 1,
                    "ic1": c1 - 1,
                }
            )
        if rel == _u_shape("right", inner_rows, inner_cols):
            matches.append(
                {
                    "direction": "right",
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "ir0": r0 + 1,
                    "ic0": c0 + 1,
                    "ir1": r1 - 1,
                    "ic1": c1,
                }
            )

    if len(matches) != 1:
        return None
    return matches[0]


def _find_containers(grid: np.ndarray, bg: int) -> list[dict[str, int | str]]:
    infos: list[dict[str, int | str]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in components_mask(grid == color):
            info = _container_info(coords)
            if info is None:
                continue
            info["color"] = color
            infos.append(info)
    infos.sort(key=lambda item: (int(item["r0"]), int(item["c0"])))
    return infos


def _bar_colors(grid: np.ndarray, info: dict[str, int | str], bg: int) -> list[int]:
    direction = str(info["direction"])
    ir0 = int(info["ir0"])
    ic0 = int(info["ic0"])
    ir1 = int(info["ir1"])
    ic1 = int(info["ic1"])

    if direction in ("up", "down"):
        bar_r = ir0 - 1 if direction == "up" else ir1 + 1
        return [int(grid[bar_r, c]) for c in range(ic0, ic1 + 1) if int(grid[bar_r, c]) != bg]

    bar_c = ic0 - 1 if direction == "left" else ic1 + 1
    return [int(grid[r, bar_c]) for r in range(ir0, ir1 + 1) if int(grid[r, bar_c]) != bg]


def _interior_colors(grid: np.ndarray, info: dict[str, int | str], bg: int) -> list[int]:
    ir0 = int(info["ir0"])
    ic0 = int(info["ic0"])
    ir1 = int(info["ir1"])
    ic1 = int(info["ic1"])
    colors: list[int] = []
    for r in range(ir0, ir1 + 1):
        for c in range(ic0, ic1 + 1):
            value = int(grid[r, c])
            if value != bg:
                colors.append(value)
    return colors


def _apply_reorder(
    out: np.ndarray, info: dict[str, int | str], ordered_colors: list[int], bg: int
) -> None:
    ir0 = int(info["ir0"])
    ic0 = int(info["ic0"])
    ir1 = int(info["ir1"])
    ic1 = int(info["ic1"])
    direction = str(info["direction"])

    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = bg

    if direction in ("up", "down"):
        center_c = (ic0 + ic1) // 2
        start_r = ir1 if direction == "up" else ir0
        step = -1 if direction == "up" else 1
        for idx, color in enumerate(ordered_colors):
            out[start_r + step * idx, center_c] = color
        return

    center_r = (ir0 + ir1) // 2
    start_c = ic1 if direction == "left" else ic0
    step = -1 if direction == "left" else 1
    for idx, color in enumerate(ordered_colors):
        out[center_r, start_c + step * idx] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the colored pixels inside each U-shaped container by the outside instruction strip and stack them toward the opening.

    Input structure:
    - The most common color is the background.
    - Several monochrome U-shaped containers appear in the grid; each container has a rectangular interior and exactly one open side facing up, down, left, or right.
    - Inside each container are 2-4 colored single pixels, and just outside the opening there is a horizontal or vertical instruction strip listing those colors in order.

    Rule:
    - Detect each container and infer its opening direction from which side of the U is missing.
    - Read the instruction strip in its visible order: left-to-right for horizontal strips, top-to-bottom for vertical strips.
    - Clear the container interior, then place the interior colors along the center line of the interior, starting from the closed side opposite the opening and continuing cell-by-cell toward the opening.
    - Leave the container walls and instruction strip unchanged.

    Key insight:
    - The missing side of the U determines both the stacking axis and which interior edge is the starting base, while the adjacent strip provides the color order.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for info in _find_containers(g, bg):
        order = _bar_colors(g, info, bg)
        interior = _interior_colors(g, info, bg)
        assert len(order) == len(interior) and len(order) > 0
        assert Counter(order) == Counter(interior)
        _apply_reorder(out, info, order, bg)

    return out


def _bg_is_unique_mode(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(values, counts)}
    return freq[bg] > max((count for color, count in freq.items() if color != bg), default=-1)


def _stack_positions(info: dict[str, int | str], colors: list[int]) -> list[tuple[int, int, int]]:
    direction = str(info["direction"])
    ir0 = int(info["ir0"])
    ic0 = int(info["ic0"])
    ir1 = int(info["ir1"])
    ic1 = int(info["ic1"])

    if direction in ("up", "down"):
        center_c = (ic0 + ic1) // 2
        start_r = ir1 if direction == "up" else ir0
        step = -1 if direction == "up" else 1
        return [(start_r + step * idx, center_c, color) for idx, color in enumerate(colors)]

    center_r = (ir0 + ir1) // 2
    start_c = ic1 if direction == "left" else ic0
    step = -1 if direction == "left" else 1
    return [(center_r, start_c + step * idx, color) for idx, color in enumerate(colors)]


def _make_spec(rng: random.Random, direction: str, wall_color: int, content_pool: list[int]) -> dict:
    n_colors = rng.randint(2, 4)
    colors = rng.sample(content_pool, n_colors)

    if direction in ("up", "down"):
        inner_rows = rng.randint(max(3, n_colors), 5)
        inner_cols = rng.choice([size for size in (3, 5) if size >= n_colors])
        bar_offset = rng.randint(0, inner_cols - n_colors)
    else:
        inner_rows = rng.choice([size for size in (3, 5) if size >= n_colors])
        inner_cols = rng.randint(max(3, n_colors), 5)
        bar_offset = rng.randint(0, inner_rows - n_colors)

    interior_cells = [(r, c) for r in range(inner_rows) for c in range(inner_cols)]
    pixel_cells = rng.sample(interior_cells, n_colors)

    return {
        "direction": direction,
        "wall_color": wall_color,
        "inner_rows": inner_rows,
        "inner_cols": inner_cols,
        "colors": colors,
        "bar_offset": bar_offset,
        "pixels": list(zip(pixel_cells, colors)),
        "foot_h": inner_rows + 2,
        "foot_w": inner_cols + 2,
    }


def _placement_candidates(occ: np.ndarray, foot_h: int, foot_w: int) -> list[tuple[int, int]]:
    h, w = occ.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, h - foot_h):
        for c0 in range(1, w - foot_w):
            if occ[r0 - 1 : r0 + foot_h + 1, c0 - 1 : c0 + foot_w + 1].any():
                continue
            candidates.append((r0, c0))
    return candidates


def _placed_info(spec: dict) -> dict[str, int | str]:
    direction = spec["direction"]
    foot_r = spec["foot_r"]
    foot_c = spec["foot_c"]
    inner_rows = spec["inner_rows"]
    inner_cols = spec["inner_cols"]

    if direction == "up":
        return {
            "direction": direction,
            "r0": foot_r + 1,
            "c0": foot_c,
            "r1": foot_r + inner_rows + 1,
            "c1": foot_c + inner_cols + 1,
            "ir0": foot_r + 1,
            "ic0": foot_c + 1,
            "ir1": foot_r + inner_rows,
            "ic1": foot_c + inner_cols,
        }
    if direction == "down":
        return {
            "direction": direction,
            "r0": foot_r,
            "c0": foot_c,
            "r1": foot_r + inner_rows,
            "c1": foot_c + inner_cols + 1,
            "ir0": foot_r + 1,
            "ic0": foot_c + 1,
            "ir1": foot_r + inner_rows,
            "ic1": foot_c + inner_cols,
        }
    if direction == "left":
        return {
            "direction": direction,
            "r0": foot_r,
            "c0": foot_c + 1,
            "r1": foot_r + inner_rows + 1,
            "c1": foot_c + inner_cols + 1,
            "ir0": foot_r + 1,
            "ic0": foot_c + 1,
            "ir1": foot_r + inner_rows,
            "ic1": foot_c + inner_cols,
        }
    return {
        "direction": direction,
        "r0": foot_r,
        "c0": foot_c,
        "r1": foot_r + inner_rows + 1,
        "c1": foot_c + inner_cols,
        "ir0": foot_r + 1,
        "ic0": foot_c + 1,
        "ir1": foot_r + inner_rows,
        "ic1": foot_c + inner_cols,
    }


def _draw_spec(grid: np.ndarray, spec: dict) -> None:
    info = _placed_info(spec)
    color = int(spec["wall_color"])
    direction = str(spec["direction"])
    inner_rows = int(spec["inner_rows"])
    inner_cols = int(spec["inner_cols"])

    r0 = int(info["r0"])
    c0 = int(info["c0"])
    r1 = int(info["r1"])
    c1 = int(info["c1"])
    ir0 = int(info["ir0"])
    ic0 = int(info["ic0"])
    ir1 = int(info["ir1"])
    ic1 = int(info["ic1"])

    if direction == "up":
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        grid[r1, c0 : c1 + 1] = color
        start = ic0 + int(spec["bar_offset"])
        for idx, value in enumerate(spec["colors"]):
            grid[r0 - 1, start + idx] = int(value)
    elif direction == "down":
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        grid[r0, c0 : c1 + 1] = color
        start = ic0 + int(spec["bar_offset"])
        for idx, value in enumerate(spec["colors"]):
            grid[r1 + 1, start + idx] = int(value)
    elif direction == "left":
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c1] = color
        start = ir0 + int(spec["bar_offset"])
        for idx, value in enumerate(spec["colors"]):
            grid[start + idx, c0 - 1] = int(value)
    else:
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        start = ir0 + int(spec["bar_offset"])
        for idx, value in enumerate(spec["colors"]):
            grid[start + idx, c1 + 1] = int(value)

    for (pr, pc), value in spec["pixels"]:
        grid[ir0 + int(pr), ic0 + int(pc)] = int(value)


def _render_output(inp: np.ndarray, specs: list[dict], bg: int) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        info = _placed_info(spec)
        _apply_reorder(out, info, [int(x) for x in spec["colors"]], bg)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, ARC_MAX)
    w = rng.randint(18, ARC_MAX)
    n_containers = rng.randint(2, 4)

    bg = rng.choice(COLORS)
    remaining = [color for color in COLORS if color != bg]
    wall_colors = rng.sample(remaining, n_containers)
    content_pool = [color for color in remaining if color not in wall_colors]
    if len(content_pool) < 4:
        return None

    directions = rng.sample(DIRECTIONS, n_containers)
    specs = [
        _make_spec(rng, direction, wall_color, content_pool)
        for direction, wall_color in zip(directions, wall_colors)
    ]

    occ = np.zeros((h, w), dtype=bool)
    rng.shuffle(specs)
    for spec in specs:
        candidates = _placement_candidates(occ, int(spec["foot_h"]), int(spec["foot_w"]))
        if not candidates:
            return None
        foot_r, foot_c = rng.choice(candidates)
        spec["foot_r"] = foot_r
        spec["foot_c"] = foot_c
        occ[foot_r - 1 : foot_r + int(spec["foot_h"]) + 1, foot_c - 1 : foot_c + int(spec["foot_w"]) + 1] = True

    grid = np.full((h, w), bg, dtype=int)
    for spec in specs:
        _draw_spec(grid, spec)

    output = _render_output(grid, specs, bg)
    if np.array_equal(grid, output):
        return None
    if not _bg_is_unique_mode(grid, bg):
        return None
    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
