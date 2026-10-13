from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


ORTH4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(10))

# Generator bounds and retry policy.
GEN_HEIGHT_RANGE = (16, 27)
GEN_WIDTH_RANGE = (16, 27)
GEN_ATTEMPTS = 800
LINE_GAP = 3


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _same_color_component(grid: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    color = int(grid[start])
    q = deque([start])
    seen = {start}
    out: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        out.append((r, c))
        for dr, dc in ORTH4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in seen and int(grid[rr, cc]) == color:
                seen.add((rr, cc))
                q.append((rr, cc))
    return out


def _line3_signature(coords: list[tuple[int, int]]) -> tuple[str, tuple[int, int]] | None:
    if len(coords) != 3:
        return None
    rows = sorted(r for r, _ in coords)
    cols = sorted(c for _, c in coords)
    if len(set(rows)) == 1 and cols == list(range(cols[0], cols[0] + 3)):
        return "h", (rows[0], cols[1])
    if len(set(cols)) == 1 and rows == list(range(rows[0], rows[0] + 3)):
        return "v", (rows[1], cols[0])
    return None


def _find_marker(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape
    visited: set[tuple[int, int]] = set()
    candidates: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if (r, c) in visited or int(g[r, c]) == bg:
                continue
            comp = _same_color_component(g, (r, c))
            visited.update(comp)
            sig = _line3_signature(comp)
            if sig is None:
                continue
            axis, center = sig
            bar_color = int(g[r, c])
            control_dirs = [(-1, 0), (1, 0)] if axis == "h" else [(0, -1), (0, 1)]
            controls: list[tuple[int, int]] = []
            for dr, dc in control_dirs:
                rr, cc = center[0] + dr, center[1] + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if int(g[rr, cc]) in (bg, bar_color):
                    continue
                if len(_same_color_component(g, (rr, cc))) == 1:
                    controls.append((rr, cc))
            if len(controls) != 1:
                continue
            control = controls[0]
            ray = (center[0] - control[0], center[1] - control[1])
            candidates.append(
                {
                    "bar_color": bar_color,
                    "control_color": int(g[control]),
                    "center": center,
                    "control": control,
                    "ray": ray,
                    "axis": axis,
                }
            )

    assert len(candidates) == 1, "expected exactly one valid marker bar with a unique control pixel"
    return candidates[0]


def _blank_rows(grid: np.ndarray, bg: int) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == bg))]


def _blank_cols(grid: np.ndarray, bg: int) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == bg))]


def _stamp_centers_from_marker(
    grid: np.ndarray, bg: int, center: tuple[int, int], ray: tuple[int, int]
) -> list[tuple[int, int]]:
    if ray[0] != 0:
        rows = _blank_rows(grid, bg)
        return [(r, center[1]) for r in rows if (r - center[0]) * ray[0] > 0]
    cols = _blank_cols(grid, bg)
    return [(center[0], c) for c in cols if (c - center[1]) * ray[1] > 0]


def _apply_stamps(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    stamp_color: int,
    bg: int,
    *,
    only_bg: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    for r, c in centers:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w and (not only_bg or int(g[rr, cc]) == bg):
                    out[rr, cc] = int(stamp_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp a control-colored 3x3 patch at each intersection of the marker ray and every perpendicular blank line.

    Input structure:
    - The dominant color is the background.
    - One foreground object is a three-pixel straight marker bar, and a singleton control pixel touches the bar center on a perpendicular side.
    - The canvas also contains some fully blank rows or columns plus scattered obstacle pixels of other colors.

    Rule:
    - Start from the bar center and cast an invisible ray in the direction away from the adjacent control pixel.
    - Take every completely background row or column whose orientation is perpendicular to that ray, and use the ray/blank-line intersection cell as the center of a 3x3 stamp colored like the control pixel.
    - Stamp only over original background cells; obstacles, the marker bar, and the control pixel stay unchanged.

    Key insight:
    - The painted cells are not the blank lines themselves: the answer comes from combining two latent structures, an unseen half-ray and the coordinates supplied by full blank lines.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2-D grid"
    bg = _get_bg(g)
    marker = _find_marker(g)
    centers = _stamp_centers_from_marker(g, bg, marker["center"], marker["ray"])
    return _apply_stamps(g, centers, int(marker["control_color"]), bg, only_bg=True)


def _sample_spaced(candidates: list[int], count_range: tuple[int, int]) -> list[int] | None:
    counts = list(range(count_range[0], count_range[1] + 1))
    random.shuffle(counts)
    for want in counts:
        if want == 0:
            return []
        for _ in range(80):
            chosen: list[int] = []
            order = candidates[:]
            random.shuffle(order)
            for value in order:
                if all(abs(value - prev) >= LINE_GAP for prev in chosen):
                    chosen.append(value)
                    if len(chosen) == want:
                        return sorted(chosen)
    return None


def _pick_layout(h: int, w: int) -> dict[str, object] | None:
    axis = random.choice(["h", "v"])
    ray_sign = random.choice([-1, 1])
    center_r = random.randint(3, h - 4)
    center_c = random.randint(3, w - 4)

    if axis == "h":
        ray = (ray_sign, 0)
        control = (center_r - ray_sign, center_c)
        positive = list(range(1, center_r - 2)) if ray_sign < 0 else list(range(center_r + 3, h - 1))
        negative = list(range(center_r + 3, h - 1)) if ray_sign < 0 else list(range(1, center_r - 2))
        parallel = [c for c in range(1, w - 1) if c not in {center_c - 1, center_c, center_c + 1}]
        stamp_side = "rows"
    else:
        ray = (0, ray_sign)
        control = (center_r, center_c - ray_sign)
        positive = list(range(1, center_c - 2)) if ray_sign < 0 else list(range(center_c + 3, w - 1))
        negative = list(range(center_c + 3, w - 1)) if ray_sign < 0 else list(range(1, center_c - 2))
        parallel = [r for r in range(1, h - 1) if r not in {center_r - 1, center_r, center_r + 1}]
        stamp_side = "cols"

    pos_lines = _sample_spaced(positive, (2, 4))
    neg_lines = _sample_spaced(negative, (1, 2))
    parallel_lines = _sample_spaced(parallel, (1, 2))
    if pos_lines is None or neg_lines is None or parallel_lines is None:
        return None

    blank_rows = sorted(pos_lines + neg_lines) if stamp_side == "rows" else sorted(parallel_lines)
    blank_cols = sorted(parallel_lines) if stamp_side == "rows" else sorted(pos_lines + neg_lines)
    return {
        "axis": axis,
        "center": (center_r, center_c),
        "control": control,
        "ray": ray,
        "blank_rows": blank_rows,
        "blank_cols": blank_cols,
        "pos_lines": pos_lines,
        "neg_lines": neg_lines,
        "parallel_lines": parallel_lines,
        "stamp_side": stamp_side,
    }


def _stamp_centers_from_info(info: dict[str, object]) -> list[tuple[int, int]]:
    center_r, center_c = info["center"]
    if info["stamp_side"] == "rows":
        return [(int(r), int(center_c)) for r in info["pos_lines"]]
    return [(int(center_r), int(c)) for c in info["pos_lines"]]


def _choose_obstacle_color(grid: np.ndarray, r: int, c: int, colors: list[int]) -> int | None:
    h, w = grid.shape
    options = colors[:]
    random.shuffle(options)
    for color in options:
        ok = True
        for dr, dc in ORTH4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == color:
                ok = False
                break
        if ok:
            return int(color)
    return None


def _place_singleton(
    grid: np.ndarray, cell: tuple[int, int], colors: list[int], forbidden: set[tuple[int, int]]
) -> bool:
    r, c = cell
    if (r, c) in forbidden:
        return False
    if int(grid[r, c]) != _get_bg(grid):
        return False
    color = _choose_obstacle_color(grid, r, c, colors)
    if color is None:
        return False
    grid[r, c] = color
    return True


def _cover_rows_and_cols(
    grid: np.ndarray,
    blank_rows: set[int],
    blank_cols: set[int],
    obstacle_colors: list[int],
    forbidden: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    while True:
        uncovered_rows = {r for r in range(h) if r not in blank_rows and bool(np.all(grid[r, :] == _get_bg(grid)))}
        uncovered_cols = {c for c in range(w) if c not in blank_cols and bool(np.all(grid[:, c] == _get_bg(grid)))}
        if not uncovered_rows and not uncovered_cols:
            return True

        best_score = -1
        best_cells: list[tuple[int, int]] = []
        for r in range(h):
            if r in blank_rows:
                continue
            for c in range(w):
                if c in blank_cols or int(grid[r, c]) != _get_bg(grid):
                    continue
                if (r, c) in forbidden:
                    continue
                score = int(r in uncovered_rows) + int(c in uncovered_cols)
                if score > best_score:
                    best_score = score
                    best_cells = [(r, c)]
                elif score == best_score:
                    best_cells.append((r, c))
        if best_score <= 0 or not best_cells:
            return False
        cell = random.choice(best_cells)
        if not _place_singleton(grid, cell, obstacle_colors, forbidden):
            return False


def _add_targeted_obstacles(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    blank_rows: set[int],
    blank_cols: set[int],
    obstacle_colors: list[int],
    forbidden: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    target_centers = centers[:]
    random.shuffle(target_centers)
    need = max(1, len(target_centers) // 2)
    placed = 0
    for r, c in target_centers:
        options = []
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if (rr, cc) == (r, c):
                    continue
                if rr in blank_rows or cc in blank_cols:
                    continue
                if (rr, cc) in forbidden:
                    continue
                if int(grid[rr, cc]) == _get_bg(grid):
                    options.append((rr, cc))
        random.shuffle(options)
        for cell in options:
            if _place_singleton(grid, cell, obstacle_colors, forbidden):
                placed += 1
                break
        if placed >= need:
            return True
    return placed >= need


def _add_extra_noise(
    grid: np.ndarray,
    blank_rows: set[int],
    blank_cols: set[int],
    obstacle_colors: list[int],
    forbidden: set[tuple[int, int]],
) -> None:
    h, w = grid.shape
    free = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if r not in blank_rows and c not in blank_cols and (r, c) not in forbidden and int(grid[r, c]) == _get_bg(grid)
    ]
    random.shuffle(free)
    budget = random.randint(4, 10)
    for cell in free:
        if budget <= 0:
            return
        if random.random() < 0.35 and _place_singleton(grid, cell, obstacle_colors, forbidden):
            budget -= 1


def _independent_expected(inp: np.ndarray, info: dict[str, object], bg: int, control_color: int) -> np.ndarray:
    return _apply_stamps(inp, _stamp_centers_from_info(info), control_color, bg, only_bg=True)


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, info: dict[str, object], colors: dict[str, int]) -> bool:
    bg = colors["bg"]
    marker = _find_marker(inp)
    if tuple(marker["center"]) != tuple(info["center"]):
        return False
    if tuple(marker["control"]) != tuple(info["control"]):
        return False
    if tuple(marker["ray"]) != tuple(info["ray"]):
        return False
    if _blank_rows(inp, bg) != info["blank_rows"]:
        return False
    if _blank_cols(inp, bg) != info["blank_cols"]:
        return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    stamp_centers = _stamp_centers_from_info(info)
    if not stamp_centers:
        return False
    expanded = {(r, c) for r, c in zip(*np.where(actual != inp))}
    if not any(cell not in stamp_centers for cell in expanded):
        return False

    wrong_both_sides = []
    center_r, center_c = info["center"]
    if info["stamp_side"] == "rows":
        wrong_both_sides = [(r, center_c) for r in info["blank_rows"]]
        wrong_parallel = stamp_centers + [(center_r, c) for c in info["parallel_lines"]]
    else:
        wrong_both_sides = [(center_r, c) for c in info["blank_cols"]]
        wrong_parallel = stamp_centers + [(r, center_c) for r in info["parallel_lines"]]
    if np.array_equal(_apply_stamps(inp, wrong_both_sides, colors["control"], bg, only_bg=True), expected):
        return False
    if np.array_equal(_apply_stamps(inp, wrong_parallel, colors["control"], bg, only_bg=True), expected):
        return False
    if np.array_equal(_apply_stamps(inp, stamp_centers, colors["control"], bg, only_bg=False), expected):
        return False

    fill_lines = inp.copy()
    fill_lines[info["blank_rows"], :] = colors["control"]
    fill_lines[:, info["blank_cols"]] = colors["control"]
    if np.array_equal(fill_lines, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    layout = _pick_layout(h, w)
    if layout is None:
        return None

    color_pool = list(PALETTE)
    random.shuffle(color_pool)
    bg, bar, control = color_pool[:3]
    obstacle_colors = color_pool[3 : random.randint(6, 8)]
    colors = {"bg": bg, "bar": bar, "control": control}

    grid = np.full((h, w), bg, dtype=int)
    center_r, center_c = layout["center"]
    control_r, control_c = layout["control"]
    if layout["axis"] == "h":
        for dc in (-1, 0, 1):
            grid[center_r, center_c + dc] = bar
    else:
        for dr in (-1, 0, 1):
            grid[center_r + dr, center_c] = bar
    grid[control_r, control_c] = control

    blank_rows = set(layout["blank_rows"])
    blank_cols = set(layout["blank_cols"])
    centers = _stamp_centers_from_info(layout)
    opposite_control = (center_r + layout["ray"][0], center_c + layout["ray"][1])
    forbidden = {
        layout["center"],
        layout["control"],
        opposite_control,
    }
    if layout["axis"] == "h":
        forbidden.update({(center_r, center_c - 1), (center_r, center_c + 1)})
    else:
        forbidden.update({(center_r - 1, center_c), (center_r + 1, center_c)})

    if not _add_targeted_obstacles(grid, centers, blank_rows, blank_cols, obstacle_colors, forbidden):
        return None
    if not _cover_rows_and_cols(grid, blank_rows, blank_cols, obstacle_colors, forbidden):
        return None
    _add_extra_noise(grid, blank_rows, blank_cols, obstacle_colors, forbidden)

    if any(int(grid[r, c]) != bg for r in blank_rows for c in range(w)):
        return None
    if any(int(grid[r, c]) != bg for r in range(h) for c in blank_cols):
        return None

    expected = _independent_expected(grid, layout, bg, control)
    if np.array_equal(grid, expected):
        return None
    if not _validate_semantics(grid, expected, layout, colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-and-blank-line stamping input")
