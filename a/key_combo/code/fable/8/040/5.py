from __future__ import annotations

import random
from collections import Counter

import numpy as np


RED = 2
COLOR_POOL = [1, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _sample_spaced_positions(count: int, low: int, high: int, gap: int) -> list[int] | None:
    if count == 0:
        return []
    positions = list(range(low, high + 1))
    for _ in range(200):
        chosen = sorted(random.sample(positions, count))
        if all(b - a >= gap for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _collect_markers(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    h, w = grid.shape
    markers: list[dict[str, int]] = []

    for c in range(1, w - 1):
        color = int(grid[0, c])
        if color != bg:
            markers.append({"orientation": "V", "side": "top", "fixed": c, "color": color, "r": 0, "c": c})
    for c in range(1, w - 1):
        color = int(grid[h - 1, c])
        if color != bg:
            markers.append(
                {"orientation": "V", "side": "bottom", "fixed": c, "color": color, "r": h - 1, "c": c}
            )
    for r in range(1, h - 1):
        color = int(grid[r, 0])
        if color != bg:
            markers.append({"orientation": "H", "side": "left", "fixed": r, "color": color, "r": r, "c": 0})
    for r in range(1, h - 1):
        color = int(grid[r, w - 1])
        if color != bg:
            markers.append(
                {"orientation": "H", "side": "right", "fixed": r, "color": color, "r": r, "c": w - 1}
            )
    return markers


def _trace_ray(grid: np.ndarray, bg: int, marker: dict[str, int]) -> dict[str, object]:
    h, w = grid.shape
    color = int(marker["color"])
    cells: list[tuple[int, int]] = []
    same_color_walls: list[tuple[int, int]] = []
    blocker: tuple[int, int] | None = None

    if marker["orientation"] == "H":
        r = int(marker["fixed"])
        if marker["side"] == "left":
            coords = [(r, c) for c in range(1, w - 1)]
        else:
            coords = [(r, c) for c in range(w - 2, 0, -1)]
    else:
        c = int(marker["fixed"])
        if marker["side"] == "top":
            coords = [(r, c) for r in range(1, h - 1)]
        else:
            coords = [(r, c) for r in range(h - 2, 0, -1)]

    for r, c in coords:
        value = int(grid[r, c])
        if value == bg:
            cells.append((r, c))
            continue
        if value == color:
            cells.append((r, c))
            same_color_walls.append((r, c))
            continue
        blocker = (r, c)
        break

    if not cells:
        interval = None
    elif marker["orientation"] == "H":
        cols = [c for _r, c in cells]
        interval = (min(cols), max(cols))
    else:
        rows = [r for r, _c in cells]
        interval = (min(rows), max(rows))

    return {"cells": cells, "interval": interval, "same_walls": same_color_walls, "blocker": blocker}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal only the intersections of inward filter-rays fired by border markers.

    Input structure:
    - Non-background singleton markers sit on the outer border, away from the corners.
    - Interior non-background pixels are wall pixels of various colors.
    - Background fills every remaining cell.

    Rule:
    - Each top/bottom marker fires a vertical ray inward; each left/right marker fires a horizontal ray inward.
    - A ray passes through background and through wall pixels of its own color, but stops immediately before the first wall pixel of a different color; if no such wall appears, it survives to the opposite side.
    - Rays themselves stay invisible. Only when a surviving horizontal ray and a surviving vertical ray both cover the same cell do we color that background cell: use red for different marker colors, otherwise use their shared color.
    - Preserve every original marker pixel and every wall pixel unchanged.

    Key insight:
    - Interior walls are color filters, not uniform blockers, so a visible crossing may exist beyond a same-color wall while a different-color wall erases later crossings by terminating the hidden ray one step early.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    markers = _collect_markers(g, bg)
    horizontals = [m for m in markers if m["orientation"] == "H"]
    verticals = [m for m in markers if m["orientation"] == "V"]

    traced_h = [(m, _trace_ray(g, bg, m)) for m in horizontals]
    traced_v = [(m, _trace_ray(g, bg, m)) for m in verticals]

    out = g.copy()
    for h_marker, h_info in traced_h:
        h_interval = h_info["interval"]
        if h_interval is None:
            continue
        row = int(h_marker["fixed"])
        h0, h1 = h_interval
        for v_marker, v_info in traced_v:
            v_interval = v_info["interval"]
            if v_interval is None:
                continue
            col = int(v_marker["fixed"])
            v0, v1 = v_interval
            if not (h0 <= col <= h1 and v0 <= row <= v1):
                continue
            if int(g[row, col]) != bg:
                continue
            color = int(h_marker["color"]) if int(h_marker["color"]) == int(v_marker["color"]) else RED
            if int(out[row, col]) == bg:
                out[row, col] = color
            elif int(out[row, col]) != color:
                out[row, col] = RED
    return out


def _plan_interval(ray: dict[str, object], axis_size: int, opposing_axes: list[int], spare_axes: list[int]) -> bool:
    mode = str(ray["mode"])
    side = str(ray["side"])
    if mode == "through":
        ray["blocker_axis"] = None
        ray["axis_interval"] = (1, axis_size - 2)
        return True

    blocker_choices = [p for p in spare_axes if min(opposing_axes) < p < max(opposing_axes)]
    if not blocker_choices:
        return False
    blocker = random.choice(blocker_choices)
    ray["blocker_axis"] = blocker
    if side in ("left", "top"):
        ray["axis_interval"] = (1, blocker - 1)
        ray["visible_opposing"] = [a for a in opposing_axes if a < blocker]
        ray["suppressed_opposing"] = [a for a in opposing_axes if a > blocker]
    else:
        ray["axis_interval"] = (blocker + 1, axis_size - 2)
        ray["visible_opposing"] = [a for a in opposing_axes if a > blocker]
        ray["suppressed_opposing"] = [a for a in opposing_axes if a < blocker]
    return bool(ray["visible_opposing"]) and bool(ray["suppressed_opposing"])


def _same_wall_candidates(ray: dict[str, object]) -> list[int]:
    lo, hi = ray["axis_interval"]
    visible = list(ray["visible_opposing"])
    if not visible:
        return []
    if ray["side"] in ("left", "top"):
        return [p for p in ray["spare_axes"] if lo <= p <= hi and any(v > p for v in visible)]
    return [p for p in ray["spare_axes"] if lo <= p <= hi and any(v < p for v in visible)]


def _visible_pairs(horizontals: list[dict[str, object]], verticals: list[dict[str, object]]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for hi, h_ray in enumerate(horizontals):
        h0, h1 = h_ray["axis_interval"]
        row = int(h_ray["fixed"])
        for vi, v_ray in enumerate(verticals):
            v0, v1 = v_ray["axis_interval"]
            col = int(v_ray["fixed"])
            if h0 <= col <= h1 and v0 <= row <= v1:
                pairs.append((hi, vi))
    return pairs


def _assign_marker_colors(horizontals: list[dict[str, object]], verticals: list[dict[str, object]]) -> bool:
    visible = _visible_pairs(horizontals, verticals)
    if len(visible) < 2:
        return False
    for _ in range(200):
        for ray in horizontals + verticals:
            ray["color"] = random.choice(COLOR_POOL)
        same_found = any(horizontals[hi]["color"] == verticals[vi]["color"] for hi, vi in visible)
        diff_found = any(horizontals[hi]["color"] != verticals[vi]["color"] for hi, vi in visible)
        if same_found and diff_found:
            return True
    return False


def _render_input(
    h: int, w: int, horizontals: list[dict[str, object]], verticals: list[dict[str, object]]
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)

    for ray in horizontals + verticals:
        if ray["orientation"] == "H":
            marker_pos = (int(ray["fixed"]), 0 if ray["side"] == "left" else w - 1)
        else:
            marker_pos = (0 if ray["side"] == "top" else h - 1, int(ray["fixed"]))
        grid[marker_pos] = int(ray["color"])

    for ray in horizontals + verticals:
        color = int(ray["color"])
        if ray["orientation"] == "H":
            r = int(ray["fixed"])
            for c in ray["same_walls"]:
                grid[r, int(c)] = color
            blocker = ray["blocker_axis"]
            if blocker is not None:
                blocker_color = random.choice([c for c in COLOR_POOL if c != color])
                grid[r, int(blocker)] = blocker_color
        else:
            c = int(ray["fixed"])
            for r in ray["same_walls"]:
                grid[int(r), c] = color
            blocker = ray["blocker_axis"]
            if blocker is not None:
                blocker_color = random.choice([c for c in COLOR_POOL if c != color])
                grid[int(blocker), c] = blocker_color
    return grid


def _expected_output(
    inp: np.ndarray, horizontals: list[dict[str, object]], verticals: list[dict[str, object]]
) -> np.ndarray:
    out = inp.copy()
    for h_ray in horizontals:
        h0, h1 = h_ray["axis_interval"]
        row = int(h_ray["fixed"])
        for v_ray in verticals:
            v0, v1 = v_ray["axis_interval"]
            col = int(v_ray["fixed"])
            if h0 <= col <= h1 and v0 <= row <= v1 and inp[row, col] == 0:
                color = int(h_ray["color"]) if int(h_ray["color"]) == int(v_ray["color"]) else RED
                if out[row, col] == 0:
                    out[row, col] = color
                elif out[row, col] != color:
                    out[row, col] = RED
    return out


def _validate_semantics(
    inp: np.ndarray,
    expected: np.ndarray,
    horizontals: list[dict[str, object]],
    verticals: list[dict[str, object]],
    blocked_id: tuple[str, int],
    transparent_id: tuple[str, int],
    through_id: tuple[str, int],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    traced_h = [_trace_ray(inp, 0, {"orientation": "H", "side": r["side"], "fixed": r["fixed"], "color": r["color"]}) for r in horizontals]
    traced_v = [_trace_ray(inp, 0, {"orientation": "V", "side": r["side"], "fixed": r["fixed"], "color": r["color"]}) for r in verticals]
    for ray, traced in zip(horizontals, traced_h):
        if traced["interval"] != ray["axis_interval"]:
            return False
    for ray, traced in zip(verticals, traced_v):
        if traced["interval"] != ray["axis_interval"]:
            return False

    visible = _visible_pairs(horizontals, verticals)
    if not visible:
        return False
    same_count = sum(1 for hi, vi in visible if horizontals[hi]["color"] == verticals[vi]["color"])
    diff_count = sum(1 for hi, vi in visible if horizontals[hi]["color"] != verticals[vi]["color"])
    if same_count == 0 or diff_count == 0:
        return False

    kind, idx = blocked_id
    blocked_ray = horizontals[idx] if kind == "H" else verticals[idx]
    if blocked_ray["mode"] != "blocked":
        return False
    if not blocked_ray["visible_opposing"] or not blocked_ray["suppressed_opposing"]:
        return False

    kind, idx = transparent_id
    transparent_ray = horizontals[idx] if kind == "H" else verticals[idx]
    if not transparent_ray["same_walls"]:
        return False
    wall_axis = transparent_ray["same_walls"][0]
    if transparent_ray["side"] in ("left", "top"):
        if not any(v > wall_axis for v in transparent_ray["visible_opposing"]):
            return False
    else:
        if not any(v < wall_axis for v in transparent_ray["visible_opposing"]):
            return False

    kind, idx = through_id
    through_ray = horizontals[idx] if kind == "H" else verticals[idx]
    if through_ray["mode"] != "through":
        return False

    if int(np.count_nonzero(inp[1:-1, 1:-1])) < 3:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 24)
    w = random.randint(14, 24)
    n_left = random.randint(1, 2)
    n_right = random.randint(1, 2)
    n_top = random.randint(1, 2)
    n_bottom = random.randint(1, 2)

    rows = _sample_spaced_positions(n_left + n_right, 1, h - 2, 2)
    cols = _sample_spaced_positions(n_top + n_bottom, 1, w - 2, 2)
    if rows is None or cols is None:
        return None
    random.shuffle(rows)
    random.shuffle(cols)

    left_rows = sorted(rows[:n_left])
    right_rows = sorted(rows[n_left:])
    top_cols = sorted(cols[:n_top])
    bottom_cols = sorted(cols[n_top:])
    selected_rows = sorted(left_rows + right_rows)
    selected_cols = sorted(top_cols + bottom_cols)

    spare_rows = [r for r in range(1, h - 1) if r not in selected_rows]
    spare_cols = [c for c in range(1, w - 1) if c not in selected_cols]
    if not spare_rows or not spare_cols:
        return None

    horizontals = [{"orientation": "H", "side": "left", "fixed": r} for r in left_rows] + [
        {"orientation": "H", "side": "right", "fixed": r} for r in right_rows
    ]
    verticals = [{"orientation": "V", "side": "top", "fixed": c} for c in top_cols] + [
        {"orientation": "V", "side": "bottom", "fixed": c} for c in bottom_cols
    ]
    all_ids = [("H", i) for i in range(len(horizontals))] + [("V", i) for i in range(len(verticals))]

    blocked_id = random.choice(all_ids)
    remaining = [rid for rid in all_ids if rid != blocked_id]
    through_id = random.choice(remaining)
    transparent_id = through_id if random.random() < 0.7 else random.choice(remaining)

    for i, ray in enumerate(horizontals):
        ray["mode"] = "blocked" if ("H", i) == blocked_id else ("through" if ("H", i) == through_id else random.choice(["through", "blocked"]))
        ray["visible_opposing"] = list(selected_cols)
        ray["suppressed_opposing"] = []
        ray["spare_axes"] = spare_cols
        if not _plan_interval(ray, w, selected_cols, spare_cols):
            return None

    for i, ray in enumerate(verticals):
        ray["mode"] = "blocked" if ("V", i) == blocked_id else ("through" if ("V", i) == through_id else random.choice(["through", "blocked"]))
        ray["visible_opposing"] = list(selected_rows)
        ray["suppressed_opposing"] = []
        ray["spare_axes"] = spare_rows
        if not _plan_interval(ray, h, selected_rows, spare_rows):
            return None

    if not _assign_marker_colors(horizontals, verticals):
        return None

    for ray in horizontals + verticals:
        ray["same_walls"] = []
        candidates = _same_wall_candidates(ray)
        if not candidates:
            continue
        want_wall = (ray["orientation"], horizontals.index(ray) if ray["orientation"] == "H" else verticals.index(ray)) == transparent_id
        if want_wall or random.random() < 0.35:
            random.shuffle(candidates)
            count = 1 if want_wall else random.randint(1, min(2, len(candidates)))
            ray["same_walls"] = sorted(candidates[:count])

    kind, idx = transparent_id
    transparent_ray = horizontals[idx] if kind == "H" else verticals[idx]
    if not transparent_ray["same_walls"]:
        candidates = _same_wall_candidates(transparent_ray)
        if not candidates:
            return None
        transparent_ray["same_walls"] = [random.choice(candidates)]

    inp = _render_input(h, w, horizontals, verticals)
    expected = _expected_output(inp, horizontals, verticals)
    if not _validate_semantics(inp, expected, horizontals, verticals, blocked_id, transparent_id, through_id):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
