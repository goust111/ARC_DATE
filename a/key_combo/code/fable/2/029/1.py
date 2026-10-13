from __future__ import annotations

from collections import Counter, defaultdict
import random
from typing import Iterable

import numpy as np


Coord = tuple[int, int]
LineSig = tuple[str, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: Iterable[int] = ()) -> list[int]:
    colors = sorted(set(map(int, grid.flatten())) - set(map(int, exclude)))
    return colors


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _line_cells(shape: tuple[int, int], kind: str, value: int) -> frozenset[Coord]:
    h, w = shape
    if kind == "H":
        return frozenset((value, c) for c in range(w))
    if kind == "V":
        return frozenset((r, value) for r in range(h))
    if kind == "D":
        return frozenset((r, r - value) for r in range(h) if 0 <= r - value < w)
    if kind == "U":
        return frozenset((r, value - r) for r in range(h) if 0 <= value - r < w)
    raise ValueError(f"unknown line kind: {kind}")


def _infer_line_signature(coords: np.ndarray) -> LineSig:
    rows = coords[:, 0]
    cols = coords[:, 1]
    candidates: list[LineSig] = []
    if np.all(rows == rows[0]):
        candidates.append(("H", int(rows[0])))
    if np.all(cols == cols[0]):
        candidates.append(("V", int(cols[0])))
    down = rows - cols
    if np.all(down == down[0]):
        candidates.append(("D", int(down[0])))
    up = rows + cols
    if np.all(up == up[0]):
        candidates.append(("U", int(up[0])))
    assert len(candidates) == 1, "visible pixels must uniquely identify one full line"
    return candidates[0]


def _bar_signature(coords: np.ndarray) -> tuple[str, Coord] | None:
    if coords.shape[0] != 3:
        return None
    rows = sorted(set(map(int, coords[:, 0])))
    cols = sorted(set(map(int, coords[:, 1])))
    if len(rows) == 1 and len(cols) == 3 and cols == list(range(cols[0], cols[0] + 3)):
        return "horizontal", (rows[0], cols[1])
    if len(cols) == 1 and len(rows) == 3 and rows == list(range(rows[0], rows[0] + 3)):
        return "vertical", (rows[1], cols[0])
    return None


def _find_marker_and_control(g: np.ndarray, bg: int) -> tuple[int, set[Coord], int, Coord, Coord, Coord]:
    colors = _colors_present(g, exclude={bg})
    bar_candidates: list[tuple[int, str, Coord]] = []
    for color in colors:
        coords = np.argwhere(g == color)
        sig = _bar_signature(coords)
        if sig is not None:
            axis, center = sig
            bar_candidates.append((int(color), axis, center))

    assert len(bar_candidates) == 1, "expected one unique three-pixel marker bar"
    bar_color, axis, center = bar_candidates[0]
    bar_cells = (
        {(center[0], center[1] - 1), center, (center[0], center[1] + 1)}
        if axis == "horizontal"
        else {(center[0] - 1, center[1]), center, (center[0] + 1, center[1])}
    )

    control_candidates: list[tuple[int, Coord]] = []
    for color in colors:
        if color == bar_color:
            continue
        coords = np.argwhere(g == color)
        if coords.shape[0] != 1:
            continue
        r, c = map(int, coords[0])
        if abs(r - center[0]) + abs(c - center[1]) == 1:
            control_candidates.append((int(color), (r, c)))

    assert len(control_candidates) == 1, "expected one singleton control pixel touching the bar center"
    control_color, control = control_candidates[0]

    step = (center[0] - control[0], center[1] - control[1])
    assert abs(step[0]) + abs(step[1]) == 1
    if axis == "horizontal":
        assert step[1] == 0, "horizontal marker bars emit vertically"
    else:
        assert step[0] == 0, "vertical marker bars emit horizontally"
    return bar_color, bar_cells, control_color, control, center, step


def _ray_cells(center: Coord, step: Coord, shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    r, c = center[0] + step[0], center[1] + step[1]
    cells: list[Coord] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += step[0]
        c += step[1]
    return cells


def _infer_lines(g: np.ndarray, bg: int, excluded_colors: set[int]) -> dict[int, frozenset[Coord]]:
    lines: dict[int, frozenset[Coord]] = {}
    used_signatures: set[LineSig] = set()
    for color in _colors_present(g, exclude=excluded_colors | {bg}):
        coords = np.argwhere(g == color)
        assert coords.shape[0] >= 2, "every line needs at least two visible pixels"
        sig = _infer_line_signature(coords)
        assert sig not in used_signatures, "different colors must be different full lines"
        used_signatures.add(sig)
        cells = _line_cells(g.shape, *sig)
        assert set(map(tuple, coords)).issubset(cells)
        lines[int(color)] = cells
    assert lines, "expected at least one line"
    return lines


def _covering_lines(lines: dict[int, frozenset[Coord]]) -> dict[Coord, list[int]]:
    cover: dict[Coord, list[int]] = defaultdict(list)
    for color, cells in lines.items():
        for cell in cells:
            cover[cell].append(int(color))
    return cover


def _visible_order_edges(
    g: np.ndarray,
    lines: dict[int, frozenset[Coord]],
    ignored_colors: set[int],
) -> set[tuple[int, int]]:
    bg = _get_bg(g)
    cover = _covering_lines(lines)
    edges: set[tuple[int, int]] = set()
    for (r, c), colors in cover.items():
        visible = int(g[r, c])
        if visible in ignored_colors:
            continue
        if visible == bg:
            assert False, "a covered cell cannot show background"
        assert visible in colors, "visible line pixel must belong to one covering line"
        for other in colors:
            if other != visible:
                edges.add((int(other), visible))
    return edges


def _all_topological_orders(colors: list[int], below_edges: set[tuple[int, int]]) -> list[tuple[int, ...]]:
    succ = {color: set() for color in colors}
    indeg = {color: 0 for color in colors}
    for lower, upper in below_edges:
        if upper not in succ[lower]:
            succ[lower].add(upper)
            indeg[upper] += 1

    orders: list[tuple[int, ...]] = []
    limit = 5000

    def dfs(prefix: list[int], indegrees: dict[int, int], available: list[int]) -> None:
        if len(orders) > limit:
            return
        if len(prefix) == len(colors):
            orders.append(tuple(prefix))
            return
        for color in sorted(available):
            next_prefix = prefix + [color]
            next_indeg = dict(indegrees)
            next_available = [item for item in available if item != color]
            for nxt in succ[color]:
                next_indeg[nxt] -= 1
                if next_indeg[nxt] == 0:
                    next_available.append(nxt)
            dfs(next_prefix, next_indeg, sorted(set(next_available)))

    start = sorted(color for color in colors if indeg[color] == 0)
    dfs([], indeg, start)
    assert orders, "visible above/below constraints must be satisfiable"
    assert len(orders) <= limit, "too many consistent layer orders"
    return orders


def _render_lines(
    shape: tuple[int, int],
    bg: int,
    lines: dict[int, frozenset[Coord]],
    order: tuple[int, ...],
    bar_color: int,
    bar_cells: set[Coord],
    control_color: int,
    control: Coord,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for color in order:
        for r, c in lines[int(color)]:
            out[r, c] = int(color)
    for r, c in bar_cells:
        out[r, c] = int(bar_color)
    out[control] = int(control_color)
    return out


def _erase_chain(
    lines: dict[int, frozenset[Coord]],
    order: tuple[int, ...],
    ray: list[Coord],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    layer = {color: idx for idx, color in enumerate(order)}
    remaining = set(lines)
    erased: list[int] = []
    while True:
        hit = None
        for cell in ray:
            visible = [color for color in remaining if cell in lines[color]]
            if visible:
                hit = max(visible, key=lambda color: layer[color])
                break
        if hit is None:
            break
        remaining.remove(int(hit))
        erased.append(int(hit))
    kept = tuple(color for color in order if color in remaining)
    return tuple(erased), kept


def _solve_all_outputs(g: np.ndarray) -> list[np.ndarray]:
    bg = _get_bg(g)
    bar_color, bar_cells, control_color, control, center, step = _find_marker_and_control(g, bg)
    lines = _infer_lines(g, bg, {bar_color, control_color})
    edges = _visible_order_edges(g, lines, {bg, bar_color, control_color})
    orders = _all_topological_orders(list(lines), edges)
    ray = _ray_cells(center, step, g.shape)
    outputs: list[np.ndarray] = []
    seen: set[bytes] = set()
    for order in orders:
        _erased, kept = _erase_chain(lines, order, ray)
        out = _render_lines(g.shape, bg, lines, kept, bar_color, bar_cells, control_color, control)
        key = out.tobytes()
        if key not in seen:
            seen.add(key)
            outputs.append(out)
    assert outputs, "expected at least one consistent output"
    return outputs


def main(grid: np.ndarray) -> np.ndarray:
    """Erase each full line struck by the marker ray, revealing lower lines until the ray is clear.

    Input structure:
    - The background is the dominant color.
    - There is one three-pixel straight marker bar plus one singleton control pixel adjacent to the bar center.
    - Every other foreground color is the visible part of one full one-cell-thick horizontal, vertical, or 45-degree diagonal line; higher lines hide lower lines at crossings.

    Rule:
    - Recover every complete line from its visible collinear pixels and infer all above/below constraints implied by visible crossings.
    - From the bar center, cast a ray in the direction away from the control pixel; repeatedly erase the whole line owning the first currently visible ray pixel, then reveal whatever lower lines remain underneath, until the ray reaches the border without hitting any line.
    - Keep the marker bar and control pixel unchanged, and re-render all unhit lines with the recovered layering.

    Key insight:
    - The ray can hit lines that were invisible at first, so the answer requires both full-line reconstruction and layered reveal, not just deleting the lines initially seen on the ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    outputs = _solve_all_outputs(g)
    assert len(outputs) == 1, "input clues must determine a unique final visible grid"
    return outputs[0]


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (14, 18)
GRID_WIDTH_RANGE = (14, 18)
LINE_COUNT_RANGE = (4, 5)
MIN_FULL_LINE_LEN = 7
MIN_RAY_LEN = 6
GEN_ATTEMPTS = 600
ORDER_ATTEMPTS = 80

ALL_COLORS = list(range(10))


def _marker_options(h: int, w: int) -> list[tuple[set[Coord], Coord, Coord, list[Coord]]]:
    options: list[tuple[set[Coord], Coord, Coord, list[Coord]]] = []
    for r in range(MIN_RAY_LEN, h - 1):
        for c in range(1, w - 1):
            center = (r, c)
            control = (r + 1, c)
            bar = {(r, c - 1), center, (r, c + 1)}
            ray = _ray_cells(center, (-1, 0), (h, w))
            if len(ray) >= MIN_RAY_LEN:
                options.append((bar, control, center, ray))
    for r in range(1, h - MIN_RAY_LEN):
        for c in range(1, w - 1):
            center = (r, c)
            control = (r - 1, c)
            bar = {(r, c - 1), center, (r, c + 1)}
            ray = _ray_cells(center, (1, 0), (h, w))
            if len(ray) >= MIN_RAY_LEN:
                options.append((bar, control, center, ray))
    for r in range(1, h - 1):
        for c in range(MIN_RAY_LEN, w - 1):
            center = (r, c)
            control = (r, c + 1)
            bar = {(r - 1, c), center, (r + 1, c)}
            ray = _ray_cells(center, (0, -1), (h, w))
            if len(ray) >= MIN_RAY_LEN:
                options.append((bar, control, center, ray))
    for r in range(1, h - 1):
        for c in range(1, w - MIN_RAY_LEN):
            center = (r, c)
            control = (r, c - 1)
            bar = {(r - 1, c), center, (r + 1, c)}
            ray = _ray_cells(center, (0, 1), (h, w))
            if len(ray) >= MIN_RAY_LEN:
                options.append((bar, control, center, ray))
    return options


def _candidate_lines(
    shape: tuple[int, int],
    reserved: set[Coord],
    ray: list[Coord],
) -> list[dict]:
    h, w = shape
    ray_set = set(ray)
    candidates: list[dict] = []
    for kind, values in (
        ("H", range(h)),
        ("V", range(w)),
        ("D", range(-(w - 1), h)),
        ("U", range(h + w - 1)),
    ):
        for value in values:
            cells = _line_cells(shape, kind, int(value))
            if len(cells) < MIN_FULL_LINE_LEN:
                continue
            if cells & reserved:
                continue
            ray_hits = sorted(cells & ray_set, key=lambda cell: ray.index(cell))
            if len(ray_hits) > 1:
                continue
            candidates.append(
                {
                    "sig": (kind, int(value)),
                    "kind": kind,
                    "value": int(value),
                    "cells": cells,
                    "hits_ray": len(ray_hits) == 1,
                    "ray_cell": ray_hits[0] if ray_hits else None,
                }
            )
    return candidates


def _pick_distinct(items: list[dict], count: int) -> list[dict] | None:
    if len(items) < count:
        return None
    chosen = items[:]
    random.shuffle(chosen)
    return chosen[:count]


def _intersects(a: dict, b: dict) -> bool:
    return bool(a["cells"] & b["cells"])


def _render_hidden_case(
    shape: tuple[int, int],
    bg: int,
    selected: list[dict],
    order: tuple[int, ...],
    bar_color: int,
    bar_cells: set[Coord],
    control_color: int,
    control: Coord,
) -> np.ndarray:
    lines = {item["color"]: item["cells"] for item in selected}
    return _render_lines(shape, bg, lines, order, bar_color, bar_cells, control_color, control)


def _line_visible_coords(grid: np.ndarray, color: int) -> np.ndarray:
    return np.argwhere(grid == int(color))


def _is_bar_like_visible(coords: np.ndarray) -> bool:
    return _bar_signature(coords) is not None


def _validate_generated_case(
    inp: np.ndarray,
    expected: np.ndarray,
    hidden: dict,
) -> bool:
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    bg = int(hidden["bg"])
    line_colors = [int(item["color"]) for item in hidden["selected"]]
    erased = tuple(map(int, hidden["erased"]))
    kept = tuple(map(int, hidden["kept"]))
    ray = list(hidden["ray"])

    if len(erased) < 2 or not kept:
        return False

    if _get_bg(inp) != bg:
        return False

    for color in line_colors:
        coords = _line_visible_coords(inp, color)
        if coords.shape[0] < 2:
            return False
        try:
            _infer_line_signature(coords)
        except AssertionError:
            return False
        if _is_bar_like_visible(coords):
            return False

    initial_visible_hit_colors = {
        int(inp[r, c])
        for r, c in ray
        if int(inp[r, c]) in set(line_colors)
    }
    if not (set(erased) - initial_visible_hit_colors):
        return False

    delete_only = inp.copy()
    for color in erased:
        delete_only[delete_only == color] = bg
    if np.array_equal(delete_only, expected):
        return False

    lines = {int(item["color"]): item["cells"] for item in hidden["selected"]}
    simple_kept = tuple(color for color in hidden["order"] if color not in initial_visible_hit_colors)
    simple_output = _render_lines(
        inp.shape,
        bg,
        lines,
        simple_kept,
        hidden["bar_color"],
        hidden["bar_cells"],
        hidden["control_color"],
        hidden["control"],
    )
    if np.array_equal(simple_output, expected):
        return False

    newly_visible = (expected != inp) & np.isin(expected, np.array(kept, dtype=int))
    if not np.any(newly_visible):
        return False

    return True


def _try_generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    shape = (h, w)

    marker_options = _marker_options(h, w)
    if not marker_options:
        return None
    bar_cells, control, center, ray = random.choice(marker_options)
    reserved = set(bar_cells) | {control}

    candidates = _candidate_lines(shape, reserved, ray)
    if not candidates:
        return None

    by_ray_cell: dict[Coord, list[dict]] = defaultdict(list)
    for cand in candidates:
        if cand["hits_ray"]:
            by_ray_cell[cand["ray_cell"]].append(cand)

    stack_cells = [cell for cell, items in by_ray_cell.items() if len(items) >= 2]
    if not stack_cells:
        return None
    stack_cell = random.choice(stack_cells)
    stack_pool = by_ray_cell[stack_cell][:]
    random.shuffle(stack_pool)
    stack_size = random.choice([size for size in (2, 3) if size <= len(stack_pool)])
    hit_selected = stack_pool[:stack_size]
    used = {item["sig"] for item in hit_selected}

    total_lines = random.randint(*LINE_COUNT_RANGE)
    max_extra_hits = min(1, total_lines - len(hit_selected) - 1)
    extra_hit_count = random.randint(0, max_extra_hits) if max_extra_hits >= 0 else 0
    extra_hit_pool = [
        cand
        for cand in candidates
        if cand["hits_ray"] and cand["ray_cell"] != stack_cell and cand["sig"] not in used
    ]
    extra_hits = _pick_distinct(extra_hit_pool, extra_hit_count)
    if extra_hits is None:
        return None
    hit_selected.extend(extra_hits)
    used |= {item["sig"] for item in extra_hits}

    non_hit_count = total_lines - len(hit_selected)
    non_hit_pool = [
        cand
        for cand in candidates
        if not cand["hits_ray"] and cand["sig"] not in used and any(_intersects(cand, hit) for hit in hit_selected)
    ]
    if len(non_hit_pool) < non_hit_count:
        return None
    random.shuffle(non_hit_pool)
    selected = hit_selected + non_hit_pool[:non_hit_count]

    random.shuffle(selected)
    if len({item["sig"] for item in selected}) != len(selected):
        return None

    palette = random.sample(ALL_COLORS, len(selected) + 3)
    bg, bar_color, control_color = palette[:3]
    for item, color in zip(selected, palette[3:]):
        item["color"] = int(color)

    lines = {int(item["color"]): item["cells"] for item in selected}
    line_colors = tuple(lines)

    for _ in range(ORDER_ATTEMPTS):
        order = tuple(random.sample(list(line_colors), len(line_colors)))
        inp = _render_lines(shape, bg, lines, order, bar_color, bar_cells, control_color, control)
        erased, kept = _erase_chain(lines, order, ray)
        if set(erased) != {int(item["color"]) for item in hit_selected}:
            continue
        expected = _render_lines(shape, bg, lines, kept, bar_color, bar_cells, control_color, control)
        hidden = {
            "bg": bg,
            "selected": selected,
            "order": order,
            "erased": erased,
            "kept": kept,
            "bar_color": bar_color,
            "bar_cells": bar_cells,
            "control_color": control_color,
            "control": control,
            "ray": ray,
        }
        if _validate_generated_case(inp, expected, hidden):
            return inp, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _try_generate_case()
        if case is not None:
            inp, _expected = case
            return inp
    raise RuntimeError("failed to generate a valid chain-reveal ray-clearing puzzle")
