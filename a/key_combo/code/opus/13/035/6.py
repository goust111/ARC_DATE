from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
RAY_COLORS = (BLUE, RED, GREEN)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_input_rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for coords in _component_coords(grid == YELLOW):
        r0, c0, r1, c1 = _bbox(coords)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert int(coords.shape[0]) == area, "yellow objects must be solid rectangles"
        rects.append({"bbox": (r0, c0, r1, c1), "coords": coords})
    assert 2 <= len(rects) <= 3, "expected 2-3 yellow rectangles"
    return rects


def _ray_priority(ray: dict) -> tuple[int, int, int]:
    if ray["orientation"] == "h":
        return (0, int(ray["index"]), 0)
    return (int(ray["index"]), 0, 1)


def _parse_rays(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    rays: list[dict] = []
    endpoint_cells: set[tuple[int, int]] = set()

    for r in range(1, h - 1):
        color = int(g[r, 0])
        if color != bg and color != YELLOW and int(g[r, w - 1]) == color:
            ray = {"orientation": "h", "index": r, "color": color}
            ray["priority"] = _ray_priority(ray)
            rays.append(ray)
            endpoint_cells.add((r, 0))
            endpoint_cells.add((r, w - 1))

    for c in range(1, w - 1):
        color = int(g[0, c])
        if color != bg and color != YELLOW and int(g[h - 1, c]) == color:
            ray = {"orientation": "v", "index": c, "color": color}
            ray["priority"] = _ray_priority(ray)
            rays.append(ray)
            endpoint_cells.add((0, c))
            endpoint_cells.add((h - 1, c))

    assert 3 <= len(rays) <= 5, "expected 3-5 border ray pairs"

    for ray in rays:
        same_axis = [other for other in rays if other["orientation"] == ray["orientation"]]
        indices = [int(other["index"]) for other in same_axis]
        assert len(indices) == len(set(indices)), "duplicate ray row/col is not allowed"

    color_counts = Counter(int(ray["color"]) for ray in rays)
    assert all(count <= 2 for count in color_counts.values()), "each ray color may appear at most twice"

    border_marks: set[tuple[int, int]] = set()
    for r in range(h):
        for c in (0, w - 1):
            if int(g[r, c]) != bg and int(g[r, c]) != YELLOW:
                border_marks.add((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if int(g[r, c]) != bg and int(g[r, c]) != YELLOW:
                border_marks.add((r, c))
    assert border_marks == endpoint_cells, "all colored border marks must belong to opposite-edge pairs"
    return rays


def _ray_maps(rays: list[dict]) -> tuple[dict[int, dict], dict[int, dict], dict[int, tuple[int, int, int]]]:
    h_rays = {int(ray["index"]): ray for ray in rays if ray["orientation"] == "h"}
    v_rays = {int(ray["index"]): ray for ray in rays if ray["orientation"] == "v"}
    color_priority: dict[int, tuple[int, int, int]] = {}
    for ray in rays:
        color = int(ray["color"])
        prio = tuple(int(x) for x in ray["priority"])
        if color not in color_priority or prio < color_priority[color]:
            color_priority[color] = prio
    return h_rays, v_rays, color_priority


def _paint_rays(grid: np.ndarray, rays: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for ray in sorted(rays, key=lambda item: item["priority"], reverse=True):
        color = int(ray["color"])
        if ray["orientation"] == "h":
            out[int(ray["index"]), :] = color
        else:
            out[:, int(ray["index"])] = color
    return out


def _colors_on_cell(r: int, c: int, h_rays: dict[int, dict], v_rays: dict[int, dict]) -> set[int]:
    colors: set[int] = set()
    if r in h_rays:
        colors.add(int(h_rays[r]["color"]))
    if c in v_rays:
        colors.add(int(v_rays[c]["color"]))
    return colors


def _choose_fragment_color(
    counts: Counter[int], color_priority: dict[int, tuple[int, int, int]]
) -> tuple[int, bool]:
    best = max(counts.values())
    winners = [int(color) for color, count in counts.items() if count == best]
    winners.sort(key=lambda color: color_priority[color])
    return winners[0], len(winners) > 1


def _solve_with_analysis(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must fit within 30x30"

    rects = _find_input_rectangles(g)
    rays = _parse_rays(g)
    h_rays, v_rays, color_priority = _ray_maps(rays)
    painted = _paint_rays(g, rays)
    out = painted.copy()

    rect_infos: list[dict] = []
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        crossings = {
            ray["orientation"]
            for ray in rays
            if (
                ray["orientation"] == "h"
                and r0 <= int(ray["index"]) <= r1
            )
            or (
                ray["orientation"] == "v"
                and c0 <= int(ray["index"]) <= c1
            )
        }

        local_mask = np.zeros((h, w), dtype=bool)
        local_mask[r0 : r1 + 1, c0 : c1 + 1] = painted[r0 : r1 + 1, c0 : c1 + 1] == YELLOW
        fragment_infos: list[dict] = []
        for coords in _component_coords(local_mask):
            counts: Counter[int] = Counter()
            for rr, cc in coords:
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w:
                        for color in _colors_on_cell(nr, nc, h_rays, v_rays):
                            counts[color] += 1

            chosen = None
            is_tie = False
            if counts:
                chosen, is_tie = _choose_fragment_color(counts, color_priority)
                out[coords[:, 0], coords[:, 1]] = chosen

            fragment_infos.append(
                {
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "adjacency": dict(counts),
                    "chosen_color": chosen,
                    "is_tie": is_tie,
                }
            )

        rect_infos.append(
            {
                "bbox": rect["bbox"],
                "crossings": crossings,
                "fragments": fragment_infos,
            }
        )

    return {
        "output": out,
        "painted": painted,
        "rays": rays,
        "rectangles": rect_infos,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend opposite-edge corridor rays, split yellow rectangles where the rays pass, then recolor each remaining fragment by its strongest adjacent ray color.

    Input structure:
    - The background is black, with 2-3 non-overlapping solid yellow rectangles inside the grid.
    - Colored border pixels appear in opposite-edge pairs on the same row or the same column.
    - Each pair defines one full horizontal or vertical corridor ray, and no two rays share a row or a column.

    Rule:
    - For every opposite-edge same-color pair, draw its 1-cell-wide ray across the entire shared row or column, overwriting any yellow cells it passes through.
    - The overwritten cells split a yellow rectangle into 4-connected yellow fragments.
    - Recolor each yellow fragment by the adjacent ray color with the largest total 4-neighbor contact length; if several colors tie, prefer the color whose border pair has the smaller left/top endpoint column.
    - Rays keep their own colors, untouched yellow rectangles stay yellow, and black background stays black.

    Key insight:
    - The answer is determined by a dependency chain: border pairs create full rays, the rays carve rectangles into emergent pieces, and each piece is colored only after measuring which neighboring ray dominates its boundary.
    """
    return _solve_with_analysis(grid)["output"]


def _rect_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _interior_rows(rect: tuple[int, int, int, int]) -> list[int]:
    r0, _c0, r1, _c1 = rect
    return list(range(r0 + 1, r1))


def _interior_cols(rect: tuple[int, int, int, int]) -> list[int]:
    _r0, c0, _r1, c1 = rect
    return list(range(c0 + 1, c1))


def _row_inside_other_rects(row: int, rects: list[tuple[int, int, int, int]]) -> bool:
    return any(r0 <= row <= r1 for r0, _c0, r1, _c1 in rects)


def _col_inside_other_rects(col: int, rects: list[tuple[int, int, int, int]]) -> bool:
    return any(c0 <= col <= c1 for _r0, c0, _r1, c1 in rects)


def _candidate_rectangles(
    h: int,
    w: int,
    existing: list[tuple[int, int, int, int]],
    used_rows: set[int],
    used_cols: set[int],
    pattern: str,
) -> list[tuple[tuple[int, int, int, int], list[int], list[int]]]:
    need_h = "H" in pattern
    need_v = "V" in pattern
    candidates: list[tuple[tuple[int, int, int, int], list[int], list[int]]] = []
    for height in range(4, 7):
        for width in range(4, 7):
            for r0 in range(2, h - height - 1):
                for c0 in range(2, w - width - 1):
                    rect = (r0, c0, r0 + height - 1, c0 + width - 1)
                    if any(_rect_close(rect, other) for other in existing):
                        continue
                    if any(r0 <= row <= rect[2] for row in used_rows):
                        continue
                    if any(c0 <= col <= rect[3] for col in used_cols):
                        continue
                    rows = [
                        row
                        for row in _interior_rows(rect)
                        if row not in used_rows and not _row_inside_other_rects(row, existing)
                    ]
                    cols = [
                        col
                        for col in _interior_cols(rect)
                        if col not in used_cols and not _col_inside_other_rects(col, existing)
                    ]
                    if need_h and not rows:
                        continue
                    if need_v and not cols:
                        continue
                    candidates.append((rect, rows, cols))
    return candidates


def _anchor_cut_candidates(rect: tuple[int, int, int, int], tie_case: bool) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    candidates: list[tuple[int, int]] = []
    for dr in range(1, height - 1):
        for dc in range(1, width - 1):
            top = dr
            bottom = height - dr - 1
            left = dc
            right = width - dc - 1
            edges = [(left, top), (right, top), (left, bottom), (right, bottom)]
            has_tie = any(a == b for a, b in edges)
            winners = set("H" if a >= b else "V" for a, b in edges)
            if tie_case != has_tie:
                continue
            if winners != {"H", "V"}:
                continue
            candidates.append((r0 + dr, c0 + dc))
    return candidates


def _assign_colors(ray_count: int) -> list[int]:
    colors = list(RAY_COLORS)
    first = random.sample(colors, 2)
    assigned = [first[0], first[1]]
    counts = Counter(assigned)
    while len(assigned) < ray_count:
        choices = [color for color in colors if counts[color] < 2]
        weights = [3 if counts[color] == 1 else 2 for color in choices]
        color = random.choices(choices, weights=weights, k=1)[0]
        assigned.append(color)
        counts[color] += 1
    return assigned


def _build_input_grid(
    h: int,
    w: int,
    rects: list[tuple[int, int, int, int]],
    rays: list[dict],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r0, c0, r1, c1 in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = YELLOW
    for ray in rays:
        color = int(ray["color"])
        if ray["orientation"] == "h":
            row = int(ray["index"])
            grid[row, 0] = color
            grid[row, w - 1] = color
        else:
            col = int(ray["index"])
            grid[0, col] = color
            grid[h - 1, col] = color
    return grid


def _validate_generated(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    patterns: list[str],
    tie_case: bool,
) -> bool:
    analysis = _solve_with_analysis(grid)
    if not np.array_equal(main(grid), analysis["output"]):
        return False

    rect_infos = analysis["rectangles"]
    if len(rect_infos) != len(rects):
        return False

    for info, rect in zip(rect_infos, rects):
        if tuple(info["bbox"]) != tuple(rect):
            return False

    if rect_infos[0]["crossings"] != {"h", "v"}:
        return False
    anchor_fragments = rect_infos[0]["fragments"]
    if len(anchor_fragments) < 4:
        return False
    anchor_colors = {frag["chosen_color"] for frag in anchor_fragments if frag["chosen_color"] is not None}
    if len(anchor_colors) < 2:
        return False
    if tie_case and not any(frag["is_tie"] for frag in anchor_fragments):
        return False
    if not tie_case and any(frag["is_tie"] for frag in anchor_fragments):
        return False

    for idx, pattern in enumerate(patterns, start=1):
        info = rect_infos[idx]
        if pattern == "U":
            if info["crossings"]:
                return False
            r0, c0, r1, c1 = info["bbox"]
            if not np.all(analysis["output"][r0 : r1 + 1, c0 : c1 + 1] == YELLOW):
                return False
            continue

        expected = set()
        if "H" in pattern:
            expected.add("h")
        if "V" in pattern:
            expected.add("v")
        if info["crossings"] != expected:
            return False
        if not info["fragments"]:
            return False
        crossing_colors = {
            int(ray["color"])
            for ray in analysis["rays"]
            if (
                ray["orientation"] == "h"
                and "H" in pattern
                and info["bbox"][0] <= int(ray["index"]) <= info["bbox"][2]
            )
            or (
                ray["orientation"] == "v"
                and "V" in pattern
                and info["bbox"][1] <= int(ray["index"]) <= info["bbox"][3]
            )
        }
        if not crossing_colors:
            return False
        for frag in info["fragments"]:
            if frag["chosen_color"] not in crossing_colors:
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 18)
    w = random.randint(15, 18)
    tie_case = random.random() < 0.45
    template = random.choice(
        [
            ["H"],
            ["V"],
            ["H", "U"],
            ["V", "U"],
            ["H", "V"],
            ["HV", "U"],
            ["HV", "H"],
        ]
    )

    rects: list[tuple[int, int, int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    anchor_candidates = _candidate_rectangles(h, w, [], set(), set(), "HV")
    random.shuffle(anchor_candidates)
    anchor_rect = None
    anchor_row = None
    anchor_col = None
    for rect, _rows, _cols in anchor_candidates:
        cuts = _anchor_cut_candidates(rect, tie_case)
        if cuts:
            anchor_rect = rect
            anchor_row, anchor_col = random.choice(cuts)
            break
    if anchor_rect is None or anchor_row is None or anchor_col is None:
        return None

    rects.append(anchor_rect)
    used_rows.add(anchor_row)
    used_cols.add(anchor_col)

    extra_rays: list[tuple[str, int]] = []
    for pattern in template:
        candidates = _candidate_rectangles(h, w, rects, used_rows, used_cols, pattern)
        if not candidates:
            return None
        rect, row_choices, col_choices = random.choice(candidates)
        rects.append(rect)
        if "H" in pattern:
            row = random.choice(row_choices)
            used_rows.add(row)
            extra_rays.append(("h", row))
        if "V" in pattern:
            col = random.choice(col_choices)
            used_cols.add(col)
            extra_rays.append(("v", col))

    ray_specs = [("h", anchor_row), ("v", anchor_col), *extra_rays]
    colors = _assign_colors(len(ray_specs))
    rays = []
    for (orientation, index), color in zip(ray_specs, colors):
        ray = {"orientation": orientation, "index": index, "color": color}
        ray["priority"] = _ray_priority(ray)
        rays.append(ray)

    grid = _build_input_grid(h, w, rects, rays)
    if not _validate_generated(grid, rects, template, tie_case):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
