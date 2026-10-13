from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_VECS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_of_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _collect_components(grid: np.ndarray, bg: int) -> dict[int, list[dict[str, object]]]:
    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components_of_mask(grid == color):
            cells = {(int(r), int(c)) for r, c in comp}
            rows = {r for r, _ in cells}
            cols = {c for _, c in cells}
            by_color[color].append(
                {
                    "color": color,
                    "cells": cells,
                    "rows": rows,
                    "cols": cols,
                }
            )
    return by_color


def _arrow_interpretations(cells: set[tuple[int, int]]) -> list[dict[str, object]]:
    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})
    out: list[dict[str, object]] = []

    if len(cells) >= 3 and len(rows) == 2:
        min_c, max_c = min(cols), max(cols)
        width = max_c - min_c + 1
        if width >= 2 and len(cols) == width and len(cells) == width + 1:
            row_to_cols = {r: sorted(c for rr, c in cells if rr == r) for r in rows}
            full_rows = [r for r in rows if row_to_cols[r] == list(range(min_c, max_c + 1))]
            single_rows = [r for r in rows if len(row_to_cols[r]) == 1]
            if len(full_rows) == 1 and len(single_rows) == 1:
                extra_row = single_rows[0]
                extra_col = row_to_cols[extra_row][0]
                if extra_col in {min_c, max_c}:
                    out.append(
                        {
                            "axis": "row",
                            "anchor": int(full_rows[0]),
                            "direction": "U" if extra_row < full_rows[0] else "D",
                            "length": int(width),
                            "cells": cells,
                        }
                    )

    if len(cells) >= 3 and len(cols) == 2:
        min_r, max_r = min(rows), max(rows)
        height = max_r - min_r + 1
        if height >= 2 and len(rows) == height and len(cells) == height + 1:
            col_to_rows = {c: sorted(r for r, cc in cells if cc == c) for c in cols}
            full_cols = [c for c in cols if col_to_rows[c] == list(range(min_r, max_r + 1))]
            single_cols = [c for c in cols if len(col_to_rows[c]) == 1]
            if len(full_cols) == 1 and len(single_cols) == 1:
                extra_col = single_cols[0]
                extra_row = col_to_rows[extra_col][0]
                if extra_row in {min_r, max_r}:
                    out.append(
                        {
                            "axis": "col",
                            "anchor": int(full_cols[0]),
                            "direction": "L" if extra_col < full_cols[0] else "R",
                            "length": int(height),
                            "cells": cells,
                        }
                    )

    unique: dict[tuple[str, int, str, int], dict[str, object]] = {}
    for item in out:
        key = (
            str(item["axis"]),
            int(item["anchor"]),
            str(item["direction"]),
            int(item["length"]),
        )
        unique[key] = item
    return list(unique.values())


def _resolve_assignments(
    targets: list[dict[str, object]],
    arrow_options: list[list[dict[str, object]]],
) -> list[dict[str, object]] | None:
    ordered = sorted(range(len(arrow_options)), key=lambda i: len(arrow_options[i]))
    assignment: list[dict[str, object] | None] = [None] * len(arrow_options)
    used_targets: set[int] = set()
    solutions: list[list[dict[str, object]]] = []

    def dfs(idx: int) -> None:
        if len(solutions) > 1:
            return
        if idx == len(ordered):
            solutions.append([item for item in assignment if item is not None])
            return
        arrow_idx = ordered[idx]
        for opt in arrow_options[arrow_idx]:
            target_id = int(opt["target_id"])
            if target_id in used_targets:
                continue
            used_targets.add(target_id)
            assignment[arrow_idx] = opt
            dfs(idx + 1)
            assignment[arrow_idx] = None
            used_targets.remove(target_id)

    dfs(0)
    if len(solutions) != 1:
        return None
    return solutions[0]


def _try_parse_with_arrow_color(
    grid: np.ndarray,
    bg: int,
    by_color: dict[int, list[dict[str, object]]],
    arrow_color: int,
) -> dict[str, object] | None:
    arrow_components = by_color[arrow_color]
    if not arrow_components:
        return None

    targets: list[dict[str, object]] = []
    for color, comps in by_color.items():
        if color == arrow_color:
            continue
        targets.extend(comps)
    if not targets:
        return None

    arrow_options: list[list[dict[str, object]]] = []
    for comp in arrow_components:
        options: list[dict[str, object]] = []
        for interp in _arrow_interpretations(comp["cells"]):
            if interp["axis"] == "row":
                matches = [i for i, tgt in enumerate(targets) if int(interp["anchor"]) in tgt["rows"]]
            else:
                matches = [i for i, tgt in enumerate(targets) if int(interp["anchor"]) in tgt["cols"]]
            if len(matches) == 1:
                option = dict(interp)
                option["target_id"] = int(matches[0])
                options.append(option)
        if not options:
            return None
        arrow_options.append(options)

    assignment = _resolve_assignments(targets, arrow_options)
    if assignment is None:
        return None

    return {
        "bg": int(bg),
        "arrow_color": int(arrow_color),
        "targets": targets,
        "arrows": assignment,
    }


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    bg = get_bg(grid)
    by_color = _collect_components(grid, bg)

    parses: list[dict[str, object]] = []
    for color, comps in by_color.items():
        if comps and all(_arrow_interpretations(comp["cells"]) for comp in comps):
            parsed = _try_parse_with_arrow_color(grid, bg, by_color, color)
            if parsed is not None:
                parses.append(parsed)

    assert len(parses) == 1, "expected a unique arrow-color interpretation"
    return parses[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read same-colored L-tip arrows as movement instructions.

    Each arrow is a horizontal or vertical line segment with one perpendicular
    single-pixel protrusion at an endpoint. The protrusion direction gives the
    movement direction, while the straight segment length gives the translation
    distance. A horizontal arrow selects the unique target object sharing its
    shaft row; a vertical arrow selects the unique target sharing its shaft
    column. Arrows are erased, and the selected targets are translated with
    out-of-bounds pixels clipped.
    """
    g = np.asarray(grid, dtype=int)
    parsed = _parse_scene(g)
    bg = int(parsed["bg"])
    arrow_color = int(parsed["arrow_color"])
    targets = list(parsed["targets"])
    arrows = list(parsed["arrows"])

    out = g.copy()
    out[g == arrow_color] = bg

    targeted = {int(arrow["target_id"]) for arrow in arrows}
    for target_id in targeted:
        for r, c in targets[target_id]["cells"]:
            out[r, c] = bg

    h, w = out.shape
    for arrow in arrows:
        target = targets[int(arrow["target_id"])]
        dr, dc = DIR_VECS[str(arrow["direction"])]
        dist = int(arrow["length"])
        color = int(target["color"])
        for r, c in target["cells"]:
            rr = r + dr * dist
            cc = c + dc * dist
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (15, 24)
GRID_W_RANGE = (15, 24)
PAIR_COUNT_RANGE = (2, 5)
ARROW_LENGTH_RANGE = (3, 5)
GAP_RANGE = (1, 3)
TARGET_AREA_RANGE = (2, 6)
TARGET_MAX_SPAN = 4
CLIP_PROB = 0.4
PAIR_TRIES = 80
SHAPE_TRIES = 80
GENERATION_ATTEMPTS = 400


def _normalize_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _rotate_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    max_r = max(r for r, _ in cells)
    return {(c, max_r - r) for r, c in cells}


def _reflect_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    max_c = max(c for _, c in cells)
    return {(r, max_c - c) for r, c in cells}


def _random_target_shape() -> list[tuple[int, int]]:
    for _ in range(SHAPE_TRIES):
        area = random.randint(*TARGET_AREA_RANGE)
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            cells.add(random.choice(list(frontier)))

        if random.random() < 0.5:
            cells = _reflect_shape(cells)
        for _ in range(random.randrange(4)):
            cells = _rotate_shape(cells)
        cells = _normalize_shape(cells)

        max_r = max(r for r, _ in cells)
        max_c = max(c for _, c in cells)
        if max_r + 1 > TARGET_MAX_SPAN or max_c + 1 > TARGET_MAX_SPAN:
            continue
        if _arrow_interpretations(cells):
            continue
        return sorted(cells)
    raise RuntimeError("failed to sample a non-arrow target shape")


def _dilate(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _direction_plan(n_pairs: int) -> list[str]:
    dirs: list[str] = []
    all_dirs = ["U", "D", "L", "R"]
    if n_pairs >= 4:
        dirs.extend(random.sample(all_dirs, 4))
        while len(dirs) < n_pairs:
            dirs.append(random.choice(all_dirs))
    elif n_pairs == 3:
        dirs = [
            random.choice(["U", "D"]),
            random.choice(["L", "R"]),
            random.choice(all_dirs),
        ]
    else:
        dirs = [random.choice(["U", "D"]), random.choice(["L", "R"])]
    random.shuffle(dirs)
    return dirs


def _build_pair(
    shape: list[tuple[int, int]],
    top: int,
    left: int,
    direction: str,
    length: int,
    side: str,
    gap: int,
    align: int,
    tip_end: str,
    h: int,
    w: int,
) -> dict[str, object] | None:
    target_cells = {(top + dr, left + dc) for dr, dc in shape}
    shape_h = max(dr for dr, _ in shape) + 1
    shape_w = max(dc for _, dc in shape) + 1

    if direction in {"U", "D"}:
        shaft_row = top + align
        if side == "left":
            c0 = left - gap - length
        else:
            c0 = left + shape_w + gap
        extra_row = shaft_row - 1 if direction == "U" else shaft_row + 1
        extra_col = c0 if tip_end == "start" else c0 + length - 1
        arrow_cells = {(shaft_row, c0 + i) for i in range(length)}
        arrow_cells.add((extra_row, extra_col))
        axis = "row"
        anchor = shaft_row
    else:
        shaft_col = left + align
        if side == "top":
            r0 = top - gap - length
        else:
            r0 = top + shape_h + gap
        extra_col = shaft_col - 1 if direction == "L" else shaft_col + 1
        extra_row = r0 if tip_end == "start" else r0 + length - 1
        arrow_cells = {(r0 + i, shaft_col) for i in range(length)}
        arrow_cells.add((extra_row, extra_col))
        axis = "col"
        anchor = shaft_col

    all_input = target_cells | arrow_cells
    if any(not (0 <= r < h and 0 <= c < w) for r, c in all_input):
        return None

    dr, dc = DIR_VECS[direction]
    moved_full = [(r + dr * length, c + dc * length) for r, c in target_cells]
    moved_cells = {(r, c) for r, c in moved_full if 0 <= r < h and 0 <= c < w}
    if not moved_cells:
        return None

    return {
        "target_cells": target_cells,
        "arrow_cells": arrow_cells,
        "moved_full": moved_full,
        "moved_cells": moved_cells,
        "axis": axis,
        "anchor": int(anchor),
        "rows": {r for r, _ in target_cells},
        "cols": {c for _, c in target_cells},
    }


def _choose_pair_placement(
    h: int,
    w: int,
    shape: list[tuple[int, int]],
    direction: str,
    blocked_input: set[tuple[int, int]],
    occupied_output: set[tuple[int, int]],
    existing_targets: list[dict[str, object]],
    existing_arrows: list[dict[str, object]],
) -> dict[str, object] | None:
    shape_rows = sorted({r for r, _ in shape})
    shape_cols = sorted({c for _, c in shape})
    shape_h = max(shape_rows) + 1
    shape_w = max(shape_cols) + 1

    for _ in range(PAIR_TRIES):
        length = random.randint(*ARROW_LENGTH_RANGE)
        gap = random.randint(*GAP_RANGE)
        clip = random.random() < CLIP_PROB
        tip_end = random.choice(["start", "end"])
        if direction in {"U", "D"}:
            side = random.choice(["left", "right"])
            align = random.choice(shape_rows)
        else:
            side = random.choice(["top", "bottom"])
            align = random.choice(shape_cols)

        candidates: list[dict[str, object]] = []
        for top in range(h - shape_h + 1):
            for left in range(w - shape_w + 1):
                built = _build_pair(shape, top, left, direction, length, side, gap, align, tip_end, h, w)
                if built is None:
                    continue
                moved_count = len(built["moved_cells"])
                if clip:
                    if moved_count == len(shape) or moved_count == 0:
                        continue
                else:
                    if moved_count != len(shape):
                        continue
                all_input = built["target_cells"] | built["arrow_cells"]
                if any(cell in blocked_input for cell in all_input):
                    continue
                if built["moved_cells"] & occupied_output:
                    continue

                new_axis = str(built["axis"])
                new_anchor = int(built["anchor"])
                new_rows = set(built["rows"])
                new_cols = set(built["cols"])

                bad = False
                for target in existing_targets:
                    if new_axis == "row" and new_anchor in target["rows"]:
                        bad = True
                        break
                    if new_axis == "col" and new_anchor in target["cols"]:
                        bad = True
                        break
                if bad:
                    continue
                for arrow in existing_arrows:
                    if arrow["axis"] == "row" and int(arrow["anchor"]) in new_rows:
                        bad = True
                        break
                    if arrow["axis"] == "col" and int(arrow["anchor"]) in new_cols:
                        bad = True
                        break
                if bad:
                    continue

                built["length"] = int(length)
                built["direction"] = direction
                built["clip"] = bool(clip)
                candidates.append(built)

        if candidates:
            return random.choice(candidates)
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    colors = list(range(10))
    bg = random.choice(colors)
    remaining = [c for c in colors if c != bg]
    arrow_color = random.choice(remaining)
    target_colors = random.sample([c for c in remaining if c != arrow_color], n_pairs)

    grid = np.full((h, w), bg, dtype=int)
    blocked_input: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set()
    existing_targets: list[dict[str, object]] = []
    existing_arrows: list[dict[str, object]] = []

    for direction, color in zip(_direction_plan(n_pairs), target_colors):
        placed = None
        for _ in range(PAIR_TRIES):
            shape = _random_target_shape()
            placed = _choose_pair_placement(
                h,
                w,
                shape,
                direction,
                blocked_input,
                occupied_output,
                existing_targets,
                existing_arrows,
            )
            if placed is not None:
                break
        if placed is None:
            return None

        for r, c in placed["target_cells"]:
            grid[r, c] = color
        for r, c in placed["arrow_cells"]:
            grid[r, c] = arrow_color

        blocked_input |= _dilate(placed["target_cells"] | placed["arrow_cells"], h, w)
        occupied_output |= set(placed["moved_cells"])
        existing_targets.append({"rows": set(placed["rows"]), "cols": set(placed["cols"])})
        existing_arrows.append({"axis": str(placed["axis"]), "anchor": int(placed["anchor"])})

    try:
        parsed = _parse_scene(grid)
        if len(parsed["arrows"]) != n_pairs:
            return None
        if len(parsed["targets"]) != n_pairs:
            return None
        out = main(grid)
        if not isinstance(out, np.ndarray) or out.shape != grid.shape:
            return None
        if np.array_equal(out, grid):
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-push input")
