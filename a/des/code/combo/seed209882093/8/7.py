from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 8
DIR_ORDER = ("up", "right", "down", "left")
DIRECTIONS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _perpendiculars(dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return (dc, -dr), (-dc, dr)


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == BG):
        return BG
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _connected_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4((int(rr), int(cc))):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append({"color": color, "cells": tuple(sorted(cells))})

    comps.sort(key=lambda comp: tuple(comp["cells"]))
    return comps


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    queue = deque([start])
    seen = {start}
    while queue:
        cell = queue.popleft()
        for nbr in _neighbors4(cell):
            if nbr in cells and nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)
    return len(seen) == len(cells)


def _is_irregular_source(cells: set[tuple[int, int]]) -> bool:
    if not (3 <= len(cells) <= 8):
        return False
    rows = {r for r, _c in cells}
    cols = {c for _r, c in cells}
    if len(rows) == 1 or len(cols) == 1:
        return False
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)
    return (r1 - r0 + 1) * (c1 - c0 + 1) > len(cells)


def _candidate_arrow(
    cells: set[tuple[int, int]],
    head: tuple[int, int],
    direction: str,
) -> dict[str, object] | None:
    dr, dc = DIRECTIONS[direction]
    (pr1, pc1), (pr2, pc2) = _perpendiculars(dr, dc)

    wing1 = (head[0] + pr1, head[1] + pc1)
    wing2 = (head[0] + pr2, head[1] + pc2)
    front = (head[0] + dr, head[1] + dc)
    if wing1 not in cells or wing2 not in cells or front in cells:
        return None

    shaft: list[tuple[int, int]] = []
    for step in range(1, 5):
        pos = (head[0] - step * dr, head[1] - step * dc)
        if pos in cells:
            shaft.append(pos)
        else:
            break
    if not shaft:
        return None

    behind_root = (head[0] - (len(shaft) + 1) * dr, head[1] - (len(shaft) + 1) * dc)
    if behind_root in cells:
        return None

    arrow = set(shaft)
    arrow.add(head)
    arrow.add(wing1)
    arrow.add(wing2)
    source = cells - arrow
    if not _is_irregular_source(source) or not _is_connected(source):
        return None

    root = shaft[-1]
    root_contacts = [nbr for nbr in _neighbors4(root) if nbr in source]
    if not root_contacts:
        return None

    for arrow_cell in arrow:
        if arrow_cell == root:
            continue
        if any(nbr in source for nbr in _neighbors4(arrow_cell)):
            return None

    return {
        "source": source,
        "arrow": arrow,
        "direction": direction,
        "length": len(shaft),
        "head": head,
        "root_contacts": len(root_contacts),
    }


def _extract_instruction(cells: set[tuple[int, int]]) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for head in sorted(cells):
        for direction in DIR_ORDER:
            cand = _candidate_arrow(cells, head, direction)
            if cand is not None:
                candidates.append(cand)
    if not candidates:
        raise ValueError("component does not contain a valid arrow/source split")

    candidates.sort(
        key=lambda cand: (
            -int(cand["length"]),
            -int(cand["root_contacts"]),
            int(cand["head"][0]),
            int(cand["head"][1]),
            DIR_ORDER.index(str(cand["direction"])),
        )
    )
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat same-colored T-ended arrows as projection instructions for nearby source objects.

    For each monochrome connected component:
    - detect the T-shaped arrow substructure,
    - remove the arrow from the output,
    - keep the source object in place,
    - stamp a translated copy of the source by the arrow-shaft length.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)

    out = g.copy()
    paints: list[tuple[int, set[tuple[int, int]], int, int]] = []

    for comp in _connected_components(g, bg):
        color = int(comp["color"])
        cells = set(comp["cells"])
        try:
            info = _extract_instruction(cells)
        except ValueError:
            continue

        for r, c in info["arrow"]:
            out[int(r), int(c)] = int(bg)

        dr, dc = DIRECTIONS[str(info["direction"])]
        shift_r = dr * int(info["length"])
        shift_c = dc * int(info["length"])
        paints.append((color, set(info["source"]), shift_r, shift_c))

    for color, source, shift_r, shift_c in paints:
        for r, c in source:
            rr = int(r) + int(shift_r)
            cc = int(c) + int(shift_c)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(color)

    return out


def _canonical_arrow(length: int) -> set[tuple[int, int]]:
    arrow = {(0, c) for c in range(length)}
    arrow.add((0, length))
    arrow.add((-1, length))
    arrow.add((1, length))
    return arrow


def _transform_point(point: tuple[int, int], direction: str) -> tuple[int, int]:
    r, c = point
    if direction == "right":
        return (r, c)
    if direction == "left":
        return (r, -c)
    if direction == "down":
        return (c, r)
    if direction == "up":
        return (-c, r)
    raise ValueError(direction)


def _transform_cells(cells: set[tuple[int, int]], direction: str) -> set[tuple[int, int]]:
    return {_transform_point(cell, direction) for cell in cells}


def _translate_cells(cells: set[tuple[int, int]], offset: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = offset
    return {(r + dr, c + dc) for r, c in cells}


def _expand8(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expanded.add((r + dr, c + dc))
    return expanded


def _make_source_shape(length: int) -> set[tuple[int, int]] | None:
    arrow = _canonical_arrow(length)
    root = (0, 0)
    anchor = random.choice([(-1, 0), (1, 0)])

    forbidden = set(arrow)
    for cell in arrow:
        if cell != root:
            forbidden.update(_neighbors4(cell))

    for nbr in _neighbors4(root):
        if nbr != anchor and nbr not in arrow:
            forbidden.add(nbr)

    target_size = random.randint(3, 8)
    cells = {anchor}
    low, high = -5, 5

    for _ in range(target_size - 1):
        frontier: set[tuple[int, int]] = set()
        for cell in cells:
            for nbr in _neighbors4(cell):
                if nbr in cells or nbr in forbidden:
                    continue
                if not (low <= nbr[0] <= high and low <= nbr[1] <= high):
                    continue
                frontier.add(nbr)
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    if not _is_irregular_source(cells):
        return None
    if not _is_connected(cells):
        return None
    return cells


def _make_blueprint(color: int) -> dict[str, object] | None:
    for _ in range(120):
        direction = random.choice(DIR_ORDER)
        length = random.randint(1, 4)
        source_right = _make_source_shape(length)
        if source_right is None:
            continue

        arrow_right = _canonical_arrow(length)
        source = _transform_cells(source_right, direction)
        arrow = _transform_cells(arrow_right, direction)
        dr, dc = DIRECTIONS[direction]
        copy = {(r + length * dr, c + length * dc) for r, c in source}
        if source & copy:
            continue

        component = source | arrow
        try:
            parsed = _extract_instruction(component)
        except ValueError:
            continue

        if set(parsed["source"]) != source:
            continue
        if set(parsed["arrow"]) != arrow:
            continue
        if str(parsed["direction"]) != direction:
            continue
        if int(parsed["length"]) != length:
            continue

        return {
            "color": int(color),
            "direction": direction,
            "length": int(length),
            "source": source,
            "arrow": arrow,
            "component": component,
            "copy": copy,
        }
    return None


def _feasible_offsets(
    component: set[tuple[int, int]],
    copy: set[tuple[int, int]],
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    all_cells = component | copy
    min_r = min(r for r, _c in all_cells)
    max_r = max(r for r, _c in all_cells)
    min_c = min(c for _r, c in all_cells)
    max_c = max(c for _r, c in all_cells)

    offsets = [
        (dr, dc)
        for dr in range(-min_r, height - max_r)
        for dc in range(-min_c, width - max_c)
    ]
    random.shuffle(offsets)
    return offsets


def _place_blueprints(
    blueprints: list[dict[str, object]],
    height: int,
    width: int,
    force_overlap: bool,
) -> list[dict[str, object]] | None:
    order = sorted(range(len(blueprints)), key=lambda i: -len(blueprints[i]["component"]))

    occupied_input: set[tuple[int, int]] = set()
    occupied_sources: set[tuple[int, int]] = set()
    occupied_arrows: set[tuple[int, int]] = set()
    occupied_copies: set[tuple[int, int]] = set()
    blocked_input: set[tuple[int, int]] = set()
    placed_by_idx: dict[int, dict[str, object]] = {}
    overlap_realized = False

    for idx in order:
        bp = blueprints[idx]
        overlap_choices: list[dict[str, object]] = []
        plain_choices: list[dict[str, object]] = []

        for offset in _feasible_offsets(bp["component"], bp["copy"], height, width):
            component = _translate_cells(bp["component"], offset)
            source = _translate_cells(bp["source"], offset)
            arrow = _translate_cells(bp["arrow"], offset)
            copy = _translate_cells(bp["copy"], offset)

            if component & blocked_input:
                continue
            if copy & occupied_copies:
                continue
            if source & copy:
                continue

            arrow_vs_copy = bool(arrow & occupied_copies)
            copy_vs_arrows = bool(copy & occupied_arrows)
            makes_source_overlap = bool(source & occupied_copies or copy & occupied_sources)

            if not force_overlap:
                if component & occupied_copies:
                    continue
                if copy & occupied_input:
                    continue
            else:
                if arrow_vs_copy or copy_vs_arrows:
                    continue

            item = {
                "color": int(bp["color"]),
                "direction": bp["direction"],
                "length": int(bp["length"]),
                "component": component,
                "source": source,
                "arrow": arrow,
                "copy": copy,
                "offset": offset,
                "makes_overlap": makes_source_overlap,
            }
            if makes_source_overlap:
                overlap_choices.append(item)
            else:
                plain_choices.append(item)

        if force_overlap and not overlap_realized:
            if overlap_choices:
                chosen = random.choice(overlap_choices)
            elif plain_choices:
                chosen = random.choice(plain_choices)
            else:
                return None
        else:
            pool = plain_choices + overlap_choices
            if not pool:
                return None
            chosen = random.choice(pool)

        placed_by_idx[idx] = chosen
        occupied_input |= set(chosen["component"])
        occupied_sources |= set(chosen["source"])
        occupied_arrows |= set(chosen["arrow"])
        occupied_copies |= set(chosen["copy"])
        blocked_input |= _expand8(set(chosen["component"]))
        overlap_realized = overlap_realized or bool(chosen["makes_overlap"])

    if force_overlap and not overlap_realized:
        return None

    return [placed_by_idx[i] for i in range(len(blueprints))]


def _render_input(height: int, width: int, placed: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for item in placed:
        color = int(item["color"])
        for r, c in item["component"]:
            grid[int(r), int(c)] = color
    return grid


def _render_output(inp: np.ndarray, placed: list[dict[str, object]]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for item in placed:
        for r, c in item["arrow"]:
            out[int(r), int(c)] = BG
    for item in placed:
        color = int(item["color"])
        for r, c in item["copy"]:
            out[int(r), int(c)] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(14, 24)
    width = random.randint(14, 24)
    n_objects = random.randint(1, 4)
    force_overlap = n_objects >= 2 and random.random() < 0.35

    palette = [color for color in range(10) if color != BG]
    colors = random.sample(palette, n_objects)

    blueprints: list[dict[str, object]] = []
    for color in colors:
        blueprint = _make_blueprint(int(color))
        if blueprint is None:
            return None
        blueprints.append(blueprint)

    placed = _place_blueprints(blueprints, height, width, force_overlap)
    if placed is None:
        return None

    inp = _render_input(height, width, placed)
    out = _render_output(inp, placed)

    if _background_color(inp) != BG:
        return None
    if np.count_nonzero(inp == BG) <= (inp.size // 2):
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
