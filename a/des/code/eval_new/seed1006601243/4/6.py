from __future__ import annotations

import random

import numpy as np


AZURE = 8
YELLOW = 4
OBJECT_COLORS = (0, 1, 2, 3, 5, 6, 7, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def _component_count(cells: set[tuple[int, int]]) -> int:
    if not cells:
        return 0

    remaining = set(cells)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIR4:
                nbr = (r + dr, c + dc)
                if nbr in remaining:
                    remaining.remove(nbr)
                    stack.append(nbr)
    return count


def _shift_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _detect_lines(g: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = g.shape
    horizontal = [r for r in range(1, h - 1) if int(g[r, 0]) == YELLOW and int(g[r, w - 1]) == YELLOW]
    vertical = [c for c in range(1, w - 1) if int(g[0, c]) == YELLOW and int(g[h - 1, c]) == YELLOW]

    expected = {(r, 0) for r in horizontal}
    expected |= {(r, w - 1) for r in horizontal}
    expected |= {(0, c) for c in vertical}
    expected |= {(h - 1, c) for c in vertical}

    actual = {tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)}
    assert actual == expected, "yellow markers must appear only as edge pairs"
    return horizontal, vertical


def _horizontal_repair(cells: set[tuple[int, int]], line_r: int) -> set[tuple[int, int]] | None:
    if any(r == line_r for r, _c in cells):
        return None

    top = {(r, c) for r, c in cells if r < line_r}
    bottom = {(r, c) for r, c in cells if r > line_r}
    if not top or not bottom:
        return None
    if not any(r == line_r - 1 for r, _c in top):
        return None
    if not any(r == line_r + 1 for r, _c in bottom):
        return None
    if _component_count(top) != 1 or _component_count(bottom) != 1:
        return None

    repaired = top | _shift_cells(bottom, -1, 0)
    if _component_count(repaired) != 1:
        return None
    return repaired


def _vertical_repair(cells: set[tuple[int, int]], line_c: int) -> set[tuple[int, int]] | None:
    if any(c == line_c for _r, c in cells):
        return None

    left = {(r, c) for r, c in cells if c < line_c}
    right = {(r, c) for r, c in cells if c > line_c}
    if not left or not right:
        return None
    if not any(c == line_c - 1 for _r, c in left):
        return None
    if not any(c == line_c + 1 for _r, c in right):
        return None
    if _component_count(left) != 1 or _component_count(right) != 1:
        return None

    repaired = left | _shift_cells(right, 0, -1)
    if _component_count(repaired) != 1:
        return None
    return repaired


def main(grid: np.ndarray) -> np.ndarray:
    """Rejoin each monochrome fragment pair by inferring the hidden horizontal or vertical erase line from yellow edge markers.

    Input structure:
    - The background is azure.
    - Yellow marker pixels appear in edge pairs: left/right on one row for a horizontal erase line, or top/bottom on one column for a vertical erase line.
    - Other colors form scattered monochrome objects; some colors are split into exactly two pieces by one marked erase line, while untouched objects stay connected.

    Rule:
    - Remove all yellow markers.
    - For each non-yellow color, test whether one marked row separates it into an upper and lower piece with a one-cell azure gap; if so, shift the lower piece up by one cell.
    - Otherwise test whether one marked column separates it into a left and right piece with a one-cell azure gap; if so, shift the right piece left by one cell.
    - Colors that are not uniquely explained by any marked erase line remain unchanged.

    Key insight:
    - The yellow pixels do not mark fragments themselves; they identify hidden cut lines, and a color should move only when its two disconnected same-color pieces sit on opposite sides of one such line and become one connected shape after a one-cell repair shift.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    horizontal, vertical = _detect_lines(g)
    out = np.full_like(g, AZURE)
    colors = [int(color) for color in np.unique(g) if int(color) not in (AZURE, YELLOW)]

    for color in colors:
        cells = {tuple(map(int, rc)) for rc in np.argwhere(g == color)}
        repaired: set[tuple[int, int]] | None = None
        if _component_count(cells) == 2:
            candidates: list[set[tuple[int, int]]] = []
            for line_r in horizontal:
                candidate = _horizontal_repair(cells, line_r)
                if candidate is not None:
                    candidates.append(candidate)
            for line_c in vertical:
                candidate = _vertical_repair(cells, line_c)
                if candidate is not None:
                    candidates.append(candidate)
            assert len(candidates) <= 1, "each color should match at most one erase line"
            if candidates:
                repaired = candidates[0]

        for r, c in repaired if repaired is not None else cells:
            out[r, c] = color

    return out


def _sample_spaced_positions(low: int, high: int, count: int, min_gap: int) -> list[int] | None:
    if count == 0:
        return []

    candidates = list(range(low, high + 1))
    random.shuffle(candidates)
    chosen: list[int] = []
    for pos in candidates:
        if all(abs(pos - other) >= min_gap for other in chosen):
            chosen.append(pos)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _grow_connected_shape(height: int, width: int, area: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return cells


def _sample_cut_shape(orientation: str) -> dict[str, object] | None:
    for _ in range(200):
        height = random.randint(4, 7)
        width = random.randint(4, 7)
        area_lo = max(5, (height * width) // 2)
        area = random.randint(area_lo, height * width)
        whole = _grow_connected_shape(height, width, area)
        if whole is None or _component_count(whole) != 1:
            continue

        if orientation == "h":
            cut = random.randint(1, height - 2)
            top = {(r, c) for r, c in whole if r < cut}
            bottom = {(r, c) for r, c in whole if r >= cut}
            if min(len(top), len(bottom)) < 2:
                continue
            if _component_count(top) != 1 or _component_count(bottom) != 1:
                continue
            if not any(r == cut - 1 for r, _c in top):
                continue
            if not any(r == cut for r, _c in bottom):
                continue
            input_cells = top | _shift_cells(bottom, 1, 0)
            if _component_count(input_cells) != 2:
                continue
            return {
                "kind": "cut",
                "orientation": "h",
                "line_offset": cut,
                "bbox_h": height + 1,
                "bbox_w": width,
                "input_local": input_cells,
                "output_local": whole,
            }

        cut = random.randint(1, width - 2)
        left = {(r, c) for r, c in whole if c < cut}
        right = {(r, c) for r, c in whole if c >= cut}
        if min(len(left), len(right)) < 2:
            continue
        if _component_count(left) != 1 or _component_count(right) != 1:
            continue
        if not any(c == cut - 1 for _r, c in left):
            continue
        if not any(c == cut for _r, c in right):
            continue
        input_cells = left | _shift_cells(right, 0, 1)
        if _component_count(input_cells) != 2:
            continue
        return {
            "kind": "cut",
            "orientation": "v",
            "line_offset": cut,
            "bbox_h": height,
            "bbox_w": width + 1,
            "input_local": input_cells,
            "output_local": whole,
        }

    return None


def _sample_untouched_shape() -> dict[str, object] | None:
    for _ in range(120):
        height = random.randint(2, 6)
        width = random.randint(2, 6)
        area = random.randint(3, max(3, height * width - 1))
        cells = _grow_connected_shape(height, width, area)
        if cells is None or _component_count(cells) != 1:
            continue
        return {
            "kind": "free",
            "bbox_h": height,
            "bbox_w": width,
            "input_local": cells,
            "output_local": cells,
        }
    return None


def _translate(cells: set[tuple[int, int]], top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in cells}


def _clear_with_gap(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    for r, c in cells:
        if (r, c) in occupied:
            return False
        for dr, dc in DIR8:
            if (r + dr, c + dc) in occupied:
                return False
    return True


def _respects_lines(
    cells: set[tuple[int, int]],
    horizontal: list[int],
    vertical: list[int],
    assigned: tuple[str, int] | None,
) -> bool:
    if any(r in horizontal for r, _c in cells):
        return False
    if any(c in vertical for _r, c in cells):
        return False

    for line_r in horizontal:
        crosses = any(r < line_r for r, _c in cells) and any(r > line_r for r, _c in cells)
        if assigned == ("h", line_r):
            if not crosses:
                return False
        elif crosses:
            return False

    for line_c in vertical:
        crosses = any(c < line_c for _r, c in cells) and any(c > line_c for _r, c in cells)
        if assigned == ("v", line_c):
            if not crosses:
                return False
        elif crosses:
            return False

    return True


def _place_object(
    spec: dict[str, object],
    color: int,
    height: int,
    width: int,
    horizontal: list[int],
    vertical: list[int],
    input_occupied: set[tuple[int, int]],
    output_occupied: set[tuple[int, int]],
) -> dict[str, object] | None:
    bbox_h = int(spec["bbox_h"])
    bbox_w = int(spec["bbox_w"])
    input_local = spec["input_local"]
    output_local = spec["output_local"]
    kind = spec["kind"]

    placements: list[tuple[int, int]] = []
    if kind == "cut":
        orientation = str(spec["orientation"])
        line_offset = int(spec["line_offset"])
        if orientation == "h":
            line_r = int(spec["line"])
            top = line_r - line_offset
            if not (1 <= top <= height - bbox_h - 1):
                return None
            placements = [(top, left) for left in range(1, width - bbox_w)]
        else:
            line_c = int(spec["line"])
            left = line_c - line_offset
            if not (1 <= left <= width - bbox_w - 1):
                return None
            placements = [(top, left) for top in range(1, height - bbox_h)]
    else:
        placements = [
            (top, left)
            for top in range(1, height - bbox_h)
            for left in range(1, width - bbox_w)
        ]

    random.shuffle(placements)
    assigned = None
    if kind == "cut":
        assigned = (str(spec["orientation"]), int(spec["line"]))

    for top, left in placements:
        input_cells = _translate(input_local, top, left)
        output_cells = _translate(output_local, top, left)
        if not _respects_lines(input_cells, horizontal, vertical, assigned):
            continue
        if not _clear_with_gap(input_cells, input_occupied):
            continue
        if not _clear_with_gap(output_cells, output_occupied):
            continue
        return {
            "color": color,
            "input_cells": input_cells,
            "output_cells": output_cells,
        }

    return None


def _choose_line_pattern() -> tuple[int, int]:
    return random.choice(
        [
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 0),
            (0, 2),
            (2, 1),
            (1, 2),
        ]
    )


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    h_count, v_count = _choose_line_pattern()
    height = random.randint(max(16, 11 + 4 * h_count), 28)
    width = random.randint(max(16, 11 + 4 * v_count), 28)

    horizontal = _sample_spaced_positions(4, height - 5, h_count, 4)
    vertical = _sample_spaced_positions(4, width - 5, v_count, 4)
    if horizontal is None or vertical is None:
        return None

    object_specs: list[dict[str, object]] = []
    for line_r in horizontal:
        for _ in range(random.randint(1, 2)):
            spec = _sample_cut_shape("h")
            if spec is None:
                return None
            spec["line"] = line_r
            object_specs.append(spec)

    for line_c in vertical:
        for _ in range(random.randint(1, 2)):
            spec = _sample_cut_shape("v")
            if spec is None:
                return None
            spec["line"] = line_c
            object_specs.append(spec)

    max_free = min(2, len(OBJECT_COLORS) - len(object_specs))
    for _ in range(random.randint(0, max_free)):
        spec = _sample_untouched_shape()
        if spec is None:
            return None
        object_specs.append(spec)

    if len(object_specs) > len(OBJECT_COLORS):
        return None

    random.shuffle(object_specs)
    object_specs.sort(
        key=lambda spec: (
            int(spec["bbox_h"]) * int(spec["bbox_w"]),
            int(spec["bbox_h"]) + int(spec["bbox_w"]),
        ),
        reverse=True,
    )
    colors = random.sample(OBJECT_COLORS, len(object_specs))

    placed: list[dict[str, object]] = []
    input_occupied: set[tuple[int, int]] = set()
    output_occupied: set[tuple[int, int]] = set()
    for spec, color in zip(object_specs, colors):
        placement = _place_object(
            spec,
            color,
            height,
            width,
            horizontal,
            vertical,
            input_occupied,
            output_occupied,
        )
        if placement is None:
            return None
        placed.append(placement)
        input_occupied |= placement["input_cells"]
        output_occupied |= placement["output_cells"]

    inp = np.full((height, width), AZURE, dtype=int)
    out = np.full((height, width), AZURE, dtype=int)
    for placement in placed:
        color = int(placement["color"])
        for r, c in placement["input_cells"]:
            inp[r, c] = color
        for r, c in placement["output_cells"]:
            out[r, c] = color

    for line_r in horizontal:
        inp[line_r, 0] = YELLOW
        inp[line_r, width - 1] = YELLOW
    for line_c in vertical:
        inp[0, line_c] = YELLOW
        inp[height - 1, line_c] = YELLOW

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        scene = _build_scene()
        if scene is None:
            continue
        inp, out = scene
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid erase-line reconstruction input")
