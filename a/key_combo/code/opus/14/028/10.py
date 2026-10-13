from __future__ import annotations

from itertools import combinations
import random

import numpy as np


BG = 0
BLUE = 1
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8

DIR_TO_COLOR = {"left": GREEN, "down": YELLOW, "up": PINK, "right": TEAL}
COLOR_TO_DIR = {value: key for key, value in DIR_TO_COLOR.items()}
DIRS = tuple(DIR_TO_COLOR)
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == int(color), dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _legend_sequence(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(np.isin(grid, list(COLOR_TO_DIR)))]
    assert 3 <= len(coords) <= 4, "expected 3-4 legend pixels"
    rows = {r for r, _c, _v in coords}
    cols = {c for _r, c, _v in coords}
    if len(rows) == 1:
        coords.sort(key=lambda item: item[1])
    else:
        assert len(cols) == 1, "legend must be a single row or single column"
        coords.sort(key=lambda item: item[0])
    return coords


def _paint_from_bbox_edge(
    grid: np.ndarray,
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: str,
) -> None:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox

    if direction == "left":
        for r in range(r0, r1 + 1):
            c = c0 - 1
            while c >= 0 and int(grid[r, c]) != BLUE:
                if int(grid[r, c]) == BG:
                    out[r, c] = ORANGE
                c -= 1
        return

    if direction == "right":
        for r in range(r0, r1 + 1):
            c = c1 + 1
            while c < w and int(grid[r, c]) != BLUE:
                if int(grid[r, c]) == BG:
                    out[r, c] = ORANGE
                c += 1
        return

    if direction == "up":
        for c in range(c0, c1 + 1):
            r = r0 - 1
            while r >= 0 and int(grid[r, c]) != BLUE:
                if int(grid[r, c]) == BG:
                    out[r, c] = ORANGE
                r -= 1
        return

    assert direction == "down"
    for c in range(c0, c1 + 1):
        r = r1 + 1
        while r < h and int(grid[r, c]) != BLUE:
            if int(grid[r, c]) == BG:
                out[r, c] = ORANGE
            r += 1


def main(grid: np.ndarray) -> np.ndarray:
    """Pair ranked orange sprites with the compact direction legend, then fill each blocker-bounded ray strip in orange.

    Input structure:
    - A black grid contains 3-4 orange 4-connected sprites, each a small 3-5 cell shape.
    - Near the upper-right corner there is a compact 3-4 pixel legend in one row or one column.
    - Legend colors encode directions: green=left, yellow=down, pink=up, teal=right; blue pixels are blockers.

    Rule:
    - Find the orange sprites and sort them by the top-left corner of their bounding boxes, first by row then by column.
    - Read the legend pixels in order, top-to-bottom if vertical or left-to-right if horizontal, and pair them with the sorted sprites.
    - For each sprite, cast rays from the full corresponding bbox edge in the assigned direction, stopping before blue pixels or the grid boundary.
    - Recolor every black pixel crossed by those rays to orange while preserving the legend, the blockers, and the original sprites.

    Key insight:
    - The legend does not say what color to paint; it says which direction each ranked sprite should project, and the blocker geometry then determines the filled strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sprites = _components_of_color(g, ORANGE)
    assert 3 <= len(sprites) <= 4, "expected 3-4 orange sprites"
    sprites_sorted = sorted(sprites, key=lambda coords: _bbox(coords)[:2])

    legend = _legend_sequence(g)
    assert len(legend) == len(sprites_sorted), "legend length must match sprite count"

    out = g.copy()
    for sprite, (_r, _c, color) in zip(sprites_sorted, legend):
        _paint_from_bbox_edge(g, out, _bbox(sprite), COLOR_TO_DIR[int(color)])
    return out


def _normalize(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _connected(cells: tuple[tuple[int, int], ...]) -> bool:
    cell_set = set(cells)
    seen = {cells[0]}
    stack = [cells[0]]
    while stack:
        r, c = stack.pop()
        for nbr in _neighbors4(r, c):
            if nbr in cell_set and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return len(seen) == len(cell_set)


def _build_shape_catalog() -> list[dict]:
    catalog: dict[tuple[tuple[int, int], ...], dict] = {}
    all_cells = [(r, c) for r in range(3) for c in range(3)]
    for area in range(3, 6):
        for picked in combinations(all_cells, area):
            if not _connected(picked):
                continue
            norm = _normalize(picked)
            rs = [r for r, _c in norm]
            cs = [c for _r, c in norm]
            h = max(rs) + 1
            w = max(cs) + 1
            if h < 2 or w < 2:
                continue
            catalog.setdefault(
                norm,
                {
                    "coords": np.asarray(norm, dtype=int),
                    "cells": set(norm),
                    "h": h,
                    "w": w,
                    "area": area,
                },
            )
    return list(catalog.values())


SHAPES = _build_shape_catalog()


def _edge_complete(shape: dict, direction: str) -> bool:
    cells = shape["cells"]
    h = int(shape["h"])
    w = int(shape["w"])
    if direction == "left":
        return all((r, 0) in cells for r in range(h))
    if direction == "right":
        return all((r, w - 1) in cells for r in range(h))
    if direction == "up":
        return all((0, c) in cells for c in range(w))
    return all((h - 1, c) in cells for c in range(w))


def _legend_positions(h: int, w: int, count: int, vertical: bool) -> list[tuple[int, int]]:
    if vertical:
        col = w - random.choice((2, 3))
        start_r = random.choice((1, 2))
        return [(start_r + i, col) for i in range(count)]
    row = random.choice((1, 2))
    start_c = w - count - random.choice((2, 3))
    return [(row, start_c + i) for i in range(count)]


def _placement_templates(h: int, w: int) -> list[list[tuple[int, int]]]:
    mid_r = max(8, h // 2 + 1)
    low_r = min(h - 5, mid_r + 3)
    mid_c = max(8, w // 2 - 1)
    return [
        [(4, 2), (4, mid_c), (mid_r, 3), (mid_r + 1, mid_c + 1)],
        [(4, 3), (7, mid_c), (mid_r, 2), (low_r, mid_c + 1)],
        [(5, 2), (5, mid_c + 1), (mid_r + 1, 4), (low_r, mid_c - 1)],
    ]


def _sprite_cells(shape: dict, top: int, left: int) -> np.ndarray:
    return shape["coords"] + np.asarray([top, left], dtype=int)


def _rects_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 2) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _cells_free(mask: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(not mask[r, c] for r, c in cells)


def _apply_cells(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        mask[r, c] = True


def _direction_options(sprite: dict, static_mask: np.ndarray) -> list[dict]:
    h, w = static_mask.shape
    r0, c0, r1, c1 = sprite["bbox"]
    options: list[dict] = []
    directions = list(DIRS)
    random.shuffle(directions)

    for direction in directions:
        if direction == "left":
            max_len = c0 - 1
            lengths = list(range(2, min(6, max_len) + 1))
            random.shuffle(lengths)
            for fill_len in lengths:
                wall_c = c0 - fill_len - 1
                fill = [(r, c) for r in range(r0, r1 + 1) for c in range(wall_c + 1, c0)]
                wall = [(r, wall_c) for r in range(r0, r1 + 1)]
                if _cells_free(static_mask, fill + wall):
                    options.append({"direction": direction, "fill": fill, "wall": wall})
            continue

        if direction == "right":
            max_len = w - c1 - 2
            lengths = list(range(2, min(6, max_len) + 1))
            random.shuffle(lengths)
            for fill_len in lengths:
                wall_c = c1 + fill_len + 1
                fill = [(r, c) for r in range(r0, r1 + 1) for c in range(c1 + 1, wall_c)]
                wall = [(r, wall_c) for r in range(r0, r1 + 1)]
                if _cells_free(static_mask, fill + wall):
                    options.append({"direction": direction, "fill": fill, "wall": wall})
            continue

        if direction == "up":
            max_len = r0 - 1
            lengths = list(range(2, min(6, max_len) + 1))
            random.shuffle(lengths)
            for fill_len in lengths:
                wall_r = r0 - fill_len - 1
                fill = [(r, c) for r in range(wall_r + 1, r0) for c in range(c0, c1 + 1)]
                wall = [(wall_r, c) for c in range(c0, c1 + 1)]
                if _cells_free(static_mask, fill + wall):
                    options.append({"direction": direction, "fill": fill, "wall": wall})
            continue

        max_len = h - r1 - 2
        lengths = list(range(2, min(6, max_len) + 1))
        random.shuffle(lengths)
        for fill_len in lengths:
            wall_r = r1 + fill_len + 1
            fill = [(r, c) for r in range(r1 + 1, wall_r) for c in range(c0, c1 + 1)]
            wall = [(wall_r, c) for c in range(c0, c1 + 1)]
            if _cells_free(static_mask, fill + wall):
                options.append({"direction": direction, "fill": fill, "wall": wall})

    return options


def _assign_rays(sprites: list[dict], static_mask: np.ndarray) -> list[dict] | None:
    option_lists = [_direction_options(sprite, static_mask) for sprite in sprites]
    if any(not options for options in option_lists):
        return None

    def rec(idx: int, taken: np.ndarray, chosen: list[dict], has_edge_gap: bool) -> list[dict] | None:
        if idx == len(sprites):
            used_dirs = {item["direction"] for item in chosen}
            if len(used_dirs) < 2 or not has_edge_gap:
                return None
            return chosen

        options = option_lists[idx][:]
        random.shuffle(options)
        for option in options:
            cells = option["fill"] + option["wall"]
            if not _cells_free(taken, cells):
                continue
            new_taken = taken.copy()
            _apply_cells(new_taken, cells)
            result = rec(
                idx + 1,
                new_taken,
                chosen + [option],
                has_edge_gap or not _edge_complete(sprites[idx]["shape"], option["direction"]),
            )
            if result is not None:
                return result
        return None

    return rec(0, static_mask.copy(), [], False)


def _distractor_candidates(sprite: dict, used_mask: np.ndarray, forbidden_direction: str) -> list[list[tuple[int, int]]]:
    h, w = used_mask.shape
    r0, c0, r1, c1 = sprite["bbox"]
    candidates: list[list[tuple[int, int]]] = []

    if forbidden_direction != "left" and c0 >= 2:
        col = c0 - 2
        for row in range(r0, r1 + 1):
            cells = [(row, col)]
            if _cells_free(used_mask, cells):
                candidates.append(cells)

    if forbidden_direction != "right" and c1 + 2 < w:
        col = c1 + 2
        for row in range(r0, r1 + 1):
            cells = [(row, col)]
            if _cells_free(used_mask, cells):
                candidates.append(cells)

    if forbidden_direction != "up" and r0 >= 2:
        row = r0 - 2
        for col in range(c0, c1 + 1):
            cells = [(row, col)]
            if _cells_free(used_mask, cells):
                candidates.append(cells)

    if forbidden_direction != "down" and r1 + 2 < h:
        row = r1 + 2
        for col in range(c0, c1 + 1):
            cells = [(row, col)]
            if _cells_free(used_mask, cells):
                candidates.append(cells)

    return candidates


def _manual_output(inp: np.ndarray, assignments: list[dict]) -> np.ndarray:
    out = inp.copy()
    for option in assignments:
        for r, c in option["fill"]:
            assert int(inp[r, c]) == BG
            out[r, c] = ORANGE
    return out


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(18, 22)
    w = random.randint(18, 22)
    count = random.randint(3, 4)
    vertical_legend = bool(random.getrandbits(1))
    legend_positions = _legend_positions(h, w, count, vertical_legend)

    legend_pad = np.zeros((h, w), dtype=bool)
    for r, c in legend_positions:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                legend_pad[rr, cc] = True

    template = random.choice(_placement_templates(h, w))
    base_slots = random.sample(template, count)
    shapes = [random.choice(SHAPES) for _ in range(count)]

    sprites: list[dict] = []
    bboxes: list[tuple[int, int, int, int]] = []
    for base, shape in zip(base_slots, shapes):
        placed = False
        for _ in range(20):
            top = base[0] + random.randint(-1, 1)
            left = base[1] + random.randint(-1, 1)
            if top < 3 or left < 1 or top + shape["h"] >= h - 2 or left + shape["w"] >= w - 3:
                continue
            coords = _sprite_cells(shape, top, left)
            bbox = _bbox(coords)
            if legend_pad[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1].any():
                continue
            if any(not _rects_separated(bbox, other) for other in bboxes):
                continue
            sprites.append({"shape": shape, "coords": coords, "bbox": bbox})
            bboxes.append(bbox)
            placed = True
            break
        if not placed:
            return None

    sprites.sort(key=lambda item: item["bbox"][:2])

    static_mask = np.zeros((h, w), dtype=bool)
    for sprite in sprites:
        _apply_cells(static_mask, [tuple(map(int, cell)) for cell in sprite["coords"]])
    _apply_cells(static_mask, legend_positions)

    assignments = _assign_rays(sprites, static_mask)
    if assignments is None:
        return None

    if random.random() < 0.45 and len({item["direction"] for item in assignments}) == len(assignments):
        return None

    used_mask = static_mask.copy()
    for option in assignments:
        _apply_cells(used_mask, option["wall"])
        _apply_cells(used_mask, option["fill"])

    distractors: list[tuple[int, int]] = []
    owner_order = list(range(len(sprites)))
    random.shuffle(owner_order)
    for owner_idx in owner_order:
        candidates = _distractor_candidates(sprites[owner_idx], used_mask, assignments[owner_idx]["direction"])
        random.shuffle(candidates)
        if not candidates:
            continue
        chosen = candidates[0]
        _apply_cells(used_mask, chosen)
        distractors.extend(chosen)
        break
    if not distractors:
        return None

    inp = np.zeros((h, w), dtype=int)
    for sprite in sprites:
        coords = sprite["coords"]
        inp[coords[:, 0], coords[:, 1]] = ORANGE
    for option in assignments:
        for r, c in option["wall"]:
            inp[r, c] = BLUE
    for r, c in distractors:
        inp[r, c] = BLUE
    for (r, c), option in zip(legend_positions, assignments):
        inp[r, c] = DIR_TO_COLOR[option["direction"]]

    out = _manual_output(inp, assignments)
    if not np.array_equal(main(inp), out):
        return None

    orange_components = _components_of_color(inp, ORANGE)
    if len(orange_components) != count:
        return None
    if any(not (3 <= len(comp) <= 5) for comp in orange_components):
        return None
    if len(_legend_sequence(inp)) != count:
        return None
    if any(set(inp[r, c] for r, c in option["fill"]) != {BG} for option in assignments):
        return None
    if any(not option["fill"] for option in assignments):
        return None
    if any(int(out[r, c]) != ORANGE for option in assignments for r, c in option["fill"]):
        return None
    if any(int(out[r, c]) != BLUE for option in assignments for r, c in option["wall"]):
        return None

    fill_sets = [set(option["fill"]) for option in assignments]
    for i in range(len(fill_sets)):
        for j in range(i + 1, len(fill_sets)):
            if fill_sets[i] & fill_sets[j]:
                return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
