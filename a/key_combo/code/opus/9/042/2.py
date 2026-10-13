from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BODY = 1
INDICATOR = 2
MARKER = 5

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in (DIR4 if connectivity == 4 else DIR8)]


def _components_from_mask(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _rotate_rel(coord: tuple[int, int], k: int) -> tuple[int, int]:
    r, c = coord
    k %= 4
    for _ in range(k):
        r, c = c, -r
    return r, c


def _rotate_set(coords: set[tuple[int, int]], k: int) -> set[tuple[int, int]]:
    return {_rotate_rel(coord, k) for coord in coords}


def _translate(coords: set[tuple[int, int]], anchor: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(ar + r, ac + c) for r, c in coords}


def _expand(coords: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _line_info(cells: set[tuple[int, int]]) -> tuple[str, tuple[int, int]]:
    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})
    if len(rows) == 1 and len(cols) in (2, 3):
        return "h", (rows[0], cols[len(cols) // 2])
    if len(cols) == 1 and len(rows) in (2, 3):
        return "v", (rows[len(rows) // 2], cols[0])
    raise ValueError("indicator region must be a straight 2-3 cell line")


def _parse_objects(grid: np.ndarray) -> tuple[dict, list[dict], list[dict]]:
    shape_components = []
    mask = np.isin(grid, [BODY, INDICATOR])
    for cells in _components_from_mask(mask, connectivity=4):
        cell_set = set(cells)
        colors = {int(grid[r, c]) for r, c in cell_set}
        shape_components.append({"cells": cell_set, "colors": colors})

    templates = [comp for comp in shape_components if comp["colors"] == {BODY, INDICATOR}]
    partials = [comp for comp in shape_components if comp["colors"] == {INDICATOR}]
    if len(templates) != 1 or not (2 <= len(partials) <= 3):
        raise ValueError("expected one full template and 2-3 indicator-only copies")

    marker_components = []
    marker_mask = grid == MARKER
    for cells in _components_from_mask(marker_mask, connectivity=8):
        cell_set = set(cells)
        if len(cell_set) != 2:
            raise ValueError("every marker glyph must contain exactly two grey pixels")
        marker_components.append({"cells": cell_set})

    if len(marker_components) != len(partials):
        raise ValueError("expected one marker per partial copy")

    return templates[0], partials, marker_components


def _glyph_rotation(marker_cells: set[tuple[int, int]]) -> int:
    (r0, c0), (r1, c1) = sorted(marker_cells)
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    if (dr, dc) == (0, 1):
        return 0
    if (dr, dc) == (1, 0):
        return 1
    if (dr, dc) == (1, 1):
        return 2
    raise ValueError("unknown marker glyph")


def _adjacent_partials(
    partials: list[dict], marker_components: list[dict]
) -> list[tuple[dict, dict]]:
    pairs = []
    for marker in marker_components:
        hits = []
        for partial in partials:
            red = partial["cells"]
            if any((nr, nc) in red for r, c in marker["cells"] for nr, nc in _neighbors(r, c, 4)):
                hits.append(partial)
        if len(hits) != 1:
            raise ValueError("each marker must touch exactly one partial indicator")
        pairs.append((hits[0], marker))
    if len({id(pair[0]) for pair in pairs}) != len(partials):
        raise ValueError("partials must have unique markers")
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each indicator-only copy by rotating the full template according to its adjacent glyph code.

    Input structure:
    - A black background contains one 4-connected full template made of blue body pixels and a red indicator line.
    - Two or three other 4-connected red-only regions repeat the template's indicator line at other positions.
    - Each red-only region has one adjacent two-pixel grey glyph: horizontal, vertical, or diagonal.

    Rule:
    - Identify the only blue+red component as the full template and the red-only components as partial copies.
    - Decode each adjacent grey glyph as a clockwise rotation: horizontal -> 0 degrees, vertical -> 90 degrees, diagonal -> 180 degrees.
    - Rotate the full template around the center cell of its red indicator line, align the rotated red line with the partial copy, draw only the missing blue body pixels, and then erase every grey glyph to black.

    Key insight:
    - The grey glyph does not choose a color; it chooses the template orientation, and only that orientation tells you which blue pixels must be restored around the shared red anchor.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    template, partials, markers = _parse_objects(g)
    template_red = {(r, c) for r, c in template["cells"] if g[r, c] == INDICATOR}
    template_blue = {(r, c) for r, c in template["cells"] if g[r, c] == BODY}
    _, anchor = _line_info(template_red)
    red_rel = {(r - anchor[0], c - anchor[1]) for r, c in template_red}
    blue_rel = {(r - anchor[0], c - anchor[1]) for r, c in template_blue}

    for partial, marker in _adjacent_partials(partials, markers):
        partial_red = partial["cells"]
        _, partial_anchor = _line_info(partial_red)
        k = _glyph_rotation(marker["cells"])
        rotated_red = _rotate_set(red_rel, k)
        rotated_blue = _rotate_set(blue_rel, k)
        expected_red = _translate(rotated_red, partial_anchor)
        if expected_red != partial_red:
            raise ValueError("marker-coded rotation does not align with the partial indicator")
        for r, c in _translate(rotated_blue, partial_anchor):
            out[r, c] = BODY

    out[out == MARKER] = BG
    return out


def _generate_blue_shape() -> set[tuple[int, int]]:
    candidates = {(r, c) for r in range(1, 5) for c in range(-2, 3)} | {(0, -2), (0, 2)}
    for _ in range(300):
        target = random.randint(5, 8)
        chosen: set[tuple[int, int]] = set()
        red = {(0, -1), (0, 0), (0, 1)}
        frontier = {cell for cell in candidates if any((nr, nc) in red for nr, nc in _neighbors(*cell, 4))}
        while len(chosen) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            chosen.add(cell)
            for nr, nc in _neighbors(*cell, 4):
                if (nr, nc) in candidates and (nr, nc) not in chosen:
                    frontier.add((nr, nc))
        if len(chosen) != target:
            continue
        if not any(r >= 2 for r, _ in chosen):
            continue
        if _rotate_set(chosen, 2) == chosen:
            continue
        if len({_freeze(_rotate_set(chosen, k)) for k in (0, 1, 2)}) < 3:
            continue
        if len({c for _, c in chosen}) < 3:
            continue
        return chosen
    raise RuntimeError("failed to generate asymmetric body")


def _freeze(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(cells))


def _paint(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _valid_full_anchor(
    coords: set[tuple[int, int]],
    anchor: tuple[int, int],
    h: int,
    w: int,
    occupied_with_margin: set[tuple[int, int]],
) -> bool:
    translated = _translate(coords, anchor)
    return (
        all(0 <= r < h and 0 <= c < w for r, c in translated)
        and translated.isdisjoint(occupied_with_margin)
    )


def _marker_candidate_sets(
    red_cells: set[tuple[int, int]],
    full_cells: set[tuple[int, int]],
    k: int,
    h: int,
    w: int,
    occupied_input: set[tuple[int, int]],
) -> list[set[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(red_cells)
    patterns = {
        0: [((0, 0), (0, 1))],
        1: [((0, 0), (1, 0))],
        2: [((0, 0), (1, 1)), ((0, 1), (1, 0))],
    }[k]

    candidates: list[set[tuple[int, int]]] = []
    red_halo = _expand(red_cells, h, w, radius=0)
    occupied_others = occupied_input - red_cells
    for base_r in range(r0 - 2, r1 + 3):
        for base_c in range(c0 - 2, c1 + 3):
            for pattern in patterns:
                cells = {(base_r + dr, base_c + dc) for dr, dc in pattern}
                if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                    continue
                if not cells.isdisjoint(full_cells):
                    continue
                if not cells.isdisjoint(occupied_input):
                    continue
                if not any((nr, nc) in red_halo for r, c in cells for nr, nc in _neighbors(r, c, 4)):
                    continue
                if any((nr, nc) in occupied_others for r, c in cells for nr, nc in _neighbors(r, c, 8)):
                    continue
                candidates.append(cells)
    random.shuffle(candidates)
    return candidates


def _rotation_choices() -> list[int]:
    if random.random() < 0.7:
        return random.sample([0, 1, 2], 3)
    pair = random.choice([(0, 2), (1, 2), (0, 1)])
    return list(pair)


def _analyze_instance(grid: np.ndarray) -> tuple[int, list[int]]:
    template, partials, markers = _parse_objects(grid)
    template_red = {(r, c) for r, c in template["cells"] if grid[r, c] == INDICATOR}
    _, template_anchor = _line_info(template_red)
    red_rel = {(r - template_anchor[0], c - template_anchor[1]) for r, c in template_red}
    rotations = []
    for partial, marker in _adjacent_partials(partials, markers):
        partial_red = partial["cells"]
        _, partial_anchor = _line_info(partial_red)
        rel = {(r - partial_anchor[0], c - partial_anchor[1]) for r, c in partial_red}
        candidates = [k for k in (0, 1, 2) if _rotate_set(red_rel, k) == rel]
        if not candidates:
            raise ValueError("partial red region is not a valid rotation of the template indicator")
        rot = _glyph_rotation(marker["cells"])
        if rot not in candidates:
            raise ValueError("glyph disagrees with the indicator orientation")
        rotations.append(rot)
    return len(partials), rotations


def _generate_instance() -> np.ndarray:
    h = random.randint(20, 26)
    w = random.randint(20, 26)
    grid = np.zeros((h, w), dtype=int)

    red_rel = {(0, -1), (0, 0), (0, 1)}
    blue_rel = _generate_blue_shape()
    full_rel = red_rel | blue_rel

    occupied_output: set[tuple[int, int]] = set()
    occupied_input: set[tuple[int, int]] = set()

    template_candidates = [
        (r, c)
        for r in range(2, h - 2)
        for c in range(2, w - 2)
        if _valid_full_anchor(full_rel, (r, c), h, w, occupied_output)
    ]
    if not template_candidates:
        raise RuntimeError("no room for template")
    template_anchor = random.choice(template_candidates)
    template_full = _translate(full_rel, template_anchor)
    _paint(grid, _translate(blue_rel, template_anchor), BODY)
    _paint(grid, _translate(red_rel, template_anchor), INDICATOR)
    occupied_output |= _expand(template_full, h, w, radius=1)
    occupied_input |= template_full

    rotations = _rotation_choices()
    random.shuffle(rotations)

    for k in rotations:
        rot_red = _rotate_set(red_rel, k)
        rot_blue = _rotate_set(blue_rel, k)
        rot_full = rot_red | rot_blue
        anchors = [
            (r, c)
            for r in range(2, h - 2)
            for c in range(2, w - 2)
            if _valid_full_anchor(rot_full, (r, c), h, w, occupied_output)
        ]
        random.shuffle(anchors)
        placed = False
        for anchor in anchors:
            full_cells = _translate(rot_full, anchor)
            red_cells = _translate(rot_red, anchor)
            marker_choices = _marker_candidate_sets(red_cells, full_cells, k, h, w, occupied_input)
            if not marker_choices:
                continue
            marker_cells = marker_choices[0]
            _paint(grid, red_cells, INDICATOR)
            _paint(grid, marker_cells, MARKER)
            occupied_output |= _expand(full_cells, h, w, radius=1)
            occupied_input |= red_cells | marker_cells
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place a partial copy")

    partial_count, realized_rotations = _analyze_instance(grid)
    if partial_count != len(rotations):
        raise RuntimeError("partial count mismatch")
    if len(rotations) == 3 and set(realized_rotations) != {0, 1, 2}:
        raise RuntimeError("three-copy case must cover all glyph rotations")
    if len(rotations) == 2 and len(set(realized_rotations)) < 2:
        raise RuntimeError("two-copy case must still vary the glyph branch")

    out = main(grid)
    if np.array_equal(grid, out):
        raise RuntimeError("degenerate instance")
    if np.count_nonzero(out == MARKER) != 0:
        raise RuntimeError("markers must be erased")
    if np.count_nonzero(out == BODY) != np.count_nonzero(grid == BODY) + len(rotations) * len(blue_rel):
        raise RuntimeError("each partial must gain one full blue body")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            return _generate_instance()
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
