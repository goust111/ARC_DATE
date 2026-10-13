from __future__ import annotations

from collections import deque
import random

import numpy as np


GREY = 5
LEGEND_H = 4
LEGEND_W = 5
H_PATTERN = np.array([[1, 1], [1, 0], [1, 1]], dtype=int)
V_PATTERN = np.array([[1, 0], [1, 0], [0, 1]], dtype=int)
DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _bg_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).reshape(-1), minlength=10)
    return int(np.argmax(counts))


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _coords_to_patch(cells: set[tuple[int, int]]) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox(cells)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        patch[r - r0, c - c0] = True
    return patch, r0, c0


def _patch_to_cells(patch: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    return {(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(patch)}


def _shift_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(r + dr), int(c + dc)) for r, c in cells}


def _flip_patch(patch: np.ndarray, kind: str) -> np.ndarray:
    if kind == "H":
        return np.fliplr(patch)
    assert kind == "V"
    return np.flipud(patch)


def _flip_cells(cells: set[tuple[int, int]], kind: str) -> set[tuple[int, int]]:
    patch, r0, c0 = _coords_to_patch(cells)
    return _patch_to_cells(_flip_patch(patch, kind), r0, c0)


def _connected_components(
    grid: np.ndarray, bg: int, ignore_mask: np.ndarray | None = None
) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    if ignore_mask is not None:
        seen |= ignore_mask

    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: set[tuple[int, int]] = set()
            while q:
                rr, cc = q.popleft()
                cells.add((int(rr), int(cc)))
                for dr, dc in DIR_TO_DELTA.values():
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "cells": cells})
    return comps


def _parse_symbol(block: np.ndarray, bg: int) -> str | None:
    if np.any((block != bg) & (block != GREY)):
        return None
    mask = (block == GREY).astype(int)
    if np.array_equal(mask, H_PATTERN):
        return "H"
    if np.array_equal(mask, V_PATTERN):
        return "V"
    return None


def _parse_legend(
    grid: np.ndarray, bg: int
) -> tuple[tuple[int, int], dict[int, str], np.ndarray]:
    h, w = grid.shape
    assert h >= LEGEND_H and w >= LEGEND_W, "grid too small for the legend"
    positions = [
        (0, 0),
        (0, w - LEGEND_W),
        (h - LEGEND_H, 0),
        (h - LEGEND_H, w - LEGEND_W),
    ]

    matches: list[tuple[tuple[int, int], dict[int, str], np.ndarray]] = []
    for r0, c0 in positions:
        window = grid[r0 : r0 + LEGEND_H, c0 : c0 + LEGEND_W]
        if window.shape != (LEGEND_H, LEGEND_W):
            continue
        if int(window[0, 2]) != bg or np.any(window[1:, 2] != bg):
            continue

        mapping: dict[int, str] = {}
        ok = True
        for start_c in (0, 3):
            swatch = window[0, start_c : start_c + 2]
            if len(set(map(int, swatch.tolist()))) != 1:
                ok = False
                break
            color = int(swatch[0])
            if color in {bg, GREY}:
                ok = False
                break
            symbol = _parse_symbol(window[1:, start_c : start_c + 2], bg)
            if symbol is None:
                ok = False
                break
            mapping[color] = symbol

        if ok and len(mapping) == 2 and set(mapping.values()) == {"H", "V"}:
            legend_mask = np.zeros_like(grid, dtype=bool)
            legend_mask[r0 : r0 + LEGEND_H, c0 : c0 + LEGEND_W] = True
            matches.append(((int(r0), int(c0)), mapping, legend_mask))

    assert len(matches) == 1, "expected one valid corner legend"
    return matches[0]


def _parse_arrow(cells: set[tuple[int, int]]) -> tuple[str, int] | None:
    if len(cells) < 3:
        return None

    candidates: list[tuple[str, int]] = []
    by_row: dict[int, list[int]] = {}
    by_col: dict[int, list[int]] = {}
    for r, c in cells:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)

    total = len(cells)
    for r, cols in by_row.items():
        if len(cols) != total - 1:
            continue
        cols = sorted(cols)
        if cols != list(range(cols[0], cols[-1] + 1)):
            continue
        shaft = {(r, c) for c in cols}
        extra = list(cells - shaft)
        if len(extra) != 1:
            continue
        er, ec = extra[0]
        if abs(er - r) != 1:
            continue
        if ec == cols[0]:
            candidates.append(("left", len(cols)))
        if ec == cols[-1]:
            candidates.append(("right", len(cols)))

    for c, rows in by_col.items():
        if len(rows) != total - 1:
            continue
        rows = sorted(rows)
        if rows != list(range(rows[0], rows[-1] + 1)):
            continue
        shaft = {(r, c) for r in rows}
        extra = list(cells - shaft)
        if len(extra) != 1:
            continue
        er, ec = extra[0]
        if abs(ec - c) != 1:
            continue
        if er == rows[0]:
            candidates.append(("up", len(rows)))
        if er == rows[-1]:
            candidates.append(("down", len(rows)))

    deduped = list(dict.fromkeys(candidates))
    if len(deduped) != 1:
        return None
    direction, length = deduped[0]
    if not (2 <= length <= 5):
        return None
    return direction, length


def _transform_object(
    cells: set[tuple[int, int]], direction: str, length: int, flip_kind: str
) -> set[tuple[int, int]]:
    dr, dc = DIR_TO_DELTA[direction]
    shifted = _shift_cells(cells, dr * length, dc * length)
    return _flip_cells(shifted, flip_kind)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each colored object by its same-color arrow length, then mirror it using the corner legend.

    Input structure:
    - Black background with exactly two moving colors and one grey legend in a corner.
    - For each moving color, one 4-connected asymmetric object is separated from one same-color arrow.
    - Each arrow is a straight horizontal or vertical segment with a one-pixel perpendicular tip attached to one endpoint.
    - The corner legend uses two top-row color swatches; below them, grey H/V glyphs indicate horizontal or vertical mirroring for those colors.

    Rule:
    - Read the legend to map each moving color to H or V.
    - For each color, identify the arrow direction from the endpoint carrying the perpendicular tip, and take the shaft length L.
    - Translate the same-color object by L cells in that direction.
    - Mirror the translated object inside its own bounding box: H means left-right reflection, V means top-bottom reflection.
    - Remove the original object and arrow, and keep the legend unchanged.

    Key insight:
    - The arrow encodes both direction and distance, while the grey corner legend independently assigns the mirror axis for each color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape
    bg = _bg_color(g)

    (legend_r0, legend_c0), mapping, legend_mask = _parse_legend(g, bg)
    comps = _connected_components(g, bg=bg, ignore_mask=legend_mask)

    by_color: dict[int, list[set[tuple[int, int]]]] = {}
    for comp in comps:
        color = int(comp["color"])
        assert color in mapping, "unexpected foreground color outside the legend"
        by_color.setdefault(color, []).append(comp["cells"])  # type: ignore[arg-type]

    assert set(by_color) == set(mapping), "every legend color must have a moving object and arrow"

    out = np.full((h, w), bg, dtype=int)
    out[legend_r0 : legend_r0 + LEGEND_H, legend_c0 : legend_c0 + LEGEND_W] = g[
        legend_r0 : legend_r0 + LEGEND_H, legend_c0 : legend_c0 + LEGEND_W
    ]

    for color, flip_kind in mapping.items():
        pieces = by_color[color]
        assert len(pieces) == 2, "each moving color must have exactly one object and one arrow"
        parsed = [(piece, _parse_arrow(piece)) for piece in pieces]
        arrows = [(piece, info) for piece, info in parsed if info is not None]
        assert len(arrows) == 1, "arrow detection must be unique"
        arrow_cells, (direction, length) = arrows[0]
        object_cells = next(piece for piece in pieces if piece is not arrow_cells)
        final_cells = _transform_object(object_cells, direction, length, flip_kind)
        for r, c in final_cells:
            assert 0 <= r < h and 0 <= c < w, "transformed object left the grid"
            assert out[r, c] in {bg, color}, "transformed objects cannot overlap"
            out[r, c] = color

    return out


def _random_polyomino(area: int) -> np.ndarray | None:
    board_h = random.randint(3, 4)
    board_w = random.randint(3, 4)
    start = (random.randrange(board_h), random.randrange(board_w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR_TO_DELTA.values():
                nr = r + dr
                nc = c + dc
                if 0 <= nr < board_h and 0 <= nc < board_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    h = max(rows) - min(rows) + 1
    w = max(cols) - min(cols) + 1
    patch = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        patch[r - min(rows), c - min(cols)] = True
    return patch


def _random_shape(flip_kind: str) -> np.ndarray | None:
    for _ in range(120):
        patch = _random_polyomino(random.randint(3, 6))
        if patch is None:
            continue
        if min(patch.shape) < 2:
            continue
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        if np.array_equal(patch, np.flipud(patch)):
            continue
        if flip_kind == "H" and np.array_equal(patch, np.fliplr(patch)):
            continue
        if flip_kind == "V" and np.array_equal(patch, np.flipud(patch)):
            continue
        cells = _patch_to_cells(patch, 0, 0)
        if _parse_arrow(cells) is not None:
            continue
        flipped_cells = _patch_to_cells(_flip_patch(patch, flip_kind), 0, 0)
        if _parse_arrow(flipped_cells) is not None:
            continue
        return patch
    return None


def _build_arrow_cells(
    r0: int, c0: int, direction: str, length: int, tip_side: int
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if direction == "right":
        shaft_row = 1 if tip_side < 0 else 0
        tip_row = 0 if tip_side < 0 else 1
        for dc in range(length):
            cells.add((r0 + shaft_row, c0 + dc))
        cells.add((r0 + tip_row, c0 + length - 1))
        return cells
    if direction == "left":
        shaft_row = 1 if tip_side < 0 else 0
        tip_row = 0 if tip_side < 0 else 1
        for dc in range(length):
            cells.add((r0 + shaft_row, c0 + dc))
        cells.add((r0 + tip_row, c0))
        return cells
    if direction == "up":
        shaft_col = 1 if tip_side < 0 else 0
        tip_col = 0 if tip_side < 0 else 1
        for dr in range(length):
            cells.add((r0 + dr, c0 + shaft_col))
        cells.add((r0, c0 + tip_col))
        return cells

    assert direction == "down"
    shaft_col = 1 if tip_side < 0 else 0
    tip_col = 0 if tip_side < 0 else 1
    for dr in range(length):
        cells.add((r0 + dr, c0 + shaft_col))
    cells.add((r0 + length - 1, c0 + tip_col))
    return cells


def _separated(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in b:
                    return False
    return True


def _all_separated(cells: set[tuple[int, int]], others: list[set[tuple[int, int]]]) -> bool:
    return all(_separated(cells, other) for other in others)


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    row_gap = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
    col_gap = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    return int(row_gap + col_gap)


def _legend_origin(h: int, w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - LEGEND_W
    if corner == "bl":
        return h - LEGEND_H, 0
    assert corner == "br"
    return h - LEGEND_H, w - LEGEND_W


def _render_legend(mapping: dict[int, str], corner: str, h: int, w: int, bg: int) -> tuple[np.ndarray, set[tuple[int, int]]]:
    legend = np.full((LEGEND_H, LEGEND_W), bg, dtype=int)
    items = list(mapping.items())
    random.shuffle(items)
    for slot, (color, symbol) in enumerate(items):
        start_c = 0 if slot == 0 else 3
        legend[0, start_c : start_c + 2] = color
        pattern = H_PATTERN if symbol == "H" else V_PATTERN
        legend[1:, start_c : start_c + 2][pattern.astype(bool)] = GREY

    r0, c0 = _legend_origin(h, w, corner)
    cells = {
        (int(r0 + r), int(c0 + c))
        for r in range(LEGEND_H)
        for c in range(LEGEND_W)
        if int(legend[r, c]) != bg
    }
    return legend, cells


def _candidate_arrow_cells(
    obj_cells: set[tuple[int, int]],
    final_cells: set[tuple[int, int]],
    direction: str,
    length: int,
    h: int,
    w: int,
    blocked: list[set[tuple[int, int]]],
) -> set[tuple[int, int]] | None:
    obj_box = _bbox(obj_cells)
    if direction in {"left", "right"}:
        arrow_h, arrow_w = 2, length
    else:
        arrow_h, arrow_w = length, 2

    candidates: list[set[tuple[int, int]]] = []
    r_lo = max(0, obj_box[0] - arrow_h - 3)
    r_hi = min(h - arrow_h, obj_box[2] + 3)
    c_lo = max(0, obj_box[1] - arrow_w - 3)
    c_hi = min(w - arrow_w, obj_box[3] + 3)
    for r0 in range(r_lo, r_hi + 1):
        for c0 in range(c_lo, c_hi + 1):
            arrow_box = (r0, c0, r0 + arrow_h - 1, c0 + arrow_w - 1)
            gap = _bbox_gap(obj_box, arrow_box)
            if not (1 <= gap <= 3):
                continue
            for tip_side in random.sample([-1, 1], 2):
                arrow_cells = _build_arrow_cells(r0, c0, direction, length, tip_side)
                if not _separated(arrow_cells, obj_cells):
                    continue
                if not _separated(arrow_cells, final_cells):
                    continue
                if not _all_separated(arrow_cells, blocked):
                    continue
                candidates.append(arrow_cells)
    if not candidates:
        return None
    return random.choice(candidates)


def _place_one_color(
    color: int,
    flip_kind: str,
    h: int,
    w: int,
    blocked: list[set[tuple[int, int]]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    _ = color
    for _attempt in range(300):
        final_patch = _random_shape(flip_kind)
        if final_patch is None:
            continue
        input_patch = _flip_patch(final_patch, flip_kind)
        direction = random.choice(list(DIR_TO_DELTA))
        length = random.randint(2, 5)
        dr, dc = DIR_TO_DELTA[direction]
        ph, pw = map(int, final_patch.shape)

        min_final_r0 = max(0, dr * length)
        max_final_r0 = min(h - ph, h - ph + dr * length)
        min_final_c0 = max(0, dc * length)
        max_final_c0 = min(w - pw, w - pw + dc * length)
        if min_final_r0 > max_final_r0 or min_final_c0 > max_final_c0:
            continue

        final_r0 = random.randint(min_final_r0, max_final_r0)
        final_c0 = random.randint(min_final_c0, max_final_c0)
        input_r0 = final_r0 - dr * length
        input_c0 = final_c0 - dc * length

        final_cells = _patch_to_cells(final_patch, final_r0, final_c0)
        input_cells = _patch_to_cells(input_patch, input_r0, input_c0)
        if not _separated(final_cells, input_cells):
            continue
        if not _all_separated(final_cells, blocked):
            continue
        if not _all_separated(input_cells, blocked):
            continue

        arrow_cells = _candidate_arrow_cells(
            obj_cells=input_cells,
            final_cells=final_cells,
            direction=direction,
            length=length,
            h=h,
            w=w,
            blocked=blocked + [input_cells, final_cells],
        )
        if arrow_cells is None:
            continue

        if _parse_arrow(arrow_cells) != (direction, length):
            continue
        return input_cells, arrow_cells, final_cells
    return None


def _render_grid(
    h: int,
    w: int,
    bg: int,
    legend: np.ndarray,
    corner: str,
    placements: list[tuple[int, set[tuple[int, int]], set[tuple[int, int]]]],
    use_final: bool,
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    r0, c0 = _legend_origin(h, w, corner)
    grid[r0 : r0 + LEGEND_H, c0 : c0 + LEGEND_W] = legend
    for color, obj_cells, arrow_or_final in placements:
        for r, c in obj_cells:
            grid[r, c] = color
        if use_final:
            continue
        for r, c in arrow_or_final:
            grid[r, c] = color
    return grid


def _expected_output(
    h: int,
    w: int,
    bg: int,
    legend: np.ndarray,
    corner: str,
    finals: list[tuple[int, set[tuple[int, int]]]],
) -> np.ndarray:
    out = np.full((h, w), bg, dtype=int)
    r0, c0 = _legend_origin(h, w, corner)
    out[r0 : r0 + LEGEND_H, c0 : c0 + LEGEND_W] = legend
    for color, cells in finals:
        for r, c in cells:
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    corner = random.choice(["tl", "tr", "bl", "br"])
    colors = random.sample(COLOR_POOL, 2)
    kinds = ["H", "V"]
    random.shuffle(kinds)
    mapping = {int(color): kind for color, kind in zip(colors, kinds)}
    legend, legend_cells = _render_legend(mapping, corner, h, w, bg)

    blocked = [legend_cells]
    input_parts: list[tuple[int, set[tuple[int, int]], set[tuple[int, int]]]] = []
    final_parts: list[tuple[int, set[tuple[int, int]]]] = []

    for color in colors:
        placed = _place_one_color(color, mapping[color], h, w, blocked)
        if placed is None:
            return None
        input_cells, arrow_cells, final_cells = placed
        blocked.extend([input_cells, arrow_cells, final_cells])
        input_parts.append((int(color), input_cells, arrow_cells))
        final_parts.append((int(color), final_cells))

    inp = _render_grid(h, w, bg, legend, corner, input_parts, use_final=False)
    expected = _expected_output(h, w, bg, legend, corner, final_parts)

    if np.array_equal(inp, expected):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
