from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS = {
    "right": (0, 1),
    "left": (0, -1),
    "up": (-1, 0),
    "down": (1, 0),
}

GEN_ATTEMPTS = 500
ITEM_ATTEMPTS = 300
GRID_MIN = 16
GRID_MAX = 30
GAP_RANGE = (1, 4)
PALETTE = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_non_bg(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(grid[sr, sc]) == bg:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            colors: set[int] = set()
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                colors.add(int(grid[r, c]))
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [r for r, _c in coords]
            cols = [c for _r, c in coords]
            comps.append(
                {
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "colors": colors,
                    "area": len(coords),
                }
            )
    return comps


def _normalize(coords: set[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    min_r = min(r for r, _c in coords)
    min_c = min(c for _r, c in coords)
    norm = frozenset((r - min_r, c - min_c) for r, c in coords)
    return norm, (int(min_r), int(min_c))


def _arrow_tip_relative(direction: str, length: int) -> set[tuple[int, int]]:
    assert 1 <= length <= 4
    if direction == "right":
        coords = {(0, 0), (0, -1), (-1, -1)}
        coords.update((0, -k) for k in range(2, length + 2))
        return coords
    if direction == "left":
        coords = {(0, 0), (0, 1), (1, 1)}
        coords.update((0, k) for k in range(2, length + 2))
        return coords
    if direction == "up":
        coords = {(0, 0), (1, 0), (1, -1)}
        coords.update((k, 0) for k in range(2, length + 2))
        return coords
    assert direction == "down"
    coords = {(0, 0), (-1, 0), (-1, 1)}
    coords.update((-k, 0) for k in range(2, length + 2))
    return coords


def _build_arrow_templates() -> dict[int, list[dict[str, object]]]:
    by_area: dict[int, list[dict[str, object]]] = {}
    for direction in DIRS:
        for length in range(1, 5):
            rel = _arrow_tip_relative(direction, length)
            norm, (min_r, min_c) = _normalize(rel)
            by_area.setdefault(len(norm), []).append(
                {
                    "direction": direction,
                    "length": length,
                    "cells": norm,
                    "tip_offset": (-min_r, -min_c),
                }
            )
    return by_area


ARROW_TEMPLATES = _build_arrow_templates()


def _arrow_cells_from_tip(direction: str, length: int, tip: tuple[int, int]) -> set[tuple[int, int]]:
    tr, tc = tip
    return {(tr + dr, tc + dc) for dr, dc in _arrow_tip_relative(direction, length)}


def _pattern_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)}


def _copy_cells(top_left: tuple[int, int], direction: str, count: int) -> set[tuple[int, int]]:
    dr, dc = DIRS[direction]
    out: set[tuple[int, int]] = set()
    for i in range(1, count + 1):
        r0 = top_left[0] + dr * 3 * i
        c0 = top_left[1] + dc * 3 * i
        out |= _pattern_cells((int(r0), int(c0)))
    return out


def _corridor_cells(
    tip: tuple[int, int],
    top_left: tuple[int, int],
    direction: str,
    count: int,
) -> set[tuple[int, int]]:
    tr, tc = tip
    r0, c0 = top_left
    if direction == "right":
        end = c0 + 3 * count + 2
        return {(tr, c) for c in range(tc + 1, end + 1)}
    if direction == "left":
        end = c0 - 3 * count
        return {(tr, c) for c in range(end, tc)}
    if direction == "down":
        end = r0 + 3 * count + 2
        return {(r, tc) for r in range(tr + 1, end + 1)}
    end = r0 - 3 * count
    return {(r, tc) for r in range(end, tr)}


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    return {
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - radius), min(h, r + radius + 1))
        for cc in range(max(0, c - radius), min(w, c + radius + 1))
    }


def _inside_all(cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _make_pattern(bg: int, avoid: int) -> np.ndarray:
    choices = [c for c in PALETTE if c not in (bg, avoid)]
    n_colors = random.randint(2, min(4, len(choices)))
    palette = random.sample(choices, n_colors)
    patch = np.array([[random.choice(palette) for _ in range(3)] for _ in range(3)], dtype=int)
    if len(set(map(int, patch.flat))) == 1:
        patch[random.randrange(3), random.randrange(3)] = random.choice([c for c in palette if c != int(patch[0, 0])])
    return patch


def _match_arrow(comp: dict[str, object]) -> dict[str, object] | None:
    colors = comp["colors"]
    assert isinstance(colors, set)
    if len(colors) != 1:
        return None
    area = int(comp["area"])
    if area not in ARROW_TEMPLATES:
        return None
    coords = comp["coords"]
    bbox = comp["bbox"]
    assert isinstance(coords, list)
    assert isinstance(bbox, tuple)
    r0, c0, _r1, _c1 = map(int, bbox)
    norm = frozenset((int(r) - r0, int(c) - c0) for r, c in coords)
    for spec in ARROW_TEMPLATES[area]:
        if norm != spec["cells"]:
            continue
        tip_dr, tip_dc = map(int, spec["tip_offset"])
        tip = (r0 + tip_dr, c0 + tip_dc)
        return {
            "coords": set(coords),
            "color": next(iter(colors)),
            "direction": spec["direction"],
            "length": spec["length"],
            "tip": tip,
        }
    return None


def _match_pattern(grid: np.ndarray, comp: dict[str, object]) -> dict[str, object] | None:
    if int(comp["area"]) != 9:
        return None
    coords = comp["coords"]
    bbox = comp["bbox"]
    assert isinstance(coords, list)
    assert isinstance(bbox, tuple)
    r0, c0, r1, c1 = map(int, bbox)
    if r1 - r0 != 2 or c1 - c0 != 2:
        return None
    if set(coords) != _pattern_cells((r0, c0)):
        return None
    return {
        "coords": set(coords),
        "top_left": (r0, c0),
        "patch": grid[r0 : r1 + 1, c0 : c1 + 1].copy(),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each arrow and repeat the first 3x3 pattern block on its ray by the arrow tail length.

    Input structure:
    - The background is the dominant color.
    - Foreground components are either monochrome arrows or fully filled 3x3 pattern blocks.
    - Each arrow consists of a 3-cell right-triangle head plus a 1-cell-wide tail of length 1-4, and the head orientation encodes one cardinal direction.
    - Each pattern block is a 3x3 non-background square that may use multiple colors.
    - Every arrow tip sees exactly one pattern block first along its forward row or column.

    Rule:
    - Detect each arrow's facing direction, tip cell, and tail length.
    - Cast a cardinal ray from the tip until it reaches the first pattern block.
    - Keep the original pattern block, then paste the same 3x3 pattern directly behind it along that direction, repeated tail_length times.
    - Repaint every arrow cell with the background color.

    Key insight:
    - The tail is a copy-count code while the triangle head fixes the ray direction, so solving means pairing each arrow with the first pattern block on its line and translating that block in 3-cell steps.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _components_non_bg(g, bg)

    arrows: list[dict[str, object]] = []
    patterns: list[dict[str, object]] = []
    covered: set[tuple[int, int]] = set()
    for comp in comps:
        arrow = _match_arrow(comp)
        if arrow is not None:
            arrows.append(arrow)
            covered |= arrow["coords"]
            continue
        pattern = _match_pattern(g, comp)
        if pattern is not None:
            patterns.append(pattern)
            covered |= pattern["coords"]
            continue
        raise AssertionError("unexpected non-background component")

    assert arrows, "expected at least one arrow"
    assert patterns, "expected at least one pattern block"

    non_bg_cells = {(int(r), int(c)) for r, c in np.argwhere(g != bg)}
    assert covered == non_bg_cells, "all foreground cells must belong to arrows or 3x3 patterns"

    pattern_by_cell: dict[tuple[int, int], int] = {}
    for pid, pattern in enumerate(patterns):
        for cell in pattern["coords"]:
            pattern_by_cell[cell] = pid

    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["coords"]:
            out[r, c] = bg

    h, w = map(int, g.shape)
    for arrow in arrows:
        tip_r, tip_c = map(int, arrow["tip"])
        dr, dc = DIRS[str(arrow["direction"])]
        pid = None
        rr, cc = tip_r + dr, tip_c + dc
        while 0 <= rr < h and 0 <= cc < w:
            pid = pattern_by_cell.get((rr, cc))
            if pid is not None:
                break
            rr += dr
            cc += dc
        assert pid is not None, "every arrow must see a pattern block"

        pattern = patterns[pid]
        pr, pc = map(int, pattern["top_left"])
        patch = pattern["patch"]
        assert isinstance(patch, np.ndarray)
        for i in range(1, int(arrow["length"]) + 1):
            nr = pr + dr * 3 * i
            nc = pc + dc * 3 * i
            assert 0 <= nr <= h - 3 and 0 <= nc <= w - 3, "copies must stay inside the grid"
            out[nr : nr + 3, nc : nc + 3] = patch

    return out


def _candidate_item(
    shape: tuple[int, int],
    direction: str,
    length: int,
    blocked: set[tuple[int, int]],
) -> dict[str, object] | None:
    h, w = map(int, shape)
    for _ in range(ITEM_ATTEMPTS):
        gap = random.randint(*GAP_RANGE)
        align = random.randint(0, 2)

        if direction == "right":
            pc_lo = gap + length + 2
            pc_hi = w - 3 * length - 3
            if pc_lo > pc_hi:
                continue
            pr = random.randint(0, h - 3)
            pc = random.randint(pc_lo, pc_hi)
            tip = (pr + align, pc - gap - 1)
        elif direction == "left":
            pc_lo = 3 * length
            pc_hi = w - gap - length - 5
            if pc_lo > pc_hi:
                continue
            pr = random.randint(0, h - 3)
            pc = random.randint(pc_lo, pc_hi)
            tip = (pr + align, pc + gap + 3)
        elif direction == "up":
            pr_lo = 3 * length
            pr_hi = h - gap - length - 5
            if pr_lo > pr_hi:
                continue
            pr = random.randint(pr_lo, pr_hi)
            pc = random.randint(0, w - 3)
            tip = (pr + gap + 3, pc + align)
        else:
            pr_lo = gap + length + 2
            pr_hi = h - 3 * length - 3
            if pr_lo > pr_hi:
                continue
            pr = random.randint(pr_lo, pr_hi)
            pc = random.randint(0, w - 3)
            tip = (pr - gap - 1, pc + align)

        pattern_top_left = (pr, pc)
        arrow_cells = _arrow_cells_from_tip(direction, length, tip)
        pattern_cells = _pattern_cells(pattern_top_left)
        copies = _copy_cells(pattern_top_left, direction, length)
        corridor = _corridor_cells(tip, pattern_top_left, direction, length)
        footprint = arrow_cells | pattern_cells | copies | corridor
        if not _inside_all(footprint, shape):
            continue
        if _halo(footprint, shape) & blocked:
            continue
        return {
            "direction": direction,
            "length": length,
            "tip": tip,
            "pattern_top_left": pattern_top_left,
            "arrow_cells": arrow_cells,
            "pattern_cells": pattern_cells,
            "footprint": footprint,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    n_items = random.randint(1, 4)
    min_side = max(GRID_MIN, 8 + 4 * n_items)
    h = random.randint(min_side, GRID_MAX)
    w = random.randint(min_side, GRID_MAX)
    bg = random.choice(PALETTE)
    grid = np.full((h, w), bg, dtype=int)

    arrow_colors = random.sample([c for c in PALETTE if c != bg], n_items)
    directions = random.sample(list(DIRS), n_items)
    blocked: set[tuple[int, int]] = set()
    items: list[dict[str, object]] = []

    for arrow_color, direction in zip(arrow_colors, directions):
        length = random.randint(1, 4)
        item = _candidate_item((h, w), direction, length, blocked)
        if item is None:
            return None
        item["arrow_color"] = arrow_color
        item["patch"] = _make_pattern(bg, arrow_color)
        items.append(item)
        blocked |= _halo(item["footprint"], (h, w))

    for item in items:
        patch = item["patch"]
        assert isinstance(patch, np.ndarray)
        for r, c in item["arrow_cells"]:
            grid[r, c] = int(item["arrow_color"])
        pr, pc = map(int, item["pattern_top_left"])
        grid[pr : pr + 3, pc : pc + 3] = patch

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if max(out.shape) > 30 or max(grid.shape) > 30:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid arrow-pattern input")
