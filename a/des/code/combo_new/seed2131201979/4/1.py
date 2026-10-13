from __future__ import annotations

import random
from typing import Iterable

import numpy as np


FRAME_COLOR = 5
ALL_COLORS = tuple(range(10))
SIDES = ("top", "bottom", "left", "right")
TEMPLATE_LENGTHS = (3, 5, 7)
GRID_MIN = 18
GRID_MAX = 30
MIN_TEMPLATE_COUNT = 2
MAX_TEMPLATE_COUNT = 4
MAX_GENERATION_ATTEMPTS = 400
TEMPLATE_PLACEMENT_ATTEMPTS = 120
RECTANGLE_PLACEMENT_ATTEMPTS = 200


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    coords.extend((r0, c) for c in range(c0, c1 + 1))
    coords.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    coords.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    coords.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return coords


def _extract_templates(grid: np.ndarray, bg: int) -> tuple[dict[int, list[int]], set[tuple[int, int]]]:
    h, w = grid.shape
    mapping: dict[int, list[int]] = {}
    template_cells: set[tuple[int, int]] = set()
    mask = (grid != bg) & (grid != FRAME_COLOR)
    for comp in _components(mask):
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        ordered: list[tuple[int, int]]
        if r0 == r1 and r0 in (0, h - 1):
            ordered = [(r0, c) for c in range(c0, c1 + 1)]
        elif c0 == c1 and c0 in (0, w - 1):
            ordered = [(r, c0) for r in range(r0, r1 + 1)]
        else:
            continue
        if set(ordered) != set(comp):
            continue
        seq = [int(grid[r, c]) for r, c in ordered]
        if len(seq) < 3:
            continue
        key = seq[0]
        if key in mapping and mapping[key] != seq:
            raise AssertionError("template identifier colors must be unique")
        mapping[key] = seq
        template_cells.update(ordered)
    assert mapping, "expected at least one edge template"
    return mapping, template_cells


def _extract_rectangles(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    rects: list[dict[str, object]] = []
    for comp in _components(grid == FRAME_COLOR):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        perimeter = _perimeter_coords(r0, c0, r1, c1)
        if set(comp) != set(perimeter):
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        seed_pts = np.argwhere((inner != bg) & (inner != FRAME_COLOR))
        if len(seed_pts) != 1:
            continue
        sr, sc = map(int, seed_pts[0])
        rects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "seed_color": int(inner[sr, sc]),
            }
        )
    assert rects, "expected at least one enclosed grey rectangle"
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each grey rectangle border with the repeated edge template selected by its interior seed color.

    Input structure:
    - The background is the most common color.
    - Several hollow grey rectangles sit away from the grid edge, and each rectangle contains exactly one non-background, non-grey seed pixel.
    - Short straight template segments lie on the outer grid edge; each template alternates between two colors, and its first pixel color is the template's identifier.

    Rule:
    - Find every edge template and map its first color to the full template color sequence.
    - For each grey rectangle, read the unique seed color inside it and select the template with the same identifier color.
    - Erase the edge templates to background, then repaint the rectangle border by repeating that template sequence while walking clockwise from the rectangle's top-left corner.
    - Leave the seed pixels unchanged.

    Key insight:
    - The interior seed is not output decoration; it is a lookup key that links a remote border template to the rectangle whose border must be rewritten.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    templates, template_cells = _extract_templates(g, bg)
    rectangles = _extract_rectangles(g, bg)

    out = g.copy()
    for r, c in template_cells:
        out[r, c] = bg

    for rect in rectangles:
        seed_color = int(rect["seed_color"])
        assert seed_color in templates, "each seed color must match exactly one template"
        seq = templates[seed_color]
        r0, c0, r1, c1 = rect["bbox"]
        for idx, (r, c) in enumerate(_perimeter_coords(r0, c0, r1, c1)):
            out[r, c] = seq[idx % len(seq)]

    return out


def _place_template(
    grid: np.ndarray,
    bg: int,
    identifier: int,
    accent: int,
    rng: random.Random,
) -> list[int] | None:
    h, w = grid.shape
    side_order = list(SIDES)
    rng.shuffle(side_order)
    for side in side_order:
        if side in ("top", "bottom"):
            lengths = [n for n in TEMPLATE_LENGTHS if n <= w - 2]
            if not lengths:
                continue
            rng.shuffle(lengths)
            row = 0 if side == "top" else h - 1
            for length in lengths:
                starts = list(range(1, w - length))
                rng.shuffle(starts)
                for start in starts:
                    cells = [(row, c) for c in range(start, start + length)]
                    gap = [(row, c) for c in range(max(0, start - 1), min(w, start + length + 1))]
                    if any(grid[r, c] != bg for r, c in gap):
                        continue
                    seq = [identifier if i % 2 == 0 else accent for i in range(length)]
                    for (r, c), value in zip(cells, seq):
                        grid[r, c] = value
                    return seq
        else:
            lengths = [n for n in TEMPLATE_LENGTHS if n <= h - 2]
            if not lengths:
                continue
            rng.shuffle(lengths)
            col = 0 if side == "left" else w - 1
            for length in lengths:
                starts = list(range(1, h - length))
                rng.shuffle(starts)
                for start in starts:
                    cells = [(r, col) for r in range(start, start + length)]
                    gap = [(r, col) for r in range(max(0, start - 1), min(h, start + length + 1))]
                    if any(grid[r, c] != bg for r, c in gap):
                        continue
                    seq = [identifier if i % 2 == 0 else accent for i in range(length)]
                    for (r, c), value in zip(cells, seq):
                        grid[r, c] = value
                    return seq
    return None


def _place_rectangle(
    grid: np.ndarray,
    reserved: np.ndarray,
    seed_color: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    max_height = min(9, h - 4)
    max_width = min(10, w - 4)
    if max_height < 4 or max_width < 4:
        return False

    for _ in range(RECTANGLE_PLACEMENT_ATTEMPTS):
        rect_h = rng.randint(4, max_height)
        rect_w = rng.randint(4, max_width)
        r0 = rng.randint(2, h - rect_h - 2)
        c0 = rng.randint(2, w - rect_w - 2)
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1
        rr0 = r0 - 1
        cc0 = c0 - 1
        rr1 = r1 + 1
        cc1 = c1 + 1
        if np.any(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1]):
            continue
        for r, c in _perimeter_coords(r0, c0, r1, c1):
            grid[r, c] = FRAME_COLOR
        sr = rng.randint(r0 + 1, r1 - 1)
        sc = rng.randint(c0 + 1, c1 - 1)
        grid[sr, sc] = seed_color
        reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        return True
    return False


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice([c for c in ALL_COLORS if c != FRAME_COLOR])
    available = [c for c in ALL_COLORS if c not in (bg, FRAME_COLOR)]
    template_count = rng.randint(MIN_TEMPLATE_COUNT, min(MAX_TEMPLATE_COUNT, len(available) // 2))
    rng.shuffle(available)
    identifiers = available[:template_count]
    accents = available[template_count : 2 * template_count]

    grid = np.full((h, w), bg, dtype=int)
    sequences: dict[int, list[int]] = {}
    for identifier, accent in zip(identifiers, accents):
        seq = _place_template(grid, bg, identifier, accent, rng)
        if seq is None:
            return None
        sequences[identifier] = seq

    extra_rectangles = rng.randint(0, min(2, template_count))
    seed_colors = identifiers + [rng.choice(identifiers) for _ in range(extra_rectangles)]
    rng.shuffle(seed_colors)

    reserved = grid != bg
    for seed_color in seed_colors:
        if not _place_rectangle(grid, reserved, seed_color, rng):
            return None

    if _background(grid) != bg:
        return None

    found_templates, template_cells = _extract_templates(grid, bg)
    if found_templates != sequences:
        return None

    rectangles = _extract_rectangles(grid, bg)
    if sorted(int(rect["seed_color"]) for rect in rectangles) != sorted(seed_colors):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if any(out[r, c] != bg for r, c in template_cells):
        return None
    if np.any(out == FRAME_COLOR):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
