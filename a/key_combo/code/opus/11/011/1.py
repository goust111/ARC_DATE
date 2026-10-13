from __future__ import annotations

from collections import deque
import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
CYCLE = (RED, GREEN, YELLOW)
N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in N4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _interior_mask(boundary_mask: np.ndarray) -> np.ndarray:
    h, w = boundary_mask.shape
    exterior = np.zeros_like(boundary_mask, dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not boundary_mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not boundary_mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not boundary_mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    return (~exterior) & (~boundary_mask)


def _ring_analysis(grid: np.ndarray) -> list[dict]:
    blue_mask = np.asarray(grid == BLUE)
    comps = _components(blue_mask)
    analysis = []
    for coords in comps:
        boundary_mask = np.zeros_like(blue_mask, dtype=bool)
        boundary_mask[coords[:, 0], coords[:, 1]] = True
        interior = _interior_mask(boundary_mask)
        anchor_r, anchor_c = map(int, coords[0])
        analysis.append(
            {
                "coords": coords,
                "mask": boundary_mask,
                "interior": interior,
                "has_inner": bool(np.any(blue_mask & interior)),
                "anchor": (anchor_r, anchor_c),
            }
        )

    for item in analysis:
        r, c = item["anchor"]
        item["depth"] = 1 + sum(
            1 for other in analysis if other is not item and bool(other["interior"][r, c])
        )

    return analysis


def _rect_ring_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _apply_inward_notch(
    cells: set[tuple[int, int]],
    outer_box: tuple[int, int, int, int],
    inner_box: tuple[int, int, int, int] | None,
) -> bool:
    r0, c0, r1, c1 = outer_box
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    safe_sides = []

    if inner_box is None:
        if width >= 6:
            safe_sides.extend(("top", "bottom"))
        if height >= 6:
            safe_sides.extend(("left", "right"))
    else:
        ir0, ic0, ir1, ic1 = inner_box
        if width >= 6 and ir0 - r0 >= 3:
            safe_sides.append("top")
        if width >= 6 and r1 - ir1 >= 3:
            safe_sides.append("bottom")
        if height >= 6 and ic0 - c0 >= 3:
            safe_sides.append("left")
        if height >= 6 and c1 - ic1 >= 3:
            safe_sides.append("right")

    if not safe_sides:
        return False

    side = random.choice(safe_sides)
    if side in ("top", "bottom"):
        start = random.randint(c0 + 2, c1 - 3)
        end = random.randint(start + 1, c1 - 2)
        row = r0 if side == "top" else r1
        notch_row = row + 1 if side == "top" else row - 1
        for c in range(start + 1, end):
            cells.discard((row, c))
        for c in range(start, end + 1):
            cells.add((notch_row, c))
        return True

    start = random.randint(r0 + 2, r1 - 3)
    end = random.randint(start + 1, r1 - 2)
    col = c0 if side == "left" else c1
    notch_col = col + 1 if side == "left" else col - 1
    for r in range(start + 1, end):
        cells.discard((r, col))
    for r in range(start, end + 1):
        cells.add((r, notch_col))
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested blue closed boundaries by cyclic colors according to containment depth, but only when they themselves contain deeper structure.

    Input structure:
    - The background is black and the foreground consists of blue 4-connected closed boundary objects.
    - The boundaries are strictly nested: one object per depth, with 3-5 layers overall.
    - Some boundaries enclose only empty background, while others enclose deeper blue boundaries inside their interior.

    Rule:
    - Treat each blue closed boundary as one object and compute its nesting depth, with the outermost boundary at depth 1.
    - For each boundary, test whether its interior contains any non-background pixel; only those boundaries participate in recoloring.
    - Sort the participating boundaries by increasing depth and recolor them cyclically red, green, yellow, red, ... .
    - Leave boundaries whose interior is empty blue, and keep the background unchanged.

    Key insight:
    - The color order is driven by topological depth rather than position, and the deepest empty ring is excluded because recoloring is gated by whether the ring contains deeper structure.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    rings = _ring_analysis(g)
    rings_to_color = [item for item in rings if item["has_inner"]]
    rings_to_color.sort(key=lambda item: int(item["depth"]))

    for idx, item in enumerate(rings_to_color):
        color = CYCLE[idx % len(CYCLE)]
        coords = item["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _build_boxes() -> list[tuple[int, int, int, int]]:
    while True:
        layers = random.choice((3, 4, 5))
        inner_h = random.randint(3, 5)
        inner_w = random.randint(3, 5)
        boxes = [(0, 0, inner_h - 1, inner_w - 1)]

        ok = True
        for _ in range(layers - 1):
            r0, c0, r1, c1 = boxes[-1]
            remaining = layers - len(boxes)
            extra_budget_h = 24 - (r1 - r0 + 1)
            extra_budget_w = 24 - (c1 - c0 + 1)
            min_needed = remaining * 4
            if extra_budget_h < 4 or extra_budget_w < 4 or extra_budget_h < min_needed or extra_budget_w < min_needed:
                ok = False
                break

            choices = [2, 3]
            if remaining >= 2:
                if extra_budget_h - min_needed < 4:
                    choices_h = [2]
                else:
                    choices_h = choices
                if extra_budget_w - min_needed < 4:
                    choices_w = [2]
                else:
                    choices_w = choices
            else:
                choices_h = choices
                choices_w = choices

            top = random.choice(choices_h)
            bottom = random.choice(choices_h)
            left = random.choice(choices_w)
            right = random.choice(choices_w)
            boxes.append((r0 - top, c0 - left, r1 + bottom, c1 + right))

        if ok:
            return list(reversed(boxes))


def _gen_attempt() -> np.ndarray | None:
    boxes = _build_boxes()
    outer_r0, outer_c0, outer_r1, outer_c1 = boxes[0]
    outer_h = outer_r1 - outer_r0 + 1
    outer_w = outer_c1 - outer_c0 + 1

    grid_h = random.randint(max(outer_h + 2, 15), min(outer_h + 6, 26))
    grid_w = random.randint(max(outer_w + 2, 15), min(outer_w + 6, 26))
    if grid_h > 30 or grid_w > 30:
        return None

    row_shift = random.randint(1 - outer_r0, grid_h - 2 - outer_r1)
    col_shift = random.randint(1 - outer_c0, grid_w - 2 - outer_c1)
    placed_boxes = [
        (r0 + row_shift, c0 + col_shift, r1 + row_shift, c1 + col_shift)
        for r0, c0, r1, c1 in boxes
    ]

    grid = np.zeros((grid_h, grid_w), dtype=int)
    ring_cells: list[set[tuple[int, int]]] = []
    notched = False
    for idx, box in enumerate(placed_boxes):
        cells = _rect_ring_cells(*box)
        inner_box = placed_boxes[idx + 1] if idx + 1 < len(placed_boxes) else None
        if random.random() < 0.7:
            notched |= _apply_inward_notch(cells, box, inner_box)
        ring_cells.append(cells)

    if not notched:
        ring_cells[-1] = _rect_ring_cells(*placed_boxes[-1])
        if not _apply_inward_notch(ring_cells[-1], placed_boxes[-1], None):
            return None

    for cells in ring_cells:
        for r, c in cells:
            grid[r, c] = BLUE

    analysis = _ring_analysis(grid)
    expected_layers = len(placed_boxes)
    if len(analysis) != expected_layers:
        return None

    depths = sorted(int(item["depth"]) for item in analysis)
    if depths != list(range(1, expected_layers + 1)):
        return None

    hole_layers = sum(1 for item in analysis if item["has_inner"])
    empty_layers = expected_layers - hole_layers
    if hole_layers < 2 or empty_layers < 1:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring input")
