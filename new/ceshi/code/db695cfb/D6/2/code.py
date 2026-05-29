# Variant of db695cfb
# Dimension: D6 — 源对象类型 (src)
# Change: 一对蓝色像素——两个孤立的蓝色像素定义对角线的起点和终点... → 箭头形状指示方向

import random
import numpy as np

from arc_utils import get_bg


# Design analysis for the D6 variant
# ==================================
# a) Concrete representation and behavior trace
#    - The old task used two isolated blue pixels to mark the ends of a diagonal segment.
#    - This variant replaces each endpoint-pair with a 3-cell blue L-shaped arrow sprite.
#    - The arrow's corner cell is the tip. Its two orthogonal legs point backward, so the
#      missing diagonal direction is the forward direction of travel.
#    - Example: blue cells at {(4, 6), (5, 6), (4, 5)} mean the tip is (4, 6); the legs are
#      down and left; therefore the diagonal extends up-right from (4, 6) until the grid edge.
#    - Magenta trigger pixels still sit on the blue diagonal. When the blue diagonal reaches a
#      magenta cell, magenta rays are emitted along the perpendicular diagonal to the boundary.
#    - Old logic on that same example would accidentally notice the incidental diagonal pair
#      {(4, 5), (5, 6)} and only draw the tiny 3-cell corner. New logic must instead parse the
#      4-connected blue component and read direction from its shape.
#
# b) Complexity calibration
#    - Only four arrow directions exist: NE, NW, SE, SW.
#    - Every active blue source uses the same 3-cell shape, so examples repeat the same cue and
#      keep the rule learnable.
#    - We keep 1-4 active sources, 0-2 triggers per source, and a few distractors, matching the
#      original task's density without introducing extra symbolic mappings.
#
# c) Implicit-assumption audit of the original solver
#    - Grouping blue pixels by diagonal no longer identifies the intended source; an L arrow
#      contains a misleading diagonal pair inside itself.
#    - Choosing the "best same-color pair" is therefore invalid in this variant.
#    - Drawing logic and trigger/ray logic remain valid once the solver receives the intended
#      oriented diagonal from each arrow tip.
#    - Overwrite safety is still handled by only painting background cells for blue fills/rays.
#
# d) Placement legality enforced by the generator
#    - Blue arrow cells cannot overlap any existing input object or any previously reserved
#      output-only cell.
#    - The forward diagonal from each tip must have step-length 3-16 to the boundary so the
#      problem stays comparable to the original.
#    - Trigger pixels are sampled strictly inside the forward diagonal, never on the tip or final
#      boundary cell.
#    - Rays cannot pass through existing input pixels or previously reserved drawn cells.
#    - Blue distractor pixels are isolated from all other blue cells in 4-neighborhood so they
#      cannot accidentally merge into a fake arrow component.
#
# e) Visual observability
#    - Humans can see the L-shaped blue sprite and its missing diagonal quadrant directly.
#    - Across examples, the visible relation "line goes away from the two blue legs" is stable.
#    - The generator samples all four orientations, so the new control variable is observable.
#
# f) Visual cleanliness
#    - Main diagonals, arrow bodies, triggers, and rays are kept non-overlapping except where a
#      trigger intentionally lies on its own main diagonal.
#    - Paths and rays are separated across different objects to avoid cluttered ambiguity.
#
# g) Color inference
#    - No new color mapping is introduced. Blue remains the source color and magenta remains the
#      trigger/ray color, so the only conceptual change is the source object's shape.


_LINE_COLOR = 1
_TRIGGER_COLOR = 6

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_SEGMENT_COUNT = (1, 4)
_SEG_LEN_RANGE = (3, 16)       # forward diagonal step-count from tip to boundary
_TRIGGERS_PER_SEG = (0, 2)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 4)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 120
_DISTRACTOR_RETRIES = 80
_GEN_RETRIES = 400

_DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _diag_path_to_boundary(tip: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = tip
    dr, dc = direction
    path = []
    while 0 <= r < h and 0 <= c < w:
        path.append((r, c))
        r += dr
        c += dc
    return path


def _arrow_cells(tip: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = tip
    dr, dc = direction
    return {(r, c), (r - dr, c), (r, c - dc)}


def _find_arrow_sources(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = grid.shape
    visited: set[tuple[int, int]] = set()
    arrows: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != _LINE_COLOR or (r, c) in visited:
                continue

            stack = [(r, c)]
            comp = set()
            visited.add((r, c))
            while stack:
                cr, cc = stack.pop()
                comp.add((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and int(grid[nr, nc]) == _LINE_COLOR:
                        visited.add((nr, nc))
                        stack.append((nr, nc))

            if len(comp) != 3:
                continue

            deg = {}
            for cell in comp:
                deg[cell] = sum((nbr in comp) for nbr in _neighbors4(*cell))
            if sorted(deg.values()) != [1, 1, 2]:
                continue

            tip = next(cell for cell, d in deg.items() if d == 2)
            rels = [(nr - tip[0], nc - tip[1]) for nr, nc in comp if (nr, nc) != tip]
            vertical = [d for d in rels if d[1] == 0 and abs(d[0]) == 1]
            horizontal = [d for d in rels if d[0] == 0 and abs(d[1]) == 1]
            if len(vertical) != 1 or len(horizontal) != 1:
                continue

            dr = -vertical[0][0]
            dc = -horizontal[0][1]
            if dr not in (-1, 1) or dc not in (-1, 1):
                continue
            arrows.append((tip, (dr, dc)))

    arrows.sort()
    return arrows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read blue arrow sprites as diagonal launchers, then emit perpendicular rays from trigger pixels touched by those diagonals.

    Input structure:
    - bg = most common color. Blue source objects are 3-cell L-shaped arrows; the corner cell is the tip.
    - Each arrow's two orthogonal legs point backward, so the diagonal extends away from them.
    - Magenta trigger pixels may lie on those diagonals; extra blue or magenta distractors can also appear off-path.

    Rule:
    - For every valid blue arrow, start at its tip and draw a blue 45-degree line in the arrow's forward diagonal direction until the grid boundary.
    - Whenever that blue diagonal passes through a non-blue, non-background trigger pixel, preserve that trigger and launch rays of the same color on the perpendicular diagonal in both directions.

    Key insight:
    - The active source is no longer a pair of blue endpoints; the direction is encoded by the local geometry of each blue L, and the line must travel away from the arrow's two legs.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    bg = get_bg(grid)

    arrows = _find_arrow_sources(grid)
    assert arrows, "grid must contain at least one valid blue arrow source"

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    for tip, (dr, dc) in arrows:
        perp_dirs = [(dr, -dc), (-dr, dc)]
        for r, c in _diag_path_to_boundary(tip, (dr, dc), h, w):
            v = int(out[r, c])
            if v == bg:
                out[r, c] = _LINE_COLOR
            elif v != _LINE_COLOR:
                triggers.setdefault((r, c, v), set()).update(perp_dirs)

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


def _compute_ray_cells(triggers: list[tuple[int, int]], perp_dirs: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    for tr, tc in triggers:
        for ddr, ddc in perp_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < h and 0 <= c < w:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(inp: np.ndarray, bg: int, segments: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]]) -> np.ndarray:
    h, w = inp.shape
    expected = inp.copy()
    for _, direction, trigger_positions, path, perp_dirs in segments:
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc in trigger_positions:
            for ddr, ddc in perp_dirs:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < h and 0 <= c < w:
                    if expected[r, c] == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _steps_to_boundary(tip: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> int:
    r, c = tip
    dr, dc = direction
    row_steps = (h - 1 - r) if dr > 0 else r
    col_steps = (w - 1 - c) if dc > 0 else c
    return min(row_steps, col_steps)


def _try_place_arrow(
    direction: tuple[int, int],
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    blue_occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
) -> tuple[tuple[int, int], set[tuple[int, int]], list[tuple[int, int]]] | None:
    dr, dc = direction
    r_lo = 0 if dr > 0 else 0
    r_hi = h - 1
    c_lo = 0 if dc > 0 else 0
    c_hi = w - 1

    if dr > 0:
        r_lo = 1
    else:
        r_hi = h - 2
    if dc > 0:
        c_lo = 1
    else:
        c_hi = w - 2

    if r_lo > r_hi or c_lo > c_hi:
        return None

    for _ in range(_SEGMENT_PLACE_RETRIES):
        tip = (random.randint(r_lo, r_hi), random.randint(c_lo, c_hi))
        steps = _steps_to_boundary(tip, direction, h, w)
        if not (_SEG_LEN_RANGE[0] <= steps <= _SEG_LEN_RANGE[1]):
            continue

        arrow_cells = _arrow_cells(tip, direction)
        path = _diag_path_to_boundary(tip, direction, h, w)
        path_tail = set(path[1:])

        if any(cell in occupied or cell in drawn for cell in arrow_cells):
            continue
        if any(cell in blue_occupied for cell in arrow_cells):
            continue
        if any(nbr in blue_occupied for cell in arrow_cells for nbr in _neighbors4(*cell)):
            continue
        if any(cell in occupied or cell in drawn for cell in path_tail):
            continue
        if any(not (0 <= r < h and 0 <= c < w) for r, c in arrow_cells):
            continue
        return tip, arrow_cells, path
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_H_RANGE)
    w = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    occupied: set[tuple[int, int]] = set()
    blue_occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    segments: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]] = []

    for _ in range(random.randint(*_SEGMENT_COUNT)):
        direction = random.choice(_DIAG_DIRS)
        placed = _try_place_arrow(direction, h, w, occupied, blue_occupied, drawn)
        if placed is None:
            continue

        tip, arrow_cells, path = placed
        path_interior = path[1:-1]
        n_triggers = random.randint(_TRIGGERS_PER_SEG[0], min(_TRIGGERS_PER_SEG[1], len(path_interior)))
        trigger_positions = random.sample(path_interior, n_triggers) if n_triggers > 0 else []
        perp_dirs = [(direction[0], -direction[1]), (-direction[0], direction[1])]
        ray_cells = _compute_ray_cells(trigger_positions, perp_dirs, h, w)

        reserved_input = occupied | arrow_cells | set(trigger_positions)
        reserved_drawn = drawn | (set(path[1:]) - set(trigger_positions))
        if any(cell in reserved_input or cell in reserved_drawn for cell in ray_cells):
            continue

        segments.append((tip, direction, trigger_positions, path, perp_dirs))
        occupied.update(arrow_cells)
        blue_occupied.update(arrow_cells)
        occupied.update(trigger_positions)
        drawn.update(set(path[1:]) - set(trigger_positions))
        drawn.update(ray_cells)

    if not segments:
        return None

    inp = np.full((h, w), bg, dtype=int)
    for tip, direction, trigger_positions, _, _ in segments:
        for r, c in _arrow_cells(tip, direction):
            inp[r, c] = _LINE_COLOR
        for r, c in trigger_positions:
            inp[r, c] = _TRIGGER_COLOR

    blue_cells = {(int(r), int(c)) for r, c in zip(*np.where(inp == _LINE_COLOR))}
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            cell = (r, c)
            if cell in occupied or cell in drawn:
                continue
            if any(abs(r - br) + abs(c - bc) == 1 for br, bc in blue_cells):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add(cell)
            blue_cells.add(cell)
            blue_occupied.add(cell)
            break

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, bg, segments)
    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
