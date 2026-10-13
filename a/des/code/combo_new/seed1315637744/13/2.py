import random
from typing import Optional

import numpy as np


ARROW_SPECS = {
    "up": {
        "shape": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
        "rod": frozenset({(0, 1), (1, 1), (2, 1)}),
        "wings": frozenset({(0, 0), (0, 2)}),
        "tip": (0, 1),
        "step": (-1, 0),
    },
    "down": {
        "shape": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
        "rod": frozenset({(0, 1), (1, 1), (2, 1)}),
        "wings": frozenset({(2, 0), (2, 2)}),
        "tip": (2, 1),
        "step": (1, 0),
    },
    "left": {
        "shape": frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}),
        "rod": frozenset({(1, 0), (1, 1), (1, 2)}),
        "wings": frozenset({(0, 0), (2, 0)}),
        "tip": (1, 0),
        "step": (0, -1),
    },
    "right": {
        "shape": frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
        "rod": frozenset({(1, 0), (1, 1), (1, 2)}),
        "wings": frozenset({(0, 2), (2, 2)}),
        "tip": (1, 2),
        "step": (0, 1),
    },
}

PALETTE = tuple(range(10))
GRID_MIN = 12
GRID_MAX = 28
ARROW_COUNT_RANGE = (1, 4)
MOVE_RANGE = (2, 8)
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def arrow_cells(origin: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    r0, c0 = origin
    return {(r0 + dr, c0 + dc) for dr, dc in ARROW_SPECS[direction]["shape"]}


def tip_cell(origin: tuple[int, int], direction: str) -> tuple[int, int]:
    r0, c0 = origin
    tr, tc = ARROW_SPECS[direction]["tip"]
    return (r0 + tr, c0 + tc)


def target_cell(origin: tuple[int, int], direction: str, move_steps: int) -> tuple[int, int]:
    tr, tc = tip_cell(origin, direction)
    dr, dc = ARROW_SPECS[direction]["step"]
    return (tr + (move_steps + 1) * dr, tc + (move_steps + 1) * dc)


def classify_arrow(grid: np.ndarray, coords: np.ndarray) -> Optional[dict]:
    if coords.shape[0] != 5:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    if (r1 - r0, c1 - c0) != (2, 2):
        return None

    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    for direction, spec in ARROW_SPECS.items():
        if rel != spec["shape"]:
            continue
        rod_colors = {int(grid[r0 + dr, c0 + dc]) for dr, dc in spec["rod"]}
        wing_colors = {int(grid[r0 + dr, c0 + dc]) for dr, dc in spec["wings"]}
        if len(rod_colors) != 1 or len(wing_colors) != 1 or rod_colors == wing_colors:
            return None
        tail_color = next(iter(rod_colors))
        wing_color = next(iter(wing_colors))
        return {
            "origin": (r0, c0),
            "direction": direction,
            "tail_color": int(tail_color),
            "wing_color": int(wing_color),
            "tip": tip_cell((r0, c0), direction),
            "step": tuple(ARROW_SPECS[direction]["step"]),
        }
    return None


def move_distance(shift_r: int, shift_c: int, dr: int, dc: int) -> int:
    if dr != 0:
        assert shift_c == 0 and shift_r % dr == 0
        steps = shift_r // dr
    else:
        assert shift_r == 0 and shift_c % dc == 0
        steps = shift_c // dc
    assert steps >= 0
    return int(steps)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each colored arrow forward until its tip reaches the matching target pixel, then recolor the arrow to the tail color.

    Input structure:
    - The background is the dominant color.
    - Foreground components are either 5-pixel two-color arrows inside a 3x3 box or isolated 1-pixel targets.
    - Each arrow has a 3-pixel straight tail in one color and two wing pixels in a second color, forming one of the four directions.
    - Every target color matches the tail color of exactly one arrow on the arrow's forward row or column ray.

    Rule:
    - Detect each arrow's direction from which side of the 3-pixel tail carries the two wing pixels.
    - For each arrow, search forward from the center tip cell to find the unique target pixel whose color matches the tail color.
    - Translate the whole arrow along that direction until the center tip cell becomes orthogonally adjacent to the target.
    - Erase the original arrow and the consumed target, then redraw the translated arrow entirely in the tail color.

    Key insight:
    - The wing placement determines movement direction, while the tail color rather than the wing color determines which target the arrow absorbs and which color survives in the output.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    arrows: list[dict] = []
    targets: dict[tuple[int, int], int] = {}
    for comp in find_components(g != bg):
        if comp.shape[0] == 1:
            r, c = map(int, comp[0])
            targets[(r, c)] = int(g[r, c])
            continue
        arrow = classify_arrow(g, comp)
        assert arrow is not None, "unexpected foreground component"
        arrows.append(arrow)

    assert arrows and targets
    arrows.sort(key=lambda item: (item["origin"][0], item["origin"][1]))

    out = g.copy()
    used_targets: set[tuple[int, int]] = set()
    redraws: list[tuple[tuple[int, int], str, int]] = []

    for arrow in arrows:
        for r, c in arrow_cells(arrow["origin"], arrow["direction"]):
            out[r, c] = bg

        dr, dc = map(int, arrow["step"])
        rr, cc = arrow["tip"][0] + dr, arrow["tip"][1] + dc
        match: Optional[tuple[int, int]] = None
        while in_bounds(g.shape, rr, cc):
            if int(g[rr, cc]) != bg:
                assert (rr, cc) in targets, "forward ray must be clear apart from targets"
                assert int(g[rr, cc]) == int(arrow["tail_color"]), "target must match the tail color"
                match = (rr, cc)
                break
            rr += dr
            cc += dc

        assert match is not None, "each arrow must see a matching target"
        tr, tc = match
        assert (tr, tc) not in used_targets
        used_targets.add((tr, tc))
        out[tr, tc] = bg

        shift_r = tr - dr - arrow["tip"][0]
        shift_c = tc - dc - arrow["tip"][1]
        _steps = move_distance(shift_r, shift_c, dr, dc)
        new_origin = (arrow["origin"][0] + shift_r, arrow["origin"][1] + shift_c)
        redraws.append((new_origin, arrow["direction"], int(arrow["tail_color"])))

    assert used_targets == set(targets), "every target must be consumed"
    for origin, direction, color in redraws:
        for r, c in arrow_cells(origin, direction):
            out[r, c] = int(color)
    return out


def protected_cells(origin: tuple[int, int], direction: str, move_steps: int) -> set[tuple[int, int]]:
    dr, dc = ARROW_SPECS[direction]["step"]
    cells: set[tuple[int, int]] = set()
    for step in range(move_steps + 1):
        shifted = (origin[0] + step * dr, origin[1] + step * dc)
        cells.update(arrow_cells(shifted, direction))
    cells.add(target_cell(origin, direction, move_steps))
    return cells


def dilate_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def placement_candidates(
    shape: tuple[int, int], direction: str, reserved: set[tuple[int, int]]
) -> list[tuple[tuple[int, int], int, set[tuple[int, int]]]]:
    h, w = shape
    candidates: list[tuple[tuple[int, int], int, set[tuple[int, int]]]] = []
    max_move = min(MOVE_RANGE[1], max(h, w) - 4)
    if max_move < MOVE_RANGE[0]:
        return candidates

    for move_steps in range(MOVE_RANGE[0], max_move + 1):
        for r0 in range(h - 2):
            for c0 in range(w - 2):
                origin = (r0, c0)
                cells = protected_cells(origin, direction, move_steps)
                if any(not in_bounds(shape, r, c) for r, c in cells):
                    continue
                if any((r, c) in reserved for r, c in cells):
                    continue
                candidates.append((origin, move_steps, cells))
    return candidates


def draw_input_arrow(
    grid: np.ndarray, origin: tuple[int, int], direction: str, tail_color: int, wing_color: int
) -> None:
    spec = ARROW_SPECS[direction]
    r0, c0 = origin
    for dr, dc in spec["rod"]:
        grid[r0 + dr, c0 + dc] = int(tail_color)
    for dr, dc in spec["wings"]:
        grid[r0 + dr, c0 + dc] = int(wing_color)


def _gen_attempt() -> Optional[np.ndarray]:
    arrow_count = random.randint(*ARROW_COUNT_RANGE)
    min_side = max(GRID_MIN, 8 + 3 * arrow_count)
    h = random.randint(min_side, GRID_MAX)
    w = random.randint(min_side, GRID_MAX)
    bg = random.choice(PALETTE)

    colors = [color for color in PALETTE if color != bg]
    random.shuffle(colors)
    tail_colors = colors[:arrow_count]
    wing_palette = colors[arrow_count:]
    if not wing_palette:
        return None

    directions = random.sample(tuple(ARROW_SPECS), arrow_count)
    random.shuffle(directions)

    reserved: set[tuple[int, int]] = set()
    placements: list[dict] = []
    for index, direction in enumerate(directions):
        candidates = placement_candidates((h, w), direction, reserved)
        if not candidates:
            return None
        origin, move_steps, cells = random.choice(candidates)
        wing_color = random.choice(wing_palette)
        placements.append(
            {
                "origin": origin,
                "direction": direction,
                "move_steps": int(move_steps),
                "tail_color": int(tail_colors[index]),
                "wing_color": int(wing_color),
            }
        )
        reserved.update(dilate_cells(cells, (h, w)))

    grid = np.full((h, w), int(bg), dtype=int)
    for placement in placements:
        draw_input_arrow(
            grid,
            placement["origin"],
            placement["direction"],
            placement["tail_color"],
            placement["wing_color"],
        )
        tr, tc = target_cell(placement["origin"], placement["direction"], placement["move_steps"])
        grid[tr, tc] = int(placement["tail_color"])

    out = main(grid)
    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
