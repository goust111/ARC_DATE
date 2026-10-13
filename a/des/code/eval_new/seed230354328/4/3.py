from __future__ import annotations

import random
from collections import deque

import numpy as np


GREY = 5
GEN_ATTEMPTS = 400
PAIR_ATTEMPTS = 200
SHAPE_ATTEMPTS = 200

RIGHT_MASK = np.array(
    [
        [1, 0, 0],
        [1, 1, 1],
        [1, 0, 0],
    ],
    dtype=bool,
)

ARROW_SPECS = {
    "right": {
        "vec": (0, 1),
        "mask": RIGHT_MASK,
        "tip": (1, 2),
        "tail_center": (1, 0),
    },
    "up": {
        "vec": (-1, 0),
        "mask": np.rot90(RIGHT_MASK, -1),
        "tip": (0, 1),
        "tail_center": (2, 1),
    },
    "left": {
        "vec": (0, -1),
        "mask": np.rot90(RIGHT_MASK, 2),
        "tip": (1, 0),
        "tail_center": (1, 2),
    },
    "down": {
        "vec": (1, 0),
        "mask": np.rot90(RIGHT_MASK, 1),
        "tip": (2, 1),
        "tail_center": (0, 1),
    },
}

DIR_ORDER = ["right", "up", "left", "down"]
BG_CHOICES = [0, 1, 2, 3, 4, 6, 7, 8, 9]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _local_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _find_arrows(grid: np.ndarray, bg: int) -> list[dict]:
    arrows: list[dict] = []
    non_arrow_colors = set(np.unique(grid)) - {bg, GREY}
    for color in sorted(int(x) for x in non_arrow_colors):
        for comp in _components_of_mask(grid == color):
            mask, bbox = _local_mask(comp)
            assert mask.shape == (3, 3), "every colored marker must fit in a 3x3 arrow bbox"
            assert int(mask.sum()) == 5, "every colored marker must be a 5-pixel arrow"
            matched = None
            for name in DIR_ORDER:
                if np.array_equal(mask, ARROW_SPECS[name]["mask"]):
                    matched = name
                    break
            assert matched is not None, "every non-grey foreground component must be a valid arrow"
            r0, c0, _, _ = bbox
            tip_r, tip_c = ARROW_SPECS[matched]["tip"]
            tail_r, tail_c = ARROW_SPECS[matched]["tail_center"]
            dr, dc = ARROW_SPECS[matched]["vec"]
            arrows.append(
                {
                    "name": matched,
                    "color": color,
                    "coords": comp,
                    "tip": (r0 + tip_r, c0 + tip_c),
                    "tail_anchor": (r0 + tail_r - dr, c0 + tail_c - dc),
                    "vec": (dr, dc),
                }
            )
    assert arrows, "expected at least one colored arrow"
    return arrows


def _target_components(grid: np.ndarray) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    comps = _components_of_mask(grid == GREY)
    assert comps, "expected at least one grey target object"
    lookup: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        for cell in comp:
            lookup[cell] = idx
    return comps, lookup


def _first_hit_object(
    grid: np.ndarray,
    lookup: dict[tuple[int, int], int],
    tip: tuple[int, int],
    vec: tuple[int, int],
) -> tuple[int, tuple[int, int]]:
    h, w = grid.shape
    dr, dc = vec
    r, c = tip[0] + dr, tip[1] + dc
    while 0 <= r < h and 0 <= c < w:
        if grid[r, c] == GREY:
            return lookup[(r, c)], (r, c)
        r += dr
        c += dc
    raise AssertionError("each arrow must encounter a grey object in its pointing direction")


def _copy_translation(
    obj: list[tuple[int, int]],
    hit: tuple[int, int],
    name: str,
    tail_anchor: tuple[int, int],
) -> tuple[int, int]:
    rows = [r for r, _ in obj]
    cols = [c for _, c in obj]
    tail_r, tail_c = tail_anchor
    hit_r, hit_c = hit
    if name == "right":
        return tail_r - hit_r, tail_c - max(cols)
    if name == "left":
        return tail_r - hit_r, tail_c - min(cols)
    if name == "down":
        return tail_r - max(rows), tail_c - hit_c
    if name == "up":
        return tail_r - min(rows), tail_c - hit_c
    raise AssertionError(f"unknown direction {name}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each colored arrow and copy the first grey object it points to onto the arrow's backside in the arrow's color.

    Input structure:
    - The background is the most common color.
    - Several small grey 4-connected target objects are scattered around the grid.
    - Several non-grey monochrome 3x3 arrow markers appear; each is one of four rotations of the same 5-pixel orthogonal arrow template.

    Rule:
    - Detect every arrow's direction and color, then scan forward from its tip along that cardinal direction until the first grey object is hit.
    - Keep the grey source object unchanged, but remove the arrow itself to background.
    - Copy the entire hit object without rotation, recolor it to the arrow's color, and translate it behind the arrow so the copy's face nearest the arrow is flush with the tail side while the hit row or column stays aligned with the tail center.

    Key insight:
    - Each arrow encodes three things at once: which object to select (its pointing direction), what color to paint the copy (its own color), and where to place that copy (the arrow tail on the opposite side).
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    assert bg != GREY, "grey target objects must differ from the background"

    arrows = _find_arrows(g, bg)
    targets, lookup = _target_components(g)

    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["coords"]:
            out[r, c] = bg

    for arrow in arrows:
        obj_idx, hit = _first_hit_object(g, lookup, arrow["tip"], arrow["vec"])
        obj = targets[obj_idx]
        delta_r, delta_c = _copy_translation(obj, hit, arrow["name"], arrow["tail_anchor"])
        for r, c in obj:
            nr = r + delta_r
            nc = c + delta_c
            assert 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1], "copied object must stay in bounds"
            assert out[nr, nc] == bg, "copied object placements must remain conflict-free"
            out[nr, nc] = arrow["color"]

    return out


def _canonical_shape() -> list[tuple[int, int]] | None:
    target_area = random.randint(3, 8)
    row_min, row_max = -2, 2
    col_min, col_max = 0, 4
    for _ in range(SHAPE_ATTEMPTS):
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if row_min <= nr <= row_max and col_min <= nc <= col_max and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != target_area:
            continue
        cols = [c for _, c in cells]
        if min(cols) != 0:
            continue
        rows = [r for r, _ in cells]
        if max(rows) == min(rows) and max(cols) == 0:
            continue
        return sorted(cells)
    return None


def _orient_shape(cells: list[tuple[int, int]], name: str) -> list[tuple[int, int]]:
    if name == "right":
        return [(r, c) for r, c in cells]
    if name == "left":
        return [(r, -c) for r, c in cells]
    if name == "down":
        return [(c, -r) for r, c in cells]
    if name == "up":
        return [(-c, r) for r, c in cells]
    raise AssertionError(f"unknown direction {name}")


def _copy_shape_for_tail(cells: list[tuple[int, int]], name: str) -> list[tuple[int, int]]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if name == "right":
        shift_c = -max(cols)
        return [(r, c + shift_c) for r, c in cells]
    if name == "left":
        shift_c = -min(cols)
        return [(r, c + shift_c) for r, c in cells]
    if name == "down":
        shift_r = -max(rows)
        return [(r + shift_r, c) for r, c in cells]
    if name == "up":
        shift_r = -min(rows)
        return [(r + shift_r, c) for r, c in cells]
    raise AssertionError(f"unknown direction {name}")


def _shift_cells(cells: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in cells]


def _set_from_cells(cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in cells}


def _in_bounds(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _arrow_cells(name: str, top: tuple[int, int]) -> list[tuple[int, int]]:
    mask = ARROW_SPECS[name]["mask"]
    cells = []
    for r in range(3):
        for c in range(3):
            if mask[r, c]:
                cells.append((top[0] + r, top[1] + c))
    return cells


def _candidate_pair(
    h: int,
    w: int,
    name: str,
    color: int,
) -> dict | None:
    shape = _canonical_shape()
    if shape is None:
        return None
    source_rel = _orient_shape(shape, name)
    copy_rel = _copy_shape_for_tail(source_rel, name)
    gap = random.randint(1, 5)
    top = (random.randint(0, h - 3), random.randint(0, w - 3))
    tip_r, tip_c = ARROW_SPECS[name]["tip"]
    tail_r, tail_c = ARROW_SPECS[name]["tail_center"]
    dr, dc = ARROW_SPECS[name]["vec"]
    tip = (top[0] + tip_r, top[1] + tip_c)
    tail_anchor = (top[0] + tail_r - dr, top[1] + tail_c - dc)
    hit = (tip[0] + gap * dr, tip[1] + gap * dc)

    source = _shift_cells(source_rel, hit[0], hit[1])
    copy = _shift_cells(copy_rel, tail_anchor[0], tail_anchor[1])
    arrow = _arrow_cells(name, top)
    corridor = [(tip[0] + step * dr, tip[1] + step * dc) for step in range(1, gap)]

    exact = _set_from_cells(source) | _set_from_cells(copy) | _set_from_cells(arrow)
    if len(exact) != len(source) + len(copy) + len(arrow):
        return None
    corridor_set = _set_from_cells(corridor)
    if exact & corridor_set:
        return None
    if not _in_bounds(exact | corridor_set, h, w):
        return None

    return {
        "name": name,
        "color": color,
        "source": _set_from_cells(source),
        "copy": _set_from_cells(copy),
        "arrow": _set_from_cells(arrow),
        "corridor": corridor_set,
    }


def _render_input(h: int, w: int, bg: int, pairs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    for pair in pairs:
        for r, c in pair["source"]:
            inp[r, c] = GREY
            out[r, c] = GREY
        for r, c in pair["arrow"]:
            inp[r, c] = pair["color"]
        for r, c in pair["copy"]:
            out[r, c] = pair["color"]
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(1, 4)
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    bg = random.choice(BG_CHOICES)
    colors = [c for c in range(10) if c not in {bg, GREY}]
    pair_colors = random.sample(colors, pair_count)
    directions = random.sample(DIR_ORDER, pair_count)

    pairs: list[dict] = []
    forbidden: set[tuple[int, int]] = set()
    for name, color in zip(directions, pair_colors):
        placed = False
        for _ in range(PAIR_ATTEMPTS):
            pair = _candidate_pair(h, w, name, color)
            if pair is None:
                continue
            exact = pair["source"] | pair["copy"] | pair["arrow"]
            if exact & forbidden:
                continue
            if pair["corridor"] & forbidden:
                continue
            pairs.append(pair)
            forbidden |= pair["corridor"]
            forbidden |= _halo(exact, h, w)
            placed = True
            break
        if not placed:
            return None

    inp, expected = _render_input(h, w, bg, pairs)
    if _most_common_color(inp) != bg:
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-copy ARC instance")
