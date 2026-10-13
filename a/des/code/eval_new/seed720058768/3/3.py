from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7
MAROON = 9

PALETTE = (BLUE, RED, YELLOW, ORANGE)
SIDES = ("top", "bottom", "left", "right")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _openings_for_box(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> list[tuple[int, int, int, int]]:
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h >= 3 and w >= 3, "container bbox must leave an interior"

    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    for r, c in corners:
        assert int(grid[r, c]) == GREY, "container corners must remain closed"

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert np.all(interior == BLACK), "container interior must be empty background"

    openings: list[tuple[int, int, int, int]] = []
    sides = (
        ("top", [(r0, c) for c in range(c0 + 1, c1)], (1, 0)),
        ("bottom", [(r1, c) for c in range(c0 + 1, c1)], (-1, 0)),
        ("left", [(r, c0) for r in range(r0 + 1, r1)], (0, 1)),
        ("right", [(r, c1) for r in range(r0 + 1, r1)], (0, -1)),
    )
    for _side, coords, direction in sides:
        prev_black = False
        for r, c in coords:
            value = int(grid[r, c])
            assert value in (BLACK, GREY), "container perimeter may only contain wall or opening cells"
            assert not (prev_black and value == BLACK), "each opening must have width exactly one"
            if value == BLACK:
                openings.append((r, c, direction[0], direction[1]))
            prev_black = value == BLACK

    assert 1 <= len(openings) <= 4, "expected one to four openings"
    return openings


def _find_containers(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], list[tuple[int, int, int, int]]]]:
    grey_coords = np.argwhere(grid == GREY)
    grey_set = {tuple(map(int, coord)) for coord in grey_coords}
    row_to_cols: dict[int, list[int]] = {}
    col_to_rows: dict[int, list[int]] = {}
    for r, c in grey_set:
        row_to_cols.setdefault(r, []).append(c)
        col_to_rows.setdefault(c, []).append(r)
    for cols in row_to_cols.values():
        cols.sort()
    for rows in col_to_rows.values():
        rows.sort()

    containers: dict[tuple[int, int, int, int], list[tuple[int, int, int, int]]] = {}
    for r0, c0 in sorted(grey_set):
        right_cols = [c1 for c1 in row_to_cols.get(r0, []) if c1 >= c0 + 2]
        lower_rows = [r1 for r1 in col_to_rows.get(c0, []) if r1 >= r0 + 2]
        for r1 in lower_rows:
            for c1 in right_cols:
                if (r1, c1) not in grey_set:
                    continue
                try:
                    openings = _openings_for_box(grid, r0, c0, r1, c1)
                except AssertionError:
                    continue
                containers[(r0, c0, r1, c1)] = openings

    boxes = sorted(containers.items())
    for i, (box_a, _openings_a) in enumerate(boxes):
        for box_b, _openings_b in boxes[i + 1 :]:
            assert not _boxes_overlap(box_a, box_b), "containers must not overlap"
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each grey container by its opening count, then fire inward rays from the openings.

    Input structure:
    - The background is black.
    - A four-pixel palette sits on the top-left row, ordered blue, red, yellow, orange.
    - Elsewhere, several one-cell-thick grey rectangular containers enclose black interiors and have one to four single-cell openings on their borders.

    Rule:
    - Count each container's openings and pick the palette color at that 1-based index.
    - For every opening, start at the opening cell and fill a straight ray inward, perpendicular to that border, until the next grey wall would be hit.
    - Pixels hit by exactly one ray take the container color; pixels hit by two or more rays become maroon. Grey walls and the palette stay unchanged.

    Key insight:
    - The opening count does double duty: it selects the fill color and also determines how many inward rays are emitted, so the interior is only partially painted rather than flood-filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert w >= 4, "palette row must fit"

    palette = tuple(int(v) for v in g[0, :4])
    assert len(palette) == 4, "palette must contain four colors"

    coverage_count = np.zeros((h, w), dtype=int)
    coverage_color = np.zeros((h, w), dtype=int)

    for _box, openings in _find_containers(g):
        fill_color = int(palette[len(openings) - 1])
        for r, c, dr, dc in openings:
            rr, cc = r, c
            while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) != GREY:
                coverage_count[rr, cc] += 1
                if coverage_count[rr, cc] == 1:
                    coverage_color[rr, cc] = fill_color
                rr += dr
                cc += dc

    out = g.copy()
    single_mask = coverage_count == 1
    overlap_mask = coverage_count >= 2
    out[single_mask] = coverage_color[single_mask]
    out[overlap_mask] = MAROON
    return out


def _random_gap(side: str, height: int, width: int) -> int:
    if side in ("top", "bottom"):
        return random.randint(2, width - 3)
    return random.randint(2, height - 3)


def _choose_openings(height: int, width: int, opening_count: int) -> list[tuple[str, int]]:
    assert 1 <= opening_count <= 4

    if opening_count == 1:
        side = random.choice(SIDES)
        return [(side, _random_gap(side, height, width))]

    if opening_count == 2:
        if random.random() < 0.5:
            vertical_side = random.choice(("top", "bottom"))
            horizontal_side = random.choice(("left", "right"))
            return [
                (vertical_side, _random_gap(vertical_side, height, width)),
                (horizontal_side, _random_gap(horizontal_side, height, width)),
            ]

        if random.random() < 0.5:
            c_top = random.randint(2, width - 3)
            choices = [c for c in range(2, width - 2) if c != c_top]
            c_bottom = random.choice(choices)
            return [("top", c_top), ("bottom", c_bottom)]

        r_left = random.randint(2, height - 3)
        choices = [r for r in range(2, height - 2) if r != r_left]
        r_right = random.choice(choices)
        return [("left", r_left), ("right", r_right)]

    if opening_count == 3:
        missing = random.choice(SIDES)
        sides = [side for side in SIDES if side != missing]
        return [(side, _random_gap(side, height, width)) for side in sides]

    return [(side, _random_gap(side, height, width)) for side in SIDES]


def _draw_container(height: int, width: int, opening_count: int) -> np.ndarray:
    patch = np.full((height, width), BLACK, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY

    for side, pos in _choose_openings(height, width, opening_count):
        if side == "top":
            patch[0, pos] = BLACK
        elif side == "bottom":
            patch[-1, pos] = BLACK
        elif side == "left":
            patch[pos, 0] = BLACK
        else:
            patch[pos, -1] = BLACK
    return patch


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded(box: tuple[int, int, int, int], pad: int, height: int, width: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(height - 1, r1 + pad),
        min(width - 1, c1 + pad),
    )


def _place_containers(
    grid: np.ndarray,
    count_choices: list[int],
) -> bool:
    height, width = grid.shape
    reserved = _expanded((0, 0, 0, 3), pad=2, height=height, width=width)
    blocked = [reserved]

    order = count_choices[:]
    random.shuffle(order)
    for opening_count in order:
        placed = False
        for _ in range(200):
            box_h = random.randint(6, min(10, height - 3))
            box_w = random.randint(6, min(10, width - 3))
            if box_h >= height or box_w >= width:
                continue
            r0 = random.randint(1, height - box_h - 1)
            c0 = random.randint(1, width - box_w - 1)
            box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            expanded = _expanded(box, pad=1, height=height, width=width)
            if any(_boxes_overlap(expanded, other) for other in blocked):
                continue
            patch = _draw_container(box_h, box_w, opening_count)
            grid[r0 : r0 + box_h, c0 : c0 + box_w] = patch
            blocked.append(expanded)
            placed = True
            break
        if not placed:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(300):
        height = random.randint(16, 30)
        width = random.randint(16, 30)
        grid = np.full((height, width), BLACK, dtype=int)
        grid[0, :4] = np.array(PALETTE, dtype=int)

        container_count = random.randint(2, 4)
        opening_counts = random.sample([1, 2, 3, 4], k=container_count)
        if not _place_containers(grid, opening_counts):
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
