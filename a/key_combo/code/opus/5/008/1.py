from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Missing-corner encoding for clockwise quarter turns.
ROT_TO_MISSING = {
    0: (1, 1),  # upper-left missing
    1: (1, 3),  # upper-right missing
    2: (3, 3),  # lower-right missing
    3: (3, 1),  # lower-left missing
}
MISSING_TO_ROT = {v: k for k, v in ROT_TO_MISSING.items()}

# Place the 5x5 square so the point touches the middle of one side.
SQUARE_SIDES = {
    "below": (1, -2),
    "above": (-5, -2),
    "right": (-2, 1),
    "left": (-2, -5),
}

GENERATE_ATTEMPTS = 600


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == int(color)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def rotate_cw(grid: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(np.array(grid, dtype=int), -int(quarter_turns))


def rect_distance_to_point(point: tuple[int, int], rect: tuple[int, int, int, int]) -> int:
    r, c = point
    r0, c0, r1, c1 = rect
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return int(dr + dc)


def rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def random_connected_cells(rows: int, cols: int, n: int) -> list[tuple[int, int]] | None:
    start = (random.randrange(rows), random.randrange(cols))
    cells = {start}
    frontier = set()
    for nr, nc in neighbors4(*start):
        if 0 <= nr < rows and 0 <= nc < cols:
            frontier.add((nr, nc))
    while len(cells) < n and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for nr, nc in neighbors4(*cell):
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in cells:
                frontier.add((nr, nc))
    if len(cells) != n:
        return None
    return sorted(cells)


def canonical_mask(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.tolist())


def random_template_patch() -> np.ndarray:
    dims = random.choice([(3, 4), (4, 3)])
    rows, cols = dims
    area_lo = max(4, rows + cols - 2)
    area_hi = min(8, rows * cols - 1)
    for _ in range(300):
        area = random.randint(area_lo, area_hi)
        cells = random_connected_cells(rows, cols, area)
        if cells is None:
            continue
        coords = np.array(cells, dtype=int)
        if bbox_coords(coords) != (0, 0, rows - 1, cols - 1):
            continue
        patch = np.zeros((rows, cols), dtype=int)
        patch[coords[:, 0], coords[:, 1]] = BLUE
        sigs = {
            (rotate_cw(patch, k).shape, canonical_mask(rotate_cw(patch, k) != 0))
            for k in range(4)
        }
        if len(sigs) < 4:
            continue
        return patch
    raise RuntimeError("failed to build an asymmetric template")


def draw_missing_corner_square(grid: np.ndarray, top: int, left: int, rotation: int) -> None:
    grid[top : top + 5, left : left + 5] = BG
    grid[top, left : left + 5] = GREY
    grid[top + 4, left : left + 5] = GREY
    grid[top : top + 5, left] = GREY
    grid[top : top + 5, left + 4] = GREY
    for rr, cc in [(1, 1), (1, 3), (3, 1), (3, 3)]:
        if (rr, cc) != ROT_TO_MISSING[int(rotation)]:
            grid[top + rr, left + cc] = GREEN


def find_missing_corner_squares(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    squares: list[dict] = []
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            sub = grid[r0 : r0 + 5, c0 : c0 + 5]
            if not (
                np.all(sub[0, :] == GREY)
                and np.all(sub[4, :] == GREY)
                and np.all(sub[:, 0] == GREY)
                and np.all(sub[:, 4] == GREY)
            ):
                continue
            inner = sub[1:4, 1:4]
            ok = True
            missing = None
            for rr in range(3):
                for cc in range(3):
                    expect_corner = (rr, cc) in {(0, 0), (0, 2), (2, 0), (2, 2)}
                    val = int(inner[rr, cc])
                    if expect_corner:
                        if val == BG:
                            if missing is not None:
                                ok = False
                            missing = (rr + 1, cc + 1)
                        elif val != GREEN:
                            ok = False
                    elif val != BG:
                        ok = False
            if not ok or missing not in MISSING_TO_ROT:
                continue
            squares.append(
                {
                    "bbox": (int(r0), int(c0), int(r0 + 4), int(c0 + 4)),
                    "rotation": int(MISSING_TO_ROT[missing]),
                }
            )
    return squares


def solve_details(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)

    blue_comps = components_of_color(g, BLUE)
    assert blue_comps, "missing blue template"
    template = max(blue_comps, key=lambda comp: int(comp.shape[0]))
    template_area = int(template.shape[0])
    assert template_area >= 4, "template must be the large blue component"
    assert sum(int(comp.shape[0]) == template_area for comp in blue_comps) == 1, "template must be uniquely largest"

    template_bbox = bbox_coords(template)
    template_patch = crop_bbox(g, template_bbox)
    templ_h = bbox_h(template_bbox)
    templ_w = bbox_w(template_bbox)

    blue_singletons = [tuple(map(int, comp[0])) for comp in blue_comps if int(comp.shape[0]) == 1]
    red_comps = components_of_color(g, RED)
    red_singletons = [tuple(map(int, comp[0])) for comp in red_comps if int(comp.shape[0]) == 1]
    assert len(blue_singletons) == 1, "expected exactly one blue singleton origin"
    assert len(red_singletons) >= 2, "expected red target singletons"

    layout_points = [(blue_singletons[0], BLUE)] + [(pt, RED) for pt in red_singletons]
    squares = find_missing_corner_squares(g)
    assert len(squares) == len(layout_points), "every layout point needs one square"

    point_infos = []
    for point, color in layout_points:
        dists = [rect_distance_to_point(point, sq["bbox"]) for sq in squares]
        best = min(dists)
        assert dists.count(best) == 1, "nearest square must be unique"
        sq = squares[dists.index(best)]
        point_infos.append({"point": point, "color": int(color), "rotation": int(sq["rotation"])})

    origin = blue_singletons[0]
    for info in point_infos:
        dr = int(info["point"][0] - origin[0])
        dc = int(info["point"][1] - origin[1])
        assert dr % templ_h == 0 and dc % templ_w == 0, "layout must use template bbox spacing"
        info["delta"] = (int(dr // templ_h), int(dc // templ_w))

    h, w = g.shape
    cover = np.zeros((h, w), dtype=int)
    origin_cover = np.zeros((h, w), dtype=bool)
    base_r, base_c = template_bbox[0], template_bbox[1]

    for info in point_infos:
        patch = rotate_cw(template_patch, info["rotation"])
        pr, pc = patch.shape
        top = int(base_r + info["delta"][0] * templ_h)
        left = int(base_c + info["delta"][1] * templ_w)
        assert 0 <= top and top + pr <= h and 0 <= left and left + pc <= w, "stamped copy must stay in bounds"
        mask = patch != BG
        cover[top : top + pr, left : left + pc] += mask.astype(int)
        if info["point"] == origin:
            origin_cover[top : top + pr, left : left + pc] |= mask

    out = np.zeros_like(g)
    out[cover > 0] = BLUE
    out[(cover == 1) & origin_cover] = RED
    out[cover >= 2] = YELLOW

    return {
        "template_patch": template_patch,
        "template_bbox": template_bbox,
        "point_infos": point_infos,
        "squares": squares,
        "cover": cover,
        "output": out,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp rotated copies of a blue template from lattice markers and missing-corner square codes.

    Input structure:
    - The background is black.
    - The largest blue 4-connected component is an asymmetric template; elsewhere there is one blue singleton origin and several red singleton lattice points.
    - Each lattice point has a nearby 5x5 grey border square whose 3x3 interior has exactly one missing green corner, encoding a rotation.

    Rule:
    - Identify the largest blue component as the template and use its bbox height and width as the lattice spacing.
    - Match each blue/red singleton to its unique nearest missing-corner square; decode upper-left, upper-right, lower-right, lower-left missing corners as 0, 90, 180, 270 degrees clockwise.
    - Convert every singleton offset from the blue origin into lattice units, rotate the template independently for that point, and stamp it at the corresponding translated location.
    - Paint the origin's stamped copy red instead of blue, repaint any cell covered by two or more copies yellow, and erase the original markers and squares.

    Key insight:
    - Position comes from the singleton lattice, but orientation comes from separate nearby missing-corner boxes, so the final overlaps can only be predicted by combining both decodings.
    """
    return solve_details(grid)["output"]


def has_required_adjacency(offsets: list[tuple[int, int]], template_patch: np.ndarray) -> bool:
    h, w = template_patch.shape
    pts = set(offsets)
    if h < w:
        return any((r + 1, c) in pts for r, c in pts)
    return any((r, c + 1) in pts for r, c in pts)


def random_offsets(template_patch: np.ndarray) -> list[tuple[int, int]]:
    for _ in range(300):
        count = random.randint(3, 5)
        rows = random.randint(2, 3)
        cols = random.randint(2, 3)
        if count > rows * cols:
            continue
        cells = random_connected_cells(rows, cols, count)
        if cells is None:
            continue
        origin = random.choice(cells)
        offsets = sorted((r - origin[0], c - origin[1]) for r, c in cells)
        if has_required_adjacency(offsets, template_patch):
            return offsets
    raise RuntimeError("failed to create layout offsets")


def choose_rotations(offsets: list[tuple[int, int]], template_patch: np.ndarray, grid_shape: tuple[int, int]) -> dict[tuple[int, int], int] | None:
    h, w = template_patch.shape
    relevant_pairs: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    pts = set(offsets)
    if h < w:
        for r, c in pts:
            if (r + 1, c) in pts:
                relevant_pairs.append(((r, c), (r + 1, c), (r, c)))
    else:
        for r, c in pts:
            if (r, c + 1) in pts:
                relevant_pairs.append(((r, c), (r, c + 1), (r, c)))
    if not relevant_pairs:
        return None

    grid_h, grid_w = grid_shape
    for _ in range(300):
        pair = random.choice(relevant_pairs)
        rotations = {offset: random.randrange(4) for offset in offsets}
        driver = pair[2]
        rotations[driver] = random.choice([1, 3])
        if len(set(rotations.values())) < 2:
            rotations[random.choice(offsets)] = (rotations[driver] + 2) % 4

        extents = []
        for dr, dc in offsets:
            patch = rotate_cw(template_patch, rotations[(dr, dc)])
            pr, pc = patch.shape
            extents.append((dr * h, dc * w, dr * h + pr - 1, dc * w + pc - 1))

        min_r = min(v[0] for v in extents)
        min_c = min(v[1] for v in extents)
        max_r = max(v[2] for v in extents)
        max_c = max(v[3] for v in extents)
        if max_r - min_r + 1 > grid_h or max_c - min_c + 1 > grid_w:
            continue

        canvas = np.zeros(grid_shape, dtype=int)
        overlap = np.zeros(grid_shape, dtype=int)
        base_r = -min_r
        base_c = -min_c
        for dr, dc in offsets:
            patch = rotate_cw(template_patch, rotations[(dr, dc)])
            pr, pc = patch.shape
            top = base_r + dr * h
            left = base_c + dc * w
            mask = patch != BG
            overlap[top : top + pr, left : left + pc] += mask.astype(int)
            canvas[top : top + pr, left : left + pc] |= mask.astype(int)

        if np.any(overlap >= 2):
            return rotations
    return None


def choose_template_position(
    grid_shape: tuple[int, int],
    template_patch: np.ndarray,
    offsets: list[tuple[int, int]],
    rotations: dict[tuple[int, int], int],
) -> tuple[int, int] | None:
    h, w = template_patch.shape
    grid_h, grid_w = grid_shape
    extents = [(0, 0, h - 1, w - 1)]
    for dr, dc in offsets:
        patch = rotate_cw(template_patch, rotations[(dr, dc)])
        pr, pc = patch.shape
        extents.append((dr * h, dc * w, dr * h + pr - 1, dc * w + pc - 1))

    min_r = min(v[0] for v in extents)
    min_c = min(v[1] for v in extents)
    max_r = max(v[2] for v in extents)
    max_c = max(v[3] for v in extents)
    lo_r = -min_r
    hi_r = grid_h - 1 - max_r
    lo_c = -min_c
    hi_c = grid_w - 1 - max_c
    if lo_r > hi_r or lo_c > hi_c:
        return None
    return random.randint(lo_r, hi_r), random.randint(lo_c, hi_c)


def square_topleft(point: tuple[int, int], side: str) -> tuple[int, int]:
    dr, dc = SQUARE_SIDES[side]
    return int(point[0] + dr), int(point[1] + dc)


def choose_layout_anchor_and_sides(
    grid_shape: tuple[int, int],
    offsets: list[tuple[int, int]],
    template_bbox: tuple[int, int, int, int],
    template_patch: np.ndarray,
    rotations: dict[tuple[int, int], int],
) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int]]] | None:
    h, w = template_patch.shape
    grid_h, grid_w = grid_shape
    min_dr = min(dr for dr, _dc in offsets)
    max_dr = max(dr for dr, _dc in offsets)
    min_dc = min(dc for _dr, dc in offsets)
    max_dc = max(dc for _dr, dc in offsets)
    lo_r = -min_dr * h
    hi_r = grid_h - 1 - max_dr * h
    lo_c = -min_dc * w
    hi_c = grid_w - 1 - max_dc * w
    if lo_r > hi_r or lo_c > hi_c:
        return None

    template_rect = template_bbox
    anchor_candidates = [
        (r, c)
        for r in range(lo_r, hi_r + 1)
        for c in range(lo_c, hi_c + 1)
    ]
    random.shuffle(anchor_candidates)

    for anchor in anchor_candidates[:120]:
        points = {
            off: (int(anchor[0] + off[0] * h), int(anchor[1] + off[1] * w))
            for off in offsets
        }
        point_list = list(points.values())
        if any(not (0 <= r < grid_h and 0 <= c < grid_w) for r, c in point_list):
            continue
        if any(template_rect[0] <= r <= template_rect[2] and template_rect[1] <= c <= template_rect[3] for r, c in point_list):
            continue

        order = sorted(offsets, key=lambda off: sum(rect_distance_to_point(points[off], template_rect) for _ in [0]))
        chosen: dict[tuple[int, int], tuple[int, int]] = {}
        rects: list[tuple[int, int, int, int]] = []

        def backtrack(idx: int) -> bool:
            if idx == len(order):
                squares = [
                    {"offset": off, "bbox": (tl[0], tl[1], tl[0] + 4, tl[1] + 4)}
                    for off, tl in chosen.items()
                ]
                for off, point in points.items():
                    dists = [rect_distance_to_point(point, sq["bbox"]) for sq in squares]
                    best = min(dists)
                    if best != 1 or dists.count(best) != 1:
                        return False
                    own_bbox = (chosen[off][0], chosen[off][1], chosen[off][0] + 4, chosen[off][1] + 4)
                    if rect_distance_to_point(point, own_bbox) != 1:
                        return False
                return True

            off = order[idx]
            point = points[off]
            sides = list(SQUARE_SIDES.keys())
            random.shuffle(sides)
            for side in sides:
                top, left = square_topleft(point, side)
                rect = (top, left, top + 4, left + 4)
                if top < 0 or left < 0 or top + 5 > grid_h or left + 5 > grid_w:
                    continue
                if rects_overlap(rect, template_rect, pad=1):
                    continue
                if any(rects_overlap(rect, prev, pad=1) for prev in rects):
                    continue
                bad_point = False
                for other in point_list:
                    if other == point:
                        continue
                    if rect_distance_to_point(other, rect) <= 1:
                        bad_point = True
                        break
                if bad_point:
                    continue
                chosen[off] = (top, left)
                rects.append(rect)
                if backtrack(idx + 1):
                    return True
                rects.pop()
                chosen.pop(off)
            return False

        if backtrack(0):
            return anchor, chosen
    return None


def validate_generated(grid: np.ndarray) -> None:
    details = solve_details(grid)
    out = details["output"]
    point_infos = details["point_infos"]
    template_patch = details["template_patch"]
    cover = details["cover"]

    assert grid.ndim == 2 and out.ndim == 2
    assert max(grid.shape) <= 30 and max(out.shape) <= 30
    assert 3 <= len(point_infos) <= 5
    assert len(set(info["rotation"] for info in point_infos)) >= 2
    assert np.any(cover >= 2), "must contain an overlap"
    assert np.any(out == RED), "origin recolor must be visible"
    assert np.any(out == BLUE), "non-origin copies must remain visible"
    assert np.any(out == YELLOW), "overlap recolor must be visible"
    assert np.sum(out == YELLOW) <= np.sum(out != BG) // 2, "overlap should stay a minority"

    sigs = {
        (rotate_cw(template_patch, k).shape, canonical_mask(rotate_cw(template_patch, k) != 0))
        for k in range(4)
    }
    assert len(sigs) == 4, "template must be asymmetric under rotation"


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid_h = random.randint(20, 28)
        grid_w = random.randint(20, 28)
        grid = np.zeros((grid_h, grid_w), dtype=int)

        template_patch = random_template_patch()
        offsets = random_offsets(template_patch)
        rotations = choose_rotations(offsets, template_patch, grid.shape)
        if rotations is None:
            continue

        template_pos = choose_template_position(grid.shape, template_patch, offsets, rotations)
        if template_pos is None:
            continue
        template_top, template_left = template_pos
        template_bbox = (
            template_top,
            template_left,
            template_top + template_patch.shape[0] - 1,
            template_left + template_patch.shape[1] - 1,
        )

        layout = choose_layout_anchor_and_sides(grid.shape, offsets, template_bbox, template_patch, rotations)
        if layout is None:
            continue
        origin_point, square_positions = layout

        grid[template_top : template_top + template_patch.shape[0], template_left : template_left + template_patch.shape[1]] = template_patch

        h, w = template_patch.shape
        for off in offsets:
            point = (int(origin_point[0] + off[0] * h), int(origin_point[1] + off[1] * w))
            grid[point] = BLUE if off == (0, 0) else RED
            sq_top, sq_left = square_positions[off]
            draw_missing_corner_square(grid, sq_top, sq_left, rotations[off])

        try:
            validate_generated(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
