from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREY = 5

ORTH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in ORTH4:
        yield r + dr, c + dc


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components_mask(mask)
    return len(comps) == 1


def _rotate90(center: tuple[int, int], cell: tuple[int, int]) -> tuple[int, int]:
    cr, cc = center
    r, c = cell
    dr = r - cr
    dc = c - cc
    return cr - dc, cc + dr


def _orbit_from_cell(center: tuple[int, int], cell: tuple[int, int]) -> list[tuple[int, int]]:
    pts = [cell]
    for _ in range(3):
        pts.append(_rotate90(center, pts[-1]))
    return pts


def _orbit_offsets(rep: tuple[int, int]) -> list[tuple[int, int]]:
    a, b = rep
    if (a, b) == (0, 0):
        return [(0, 0)]
    return [(-a, b), (b, a), (a, -b), (-b, -a)]


def _orbit_cells(center: tuple[int, int], rep: tuple[int, int]) -> list[tuple[int, int]]:
    cr, cc = center
    return [(cr + dr, cc + dc) for dr, dc in _orbit_offsets(rep)]


def _line_component(grid: np.ndarray) -> tuple[str, np.ndarray]:
    grey = _components_mask(grid == GREY)
    assert len(grey) == 1, "expected exactly one grey line component"
    coords = grey[0]
    r0, c0, r1, c1 = _bbox_coords(coords)
    if r0 == r1:
        assert len(coords) == c1 - c0 + 1, "grey component must be a 1-pixel-thick straight line"
        return "horizontal", coords
    assert c0 == c1, "grey component must be horizontal or vertical"
    assert len(coords) == r1 - r0 + 1, "grey component must be a 1-pixel-thick straight line"
    return "vertical", coords


def _largest_blue_component(grid: np.ndarray) -> np.ndarray:
    blue = _components_mask(grid == BLUE)
    assert blue, "expected at least one blue component"
    sizes = [len(comp) for comp in blue]
    best = max(sizes)
    assert sizes.count(best) == 1, "expected a unique largest blue component"
    return blue[sizes.index(best)]


def _infer_center(grid: np.ndarray) -> tuple[int, int]:
    orientation, line = _line_component(grid)
    blue = _largest_blue_component(grid)
    br0, bc0, br1, bc1 = _bbox_coords(blue)
    if orientation == "horizontal":
        row = int(line[0, 0])
        col = (bc0 + bc1) // 2
        return row, col
    col = int(line[0, 1])
    row = (br0 + br1) // 2
    return row, col


def _classify_reds(grid: np.ndarray, center: tuple[int, int]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    red_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == RED)}
    defects: set[tuple[int, int]] = set()
    features: set[tuple[int, int]] = set()
    for cell in red_cells:
        peers = _orbit_from_cell(center, cell)[1:]
        if any(peer in red_cells for peer in peers):
            features.add(cell)
        else:
            defects.add(cell)
    return defects, features


def _apply_with_center(grid: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    line_mask = out == GREY

    defects, features = _classify_reds(out, center)
    for r, c in defects:
        out[r, c] = BLUE

    blue_cells = [tuple(map(int, rc)) for rc in np.argwhere(out == BLUE)]
    for cell in blue_cells:
        for rr, cc in _orbit_from_cell(center, cell):
            if _in_bounds(out.shape, rr, cc) and out[rr, cc] == BACKGROUND:
                out[rr, cc] = BLUE

    feature_cells = [tuple(map(int, rc)) for rc in np.argwhere(out == RED)]
    for cell in feature_cells:
        for rr, cc in _orbit_from_cell(center, cell):
            if _in_bounds(out.shape, rr, cc) and not line_mask[rr, cc]:
                out[rr, cc] = RED

    out[line_mask] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair singleton red defects and then complete the blue object by quarter-turn rotation around the line-indicated center.

    Input structure:
    - The grid has black background, one straight grey cue line, a large blue 4-connected object, and 2-4 red marker pixels.
    - The grey line is either horizontal or vertical and supplies one coordinate of the rotation center.
    - The blue object's tight bbox supplies the other center coordinate, while red markers may be true rotational features or embedded defects.

    Rule:
    - Find the grey line and the largest blue component.
    - If the grey line is horizontal, the center is `(line_row, blue_bbox_center_col)`; if vertical, it is `(blue_bbox_center_row, line_col)`.
    - For each red marker, inspect its 90/180/270-degree rotated counterparts around that center: if none of them is red, recolor that marker blue as a defect; otherwise keep it red as a true feature.
    - After repairing defects, close the blue object under 0/90/180/270-degree rotation, then close the surviving red features the same way, with red overwriting blue and the grey line preserved.

    Key insight:
    - The center cannot be recovered from the blue object alone: one coordinate comes from the grey line, and that center is needed to decide which red pixels are defects before the rotational completion becomes valid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    center = _infer_center(g)
    return _apply_with_center(g, center)


def _support_reps(radius: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    mandatory = [(0, 0)] + [(k, 0) for k in range(1, radius + 1)] + [(1, 1)]
    extra_pool = []
    for a in range(2, radius + 1):
        for b in range(1, a + 1):
            rep = (a, b)
            if rep not in mandatory:
                extra_pool.append(rep)
    extra_pool.sort(key=lambda rep: (rep[0] + rep[1], rep[0], rep[1]))
    return mandatory, extra_pool


def _touches_selected(rep: tuple[int, int], selected: set[tuple[int, int]], center: tuple[int, int]) -> bool:
    cells = set(_orbit_cells(center, rep))
    for other in selected:
        others = set(_orbit_cells(center, other))
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in others:
                    return True
    return False


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _expected_output(
    shape: tuple[int, int],
    center: tuple[int, int],
    line_cells: list[tuple[int, int]],
    support: list[tuple[int, int]],
    feature_rep: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    _paint_cells(out, line_cells, GREY)
    for rep in support:
        _paint_cells(out, _orbit_cells(center, rep), BLUE)
    _paint_cells(out, _orbit_cells(center, feature_rep), RED)
    return out


def _adjacent_subset(center: tuple[int, int], rep: tuple[int, int], anchor_blue: set[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    cells = _orbit_cells(center, rep)
    good = []
    for cell in cells:
        if any(nb in anchor_blue for nb in _neighbors4(*cell)):
            good.append(cell)
    if not good:
        good = cells[:]
    random.shuffle(good)
    chosen = good[:count]
    if len(chosen) < count:
        for cell in cells:
            if cell not in chosen:
                chosen.append(cell)
            if len(chosen) == count:
                break
    return chosen


def _axis_pattern(
    center: tuple[int, int],
    radius: int,
    orientation: str,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    cells = _orbit_cells(center, (radius, 0))
    # [top, right, bottom, left]
    if orientation == "horizontal":
        defect_idx = random.choice([0, 2])
        keep = [1, 2, 3] if defect_idx == 0 else [0, 1, 3]
    else:
        defect_idx = random.choice([1, 3])
        keep = [0, 2, 3] if defect_idx == 1 else [0, 1, 2]
    return [cells[i] for i in keep], cells[defect_idx]


def _alt_centers(inp: np.ndarray) -> list[tuple[int, int]]:
    centers = []
    blue = np.argwhere(inp == BLUE)
    if len(blue):
        br0, bc0, br1, bc1 = _bbox_coords(blue)
        centers.append(((br0 + br1) // 2, (bc0 + bc1) // 2))
    fg = np.argwhere((inp != BACKGROUND) & (inp != GREY))
    if len(fg):
        fr0, fc0, fr1, fc1 = _bbox_coords(fg)
        centers.append(((fr0 + fr1) // 2, (fc0 + fc1) // 2))
    uniq = []
    seen = set()
    for center in centers:
        if center not in seen:
            seen.add(center)
            uniq.append(center)
    return uniq


def _validate_example(
    inp: np.ndarray,
    expected: np.ndarray,
    true_center: tuple[int, int],
    orientation: str,
    feature_rep: tuple[int, int],
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False
    if int(np.count_nonzero(inp == RED)) not in {2, 3, 4}:
        return False

    try:
        guessed = _infer_center(inp)
    except AssertionError:
        return False
    if guessed != true_center:
        return False

    blue_mask = inp == BLUE
    if not blue_mask.any() or not _is_connected(blue_mask):
        return False

    defects, features = _classify_reds(inp, true_center)
    if not defects or not features:
        return False

    if not np.array_equal(main(inp), expected):
        return False

    no_repair = inp.copy()
    # Wrong shortcut: treat every red pixel as a real feature.
    reds = [tuple(map(int, rc)) for rc in np.argwhere(no_repair == RED)]
    naive_all_red = no_repair.copy()
    line_mask = naive_all_red == GREY
    blue_cells = [tuple(map(int, rc)) for rc in np.argwhere(naive_all_red == BLUE)]
    for cell in blue_cells:
        for rr, cc in _orbit_from_cell(true_center, cell):
            if _in_bounds(naive_all_red.shape, rr, cc) and naive_all_red[rr, cc] == BACKGROUND:
                naive_all_red[rr, cc] = BLUE
    for cell in reds:
        for rr, cc in _orbit_from_cell(true_center, cell):
            if _in_bounds(naive_all_red.shape, rr, cc) and not line_mask[rr, cc]:
                naive_all_red[rr, cc] = RED
    if np.array_equal(naive_all_red, expected):
        return False

    repaired_only = inp.copy()
    for r, c in defects:
        repaired_only[r, c] = BLUE
    if np.array_equal(repaired_only, expected):
        return False

    feature_present = sum(int(inp[cell] == RED) for cell in _orbit_cells(true_center, feature_rep))
    if feature_present not in {2, 3}:
        return False

    for center in _alt_centers(inp):
        if center != true_center and np.array_equal(_apply_with_center(inp, center), expected):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = random.randint(14, 18)
        w = random.randint(14, 18)
        orientation = random.choice(["horizontal", "vertical"])
        radius = random.choice([3, 4])

        if orientation == "horizontal":
            if w < 2 * radius + 8 or h < 2 * radius + 3:
                continue
            cr = random.randint(radius + 1, h - radius - 2)
            cc = random.randint(radius + 6, w - radius - 2)
        else:
            if h < 2 * radius + 8 or w < 2 * radius + 3:
                continue
            cr = random.randint(radius + 6, h - radius - 2)
            cc = random.randint(radius + 1, w - radius - 2)
        center = (cr, cc)

        mandatory, extra_pool = _support_reps(radius)
        selected: set[tuple[int, int]] = set(mandatory)
        extra_min = 1 if radius == 3 else 0
        extra_max = 2
        extra_count = random.randint(extra_min, min(extra_max, len(extra_pool)))
        for rep in extra_pool:
            if len(selected) - len(mandatory) >= extra_count:
                break
            if _touches_selected(rep, selected, center):
                if random.random() < 0.7:
                    selected.add(rep)
        while len(selected) - len(mandatory) < extra_count:
            candidates = [rep for rep in extra_pool if rep not in selected and _touches_selected(rep, selected, center)]
            if not candidates:
                break
            selected.add(random.choice(candidates))
        support = sorted(selected, key=lambda rep: (rep[0], rep[1]))

        feature_candidates = [rep for rep in support if rep not in {(0, 0), (radius, 0)} and rep[1] > 0]
        if not feature_candidates:
            continue
        feature_rep = random.choice(feature_candidates)

        spare_for_second_defect = [rep for rep in support if rep not in {(0, 0), feature_rep, (radius, 0)} and rep[1] > 0]
        second_defect_rep = None
        feature_count = random.choice([2, 3])
        if feature_count == 2 and spare_for_second_defect and random.random() < 0.5:
            second_defect_rep = random.choice(spare_for_second_defect)

        expected = _expected_output((h, w), center, [], support, feature_rep)
        visible_blue = int(np.count_nonzero(expected == BLUE))
        if not (15 <= visible_blue <= 25):
            continue

        inp = np.zeros((h, w), dtype=int)

        left_edge = cc - radius
        top_edge = cr - radius
        if orientation == "horizontal":
            line_end = left_edge - 2
            if line_end < 4:
                continue
            line_len = random.randint(4, min(7, line_end))
            c0 = line_end - line_len + 1
            line_cells = [(cr, c) for c in range(c0, line_end + 1)]
        else:
            line_end = top_edge - 2
            if line_end < 4:
                continue
            line_len = random.randint(4, min(7, line_end))
            r0 = line_end - line_len + 1
            line_cells = [(r, cc) for r in range(r0, line_end + 1)]
        _paint_cells(inp, line_cells, GREY)

        # Stable connected blue core.
        _paint_cells(inp, _orbit_cells(center, (0, 0)), BLUE)
        for rep in [(k, 0) for k in range(1, radius)]:
            _paint_cells(inp, _orbit_cells(center, rep), BLUE)

        axis_keep, axis_defect = _axis_pattern(center, radius, orientation)
        _paint_cells(inp, axis_keep, BLUE)
        inp[axis_defect] = RED

        feature_cells = _orbit_cells(center, feature_rep)
        random.shuffle(feature_cells)
        for cell in feature_cells[:feature_count]:
            inp[cell] = RED

        if second_defect_rep is not None:
            second_cells = _orbit_cells(center, second_defect_rep)
            inp[random.choice(second_cells)] = RED

        anchor_blue = {tuple(map(int, rc)) for rc in np.argwhere(inp == BLUE)}
        partial_candidates = [
            rep
            for rep in support
            if rep not in {(0, 0), (radius, 0), feature_rep}
            and rep != second_defect_rep
            and rep[1] > 0
        ]
        partial_rep = random.choice(partial_candidates) if partial_candidates and random.random() < 0.65 else None

        for rep in support:
            if rep == (0, 0) or rep in [(k, 0) for k in range(1, radius)] or rep == (radius, 0):
                continue
            if rep == feature_rep or rep == second_defect_rep:
                continue
            cells = _orbit_cells(center, rep)
            if rep == partial_rep:
                keep_count = random.choice([1, 2])
                keep = _adjacent_subset(center, rep, anchor_blue, keep_count)
                _paint_cells(inp, keep, BLUE)
            else:
                _paint_cells(inp, cells, BLUE)
            anchor_blue = {tuple(map(int, rc)) for rc in np.argwhere(inp == BLUE)}

        if np.any((inp == GREY) & ((inp == BLUE) | (inp == RED))):
            continue

        expected = _expected_output((h, w), center, line_cells, support, feature_rep)
        if not _validate_example(inp, expected, center, orientation, feature_rep):
            continue
        return inp

    raise RuntimeError("failed to generate a valid example")
