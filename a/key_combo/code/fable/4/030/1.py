import random
from collections import deque

import numpy as np


BG = 0
DOT = 5
CUP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
ORIENTATIONS = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_monochrome_components(grid: np.ndarray, ignore: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in ignore:
                continue

            seen[r, c] = True
            queue = deque([(r, c)])
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            components.append({"color": color, "coords": coords})

    return components


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    coords = set()
    for c in range(c0, c1 + 1):
        coords.add((r0, c))
        coords.add((r1, c))
    for r in range(r0, r1 + 1):
        coords.add((r, c0))
        coords.add((r, c1))
    return coords


def _detect_cup(component: dict) -> dict:
    coords = component["coords"]
    color = int(component["color"])
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    cells = set(coords)

    border = _border_coords(r0, c0, r1, c1)
    if not cells <= border:
        raise AssertionError("cup must lie on a rectangle border")

    missing = border - cells
    if len(missing) < 2:
        raise AssertionError("cup must have an opening of width at least 2")

    side_spans = {
        "top": [c for c in range(c0, c1 + 1) if (r0, c) in missing],
        "bottom": [c for c in range(c0, c1 + 1) if (r1, c) in missing],
        "left": [r for r in range(r0, r1 + 1) if (r, c0) in missing],
        "right": [r for r in range(r0, r1 + 1) if (r, c1) in missing],
    }
    open_sides = [side for side, span in side_spans.items() if span]
    if len(open_sides) != 1:
        raise AssertionError("cup must have exactly one open side segment")

    side = open_sides[0]
    span = side_spans[side]
    if span[-1] - span[0] + 1 != len(span):
        raise AssertionError("opening must be contiguous")

    if side in ("top", "bottom"):
        if span[0] == c0 or span[-1] == c1:
            raise AssertionError("opening must leave both corners intact")
        opening_width = len(span)
        open_range = (span[0], span[-1])
    else:
        if span[0] == r0 or span[-1] == r1:
            raise AssertionError("opening must leave both corners intact")
        opening_width = len(span)
        open_range = (span[0], span[-1])

    if len(cells) != len(border) - opening_width:
        raise AssertionError("unexpected extra holes in cup outline")

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "opening_width": opening_width,
        "open_range": open_range,
    }


def _band_membership(cup: dict, r: int, c: int) -> tuple[bool, int]:
    r0, c0, r1, c1 = cup["bbox"]
    start, end = cup["open_range"]
    side = cup["side"]

    if side == "top":
        if r < r0 and start <= c <= end:
            return True, r0 - r
    elif side == "bottom":
        if r > r1 and start <= c <= end:
            return True, r - r1
    elif side == "left":
        if c < c0 and start <= r <= end:
            return True, c0 - c
    else:
        if c > c1 and start <= r <= end:
            return True, c - c1

    return False, -1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only the gray dots that land in a cup's opening band at distances matching that opening width.

    Input structure:
    - The background is black.
    - Colored foreground objects are rectangular cup outlines: one-pixel-thick rectangle borders with one contiguous opening on exactly one side.
    - Gray singleton dots are scattered elsewhere on the canvas.

    Rule:
    - For each cup, read its opening side and opening width from the missing border segment.
    - The opening defines an outward strip whose width equals the opening width.
    - A gray dot inside that strip is recolored to the cup color exactly when its perpendicular distance from the cup side is a positive multiple of the opening width; every other gray dot is erased to background.

    Key insight:
    - The missing border segment is both a direction cue and a period cue: the cup claims only those strip-aligned dots whose distances are in phase with the opening width.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    components = _find_monochrome_components(g, ignore={bg, DOT})
    cups = [_detect_cup(component) for component in components]
    assert cups, "expected at least one cup"

    out = g.copy()
    out[g == DOT] = bg

    dot_coords = np.argwhere(g == DOT)
    for rr, cc in dot_coords:
        r, c = int(rr), int(cc)
        matches = []
        for cup in cups:
            inside, dist = _band_membership(cup, r, c)
            if inside:
                matches.append((cup, dist))

        assert len(matches) <= 1, "dots must belong to at most one cup band"
        if not matches:
            continue

        cup, dist = matches[0]
        if dist > 0 and dist % cup["opening_width"] == 0:
            out[r, c] = cup["color"]

    return out


def _make_cup_patch(side: str, height: int, width: int, opening_width: int, offset: int, color: int) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    if side == "top":
        patch[0, offset : offset + opening_width] = BG
    elif side == "bottom":
        patch[-1, offset : offset + opening_width] = BG
    elif side == "left":
        patch[offset : offset + opening_width, 0] = BG
    else:
        patch[offset : offset + opening_width, -1] = BG

    return patch


def _choose_box_dims(side: str, opening_width: int) -> tuple[int, int, int]:
    if side in ("top", "bottom"):
        width = random.randint(opening_width + 2, min(10, opening_width + 5))
        height = random.randint(4, 8)
        offset = random.randint(1, width - opening_width - 1)
    else:
        height = random.randint(opening_width + 2, min(10, opening_width + 5))
        width = random.randint(4, 8)
        offset = random.randint(1, height - opening_width - 1)
    return height, width, offset


def _placement_ranges(shape: tuple[int, int], side: str, height: int, width: int, opening_width: int) -> tuple[range, range]:
    h, w = shape
    if side == "top":
        r_lo, r_hi = opening_width + 1, h - height - 1
        c_lo, c_hi = 1, w - width - 1
    elif side == "bottom":
        r_lo, r_hi = 1, h - height - opening_width - 2
        c_lo, c_hi = 1, w - width - 1
    elif side == "left":
        r_lo, r_hi = 1, h - height - 1
        c_lo, c_hi = opening_width + 1, w - width - 1
    else:
        r_lo, r_hi = 1, h - height - 1
        c_lo, c_hi = 1, w - width - opening_width - 2

    if r_lo > r_hi or c_lo > c_hi:
        raise AssertionError("grid too small for cup placement")
    return range(r_lo, r_hi + 1), range(c_lo, c_hi + 1)


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _cup_band_cells(cup: dict, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = cup["bbox"]
    start, end = cup["open_range"]
    side = cup["side"]

    cells = []
    if side == "top":
        for r in range(0, r0):
            for c in range(start, end + 1):
                cells.append((r, c))
    elif side == "bottom":
        for r in range(r1 + 1, h):
            for c in range(start, end + 1):
                cells.append((r, c))
    elif side == "left":
        for r in range(start, end + 1):
            for c in range(0, c0):
                cells.append((r, c))
    else:
        for r in range(start, end + 1):
            for c in range(c1 + 1, w):
                cells.append((r, c))
    return cells


def _random_positions(cells: list[tuple[int, int]], count: int, used: set[tuple[int, int]]) -> list[tuple[int, int]]:
    choices = [cell for cell in cells if cell not in used]
    if len(choices) < count:
        raise AssertionError("not enough free cells")
    return random.sample(choices, count)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    grid = np.full((h, w), BG, dtype=int)

    cup_count = random.choices([2, 3, 4], weights=[1, 2, 3], k=1)[0]
    sides = random.sample(list(ORIENTATIONS), cup_count)
    opening_widths = random.sample(range(2, 7), cup_count)
    colors = random.sample(list(CUP_COLORS), cup_count)

    placed_cups = []
    occupied_boxes = []

    for side, opening_width, color in zip(sides, opening_widths, colors):
        height, width, offset = _choose_box_dims(side, opening_width)
        try:
            row_range, col_range = _placement_ranges((h, w), side, height, width, opening_width)
        except AssertionError:
            return None

        candidates = [(r, c) for r in row_range for c in col_range]
        random.shuffle(candidates)
        placed = False
        for top, left in candidates:
            bbox = (top, left, top + height - 1, left + width - 1)
            if any(not _boxes_separate(bbox, other) for other in occupied_boxes):
                continue

            patch = _make_cup_patch(side, height, width, opening_width, offset, color)
            region = grid[top : top + height, left : left + width]
            if np.any((patch != BG) & (region != BG)):
                continue

            region[patch != BG] = color
            if side in ("top", "bottom"):
                open_range = (left + offset, left + offset + opening_width - 1)
            else:
                open_range = (top + offset, top + offset + opening_width - 1)
            placed_cups.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "side": side,
                    "opening_width": opening_width,
                    "open_range": open_range,
                }
            )
            occupied_boxes.append(bbox)
            placed = True
            break

        if not placed:
            return None

    membership_count = {}
    cup_band_maps = []
    for cup in placed_cups:
        band_cells = _cup_band_cells(cup, grid.shape)
        cup_band_maps.append(set(band_cells))
        for cell in band_cells:
            membership_count[cell] = membership_count.get(cell, 0) + 1

    used = {(r, c) for cup in placed_cups for r in range(cup["bbox"][0], cup["bbox"][2] + 1) for c in range(cup["bbox"][1], cup["bbox"][3] + 1) if grid[r, c] != BG}
    recolored_total = 0
    erased_total = 0

    for cup, band_cells in zip(placed_cups, cup_band_maps):
        by_distance: dict[int, list[tuple[int, int]]] = {}
        unique_cells = [cell for cell in band_cells if membership_count[cell] == 1 and cell not in used]
        for r, c in unique_cells:
            inside, dist = _band_membership(cup, r, c)
            if inside:
                by_distance.setdefault(dist, []).append((r, c))

        good_distances = [dist for dist in by_distance if dist % cup["opening_width"] == 0]
        bad_distances = [dist for dist in by_distance if dist % cup["opening_width"] != 0]
        if not good_distances or not bad_distances:
            return None

        chosen_good = random.sample(good_distances, k=min(len(good_distances), random.randint(1, 2)))
        chosen_bad = random.sample(bad_distances, k=min(len(bad_distances), random.randint(1, 2)))

        for dist in chosen_good:
            pos = random.choice([cell for cell in by_distance[dist] if cell not in used])
            grid[pos] = DOT
            used.add(pos)
            recolored_total += 1

        for dist in chosen_bad:
            pos = random.choice([cell for cell in by_distance[dist] if cell not in used])
            grid[pos] = DOT
            used.add(pos)
            erased_total += 1

    all_band_cells = set().union(*cup_band_maps) if cup_band_maps else set()
    stray_candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if grid[r, c] == BG and (r, c) not in all_band_cells
    ]
    if len(stray_candidates) < 2:
        return None
    for pos in random.sample(stray_candidates, k=random.randint(2, min(5, len(stray_candidates)))):
        grid[pos] = DOT
        erased_total += 1

    out = main(grid)
    if np.count_nonzero(out == DOT) != 0:
        return None
    if recolored_total == 0 or erased_total == 0:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
