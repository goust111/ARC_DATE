"""ARC task 036-6: long-arm crosses place recolored template stamps."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
MAX_SIZE = 30
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
CORNERS = ("tl", "tr", "bl", "br")


def _map_from_corner(
    corner: str, shape: tuple[int, int], r: int, c: int
) -> tuple[int, int]:
    """Map corner-local (r, c), whose origin is the corner, to grid coordinates."""
    h, w = shape
    if corner == "tl":
        return r, c
    if corner == "tr":
        return r, w - 1 - c
    if corner == "bl":
        return h - 1 - r, c
    if corner == "br":
        return h - 1 - r, w - 1 - c
    raise ValueError("unknown corner")


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Inclusive bbox in the common ARC (row, column) coordinate convention."""
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[set[tuple[int, int]]]:
    """Return 4-connected monochromatic components as coordinate sets."""
    remaining = set(map(tuple, np.argwhere(grid == color)))
    components: list[set[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        queue = deque([seed])
        while queue:
            r, c = queue.popleft()
            for dr, dc in DIRECTIONS:
                nxt = (r + dr, c + dc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    component.add(nxt)
                    queue.append(nxt)
        components.append(component)
    return components


def _find_template(grid: np.ndarray) -> dict:
    """Find the corner L separator and the tight monochrome template inside it."""
    h, w = grid.shape
    candidates = []
    colors = [int(c) for c in np.unique(grid) if int(c) != BG]
    components = {
        color: _components_of_color(grid, color) for color in colors
    }

    for corner in CORNERS:
        for k in range(6, min(h, w) + 1):
            # The separator is the two inner-facing edges of a k x k corner box.
            expected = {
                _map_from_corner(corner, grid.shape, k - 1, i) for i in range(k)
            } | {
                _map_from_corner(corner, grid.shape, i, k - 1) for i in range(k)
            }
            sep_color = int(grid[_map_from_corner(corner, grid.shape, k - 1, k - 1)])
            if sep_color == BG or expected not in components.get(sep_color, []):
                continue

            compartment = {
                _map_from_corner(corner, grid.shape, r, c)
                for r in range(k - 1)
                for c in range(k - 1)
            }
            pattern = {p for p in compartment if int(grid[p]) != BG}
            if not pattern:
                continue
            pattern_colors = {int(grid[p]) for p in pattern}
            if len(pattern_colors) != 1:
                continue
            r0, c0, r1, c1 = _bbox(pattern)
            side = r1 - r0 + 1
            if side != c1 - c0 + 1 or side < 3 or side % 2 == 0:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            mask = patch != BG
            # The stated bbox must really be tight on all four sides.
            if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                continue
            protected = {
                _map_from_corner(corner, grid.shape, r, c)
                for r in range(k)
                for c in range(k)
            }
            candidates.append(
                {
                    "corner": corner,
                    "k": k,
                    "separator_color": sep_color,
                    "template_color": next(iter(pattern_colors)),
                    "mask": mask,
                    "side": side,
                    "protected": protected,
                }
            )

    assert len(candidates) == 1, "expected one uniquely delimited corner template"
    return candidates[0]


def _cross_info(component: set[tuple[int, int]]) -> dict | None:
    """Recognize a symmetric 5-cell cross or a 6-cell one-long-arm cross."""
    if len(component) not in (5, 6):
        return None
    matches = []
    for center in component:
        if not all(
            (center[0] + dr, center[1] + dc) in component
            for dr, dc in DIRECTIONS
        ):
            continue
        lengths = []
        rebuilt = {center}
        for dr, dc in DIRECTIONS:
            length = 0
            for distance in (1, 2, 3):
                cell = (center[0] + distance * dr, center[1] + distance * dc)
                if cell not in component:
                    break
                length = distance
                rebuilt.add(cell)
            lengths.append(length)
        if rebuilt == component and all(length in (1, 2) for length in lengths):
            matches.append((center, lengths))
    if len(matches) != 1:
        return None
    center, lengths = matches[0]
    long_indices = [i for i, length in enumerate(lengths) if length == 2]
    if not long_indices:
        return {"center": center, "direction": None, "symmetric": True}
    if len(long_indices) == 1 and lengths.count(1) == 3:
        return {
            "center": center,
            "direction": DIRECTIONS[long_indices[0]],
            "symmetric": False,
        }
    return None


def _main_components(grid: np.ndarray, protected: set[tuple[int, int]]) -> list[dict]:
    """Extract all monochromatic objects outside the template compartment."""
    work = np.array(grid, dtype=int, copy=True)
    for r, c in protected:
        work[r, c] = BG
    objects = []
    for color in (int(c) for c in np.unique(work) if int(c) != BG):
        for component in _components_of_color(work, color):
            info = _cross_info(component)
            assert info is not None, "every main-area object must be a valid cross"
            info.update({"color": color, "coords": component})
            objects.append(info)
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """Replace each long-arm cross with a directionally offset, recolored template stamp.

    Input structure:
    - A black grid has one odd-square monochrome template in a corner compartment
      bounded by an L-shaped separator, plus isolated monochrome crosses in the main area.
    - A directed cross has three arms of length 1 and one arm of length 2; a symmetric
      cross has four arms of length 1.

    Rule:
    - For every directed cross, erase all of its cells and translate the corner template
      mask without rotation so its center is one template side length from the cross
      center along the long arm; recolor that stamp with the cross's own color.
    - Preserve symmetric crosses, the separator, and the corner template exactly.

    Key insight:
    - Each asymmetric cross is a three-part instruction: its center is the reference,
      its unique long arm supplies direction, and its color supplies stamp color, while
      the separate corner object supplies the stamp shape; symmetric crosses are decoys.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    assert max(g.shape) <= MAX_SIZE, "ARC grids cannot exceed 30 x 30"
    template = _find_template(g)
    side = int(template["side"])
    radius = side // 2
    mask = np.asarray(template["mask"], dtype=bool)
    objects = _main_components(g, template["protected"])
    assert objects, "expected at least one main-area cross"

    out = g.copy()
    directed = [obj for obj in objects if not obj["symmetric"]]
    assert directed, "expected at least one long-arm instruction cross"
    for obj in directed:
        for r, c in obj["coords"]:
            out[r, c] = BG

    occupied_stamp_cells: set[tuple[int, int]] = set()
    for obj in directed:
        cr, cc = obj["center"]
        dr, dc = obj["direction"]
        stamp_center = (cr + side * dr, cc + side * dc)
        top = stamp_center[0] - radius
        left = stamp_center[1] - radius
        assert 0 <= top and 0 <= left
        assert top + side <= g.shape[0] and left + side <= g.shape[1]
        stamp_cells = {
            (top + r, left + c) for r, c in map(tuple, np.argwhere(mask))
        }
        assert not (stamp_cells & template["protected"]), "stamp must stay in the main area"
        assert not (stamp_cells & occupied_stamp_cells), "stamps must not overlap"
        occupied_stamp_cells |= stamp_cells
        for r, c in stamp_cells:
            out[r, c] = obj["color"]
    return out


def _make_template(side: int) -> np.ndarray | None:
    """Create a connected, tight-bbox, visibly asymmetric template mask."""
    center = side // 2
    mask = np.zeros((side, side), dtype=bool)
    mask[center, center] = True
    targets = [
        (0, random.randrange(side)),
        (side - 1, random.randrange(side)),
        (random.randrange(side), 0),
        (random.randrange(side), side - 1),
    ]
    for tr, tc in targets:
        r, c = center, center
        while (r, c) != (tr, tc):
            choices = []
            if r != tr:
                choices.append((r + (1 if tr > r else -1), c))
            if c != tc:
                choices.append((r, c + (1 if tc > c else -1)))
            r, c = random.choice(choices)
            mask[r, c] = True

    maximum = int(0.62 * side * side)
    if int(mask.sum()) > maximum:
        return None
    target_area = random.randint(max(int(mask.sum()), int(0.34 * side * side)), maximum)
    while int(mask.sum()) < target_area:
        cells = list(map(tuple, np.argwhere(mask)))
        r, c = random.choice(cells)
        dr, dc = random.choice(DIRECTIONS)
        nr, nc = r + dr, c + dc
        if 0 <= nr < side and 0 <= nc < side:
            mask[nr, nc] = True

    transforms = [np.rot90(mask, k) for k in (1, 2, 3)] + [np.fliplr(mask), np.flipud(mask)]
    if any(np.array_equal(mask, transformed) for transformed in transforms):
        return None
    if int(mask.sum()) == side * side:
        return None
    return mask


def _cross_cells(
    center: tuple[int, int], direction: tuple[int, int] | None
) -> set[tuple[int, int]]:
    cells = {center}
    for dr, dc in DIRECTIONS:
        cells.add((center[0] + dr, center[1] + dc))
    if direction is not None:
        cells.add((center[0] + 2 * direction[0], center[1] + 2 * direction[1]))
    return cells


def _too_close(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    """Whether two visible/reserved regions touch even diagonally."""
    return any(abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1 for r1, c1 in a for r2, c2 in b)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    h = random.randint(24, 30)
    w = random.randint(24, 30)
    side = random.choice((5, 7))
    k = side + 3
    corner = random.choice(CORNERS)
    template_mask = _make_template(side)
    if template_mask is None:
        return None

    separator_color, template_color, color_a, color_b = random.sample(range(1, 10), 4)
    grid = np.zeros((h, w), dtype=int)
    protected = {
        _map_from_corner(corner, grid.shape, r, c)
        for r in range(k)
        for c in range(k)
    }
    separator = {
        _map_from_corner(corner, grid.shape, k - 1, i) for i in range(k)
    } | {
        _map_from_corner(corner, grid.shape, i, k - 1) for i in range(k)
    }
    for cell in separator:
        grid[cell] = separator_color
    pattern_cells = set()
    for r, c in map(tuple, np.argwhere(template_mask)):
        cell = _map_from_corner(corner, grid.shape, r + 1, c + 1)
        grid[cell] = template_color
        pattern_cells.add(cell)
    pr0, pc0, pr1, pc1 = _bbox(pattern_cells)
    # This is the template exactly as it is visibly oriented in global grid coordinates.
    visible_template_mask = grid[pr0 : pr1 + 1, pc0 : pc1 + 1] != BG

    n_directed = random.choice((2, 3))
    n_symmetric = random.choice((1, 2))
    vertical = random.choice((DIRECTIONS[0], DIRECTIONS[2]))
    horizontal = random.choice((DIRECTIONS[1], DIRECTIONS[3]))
    directed_directions = [vertical, horizontal]
    if n_directed == 3:
        directed_directions.append(random.choice(DIRECTIONS))
    random.shuffle(directed_directions)
    directed_colors = [color_a, color_b]
    if n_directed == 3:
        directed_colors.append(random.choice((color_a, color_b)))
    random.shuffle(directed_colors)

    reserved = set(protected)
    directed_meta = []
    mask_offsets = [
        (r - side // 2, c - side // 2)
        for r, c in map(tuple, np.argwhere(visible_template_mask))
    ]
    radius = side // 2

    for direction, color in zip(directed_directions, directed_colors):
        placed = False
        for _ in range(2500):
            center = (random.randint(2, h - 3), random.randint(2, w - 3))
            cross = _cross_cells(center, direction)
            sr, sc = center[0] + side * direction[0], center[1] + side * direction[1]
            if not (radius <= sr < h - radius and radius <= sc < w - radius):
                continue
            stamp_box = {
                (r, c)
                for r in range(sr - radius, sr + radius + 1)
                for c in range(sc - radius, sc + radius + 1)
            }
            stamp = {(sr + dr, sc + dc) for dr, dc in mask_offsets}
            visual_region = cross | stamp_box
            if cross & stamp or _too_close(visual_region, reserved):
                continue
            for cell in cross:
                grid[cell] = color
            reserved |= visual_region
            directed_meta.append(
                {
                    "center": center,
                    "direction": direction,
                    "color": color,
                    "cross": cross,
                    "stamp": stamp,
                    "stamp_center": (sr, sc),
                }
            )
            placed = True
            break
        if not placed:
            return None

    symmetric_meta = []
    for _ in range(n_symmetric):
        placed = False
        for _ in range(2500):
            center = (random.randint(1, h - 2), random.randint(1, w - 2))
            cross = _cross_cells(center, None)
            if _too_close(cross, reserved):
                continue
            color = random.choice((color_a, color_b))
            for cell in cross:
                grid[cell] = color
            reserved |= cross
            symmetric_meta.append({"center": center, "color": color, "cross": cross})
            placed = True
            break
        if not placed:
            return None

    expected = grid.copy()
    for obj in directed_meta:
        for cell in obj["cross"]:
            expected[cell] = BG
    for obj in directed_meta:
        for cell in obj["stamp"]:
            expected[cell] = obj["color"]

    metadata = {
        "side": side,
        "corner": corner,
        "template_mask": visible_template_mask,
        "directed": directed_meta,
        "symmetric": symmetric_meta,
    }
    return grid, expected, metadata


def _semantically_valid(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    """Check the stated rule and anti-shortcut properties, not just solver closure."""
    if grid.shape != expected.shape or max(grid.shape) > MAX_SIZE:
        return False
    directed = meta["directed"]
    if len({obj["color"] for obj in directed}) < 2:
        return False
    if not any(obj["direction"][0] for obj in directed):
        return False
    if not any(obj["direction"][1] for obj in directed):
        return False
    if not meta["symmetric"]:
        return False
    directed_colors = {obj["color"] for obj in directed}
    if not all(obj["color"] in directed_colors for obj in meta["symmetric"]):
        return False
    if len({obj["stamp_center"] for obj in directed}) != len(directed):
        return False

    side = meta["side"]
    for obj in directed:
        cr, cc = obj["center"]
        dr, dc = obj["direction"]
        if obj["stamp_center"] != (cr + side * dr, cc + side * dc):
            return False
        if any(int(expected[cell]) != BG for cell in obj["cross"]):
            return False
        if any(int(expected[cell]) != obj["color"] for cell in obj["stamp"]):
            return False
    for obj in meta["symmetric"]:
        if any(int(expected[cell]) != int(grid[cell]) for cell in obj["cross"]):
            return False
    try:
        parsed_template = _find_template(grid)
        parsed_objects = _main_components(grid, parsed_template["protected"])
        actual = main(grid)
    except (AssertionError, ValueError, IndexError):
        return False
    if parsed_template["side"] != side:
        return False
    if sum(not obj["symmetric"] for obj in parsed_objects) != len(directed):
        return False
    if sum(obj["symmetric"] for obj in parsed_objects) != len(meta["symmetric"]):
        return False
    return not np.array_equal(grid, expected) and np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    """Generate a diverse, oracle-checked long-arm anchor puzzle input."""
    for _ in range(500):
        case = _gen_attempt()
        if case is None:
            continue
        grid, expected, metadata = case
        if _semantically_valid(grid, expected, metadata):
            return grid
    raise RuntimeError("failed to generate a non-degenerate long-arm anchor input")
