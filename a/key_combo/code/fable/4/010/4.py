from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_PALETTE = tuple(range(1, 10))
_MAX_GENERATION_ATTEMPTS = 400


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[Coord]]:
    """Return 4-connected True components using canonical (row, column) coordinates."""
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    found: list[list[Coord]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        component: list[Coord] = []
        while queue:
            r, c = queue.popleft()
            component.append((r, c))
            for dr, dc in _DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        found.append(component)
    return found


def _bbox(coords: list[Coord]) -> BBox:
    rows, cols = zip(*coords)
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    return (
        {(r0, c) for c in range(c0, c1 + 1)}
        | {(r1, c) for c in range(c0, c1 + 1)}
        | {(r, c0) for r in range(r0, r1 + 1)}
        | {(r, c1) for r in range(r0, r1 + 1)}
    )


def _parse_scene(grid: np.ndarray) -> tuple[int, list[BBox], list[dict]]:
    """Separate exact hollow rectangles from straight, two-marker rods."""
    bg = _background(grid)
    frames: list[BBox] = []
    rods: list[dict] = []

    for coords in _components(grid != bg):
        box = _bbox(coords)
        values = [int(grid[cell]) for cell in coords]
        coord_set = set(coords)
        r0, c0, r1, c1 = box

        if r1 - r0 >= 2 and c1 - c0 >= 2 and coord_set == _perimeter(box) and len(set(values)) == 1:
            frames.append(box)
            continue

        horizontal = r0 == r1 and c1 > c0
        vertical = c0 == c1 and r1 > r0
        assert horizontal ^ vertical, "each non-frame object must be one straight rod"
        ordered = (
            [(r0, c) for c in range(c0, c1 + 1)]
            if horizontal
            else [(r, c0) for r in range(r0, r1 + 1)]
        )
        assert coord_set == set(ordered), "rod must be contiguous"

        counts = Counter(int(grid[cell]) for cell in ordered)
        singleton_cells = [cell for cell in ordered if counts[int(grid[cell])] == 1]
        repeated = [color for color, count in counts.items() if count == len(ordered) - 2]
        assert len(ordered) >= 5 and len(singleton_cells) == 2 and len(repeated) == 1, (
            "rod needs one repeated line color and two singleton markers"
        )
        endpoint_cells = [cell for cell in singleton_cells if cell in (ordered[0], ordered[-1])]
        hinge_cells = [cell for cell in singleton_cells if cell not in (ordered[0], ordered[-1])]
        assert len(endpoint_cells) == len(hinge_cells) == 1, (
            "one singleton must mark an endpoint and the other an internal hinge"
        )

        endpoint = endpoint_cells[0]
        hinge = hinge_cells[0]
        endpoint_index = ordered.index(endpoint)
        hinge_index = ordered.index(hinge)
        assert endpoint_index in (0, len(ordered) - 1)
        tail_step = 1 if endpoint_index < hinge_index else -1
        tail_indices = list(range(hinge_index + tail_step, len(ordered) if tail_step > 0 else -1, tail_step))
        assert tail_indices and all(int(grid[ordered[i]]) == repeated[0] for i in tail_indices)
        fixed_indices = list(range(endpoint_index, hinge_index + tail_step, tail_step))
        assert fixed_indices[0] == endpoint_index and fixed_indices[-1] == hinge_index

        rods.append(
            {
                "horizontal": horizontal,
                "line_color": repeated[0],
                "endpoint": endpoint,
                "hinge": hinge,
                "tail": [ordered[i] for i in tail_indices],
                "fixed": [ordered[i] for i in fixed_indices],
            }
        )

    assert frames, "expected at least one hollow rectangle"
    assert rods, "expected at least one marked straight rod"
    return bg, frames, rods


def _strictly_inside(cells: list[Coord], box: BBox) -> bool:
    r0, c0, r1, c1 = box
    return all(r0 < r < r1 and c0 < c < c1 for r, c in cells)


def _fold_target(grid: np.ndarray, bg: int, frames: list[BBox], rod: dict) -> tuple[list[Coord], BBox]:
    hr, hc = rod["hinge"]
    length = len(rod["tail"])
    directions = ((-1, 0), (1, 0)) if rod["horizontal"] else ((0, -1), (0, 1))
    candidates: list[tuple[list[Coord], BBox]] = []
    for dr, dc in directions:
        cells = [(hr + step * dr, hc + step * dc) for step in range(1, length + 1)]
        if not all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] for r, c in cells):
            continue
        for frame in frames:
            if _strictly_inside(cells, frame) and all(int(grid[cell]) == bg for cell in cells):
                candidates.append((cells, frame))
    assert len(candidates) == 1, "each tail must have exactly one frame-and-direction fit"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Fold every marked straight rod by ninety degrees so its tail fits wholly inside a hollow frame.

    Input structure:
    - A dominant background contains several monochrome hollow rectangular frames and several separate
      horizontal or vertical rods. Each rod has a repeated line color, a singleton marker at one endpoint,
      and a differently colored singleton hinge strictly inside the rod.

    Rule:
    - Keep the endpoint-to-hinge segment, both markers, and every frame unchanged. Erase the line-colored
      tail beyond the hinge, rotate that tail ninety degrees about the hinge, and choose the one perpendicular
      direction for which every rotated tail cell lies strictly inside one frame without touching its border.

    Key insight:
    - The locally available space is a distraction: the complete tail, measured from its own hinge in grid
      row/column coordinates, must be contained by exactly one hollow rectangle interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size, "expected a nonempty 2D grid"
    bg, frames, rods = _parse_scene(g)

    plans: list[tuple[dict, list[Coord]]] = []
    claimed: set[Coord] = set()
    for rod in rods:
        target, _frame = _fold_target(g, bg, frames, rod)
        assert not (claimed & set(target)), "folded tails must not overlap"
        claimed.update(target)
        plans.append((rod, target))

    out = g.copy()
    for rod, _target in plans:
        for cell in rod["tail"]:
            out[cell] = bg
    for rod, target in plans:
        for cell in target:
            assert int(out[cell]) == bg, "folded tail must land on empty frame interior"
            out[cell] = rod["line_color"]
    return out


def _boxes_separated(a: BBox, b: BBox, gap: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_frames(height: int, width: int, sizes: list[tuple[int, int]]) -> list[BBox] | None:
    boxes: list[BBox] = []
    for frame_h, frame_w in sorted(sizes, reverse=True):
        legal: list[BBox] = []
        for r0 in range(1, height - frame_h):
            for c0 in range(1, width - frame_w):
                box = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
                if all(_boxes_separated(box, old) for old in boxes):
                    legal.append(box)
        if not legal:
            return None
        boxes.append(random.choice(legal))
    random.shuffle(boxes)
    return boxes


def _rod_for_frame(
    box: BBox,
    horizontal: bool,
    fold_sign: int,
    endpoint_sign: int,
    line_color: int,
    endpoint_color: int,
    hinge_color: int,
) -> dict | None:
    """Construct a rod whose known fold sign is the sole strict-interior fit."""
    r0, c0, r1, c1 = box
    tail_len = random.choice((2, 2, 3))
    fixed_len = random.choice((2, 2, 3))

    if horizontal:
        hinge_r = r0 + 2 if fold_sign > 0 else r1 - 2
        axis_min, axis_max = c0 + 2, c1 - 2
        fold_direction = (fold_sign, 0)
    else:
        hinge_c = c0 + 2 if fold_sign > 0 else c1 - 2
        axis_min, axis_max = r0 + 2, r1 - 2
        fold_direction = (0, fold_sign)

    # The endpoint and tail occupy opposite sides of the hinge along the original rod axis.
    tail_axis_sign = -endpoint_sign
    possible_hinges = [
        value
        for value in range(axis_min, axis_max + 1)
        if axis_min <= value + endpoint_sign * fixed_len <= axis_max
        and axis_min <= value + tail_axis_sign * tail_len <= axis_max
    ]
    if not possible_hinges:
        return None
    axis_hinge = random.choice(possible_hinges)
    if horizontal:
        hinge = (hinge_r, axis_hinge)
        axis_direction = (0, 1)
    else:
        hinge = (axis_hinge, hinge_c)
        axis_direction = (1, 0)

    ar, ac = axis_direction
    endpoint = (hinge[0] + endpoint_sign * fixed_len * ar, hinge[1] + endpoint_sign * fixed_len * ac)
    tail = [
        (hinge[0] + tail_axis_sign * k * ar, hinge[1] + tail_axis_sign * k * ac)
        for k in range(1, tail_len + 1)
    ]
    fixed = [
        (endpoint[0] - endpoint_sign * k * ar, endpoint[1] - endpoint_sign * k * ac)
        for k in range(fixed_len + 1)
    ]
    target = [
        (hinge[0] + fold_direction[0] * k, hinge[1] + fold_direction[1] * k)
        for k in range(1, tail_len + 1)
    ]
    opposite = [
        (hinge[0] - fold_direction[0] * k, hinge[1] - fold_direction[1] * k)
        for k in range(1, tail_len + 1)
    ]
    if not _strictly_inside(target, box) or _strictly_inside(opposite, box):
        return None
    return {
        "horizontal": horizontal,
        "fold_sign": fold_sign,
        "endpoint_sign": endpoint_sign,
        "line_color": line_color,
        "endpoint_color": endpoint_color,
        "hinge_color": hinge_color,
        "endpoint": endpoint,
        "hinge": hinge,
        "tail": tail,
        "fixed": fixed,
        "target": target,
        "frame": box,
    }


def _render_case(shape: tuple[int, int], frames: list[BBox], rods: list[dict], colors: tuple[int, int, int, int]):
    frame_color, line_color, endpoint_color, hinge_color = colors
    inp = np.zeros(shape, dtype=int)
    expected = np.zeros(shape, dtype=int)
    for frame in frames:
        for cell in _perimeter(frame):
            inp[cell] = expected[cell] = frame_color

    for rod in rods:
        for cell in rod["fixed"][1:-1] + rod["tail"]:
            inp[cell] = line_color
        inp[rod["endpoint"]] = endpoint_color
        inp[rod["hinge"]] = hinge_color

        for cell in rod["fixed"][1:-1]:
            expected[cell] = line_color
        expected[rod["endpoint"]] = endpoint_color
        expected[rod["hinge"]] = hinge_color
        for cell in rod["target"]:
            expected[cell] = line_color
    return inp, expected


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, frames: list[BBox], rods: list[dict]) -> bool:
    """Check the stated geometry directly, independently of the solver's scene parser."""
    if np.array_equal(inp, expected):
        return False
    if {rod["horizontal"] for rod in rods} != {False, True}:
        return False
    if {rod["fold_sign"] for rod in rods} != {-1, 1}:
        return False

    rotation_senses: set[int] = set()
    edge_space_comparisons: set[int] = set()

    occupied_targets: set[Coord] = set()
    for rod in rods:
        # Endpoint-to-hinge cells and both visible markers are exactly preserved.
        if any(int(inp[cell]) != int(expected[cell]) for cell in rod["fixed"]):
            return False
        if any(int(expected[cell]) != 0 for cell in rod["tail"]):
            return False
        if any(int(inp[cell]) != 0 or int(expected[cell]) != rod["line_color"] for cell in rod["target"]):
            return False

        hr, hc = rod["hinge"]
        tr, tc = rod["target"][0]
        fdr, fdc = tr - hr, tc - hc
        tail_r, tail_c = rod["tail"][0]
        tdr, tdc = tail_r - hr, tail_c - hc
        cross = tdr * fdc - tdc * fdr
        rotation_senses.add(1 if cross > 0 else -1)

        if fdr:
            chosen_space = inp.shape[0] - 1 - hr if fdr > 0 else hr
            opposite_space = hr if fdr > 0 else inp.shape[0] - 1 - hr
        else:
            chosen_space = inp.shape[1] - 1 - hc if fdc > 0 else hc
            opposite_space = hc if fdc > 0 else inp.shape[1] - 1 - hc
        if chosen_space != opposite_space:
            edge_space_comparisons.add(1 if chosen_space > opposite_space else -1)

        directions = ((-1, 0), (1, 0)) if rod["horizontal"] else ((0, -1), (0, 1))
        fits = []
        for dr, dc in directions:
            candidate = [(hr + k * dr, hc + k * dc) for k in range(1, len(rod["tail"]) + 1)]
            for frame in frames:
                if _strictly_inside(candidate, frame):
                    fits.append((candidate, frame))
        if len(fits) != 1 or fits[0][0] != rod["target"] or fits[0][1] != rod["frame"]:
            return False
        if occupied_targets & set(rod["target"]):
            return False
        occupied_targets.update(rod["target"])

    # Each example includes both rotation senses, and includes a rod folding toward the globally
    # narrower canvas side as a counterexample to the "more open side" shortcut.
    if rotation_senses != {-1, 1} or edge_space_comparisons != {-1, 1}:
        return False

    # All frame pixels are unchanged, including their corners and narrow sides.
    return all(int(inp[cell]) == int(expected[cell]) for frame in frames for cell in _perimeter(frame))


def _gen_attempt() -> np.ndarray | None:
    height, width = random.randint(23, 30), random.randint(23, 30)
    rod_count = random.choice((2, 2, 3))
    sizes = [(random.randint(9, 11), random.randint(9, 11)) for _ in range(rod_count)]
    frames = _place_frames(height, width, sizes)
    if frames is None:
        return None

    frame_color, line_color, endpoint_color, hinge_color = random.sample(_PALETTE, 4)
    orientations = [True, False] + [bool(random.getrandbits(1)) for _ in range(rod_count - 2)]
    fold_signs = [1, -1] + [random.choice((-1, 1)) for _ in range(rod_count - 2)]
    rotation_senses = [1, -1] + [random.choice((-1, 1)) for _ in range(rod_count - 2)]
    random.shuffle(orientations)
    random.shuffle(fold_signs)
    random.shuffle(rotation_senses)

    rods: list[dict] = []
    for box, horizontal, fold_sign, rotation_sense in zip(frames, orientations, fold_signs, rotation_senses):
        endpoint_sign = rotation_sense * fold_sign if horizontal else -rotation_sense * fold_sign
        rod = _rod_for_frame(
            box,
            horizontal,
            fold_sign,
            endpoint_sign,
            line_color,
            endpoint_color,
            hinge_color,
        )
        if rod is None:
            return None
        rods.append(rod)

    inp, expected = _render_case(
        (height, width), frames, rods, (frame_color, line_color, endpoint_color, hinge_color)
    )
    if not _semantic_checks(inp, expected, frames, rods):
        return None
    try:
        actual = main(inp)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked multi-frame folding instance."""
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-into-frame instance")
