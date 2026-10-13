import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GRAY = 5

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GEN_RETRIES = 500
PLACE_RETRIES = 250


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_rectangles(grid: np.ndarray, color: int) -> list[dict]:
    rectangles: list[dict] = []
    for comp in _components(grid == color):
        r0, c0, r1, c1 = _bbox(comp)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if len(comp) != area:
            continue
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            continue
        rectangles.append(
            {
                "bbox": (r0, c0, r1, c1),
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
            }
        )
    return rectangles


def _infer_rectangle_color(grid: np.ndarray) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) != BLACK]
    assert len(colors) == 2, "expected exactly two foreground colors"

    best_color = None
    best_area = -1
    for color in colors:
        total_rect_area = sum(
            (rect["height"] * rect["width"]) for rect in _find_rectangles(grid, color)
        )
        if total_rect_area > best_area:
            best_color = color
            best_area = total_rect_area

    assert best_color is not None and best_area > 0, "expected at least one solid rectangle"
    return int(best_color)


def _parse_arrow(component: list[tuple[int, int]]) -> dict | None:
    cells = set(component)
    degrees: dict[tuple[int, int], int] = {}
    for r, c in cells:
        degrees[(r, c)] = sum((r + dr, c + dc) in cells for dr, dc in DIR4)

    tips = [cell for cell, degree in degrees.items() if degree == 3]
    if len(tips) != 1:
        return None

    tip = tips[0]
    neighbors: list[tuple[int, int]] = []
    vectors: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        nr, nc = tip[0] + dr, tip[1] + dc
        if (nr, nc) in cells:
            neighbors.append((nr, nc))
            vectors.append((dr, dc))

    wing_pair: set[tuple[int, int]] | None = None
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if vectors[i][0] == -vectors[j][0] and vectors[i][1] == -vectors[j][1]:
                wing_pair = {vectors[i], vectors[j]}
                break
        if wing_pair is not None:
            break
    if wing_pair is None:
        return None

    shaft_vec = None
    for vec in vectors:
        if vec not in wing_pair:
            shaft_vec = vec
            break
    if shaft_vec is None:
        return None

    shaft_cells: list[tuple[int, int]] = []
    cur = (tip[0] + shaft_vec[0], tip[1] + shaft_vec[1])
    while cur in cells:
        shaft_cells.append(cur)
        cur = (cur[0] + shaft_vec[0], cur[1] + shaft_vec[1])

    if not (1 <= len(shaft_cells) <= 4):
        return None

    expected = {tip}
    for cell in shaft_cells:
        expected.add(cell)
    for wr, wc in wing_pair:
        expected.add((tip[0] + wr, tip[1] + wc))
    if expected != cells:
        return None

    toward_rect = (-shaft_vec[0], -shaft_vec[1])
    return {"tip": tip, "toward": toward_rect, "layers": len(shaft_cells)}


def _direction_to_side(toward: tuple[int, int]) -> str:
    mapping = {
        (1, 0): "top",
        (-1, 0): "bottom",
        (0, 1): "left",
        (0, -1): "right",
    }
    return mapping[toward]


def _paint_layers(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
    layers: int,
    fill_color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if side in ("top", "bottom"):
        depth = height
    else:
        depth = width

    actual = min(layers, (depth + 1) // 2)
    for layer in range(actual):
        if side == "top":
            out[r0 + layer, c0 : c1 + 1] = fill_color
        elif side == "bottom":
            out[r1 - layer, c0 : c1 + 1] = fill_color
        elif side == "left":
            out[r0 : r1 + 1, c0 + layer] = fill_color
        else:
            out[r0 : r1 + 1, c1 - layer] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each red arrow to choose a gray rectangle side, then repaint that many outer layers from the pointed side inward.

    Input structure:
    - The background is black.
    - One foreground color forms solid gray axis-aligned rectangles of varying sizes.
    - The other foreground color forms thin orthogonal arrows: a straight shaft of length 1 to 4 ending in a three-pixel arrowhead whose tip points at a rectangle side.

    Rule:
    - Identify every solid gray rectangle and every red arrow.
    - The arrow's pointing direction selects which side of the touched rectangle is active.
    - The shaft length N determines how many outer layers of that rectangle are repainted red from the pointed side inward, capped at the rectangle center.
    - Remove the arrow itself to black and leave rectangles without arrows unchanged.

    Key insight:
    - Each arrow encodes two independent variables at once: direction chooses the rectangle side, while shaft length chooses the number of onion-like layers to repaint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rect_color = _infer_rectangle_color(g)
    arrow_color = RED if rect_color != RED else GRAY

    rectangles = _find_rectangles(g, rect_color)
    assert rectangles, "expected at least one solid rectangle"

    rect_by_cell: dict[tuple[int, int], int] = {}
    for idx, rect in enumerate(rectangles):
        r0, c0, r1, c1 = rect["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                rect_by_cell[(r, c)] = idx

    out = g.copy()
    out[g == arrow_color] = BLACK

    for comp in _components(g == arrow_color):
        arrow = _parse_arrow(comp)
        assert arrow is not None, "expected every non-rectangle foreground component to be an arrow"
        tip_r, tip_c = arrow["tip"]
        dr, dc = arrow["toward"]
        target = (tip_r + dr, tip_c + dc)
        assert target in rect_by_cell, "expected arrow tip to touch a rectangle"
        side = _direction_to_side((dr, dc))
        rect = rectangles[rect_by_cell[target]]
        _paint_layers(out, rect["bbox"], side, arrow["layers"], arrow_color)

    return out


def _choose_depth(rng: random.Random, layers: int, branch: str) -> int:
    if branch == "partial":
        low, high = 2 * layers + 1, 10
    else:
        low, high = 2, min(2 * layers, 10)
    assert low <= high
    return rng.randint(low, high)


def _make_specs(rng: random.Random) -> list[dict]:
    total_rects = rng.randint(2, 5)
    plain_count = 0
    if total_rects > 1 and rng.random() < 0.5:
        plain_count = rng.randint(1, total_rects - 1)
    arrow_count = total_rects - plain_count

    directions = ["top", "bottom", "left", "right"]
    rng.shuffle(directions)
    direction_list = [
        directions[i] if i < len(directions) else rng.choice(directions)
        for i in range(arrow_count)
    ]

    lengths = [1, 2, 3, 4]
    rng.shuffle(lengths)
    length_list = [
        lengths[i] if i < len(lengths) else rng.randint(1, 4)
        for i in range(arrow_count)
    ]

    branches: list[str]
    if arrow_count == 1:
        branches = [rng.choice(["partial", "center"])]
    else:
        branches = ["partial", "center"]
        while len(branches) < arrow_count:
            branches.append(rng.choice(["partial", "center"]))
        rng.shuffle(branches)

    specs: list[dict] = []
    for side, layers, branch in zip(direction_list, length_list, branches):
        if side in ("top", "bottom"):
            height = _choose_depth(rng, layers, branch)
            width = rng.randint(3, 9)
        else:
            width = _choose_depth(rng, layers, branch)
            height = rng.randint(3, 9)
        specs.append(
            {
                "kind": "arrowed",
                "side": side,
                "layers": layers,
                "branch": branch,
                "height": height,
                "width": width,
            }
        )

    for _ in range(plain_count):
        specs.append(
            {
                "kind": "plain",
                "side": None,
                "layers": 0,
                "branch": None,
                "height": rng.randint(2, 10),
                "width": rng.randint(2, 10),
            }
        )

    specs.sort(
        key=lambda spec: (
            spec["height"] * spec["width"] + (spec["layers"] + 3 if spec["kind"] == "arrowed" else 0)
        ),
        reverse=True,
    )
    return specs


def _build_arrow_cells(
    r0: int,
    c0: int,
    height: int,
    width: int,
    side: str,
    layers: int,
    offset: int,
) -> list[tuple[int, int]]:
    if side == "top":
        tip = (r0 - 1, c0 + offset)
        shaft = [(r0 - 2 - i, c0 + offset) for i in range(layers)]
        wings = [(tip[0], tip[1] - 1), (tip[0], tip[1] + 1)]
    elif side == "bottom":
        tip = (r0 + height, c0 + offset)
        shaft = [(r0 + height + 1 + i, c0 + offset) for i in range(layers)]
        wings = [(tip[0], tip[1] - 1), (tip[0], tip[1] + 1)]
    elif side == "left":
        tip = (r0 + offset, c0 - 1)
        shaft = [(r0 + offset, c0 - 2 - i) for i in range(layers)]
        wings = [(tip[0] - 1, tip[1]), (tip[0] + 1, tip[1])]
    else:
        tip = (r0 + offset, c0 + width)
        shaft = [(r0 + offset, c0 + width + 1 + i) for i in range(layers)]
        wings = [(tip[0] - 1, tip[1]), (tip[0] + 1, tip[1])]
    return shaft + [tip] + wings


def _composite_cells(spec: dict, r0: int, c0: int, offset: int | None) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    rect_cells = [
        (r, c)
        for r in range(r0, r0 + spec["height"])
        for c in range(c0, c0 + spec["width"])
    ]
    arrow_cells: list[tuple[int, int]] = []
    if spec["kind"] == "arrowed":
        assert offset is not None
        arrow_cells = _build_arrow_cells(
            r0,
            c0,
            spec["height"],
            spec["width"],
            spec["side"],
            spec["layers"],
            offset,
        )
    return rect_cells, arrow_cells


def _fits(shape: tuple[int, int], cells: list[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _well_separated(occupied: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if occupied[r, c]:
            return False
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return False
    return True


def _candidate_ranges(shape: tuple[int, int], spec: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    height = spec["height"]
    width = spec["width"]
    side = spec["side"]
    layers = spec["layers"]

    if side == "top":
        r_min, r_max = layers + 1, h - height
        c_min, c_max = 0, w - width
    elif side == "bottom":
        r_min, r_max = 0, h - height - layers - 1
        c_min, c_max = 0, w - width
    elif side == "left":
        r_min, r_max = 0, h - height
        c_min, c_max = layers + 1, w - width
    elif side == "right":
        r_min, r_max = 0, h - height
        c_min, c_max = 0, w - width - layers - 1
    else:
        r_min, r_max = 0, h - height
        c_min, c_max = 0, w - width
    return (r_min, r_max), (c_min, c_max)


def _place_specs(rng: random.Random, shape: tuple[int, int], specs: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full(shape, BLACK, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    placed: list[dict] = []

    for spec in specs:
        (r_min, r_max), (c_min, c_max) = _candidate_ranges(shape, spec)
        if r_min > r_max or c_min > c_max:
            return None

        success = False
        for _ in range(PLACE_RETRIES):
            r0 = rng.randint(r_min, r_max)
            c0 = rng.randint(c_min, c_max)
            offset = None
            if spec["kind"] == "arrowed":
                if spec["side"] in ("top", "bottom"):
                    offset = rng.randint(1, spec["width"] - 2)
                else:
                    offset = rng.randint(1, spec["height"] - 2)

            rect_cells, arrow_cells = _composite_cells(spec, r0, c0, offset)
            all_cells = rect_cells + arrow_cells
            if not _fits(shape, all_cells):
                continue
            if not _well_separated(occupied, all_cells):
                continue

            for r, c in rect_cells:
                grid[r, c] = GRAY
                occupied[r, c] = True
            for r, c in arrow_cells:
                grid[r, c] = RED
                occupied[r, c] = True

            placed.append(
                {
                    **spec,
                    "bbox": (r0, c0, r0 + spec["height"] - 1, c0 + spec["width"] - 1),
                    "arrow_cells": arrow_cells,
                }
            )
            success = True
            break

        if not success:
            return None

    return grid, placed


def _expected_output(grid: np.ndarray, placed: list[dict]) -> np.ndarray:
    out = grid.copy()
    for item in placed:
        for r, c in item["arrow_cells"]:
            out[r, c] = BLACK
        if item["kind"] == "arrowed":
            _paint_layers(out, item["bbox"], item["side"], item["layers"], RED)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    specs = _make_specs(rng)
    base = 12 + 2 * len(specs)
    shape = (rng.randint(base, 30), rng.randint(base, 30))
    if min(shape) < 2:
        return None

    placed = _place_specs(rng, shape, specs)
    if placed is None:
        return None
    grid, placements = placed

    expected = _expected_output(grid, placements)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
