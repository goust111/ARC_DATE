import random
from collections import defaultdict

import numpy as np


ORANGE = 7
GREY = 5
RED = 2
MAROON = 9
MARKER_COLORS = (1, 3, 4, 6, 8)

WORK_SIZE = 30
MIN_COMPONENT_GAP = 2
GEN_ATTEMPTS = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _intersection_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)
    if h < 3 or w < 3:
        return mask

    occupied = grid != bg
    inner = grid[1:-1, 1:-1] == GREY
    inner &= occupied[:-2, 1:-1]
    inner &= occupied[2:, 1:-1]
    inner &= occupied[1:-1, :-2]
    inner &= occupied[1:-1, 2:]
    mask[1:-1, 1:-1] = inner
    return mask


def _marker_direction(grid: np.ndarray, bg: int, r: int, c: int) -> tuple[int, int]:
    h, w = grid.shape
    neighbors: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != bg:
            neighbors.append((dr, dc))
    assert len(neighbors) == 1, "each marker must sit at exactly one line endpoint"
    return neighbors[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather colored endpoint markers to the nearest grey-line intersections.

    Grey lines are transmission channels only. Markers slide along their unique
    line toward the first intersection they encounter, then intersections are
    recolored from the number of incoming markers:
    1 -> keep the marker color, 2 -> red, 3+ -> maroon.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _bg_color(g)

    intersections = _intersection_mask(g, bg)
    incoming: defaultdict[tuple[int, int], list[int]] = defaultdict(list)

    marker_coords = np.argwhere((g != bg) & (g != GREY))
    for r, c in marker_coords:
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        dr, dc = _marker_direction(g, bg, r, c)
        rr = r + dr
        cc = c + dc
        found = False

        while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) != bg:
            if intersections[rr, cc]:
                incoming[(rr, cc)].append(color)
                found = True
                break
            assert int(g[rr, cc]) == GREY, "markers may travel only along grey lines"
            rr += dr
            cc += dc

        assert found, "every marker must reach an intersection on its line"

    out = np.full_like(g, bg)
    for r, c in np.argwhere(intersections):
        key = (int(r), int(c))
        count = len(incoming[key])
        if count == 1:
            out[key] = incoming[key][0]
        elif count == 2:
            out[key] = RED
        elif count >= 3:
            out[key] = MAROON

    return out


def _component_kind() -> list[tuple[int, int]]:
    rich = (random.randint(3, 4), random.randint(3, 4))
    high = random.choice(((1, 1), (1, 2), (2, 1), (1, 3), (3, 1)))
    extras = ((2, 2), (2, 3), (3, 2), (1, 4), (4, 1), (2, 4), (4, 2))
    kinds = [rich, high]
    if random.random() < 0.65:
        kinds.append(random.choice(extras))
    return kinds


def _make_component(rows_count: int, cols_count: int) -> dict:
    top_arm = random.randint(1, 3)
    bottom_arm = random.randint(1, 3)
    left_arm = random.randint(1, 3)
    right_arm = random.randint(1, 3)
    row_gap = random.randint(2, 4) if rows_count > 1 else 0
    col_gap = random.randint(2, 4) if cols_count > 1 else 0

    rows = [top_arm + i * row_gap for i in range(rows_count)]
    cols = [left_arm + j * col_gap for j in range(cols_count)]
    height = rows[-1] + bottom_arm + 1
    width = cols[-1] + right_arm + 1

    grid = np.full((height, width), ORANGE, dtype=int)
    for r in rows:
        grid[r, 1:width - 1] = GREY
    for c in cols:
        grid[1:height - 1, c] = GREY

    left_colors = [random.choice(MARKER_COLORS) for _ in rows]
    right_colors = [random.choice(MARKER_COLORS) for _ in rows]
    top_colors = [random.choice(MARKER_COLORS) for _ in cols]
    bottom_colors = [random.choice(MARKER_COLORS) for _ in cols]

    for i, r in enumerate(rows):
        grid[r, 0] = left_colors[i]
        grid[r, width - 1] = right_colors[i]
    for j, c in enumerate(cols):
        grid[0, c] = top_colors[j]
        grid[height - 1, c] = bottom_colors[j]

    return {
        "grid": grid,
        "height": height,
        "width": width,
        "rows": rows,
        "cols": cols,
        "left_colors": left_colors,
        "right_colors": right_colors,
        "top_colors": top_colors,
        "bottom_colors": bottom_colors,
    }


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_components(components: list[dict]) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []

    order = sorted(range(len(components)), key=lambda i: components[i]["height"] * components[i]["width"], reverse=True)
    placed = [None] * len(components)

    for idx in order:
        comp = components[idx]
        h = int(comp["height"])
        w = int(comp["width"])
        options = [
            (r0, c0)
            for r0 in range(WORK_SIZE - h + 1)
            for c0 in range(WORK_SIZE - w + 1)
        ]
        random.shuffle(options)

        chosen = None
        for r0, c0 in options:
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_boxes_overlap(box, other, MIN_COMPONENT_GAP) for other in boxes):
                continue
            chosen = (r0, c0)
            boxes.append(box)
            placed[idx] = chosen
            break
        if chosen is None:
            return None

    return [(int(r0), int(c0)) for r0, c0 in placed]


def _paint_expected(out: np.ndarray, comp: dict, top: int, left: int) -> None:
    rows = comp["rows"]
    cols = comp["cols"]
    left_colors = comp["left_colors"]
    right_colors = comp["right_colors"]
    top_colors = comp["top_colors"]
    bottom_colors = comp["bottom_colors"]

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            incoming = []
            if j == 0:
                incoming.append(left_colors[i])
            if j == len(cols) - 1:
                incoming.append(right_colors[i])
            if i == 0:
                incoming.append(top_colors[j])
            if i == len(rows) - 1:
                incoming.append(bottom_colors[j])

            rr = top + r
            cc = left + c
            if len(incoming) == 1:
                out[rr, cc] = incoming[0]
            elif len(incoming) == 2:
                out[rr, cc] = RED
            elif len(incoming) >= 3:
                out[rr, cc] = MAROON


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != ORANGE)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())

    top_extra = random.randint(0, min(3, r0))
    left_extra = random.randint(0, min(3, c0))
    bottom_extra = random.randint(0, min(3, WORK_SIZE - 1 - r1))
    right_extra = random.randint(0, min(3, WORK_SIZE - 1 - c1))

    rr0 = r0 - top_extra
    cc0 = c0 - left_extra
    rr1 = r1 + bottom_extra
    cc1 = c1 + right_extra
    return inp[rr0:rr1 + 1, cc0:cc1 + 1], out[rr0:rr1 + 1, cc0:cc1 + 1]


def _valid_branch_coverage(expected: np.ndarray) -> bool:
    colors = set(map(int, expected.ravel()))
    return any(color in MARKER_COLORS for color in colors) and RED in colors and MAROON in colors


def _gen_attempt() -> np.ndarray | None:
    components = [_make_component(rows_count, cols_count) for rows_count, cols_count in _component_kind()]
    placements = _place_components(components)
    if placements is None:
        return None

    inp = np.full((WORK_SIZE, WORK_SIZE), ORANGE, dtype=int)
    expected = np.full((WORK_SIZE, WORK_SIZE), ORANGE, dtype=int)

    for comp, (r0, c0) in zip(components, placements):
        patch = comp["grid"]
        h, w = patch.shape
        inp[r0:r0 + h, c0:c0 + w] = patch
        _paint_expected(expected, comp, r0, c0)

    inp, expected = _crop_pair(inp, expected)

    if not _valid_branch_coverage(expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
