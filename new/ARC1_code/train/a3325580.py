from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summarize the largest foreground object-size class as solid vertical color bars.

    Input structure:
    - The input has a dominant background color and several monochromatic
      4-connected foreground objects.
    - Each foreground color appears as exactly one object.
    - The largest objects have distinct left edges, giving a unique left-to-right
      ordering.

    Rule:
    - Find the maximum foreground object area.
    - Keep exactly the objects with that area.
    - Build an output whose height is that maximum area and whose columns are
      filled with the kept object colors in left-to-right input order.

    Key insight:
    - The original shapes are distractors; only object area and horizontal order
      survive, so each selected object collapses to a same-color vertical bar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    objects = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert objects, "expected at least one foreground object"

    colors = [int(obj["color"]) for obj in objects]
    assert len(colors) == len(set(colors)), "each foreground color must form one object"

    max_area = max(int(obj["area"]) for obj in objects)
    largest = [obj for obj in objects if int(obj["area"]) == max_area]

    left_edges = [int(obj["bbox"][1]) for obj in largest]
    assert len(left_edges) == len(
        set(left_edges)
    ), "largest objects need unique left-to-right order"
    largest.sort(key=lambda obj: int(obj["bbox"][1]))

    out = np.empty((max_area, len(largest)), dtype=int)
    for col, obj in enumerate(largest):
        out[:, col] = int(obj["color"])
    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
CANVAS_SHAPE = (10, 10)
BACKGROUND = 0
COLOR_MIN, COLOR_MAX = 1, 9

# Blob shape bounds.
BLOB_MIN_HEIGHT, BLOB_MAX_HEIGHT = 1, 6
BLOB_MIN_WIDTH, BLOB_MAX_WIDTH = 1, 3
FOUR_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Object counts and sizes.
COUNT_PAIRS = (
    (1, 1), (1, 2), (1, 3), (2, 0), (2, 1),
    (2, 2), (2, 3), (3, 0), (3, 1), (3, 2),
)
LARGEST_AREA_MIN = 3
LARGEST_AREA_MAX = 9
DISTRACTOR_AREA_MIN = 2

# Placement and retry policy.
PLACEMENT_PADDING = 1
GENERATION_RETRY_LIMIT = 300


def _blob_dim_choices(area: int) -> list[tuple[int, int]]:
    return [
        (h, w)
        for h in range(BLOB_MIN_HEIGHT, min(BLOB_MAX_HEIGHT, area) + 1)
        for w in range(BLOB_MIN_WIDTH, BLOB_MAX_WIDTH + 1)
        if h * w >= area
    ]


def _gen_blob(area: int, color: int) -> np.ndarray:
    area = int(area)
    color = int(color)
    h, w = random.choice(_blob_dim_choices(area))

    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in FOUR_NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))

    coords = np.array(sorted(cells), dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = color
    return patch


def _gen_locations(
    canvas: np.ndarray, patch: np.ndarray, forbidden_lefts: set[int] | None = None
) -> list[tuple[int, int]]:
    forbidden_lefts = forbidden_lefts or set()
    H, W = canvas.shape
    h, w = patch.shape
    filled = np.argwhere(patch != 0)
    locs = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if c0 in forbidden_lefts:
                continue
            ok = True
            for pr, pc in filled:
                rr, cc = r0 + int(pr), c0 + int(pc)
                r_min = max(0, rr - PLACEMENT_PADDING)
                r_max = min(H, rr + PLACEMENT_PADDING + 1)
                c_min = max(0, cc - PLACEMENT_PADDING)
                c_max = min(W, cc + PLACEMENT_PADDING + 1)
                if np.any(canvas[r_min:r_max, c_min:c_max] != BACKGROUND):
                    ok = False
                    break
            if ok:
                locs.append((r0, c0))
    return locs


def _gen_place(
    canvas: np.ndarray, patch: np.ndarray, forbidden_lefts: set[int] | None = None
) -> tuple[int, int] | None:
    locs = _gen_locations(canvas, patch, forbidden_lefts=forbidden_lefts)
    if not locs:
        return None
    r0, c0 = random.choice(locs)
    paste(canvas, patch, r0, c0, transparent=BACKGROUND)
    return r0, c0


def _make_object(area: int, color: int, largest: bool) -> dict:
    return {
        "area": area,
        "color": color,
        "largest": largest,
        "patch": _gen_blob(area, color),
    }


def _sample_objects() -> tuple[int, int, list[dict]]:
    n_largest, n_distractors = random.choice(COUNT_PAIRS)
    max_area = random.randint(LARGEST_AREA_MIN, LARGEST_AREA_MAX)
    total = n_largest + n_distractors
    colors = random.sample(range(COLOR_MIN, COLOR_MAX + 1), total)

    objects = []
    for i in range(n_largest):
        objects.append(_make_object(max_area, colors[i], True))
    for i in range(n_distractors):
        color = colors[n_largest + i]
        area = random.randint(DISTRACTOR_AREA_MIN, max_area - 1)
        objects.append(_make_object(area, color, False))
    return max_area, n_largest, objects


def _place_objects(canvas: np.ndarray, objects: list[dict]) -> bool:
    """Place patches and record each object's top-left location."""
    largest_lefts: set[int] = set()
    order = list(range(len(objects)))
    random.shuffle(order)
    order.sort(key=lambda i: (not objects[i]["largest"], -int(objects[i]["area"])))
    for i in order:
        obj = objects[i]
        loc = _gen_place(
            canvas,
            obj["patch"],
            forbidden_lefts=largest_lefts if obj["largest"] else None,
        )
        if loc is None:
            return False
        obj["loc"] = loc
        if obj["largest"]:
            largest_lefts.add(int(loc[1]))
    return True


def _expected_output(max_area: int, n_largest: int, objects: list[dict]) -> np.ndarray:
    largest = [obj for obj in objects if obj["largest"]]
    largest.sort(key=lambda obj: int(obj["loc"][1]))
    expected = np.empty((max_area, n_largest), dtype=int)
    for col, obj in enumerate(largest):
        expected[:, col] = int(obj["color"])
    return expected


def _gen_attempt() -> np.ndarray | None:
    max_area, n_largest, objects = _sample_objects()
    canvas = np.zeros(CANVAS_SHAPE, dtype=int)
    if not _place_objects(canvas, objects):
        return None

    expected = _expected_output(max_area, n_largest, objects)
    if not np.array_equal(main(canvas), expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid a3325580 input")
