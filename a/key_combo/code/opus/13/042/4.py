import random
from typing import Iterable

import numpy as np


BG = 0
GROUND = 5
OBJECT_COLORS = (1, 2, 3)
SIDES = ("top", "left", "right")
PATTERNS = [
    ("top", "left"),
    ("top", "right"),
    ("top", "left", "right"),
    ("top", "top", "left"),
    ("top", "top", "right"),
]
MAX_GENERATION_ATTEMPTS = 500


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _edge_of_rect(rect: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    h, w = shape
    r0, c0, r1, c1 = rect
    touched = []
    if r0 == 0:
        touched.append("top")
    if c0 == 0:
        touched.append("left")
    if c1 == w - 1:
        touched.append("right")
    if r1 == h - 1:
        touched.append("bottom")
    assert touched in (["top"], ["left"], ["right"]), "object must touch exactly one allowed edge"
    return touched[0]


def _destination(rect: tuple[int, int, int, int], side: str, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = rect
    obj_h = r1 - r0 + 1
    obj_w = c1 - c0 + 1
    if side == "top":
        return h - 1 - obj_h, c0, h - 2, c1
    if side == "left":
        return r0, w - obj_w, r1, w - 1
    assert side == "right"
    return r0, 0, r1, obj_w - 1


def _travel_distance(rect: tuple[int, int, int, int], side: str, shape: tuple[int, int]) -> int:
    dst = _destination(rect, side, shape)
    if side == "top":
        return dst[0] - rect[0]
    return abs(dst[1] - rect[1])


def _trail_cells(
    rect: tuple[int, int, int, int], side: str, shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = rect
    obj_h = r1 - r0 + 1
    obj_w = c1 - c0 + 1
    dst = _destination(rect, side, shape)
    cells: list[tuple[int, int]] = []

    if side == "top":
        for step, r in enumerate(range(r1 + 1, dst[0])):
            start_c = c0 + step
            for c in range(start_c, min(start_c + obj_w, w)):
                cells.append((r, c))
        return cells

    limit_row = h - 1  # keep the grey floor row untouched
    if side == "left":
        for step, c in enumerate(range(c1 + 1, dst[1])):
            start_r = r0 + step
            for r in range(start_r, min(start_r + obj_h, limit_row)):
                cells.append((r, c))
        return cells

    assert side == "right"
    for step, c in enumerate(range(c0 - 1, dst[3], -1)):
        start_r = r0 + step
        for r in range(start_r, min(start_r + obj_h, limit_row)):
            cells.append((r, c))
    return cells


def _build_object(
    side: str, color: int, obj_h: int, obj_w: int, anchor: int, shape: tuple[int, int]
) -> dict:
    h, w = shape
    if side == "top":
        rect = (0, anchor, obj_h - 1, anchor + obj_w - 1)
    elif side == "left":
        rect = (anchor, 0, anchor + obj_h - 1, obj_w - 1)
    else:
        rect = (anchor, w - obj_w, anchor + obj_h - 1, w - 1)

    dst = _destination(rect, side, shape)
    trail = _trail_cells(rect, side, shape)
    trail_steps = 0
    if side == "top":
        trail_steps = max(0, dst[0] - rect[2] - 1)
    elif side == "left":
        trail_steps = max(0, dst[1] - rect[3] - 1)
    else:
        trail_steps = max(0, rect[1] - dst[3] - 1)
    unclipped = trail_steps * (obj_w if side == "top" else obj_h)
    return {
        "side": side,
        "color": int(color),
        "src": rect,
        "dst": dst,
        "trail": tuple(trail),
        "distance": _travel_distance(rect, side, shape),
        "clipped": len(trail) < unclipped,
        "output_cells": _rect_cells(dst) | set(trail),
        "input_cells": _rect_cells(rect),
    }


def _render_from_objects(objects: Iterable[dict], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    out[h - 1, :] = GROUND

    best_color: dict[tuple[int, int], int] = {}
    best_priority: dict[tuple[int, int], tuple[int, int]] = {}
    for obj in objects:
        priority = (int(obj["distance"]), -int(obj["color"]))
        for cell in obj["trail"]:
            prev = best_priority.get(cell)
            if prev is None or priority > prev:
                best_priority[cell] = priority
                best_color[cell] = int(obj["color"])

    for (r, c), color in best_color.items():
        out[r, c] = color

    occupied_dsts: set[tuple[int, int]] = set()
    for obj in objects:
        dst_cells = _rect_cells(obj["dst"])
        assert occupied_dsts.isdisjoint(dst_cells), "destination rectangles must be disjoint"
        occupied_dsts |= dst_cells
        for r, c in dst_cells:
            out[r, c] = int(obj["color"])

    out[h - 1, :] = GROUND
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide edge-touching colored rectangles to the opposite side and leave a sheared trail behind them.

    Input structure:
    - The grid has a black background and a full grey ground row on the bottom edge.
    - Two or three solid colored rectangles of distinct colors touch exactly one allowed outer edge: top, left, or right.
    - Each rectangle is 2x2, 2x3, 3x2, or 3x3, and no rectangle touches the grey ground row.

    Rule:
    - A top-touching rectangle slides downward until it rests immediately above the grey ground row; a left-touching rectangle slides to the right edge; a right-touching rectangle slides to the left edge.
    - Clear the source rectangle, redraw it at its destination, and fill the swept region with a same-color sheared trail: vertical slides create one width-sized segment per intermediate row, offset one cell further right each row; horizontal slides create one height-sized segment per intermediate column, offset one cell further down each column.
    - Clip trail pixels that would go past the right edge or below the row just above the grey ground, resolve trail overlaps by longer slide distance then smaller color id, and never alter the grey ground row.

    Key insight:
    - The touched edge sets the slide direction, while the rectangle's intermediate row or column index sets the perpendicular one-cell shear that turns a straight sweep into a parallelogram trail.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 7 and w >= 7
    assert np.all(g[h - 1] == GROUND), "bottom row must be the grey ground"

    colors = [int(v) for v in np.unique(g) if int(v) not in (BG, GROUND)]
    assert 2 <= len(colors) <= 3
    assert len(colors) == len(set(colors))

    objects: list[dict] = []
    seen_sides = []
    for color in sorted(colors):
        coords = np.argwhere(g == color)
        assert coords.size > 0
        rect = _bbox(coords)
        r0, c0, r1, c1 = rect
        assert (r1 - r0 + 1) in (2, 3) and (c1 - c0 + 1) in (2, 3)
        assert coords.shape[0] == (r1 - r0 + 1) * (c1 - c0 + 1), "object must be a solid rectangle"
        side = _edge_of_rect(rect, g.shape)
        seen_sides.append(side)
        assert r1 <= h - 5, "object must stay at least three rows above the ground"
        objects.append(
            {
                "side": side,
                "color": color,
                "src": rect,
                "dst": _destination(rect, side, g.shape),
                "trail": tuple(_trail_cells(rect, side, g.shape)),
                "distance": _travel_distance(rect, side, g.shape),
            }
        )

    assert "top" in seen_sides
    assert any(side in ("left", "right") for side in seen_sides)
    return _render_from_objects(objects, g.shape)


def _candidate_anchors(side: str, obj_h: int, obj_w: int, shape: tuple[int, int], require_clip: bool) -> list[int]:
    h, w = shape
    if side == "top":
        values = list(range(1, w - obj_w))
    else:
        values = list(range(1, h - obj_h - 3))
    random.shuffle(values)

    ranked = []
    for anchor in values:
        obj = _build_object(side, 1, obj_h, obj_w, anchor, shape)
        if require_clip and not obj["clipped"]:
            continue
        score = 0
        if side == "top":
            score = anchor
        else:
            score = anchor
        ranked.append((score, anchor))
    ranked.sort(reverse=bool(require_clip))
    return [anchor for _, anchor in ranked]


def _partial_compatible(objects: list[dict], shape: tuple[int, int]) -> bool:
    used_input: set[tuple[int, int]] = set()
    used_output: set[tuple[int, int]] = set()
    for obj in objects:
        r0, c0, r1, c1 = obj["src"]
        if not ((r1 - r0 + 1) in (2, 3) and (c1 - c0 + 1) in (2, 3)):
            return False
        if _edge_of_rect(obj["src"], shape) != obj["side"]:
            return False
        if r1 > shape[0] - 5:
            return False
        if not used_input.isdisjoint(obj["input_cells"]):
            return False
        if not used_output.isdisjoint(obj["output_cells"]):
            return False
        used_input |= obj["input_cells"]
        used_output |= obj["output_cells"]
    return True


def _objects_are_valid(objects: list[dict], shape: tuple[int, int]) -> bool:
    if not (2 <= len(objects) <= 3):
        return False
    if len({int(obj["color"]) for obj in objects}) != len(objects):
        return False
    sides = [obj["side"] for obj in objects]
    if "top" not in sides or not any(side in ("left", "right") for side in sides):
        return False
    if not _partial_compatible(objects, shape):
        return False

    out = _render_from_objects(objects, shape)
    non_ground = int(np.count_nonzero(out[:-1] != BG))
    return non_ground < (shape[0] * shape[1]) // 2


def _render_input(objects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    grid[h - 1, :] = GROUND
    for obj in objects:
        r0, c0, r1, c1 = obj["src"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(obj["color"])
    return grid


def _make_objects(shape: tuple[int, int], pattern: tuple[str, ...]) -> list[dict] | None:
    colors = random.sample(list(OBJECT_COLORS), len(pattern))
    specs = []
    top_indices = [i for i, side in enumerate(pattern) if side == "top"]
    side_indices = [i for i, side in enumerate(pattern) if side in ("left", "right")]
    clip_top = random.choice(top_indices) if top_indices and random.random() < 0.6 else None
    clip_side = random.choice(side_indices) if side_indices and random.random() < 0.6 else None

    for idx, side in enumerate(pattern):
        specs.append(
            {
                "side": side,
                "color": colors[idx],
                "obj_h": random.randint(2, 3),
                "obj_w": random.randint(2, 3),
                "require_clip": idx in (clip_top, clip_side),
            }
        )

    placed: list[dict] = []

    def backtrack(i: int) -> bool:
        if i == len(specs):
            return True

        spec = specs[i]
        anchors = _candidate_anchors(
            spec["side"], spec["obj_h"], spec["obj_w"], shape, bool(spec["require_clip"])
        )
        for anchor in anchors:
            obj = _build_object(spec["side"], spec["color"], spec["obj_h"], spec["obj_w"], anchor, shape)
            if not _partial_compatible(placed + [obj], shape):
                continue
            placed.append(obj)
            if backtrack(i + 1):
                return True
            placed.pop()
        return False

    return placed if backtrack(0) else None


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(13, 18), random.randint(13, 18))
    pattern = random.choice(PATTERNS)
    objects = _make_objects(shape, pattern)
    if objects is None:
        return None

    inp = _render_input(objects, shape)
    expected = _render_from_objects(objects, shape)

    # Semantic checks beyond the oracle: structural validity, non-empty trails, and branch variety.
    if not _objects_are_valid(objects, shape):
        return None
    if not any(obj["trail"] for obj in objects):
        return None
    if not np.any(expected[:-1] != inp[:-1]):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sheared-slide ARC input")
