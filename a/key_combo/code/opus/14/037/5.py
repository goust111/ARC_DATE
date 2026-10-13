from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
ORANGE = 7
FRAME_COLORS = (1, 2, 3, 4)

GEN_GRID_RANGE = (16, 22)
GEN_OBJECT_COUNT_CHOICES = (3, 4)
GEN_RECT_SHORT = 5
GEN_RECT_LONG_RANGE = (4, 6)
GEN_RETRIES = 500
GEN_CROSS_PROB = 0.6


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                cells.append((r, c))
    return cells


def _interior_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _connected_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BACKGROUND, ORANGE) or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        _in_bounds(grid.shape, nr, nc)
                        and not visited[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": np.array(cells, dtype=int)})
    return comps


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    objects: list[dict] = []
    seen_protrusions: set[tuple[int, int]] = set()

    for comp in _connected_components(g):
        color = int(comp["color"])
        coords = comp["coords"]
        r0, c0, r1, c1 = _bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1

        assert 4 <= h <= 6 and 4 <= w <= 6, "frame size must stay within 4..6"
        expected_frame = set(_frame_cells(r0, c0, r1, c1))
        actual_frame = {(int(r), int(c)) for r, c in coords}
        assert actual_frame == expected_frame, "foreground component must be a 1-pixel rectangular frame"
        assert all(int(g[r, c]) == BACKGROUND for r, c in _interior_cells(r0, c0, r1, c1)), "frame interior must be black"

        candidates: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
        for c in range(c0 + 1, c1):
            if r0 > 0 and int(g[r0 - 1, c]) == ORANGE:
                candidates.append(("top", (r0 - 1, c), (-1, 0)))
            if r1 + 1 < g.shape[0] and int(g[r1 + 1, c]) == ORANGE:
                candidates.append(("bottom", (r1 + 1, c), (1, 0)))
        for r in range(r0 + 1, r1):
            if c0 > 0 and int(g[r, c0 - 1]) == ORANGE:
                candidates.append(("left", (r, c0 - 1), (0, -1)))
            if c1 + 1 < g.shape[1] and int(g[r, c1 + 1]) == ORANGE:
                candidates.append(("right", (r, c1 + 1), (0, 1)))

        assert len(candidates) == 1, "each frame must have exactly one orange protrusion"
        side, protrusion, ray_dir = candidates[0]
        pr, pc = protrusion
        seen_protrusions.add((pr, pc))

        non_black_neighbors = [
            (nr, nc)
            for nr, nc in _neighbors4(pr, pc)
            if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) != BACKGROUND
        ]
        assert non_black_neighbors == [((pr - ray_dir[0]), (pc - ray_dir[1]))], "protrusion must touch only its frame"

        objects.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "protrusion": protrusion,
                "side": side,
                "ray_dir": ray_dir,
                "branch": "vertical" if side in ("top", "bottom") else "horizontal",
            }
        )

    orange_cells = {(int(r), int(c)) for r, c in np.argwhere(g == ORANGE)}
    assert orange_cells == seen_protrusions, "orange cells must be exactly the protrusions"
    assert 1 <= len(objects) <= 4
    return sorted(objects, key=lambda obj: (obj["bbox"], obj["color"]))


def _apply_rule_from_objects(grid: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)

    for obj in objects:
        pr, pc = obj["protrusion"]
        out[pr, pc] = BACKGROUND

    for obj in objects:
        if obj["branch"] != "horizontal":
            continue
        r0, c0, r1, c1 = obj["bbox"]
        for r, c in _frame_cells(r0, c0, r1, c1):
            out[r, c] = BACKGROUND
        for r, c in _interior_cells(r0, c0, r1, c1):
            out[r, c] = ORANGE

    for obj in objects:
        if obj["branch"] != "vertical":
            continue
        rr, cc = obj["protrusion"]
        dr, dc = obj["ray_dir"]
        color = int(obj["color"])
        while _in_bounds(out.shape, rr, cc) and int(out[rr, cc]) in (BACKGROUND, ORANGE):
            out[rr, cc] = color
            rr += dr
            cc += dc

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Branch on each rectangle's orange leaf direction, swapping horizontal ones into fills and vertical ones into rays.

    Input structure:
    - A black background contains several 1-pixel-thick hollow rectangular frames in distinct non-black colors.
    - Each frame has a single orange one-cell protrusion attached just outside one side of the frame.
    - A left/right protrusion marks a horizontal case, while a top/bottom protrusion marks a vertical case.

    Rule:
    - Detect each rectangular frame and its orange protrusion direction.
    - For a horizontal protrusion, delete the protrusion, turn the whole frame border black, and fill the enclosed interior with orange.
    - For a vertical protrusion, keep the frame, recolor the protrusion into the frame color, and extend a ray from that cell outward in the protrusion direction until the grid edge or the first non-black, non-orange blocker.
    - Apply all horizontal swaps before any vertical rays, so vertical rays may overwrite orange fills they pass through.

    Key insight:
    - The orange leaf is not merely a locator: its attachment side selects one of two qualitatively different transformations, and the later ray branch has priority over earlier orange fills where they overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    objects = _find_objects(g)
    return _apply_rule_from_objects(g, objects)


def _make_object(
    color: int,
    branch: str,
    side: str,
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> dict:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if side == "left":
        protrusion = (r0 + h // 2, c0 - 1)
        ray_dir = (0, -1)
    elif side == "right":
        protrusion = (r0 + h // 2, c1 + 1)
        ray_dir = (0, 1)
    elif side == "top":
        protrusion = (r0 - 1, c0 + w // 2)
        ray_dir = (-1, 0)
    else:
        protrusion = (r1 + 1, c0 + w // 2)
        ray_dir = (1, 0)

    return {
        "color": int(color),
        "branch": branch,
        "side": side,
        "bbox": (r0, c0, r1, c1),
        "protrusion": protrusion,
        "ray_dir": ray_dir,
    }


def _object_input_cells(obj: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = obj["bbox"]
    cells = set(_frame_cells(r0, c0, r1, c1))
    cells.add(tuple(obj["protrusion"]))
    return cells


def _expanded_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded = set(cells)
    for r, c in list(cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expanded.add((r + dr, c + dc))
    return expanded


def _can_place(obj: dict, blocked: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    cells = _object_input_cells(obj)
    if not all(_in_bounds(shape, r, c) for r, c in cells):
        return False
    return not (_expanded_cells(cells) & blocked)


def _add_to_blocked(blocked: set[tuple[int, int]], obj: dict) -> None:
    blocked.update(_expanded_cells(_object_input_cells(obj)))


def _draw_input(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in objects:
        color = int(obj["color"])
        r0, c0, r1, c1 = obj["bbox"]
        for r, c in _frame_cells(r0, c0, r1, c1):
            grid[r, c] = color
        pr, pc = obj["protrusion"]
        grid[pr, pc] = ORANGE
    return grid


def _expected_from_metadata(inp: np.ndarray, objects: list[dict]) -> tuple[np.ndarray, dict]:
    out = np.array(inp, copy=True)
    info = {
        "horizontal_count": 0,
        "vertical_count": 0,
        "cross_fill_overwrites": 0,
        "ray_lengths": [],
    }

    for obj in objects:
        pr, pc = obj["protrusion"]
        out[pr, pc] = BACKGROUND

    for obj in objects:
        if obj["branch"] != "horizontal":
            continue
        info["horizontal_count"] += 1
        r0, c0, r1, c1 = obj["bbox"]
        for r, c in _frame_cells(r0, c0, r1, c1):
            out[r, c] = BACKGROUND
        for r, c in _interior_cells(r0, c0, r1, c1):
            out[r, c] = ORANGE

    for obj in sorted(objects, key=lambda item: (item["color"], item["bbox"])):
        if obj["branch"] != "vertical":
            continue
        info["vertical_count"] += 1
        rr, cc = obj["protrusion"]
        dr, dc = obj["ray_dir"]
        length = 0
        overwrites = 0
        while _in_bounds(out.shape, rr, cc) and int(out[rr, cc]) in (BACKGROUND, ORANGE):
            if int(out[rr, cc]) == ORANGE:
                overwrites += 1
            out[rr, cc] = int(obj["color"])
            length += 1
            rr += dr
            cc += dc
        info["cross_fill_overwrites"] += overwrites
        info["ray_lengths"].append(length)

    return out, info


def _branch_mix(count: int) -> list[str]:
    if count == 3:
        return random.choice([["horizontal", "horizontal", "vertical"], ["horizontal", "vertical", "vertical"]])
    options = [
        ["horizontal", "horizontal", "vertical", "vertical"],
        ["horizontal", "vertical", "vertical", "vertical"],
        ["horizontal", "horizontal", "horizontal", "vertical"],
    ]
    return random.choice(options)


def _sample_random_object(
    shape: tuple[int, int],
    color: int,
    branch: str,
    used_vertical_cols: set[int],
) -> dict | None:
    h_total, w_total = shape
    for _ in range(160):
        if branch == "horizontal":
            h = GEN_RECT_SHORT
            w = random.randint(*GEN_RECT_LONG_RANGE)
            side = random.choice(["left", "right"])
        else:
            h = random.randint(*GEN_RECT_LONG_RANGE)
            w = GEN_RECT_SHORT
            side = random.choice(["top", "bottom"])

        r0 = random.randint(1, h_total - h - 1)
        c0 = random.randint(1, w_total - w - 1)
        obj = _make_object(color, branch, side, r0, c0, h, w)
        if branch == "vertical":
            col = int(obj["protrusion"][1])
            if col in used_vertical_cols:
                continue
        return obj
    return None


def _place_cross_pair(
    shape: tuple[int, int],
    h_color: int,
    v_color: int,
    blocked: set[tuple[int, int]],
    used_vertical_cols: set[int],
) -> list[dict] | None:
    h_total, w_total = shape
    for _ in range(220):
        h_h = GEN_RECT_SHORT
        h_w = random.randint(*GEN_RECT_LONG_RANGE)
        h_r0 = random.randint(4, h_total - h_h - 4)
        h_c0 = random.randint(2, w_total - h_w - 3)
        h_side = random.choice(["left", "right"])
        h_obj = _make_object(h_color, "horizontal", h_side, h_r0, h_c0, h_h, h_w)
        if not _can_place(h_obj, blocked, shape):
            continue

        cross_col = random.randint(h_c0 + 1, h_c0 + h_w - 2)
        v_h = random.randint(*GEN_RECT_LONG_RANGE)
        from_above = random.choice([True, False])
        if from_above:
            v_side = "bottom"
            v_r1 = h_r0 - 3
            v_r0 = v_r1 - v_h + 1
        else:
            v_side = "top"
            v_r0 = h_r0 + h_h + 2
        v_c0 = cross_col - 2
        if from_above:
            if v_r0 < 1:
                continue
        else:
            if v_r0 + v_h > h_total - 1:
                continue
        if v_c0 < 1 or v_c0 + GEN_RECT_SHORT > w_total - 1:
            continue

        v_obj = _make_object(v_color, "vertical", v_side, v_r0, v_c0, v_h, GEN_RECT_SHORT)
        if int(v_obj["protrusion"][1]) in used_vertical_cols:
            continue
        if not _can_place(v_obj, blocked | _expanded_cells(_object_input_cells(h_obj)), shape):
            continue
        return [h_obj, v_obj]
    return None


def _validate_generated(inp: np.ndarray, objects: list[dict], cross_required: bool) -> bool:
    solver_objects = _find_objects(inp)
    if len(solver_objects) != len(objects):
        return False

    expected, info = _expected_from_metadata(inp, objects)
    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    if info["horizontal_count"] < 1 or info["vertical_count"] < 1:
        return False
    if any(length < 2 for length in info["ray_lengths"]):
        return False
    if np.array_equal(inp, expected):
        return False
    if cross_required and info["cross_fill_overwrites"] == 0:
        return False

    colors_in_output = set(map(int, np.unique(expected)))
    if len(colors_in_output) < 3:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GEN_GRID_RANGE)
    w = random.randint(*GEN_GRID_RANGE)
    count = random.choice(GEN_OBJECT_COUNT_CHOICES)
    cross_required = random.random() < GEN_CROSS_PROB
    branches = _branch_mix(count)
    random.shuffle(branches)
    colors = random.sample(FRAME_COLORS, count)

    objects: list[dict] = []
    blocked: set[tuple[int, int]] = set()
    used_vertical_cols: set[int] = set()

    if cross_required:
        try:
            h_idx = branches.index("horizontal")
            v_idx = branches.index("vertical")
        except ValueError:
            return None
        pair = _place_cross_pair((h, w), colors[h_idx], colors[v_idx], blocked, used_vertical_cols)
        if pair is None:
            return None
        for obj in pair:
            objects.append(obj)
            _add_to_blocked(blocked, obj)
            if obj["branch"] == "vertical":
                used_vertical_cols.add(int(obj["protrusion"][1]))
        skip_indices = {h_idx, v_idx}
    else:
        skip_indices = set()

    for idx, (color, branch) in enumerate(zip(colors, branches)):
        if idx in skip_indices:
            continue
        obj = _sample_random_object((h, w), color, branch, used_vertical_cols)
        if obj is None:
            return None
        placed = False
        for _ in range(180):
            if _can_place(obj, blocked, (h, w)):
                placed = True
                break
            obj = _sample_random_object((h, w), color, branch, used_vertical_cols)
            if obj is None:
                break
        if not placed or obj is None:
            return None
        objects.append(obj)
        _add_to_blocked(blocked, obj)
        if obj["branch"] == "vertical":
            used_vertical_cols.add(int(obj["protrusion"][1]))

    inp = _draw_input((h, w), objects)
    if not _validate_generated(inp, objects, cross_required):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
