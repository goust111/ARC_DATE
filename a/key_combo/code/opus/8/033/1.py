from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
NONBLACK = tuple(range(1, 10))
CORNER_ORDER = ("TL", "TR", "BR", "BL")
SIDE_ORDER = ("T", "R", "B", "L")
SIDE_DIRS = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
RING_OFFSETS = {
    "TL": (0, 0),
    "T": (0, 1),
    "TR": (0, 2),
    "L": (1, 0),
    "R": (1, 2),
    "BL": (2, 0),
    "B": (2, 1),
    "BR": (2, 2),
}
ALL_RING_SLOTS = ("TL", "T", "TR", "L", "R", "BL", "B", "BR")
GEN_RETRIES = 600


def _dominant_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _ring_bbox(r0: int, c0: int) -> tuple[int, int, int, int]:
    return r0, c0, r0 + 2, c0 + 2


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    row_gap = max(b[0] - a[2] - 1, a[0] - b[2] - 1, 0)
    col_gap = max(b[1] - a[3] - 1, a[1] - b[3] - 1, 0)
    return row_gap, col_gap


def _boxes_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int], min_gap: int = 2) -> bool:
    row_gap, col_gap = _bbox_gap(a, b)
    return row_gap >= min_gap or col_gap >= min_gap


def _slot_coord(r0: int, c0: int, slot: str) -> tuple[int, int]:
    dr, dc = RING_OFFSETS[slot]
    return r0 + dr, c0 + dc


def _trace_ray(
    shape: tuple[int, int],
    blocker: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[list[tuple[int, int]], str]:
    h, w = shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    path: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        if blocker[r, c]:
            return path, "blocked"
        path.append((r, c))
        r += dr
        c += dc
    return path, "boundary"


def _scan_rings(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert np.any(g == BLACK), "expected black cells for the background and ring centers"
    h, w = g.shape

    rings: list[dict] = []
    cover = np.zeros_like(g, dtype=int)

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if g[r, c] != BLACK:
                continue
            r0, c0 = r - 1, c - 1
            patch = g[r0 : r0 + 3, c0 : c0 + 3]
            if patch.shape != (3, 3):
                continue
            if np.count_nonzero(patch) != 7:
                continue
            if patch[1, 1] != BLACK:
                continue
            if any(patch[dr, dc] == BLACK for dr, dc in (RING_OFFSETS["TL"], RING_OFFSETS["TR"], RING_OFFSETS["BL"], RING_OFFSETS["BR"])):
                continue

            zero_sides = [side for side in SIDE_ORDER if patch[RING_OFFSETS[side]] == BLACK]
            if len(zero_sides) != 1:
                continue
            gap_side = zero_sides[0]
            visible_colors = {int(v) for v in patch.flat if v != BLACK}
            if not 2 <= len(visible_colors) <= 3:
                continue

            visible = {
                _slot_coord(r0, c0, slot)
                for slot in ALL_RING_SLOTS
                if slot != gap_side
            }
            for rr, cc in visible:
                cover[rr, cc] += 1

            rings.append(
                {
                    "center": (r, c),
                    "r0": r0,
                    "c0": c0,
                    "bbox": (r0, c0, r0 + 2, c0 + 2),
                    "gap_side": gap_side,
                    "gap_coord": _slot_coord(r0, c0, gap_side),
                }
            )

    assert rings, "expected at least one one-gap ring"
    nonblack = g != BLACK
    assert np.all(cover[nonblack] == 1), "every non-black pixel must belong to exactly one ring"
    assert np.all(cover[~nonblack] == 0), "only ring pixels may be non-black"
    return rings


def _transform_details(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    rings = _scan_rings(g)
    pre_ray = np.zeros_like(g)
    ray_specs: list[dict] = []

    for ring in rings:
        r0, c0 = ring["r0"], ring["c0"]
        patch = g[r0 : r0 + 3, c0 : c0 + 3]

        old_corners = [int(patch[RING_OFFSETS[slot]]) for slot in CORNER_ORDER]
        new_corners = [old_corners[3], old_corners[0], old_corners[1], old_corners[2]]
        for slot, value in zip(CORNER_ORDER, new_corners):
            rr, cc = _slot_coord(r0, c0, slot)
            pre_ray[rr, cc] = value

        old_sides = [int(patch[RING_OFFSETS[slot]]) for slot in SIDE_ORDER]
        new_sides = [old_sides[1], old_sides[2], old_sides[3], old_sides[0]]
        for slot, value in zip(SIDE_ORDER, new_sides):
            rr, cc = _slot_coord(r0, c0, slot)
            pre_ray[rr, cc] = value

        gap_idx = SIDE_ORDER.index(ring["gap_side"])
        ray_specs.append(
            {
                **ring,
                "ray_color": int(new_sides[gap_idx]),
                "direction": SIDE_DIRS[ring["gap_side"]],
            }
        )

    blocker = pre_ray != BLACK
    ray_paths: list[set[tuple[int, int]]] = []
    gap_coords = [spec["gap_coord"] for spec in ray_specs]

    for idx, spec in enumerate(ray_specs):
        path, stop_reason = _trace_ray(g.shape, blocker, spec["gap_coord"], spec["direction"])
        spec["ray_path"] = path
        spec["stop_reason"] = stop_reason
        spec["ray_length"] = len(path)
        path_set = set(path)
        ray_paths.append(path_set)
        assert all(cell not in path_set for j, cell in enumerate(gap_coords) if j != idx), "rays may not pass through another gap cell"

    for i in range(len(ray_paths)):
        for j in range(i + 1, len(ray_paths)):
            assert not (ray_paths[i] & ray_paths[j]), "ray paths must be disjoint"

    out = pre_ray.copy()
    for spec in ray_specs:
        for r, c in spec["ray_path"]:
            out[r, c] = spec["ray_color"]

    return out, ray_specs


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each 3x3 one-gap ring, use the cycled gap color, and emit an outward ray from the original gap.

    Input structure:
    - The grid has black background and several separated 3x3 ring objects.
    - Each ring has a black center, four non-black corners, three non-black side-centers, and exactly one missing side-center gap.
    - Rings may use 2-3 visible colors per object and are spaced so their outward rays do not overlap.

    Rule:
    - For each ring, cycle the four corner colors one step clockwise.
    - Cycle the four side-center values one step counterclockwise, treating the missing gap as a real black slot in that four-cycle.
    - The original gap location receives its post-cycle color and then emits a straight ray outward, painting background cells of that color until the next non-black pixel or the grid boundary.
    - The ring itself is redrawn with its cycled colors, so the black gap moves to the side-center that receives the black slot.

    Key insight:
    - The missing side-center is not ignored: it behaves as a black placeholder inside the side-center cycle, and the color moved into that original gap is exactly the ray color.
    """
    out, _ = _transform_details(np.asarray(grid, dtype=int))
    return out


def _make_ring_patch(gap_side: str, palette_size: int) -> np.ndarray:
    colors = random.sample(NONBLACK, palette_size)
    patch = np.zeros((3, 3), dtype=int)

    while True:
        for slot in ALL_RING_SLOTS:
            rr, cc = RING_OFFSETS[slot]
            if slot == gap_side:
                patch[rr, cc] = BLACK
            else:
                patch[rr, cc] = random.choice(colors)
        patch[1, 1] = BLACK

        corner_vals = [int(patch[RING_OFFSETS[slot]]) for slot in CORNER_ORDER]
        side_vals = [int(patch[RING_OFFSETS[slot]]) for slot in SIDE_ORDER if slot != gap_side]
        if len(set(corner_vals)) < 2:
            continue
        if len(set(side_vals)) < 2:
            continue
        if len(set(int(v) for v in patch.flat if v != BLACK)) != palette_size:
            continue

        transformed = main(patch)
        if np.array_equal(transformed, patch):
            continue
        if int(np.count_nonzero(transformed != patch)) < 4:
            continue
        return patch.copy()


def _render_grid(shape: tuple[int, int], ring_specs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in ring_specs:
        r0, c0 = spec["pos"]
        grid[r0 : r0 + 3, c0 : c0 + 3] = spec["patch"]
    return grid


def _allowed_gap_dirs(r0: int, c0: int, h: int, w: int, min_clear: int = 1) -> list[str]:
    choices = []
    if r0 >= min_clear:
        choices.append("T")
    if c0 + 2 + min_clear < w:
        choices.append("R")
    if r0 + 2 + min_clear < h:
        choices.append("B")
    if c0 >= min_clear:
        choices.append("L")
    return choices


def _random_free_position(
    h: int,
    w: int,
    boxes: list[tuple[int, int, int, int]],
    forbidden: set[tuple[int, int]],
    gap_dirs: tuple[str, ...],
) -> tuple[tuple[int, int], str] | None:
    candidates = [(r0, c0) for r0 in range(h - 2) for c0 in range(w - 2)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        box = _ring_bbox(r0, c0)
        if any(not _boxes_far_enough(box, other) for other in boxes):
            continue
        box_cells = {(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)}
        if box_cells & forbidden:
            continue
        dirs = [d for d in gap_dirs if d in _allowed_gap_dirs(r0, c0, h, w, min_clear=1)]
        if not dirs:
            continue
        return (r0, c0), random.choice(dirs)
    return None


def _place_pair(h: int, w: int, source_dir: str) -> list[dict] | None:
    if source_dir == "R":
        target_gap_options = []
        for target_gap in ("T", "B"):
            row_choices = list(range(h - 2))
            if target_gap == "T":
                row_choices = [r0 for r0 in row_choices if r0 >= 2]
            else:
                row_choices = [r0 for r0 in row_choices if r0 + 4 < h]
            for r0 in row_choices:
                for c0_t in range(5, w - 2):
                    if c0_t + 2 >= w:
                        continue
                    for c0_s in range(0, c0_t - 4):
                        target_gap_coord = _slot_coord(r0, c0_t, target_gap)
                        path_cells = {(r0 + 1, c) for c in range(c0_s + 3, c0_t)}
                        if target_gap_coord in path_cells:
                            continue
                        target_path, _ = _trace_ray((h, w), np.zeros((h, w), dtype=bool), target_gap_coord, SIDE_DIRS[target_gap])
                        if len(target_path) < 2:
                            continue
                        target_gap_line = set(target_path)
                        if path_cells & target_gap_line:
                            continue
                        target_gap_cell = target_gap_coord
                        if target_gap_cell in path_cells:
                            continue
                        target_gap_options.append(
                            [
                                {"pos": (r0, c0_s), "gap_side": "R"},
                                {"pos": (r0, c0_t), "gap_side": target_gap},
                            ]
                        )
        return random.choice(target_gap_options) if target_gap_options else None

    if source_dir == "L":
        target_gap_options = []
        for target_gap in ("T", "B"):
            row_choices = list(range(h - 2))
            if target_gap == "T":
                row_choices = [r0 for r0 in row_choices if r0 >= 2]
            else:
                row_choices = [r0 for r0 in row_choices if r0 + 4 < h]
            for r0 in row_choices:
                for c0_t in range(0, w - 7):
                    for c0_s in range(c0_t + 5, w - 2):
                        target_gap_coord = _slot_coord(r0, c0_t, target_gap)
                        path_cells = {(r0 + 1, c) for c in range(c0_t + 3, c0_s)}
                        if target_gap_coord in path_cells:
                            continue
                        target_path, _ = _trace_ray((h, w), np.zeros((h, w), dtype=bool), target_gap_coord, SIDE_DIRS[target_gap])
                        if len(target_path) < 2:
                            continue
                        if path_cells & set(target_path):
                            continue
                        target_gap_options.append(
                            [
                                {"pos": (r0, c0_s), "gap_side": "L"},
                                {"pos": (r0, c0_t), "gap_side": target_gap},
                            ]
                        )
        return random.choice(target_gap_options) if target_gap_options else None

    if source_dir == "B":
        target_gap_options = []
        for target_gap in ("L", "R"):
            col_choices = list(range(w - 2))
            if target_gap == "L":
                col_choices = [c0 for c0 in col_choices if c0 >= 2]
            else:
                col_choices = [c0 for c0 in col_choices if c0 + 4 < w]
            for c0 in col_choices:
                for r0_t in range(5, h - 2):
                    for r0_s in range(0, r0_t - 4):
                        target_gap_coord = _slot_coord(r0_t, c0, target_gap)
                        path_cells = {(r, c0 + 1) for r in range(r0_s + 3, r0_t)}
                        if target_gap_coord in path_cells:
                            continue
                        target_path, _ = _trace_ray((h, w), np.zeros((h, w), dtype=bool), target_gap_coord, SIDE_DIRS[target_gap])
                        if len(target_path) < 2:
                            continue
                        if path_cells & set(target_path):
                            continue
                        target_gap_options.append(
                            [
                                {"pos": (r0_s, c0), "gap_side": "B"},
                                {"pos": (r0_t, c0), "gap_side": target_gap},
                            ]
                        )
        return random.choice(target_gap_options) if target_gap_options else None

    target_gap_options = []
    for target_gap in ("L", "R"):
        col_choices = list(range(w - 2))
        if target_gap == "L":
            col_choices = [c0 for c0 in col_choices if c0 >= 2]
        else:
            col_choices = [c0 for c0 in col_choices if c0 + 4 < w]
        for c0 in col_choices:
            for r0_t in range(0, h - 7):
                for r0_s in range(r0_t + 5, h - 2):
                    target_gap_coord = _slot_coord(r0_t, c0, target_gap)
                    path_cells = {(r, c0 + 1) for r in range(r0_t + 3, r0_s)}
                    if target_gap_coord in path_cells:
                        continue
                    target_path, _ = _trace_ray((h, w), np.zeros((h, w), dtype=bool), target_gap_coord, SIDE_DIRS[target_gap])
                    if len(target_path) < 2:
                        continue
                    if path_cells & set(target_path):
                        continue
                    target_gap_options.append(
                        [
                            {"pos": (r0_s, c0), "gap_side": "T"},
                            {"pos": (r0_t, c0), "gap_side": target_gap},
                        ]
                    )
    return random.choice(target_gap_options) if target_gap_options else None


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        out, rings = _transform_details(grid)
    except AssertionError:
        return False

    if len(rings) < 2:
        return False
    if np.array_equal(out, grid):
        return False
    if int(np.count_nonzero(out)) <= int(np.count_nonzero(grid)):
        return False

    reasons = Counter(spec["stop_reason"] for spec in rings)
    if reasons["blocked"] < 1 or reasons["boundary"] < 1:
        return False
    if any(spec["ray_length"] < 1 for spec in rings):
        return False

    gap_dirs = {spec["gap_side"] for spec in rings}
    if len(gap_dirs) < 2:
        return False

    boxes = [spec["bbox"] for spec in rings]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if not _boxes_far_enough(boxes[i], boxes[j], min_gap=2):
                return False

    if not np.array_equal(main(grid), out):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    ring_count = random.randint(2, 4)
    source_dir = random.choice(SIDE_ORDER)

    placed = _place_pair(h, w, source_dir)
    if placed is None:
        return None

    boxes = [_ring_bbox(*spec["pos"]) for spec in placed]
    forbidden: set[tuple[int, int]] = set()
    palette_sizes = [2, 3] + [random.choice((2, 3)) for _ in range(ring_count - 2)]

    for spec in placed:
        r0, c0 = spec["pos"]
        gap = _slot_coord(r0, c0, spec["gap_side"])
        path, _ = _trace_ray((h, w), np.zeros((h, w), dtype=bool), gap, SIDE_DIRS[spec["gap_side"]])
        forbidden.update(path)

    extras: list[dict] = []
    for idx in range(ring_count - 2):
        preferred = tuple(side for side in SIDE_ORDER if side not in {spec["gap_side"] for spec in placed + extras})
        gap_pool = preferred if preferred else SIDE_ORDER
        choice = _random_free_position(h, w, boxes, forbidden, tuple(gap_pool))
        if choice is None:
            return None
        (r0, c0), gap_side = choice
        extras.append({"pos": (r0, c0), "gap_side": gap_side})
        boxes.append(_ring_bbox(r0, c0))

    ring_specs = placed + extras
    random.shuffle(ring_specs)
    for spec, palette_size in zip(ring_specs, palette_sizes):
        spec["patch"] = _make_ring_patch(spec["gap_side"], palette_size)

    grid = _render_grid((h, w), ring_specs)
    return grid if _semantic_ok(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-gap ring ray input")
