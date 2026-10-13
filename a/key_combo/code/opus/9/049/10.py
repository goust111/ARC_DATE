from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREEN = 3

GRID_MIN = 11
GRID_MAX = 16
OBJECT_AREA_MIN = 5
OBJECT_AREA_MAX = 10
OBJECT_DIM_MIN = 3
OBJECT_DIM_MAX = 5
MAX_ATTEMPTS = 400

ALL_CASES = [
    ("horizontal", 1),
    ("horizontal", 2),
    ("horizontal", 3),
    ("vertical", 1),
    ("vertical", 2),
    ("vertical", 3),
]
_CASE_BAG: list[tuple[str, int]] = []


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _stripe_orientation(stripe_comps: list[list[tuple[int, int]]]) -> str:
    assert 1 <= len(stripe_comps) <= 3, "expected 1-3 green stripe components"
    orientation: str | None = None
    used_lines: set[int] = set()
    for comp in stripe_comps:
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = len(comp)
        if h == 1 and 2 <= w <= 3 and area == w:
            cur = "horizontal"
            line_id = r0
        elif w == 1 and 2 <= h <= 3 and area == h:
            cur = "vertical"
            line_id = c0
        else:
            raise AssertionError("green components must be short solid stripes")
        if orientation is None:
            orientation = cur
        else:
            assert orientation == cur, "all stripes must share the same orientation"
        assert line_id not in used_lines, "stripes must occupy distinct rows or columns"
        used_lines.add(line_id)
    assert orientation is not None
    return orientation


def _transform_patch(patch: np.ndarray, orientation: str, k: int) -> np.ndarray:
    if orientation == "horizontal":
        return np.rot90(patch, -k).copy()
    if k % 2 == 1:
        return np.fliplr(patch).copy()
    return np.flipud(patch).copy()


def _operation_name(orientation: str, k: int) -> str:
    if orientation == "horizontal":
        return f"rot{k * 90}"
    return "fliplr" if k % 2 == 1 else "flipud"


def _render_output(
    grid: np.ndarray,
    obj_box: tuple[int, int, int, int],
    transformed_patch: np.ndarray,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    out[out == GREEN] = BG
    out[out == RED] = BG

    r0, c0, _r1, _c1 = obj_box
    h, w = transformed_patch.shape
    for dr in range(h):
        for dc in range(w):
            if transformed_patch[dr, dc] != RED:
                continue
            rr = r0 + dr
            cc = c0 + dc
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = RED
    return out


def _full_grid_outputs(
    grid: np.ndarray,
    obj_box: tuple[int, int, int, int],
    patch: np.ndarray,
) -> dict[str, np.ndarray]:
    ops = {
        "rot90": np.rot90(patch, -1).copy(),
        "rot180": np.rot90(patch, -2).copy(),
        "rot270": np.rot90(patch, -3).copy(),
        "fliplr": np.fliplr(patch).copy(),
        "flipud": np.flipud(patch).copy(),
    }
    return {name: _render_output(grid, obj_box, candidate) for name, candidate in ops.items()}


def main(grid: np.ndarray) -> np.ndarray:
    """Use green stripe orientation and count as a geometric command that rotates or mirrors the red object, then erase the stripes.

    Input structure:
    - A black background contains one 4-connected red object and 1-3 separate green stripe segments.
    - Every green segment is a short solid stripe, and all green stripes are either horizontal or vertical.
    - The red object is transformed inside its own bounding box while the grid size stays fixed.

    Rule:
    - Count the green stripes and determine whether they are horizontal or vertical.
    - Crop the red object to its minimal bounding rectangle.
    - Horizontal stripes mean clockwise rotation by 90 degrees times the stripe count.
    - Vertical stripes mean a mirror: odd counts flip left-right, even counts flip up-down.
    - Erase all green pixels, clear the original red object, and paste the transformed red patch back at the original bounding-box top-left, clipping anything that falls outside the grid.

    Key insight:
    - The green stripes are not content to preserve; their direction chooses the transformation family and their count chooses the transformation parameter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    stripe_comps = _components(g == GREEN)
    orientation = _stripe_orientation(stripe_comps)
    k = len(stripe_comps)

    red_comps = _components(g == RED)
    assert len(red_comps) == 1, "expected exactly one red object"
    obj_box = _bbox(red_comps[0])
    patch = _crop(g, obj_box)

    transformed_patch = _transform_patch(patch, orientation, k)
    return _render_output(g, obj_box, transformed_patch)


def _shape_signature(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return arr.shape, arr.tobytes()


def _patch_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    shifted = {(r - r0, c - c0) for r, c in cells}
    h = 1 + max(r for r, _c in shifted)
    w = 1 + max(c for _r, c in shifted)
    patch = np.zeros((h, w), dtype=int)
    for r, c in shifted:
        patch[r, c] = RED
    return patch


def _random_connected_patch() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        base_h = random.randint(OBJECT_DIM_MIN, OBJECT_DIM_MAX)
        base_w = random.randint(OBJECT_DIM_MIN, OBJECT_DIM_MAX)
        max_area = min(OBJECT_AREA_MAX, base_h * base_w - 1)
        if max_area < OBJECT_AREA_MIN:
            continue
        target_area = random.randint(OBJECT_AREA_MIN, max_area)
        start = (random.randrange(base_h), random.randrange(base_w))
        cells = {start}
        while len(cells) < target_area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < base_h and 0 <= nc < base_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_area:
            continue

        patch = _patch_from_cells(cells)
        if random.random() < 0.5:
            patch = np.fliplr(patch)
        patch = np.rot90(patch, -random.randrange(4)).copy()

        red_cells = int(np.count_nonzero(patch == RED))
        if not (OBJECT_AREA_MIN <= red_cells <= OBJECT_AREA_MAX):
            continue
        if min(patch.shape) < 2 or red_cells == patch.size:
            continue

        signatures = {
            "orig": _shape_signature(patch),
            "rot90": _shape_signature(np.rot90(patch, -1)),
            "rot180": _shape_signature(np.rot90(patch, -2)),
            "rot270": _shape_signature(np.rot90(patch, -3)),
            "fliplr": _shape_signature(np.fliplr(patch)),
            "flipud": _shape_signature(np.flipud(patch)),
        }
        if len(set(signatures.values())) != len(signatures):
            continue
        return patch.copy()
    raise RuntimeError("failed to construct a non-degenerate red object")


def _refill_case_bag() -> None:
    global _CASE_BAG
    first_three = [
        ("vertical", 2),
        random.choice([("horizontal", 1), ("horizontal", 2), ("horizontal", 3)]),
        random.choice([("vertical", 1), ("vertical", 3)]),
    ]
    random.shuffle(first_three)
    remaining = [case for case in ALL_CASES if case not in first_three]
    random.shuffle(remaining)
    _CASE_BAG = first_three + remaining


def _next_case() -> tuple[str, int]:
    if not _CASE_BAG:
        _refill_case_bag()
    return _CASE_BAG.pop(0)


def _pick_grid_and_position(
    patch: np.ndarray,
    orientation: str,
    k: int,
) -> tuple[int, int, int, int, bool]:
    ph, pw = patch.shape
    transformed = _transform_patch(patch, orientation, k)
    th, tw = transformed.shape
    can_force_clip = orientation == "horizontal" and (k % 2 == 1) and (ph != pw)

    for _ in range(MAX_ATTEMPTS):
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)
        if ph > h or pw > w or th > h + ph or tw > w + pw:
            continue

        row_min = 6 if orientation == "horizontal" else 3
        col_min = 6 if orientation == "vertical" else 3
        if row_min > h - ph or col_min > w - pw:
            continue

        want_clip = can_force_clip and random.random() < 0.35
        clip_ranges: list[tuple[str, int, int]] = []
        if want_clip and th > ph:
            lo = max(row_min, h - th + 1)
            hi = h - ph
            if lo <= hi:
                clip_ranges.append(("row", lo, hi))
        if want_clip and tw > pw:
            lo = max(col_min, w - tw + 1)
            hi = w - pw
            if lo <= hi:
                clip_ranges.append(("col", lo, hi))

        if clip_ranges:
            axis, lo, hi = random.choice(clip_ranges)
            r0_lo = row_min
            r0_hi = h - ph
            c0_lo = col_min
            c0_hi = w - pw
            if axis == "row":
                r0_lo, r0_hi = lo, hi
            else:
                c0_lo, c0_hi = lo, hi
            r0 = random.randint(r0_lo, r0_hi)
            c0 = random.randint(c0_lo, c0_hi)
            return h, w, r0, c0, True

        r0_hi = min(h - ph, h - th)
        c0_hi = min(w - pw, w - tw)
        if row_min > r0_hi or col_min > c0_hi:
            continue
        r0 = random.randint(row_min, r0_hi)
        c0 = random.randint(col_min, c0_hi)
        return h, w, r0, c0, False

    raise RuntimeError("failed to place object")


def _place_stripes(grid: np.ndarray, orientation: str, k: int) -> None:
    if orientation == "horizontal":
        rows = random.sample([0, 2, 4], k)
        for r in rows:
            length = random.randint(2, 3)
            c0 = random.randint(0, grid.shape[1] - length)
            grid[r, c0 : c0 + length] = GREEN
    else:
        cols = random.sample([0, 2, 4], k)
        for c in cols:
            length = random.randint(2, 3)
            r0 = random.randint(0, grid.shape[0] - length)
            grid[r0 : r0 + length, c] = GREEN


def _paint_object(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    for dr in range(patch.shape[0]):
        for dc in range(patch.shape[1]):
            if patch[dr, dc] == RED:
                grid[r0 + dr, c0 + dc] = RED


def _touching(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    h, w = mask_a.shape
    for r in range(h):
        for c in range(w):
            if not mask_a[r, c]:
                continue
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask_b[nr, nc]:
                    return True
    return False


def _validate_generated_example(
    grid: np.ndarray,
    orientation: str,
    k: int,
    patch: np.ndarray,
    obj_box: tuple[int, int, int, int],
    requested_clip: bool,
) -> bool:
    if set(np.unique(grid)) - {BG, RED, GREEN}:
        return False

    stripe_comps = _components(grid == GREEN)
    if len(stripe_comps) != k:
        return False
    if _stripe_orientation(stripe_comps) != orientation:
        return False

    red_comps = _components(grid == RED)
    if len(red_comps) != 1:
        return False
    if len(red_comps[0]) != int(np.count_nonzero(patch == RED)):
        return False
    if _touching(grid == RED, grid == GREEN):
        return False

    out = main(grid)
    if np.any(out == GREEN):
        return False

    target_name = _operation_name(orientation, k)
    all_outputs = _full_grid_outputs(grid, obj_box, patch)
    target = all_outputs[target_name]
    if not np.array_equal(out, target):
        return False

    target_sig = _shape_signature(target)
    alt_sigs = {
        name: _shape_signature(candidate)
        for name, candidate in all_outputs.items()
        if name != target_name
    }
    if target_sig in alt_sigs.values():
        return False

    erased_only = grid.copy()
    erased_only[erased_only == GREEN] = BG
    if np.array_equal(out, erased_only):
        return False

    in_red = int(np.count_nonzero(grid == RED))
    out_red = int(np.count_nonzero(out == RED))
    if requested_clip:
        if not (out_red < in_red and out_red >= max(4, in_red - 2)):
            return False
    elif out_red != in_red:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        orientation, k = _next_case()
        patch = _random_connected_patch()
        h, w, r0, c0, requested_clip = _pick_grid_and_position(patch, orientation, k)

        grid = np.zeros((h, w), dtype=int)
        _paint_object(grid, patch, r0, c0)
        _place_stripes(grid, orientation, k)
        obj_box = (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1)

        if _validate_generated_example(grid, orientation, k, patch, obj_box, requested_clip):
            return grid

    raise RuntimeError("failed to generate a valid input grid")
