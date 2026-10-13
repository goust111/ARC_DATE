from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_holes(local_mask: np.ndarray) -> list[np.ndarray]:
    h, w = local_mask.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not local_mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not local_mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not local_mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    holes = (~local_mask) & (~exterior)
    return _components_mask(holes)


def _parse_objects(grid: np.ndarray) -> list[dict]:
    objects: list[dict] = []
    for coords in _components_mask(grid != BG):
        r0, c0, r1, c1 = _bbox_coords(coords)
        values = {int(grid[r, c]) for r, c in coords}
        colors = values - {GREY}
        assert len(colors) == 1, "each object must have exactly one non-grey color"
        color = int(next(iter(colors)))

        grey_coords = coords[np.array([grid[r, c] == GREY for r, c in coords], dtype=bool)]
        assert len(grey_coords) > 0, "each object must contain one grey fracture line"
        grey_rows = sorted({int(r) for r, _ in grey_coords})
        grey_cols = sorted({int(c) for _, c in grey_coords})

        vertical = (
            len(grey_cols) == 1
            and grey_rows == list(range(r0, r1 + 1))
        )
        horizontal = (
            len(grey_rows) == 1
            and grey_cols == list(range(c0, c1 + 1))
        )
        assert vertical ^ horizontal, "fracture line must be a full row or column"

        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        hole_components = _find_holes(local_mask)
        hole_coords_global: list[np.ndarray] = []
        for hole in hole_components:
            hole_coords_global.append(hole + np.array([r0, c0], dtype=int))

        objects.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "color": color,
                "hole_coords": hole_coords_global,
                "hole_count": len(hole_components),
                "expand_axis": "horizontal" if vertical else "vertical",
            }
        )
    return objects


def _render_from_objects(
    grid: np.ndarray,
    objects: list[dict],
    *,
    parallel_to_fracture: bool = False,
    force_h: int | None = None,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    height, width = out.shape

    for obj in objects:
        color = int(obj["color"])
        coords = obj["coords"]
        r0, c0, r1, c1 = obj["bbox"]
        h = int(obj["hole_count"]) if force_h is None else int(force_h)

        out[coords[:, 0], coords[:, 1]] = color
        for hole in obj["hole_coords"]:
            out[hole[:, 0], hole[:, 1]] = color

        axis = obj["expand_axis"]
        if parallel_to_fracture:
            axis = "vertical" if axis == "horizontal" else "horizontal"

        if axis == "horizontal" and h > 0:
            lc0 = max(0, c0 - h)
            rc1 = min(width - 1, c1 + h)
            if lc0 <= c0 - 1:
                out[r0 : r1 + 1, lc0:c0] = color
            if c1 + 1 <= rc1:
                out[r0 : r1 + 1, c1 + 1 : rc1 + 1] = color
        elif axis == "vertical" and h > 0:
            tr0 = max(0, r0 - h)
            br1 = min(height - 1, r1 + h)
            if tr0 <= r0 - 1:
                out[tr0:r0, c0 : c1 + 1] = color
            if r1 + 1 <= br1:
                out[r1 + 1 : br1 + 1, c0 : c1 + 1] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each cracked holed shape, then expand it perpendicular to the crack by its hole count.

    Input structure:
    - The grid has black background and 2 to 4 separated foreground objects.
    - Each object contains one non-grey shape color plus one grey fracture line.
    - Every object encloses 1 to 3 black rectangular holes, and its fracture line
      spans the full bbox height or width without crossing any hole.

    Rule:
    - Treat each non-black 4-connected component as one object.
    - Count the object's enclosed black 4-connected hole regions inside its bbox.
    - If the grey fracture is vertical, replace it with the shape color and extend
      the bbox left and right by that hole count; if the fracture is horizontal,
      extend the bbox upward and downward by that hole count.
    - Also fill all enclosed holes with the shape color. Clip any extension that
      would go beyond the grid boundary.

    Key insight:
    - The hole count determines how far to expand, while the grey fracture gives
      only the perpendicular expansion axis, so both analyses are necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {0, 1, 2, 3, 4, 5}

    objects = _parse_objects(g)
    assert 2 <= len(objects) <= 4, "expected 2 to 4 objects"
    return _render_from_objects(g, objects)


def _rect_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    row_gap = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
    col_gap = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    return int(row_gap), int(col_gap)


def _rects_separated_by(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    row_gap, col_gap = _rect_gap(a, b)
    return max(row_gap, col_gap) >= int(gap)


def _rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded_bbox(
    top: int,
    left: int,
    patch: np.ndarray,
    hole_count: int,
    expand_axis: str,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    h, w = patch.shape
    r0, c0 = int(top), int(left)
    r1, c1 = r0 + h - 1, c0 + w - 1
    gh, gw = shape
    if expand_axis == "horizontal":
        c0 = max(0, c0 - hole_count)
        c1 = min(gw - 1, c1 + hole_count)
    else:
        r0 = max(0, r0 - hole_count)
        r1 = min(gh - 1, r1 + hole_count)
    return int(r0), int(c0), int(r1), int(c1)


def _sample_hole_rectangles(
    outer_h: int,
    outer_w: int,
    hole_count: int,
    orientation: str,
    fracture_index: int,
) -> list[tuple[int, int, int, int]] | None:
    candidates: list[tuple[int, int, int, int]] = []
    max_hole_h = min(2, outer_h - 2)
    max_hole_w = min(2, outer_w - 2)
    for hh in range(1, max_hole_h + 1):
        for ww in range(1, max_hole_w + 1):
            for r in range(1, outer_h - hh):
                for c in range(1, outer_w - ww):
                    if orientation == "wide" and c <= fracture_index < c + ww:
                        continue
                    if orientation == "tall" and r <= fracture_index < r + hh:
                        continue
                    candidates.append((r, c, hh, ww))
    random.shuffle(candidates)

    target_min_removed = max(0, outer_h * outer_w - 25)
    target_max_removed = outer_h * outer_w - 10
    chosen: list[tuple[int, int, int, int]] = []

    def separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        ar, ac, ah, aw = a
        br, bc, bh, bw = b
        return (
            ar + ah < br - 1
            or br + bh < ar - 1
            or ac + aw < bc - 1
            or bc + bw < ac - 1
        )

    def rec(idx: int, removed: int) -> bool:
        if len(chosen) == hole_count:
            return target_min_removed <= removed <= target_max_removed
        if idx >= len(candidates):
            return False

        need = hole_count - len(chosen)
        if len(candidates) - idx < need:
            return False

        for j in range(idx, len(candidates)):
            rect = candidates[j]
            area = rect[2] * rect[3]
            if removed + area > target_max_removed:
                continue
            if any(not separated(rect, other) for other in chosen):
                continue
            chosen.append(rect)
            if rec(j + 1, removed + area):
                return True
            chosen.pop()
        return False

    if rec(0, 0):
        return chosen[:]
    return None


def _build_patch(color: int, orientation: str, hole_count: int) -> np.ndarray | None:
    if orientation == "wide":
        dims = [(4, 5), (4, 6), (5, 6)]
    else:
        dims = [(5, 4), (6, 4), (6, 5)]

    random.shuffle(dims)
    for outer_h, outer_w in dims:
        indexes = list(range(1, outer_w - 1)) if orientation == "wide" else list(range(1, outer_h - 1))
        random.shuffle(indexes)
        for fracture_index in indexes:
            holes = _sample_hole_rectangles(outer_h, outer_w, hole_count, orientation, fracture_index)
            if holes is None:
                continue
            patch = np.full((outer_h, outer_w), color, dtype=int)
            for r, c, hh, ww in holes:
                patch[r : r + hh, c : c + ww] = BG
            if orientation == "wide":
                patch[:, fracture_index] = GREY
            else:
                patch[fracture_index, :] = GREY

            area = int(np.count_nonzero(patch))
            if 10 <= area <= 25:
                return patch
    return None


def _sample_specs() -> list[dict] | None:
    count = random.randint(2, 4)
    colors = random.sample(PALETTE, count)

    orientations = ["wide", "tall"]
    while len(orientations) < count:
        orientations.append(random.choice(["wide", "tall"]))
    random.shuffle(orientations)

    hole_counts = [random.randint(1, 3) for _ in range(count)]
    if len(set(hole_counts)) == 1:
        idx = random.randrange(count)
        hole_counts[idx] = 1 + (hole_counts[idx] % 3)

    specs: list[dict] = []
    for color, orientation, hole_count in zip(colors, orientations, hole_counts):
        patch = _build_patch(color, orientation, hole_count)
        if patch is None:
            return None
        specs.append(
            {
                "color": color,
                "orientation": orientation,
                "hole_count": hole_count,
                "patch": patch,
                "expand_axis": "horizontal" if orientation == "wide" else "vertical",
            }
        )
    return specs


def _candidate_positions(
    shape: tuple[int, int],
    patch: np.ndarray,
    hole_count: int,
    expand_axis: str,
    want_clipped: bool,
) -> list[tuple[int, int]]:
    gh, gw = shape
    ph, pw = patch.shape
    positions = [(r, c) for r in range(gh - ph + 1) for c in range(gw - pw + 1)]
    if want_clipped:
        clipped: list[tuple[int, int]] = []
        for r, c in positions:
            if expand_axis == "horizontal":
                if c < hole_count or c + pw - 1 + hole_count >= gw:
                    clipped.append((r, c))
            else:
                if r < hole_count or r + ph - 1 + hole_count >= gh:
                    clipped.append((r, c))
        if clipped:
            positions = clipped
    random.shuffle(positions)
    return positions


def _place_specs(grid_shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    gh, gw = grid_shape
    placed: list[dict] = []
    order = sorted(range(len(specs)), key=lambda i: specs[i]["patch"].size + 10 * specs[i]["hole_count"], reverse=True)
    clipped_index = random.choice(order) if random.random() < 0.35 else None

    def rec(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        spec = specs[idx]
        patch = spec["patch"]
        candidates = _candidate_positions(
            (gh, gw),
            patch,
            int(spec["hole_count"]),
            str(spec["expand_axis"]),
            idx == clipped_index,
        )
        for top, left in candidates[:250]:
            in_bbox = (top, left, top + patch.shape[0] - 1, left + patch.shape[1] - 1)
            out_bbox = _expanded_bbox(top, left, patch, int(spec["hole_count"]), str(spec["expand_axis"]), (gh, gw))
            if any(not _rects_separated_by(in_bbox, prev["in_bbox"], 6) for prev in placed):
                continue
            if any(_rects_intersect(out_bbox, prev["out_bbox"]) for prev in placed):
                continue
            placed.append(
                {
                    **spec,
                    "top": top,
                    "left": left,
                    "in_bbox": in_bbox,
                    "out_bbox": out_bbox,
                }
            )
            if rec(k + 1):
                return True
            placed.pop()
        return False

    if rec(0):
        return placed
    return None


def _render_input(grid_shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    for spec in placed:
        top = int(spec["top"])
        left = int(spec["left"])
        patch = spec["patch"]
        ph, pw = patch.shape
        grid[top : top + ph, left : left + pw] = patch
    return grid


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        objects = _parse_objects(grid)
    except AssertionError:
        return False

    if len(objects) < 2:
        return False
    if len({obj["color"] for obj in objects}) != len(objects):
        return False
    if {obj["expand_axis"] for obj in objects} != {"horizontal", "vertical"}:
        return False
    hole_counts = {int(obj["hole_count"]) for obj in objects}
    if len(hole_counts) < 2:
        return False
    if not all(1 <= int(obj["hole_count"]) <= 3 for obj in objects):
        return False

    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            if not _rects_separated_by(a["bbox"], b["bbox"], 6):
                return False

    out = main(grid)
    for i, a in enumerate(objects):
        a_box = _expanded_bbox(a["bbox"][0], a["bbox"][1], np.zeros((a["bbox"][2] - a["bbox"][0] + 1, a["bbox"][3] - a["bbox"][1] + 1), dtype=int), int(a["hole_count"]), str(a["expand_axis"]), grid.shape)
        for b in objects[i + 1 :]:
            b_box = _expanded_bbox(b["bbox"][0], b["bbox"][1], np.zeros((b["bbox"][2] - b["bbox"][0] + 1, b["bbox"][3] - b["bbox"][1] + 1), dtype=int), int(b["hole_count"]), str(b["expand_axis"]), grid.shape)
            if _rects_intersect(a_box, b_box):
                return False

    wrong_parallel = _render_from_objects(grid, objects, parallel_to_fracture=True)
    wrong_fixed = _render_from_objects(grid, objects, force_h=1)
    if np.array_equal(out, wrong_parallel):
        return False
    if np.array_equal(out, wrong_fixed):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        shape = (random.randint(24, 30), random.randint(24, 30))
        specs = _sample_specs()
        if specs is None:
            continue
        placed = _place_specs(shape, specs)
        if placed is None:
            continue
        grid = _render_input(shape, placed)
        if _validate_semantics(grid):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
