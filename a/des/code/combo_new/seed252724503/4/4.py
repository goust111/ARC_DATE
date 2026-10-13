from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
OVERLAP_COLOR = 2
OBJECT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

MAX_GENERATION_TRIES = 300
OBJECT_SHAPE_TRIES = 200
MISSING_SIDE_TRIES = 200
PACKING_TRIES = 100


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            components.append(np.asarray(coords, dtype=int))

    return components


def _is_connected(mask: np.ndarray) -> bool:
    return len(_component_coords(mask)) == 1


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    objects: list[dict] = []

    for color in sorted(set(map(int, g.flat)) - {BACKGROUND}):
        for coords in _component_coords(g == color):
            rows = coords[:, 0]
            cols = coords[:, 1]
            r0, r1 = int(rows.min()), int(rows.max())
            c0, c1 = int(cols.min()), int(cols.max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[rows - r0, cols - c0] = True
            objects.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )

    objects.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return objects


def _candidate_axes(length: int) -> list[float]:
    return [float(i) for i in range(length)] + [i + 0.5 for i in range(length - 1)]


def _side_name(index: int, orientation: str, axis: float) -> str:
    delta = index - axis
    if abs(delta) < 1e-9:
        return "axis"
    if orientation == "v":
        return "right" if delta > 0 else "left"
    return "bottom" if delta > 0 else "top"


def _mirror_coord(r: int, c: int, orientation: str, axis: float) -> tuple[int, int] | None:
    if orientation == "v":
        cc = 2 * axis - c
        if abs(cc - round(cc)) > 1e-9:
            return None
        return r, int(round(cc))

    rr = 2 * axis - r
    if abs(rr - round(rr)) > 1e-9:
        return None
    return int(round(rr)), c


def _evaluate_fold(mask: np.ndarray, orientation: str, axis: float, keep_side: str) -> dict | None:
    h, w = mask.shape
    donor_count = 0
    keep_count = 0
    extra_keep = 0
    axis_count = 0

    for r, c in np.argwhere(mask):
        idx = c if orientation == "v" else r
        side = _side_name(int(idx), orientation, axis)
        if side == "axis":
            axis_count += 1
            continue

        mirror = _mirror_coord(int(r), int(c), orientation, axis)
        if side == keep_side:
            keep_count += 1
            if mirror is None:
                extra_keep += 1
                continue
            rr, cc = mirror
            if not (0 <= rr < h and 0 <= cc < w and mask[rr, cc]):
                extra_keep += 1
        else:
            donor_count += 1
            if mirror is None:
                return None
            rr, cc = mirror
            if not (0 <= rr < h and 0 <= cc < w and mask[rr, cc]):
                return None

    if donor_count == 0 or keep_count == 0 or extra_keep == 0:
        return None

    axis_preference = int(abs(axis - round(axis)) < 1e-9)
    return {
        "orientation": orientation,
        "axis": axis,
        "keep_side": keep_side,
        "donor_count": donor_count,
        "keep_count": keep_count,
        "extra_keep": extra_keep,
        "axis_count": axis_count,
        "axis_preference": axis_preference,
        "score": donor_count * 1000 + keep_count * 10 + extra_keep,
    }


def _fold_candidates(mask: np.ndarray) -> list[dict]:
    candidates: list[dict] = []
    for orientation in ("v", "h"):
        length = mask.shape[1] if orientation == "v" else mask.shape[0]
        keep_sides = ("left", "right") if orientation == "v" else ("top", "bottom")
        for axis in _candidate_axes(length):
            for keep_side in keep_sides:
                candidate = _evaluate_fold(mask, orientation, axis, keep_side)
                if candidate is not None:
                    candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            item["score"],
            item["axis_preference"],
            item["donor_count"],
            item["keep_count"],
        ),
        reverse=True,
    )
    return candidates


def _choose_fold_spec(mask: np.ndarray) -> dict:
    candidates = _fold_candidates(mask)
    assert candidates, "object has no valid fold axis"

    best = candidates[0]
    best_key = (
        best["score"],
        best["axis_preference"],
        best["donor_count"],
        best["keep_count"],
    )
    ties = [
        item
        for item in candidates
        if (
            item["score"],
            item["axis_preference"],
            item["donor_count"],
            item["keep_count"],
        )
        == best_key
    ]
    assert len(ties) == 1, "object has an ambiguous fold axis"
    return best


def _fold_local(mask: np.ndarray, color: int, spec: dict) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=int)

    for r in range(h):
        for c in range(w):
            idx = c if spec["orientation"] == "v" else r
            side = _side_name(idx, spec["orientation"], spec["axis"])
            if side not in {"axis", spec["keep_side"]}:
                continue

            keep_present = bool(mask[r, c])
            if side == "axis":
                if keep_present:
                    out[r, c] = color
                continue

            donor_present = False
            mirror = _mirror_coord(r, c, spec["orientation"], spec["axis"])
            if mirror is not None:
                rr, cc = mirror
                if 0 <= rr < h and 0 <= cc < w:
                    donor_present = bool(mask[rr, cc])

            if keep_present or donor_present:
                out[r, c] = OVERLAP_COLOR if keep_present and donor_present else color

    return out


def _is_exact_symmetry(mask: np.ndarray, orientation: str, axis: float) -> bool:
    h, w = mask.shape
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    for r, c in coords:
        mirror = _mirror_coord(int(r), int(c), orientation, axis)
        if mirror is None:
            return False
        rr, cc = mirror
        if not (0 <= rr < h and 0 <= cc < w and mask[rr, cc]):
            return False
    return True


def _exact_axes(mask: np.ndarray) -> list[tuple[str, float]]:
    axes: list[tuple[str, float]] = []
    for orientation in ("v", "h"):
        length = mask.shape[1] if orientation == "v" else mask.shape[0]
        for axis in _candidate_axes(length):
            if _is_exact_symmetry(mask, orientation, axis):
                axes.append((orientation, axis))
    return axes


def _crop_nonzero(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BACKGROUND)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each nearly symmetric monochrome object onto its own hidden axis and recolor the overlaps red.

    Input structure:
    - The background is black.
    - The grid contains several disjoint 4-connected monochrome objects.
    - Each object is almost mirror-symmetric about exactly one horizontal or vertical axis, but one side is missing some pixels.

    Rule:
    - For each object, locate the unique axis whose opposite side is a subset of the other side under reflection.
    - Keep the more complete side together with the axis, and fold the smaller side onto it.
    - Cells occupied by both sides after folding become red; cells occupied by only one side keep the object's original color.
    - Combine all folded objects in their original locations and crop the result to the tight non-background bounding box.

    Key insight:
    - The object is not arbitrary noise: one side is a damaged copy of the other across a hidden axis, so folding is detected by a one-sided mirror-containment test rather than exact symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.zeros_like(g)
    for obj in _extract_objects(g):
        spec = _choose_fold_spec(obj["mask"])
        folded = _fold_local(obj["mask"], int(obj["color"]), spec)
        r0, c0, _, _ = obj["bbox"]
        h, w = folded.shape
        region = out[r0 : r0 + h, c0 : c0 + w]
        region[folded != BACKGROUND] = folded[folded != BACKGROUND]

    return _crop_nonzero(out)


def _random_half_mask(height: int, side_extent: int) -> np.ndarray | None:
    width = side_extent + 1
    half = np.zeros((height, width), dtype=bool)

    segment_len = random.randint(2, height)
    start = random.randint(0, height - segment_len)
    half[start : start + segment_len, 0] = True

    area_min = int(half.sum()) + 2
    area_max = height * width
    if area_min > area_max:
        return None
    target = random.randint(area_min, area_max)

    while int(half.sum()) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(half):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < height and 0 <= nc < width and not half[nr, nc]:
                    frontier.add((nr, nc))

        if not frontier:
            break

        choices = list(frontier)
        weights = [3 if c > 0 else 1 for _, c in choices]
        rr, cc = random.choices(choices, weights=weights, k=1)[0]
        half[rr, cc] = True

    if not np.any(half[:, 1:]):
        return None
    if np.count_nonzero(half.any(axis=0)) < 2:
        return None
    return half


def _vertical_from_half(half: np.ndarray) -> np.ndarray:
    h, width = half.shape
    side_extent = width - 1
    full = np.zeros((h, side_extent * 2 + 1), dtype=bool)
    axis = side_extent
    full[:, axis:] = half
    full[:, :axis] = half[:, 1:][:, ::-1]
    return full


def _generate_vertical_object() -> np.ndarray | None:
    for _ in range(OBJECT_SHAPE_TRIES):
        height = random.randint(5, 10)
        side_extent = random.randint(2, 5)
        half = _random_half_mask(height, side_extent)
        if half is None:
            continue

        full = _vertical_from_half(half)
        axes = _exact_axes(full)
        if len(axes) == 1 and axes[0][0] == "v" and abs(axes[0][1] - side_extent) < 1e-9:
            return full

    return None


def _remove_from_one_side(full: np.ndarray, keep_side: str) -> np.ndarray | None:
    axis = (full.shape[1] - 1) / 2
    donor_side = "left" if keep_side == "right" else "right"
    donor_cells = [
        (int(r), int(c))
        for r, c in np.argwhere(full)
        if _side_name(int(c), "v", axis) == donor_side
    ]
    if len(donor_cells) < 2:
        return None

    for _ in range(MISSING_SIDE_TRIES):
        near = full.copy()
        max_remove = min(5, len(donor_cells) - 1)
        remove_count = random.randint(1, max_remove)
        weights = [1 + abs(c - axis) for _, c in donor_cells]
        chosen_idx = set()
        while len(chosen_idx) < remove_count:
            chosen_idx.add(random.choices(range(len(donor_cells)), weights=weights, k=1)[0])
        for idx in chosen_idx:
            rr, cc = donor_cells[idx]
            near[rr, cc] = False

        if not _is_connected(near):
            continue

        try:
            spec = _choose_fold_spec(near)
        except AssertionError:
            continue

        if spec["orientation"] != "v" or abs(spec["axis"] - axis) > 1e-9 or spec["keep_side"] != keep_side:
            continue

        folded = _fold_local(near, 3, spec)
        if not np.any(folded == OVERLAP_COLOR):
            continue

        keep_has_extra = False
        for r, c in np.argwhere(folded == 3):
            if _side_name(int(c), "v", axis) == keep_side:
                keep_has_extra = True
                break
        if not keep_has_extra:
            continue

        return near

    return None


def _generate_object_mask(orientation: str) -> np.ndarray | None:
    for _ in range(OBJECT_SHAPE_TRIES):
        full = _generate_vertical_object()
        if full is None:
            continue

        keep_side = random.choice(("left", "right"))
        near = _remove_from_one_side(full, keep_side)
        if near is None:
            continue

        obj = near if orientation == "v" else near.T
        try:
            spec = _choose_fold_spec(obj)
        except AssertionError:
            continue
        if spec["orientation"] != orientation:
            continue
        return obj

    return None


def _pack_objects(masks: list[np.ndarray], colors: list[int]) -> np.ndarray | None:
    items = list(zip(masks, colors))

    for _ in range(PACKING_TRIES):
        random.shuffle(items)
        if len(items) == 1:
            row_items = [items]
        else:
            row_count = random.choice((1, 2))
            row_items = [[] for _ in range(row_count)]
            for idx, item in enumerate(items):
                row_items[idx % row_count].append(item)
            for row in row_items:
                random.shuffle(row)
            row_items = [row for row in row_items if row]

        row_specs = []
        for row in row_items:
            gaps = [random.randint(2, 4) for _ in range(max(0, len(row) - 1))]
            width = sum(mask.shape[1] for mask, _ in row) + sum(gaps)
            height = max(mask.shape[0] for mask, _ in row)
            row_specs.append((row, gaps, width, height))

        v_gaps = [random.randint(2, 4) for _ in range(max(0, len(row_specs) - 1))]
        inner_w = max(width for _, _, width, _ in row_specs)
        inner_h = sum(height for _, _, _, height in row_specs)
        inner_h += sum(v_gaps)

        if inner_w > 30 or inner_h > 30:
            continue

        grid_h = random.randint(inner_h, min(30, inner_h + 6))
        grid_w = random.randint(inner_w, min(30, inner_w + 6))
        grid = np.zeros((grid_h, grid_w), dtype=int)

        row_tops: list[int] = []
        current_r = random.randint(0, grid_h - inner_h)
        for row_idx, (_, _, _, height) in enumerate(row_specs):
            row_tops.append(current_r)
            current_r += height
            if row_idx < len(v_gaps):
                current_r += v_gaps[row_idx]

        for row_top, (row, gaps, width, height) in zip(row_tops, row_specs):
            current_c = random.randint(0, grid_w - width)
            for item_idx, (mask, color) in enumerate(row):
                obj_h, obj_w = mask.shape
                row_offset = random.randint(0, height - obj_h)
                region = grid[row_top + row_offset : row_top + row_offset + obj_h, current_c : current_c + obj_w]
                region[mask] = color
                if item_idx < len(gaps):
                    current_c += obj_w + gaps[item_idx]

        if len(_extract_objects(grid)) != len(masks):
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(1, 4)
    colors = random.sample(OBJECT_COLORS, object_count)
    masks: list[np.ndarray] = []

    for _ in range(object_count):
        orientation = random.choice(("v", "h"))
        mask = _generate_object_mask(orientation)
        if mask is None:
            return None
        masks.append(mask)

    grid = _pack_objects(masks, colors)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.ndim != 2 or max(out.shape) > 30:
        return None
    if not np.any(out == OVERLAP_COLOR):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_TRIES} attempts")
