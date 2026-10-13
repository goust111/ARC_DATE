from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GRAY = 5

MAX_ATTEMPTS = 400
GRID_MIN = 24
GRID_MAX = 30
OBJECT_COUNT_RANGE = (3, 5)
OBJECT_SIDE_RANGE = (7, 11)
PLACEMENT_PADDING = 1


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _interior_zero_mask(patch: np.ndarray) -> np.ndarray:
    h, w = patch.shape
    zero = patch == BG
    exterior = np.zeros_like(zero, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if zero[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if zero[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and zero[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    return zero & ~exterior


def _line_runs(pattern: np.ndarray, axis: int) -> list[tuple[int, int, tuple[int, ...]]]:
    lines = pattern if axis == 0 else pattern.T
    runs: list[tuple[int, int, tuple[int, ...]]] = []
    start = 0
    for i in range(1, len(lines) + 1):
        if i == len(lines) or not np.array_equal(lines[i], lines[start]):
            runs.append((start, i, tuple(map(int, lines[start]))))
            start = i
    return runs


def _stretch_run_index(runs: list[tuple[int, int, tuple[int, ...]]]) -> int:
    color_counts = [len(set(line)) for _start, _end, line in runs]
    max_count = max(color_counts)
    candidates = [i for i, count in enumerate(color_counts) if count == max_count]
    if len(candidates) == 1:
        return candidates[0]

    lengths = [end - start for start, end, _line in runs]
    max_length = max(lengths[i] for i in candidates)
    candidates = [i for i in candidates if lengths[i] == max_length]
    return candidates[0]


def _axis_indices(pattern: np.ndarray, axis: int, target_length: int) -> list[int]:
    runs = _line_runs(pattern, axis)
    stretch_i = _stretch_run_index(runs)
    fixed_length = sum(end - start for i, (start, end, _line) in enumerate(runs) if i != stretch_i)
    stretch_length = int(target_length) - fixed_length
    if stretch_length < 0:
        raise ValueError("target is too small for the fixed exemplar bands")

    indices: list[int] = []
    for i, (start, end, _line) in enumerate(runs):
        if i == stretch_i:
            indices.extend([start] * stretch_length)
        else:
            indices.extend(range(start, end))
    if len(indices) != int(target_length):
        raise ValueError("stretch produced an incorrect axis length")
    return indices


def _stretch_pattern(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    row_indices = _axis_indices(pattern, 0, int(shape[0]))
    col_indices = _axis_indices(pattern, 1, int(shape[1]))
    return pattern[np.ix_(row_indices, col_indices)]


def _exemplar_fixed_lengths(pattern: np.ndarray) -> tuple[int, int]:
    row_runs = _line_runs(pattern, 0)
    col_runs = _line_runs(pattern, 1)
    row_stretch = _stretch_run_index(row_runs)
    col_stretch = _stretch_run_index(col_runs)
    min_h = sum(end - start for i, (start, end, _line) in enumerate(row_runs) if i != row_stretch)
    min_w = sum(end - start for i, (start, end, _line) in enumerate(col_runs) if i != col_stretch)
    return int(min_h), int(min_w)


def _extract_exemplar(grid: np.ndarray) -> dict:
    comps = _components_from_mask(grid != BG)
    candidates = []
    for coords in comps:
        bbox = _bbox_coords(coords)
        patch = _crop_bbox(grid, bbox)
        if patch.shape[0] < 3 or patch.shape[1] < 3:
            continue
        if np.any(patch == BG):
            continue
        if not (
            np.all(patch[0, :] == GRAY)
            and np.all(patch[-1, :] == GRAY)
            and np.all(patch[:, 0] == GRAY)
            and np.all(patch[:, -1] == GRAY)
        ):
            continue
        inner = patch[1:-1, 1:-1]
        inner_colors = set(map(int, np.unique(inner)))
        if inner_colors != {BLUE, RED}:
            continue
        mask = np.zeros_like(grid, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        candidates.append({"bbox": bbox, "patch": patch, "mask": mask})
    if len(candidates) != 1:
        raise ValueError("expected exactly one multicolor exemplar rectangle")
    return candidates[0]


def _extract_gray_objects(grid: np.ndarray, exemplar_mask: np.ndarray) -> list[dict]:
    gray_mask = (grid == GRAY) & ~exemplar_mask
    objects = []
    for coords in _components_from_mask(gray_mask):
        bbox = _bbox_coords(coords)
        patch = _crop_bbox(grid, bbox)
        if not set(map(int, np.unique(patch))) <= {BG, GRAY}:
            raise ValueError("gray objects may only contain gray shell and black hole")
        hole_mask = _interior_zero_mask(patch)
        hole_comps = _components_from_mask(hole_mask)
        if len(hole_comps) != 1:
            raise ValueError("each gray object must contain exactly one interior hole")
        hb = _bbox_coords(hole_comps[0])
        hr0, hc0, hr1, hc1 = hb
        hole_crop = hole_mask[hr0 : hr1 + 1, hc0 : hc1 + 1]
        if int(hole_crop.sum()) != hole_crop.size:
            raise ValueError("hole must be a filled rectangle")
        r0, c0, _r1, _c1 = bbox
        objects.append(
            {
                "bbox": bbox,
                "patch": patch,
                "hole_bbox_local": hb,
                "hole_bbox_global": (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1),
                "hole_shape": hole_crop.shape,
            }
        )
    if not objects:
        raise ValueError("expected at least one gray object")
    return objects


def _transform_from_parts(grid: np.ndarray, exemplar: dict, objects: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    er0, ec0, er1, ec1 = exemplar["bbox"]
    out[er0 : er1 + 1, ec0 : ec1 + 1] = BG
    pattern = exemplar["patch"]

    for obj in objects:
        hr0, hc0, hr1, hc1 = obj["hole_bbox_global"]
        hh, hw = obj["hole_shape"]
        if hh == hw:
            out[hr0 : hr1 + 1, hc0 : hc1 + 1] = RED
        else:
            out[hr0 : hr1 + 1, hc0 : hc1 + 1] = _stretch_pattern(pattern, (hh, hw))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill gray-object holes by geometry-gated red fill or exemplar-driven elastic stretching.

    Input structure:
    - The background is black.
    - One foreground component is a multicolor exemplar rectangle whose full outer border is gray and whose interior uses red and blue.
    - Several separate gray rectangular objects each contain one enclosed black rectangular hole.

    Rule:
    - Identify the unique multicolor exemplar and crop it to its tight bbox.
    - Run-length encode the exemplar's identical rows and columns; in each axis, treat the run whose representative line has the most colors as the unique internal stretch run.
    - For every gray object's hole, fill square holes solid red.
    - Fill non-square rectangular holes with the exemplar resized to the hole by repeating or truncating only the internal stretch row run and internal stretch column run.
    - Remove the original exemplar by turning its entire bbox black.

    Key insight:
    - Hole geometry chooses the branch: square holes ignore the exemplar and become red, while non-square holes require extracting the exemplar's internal stretch row and column before filling.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    exemplar = _extract_exemplar(g)
    objects = _extract_gray_objects(g, exemplar["mask"])
    return _transform_from_parts(g, exemplar, objects)


def _make_exemplar() -> np.ndarray:
    inner_h = random.choice((3, 4))
    inner_w = random.choice((3, 4))
    patch = np.full((inner_h + 2, inner_w + 2), GRAY, dtype=int)
    patch[1:-1, 1:-1] = BLUE

    candidates = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if len(candidates) > 1:
        center = (inner_h // 2, inner_w // 2)
        off_center = [pos for pos in candidates if pos != center]
        if off_center:
            candidates = off_center
    sr, sc = random.choice(candidates)
    patch[1 + sr, 1 + sc] = RED
    return patch


def _make_object(hole_h: int, hole_w: int) -> np.ndarray:
    outer_h_lo = max(OBJECT_SIDE_RANGE[0], hole_h + 2)
    outer_h_hi = min(OBJECT_SIDE_RANGE[1], hole_h + 5)
    outer_w_lo = max(OBJECT_SIDE_RANGE[0], hole_w + 2)
    outer_w_hi = min(OBJECT_SIDE_RANGE[1], hole_w + 5)
    if outer_h_lo > outer_h_hi or outer_w_lo > outer_w_hi:
        raise ValueError("hole does not fit object size bounds")

    outer_h = random.randint(outer_h_lo, outer_h_hi)
    outer_w = random.randint(outer_w_lo, outer_w_hi)
    top = random.randint(1, outer_h - hole_h - 1)
    left = random.randint(1, outer_w - hole_w - 1)

    patch = np.full((outer_h, outer_w), GRAY, dtype=int)
    patch[top : top + hole_h, left : left + hole_w] = BG
    return patch


def _non_square_hole_sizes(exemplar: np.ndarray) -> list[tuple[int, int]]:
    min_h, min_w = _exemplar_fixed_lengths(exemplar)
    inner_h = exemplar.shape[0] - 2
    inner_w = exemplar.shape[1] - 2
    max_h = min(8, inner_h * 3)
    max_w = min(8, inner_w * 3)
    sizes = []
    for hh in range(min_h, max_h + 1):
        for hw in range(min_w, max_w + 1):
            if hh != hw:
                sizes.append((hh, hw))
    return sizes


def _square_hole_sizes(exemplar: np.ndarray) -> list[int]:
    _min_h, _min_w = _exemplar_fixed_lengths(exemplar)
    inner_h = exemplar.shape[0] - 2
    inner_w = exemplar.shape[1] - 2
    max_side = min(6, inner_h * 3, inner_w * 3, OBJECT_SIDE_RANGE[1] - 2)
    min_side = 3
    return [side for side in range(min_side, max_side + 1)]


def _place_patch(canvas: np.ndarray, blocked: np.ndarray, patch: np.ndarray) -> bool:
    h, w = patch.shape
    ch, cw = canvas.shape
    candidates = [(r, c) for r in range(ch - h + 1) for c in range(cw - w + 1)]
    random.shuffle(candidates)

    for r0, c0 in candidates:
        if blocked[r0 : r0 + h, c0 : c0 + w].any():
            continue
        canvas[r0 : r0 + h, c0 : c0 + w] = patch
        r1 = min(ch, r0 + h + PLACEMENT_PADDING)
        c1 = min(cw, c0 + w + PLACEMENT_PADDING)
        rr0 = max(0, r0 - PLACEMENT_PADDING)
        cc0 = max(0, c0 - PLACEMENT_PADDING)
        blocked[rr0:r1, cc0:c1] = True
        return True
    return False


def _unique_max_color_run(pattern: np.ndarray, axis: int) -> bool:
    runs = _line_runs(pattern, axis)
    counts = [len(set(line)) for _start, _end, line in runs]
    return counts.count(max(counts)) == 1


def _validate_generated_case(grid: np.ndarray) -> None:
    exemplar = _extract_exemplar(grid)
    objects = _extract_gray_objects(grid, exemplar["mask"])

    if not (OBJECT_COUNT_RANGE[0] <= len(objects) <= OBJECT_COUNT_RANGE[1]):
        raise ValueError("unexpected object count")
    if not _unique_max_color_run(exemplar["patch"], 0):
        raise ValueError("row stretch run must be unique")
    if not _unique_max_color_run(exemplar["patch"], 1):
        raise ValueError("column stretch run must be unique")

    square_count = 0
    non_square_count = 0
    min_h, min_w = _exemplar_fixed_lengths(exemplar["patch"])
    for obj in objects:
        hh, hw = obj["hole_shape"]
        if hh == hw:
            square_count += 1
        else:
            non_square_count += 1
            if hh < min_h or hw < min_w:
                raise ValueError("non-square hole is too small for fixed exemplar bands")
    if square_count == 0 or non_square_count == 0:
        raise ValueError("both geometric branches must appear")

    out = main(grid)
    expected = _transform_from_parts(grid, exemplar, objects)
    if not np.array_equal(out, expected):
        raise ValueError("main output does not match the parsed rule")

    er0, ec0, er1, ec1 = exemplar["bbox"]
    if np.any(out[er0 : er1 + 1, ec0 : ec1 + 1] != BG):
        raise ValueError("exemplar was not removed")
    for obj in objects:
        hr0, hc0, hr1, hc1 = obj["hole_bbox_global"]
        hh, hw = obj["hole_shape"]
        fill = out[hr0 : hr1 + 1, hc0 : hc1 + 1]
        if hh == hw:
            if not np.all(fill == RED):
                raise ValueError("square hole must be solid red")
        else:
            target = _stretch_pattern(exemplar["patch"], (hh, hw))
            if not np.array_equal(fill, target):
                raise ValueError("non-square hole fill mismatch")
            if set(map(int, np.unique(fill))) <= {RED}:
                raise ValueError("non-square fill degenerated to a trivial red block")


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        exemplar = _make_exemplar()
        non_square_sizes = _non_square_hole_sizes(exemplar)
        square_sizes = _square_hole_sizes(exemplar)
        if not non_square_sizes or not square_sizes:
            continue

        n_objects = random.randint(*OBJECT_COUNT_RANGE)
        branch_types = ["square", "non_square"] + [random.choice(("square", "non_square")) for _ in range(n_objects - 2)]
        random.shuffle(branch_types)

        patches: list[np.ndarray] = [exemplar]
        try:
            for branch in branch_types:
                if branch == "square":
                    side = random.choice(square_sizes)
                    patches.append(_make_object(side, side))
                else:
                    hole_h, hole_w = random.choice(non_square_sizes)
                    patches.append(_make_object(hole_h, hole_w))
        except ValueError:
            continue

        grid_h = random.randint(GRID_MIN, GRID_MAX)
        grid_w = random.randint(GRID_MIN, GRID_MAX)
        grid = np.full((grid_h, grid_w), BG, dtype=int)
        blocked = np.zeros((grid_h, grid_w), dtype=bool)

        ok = True
        for patch in sorted(patches, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
            if not _place_patch(grid, blocked, patch):
                ok = False
                break
        if not ok:
            continue

        try:
            _validate_generated_case(grid)
        except Exception:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
