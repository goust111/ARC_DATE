from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 400
ROW_RUN_COUNTS = (5, 6)
COL_RUN_COUNTS = (5, 6, 7)
FIXED_RUN_WIDTHS = (1, 2)
STRETCH_RUN_WIDTHS = (2, 3, 4)
PALETTE = tuple(range(1, 10))
COPY_COUNTS = (2, 3)
TARGET_COUNT_OPTIONS = (1, 2, 3)
GRID_MIN = 22
GRID_MAX = 30
OBJECT_BORDER = 1
OBJECT_PADDING = 1


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


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
    if len(runs) < 3:
        raise AssertionError("expected boundary runs plus at least one interior run")
    interior = list(range(1, len(runs) - 1))
    if len(interior) == 1:
        return interior[0]

    color_counts = {i: len(set(runs[i][2])) for i in interior}
    best = max(color_counts.values())
    candidates = [i for i in interior if color_counts[i] == best]
    if len(candidates) == 1:
        return candidates[0]

    lengths = {i: runs[i][1] - runs[i][0] for i in candidates}
    best_len = max(lengths.values())
    candidates = [i for i in candidates if lengths[i] == best_len]
    if len(candidates) != 1:
        raise AssertionError("stretch run must be uniquely identifiable")
    return candidates[0]


def _axis_indices(pattern: np.ndarray, axis: int, target_length: int) -> list[int]:
    runs = _line_runs(pattern, axis)
    stretch_i = _stretch_run_index(runs)
    fixed = sum(end - start for i, (start, end, _line) in enumerate(runs) if i != stretch_i)
    stretch = int(target_length) - fixed
    if stretch <= 0:
        raise AssertionError("target size is too small for fixed bands")

    indices: list[int] = []
    for i, (start, end, _line) in enumerate(runs):
        if i == stretch_i:
            indices.extend([start] * stretch)
        else:
            indices.extend(range(start, end))
    if len(indices) != int(target_length):
        raise AssertionError("axis stretch produced the wrong length")
    return indices


def _stretch_pattern(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    row_idx = _axis_indices(pattern, 0, int(shape[0]))
    col_idx = _axis_indices(pattern, 1, int(shape[1]))
    return pattern[np.ix_(row_idx, col_idx)]


def _detect_candidate_objects(grid: np.ndarray, bg: int) -> list[dict]:
    candidates: list[dict] = []
    seen_keys: set[tuple[int, int, int, int, int]] = set()
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for comp in _find_components_of_color(grid, color):
            r0, c0, r1, c1 = _bbox_from_coords(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            patch = _crop(grid, (r0, c0, r1, c1))
            if not (
                np.all(patch[0, :] == color)
                and np.all(patch[-1, :] == color)
                and np.all(patch[:, 0] == color)
                and np.all(patch[:, -1] == color)
            ):
                continue
            key = (r0, c0, r1, c1, color)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(
                {
                    "outer": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "patch": patch,
                    "shape": patch.shape,
                    "solid": bool(np.all(patch == int(color))),
                }
            )
    return candidates


def _select_object_group(candidates: list[dict]) -> tuple[int, list[dict]]:
    by_color: dict[int, list[dict]] = {}
    for obj in candidates:
        by_color.setdefault(int(obj["outer"]), []).append(obj)

    viable: list[tuple[int, list[dict], tuple[int, int]]] = []
    for color, objs in by_color.items():
        solids = [obj for obj in objs if obj["solid"]]
        exemplars = [obj for obj in objs if not obj["solid"]]
        if len(solids) < 1 or len(exemplars) < 2:
            continue
        shapes = Counter(obj["shape"] for obj in exemplars)
        exemplar_shape, count = shapes.most_common(1)[0]
        if count < 2:
            continue
        matched = [obj for obj in exemplars if obj["shape"] == exemplar_shape]
        if len(matched) < 2:
            continue
        viable.append((color, solids + matched, (len(solids + matched), matched[0]["shape"][0] * matched[0]["shape"][1])))

    if not viable:
        raise AssertionError("failed to identify the outer-color object family")
    viable.sort(key=lambda item: item[2], reverse=True)
    color, objs, _score = viable[0]
    return int(color), objs


def _reconstruct_exemplar(exemplars: list[np.ndarray], bg: int) -> np.ndarray:
    shape = exemplars[0].shape
    full = np.full(shape, int(bg), dtype=int)
    for r in range(shape[0]):
        for c in range(shape[1]):
            vals = {int(patch[r, c]) for patch in exemplars if int(patch[r, c]) != int(bg)}
            if not vals:
                raise AssertionError("exemplar copies must complete each other")
            if len(vals) != 1:
                raise AssertionError("exemplar copies disagree on a visible cell")
            full[r, c] = vals.pop()
    if np.any(full == int(bg)):
        raise AssertionError("reconstructed exemplar must be fully colored")
    return full


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the damaged exemplar copies by union, then stretch the recovered pattern into each solid target rectangle.

    Input structure:
    - A black background contains several separated rectangles whose outer border uses one shared color.
    - Two or more same-sized multicolor rectangles are damaged copies of one exemplar, with black holes punched out of different interior parts.
    - The remaining rectangles are solid blocks of the shared outer color and have sizes different from the exemplar.

    Rule:
    - Detect the rectangle family by its shared outer-border color and split it into damaged exemplars versus solid targets.
    - Reconstruct the full exemplar by taking the cellwise union of all damaged copies in exemplar-relative coordinates, then refill every exemplar location with that complete pattern.
    - Run-length encode identical rows and identical columns of the complete exemplar; keep boundary runs fixed and resize the unique interior stretch run in each axis to fit each target rectangle.
    - Replace every solid target block in place with the stretched exemplar.

    Key insight:
    - No single exemplar copy is sufficient: the template only becomes usable after cross-copy union completion, and size changes happen by stretching the interior row/column runs rather than by geometric scaling.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    candidates = _detect_candidate_objects(g, bg)
    outer, family = _select_object_group(candidates)
    solids = [obj for obj in family if obj["solid"]]
    exemplars = [obj for obj in family if not obj["solid"]]
    if len(solids) < 1 or len(exemplars) < 2:
        raise AssertionError("expected at least one target and two exemplar copies")

    exemplar_shapes = Counter(obj["shape"] for obj in exemplars)
    exemplar_shape, _count = exemplar_shapes.most_common(1)[0]
    exemplar_objs = [obj for obj in exemplars if obj["shape"] == exemplar_shape]
    if len(exemplar_objs) < 2:
        raise AssertionError("expected multiple damaged copies of one exemplar size")
    for patch in (obj["patch"] for obj in exemplar_objs):
        if np.all(patch == int(outer)):
            raise AssertionError("damaged exemplar cannot be a solid target")

    full = _reconstruct_exemplar([obj["patch"] for obj in exemplar_objs], bg)

    out = g.copy()
    for obj in exemplar_objs:
        r0, c0, r1, c1 = obj["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = full
    for obj in solids:
        r0, c0, r1, c1 = obj["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = _stretch_pattern(full, obj["shape"])
    return out


def _choose_distinct(colors: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if count <= len(colors):
        picked = random.sample(colors, count)
    else:
        picked = []
        while len(picked) < count:
            pool = colors.copy()
            random.shuffle(pool)
            for color in pool:
                if not picked or color != picked[-1]:
                    picked.append(color)
                if len(picked) == count:
                    break
    for i in range(1, len(picked)):
        if picked[i] == picked[i - 1]:
            for j, color in enumerate(colors):
                if color != picked[i - 1]:
                    picked[i] = color
                    break
    return [int(x) for x in picked]


def _build_exemplar() -> tuple[np.ndarray, int]:
    palette = random.sample(PALETTE, 5)
    outer = int(palette[0])
    non_outer = [int(c) for c in palette[1:]]
    base = int(non_outer[0])
    center = int(non_outer[1])

    row_count = random.choice(ROW_RUN_COUNTS)
    col_count = random.choice(COL_RUN_COUNTS)
    stretch_r = random.randrange(1, row_count - 1)
    stretch_c = random.randrange(1, col_count - 1)

    row_widths = [random.choice(FIXED_RUN_WIDTHS) for _ in range(row_count)]
    col_widths = [random.choice(FIXED_RUN_WIDTHS) for _ in range(col_count)]
    row_widths[stretch_r] = random.choice(STRETCH_RUN_WIDTHS)
    col_widths[stretch_c] = random.choice(STRETCH_RUN_WIDTHS)

    seed = np.full((row_count, col_count), int(outer), dtype=int)
    for r in range(1, row_count - 1):
        for c in range(1, col_count - 1):
            seed[r, c] = int(base)

    other_rows = [r for r in range(1, row_count - 1) if r != stretch_r]
    other_cols = [c for c in range(1, col_count - 1) if c != stretch_c]
    row_colors = _choose_distinct(non_outer, len(other_rows))
    col_colors = _choose_distinct(non_outer, len(other_cols))

    for r, color in zip(other_rows, row_colors):
        seed[r, stretch_c] = int(color)
    for c, color in zip(other_cols, col_colors):
        seed[stretch_r, c] = int(color)
    seed[stretch_r, stretch_c] = int(center)

    exemplar = np.repeat(np.repeat(seed, row_widths, axis=0), col_widths, axis=1)
    if exemplar.shape[0] > 14 or exemplar.shape[1] > 16:
        raise AssertionError("exemplar unexpectedly large")
    if _stretch_run_index(_line_runs(exemplar, 0)) != stretch_r:
        raise AssertionError("row stretch run is ambiguous")
    if _stretch_run_index(_line_runs(exemplar, 1)) != stretch_c:
        raise AssertionError("col stretch run is ambiguous")
    if np.any(exemplar[0, :] != outer) or np.any(exemplar[-1, :] != outer):
        raise AssertionError("top/bottom border must stay in the outer color")
    if np.any(exemplar[:, 0] != outer) or np.any(exemplar[:, -1] != outer):
        raise AssertionError("left/right border must stay in the outer color")
    return exemplar, outer


def _damage_copies(exemplar: np.ndarray, copy_count: int) -> list[np.ndarray]:
    h, w = exemplar.shape
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    if len(interior) < copy_count:
        raise AssertionError("not enough interior cells to damage")

    for _ in range(80):
        anchors = random.sample(interior, copy_count)
        parts: list[list[tuple[int, int]]] = [[] for _ in range(copy_count)]
        for r, c in interior:
            distances = [abs(r - ar) + abs(c - ac) for ar, ac in anchors]
            best = min(distances)
            choices = [i for i, d in enumerate(distances) if d == best]
            owner = random.choice(choices)
            parts[owner].append((r, c))
        if all(parts):
            break
    else:
        raise AssertionError("failed to partition interior for damage masks")

    copies: list[np.ndarray] = []
    for cells in parts:
        patch = exemplar.copy()
        for r, c in cells:
            patch[r, c] = BG
        copies.append(patch)
    return copies


def _target_shapes(exemplar: np.ndarray, n_targets: int) -> list[tuple[int, int]]:
    row_runs = _line_runs(exemplar, 0)
    col_runs = _line_runs(exemplar, 1)
    row_stretch = _stretch_run_index(row_runs)
    col_stretch = _stretch_run_index(col_runs)
    fixed_h = sum(end - start for i, (start, end, _line) in enumerate(row_runs) if i != row_stretch)
    fixed_w = sum(end - start for i, (start, end, _line) in enumerate(col_runs) if i != col_stretch)
    original = exemplar.shape

    shapes: set[tuple[int, int]] = set()
    for _ in range(120):
        h = fixed_h + random.randint(1, 8)
        w = fixed_w + random.randint(1, 10)
        shape = (int(h), int(w))
        if shape == original or max(shape) > 18:
            continue
        shapes.add(shape)
        if len(shapes) == n_targets:
            return list(shapes)
    raise AssertionError("failed to sample distinct target shapes")


def _place_objects(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(pairs, key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True)
    for _ in range(120):
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)
        inp = np.full((h, w), BG, dtype=int)
        out = np.full((h, w), BG, dtype=int)
        occ = np.zeros((h, w), dtype=bool)
        ok = True

        for in_patch, out_patch in ordered:
            ph, pw = in_patch.shape
            positions: list[tuple[int, int]] = []
            for r0 in range(OBJECT_BORDER, h - ph - OBJECT_BORDER + 1):
                for c0 in range(OBJECT_BORDER, w - pw - OBJECT_BORDER + 1):
                    rr0 = max(0, r0 - OBJECT_PADDING)
                    cc0 = max(0, c0 - OBJECT_PADDING)
                    rr1 = min(h, r0 + ph + OBJECT_PADDING)
                    cc1 = min(w, c0 + pw + OBJECT_PADDING)
                    if occ[rr0:rr1, cc0:cc1].any():
                        continue
                    positions.append((r0, c0))
            if not positions:
                ok = False
                break

            r0, c0 = random.choice(positions)
            _paste(inp, in_patch, r0, c0, transparent=BG)
            _paste(out, out_patch, r0, c0, transparent=BG)
            occ[r0 : r0 + ph, c0 : c0 + pw] = True

        if ok and _get_bg(inp) == BG:
            return inp, out
    raise AssertionError("failed to place all objects")


def _gen_attempt() -> np.ndarray:
    exemplar, outer = _build_exemplar()
    copy_count = random.choice(COPY_COUNTS)
    target_count = random.choice(TARGET_COUNT_OPTIONS)
    damaged = _damage_copies(exemplar, copy_count)
    targets = _target_shapes(exemplar, target_count)

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for patch in damaged:
        pairs.append((patch, exemplar))
    for shape in targets:
        solid = np.full(shape, int(outer), dtype=int)
        pairs.append((solid, _stretch_pattern(exemplar, shape)))

    inp, expected = _place_objects(pairs)
    solved = main(inp)
    if not np.array_equal(solved, expected):
        raise AssertionError("generator produced an unsolved sample")
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _gen_attempt()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input for puzzle 5")
