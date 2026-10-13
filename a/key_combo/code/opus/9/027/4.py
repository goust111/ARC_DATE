from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
FRAME = 4
HALO = 7
INPUT_COLORS = (1, 2, 3, 5, 6, 8, 9)


@dataclass(frozen=True)
class Rect:
    bbox: tuple[int, int, int, int]
    patch: np.ndarray

    @property
    def area(self) -> int:
        return int(self.patch.shape[0] * self.patch.shape[1])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _perimeter_mask(h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=bool)
    out[0, :] = True
    out[-1, :] = True
    out[:, 0] = True
    out[:, -1] = True
    return out


def _extract_rectangles(grid: np.ndarray) -> list[Rect]:
    rects: list[Rect] = []
    for coords in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(coords)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        local = np.zeros((h, w), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if np.array_equal(local, _perimeter_mask(h, w)):
            rects.append(Rect((r0, c0, r1, c1), grid[r0 : r1 + 1, c0 : c1 + 1].copy()))
    assert len(rects) >= 2, "expected one exemplar and at least one target rectangle"
    return rects


def _line_runs(pattern: np.ndarray, axis: int) -> list[tuple[int, int, tuple[int, ...]]]:
    lines = pattern if axis == 0 else pattern.T
    runs = []
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
    max_len = max(lengths[i] for i in candidates)
    candidates = [i for i in candidates if lengths[i] == max_len]
    assert len(candidates) == 1, "stretchable run must be unique"
    return candidates[0]


def _axis_indices(pattern: np.ndarray, axis: int, target_length: int) -> list[int]:
    runs = _line_runs(pattern, axis)
    stretch_i = _stretch_run_index(runs)
    fixed = sum(end - start for i, (start, end, _line) in enumerate(runs) if i != stretch_i)
    stretch_len = int(target_length) - fixed
    assert stretch_len > 0, "target rectangle is too small"
    indices: list[int] = []
    for i, (start, end, _line) in enumerate(runs):
        if i == stretch_i:
            indices.extend([start] * stretch_len)
        else:
            indices.extend(range(start, end))
    assert len(indices) == int(target_length)
    return indices


def _stretch_patch(exemplar: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    inner = exemplar[1:-1, 1:-1]
    row_idx = _axis_indices(inner, 0, h - 2)
    col_idx = _axis_indices(inner, 1, w - 2)
    out = np.full((h, w), int(exemplar[0, 0]), dtype=int)
    out[1:-1, 1:-1] = inner[np.ix_(row_idx, col_idx)]
    return out


def _apply_halos(base: np.ndarray, defects: list[tuple[int, int]]) -> np.ndarray:
    out = base.copy()
    defect_set = set(defects)
    h, w = out.shape
    for r, c in defects:
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in defect_set and out[nr, nc] == BG:
                out[nr, nc] = HALO
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect stretched-template defects, repair them, and add orange halos around the defect sites.

    Input structure:
    - Black background with several yellow-framed axis-aligned rectangles.
    - The smallest framed rectangle is a multicolor exemplar whose interior contains repeated row/column bands, including one stretchable interior run on each axis.
    - Larger framed rectangles contain stretched copies of that exemplar pattern, except that some interior pixels are wrong.

    Rule:
    - Identify the smallest yellow-framed rectangle as the exemplar.
    - Stretch the exemplar to each larger rectangle by keeping all non-stretch row/column runs fixed and expanding only the interior run whose representative line has the most colors.
    - Compare each target rectangle to its expected stretched content; every mismatch is a defect and is corrected to the expected color.
    - After repairing the defects, paint orange on every black pixel in each defect's 8-neighborhood, without overwriting the defect pixel itself.

    Key insight:
    - A wrong pixel is not detectable from its color alone; it is defined only by disagreement with the uniquely determined stretched exemplar.
    """
    g = np.asarray(grid, dtype=int)
    rects = _extract_rectangles(g)
    min_area = min(rect.area for rect in rects)
    exemplar_rects = [rect for rect in rects if rect.area == min_area]
    assert len(exemplar_rects) == 1, "expected a unique smallest exemplar rectangle"
    exemplar = exemplar_rects[0]

    out = g.copy()
    defects: list[tuple[int, int]] = []
    for rect in rects:
        if rect.bbox == exemplar.bbox:
            continue
        expected = _stretch_patch(exemplar.patch, rect.patch.shape)
        mismatch = rect.patch != expected
        r0, c0, r1, c1 = rect.bbox
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        sub[mismatch] = expected[mismatch]
        for rr, cc in np.argwhere(mismatch):
            defects.append((int(r0 + rr), int(c0 + cc)))

    return _apply_halos(out, defects)


def _seed_pattern(colors: list[int]) -> np.ndarray:
    top, left, center, right, bottom = map(int, colors)
    return np.array(
        [
            [BG, top, BG],
            [left, center, right],
            [BG, bottom, BG],
        ],
        dtype=int,
    )


def _run_expanded(seed: np.ndarray, row_runs: list[int], col_runs: list[int]) -> np.ndarray:
    row_idx: list[int] = []
    col_idx: list[int] = []
    for i, width in enumerate(row_runs):
        row_idx.extend([i] * int(width))
    for i, width in enumerate(col_runs):
        col_idx.extend([i] * int(width))
    return seed[np.ix_(row_idx, col_idx)]


def _make_exemplar(seed_colors: list[int], row_runs: list[int], col_runs: list[int]) -> np.ndarray:
    inner = _run_expanded(_seed_pattern(seed_colors), row_runs, col_runs)
    patch = np.full((inner.shape[0] + 2, inner.shape[1] + 2), FRAME, dtype=int)
    patch[1:-1, 1:-1] = inner
    return patch


def _candidate_defects(expected: np.ndarray) -> list[tuple[int, int]]:
    inner = expected[1:-1, 1:-1]
    candidates = []
    for rr in range(inner.shape[0]):
        for cc in range(inner.shape[1]):
            if inner[rr, cc] == BG:
                continue
            has_black_neighbor = False
            for nr, nc in _neighbors8(rr, cc):
                if 0 <= nr < inner.shape[0] and 0 <= nc < inner.shape[1] and inner[nr, nc] == BG:
                    has_black_neighbor = True
                    break
            if has_black_neighbor:
                candidates.append((rr + 1, cc + 1))
    return candidates


def _choose_wrong_color(expected_color: int, palette: list[int]) -> int:
    options = [c for c in palette if c not in (expected_color, FRAME, HALO, BG)]
    return int(random.choice(options))


def _too_close(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _inject_defects(expected: np.ndarray, palette: list[int], defect_count: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    actual = expected.copy()
    candidates = _candidate_defects(expected)
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for pos in candidates:
        if any(_too_close(pos, other) for other in chosen):
            continue
        chosen.append(pos)
        if len(chosen) == defect_count:
            break
    assert len(chosen) == defect_count, "not enough valid defect locations"
    for rr, cc in chosen:
        actual[rr, cc] = _choose_wrong_color(int(expected[rr, cc]), palette)
    return actual, chosen


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _can_place(occupied: np.ndarray, shape: tuple[int, int], r0: int, c0: int, pad: int = 1) -> bool:
    h, w = shape
    r1, c1 = r0 + h - 1, c0 + w - 1
    if r0 < 0 or c0 < 0 or r1 >= occupied.shape[0] or c1 >= occupied.shape[1]:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0] - 1, r1 + pad)
    cc1 = min(occupied.shape[1] - 1, c1 + pad)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _search_layout(shapes: list[tuple[int, int]], grid_shape: tuple[int, int], anchor: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    occupied = np.zeros((h, w), dtype=bool)
    placements = [anchor]
    ah, aw = shapes[0]
    occupied[anchor[0] : anchor[0] + ah, anchor[1] : anchor[1] + aw] = True

    rest = list(enumerate(shapes[1:], start=1))
    rest.sort(key=lambda item: item[1][0] * item[1][1], reverse=True)
    ordered_positions: dict[int, tuple[int, int]] = {}

    def rec(i: int) -> bool:
        if i == len(rest):
            return True
        idx, (ph, pw) = rest[i]
        candidates = [(r, c) for r in range(1, h - ph) for c in range(1, w - pw)]
        random.shuffle(candidates)
        for r0, c0 in candidates:
            if not _can_place(occupied, (ph, pw), r0, c0, pad=1):
                continue
            occupied[r0 : r0 + ph, c0 : c0 + pw] = True
            ordered_positions[idx] = (r0, c0)
            if rec(i + 1):
                return True
            occupied[r0 : r0 + ph, c0 : c0 + pw] = False
            del ordered_positions[idx]
        return False

    if not rec(0):
        return None
    for i in range(1, len(shapes)):
        placements.append(ordered_positions[i])
    return placements


def _make_output(clean_grid: np.ndarray, defect_positions: list[tuple[int, int]]) -> np.ndarray:
    return _apply_halos(clean_grid, defect_positions)


def _layout_shapes(shapes: list[tuple[int, int]]) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    gap = 1
    margin = 1
    ex_h, ex_w = shapes[0]
    t0_h, t0_w = shapes[1]
    t1_h, t1_w = shapes[2]
    t2_h, t2_w = shapes[3]
    row1_h = max(ex_h, t0_h)
    row2_h = max(t1_h, t2_h)
    row1_w = margin + ex_w + gap + t0_w + margin
    row2_w = margin + t1_w + gap + t2_w + margin

    if len(shapes) == 5:
        t3_h, t3_w = shapes[4]
        row3_h = t3_h
        row3_w = margin + t3_w + margin
        grid_h = margin + row1_h + gap + row2_h + gap + row3_h + margin
        grid_w = max(row1_w, row2_w, row3_w)
    else:
        row3_h = 0
        grid_h = margin + row1_h + gap + row2_h + margin
        grid_w = max(row1_w, row2_w)

    grid_h += random.randint(0, 2)
    grid_w += random.randint(0, 2)
    if grid_h > 30 or grid_w > 30:
        return None

    placements: list[tuple[int, int]] = []
    ex_r = margin + random.randint(0, row1_h - ex_h)
    ex_c = margin
    placements.append((ex_r, ex_c))

    t0_r = margin + random.randint(0, row1_h - t0_h)
    t0_c = ex_c + ex_w + gap
    placements.append((t0_r, t0_c))

    row2_top = margin + row1_h + gap
    t1_r = row2_top + random.randint(0, row2_h - t1_h)
    t1_c = margin
    placements.append((t1_r, t1_c))

    t2_r = row2_top + random.randint(0, row2_h - t2_h)
    t2_c = t1_c + t1_w + gap
    placements.append((t2_r, t2_c))

    if len(shapes) == 5:
        row3_top = row2_top + row2_h + gap
        t3_h, t3_w = shapes[4]
        t3_r = row3_top
        t3_c = margin + random.randint(0, grid_w - 2 - t3_w)
        placements.append((t3_r, t3_c))

    return (grid_h, grid_w), placements


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(400):
        palette = random.sample(list(INPUT_COLORS), 5)
        row_runs = [random.randint(1, 2), random.randint(1, 2), random.randint(1, 2)]
        col_runs = [random.randint(1, 2), random.randint(1, 2), random.randint(1, 2)]
        exemplar = _make_exemplar(palette, row_runs, col_runs)

        target_count = random.randint(3, 4)
        target_specs: list[tuple[int, int, int, int]] = []
        fixed_h = row_runs[0] + row_runs[2]
        fixed_w = col_runs[0] + col_runs[2]
        for _idx in range(target_count):
            mid_h = random.randint(row_runs[1] + 1, row_runs[1] + 2)
            mid_w = random.randint(col_runs[1] + 1, col_runs[1] + 4)
            target_h = fixed_h + mid_h + 2
            target_w = fixed_w + mid_w + 2
            target_specs.append((target_h, target_w, mid_h, mid_w))

        shapes = [tuple(exemplar.shape)] + [(h, w) for h, w, _mh, _mw in target_specs]
        layout = _layout_shapes(shapes)
        if layout is None:
            continue
        (grid_h, grid_w), placements = layout

        target_expected = [_stretch_patch(exemplar, (h, w)) for h, w, _mh, _mw in target_specs]
        defect_target_count = random.randint(1, min(2, target_count))
        defect_target_ids = set(random.sample(range(target_count), defect_target_count))

        clean_grid = np.zeros((grid_h, grid_w), dtype=int)
        input_grid = np.zeros((grid_h, grid_w), dtype=int)
        _paste(clean_grid, exemplar, placements[0][0], placements[0][1])
        _paste(input_grid, exemplar, placements[0][0], placements[0][1])

        all_defects: list[tuple[int, int]] = []
        clean_targets = 0
        defective_targets = 0

        for i, expected in enumerate(target_expected):
            r0, c0 = placements[i + 1]
            _paste(clean_grid, expected, r0, c0)
            if i in defect_target_ids:
                defect_count = random.randint(1, 2)
                try:
                    actual, local_defects = _inject_defects(expected, palette, defect_count)
                except AssertionError:
                    break
                defective_targets += 1
                for rr, cc in local_defects:
                    all_defects.append((r0 + rr, c0 + cc))
            else:
                actual = expected
                clean_targets += 1
            _paste(input_grid, actual, r0, c0)
        else:
            if clean_targets == 0 or defective_targets == 0:
                continue
            output_grid = _make_output(clean_grid, all_defects)
            if not np.any(output_grid == HALO):
                continue

            rects = _extract_rectangles(input_grid)
            areas = sorted(rect.area for rect in rects)
            if areas[0] == areas[1]:
                continue
            if np.array_equal(input_grid, output_grid):
                continue
            if not np.array_equal(main(input_grid), output_grid):
                continue
            return input_grid, output_grid
    raise RuntimeError("failed to generate a valid ARC sample")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
