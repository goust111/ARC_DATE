from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random
from typing import Iterable

import numpy as np

try:
    from arc_utils import bbox, get_bg
except Exception:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_shared", "/home/lixinyu/snap/new/solvers/arc_utils.py"
    ).load_module()
    bbox = _ARC_UTILS.bbox
    get_bg = _ARC_UTILS.get_bg


BG = 0
BLUE = 1
RED = 2

ORTH_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

GRID_SIDE_MIN = 20
GRID_SIDE_MAX = 24
MAIN_ROW_GAP = 3
MAIN_COL_GAP = 5
GEN_ATTEMPTS = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _segments_1d(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    segs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for x in indices[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
            continue
        segs.append((int(start), int(prev)))
        start = prev = x
    segs.append((int(start), int(prev)))
    return segs


def _find_diagonal_noise(mask: np.ndarray) -> np.ndarray:
    h, w = map(int, mask.shape)
    out = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            if any(
                _in_bounds(mask.shape, r + dr, c + dc) and bool(mask[r + dr, c + dc])
                for dr, dc in ORTH_DIRS
            ):
                continue
            if any(
                _in_bounds(mask.shape, r + dr, c + dc) and bool(mask[r + dr, c + dc])
                for dr, dc in DIAG_DIRS
            ):
                out[r, c] = True
    return out


def _fill_gaps(erased_mask: np.ndarray, removed_noise: np.ndarray) -> np.ndarray:
    h, w = map(int, erased_mask.shape)
    out = erased_mask.copy()

    for r in range(h):
        segs = _segments_1d(np.where(erased_mask[r])[0])
        for (a0, a1), (b0, _b1) in zip(segs, segs[1:]):
            gap0, gap1 = int(a1 + 1), int(b0 - 1)
            if gap0 > gap1:
                continue
            gap_len = int(gap1 - gap0 + 1)
            rr0 = max(0, int(r - 1))
            rr1 = min(int(h - 1), int(r + 1))
            nearby = int(np.count_nonzero(removed_noise[rr0 : rr1 + 1, gap0 : gap1 + 1]))
            if gap_len <= nearby:
                out[r, gap0 : gap1 + 1] = True

    for c in range(w):
        segs = _segments_1d(np.where(erased_mask[:, c])[0])
        for (a0, a1), (b0, _b1) in zip(segs, segs[1:]):
            gap0, gap1 = int(a1 + 1), int(b0 - 1)
            if gap0 > gap1:
                continue
            gap_len = int(gap1 - gap0 + 1)
            cc0 = max(0, int(c - 1))
            cc1 = min(int(w - 1), int(c + 1))
            nearby = int(np.count_nonzero(removed_noise[gap0 : gap1 + 1, cc0 : cc1 + 1]))
            if gap_len <= nearby:
                out[gap0 : gap1 + 1, c] = True

    return out


def _run_length(mask: np.ndarray, r: int, c: int, *, axis: int) -> int:
    if not bool(mask[r, c]):
        return 0
    h, w = map(int, mask.shape)
    total = 1
    if axis == 0:
        rr = int(r - 1)
        while rr >= 0 and bool(mask[rr, c]):
            total += 1
            rr -= 1
        rr = int(r + 1)
        while rr < h and bool(mask[rr, c]):
            total += 1
            rr += 1
    else:
        cc = int(c - 1)
        while cc >= 0 and bool(mask[r, cc]):
            total += 1
            cc -= 1
        cc = int(c + 1)
        while cc < w and bool(mask[r, cc]):
            total += 1
            cc += 1
    return int(total)


def _intersections(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    pts: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            if _run_length(mask, r, c, axis=1) >= 3 and _run_length(mask, r, c, axis=0) >= 3:
                pts.append((int(r), int(c)))
    return pts


def _extract_template(grid: np.ndarray) -> np.ndarray:
    region = np.asarray(grid[-4:, -4:], dtype=int)
    red = region == RED
    b = bbox(red)
    assert b is not None, "expected a red template in the lower-right 4x4 zone"
    r0, c0, r1, c1 = map(int, b)
    patch = region[r0 : r1 + 1, c0 : c1 + 1]
    assert patch.shape == (3, 3), f"template bbox must be 3x3, got {patch.shape}"
    assert int(patch[1, 1]) == RED, "template center must be red"
    return patch


def _stamp_red(out: np.ndarray, template: np.ndarray, center: tuple[int, int]) -> None:
    h, w = map(int, out.shape)
    cr, cc = map(int, center)
    for tr in range(3):
        for tc in range(3):
            if int(template[tr, tc]) != RED:
                continue
            rr = int(cr + tr - 1)
            cc2 = int(cc + tc - 1)
            if 0 <= rr < h and 0 <= cc2 < w:
                out[rr, cc2] = RED


def _solve_blue(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blue = np.asarray(grid == BLUE, dtype=bool)
    noise = _find_diagonal_noise(blue)
    erased = blue & (~noise)
    completed = _fill_gaps(erased, noise)
    return noise, erased, completed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove diagonal blue noise, complete hinted orthogonal gaps, then stamp the lower-right template on every emergent line crossing.

    Input structure
    - The background is black, blue pixels form one-pixel-wide horizontal and vertical line segments, and some blue pixels are diagonal-only staircase noise near broken gaps.
    - The lower-right 4x4 area contains a red template whose non-black pixels occupy a 3x3 bbox and whose center cell is red.

    Rule
    - Identify blue noise pixels as blue cells with no orthogonal blue neighbor and at least one diagonal blue neighbor, then erase them.
    - On the erased grid, inspect each row and each column separately; whenever two blue segments are separated by a gap whose length is no larger than the number of erased noise pixels in the corresponding local 3-cell-thick band, fill that gap blue.
    - Find every cell that lies on both a continuous horizontal blue run of length at least 3 and a continuous vertical blue run of length at least 3.
    - Copy the 3x3 red template onto the output with its center aligned to each intersection; red cells overwrite existing content while black template cells stay transparent.

    Key insight
    The crossing locations are not visible in the raw input: diagonal-only noise first reveals where lines are truly broken, and only the noise-count-limited repairs create the final stamp centers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    _ = int(get_bg(g))

    template = _extract_template(g)
    _noise, _erased, completed = _solve_blue(g)

    out = g.copy()
    out[out == BLUE] = BG
    out[completed] = BLUE

    for pt in sorted(_intersections(completed)):
        _stamp_red(out, template, pt)
    return out


def _pick_spaced(rng: random.Random, low: int, high: int, count: int, gap: int) -> list[int] | None:
    values = list(range(int(low), int(high) + 1))
    rng.shuffle(values)
    picked: list[int] = []
    for v in values:
        if all(abs(int(v) - int(p)) >= int(gap) for p in picked):
            picked.append(int(v))
            if len(picked) == count:
                return sorted(picked)
    return None


def _transpose_rels(rels: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(int(dc), int(dr)) for dr, dc in rels]


def _noise_rel_candidates(orientation: str, gap_len: int, count: int) -> list[list[tuple[int, int]]]:
    assert 1 <= count <= gap_len <= 3
    horizontal: list[list[tuple[int, int]]] = []
    if count == gap_len:
        if gap_len == 1:
            horizontal = [[(1, 0)], [(-1, 0)]]
        elif gap_len == 2:
            horizontal = [[(1, 0), (-1, 1)], [(-1, 0), (1, 1)]]
        else:
            horizontal = [
                [(-1, 0), (0, 1), (1, 2)],
                [(1, 0), (0, 1), (-1, 2)],
            ]
    elif count == 1:
        horizontal = [[(1, 0)], [(-1, 0)], [(1, gap_len - 1)], [(-1, gap_len - 1)]]
    else:
        horizontal = [[(1, 0), (-1, gap_len - 1)], [(-1, 0), (1, gap_len - 1)]]

    if orientation == "h":
        return horizontal
    assert orientation == "v"
    return [_transpose_rels(rels) for rels in horizontal]


def _gap_cells(spec: dict) -> list[tuple[int, int]]:
    if spec["orientation"] == "h":
        return [(int(spec["fixed"]), int(spec["gap_start"]) + i) for i in range(int(spec["gap_len"]))]
    return [(int(spec["gap_start"]) + i, int(spec["fixed"])) for i in range(int(spec["gap_len"]))]


def _apply_line(mask: np.ndarray, spec: dict, *, broken: bool) -> None:
    if spec["orientation"] == "h":
        r = int(spec["fixed"])
        if not broken:
            mask[r, int(spec["start"]) : int(spec["end"]) + 1] = True
            return
        gap_cells = {tuple(cell) for cell in _gap_cells(spec)}
        for c in range(int(spec["start"]), int(spec["end"]) + 1):
            if (r, int(c)) not in gap_cells:
                mask[r, int(c)] = True
        return

    c = int(spec["fixed"])
    if not broken:
        mask[int(spec["start"]) : int(spec["end"]) + 1, c] = True
        return
    gap_cells = {tuple(cell) for cell in _gap_cells(spec)}
    for r in range(int(spec["start"]), int(spec["end"]) + 1):
        if (int(r), c) not in gap_cells:
            mask[int(r), c] = True


def _place_gap_noise(
    input_blue: np.ndarray,
    occupied_forbidden: np.ndarray,
    spec: dict,
    count: int,
) -> list[tuple[int, int]] | None:
    cells = _gap_cells(spec)
    anchor_r, anchor_c = map(int, cells[0])
    if spec["orientation"] == "h":
        base_r = anchor_r
        base_c = anchor_c
    else:
        base_r = anchor_r
        base_c = anchor_c

    for rels in _noise_rel_candidates(spec["orientation"], int(spec["gap_len"]), int(count)):
        coords = [(int(base_r + dr), int(base_c + dc)) for dr, dc in rels]
        if len(set(coords)) != len(coords):
            continue
        if any(not _in_bounds(input_blue.shape, r, c) for r, c in coords):
            continue
        if any(bool(occupied_forbidden[r, c]) or bool(input_blue[r, c]) for r, c in coords):
            continue
        return coords
    return None


def _random_template(rng: random.Random) -> np.ndarray:
    while True:
        patch = np.zeros((3, 3), dtype=int)
        cells = {(1, 1)}
        target = rng.randint(4, 6)
        frontier = {(1, 0), (0, 1), (1, 2), (2, 1)}
        while len(cells) < target and frontier:
            cell = rng.choice(sorted(frontier))
            frontier.remove(cell)
            cells.add(cell)
            r, c = cell
            for dr, dc in ORTH_DIRS:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < 3 and 0 <= cc < 3 and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if len(cells) != target:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        if max(rs) - min(rs) != 2 or max(cs) - min(cs) != 2:
            continue
        for r, c in cells:
            patch[r, c] = RED
        return patch


def _paint_template_zone(grid: np.ndarray, template: np.ndarray, offset_r: int, offset_c: int) -> None:
    zone_r0 = int(grid.shape[0] - 4)
    zone_c0 = int(grid.shape[1] - 4)
    grid[zone_r0 + offset_r : zone_r0 + offset_r + 3, zone_c0 + offset_c : zone_c0 + offset_c + 3] = template


def _add_stamps(base: np.ndarray, template: np.ndarray, intersections: list[tuple[int, int]]) -> np.ndarray:
    out = base.copy()
    for pt in sorted(intersections):
        _stamp_red(out, template, pt)
    return out


def _build_example(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h = rng.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    w = rng.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    work_r_max = int(h - 5)
    work_c_max = int(w - 5)

    n_rows = 3 if h >= 22 and rng.random() < 0.45 else 2
    row_gap = 3
    row_span = int((n_rows - 1) * row_gap)
    row0_low = 6
    row0_high = int(work_r_max - 5 - row_span)
    if row0_low > row0_high:
        raise ValueError("failed to place main rows/cols")
    row0 = rng.randint(row0_low, row0_high)
    rows = [int(row0 + i * row_gap) for i in range(n_rows)]

    n_cols = 3 if w >= 24 and rng.random() < 0.45 else 2
    col_gap = 5
    col_span = int((n_cols - 1) * col_gap)
    col0_low = 4
    col0_high = int(work_c_max - 4 - col_span)
    if col0_low > col0_high:
        raise ValueError("failed to place main rows/cols")
    col0 = rng.randint(col0_low, col0_high)
    cols = [int(col0 + i * col_gap) for i in range(n_cols)]

    expected_blue = np.zeros((h, w), dtype=bool)
    main_specs: list[dict] = []

    for r in rows:
        side = rng.choice(["left", "right"])
        if side == "left":
            gap_len = 1
            start = int(cols[0])
            gap_start = int(cols[0] + 2)
            min_end = max(int(cols[-1]), int(gap_start + gap_len + 2))
            if min_end > work_c_max:
                raise ValueError("horizontal right extent too short")
            end = rng.randint(min_end, work_c_max)
        else:
            gap_len = 1
            end = int(cols[-1])
            gap_start = int(cols[-1] - 2)
            max_start = min(int(cols[0] - 1), int(gap_start - 2))
            if max_start < 1:
                raise ValueError("horizontal left extent too short")
            start = rng.randint(1, max_start)

        spec = {
            "orientation": "h",
            "fixed": int(r),
            "start": int(start),
            "end": int(end),
            "gap_start": int(gap_start),
            "gap_len": int(gap_len),
            "fillable": True,
        }
        _apply_line(expected_blue, spec, broken=False)
        main_specs.append(spec)

    for c in cols:
        start = rng.randint(1, int(rows[0] - 1))
        end = rng.randint(int(rows[-1] + 1), work_r_max)
        spec = {
            "orientation": "v",
            "fixed": int(c),
            "start": int(start),
            "end": int(end),
        }
        _apply_line(expected_blue, spec, broken=False)

    decoy_rows = [
        r
        for r in range(1, work_r_max + 1)
        if all(abs(int(r) - int(main_r)) >= 2 for main_r in rows)
    ]
    if not decoy_rows:
        raise ValueError("no decoy row")
    decoy_r = int(rng.choice(decoy_rows))
    decoy_len = rng.randint(8, min(12, work_c_max - 1))
    decoy_start = rng.randint(1, work_c_max - decoy_len + 1)
    decoy_end = int(decoy_start + decoy_len - 1)
    decoy_gap_len = rng.choice([2, 3])
    decoy_gap_start = rng.randint(decoy_start + 2, decoy_end - decoy_gap_len - 1)
    decoy_spec = {
        "orientation": "h",
        "fixed": int(decoy_r),
        "start": int(decoy_start),
        "end": int(decoy_end),
        "gap_start": int(decoy_gap_start),
        "gap_len": int(decoy_gap_len),
        "fillable": False,
    }
    _apply_line(expected_blue, decoy_spec, broken=True)

    input_blue = expected_blue.copy()
    for spec in main_specs:
        for r, c in _gap_cells(spec):
            input_blue[r, c] = False

    forbidden = np.zeros((h, w), dtype=bool)
    forbidden[h - 4 :, w - 4 :] = True
    noise_jobs = [*[(spec, int(spec["gap_len"])) for spec in main_specs]]
    candidate_lists: list[list[tuple[tuple[int, int], ...]]] = []

    for spec, count in noise_jobs:
        cells = _gap_cells(spec)
        base_r, base_c = map(int, cells[0])
        options: list[tuple[tuple[int, int], ...]] = []
        seen_options: set[tuple[tuple[int, int], ...]] = set()
        for rels in _noise_rel_candidates(spec["orientation"], int(spec["gap_len"]), int(count)):
            coords = tuple(sorted((int(base_r + dr), int(base_c + dc)) for dr, dc in rels))
            if len(set(coords)) != len(coords):
                continue
            if any(not _in_bounds(input_blue.shape, r, c) for r, c in coords):
                continue
            if any(bool(forbidden[r, c]) or bool(input_blue[r, c]) for r, c in coords):
                continue
            if coords not in seen_options:
                seen_options.add(coords)
                options.append(coords)
        if not options:
            raise ValueError("failed to place diagonal hints")
        candidate_lists.append(options)

    order = sorted(range(len(candidate_lists)), key=lambda i: len(candidate_lists[i]))
    chosen_noise: list[tuple[int, int]] | None = None

    def backtrack(pos: int, used: set[tuple[int, int]]) -> bool:
        nonlocal chosen_noise
        if pos == len(order):
            trial = input_blue.copy()
            for r, c in used:
                trial[r, c] = True
            found_noise, erased, completed = _solve_blue(np.where(trial, BLUE, BG))
            if not np.array_equal(completed, expected_blue):
                return False
            if len(_intersections(erased)) >= len(_intersections(expected_blue)):
                return False
            for spec in main_specs:
                if not all(bool(completed[r, c]) for r, c in _gap_cells(spec)):
                    return False
                r = int(spec["fixed"])
                c0 = int(spec["gap_start"])
                rr0 = max(0, r - 1)
                rr1 = min(h - 1, r + 1)
                if int(np.count_nonzero(found_noise[rr0 : rr1 + 1, c0 : c0 + spec["gap_len"]])) < int(spec["gap_len"]):
                    return False
            if not all(not bool(completed[r, c]) for r, c in _gap_cells(decoy_spec)):
                return False
            chosen_noise = sorted(used)
            return True

        job_idx = order[pos]
        for coords in candidate_lists[job_idx]:
            coord_set = set(coords)
            if used & coord_set:
                continue
            if backtrack(pos + 1, used | coord_set):
                return True
        return False

    if not backtrack(0, set()):
        raise ValueError("failed to place diagonal hints")

    for r, c in chosen_noise:
        input_blue[r, c] = True

    template = _random_template(rng)
    template_offset_r = rng.randint(0, 1)
    template_offset_c = rng.randint(0, 1)
    inp = np.zeros((h, w), dtype=int)
    inp[input_blue] = BLUE
    _paint_template_zone(inp, template, template_offset_r, template_offset_c)

    base_out = np.zeros((h, w), dtype=int)
    base_out[expected_blue] = BLUE
    _paint_template_zone(base_out, template, template_offset_r, template_offset_c)
    exp = _add_stamps(base_out, template, _intersections(expected_blue))

    found_noise, erased, completed = _solve_blue(inp)
    assert np.array_equal(completed, expected_blue)
    assert len(_intersections(expected_blue)) >= 2
    assert len(_intersections(erased)) < len(_intersections(expected_blue))

    for spec in main_specs:
        assert all(not bool(erased[r, c]) for r, c in _gap_cells(spec))
        assert all(bool(completed[r, c]) for r, c in _gap_cells(spec))
        r = int(spec["fixed"])
        c0 = int(spec["gap_start"])
        rr0 = max(0, r - 1)
        rr1 = min(h - 1, r + 1)
        assert int(np.count_nonzero(found_noise[rr0 : rr1 + 1, c0 : c0 + spec["gap_len"]])) >= int(spec["gap_len"])

    assert all(not bool(completed[r, c]) for r, c in _gap_cells(decoy_spec))
    assert np.array_equal(main(inp), exp)
    return inp, exp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        try:
            inp, _exp = _build_example(rng)
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
