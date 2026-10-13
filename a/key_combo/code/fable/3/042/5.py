from __future__ import annotations

import random

import numpy as np


SEPARATOR = 5
MISSING = 6
CANVAS_COLORS = (0, 1, 2, 3, 4, 7, 8, 9)


Span = tuple[int, int]
Offset = tuple[int, int]


def _groups(indices: list[int]) -> list[Span]:
    assert indices, "expected separator lines"
    groups: list[Span] = []
    start = prev = int(indices[0])
    for idx0 in indices[1:]:
        idx = int(idx0)
        if idx == prev + 1:
            prev = idx
        else:
            groups.append((start, prev))
            start = prev = idx
    groups.append((start, prev))
    return groups


def _spans_between(line_groups: list[Span], limit: int) -> list[Span]:
    spans: list[Span] = []
    start = 0
    for line_start, line_end in line_groups:
        if start < line_start:
            spans.append((start, line_start - 1))
        start = line_end + 1
    if start < limit:
        spans.append((start, limit - 1))

    assert len(spans) >= 2, "expected at least two partitions on each axis"
    sizes = {end - start + 1 for start, end in spans}
    assert len(sizes) == 1, "all partitions on an axis must have equal size"
    return spans


def _partition_spans(g: np.ndarray) -> tuple[list[Span], list[Span]]:
    row_lines = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEPARATOR)]
    col_lines = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEPARATOR)]
    assert row_lines and col_lines, "expected full gray row and column separators"
    row_spans = _spans_between(_groups(row_lines), g.shape[0])
    col_spans = _spans_between(_groups(col_lines), g.shape[1])
    return row_spans, col_spans


def _extract_windows(g: np.ndarray) -> tuple[list[Span], list[Span], list[np.ndarray]]:
    row_spans, col_spans = _partition_spans(g)
    windows: list[np.ndarray] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(patch == SEPARATOR), "gray belongs only to separators"
            windows.append(patch.copy())
    return row_spans, col_spans, windows


def _min_overlap(h: int, w: int) -> int:
    return max(4, (int(h) * int(w) + 2) // 3)


def _overlap_slices(h: int, w: int, dr: int, dc: int) -> tuple[slice, slice, slice, slice] | None:
    ar0 = max(0, int(dr))
    ar1 = min(h, h + int(dr))
    ac0 = max(0, int(dc))
    ac1 = min(w, w + int(dc))
    if ar0 >= ar1 or ac0 >= ac1:
        return None
    return (
        slice(ar0, ar1),
        slice(ac0, ac1),
        slice(ar0 - int(dr), ar1 - int(dr)),
        slice(ac0 - int(dc), ac1 - int(dc)),
    )


def _pair_consistent(a: np.ndarray, b: np.ndarray, dr: int, dc: int) -> bool:
    h, w = a.shape
    assert b.shape == (h, w), "all windows must have the same shape"
    slices = _overlap_slices(h, w, dr, dc)
    if slices is None:
        return False
    ar, ac, br, bc = slices
    overlap_area = int((ar.stop - ar.start) * (ac.stop - ac.start))
    if overlap_area < _min_overlap(h, w):
        return False

    aa = a[ar, ac]
    bb = b[br, bc]
    known = (aa != MISSING) & (bb != MISSING)
    known_count = int(np.count_nonzero(known))
    if known_count < max(4, overlap_area // 2):
        return False
    return bool(np.all(aa[known] == bb[known]))


def _candidate_offsets(reference: np.ndarray, patch: np.ndarray) -> list[Offset]:
    h, w = reference.shape
    candidates: list[tuple[int, int, int]] = []
    for dr in range(1 - h, h):
        for dc in range(1 - w, w):
            if dr == 0 and dc == 0:
                continue
            if not _pair_consistent(reference, patch, dr, dc):
                continue
            slices = _overlap_slices(h, w, dr, dc)
            assert slices is not None
            ar, ac, br, bc = slices
            known = (reference[ar, ac] != MISSING) & (patch[br, bc] != MISSING)
            candidates.append((dr, dc, int(np.count_nonzero(known))))

    candidates.sort(key=lambda item: (-item[2], abs(item[0]) + abs(item[1]), item[0], item[1]))
    return [(int(dr), int(dc)) for dr, dc, _known in candidates]


def _recover_offsets(windows: list[np.ndarray]) -> list[Offset]:
    assert len(windows) >= 4, "expected several overlapping windows"
    shape = windows[0].shape
    assert all(window.shape == shape for window in windows)

    candidates: dict[int, list[Offset]] = {0: [(0, 0)]}
    for idx in range(1, len(windows)):
        cands = _candidate_offsets(windows[0], windows[idx])
        assert cands, "each window must align with the first window"
        candidates[idx] = cands

    assigned: dict[int, Offset] = {0: (0, 0)}
    unassigned = set(range(1, len(windows)))
    solutions: list[list[Offset]] = []

    def valid(idx: int, offset: Offset) -> bool:
        if offset in assigned.values():
            return False
        for other_idx, other_offset in assigned.items():
            dr = int(offset[0] - other_offset[0])
            dc = int(offset[1] - other_offset[1])
            if not _pair_consistent(windows[other_idx], windows[idx], dr, dc):
                return False
        return True

    def search() -> None:
        if len(solutions) > 1:
            return
        if not unassigned:
            solutions.append([assigned[i] for i in range(len(windows))])
            return

        best_idx = None
        best_options: list[Offset] | None = None
        for idx in sorted(unassigned):
            options = [offset for offset in candidates[idx] if valid(idx, offset)]
            if best_options is None or len(options) < len(best_options):
                best_idx = idx
                best_options = options
        assert best_idx is not None and best_options is not None

        idx = best_idx
        unassigned.remove(idx)
        for offset in best_options:
            assigned[idx] = offset
            search()
            assigned.pop(idx)
            if len(solutions) > 1:
                break
        unassigned.add(idx)

    search()
    assert len(solutions) == 1, "window alignment must be uniquely determined"
    return solutions[0]


def _visible_evidence(windows: list[np.ndarray], offsets: list[Offset]) -> dict[tuple[int, int], int]:
    evidence: dict[tuple[int, int], int] = {}
    for patch, (orow, ocol) in zip(windows, offsets):
        for r, c in zip(*np.where(patch != MISSING)):
            key = (int(orow + r), int(ocol + c))
            color = int(patch[r, c])
            if key in evidence:
                assert evidence[key] == color, "aligned visible overlap must agree"
            else:
                evidence[key] = color
    return evidence


def main(grid: np.ndarray) -> np.ndarray:
    """Repair magenta erasures by aligning partition windows of one hidden canvas.

    Input structure:
    - Gray full-row and full-column separator lines divide the grid into equal rectangular windows.
    - Every window shows a different offset crop of the same hidden multicolor canvas.
    - A few pixels inside each window are magenta erasure markers; gray and magenta are not canvas colors.

    Rule:
    - Use only non-magenta pixels in the large pairwise overlaps to recover the unique relative window offsets.
    - For each magenta pixel, translate its local coordinate through the recovered offset into hidden-canvas coordinates.
    - Replace it with the color seen at that same hidden coordinate in another aligned window, preserving separators and all visible pixels.

    Key insight:
    - A single window has no local pattern to complete; the missing colors are determined only after treating the regions as misaligned views of one shared hidden canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    row_spans, col_spans, windows = _extract_windows(g)
    assert all(np.any(window == MISSING) for window in windows), "each window should contain erasures"

    offsets = _recover_offsets(windows)
    evidence = _visible_evidence(windows, offsets)

    repaired = [window.copy() for window in windows]
    for idx, patch in enumerate(repaired):
        orow, ocol = offsets[idx]
        for r, c in zip(*np.where(patch == MISSING)):
            key = (int(orow + r), int(ocol + c))
            assert key in evidence, "every erasure needs evidence from another window"
            patch[r, c] = evidence[key]

    out = g.copy()
    idx = 0
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            out[r0 : r1 + 1, c0 : c1 + 1] = repaired[idx]
            idx += 1
    return out


# --------------- generator ---------------

LAYOUTS = ((2, 2), (2, 3), (3, 2), (3, 3))
WINDOW_SIDE_RANGE = (6, 9)
GEN_ATTEMPTS = 500


def _gen_layout() -> tuple[int, int, int, int]:
    part_rows, part_cols = random.choice(LAYOUTS)
    max_h = min(WINDOW_SIDE_RANGE[1], (30 - (part_rows - 1)) // part_rows)
    max_w = min(WINDOW_SIDE_RANGE[1], (30 - (part_cols - 1)) // part_cols)
    assert max_h >= WINDOW_SIDE_RANGE[0] and max_w >= WINDOW_SIDE_RANGE[0]
    win_h = random.randint(WINDOW_SIDE_RANGE[0], max_h)
    win_w = random.randint(WINDOW_SIDE_RANGE[0], max_w)
    return part_rows, part_cols, win_h, win_w


def _gen_pair_overlap_ok(a: Offset, b: Offset, win_h: int, win_w: int) -> bool:
    dr = abs(int(a[0] - b[0]))
    dc = abs(int(a[1] - b[1]))
    return (win_h - dr) * (win_w - dc) >= _min_overlap(win_h, win_w)


def _gen_origins(count: int, win_h: int, win_w: int) -> list[Offset] | None:
    row_spans = [2] + ([3] if win_h >= 8 else [])
    col_spans = [2] + ([3] if win_w >= 8 else [])
    for _ in range(80):
        max_r = random.choice(row_spans)
        max_c = random.choice(col_spans)
        pool = [(r, c) for r in range(max_r + 1) for c in range(max_c + 1)]
        random.shuffle(pool)
        chosen: list[Offset] = []
        for pos in pool:
            if all(_gen_pair_overlap_ok(pos, prev, win_h, win_w) for prev in chosen):
                chosen.append(pos)
                if len(chosen) == count:
                    break
        if len(chosen) != count:
            continue
        if len({r for r, _c in chosen}) < 2 or len({c for _r, c in chosen}) < 2:
            continue
        random.shuffle(chosen)
        return chosen
    return None


def _gen_hidden_canvas(origins: list[Offset], win_h: int, win_w: int) -> np.ndarray:
    max_r = max(r for r, _c in origins)
    max_c = max(c for _r, c in origins)
    palette_size = random.randint(4, len(CANVAS_COLORS))
    palette = random.sample(CANVAS_COLORS, palette_size)
    return np.random.choice(np.array(palette, dtype=int), size=(win_h + max_r, win_w + max_c))


def _gen_clean_windows(hidden: np.ndarray, origins: list[Offset], win_h: int, win_w: int) -> list[np.ndarray]:
    return [hidden[r : r + win_h, c : c + win_w].copy() for r, c in origins]


def _gen_coverage(origins: list[Offset], win_h: int, win_w: int) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    coverage: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for idx, (orow, ocol) in enumerate(origins):
        for r in range(win_h):
            for c in range(win_w):
                coverage.setdefault((orow + r, ocol + c), []).append((idx, r, c))
    return coverage


def _gen_erased_windows(clean_windows: list[np.ndarray], origins: list[Offset]) -> tuple[list[np.ndarray], list[int]] | None:
    count = len(clean_windows)
    win_h, win_w = clean_windows[0].shape
    coverage = _gen_coverage(origins, win_h, win_w)
    erased = [patch.copy() for patch in clean_windows]
    used_hidden: set[tuple[int, int]] = set()
    missing_colors: list[int] = []

    for idx in range(count):
        if count <= 4:
            erase_count = random.randint(2, 4)
        elif count <= 6:
            erase_count = random.randint(1, 3)
        else:
            erase_count = random.randint(1, 2)

        chosen_local: set[tuple[int, int]] = set()
        for step in range(erase_count):
            candidates = []
            edge_candidates = []
            for r in range(win_h):
                for c in range(win_w):
                    key = (origins[idx][0] + r, origins[idx][1] + c)
                    if key in used_hidden or (r, c) in chosen_local:
                        continue
                    if len(coverage[key]) < 2:
                        continue
                    item = (r, c, key)
                    candidates.append(item)
                    if r in (0, win_h - 1) or c in (0, win_w - 1):
                        edge_candidates.append(item)
            if not candidates:
                return None
            pool = edge_candidates if step == 0 and edge_candidates and random.random() < 0.55 else candidates
            r, c, key = random.choice(pool)
            erased[idx][r, c] = MISSING
            chosen_local.add((r, c))
            used_hidden.add(key)
            missing_colors.append(int(clean_windows[idx][r, c]))

    if len(set(missing_colors)) < 2:
        return None
    return erased, missing_colors


def _gen_compose(part_rows: int, part_cols: int, windows: list[np.ndarray]) -> np.ndarray:
    win_h, win_w = windows[0].shape
    h = part_rows * win_h + (part_rows - 1)
    w = part_cols * win_w + (part_cols - 1)
    grid = np.full((h, w), SEPARATOR, dtype=int)
    idx = 0
    for pr in range(part_rows):
        r0 = pr * (win_h + 1)
        for pc in range(part_cols):
            c0 = pc * (win_w + 1)
            grid[r0 : r0 + win_h, c0 : c0 + win_w] = windows[idx]
            idx += 1
    return grid


def _gen_no_large_self_match(window: np.ndarray) -> bool:
    h, w = window.shape
    for dr in range(1 - h, h):
        for dc in range(1 - w, w):
            if dr == 0 and dc == 0:
                continue
            slices = _overlap_slices(h, w, dr, dc)
            if slices is None:
                continue
            ar, ac, br, bc = slices
            overlap_area = int((ar.stop - ar.start) * (ac.stop - ac.start))
            if overlap_area < _min_overlap(h, w):
                continue
            if np.array_equal(window[ar, ac], window[br, bc]):
                return False
    return True


def _gen_has_aligned_witness(erased_windows: list[np.ndarray], origins: list[Offset]) -> bool:
    win_h, win_w = erased_windows[0].shape
    for idx, patch in enumerate(erased_windows):
        for r, c in zip(*np.where(patch == MISSING)):
            hidden_key = (origins[idx][0] + int(r), origins[idx][1] + int(c))
            witnesses = 0
            for j, other in enumerate(erased_windows):
                if j == idx:
                    continue
                rr = hidden_key[0] - origins[j][0]
                cc = hidden_key[1] - origins[j][1]
                if 0 <= rr < win_h and 0 <= cc < win_w and other[rr, cc] != MISSING:
                    witnesses += 1
            if witnesses == 0:
                return False
    return True


def _gen_reject_same_local_shortcut(clean_windows: list[np.ndarray], erased_windows: list[np.ndarray]) -> bool:
    for idx, patch in enumerate(erased_windows):
        for r, c in zip(*np.where(patch == MISSING)):
            true_color = int(clean_windows[idx][r, c])
            same_local = {
                int(other[r, c])
                for j, other in enumerate(erased_windows)
                if j != idx and int(other[r, c]) != MISSING
            }
            if same_local and true_color not in same_local:
                return True
    return False


def _gen_semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    clean_windows: list[np.ndarray],
    erased_windows: list[np.ndarray],
    origins: list[Offset],
) -> bool:
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if not all(_gen_no_large_self_match(window) for window in clean_windows):
        return False
    if not _gen_has_aligned_witness(erased_windows, origins):
        return False
    if not _gen_reject_same_local_shortcut(clean_windows, erased_windows):
        return False

    true_offsets = [(r - origins[0][0], c - origins[0][1]) for r, c in origins]
    try:
        recovered = _recover_offsets(erased_windows)
    except AssertionError:
        return False
    if recovered != true_offsets:
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    part_rows, part_cols, win_h, win_w = _gen_layout()
    count = part_rows * part_cols
    origins = _gen_origins(count, win_h, win_w)
    if origins is None:
        return None

    hidden = _gen_hidden_canvas(origins, win_h, win_w)
    clean_windows = _gen_clean_windows(hidden, origins, win_h, win_w)
    erased_result = _gen_erased_windows(clean_windows, origins)
    if erased_result is None:
        return None
    erased_windows, _missing_colors = erased_result

    inp = _gen_compose(part_rows, part_cols, erased_windows)
    expected = _gen_compose(part_rows, part_cols, clean_windows)
    if _gen_semantic_checks(inp, expected, clean_windows, erased_windows, origins):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fable/3/042/5 input")
