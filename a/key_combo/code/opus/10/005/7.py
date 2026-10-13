from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
YELLOW = 4
TEAL = 8
GEN_ATTEMPTS = 2000
PALETTE = (1, 2, 3, 6, 7, 9)
RECT_CORNERS = ("TL", "TR", "BL", "BR")


def _contiguous(values: list[int]) -> bool:
    return values == list(range(values[0], values[-1] + 1))


def _find_axis(grid: np.ndarray) -> tuple[str, int, tuple[int, int], tuple[int, int]]:
    yellow = [(int(r), int(c)) for r, c in np.argwhere(grid == YELLOW)]
    assert len(yellow) >= 3, "expected one yellow axis and one off-axis marker"

    candidates: list[tuple[str, int, int]] = []
    row_counts = Counter(r for r, _ in yellow)
    col_counts = Counter(c for _, c in yellow)

    for row, count in row_counts.items():
        if count >= 2:
            cols = sorted(c for r, c in yellow if r == row)
            if _contiguous(cols):
                candidates.append(("h", int(row), len(cols)))

    for col, count in col_counts.items():
        if count >= 2:
            rows = sorted(r for r, c in yellow if c == col)
            if _contiguous(rows):
                candidates.append(("v", int(col), len(rows)))

    assert candidates, "expected a contiguous yellow axis"
    max_len = max(length for _, _, length in candidates)
    best = [cand for cand in candidates if cand[2] == max_len]
    assert len(best) == 1, "axis must be uniquely determined"
    orient, axis, _ = best[0]

    if orient == "h":
        axis_cells = {(axis, c) for _, c in yellow if _ == axis}
        marker = [cell for cell in yellow if cell[0] != axis]
    else:
        axis_cells = {(r, axis) for r, _ in yellow if _ == axis}
        marker = [cell for cell in yellow if cell[1] != axis]

    assert len(marker) == 1, "expected exactly one off-axis yellow marker"
    return orient, int(axis), marker[0], min(axis_cells), max(axis_cells)


def _marker_side(orient: str, axis: int, marker: tuple[int, int]) -> int:
    r, c = marker
    if orient == "h":
        assert r != axis
        return -1 if r < axis else 1
    assert c != axis
    return -1 if c < axis else 1


def _is_source_cell(r: int, c: int, orient: str, axis: int, side: int) -> bool:
    return (r < axis if side < 0 else r > axis) if orient == "h" else (c < axis if side < 0 else c > axis)


def _reflect_coord(r: int, c: int, orient: str, axis: int) -> tuple[int, int]:
    return (2 * axis - r, c) if orient == "h" else (r, 2 * axis - c)


def _foreground_colors(grid: np.ndarray) -> tuple[int, int]:
    colors = [int(v) for v in np.unique(grid) if int(v) not in (BG, YELLOW, TEAL)]
    assert len(colors) == 2, f"expected exactly two foreground colors, got {colors}"
    return int(colors[0]), int(colors[1])


def _local_color_counts(
    grid: np.ndarray,
    r: int,
    c: int,
    orient: str,
    axis: int,
    side: int,
    colors: tuple[int, int],
) -> dict[int, int]:
    h, w = grid.shape
    counts = {int(colors[0]): 0, int(colors[1]): 0}
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if not _is_source_cell(rr, cc, orient, axis, side):
                continue
            val = int(grid[rr, cc])
            if val in counts:
                counts[val] += 1
    return counts


def _majority_color(counts: dict[int, int]) -> int | None:
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    if ranked[0][1] >= 5 and ranked[0][1] > ranked[1][1]:
        return int(ranked[0][0])
    return None


def _clean_source(
    grid: np.ndarray,
    orient: str,
    axis: int,
    side: int,
    colors: tuple[int, int],
    passes: int = 2,
) -> np.ndarray:
    cur = np.array(grid, dtype=int, copy=True)
    h, w = cur.shape
    for _ in range(passes):
        nxt = cur.copy()
        for r in range(h):
            for c in range(w):
                if not _is_source_cell(r, c, orient, axis, side):
                    continue
                if int(cur[r, c]) == YELLOW:
                    continue
                majority = _majority_color(_local_color_counts(cur, r, c, orient, axis, side, colors))
                if majority is not None and int(cur[r, c]) != majority:
                    nxt[r, c] = majority
        cur = nxt
    return cur


def _dent_centers(
    original: np.ndarray,
    clean: np.ndarray,
    orient: str,
    axis: int,
    side: int,
    colors: tuple[int, int],
) -> list[tuple[int, int, int]]:
    dents: list[tuple[int, int, int]] = []
    h, w = original.shape
    for r in range(h):
        for c in range(w):
            if not _is_source_cell(r, c, orient, axis, side):
                continue
            if int(original[r, c]) != BG:
                continue
            majority = _majority_color(_local_color_counts(clean, r, c, orient, axis, side, colors))
            if majority is not None:
                dents.append((r, c, majority))
    return dents


def _apply_defect_marks(
    clean: np.ndarray,
    dents: list[tuple[int, int, int]],
    colors: tuple[int, int],
) -> np.ndarray:
    out = np.array(clean, dtype=int, copy=True)
    h, w = out.shape
    a, b = map(int, colors)
    for r, c, host in dents:
        other = b if int(host) == a else a
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = other
    for r, c, _host in dents:
        out[r, c] = TEAL
    return out


def _mirror_source(grid: np.ndarray, orient: str, axis: int, side: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    for r in range(h):
        for c in range(w):
            if not _is_source_cell(r, c, orient, axis, side):
                continue
            if int(grid[r, c]) == YELLOW:
                continue
            rr, cc = _reflect_coord(r, c, orient, axis)
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != YELLOW:
                out[rr, cc] = int(grid[r, c])
    return out


def _pipeline(
    grid: np.ndarray,
    *,
    do_clean: bool = True,
    do_mark: bool = True,
    mirror_first: bool = False,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    orient, axis, marker, _axis_start, _axis_end = _find_axis(g)
    side = _marker_side(orient, axis, marker)
    colors = _foreground_colors(g)

    if mirror_first:
        mirrored = _mirror_source(g, orient, axis, side)
        work = _clean_source(mirrored, orient, axis, side, colors) if do_clean else mirrored.copy()
        if do_mark:
            dents = _dent_centers(g, work, orient, axis, side, colors)
            work = _apply_defect_marks(work, dents, colors)
        return work

    work = _clean_source(g, orient, axis, side, colors) if do_clean else g.copy()
    if do_mark:
        dents = _dent_centers(g, work, orient, axis, side, colors)
        work = _apply_defect_marks(work, dents, colors)
    return _mirror_source(work, orient, axis, side)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise the marked source half, mark its original black dents, then mirror-complete it across the yellow axis.

    Input structure:
    - The background is black, with exactly two non-yellow foreground colors forming a large source-side region.
    - One contiguous yellow horizontal or vertical line is the symmetry axis, and one off-axis yellow pixel marks which side is the trusted source side.
    - The source side may contain one-cell wrong-color noise and black dent cells inside the foreground; the opposite side may be empty or contain incomplete residue.

    Rule:
    - Use the side containing the off-axis yellow marker as the source side.
    - First denoise only that source side: any source-side cell whose source-side 3x3 neighborhood has a strict foreground-color majority of at least five cells is recolored to that majority.
    - For every source-side cell that was originally black and still has such a cleaned majority neighborhood, fill its 3x3 neighborhood with the opposite foreground color and recolor the dent center teal.
    - Finally mirror the entire processed source side, including black background and teal dent marks, across the yellow axis onto the opposite side; keep the yellow axis and the single yellow marker unchanged.

    Key insight:
    - The output depends on an order-constrained chain: dents are identified from the original black cells but interpreted after denoising, and only the cleaned-and-marked source half is trustworthy enough to mirror.
    """
    return _pipeline(grid, do_clean=True, do_mark=True, mirror_first=False)


def _paint_rect(patch: np.ndarray, corner: str, color: int) -> None:
    h, w = patch.shape
    rh = random.randint(max(4, int(np.ceil(0.55 * h))), h)
    rw = random.randint(max(4, int(np.ceil(0.55 * w))), w)
    r_margin = random.randint(0, max(0, h - rh))
    c_margin = random.randint(0, max(0, w - rw))
    if corner == "TL":
        r0, c0 = 0, 0
    elif corner == "TR":
        r0, c0 = 0, w - rw - c_margin
    elif corner == "BL":
        r0, c0 = h - rh - r_margin, 0
    else:
        r0, c0 = h - rh - r_margin, w - rw - c_margin
    patch[r0 : r0 + rh, c0 : c0 + rw] = color


def _build_clean_patch(h: int, w: int, colors: tuple[int, int]) -> np.ndarray | None:
    a, b = map(int, colors)
    for _ in range(80):
        patch = np.zeros((h, w), dtype=int)
        n_rects = random.choice([2, 3])
        corners = random.sample(RECT_CORNERS, n_rects)
        if n_rects == 2:
            seq = [a, b]
            random.shuffle(seq)
        else:
            lead = random.choice([a, b])
            seq = [lead, b if lead == a else a, lead]
        for corner, color in zip(corners, seq):
            _paint_rect(patch, corner, color)

        count_a = int(np.sum(patch == a))
        count_b = int(np.sum(patch == b))
        fill = int(np.sum(patch != BG))
        if count_a < 14 or count_b < 14:
            continue
        if fill >= h * w - 2:
            continue
        if len(np.unique(patch)) < 3:
            continue
        return patch
    return None


def _patch_majority(clean_patch: np.ndarray, r: int, c: int, colors: tuple[int, int]) -> int | None:
    a, b = map(int, colors)
    sub = clean_patch[max(0, r - 1) : min(clean_patch.shape[0], r + 2), max(0, c - 1) : min(clean_patch.shape[1], c + 2)]
    counts = {a: int(np.sum(sub == a)), b: int(np.sum(sub == b))}
    return _majority_color(counts)


def _pick_spaced_cells(candidates: list[tuple[int, int]], wanted: int, gap: int) -> list[tuple[int, int]]:
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= gap for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) >= wanted:
                break
    return chosen


def _dent_and_noise_plan(clean_patch: np.ndarray, colors: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    dent_candidates: list[tuple[int, int]] = []
    boundaryish_dents: list[tuple[int, int]] = []
    noise_candidates: list[tuple[int, int]] = []
    boundaryish_noise: list[tuple[int, int]] = []
    h, w = clean_patch.shape

    for r in range(h):
        for c in range(w):
            val = int(clean_patch[r, c])
            if val == BG:
                continue
            majority = _patch_majority(clean_patch, r, c, colors)
            if majority is None or majority != val:
                continue

            sub = clean_patch[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]
            counts = {color: int(np.sum(sub == color)) for color in colors}
            host_count = counts[val]
            edge_like = host_count <= 6 or bool(np.any(sub == BG)) or any(
                0 <= rr < h and 0 <= cc < w and int(clean_patch[rr, cc]) not in (BG, val)
                for rr in range(r - 1, r + 2)
                for cc in range(c - 1, c + 2)
            )

            dent_candidates.append((r, c))
            noise_candidates.append((r, c))
            if edge_like:
                boundaryish_dents.append((r, c))
                boundaryish_noise.append((r, c))

    if len(dent_candidates) < 3 or len(noise_candidates) < 4:
        return None

    wanted_dents = random.randint(2, min(4, len(dent_candidates)))
    dents = _pick_spaced_cells(boundaryish_dents or dent_candidates, wanted_dents, gap=3)
    if len(dents) < 2:
        dents = _pick_spaced_cells(dent_candidates, wanted_dents, gap=3)
    if len(dents) < 2:
        return None

    dent_forbid = {(rr, cc) for r, c in dents for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)}
    far_noise = [cell for cell in (boundaryish_noise or noise_candidates) if cell not in dent_forbid]
    if not far_noise:
        far_noise = [cell for cell in noise_candidates if cell not in dent_forbid]
    if not far_noise:
        return None

    wanted_noise = random.randint(1, min(3, len(far_noise)))
    noises = _pick_spaced_cells(far_noise, wanted_noise, gap=2)
    if not noises:
        return None
    return dents, noises


def _source_layout(
    orient: str,
    side: int,
    source_depth: int,
    target_depth: int,
) -> tuple[int, int, int, int, int, int, int]:
    if orient == "v":
        h = random.randint(14, 18)
        w = source_depth + 1 + target_depth
        axis = source_depth if side < 0 else target_depth
        patch_h = random.randint(8, min(10, h - 2))
        patch_w = random.randint(5, max(5, source_depth - 1))
        r0 = random.randint(1, h - patch_h - 1)
        touch_axis = random.random() < 0.5
        if side < 0:
            c1 = axis - 1 if touch_axis else axis - random.choice([2, 3])
            c0 = c1 - patch_w + 1
        else:
            c0 = axis + 1 if touch_axis else axis + random.choice([2, 3])
            c1 = c0 + patch_w - 1
        if c0 < 1 or c1 > w - 2:
            raise ValueError("patch placement out of bounds")
        return h, w, axis, r0, c0, patch_h, patch_w

    h = source_depth + 1 + target_depth
    w = random.randint(14, 18)
    axis = source_depth if side < 0 else target_depth
    patch_h = random.randint(5, max(5, source_depth - 1))
    patch_w = random.randint(8, min(10, w - 2))
    c0 = random.randint(1, w - patch_w - 1)
    touch_axis = random.random() < 0.5
    if side < 0:
        r1 = axis - 1 if touch_axis else axis - random.choice([2, 3])
        r0 = r1 - patch_h + 1
    else:
        r0 = axis + 1 if touch_axis else axis + random.choice([2, 3])
        r1 = r0 + patch_h - 1
    if r0 < 1 or r1 > h - 2:
        raise ValueError("patch placement out of bounds")
    return h, w, axis, r0, c0, patch_h, patch_w


def _make_empty_canvas(shape: tuple[int, int], orient: str, axis: int, marker: tuple[int, int], axis_span: tuple[int, int]) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    start, end = axis_span
    if orient == "v":
        grid[start : end + 1, axis] = YELLOW
    else:
        grid[axis, start : end + 1] = YELLOW
    grid[marker] = YELLOW
    return grid


def _embed_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _full_processed_output(
    shape: tuple[int, int],
    orient: str,
    axis: int,
    side: int,
    marker: tuple[int, int],
    axis_span: tuple[int, int],
    processed_patch: np.ndarray,
    patch_r0: int,
    patch_c0: int,
) -> np.ndarray:
    base = _make_empty_canvas(shape, orient, axis, marker, axis_span)
    _embed_patch(base, processed_patch, patch_r0, patch_c0)
    return _mirror_source(base, orient, axis, side)


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    orient = random.choice(["h", "v"])
    side = random.choice([-1, 1])
    colors = tuple(random.sample(PALETTE, 2))
    source_depth = random.randint(6, 8)
    target_depth = random.randint(source_depth, min(10, source_depth + 2))

    try:
        h, w, axis, patch_r0, patch_c0, patch_h, patch_w = _source_layout(orient, side, source_depth, target_depth)
    except ValueError:
        return None

    clean_patch = _build_clean_patch(patch_h, patch_w, colors)
    if clean_patch is None:
        return None

    plan = _dent_and_noise_plan(clean_patch, colors)
    if plan is None:
        return None
    dents_local, noise_local = plan

    noisy_patch = clean_patch.copy()
    for r, c in dents_local:
        noisy_patch[r, c] = BG

    for r, c in noise_local:
        host = int(clean_patch[r, c])
        other = int(colors[1] if host == colors[0] else colors[0])
        noisy_patch[r, c] = other

    axis_start = max(0, patch_r0 - 1) if orient == "v" else max(0, patch_c0 - 1)
    axis_end = min(h - 1 if orient == "v" else w - 1, (patch_r0 + patch_h) if orient == "v" else (patch_c0 + patch_w))
    if axis_end - axis_start < 4:
        return None

    if orient == "v":
        marker_row_choices = [axis_start, axis_end]
        marker_col = axis - random.choice([1, 2]) if side < 0 else axis + random.choice([1, 2])
        marker = (random.choice(marker_row_choices), marker_col)
    else:
        marker_col_choices = [axis_start, axis_end]
        marker_row = axis - random.choice([1, 2]) if side < 0 else axis + random.choice([1, 2])
        marker = (marker_row, random.choice(marker_col_choices))

    if not (0 <= marker[0] < h and 0 <= marker[1] < w):
        return None

    raw = _make_empty_canvas((h, w), orient, axis, marker, (axis_start, axis_end))
    _embed_patch(raw, noisy_patch, patch_r0, patch_c0)

    dents_global = [(patch_r0 + r, patch_c0 + c, int(clean_patch[r, c])) for r, c in dents_local]
    processed_patch = clean_patch.copy()
    for r, c in dents_local:
        host = int(clean_patch[r, c])
        other = int(colors[1] if host == colors[0] else colors[0])
        processed_patch[max(0, r - 1) : min(patch_h, r + 2), max(0, c - 1) : min(patch_w, c + 2)] = other
    for r, c in dents_local:
        processed_patch[r, c] = TEAL

    if random.random() < 0.55:
        mirrored_raw = _make_empty_canvas((h, w), orient, axis, marker, (axis_start, axis_end))
        _embed_patch(mirrored_raw, noisy_patch, patch_r0, patch_c0)
        mirrored_raw = _mirror_source(mirrored_raw, orient, axis, side)
        for r in range(h):
            for c in range(w):
                if _is_source_cell(r, c, orient, axis, side):
                    continue
                if int(mirrored_raw[r, c]) in colors and random.random() < 0.35:
                    raw[r, c] = int(mirrored_raw[r, c] if random.random() < 0.7 else random.choice(colors))
    else:
        mirror_truth = _full_processed_output((h, w), orient, axis, side, marker, (axis_start, axis_end), processed_patch, patch_r0, patch_c0)
        target_fg = [(r, c) for r in range(h) for c in range(w) if not _is_source_cell(r, c, orient, axis, side) and int(mirror_truth[r, c]) in colors]
        random.shuffle(target_fg)
        for r, c in target_fg[: random.randint(3, min(8, len(target_fg)))]:
            raw[r, c] = int(random.choice(colors))

    expected = _full_processed_output((h, w), orient, axis, side, marker, (axis_start, axis_end), processed_patch, patch_r0, patch_c0)

    solved = main(raw)
    if not np.array_equal(solved, expected):
        return None

    clean_full = _clean_source(raw, orient, axis, side, colors)
    if not np.array_equal(clean_full[patch_r0 : patch_r0 + patch_h, patch_c0 : patch_c0 + patch_w], clean_patch):
        return None

    detected = _dent_centers(raw, clean_full, orient, axis, side, colors)
    if sorted(detected) != sorted(dents_global):
        return None

    if all(np.array_equal(expected, _pipeline(raw, do_clean=do_clean, do_mark=do_mark, mirror_first=mirror_first))
           for do_clean, do_mark, mirror_first in [
               (False, True, False),
               (True, False, False),
               (True, True, True),
           ]):
        return None

    if np.array_equal(raw, expected):
        return None

    target_changed = False
    for r in range(h):
        for c in range(w):
            if _is_source_cell(r, c, orient, axis, side) or int(raw[r, c]) == YELLOW:
                continue
            if int(raw[r, c]) != int(expected[r, c]):
                target_changed = True
                break
        if target_changed:
            break
    if not target_changed:
        return None

    return raw, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
