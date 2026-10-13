from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

CELL = 4
CELL_GAP = 1
ARRAY_ROWS = 2
ARRAY_COLS = 3
ARRAY_H = ARRAY_ROWS * CELL + (ARRAY_ROWS - 1) * CELL_GAP
ARRAY_W = ARRAY_COLS * CELL + (ARRAY_COLS - 1) * CELL_GAP
TOP_CELL_ORIGINS = (
    (0, 0),
    (0, 5),
    (0, 10),
    (5, 0),
    (5, 5),
    (5, 10),
)

GEN_RETRY_LIMIT = 400


def _active_runs(active: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(active):
        if bool(value) and start is None:
            start = idx
        elif not bool(value) and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(active) - 1))
    return runs


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    box = _mask_bbox(mask)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return None

    expected = np.zeros_like(mask, dtype=bool)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    if not np.array_equal(mask, expected):
        return None
    return r0, c0, r1, c1


def _interior_bbox(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _find_frame(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    box = _rect_frame_bbox(grid == color)
    assert box is not None, f"expected one hollow frame of color {color}"
    ir0, ic0, ir1, ic1 = _interior_bbox(box)
    assert ir1 - ir0 + 1 >= 3 and ic1 - ic0 + 1 >= 3
    assert np.all(grid[ir0 : ir1 + 1, ic0 : ic1 + 1] == BLACK)
    return box


def _draw_axis_line(canvas: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        c_lo, c_hi = sorted((c0, c1))
        canvas[r0, c_lo : c_hi + 1] = True
        return
    assert c0 == c1
    r_lo, r_hi = sorted((r0, r1))
    canvas[r_lo : r_hi + 1, c0] = True


def _fill_enclosed(stroke: np.ndarray) -> np.ndarray:
    h, w = stroke.shape
    open_space = ~stroke
    padded = np.pad(open_space, 1, constant_values=True)
    seen = np.zeros_like(padded, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    ph, pw = padded.shape

    for r in range(ph):
        for c in (0, pw - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(pw):
        for r in (0, ph - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < ph and 0 <= nc < pw and padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))

    holes = open_space & ~seen[1 : h + 1, 1 : w + 1]
    return stroke | holes


def _scaled_index(i: int, src_len: int, dst_len: int) -> int:
    assert src_len >= 1 and dst_len >= 1
    if src_len == 1:
        return (dst_len - 1) // 2
    return int(round(i * (dst_len - 1) / (src_len - 1)))


def _stretch_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    box = _mask_bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    src = mask[r0 : r1 + 1, c0 : c1 + 1]
    src_h, src_w = src.shape
    dst_h, dst_w = target_shape
    stroke = np.zeros((dst_h, dst_w), dtype=bool)

    coords = np.argwhere(src)
    mapped: dict[tuple[int, int], tuple[int, int]] = {}
    for sr_raw, sc_raw in coords:
        sr = int(sr_raw)
        sc = int(sc_raw)
        tr = _scaled_index(sr, src_h, dst_h)
        tc = _scaled_index(sc, src_w, dst_w)
        mapped[(sr, sc)] = (tr, tc)
        stroke[tr, tc] = True

    for sr_raw, sc_raw in coords:
        sr = int(sr_raw)
        sc = int(sc_raw)
        for nr, nc in ((sr + 1, sc), (sr, sc + 1)):
            if (nr, nc) in mapped:
                r_a, c_a = mapped[(sr, sc)]
                r_b, c_b = mapped[(nr, nc)]
                _draw_axis_line(stroke, r_a, c_a, r_b, c_b)

    return _fill_enclosed(stroke)


def _extract_sprite_windows(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]]:
    red = grid == RED
    row_runs = _active_runs(np.any(red, axis=1))
    col_runs = _active_runs(np.any(red, axis=0))
    assert len(row_runs) == ARRAY_ROWS and len(col_runs) == ARRAY_COLS
    assert all(r1 - r0 + 1 == CELL for r0, r1 in row_runs)
    assert all(c1 - c0 + 1 == CELL for c0, c1 in col_runs)

    windows: list[tuple[int, int, int, int]] = []
    masks: list[np.ndarray] = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert patch.shape == (CELL, CELL)
            assert np.all((patch == BLACK) | (patch == RED))
            mask = patch == RED
            assert np.any(mask)
            windows.append((r0, r1, c0, c1))
            masks.append(mask)
    return windows, masks


def _find_union_candidates(masks: list[np.ndarray]) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for dst, dst_mask in enumerate(masks):
        for a in range(len(masks)):
            if a == dst:
                continue
            for b in range(a + 1, len(masks)):
                if b == dst:
                    continue
                mask_a = masks[a]
                mask_b = masks[b]
                if not np.array_equal(mask_a | mask_b, dst_mask):
                    continue
                if not (np.any(mask_a & ~mask_b) and np.any(mask_b & ~mask_a)):
                    continue
                candidates.append((a, b, dst))
    return candidates


def _select_union_candidate(candidates: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    assert candidates, "expected at least one union candidate"
    return min(candidates, key=lambda item: (item[0], item[1], item[2]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the unique mask-union roles, stretch the two operands into the blue and green frames, and recolor the union result.

    Input structure:
    - The background is black.
    - The upper part contains six non-empty red 4x4 sprite masks arranged as a
      3-column by 2-row array with one-cell gaps.
    - The lower part contains one blue hollow rectangle and one green hollow
      rectangle, each with a black interior and a different size.

    Rule:
    - Among the six red masks, find a triple (A, B, D) whose pixelwise union
      satisfies A | B == D and where A and B each contribute at least one pixel
      not present in the other; if several such triples exist because of
      duplicates, use the row-major earliest A, then the earliest remaining tie.
    - Stretch A from its own red-mask bounding box into the interior of the blue
      frame, and stretch B the same way into the interior of the green frame:
      bbox extremes map to interior extremes, 4-neighbor source adjacencies draw
      straight segments, and any enclosed area is filled.
    - Recolor the red pixels of D to yellow, keep the other three upper masks
      unchanged, and preserve both frame borders.

    Key insight:
    - The union relation assigns semantic roles first; only after identifying
      which masks are the two operands and the union result can the solver know
      which two sprites must undergo geometric stretching and which sprite is
      merely recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    windows, masks = _extract_sprite_windows(g)
    blue_frame = _find_frame(g, BLUE)
    green_frame = _find_frame(g, GREEN)
    src_a, src_b, dst = _select_union_candidate(_find_union_candidates(masks))

    out = g.copy()

    d_r0, d_r1, d_c0, d_c1 = windows[dst]
    out[d_r0 : d_r1 + 1, d_c0 : d_c1 + 1][masks[dst]] = YELLOW

    for src_idx, frame_color, frame_box in (
        (src_a, BLUE, blue_frame),
        (src_b, GREEN, green_frame),
    ):
        ir0, ic0, ir1, ic1 = _interior_bbox(frame_box)
        stretched = _stretch_mask(masks[src_idx], (ir1 - ir0 + 1, ic1 - ic0 + 1))
        out[ir0 : ir1 + 1, ic0 : ic1 + 1][stretched] = frame_color

    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask)


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
    return count


def _has_adjacent_pair(mask: np.ndarray) -> bool:
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c]:
                continue
            if r + 1 < mask.shape[0] and mask[r + 1, c]:
                return True
            if c + 1 < mask.shape[1] and mask[r, c + 1]:
                return True
    return False


def _is_solid_bbox(mask: np.ndarray) -> bool:
    box = _mask_bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))


def _cycle_operand_mask(rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((CELL, CELL), dtype=bool)
    r0 = int(rng.integers(0, CELL - 1))
    c0 = int(rng.integers(0, CELL - 1))
    mask[r0 : r0 + 2, c0 : c0 + 2] = True
    return mask


def _open_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(300):
        count = int(rng.integers(3, 7))
        mask = np.zeros((CELL, CELL), dtype=bool)
        mask.flat[rng.choice(CELL * CELL, size=count, replace=False)] = True
        box = _mask_bbox(mask)
        assert box is not None
        r0, c0, r1, c1 = box
        if r1 == r0 or c1 == c0:
            continue
        if _is_solid_bbox(mask):
            continue
        if not _has_adjacent_pair(mask):
            continue
        if _component_count(mask) > 2:
            continue
        return mask
    raise AssertionError("failed to sample an open operand mask")


def _distractor_mask(rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.25:
        return _cycle_operand_mask(rng)
    return _open_mask(rng)


def _generate_operands(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    fill_branch = bool(rng.random() < 0.45)
    for _ in range(400):
        if fill_branch:
            if rng.random() < 0.5:
                a = _cycle_operand_mask(rng)
                b = _open_mask(rng)
            else:
                a = _open_mask(rng)
                b = _cycle_operand_mask(rng)
        else:
            a = _open_mask(rng)
            b = _open_mask(rng)

        if np.array_equal(a, b):
            continue
        d = a | b
        if np.array_equal(d, a) or np.array_equal(d, b):
            continue
        if not (np.any(a & ~b) and np.any(b & ~a)):
            continue
        if not (4 <= int(d.sum()) <= 8):
            continue
        return a, b, d, fill_branch
    raise AssertionError("failed to generate operand masks")


def _choose_positions(
    rng: np.random.Generator, duplicate_branch: bool
) -> tuple[int, int, int, int | None, list[int]] | None:
    if not duplicate_branch:
        a_idx = int(rng.integers(0, 5))
        b_idx = int(rng.integers(a_idx + 1, 6))
        d_idx = int(rng.choice([i for i in range(6) if i not in (a_idx, b_idx)]))
        rest = [i for i in range(6) if i not in (a_idx, b_idx, d_idx)]
        return a_idx, b_idx, d_idx, None, rest

    for _ in range(100):
        a_idx = int(rng.integers(0, 4))
        b_idx = int(rng.integers(a_idx + 1, 5))
        remaining = [i for i in range(6) if i not in (a_idx, b_idx)]
        d_idx = int(rng.choice(remaining))
        remaining = [i for i in remaining if i != d_idx]
        later = [i for i in remaining if i > b_idx]
        if not later:
            continue
        b_dup_idx = int(rng.choice(later))
        rest = [i for i in remaining if i != b_dup_idx]
        return a_idx, b_idx, d_idx, b_dup_idx, rest
    return None


def _band_coverage_visible(masks_by_pos: list[np.ndarray]) -> bool:
    for row_idx in range(ARRAY_ROWS):
        row_masks = masks_by_pos[row_idx * ARRAY_COLS : (row_idx + 1) * ARRAY_COLS]
        row_union = np.any(np.stack(row_masks, axis=0), axis=0)
        if not np.all(np.any(row_union, axis=1)):
            return False
    for col_idx in range(ARRAY_COLS):
        col_union = np.any(np.stack((masks_by_pos[col_idx], masks_by_pos[col_idx + 3]), axis=0), axis=0)
        if not np.all(np.any(col_union, axis=0)):
            return False
    return True


def _make_mask_assignment(
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], dict[str, int | bool]] | None:
    duplicate_branch = bool(rng.random() < 0.35)
    chosen = _choose_positions(rng, duplicate_branch)
    if chosen is None:
        return None
    a_idx, b_idx, d_idx, b_dup_idx, rest = chosen

    a_mask, b_mask, d_mask, fill_branch = _generate_operands(rng)
    masks: list[np.ndarray | None] = [None] * 6
    masks[a_idx] = a_mask
    masks[b_idx] = b_mask
    masks[d_idx] = d_mask
    if b_dup_idx is not None:
        masks[b_dup_idx] = b_mask.copy()

    existing_keys = {_mask_key(a_mask), _mask_key(b_mask), _mask_key(d_mask)}
    for pos in rest:
        for _ in range(200):
            candidate = _distractor_mask(rng)
            key = _mask_key(candidate)
            if key in existing_keys:
                continue
            masks[pos] = candidate
            existing_keys.add(key)
            break
        else:
            return None

    full_masks = [mask for mask in masks if mask is not None]
    assert len(full_masks) == 6
    full_masks = [np.asarray(mask, dtype=bool) for mask in full_masks]

    if not _band_coverage_visible(full_masks):
        return None

    candidates = _find_union_candidates(full_masks)
    expected = {(a_idx, b_idx, d_idx)}
    if b_dup_idx is not None:
        expected.add((a_idx, b_dup_idx, d_idx))
    if set(candidates) != expected:
        return None
    if _select_union_candidate(candidates) != (a_idx, b_idx, d_idx):
        return None

    return full_masks, {
        "a_idx": a_idx,
        "b_idx": b_idx,
        "d_idx": d_idx,
        "duplicate_branch": b_dup_idx is not None,
        "fill_branch": fill_branch,
    }


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _pair_matches_arrangement(
    blue_box: tuple[int, int, int, int],
    green_box: tuple[int, int, int, int],
    arrangement: str,
) -> bool:
    if arrangement == "side":
        return _overlap_len(blue_box[0], blue_box[2], green_box[0], green_box[2]) >= 2
    return _overlap_len(blue_box[1], blue_box[3], green_box[1], green_box[3]) >= 2


def _sample_layout(
    rng: np.random.Generator,
    blue_outer: tuple[int, int],
    green_outer: tuple[int, int],
) -> tuple[int, int, int, int, tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    blue_h, blue_w = blue_outer
    green_h, green_w = green_outer
    for _ in range(150):
        w_min = max(18, ARRAY_W + 3, blue_w + 2, green_w + 2)
        w = int(rng.integers(w_min, 31))
        array_c0 = int(rng.integers(1, w - ARRAY_W))
        array_r0 = int(rng.integers(1, 4))
        lower_gap = int(rng.integers(2, 5))
        min_frame_r = array_r0 + ARRAY_H + lower_gap
        h_min = max(18, min_frame_r + min(blue_h, green_h) + 1)
        h = int(rng.integers(h_min, 31))
        if min_frame_r + max(blue_h, green_h) > h:
            continue

        blue_candidates = [
            (r0, c0, r0 + blue_h - 1, c0 + blue_w - 1)
            for r0 in range(min_frame_r, h - blue_h + 1)
            for c0 in range(0, w - blue_w + 1)
        ]
        green_candidates = [
            (r0, c0, r0 + green_h - 1, c0 + green_w - 1)
            for r0 in range(min_frame_r, h - green_h + 1)
            for c0 in range(0, w - green_w + 1)
        ]
        if not blue_candidates or not green_candidates:
            continue

        arrangement = ("side", "stack")[int(rng.integers(0, 2))]
        for _ in range(120):
            blue_box = blue_candidates[int(rng.integers(0, len(blue_candidates)))]
            green_box = green_candidates[int(rng.integers(0, len(green_candidates)))]
            if not _boxes_separate(blue_box, green_box, gap=1):
                continue
            if not _pair_matches_arrangement(blue_box, green_box, arrangement):
                continue
            return h, w, array_r0, array_c0, blue_box, green_box
    return None


def _render_input(
    masks_by_pos: list[np.ndarray],
    h: int,
    w: int,
    array_r0: int,
    array_c0: int,
    blue_box: tuple[int, int, int, int],
    green_box: tuple[int, int, int, int],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for idx, mask in enumerate(masks_by_pos):
        off_r, off_c = TOP_CELL_ORIGINS[idx]
        r0 = array_r0 + off_r
        c0 = array_c0 + off_c
        grid[r0 : r0 + CELL, c0 : c0 + CELL][mask] = RED
    _draw_frame(grid, blue_box, BLUE)
    _draw_frame(grid, green_box, GREEN)
    return grid


def _expected_output(
    inp: np.ndarray,
    masks_by_pos: list[np.ndarray],
    meta: dict[str, int | bool],
    array_r0: int,
    array_c0: int,
    blue_box: tuple[int, int, int, int],
    green_box: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = inp.copy()

    a_idx = int(meta["a_idx"])
    b_idx = int(meta["b_idx"])
    d_idx = int(meta["d_idx"])

    d_r = array_r0 + TOP_CELL_ORIGINS[d_idx][0]
    d_c = array_c0 + TOP_CELL_ORIGINS[d_idx][1]
    out[d_r : d_r + CELL, d_c : d_c + CELL][masks_by_pos[d_idx]] = YELLOW

    blue_inner = _interior_bbox(blue_box)
    green_inner = _interior_bbox(green_box)
    blue_mask = _stretch_mask(
        masks_by_pos[a_idx], (blue_inner[2] - blue_inner[0] + 1, blue_inner[3] - blue_inner[1] + 1)
    )
    green_mask = _stretch_mask(
        masks_by_pos[b_idx], (green_inner[2] - green_inner[0] + 1, green_inner[3] - green_inner[1] + 1)
    )
    out[blue_inner[0] : blue_inner[2] + 1, blue_inner[1] : blue_inner[3] + 1][blue_mask] = BLUE
    out[green_inner[0] : green_inner[2] + 1, green_inner[1] : green_inner[3] + 1][green_mask] = GREEN
    return out, blue_mask, green_mask


def _case_is_valid(
    inp: np.ndarray,
    expected: np.ndarray,
    masks_by_pos: list[np.ndarray],
    meta: dict[str, int | bool],
    blue_box: tuple[int, int, int, int],
    green_box: tuple[int, int, int, int],
    blue_mask: np.ndarray,
    green_mask: np.ndarray,
) -> bool:
    try:
        windows, parsed_masks = _extract_sprite_windows(inp)
        _ = windows
        if any(not np.array_equal(a, b) for a, b in zip(parsed_masks, masks_by_pos)):
            return False

        candidates = _find_union_candidates(parsed_masks)
        selected = _select_union_candidate(candidates)
        if selected != (int(meta["a_idx"]), int(meta["b_idx"]), int(meta["d_idx"])):
            return False

        if bool(meta["duplicate_branch"]):
            if len(candidates) != 2:
                return False
        else:
            if len(candidates) != 1:
                return False

        if not np.any(blue_mask) or not np.any(green_mask):
            return False
        if np.array_equal(inp, expected):
            return False

        d_idx = int(meta["d_idx"])
        top_rows = _active_runs(np.any(inp == RED, axis=1))
        top_cols = _active_runs(np.any(inp == RED, axis=0))
        d_r = top_rows[0][0] + TOP_CELL_ORIGINS[d_idx][0]
        d_c = top_cols[0][0] + TOP_CELL_ORIGINS[d_idx][1]
        if not np.all(expected[d_r : d_r + CELL, d_c : d_c + CELL][masks_by_pos[d_idx]] == YELLOW):
            return False

        if bool(meta["fill_branch"]):
            a_idx = int(meta["a_idx"])
            b_idx = int(meta["b_idx"])
            if _is_solid_bbox(masks_by_pos[a_idx]):
                if not np.all(blue_mask):
                    return False
            if _is_solid_bbox(masks_by_pos[b_idx]):
                if not np.all(green_mask):
                    return False
        if np.all(blue_mask) and np.all(green_mask):
            return False

        actual = main(inp)
        return np.array_equal(actual, expected)
    except Exception:
        return False


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    mask_pack = _make_mask_assignment(rng)
    if mask_pack is None:
        return None
    masks_by_pos, meta = mask_pack

    for _ in range(120):
        blue_inner = (int(rng.integers(3, 7)), int(rng.integers(3, 7)))
        green_inner = (int(rng.integers(3, 7)), int(rng.integers(3, 7)))
        if blue_inner == green_inner:
            continue
        blue_outer = (blue_inner[0] + 2, blue_inner[1] + 2)
        green_outer = (green_inner[0] + 2, green_inner[1] + 2)
        layout = _sample_layout(rng, blue_outer, green_outer)
        if layout is None:
            continue
        h, w, array_r0, array_c0, blue_box, green_box = layout
        inp = _render_input(masks_by_pos, h, w, array_r0, array_c0, blue_box, green_box)
        expected, blue_mask, green_mask = _expected_output(
            inp, masks_by_pos, meta, array_r0, array_c0, blue_box, green_box
        )
        if _case_is_valid(inp, expected, masks_by_pos, meta, blue_box, green_box, blue_mask, green_mask):
            return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid input")
