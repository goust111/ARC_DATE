from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

PARTIAL_COLORS = [BLUE, GREEN, YELLOW, TEAL, MAROON]
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    cells = {tuple(map(int, rc)) for rc in coords}
    stack = [next(iter(cells))]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _anchor_offset(canvas_len: int, obj_len: int, anchor_idx: int) -> int:
    if anchor_idx == 0:
        return 0
    if anchor_idx == 1:
        return (canvas_len - obj_len) // 2
    if anchor_idx == 2:
        return canvas_len - obj_len
    raise ValueError(f"invalid anchor index: {anchor_idx}")


def _find_marker_anchor(right: np.ndarray) -> tuple[int, int]:
    h, w = right.shape
    matches: list[tuple[int, int]] = []
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            patch = right[r0 : r0 + 5, c0 : c0 + 5]
            border_ok = (
                np.all(patch[0, :] == GREY)
                and np.all(patch[4, :] == GREY)
                and np.all(patch[:, 0] == GREY)
                and np.all(patch[:, 4] == GREY)
            )
            if not border_ok:
                continue
            inner = patch[1:4, 1:4]
            red_cells = np.argwhere(inner == RED)
            if red_cells.shape[0] != 1:
                continue
            if np.any((inner != BLACK) & (inner != RED)):
                continue
            rr, cc = map(int, red_cells[0])
            matches.append((rr, cc))
    assert len(matches) == 1, "expected exactly one 5x5 marker square"
    return matches[0]


def _recover_mask(left: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    colors = [int(c) for c in np.unique(left) if int(c) != BLACK]
    assert 3 <= len(colors) <= 4, "expected 3-4 partial copies"
    partials: list[np.ndarray] = []
    for color in colors:
        local_mask = left == color
        box = _bbox(local_mask)
        assert box is not None, "foreground color must be present"
        r0, c0, r1, c1 = box
        crop = local_mask[r0 : r1 + 1, c0 : c1 + 1]
        assert _is_connected(crop), "each partial sprite must stay 4-connected"
        partials.append(crop)

    full_h = max(mask.shape[0] for mask in partials)
    full_w = max(mask.shape[1] for mask in partials)
    full_mask = np.zeros((full_h, full_w), dtype=bool)
    for part in partials:
        h, w = part.shape
        full_mask[:h, :w] |= part
    assert np.any(full_mask), "recovered mask must be non-empty"
    return full_mask, partials


def _build_composite(full_mask: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    mh, mw = full_mask.shape
    fh, fw = frame_mask.shape
    composite = np.zeros((fh * mh, fw * mw), dtype=int)
    for fr in range(fh):
        for fc in range(fw):
            if not frame_mask[fr, fc]:
                continue
            r0 = fr * mh
            c0 = fc * mw
            block = composite[r0 : r0 + mh, c0 : c0 + mw]
            block[full_mask] = RED
    return composite


def _stamp_clipped(canvas: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    ch, cw = canvas.shape
    sh, sw = sprite.shape
    src_r0 = max(0, -top)
    src_c0 = max(0, -left)
    dst_r0 = max(0, top)
    dst_c0 = max(0, left)
    h = min(sh - src_r0, ch - dst_r0)
    w = min(sw - src_c0, cw - dst_c0)
    if h <= 0 or w <= 0:
        return
    patch = sprite[src_r0 : src_r0 + h, src_c0 : src_c0 + w]
    mask = patch != BLACK
    canvas[dst_r0 : dst_r0 + h, dst_c0 : dst_c0 + w][mask] = patch[mask]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    sep_cols = [c for c in range(w) if np.all(g[:, c] == PINK)]
    assert len(sep_cols) == 1, "expected one full magenta separator column"
    sep = sep_cols[0]
    left = g[:, :sep]
    right = g[:, sep + 1 :]

    full_mask, _partials = _recover_mask(left)

    frame_bbox = _bbox(right == ORANGE)
    assert frame_bbox is not None, "expected an orange frame template"
    fr0, fc0, fr1, fc1 = frame_bbox
    frame_mask = right[fr0 : fr1 + 1, fc0 : fc1 + 1] == ORANGE
    assert 3 <= int(frame_mask.sum()) <= 6, "frame activity count must be 3-6"

    anchor_r, anchor_c = _find_marker_anchor(right)
    composite = _build_composite(full_mask, frame_mask)

    out = np.zeros_like(right)
    top = _anchor_offset(out.shape[0], composite.shape[0], anchor_r)
    left_offset = _anchor_offset(out.shape[1], composite.shape[1], anchor_c)
    _stamp_clipped(out, composite, top, left_offset)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a complete mask from complementary partial sprites, tile it through a frame, and anchor-place the result.

    Input structure:
    - A full magenta separator column splits the grid into a left partial-sprite region and a right control region.
    - The left region contains 3-4 vertically separated single-color partial copies of one common sprite window.
    - The right region contains an orange/black frame template plus a 5x5 grey border square whose lone red inner pixel encodes anchor position.

    Rule:
    - Crop each partial copy to its bounding box, align those crops at the top-left, and OR them to recover the full binary mask.
    - Replace every orange frame cell by one red copy of the recovered mask and every inactive frame cell by an equally sized black block.
    - Output a black canvas with the same shape as the right region and place the tiled composite so its top/center/bottom and left/center/right anchor matches the marker square, clipping any overflow.

    Key insight:
    - No single left-side copy is complete; only their top-left-aligned union reveals the mask that must then be reused as the tiling unit before anchor-based placement.
    """
    return _solve(grid)


def _grow_connected_cells(height: int, width: int, area: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _cells_to_mask(cells: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_full_mask(num_copies: int) -> np.ndarray | None:
    for _ in range(120):
        mh = random.randint(2, 4)
        mw = random.randint(2, 4)
        max_area = min(8, mh * mw)
        min_area = max(num_copies + 1, 5, max(mh, mw))
        if min_area > max_area:
            continue
        area = random.randint(min_area, max_area)
        cells = _grow_connected_cells(mh, mw, area)
        if cells is None:
            continue
        cells = _normalize_cells(cells)
        if max(r for r, _ in cells) != mh - 1 or max(c for _, c in cells) != mw - 1:
            continue
        mask = _cells_to_mask(cells, (mh, mw))
        if not all(mask[r, :].any() for r in range(mh)):
            continue
        if not all(mask[:, c].any() for c in range(mw)):
            continue
        return mask
    return None


def _subset_connected(indices: frozenset[int], coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    mask = np.zeros(shape, dtype=bool)
    for idx in indices:
        r, c = coords[idx]
        mask[r, c] = True
    return _is_connected(mask)


def _subset_shape(indices: frozenset[int], coords: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [coords[idx][0] for idx in indices]
    cols = [coords[idx][1] for idx in indices]
    return max(rows) + 1, max(cols) + 1


def _enumerate_partial_candidates(full_mask: np.ndarray) -> list[frozenset[int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(full_mask)]
    area = len(coords)
    max_size = min(7, area - 1)
    candidates: list[frozenset[int]] = []
    for size in range(3, max_size + 1):
        for combo in combinations(range(area), size):
            subset = frozenset(int(idx) for idx in combo)
            rows = {coords[idx][0] for idx in subset}
            cols = {coords[idx][1] for idx in subset}
            if 0 not in rows or 0 not in cols:
                continue
            if not _subset_connected(subset, coords, full_mask.shape):
                continue
            candidates.append(subset)
    return candidates


def _choose_partial_masks(full_mask: np.ndarray, num_copies: int) -> list[np.ndarray] | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(full_mask)]
    area = len(coords)
    candidates = _enumerate_partial_candidates(full_mask)
    if len(candidates) < num_copies:
        return None

    full_set = set(range(area))
    for _ in range(240):
        chosen = random.sample(candidates, num_copies)
        union = set().union(*chosen)
        if union != full_set:
            continue
        necessary = True
        for idx, subset in enumerate(chosen):
            others = set().union(*(chosen[j] for j in range(num_copies) if j != idx))
            if others == full_set:
                necessary = False
                break
        if not necessary:
            continue
        full_h, full_w = full_mask.shape
        smaller_exists = any(_subset_shape(subset, coords) != (full_h, full_w) for subset in chosen)
        if not smaller_exists:
            continue
        masks: list[np.ndarray] = []
        for subset in chosen:
            part = np.zeros_like(full_mask, dtype=bool)
            for point_idx in subset:
                r, c = coords[point_idx]
                part[r, c] = True
            masks.append(part)
        return masks
    return None


def _random_frame_mask() -> np.ndarray | None:
    for _ in range(120):
        fh = random.randint(3, 4)
        fw = random.randint(3, 4)
        min_active = max(3, fh, fw)
        max_active = min(6, fh * fw - 1)
        if min_active > max_active:
            continue
        area = random.randint(min_active, max_active)
        cells = _grow_connected_cells(fh, fw, area)
        if cells is None:
            continue
        mask = _cells_to_mask(cells, (fh, fw))
        if not all(mask[r, :].any() for r in range(fh)):
            continue
        if not all(mask[:, c].any() for c in range(fw)):
            continue
        return mask
    return None


def _sample_canvas_dim(required: int, obj: int, upper: int = 15) -> int | None:
    if required > upper:
        return None
    options: list[tuple[int, int]] = []
    if required < obj:
        options.append((required, min(upper, obj - 1)))
    if obj <= upper:
        options.append((max(required, obj), upper))
    if not options:
        options.append((required, upper))
    low, high = random.choice(options)
    if low > high:
        return None
    return random.randint(low, high)


def _render_right_region(
    right_h: int,
    right_w: int,
    frame_mask: np.ndarray,
    frame_pos: tuple[int, int],
    anchor: tuple[int, int],
    marker_pos: tuple[int, int],
) -> np.ndarray:
    right = np.zeros((right_h, right_w), dtype=int)
    fr0, fc0 = frame_pos
    fh, fw = frame_mask.shape
    frame_block = right[fr0 : fr0 + fh, fc0 : fc0 + fw]
    frame_block[frame_mask] = ORANGE

    mr0, mc0 = marker_pos
    right[mr0 : mr0 + 5, mc0 : mc0 + 5] = BLACK
    right[mr0, mc0 : mc0 + 5] = GREY
    right[mr0 + 4, mc0 : mc0 + 5] = GREY
    right[mr0 : mr0 + 5, mc0] = GREY
    right[mr0 : mr0 + 5, mc0 + 4] = GREY
    right[mr0 + 1 + anchor[0], mc0 + 1 + anchor[1]] = RED
    return right


def _render_left_region(
    grid_h: int,
    left_w: int,
    partial_masks: list[np.ndarray],
    colors: list[int],
    copy_rows: list[int],
    copy_col: int,
) -> np.ndarray:
    left = np.zeros((grid_h, left_w), dtype=int)
    for part, color, row0 in zip(partial_masks, colors, copy_rows):
        coords = np.argwhere(part)
        for r, c in coords:
            left[row0 + int(r), copy_col + int(c)] = int(color)
    return left


def _expected_output(
    full_mask: np.ndarray,
    frame_mask: np.ndarray,
    anchor: tuple[int, int],
    canvas_shape: tuple[int, int],
) -> np.ndarray:
    composite = _build_composite(full_mask, frame_mask)
    out = np.zeros(canvas_shape, dtype=int)
    top = _anchor_offset(canvas_shape[0], composite.shape[0], anchor[0])
    left = _anchor_offset(canvas_shape[1], composite.shape[1], anchor[1])
    _stamp_clipped(out, composite, top, left)
    return out


def _semantic_checks(
    inp: np.ndarray,
    full_mask: np.ndarray,
    partial_masks: list[np.ndarray],
    frame_mask: np.ndarray,
    anchor: tuple[int, int],
) -> bool:
    sep = int(np.where(np.all(inp == PINK, axis=0))[0][0])
    canvas_shape = (inp.shape[0], inp.shape[1] - sep - 1)
    expected = _expected_output(full_mask, frame_mask, anchor, canvas_shape)
    if not np.array_equal(main(inp), expected):
        return False

    if any(int(mask.sum()) < 3 or int(mask.sum()) > 7 for mask in partial_masks):
        return False
    if any(np.array_equal(mask, full_mask) for mask in partial_masks):
        return False

    union = np.zeros_like(full_mask, dtype=bool)
    for part in partial_masks:
        union |= part
    if not np.array_equal(union, full_mask):
        return False

    for idx in range(len(partial_masks)):
        partial_union = np.zeros_like(full_mask, dtype=bool)
        for j, other in enumerate(partial_masks):
            if j != idx:
                partial_union |= other
        if np.array_equal(partial_union, full_mask):
            return False

    full_shape = full_mask.shape
    smaller = False
    for part in partial_masks:
        box = _bbox(part)
        if box is None:
            return False
        if (box[2] - box[0] + 1, box[3] - box[1] + 1) != full_shape:
            smaller = True
    if not smaller:
        return False

    if not (3 <= int(frame_mask.sum()) <= 6):
        return False
    if int(frame_mask.sum()) == frame_mask.size:
        return False
    if not all(frame_mask[r, :].any() for r in range(frame_mask.shape[0])):
        return False
    if not all(frame_mask[:, c].any() for c in range(frame_mask.shape[1])):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    num_copies = random.randint(3, 4)
    full_mask = _random_full_mask(num_copies)
    if full_mask is None:
        return None

    partial_masks = _choose_partial_masks(full_mask, num_copies)
    if partial_masks is None:
        return None

    frame_mask = _random_frame_mask()
    if frame_mask is None:
        return None

    anchor = (random.randint(0, 2), random.randint(0, 2))
    full_h, full_w = full_mask.shape
    frame_h, frame_w = frame_mask.shape
    comp_h = full_h * frame_h
    comp_w = full_w * frame_w

    top_margin = random.randint(0, 1)
    gap = random.randint(1, 3)
    bottom_margin = random.randint(0, 2)
    required_right_h = top_margin + frame_h + gap + 5 + bottom_margin
    required_right_w = max(5, frame_w)

    right_h = _sample_canvas_dim(required_right_h, comp_h)
    right_w = _sample_canvas_dim(required_right_w, comp_w)
    if right_h is None or right_w is None:
        return None

    frame_r0 = top_margin
    frame_c0 = random.randint(0, right_w - frame_w)
    marker_r0 = top_margin + frame_h + gap
    marker_c0 = random.randint(0, right_w - 5)
    right = _render_right_region(
        right_h,
        right_w,
        frame_mask,
        (frame_r0, frame_c0),
        anchor,
        (marker_r0, marker_c0),
    )

    left_w = full_w + 2
    copy_col = 1
    stack_top = random.randint(0, 1)
    row_gap = random.randint(2, 3)
    copy_rows = [stack_top + idx * (full_h + row_gap) for idx in range(num_copies)]
    grid_h = max(right_h, copy_rows[-1] + full_h + random.randint(0, 1))
    total_w = left_w + 1 + right_w
    if total_w > 30 or grid_h > 30:
        return None

    colors = random.sample(PARTIAL_COLORS, num_copies)
    left = _render_left_region(grid_h, left_w, partial_masks, colors, copy_rows, copy_col)
    right_full = np.zeros((grid_h, right_w), dtype=int)
    right_full[:right_h, :] = right

    grid = np.zeros((grid_h, total_w), dtype=int)
    grid[:, :left_w] = left
    grid[:, left_w] = PINK
    grid[:, left_w + 1 :] = right_full

    if not _semantic_checks(grid, full_mask, partial_masks, frame_mask, anchor):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
