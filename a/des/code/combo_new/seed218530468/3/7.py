from collections import deque
import random

import numpy as np


BG = 0
RED = 2
MAGENTA = 6
AZURE = 8
FILL_COLORS = (1, 3, 4, 5, 7, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[dict[str, object]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )

    return comps


def _bbox_of_true(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1 if np.any(mask) else False


def _find_separator_row(grid: np.ndarray) -> int:
    rows = []
    h = grid.shape[0]
    for r in range(h):
        value = int(grid[r, 0])
        if value == BG or not np.all(grid[r] == value):
            continue
        if np.any(grid[:r] != BG) and np.any(grid[r + 1 :] != BG):
            rows.append(r)
    assert len(rows) == 1, "expected exactly one horizontal separator row"
    return int(rows[0])


def _extract_stamp(top: np.ndarray) -> np.ndarray:
    rects = []
    for comp in _components(top != BG):
        r0, c0, r1, c1 = comp["bbox"]
        rects.append((comp["bbox"], top[r0 : r1 + 1, c0 : c1 + 1]))

    assert len(rects) == 2, "expected exactly two upper pattern blocks"
    rects.sort(key=lambda item: item[0][1])
    (_, left), (_, right) = rects
    assert left.shape == right.shape, "upper blocks must have the same size"

    diff = (left != BG) & (right == BG)
    r0, c0, r1, c1 = _bbox_of_true(diff)
    return diff[r0 : r1 + 1, c0 : c1 + 1]


def _find_marker_color(lower: np.ndarray) -> int:
    candidates = []
    for color in sorted(set(map(int, lower.ravel())) - {BG}):
        comps = _components(lower == color)
        if comps and all(int(comp["area"]) == 1 for comp in comps):
            candidates.append(color)
    assert len(candidates) == 1, "expected exactly one singleton marker color"
    return int(candidates[0])


def _stamp_lower(lower: np.ndarray, stamp: np.ndarray, marker_color: int) -> np.ndarray:
    out = lower.copy()
    region_mask = lower != BG
    for mr, mc in np.argwhere(lower == marker_color):
        mr = int(mr)
        mc = int(mc)
        out[mr, mc] = MAGENTA
        for sr, sc in np.argwhere(stamp):
            rr = mr + int(sr)
            cc = mc + int(sc)
            if 0 <= rr < lower.shape[0] and 0 <= cc < lower.shape[1] and region_mask[rr, cc]:
                out[rr, cc] = MAGENTA
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a hidden stamp from the top pair and imprint it at every marked point in the lower shape.

    Input structure:
    - The background is black, and one full-width azure row separates the grid into upper and lower zones.
    - Above the separator sit two same-sized side-by-side colored rectangles with a few black-vs-colored disagreements.
    - Below the separator is one larger colored region containing several single-pixel markers of another color.

    Rule:
    - Compare the two upper rectangles and collect the cells where the left rectangle is colored while the right rectangle is black.
    - Crop that left-only difference to its tight bounding box; this cropped boolean mask is the stamp.
    - For every lower marker, place the stamp with the marker as the stamp bbox's top-left anchor, paint all covered in-region cells magenta, and also recolor the marker cell itself magenta.
    - Clip every stamp to the lower colored region, and leave the upper area plus the separator unchanged.

    Key insight:
    - The stamp is not drawn explicitly; it is encoded as the left-only part of a small find-the-difference pair, and the lower markers specify where that hidden shape must be applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator_row(g)
    stamp = _extract_stamp(g[:sep])
    lower = g[sep + 1 :]
    marker_color = _find_marker_color(lower)

    out = g.copy()
    out[sep + 1 :] = _stamp_lower(lower, stamp, marker_color)
    return out


def _make_stamp_mask(
    rng: random.Random,
    rect_h: int,
    rect_w: int,
    anchor_filled: bool,
    need_clip: bool,
) -> np.ndarray | None:
    max_h = min(4, rect_h - 2)
    max_w = min(5, rect_w - 2)
    min_h = 1 if anchor_filled else 2
    min_w = 1 if anchor_filled else 2
    if max_h < min_h or max_w < min_w:
        return None

    for _ in range(200):
        sh = rng.randint(min_h, max_h)
        sw = rng.randint(min_w, max_w)
        if need_clip and sh == 1 and sw == 1:
            continue

        forced: set[tuple[int, int]] = set()
        if anchor_filled:
            forced.add((0, 0))
        else:
            forced.add((0, rng.randrange(1, sw)))
            forced.add((rng.randrange(1, sh), 0))
        forced.add((sh - 1, rng.randrange(sw)))
        forced.add((rng.randrange(sh), sw - 1))

        max_area = sh * sw
        area_hi = max_area if rng.random() < 0.3 else max(len(forced), max_area - 1)
        target_area = rng.randint(len(forced), area_hi)

        remaining = [(r, c) for r in range(sh) for c in range(sw) if (r, c) not in forced]
        rng.shuffle(remaining)
        cells = set(forced)
        for cell in remaining[: max(0, target_area - len(cells))]:
            cells.add(cell)

        mask = np.zeros((sh, sw), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if bool(mask[0, 0]) != anchor_filled:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        return mask

    return None


def _make_region_mask(rng: random.Random, height: int, width: int) -> np.ndarray | None:
    if height < 8 or width < 12:
        return None

    for _ in range(200):
        mask = np.zeros((height, width), dtype=bool)
        main_h = rng.randint(max(6, height // 2), height - 2)
        main_w = rng.randint(max(8, width // 2), width - 2)
        r0 = rng.randint(1, height - main_h - 1)
        c0 = rng.randint(1, width - main_w - 1)
        r1 = r0 + main_h
        c1 = c0 + main_w
        mask[r0:r1, c0:c1] = True

        sides = ["top", "bottom", "left", "right"]
        rng.shuffle(sides)
        for side in sides[: rng.randint(0, 2)]:
            if side in ("top", "bottom"):
                max_depth = min(3, main_h - 2)
                max_span = main_w - 2
                if max_depth < 1 or max_span < 2:
                    continue
                depth = rng.randint(1, max_depth)
                span = rng.randint(2, max_span)
                start = rng.randint(c0 + 1, c1 - span - 1)
                rows = slice(r0, r0 + depth) if side == "top" else slice(r1 - depth, r1)
                mask[rows, start : start + span] = False
            else:
                max_depth = min(3, main_w - 2)
                max_span = main_h - 2
                if max_depth < 1 or max_span < 2:
                    continue
                depth = rng.randint(1, max_depth)
                span = rng.randint(2, max_span)
                start = rng.randint(r0 + 1, r1 - span - 1)
                cols = slice(c0, c0 + depth) if side == "left" else slice(c1 - depth, c1)
                mask[start : start + span, cols] = False

        if int(mask.sum()) < 45:
            continue
        if _is_connected(mask):
            return mask

    return None


def _stamp_fits(region_mask: np.ndarray, anchor: tuple[int, int], stamp: np.ndarray) -> bool:
    ar, ac = anchor
    for sr, sc in np.argwhere(stamp):
        rr = ar + int(sr)
        cc = ac + int(sc)
        if not (0 <= rr < region_mask.shape[0] and 0 <= cc < region_mask.shape[1]):
            return False
        if not region_mask[rr, cc]:
            return False
    return True


def _pick_spaced(
    rng: random.Random,
    cells: list[tuple[int, int]],
    count: int,
    min_gap: int,
    seed: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    order = list(cells)
    rng.shuffle(order)
    picked = [] if seed is None else list(seed)

    for r, c in order:
        if (r, c) in picked:
            continue
        if all(max(abs(r - pr), abs(c - pc)) >= min_gap for pr, pc in picked):
            picked.append((r, c))
            if len(picked) == count:
                return picked

    return picked if len(picked) == count else None


def _render_expected(
    grid: np.ndarray,
    sep_row: int,
    stamp: np.ndarray,
    markers: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    lower = out[sep_row + 1 :]
    region_mask = lower != BG
    for mr, mc in markers:
        lower[mr, mc] = MAGENTA
        for sr, sc in np.argwhere(stamp):
            rr = mr + int(sr)
            cc = mc + int(sc)
            if 0 <= rr < lower.shape[0] and 0 <= cc < lower.shape[1] and region_mask[rr, cc]:
                lower[rr, cc] = MAGENTA
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    top_color, region_color = rng.sample(FILL_COLORS, 2)
    clip_branch = rng.random() < 0.5
    anchor_filled = rng.random() < 0.5
    extra_branch = rng.random() < 0.5

    rect_h = rng.randint(5, 8)
    rect_w = rng.randint(6, 10)
    stamp = _make_stamp_mask(rng, rect_h, rect_w, anchor_filled, clip_branch)
    if stamp is None:
        return None
    stamp_h, stamp_w = stamp.shape

    top_margin = rng.randint(1, 3)
    between_gap = rng.randint(2, 5)
    below_gap = rng.randint(1, 2)
    min_width = 2 * rect_w + between_gap + 2
    if min_width > 28:
        return None
    width = rng.randint(max(18, min_width), 28)
    left_col = rng.randint(1, width - (2 * rect_w + between_gap) - 1)
    right_col = left_col + rect_w + between_gap

    sep_row = top_margin + rect_h + below_gap
    lower_h_lo = max(8, stamp_h + 5)
    lower_h_hi = min(15, 29 - sep_row)
    if lower_h_lo > lower_h_hi:
        return None
    lower_h = rng.randint(lower_h_lo, lower_h_hi)
    height = sep_row + 1 + lower_h

    grid = np.full((height, width), BG, dtype=int)
    grid[sep_row, :] = AZURE

    left_rect = np.full((rect_h, rect_w), top_color, dtype=int)
    right_rect = np.full((rect_h, rect_w), top_color, dtype=int)

    offset_r = rng.randint(1, rect_h - stamp_h - 1)
    offset_c = rng.randint(1, rect_w - stamp_w - 1)
    right_patch = right_rect[offset_r : offset_r + stamp_h, offset_c : offset_c + stamp_w]
    right_patch[stamp] = BG

    interior = [(r, c) for r in range(1, rect_h - 1) for c in range(1, rect_w - 1)]
    occupied = {
        (offset_r + int(sr), offset_c + int(sc))
        for sr, sc in np.argwhere(stamp)
    }
    common_pool = [cell for cell in interior if cell not in occupied]
    rng.shuffle(common_pool)
    for r, c in common_pool[: rng.randint(0, min(3, len(common_pool)))]:
        left_rect[r, c] = BG
        right_rect[r, c] = BG

    if extra_branch:
        extra_pool = [cell for cell in interior if cell not in occupied and left_rect[cell] != BG]
        rng.shuffle(extra_pool)
        for r, c in extra_pool[: rng.randint(1, min(4, len(extra_pool)))]:
            left_rect[r, c] = BG

    if not (_is_connected(left_rect != BG) and _is_connected(right_rect != BG)):
        return None

    row0 = top_margin
    grid[row0 : row0 + rect_h, left_col : left_col + rect_w] = left_rect
    grid[row0 : row0 + rect_h, right_col : right_col + rect_w] = right_rect

    region_mask = _make_region_mask(rng, lower_h, width)
    if region_mask is None:
        return None

    lower = grid[sep_row + 1 :]
    lower[region_mask] = region_color

    region_cells = [tuple(map(int, cell)) for cell in np.argwhere(region_mask)]
    safe_cells = [cell for cell in region_cells if _stamp_fits(region_mask, cell, stamp)]
    clip_cells = [cell for cell in region_cells if not _stamp_fits(region_mask, cell, stamp)]
    if not safe_cells or (clip_branch and not clip_cells):
        return None

    min_gap = max(2, max(stamp_h, stamp_w) - 1)
    markers = None
    counts = [2, 3, 4]
    rng.shuffle(counts)
    for marker_count in counts:
        if clip_branch:
            seed = [rng.choice(clip_cells)]
            markers = _pick_spaced(rng, safe_cells + clip_cells, marker_count, min_gap, seed=seed)
        else:
            markers = _pick_spaced(rng, safe_cells, marker_count, min_gap)
        if markers is not None:
            break
    if markers is None:
        return None

    for r, c in markers:
        lower[r, c] = RED

    expected = _render_expected(grid, sep_row, stamp, markers)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_ATTEMPTS} attempts")
