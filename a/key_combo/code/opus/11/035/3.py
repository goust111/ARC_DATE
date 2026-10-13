from __future__ import annotations

import math
import random

import numpy as np


BG = 0
FILL = 1
MARK = 2

GRID_ROWS = (18, 24)
GRID_COLS = (18, 24)
RECT_CHOICES = ((3, 4), (3, 5), (4, 4), (4, 5))
MATCH_COUNT_CHOICES = (2, 3)
DISTRACTOR_COUNT_CHOICES = (1, 2)
PLACEMENT_GAP = 1
GEN_ATTEMPTS = 500


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _is_connected(mask: np.ndarray) -> bool:
    total = int(mask.sum())
    if total == 0:
        return False
    return len(_components(mask)) == 1


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for transformed in (rot, np.fliplr(rot)):
            key = (tuple(map(int, transformed.shape)), transformed.astype(np.uint8).tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(transformed.copy())
    return out


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _find_completion_candidates(
    grid: np.ndarray,
) -> list[tuple[int, tuple[int, int, int, int], frozenset[tuple[int, int]]]]:
    blue_parts = _components(grid == FILL)
    if len(blue_parts) != 2:
        return []

    comps = []
    for coords in blue_parts:
        bbox = _bbox_from_coords(coords)
        cells = frozenset((int(r), int(c)) for r, c in coords)
        comps.append((bbox, cells))

    left = [item for item in comps if item[0][1] == 0]
    right = [item for item in comps if item[0][1] != 0]
    if len(left) != 1 or len(right) != 1:
        return []

    fragment_bbox, fragment = left[0]
    _anchor_bbox, anchor = right[0]
    h, w = grid.shape
    candidates: list[tuple[int, tuple[int, int, int, int], frozenset[tuple[int, int]]]] = []

    for dc in range(1, w):
        moved = {(r, c + dc) for r, c in fragment}
        if any(c >= w for _r, c in moved):
            continue
        if any(int(grid[r, c]) != BG for r, c in moved):
            continue

        union = frozenset(moved | anchor)
        coords = np.asarray(list(union), dtype=int)
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        ok = True
        marker_count = 0
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if (rr, cc) in union:
                    continue
                if int(grid[rr, cc]) != MARK:
                    ok = False
                    break
                marker_count += 1
            if not ok:
                break
        if ok and marker_count > 0:
            candidates.append((dc, (r0, c0, r1, c1), union))

    return candidates


def _find_stamp_matches(
    grid: np.ndarray, marker_mask: np.ndarray
) -> list[tuple[tuple[int, int, int, int], frozenset[tuple[int, int]]]]:
    h_grid, w_grid = grid.shape
    matches: list[tuple[tuple[int, int, int, int], frozenset[tuple[int, int]]]] = []
    seen: set[frozenset[tuple[int, int]]] = set()

    for transformed in _unique_transforms(marker_mask):
        h, w = transformed.shape
        marker_total = int(transformed.sum())
        for r0 in range(h_grid - h + 1):
            for c0 in range(w_grid - w + 1):
                window = grid[r0 : r0 + h, c0 : c0 + w]
                if np.count_nonzero(window == MARK) != marker_total:
                    continue
                if np.any((window != BG) & (window != MARK)):
                    continue
                if not np.array_equal(window == MARK, transformed):
                    continue

                fill_cells = frozenset(
                    (r0 + rr, c0 + cc)
                    for rr in range(h)
                    for cc in range(w)
                    if not transformed[rr, cc]
                )
                if fill_cells in seen:
                    continue
                seen.add(fill_cells)
                matches.append(((r0, c0, r0 + h - 1, c0 + w - 1), fill_cells))

    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the left-edge blue fragment to complete its marked rectangle, then use that synthesized marker template to stamp matching locations.

    Input structure:
    - The background is black.
    - Blue cells form exactly two 4-connected objects: one fragment touches the left edge, and one anchor object sits away from the edge.
    - At the anchor site, red marker cells lie inside the rectangle that would be completed by translating the left fragment horizontally.
    - Elsewhere, 2-3 disjoint red pixel groups reproduce the completed rectangle's marker mask under rotation or reflection, with the rest of each candidate rectangle still black.

    Rule:
    - Find the unique horizontal translation of the left-edge blue fragment that, together with the anchor blue cells and the red cells already at that site, forms a fully occupied axis-aligned rectangle.
    - Erase the fragment from the left edge and redraw it at that completing offset.
    - Read the completed rectangle's red marker positions as a template mask, enumerate all distinct rotations and mirror variants, and search the original grid for every location where one transformed mask matches red cells exactly while every other cell of the corresponding rectangle is black.
    - For each such match, fill the black cells of that rectangle with blue and keep the red marker cells unchanged.

    Key insight:
    - The stamp template is not directly given as a full object; it only becomes fully defined after the left fragment is moved to reconstruct the hidden source rectangle.
    """
    g = np.asarray(grid, dtype=int)
    candidates = _find_completion_candidates(g)
    assert len(candidates) == 1, "expected a unique completing horizontal translation"
    dc, source_bbox, source_union = candidates[0]

    blue_parts = _components(g == FILL)
    fragment = [coords for coords in blue_parts if _bbox_from_coords(coords)[1] == 0][0]

    out = g.copy()
    for r, c in fragment:
        out[int(r), int(c)] = BG
        out[int(r), int(c) + dc] = FILL

    r0, c0, r1, c1 = source_bbox
    marker_mask = g[r0 : r1 + 1, c0 : c1 + 1] == MARK
    assert marker_mask.any(), "expected nonempty marker mask in the completed rectangle"

    for _bbox, fill_cells in _find_stamp_matches(g, marker_mask):
        for r, c in fill_cells:
            out[r, c] = FILL

    return out


def _sample_fragment_widths(h: int, w: int) -> list[int] | None:
    max_frag = max(1, w - 2)
    upper = min(max_frag, 2)
    if upper < 1:
        return None
    for _ in range(80):
        widths = [random.randint(1, upper)]
        for _row in range(1, h):
            lo = max(1, widths[-1] - 1)
            hi = min(upper, widths[-1] + 1)
            widths.append(random.randint(lo, hi))
        candidate_area = sum(w - widths[r] for r in range(1, h))
        if candidate_area >= max(3, math.ceil(0.2 * h * w)):
            return widths
    return None


def _sample_source_pattern(
    h: int, w: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    widths = _sample_fragment_widths(h, w)
    if widths is None:
        return None

    fragment_mask = np.zeros((h, w), dtype=bool)
    anchor_region = np.zeros((h, w), dtype=bool)
    for r in range(h):
        fragment_mask[r, : widths[r]] = True
        anchor_region[r, widths[r] :] = True

    candidates = [(r, c) for r in range(1, h) for c in range(widths[r], w)]
    area = h * w
    count_lo = max(3, math.ceil(area * 0.2))
    count_hi = min(len(candidates), math.floor(area * 0.4))
    if count_lo > count_hi or (h - 1, w - 1) not in candidates:
        return None

    forced = {(h - 1, w - 1)}
    pool = [cell for cell in candidates if cell not in forced]
    for _ in range(300):
        count = random.randint(count_lo, count_hi)
        chosen = set(random.sample(pool, count - len(forced))) | forced
        marker_mask = np.zeros((h, w), dtype=bool)
        for r, c in chosen:
            marker_mask[r, c] = True

        if marker_mask[0].any():
            continue
        if marker_mask[:, 0].any():
            continue
        if marker_mask[:, -1].sum() == 0:
            continue
        if len(_unique_transforms(marker_mask)) != 8:
            continue

        anchor_mask = anchor_region & ~marker_mask
        if not _is_connected(fragment_mask) or not _is_connected(anchor_mask):
            continue
        if not np.any(anchor_mask[:, -1]):
            continue
        if not np.any(anchor_mask[-1] | fragment_mask[-1]):
            continue
        return fragment_mask, anchor_mask, marker_mask

    return None


def _box_coords(r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    return r0, c0, r0 + h - 1, c0 + w - 1


def _place_box(
    h: int,
    w: int,
    rows: int,
    cols: int,
    reserved: list[tuple[int, int, int, int]],
    *,
    min_col: int = 0,
) -> tuple[int, int] | None:
    cells = [(r, c) for r in range(rows - h + 1) for c in range(min_col, cols - w + 1)]
    random.shuffle(cells)
    for r0, c0 in cells:
        box = _box_coords(r0, c0, h, w)
        if any(_boxes_too_close(box, old, PLACEMENT_GAP) for old in reserved):
            continue
        return r0, c0
    return None


def _paint_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    canvas[r0 : r0 + h, c0 : c0 + w] = patch


def _attempt_generate() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS)
    cols = random.randint(*GRID_COLS)
    rect_h, rect_w = random.choice(RECT_CHOICES)
    sampled = _sample_source_pattern(rect_h, rect_w)
    if sampled is None:
        return None
    fragment_mask, anchor_mask, marker_mask = sampled

    src_r0 = random.randint(0, rows - rect_h)
    src_c0_min = max(rect_w + 2, 5)
    src_c0_max = cols - rect_w - 1
    if src_c0_min > src_c0_max:
        return None
    src_c0 = random.randint(src_c0_min, src_c0_max)

    source_patch = np.full((rect_h, rect_w), FILL, dtype=int)
    source_patch[marker_mask] = MARK
    source_box = _box_coords(src_r0, src_c0, rect_h, rect_w)
    fragment_box = _box_coords(src_r0, 0, rect_h, rect_w)

    out = np.full((rows, cols), BG, dtype=int)
    _paint_patch(out, source_patch, src_r0, src_c0)

    reserved = [source_box, fragment_box]
    inp = out.copy()
    inp[src_r0 : src_r0 + rect_h, src_c0 : src_c0 + rect_w][fragment_mask] = BG
    inp[src_r0 : src_r0 + rect_h, :rect_w][fragment_mask] = FILL

    transforms = _unique_transforms(marker_mask)
    non_identity = [mask for mask in transforms if not np.array_equal(mask, marker_mask)]
    if not non_identity:
        return None

    target_count = random.choice(MATCH_COUNT_CHOICES)
    target_boxes: list[tuple[int, int, int, int]] = []
    chosen_transforms = [random.choice(non_identity)]
    while len(chosen_transforms) < target_count:
        chosen_transforms.append(random.choice(transforms))

    for transformed in chosen_transforms:
        h, w = transformed.shape
        loc = _place_box(h, w, rows, cols, reserved, min_col=1)
        if loc is None:
            return None
        r0, c0 = loc
        patch = np.full((h, w), FILL, dtype=int)
        patch[transformed] = MARK
        _paint_patch(out, patch, r0, c0)
        inp[r0 : r0 + h, c0 : c0 + w] = np.where(transformed, MARK, BG)
        box = _box_coords(r0, c0, h, w)
        reserved.append(box)
        target_boxes.append(box)

    distractor_count = random.choice(DISTRACTOR_COUNT_CHOICES)
    for _ in range(distractor_count):
        cells = [(r, c) for r in range(rows) for c in range(cols) if int(inp[r, c]) == BG]
        random.shuffle(cells)
        placed = False
        for r, c in cells:
            if any(box[0] <= r <= box[2] and box[1] <= c <= box[3] for box in target_boxes + [source_box, fragment_box]):
                continue
            if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in zip(*np.where(inp == MARK))):
                continue
            inp[r, c] = MARK
            out[r, c] = MARK
            placed = True
            break
        if not placed:
            return None

    blue_parts = _components(inp == FILL)
    if len(blue_parts) != 2:
        return None
    if sum(1 for coords in blue_parts if _bbox_from_coords(coords)[1] == 0) != 1:
        return None

    completions = _find_completion_candidates(inp)
    if len(completions) != 1:
        return None
    _dc, bbox, _union = completions[0]
    if bbox != source_box:
        return None

    found_matches = {box for box, _fill in _find_stamp_matches(inp, marker_mask)}
    if found_matches != set(target_boxes):
        return None
    if len(found_matches) != target_count:
        return None
    if len(_unique_transforms(marker_mask)) != 8:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
