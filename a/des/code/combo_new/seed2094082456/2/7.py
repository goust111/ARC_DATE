import random
from collections import Counter, deque

import numpy as np


BG = 0
RED = 2
GRAY = 5
PATTERN_COLORS = tuple(color for color in range(10) if color not in {BG, RED, GRAY})

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = [UP, DOWN, LEFT, RIGHT]

GEN_ATTEMPTS = 400
LOCAL_PLACE_ATTEMPTS = 180


def neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(max(counts, key=counts.get))


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        pts: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            pts.append((cr, cc))
            for nr, nc in neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def rect_border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _parse_arrow_component(coords: np.ndarray) -> tuple[int, int] | None:
    pts = [(int(r), int(c)) for r, c in coords]
    if len(pts) != 4:
        return None

    row_counts = Counter(r for r, _c in pts)
    col_counts = Counter(c for _r, c in pts)

    shaft_rows = [row for row, count in row_counts.items() if count == 3]
    if len(shaft_rows) == 1:
        shaft_row = shaft_rows[0]
        shaft = sorted((r, c) for r, c in pts if r == shaft_row)
        cols = [c for _r, c in shaft]
        if cols == list(range(cols[0], cols[0] + 3)):
            head = next((r, c) for r, c in pts if r != shaft_row)
            if abs(head[0] - shaft_row) == 1 and head[1] in {cols[0], cols[-1]}:
                return UP if head[0] < shaft_row else DOWN

    shaft_cols = [col for col, count in col_counts.items() if count == 3]
    if len(shaft_cols) == 1:
        shaft_col = shaft_cols[0]
        shaft = sorted((r, c) for r, c in pts if c == shaft_col)
        rows = [r for r, _c in shaft]
        if rows == list(range(rows[0], rows[0] + 3)):
            head = next((r, c) for r, c in pts if c != shaft_col)
            if abs(head[1] - shaft_col) == 1 and head[0] in {rows[0], rows[-1]}:
                return LEFT if head[1] < shaft_col else RIGHT

    return None


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for coords in components_mask(grid == GRAY):
        r0, c0, r1, c1 = bbox_coords(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = set(rect_border_cells(r0, c0, r1, c1))
        pts = {(int(r), int(c)) for r, c in coords}
        if pts != border:
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not np.all(inner == bg):
            continue
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
            }
        )
    assert frames, "expected at least one gray hollow frame"
    return frames


def _find_patterns(grid: np.ndarray, bg: int) -> list[dict]:
    red_components = components_mask(grid == RED)
    arrows: list[dict] = []
    arrow_lookup: dict[tuple[int, int], int] = {}
    for arrow_id, coords in enumerate(red_components):
        direction = _parse_arrow_component(coords)
        assert direction is not None, "every red component must be a valid arrow"
        cells = {(int(r), int(c)) for r, c in coords}
        arrows.append({"dir": direction, "cells": cells})
        for cell in cells:
            arrow_lookup[cell] = arrow_id

    special = {bg, RED, GRAY}
    patterns: list[dict] = []
    used_arrows: set[int] = set()
    for color in np.unique(grid):
        color = int(color)
        if color in special:
            continue
        comps = components_mask(grid == color)
        assert len(comps) == 1, "each pattern color must form exactly one component"
        coords = comps[0]
        r0, c0, r1, c1 = bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert 2 <= h <= 3 and 2 <= w <= 3, "pattern bbox must be between 2x2 and 3x3"

        touching_arrows: set[int] = set()
        for r, c in coords:
            for nr, nc in neighbors4(int(r), int(c)):
                arrow_id = arrow_lookup.get((nr, nc))
                if arrow_id is not None:
                    touching_arrows.add(arrow_id)
        assert len(touching_arrows) == 1, "each pattern must touch exactly one arrow"
        arrow_id = next(iter(touching_arrows))
        used_arrows.add(arrow_id)

        patch = np.where(grid[r0 : r1 + 1, c0 : c1 + 1] == color, color, bg)
        patterns.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "dir": arrows[arrow_id]["dir"],
            }
        )

    assert patterns, "expected at least one source pattern"
    assert used_arrows == set(range(len(arrows))), "all arrows must attach to a pattern"
    return patterns


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _match_frame(pattern: dict, frames: list[dict]) -> int:
    r0, c0, r1, c1 = pattern["bbox"]
    direction = pattern["dir"]
    candidates: list[tuple[int, int]] = []

    for idx, frame in enumerate(frames):
        fr0, fc0, fr1, fc1 = frame["bbox"]
        if direction == RIGHT:
            if fc0 > c1 and _ranges_overlap(r0, r1, fr0, fr1):
                candidates.append((fc0 - c1, idx))
        elif direction == LEFT:
            if fc1 < c0 and _ranges_overlap(r0, r1, fr0, fr1):
                candidates.append((c0 - fc1, idx))
        elif direction == DOWN:
            if fr0 > r1 and _ranges_overlap(c0, c1, fc0, fc1):
                candidates.append((fr0 - r1, idx))
        else:
            assert direction == UP
            if fr1 < r0 and _ranges_overlap(c0, c1, fc0, fc1):
                candidates.append((r0 - fr1, idx))

    assert candidates, "each pattern must hit at least one frame in its arrow direction"
    candidates.sort()
    if len(candidates) > 1:
        assert candidates[0][0] < candidates[1][0], "ray match must be unambiguous"
    return int(candidates[0][1])


def _periodic_indices(length: int, period: int, anchor_end: bool) -> list[int]:
    offset = (-length) % period if anchor_end else 0
    return [(offset + i) % period for i in range(length)]


def _tile_frame(out: np.ndarray, frame: dict, patch: np.ndarray, direction: tuple[int, int]) -> None:
    ir0, ic0, ir1, ic1 = frame["inner"]
    inner_h = ir1 - ir0 + 1
    inner_w = ic1 - ic0 + 1
    ph, pw = patch.shape

    row_ids = _periodic_indices(inner_h, ph, anchor_end=(direction == UP))
    col_ids = _periodic_indices(inner_w, pw, anchor_end=(direction == LEFT))

    for rr, src_r in enumerate(row_ids):
        for cc, src_c in enumerate(col_ids):
            out[ir0 + rr, ic0 + cc] = int(patch[src_r, src_c])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each arrow-marked source pattern onto the first frame in its ray and tile that pattern inside the frame.

    Input structure:
    - The background is the dominant color.
    - Several small monochrome 4-connected pattern blocks occupy 2x2 to 3x3 bounding boxes, and each block uses its own non-background color.
    - Every pattern block touches one red 4-cell arrow marker whose 3-pixel shaft hugs one bbox side and whose single offset tip indicates one of the four cardinal projection directions.
    - Separate gray one-pixel-thick hollow rectangles act as target frames, with background-filled interiors.

    Rule:
    - For each pattern block, read the adjacent arrow direction and cast a straight ray away from the block; the first gray frame hit along that direction is the block's target.
    - Copy the pattern block's full bounding-box contents, including any background holes, periodically across the target frame interior.
    - Anchor the periodic tiling on the interior side where the ray enters the frame, clip any partial repeat at the far edge, erase every red arrow pixel to background, and leave the source blocks and gray frame borders unchanged.

    Key insight:
    - The red arrow is a dual-purpose control: it chooses both which frame receives the pattern and which side of that frame starts the periodic tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    frames = _find_frames(g, bg)
    patterns = _find_patterns(g, bg)

    targets: dict[int, int] = {}
    used_frames: set[int] = set()
    for idx, pattern in enumerate(patterns):
        frame_idx = _match_frame(pattern, frames)
        assert frame_idx not in used_frames, "each target frame should receive at most one pattern"
        used_frames.add(frame_idx)
        targets[idx] = frame_idx

    out = g.copy()
    out[out == RED] = bg
    for idx, pattern in enumerate(patterns):
        _tile_frame(out, frames[targets[idx]], pattern["patch"], pattern["dir"])
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = (int(pts[0, 0]), int(pts[0, 1]))
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _build_shape_library() -> dict[tuple[int, int], list[np.ndarray]]:
    library: dict[tuple[int, int], list[np.ndarray]] = {}
    for h in (2, 3):
        for w in (2, 3):
            shapes: list[np.ndarray] = []
            for bits in range(1 << (h * w)):
                mask = np.array(
                    [(bits >> i) & 1 for i in range(h * w)],
                    dtype=bool,
                ).reshape(h, w)
                area = int(mask.sum())
                if area < 3 or not _mask_connected(mask):
                    continue
                rows, cols = np.where(mask)
                if rows.min() != 0 or rows.max() != h - 1 or cols.min() != 0 or cols.max() != w - 1:
                    continue
                shapes.append(mask)
            library[(h, w)] = shapes
    return library


SHAPES_BY_SIZE = _build_shape_library()


def _mark_rect(mask: np.ndarray, rect: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = rect
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _cells_bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _overlap_start(rng: random.Random, seg0: int, seg1: int, size: int, limit: int) -> int | None:
    low = max(0, seg0 - size + 1)
    high = min(limit - size, seg1)
    if low > high:
        return None
    return int(rng.randint(low, high))


def _arrow_variants(
    r0: int,
    c0: int,
    h: int,
    w: int,
    direction: tuple[int, int],
    grid_h: int,
    grid_w: int,
) -> list[list[tuple[int, int]]]:
    variants: list[list[tuple[int, int]]] = []

    if direction in (LEFT, RIGHT):
        start_rows = [r0] if h == 3 else [r0 - 1, r0]
        shaft_col = c0 + w if direction == RIGHT else c0 - 1
        head_col = shaft_col + 1 if direction == RIGHT else shaft_col - 1
        for start in start_rows:
            shaft_rows = [start, start + 1, start + 2]
            for head_row in (shaft_rows[0], shaft_rows[-1]):
                cells = [(row, shaft_col) for row in shaft_rows] + [(head_row, head_col)]
                if all(0 <= rr < grid_h and 0 <= cc < grid_w for rr, cc in cells):
                    variants.append(cells)
    else:
        start_cols = [c0] if w == 3 else [c0 - 1, c0]
        shaft_row = r0 + h if direction == DOWN else r0 - 1
        head_row = shaft_row + 1 if direction == DOWN else shaft_row - 1
        for start in start_cols:
            shaft_cols = [start, start + 1, start + 2]
            for head_col in (shaft_cols[0], shaft_cols[-1]):
                cells = [(shaft_row, col) for col in shaft_cols] + [(head_row, head_col)]
                if all(0 <= rr < grid_h and 0 <= cc < grid_w for rr, cc in cells):
                    variants.append(cells)

    dedup: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
    for cells in variants:
        key = tuple(sorted(cells))
        dedup[key] = cells
    return list(dedup.values())


def _band_rect(
    grid_h: int,
    grid_w: int,
    bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    if direction in (LEFT, RIGHT):
        return r0, 0, r1, grid_w - 1
    return 0, c0, grid_h - 1, c1


def _try_place_projector(
    rng: random.Random,
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    direction: tuple[int, int],
) -> bool:
    grid_h, grid_w = grid.shape

    ph = rng.choice([2, 3])
    pw = rng.choice([2, 3])
    shape = rng.choice(SHAPES_BY_SIZE[(ph, pw)])

    inner_h = rng.randint(2, 7)
    inner_w = rng.randint(2, 7)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    use_decoy = rng.random() < 0.5
    if use_decoy:
        decoy_inner_h = rng.randint(2, 6)
        decoy_inner_w = rng.randint(2, 6)
        decoy_h = decoy_inner_h + 2
        decoy_w = decoy_inner_w + 2
    else:
        decoy_h = decoy_w = 0

    gap1 = rng.randint(1, 5)
    gap2 = rng.randint(1, 4)

    for _ in range(LOCAL_PLACE_ATTEMPTS):
        if direction == RIGHT:
            total_w = pw + 2 + gap1 + frame_w + (gap2 + decoy_w if use_decoy else 0)
            if total_w > grid_w:
                return False
            src_r0 = rng.randint(0, grid_h - ph)
            src_c0 = rng.randint(0, grid_w - total_w)
            frame_c0 = src_c0 + pw + 2 + gap1
            decoy_c0 = frame_c0 + frame_w + gap2 if use_decoy else None
            frame_r0 = _overlap_start(rng, src_r0, src_r0 + ph - 1, frame_h, grid_h)
            decoy_r0 = (
                _overlap_start(rng, src_r0, src_r0 + ph - 1, decoy_h, grid_h) if use_decoy else None
            )
        elif direction == LEFT:
            total_left = frame_w + gap1 + 2 + (decoy_w + gap2 if use_decoy else 0)
            if total_left + pw > grid_w:
                return False
            src_r0 = rng.randint(0, grid_h - ph)
            src_c0 = rng.randint(total_left, grid_w - pw)
            frame_c0 = src_c0 - frame_w - gap1 - 2
            decoy_c0 = frame_c0 - gap2 - decoy_w if use_decoy else None
            frame_r0 = _overlap_start(rng, src_r0, src_r0 + ph - 1, frame_h, grid_h)
            decoy_r0 = (
                _overlap_start(rng, src_r0, src_r0 + ph - 1, decoy_h, grid_h) if use_decoy else None
            )
        elif direction == DOWN:
            total_h = ph + 2 + gap1 + frame_h + (gap2 + decoy_h if use_decoy else 0)
            if total_h > grid_h:
                return False
            src_r0 = rng.randint(0, grid_h - total_h)
            src_c0 = rng.randint(0, grid_w - pw)
            frame_r0 = src_r0 + ph + 2 + gap1
            decoy_r0 = frame_r0 + frame_h + gap2 if use_decoy else None
            frame_c0 = _overlap_start(rng, src_c0, src_c0 + pw - 1, frame_w, grid_w)
            decoy_c0 = (
                _overlap_start(rng, src_c0, src_c0 + pw - 1, decoy_w, grid_w) if use_decoy else None
            )
        else:
            assert direction == UP
            total_up = frame_h + gap1 + 2 + (decoy_h + gap2 if use_decoy else 0)
            if total_up + ph > grid_h:
                return False
            src_r0 = rng.randint(total_up, grid_h - ph)
            src_c0 = rng.randint(0, grid_w - pw)
            frame_r0 = src_r0 - frame_h - gap1 - 2
            decoy_r0 = frame_r0 - gap2 - decoy_h if use_decoy else None
            frame_c0 = _overlap_start(rng, src_c0, src_c0 + pw - 1, frame_w, grid_w)
            decoy_c0 = (
                _overlap_start(rng, src_c0, src_c0 + pw - 1, decoy_w, grid_w) if use_decoy else None
            )

        if frame_r0 is None or frame_c0 is None or (use_decoy and (decoy_r0 is None or decoy_c0 is None)):
            continue

        arrow_options = _arrow_variants(src_r0, src_c0, ph, pw, direction, grid_h, grid_w)
        if not arrow_options:
            continue
        arrow_cells = rng.choice(arrow_options)

        source_bbox = (src_r0, src_c0, src_r0 + ph - 1, src_c0 + pw - 1)
        frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)
        rects = [source_bbox, frame_bbox, _cells_bbox(arrow_cells), _band_rect(grid_h, grid_w, source_bbox, direction)]
        decoy_bbox = None
        if use_decoy:
            decoy_bbox = (decoy_r0, decoy_c0, decoy_r0 + decoy_h - 1, decoy_c0 + decoy_w - 1)
            rects.append(decoy_bbox)

        reserve = np.zeros_like(blocked)
        for rect in rects:
            _mark_rect(reserve, rect, pad=1)
        if np.any(reserve & blocked):
            continue

        draw_cells: dict[tuple[int, int], int] = {}
        for rr in range(ph):
            for cc in range(pw):
                if shape[rr, cc]:
                    draw_cells[(src_r0 + rr, src_c0 + cc)] = color
        for cell in arrow_cells:
            draw_cells[cell] = RED
        for cell in rect_border_cells(*frame_bbox):
            draw_cells[cell] = GRAY
        if decoy_bbox is not None:
            for cell in rect_border_cells(*decoy_bbox):
                draw_cells[cell] = GRAY

        if any(grid[r, c] != BG for r, c in draw_cells):
            continue

        for (r, c), value in draw_cells.items():
            grid[r, c] = value
        blocked |= reserve
        return True

    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    projector_count = rng.choice([1, 2, 2, 3, 3])
    min_side = min(30, 12 + 4 * projector_count)
    grid_h = rng.randint(min_side, 30)
    grid_w = rng.randint(min_side, 30)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    directions = DIRS[:]
    rng.shuffle(directions)
    colors = rng.sample(PATTERN_COLORS, projector_count)

    for color, direction in zip(colors, directions[:projector_count]):
        if not _try_place_projector(rng, grid, blocked, color, direction):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if np.any(out == RED):
        return None
    if int(np.count_nonzero(out != grid)) < 6:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid projector-frame input")
