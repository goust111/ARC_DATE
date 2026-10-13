from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
FRAME = 5

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_INTERVALS_4 = {
    2: [tuple(range(0, 2)), tuple(range(1, 3)), tuple(range(2, 4))],
    3: [tuple(range(0, 3)), tuple(range(1, 4))],
    4: [tuple(range(0, 4))],
}
_DOUBLE_ROW_TEMPLATES = [
    ((0, 0), (0, 1), (0, 2), (1, 2), (1, 3)),
    ((0, 1), (0, 2), (0, 3), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (1, 3)),
    ((0, 2), (0, 3), (1, 0), (1, 1), (1, 2)),
]
_DOUBLE_COL_TEMPLATES = [
    ((0, 0), (1, 0), (2, 0), (2, 1), (3, 1)),
    ((0, 1), (1, 1), (2, 1), (2, 0), (3, 0)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (3, 1)),
    ((0, 1), (1, 1), (1, 0), (2, 0), (3, 0)),
]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in _DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_rectangular_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(local, expected))


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for coords in _components(grid == FRAME):
        if not _is_rectangular_frame(coords):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "interior_area": (r1 - r0 - 1) * (c1 - c0 - 1),
            }
        )
    return frames


def _largest_frame(grid: np.ndarray) -> dict:
    frames = _find_frames(grid)
    assert frames, "expected at least one rectangular frame"
    frames.sort(key=lambda item: item["interior_area"], reverse=True)
    assert len(frames) == 1 or frames[0]["interior_area"] > frames[1]["interior_area"], "expected a unique largest frame"
    return frames[0]


def _find_axis(inner: np.ndarray) -> tuple[str, int]:
    empty_rows = [r for r in range(inner.shape[0]) if np.all(inner[r, :] == BG)]
    empty_cols = [c for c in range(inner.shape[1]) if np.all(inner[:, c] == BG)]
    axes = [("row", r) for r in empty_rows] + [("col", c) for c in empty_cols]
    assert len(axes) == 1, "expected exactly one empty interior row or column"
    return axes[0]


def _minority_color(inner: np.ndarray) -> int:
    counts = Counter(int(v) for v in inner.flatten() if int(v) not in {BG, FRAME})
    assert len(counts) == 2, "expected exactly two interior foreground colors"
    ranked = counts.most_common()
    assert ranked[0][1] != ranked[1][1], "expected a unique minority color"
    return int(ranked[-1][0])


def _reflect(kind: str, axis_index: int, r: int, c: int) -> tuple[int, int]:
    if kind == "row":
        return 2 * axis_index - r, c
    return r, 2 * axis_index - c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the largest gray frame and mirror its locally rarer interior color across the unique empty interior axis.

    Input structure:
    - A black background contains 2-3 complete one-pixel-thick gray rectangular frames.
    - The unique largest frame by interior area contains blue and red foreground objects plus exactly one fully black interior row or column.
    - Smaller gray frames may contain extra blue/red distractor pixels that do not affect the target crop.

    Rule:
    - Identify the gray rectangular frame with the unique largest interior area and crop its full bounding box.
    - Inside that crop, find the unique all-black interior row or column and use it as the reflection axis.
    - Count only the cropped frame's interior foreground pixels, choose the unique minority color, and reflect each of its pixels across the axis.
    - Paint reflected pixels only onto black cells, preserving any pre-existing non-black pixels, and return the cropped frame.

    Key insight:
    - Three independent local decisions must agree: the largest frame chooses the crop, the empty interior line chooses the axis, and the minority count must be computed only inside that crop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    target = _largest_frame(g)
    r0, c0, r1, c1 = target["bbox"]
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    inner = crop[1:-1, 1:-1]

    axis_kind, axis_index = _find_axis(inner)
    color = _minority_color(inner)

    for rr_raw, cc_raw in np.argwhere(inner == color):
        rr = int(rr_raw)
        cc = int(cc_raw)
        mr, mc = _reflect(axis_kind, axis_index, rr, cc)
        assert 0 <= mr < inner.shape[0] and 0 <= mc < inner.shape[1], "reflected cell left the frame interior"
        if crop[mr + 1, mc + 1] == BG:
            crop[mr + 1, mc + 1] = color
    return crop


def _double_row_cells(axis_row: int, double_side: str, variant: int) -> set[tuple[int, int]]:
    rows = [0, 1] if double_side == "top" else [2, 3]
    near_is_top = double_side == "bottom"
    cells = set()
    for rr, cc in _DOUBLE_ROW_TEMPLATES[variant]:
        base_r = rr if near_is_top else 1 - rr
        cells.add((rows[base_r], cc))
    if axis_row == 2:
        cells = {(3 - r, c) for r, c in cells}
    return cells


def _double_col_cells(axis_col: int, double_side: str, variant: int) -> set[tuple[int, int]]:
    cols = [0, 1] if double_side == "left" else [2, 3]
    near_is_left = double_side == "right"
    cells = set()
    for rr, cc in _DOUBLE_COL_TEMPLATES[variant]:
        base_c = cc if near_is_left else 1 - cc
        cells.add((rr, cols[base_c]))
    if axis_col == 2:
        cells = {(r, 3 - c) for r, c in cells}
    return cells


def _choose_interval(length: int, must_hit: set[int] | None = None, cannot_cover: set[int] | None = None) -> tuple[int, ...]:
    candidates = list(_INTERVALS_4[length])
    random.shuffle(candidates)
    for interval in candidates:
        cells = set(interval)
        if must_hit is not None and not (cells & must_hit):
            continue
        if cannot_cover is not None and cells >= cannot_cover:
            continue
        return interval
    raise RuntimeError("no valid interval found")


def _build_largest_inner(axis_kind: str, axis_index: int, source_side: str, minority: int, majority: int) -> np.ndarray:
    inner = np.zeros((4, 4), dtype=int)
    if axis_kind == "row":
        if axis_index == 1:
            single_side, double_side = "top", "bottom"
            single_row, near_row, far_row = 0, 2, 3
        else:
            single_side, double_side = "bottom", "top"
            single_row, near_row, far_row = 3, 1, 0

        if source_side == single_side:
            major_cells = _double_row_cells(axis_index, double_side, random.randrange(len(_DOUBLE_ROW_TEMPLATES)))
            near_occ = {c for r, c in major_cells if r == near_row}
            minority_cols = _choose_interval(random.choice((3, 4)), must_hit=set(range(4)) - near_occ)
            for r, c in major_cells:
                inner[r, c] = majority
            for c in minority_cols:
                inner[single_row, c] = minority
        else:
            minority_cols = _choose_interval(random.choice((3, 4)))
            blocker_cols = _choose_interval(random.choice((2, 3)), cannot_cover=set(minority_cols))
            for c in range(4):
                inner[far_row, c] = majority
            for c in blocker_cols:
                inner[single_row, c] = majority
            for c in minority_cols:
                inner[near_row, c] = minority
    else:
        if axis_index == 1:
            single_side, double_side = "left", "right"
            single_col, near_col, far_col = 0, 2, 3
        else:
            single_side, double_side = "right", "left"
            single_col, near_col, far_col = 3, 1, 0

        if source_side == single_side:
            major_cells = _double_col_cells(axis_index, double_side, random.randrange(len(_DOUBLE_COL_TEMPLATES)))
            near_occ = {r for r, c in major_cells if c == near_col}
            minority_rows = _choose_interval(random.choice((3, 4)), must_hit=set(range(4)) - near_occ)
            for r, c in major_cells:
                inner[r, c] = majority
            for r in minority_rows:
                inner[r, single_col] = minority
        else:
            minority_rows = _choose_interval(random.choice((3, 4)))
            blocker_rows = _choose_interval(random.choice((2, 3)), cannot_cover=set(minority_rows))
            for r in range(4):
                inner[r, far_col] = majority
            for r in blocker_rows:
                inner[r, single_col] = majority
            for r in minority_rows:
                inner[r, near_col] = minority

    assert _find_axis(inner) == (axis_kind, axis_index)
    assert _minority_color(inner) == minority
    for coords in _components(inner == majority):
        assert 2 <= len(coords) <= 5
    for coords in _components(inner == minority):
        assert 2 <= len(coords) <= 5
    return inner


def _frame_patch(inner: np.ndarray) -> np.ndarray:
    h, w = inner.shape
    patch = np.full((h + 2, w + 2), FRAME, dtype=int)
    patch[1:-1, 1:-1] = inner
    return patch


def _boxes_separate(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_frame(
    H: int,
    W: int,
    height: int,
    width: int,
    existing: list[tuple[int, int, int, int]],
    *,
    target_line: int | None = None,
    target_kind: str | None = None,
) -> tuple[int, int]:
    candidates = []
    for r0 in range(0, H - height + 1):
        for c0 in range(0, W - width + 1):
            box = (r0, c0, r0 + height - 1, c0 + width - 1)
            if any(not _boxes_separate(box, other) for other in existing):
                continue
            if target_line is not None:
                if target_kind == "row" and not (r0 + 1 <= target_line <= r0 + height - 2):
                    continue
                if target_kind == "col" and not (c0 + 1 <= target_line <= c0 + width - 2):
                    continue
            candidates.append((r0, c0))
    if not candidates:
        raise RuntimeError("failed to place frame")
    return random.choice(candidates)


def _fill_small_frame(
    grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
    axis_kind: str,
    axis_global: int,
    minority: int,
    majority: int,
    extra_minority: int,
) -> None:
    r0, c0, r1, c1 = frame_box
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    random.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    if axis_kind == "row":
        line_cells = [(axis_global, c) for c in range(c0 + 1, c1)]
    else:
        line_cells = [(r, axis_global) for r in range(r0 + 1, r1)]
    line_cells = [cell for cell in line_cells if cell in cells]
    assert line_cells, "blocker frame must cross the target axis line"
    blocker = random.choice(line_cells)
    chosen.append(blocker)
    remaining = [cell for cell in cells if cell != blocker]
    chosen.extend(remaining[: extra_minority - 1])
    for r, c in chosen:
        grid[r, c] = minority
    extra_major = random.randint(0, 1)
    if extra_major:
        leftovers = [cell for cell in remaining if cell not in chosen]
        if leftovers:
            rr, cc = random.choice(leftovers)
            grid[rr, cc] = majority


def _global_minority_solver(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    target = _largest_frame(g)
    r0, c0, r1, c1 = target["bbox"]
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    inner = crop[1:-1, 1:-1]
    axis_kind, axis_index = _find_axis(inner)
    counts = Counter(int(v) for v in g.flatten() if int(v) not in {BG, FRAME})
    color = int(min(counts, key=counts.get))
    for rr_raw, cc_raw in np.argwhere(inner == color):
        rr = int(rr_raw)
        cc = int(cc_raw)
        mr, mc = _reflect(axis_kind, axis_index, rr, cc)
        if 0 <= mr < inner.shape[0] and 0 <= mc < inner.shape[1] and crop[mr + 1, mc + 1] == BG:
            crop[mr + 1, mc + 1] = color
    return crop


def _center_axis_solver(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    target = _largest_frame(g)
    r0, c0, r1, c1 = target["bbox"]
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    inner = crop[1:-1, 1:-1]
    axis_kind, _ = _find_axis(inner)
    color = _minority_color(inner)
    for rr_raw, cc_raw in np.argwhere(inner == color):
        rr = int(rr_raw)
        cc = int(cc_raw)
        if axis_kind == "row":
            mr = inner.shape[0] - 1 - rr
            mc = cc
        else:
            mr = rr
            mc = inner.shape[1] - 1 - cc
        if crop[mr + 1, mc + 1] == BG:
            crop[mr + 1, mc + 1] = color
    return crop


def _validate_semantics(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    assert 2 <= len(frames) <= 3
    largest = _largest_frame(g)
    r0, c0, r1, c1 = largest["bbox"]
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    inner = crop[1:-1, 1:-1]
    axis_kind, axis_index = _find_axis(inner)
    minority = _minority_color(inner)
    majority = BLUE if minority == RED else RED
    local_counts = Counter(int(v) for v in inner.flatten() if int(v) not in {BG, FRAME})
    assert local_counts[minority] < local_counts[majority]

    axis_global = (r0 + 1 + axis_index) if axis_kind == "row" else (c0 + 1 + axis_index)
    if axis_kind == "row":
        assert not np.all(g[axis_global, :] == BG), "global row shortcut should fail"
    else:
        assert not np.all(g[:, axis_global] == BG), "global column shortcut should fail"

    global_counts = Counter(int(v) for v in g.flatten() if int(v) not in {BG, FRAME})
    assert len(global_counts) == 2
    global_minority = int(min(global_counts, key=global_counts.get))
    assert global_minority != minority, "global counting shortcut should be wrong"

    out = main(g)
    assert out.shape == crop.shape
    assert np.any(out != crop), "output must differ from the cropped input"
    assert not np.array_equal(_global_minority_solver(g), out), "global counting shortcut accidentally works"
    assert not np.array_equal(_center_axis_solver(g), out), "center-axis shortcut accidentally works"


def _attempt_generate() -> np.ndarray:
    H = random.randint(16, 18)
    W = random.randint(17, 19)
    grid = np.zeros((H, W), dtype=int)

    axis_kind = random.choice(("row", "col"))
    axis_index = random.choice((1, 2))
    source_side = random.choice(("top", "bottom")) if axis_kind == "row" else random.choice(("left", "right"))
    minority, majority = random.choice(((BLUE, RED), (RED, BLUE)))

    inner = _build_largest_inner(axis_kind, axis_index, source_side, minority, majority)
    largest_patch = _frame_patch(inner)
    largest_h, largest_w = largest_patch.shape

    blocker_h = random.choice((5, 6))
    blocker_w = random.choice((5, 6))
    if blocker_h < 5:
        blocker_h = 5
    if blocker_w < 5:
        blocker_w = 5
    extra_frames = 1 if random.random() < 0.55 else 2

    largest_r0, largest_c0 = _place_frame(H, W, largest_h, largest_w, [])
    largest_box = (largest_r0, largest_c0, largest_r0 + largest_h - 1, largest_c0 + largest_w - 1)
    axis_global = largest_r0 + 1 + axis_index if axis_kind == "row" else largest_c0 + 1 + axis_index

    blocker_r0, blocker_c0 = _place_frame(
        H,
        W,
        blocker_h,
        blocker_w,
        [largest_box],
        target_line=axis_global,
        target_kind=axis_kind,
    )
    blocker_box = (blocker_r0, blocker_c0, blocker_r0 + blocker_h - 1, blocker_c0 + blocker_w - 1)

    other_boxes = [largest_box, blocker_box]
    extra_box = None
    if extra_frames == 2:
        small_h = random.choice((4, 5))
        small_w = random.choice((4, 5))
        extra_r0, extra_c0 = _place_frame(H, W, small_h, small_w, other_boxes)
        extra_box = (extra_r0, extra_c0, extra_r0 + small_h - 1, extra_c0 + small_w - 1)
        other_boxes.append(extra_box)

    for box in other_boxes:
        r0, c0, r1, c1 = box
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME
        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME

    grid[largest_r0 : largest_r0 + largest_h, largest_c0 : largest_c0 + largest_w] = largest_patch

    local_counts = Counter(int(v) for v in inner.flatten() if int(v) not in {BG, FRAME})
    extra_minority = local_counts[majority] - local_counts[minority] + 1
    _fill_small_frame(grid, blocker_box, axis_kind, axis_global, minority, majority, extra_minority)

    if extra_box is not None:
        r0, c0, r1, c1 = extra_box
        free_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        random.shuffle(free_cells)
        extras = random.randint(0, min(2, len(free_cells)))
        for r, c in free_cells[:extras]:
            grid[r, c] = random.choice((minority, majority))

    _validate_semantics(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            return _attempt_generate()
        except (AssertionError, RuntimeError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid input")
