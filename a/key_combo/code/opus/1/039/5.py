from __future__ import annotations

import numpy as np


BG = 0
SPRITE = 1
RED = 2
FRAME = 4

GEN_MAX_ATTEMPTS = 600
GEN_BRANCHES = ("no_wrap", "wrap", "overflow")
GEN_DIM_RANGES = {
    "no_wrap": ((16, 25), (16, 25)),
    "wrap": ((15, 23), (15, 23)),
    "overflow": ((13, 20), (13, 20)),
}
SPRITE_LIBRARY = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
)


def _frame_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    mask[1, 1 : width - 1] = True
    mask[height - 2, 1 : width - 1] = True
    mask[1 : height - 1, 1] = True
    mask[1 : height - 1, width - 2] = True
    return mask


def _frame_path(shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = shape
    path: list[tuple[int, int]] = []
    path.extend((1, c) for c in range(1, width - 1))
    path.extend((r, width - 2) for r in range(2, height - 1))
    path.extend((height - 2, c) for c in range(width - 3, 0, -1))
    path.extend((r, 1) for r in range(height - 3, 1, -1))
    return path


def _single_red_run(top_edge: np.ndarray) -> tuple[int, int]:
    assert top_edge.ndim == 1, "top edge must be one-dimensional"
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(map(int, top_edge)):
        if value == RED and start is None:
            start = idx
        elif value != RED and start is not None:
            runs.append((start, idx - start))
            start = None
    if start is not None:
        runs.append((start, len(top_edge) - start))
    assert len(runs) == 1, "expected exactly one red segment on the frame top edge"
    start, length = runs[0]
    assert 2 <= length <= 4, "red frame segment length must be between 2 and 4"
    for idx, value in enumerate(map(int, top_edge)):
        if start <= idx < start + length:
            assert value == RED, "segment cells must be red"
        else:
            assert value == FRAME, "non-segment top-edge frame cells must stay yellow"
    return start, length


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _normalize_shape(coords: list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    min_r, min_c = min(rows), min(cols)
    rel = tuple(sorted((r - min_r, c - min_c) for r, c in coords))
    height = max(r for r, _ in rel) + 1
    width = max(c for _, c in rel) + 1
    return rel, height, width


def _count_border_copies(path_len: int, seg_start: int, seg_len: int) -> int:
    pos = seg_start
    count = 0
    while pos + seg_len <= path_len:
        count += 1
        pos += 2 * seg_len
    return count


def _paint_border_copies(out: np.ndarray, path: list[tuple[int, int]], seg_start: int, seg_len: int) -> None:
    pos = seg_start
    while pos + seg_len <= len(path):
        for idx in range(pos, pos + seg_len):
            r, c = path[idx]
            out[r, c] = RED
        pos += 2 * seg_len


def _stamp_anchors(
    start_row: int,
    start_col: int,
    left_col: int,
    right_col: int,
    bottom_row: int,
    sprite_h: int,
    sprite_w: int,
) -> list[tuple[int, int]]:
    anchors: list[tuple[int, int]] = []
    row = start_row
    col = start_col
    while row + sprite_h - 1 <= bottom_row:
        if col + sprite_w - 1 <= right_col:
            anchors.append((row, col))
            col += sprite_w + 1
        else:
            row += sprite_h + 1
            col = left_col
    return anchors


def _paint_sprite(out: np.ndarray, sprite_rel: tuple[tuple[int, int], ...], anchor: tuple[int, int]) -> None:
    ar, ac = anchor
    for dr, dc in sprite_rel:
        out[ar + dr, ac + dc] = SPRITE


def _analyze_input(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert 8 <= height <= 30 and 8 <= width <= 30, "grid must fit ARC bounds"

    assert np.all(g[0, :] == BG) and np.all(g[-1, :] == BG), "outer rows must be background"
    assert np.all(g[:, 0] == BG) and np.all(g[:, -1] == BG), "outer columns must be background"

    frame_mask = _frame_mask(g.shape)
    assert np.all(np.isin(g[frame_mask], [RED, FRAME])), "frame may only use red and yellow"

    non_top_frame = frame_mask.copy()
    non_top_frame[1, 1 : width - 1] = False
    assert not np.any(g[non_top_frame] == RED), "only the top frame edge may contain red"
    assert np.all(g[non_top_frame] == FRAME), "non-top frame cells must stay yellow"

    seg_start, seg_len = _single_red_run(g[1, 1 : width - 1])
    path = _frame_path(g.shape)
    copy_count = _count_border_copies(len(path), seg_start, seg_len)

    placeholder_cells = np.argwhere((g == RED) & ~frame_mask)
    assert placeholder_cells.shape == (1, 2), "expected one interior red placeholder"
    placeholder = tuple(map(int, placeholder_cells[0]))
    assert placeholder == (2, 2), "placeholder must occupy the top-left cell inside the frame"

    blue_components = _components(g == SPRITE)
    assert len(blue_components) == 1, "expected exactly one blue source sprite"
    sprite_abs = blue_components[0]
    sprite_rel, sprite_h, sprite_w = _normalize_shape(sprite_abs)
    assert 3 <= len(sprite_rel) <= 5, "source sprite must contain 3 to 5 pixels"
    assert (0, 0) in sprite_rel, "the source sprite must cover its own bbox top-left cell"

    interior_left = 2
    interior_right = width - 3
    interior_bottom = height - 3
    anchors = _stamp_anchors(
        placeholder[0],
        placeholder[1],
        interior_left,
        interior_right,
        interior_bottom,
        sprite_h,
        sprite_w,
    )
    return {
        "grid": g,
        "height": height,
        "width": width,
        "frame_mask": frame_mask,
        "path": path,
        "seg_start": seg_start,
        "seg_len": seg_len,
        "copy_count": copy_count,
        "placeholder": placeholder,
        "sprite_rel": sprite_rel,
        "sprite_h": sprite_h,
        "sprite_w": sprite_w,
        "anchors": anchors,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Tile the red border segment along the frame and use the copy count to stamp blue sprites inside.

    Input structure:
    - The grid has black background and a one-cell-thick yellow rectangular frame inset by one cell from the outer edge.
    - A single contiguous red segment replaces part of the frame's top edge, and one red placeholder pixel sits at the frame interior's top-left cell.
    - A single small blue 4-connected sprite inside the frame serves as the stamp template.

    Rule:
    - Unroll the frame boundary clockwise from the frame's top-left corner.
    - If the top red segment has length n, place copies of that length-n segment every 2n positions along the unrolled boundary, counting only complete copies.
    - Repaint the frame with those tiled red segments, then stamp that many blue sprite copies from the placeholder position, moving left-to-right with one blank column between copies and wrapping to the next sprite row with one blank row when needed.

    Key insight:
    - The red border segment encodes a number through its tiling count on the frame perimeter, and that derived count controls how many sprite copies appear in a different region of the grid.
    """
    meta = _analyze_input(grid)
    out = np.array(meta["grid"], copy=True)
    frame_mask = meta["frame_mask"]
    path = meta["path"]
    seg_start = int(meta["seg_start"])
    seg_len = int(meta["seg_len"])
    copy_count = int(meta["copy_count"])
    sprite_rel = meta["sprite_rel"]
    anchors = meta["anchors"]

    out[frame_mask] = FRAME
    _paint_border_copies(out, path, seg_start, seg_len)
    for anchor in anchors[:copy_count]:
        _paint_sprite(out, sprite_rel, anchor)
    return out


def _choose_sprite(rng: np.random.Generator) -> tuple[tuple[tuple[int, int], ...], int, int]:
    sprite = tuple(tuple(cell) for cell in SPRITE_LIBRARY[int(rng.integers(len(SPRITE_LIBRARY)))])
    height = max(r for r, _ in sprite) + 1
    width = max(c for _, c in sprite) + 1
    return sprite, height, width


def _branch_matches(branch: str, count: int, per_row: int, capacity: int) -> bool:
    if branch == "no_wrap":
        return 2 <= count <= per_row
    if branch == "wrap":
        return per_row + 1 <= count <= capacity and capacity > per_row
    if branch == "overflow":
        return count > capacity >= 1
    raise ValueError(f"unknown branch {branch}")


def _used_mask(
    shape: tuple[int, int],
    sprite_rel: tuple[tuple[int, int], ...],
    anchors: list[tuple[int, int]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for anchor in anchors:
        ar, ac = anchor
        for dr, dc in sprite_rel:
            mask[ar + dr, ac + dc] = True
    return mask


def _find_source_origin(
    height: int,
    width: int,
    sprite_rel: tuple[tuple[int, int], ...],
    sprite_h: int,
    sprite_w: int,
    occupied: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    top, left = 2, 2
    bottom, right = height - 3, width - 3

    def scan(require_gap: bool) -> list[tuple[int, int]]:
        candidates: list[tuple[int, int]] = []
        for row in range(bottom - sprite_h + 1, top - 1, -1):
            for col in range(right - sprite_w + 1, left - 1, -1):
                if row < top + (bottom - top + 1) // 2 or col < left + (right - left + 1) // 2:
                    continue
                if any(occupied[row + dr, col + dc] for dr, dc in sprite_rel):
                    continue
                if require_gap:
                    r0 = max(top, row - 1)
                    r1 = min(bottom, row + sprite_h)
                    c0 = max(left, col - 1)
                    c1 = min(right, col + sprite_w)
                    if np.any(occupied[r0 : r1 + 1, c0 : c1 + 1]):
                        continue
                candidates.append((row, col))
        return candidates

    candidates = scan(require_gap=True)
    if not candidates:
        candidates = scan(require_gap=False)
    if not candidates:
        return None
    limit = min(6, len(candidates))
    return candidates[int(rng.integers(limit))]


def _pick_border_params(
    height: int,
    width: int,
    branch: str,
    per_row: int,
    capacity: int,
    rng: np.random.Generator,
) -> tuple[int, int, int] | None:
    top_len = width - 2
    path_len = len(_frame_path((height, width)))
    preferred: list[tuple[int, int, int]] = []
    fallback: list[tuple[int, int, int]] = []
    for seg_len in (2, 3, 4):
        if top_len < seg_len:
            continue
        baseline = path_len // (2 * seg_len)
        for seg_start in range(0, top_len - seg_len + 1):
            count = _count_border_copies(path_len, seg_start, seg_len)
            if not _branch_matches(branch, count, per_row, capacity):
                continue
            record = (seg_len, seg_start, count)
            if seg_start > 0 and count != baseline:
                preferred.append(record)
            else:
                fallback.append(record)
    if preferred and rng.random() < 0.8:
        pool = preferred
    else:
        pool = preferred + fallback
    if not pool:
        return None
    return pool[int(rng.integers(len(pool)))]


def _validate_generated_case(grid: np.ndarray, branch: str, expected_count: int) -> None:
    meta = _analyze_input(grid)
    copy_count = int(meta["copy_count"])
    assert copy_count == expected_count, "generator and parser disagree on the border copy count"

    anchors = meta["anchors"]
    per_row = sum(1 for row, _ in anchors if row == anchors[0][0]) if anchors else 0
    capacity = len(anchors)
    assert _branch_matches(branch, copy_count, per_row, capacity), "sample does not realize the requested branch"

    out = main(grid)
    frame_mask = meta["frame_mask"]
    expected_frame = np.full_like(out, BG)
    expected_frame[frame_mask] = FRAME
    _paint_border_copies(expected_frame, meta["path"], int(meta["seg_start"]), int(meta["seg_len"]))
    assert np.array_equal(out[frame_mask], expected_frame[frame_mask]), "border tiling is incorrect"

    stamp_count = min(copy_count, capacity)
    occupied = _used_mask(out.shape, meta["sprite_rel"], anchors[:stamp_count])
    stamp_cells = int(np.sum(occupied))
    source_area = len(meta["sprite_rel"])
    blue_cells = int(np.count_nonzero(out == SPRITE))
    assert blue_cells == stamp_cells + source_area, "output blue cells must equal source sprite plus placed copies"
    assert out[2, 2] == SPRITE, "the first stamped sprite must cover the placeholder"


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    branch = str(rng.choice(np.array(GEN_BRANCHES, dtype=object)))
    sprite_rel, sprite_h, sprite_w = _choose_sprite(rng)
    (h_lo, h_hi), (w_lo, w_hi) = GEN_DIM_RANGES[branch]

    for _ in range(120):
        height = int(rng.integers(h_lo, h_hi))
        width = int(rng.integers(w_lo, w_hi))
        if max(height, width) > 30:
            continue

        anchors = _stamp_anchors(2, 2, 2, width - 3, height - 3, sprite_h, sprite_w)
        if not anchors:
            continue
        per_row = sum(1 for row, _ in anchors if row == anchors[0][0])
        capacity = len(anchors)

        border_params = _pick_border_params(height, width, branch, per_row, capacity, rng)
        if border_params is None:
            continue
        seg_len, seg_start, copy_count = border_params

        placed = anchors[: min(copy_count, capacity)]
        occupied = _used_mask((height, width), sprite_rel, placed)
        source_origin = _find_source_origin(height, width, sprite_rel, sprite_h, sprite_w, occupied, rng)
        if source_origin is None:
            continue

        grid = np.full((height, width), BG, dtype=int)
        frame_mask = _frame_mask(grid.shape)
        grid[frame_mask] = FRAME
        grid[1, 1 + seg_start : 1 + seg_start + seg_len] = RED
        grid[2, 2] = RED
        sr, sc = source_origin
        for dr, dc in sprite_rel:
            grid[sr + dr, sc + dc] = SPRITE

        baseline = len(_frame_path((height, width))) // (2 * seg_len)
        if branch != "overflow" and rng.random() < 0.65 and copy_count == baseline:
            continue

        try:
            _validate_generated_case(grid, branch, copy_count)
        except AssertionError:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
