from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, get_bg  # type: ignore  # noqa: E402


BG = 0
FRAME = 6
GREY = 5
OBJECT_COLORS = (1, 2, 3, 4, 7, 8, 9)
BODY_SIZES = ((2, 2), (2, 3), (3, 2))
DIRECTIONS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}
MAX_ATTEMPTS = 500


@dataclass(frozen=True)
class FrameInfo:
    bbox: tuple[int, int, int, int]
    line_rows: tuple[int, ...]
    line_cols: tuple[int, ...]
    n_cells: int
    cell_size: int


@dataclass(frozen=True)
class TailObject:
    color: int
    body_bbox: tuple[int, int, int, int]
    direction: tuple[int, int]
    length: int
    tail_anchor: tuple[int, int]
    target_axis: str
    target_index: int
    fill_color: int
    priority_distance: int


def _cell_slices(frame: FrameInfo, row_idx: int, col_idx: int) -> tuple[slice, slice]:
    return (
        slice(frame.line_rows[row_idx] + 1, frame.line_rows[row_idx + 1]),
        slice(frame.line_cols[col_idx] + 1, frame.line_cols[col_idx + 1]),
    )


def _index_between(lines: tuple[int, ...], value: int) -> int:
    for idx in range(len(lines) - 1):
        if lines[idx] < value < lines[idx + 1]:
            return idx
    raise AssertionError("tail ray must pass through a cell interior, not a frame line")


def _detect_frame(grid: np.ndarray) -> FrameInfo:
    mask = grid == FRAME
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a magenta lattice frame"
    r0, c0, r1, c1 = bbox_coords(coords)
    box = mask[r0 : r1 + 1, c0 : c1 + 1]

    line_rows = tuple(int(r0 + i) for i in np.where(np.all(box, axis=1))[0])
    line_cols = tuple(int(c0 + i) for i in np.where(np.all(box, axis=0))[0])
    assert len(line_rows) == len(line_cols) and len(line_rows) in (4, 5), "expected a 3x3 or 4x4 full frame"

    row_steps = {line_rows[i + 1] - line_rows[i] for i in range(len(line_rows) - 1)}
    col_steps = {line_cols[i + 1] - line_cols[i] for i in range(len(line_cols) - 1)}
    assert len(row_steps) == 1 and len(col_steps) == 1, "frame spacing must be uniform"
    row_step = next(iter(row_steps))
    col_step = next(iter(col_steps))
    assert row_step == col_step and row_step >= 3, "cell size must be consistent and at least 2"

    expected = np.zeros_like(box, dtype=bool)
    expected[np.array(line_rows) - r0, :] = True
    expected[:, np.array(line_cols) - c0] = True
    assert np.array_equal(box, expected), "frame must be a clean rectangular lattice with outer border"

    return FrameInfo(
        bbox=(r0, c0, r1, c1),
        line_rows=line_rows,
        line_cols=line_cols,
        n_cells=len(line_rows) - 1,
        cell_size=row_step - 1,
    )


def _single_component(mask: np.ndarray) -> np.ndarray:
    comps = components_mask(mask, connectivity=4)
    assert len(comps) == 1, "each object color must form exactly one 4-connected component"
    return comps[0]


def _match_tail_shape(local: np.ndarray, body_h: int, body_w: int, direction_name: str) -> tuple[int, int] | None:
    h, w = local.shape
    if direction_name == "right":
        if h != body_h or w <= body_w or not np.all(local[:, :body_w]):
            return None
        tail = local[:, body_w:]
        if int(tail.sum()) != w - body_w:
            return None
        rows = np.where(np.all(tail, axis=1))[0]
        if len(rows) != 1:
            return None
        if int(local[rows[0]].sum()) != w:
            return None
        return int(rows[0]), w - body_w
    if direction_name == "left":
        if h != body_h or w <= body_w or not np.all(local[:, w - body_w :]):
            return None
        tail = local[:, : w - body_w]
        if int(tail.sum()) != w - body_w:
            return None
        rows = np.where(np.all(tail, axis=1))[0]
        if len(rows) != 1:
            return None
        if int(local[rows[0]].sum()) != w:
            return None
        return int(rows[0]), w - body_w
    if direction_name == "down":
        if w != body_w or h <= body_h or not np.all(local[:body_h, :]):
            return None
        tail = local[body_h:, :]
        if int(tail.sum()) != h - body_h:
            return None
        cols = np.where(np.all(tail, axis=0))[0]
        if len(cols) != 1:
            return None
        if int(local[:, cols[0]].sum()) != h:
            return None
        return int(cols[0]), h - body_h
    if direction_name == "up":
        if w != body_w or h <= body_h or not np.all(local[h - body_h :, :]):
            return None
        tail = local[: h - body_h, :]
        if int(tail.sum()) != h - body_h:
            return None
        cols = np.where(np.all(tail, axis=0))[0]
        if len(cols) != 1:
            return None
        if int(local[:, cols[0]].sum()) != h:
            return None
        return int(cols[0]), h - body_h
    return None


def _detect_object(grid: np.ndarray, frame: FrameInfo, color: int) -> TailObject:
    coords = _single_component(grid == color)
    r0, c0, r1, c1 = bbox_coords(coords)
    local = (grid[r0 : r1 + 1, c0 : c1 + 1] == color)

    matches: list[tuple[int, str, int]] = []
    for body_h, body_w in BODY_SIZES:
        for direction_name in DIRECTIONS:
            hit = _match_tail_shape(local, body_h, body_w, direction_name)
            if hit is None:
                continue
            tail_pos, length = hit
            matches.append((body_h * body_w, direction_name, tail_pos))

    assert matches, f"color {color} is not a valid tailed rectangle"
    max_area = max(area for area, _, _ in matches)
    best = [(name, tail_pos) for area, name, tail_pos in matches if area == max_area]
    assert len(best) == 1, f"color {color} has an ambiguous body/tail decomposition"
    direction_name, tail_pos = best[0]
    dr, dc = DIRECTIONS[direction_name]

    h, w = local.shape
    if direction_name == "right":
        body_w = min(int(local[row].sum()) for row in range(h))
        body_bbox = (r0, c0, r1, c0 + body_w - 1)
        length = w - body_w
        tail_anchor = (r0 + tail_pos, c0 + body_w)
        target_axis = "row"
        target_index = _index_between(frame.line_rows, tail_anchor[0])
        priority_distance = frame.bbox[1] - body_bbox[3] - 1
    elif direction_name == "left":
        row_sums = [int(local[row].sum()) for row in range(h)]
        body_w = min(row_sums)
        body_bbox = (r0, c1 - body_w + 1, r1, c1)
        length = w - body_w
        tail_anchor = (r0 + tail_pos, c1 - body_w)
        target_axis = "row"
        target_index = _index_between(frame.line_rows, tail_anchor[0])
        priority_distance = body_bbox[1] - frame.bbox[3] - 1
    elif direction_name == "down":
        body_h = min(int(local[:, col].sum()) for col in range(w))
        body_bbox = (r0, c0, r0 + body_h - 1, c1)
        length = h - body_h
        tail_anchor = (r0 + body_h, c0 + tail_pos)
        target_axis = "col"
        target_index = _index_between(frame.line_cols, tail_anchor[1])
        priority_distance = frame.bbox[0] - body_bbox[2] - 1
    else:
        col_sums = [int(local[:, col].sum()) for col in range(w)]
        body_h = min(col_sums)
        body_bbox = (r1 - body_h + 1, c0, r1, c1)
        length = h - body_h
        tail_anchor = (r1 - body_h, c0 + tail_pos)
        target_axis = "col"
        target_index = _index_between(frame.line_cols, tail_anchor[1])
        priority_distance = body_bbox[0] - frame.bbox[2] - 1

    assert length >= 2, "tail must have length at least 2"
    assert priority_distance > length, "tail must stop before the frame"
    return TailObject(
        color=int(color),
        body_bbox=tuple(map(int, body_bbox)),
        direction=(int(dr), int(dc)),
        length=int(length),
        tail_anchor=tuple(map(int, tail_anchor)),
        target_axis=target_axis,
        target_index=int(target_index),
        fill_color=int(color if length % 2 == 1 else GREY),
        priority_distance=int(priority_distance),
    )


def _analyze(grid: np.ndarray) -> tuple[FrameInfo, list[TailObject]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert get_bg(g) == BG, "expected black background"
    frame = _detect_frame(g)
    colors = [int(color) for color in np.unique(g) if color not in (BG, FRAME)]
    assert 3 <= len(colors) <= 5, "expected 3 to 5 tailed rectangles"
    objects = [_detect_object(g, frame, color) for color in colors]
    row_targets = [obj.target_index for obj in objects if obj.target_axis == "row"]
    col_targets = [obj.target_index for obj in objects if obj.target_axis == "col"]
    assert row_targets and col_targets, "need at least one row-target and one column-target object"
    assert len(row_targets) == len(set(row_targets)), "row-target objects must use distinct cell rows"
    assert len(col_targets) == len(set(col_targets)), "column-target objects must use distinct cell columns"
    return frame, objects


def _winner(current: TailObject | None, challenger: TailObject) -> TailObject:
    if current is None:
        return challenger
    cur_key = (current.length, -current.priority_distance)
    new_key = (challenger.length, -challenger.priority_distance)
    assert cur_key != new_key, "overlapping objects cannot have identical priority"
    return challenger if new_key > cur_key else current


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill selected lattice cells using the tailed outer rectangles while erasing those rectangles afterward.

    Input structure:
    - The background is black, and the central structure is a full magenta lattice with an outer border plus uniform internal lines, making a 3x3 or 4x4 array of closed cells.
    - Outside the lattice are 3 to 5 distinct solid colored rectangles, each with a same-colored 1-pixel-wide cardinal tail pointing toward the lattice but stopping short of it.
    - Each tail lies on a unique lattice row or unique lattice column, and at least one horizontal-tail object and one vertical-tail object are present.

    Rule:
    - Decode each outer object from its tail: the tail direction selects either one cell row or one cell column of the lattice, the tail length parity selects the fill color (odd -> the rectangle's own color, even -> grey), and the absolute tail length sets overlap priority.
    - A horizontal tail targets every cell in the lattice row crossed by that ray; a vertical tail targets every cell in the lattice column crossed by that ray.
    - When multiple objects target the same cell, the longer tail wins; if tail lengths tie, the rectangle whose body is closer to the lattice wins.
    - Output only the magenta lattice plus the resolved filled cell interiors; all outer rectangles and tails are removed, and untouched cells stay black.

    Key insight:
    - The same geometric tail simultaneously encodes the target line, the fill-color rule through odd/even length, and the conflict priority through absolute length and body distance.
    """
    g = np.asarray(grid, dtype=int)
    frame, objects = _analyze(g)
    out = np.full_like(g, BG)
    out[g == FRAME] = FRAME

    winners: list[list[TailObject | None]] = [[None for _ in range(frame.n_cells)] for _ in range(frame.n_cells)]
    for obj in objects:
        if obj.target_axis == "row":
            targets = [(obj.target_index, col) for col in range(frame.n_cells)]
        else:
            targets = [(row, obj.target_index) for row in range(frame.n_cells)]
        for row_idx, col_idx in targets:
            winners[row_idx][col_idx] = _winner(winners[row_idx][col_idx], obj)

    for row_idx in range(frame.n_cells):
        for col_idx in range(frame.n_cells):
            winner = winners[row_idx][col_idx]
            if winner is None:
                continue
            rs, cs = _cell_slices(frame, row_idx, col_idx)
            out[rs, cs] = winner.fill_color
    return out


def _build_frame(n_cells: int, cell_size: int) -> np.ndarray:
    span = n_cells * cell_size + n_cells + 1
    grid = np.zeros((span, span), dtype=int)
    lines = [i * (cell_size + 1) for i in range(n_cells + 1)]
    for r in lines:
        grid[r, :] = FRAME
    for c in lines:
        grid[:, c] = FRAME
    return grid


def _dilated(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    coords = np.argwhere(mask)
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _paint_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    side: str,
    frame: FrameInfo,
    target_index: int,
    color: int,
    body_h: int,
    body_w: int,
    tail_len: int,
    clear: int,
    rng: np.random.Generator,
) -> bool:
    fr0, fc0, fr1, fc1 = frame.bbox
    cells = np.zeros_like(grid, dtype=bool)

    if side in ("left", "right"):
        band_top = frame.line_rows[target_index] + 1
        band_bottom = frame.line_rows[target_index + 1] - 1
        candidate_rows = list(range(max(0, band_top - body_h + 1), min(grid.shape[0] - body_h, band_bottom) + 1))
        rng.shuffle(candidate_rows)
        if side == "left":
            body_c1 = fc0 - clear - tail_len - 1
            body_c0 = body_c1 - body_w + 1
            tail_c0 = body_c1 + 1
            tail_c1 = body_c1 + tail_len
        else:
            body_c0 = fc1 + clear + tail_len + 1
            body_c1 = body_c0 + body_w - 1
            tail_c0 = body_c0 - tail_len
            tail_c1 = body_c0 - 1
        if body_c0 < 0 or body_c1 >= grid.shape[1]:
            return False
        for body_r0 in candidate_rows:
            cells[:, :] = False
            body_r1 = body_r0 + body_h - 1
            row_choices = [r for r in range(max(band_top, body_r0), min(band_bottom, body_r1) + 1)]
            rng.shuffle(row_choices)
            for tail_r in row_choices:
                cells[body_r0 : body_r1 + 1, body_c0 : body_c1 + 1] = True
                cells[tail_r, tail_c0 : tail_c1 + 1] = True
                if np.any(cells & occupied):
                    cells[:, :] = False
                    continue
                grid[cells] = color
                occupied |= cells
                return True
    else:
        band_left = frame.line_cols[target_index] + 1
        band_right = frame.line_cols[target_index + 1] - 1
        candidate_cols = list(range(max(0, band_left - body_w + 1), min(grid.shape[1] - body_w, band_right) + 1))
        rng.shuffle(candidate_cols)
        if side == "top":
            body_r1 = fr0 - clear - tail_len - 1
            body_r0 = body_r1 - body_h + 1
            tail_r0 = body_r1 + 1
            tail_r1 = body_r1 + tail_len
        else:
            body_r0 = fr1 + clear + tail_len + 1
            body_r1 = body_r0 + body_h - 1
            tail_r0 = body_r0 - tail_len
            tail_r1 = body_r0 - 1
        if body_r0 < 0 or body_r1 >= grid.shape[0]:
            return False
        for body_c0 in candidate_cols:
            cells[:, :] = False
            body_c1 = body_c0 + body_w - 1
            col_choices = [c for c in range(max(band_left, body_c0), min(band_right, body_c1) + 1)]
            rng.shuffle(col_choices)
            for tail_c in col_choices:
                cells[body_r0 : body_r1 + 1, body_c0 : body_c1 + 1] = True
                cells[tail_r0 : tail_r1 + 1, tail_c] = True
                if np.any(cells & occupied):
                    cells[:, :] = False
                    continue
                grid[cells] = color
                occupied |= cells
                return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_cells = int(rng.choice([3, 4]))
    cell_size = int(rng.choice([2, 3]))
    if n_cells == 4 and cell_size == 3:
        cell_size = 2

    total = int(rng.choice([3, 4] if n_cells == 3 else [3, 4, 5]))
    num_h = int(rng.integers(1, min(3, total - 1, n_cells - 1) + 1))
    num_v = total - num_h
    if num_v < 1 or num_v >= n_cells:
        return None

    row_indices = list(map(int, rng.choice(np.arange(n_cells), size=num_h, replace=False)))
    col_indices = list(map(int, rng.choice(np.arange(n_cells), size=num_v, replace=False)))
    rng.shuffle(row_indices)
    rng.shuffle(col_indices)

    h_side = str(rng.choice(["left", "right"]))
    v_side = str(rng.choice(["top", "bottom"]))
    colors = list(map(int, rng.choice(np.array(OBJECT_COLORS, dtype=int), size=total, replace=False)))
    rng.shuffle(colors)

    body_sizes = [BODY_SIZES[int(rng.integers(len(BODY_SIZES)))] for _ in range(total)]
    lengths = [int(rng.integers(2, 7)) for _ in range(total)]
    clears = [int(rng.integers(1, 4)) for _ in range(total)]

    tie_case = bool(rng.integers(2))
    hi = 0
    vi = num_h
    if tie_case:
        shared = int(rng.choice([3, 5]))
        lengths[hi] = shared
        lengths[vi] = shared
        clears[hi] = 1
        clears[vi] = 3
    else:
        lengths[hi] = int(rng.choice([2, 4, 6]))
        lengths[vi] = int(rng.choice([3, 5]))
        clears[hi] = int(rng.integers(1, 3))
        clears[vi] = int(rng.integers(1, 3))

    adjust_idx = next(idx for idx in range(total) if idx not in (hi, vi))
    if all(length % 2 == 0 for length in lengths):
        lengths[adjust_idx] += 1 if lengths[adjust_idx] < 6 else -1
    if all(length % 2 == 1 for length in lengths):
        lengths[adjust_idx] += 1 if lengths[adjust_idx] < 6 else -1

    frame_only = _build_frame(n_cells, cell_size)
    span = frame_only.shape[0]

    left_need = 2
    right_need = 2
    top_need = 2
    bottom_need = 2

    for idx in range(num_h):
        body_h, body_w = body_sizes[idx]
        need = body_w + lengths[idx] + clears[idx]
        if h_side == "left":
            left_need = max(left_need, need)
        else:
            right_need = max(right_need, need)
        top_need = max(top_need, body_h - 1)
        bottom_need = max(bottom_need, body_h - 1)

    for idx in range(num_h, total):
        body_h, body_w = body_sizes[idx]
        need = body_h + lengths[idx] + clears[idx]
        if v_side == "top":
            top_need = max(top_need, need)
        else:
            bottom_need = max(bottom_need, need)
        left_need = max(left_need, body_w - 1)
        right_need = max(right_need, body_w - 1)

    height = top_need + span + bottom_need
    width = left_need + span + right_need
    if height > 30 or width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)

    fr0 = top_need
    fc0 = left_need
    grid[fr0 : fr0 + span, fc0 : fc0 + span] = frame_only
    occupied |= (grid == FRAME)
    frame = _detect_frame(grid)

    specs = []
    for idx in range(num_h):
        specs.append((h_side, row_indices[idx], colors[idx], *body_sizes[idx], lengths[idx], clears[idx]))
    for offset, idx in enumerate(range(num_h, total)):
        specs.append((v_side, col_indices[offset], colors[idx], *body_sizes[idx], lengths[idx], clears[idx]))
    rng.shuffle(specs[2:])

    for side, target_idx, color, body_h, body_w, tail_len, clear in specs:
        ok = _paint_object(
            grid,
            occupied,
            side,
            frame,
            int(target_idx),
            int(color),
            int(body_h),
            int(body_w),
            int(tail_len),
            int(clear),
            rng,
        )
        if not ok:
            return None

    try:
        frame_info, objects = _analyze(grid)
        out = main(grid)
    except Exception:
        return None

    filled_colors = {int(v) for v in np.unique(out) if v not in (BG, FRAME)}
    if GREY not in filled_colors or not any(color in filled_colors for color in OBJECT_COLORS):
        return None

    winners = [[None for _ in range(frame_info.n_cells)] for _ in range(frame_info.n_cells)]
    contenders: dict[tuple[int, int], list[TailObject]] = {}
    for obj in objects:
        if obj.target_axis == "row":
            cells = [(obj.target_index, col) for col in range(frame_info.n_cells)]
        else:
            cells = [(row, obj.target_index) for row in range(frame_info.n_cells)]
        for cell in cells:
            contenders.setdefault(cell, []).append(obj)
            winners[cell[0]][cell[1]] = _winner(winners[cell[0]][cell[1]], obj)

    conflict_cells = [cell for cell, objs in contenders.items() if len(objs) >= 2]
    if not conflict_cells:
        return None

    visible_conflict = False
    tie_visible = False
    black_cell_exists = False
    for row_idx in range(frame_info.n_cells):
        for col_idx in range(frame_info.n_cells):
            rs, cs = _cell_slices(frame_info, row_idx, col_idx)
            cell_vals = set(map(int, out[rs, cs].flatten()))
            if cell_vals == {BG}:
                black_cell_exists = True
            objs = contenders.get((row_idx, col_idx), [])
            if len(objs) >= 2:
                colors_here = {obj.fill_color for obj in objs}
                if len(colors_here) >= 2:
                    visible_conflict = True
                if len(objs) == 2 and objs[0].length == objs[1].length and objs[0].priority_distance != objs[1].priority_distance:
                    if objs[0].fill_color != objs[1].fill_color:
                        tie_visible = True

    if not black_cell_exists or not visible_conflict:
        return None
    if tie_case and not tie_visible:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tailed-lattice input")
