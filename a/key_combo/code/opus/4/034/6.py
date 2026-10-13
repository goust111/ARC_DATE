import random
import sys
from pathlib import Path

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg


BG = 0
BAR_COLORS = (1, 2, 3)
BASE_COLOR = 4
MARKER_COLOR = 7
MAX_ATTEMPTS = 400


def _map_margin_index(idx: int, src_len: int, dst_len: int) -> int | None:
    near_start = int(idx)
    near_end = int(src_len - 1 - idx)
    if near_start == near_end:
        return None
    if near_start < near_end:
        mapped = near_start
    else:
        mapped = int(dst_len - 1 - near_end)
    if 0 <= mapped < dst_len:
        return int(mapped)
    return None


def _bar_rows(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bars: list[tuple[int, int, int]] = []
    for r in range(h):
        row = g[r, :]
        bg_count = int(np.sum(row == BG))
        if bg_count != 1:
            continue
        fg = row[row != BG]
        if fg.size != w - 1:
            continue
        colors = {int(x) for x in fg}
        if len(colors) != 1:
            continue
        color = colors.pop()
        if color not in BAR_COLORS:
            continue
        gap_c = int(np.flatnonzero(row == BG)[0])
        bars.append((int(r), gap_c, color))
    bars.sort()
    return bars


def _rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for comp in find_components(grid, bg=BG, connectivity=4, monochromatic=False, return_format="dict"):
        bbox = tuple(map(int, comp["bbox"]))
        h = bbox_h(bbox)
        w = bbox_w(bbox)
        if h < 3 or w < 4:
            continue
        if int(comp["area"]) != h * w:
            continue
        sub = crop_bbox(grid, bbox)
        if not np.all(sub != BG):
            continue
        colors = {int(x) for x in sub.reshape(-1)}
        if colors <= {BASE_COLOR, MARKER_COLOR} and BASE_COLOR in colors:
            rects.append({"bbox": bbox, "sub": sub, "colors": colors})
    rects.sort(key=lambda item: item["bbox"])
    return rects


def _split_rectangles(grid: np.ndarray) -> tuple[dict, dict]:
    rects = _rectangles(grid)
    assert len(rects) == 2, f"expected exactly two rectangles, found {len(rects)}"
    marked = [rect for rect in rects if MARKER_COLOR in rect["colors"]]
    plain = [rect for rect in rects if rect["colors"] == {BASE_COLOR}]
    assert len(marked) == 1 and len(plain) == 1, "expected one marked rectangle and one plain rectangle"
    return plain[0], marked[0]


def _transfer_positions(marked_sub: np.ndarray, target_shape: tuple[int, int]) -> set[tuple[int, int]]:
    src_h, src_w = map(int, marked_sub.shape)
    dst_h, dst_w = map(int, target_shape)
    positions: set[tuple[int, int]] = set()
    for r, c in np.argwhere(marked_sub == MARKER_COLOR):
        rr = _map_margin_index(int(r), src_h, dst_h)
        cc = _map_margin_index(int(c), src_w, dst_w)
        if rr is None or cc is None:
            continue
        if 0 <= rr < dst_h and 0 <= cc < dst_w:
            positions.add((int(rr), int(cc)))
    return positions


def _fill_segments(bars: list[tuple[int, int, int]], height: int) -> list[tuple[int, int, int, int]]:
    segments: list[tuple[int, int, int, int]] = []
    for i, (row, col, color) in enumerate(bars):
        end_row = int(height - 1 if i + 1 == len(bars) else bars[i + 1][0] - 1)
        if end_row >= row:
            segments.append((int(row), end_row, int(col), int(color)))
    return segments


def _plain_overlay_data(
    grid: np.ndarray,
) -> tuple[dict, dict, list[tuple[int, int, int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    bars = _bar_rows(grid)
    assert 2 <= len(bars) <= 3, "expected 2 or 3 broken horizontal bars"
    plain, marked = _split_rectangles(grid)
    pr0, pc0, pr1, pc1 = plain["bbox"]
    mr0, _mc0, mr1, _mc1 = marked["bbox"]
    assert pr0 > bars[-1][0] and mr0 > bars[-1][0], "bars must be above both rectangles"
    assert max(pr0, mr0) <= min(pr1, mr1), "rectangles must overlap in rows"

    target_shape = (bbox_h(plain["bbox"]), bbox_w(plain["bbox"]))
    transferred = _transfer_positions(marked["sub"], target_shape)

    stripe_cells: set[tuple[int, int]] = set()
    stripe_segments: list[tuple[int, int, int, int]] = []
    for start_r, end_r, col, color in _fill_segments(bars, grid.shape[0]):
        if not (pc0 <= col <= pc1):
            continue
        rr0 = max(int(start_r), pr0)
        rr1 = min(int(end_r), pr1)
        if rr0 > rr1:
            continue
        local_c = int(col - pc0)
        stripe_segments.append((rr0, rr1, local_c, color))
        for r in range(rr0, rr1 + 1):
            stripe_cells.add((int(r - pr0), local_c))

    return plain, marked, stripe_segments, transferred, stripe_cells


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Crop the plain rectangle after transferring the marked rectangle's edge-relative orange pixels and overwriting them with the gap-column stripe.

    Input structure:
    - Black background surrounds 2 or 3 full-width horizontal bars near the top; each bar has exactly one black gap cell inside an otherwise solid row.
    - Lower in the grid there are exactly two dense yellow rectangles with overlapping row spans: one is plain, the other contains orange marker pixels.
    - The orange markers sit at asymmetric edge-relative offsets inside the marked rectangle, and at least one lowest-bar gap column passes through the plain rectangle.

    Rule:
    - Detect each broken horizontal bar and its gap column.
    - Transfer every orange marker from the marked rectangle to the plain rectangle by preserving its row margin to the nearer top/bottom edge and its column margin to the nearer left/right edge.
    - For every gap column, fill downward from that bar row in the bar color until the next bar row or the bottom of the grid; when this vertical fill crosses the plain rectangle, it overwrites any transferred orange marker there.
    - Return only the cropped plain rectangle.

    Key insight:
    - The output is the plain rectangle's crop after combining two independent mechanisms: edge-margin pattern transfer from the marked rectangle and vertical stripe painting controlled by the broken full-width bars, with the stripe winning on conflicts.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2
    _ = get_bg(grid)

    plain, _marked, stripe_segments, transferred, _stripe_cells = _plain_overlay_data(grid)
    pr0, _pc0, pr1, _pc1 = plain["bbox"]
    out = np.full((bbox_h(plain["bbox"]), bbox_w(plain["bbox"])), BASE_COLOR, dtype=int)

    for r, c in transferred:
        out[int(r), int(c)] = MARKER_COLOR

    for rr0, rr1, local_c, color in stripe_segments:
        local_r0 = int(rr0 - pr0)
        local_r1 = int(rr1 - pr0 + 1)
        out[local_r0:local_r1, int(local_c)] = int(color)

    return out


def _target_of_cell(r: int, c: int, src_shape: tuple[int, int], dst_shape: tuple[int, int]) -> tuple[int, int] | None:
    rr = _map_margin_index(r, src_shape[0], dst_shape[0])
    cc = _map_margin_index(c, src_shape[1], dst_shape[1])
    if rr is None or cc is None:
        return None
    if 0 <= rr < dst_shape[0] and 0 <= cc < dst_shape[1]:
        return int(rr), int(cc)
    return None


def _marker_candidates(src_shape: tuple[int, int], dst_shape: tuple[int, int]) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int]]]:
    visible: list[tuple[int, int, int, int]] = []
    dropped: list[tuple[int, int]] = []
    for r in range(src_shape[0]):
        for c in range(src_shape[1]):
            target = _target_of_cell(r, c, src_shape, dst_shape)
            if target is None:
                dropped.append((int(r), int(c)))
            else:
                visible.append((int(r), int(c), int(target[0]), int(target[1])))
    return visible, dropped


def _pick_marker_cells(marked_shape: tuple[int, int], plain_shape: tuple[int, int]) -> tuple[set[tuple[int, int]], tuple[int, int], set[tuple[int, int]]]:
    visible, dropped = _marker_candidates(marked_shape, plain_shape)
    random.shuffle(visible)

    by_target: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for item in visible:
        by_target.setdefault((item[2], item[3]), item)
    unique_visible = list(by_target.values())
    assert len(unique_visible) >= 2

    conflict = random.choice(unique_visible)
    alternatives = [item for item in unique_visible if item[3] != conflict[3]]
    assert alternatives
    survivor = random.choice(alternatives)

    chosen_src = {(conflict[0], conflict[1]), (survivor[0], survivor[1])}
    used_targets = {(conflict[2], conflict[3]), (survivor[2], survivor[3])}

    extra_count = random.randint(1, min(3, max(1, len(unique_visible) - 2)))
    extra_pool = [item for item in unique_visible if (item[2], item[3]) not in used_targets]
    random.shuffle(extra_pool)
    for item in extra_pool[:extra_count]:
        chosen_src.add((item[0], item[1]))
        used_targets.add((item[2], item[3]))

    if dropped and random.random() < 0.35:
        chosen_src.add(random.choice(dropped))

    survivor_targets = {(item[2], item[3]) for item in unique_visible if (item[0], item[1]) in chosen_src and item[3] != conflict[3]}
    return chosen_src, (int(conflict[2]), int(conflict[3])), survivor_targets


def _choose_bar_rows(height: int, rect_top: int, bar_count: int) -> list[int]:
    usable = list(range(1, rect_top - 1))
    if len(usable) < bar_count:
        raise ValueError("not enough rows for bars")
    while True:
        rows = sorted(random.sample(usable, bar_count))
        if rows[-1] <= rect_top - 2:
            return rows


def _render_rectangle(grid: np.ndarray, top: int, left: int, height: int, width: int, marked_cells: set[tuple[int, int]] | None = None) -> None:
    grid[top : top + height, left : left + width] = BASE_COLOR
    if marked_cells:
        for r, c in marked_cells:
            grid[top + int(r), left + int(c)] = MARKER_COLOR


def _make_attempt() -> np.ndarray | None:
    marked_h = random.randint(4, 6)
    plain_h = random.randint(4, 6)
    marked_w = random.randint(5, 8)
    plain_w = random.randint(marked_w, min(9, marked_w + 3))

    rect_gap = random.randint(2, 4)
    total_rect_w = plain_w + marked_w + rect_gap
    min_w = max(15, total_rect_w + 3)
    max_w = min(22, total_rect_w + 7)
    if min_w > max_w:
        return None
    width = random.randint(min_w, max_w)

    height = random.randint(max(15, max(plain_h, marked_h) + 9), 20)
    grid = np.zeros((height, width), dtype=int)

    pair_left = random.randint(1, width - total_rect_w - 1)
    plain_on_left = random.random() < 0.5
    if plain_on_left:
        plain_c0 = pair_left
        marked_c0 = pair_left + plain_w + rect_gap
    else:
        marked_c0 = pair_left
        plain_c0 = pair_left + marked_w + rect_gap

    base_top_min = max(height // 2, 5)
    base_top_max = height - max(plain_h, marked_h) - 1
    if base_top_min > base_top_max:
        return None
    anchor_top = random.randint(base_top_min, base_top_max)

    plain_r0 = anchor_top + random.choice([0, 0, 1, -1])
    marked_r0 = anchor_top + random.choice([0, 0, 1, -1])
    plain_r0 = max(5, min(plain_r0, height - plain_h - 1))
    marked_r0 = max(5, min(marked_r0, height - marked_h - 1))
    if max(plain_r0, marked_r0) > min(plain_r0 + plain_h - 1, marked_r0 + marked_h - 1):
        return None

    chosen_markers, conflict_target, survivor_targets = _pick_marker_cells((marked_h, marked_w), (plain_h, plain_w))
    if not survivor_targets:
        return None

    _render_rectangle(grid, plain_r0, plain_c0, plain_h, plain_w)
    _render_rectangle(grid, marked_r0, marked_c0, marked_h, marked_w, chosen_markers)

    bar_count = random.choice([2, 3])
    rect_top = min(plain_r0, marked_r0)
    try:
        bar_rows = _choose_bar_rows(height, rect_top, bar_count)
    except ValueError:
        return None
    bar_colors = random.sample(list(BAR_COLORS), bar_count)

    conflict_col = plain_c0 + conflict_target[1]
    plain_cols = [plain_c0 + i for i in range(plain_w)]
    distractor_cols = [c for c in plain_cols if c != conflict_col]
    if not distractor_cols:
        return None
    distractor_col = random.choice(distractor_cols)

    outer_cols = [c for c in range(1, width - 1) if c not in {conflict_col, distractor_col}]
    if not outer_cols:
        return None

    gap_cols = [0] * bar_count
    gap_cols[-1] = conflict_col
    if bar_count == 2:
        gap_cols[0] = distractor_col
    else:
        gap_cols[1] = distractor_col
        gap_cols[0] = random.choice([conflict_col, random.choice(outer_cols)])

    for r, color, gap_c in zip(bar_rows, bar_colors, gap_cols):
        grid[int(r), :] = int(color)
        grid[int(r), int(gap_c)] = BG

    return grid


def _validate_instance(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30

    bars = _bar_rows(grid)
    assert 2 <= len(bars) <= 3
    plain, marked, stripe_segments, transferred, stripe_cells = _plain_overlay_data(grid)

    pr0, pc0, pr1, pc1 = plain["bbox"]
    mr0, _mc0, mr1, _mc1 = marked["bbox"]
    assert pr0 > bars[-1][0] and mr0 > bars[-1][0]
    assert max(pr0, mr0) <= min(pr1, mr1)
    assert bbox_w(plain["bbox"]) >= bbox_w(marked["bbox"])

    assert stripe_segments, "plain rectangle must receive at least one vertical stripe"
    assert transferred, "there must be transferable marker cells"
    conflict = transferred & stripe_cells
    survivors = transferred - stripe_cells
    assert conflict, "there must be at least one overwrite conflict"
    assert survivors, "there must be at least one surviving transferred marker"

    earlier_plain_gaps = [col for row, col, _color in bars[:-1] if pc0 <= col <= pc1 and col != bars[-1][1]]
    if len(bars) >= 2:
        assert earlier_plain_gaps, "need a distractor gap inside the plain rectangle from an upper bar"

    out = main(grid)
    assert out.shape == (bbox_h(plain["bbox"]), bbox_w(plain["bbox"]))
    assert np.any(np.isin(out, BAR_COLORS)), "output must contain stripe color"
    assert np.any(out == MARKER_COLOR), "output must contain surviving transferred markers"
    assert np.any(out == BASE_COLOR), "output must retain base color"


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _make_attempt()
        if grid is None:
            continue
        try:
            _validate_instance(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("generate_input failed after repeated attempts")
