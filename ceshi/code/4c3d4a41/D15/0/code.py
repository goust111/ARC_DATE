# Variant of 4c3d4a41
# Dimension: D15 — src-dst基数关系 (multi_step)
# Change: 1:1关系 - 1个杆对1个框架... → 多杆对多框架(N:N)

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import get_bg, paste


# Design analysis
# yaml-like summary:
#   concretization:
#     - Keep the original per-object behavior: a grey bar with upward teeth is inserted
#       into a half-open grey frame, and each colored slot column is restacked above the
#       placed bar.
#     - Replace the single instance with N instances, N in {2, 3}.
#     - Each bar/frame pair is matched by visible size class K in {2, 3, 4}, where:
#         frame width = 2K + 3
#         bar width   = 2K - 1
#         slot count  = K
#       This makes pairing human-visible without using color ids or hidden metadata.
#     - Frames stay on the right, bars stay on the left, but size assignments to rows are
#       independently shuffled so at least one pair is cross-row. That makes the N:N
#       behavior observable instead of collapsing into several trivial 1:1 rowwise copies.
#   behavior_trace:
#     - Original: one largest grey component is the frame, one second-largest grey component
#       is the bar, then a single dx/dy is computed and one transfer happens.
#     - Variant: detect every frame component and every bar component; infer each object's
#       size class K from its bbox width; pair each frame with the bar of the same K; apply
#       the original insert-and-restack rule separately for each pair.
#     - Example:
#         frame K=2 at rows 18..25, bar K=2 at rows 2..5
#         frame K=4 at rows 0..7,  bar K=4 at rows 20..23
#       The old solver would wrongly keep only one frame and one bar. The new solver builds
#       two independent transfers, so both frames receive their matching bars.
#   complexity_calibration:
#     - Only three size classes are used. This keeps pairing unambiguous and visually easy
#       to infer from 2-3 examples.
#     - N is limited to 2 or 3, which is enough to demonstrate multi-step matching without
#       cluttering the grid.
#     - Colors remain local content markers inside each frame; they do not control pairing.
#   implicit_assumption_audit:
#     - Old assumption "largest grey component is the only frame": no longer valid.
#       Replaced with explicit frame-shape detection.
#     - Old assumption "next-largest grey component is the only bar": no longer valid.
#       Replaced with explicit bar-shape detection and per-size grouping.
#     - Old assumption "one global slot bbox/output region": no longer valid.
#       Replaced with per-frame output preparation.
#     - Old assumption "one global dx": no longer valid.
#       Replaced with per-pair placement derived from frame-local slot columns.
#   placement_legality:
#     - Canvas is enlarged to 26x30 so three 8-row frames can stack vertically with gaps.
#     - Frames occupy fixed right-side lanes; bars occupy fixed left-side lanes.
#     - Width classes 7/9/11 for frames and 3/5/7 for bars fit without overlap.
#     - Generator independently shuffles size-to-row assignments but keeps objects in
#       disjoint lanes, so solver-visible geometry is always legal.
#   visual_observability:
#     - Matching is by width/slot-count, not by color id or hidden ordering.
#     - At least one bar/frame pair is row-mismatched, so the scene clearly differs from
#       "three unrelated copies of the old task".
#   visual_cleanliness:
#     - Only one frame per horizontal lane and one bar per left lane.
#     - No annotations are added on top of objects, so the frame interiors remain readable.
#   color_change_inference:
#     - No cross-color remapping is introduced. Colors only keep their original role as
#       per-slot stacks that are trimmed by the inserted bar height.
#   new_solver_assumptions_checked_against_generator:
#     - Frames use height 8 and widths in {7, 9, 11}: guaranteed by generator.
#     - Bars use widths in {3, 5, 7} with a full base row: guaranteed by generator.
#     - Each chosen size class appears exactly once as a frame and once as a bar, so
#       size-based pairing is well-defined.
#     - Every frame has colored cells in each designated slot column, so frame-local
#       destination columns are recoverable from the input.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 26, 30

FRAME_H = 8
FRAME_C0 = 18
FRAME_ROW_OPTIONS = [0, 9, 18]
LEFT_WALL_OPEN_ROWS = (4, 5)

BAR_H = 4
BAR_C0 = 1
BAR_ROW_OPTIONS = [2, 11, 20]

SIZE_OPTIONS = (2, 3, 4)
HEIGHT0_RANGE = (2, 5)
HEIGHT_REST_RANGE = (1, 5)
MAX_ATTEMPTS = 300


def _slot_lengths(k: int) -> list[int]:
    if k == 2:
        return [3, 4]
    return [3] + [4] * (k - 2) + [3]


def _frame_width(k: int) -> int:
    return 2 * k + 3


def _bar_width(k: int) -> int:
    return 2 * k - 1


def _slot_local_cols(k: int) -> list[int]:
    return [2 + 2 * i for i in range(k)]


def _bar_proj_local_cols(k: int) -> list[int]:
    return [2 * i for i in range(k)]


def _frame_patch(k: int) -> np.ndarray:
    width = _frame_width(k)
    patch = np.zeros((FRAME_H, width), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, -1] = GREY
    patch[:, 0] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _bar_patch(heights: list[int]) -> np.ndarray:
    k = len(heights)
    width = _bar_width(k)
    patch = np.zeros((BAR_H, width), dtype=int)
    patch[-1, :] = GREY
    for local_c, height in zip(_bar_proj_local_cols(k), heights):
        patch[BAR_H - height : BAR_H, local_c] = GREY
    return patch


def _paint_slot_column(
    grid: np.ndarray,
    *,
    frame_r0: int,
    frame_c0: int,
    local_c: int,
    length: int,
    color: int,
    count: int,
    height: int | None,
) -> None:
    abs_c = frame_c0 + local_c
    top = frame_r0 + 1
    cap = length - count
    grid[top : top + cap, abs_c] = GREY
    if height is None:
        grid[top + cap : top + length, abs_c] = color
        return

    visible = min(count, 5 - int(height))
    if visible <= 0:
        return
    row_end = frame_r0 + 5 - int(height)
    grid[row_end - visible + 1 : row_end + 1, abs_c] = color


def _frame_kind(pts: np.ndarray) -> int | None:
    r0, c0, r1, c1 = _bbox(pts)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height != FRAME_H or width not in {_frame_width(k) for k in SIZE_OPTIONS}:
        return None
    ptset = {tuple(map(int, p)) for p in pts}
    if not all((r0, c) in ptset for c in range(c0, c1 + 1)):
        return None
    if not all((r1, c) in ptset for c in range(c0, c1 + 1)):
        return None
    if not all((r, c1) in ptset for r in range(r0, r1 + 1)):
        return None
    expected_left = {r0 + off for off in (0, 1, 2, 3, 6, 7)}
    actual_left = {r for r in range(r0, r1 + 1) if (r, c0) in ptset}
    if actual_left != expected_left:
        return None
    return (width - 3) // 2


def _bar_kind(pts: np.ndarray) -> int | None:
    r0, c0, r1, c1 = _bbox(pts)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height > BAR_H or width not in {_bar_width(k) for k in SIZE_OPTIONS}:
        return None
    ptset = {tuple(map(int, p)) for p in pts}
    if not all((r1, c) in ptset for c in range(c0, c1 + 1)):
        return None
    return (width + 1) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move several grey bars into their matching grey frames, then restack each frame's colors above its own bar.

    Input structure:
    - Background plus multiple grey structural components: N frames on the right and N bars on the left.
    - Each frame contains K colored slot columns and each bar has the corresponding width class K, where K is visibly encoded by object width.

    Rule:
    - Pair every frame with the bar of the same width class.
    - For each matched pair, remove the bar from its input location, place it into the frame so its base sits one cell above the interior floor, and align the bar's every-other-column teeth with that frame's colored slot columns.
    - For each slot column in that frame, keep its original top-to-bottom color order and stack the visible colored cells immediately above the placed bar in that same column.

    Key insight:
    - The single-instance "find one frame and one bar" logic no longer works; the solver must first decompose the scene into multiple grey objects, infer each object's size class, and then perform the original insertion rule independently for every matched class.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    other = g[g != bg]
    if other.size == 0:
        return g.copy()
    grey = _most_common(other)

    frames_by_k: dict[int, list[dict]] = {}
    bars_by_k: dict[int, list[dict]] = {}

    for pts in _cc4(g == grey):
        bbox = _bbox(pts)
        fk = _frame_kind(pts)
        if fk is not None:
            frames_by_k.setdefault(fk, []).append({"bbox": bbox, "pts": pts})
            continue
        bk = _bar_kind(pts)
        if bk is not None:
            bars_by_k.setdefault(bk, []).append({"bbox": bbox, "pts": pts})

    frame_count = sum(len(v) for v in frames_by_k.values())
    bar_count = sum(len(v) for v in bars_by_k.values())
    if frame_count == 0 or frame_count != bar_count:
        return g.copy()
    if set(frames_by_k) != set(bars_by_k):
        return g.copy()
    if any(len(frames_by_k[k]) != len(bars_by_k[k]) for k in frames_by_k):
        return g.copy()

    for comps in frames_by_k.values():
        comps.sort(key=lambda item: item["bbox"])
    for comps in bars_by_k.values():
        comps.sort(key=lambda item: item["bbox"])

    out = np.full_like(g, bg)
    for k in sorted(frames_by_k):
        for frame in frames_by_k[k]:
            sr0, sc0, sr1, sc1 = frame["bbox"]
            out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
            region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
            region[(region != bg) & (region != grey)] = bg

    for k in sorted(frames_by_k):
        for frame, bar in zip(frames_by_k[k], bars_by_k[k]):

            sr0, sc0, sr1, sc1 = frame["bbox"]
            _br0, bc0, br1, _bc1 = bar["bbox"]
            bar_pts = bar["pts"]

            dst_cols = sorted(
                {
                    int(c)
                    for r, c in np.argwhere((g != bg) & (g != grey))
                    if sr0 <= int(r) <= sr1 and sc0 <= int(c) <= sc1
                }
            )
            if not dst_cols:
                continue

            bar_base_row = max(int(r) for r, _ in bar_pts)
            target_base_row = sr1 - 2
            dy = target_base_row - bar_base_row
            dx = dst_cols[0] - bc0

            placed: list[tuple[int, int]] = []
            for r, c in bar_pts:
                rr = int(r) + dy
                cc = int(c) + dx
                if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
                    out[rr, cc] = grey
                    placed.append((rr, cc))

            placed_by_col: dict[int, list[int]] = {}
            for rr, cc in placed:
                placed_by_col.setdefault(cc, []).append(rr)

            interior_top = sr0 + 1
            interior_bottom = sr1 - 1
            for col in dst_cols:
                if col not in placed_by_col:
                    continue
                bar_top = min(placed_by_col[col])
                desired_bottom = bar_top - 1
                if desired_bottom < interior_top:
                    continue

                rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != grey]
                if not rows:
                    continue
                colors = [int(g[r, col]) for r in rows]

                max_len = desired_bottom - interior_top + 1
                if len(colors) > max_len:
                    colors = colors[-max_len:]
                start_row = max(desired_bottom - len(colors) + 1, interior_top)
                for i, colv in enumerate(colors):
                    rr = start_row + i
                    if interior_top <= rr <= interior_bottom:
                        out[rr, col] = colv

    return out


def _sample_heights(rng: np.random.Generator, k: int) -> list[int]:
    heights = [int(rng.integers(*HEIGHT0_RANGE))]
    heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(k - 1))
    return heights


def _build_input(
    frame_rows: list[int],
    frame_sizes: list[int],
    bar_rows: list[int],
    bar_sizes: list[int],
    records_by_k: dict[int, dict],
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    for frame_r0, k in zip(frame_rows, frame_sizes):
        paste(grid, _frame_patch(k), frame_r0, FRAME_C0)
    for bar_r0, k in zip(bar_rows, bar_sizes):
        paste(grid, _bar_patch(records_by_k[k]["heights"]), bar_r0, BAR_C0)
    for frame_r0, k in zip(frame_rows, frame_sizes):
        rec = records_by_k[k]
        for local_c, length, color, count in zip(
            _slot_local_cols(k), _slot_lengths(k), rec["colors"], rec["counts"]
        ):
            _paint_slot_column(
                grid,
                frame_r0=frame_r0,
                frame_c0=FRAME_C0,
                local_c=local_c,
                length=length,
                color=color,
                count=count,
                height=None,
            )
    return grid


def _build_output(frame_rows: list[int], frame_sizes: list[int], records_by_k: dict[int, dict]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    for frame_r0, k in zip(frame_rows, frame_sizes):
        paste(grid, _frame_patch(k), frame_r0, FRAME_C0)
        paste(grid, _bar_patch(records_by_k[k]["heights"]), frame_r0 + 2, FRAME_C0 + 2)
        rec = records_by_k[k]
        for local_c, length, color, count, height in zip(
            _slot_local_cols(k),
            _slot_lengths(k),
            rec["colors"],
            rec["counts"],
            rec["heights"],
        ):
            _paint_slot_column(
                grid,
                frame_r0=frame_r0,
                frame_c0=FRAME_C0,
                local_c=local_c,
                length=length,
                color=color,
                count=count,
                height=height,
            )
    return grid


def _sample_scene(rng: np.random.Generator) -> tuple[list[int], list[int], list[int], list[int], dict[int, dict]]:
    n = int(rng.choice([2, 3]))
    sizes = [int(v) for v in rng.choice(SIZE_OPTIONS, size=n, replace=False)]

    frame_rows = sorted(int(v) for v in rng.choice(FRAME_ROW_OPTIONS, size=n, replace=False))
    bar_rows = sorted(int(v) for v in rng.choice(BAR_ROW_OPTIONS, size=n, replace=False))

    frame_sizes = sizes.copy()
    bar_sizes = sizes.copy()
    rng.shuffle(frame_sizes)
    rng.shuffle(bar_sizes)
    if frame_sizes == bar_sizes:
        bar_sizes = bar_sizes[1:] + bar_sizes[:1]

    records_by_k: dict[int, dict] = {}
    for k in sizes:
        records_by_k[k] = {
            "colors": [int(v) for v in rng.choice(PALETTE, size=k, replace=False)],
            "counts": [int(rng.integers(1, L + 1)) for L in _slot_lengths(k)],
            "heights": _sample_heights(rng, k),
        }

    return frame_rows, frame_sizes, bar_rows, bar_sizes, records_by_k


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_rows, frame_sizes, bar_rows, bar_sizes, records_by_k = _sample_scene(rng)
    grid = _build_input(frame_rows, frame_sizes, bar_rows, bar_sizes, records_by_k)
    expected = _build_output(frame_rows, frame_sizes, records_by_k)

    if get_bg(grid) != BG:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
