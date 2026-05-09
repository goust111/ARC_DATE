# Combo variant of 2ba387bc (index 35)
# combo_spec: D12/2+D15/1
# Name: 彩色分隔线+不同输出背景

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


OUTPUT_BG = 0
SEPARATOR_COLOR = 1
INPUT_BGS = (2, 4, 8)


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and small trace
#    - Original task: detect scattered 4x4 objects on a uniform input background.
#      Hollow 4x4 frames go to the left output column, full 4x4 solids go to the
#      right output column. Each side is sorted by input row and stacked with no gap.
#    - Combo behavior keeps that semantic split and ordering, but changes two output
#      rendering rules at the same time:
#         1. output background is a new fixed color 0, not the input background;
#         2. a full-height colored separator column is inserted between the two
#            packed columns.
#    - Therefore output width changes from 8 to 9:
#         cols 0..3 = frames
#         col 4     = colored separator
#         cols 5..8 = solids
#    - Small trace:
#         input bg = 4
#         one frame of color 6, one solid of color 7
#      Original output would be:
#         66667777
#         64467777
#         64467777
#         66667777
#      Combo output must be:
#         666617777
#         600617777
#         600617777
#         666617777
#      Differences are visible in two places at once:
#         - the frame center switched from input bg 4 to output bg 0,
#         - the new blue separator line sits between the columns.
#
# b) Complexity calibration
#    - The new background uses one fixed output color (0) while input backgrounds are
#      sampled only from {2,4,8}. This makes the background-role change obvious
#      without forcing a hard per-sample color mapping.
#    - The separator uses one reserved constant color (1). Keeping it constant avoids
#      arbitrary per-sample color remapping while still making D15/1 immediately
#      visible. Humans can infer "this colored line is the boundary between groups"
#      from 2-4 examples.
#    - Generator always produces at least one frame and one solid, so the separator
#      is never decorative-only; both sides of the split are populated.
#
# c) Original solver assumption audit
#    - `bg = get_bg(g)`: still valid, but only for interpreting the input.
#    - `find_components(..., bg=bg)`: still valid because input structure is unchanged.
#    - `bbox_h == bbox_w == 4`: still valid because only sortable objects are 4x4.
#    - `sub[1:3,1:3] == bg` for frame detection: still valid on the input side.
#    - `out = np.full((outH, 8), bg)`: invalid now. Output width is 9 and background
#      is OUTPUT_BG rather than input bg.
#    - `paste(out, sub, ...)` for frames: invalid now. Reusing the raw input crop would
#      keep the old input background inside the frame center. Frames must be rebuilt on
#      the new output background.
#    - Right-column origin `4`: invalid now. Solids start at col 5 because col 4 is the
#      separator line.
#    - Multi-change interaction: if we inserted the separator but still pasted raw frame
#      crops, the output would show three roles for the old input background
#      (outer blank space, frame centers, and not-separator), making the D12 rule much
#      less inferable. Rebuilding frames against OUTPUT_BG removes that conflict.
#
# d) Placement legality
#    - Generator still uses `random_free_location(... padding=1, connectivity=4)` on a
#      uniform input background, so 4x4 objects never overlap or touch orthogonally.
#    - Every placed object is exactly 4x4 and stays inside bounds.
#    - Since the separator exists only in the output, it cannot create accidental input
#      components or touch any scattered object.
#
# e) Visual observability
#    - Across examples, a human sees the same left/right sorting as the source task, but
#      every output also has:
#         * a new black background even when the input background is not black;
#         * a full-height colored divider between the two stacks.
#    - The rule depends only on visible structure and color roles, not on hidden numeric
#      properties of color ids.
#
# f) Visual cleanliness
#    - The output contains only three clean visual regions: left pile, 1-column
#      separator, right pile. No overlaps or layered markers occur.
#    - The separator color is reserved and never used by the input generator for object
#      colors or input background, so it reads as a deliberate symbol rather than a
#      stray object color.
#
# g) Change inferability and non-degeneration metric
#    - Core metric:
#         * `input_bg != OUTPUT_BG`
#         * output column 4 is uniformly `SEPARATOR_COLOR`
#         * combo output differs from the original-style output on the same input
#      The generator audits and rejects any sample violating these conditions.
#    - Because the original-style output is 8 columns wide and preserves the input
#      background, every accepted sample visibly exercises both fused changes.
#
# h) New solver assumptions introduced by this combo
#    - Assumption: every non-frame 4x4 object is a monochrome solid block with exactly
#      one non-background color. Generator guarantee: yes, solids are created only by
#      `_make_solid`.
#    - Assumption: separator color 1 is reserved and never appears inside input objects.
#      Generator guarantee: object palette excludes `SEPARATOR_COLOR`.
#    - Assumption: at least one frame exists, so the background-change effect is actually
#      visible inside an object, not only in blank space. Generator guarantee: number of
#      frames is sampled from 1..4.


def _make_frame(col: int, bg: int) -> np.ndarray:
    tile = np.full((4, 4), int(col), dtype=int)
    tile[1:3, 1:3] = int(bg)
    return tile


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _frame_color(sub: np.ndarray, input_bg: int) -> int:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    bcols = set(map(int, border.flatten()))
    if len(bcols) != 1:
        raise ValueError("frame border is not monochrome")
    if not bool((sub[1:3, 1:3] == int(input_bg)).all()):
        raise ValueError("frame center does not match input background")
    return int(next(iter(bcols)))


def _solid_color(sub: np.ndarray, input_bg: int) -> int:
    cols = sorted(set(map(int, sub[sub != int(input_bg)].flatten())))
    if len(cols) != 1:
        raise ValueError("solid is not monochrome")
    return int(cols[0])


def _assemble_output(
    frames: list[tuple[int, int, int]],
    solids: list[tuple[int, int, int]],
    *,
    output_bg: int = OUTPUT_BG,
    separator_color: int = SEPARATOR_COLOR,
) -> np.ndarray:
    out_h = 4 * max(len(frames), len(solids))
    out = np.full((out_h, 9), int(output_bg), dtype=int)
    out[:, 4] = int(separator_color)

    for i, (_r0, _c0, col) in enumerate(frames):
        paste(out, _make_frame(int(col), int(output_bg)), int(i * 4), 0)
    for i, (_r0, _c0, col) in enumerate(solids):
        paste(out, _make_solid(int(col)), int(i * 4), 5)

    return out


def _assemble_original_style(
    frames: list[tuple[int, int, int]],
    solids: list[tuple[int, int, int]],
    *,
    input_bg: int,
) -> np.ndarray:
    out_h = 4 * max(len(frames), len(solids))
    out = np.full((out_h, 8), int(input_bg), dtype=int)

    for i, (_r0, _c0, col) in enumerate(frames):
        paste(out, _make_frame(int(col), int(input_bg)), int(i * 4), 0)
    for i, (_r0, _c0, col) in enumerate(solids):
        paste(out, _make_solid(int(col)), int(i * 4), 4)

    return out


def _extract_sorted_objects(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    g = np.array(grid, dtype=int)
    input_bg = int(get_bg(g))
    comps = find_components(g, bg=input_bg, connectivity=4, monochromatic=False, return_format="dict")

    frames: list[tuple[int, int, int]] = []
    solids: list[tuple[int, int, int]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue

        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        try:
            frames.append((int(r0), int(c0), _frame_color(sub, input_bg)))
        except ValueError:
            solids.append((int(r0), int(c0), _solid_color(sub, input_bg)))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return input_bg, frames, solids


def combo_metrics(grid: np.ndarray) -> dict[str, int]:
    input_bg, frames, solids = _extract_sorted_objects(grid)
    combo = _assemble_output(frames, solids)
    original = _assemble_original_style(frames, solids, input_bg=input_bg)

    frame_center_count = 0
    for i in range(len(frames)):
        if bool((combo[i * 4 + 1:i * 4 + 3, 1:3] == int(OUTPUT_BG)).all()):
            frame_center_count += 1

    return {
        "input_bg": int(input_bg),
        "output_bg": int(OUTPUT_BG),
        "separator_color": int(SEPARATOR_COLOR),
        "frame_count": int(len(frames)),
        "solid_count": int(len(solids)),
        "output_width": int(combo.shape[1]),
        "separator_cells": int(np.sum(combo[:, 4] == int(SEPARATOR_COLOR))),
        "frame_centers_on_output_bg": int(frame_center_count),
        "differs_from_original": int(
            combo.shape != original.shape or not np.array_equal(combo, original)
        ),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Detect scattered 4x4 objects exactly as in the source task.
    - Classify hollow 4x4 objects as frames and full 4x4 objects as solids.
    - Sort each group by input row and stack frames left, solids right.
    - Rebuild the output on a new background color and insert a full-height colored
      separator column between the two packed groups.
    """
    _input_bg, frames, solids = _extract_sorted_objects(grid)
    return _assemble_output(frames, solids)


def _gen_attempt() -> np.ndarray | None:
    input_bg = int(np.random.choice(INPUT_BGS))
    h = int(np.random.randint(18, 30))
    w = int(np.random.randint(18, 30))
    g = np.full((h, w), input_bg, dtype=int)

    palette = [col for col in range(1, 10) if col not in {int(input_bg), int(OUTPUT_BG), int(SEPARATOR_COLOR)}]
    max_total = len(palette)
    if max_total < 2:
        return None

    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))
    if n_frames + n_solids > max_total:
        return None

    try:
        np.random.shuffle(palette)
        colors = [int(col) for col in palette[:n_frames + n_solids]]
        placed_frames: list[tuple[int, int, int]] = []
        placed_solids: list[tuple[int, int, int]] = []

        for col in colors[:n_frames]:
            patch = _make_frame(col, input_bg)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=input_bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=input_bg)
            placed_frames.append((int(r0), int(c0), int(col)))

        for col in colors[n_frames:]:
            patch = _make_solid(col)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=input_bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=input_bg)
            placed_solids.append((int(r0), int(c0), int(col)))

        placed_frames.sort(key=lambda t: (int(t[0]), int(t[1])))
        placed_solids.sort(key=lambda t: (int(t[0]), int(t[1])))

        expected = _assemble_output(placed_frames, placed_solids)
        actual = main(g)
        if not np.array_equal(actual, expected):
            raise ValueError("generator produced a grid inconsistent with main()")

        metrics = combo_metrics(g)
        if metrics["input_bg"] == metrics["output_bg"]:
            raise ValueError("background change degenerated")
        if metrics["separator_cells"] != int(expected.shape[0]):
            raise ValueError("separator line is not full height")
        if metrics["frame_centers_on_output_bg"] != len(placed_frames):
            raise ValueError("frame centers did not switch to output background")
        if metrics["differs_from_original"] != 1:
            raise ValueError("combo output degenerated to original behavior")

        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
