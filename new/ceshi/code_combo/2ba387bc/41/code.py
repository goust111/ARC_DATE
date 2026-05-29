# Combo variant of 2ba387bc (index 41)
# combo_spec: D2/1+D5/3+D7/2
# Name: 重力堆叠+颜色互换+降序排序

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

# Design analysis for this combo variant
# a) Concrete behavior and trace:
#    - Input is still a sparse background with isolated 4x4 objects of two types:
#      hollow frames and filled solids.
#    - Original seed behavior:
#      1. classify frame vs solid,
#      2. sort each group by (row ascending, col ascending),
#      3. paste frames top-down on the left and solids top-down on the right,
#      4. keep every object's own color.
#    - Combo behavior:
#      1. classify the same way,
#      2. sort both groups by (row descending, col ascending),
#      3. pair the i-th ordered frame with the i-th ordered solid and swap their colors,
#      4. paste frames from the bottom upward on the left, solids from the top downward on the right.
#    - Small trace:
#      frames at rows [2, 10, 17] with colors [2, 4, 6], solids at rows [5, 14, 19] with colors [3, 7, 8].
#      Seed order is frames [2,4,6], solids [3,7,8].
#      Combo order becomes frames [6,4,2], solids [8,7,3].
#      After paired color swap by ordered index:
#        ordered frames carry colors [8,7,3], ordered solids carry colors [6,4,2].
#      Then solids are pasted top-down, so the right column top-to-bottom shows colors [6,4,2].
#      Frames are pasted bottom-up, so the bottommost left frame gets color 8, the one above gets 7,
#      and the topmost left frame gets 3. The left visible top-to-bottom sequence can differ from the
#      ordered sequence because gravity reverses its physical placement direction.
#    - The key difference from naive "same-row swap" is that color exchange is by ordered index before
#      placement; opposite gravity can move the swapped colors to different final heights.
# b) Complexity calibration:
#    - The combo already has three moving parts, so the generator keeps shape vocabulary minimal:
#      only classic 4x4 frames and solids.
#    - To make all three factors visible, samples use at least 2 frames and 3 solids, unique colors
#      across all objects, and distinct rows inside each type. More solids than frames ensures the
#      left column has visible top padding from bottom gravity.
#    - This avoids over-complicating the task while still forcing the solver to track ordering,
#      destination slots, and cross-column color transport.
# c) Implicit assumption review of the original main():
#    - `find_components(..., monochromatic=False)` assumes objects are disconnected by background.
#      Still required, so the generator uses padded placement.
#    - `bbox_h == bbox_w == 4` assumes every relevant object is exactly 4x4. Still true.
#    - The frame test assumes one non-bg border color plus bg 2x2 center. Still true.
#    - The old `sort(key=(row, col))` no longer matches the task; it must become row descending.
#    - The old frame destination `i*4` assumes top gravity; the combo needs `out_h - 4*(i+1)`.
#    - The old "keep patch unchanged" assumption fails because color now depends on its paired object.
# d) Placement legality:
#    - Generator placement uses `random_free_location(..., border=1, padding=1, padding_connectivity=4)`.
#      This keeps every 4x4 object in bounds, non-overlapping, and 4-connected only to itself.
#    - Rows are forced distinct within each type so descending order is unambiguous to a human.
# e) Visual observability:
#    - No invisible metadata is used. Ordering comes only from visible input row positions.
#    - Unique colors make the color swap traceable across columns.
#    - Solids visibly reverse to bottom-to-top input order on the right, while frames visibly sit against
#      the bottom edge on the left because solid count is always larger.
#    - Because swapping is done on paired ordered objects rather than on already aligned output rows,
#      the examples show a real cross-column mapping rather than a trivial recolor.
# f) Visual cleanliness:
#    - Objects stay sparse and isolated in the input.
#    - Output stays as two clean 4-column strips with no overlapping annotations or multi-layer marks.
#    - Restricting to a small count range avoids clutter while still giving repeated evidence.
# g) Color inference:
#    - Colors are not remapped by numeric identity. Each object's single non-bg color is swapped with
#      its ordered partner's single non-bg color, so the pattern is inferable from position + type.
#    - Keeping all object colors unique prevents invisible "swap with same color" degeneracy.
# 9) New assumptions introduced by the modified main():
#    - Every valid 4x4 object has exactly one non-bg color, so recoloring can safely replace all
#      non-bg cells with the partner color. The generator guarantees this by emitting only true frames
#      and true solids.
#    - Cross-column swapping pairs objects by sorted index, not by final row alignment. The generator
#      guarantees enough paired objects (`min(frames, solids) >= 2`) so this rule is visible.
#    - The left-column gravity cue is easiest to observe when there are fewer frames than solids;
#      the generator guarantees that asymmetry.


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    patch = np.full((4, 4), int(col), dtype=int)
    patch[1:3, 1:3] = int(bg)
    return patch


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    return len(set(map(int, border.flatten()))) == 1 and bool((sub[1:3, 1:3] == int(bg)).all())


def _extract_objects(grid: np.ndarray) -> tuple[int, list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = map(int, b)
        sub = crop_bbox(g, b)
        if _is_frame(sub, bg):
            frames.append((r0, c0, sub))
        else:
            solids.append((r0, c0, sub))

    return bg, frames, solids


def _sort_group(group: list[tuple[int, int, np.ndarray]]) -> list[tuple[int, int, np.ndarray]]:
    return sorted(group, key=lambda t: (-int(t[0]), int(t[1])))


def _primary_color(sub: np.ndarray, bg: int) -> int:
    cols = set(map(int, sub.flatten()))
    cols.discard(int(bg))
    if len(cols) != 1:
        raise ValueError("expected exactly one non-background color")
    return int(next(iter(cols)))


def _recolor_patch(sub: np.ndarray, bg: int, new_color: int) -> np.ndarray:
    out = np.array(sub, copy=True)
    out[out != int(bg)] = int(new_color)
    return out


def _build_output(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
    *,
    descending: bool,
    frame_from_bottom: bool,
    swap_colors: bool,
) -> np.ndarray:
    ordered_frames = _sort_group(frames) if descending else sorted(frames, key=lambda t: (int(t[0]), int(t[1])))
    ordered_solids = _sort_group(solids) if descending else sorted(solids, key=lambda t: (int(t[0]), int(t[1])))

    frame_patches = [np.array(sub, copy=True) for _r0, _c0, sub in ordered_frames]
    solid_patches = [np.array(sub, copy=True) for _r0, _c0, sub in ordered_solids]

    if swap_colors:
        pair_count = min(len(frame_patches), len(solid_patches))
        for i in range(pair_count):
            frame_color = _primary_color(frame_patches[i], bg)
            solid_color = _primary_color(solid_patches[i], bg)
            frame_patches[i] = _recolor_patch(frame_patches[i], bg, solid_color)
            solid_patches[i] = _recolor_patch(solid_patches[i], bg, frame_color)

    out_h = 4 * max(int(len(frame_patches)), int(len(solid_patches)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, sub in enumerate(solid_patches):
        paste(out, sub, int(i * 4), 4)

    for i, sub in enumerate(frame_patches):
        dest_r = int(out_h - 4 * (i + 1)) if frame_from_bottom else int(i * 4)
        paste(out, sub, dest_r, 0)

    return out


def _variant_metrics_from_parts(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
) -> dict[str, int]:
    combo = _build_output(bg, frames, solids, descending=True, frame_from_bottom=True, swap_colors=True)
    no_swap = _build_output(bg, frames, solids, descending=True, frame_from_bottom=True, swap_colors=False)
    seed = _build_output(bg, frames, solids, descending=False, frame_from_bottom=False, swap_colors=False)
    return {
        "n_frames": int(len(frames)),
        "n_solids": int(len(solids)),
        "pair_count": int(min(len(frames), len(solids))),
        "frame_padding_blocks": int(max(len(frames), len(solids)) - len(frames)),
        "pixel_diff_vs_no_swap": int(np.count_nonzero(combo != no_swap)),
        "pixel_diff_vs_seed": int(np.count_nonzero(combo != seed)),
    }


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    bg, frames, solids = _extract_objects(grid)
    return _variant_metrics_from_parts(bg, frames, solids)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule for 2ba387bc:
    - classify 4x4 objects into frames vs solids,
    - sort each type by row descending (row ties use col ascending),
    - swap the colors of the i-th ordered frame and i-th ordered solid,
    - stack frames bottom-up in cols 0..3 and solids top-down in cols 4..7.
    """
    bg, frames, solids = _extract_objects(np.array(grid, dtype=int))
    return _build_output(bg, frames, solids, descending=True, frame_from_bottom=True, swap_colors=True)


def _pick_counts() -> tuple[int, int]:
    configs = [
        (2, 3),
        (2, 4),
        (3, 4),
    ]
    idx = int(np.random.randint(0, len(configs)))
    return configs[idx]


def _place_with_new_row(
    grid: np.ndarray,
    patch: np.ndarray,
    *,
    bg: int,
    used_rows: set[int],
) -> tuple[int, int]:
    for _ in range(100):
        r0, c0 = random_free_location(
            grid,
            patch,
            transparent=bg,
            border=1,
            padding=1,
            padding_connectivity=4,
        )
        if int(r0) not in used_rows:
            return int(r0), int(c0)
    raise ValueError("failed to place object on a fresh row")


def _variant_is_observable(grid: np.ndarray) -> bool:
    bg, frames, solids = _extract_objects(grid)
    if len(frames) < 2 or len(solids) < 3:
        return False

    frame_rows = [int(r0) for r0, _c0, _sub in frames]
    solid_rows = [int(r0) for r0, _c0, _sub in solids]
    if len(set(frame_rows)) != len(frame_rows):
        return False
    if len(set(solid_rows)) != len(solid_rows):
        return False

    metrics = _variant_metrics_from_parts(bg, frames, solids)
    if metrics["pair_count"] < 2:
        return False
    if metrics["frame_padding_blocks"] <= 0:
        return False
    if metrics["pixel_diff_vs_no_swap"] <= 0:
        return False
    if metrics["pixel_diff_vs_seed"] <= 0:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(22, 31))
    w = int(np.random.randint(22, 31))
    grid = np.full((h, w), bg, dtype=int)
    n_frames, n_solids = _pick_counts()

    try:
        colors = list(map(int, np.random.permutation(np.arange(1, 10))))
        if len(colors) < n_frames + n_solids:
            return None

        frame_rows: set[int] = set()
        solid_rows: set[int] = set()

        for i in range(n_frames):
            patch = _make_frame(colors[i], bg)
            r0, c0 = _place_with_new_row(grid, patch, bg=bg, used_rows=frame_rows)
            paste(grid, patch, r0, c0, transparent=bg)
            frame_rows.add(int(r0))

        for i in range(n_solids):
            patch = _make_solid(colors[n_frames + i])
            r0, c0 = _place_with_new_row(grid, patch, bg=bg, used_rows=solid_rows)
            paste(grid, patch, r0, c0, transparent=bg)
            solid_rows.add(int(r0))

        parsed_bg, frames, solids = _extract_objects(grid)
        if len(frames) != n_frames or len(solids) != n_solids:
            raise ValueError("component extraction mismatch")
        if not _variant_is_observable(grid):
            raise ValueError("variant observability check failed")

        expected = _build_output(parsed_bg, frames, solids, descending=True, frame_from_bottom=True, swap_colors=True)
        if not np.array_equal(main(grid), expected):
            raise ValueError("round-trip check failed")
        return grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
