# Combo variant of 2ba387bc (index 8)
# combo_spec: D2/1+D5/3
# Name: 降序排列+重力堆叠

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

# Design analysis for this combo variant
# a) Concrete behavior and trace:
#    - Input still consists of isolated 4x4 objects on a uniform background.
#    - Frames are hollow 4x4 squares with a bg-filled 2x2 center; solids are fully filled 4x4 blocks.
#    - Fusion rule:
#      1. Within each type-group, sort by input top-row in descending order (bottom-to-top), with col
#         ascending only as a tie-break.
#      2. Frames still go to cols 0..3 but now stack from the output bottom upward.
#      3. Solids still go to cols 4..7 and stack from the output top downward.
#    - Small trace with frame rows [2, 11] and solid rows [4, 10, 16]:
#      original seed output:
#        left  = rows 2 -> 11 from top,
#        right = rows 4 -> 10 -> 16 from top.
#      combo output after descending sort:
#        frame order list becomes 11 -> 2, but bottom-stacking pastes 11 into the bottom slot and 2 above it,
#        so the visible top-to-bottom order on the left becomes 2 -> 11 again, with empty bg rows above if
#        frames are fewer than solids.
#        solid order list becomes 16 -> 10 -> 4, and top-stacking preserves that visible order on the right.
#      Difference from seed:
#        solids visibly reverse, while frames reveal the new gravity mainly through top padding rather than
#        through a visible order reversal. This is the key interaction of the two dimensions.
# b) Complexity calibration:
#    - The combo is easy to under-specify because frame descending order and bottom gravity partially cancel.
#    - To keep both dimensions learnable, the generator forces:
#      1. at least two solids with distinct rows, so right-column descending order is visible;
#      2. strictly more solids than frames, so left-column bottom gravity creates visible top padding;
#      3. unique colors across all objects, so identity tracking is visual rather than numeric.
#    - This avoids degenerate samples that look like the seed or like a single-dimension variant.
# c) Implicit assumption audit of the original main():
#    - `find_components(..., monochromatic=False)` assumes each object is isolated by background.
#      The generator preserves a 1-cell gap, so this still holds.
#    - `bbox_h == bbox_w == 4` assumes every relevant object occupies a 4x4 bbox. Still true.
#    - Frame detection via "single border color + bg 2x2 interior" still holds unchanged.
#    - The original `sort(key=(row, col))` no longer matches the combo; it must become row descending.
#    - The original frame placement `dest_r = i * 4` assumes top-down stacking and is invalid here;
#      the combo needs `dest_r = out_h - 4 * (i + 1)` for frames.
#    - Solid placement still uses top-down `i * 4`.
# d) Placement legality:
#    - Generator placement uses `random_free_location(..., border=1, padding=1, padding_connectivity=4)`,
#      so objects stay in bounds, do not overlap, and remain disconnected under 4-connectivity.
#    - Post-placement parsing is re-run to verify that the intended object counts survive segmentation.
# e) Visual observability:
#    - No hidden factor such as color id or generation order is used for sorting; only visible row position.
#    - The generator rejects any input whose combo output equals the seed output, the D2-only output,
#      or the D5-only output on the same input.
#    - This ensures both the descending-order control and the gravity control materially affect examples.
# f) Visual cleanliness:
#    - Objects remain sparse, non-touching, and all share the same simple 4x4 vocabulary.
#    - Output remains two clean 4-column stacks with bg padding only where gravity exposes it.
# g) Color inference:
#    - Colors are identity markers only. There is no color remapping rule to infer.
# 9) New implicit assumptions introduced by the modified main():
#    - If same-type objects share the same top row, descending order falls back to column ascending.
#      The generator avoids same-type row ties so the descending rule is unambiguous.
#    - The combo relies on there being more solids than frames for left-column gravity to be visible.
#      `generate_input()` explicitly enforces `len(solids) > len(frames)`.


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


def _sort_group(group: list[tuple[int, int, np.ndarray]], *, descending: bool) -> list[tuple[int, int, np.ndarray]]:
    if descending:
        return sorted(group, key=lambda t: (-int(t[0]), int(t[1])))
    return sorted(group, key=lambda t: (int(t[0]), int(t[1])))


def _build_output(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
    *,
    descending: bool,
    frame_from_bottom: bool,
) -> np.ndarray:
    ordered_frames = _sort_group(frames, descending=descending)
    ordered_solids = _sort_group(solids, descending=descending)

    out_h = 4 * max(int(len(ordered_frames)), int(len(ordered_solids)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(ordered_solids):
        paste(out, sub, int(i * 4), 4)

    for i, (_r0, _c0, sub) in enumerate(ordered_frames):
        dest_r = int(out_h - 4 * (i + 1)) if frame_from_bottom else int(i * 4)
        paste(out, sub, dest_r, 0)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule for 2ba387bc:
    - classify 4x4 objects as frame vs solid;
    - sort each type by input row descending (bottom-to-top), col ascending on ties;
    - paste frames into cols 0..3 from the output bottom upward;
    - paste solids into cols 4..7 from the output top downward.
    """
    bg, frames, solids = _extract_objects(np.array(grid, dtype=int))
    return _build_output(bg, frames, solids, descending=True, frame_from_bottom=True)


def _seed_output(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    return _build_output(bg, frames, solids, descending=False, frame_from_bottom=False)


def _d2_only_output(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    return _build_output(bg, frames, solids, descending=True, frame_from_bottom=False)


def _d5_only_output(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    return _build_output(bg, frames, solids, descending=False, frame_from_bottom=True)


def _pick_counts() -> tuple[int, int]:
    # Keep `solids > frames` so left-column bottom gravity stays visible via top padding.
    configs = [
        (2, 3),
        (2, 4),
        (3, 4),
    ]
    idx = int(np.random.randint(0, len(configs)))
    return configs[idx]


def _place_with_distinct_row(
    grid: np.ndarray,
    patch: np.ndarray,
    *,
    bg: int,
    used_rows: set[int],
) -> tuple[int, int]:
    for _ in range(80):
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
    raise ValueError("failed to find a placement with a fresh row")


def _combo_is_observable(grid: np.ndarray) -> bool:
    bg, frames, solids = _extract_objects(grid)
    if len(frames) < 2 or len(solids) < 2:
        return False
    if len(solids) <= len(frames):
        return False

    frame_rows = [int(r0) for r0, _c0, _sub in frames]
    solid_rows = [int(r0) for r0, _c0, _sub in solids]
    if len(set(frame_rows)) != len(frame_rows):
        return False
    if len(set(solid_rows)) != len(solid_rows):
        return False

    combo = _build_output(bg, frames, solids, descending=True, frame_from_bottom=True)
    seed = _seed_output(bg, frames, solids)
    d2_only = _d2_only_output(bg, frames, solids)
    d5_only = _d5_only_output(bg, frames, solids)

    return (
        not np.array_equal(combo, seed)
        and not np.array_equal(combo, d2_only)
        and not np.array_equal(combo, d5_only)
    )


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(20, 31))
    w = int(np.random.randint(20, 31))
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
            r0, c0 = _place_with_distinct_row(grid, patch, bg=bg, used_rows=frame_rows)
            paste(grid, patch, r0, c0, transparent=bg)
            frame_rows.add(int(r0))

        for i in range(n_solids):
            patch = _make_solid(colors[n_frames + i])
            r0, c0 = _place_with_distinct_row(grid, patch, bg=bg, used_rows=solid_rows)
            paste(grid, patch, r0, c0, transparent=bg)
            solid_rows.add(int(r0))

        parsed_bg, frames, solids = _extract_objects(grid)
        if len(frames) != n_frames or len(solids) != n_solids:
            raise ValueError("component extraction mismatch")
        if not _combo_is_observable(grid):
            raise ValueError("combo not visually observable")

        expected = _build_output(parsed_bg, frames, solids, descending=True, frame_from_bottom=True)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            raise ValueError("round-trip check failed")
        return grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
