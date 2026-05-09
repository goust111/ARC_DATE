# Variant of 2ba387bc
# Dimension: D5 — 目标布局——两列排列方式 (dst)
# Change: 框架放入左列（cols 0-3），实心放入右列（cols 4-7），两列紧邻无间隙... → 重力堆叠——框架从底部向上堆叠，实心从顶部向下堆叠

from __future__ import annotations

import numpy as np
from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

# Design analysis for this D5 variant
# a) Concrete behavior:
#    Input stays identical to the seed task: scattered 4x4 objects on a uniform background.
#    Frames are 1-pixel hollow squares with a bg-colored 2x2 center; solids are fully filled 4x4 blocks.
#    Classification and per-type sorting key stay the same: sort by (top row, then left col).
#    The only rule change is the destination stack direction:
#    - solids still occupy cols 4..7 and are pasted from row 0 downward;
#    - frames still occupy cols 0..3 but are pasted from the bottom slot upward.
#    Example with two frames F1,F2 and three solids S1,S2,S3 after row-sorting:
#    original output rows were [F1,F2, . ] on the left and [S1,S2,S3] on the right;
#    variant output rows become [ . ,F2,F1] on the left and [S1,S2,S3] on the right.
#    So the left column now shows gravity toward the bottom edge while the right column keeps top gravity.
# b) Complexity calibration:
#    No new object types or hidden metadata are introduced. The observable novelty comes from
#    opposite stack directions, which is visible when there are >=2 frames (order reversal)
#    and especially clear when frame/solid counts differ (left-side top padding vs right-side bottom padding).
#    The generator therefore enforces at least two frames and biases counts toward unequal pairs.
# c) Old assumptions in main():
#    - 4x4 bbox filter is still valid because object geometry is unchanged.
#    - frame detection via "single border color + bg 2x2 interior" is still valid.
#    - sorting by input row still applies.
#    - the old paste formula `i*4` for frames is no longer valid because it assumes top-down stacking.
#      The new frame destination must be `outH - 4*(i+1)`.
#    - solids keep the old top-down destination formula.
# d) Placement legality:
#    Generator placement still uses 1-cell orthogonal padding through `random_free_location`, so
#    components remain disjoint under 4-connectivity, stay in-bounds, and preserve 4x4 bboxes.
#    The generator also retries until same-type objects have distinct input rows, making the
#    row-based ordering legible and preventing ambiguous "same row" examples from dominating.
# e) Visual observability:
#    Sorting never depends on color ids; colors only identify objects visually.
#    Unique object colors ensure that reversing the frame stack is visible. At least two frames
#    guarantee the new output is not identical to the legacy top-down layout.
# f) Visual cleanliness:
#    Only one object occupies each 4x4 footprint, with background gaps between objects. The output
#    remains two clean 4-column strips with padding only at the exposed gravity end.
# g) New implicit assumptions:
#    `main()` now assumes only that there is at least one 4x4 object and that frames/solids are
#    recognized by the original geometry rule. `generate_input()` guarantees both.


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


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

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return bg, frames, solids


def _build_output(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
    *,
    frame_from_bottom: bool,
) -> np.ndarray:
    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)

    for i, (_r0, _c0, sub) in enumerate(frames):
        dest_r = int(out_h - 4 * (i + 1)) if frame_from_bottom else int(i * 4)
        paste(out, sub, dest_r, 0)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 objects by type, then stack them into two 4-column outputs.

    Variant rule:
    - Frames (hollow squares) go to cols 0..3 and stack from the bottom upward.
    - Solids go to cols 4..7 and stack from the top downward.
    - Within each type, the order source is still the input top-to-bottom order.
    """
    bg, frames, solids = _extract_objects(grid)
    return _build_output(bg, frames, solids, frame_from_bottom=True)


def _legacy_output(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    return _build_output(bg, frames, solids, frame_from_bottom=False)


def _pick_counts() -> tuple[int, int]:
    configs = [
        (2, 3),
        (2, 4),
        (3, 2),
        (3, 4),
        (4, 2),
        (4, 3),
        (2, 2),
        (3, 3),
    ]
    idx = int(np.random.randint(0, len(configs)))
    return configs[idx]


def _place_with_distinct_row(
    grid: np.ndarray,
    patch: np.ndarray,
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
    raise ValueError("failed to find a distinct row for object placement")


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(20, 31))
    w = int(np.random.randint(20, 31))
    g = np.full((h, w), bg, dtype=int)
    n_frames, n_solids = _pick_counts()

    try:
        colors = list(map(int, np.random.permutation(np.arange(1, 10))))
        if len(colors) < n_frames + n_solids:
            return None

        frame_rows: set[int] = set()
        solid_rows: set[int] = set()

        for i in range(n_frames):
            patch = _make_frame(colors[i], bg)
            r0, c0 = _place_with_distinct_row(g, patch, bg, frame_rows)
            paste(g, patch, r0, c0, transparent=bg)
            frame_rows.add(int(r0))

        for i in range(n_solids):
            patch = _make_solid(colors[n_frames + i])
            r0, c0 = _place_with_distinct_row(g, patch, bg, solid_rows)
            paste(g, patch, r0, c0, transparent=bg)
            solid_rows.add(int(r0))

        parsed_bg, frames, solids = _extract_objects(g)
        if len(frames) != n_frames or len(solids) != n_solids:
            raise ValueError("component extraction mismatch")

        expected = _build_output(parsed_bg, frames, solids, frame_from_bottom=True)
        legacy = _legacy_output(parsed_bg, frames, solids)
        actual = main(g)

        if np.array_equal(expected, legacy):
            raise ValueError("variant degenerated to legacy layout")
        if not np.array_equal(actual, expected):
            raise ValueError("round-trip check failed")
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
