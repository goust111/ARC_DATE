# Combo variant of 2ba387bc (index 34)
# combo_spec: D14/0+D16/0
# Name: 反向堆叠+框架旋转

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace:
#    Original task:
#    - detect scattered 4x4 objects
#    - group by type: frame left, solid right
#    - sort each group by input row
#    - stack both groups from top to bottom
#    Fused task:
#    - frame-like objects are still sent to the left column, but each one is rotated
#      90 degrees CCW before placement
#    - solid 4x4 blocks still go to the right column, but are stacked from bottom to top
#    Example with two frames F1,F2 and two solids S1,S2, ordered by input row:
#      original output rows: left=[F1,F2], right=[S1,S2]
#      fused output rows:    left=[rot(F1),rot(F2)], right=[S2 on top, S1 on bottom]
#    Difference:
#    - left column changes object orientation
#    - right column reverses the vertical placement direction without changing the sort rule
#
# b) Complexity calibration:
#    A plain 4x4 hollow square is rotationally symmetric, so "rotate frame by 90 degrees"
#    would be visually invisible and the variant would collapse back toward the base task.
#    To keep D16/0 observable, the generator uses frame-like objects with:
#    - a uniform 1px border
#    - a 2x2 interior containing exactly one same-colored marker cell
#    They remain clearly distinct from solids, while the marker makes rotation visible.
#    The marker only has four corner positions inside the 2x2 interior, so the freedom is
#    small and repeated enough for humans to infer.
#
# c) Implicit assumptions review:
#    - original code assumes every relevant component has bbox 4x4: preserved
#    - original frame test assumes interior is all background: relaxed to "mostly hollow"
#      with 0 or 1 interior marker matching the border color
#    - original output assumes both columns fill top-down: broken by D14/0, so right-column
#      row calculation must be inverted
#    - original code pastes the original frame unchanged: broken by D16/0, so frames must
#      be rotated before pasting
#
# d) Placement legality:
#    The generator still uses random_free_location with border=1 and padding=1, so objects
#    do not overlap, do not touch, and stay within bounds.
#
# e) Visual observability:
#    - reverse stacking is visible because each sample contains at least two solids
#    - rotation is visible because each frame has an interior marker and each sample contains
#      at least two frames
#    - colors are unique per object, so object identity can be tracked visually
#
# f) Visual neatness:
#    Objects are still isolated scattered 4x4 shapes on a blank background. There are no
#    extra guide marks or overlays, so the input remains clean.
#
# g) Inferability:
#    No hidden numeric ordering is used. Sorting depends only on visible row position in the
#    input; the changed behavior is visible in output geometry alone.


def _frame_marker_positions() -> list[tuple[int, int]]:
    return [(1, 1), (1, 2), (2, 2), (2, 1)]


def _make_frame(col: int, bg: int, marker_idx: int) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    mr, mc = _frame_marker_positions()[int(marker_idx) % 4]
    a[mr, mc] = int(col)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _is_frame_like(sub: np.ndarray, bg: int) -> bool:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    border_colors = set(map(int, border.flatten()))
    if len(border_colors) != 1:
        return False

    border_col = int(next(iter(border_colors)))
    if border_col == int(bg):
        return False

    interior = sub[1:3, 1:3]
    nz = np.argwhere(interior != int(bg))
    if int(nz.shape[0]) == 0:
        return True
    if int(nz.shape[0]) != 1:
        return False

    rr, cc = map(int, nz[0])
    return int(interior[rr, cc]) == int(border_col)


def _is_solid(sub: np.ndarray, bg: int) -> bool:
    cols = set(map(int, sub.flatten()))
    return len(cols) == 1 and int(bg) not in cols


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
        if _is_frame_like(sub, bg):
            frames.append((int(r0), int(c0), sub.copy()))
        elif _is_solid(sub, bg):
            solids.append((int(r0), int(c0), sub.copy()))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return int(bg), frames, solids


def _build_output(bg: int, frames: list[tuple[int, int, np.ndarray]], solids: list[tuple[int, int, np.ndarray]]) -> np.ndarray:
    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, np.rot90(sub, k=1), int(i * 4), 0)

    for i, (_r0, _c0, sub) in enumerate(solids):
        dst_r = int(out_h - 4 * (i + 1))
        paste(out, sub, dst_r, 4)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - classify 4x4 objects into frame-like vs solid
    - sort each class by input row, then column
    - rotate each frame 90 degrees CCW and stack it top-down in the left column
    - stack solids bottom-up in the right column without rotation
    """
    bg, frames, solids = _extract_objects(grid)
    return _build_output(bg, frames, solids)


def _original_style_output(bg: int, frames: list[tuple[int, int, np.ndarray]], solids: list[tuple[int, int, np.ndarray]]) -> np.ndarray:
    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(bg), dtype=int)
    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)
    return out


def _distinct_rows(entries: list[tuple[int, int, np.ndarray]]) -> bool:
    rows = [int(r0) for r0, _c0, _sub in entries]
    return len(rows) == len(set(rows))


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(22, 31))
    w = int(np.random.randint(22, 31))
    g = np.full((h, w), bg, dtype=int)

    n_frames = int(np.random.randint(2, 5))
    n_solids = int(np.random.randint(2, 5))
    total = int(n_frames + n_solids)
    palette = np.random.choice(np.arange(1, 10), size=total, replace=False)

    try:
        color_idx = 0
        for _ in range(n_frames):
            col = int(palette[color_idx])
            color_idx += 1
            marker_idx = int(np.random.randint(0, 4))
            frame = _make_frame(col, bg, marker_idx)
            r0, c0 = random_free_location(
                g,
                frame,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, frame, r0, c0, transparent=bg)

        for _ in range(n_solids):
            col = int(palette[color_idx])
            color_idx += 1
            solid = _make_solid(col)
            r0, c0 = random_free_location(
                g,
                solid,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, solid, r0, c0, transparent=bg)

        parsed_bg, frames, solids = _extract_objects(g)
        if parsed_bg != int(bg):
            return None
        if len(frames) != int(n_frames) or len(solids) != int(n_solids):
            return None
        if not _distinct_rows(frames) or not _distinct_rows(solids):
            return None

        fused = _build_output(bg, frames, solids)
        base = _original_style_output(bg, frames, solids)
        if np.array_equal(fused, base):
            return None

        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
