from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_shared_arc_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_shared_arc_utils", module)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
bbox = _ARC.bbox
bbox_h = _ARC.bbox_h
bbox_w = _ARC.bbox_w
crop_bbox = _ARC.crop_bbox
find_components = _ARC.find_components
get_bg = _ARC.get_bg


BG = Color.BLACK
FRAME = Color.GREY
CONTROL = Color.ORANGE
PALETTE_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.TEAL, Color.MAROON]
GEN_RETRIES = 400


def _parse_input(grid: np.ndarray) -> tuple[list[int], tuple[int, int, int, int], tuple[int, int], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    bg = get_bg(g)
    assert bg == BG, "expected black background"

    frame_components = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[FRAME])
    assert len(frame_components) == 1, "expected exactly one gray frame"
    frame_bbox = frame_components[0]["bbox"]
    r0, c0, r1, c1 = frame_bbox
    assert bbox_h(frame_bbox) >= 4 and bbox_w(frame_bbox) >= 4, "frame must have interior"

    frame_crop = crop_bbox(g, frame_bbox)
    assert np.all(frame_crop[0, :] == FRAME)
    assert np.all(frame_crop[-1, :] == FRAME)
    assert np.all(frame_crop[:, 0] == FRAME)
    assert np.all(frame_crop[:, -1] == FRAME)
    assert not np.any(frame_crop[1:-1, 1:-1] == FRAME), "frame must be one-pixel thick"

    control_coords = np.argwhere(g == CONTROL)
    assert control_coords.shape[0] == 1, "expected exactly one orange control pixel"
    ctrl_r, ctrl_c = map(int, control_coords[0])
    assert r0 < ctrl_r < r1 and c0 < ctrl_c < c1, "control pixel must lie inside the frame"

    palette_cols: list[tuple[int, int]] = []
    for c in range(c0):
        col = g[:, c]
        if int(col[0]) != bg and np.all(col == col[0]):
            palette_cols.append((c, int(col[0])))
    assert 3 <= len(palette_cols) <= 5, "expected 3-5 palette columns"

    cols = [c for c, _ in palette_cols]
    assert all(cols[i + 1] - cols[i] == 2 for i in range(len(cols) - 1)), "palette columns must be separated by one gap column"
    for c in range(cols[0], cols[-1] + 1):
        if c not in cols:
            assert np.all(g[:, c] == bg), "palette gap columns must be black"

    palette = [color for _, color in palette_cols]
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    assert all(color not in {BG, FRAME, CONTROL} for color in palette)
    return palette, frame_bbox, (ctrl_r, ctrl_c), (cols[0], cols[-1])


def _paint_rings(
    grid: np.ndarray,
    palette: list[int],
    frame_bbox: tuple[int, int, int, int],
    control: tuple[int, int],
    palette_span: tuple[int, int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = frame_bbox
    ctrl_r, ctrl_c = control
    period = len(palette)

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            dist = max(abs(r - ctrl_r), abs(c - ctrl_c))
            out[r, c] = int(palette[dist % period])

    pc0, pc1 = palette_span
    out[:, pc0 : pc1 + 1] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the left palette columns and use them to color Chebyshev-distance rings inside the right frame, then remove the palette area.

    Input structure:
    - The background is black.
    - On the left, 3-5 full-height single-color palette columns are separated by one black column each.
    - On the right, a one-pixel-thick gray rectangular frame contains exactly one orange control pixel on an otherwise black interior.

    Rule:
    - Read the palette colors from left to right.
    - Inside the gray frame, compute each interior cell's Chebyshev distance from the orange control pixel.
    - Recolor every interior cell, including the control pixel itself, with the palette color at index `distance mod len(palette)`.
    - Keep the gray frame unchanged and set the entire palette strip, including its gap columns, to black.

    Key insight:
    - The frame supplies the region, the orange pixel supplies the ring center, and the ordered palette columns supply the cyclic ring-color sequence.
    """
    palette, frame_bbox, control, palette_span = _parse_input(np.asarray(grid, dtype=int))
    return _paint_rings(np.asarray(grid, dtype=int), palette, frame_bbox, control, palette_span)


def _all_control_candidates(h: int, w: int, palette_len: int) -> list[tuple[int, int]]:
    need = 2 * palette_len - 1
    out = []
    for r in range(h):
        for c in range(w):
            max_dist = max(r, h - 1 - r, c, w - 1 - c)
            if max_dist >= need:
                out.append((r, c))
    return out


def _sample_control(h: int, w: int, palette_len: int) -> tuple[int, int]:
    candidates = _all_control_candidates(h, w, palette_len)
    assert candidates, "interior must admit a control position with two visible cycles"

    cornerish = [(r, c) for r, c in candidates if (r in {0, 1, h - 2, h - 1}) and (c in {0, 1, w - 2, w - 1})]
    edgeish = [
        (r, c)
        for r, c in candidates
        if (r in {0, h - 1} or c in {0, w - 1} or r in {1, h - 2} or c in {1, w - 2})
        and (r, c) not in cornerish
    ]
    central = [(r, c) for r, c in candidates if (r, c) not in cornerish and (r, c) not in edgeish]

    labeled_pools = []
    if cornerish:
        labeled_pools.append((cornerish, 0.4))
    if edgeish:
        labeled_pools.append((edgeish, 0.35))
    if central:
        labeled_pools.append((central, 0.25))

    chosen_pool = random.choices(
        [pool for pool, _ in labeled_pools],
        weights=[weight for _, weight in labeled_pools],
        k=1,
    )[0]
    return random.choice(chosen_pool)


def _sample_dimensions(palette_len: int, frame_c0_min: int) -> tuple[int, int, int, int, int, int]:
    for _ in range(200):
        interior_h = random.randint(2 * palette_len, min(12, 2 * palette_len + 4))
        interior_w = random.randint(2 * palette_len, min(14, 2 * palette_len + 6))
        outer_h = interior_h + 2
        outer_w = interior_w + 2

        top_margin = random.randint(1, 4)
        bottom_margin = random.randint(1, 4)
        right_margin = random.randint(1, 4)
        H = top_margin + outer_h + bottom_margin
        W_min = frame_c0_min + outer_w + right_margin
        if H > 30 or W_min > 30:
            continue

        W = random.randint(W_min, min(30, W_min + 4))
        r0 = random.randint(top_margin - 1, H - outer_h - bottom_margin)
        c0 = random.randint(frame_c0_min, W - outer_w - right_margin)
        return H, W, interior_h, interior_w, r0, c0
    raise RuntimeError("failed to sample valid grid dimensions")


def _expected_from_spec(grid: np.ndarray) -> np.ndarray:
    palette, frame_bbox, control, palette_span = _parse_input(grid)
    return _paint_rings(grid, palette, frame_bbox, control, palette_span)


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        palette, frame_bbox, control, palette_span = _parse_input(inp)
        out = main(inp)
    except Exception:
        return False

    expected = _paint_rings(inp, palette, frame_bbox, control, palette_span)
    if not np.array_equal(out, expected):
        return False

    r0, c0, r1, c1 = frame_bbox
    ctrl_r, ctrl_c = control
    period = len(palette)
    seen_distances = set()
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            d = max(abs(r - ctrl_r), abs(c - ctrl_c))
            seen_distances.add(d)
            if int(out[r, c]) != int(palette[d % period]):
                return False
    if not all(d in seen_distances for d in range(2 * period)):
        return False

    if not np.array_equal(out[r0, c0 : c1 + 1], inp[r0, c0 : c1 + 1]):
        return False
    if not np.array_equal(out[r1, c0 : c1 + 1], inp[r1, c0 : c1 + 1]):
        return False
    if not np.array_equal(out[r0 : r1 + 1, c0], inp[r0 : r1 + 1, c0]):
        return False
    if not np.array_equal(out[r0 : r1 + 1, c1], inp[r0 : r1 + 1, c1]):
        return False

    pc0, pc1 = palette_span
    if np.any(out[:, pc0 : pc1 + 1] != BG):
        return False

    reversed_out = _paint_rings(inp, list(reversed(palette)), frame_bbox, control, palette_span)
    if np.array_equal(out, reversed_out):
        return False

    if len(palette) > 3:
        shorter_out = _paint_rings(inp, palette[:-1], frame_bbox, control, palette_span)
        if np.array_equal(out, shorter_out):
            return False

    center_control = ((r0 + r1) // 2, (c0 + c1) // 2)
    if center_control != control:
        centered_out = _paint_rings(inp, palette, frame_bbox, center_control, palette_span)
        if np.array_equal(out, centered_out):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    palette_len = random.randint(3, 5)
    palette = random.sample(PALETTE_COLORS, palette_len)
    palette_start = random.randint(0, 1)
    palette_cols = [palette_start + 2 * i for i in range(palette_len)]
    frame_gap = random.randint(2, 4)
    frame_c0_min = palette_cols[-1] + frame_gap

    try:
        H, W, interior_h, interior_w, frame_r0, frame_c0 = _sample_dimensions(palette_len, frame_c0_min)
    except RuntimeError:
        return None

    grid = np.zeros((H, W), dtype=int)
    for c, color in zip(palette_cols, palette):
        grid[:, c] = int(color)

    frame_r1 = frame_r0 + interior_h + 1
    frame_c1 = frame_c0 + interior_w + 1
    grid[frame_r0, frame_c0 : frame_c1 + 1] = FRAME
    grid[frame_r1, frame_c0 : frame_c1 + 1] = FRAME
    grid[frame_r0 : frame_r1 + 1, frame_c0] = FRAME
    grid[frame_r0 : frame_r1 + 1, frame_c1] = FRAME

    local_r, local_c = _sample_control(interior_h, interior_w, palette_len)
    control_r = frame_r0 + 1 + local_r
    control_c = frame_c0 + 1 + local_c
    grid[control_r, control_c] = CONTROL

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
