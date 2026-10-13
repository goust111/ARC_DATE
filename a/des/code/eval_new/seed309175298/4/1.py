from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

FRAME_COLOR = GRAY
END_COLOR = MAROON
SEED_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE)

SIDES = ("top", "bottom", "left", "right")
GAP_WIDTHS = (2, 3)
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
FRAME_COUNT_RANGE = (1, 3)
INTERIOR_H_RANGE = (3, 7)
INTERIOR_W_RANGE = (3, 7)
PLACEMENT_RETRIES = 200
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _sorted_components(mask: np.ndarray) -> list[np.ndarray]:
    comps = _components(mask)
    comps.sort(key=lambda coords: tuple(coords.min(axis=0).tolist()))
    return comps


def _expand_box(box: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _analyze_frame(grid: np.ndarray, coords: np.ndarray) -> dict[str, object] | None:
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    frame_mask = sub == FRAME_COLOR
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if np.any(frame_mask & ~border):
        return None

    missing = np.argwhere(border & ~frame_mask)
    if len(missing) not in GAP_WIDTHS:
        return None

    side: str
    gap_indices: list[int]
    if np.all(missing[:, 0] == 0):
        side = "top"
        gap_indices = sorted(int(v) for v in missing[:, 1])
    elif np.all(missing[:, 0] == h - 1):
        side = "bottom"
        gap_indices = sorted(int(v) for v in missing[:, 1])
    elif np.all(missing[:, 1] == 0):
        side = "left"
        gap_indices = sorted(int(v) for v in missing[:, 0])
    elif np.all(missing[:, 1] == w - 1):
        side = "right"
        gap_indices = sorted(int(v) for v in missing[:, 0])
    else:
        return None

    if gap_indices[-1] - gap_indices[0] + 1 != len(gap_indices):
        return None
    if gap_indices[0] == 0 or gap_indices[-1] == (w - 1 if side in ("top", "bottom") else h - 1):
        return None
    if int(frame_mask.sum()) != int(border.sum()) - len(gap_indices):
        return None

    inner = sub[1:-1, 1:-1]
    seed_locs = np.argwhere((inner != BLACK) & (inner != FRAME_COLOR))
    if not (1 <= len(seed_locs) <= 3):
        return None

    seeds: list[tuple[int, int, int]] = []
    colors: list[int] = []
    for ir, ic in seed_locs:
        color = int(inner[ir, ic])
        if color == END_COLOR:
            return None
        seeds.append((r0 + 1 + int(ir), c0 + 1 + int(ic), color))
        colors.append(color)
    if len(set(colors)) != len(colors):
        return None

    if side in ("top", "bottom"):
        line_coords = [c for _r, c, _color in seeds]
        depth = h - 2
        gap_center = c0 + (gap_indices[0] + gap_indices[-1]) / 2.0
        gap_span = (c0 + gap_indices[0], c0 + gap_indices[-1])
    else:
        line_coords = [r for r, _c, _color in seeds]
        depth = w - 2
        gap_center = r0 + (gap_indices[0] + gap_indices[-1]) / 2.0
        gap_span = (r0 + gap_indices[0], r0 + gap_indices[-1])
    if len(set(line_coords)) != len(line_coords):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "gap_span": gap_span,
        "gap_center": gap_center,
        "depth": depth,
        "seeds": sorted(seeds),
    }


def _growth_cells(frame: dict[str, object], seed: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
    side = frame["side"]  # type: ignore[assignment]
    depth = int(frame["depth"])
    gap_center = float(frame["gap_center"])
    sr, sc, color = seed

    cells: list[tuple[int, int, int]] = []
    if side == "top":
        offset = sc - gap_center
        line_c = int(round(gap_center + offset))
        start_r = r0 - 1
        for step in range(depth):
            cells.append((start_r - step, line_c, color if step + 1 < depth else END_COLOR))
    elif side == "bottom":
        offset = sc - gap_center
        line_c = int(round(gap_center + offset))
        start_r = r1 + 1
        for step in range(depth):
            cells.append((start_r + step, line_c, color if step + 1 < depth else END_COLOR))
    elif side == "left":
        offset = sr - gap_center
        line_r = int(round(gap_center + offset))
        start_c = c0 - 1
        for step in range(depth):
            cells.append((line_r, start_c - step, color if step + 1 < depth else END_COLOR))
    else:
        assert side == "right"
        offset = sr - gap_center
        line_r = int(round(gap_center + offset))
        start_c = c1 + 1
        for step in range(depth):
            cells.append((line_r, start_c + step, color if step + 1 < depth else END_COLOR))
    return cells


def _paint_growth(out: np.ndarray, original: np.ndarray, frame: dict[str, object]) -> None:
    h, w = out.shape
    for seed in frame["seeds"]:  # type: ignore[assignment]
        for rr, cc, value in _growth_cells(frame, seed):
            assert 0 <= rr < h and 0 <= cc < w, "growth line must stay inside the grid"
            assert int(original[rr, cc]) == BLACK, "growth area must start on empty background"
            assert int(out[rr, cc]) == BLACK, "growth regions must not overlap"
            out[rr, cc] = int(value)


def _draw_frame(inp: np.ndarray, frame: dict[str, object]) -> None:
    r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
    inp[r0 : r1 + 1, c0] = FRAME_COLOR
    inp[r0 : r1 + 1, c1] = FRAME_COLOR
    inp[r0, c0 : c1 + 1] = FRAME_COLOR
    inp[r1, c0 : c1 + 1] = FRAME_COLOR

    gap_lo, gap_hi = frame["gap_span"]  # type: ignore[misc]
    side = frame["side"]  # type: ignore[assignment]
    if side == "top":
        inp[r0, gap_lo : gap_hi + 1] = BLACK
    elif side == "bottom":
        inp[r1, gap_lo : gap_hi + 1] = BLACK
    elif side == "left":
        inp[gap_lo : gap_hi + 1, c0] = BLACK
    else:
        inp[gap_lo : gap_hi + 1, c1] = BLACK

    for sr, sc, color in frame["seeds"]:  # type: ignore[assignment]
        inp[sr, sc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored exterior branches from each gapped gray frame using the seed pixels inside the frame.

    Input structure:
    - The background is black.
    - One or more gray rectangular one-cell-thick frames are scattered on the grid, and each frame has exactly one contiguous gap of width 2 or 3 on a single side.
    - Each frame contains 1 to 3 single-pixel colored seeds in its interior, while the exterior of every frame is otherwise black.

    Rule:
    - Keep every frame and every interior seed unchanged.
    - For each frame, detect which side contains the gap; that side determines the outward growth direction, and the frame interior depth along that direction determines the branch length.
    - For each seed, preserve its offset along the axis parallel to the gap and draw a straight branch outside the frame in the gap's outward direction with the seed's color; recolor the last pixel of that branch to maroon.

    Key insight:
    - The gapped frame is a directional encoder: the missing side chooses the growth axis, the frame depth chooses the branch length, and each seed's parallel offset chooses where its exterior branch is placed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames: list[dict[str, object]] = []
    for coords in _sorted_components(g == FRAME_COLOR):
        frame = _analyze_frame(g, coords)
        assert frame is not None, "every gray component must be a valid gapped rectangular frame"
        frames.append(frame)
    assert frames, "expected at least one gapped frame"

    out = g.copy()
    for frame in sorted(frames, key=lambda f: f["bbox"]):  # type: ignore[index]
        _paint_growth(out, g, frame)
    return out


def _sample_frame(shape: tuple[int, int], occupied: np.ndarray) -> dict[str, object] | None:
    h, w = shape
    side = random.choice(SIDES)
    gap_width = random.choice(GAP_WIDTHS)
    seed_count = random.randint(1, 3)

    for _ in range(PLACEMENT_RETRIES):
        inner_h = random.randint(*INTERIOR_H_RANGE)
        inner_w = random.randint(*INTERIOR_W_RANGE)
        if side in ("top", "bottom"):
            if inner_w < max(seed_count, gap_width):
                continue
            total_h = inner_h + (inner_h + 2)
            total_w = inner_w + 2
            if total_h > h or total_w > w:
                continue
            total_r0 = random.randint(0, h - total_h)
            total_c0 = random.randint(0, w - total_w)
            frame_r0 = total_r0 + inner_h if side == "top" else total_r0
            frame_c0 = total_c0
        else:
            if inner_h < max(seed_count, gap_width):
                continue
            total_h = inner_h + 2
            total_w = inner_w + (inner_w + 2)
            if total_h > h or total_w > w:
                continue
            total_r0 = random.randint(0, h - total_h)
            total_c0 = random.randint(0, w - total_w)
            frame_r0 = total_r0
            frame_c0 = total_c0 + inner_w if side == "left" else total_c0

        frame_r1 = frame_r0 + inner_h + 1
        frame_c1 = frame_c0 + inner_w + 1

        if side == "top":
            total_box = (frame_r0 - inner_h, frame_c0, frame_r1, frame_c1)
        elif side == "bottom":
            total_box = (frame_r0, frame_c0, frame_r1 + inner_h, frame_c1)
        elif side == "left":
            total_box = (frame_r0, frame_c0 - inner_w, frame_r1, frame_c1)
        else:
            total_box = (frame_r0, frame_c0, frame_r1, frame_c1 + inner_w)

        reserve = _expand_box(total_box, 1, shape)
        rr0, rc0, rr1, rc1 = reserve
        if occupied[rr0 : rr1 + 1, rc0 : rc1 + 1].any():
            continue

        if side in ("top", "bottom"):
            gap_start = random.randint(1, inner_w - gap_width + 1)
            gap_span = (frame_c0 + gap_start, frame_c0 + gap_start + gap_width - 1)
            seed_cols = random.sample(range(frame_c0 + 1, frame_c1), seed_count)
            colors = random.sample(SEED_COLORS, seed_count)
            seeds = [
                (random.randint(frame_r0 + 1, frame_r1 - 1), col, color)
                for col, color in zip(seed_cols, colors)
            ]
            depth = inner_h
            gap_center = (gap_span[0] + gap_span[1]) / 2.0
        else:
            gap_start = random.randint(1, inner_h - gap_width + 1)
            gap_span = (frame_r0 + gap_start, frame_r0 + gap_start + gap_width - 1)
            seed_rows = random.sample(range(frame_r0 + 1, frame_r1), seed_count)
            colors = random.sample(SEED_COLORS, seed_count)
            seeds = [
                (row, random.randint(frame_c0 + 1, frame_c1 - 1), color)
                for row, color in zip(seed_rows, colors)
            ]
            depth = inner_w
            gap_center = (gap_span[0] + gap_span[1]) / 2.0

        frame = {
            "bbox": (frame_r0, frame_c0, frame_r1, frame_c1),
            "side": side,
            "gap_span": gap_span,
            "gap_center": gap_center,
            "depth": depth,
            "seeds": sorted(seeds),
        }
        occupied[rr0 : rr1 + 1, rc0 : rc1 + 1] = True
        return frame
    return None


def _render_case(shape: tuple[int, int], frames: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BLACK, dtype=int)
    for frame in frames:
        _draw_frame(inp, frame)
    out = inp.copy()
    for frame in sorted(frames, key=lambda f: f["bbox"]):  # type: ignore[index]
        _paint_growth(out, inp, frame)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_H_RANGE),
        random.randint(*GRID_W_RANGE),
    )
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    occupied = np.zeros(shape, dtype=bool)
    frames: list[dict[str, object]] = []
    for _ in range(frame_count):
        frame = _sample_frame(shape, occupied)
        if frame is None:
            return None
        frames.append(frame)

    inp, expected = _render_case(shape, frames)
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
