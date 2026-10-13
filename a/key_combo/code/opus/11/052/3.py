from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SEP = 5
FRAME_COLORS = (1, 2, 3, 4, 6)
RIGHT_PANEL_WIDTH = 6
SPRITE_SIZE = 6
SPRITE_STEP = 4
MAX_ATTEMPTS = 400

# A 16-pixel hollow 5x5 rectangle embedded in a 6x6 sprite box.
RING_CELLS = tuple(
    [(0, c) for c in range(5)]
    + [(4, c) for c in range(5)]
    + [(r, 0) for r in range(1, 4)]
    + [(r, 4) for r in range(1, 4)]
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _checker_count(ih: int, iw: int) -> int:
    return (int(ih) * int(iw) + 1) // 2


def _frame_patch(color: int, ih: int, iw: int) -> np.ndarray:
    patch = np.full((ih + 2, iw + 2), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    for r in range(ih):
        for c in range(iw):
            if (r + c) % 2 == 0:
                patch[r + 1, c + 1] = color
    return patch


def _analyze_input(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    sep_cols = [c for c in range(w) if np.all(g[:, c] == SEP)]
    assert len(sep_cols) == 1, "expected exactly one full-height grey separator column"
    sep = int(sep_cols[0])
    assert 0 < sep < w - 1, "separator must split the grid into two non-empty panels"

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    right_w = int(right.shape[1])
    assert right_w >= SPRITE_SIZE, "right panel must be at least 6 columns wide"
    assert np.all(right == BG), "right panel must start empty"

    colors = sorted(set(map(int, left.ravel())) - {BG})
    assert 3 <= len(colors) <= 5, "expected 3-5 colored frames on the left panel"

    frames: list[dict[str, int]] = []
    for color in colors:
        mask = left == color
        r0, c0, r1, c1 = _bbox(mask)
        patch = left[r0 : r1 + 1, c0 : c1 + 1]
        oh, ow = patch.shape
        assert oh >= 3 and ow >= 3, "frame bbox must include a non-empty interior"
        assert np.all(patch[0, :] == color)
        assert np.all(patch[-1, :] == color)
        assert np.all(patch[:, 0] == color)
        assert np.all(patch[:, -1] == color)

        ih, iw = oh - 2, ow - 2
        fill_count = 0
        for ir in range(ih):
            for ic in range(iw):
                expected = color if (ir + ic) % 2 == 0 else BG
                cell = int(patch[ir + 1, ic + 1])
                assert cell == expected, "frame interior must be an upper-left-starting checkerboard"
                if cell == color:
                    fill_count += 1

        frames.append(
            {
                "color": int(color),
                "count": int(fill_count),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "ih": int(ih),
                "iw": int(iw),
                "outer_area": int(oh * ow),
                "border_pixels": int(2 * oh + 2 * ow - 4),
                "component_pixels": int(np.sum(mask)),
            }
        )

    counts = [frame["count"] for frame in frames]
    assert len(set(counts)) == len(counts), "checker counts must be unique"
    return {"grid": g, "left_width": int(sep), "right_width": right_w, "frames": frames}


def _render_output(frames_sorted_asc: list[dict[str, int]], right_w: int) -> np.ndarray:
    out_h = SPRITE_SIZE + SPRITE_STEP * (len(frames_sorted_asc) - 1)
    out = np.full((out_h, right_w), BG, dtype=int)

    # Spatial order is smallest-count at the top; drawing happens in reverse so
    # smaller counts stay visually on top inside the overlap rows.
    for i in range(len(frames_sorted_asc) - 1, -1, -1):
        frame = frames_sorted_asc[i]
        base_r = SPRITE_STEP * i
        for dr, dc in RING_CELLS:
            out[base_r + dr, dc] = int(frame["color"])
    return out


def _sorted_by_key(frames: list[dict[str, int]], key: str) -> list[dict[str, int]]:
    return sorted(frames, key=lambda frame: (int(frame[key]), int(frame["color"])))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the colored checker cells inside left-panel frames and use that order to stack right-panel ring sprites.

    Input structure:
    - A full-height grey column splits the grid into a left analysis panel and a right empty panel.
    - The left panel has 3-5 one-cell-thick hollow rectangular frames, each in a unique non-grey color.
    - Every frame interior is filled with that same color on an upper-left-starting checkerboard; the right panel is all black.

    Rule:
    - For each left-panel color, locate its frame and count only the colored checkerboard cells inside the border, excluding the border itself.
    - Sort the colors by that interior fill count from small to large.
    - Output only the right-panel width, and place one 6x6 ring sprite per sorted color from top to bottom with a vertical step of 4 rows.
    - Draw the sprites in reverse sorted order, so the smallest-count color is painted last and therefore stays on top in the overlap rows.

    Key insight:
    - The right panel contains no direct clue at all; the visible layer order comes entirely from a geometric count computed in the separate left panel.
    """
    info = _analyze_input(grid)
    frames = _sorted_by_key(info["frames"], "count")
    return _render_output(frames, int(info["right_width"]))


def _shape_pool() -> list[dict[str, int]]:
    pool: list[dict[str, int]] = []
    for ih in range(1, 6):
        for iw in range(2, 8):
            oh, ow = ih + 2, iw + 2
            count = _checker_count(ih, iw)
            pool.append(
                {
                    "ih": ih,
                    "iw": iw,
                    "oh": oh,
                    "ow": ow,
                    "count": count,
                    "outer_area": oh * ow,
                    "border_pixels": 2 * oh + 2 * ow - 4,
                    "component_pixels": (2 * oh + 2 * ow - 4) + count,
                }
            )
    return pool


def _has_inversion(shapes: list[dict[str, int]], key: str) -> bool:
    count_order = [shape["count"] for shape in sorted(shapes, key=lambda s: (s["count"], s["ih"], s["iw"]))]
    other_order = [shape["count"] for shape in sorted(shapes, key=lambda s: (s[key], s["count"], s["ih"], s["iw"]))]
    return count_order != other_order


def _choose_shapes(n_frames: int) -> list[dict[str, int]] | None:
    pool = _shape_pool()
    max_total_height = 20 if n_frames == 5 else 22
    for _ in range(200):
        shapes = random.sample(pool, n_frames)
        counts = [shape["count"] for shape in shapes]
        if len(set(counts)) != n_frames:
            continue
        if sum(shape["oh"] for shape in shapes) > max_total_height:
            continue
        if max(shape["ow"] for shape in shapes) > 9:
            continue
        if not (_has_inversion(shapes, "outer_area") or _has_inversion(shapes, "border_pixels") or _has_inversion(shapes, "component_pixels")):
            continue
        return [dict(shape) for shape in shapes]
    return None


def _place_frames(left_h: int, left_w: int, frames: list[dict[str, int]]) -> bool:
    occupied = np.zeros((left_h, left_w), dtype=bool)
    for frame in sorted(frames, key=lambda item: item["oh"] * item["ow"], reverse=True):
        oh, ow = int(frame["oh"]), int(frame["ow"])
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, left_h - oh):
            for c0 in range(1, left_w - ow):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(left_h, r0 + oh + 1)
                cc1 = min(left_w, c0 + ow + 1)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        frame["r0"] = int(r0)
        frame["c0"] = int(c0)
        occupied[max(0, r0 - 1) : min(left_h, r0 + oh + 1), max(0, c0 - 1) : min(left_w, c0 + ow + 1)] = True
    return True


def _overlap_pixel_count(n_frames: int) -> int:
    count = 0
    for i in range(n_frames - 1):
        top = {(SPRITE_STEP * i + dr, dc) for dr, dc in RING_CELLS}
        below = {(SPRITE_STEP * (i + 1) + dr, dc) for dr, dc in RING_CELLS}
        count += len(top & below)
    return count


def _generate_attempt() -> np.ndarray | None:
    n_frames = random.choices([3, 4, 5], weights=[3, 3, 2], k=1)[0]
    shapes = _choose_shapes(n_frames)
    if shapes is None:
        return None

    colors = random.sample(FRAME_COLORS, n_frames)
    for frame, color in zip(shapes, colors):
        frame["color"] = int(color)

    max_ow = max(frame["ow"] for frame in shapes)
    left_w_min = max(11, max_ow + 2)
    left_w_max = min(13, max_ow + 4)
    if left_w_min > left_w_max:
        return None
    left_w = random.randint(left_w_min, left_w_max)

    output_h = SPRITE_SIZE + SPRITE_STEP * (n_frames - 1)
    packed_h = sum(frame["oh"] for frame in shapes) + n_frames
    left_h = max(output_h, packed_h) + random.randint(0, 3)
    if left_h > 28:
        return None

    if not _place_frames(left_h, left_w, shapes):
        return None

    grid = np.full((left_h, left_w + 1 + RIGHT_PANEL_WIDTH), BG, dtype=int)
    grid[:, left_w] = SEP
    for frame in shapes:
        patch = _frame_patch(int(frame["color"]), int(frame["ih"]), int(frame["iw"]))
        r0, c0 = int(frame["r0"]), int(frame["c0"])
        oh, ow = patch.shape
        grid[r0 : r0 + oh, c0 : c0 + ow] = patch

    info = _analyze_input(grid)
    frames = list(info["frames"])
    true_sorted = _sorted_by_key(frames, "count")
    true_out = _render_output(true_sorted, RIGHT_PANEL_WIDTH)

    if _overlap_pixel_count(n_frames) == 0:
        return None
    if not np.array_equal(main(grid), true_out):
        return None

    # Reject degenerate sets where simpler geometric keys would accidentally
    # produce the same vertical/layer order and therefore the same output.
    for alt_key in ("outer_area", "border_pixels", "component_pixels"):
        alt_sorted = _sorted_by_key(frames, alt_key)
        alt_out = _render_output(alt_sorted, RIGHT_PANEL_WIDTH)
        if alt_sorted != true_sorted and np.array_equal(alt_out, true_out):
            return None

    # The output must depend on the count order, not just the color set.
    reversed_out = _render_output(list(reversed(true_sorted)), RIGHT_PANEL_WIDTH)
    if np.array_equal(reversed_out, true_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-3 input")
