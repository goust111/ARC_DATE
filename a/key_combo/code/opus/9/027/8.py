from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GRAY = 5

GEN_RETRY_LIMIT = 400
FRAME_SIZES = (10, 12)
CONTROL_PATTERNS = [
    (BLUE, RED, BLUE, RED),
    (RED, BLUE, RED, BLUE),
    (BLUE, RED, RED, BLUE),
    (RED, BLUE, BLUE, RED),
    (BLUE, BLUE, RED, BLUE),
    (RED, RED, BLUE, RED),
    (BLUE, RED, BLUE, BLUE),
    (RED, BLUE, RED, RED),
]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _majority_color(values: np.ndarray) -> int:
    flat = [int(v) for v in np.asarray(values).ravel()]
    counts = Counter(flat)
    ranked = counts.most_common()
    assert ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1])
    return int(ranked[0][0])


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    box = _bbox(grid == GRAY)
    assert box is not None, "expected one gray frame"
    r0, c0, r1, c1 = box
    frame = _crop(grid, box)
    assert frame.shape[0] == frame.shape[1], "frame must be square"
    assert frame.shape[0] in FRAME_SIZES, "unexpected frame size"
    assert frame.shape[0] % 2 == 0, "frame size must be even"
    assert np.all(frame[0, :] == GRAY)
    assert np.all(frame[-1, :] == GRAY)
    assert np.all(frame[:, 0] == GRAY)
    assert np.all(frame[:, -1] == GRAY)
    assert not np.any(frame[1:-1, 1:-1] == GRAY)
    return box


def _find_green_box(grid: np.ndarray, frame_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = frame_box
    inside = np.zeros_like(grid, dtype=bool)
    inside[r0 + 1 : r1, c0 + 1 : c1] = True
    green_box = _bbox((grid == GREEN) & inside)
    assert green_box is not None, "expected one 2x2 green block"
    gr0, gc0, gr1, gc1 = green_box
    assert (gr1 - gr0 + 1, gc1 - gc0 + 1) == (2, 2)
    assert np.all(grid[gr0 : gr1 + 1, gc0 : gc1 + 1] == GREEN)
    return green_box


def _frame_side_colors(grid: np.ndarray, frame_box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_box
    left_vals = grid[r0 + 2 : r1 - 1, c0 + 1]
    right_vals = grid[r0 + 2 : r1 - 1, c1 - 1]
    left_color = _majority_color(left_vals)
    right_color = _majority_color(right_vals)
    assert left_color != right_color
    return left_color, right_color


def _find_source_motif(
    grid: np.ndarray, frame_box: tuple[int, int, int, int], colors: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = frame_box
    outside = np.ones_like(grid, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    mask = outside & np.isin(grid, np.array(colors, dtype=int))
    box = _bbox(mask)
    assert box is not None, "expected one external motif"
    motif = _crop(grid, box)
    assert set(map(int, np.unique(motif))) <= {BLACK, colors[0], colors[1]}
    return motif, box


def _orient_motif(source: np.ndarray, left_color: int, right_color: int) -> np.ndarray:
    left_majority = _majority_color(source[:, 0])
    right_majority = _majority_color(source[:, -1])
    if (left_majority, right_majority) == (left_color, right_color):
        return source.copy()
    if (left_majority, right_majority) == (right_color, left_color):
        return np.fliplr(source)
    raise AssertionError("source motif edge colors do not match the frame in either orientation")


def _fill_interior(
    out: np.ndarray,
    frame_box: tuple[int, int, int, int],
    motif: np.ndarray,
    green_box: tuple[int, int, int, int],
    left_color: int,
    right_color: int,
) -> None:
    r0, c0, r1, c1 = frame_box
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    assert motif.shape == (inner_h, inner_w - 2)

    # Keep the visible side labels while the pattern fills the center strip.
    out[r0 + 2 : r1 - 1, c0 + 1] = left_color
    out[r0 + 2 : r1 - 1, c1 - 1] = right_color
    out[r0 + 1 : r1, c0 + 2 : c1 - 1] = motif

    # The four corner-adjacent control pixels are copied from the oriented motif corners.
    out[r0 + 1, c0 + 1] = int(motif[0, 0])
    out[r0 + 1, c1 - 1] = int(motif[0, -1])
    out[r1 - 1, c0 + 1] = int(motif[-1, 0])
    out[r1 - 1, c1 - 1] = int(motif[-1, -1])

    gr0, gc0, gr1, gc1 = green_box
    out[gr0 : gr1 + 1, gc0 : gc1 + 1] = GREEN


def _paint_rays(out: np.ndarray, frame_box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_box
    controls = [
        ((r0, c0), (r0 + 1, c0 + 1), (-1, -1)),
        ((r0, c1), (r0 + 1, c1 - 1), (-1, 1)),
        ((r1, c0), (r1 - 1, c0 + 1), (1, -1)),
        ((r1, c1), (r1 - 1, c1 - 1), (1, 1)),
    ]
    for (fr, fc), (cr, cc), (dr, dc) in controls:
        if int(out[cr, cc]) != BLUE:
            continue
        rr = fr + dr
        cc2 = fc + dc
        painted = 0
        while _in_bounds(out.shape, rr, cc2):
            if int(out[rr, cc2]) == BLACK:
                out[rr, cc2] = GREEN
            painted += 1
            rr += dr
            cc2 += dc
        assert painted > 0


def _transform(grid: np.ndarray, force_flip: bool | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frame_box = _find_frame(g)
    green_box = _find_green_box(g, frame_box)
    left_color, right_color = _frame_side_colors(g, frame_box)
    motif, _motif_box = _find_source_motif(g, frame_box, (left_color, right_color))

    if force_flip is None:
        oriented = _orient_motif(motif, left_color, right_color)
    else:
        oriented = np.fliplr(motif) if force_flip else motif.copy()

    out = g.copy()
    _fill_interior(out, frame_box, oriented, green_box, left_color, right_color)
    _paint_rays(out, frame_box)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the external two-color pattern into the frame, then let its corner colors gate diagonal green rays.

    Input structure:
    - A black background contains one even-sized gray hollow square frame, a central 2x2 green solid block, and one external blue/red motif to the right.
    - Just inside the frame, the left inner wall is marked mostly blue and the right inner wall mostly red, leaving a center strip for the motif.
    - The four diagonal cells just inside the frame corners are not fixed markers; after insertion they become the per-corner ray-control pixels.

    Rule:
    - Compare the external motif's left and right edge-majority colors with the frame's side markers; if they are reversed, mirror the motif horizontally.
    - Copy the oriented motif into the frame's center strip, keep the side markers, and copy the motif's four corner colors into the four corner-adjacent control pixels; the 2x2 green block stays on top.
    - For each frame corner, if its inward diagonal control pixel is blue, extend a green diagonal ray from that frame corner outward to the grid edge, painting only black cells beyond the frame.

    Key insight:
    - The mirror step is a remote switch: it changes which motif corner colors land beside the frame corners, and those corner-adjacent colors decide which diagonal rays are released.
    """
    return _transform(grid)


def _random_rect_patch(h: int, w: int) -> tuple[int, int, int, int]:
    r0 = random.randint(0, h - 1)
    r1 = random.randint(r0, h - 1)
    c0 = random.randint(0, w - 1)
    c1 = random.randint(c0, w - 1)
    return r0, c0, r1, c1


def _build_canonical_motif(inner_h: int, width: int, controls: tuple[int, int, int, int]) -> np.ndarray:
    motif = np.full((inner_h, width), RED, dtype=int)
    split = random.randint(max(1, width // 2 - 1), min(width - 1, width // 2 + 1))
    motif[:, :split] = BLUE

    for _ in range(random.randint(2, 5)):
        r0, c0, r1, c1 = _random_rect_patch(inner_h, width)
        if c0 == 0 or c1 == width - 1:
            continue
        motif[r0 : r1 + 1, c0 : c1 + 1] = random.choice((BLUE, RED))

    motif[1:-1, 0] = BLUE
    motif[1:-1, -1] = RED
    motif[0, 0], motif[0, -1], motif[-1, 0], motif[-1, -1] = controls

    # Keep the side-majority signal unique while leaving the corners free.
    assert _majority_color(motif[:, 0]) == BLUE
    assert _majority_color(motif[:, -1]) == RED
    assert not np.array_equal(motif, np.fliplr(motif))
    return motif


def _generate_case() -> np.ndarray:
    frame_size = random.choice(FRAME_SIZES)
    inner_h = frame_size - 2
    motif_w = frame_size - 4

    top_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)
    left_margin = random.randint(1, 4)
    gap = random.randint(1, 3)
    right_margin = random.randint(1, 3)

    height = top_margin + frame_size + bottom_margin
    width = left_margin + frame_size + gap + motif_w + right_margin
    if max(height, width) > 30:
        raise AssertionError("grid exceeds ARC size limit")

    grid = np.zeros((height, width), dtype=int)
    r0 = top_margin
    c0 = left_margin
    r1 = r0 + frame_size - 1
    c1 = c0 + frame_size - 1

    grid[r0, c0 : c1 + 1] = GRAY
    grid[r1, c0 : c1 + 1] = GRAY
    grid[r0 : r1 + 1, c0] = GRAY
    grid[r0 : r1 + 1, c1] = GRAY

    grid[r0 + 2 : r1 - 1, c0 + 1] = BLUE
    grid[r0 + 2 : r1 - 1, c1 - 1] = RED

    center = frame_size // 2
    gr0 = r0 + center - 1
    gc0 = c0 + center - 1
    grid[gr0 : gr0 + 2, gc0 : gc0 + 2] = GREEN

    controls = random.choice(CONTROL_PATTERNS)
    canonical = _build_canonical_motif(inner_h, motif_w, controls)
    needs_flip = random.random() < 0.5
    source = np.fliplr(canonical) if needs_flip else canonical

    motif_top = r0 + 1
    motif_left = c1 + gap + 1
    grid[motif_top : motif_top + inner_h, motif_left : motif_left + motif_w] = source

    correct = _transform(grid)
    wrong = _transform(grid, force_flip=not needs_flip)
    assert not np.array_equal(correct, wrong), "mirror step must matter"

    active = {
        "tl": int(correct[r0 + 1, c0 + 1]) == BLUE,
        "tr": int(correct[r0 + 1, c1 - 1]) == BLUE,
        "bl": int(correct[r1 - 1, c0 + 1]) == BLUE,
        "br": int(correct[r1 - 1, c1 - 1]) == BLUE,
    }
    assert 0 < sum(active.values()) < 4, "need both active and blocked rays"

    for (fr, fc), flag, (dr, dc) in [
        ((r0, c0), active["tl"], (-1, -1)),
        ((r0, c1), active["tr"], (-1, 1)),
        ((r1, c0), active["bl"], (1, -1)),
        ((r1, c1), active["br"], (1, 1)),
    ]:
        rr = fr + dr
        cc = fc + dc
        ray_cells = []
        while _in_bounds(grid.shape, rr, cc):
            ray_cells.append((rr, cc))
            rr += dr
            cc += dc
        assert ray_cells, "each ray branch must have visible space beyond the frame"
        greens = sum(int(correct[r, c] == GREEN) for r, c in ray_cells)
        if flag:
            assert greens == len(ray_cells)
        else:
            assert greens == 0

    inner = correct[r0 + 1 : r1, c0 + 1 : c1]
    assert np.count_nonzero(inner == BLUE) > 0 and np.count_nonzero(inner == RED) > 0
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        try:
            return _generate_case()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid mirror-ray ARC input")
