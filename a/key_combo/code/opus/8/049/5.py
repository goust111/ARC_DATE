import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
STRIPE_COLORS = (1, 2, 3, 4, 6, 7)
GEN_ATTEMPTS = 400


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _find_separator(grid: np.ndarray) -> int:
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(sep_cols) == 1, "expected exactly one full-height grey separator column"
    sep = sep_cols[0]
    assert 1 <= sep < grid.shape[1] - 1, "separator must split the grid into two non-empty zones"
    return sep


def _compress_stripes(left: np.ndarray) -> tuple[list[int], list[int]]:
    seq: list[int] = []
    thicknesses: list[int] = []
    prev = None
    run = 0
    for r in range(left.shape[0]):
        color = int(left[r, 0])
        assert color in STRIPE_COLORS, "left zone must use instruction stripe colors"
        assert np.all(left[r, :] == color), "each left-zone row must be uniform"
        if color != prev:
            if prev is not None:
                thicknesses.append(run)
            seq.append(color)
            prev = color
            run = 1
        else:
            run += 1
    thicknesses.append(run)
    return seq, thicknesses


def _extract_frames(right: np.ndarray) -> list[tuple[int, int, int, int]]:
    assert np.all(np.isin(right, (BLACK, MAROON))), "right zone must contain only black background and maroon frames"
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components4(right == MAROON):
        r0, c0, r1, c1 = _bbox(comp)
        sub = right[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape
        assert h >= 3 and w >= 3, "frames must have a non-empty strict interior"
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.array_equal(sub == MAROON, border), "every maroon component must be a one-cell-thick hollow rectangle"
        frames.append((r0, c0, r1, c1))
    assert frames, "expected at least one frame"
    return sorted(frames, key=lambda box: (box[0], box[1]))


def _expected_output_from_parts(colors: list[int], frames: list[tuple[int, int, int, int]], right_shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(right_shape, dtype=int)
    for color, (r0, c0, r1, c1) in zip(colors, frames):
        out[r0 : r1 + 1, c0] = MAROON
        out[r0 : r1 + 1, c1] = MAROON
        out[r0, c0 : c1 + 1] = MAROON
        out[r1, c0 : c1 + 1] = MAROON
        out[r0 + 1 : r1, c0 + 1 : c1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the left instruction stripes and use them to fill the sorted hollow frames on the right.

    Input structure:
    - A full-height grey separator column splits the grid into a left stripe zone and a right frame zone.
    - The left zone is made of full-width horizontal color stripes; consecutive rows of the same color may repeat to create varying stripe thickness.
    - The right zone has a black background with several non-touching one-cell-thick maroon hollow rectangles whose interiors are black.

    Rule:
    - Scan the left zone from top to bottom and collapse each maximal run of equal-colored rows into one color instruction.
    - Detect every maroon hollow rectangle in the right zone and sort them by the row, then column, of the rectangle's top-left corner.
    - Fill the strict interior of the k-th sorted rectangle with the k-th compressed stripe color.
    - Return only the transformed right zone, keeping frame borders maroon and everything outside the frames black.

    Key insight:
    - Stripe thickness is redundant control noise, so the usable program is the compressed color order rather than the raw row count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    colors, _ = _compress_stripes(left)
    frames = _extract_frames(right)
    assert len(colors) == len(frames), "compressed stripe count must match frame count"

    out = right.copy()
    for color, (r0, c0, r1, c1) in zip(colors, frames):
        out[r0 + 1 : r1, c0 + 1 : c1] = color
    return out


def _sample_stripe_colors(count: int) -> list[int]:
    colors: list[int] = []
    for idx in range(count):
        choices = [color for color in STRIPE_COLORS if color != (colors[-1] if colors else None)]
        if idx >= 2 and random.random() < 0.4 and colors[-2] in choices:
            color = colors[-2]
        else:
            color = random.choice(choices)
        colors.append(color)
    assert len(set(colors)) >= 2
    return colors


def _sample_thicknesses(count: int) -> list[int]:
    for _ in range(200):
        vals = [random.randint(1, 4) for _ in range(count)]
        total = sum(vals)
        if 10 <= total <= 14 and max(vals) > 1 and len(set(vals)) > 1:
            return vals
    raise RuntimeError("failed to sample valid stripe thicknesses")


def _compose_nonnegative(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _sample_shelves(height: int, frame_count: int) -> list[tuple[int, int, int]]:
    for _ in range(200):
        shelf_count = 2 if frame_count == 3 else random.choice((2, 2, 3))
        if shelf_count > frame_count:
            continue
        shelf_heights = [random.randint(3, 5) for _ in range(shelf_count)]
        required = sum(shelf_heights) + (shelf_count - 1)
        if required > height:
            continue
        extra = height - required
        pads = _compose_nonnegative(extra, shelf_count + 1)
        counts = [1] * shelf_count
        for _ in range(frame_count - shelf_count):
            counts[random.randrange(shelf_count)] += 1
        if max(counts) < 2:
            continue
        shelves: list[tuple[int, int, int]] = []
        top = pads[0]
        for idx, shelf_h in enumerate(shelf_heights):
            shelves.append((top, shelf_h, counts[idx]))
            top += shelf_h + 1 + pads[idx + 1]
        return shelves
    raise RuntimeError("failed to sample shelf layout")


def _sample_widths(count: int, right_width: int) -> list[int] | None:
    for _ in range(200):
        widths = [random.randint(3, 6) for _ in range(count)]
        if sum(widths) + (count - 1) <= right_width:
            return widths
    return None


def _spread_positions(widths: list[int], total_width: int, bias: str) -> list[int]:
    required = sum(widths) + (len(widths) - 1)
    extra = total_width - required
    inner_slots = len(widths) - 1
    if inner_slots:
        inner = _compose_nonnegative(extra, inner_slots + 2)
        left_pad = inner[0]
        mid_extras = inner[1:-1]
        right_pad = inner[-1]
    else:
        left_pad = extra
        mid_extras = []
        right_pad = 0

    if bias == "right" and extra > 0:
        shift = random.randint(extra // 2, extra)
        left_pad = shift
        remain = extra - shift
        spread = _compose_nonnegative(remain, inner_slots + 1)
        mid_extras = spread[:-1]
        right_pad = spread[-1]
    elif bias == "left" and extra > 0:
        shift = random.randint(0, extra // 2)
        left_pad = shift
        remain = extra - shift
        spread = _compose_nonnegative(remain, inner_slots + 1)
        mid_extras = spread[:-1]
        right_pad = spread[-1]

    starts: list[int] = []
    pos = left_pad
    for idx, width in enumerate(widths):
        starts.append(pos)
        pos += width
        if idx < len(widths) - 1:
            pos += 1 + mid_extras[idx]
    assert pos + right_pad == total_width
    return starts


def _sample_frames(height: int, right_width: int, frame_count: int) -> list[tuple[int, int, int, int]] | None:
    shelves = _sample_shelves(height, frame_count)
    frames: list[tuple[int, int, int, int]] = []
    for shelf_idx, (top, shelf_h, count) in enumerate(shelves):
        widths = _sample_widths(count, right_width)
        if widths is None:
            return None
        bias = "right" if shelf_idx == 0 else "left" if shelf_idx == 1 else "mixed"
        starts = _spread_positions(widths, right_width, bias)
        shared_top = random.random() < 0.85
        for idx in range(count):
            frame_h = random.randint(3, shelf_h)
            frame_w = widths[idx]
            r0 = top if shared_top else top + random.randint(0, shelf_h - frame_h)
            c0 = starts[idx]
            frames.append((r0, c0, r0 + frame_h - 1, c0 + frame_w - 1))
    frames = sorted(frames, key=lambda box: (box[0], box[1]))
    col_major = sorted(frames, key=lambda box: (box[1], box[0]))
    if frames == col_major:
        return None
    if len({box[0] for box in frames}) == len(frames):
        return None
    return frames


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], col_offset: int) -> None:
    r0, c0, r1, c1 = box
    c0 += col_offset
    c1 += col_offset
    grid[r0 : r1 + 1, c0] = MAROON
    grid[r0 : r1 + 1, c1] = MAROON
    grid[r0, c0 : c1 + 1] = MAROON
    grid[r1, c0 : c1 + 1] = MAROON


def _validate_generated(grid: np.ndarray) -> None:
    sep = _find_separator(grid)
    left = grid[:, :sep]
    right = grid[:, sep + 1 :]
    colors, thicknesses = _compress_stripes(left)
    frames = _extract_frames(right)
    assert 3 <= len(colors) <= 5
    assert len(colors) == len(frames)
    assert len(set(colors)) >= 2
    assert max(thicknesses) > 1
    assert len(set(thicknesses)) > 1
    assert frames != sorted(frames, key=lambda box: (box[1], box[0]))

    out = main(grid)
    expected = _expected_output_from_parts(colors, frames, right.shape)
    assert np.array_equal(out, expected), "solver output must match the semantic construction"

    for color, (r0, c0, r1, c1) in zip(colors, frames):
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0
        assert np.all(interior == color)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        frame_count = random.randint(3, 5)
        thicknesses = _sample_thicknesses(frame_count)
        colors = _sample_stripe_colors(frame_count)
        height = sum(thicknesses)
        left_width = random.randint(4, 7)
        right_min = max(10, frame_count * 3 + 1)
        right_max = min(18, 30 - left_width - 1)
        if right_min > right_max:
            continue
        right_width = random.randint(right_min, right_max)
        frames = _sample_frames(height, right_width, frame_count)
        if frames is None:
            continue

        grid = np.zeros((height, left_width + 1 + right_width), dtype=int)
        row = 0
        for color, thickness in zip(colors, thicknesses):
            grid[row : row + thickness, :left_width] = color
            row += thickness
        grid[:, left_width] = GREY
        for box in frames:
            _draw_frame(grid, box, left_width + 1)

        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
