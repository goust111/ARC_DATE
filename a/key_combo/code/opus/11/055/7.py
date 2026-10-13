import random

import numpy as np


BG = 0
MARKER_COLORS = (2, 3, 4, 7)
NON_MARKER_COLORS = (1, 5, 6, 8, 9)
GEN_RETRIES = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_complete_frame(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in np.unique(grid):
        color = int(color)
        mask = grid == color
        box = _bbox(mask)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        frame = np.zeros_like(mask, dtype=bool)
        frame[r0 : r1 + 1, c0] = True
        frame[r0 : r1 + 1, c1] = True
        frame[r0, c0 : c1 + 1] = True
        frame[r1, c0 : c1 + 1] = True
        if np.array_equal(mask, frame):
            candidates.append((color, box))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one complete frame, found {len(candidates)}")
    return candidates[0]


def _find_marker(grid: np.ndarray, frame_box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    r0, c0, r1, c1 = frame_box
    markers: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(np.isin(grid, MARKER_COLORS)):
        rr = int(r)
        cc = int(c)
        if r0 <= rr <= r1 and c0 <= cc <= c1:
            continue
        markers.append((rr, cc, int(grid[rr, cc])))
    if len(markers) != 1:
        raise ValueError(f"expected exactly one external marker, found {len(markers)}")
    return markers[0]


def _marker_axis(
    marker_r: int, marker_c: int, frame_box: tuple[int, int, int, int]
) -> str:
    r0, c0, r1, c1 = frame_box
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    dr = marker_r - center_r
    dc = marker_c - center_c
    if abs(dc) >= abs(dr):
        return "vertical"
    return "horizontal"


def _recolor_border(tile: np.ndarray, color: int) -> np.ndarray:
    out = tile.copy()
    out[_border_mask(tile.shape)] = int(color)
    return out


def _compose(tile: np.ndarray, axis: str) -> np.ndarray:
    if axis == "vertical":
        mirrored = np.fliplr(tile)
        return np.concatenate([tile, mirrored[:, 1:]], axis=1)
    if axis == "horizontal":
        mirrored = np.flipud(tile)
        return np.concatenate([tile, mirrored[1:, :]], axis=0)
    raise ValueError(f"unknown mirror axis: {axis}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the unique framed region, recolor its border by a marker, then mirror-join it.

    Input structure:
    - A black background contains scattered noise pixels, exactly one complete monochrome
      rectangular frame, and exactly one external marker pixel.
    - The frame color appears only on that one-pixel-thick perimeter and encloses a small
      multi-color non-symmetric interior pattern.
    - The marker color is one of red(2), green(3), yellow(4), or orange(7), and its
      position relative to the frame center is not on the horizontal or vertical midline.

    Rule:
    - Find the only color whose full set of pixels is exactly a complete rectangular
      perimeter, then crop that frame together with everything inside it.
    - Compare the marker position to the frame center: if the horizontal offset is at
      least as large as the vertical offset, use a vertical mirror axis and horizontal
      concatenation; otherwise use a horizontal mirror axis and vertical concatenation.
    - Recolor the border of the crop to the marker color, build the corresponding mirror
      copy, and join the two halves while merging the shared seam border into one row or
      one column.

    Key insight:
    - The frame isolates the only relevant subgrid, while the single external marker
      simultaneously encodes the new border color and which mirror-and-join orientation
      must be used.
    """
    g = np.asarray(grid, dtype=int)
    frame_color, frame_box = _find_complete_frame(g)
    del frame_color
    marker_r, marker_c, marker_color = _find_marker(g, frame_box)
    r0, c0, r1, c1 = frame_box
    tile = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    tile = _recolor_border(tile, marker_color)
    axis = _marker_axis(marker_r, marker_c, frame_box)
    return _compose(tile, axis)


def _candidate_marker_cells(
    shape: tuple[int, int], frame_box: tuple[int, int, int, int], mode: str
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = frame_box
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            dr = r - center_r
            dc = c - center_c
            if dr == 0 or dc == 0:
                continue
            if mode == "horizontal" and abs(dc) > abs(dr):
                cells.append((r, c))
            elif mode == "vertical" and abs(dr) > abs(dc):
                cells.append((r, c))
            elif mode == "tie" and abs(dc) == abs(dr):
                cells.append((r, c))
    return cells


def _make_interior_pattern(
    height: int, width: int, colors: tuple[int, ...]
) -> np.ndarray | None:
    area = height * width
    for _ in range(200):
        pat = np.zeros((height, width), dtype=int)
        for color in colors:
            r = random.randrange(height)
            c = random.randrange(width)
            steps = random.randint(2, max(2, min(area - 1, (height + width) // 2 + 1)))
            for _ in range(steps):
                pat[r, c] = int(color)
                candidates = [(r, c)]
                if r > 0:
                    candidates.append((r - 1, c))
                if r + 1 < height:
                    candidates.append((r + 1, c))
                if c > 0:
                    candidates.append((r, c - 1))
                if c + 1 < width:
                    candidates.append((r, c + 1))
                r, c = random.choice(candidates)
        extras = random.randint(1, max(1, area // 4))
        for _ in range(extras):
            r = random.randrange(height)
            c = random.randrange(width)
            pat[r, c] = int(random.choice(colors))
        used = tuple(sorted(int(v) for v in np.unique(pat) if v != BG))
        if len(used) < 2:
            continue
        if np.count_nonzero(pat) < len(colors) + 2:
            continue
        if np.array_equal(pat, np.fliplr(pat)):
            continue
        if np.array_equal(pat, np.flipud(pat)):
            continue
        for color in used:
            mask = pat == color
            box = _bbox(mask)
            if box is None:
                continue
            rr0, cc0, rr1, cc1 = box
            sub = mask[rr0 : rr1 + 1, cc0 : cc1 + 1]
            if sub.shape[0] >= 3 and sub.shape[1] >= 3:
                frame = _border_mask(sub.shape)
                if np.array_equal(sub, frame):
                    break
        else:
            return pat
    return None


def _paint_noise(
    grid: np.ndarray,
    forbidden: np.ndarray,
    color_choices: tuple[int, ...],
    count: int,
) -> bool:
    h, w = grid.shape
    blobs_done = 0
    for _ in range(200):
        if blobs_done >= count:
            return True
        free = np.argwhere((grid == BG) & (~forbidden))
        if free.size == 0:
            break
        sr, sc = map(int, random.choice(free))
        length = random.randint(1, 3)
        color = int(random.choice(color_choices))
        cells = [(sr, sc)]
        for _ in range(length - 1):
            next_steps: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BG and not forbidden[nr, nc]:
                        next_steps.append((nr, nc))
            if not next_steps:
                break
            cells.append(random.choice(next_steps))
        for r, c in cells:
            grid[r, c] = color
            rr0 = max(0, r - 1)
            rr1 = min(h, r + 2)
            cc0 = max(0, c - 1)
            cc1 = min(w, c + 2)
            forbidden[rr0:rr1, cc0:cc1] = True
        blobs_done += 1
    return blobs_done >= count


def _expanded_box_mask(
    shape: tuple[int, int], box: tuple[int, int, int, int], pad: int
) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = box
    mask = np.zeros((h, w), dtype=bool)
    rr0 = max(0, r0 - pad)
    rr1 = min(h, r1 + pad + 1)
    cc0 = max(0, c0 - pad)
    cc1 = min(w, c1 + pad + 1)
    mask[rr0:rr1, cc0:cc1] = True
    return mask


def _validate_generated_case(grid: np.ndarray) -> bool:
    try:
        frame_color, frame_box = _find_complete_frame(grid)
        marker_r, marker_c, marker_color = _find_marker(grid, frame_box)
        axis = _marker_axis(marker_r, marker_c, frame_box)
        r0, c0, r1, c1 = frame_box
        tile = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        interior = tile[1:-1, 1:-1]
        interior_colors = tuple(sorted(int(v) for v in np.unique(interior) if v != BG))
        if frame_color == marker_color or marker_color in interior_colors:
            return False
        if len(interior_colors) < 2 or len(interior_colors) > 3:
            return False
        if np.array_equal(tile, np.fliplr(tile)) or np.array_equal(tile, np.flipud(tile)):
            return False
        if np.count_nonzero(grid) - np.count_nonzero(tile) <= 1:
            return False
        expected = main(grid)
        if max(expected.shape) > 30:
            return False
        recolored = _recolor_border(tile, marker_color)
        other_axis = "horizontal" if axis == "vertical" else "vertical"
        other = _compose(recolored, other_axis)
        if expected.shape == other.shape and np.array_equal(expected, other):
            return False
        if axis == "vertical":
            doubled = np.concatenate([recolored, np.fliplr(recolored)], axis=1)
        else:
            doubled = np.concatenate([recolored, np.flipud(recolored)], axis=0)
        if expected.shape == doubled.shape and np.array_equal(expected, doubled):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid_h = random.randint(14, 20)
        grid_w = random.randint(14, 20)
        grid = np.zeros((grid_h, grid_w), dtype=int)

        frame_h = random.randint(5, 9)
        frame_w = random.randint(5, 9)
        if frame_h >= grid_h - 2 or frame_w >= grid_w - 2:
            continue
        r0 = random.randint(1, grid_h - frame_h - 1)
        c0 = random.randint(1, grid_w - frame_w - 1)
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1
        frame_box = (r0, c0, r1, c1)

        frame_color = int(random.choice(NON_MARKER_COLORS))
        marker_color = int(random.choice(MARKER_COLORS))
        interior_pool = tuple(color for color in NON_MARKER_COLORS if color != frame_color)
        color_count = random.choice((2, 3))
        interior_colors = tuple(random.sample(interior_pool, color_count))
        pattern = _make_interior_pattern(frame_h - 2, frame_w - 2, interior_colors)
        if pattern is None:
            continue

        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color
        grid[r0 + 1 : r1, c0 + 1 : c1] = pattern

        mode = random.choice(("horizontal", "vertical", "tie"))
        candidates = _candidate_marker_cells(grid.shape, frame_box, mode)
        if not candidates:
            continue
        marker_r, marker_c = random.choice(candidates)
        grid[marker_r, marker_c] = marker_color

        forbidden = _expanded_box_mask(grid.shape, frame_box, pad=1)
        forbidden[marker_r, marker_c] = True
        rr0 = max(0, marker_r - 1)
        rr1 = min(grid_h, marker_r + 2)
        cc0 = max(0, marker_c - 1)
        cc1 = min(grid_w, marker_c + 2)
        forbidden[rr0:rr1, cc0:cc1] = True
        noise_count = random.randint(3, 7)
        noise_palette = tuple(
            color for color in NON_MARKER_COLORS if color != frame_color
        )
        if not _paint_noise(grid, forbidden, noise_palette, noise_count):
            continue

        if _validate_generated_case(grid):
            return grid

    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
