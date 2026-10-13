from __future__ import annotations

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, paste
import random


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, int, int, int, int, int, int]:
    candidates: list[tuple[int, int, int, int, int, int, int]] = []
    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == int(color))
        if coords.shape[0] != 4:
            continue

        rows = sorted(set(map(int, coords[:, 0])))
        cols = sorted(set(map(int, coords[:, 1])))
        if len(rows) != 2 or len(cols) != 2:
            continue

        r0, r1 = rows
        c0, c1 = cols
        if r1 <= r0 + 1 or c1 <= c0 + 1:
            continue
        if set(map(tuple, coords)) != {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            continue

        left_colors = set(map(int, g[r0 + 1 : r1, c0]))
        right_colors = set(map(int, g[r0 + 1 : r1, c1]))
        if len(left_colors) != 1 or len(right_colors) != 1:
            continue

        left_color = next(iter(left_colors))
        right_color = next(iter(right_colors))
        if left_color == int(bg) or right_color == int(bg) or left_color == right_color:
            continue

        candidates.append((int(color), r0, c0, r1, c1, left_color, right_color))

    assert len(candidates) == 1, "expected a unique four-corner frame"
    return candidates[0]


def _edge_colors(sprite: np.ndarray, bg: int) -> tuple[int, int]:
    coords = np.argwhere(sprite != int(bg))
    assert coords.size, "expected a non-empty source motif"

    min_c = int(coords[:, 1].min())
    max_c = int(coords[:, 1].max())
    left_rows = coords[coords[:, 1] == min_c, 0]
    right_rows = coords[coords[:, 1] == max_c, 0]

    left_colors = set(map(int, sprite[left_rows, min_c]))
    right_colors = set(map(int, sprite[right_rows, max_c]))
    assert len(left_colors) == 1 and len(right_colors) == 1, "motif side colors must be unique"
    return next(iter(left_colors)), next(iter(right_colors))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the external two-color motif into the marked frame, mirroring it when needed so its side colors match the frame sides.

    Input structure:
    - The background contains four equal marker pixels at the corners of an empty
      rectangle.
    - The two vertical sides of that rectangle are uniform foreground bars with
      different colors.
    - A separate two-color motif has the same height and width as the rectangle's
      interior, and uses exactly the two side-bar colors.

    Rule:
    - Crop the marked rectangle to form the output canvas.
    - Insert the separate motif into the rectangle interior.
    - If the motif's left and right edge colors are reversed relative to the
      frame side colors, mirror the motif horizontally before insertion.

    Key insight:
    - The side bars are color labels for the motif orientation: the motif is
      placed in the only horizontal orientation whose visible side colors agree
      with the frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    marker, r0, c0, r1, c1, left_color, right_color = _find_frame(g, bg)

    side_mask = np.zeros(g.shape, dtype=bool)
    side_mask[r0 + 1 : r1, c0] = True
    side_mask[r0 + 1 : r1, c1] = True

    source_mask = (g != int(bg)) & (g != int(marker)) & ~side_mask
    source_bbox = bbox(source_mask)
    assert source_bbox is not None, "expected one source motif outside the frame"

    source = crop_bbox(g, source_bbox)
    interior_shape = (r1 - r0 - 1, c1 - c0 - 1)
    assert source.shape == interior_shape, "source motif must fit the frame interior"

    source_colors = set(map(int, np.unique(source[source != int(bg)])))
    assert source_colors == {left_color, right_color}, "source colors must match frame side colors"

    source_left, source_right = _edge_colors(source, bg)
    if (source_left, source_right) == (left_color, right_color):
        oriented = source
    elif (source_left, source_right) == (right_color, left_color):
        oriented = np.fliplr(source)
    else:
        raise AssertionError("source side colors must match the frame in one horizontal orientation")

    out = crop_bbox(g, (r0, c0, r1, c1))
    assert np.all(out[1:-1, 1:-1] == int(bg)), "frame interior must be empty"
    paste(out, oriented, 1, 1)
    return out

# --------------- generator (o5x) ---------------

# Canvas and color constants.
GEN_CANVAS_SIZE = 13
GEN_MARKER_COLOR = 4
GEN_PALETTE = [c for c in range(1, 10) if c != GEN_MARKER_COLOR]

# Interior shape and motif density.
GEN_INTERIOR_H_MIN, GEN_INTERIOR_H_MAX = 2, 5
GEN_INTERIOR_WIDTHS = [4, 6]
GEN_DENSITY_MIN, GEN_DENSITY_MAX = 0.45, 0.75

# Layout/orientation sampling and retry limits.
GEN_ROW_GAPS = range(5)
GEN_PATH_HORIZONTAL_FIRST_PROB = 0.5
GEN_EDGE_ROW_LEFT_PATH_PROB = 0.5
GEN_SOURCE_FLIP_PROB = 0.5
GEN_RETRY_LIMIT = 300


def _gen_add_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    er, ec = end
    mask[r, c] = True

    if random.random() < GEN_PATH_HORIZONTAL_FIRST_PROB:
        while c != ec:
            c += 1 if ec > c else -1
            mask[r, c] = True
        while r != er:
            r += 1 if er > r else -1
            mask[r, c] = True
    else:
        while r != er:
            r += 1 if er > r else -1
            mask[r, c] = True
        while c != ec:
            c += 1 if ec > c else -1
            mask[r, c] = True


def _gen_frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                cells.add((nr, nc))
    return list(cells)


def _gen_grow_regions(left: np.ndarray, right: np.ndarray, target: int) -> bool:
    """Mutate the two half-masks until their combined cell count reaches target."""
    while int(left.sum() + right.sum()) < target:
        left_frontier = _gen_frontier(left)
        right_frontier = _gen_frontier(right)
        choices: list[str] = []
        if left_frontier:
            choices.extend(["left"] * len(left_frontier))
        if right_frontier:
            choices.extend(["right"] * len(right_frontier))
        if not choices:
            return False

        side = random.choice(choices)
        if side == "left":
            r, c = random.choice(left_frontier)
            left[r, c] = True
        else:
            r, c = random.choice(right_frontier)
            right[r, c] = True
    return True


def _gen_build_motif(h: int, w: int, left_color: int, right_color: int) -> np.ndarray | None:
    """Return a connected two-color motif, or None if density bounds are impossible."""
    half = w // 2
    contact_r = random.randrange(h)
    left = np.zeros((h, half), dtype=bool)
    right = np.zeros((h, half), dtype=bool)

    if contact_r == 0:
        left_edge_r = 0
        right_edge_r = random.randrange(h)
    elif contact_r == h - 1:
        left_edge_r = random.randrange(h)
        right_edge_r = h - 1
    else:
        left_edge_r = contact_r
        right_edge_r = contact_r

    _gen_add_path(left, (contact_r, half - 1), (left_edge_r, 0))
    _gen_add_path(right, (contact_r, 0), (right_edge_r, half - 1))

    for row in (0, h - 1):
        if left[row].any() or right[row].any():
            continue
        if random.random() < GEN_EDGE_ROW_LEFT_PATH_PROB:
            _gen_add_path(left, (contact_r, half - 1), (row, random.randrange(half)))
        else:
            _gen_add_path(right, (contact_r, 0), (row, random.randrange(half)))

    min_cells = int(np.ceil(GEN_DENSITY_MIN * h * w))
    max_cells = int(np.floor(GEN_DENSITY_MAX * h * w))
    base_cells = int(left.sum() + right.sum())
    if base_cells > max_cells:
        return None

    target = random.randint(max(min_cells, base_cells), max_cells)
    if not _gen_grow_regions(left, right, target):
        return None

    motif = np.zeros((h, w), dtype=int)
    motif[:, :half][left] = left_color
    motif[:, half:][right] = right_color
    return motif


def _gen_row_placements(h: int, gap: int) -> list[tuple[int, int]]:
    frame_h = h + 2
    pairs: list[tuple[int, int]] = []
    for frame_r in range(GEN_CANVAS_SIZE - frame_h + 1):
        motif_above_r = frame_r - gap - h
        if motif_above_r >= 0:
            pairs.append((frame_r, motif_above_r))

        motif_below_r = frame_r + frame_h + gap
        if motif_below_r + h <= GEN_CANVAS_SIZE:
            pairs.append((frame_r, motif_below_r))
    return pairs


def _gen_draw_frame(
    grid: np.ndarray, top: int, left: int, h: int, w: int, left_color: int, right_color: int
) -> None:
    bottom = top + h + 1
    right = left + w + 1
    grid[top, left] = GEN_MARKER_COLOR
    grid[top, right] = GEN_MARKER_COLOR
    grid[bottom, left] = GEN_MARKER_COLOR
    grid[bottom, right] = GEN_MARKER_COLOR
    grid[top + 1 : bottom, left] = left_color
    grid[top + 1 : bottom, right] = right_color


def _gen_sample_layout(h: int, w: int) -> tuple[int, int, int, int]:
    valid_gaps = [gap for gap in GEN_ROW_GAPS if _gen_row_placements(h, gap)]
    gap = random.choice(valid_gaps)
    frame_r, motif_r = random.choice(_gen_row_placements(h, gap))
    frame_c = random.randrange(GEN_CANVAS_SIZE - (w + 2) + 1)
    motif_c = random.randrange(GEN_CANVAS_SIZE - w + 1)
    return frame_r, motif_r, frame_c, motif_c


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GEN_INTERIOR_H_MIN, GEN_INTERIOR_H_MAX)
    w = random.choice(GEN_INTERIOR_WIDTHS)

    left_color, right_color = random.sample(GEN_PALETTE, 2)
    oriented = _gen_build_motif(h, w, left_color, right_color)
    if oriented is None:
        return None

    out = np.zeros((h + 2, w + 2), dtype=int)
    _gen_draw_frame(out, 0, 0, h, w, left_color, right_color)
    paste(out, oriented, 1, 1)

    frame_r, motif_r, frame_c, motif_c = _gen_sample_layout(h, w)

    inp = np.zeros((GEN_CANVAS_SIZE, GEN_CANVAS_SIZE), dtype=int)
    _gen_draw_frame(inp, frame_r, frame_c, h, w, left_color, right_color)

    source = np.fliplr(oriented) if random.random() < GEN_SOURCE_FLIP_PROB else oriented
    paste(inp, source, motif_r, motif_c)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 846bdb03 input")
