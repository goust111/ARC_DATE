from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5
MARKER = 6
PALETTE = (1, 2, 3)

GEN_RETRIES = 500
HEIGHT_RANGE = (12, 18)
WIDTH_RANGE = (12, 18)
BAND_WIDTH_RANGE = (2, 4)
SIDE_CHOICES = ("left", "right")


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(counts.most_common(1)[0][0])


def _find_palette(grid: np.ndarray) -> tuple[int, np.ndarray]:
    bottom = np.asarray(grid, dtype=int)[-1]
    runs: list[tuple[int, np.ndarray]] = []
    c = 0
    while c < bottom.size:
        if int(bottom[c]) not in (BG, GREY, MARKER):
            start = c
            vals = []
            while c < bottom.size and int(bottom[c]) not in (BG, GREY, MARKER):
                vals.append(int(bottom[c]))
                c += 1
            runs.append((start, np.asarray(vals, dtype=int)))
        else:
            c += 1
    candidates = [(start, vals) for start, vals in runs if len(vals) == len(PALETTE)]
    assert len(candidates) == 1, "expected exactly one detached 3-cell palette"
    return candidates[0]


def _band_only_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int).copy()
    start, palette = _find_palette(g)
    g[-1, start : start + len(palette)] = BG
    return g


def _active_side(g: np.ndarray) -> str:
    w = g.shape[1]
    left_markers = int(np.count_nonzero(g[:, : min(4, w)] == MARKER))
    right_markers = int(np.count_nonzero(g[:, max(0, w - 4) :] == MARKER))
    assert max(left_markers, right_markers) > 0, "expected marker pixels inside the active edge band"
    assert left_markers != right_markers, "expected one edge to have the unique marker majority"
    return "left" if left_markers > right_markers else "right"


def _band_width(g: np.ndarray, side: str) -> int:
    h, w = g.shape
    if side == "left":
        width = 0
        for c in range(min(4, w)):
            if np.any(np.isin(g[:, c], (GREY, MARKER))):
                width += 1
            elif width:
                break
        assert 2 <= width <= 4
        assert not np.any(np.isin(g[:, width:], (GREY, MARKER)))
        return width

    width = 0
    for c in range(w - 1, max(-1, w - 5), -1):
        if np.any(np.isin(g[:, c], (GREY, MARKER))):
            width += 1
        elif width:
            break
    assert 2 <= width <= 4
    assert not np.any(np.isin(g[:, : w - width], (GREY, MARKER)))
    return width


def _source_band(g: np.ndarray, side: str, width: int) -> np.ndarray:
    if side == "left":
        return g[:, :width].copy()
    return g[:, g.shape[1] - width :].copy()


def _paint_translated_band(
    out: np.ndarray, band: np.ndarray, side: str, width: int, palette: np.ndarray
) -> np.ndarray:
    h, w = out.shape
    assert width == band.shape[1]
    plen = len(palette)
    source_mask = np.isin(band, (GREY, MARKER))
    for r, c in np.argwhere(source_mask):
        rr = int(r)
        cc = int(c)
        if side == "left":
            out[rr, cc] = BG
            dst_c = w - width + cc
            out[rr, dst_c] = int(palette[(width - 1 - cc) % plen])
        else:
            src_c = w - width + cc
            out[rr, src_c] = BG
            out[rr, cc] = int(palette[cc % plen])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the marked edge band to the opposite side, then recolor its translated layers by distance from the new outer edge.

    Input structure:
    - The background is black.
    - Exactly one vertical edge band of width 2-4 touches the left or right border; inside that band, only grey body pixels and magenta marker pixels appear.
    - The opposite edge band of the same width is empty, and the bottom row contains one detached 3-cell palette strip.

    Rule:
    - Detect the active edge as the side whose edge band contains the unique majority of magenta marker pixels.
    - Measure the active band width by counting consecutive non-empty columns from that border.
    - Erase all grey and magenta pixels from the source band.
    - Translate the same band pattern straight across to the opposite edge without mirroring it.
    - Recolor every translated pixel by its distance d from the new outer edge: use palette[d mod 3]. The palette strip itself stays unchanged.

    Key insight:
    - The moved pattern keeps its shape, but the color of each pixel is determined only after translation, because the relevant layer index is its depth inside the destination edge band.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC grid"
    assert _get_bg(g) == BG, "expected black background"

    palette_start, palette = _find_palette(g)
    assert palette.size == 3
    out = g.copy()
    band_grid = _band_only_grid(g)
    side = _active_side(band_grid)
    width = _band_width(band_grid, side)
    band = _source_band(g, side, width)

    out = _paint_translated_band(out, band, side, width, palette)
    out[-1, palette_start : palette_start + len(palette)] = palette
    return out


def _build_band_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    prev_start = rng.randint(0, max(0, height - 4))
    prev_len = rng.randint(2, min(4, height))
    prev_end = min(height - 1, prev_start + prev_len - 1)

    for c in range(width):
        if c == 0:
            start, end = prev_start, prev_end
        else:
            anchor = rng.randint(max(0, prev_start - 2), min(height - 1, prev_end + 2))
            span = rng.randint(2, min(4, height))
            shift = rng.randint(0, span - 1)
            start = max(0, min(height - span, anchor - shift))
            end = start + span - 1
            if start > prev_end + 1:
                start = max(0, prev_end - span + 1)
                end = start + span - 1
            if end < prev_start - 1:
                end = min(height - 1, prev_start + span - 1)
                start = end - span + 1
        mask[start : end + 1, c] = True
        prev_start, prev_end = start, end

    extras = rng.randint(0, max(1, width))
    for _ in range(extras):
        frontier = []
        for r, c in np.argwhere(mask):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        rr, cc = rng.choice(frontier)
        mask[int(rr), int(cc)] = True

    return mask


def _marker_positions(mask: np.ndarray) -> list[tuple[int, int]]:
    occ = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    assert len(occ) >= 4
    mid_r = (mask.shape[0] - 1) / 2.0
    candidates = [
        min(occ, key=lambda rc: (rc[0], rc[1])),
        max(occ, key=lambda rc: (rc[0], rc[1])),
        min(occ, key=lambda rc: (abs(rc[0] - mid_r), rc[1], rc[0])),
    ]
    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        if cell not in chosen:
            chosen.append(cell)
    desired = 2 + (len(occ) % 2)
    if desired == 3:
        for cell in occ:
            if cell not in chosen:
                chosen.append(cell)
            if len(chosen) == 3:
                break
    return chosen[:desired]


def _render_input(
    height: int,
    width: int,
    side: str,
    band_width: int,
    mask: np.ndarray,
    palette_col: int,
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    markers = set(_marker_positions(mask))
    row_limit = height - 1
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        gc = cc if side == "left" else width - band_width + cc
        grid[rr, gc] = MARKER if (rr, cc) in markers else GREY
    grid[row_limit, palette_col : palette_col + len(PALETTE)] = np.asarray(PALETTE, dtype=int)
    return grid


def _expected_output(
    height: int,
    width: int,
    side: str,
    band_width: int,
    mask: np.ndarray,
    palette_col: int,
) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    out[-1, palette_col : palette_col + len(PALETTE)] = np.asarray(PALETTE, dtype=int)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if side == "left":
            dst_c = width - band_width + cc
            out[rr, dst_c] = PALETTE[(band_width - 1 - cc) % len(PALETTE)]
        else:
            out[rr, cc] = PALETTE[cc % len(PALETTE)]
    return out


def _wrong_source_depth_output(
    height: int,
    width: int,
    side: str,
    band_width: int,
    mask: np.ndarray,
    palette_col: int,
) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    out[-1, palette_col : palette_col + len(PALETTE)] = np.asarray(PALETTE, dtype=int)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if side == "left":
            dst_c = width - band_width + cc
            out[rr, dst_c] = PALETTE[cc % len(PALETTE)]
        else:
            out[rr, cc] = PALETTE[(band_width - 1 - cc) % len(PALETTE)]
    return out


def _mirrored_output(
    height: int,
    width: int,
    side: str,
    band_width: int,
    mask: np.ndarray,
    palette_col: int,
) -> np.ndarray:
    return _expected_output(height, width, side, band_width, np.fliplr(mask), palette_col)


def _sample_palette_col(width: int, band_width: int, rng: random.Random) -> int:
    left_bound = band_width + 1
    right_bound = width - band_width - len(PALETTE) - 1
    assert left_bound <= right_bound, "grid too narrow for a detached palette strip"
    return rng.randint(left_bound, right_bound)


def _nontrivial_mask(mask: np.ndarray) -> bool:
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if len({int(c) for _r, c in np.argwhere(mask)}) != mask.shape[1]:
        return False
    return True


def _candidate_input(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*HEIGHT_RANGE)
    width = rng.randint(*WIDTH_RANGE)
    band_width = rng.randint(*BAND_WIDTH_RANGE)
    if width < 2 * band_width + len(PALETTE) + 3:
        return None

    side = rng.choice(SIDE_CHOICES)
    band_height = height - 1
    mask = _build_band_mask(band_height, band_width, rng)
    if not _nontrivial_mask(mask):
        return None

    palette_col = _sample_palette_col(width, band_width, rng)
    inp = _render_input(height, width, side, band_width, mask, palette_col)
    expected = _expected_output(height, width, side, band_width, mask, palette_col)
    wrong_depth = _wrong_source_depth_output(height, width, side, band_width, mask, palette_col)
    mirrored = _mirrored_output(height, width, side, band_width, mask, palette_col)

    if np.array_equal(expected, wrong_depth) or np.array_equal(expected, mirrored):
        return None
    if np.array_equal(inp, expected):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _candidate_input(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid edge-band translation sample")
