from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
MASK = 5
BAR_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _majority_nonzero(values: np.ndarray) -> int:
    colors = [int(v) for v in values.ravel() if int(v) != BG]
    assert colors
    return int(Counter(colors).most_common(1)[0][0])


def _contiguous_band(indices: np.ndarray) -> tuple[int, int]:
    idx = np.asarray(indices, dtype=int)
    assert idx.size in (1, 2), f"expected 1-2 band cells, got {idx.tolist()}"
    assert int(idx[-1] - idx[0] + 1) == int(idx.size), f"expected contiguous band, got {idx.tolist()}"
    return int(idx[0]), int(idx[-1])


def _quadrant_slices(qh: int, qw: int, hband: int, vband: int) -> dict[str, tuple[slice, slice]]:
    return {
        "tl": (slice(0, qh), slice(0, qw)),
        "tr": (slice(0, qh), slice(qw + vband, 2 * qw + vband)),
        "bl": (slice(qh + hband, 2 * qh + hband), slice(0, qw)),
        "br": (slice(qh + hband, 2 * qh + hband), slice(qw + vband, 2 * qw + vband)),
    }


def _mirror_partners(top_orientation: str) -> dict[str, str]:
    if top_orientation == "h":
        return {"tl": "bl", "tr": "br", "bl": "tl", "br": "tr"}
    return {"tl": "tr", "tr": "tl", "bl": "br", "br": "bl"}


def _wrong_axis_partners(top_orientation: str) -> dict[str, str]:
    if top_orientation == "h":
        return {"tl": "tr", "tr": "tl", "bl": "br", "br": "bl"}
    return {"tl": "bl", "tr": "br", "bl": "tl", "br": "tr"}


def _make_sparse_pattern(h: int, w: int, palette: list[int]) -> np.ndarray:
    patch = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    inner_rows = range(1, h - 1)
    inner_cols = range(1, w - 1)
    target_shapes = random.randint(3, min(6, h + w - 4))

    for _ in range(target_shapes):
        placed = False
        for _ in range(60):
            sh = random.randint(1, min(3, h - 2))
            sw = random.randint(1, min(3, w - 2))
            r0 = random.randint(1, h - sh - 1)
            c0 = random.randint(1, w - sw - 1)
            r1 = r0 + sh - 1
            c1 = c0 + sw - 1
            if blocked[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)].any():
                continue
            color = random.choice(palette)
            patch[r0 : r1 + 1, c0 : c1 + 1] = int(color)
            blocked[r0 : r1 + 1, c0 : c1 + 1] = True
            placed = True
            break
        if not placed:
            continue

    # Add a few isolated pixels so mask candidates can expose mixed rectangles.
    singles = random.randint(1, 3)
    cells = [(r, c) for r in inner_rows for c in inner_cols if not blocked[r, c]]
    random.shuffle(cells)
    for r, c in cells[:singles]:
        patch[r, c] = int(random.choice(palette))
        blocked[r, c] = True

    if np.count_nonzero(patch) < 3:
        refill = [(r, c) for r in inner_rows for c in inner_cols if patch[r, c] == BG]
        random.shuffle(refill)
        for r, c in refill[: 3 - int(np.count_nonzero(patch))]:
            patch[r, c] = int(random.choice(palette))
    return patch


def _render_full_grid(
    qh: int,
    qw: int,
    hband: int,
    vband: int,
    hcolor: int,
    vcolor: int,
    top_orientation: str,
    quadrants: dict[str, np.ndarray],
) -> np.ndarray:
    grid = np.full((2 * qh + hband, 2 * qw + vband), BG, dtype=int)
    grid[qh : qh + hband, :] = int(hcolor)
    grid[:, qw : qw + vband] = int(vcolor)
    cross_color = hcolor if top_orientation == "h" else vcolor
    grid[qh : qh + hband, qw : qw + vband] = int(cross_color)

    for name, (rs, cs) in _quadrant_slices(qh, qw, hband, vband).items():
        grid[rs, cs] = quadrants[name]
    return grid


def _find_mask_candidates(
    quadrants: dict[str, np.ndarray],
    qh: int,
    qw: int,
    top_orientation: str,
) -> list[tuple[str, int, int, int, int]]:
    true_partner = _mirror_partners(top_orientation)
    wrong_partner = _wrong_axis_partners(top_orientation)
    candidates: list[tuple[str, int, int, int, int]] = []

    for target in ("tl", "tr", "bl", "br"):
        src = quadrants[true_partner[target]]
        wrong = quadrants[wrong_partner[target]]
        for rh in range(2, min(4, qh - 1) + 1):
            for rw in range(2, min(4, qw - 1) + 1):
                for r0 in range(1, qh - rh):
                    for c0 in range(1, qw - rw):
                        sl = (slice(r0, r0 + rh), slice(c0, c0 + rw))
                        src_rect = src[sl]
                        wrong_rect = wrong[sl]
                        nonzero = int(np.count_nonzero(src_rect))
                        if nonzero < 2:
                            continue
                        if len(set(map(int, src_rect.ravel()))) < 2:
                            continue
                        if np.array_equal(src_rect, wrong_rect):
                            continue
                        candidates.append((target, r0, c0, rh, rw))
    return candidates


def _expected_from_full(full: np.ndarray, qh: int, qw: int, hband: int, vband: int, top_orientation: str, hcolor: int, vcolor: int) -> np.ndarray:
    out = full.copy()
    out[qh : qh + hband, qw : qw + vband] = int(vcolor if top_orientation == "h" else hcolor)
    return out


def _wrong_axis_fill(inp: np.ndarray) -> np.ndarray:
    g = np.asarray(inp, dtype=int)
    rows = np.where(np.all(g != BG, axis=1))[0]
    cols = np.where(np.all(g != BG, axis=0))[0]
    hr0, hr1 = _contiguous_band(rows)
    vc0, vc1 = _contiguous_band(cols)

    out = g.copy()
    mask_coords = np.argwhere(g == MASK)
    assert mask_coords.size > 0

    # Deliberately use the opposite axis from the visible crossing color.
    bar_rows = np.r_[0:hr0, hr1 + 1 : g.shape[0]]
    bar_cols = np.r_[0:vc0, vc1 + 1 : g.shape[1]]
    hcolor = _majority_nonzero(g[hr0 : hr1 + 1][:, bar_cols])
    vcolor = _majority_nonzero(g[bar_rows][:, vc0 : vc1 + 1])
    cross = set(map(int, g[hr0 : hr1 + 1, vc0 : vc1 + 1].ravel()))
    assert len(cross) == 1
    top_orientation = "h" if next(iter(cross)) == hcolor else "v"
    wrong_axis = "v" if top_orientation == "h" else "h"

    for r, c in mask_coords:
        rr = hr0 + hr1 - int(r) if wrong_axis == "h" else int(r)
        cc = int(c) if wrong_axis == "h" else vc0 + vc1 - int(c)
        out[int(r), int(c)] = int(g[rr, cc])
    out[hr0 : hr1 + 1, vc0 : vc1 + 1] = int(vcolor if top_orientation == "h" else hcolor)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the crossing color to choose the repair axis, mirror-fill the gray damage, and swap the crossing color.

    Input structure:
    - Black background surrounds one full-width horizontal bar and one full-height vertical bar, each 1-2 cells thick.
    - Their crossing rectangle is uniformly colored as either the horizontal bar color or the vertical bar color.
    - The four regions around the bars contain sparse decorative pixels, and one region has a solid gray rectangular mask.

    Rule:
    - Detect the full foreground rows and columns to locate the horizontal bar, vertical bar, and crossing rectangle.
    - If the crossing color equals the horizontal bar color, use the horizontal bar as the symmetry axis; otherwise use the vertical bar.
    - For every gray cell, reflect its coordinate across the chosen bar band and copy the visible mirrored color into the mask.
    - Recolor the whole crossing rectangle to the other bar's color.

    Key insight:
    - The crossing color is both a layering clue and a control signal: it says which bar is on top, and therefore which symmetry axis must be used to recover the hidden pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    full_rows = np.where(np.all(g != BG, axis=1))[0]
    full_cols = np.where(np.all(g != BG, axis=0))[0]
    hr0, hr1 = _contiguous_band(full_rows)
    vc0, vc1 = _contiguous_band(full_cols)

    outside_cols = np.r_[0:vc0, vc1 + 1 : g.shape[1]]
    outside_rows = np.r_[0:hr0, hr1 + 1 : g.shape[0]]
    assert outside_cols.size > 0 and outside_rows.size > 0

    hcolor = _majority_nonzero(g[hr0 : hr1 + 1][:, outside_cols])
    vcolor = _majority_nonzero(g[outside_rows][:, vc0 : vc1 + 1])

    cross_colors = set(map(int, g[hr0 : hr1 + 1, vc0 : vc1 + 1].ravel()))
    assert len(cross_colors) == 1, "crossing rectangle must be uniform"
    cross_color = next(iter(cross_colors))
    assert cross_color in {hcolor, vcolor}, "crossing must show one of the two bar colors"
    top_orientation = "h" if cross_color == hcolor else "v"

    out = g.copy()
    mask_coords = np.argwhere(g == MASK)
    assert mask_coords.size > 0, "expected one gray mask"
    for r, c in mask_coords:
        rr = hr0 + hr1 - int(r) if top_orientation == "h" else int(r)
        cc = int(c) if top_orientation == "h" else vc0 + vc1 - int(c)
        assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
        assert int(g[rr, cc]) != MASK
        out[int(r), int(c)] = int(g[rr, cc])

    out[hr0 : hr1 + 1, vc0 : vc1 + 1] = int(vcolor if top_orientation == "h" else hcolor)
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        qh = random.randint(4, 11)
        qw = random.randint(4, 11)
        hband = random.randint(1, 2)
        vband = random.randint(1, 2)

        hcolor, vcolor = random.sample(BAR_COLORS, 2)
        deco_palette = [c for c in BAR_COLORS if c not in {hcolor, vcolor, MASK}]
        deco_count = random.randint(2, min(4, len(deco_palette)))
        deco_palette = random.sample(deco_palette, deco_count)
        top_orientation = random.choice(["h", "v"])

        if top_orientation == "h":
            top_left = _make_sparse_pattern(qh, qw, deco_palette)
            top_right = _make_sparse_pattern(qh, qw, deco_palette)
            if np.array_equal(top_left, top_right):
                continue
            quadrants = {
                "tl": top_left,
                "tr": top_right,
                "bl": top_left.copy(),
                "br": top_right.copy(),
            }
        else:
            top_left = _make_sparse_pattern(qh, qw, deco_palette)
            bottom_left = _make_sparse_pattern(qh, qw, deco_palette)
            if np.array_equal(top_left, bottom_left):
                continue
            quadrants = {
                "tl": top_left,
                "tr": top_left.copy(),
                "bl": bottom_left,
                "br": bottom_left.copy(),
            }

        candidates = _find_mask_candidates(quadrants, qh, qw, top_orientation)
        if not candidates:
            continue

        full = _render_full_grid(qh, qw, hband, vband, hcolor, vcolor, top_orientation, quadrants)
        expected = _expected_from_full(full, qh, qw, hband, vband, top_orientation, hcolor, vcolor)
        inp = full.copy()

        target, r0, c0, rh, rw = random.choice(candidates)
        rs, cs = _quadrant_slices(qh, qw, hband, vband)[target]
        inp[rs.start + r0 : rs.start + r0 + rh, cs.start + c0 : cs.start + c0 + rw] = MASK

        mask_coords = np.argwhere(inp == MASK)
        assert mask_coords.size > 0
        mask_bbox = _bbox_coords(mask_coords)
        mr0, mc0, mr1, mc1 = mask_bbox
        assert int((inp == MASK).sum()) == (mr1 - mr0 + 1) * (mc1 - mc0 + 1)

        # Hard semantic checks: true-axis recovery must differ from false-axis recovery.
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        if np.array_equal(_wrong_axis_fill(inp), expected):
            continue
        if int(np.count_nonzero(expected != inp)) <= hband * vband:
            continue
        return inp

    raise RuntimeError("failed to generate a valid crossing-driven symmetry repair input")
