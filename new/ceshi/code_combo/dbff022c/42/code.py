# Combo variant of dbff022c (index 42)
# combo_spec: D4/2+D5/2+D9/0
# Name: 面积排序覆盖 + 多色边框 + 仅填充内边缘

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
import random
import sys
from typing import NamedTuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.insert(0, str(_SOLVERS))

from arc_utils import neighbors4


# Design analysis for the fused combo variant
# -------------------------------------------
# a) Concrete behavior and trace
# - The original task can discover each sprite by looking for a clean enclosed background hole,
#   then flood-fill the full hole from a palette lookup keyed by one border color.
# - This combo breaks that pipeline in three places at once:
#   1. D4/2: a border is now multicolor, so the lookup key must be the shell's majority color.
#   2. D5/2: only the hole's inner edge is painted; any deeper core stays background.
#   3. D9/0: sprites are processed from smaller area to larger area, and later fills may cover
#      earlier fills on the same background cells.
# - A concrete example is two overlapping 2-cell-thick rectangular rings. Their borders share
#   some visible pixels, their writable inner-edge rings overlap on a couple of background cells,
#   and both borders use a common accent color on the inner layer plus different majority colors
#   on the outer layer. The old local rule would see the accent and choose the accent mapping;
#   the fused rule must count the full shell, choose the outer majority color, paint only the
#   ring of writable hole cells, then let the larger rectangle repaint any shared ring cells.
#
# b) Complexity calibration
# - Plain 1-pixel rectangular frames are too weak for this combo: they do not provide enough
#   "outer-only" support for a majority color to beat the inner accent, and their ring-overlap
#   behavior is too fragile.
# - The generator therefore uses 2-cell-thick rectangular annuli. This makes all three changes
#   visible at once:
#   - the inner border layer can be accent-dominant locally,
#   - the outer layer can still dominate globally,
#   - the hole can retain a non-empty core after only the inner edge is filled.
# - Only two filled sprites are generated. That keeps the scene readable while still providing
#   repeated evidence for majority lookup and one visible overlap where area ordering matters.
#
# c) Hidden assumptions audit
# - `_best_palette` / `_palette_mapping` are still valid because the palette geometry did not
#   change; they depend on edge isolation, not on sprite internals.
# - The original solver assumed "sprite interior = one enclosed background component". That is
#   false now because one sprite's border may intrude into another sprite's hole before any fill
#   is applied. The new solver therefore recognizes sprites directly as 2-thick ring templates
#   instead of deriving them from clean background holes.
# - The original fill loop assumed "write every cell of the hole". That is false under D5/2, so
#   the new write set is restricted to ring cells of the hole whose input value is background.
# - The new order rule assumes area ties do not matter. The generator guarantees distinct border
#   areas for the two filled sprites, so the small-to-large ordering is visible and deterministic.
#
# d) Placement legality
# - The palette remains isolated on one grid edge and is separated from the sprite cluster by at
#   least one background row or column.
# - The two filled sprites are positioned so that:
#   - their ring write-sets overlap on background cells,
#   - their shared border pixels never require writing during filling,
#   - each sprite still keeps at least one untouched core background cell.
#
# e) Visual observability
# - The majority/accent distinction is visible because the inner border layer uses one repeated
#   accent color while the outer layer uses the sprite's majority key color.
# - The D5 change is visible because each hole keeps a blank core in the output.
# - The D9 change is visible because one small overlap region ends up taking the larger sprite's
#   fill color, even though the smaller sprite would have painted it earlier.
#
# f) Visual cleanliness
# - Only one overlap pair is used, and both shapes are simple axis-aligned annuli, so the grid
#   stays legible. There is no third overlapping object and no decorative noise.
# - The palette remains compact (2x4 or 4x2), and one palette entry is intentionally unused so
#   the mapping is not overloaded with too many active colors.
#
# g) Inferability of the color change
# - The accent color also appears in the palette, so a human can see that the wrong local choice
#   would produce a different fill. But both filled sprites consistently use the majority shell
#   color instead, making the global-count rule learnable from a few examples.
# - The output colors depend only on visible palette entries, visible shell frequencies, and the
#   visible size ordering of the two sprites. No hidden tie-break by color id is used.


_GRID_H = 15
_GRID_W = 15
_PALETTE_N = 4
_MAX_RETRIES = 400

_CLUSTER_LAYOUT = (
    (0, 0, 8, 9),  # smaller sprite
    (2, 2, 9, 9),  # larger sprite
)


class FrameCandidate(NamedTuple):
    r0: int
    c0: int
    h: int
    w: int
    area: int
    major_color: int
    accent_color: int
    ring_fillable: tuple[tuple[int, int], ...]
    core_bg: tuple[tuple[int, int], ...]


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the 2xN or Nx2 palette block on a grid edge.

    Returns (side, start, end) where:
      - side in {"top","bottom","left","right"}
      - for top/bottom: start/end are column indices [start,end)
      - for left/right: start/end are row indices [start,end)
    """
    h, w = g.shape
    best = None  # (score_tuple, side, start, end)

    def isolated(side: str, start: int, end: int) -> bool:
        pal = np.zeros((h, w), dtype=bool)
        if side == "top":
            pal[0:2, start:end] = True
        elif side == "bottom":
            pal[h - 2 : h, start:end] = True
        elif side == "left":
            pal[start:end, 0:2] = True
        else:
            pal[start:end, w - 2 : w] = True

        for r, c in zip(*np.where(pal), strict=False):
            r = int(r)
            c = int(c)
            v = int(g[r, c])
            if v == bg:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc]:
                    if int(g[rr, cc]) == v:
                        return False
        return True

    def consider_top() -> None:
        nonlocal best
        if h < 2:
            return
        a = g[0, :]
        b = g[1, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_a = a[c0:c1]
                seg_b = b[c0:c1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                cand = ((len(set(border_keys)), len(d), -(c1 - c0), nonbg), "top", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_bottom() -> None:
        nonlocal best
        if h < 2:
            return
        a = g[h - 1, :]
        b = g[h - 2, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_a = a[c0:c1]
                seg_b = b[c0:c1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                cand = ((len(set(border_keys)), len(d), -(c1 - c0), nonbg), "bottom", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_left() -> None:
        nonlocal best
        if w < 2:
            return
        a = g[:, 0]
        b = g[:, 1]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_a = a[r0:r1]
                seg_b = b[r0:r1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                cand = ((len(set(border_keys)), len(d), -(r1 - r0), nonbg), "left", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_right() -> None:
        nonlocal best
        if w < 2:
            return
        a = g[:, w - 1]
        b = g[:, w - 2]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_a = a[r0:r1]
                seg_b = b[r0:r1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                cand = ((len(set(border_keys)), len(d), -(r1 - r0), nonbg), "right", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    consider_top()
    consider_bottom()
    consider_left()
    consider_right()

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """Return (mapping border->fill, palette_mask)."""
    h, w = g.shape
    side, start, end = _best_palette(g, bg)
    pal_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, int] = {}

    if side == "top":
        pal_mask[0:2, start:end] = True
        border_line = g[0, start:end]
        fill_line = g[1, start:end]
    elif side == "bottom":
        pal_mask[h - 2 : h, start:end] = True
        border_line = g[h - 1, start:end]
        fill_line = g[h - 2, start:end]
    elif side == "left":
        pal_mask[start:end, 0:2] = True
        border_line = g[start:end, 0]
        fill_line = g[start:end, 1]
    else:
        pal_mask[start:end, w - 2 : w] = True
        border_line = g[start:end, w - 1]
        fill_line = g[start:end, w - 2]

    for b, f in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        b = int(b)
        if b != bg:
            mapping[b] = int(f)

    return mapping, pal_mask


@lru_cache(maxsize=None)
def _frame_template(h: int, w: int) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]:
    """
    Return relative coordinates for a 2-cell-thick rectangular annulus.

    Order: border cells, inner-border layer, hole ring, hole core.
    """
    assert h >= 7 and w >= 7

    border: list[tuple[int, int]] = []
    inner_border: list[tuple[int, int]] = []
    hole_ring: list[tuple[int, int]] = []
    hole_core: list[tuple[int, int]] = []
    hole_set = {
        (r, c)
        for r in range(2, h - 2)
        for c in range(2, w - 2)
    }

    for r in range(h):
        for c in range(w):
            if r < 2 or r >= h - 2 or c < 2 or c >= w - 2:
                border.append((r, c))
                if any((rr, cc) in hole_set for rr, cc in neighbors4(r, c)):
                    inner_border.append((r, c))

    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if r in (2, h - 3) or c in (2, w - 3):
                hole_ring.append((r, c))
            else:
                hole_core.append((r, c))

    return tuple(border), tuple(inner_border), tuple(hole_ring), tuple(hole_core)


def _translated(points: tuple[tuple[int, int], ...], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + dr, c0 + dc) for dr, dc in points]


def _find_frame_candidates(
    g: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> list[FrameCandidate]:
    """
    Recognize 2-cell-thick multicolor annuli directly from their border template.

    This replaces the original "discover clean background holes first" logic, which no longer
    works once one sprite can sit inside another sprite's hole before any fill is applied.
    """
    h, w = g.shape
    orig_bg = (g == bg) & (~pal_mask)
    out: list[FrameCandidate] = []

    for sh in range(7, h + 1):
        for sw in range(7, w + 1):
            border_rel, inner_rel, ring_rel, core_rel = _frame_template(sh, sw)
            area = len(border_rel)
            inner_need = len(inner_rel) // 2 + 1
            major_need = area // 2 + 1
            for r0 in range(h - sh + 1):
                for c0 in range(w - sw + 1):
                    border = _translated(border_rel, r0, c0)
                    if any(pal_mask[r, c] or int(g[r, c]) == bg for r, c in border):
                        continue

                    counts = Counter(int(g[r, c]) for r, c in border)
                    if len(counts) < 2:
                        continue
                    major_color, major_count = counts.most_common(1)[0]
                    if major_count < major_need:
                        continue

                    inner_border = _translated(inner_rel, r0, c0)
                    inner_counts = Counter(int(g[r, c]) for r, c in inner_border)
                    accent_color, accent_count = inner_counts.most_common(1)[0]
                    if accent_color == major_color or accent_count < inner_need:
                        continue

                    ring_cells = _translated(ring_rel, r0, c0)
                    fillable = [(r, c) for r, c in ring_cells if bool(orig_bg[r, c])]
                    if len(fillable) < 2:
                        continue

                    core_cells = _translated(core_rel, r0, c0)
                    core_bg = [(r, c) for r, c in core_cells if bool(orig_bg[r, c])]
                    if not core_bg:
                        continue

                    if major_color in mapping:
                        out.append(
                            FrameCandidate(
                                r0=r0,
                                c0=c0,
                                h=sh,
                                w=sw,
                                area=area,
                                major_color=int(major_color),
                                accent_color=int(accent_color),
                                ring_fillable=tuple(fillable),
                                core_bg=tuple(core_bg),
                            )
                        )

    return out


def _apply_sorted_ring_fills(
    g: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> np.ndarray:
    """
    Apply fills from smaller visible border-area to larger border-area.

    Later sprites may overwrite earlier fills, but only on cells that were background in the
    input; original border pixels are never rewritten.
    """
    out = g.copy()
    cands = _find_frame_candidates(g, bg, mapping, pal_mask)
    cands.sort(key=lambda cand: (cand.area, cand.r0, cand.c0, cand.h, cand.w))

    for cand in cands:
        fill = mapping.get(cand.major_color)
        if fill is None:
            continue
        for r, c in cand.ring_fillable:
            out[r, c] = int(fill)

    return out


def _behavior_stats(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether the three fused mechanisms are actually present in one sample."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    cands = [
        cand
        for cand in _find_frame_candidates(g, bg, mapping, pal_mask)
        if cand.major_color in mapping
    ]

    fill_hits: Counter[tuple[int, int]] = Counter()
    majority_disagreements = 0
    core_bg_total = 0
    for cand in cands:
        for cell in cand.ring_fillable:
            fill_hits[cell] += 1
        if cand.accent_color != cand.major_color:
            majority_disagreements += 1
        core_bg_total += len(cand.core_bg)

    overlap_cells = sum(1 for cnt in fill_hits.values() if cnt >= 2)
    return {
        "mapped_sprites": len(cands),
        "overlap_cells": overlap_cells,
        "majority_disagreements": majority_disagreements,
        "core_bg_total": core_bg_total,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Refill overlapping hollow sprites from an edge palette using majority-color lookup.

    Input structure:
    - Background dominates the outer boundary, and a compact 2xN or Nx2 palette sits on one
      grid edge with border-key colors on the outer line and fill colors on the inner line.
    - Away from the palette are two overlapping 2-cell-thick hollow annuli whose borders mix
      a repeated accent color with a different majority shell color.

    Rule:
    - Read the edge legend as a border-color -> fill-color mapping.
    - Detect each hollow annulus directly from its 2-thick ring border, because another sprite's
      border may already intrude into its hole.
    - For each sprite, count all border pixels of that annulus and use the most frequent color
      as the palette lookup key.
    - Process sprites from smaller visible area to larger visible area, and for each one paint
      only the hole cells that lie on the inner edge and were background in the input.

    Key insight:
    - The correct color comes from the full multicolor shell, not from the inner accent layer,
      and the correct write-set is only the hole's boundary ring; any shared writable cells end
      up with the larger sprite's color because it is applied later.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"
    return _apply_sorted_ring_fills(g, bg, mapping, pal_mask)


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    n = len(border_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        pal[outer] = border_keys
        pal[inner] = fill_values
    else:
        pal = np.zeros((n, 2), dtype=int)
        outer, inner = (0, 1) if side == "left" else (1, 0)
        pal[:, outer] = border_keys
        pal[:, inner] = fill_values
    return pal


def _place_palette(grid: np.ndarray, side: str, pal: np.ndarray, rng: random.Random) -> np.ndarray:
    """Place the palette on one edge and return its mask."""
    h, w = grid.shape
    ph, pw = pal.shape
    pal_mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        r0, c0 = 0, rng.randint(0, w - pw)
    elif side == "bottom":
        r0, c0 = h - ph, rng.randint(0, w - pw)
    elif side == "left":
        r0, c0 = rng.randint(0, h - ph), 0
    else:
        r0, c0 = rng.randint(0, h - ph), w - pw
    grid[r0 : r0 + ph, c0 : c0 + pw] = pal
    pal_mask[r0 : r0 + ph, c0 : c0 + pw] = True
    return pal_mask


def _orient_layout(
    layout: tuple[tuple[int, int, int, int], ...],
    transpose: bool,
    flip_r: bool,
    flip_c: bool,
) -> tuple[tuple[int, int, int, int], ...]:
    rects = list(layout)
    max_r = max(r0 + h for r0, _c0, h, _w in rects)
    max_c = max(c0 + w for _r0, c0, _h, w in rects)

    def apply_point(r: int, c: int) -> tuple[int, int]:
        rr, cc = r, c
        hh, ww = max_r, max_c
        if flip_r:
            rr = hh - 1 - rr
        if flip_c:
            cc = ww - 1 - cc
        if transpose:
            rr, cc = cc, rr
        return rr, cc

    out: list[tuple[int, int, int, int]] = []
    for r0, c0, h, w in rects:
        pts = [apply_point(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        nr0, nc0 = min(rows), min(cols)
        nh, nw = max(rows) - nr0 + 1, max(cols) - nc0 + 1
        out.append((nr0, nc0, nh, nw))

    min_r = min(r0 for r0, _c0, _h, _w in out)
    min_c = min(c0 for _r0, c0, _h, _w in out)
    return tuple((r0 - min_r, c0 - min_c, h, w) for r0, c0, h, w in out)


def _paint_annulus(
    grid: np.ndarray,
    frame: tuple[int, int, int, int],
    major_color: int,
    accent_color: int,
    shared_border: set[tuple[int, int]],
) -> dict[str, object]:
    r0, c0, h, w = frame
    border_rel, inner_rel, ring_rel, core_rel = _frame_template(h, w)
    border = {(r0 + dr, c0 + dc) for dr, dc in border_rel}
    inner_border = {(r0 + dr, c0 + dc) for dr, dc in inner_rel}
    ring = {(r0 + dr, c0 + dc) for dr, dc in ring_rel}
    core = {(r0 + dr, c0 + dc) for dr, dc in core_rel}

    for r, c in border:
        grid[r, c] = int(accent_color if (r, c) in inner_border or (r, c) in shared_border else major_color)

    return {
        "frame": frame,
        "border": border,
        "ring": ring,
        "core": core,
        "area": len(border),
        "major": int(major_color),
    }


def _cluster_origin(side: str, cluster_h: int, cluster_w: int, rng: random.Random) -> tuple[int, int]:
    if side == "top":
        return 3, rng.randint(1, _GRID_W - cluster_w - 1)
    if side == "bottom":
        return 1, rng.randint(1, _GRID_W - cluster_w - 1)
    if side == "left":
        return rng.randint(1, _GRID_H - cluster_h - 1), 3
    return rng.randint(1, _GRID_H - cluster_h - 1), 1


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0
    side = rng.choice(["top", "bottom", "left", "right"])

    all_colors = list(range(1, 10))
    rng.shuffle(all_colors)
    border_keys = all_colors[:_PALETTE_N]
    fill_values = all_colors[_PALETTE_N : 2 * _PALETTE_N]
    mapping = dict(zip(border_keys, fill_values))

    small_major = int(border_keys[0])
    large_major = int(border_keys[1])
    accent_key = int(border_keys[2])

    pal = _build_palette(side, border_keys, fill_values)
    grid = np.full((_GRID_H, _GRID_W), bg, dtype=int)
    pal_mask = _place_palette(grid, side, pal, rng)

    layout = _orient_layout(
        _CLUSTER_LAYOUT,
        transpose=bool(rng.randint(0, 1)),
        flip_r=bool(rng.randint(0, 1)),
        flip_c=bool(rng.randint(0, 1)),
    )
    cluster_h = max(r0 + h for r0, _c0, h, _w in layout)
    cluster_w = max(c0 + w for _r0, c0, _h, w in layout)
    base_r, base_c = _cluster_origin(side, cluster_h, cluster_w, rng)
    frames = tuple((base_r + r0, base_c + c0, h, w) for r0, c0, h, w in layout)

    small_frame, large_frame = frames
    border_small = set(_translated(_frame_template(small_frame[2], small_frame[3])[0], small_frame[0], small_frame[1]))
    border_large = set(_translated(_frame_template(large_frame[2], large_frame[3])[0], large_frame[0], large_frame[1]))
    shared_border = border_small & border_large
    if not shared_border:
        return None
    if any(pal_mask[r, c] for r, c in border_small | border_large):
        return None

    infos = []
    infos.append(_paint_annulus(grid, small_frame, small_major, accent_key, shared_border))
    infos.append(_paint_annulus(grid, large_frame, large_major, accent_key, shared_border))

    border_union = set().union(*(info["border"] for info in infos))
    if any((r == 0 or c == 0 or r == _GRID_H - 1 or c == _GRID_W - 1) for r, c in border_union):
        return None

    for info in infos:
        if not (info["core"] - (border_union - info["border"])):
            return None

    output = grid.copy()
    for info in sorted(infos, key=lambda x: int(x["area"])):
        fill = int(mapping[int(info["major"])])
        ring_fillable = [cell for cell in info["ring"] if int(grid[cell]) == bg]
        if len(ring_fillable) < 2:
            return None
        for r, c in ring_fillable:
            output[r, c] = fill

    stats = _behavior_stats(grid)
    if stats["mapped_sprites"] < 2 or stats["overlap_cells"] < 1:
        return None
    if stats["majority_disagreements"] < 2 or stats["core_bg_total"] < 2:
        return None

    try:
        if np.array_equal(main(grid), output):
            return grid
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
