from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400
PANEL_ATTEMPTS = 250
STAMP_ATTEMPTS = 200
ANCHOR_ATTEMPTS = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    return len(_components(mask)) == 1


def _panel_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == MAROON)]
    assert len(sep_cols) in (2, 3), "expected 3 or 4 panels"
    spans: list[tuple[int, int]] = []
    left = 0
    for sep in sep_cols:
        spans.append((left, sep))
        left = sep + 1
    spans.append((left, grid.shape[1]))
    widths = {end - start for start, end in spans}
    assert len(widths) == 1, "panels must have equal width"
    return spans, sep_cols


def _largest_canvas(panel: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    best: tuple[int, int, tuple[int, int, int, int]] | None = None
    second_size = -1
    for color in sorted(set(map(int, np.unique(panel)))):
        if color in (BLACK, MAROON):
            continue
        for comp in _components(panel == color):
            size = len(comp)
            bbox = _bbox(comp)
            if best is None or size > best[0]:
                if best is not None:
                    second_size = max(second_size, best[0])
                best = (size, color, bbox)
            else:
                second_size = max(second_size, size)
    assert best is not None, "panel must contain a canvas"
    assert best[0] > second_size, "canvas must be the unique largest monochrome component"
    _size, color, bbox = best
    r0, c0, r1, c1 = bbox
    inside = panel[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((inside != BLACK) & (inside != MAROON)), "canvas bbox must be fully filled"
    return color, bbox


def _stamp_relative_pixels(panel: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, list[tuple[int, int, int]]]:
    r0, c0, r1, c1 = bbox
    mask = np.ones(panel.shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    stamp_cells = [tuple(map(int, pt)) for pt in np.argwhere(mask & (panel != BLACK))]
    assert stamp_cells, "panel must contain a stamp outside the canvas"
    counts = Counter(int(panel[r, c]) for r, c in stamp_cells)
    min_count = min(counts.values())
    anchor_colors = [color for color, count in counts.items() if count == min_count]
    assert len(anchor_colors) == 1, "stamp must have a unique smallest-area color"
    anchor_color = int(anchor_colors[0])
    anchor_cells = [(r, c) for r, c in stamp_cells if int(panel[r, c]) == anchor_color]
    assert len(anchor_cells) == 1, "smallest-area stamp color must occupy one pixel"
    ar, ac = anchor_cells[0]
    rel_pixels = [(r - ar, c - ac, int(panel[r, c])) for r, c in stamp_cells]
    return anchor_color, rel_pixels


def _render_canvas(
    shape: tuple[int, int],
    base_color: int,
    anchor_color: int,
    rel_pixels: list[tuple[int, int, int]],
    anchors: list[tuple[int, int]],
) -> np.ndarray:
    h, w = shape
    canvas = np.full((h, w), base_color, dtype=int)
    for ar, ac in anchors:
        assert 0 <= ar < h and 0 <= ac < w
        for dr, dc, color in rel_pixels:
            rr = ar + dr
            cc = ac + dc
            if 0 <= rr < h and 0 <= cc < w:
                canvas[rr, cc] = color
    assert all(canvas[ar, ac] == anchor_color for ar, ac in anchors)
    return canvas


def _solve_panel(panel: np.ndarray) -> np.ndarray:
    base_color, bbox = _largest_canvas(panel)
    anchor_color, rel_pixels = _stamp_relative_pixels(panel, bbox)
    r0, c0, r1, c1 = bbox
    inside = panel[r0 : r1 + 1, c0 : c1 + 1]
    inside_colors = set(map(int, np.unique(inside)))
    assert inside_colors <= {base_color, anchor_color}, "canvas may only contain base color and anchors"
    anchors = [tuple(map(int, pt)) for pt in np.argwhere(inside == anchor_color)]
    assert anchors, "canvas must contain anchors"
    rendered = _render_canvas((inside.shape[0], inside.shape[1]), base_color, anchor_color, rel_pixels, anchors)
    out = np.full(panel.shape, BLACK, dtype=int)
    out[r0 : r1 + 1, c0 : c1 + 1] = rendered
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each external stamp sprite and replay it on the internal canvas at every anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    spans, sep_cols = _panel_spans(g)
    out = np.full_like(g, BLACK)
    for c in sep_cols:
        out[:, c] = MAROON
    for start, end in spans:
        out[:, start:end] = _solve_panel(g[:, start:end])
    return out


def _non_overlapping_choice(
    rng: random.Random,
    candidates: list[dict[str, object]],
    count: int,
    require_clip: bool,
    canvas_shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    fit = [cand for cand in candidates if bool(cand["fit"])]
    clipped = [cand for cand in candidates if not bool(cand["fit"])]
    if require_clip and not clipped:
        return None
    if not require_clip and not fit:
        return None

    for _ in range(ANCHOR_ATTEMPTS):
        selected: list[dict[str, object]] = []
        covered: set[tuple[int, int]] = set()
        need_clip = require_clip
        for _idx in range(count):
            if need_clip:
                pool = clipped
            elif not require_clip:
                pool = fit
            else:
                pool = candidates
            valid = [
                cand
                for cand in pool
                if not covered.intersection(cand["cover"]) and cand not in selected
            ]
            if not valid:
                break
            chosen = rng.choice(valid)
            selected.append(chosen)
            covered.update(chosen["cover"])
            if not bool(chosen["fit"]):
                need_clip = False
        if len(selected) != count:
            continue
        anchors = [tuple(cand["pos"]) for cand in selected]
        if _rect_minus_holes_connected(canvas_shape, anchors):
            return anchors
    return None


def _rect_minus_holes_connected(shape: tuple[int, int], holes: list[tuple[int, int]]) -> bool:
    h, w = shape
    mask = np.ones((h, w), dtype=bool)
    for r, c in holes:
        mask[r, c] = False
    return _mask_connected(mask)


def _candidate_anchors(
    shape: tuple[int, int],
    rel_pixels: list[tuple[int, int, int]],
) -> list[dict[str, object]]:
    h, w = shape
    rel_pairs = [(dr, dc) for dr, dc, _color in rel_pixels]
    min_dr = min(dr for dr, _dc in rel_pairs)
    max_dr = max(dr for dr, _dc in rel_pairs)
    min_dc = min(dc for _dr, dc in rel_pairs)
    max_dc = max(dc for _dr, dc in rel_pairs)

    candidates: list[dict[str, object]] = []
    for ar in range(h):
        for ac in range(w):
            cover: set[tuple[int, int]] = set()
            for dr, dc, _color in rel_pixels:
                rr = ar + dr
                cc = ac + dc
                if 0 <= rr < h and 0 <= cc < w:
                    cover.add((rr, cc))
            fit = (-min_dr <= ar <= h - 1 - max_dr) and (-min_dc <= ac <= w - 1 - max_dc)
            candidates.append({"pos": (ar, ac), "fit": fit, "cover": cover})
    return candidates


def _make_stamp(rng: random.Random) -> tuple[np.ndarray, int, tuple[int, int]]:
    for _ in range(STAMP_ATTEMPTS):
        sh = rng.randint(3, 5)
        sw = rng.randint(3, 5)
        mask = np.ones((sh, sw), dtype=bool)
        corners = [(0, 0), (0, sw - 1), (sh - 1, 0), (sh - 1, sw - 1)]
        rng.shuffle(corners)
        remove_count = rng.randint(0, min(2, len(corners)))
        for rr, cc in corners[:remove_count]:
            mask[rr, cc] = False
        if not _mask_connected(mask):
            continue
        cells = [tuple(map(int, pt)) for pt in np.argwhere(mask)]
        max_extra = min(3, (len(cells) - 1) // 2)
        if max_extra < 1:
            continue
        extra_count = rng.randint(1, max_extra)
        anchor = rng.choice(cells)
        rest = [cell for cell in cells if cell != anchor]
        seeds = rng.sample(rest, extra_count)
        regions: list[list[tuple[int, int]]] = [[] for _ in range(extra_count)]
        for cell in rest:
            best_idx = min(
                range(extra_count),
                key=lambda idx: (
                    abs(cell[0] - seeds[idx][0]) + abs(cell[1] - seeds[idx][1]),
                    seeds[idx][0],
                    seeds[idx][1],
                    idx,
                ),
            )
            regions[best_idx].append(cell)
        if min(len(region) for region in regions) < 2:
            continue

        colors = rng.sample(COLOR_POOL, extra_count + 1)
        anchor_color = colors[0]
        sprite = np.full((sh, sw), BLACK, dtype=int)
        sprite[anchor] = anchor_color
        for region, color in zip(regions, colors[1:]):
            for r, c in region:
                sprite[r, c] = color
        return sprite, anchor_color, anchor
    raise RuntimeError("failed to build stamp")


def _choose_canvas_and_stamp_positions(
    rng: random.Random,
    panel_h: int,
    panel_w: int,
    stamp_shape: tuple[int, int],
) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    sh, sw = stamp_shape
    gap = 1
    margin = 1
    sides = ["left", "right", "top", "bottom"]
    rng.shuffle(sides)

    for side in sides:
        if side in ("left", "right"):
            max_ch = panel_h - 2 * margin
            max_cw = panel_w - sw - gap - 2 * margin
        else:
            max_ch = panel_h - sh - gap - 2 * margin
            max_cw = panel_w - 2 * margin
        min_ch = max(6, sh)
        min_cw = max(6, sw)
        if max_ch < min_ch or max_cw < min_cw:
            continue

        ch = rng.randint(min_ch, max_ch)
        cw = rng.randint(min_cw, max_cw)

        if side == "left":
            c0_lo = margin + sw + gap
            c0_hi = panel_w - margin - cw
            if c0_lo > c0_hi:
                continue
            c0 = rng.randint(c0_lo, c0_hi)
            r0 = rng.randint(margin, panel_h - margin - ch)
            stamp_c = rng.randint(margin, c0 - gap - sw)
            stamp_r = rng.randint(0, panel_h - sh)
        elif side == "right":
            c0_lo = margin
            c0_hi = panel_w - margin - cw - sw - gap
            if c0_lo > c0_hi:
                continue
            c0 = rng.randint(c0_lo, c0_hi)
            r0 = rng.randint(margin, panel_h - margin - ch)
            stamp_c = rng.randint(c0 + cw + gap, panel_w - sw - margin)
            stamp_r = rng.randint(0, panel_h - sh)
        elif side == "top":
            r0_lo = margin + sh + gap
            r0_hi = panel_h - margin - ch
            if r0_lo > r0_hi:
                continue
            r0 = rng.randint(r0_lo, r0_hi)
            c0 = rng.randint(margin, panel_w - margin - cw)
            stamp_r = rng.randint(margin, r0 - gap - sh)
            stamp_c = rng.randint(0, panel_w - sw)
        else:
            r0_lo = margin
            r0_hi = panel_h - margin - ch - sh - gap
            if r0_lo > r0_hi:
                continue
            r0 = rng.randint(r0_lo, r0_hi)
            c0 = rng.randint(margin, panel_w - margin - cw)
            stamp_r = rng.randint(r0 + ch + gap, panel_h - sh - margin)
            stamp_c = rng.randint(0, panel_w - sw)

        return (r0, c0, ch, cw), (stamp_r, stamp_c)
    raise RuntimeError("failed to place canvas and stamp")


def _build_panel(
    rng: random.Random,
    panel_h: int,
    panel_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(PANEL_ATTEMPTS):
        stamp, anchor_color, anchor_in_stamp = _make_stamp(rng)
        sh, sw = stamp.shape
        rel_pixels = []
        for r, c in np.argwhere(stamp != BLACK):
            rel_pixels.append((int(r - anchor_in_stamp[0]), int(c - anchor_in_stamp[1]), int(stamp[r, c])))

        try:
            (r0, c0, ch, cw), (stamp_r, stamp_c) = _choose_canvas_and_stamp_positions(
                rng, panel_h, panel_w, (sh, sw)
            )
        except RuntimeError:
            continue

        sprite_colors = set(map(int, np.unique(stamp))) - {BLACK}
        canvas_choices = [color for color in COLOR_POOL if color not in sprite_colors]
        if not canvas_choices:
            continue
        canvas_color = rng.choice(canvas_choices)

        candidates = _candidate_anchors((ch, cw), rel_pixels)
        if not candidates:
            continue
        fit_count = sum(bool(cand["fit"]) for cand in candidates)
        clip_count = len(candidates) - fit_count
        if fit_count == 0 or clip_count == 0:
            continue

        want_clip = bool(rng.randint(0, 1))
        anchor_total = rng.randint(1, min(4, len(candidates)))
        anchors = _non_overlapping_choice(rng, candidates, anchor_total, want_clip, (ch, cw))
        if anchors is None:
            continue

        panel_in = np.full((panel_h, panel_w), BLACK, dtype=int)
        panel_out = np.full((panel_h, panel_w), BLACK, dtype=int)
        panel_in[r0 : r0 + ch, c0 : c0 + cw] = canvas_color
        for ar, ac in anchors:
            panel_in[r0 + ar, c0 + ac] = anchor_color
        stamp_mask = stamp != BLACK
        target = panel_in[stamp_r : stamp_r + sh, stamp_c : stamp_c + sw]
        if np.any(target[stamp_mask] != BLACK):
            continue
        target[stamp_mask] = stamp[stamp_mask]

        rendered = _render_canvas((ch, cw), canvas_color, anchor_color, rel_pixels, anchors)
        panel_out[r0 : r0 + ch, c0 : c0 + cw] = rendered

        if not np.array_equal(_solve_panel(panel_in), panel_out):
            continue
        return panel_in, panel_out
    raise RuntimeError("failed to build panel")


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    panel_count = rng.choice((3, 4))
    panel_h = rng.randint(14, 19)
    panel_w = rng.randint(13, 17)
    try:
        panels = [_build_panel(rng, panel_h, panel_w) for _ in range(panel_count)]
    except RuntimeError:
        return None

    width = panel_count * panel_w + (panel_count - 1)
    grid = np.full((panel_h, width), BLACK, dtype=int)
    out = np.full_like(grid, BLACK)
    cursor = 0
    for idx, (panel_in, panel_out) in enumerate(panels):
        grid[:, cursor : cursor + panel_w] = panel_in
        out[:, cursor : cursor + panel_w] = panel_out
        cursor += panel_w
        if idx != panel_count - 1:
            grid[:, cursor] = MAROON
            out[:, cursor] = MAROON
            cursor += 1

    if not np.array_equal(main(grid), out):
        return None
    return grid, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input")
