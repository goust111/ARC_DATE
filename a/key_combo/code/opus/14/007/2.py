from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
TEMPLATE_COLORS = (1, 3, 4, 6, 7, 8, 9)
GEN_RETRY_LIMIT = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _infer_vertical_axis(sprite: np.ndarray) -> int:
    h, w = sprite.shape
    known = sprite != BLACK
    best_a2 = w - 1
    best_key = None
    for a2 in range(0, 2 * (w - 1) + 1):
        mismatches = 0
        paired = 0
        in_bounds = 0
        for r in range(h):
            for c in np.where(known[r])[0]:
                c2 = a2 - int(c)
                if 0 <= c2 < w:
                    in_bounds += 1
                    if known[r, c2]:
                        paired += 1
                        if int(sprite[r, c]) != int(sprite[r, c2]):
                            mismatches += 1
        key = (mismatches, -paired, -in_bounds, abs(a2 - (w - 1)))
        if best_key is None or key < best_key:
            best_key = key
            best_a2 = a2
    return int(best_a2)


def _infer_horizontal_axis(sprite: np.ndarray) -> int:
    h, w = sprite.shape
    known = sprite != BLACK
    best_b2 = h - 1
    best_key = None
    for b2 in range(0, 2 * (h - 1) + 1):
        mismatches = 0
        paired = 0
        in_bounds = 0
        for r in range(h):
            r2 = b2 - r
            if not (0 <= r2 < h):
                continue
            for c in np.where(known[r])[0]:
                in_bounds += 1
                if known[r2, c]:
                    paired += 1
                    if int(sprite[r, c]) != int(sprite[r2, c]):
                        mismatches += 1
        key = (mismatches, -paired, -in_bounds, abs(b2 - (h - 1)))
        if best_key is None or key < best_key:
            best_key = key
            best_b2 = b2
    return int(best_b2)


def _choose_fill_color(v: int, h: int, d: int) -> int:
    v_nonblack = v != BLACK
    h_nonblack = h != BLACK
    d_nonblack = d != BLACK

    if v_nonblack and h_nonblack:
        if v == h:
            return int(v)
        if d_nonblack:
            return int(d)
        return int(v)
    if v_nonblack:
        return int(v)
    if h_nonblack:
        return int(h)
    if d_nonblack:
        return int(d)
    return BLACK


def _repair_sprite(sprite: np.ndarray, a2: int, b2: int) -> np.ndarray:
    out = np.array(sprite, dtype=int, copy=True)
    h, w = out.shape
    for _ in range(max(h, w)):
        updates: list[tuple[int, int, int]] = []
        for r, c in np.argwhere(out == BLACK):
            r = int(r)
            c = int(c)
            c2 = a2 - c
            r2 = b2 - r
            v = int(out[r, c2]) if 0 <= c2 < w else BLACK
            hcol = int(out[r2, c]) if 0 <= r2 < h else BLACK
            d = int(out[r2, c2]) if 0 <= r2 < h and 0 <= c2 < w else BLACK
            color = _choose_fill_color(v, hcol, d)
            if color != BLACK:
                updates.append((r, c, color))
        if not updates:
            break
        for r, c, color in updates:
            out[r, c] = int(color)
    return out


def _stamp(out: np.ndarray, template: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    th, tw = template.shape
    cr = th // 2
    cc = tw // 2
    r0 = anchor_r - cr
    c0 = anchor_c - cc
    for tr in range(th):
        for tc in range(tw):
            color = int(template[tr, tc])
            if color != BLACK:
                out[r0 + tr, c0 + tc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Repair the damaged symmetric top motif, crop it into a template, and stamp it at each lower anchor.

    Input structure:
    - The grid has black background and one solid grey separator row splitting an upper motif area and a lower anchor area.
    - Above the separator is a multicolor pattern whose completed form is symmetric across both its vertical and horizontal center axes, but 2-4 pixels have been erased to black.
    - Below the separator are 2-4 red anchor pixels on black background, with enough room to center the repaired motif on each anchor.

    Rule:
    - Infer the vertical and horizontal symmetry axes from the visible non-black pixels in the upper pattern.
    - Fill erased black cells from their mirrored counterparts, using the two-axis mirror as the tie-breaker when vertical and horizontal mirrors disagree.
    - Crop the repaired upper pattern to its minimal non-black bounding box to obtain the stamp template.
    - In the lower region, place the template so its center cell aligns with every red anchor, copying only the template's non-black pixels, and output only the lower region.

    Key insight:
    - The lower stamps are not given directly: they are defined by first reconstructing the damaged top figure via bilateral symmetry, then reusing that repaired figure as the template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep_rows = np.flatnonzero(np.all(g == GREY, axis=1))
    assert len(sep_rows) == 1, "expected exactly one solid grey separator row"
    sep = int(sep_rows[0])

    top = g[:sep]
    bottom = g[sep + 1 :]
    assert top.size > 0 and bottom.size > 0

    top_box = _bbox(top != BLACK)
    assert top_box is not None, "upper region must contain a visible pattern"
    damaged = _crop_bbox(top, top_box)

    a2 = _infer_vertical_axis(damaged)
    b2 = _infer_horizontal_axis(damaged)
    repaired = _repair_sprite(damaged, a2, b2)

    repaired_box = _bbox(repaired != BLACK)
    assert repaired_box is not None
    template = _crop_bbox(repaired, repaired_box)

    out = np.zeros_like(bottom)
    for anchor_r, anchor_c in np.argwhere(bottom == RED):
        _stamp(out, template, int(anchor_r), int(anchor_c))
    return out


def _orbit_coords(h: int, w: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    coords = {
        (int(r), int(c)),
        (int(r), int(w - 1 - c)),
        (int(h - 1 - r), int(c)),
        (int(h - 1 - r), int(w - 1 - c)),
    }
    return tuple(sorted(coords))


def _make_symmetric_template(h: int, w: int) -> np.ndarray | None:
    center_r = h // 2
    center_c = w // 2
    palette = random.sample(TEMPLATE_COLORS, random.randint(2, 4))
    template = np.zeros((h, w), dtype=int)

    forced_orbits = {
        _orbit_coords(h, w, 0, 0),
        _orbit_coords(h, w, center_r, center_c),
        _orbit_coords(h, w, 0, center_c),
        _orbit_coords(h, w, center_r, 0),
    }

    seen: set[tuple[tuple[int, int], ...]] = set()
    for r in range((h + 1) // 2):
        for c in range((w + 1) // 2):
            orbit = _orbit_coords(h, w, r, c)
            if orbit in seen:
                continue
            seen.add(orbit)
            if orbit in forced_orbits:
                color = random.choice(palette)
            else:
                if random.random() < 0.35:
                    color = BLACK
                else:
                    color = random.choice(palette)
            for rr, cc in orbit:
                template[rr, cc] = int(color)

    template[center_r, center_c] = random.choice(palette)

    colors = {int(v) for v in np.unique(template) if int(v) != BLACK}
    if len(colors) < 2:
        return None
    filled = int(np.count_nonzero(template))
    if not (h * w // 3 <= filled <= (3 * h * w) // 4):
        return None
    if _bbox(template != BLACK) != (0, 0, h - 1, w - 1):
        return None
    if _infer_vertical_axis(template) != w - 1:
        return None
    if _infer_horizontal_axis(template) != h - 1:
        return None
    return template


def _general_orbits(template: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    h, w = template.shape
    center_r = h // 2
    center_c = w // 2
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(1, center_r + 1):
        if r == center_r:
            continue
        for c in range(1, center_c + 1):
            if c == center_c:
                continue
            orbit = _orbit_coords(h, w, r, c)
            if len(orbit) != 4:
                continue
            if all(int(template[rr, cc]) != BLACK for rr, cc in orbit):
                orbits.append(orbit)
    return orbits


def _axis_cells(template: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = template.shape
    center_r = h // 2
    center_c = w // 2
    v_cells = []
    for r in range(1, h - 1):
        if r == center_r:
            continue
        if int(template[r, center_c]) != BLACK and int(template[h - 1 - r, center_c]) != BLACK:
            v_cells.append((r, center_c))

    h_cells = []
    for c in range(1, w - 1):
        if c == center_c:
            continue
        if int(template[center_r, c]) != BLACK and int(template[center_r, w - 1 - c]) != BLACK:
            h_cells.append((center_r, c))
    return v_cells, h_cells


def _pick_holes(template: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = template.shape
    center_r = h // 2
    center_c = w // 2
    general = _general_orbits(template)
    v_cells, h_cells = _axis_cells(template)

    strategies: list[str] = []
    if v_cells and h_cells:
        strategies.extend(["axis_pair_2", "axis_pair_3"])
    if len(general) >= 2:
        strategies.append("pair_combo_4")
    if not strategies:
        return None

    random.shuffle(strategies)
    for strategy in strategies:
        if strategy == "axis_pair_2":
            vcell = random.choice(v_cells)
            h_choices = [cell for cell in h_cells if cell != (center_r, center_c)]
            if not h_choices:
                continue
            hcell = random.choice(h_choices)
            return [vcell, hcell]

        if strategy == "axis_pair_3":
            if not general:
                continue
            vcell = random.choice(v_cells)
            hcell = random.choice(h_cells)
            extra_orbit = random.choice(general)
            extra = random.choice(extra_orbit)
            if extra in {vcell, hcell}:
                continue
            return [vcell, hcell, extra]

        if strategy == "pair_combo_4":
            orbit1, orbit2 = random.sample(general, 2)
            o1 = sorted(orbit1)
            o2 = sorted(orbit2)
            r1 = min(rr for rr, _ in o1)
            c1 = min(cc for _, cc in o1)
            r2 = min(rr for rr, _ in o2)
            c2 = min(cc for _, cc in o2)
            holes = [(r1, c1), (r1, w - 1 - c1), (r2, c2), (h - 1 - r2, c2)]
            if len(set(holes)) == 4:
                return holes
    return None


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _choose_anchors(bottom_h: int, bottom_w: int, template_h: int, template_w: int, count: int) -> list[tuple[int, int]] | None:
    cr = template_h // 2
    cc = template_w // 2
    candidates = [
        (r, c)
        for r in range(cr, bottom_h - cr)
        for c in range(cc, bottom_w - cc)
    ]
    random.shuffle(candidates)

    anchors: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for r, c in candidates:
        box = (r - cr, c - cc, r + cr, c + cc)
        if all(not _boxes_overlap(box, other, gap=1) for other in boxes):
            anchors.append((r, c))
            boxes.append(box)
            if len(anchors) == count:
                return anchors
    return None


def _expected_output(template: np.ndarray, bottom_shape: tuple[int, int], anchors: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(bottom_shape, dtype=int)
    for r, c in anchors:
        _stamp(out, template, int(r), int(c))
    return out


def _build_example() -> np.ndarray | None:
    template_h = random.choice((5, 7))
    template_w = random.choice((5, 7))
    template = _make_symmetric_template(template_h, template_w)
    if template is None:
        return None

    holes = _pick_holes(template)
    if holes is None:
        return None
    damaged = template.copy()
    for r, c in holes:
        damaged[r, c] = BLACK

    if np.array_equal(damaged, template):
        return None

    top_h = template_h + random.randint(2, 5)
    left_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)

    width = max(template_w + left_margin + right_margin, random.randint(13, 22))
    bottom_h = random.randint(max(template_h + 2, 9), 16)
    if top_h + 1 + bottom_h > 30 or width > 30:
        return None

    if width - template_w - left_margin < 1:
        return None
    template_c0 = random.randint(left_margin, width - template_w - right_margin)
    template_r0 = random.randint(1, top_h - template_h - 1)

    max_anchors = 4
    anchor_count = random.randint(2, max_anchors)
    anchors = _choose_anchors(bottom_h, width, template_h, template_w, anchor_count)
    if anchors is None:
        return None

    top = np.zeros((top_h, width), dtype=int)
    top[template_r0 : template_r0 + template_h, template_c0 : template_c0 + template_w] = damaged
    bottom = np.zeros((bottom_h, width), dtype=int)
    for r, c in anchors:
        bottom[r, c] = RED
    separator = np.full((1, width), GREY, dtype=int)
    inp = np.vstack([top, separator, bottom])

    expected = _expected_output(template, bottom.shape, anchors)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    if _infer_vertical_axis(damaged) != template_w - 1:
        return None
    if _infer_horizontal_axis(damaged) != template_h - 1:
        return None
    if len({tuple(row) for row in expected}) == 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _build_example()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
