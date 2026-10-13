from __future__ import annotations

import random
from typing import Any

import numpy as np


BG = 0
RECT_COLORS = (1, 2, 3, 4)
RULER_COLOR = 5
PATTERN_COLORS = (6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_pattern_cell(value: int) -> bool:
    return int(value) not in {BG, RULER_COLOR}


def _extract_pattern_from_top(g: np.ndarray, c0: int) -> np.ndarray:
    h, w = g.shape
    assert h >= 3 and 0 <= c0 < w
    assert _is_pattern_cell(g[1, c0])

    width = 0
    while c0 + width < w and _is_pattern_cell(g[1, c0 + width]):
        width += 1
    height = 0
    while 1 + height < h and np.all([_is_pattern_cell(v) for v in g[1 + height, c0 : c0 + width]]):
        height += 1

    pattern = g[1 : 1 + height, c0 : c0 + width]
    assert pattern.size > 0
    assert np.all((pattern != BG) & (pattern != RULER_COLOR))
    return pattern.copy()


def _extract_pattern_from_right(g: np.ndarray, r0: int) -> np.ndarray:
    h, w = g.shape
    assert w >= 3 and 0 <= r0 < h
    assert _is_pattern_cell(g[r0, w - 2])

    width = 0
    while w - 2 - width >= 0 and _is_pattern_cell(g[r0, w - 2 - width]):
        width += 1
    c_left = w - 1 - width

    height = 0
    while r0 + height < h and np.all([_is_pattern_cell(v) for v in g[r0 + height, c_left : w - 1]]):
        height += 1

    pattern = g[r0 : r0 + height, c_left : w - 1]
    assert pattern.size > 0
    assert np.all((pattern != BG) & (pattern != RULER_COLOR))
    return pattern.copy()


def _detect_rulers(g: np.ndarray) -> list[dict[str, Any]]:
    h, w = g.shape
    rulers = []
    for coords in _find_components(g == RULER_COLOR):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        if r0 == r1 == 0:
            length = c1 - c0 + 1
            pattern = _extract_pattern_from_top(g, c0)
            rulers.append({"side": "top", "length": int(length), "pattern": pattern})
        elif c0 == c1 == w - 1:
            length = r1 - r0 + 1
            pattern = _extract_pattern_from_right(g, r0)
            rulers.append({"side": "right", "length": int(length), "pattern": pattern})
        else:
            raise AssertionError(f"unexpected grey component bbox: {(r0, c0, r1, c1)}")
    lengths = [r["length"] for r in rulers]
    assert len(lengths) == len(set(lengths)), "ruler lengths must be unique"
    return rulers


def _detect_rectangles(g: np.ndarray) -> list[dict[str, Any]]:
    rects = []
    for color in RECT_COLORS:
        coords = np.argwhere(g == int(color))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        assert not np.any(region == BG), "rectangle footprint must contain no background"
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        rects.append(
            {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "height": int(h),
                "width": int(w),
                "area": int(h * w),
                "long_len": int(max(h, w)),
                "occluded": bool(np.any(region != int(color))),
            }
        )
    assert 3 <= len(rects) <= 4, "expected 3-4 rectangles"
    return rects


def _tile_crop(pattern: np.ndarray, h: int, w: int) -> np.ndarray:
    if h <= 0 or w <= 0:
        return np.zeros((max(h, 0), max(w, 0)), dtype=int)
    ph, pw = pattern.shape
    reps_h = (h + ph - 1) // ph
    reps_w = (w + pw - 1) // pw
    return np.tile(pattern, (reps_h, reps_w))[:h, :w]


def _render_output(shape: tuple[int, int], rects: list[dict[str, Any]], ruler_map: dict[int, np.ndarray]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rect in sorted(rects, key=lambda item: item["area"], reverse=True):
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])
        if rect["long_len"] not in ruler_map:
            continue
        if rect["height"] <= 2 or rect["width"] <= 2:
            continue
        tile = _tile_crop(ruler_map[rect["long_len"]], rect["height"] - 2, rect["width"] - 2)
        out[r0 + 1 : r1, c0 + 1 : c1] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each occluded rectangle, match its restored long side to a ruler, and tile the ruler's motif inside it.

    Input structure:
    - Black background, with 3-4 solid rectangles colored from blue/red/green/yellow.
    - Some rectangles are partially hidden by smaller rectangles, so a color's bbox has no background but may contain other foreground colors.
    - Along the top edge or right edge sit 2-3 grey ruler bars, each with a small colored rectangular motif attached at one end.

    Rule:
    - Detect each rectangle color by its visible bbox.
    - For every rectangle color, restore its full rectangular footprint; when restored footprints overlap, smaller rectangles stay on top of larger ones.
    - Measure each restored rectangle's longer side, pair it with the unique ruler of the same length, and tile that ruler motif across the rectangle interior while keeping a one-cell border of the rectangle color.
    - Rectangles without a matching ruler stay solid, and all rulers plus motifs are erased.

    Key insight:
    - The ruler match depends on the rectangle's true long side, which is only recoverable after undoing the occlusion inside each color bbox.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rulers = _detect_rulers(g)
    rects = _detect_rectangles(g)
    ruler_map = {r["length"]: r["pattern"] for r in rulers}

    occluded = [rect for rect in rects if rect["occluded"]]
    assert len(occluded) == len(rulers), "each ruler should match one occluded rectangle"

    matched = [rect for rect in rects if rect["long_len"] in ruler_map]
    assert len(matched) == len(rulers), "ruler pairing must be unique"
    return _render_output(g.shape, rects, ruler_map)


def _random_pattern() -> np.ndarray:
    h = 2
    w = random.choice([2, 3])
    a, b = random.sample(PATTERN_COLORS, 2)
    kind = random.choice(["checker", "vstripe", "hstripe", "random"])
    pat = np.full((h, w), a, dtype=int)
    if kind == "checker":
        for r in range(h):
            for c in range(w):
                pat[r, c] = a if (r + c) % 2 == 0 else b
    elif kind == "vstripe":
        for c in range(w):
            pat[:, c] = a if c % 2 == 0 else b
    elif kind == "hstripe":
        for r in range(h):
            pat[r, :] = a if r % 2 == 0 else b
    else:
        pat = np.array([[random.choice([a, b]) for _ in range(w)] for _ in range(h)], dtype=int)
        if not np.any(pat == a) or not np.any(pat == b):
            pat[0, 0] = a
            pat[-1, -1] = b
    return pat


def _make_nested_boxes(n_rect: int, max_h: int, max_w: int) -> list[tuple[int, int, int, int]] | None:
    min_inner = 4
    for _ in range(200):
        inner_h = random.randint(min_inner, 6)
        inner_w = random.randint(min_inner, 6)
        boxes = [(0, 0, inner_h - 1, inner_w - 1)]
        top = bottom = left = right = 0
        cur_h = inner_h
        cur_w = inner_w
        for _level in range(1, n_rect):
            dt = random.randint(1, 2)
            db = random.randint(1, 2)
            dl = random.randint(1, 2)
            dr = random.randint(1, 2)
            top += dt
            bottom += db
            left += dl
            right += dr
            cur_h += dt + db
            cur_w += dl + dr
            boxes.append((-top, -left, bottom + inner_h - 1, right + inner_w - 1))

        outer = boxes[-1]
        outer_h = outer[2] - outer[0] + 1
        outer_w = outer[3] - outer[1] + 1
        if outer_h > max_h or outer_w > max_w:
            continue

        norm_boxes = []
        shift_r = -outer[0]
        shift_c = -outer[1]
        for r0, c0, r1, c1 in reversed(boxes):
            norm_boxes.append((r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c))

        longs = [max(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in norm_boxes]
        if len(set(longs)) != len(longs):
            continue
        return norm_boxes
    return None


def _translate_boxes(
    boxes: list[tuple[int, int, int, int]], scene_r0: int, scene_r1: int, scene_c0: int, scene_c1: int
) -> list[tuple[int, int, int, int]] | None:
    outer = boxes[0]
    outer_h = outer[2] - outer[0] + 1
    outer_w = outer[3] - outer[1] + 1
    max_r0 = scene_r1 - outer_h + 1
    max_c0 = scene_c1 - outer_w + 1
    if max_r0 < scene_r0 or max_c0 < scene_c0:
        return None
    dr = random.randint(scene_r0, max_r0)
    dc = random.randint(scene_c0, max_c0)
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in boxes]


def _orientation_choices(count: int) -> list[str]:
    if count == 2:
        options = [["top", "top"], ["right", "right"], ["top", "right"]]
        weights = [1, 1, 2]
    else:
        options = [
            ["top", "top", "right"],
            ["top", "right", "right"],
            ["top", "top", "top"],
            ["right", "right", "right"],
        ]
        weights = [2, 2, 1, 1]
    return random.choices(options, weights=weights, k=1)[0][:]


def _interval_disjoint(existing: list[tuple[int, int]], start: int, end: int, gap: int = 1) -> bool:
    for a, b in existing:
        if not (end + gap < a or b + gap < start):
            return False
    return True


def _place_rulers(h: int, w: int, lengths: list[int]) -> list[dict[str, Any]] | None:
    for _ in range(200):
        sides = _orientation_choices(len(lengths))
        random.shuffle(sides)
        specs = [{"length": int(length), "side": side} for length, side in zip(lengths, sides)]
        has_top = any(spec["side"] == "top" for spec in specs)
        has_right = any(spec["side"] == "right" for spec in specs)
        top_used: list[tuple[int, int]] = []
        right_used: list[tuple[int, int]] = []
        placed = []
        ok = True
        for spec in sorted(specs, key=lambda item: item["length"], reverse=True):
            length = spec["length"]
            pattern = _random_pattern()
            if spec["side"] == "top":
                end_cap = w - 3 if has_right else w - 1
                starts = [
                    c0
                    for c0 in range(0, end_cap - length + 2)
                    if _interval_disjoint(top_used, c0, c0 + length - 1)
                ]
                if not starts:
                    ok = False
                    break
                c0 = random.choice(starts)
                top_used.append((c0, c0 + length - 1))
                placed.append({"side": "top", "length": length, "c0": c0, "pattern": pattern})
            else:
                start_min = 3 if has_top else 0
                starts = [
                    r0
                    for r0 in range(start_min, h - length + 1)
                    if _interval_disjoint(right_used, r0, r0 + length - 1)
                ]
                if not starts:
                    ok = False
                    break
                r0 = random.choice(starts)
                right_used.append((r0, r0 + length - 1))
                placed.append({"side": "right", "length": length, "r0": r0, "pattern": pattern})
        if ok:
            return placed
    return None


def _paint_input_grid(shape: tuple[int, int], rects: list[dict[str, Any]], rulers: list[dict[str, Any]]) -> np.ndarray:
    h, w = shape
    g = np.zeros((h, w), dtype=int)
    for rect in sorted(rects, key=lambda item: item["area"], reverse=True):
        r0, c0, r1, c1 = rect["bbox"]
        g[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])
    for ruler in rulers:
        if ruler["side"] == "top":
            c0 = ruler["c0"]
            g[0, c0 : c0 + ruler["length"]] = RULER_COLOR
            pat = ruler["pattern"]
            ph, pw = pat.shape
            g[1 : 1 + ph, c0 : c0 + pw] = pat
        else:
            r0 = ruler["r0"]
            g[r0 : r0 + ruler["length"], w - 1] = RULER_COLOR
            pat = ruler["pattern"]
            ph, pw = pat.shape
            g[r0 : r0 + ph, w - 1 - pw : w - 1] = pat
    return g


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(20, 26)
    w = random.randint(20, 26)
    n_rect = random.choice([3, 4])
    scene_r0, scene_r1 = 4, h - 3
    scene_c0, scene_c1 = 1, w - 6
    max_scene_h = scene_r1 - scene_r0 + 1
    max_scene_w = scene_c1 - scene_c0 + 1
    boxes = _make_nested_boxes(n_rect, max_scene_h, max_scene_w)
    if boxes is None:
        return None
    boxes = _translate_boxes(boxes, scene_r0, scene_r1, scene_c0, scene_c1)
    if boxes is None:
        return None

    colors = random.sample(list(RECT_COLORS), n_rect)
    rects = []
    for color, bbox in zip(colors, boxes):
        r0, c0, r1, c1 = bbox
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        rects.append(
            {
                "color": int(color),
                "bbox": bbox,
                "height": int(hh),
                "width": int(ww),
                "area": int(hh * ww),
                "long_len": int(max(hh, ww)),
                "occluded": False,
            }
        )

    matched_rects = rects[:-1]
    unmatched_rect = rects[-1]
    for rect in matched_rects:
        rect["occluded"] = True

    matched_lengths = [rect["long_len"] for rect in matched_rects]
    if len(set(matched_lengths)) != len(matched_lengths):
        return None
    if unmatched_rect["long_len"] in matched_lengths:
        return None

    rulers = _place_rulers(h, w, matched_lengths)
    if rulers is None:
        return None

    inp = _paint_input_grid((h, w), rects, rulers)
    ruler_map = {ruler["length"]: ruler["pattern"] for ruler in rulers}
    out = _render_output((h, w), rects, ruler_map)

    detected_rects = _detect_rectangles(inp)
    detected_rulers = _detect_rulers(inp)
    if len([rect for rect in detected_rects if rect["occluded"]]) != len(detected_rulers):
        return None
    if np.any(out == RULER_COLOR):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _build_case()
        if case is None:
            continue
        inp, out = case
        if not np.array_equal(main(inp), out):
            continue
        return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
