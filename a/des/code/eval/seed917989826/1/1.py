from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRS = ("up", "down", "left", "right")

# Generator tuning.
GRID_H_RANGE = (24, 32)
GRID_W_RANGE = (24, 32)
FRAME_COUNT_OPTIONS = (2, 3, 4, 5)
FRAME_INNER_H_RANGE = (4, 7)
FRAME_INNER_W_RANGE = (4, 7)
PATTERN_BBOX_MIN = 2
PATTERN_BBOX_MAX = 5
PATTERN_COUNT_RANGE = (3, 12)
CANVAS_SLACK_RANGE = (0, 2)
PLACEMENT_TRIES = 200
GEN_RETRIES = 400


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 8:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 <= br1 and br0 <= ar1 and ac0 <= bc1 and bc0 <= ac1


def _expand_rect(rect: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _crop_pattern(cells: list[tuple[int, int]], color: int, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = int(color)
    return patch, (r0, c0, r1, c1)


def _pattern_push_top_left(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    pr0, pc0, pr1, pc1 = frame["pattern_bbox"]
    ph = pr1 - pr0 + 1
    pw = pc1 - pc0 + 1
    direction = frame["direction"]
    if direction == "up":
        return r0 - ph, pc0
    if direction == "down":
        return r1 + 1, pc0
    if direction == "left":
        return pr0, c0 - pw
    return pr0, c1 + 1


def _frame_union_rect(frame_bbox: tuple[int, int, int, int], push_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    fr0, fc0, fr1, fc1 = frame_bbox
    pr0, pc0, pr1, pc1 = push_bbox
    return min(fr0, pr0), min(fc0, pc0), max(fr1, pr1), max(fc1, pc1)


def _pack_pattern_crops(
    canvas_shape: tuple[int, int],
    items: list[dict],
    bg: int,
    canvas_color: int,
) -> np.ndarray:
    ch, cw = canvas_shape
    out = np.full((ch, cw), int(canvas_color), dtype=int)
    cursor_r = 0
    cursor_c = 0
    row_h = 0
    for item in items:
        patch = item["pattern_patch"]
        ph, pw = patch.shape
        if cursor_c > 0 and cursor_c + pw > cw:
            cursor_r += row_h
            cursor_c = 0
            row_h = 0
        assert cursor_r + ph <= ch and cursor_c + pw <= cw
        view = out[cursor_r : cursor_r + ph, cursor_c : cursor_c + pw]
        mask = patch != bg
        view[mask] = patch[mask]
        cursor_c += pw
        row_h = max(row_h, ph)
    return out


def _parse_scene(grid: np.ndarray) -> tuple[int, dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    h, w = g.shape

    frame_infos: list[dict] = []
    solid_candidates: list[dict] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for comp in _components(g == int(color), connectivity=4):
            r0, c0, r1, c1 = _bbox(comp)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            comp_set = set(comp)

            if len(comp) == hh * ww:
                solid_candidates.append(
                    {
                        "color": int(color),
                        "bbox": (r0, c0, r1, c1),
                        "area": hh * ww,
                    }
                )

            if hh < 3 or ww < 3:
                continue

            border = _border_cells(r0, c0, r1, c1)
            if not comp_set.issubset(border):
                continue
            missing = list(border - comp_set)
            if len(missing) != 1:
                continue
            gap_r, gap_c = missing[0]
            if gap_r in (r0, r1) and gap_c in (c0, c1):
                continue

            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            inner_mask = (inner != bg) & (inner != int(color))
            if not bool(inner_mask.any()):
                continue
            inner_comps = _components(inner_mask, connectivity=4)
            if len(inner_comps) != 1:
                continue

            pattern_local = inner_comps[0]
            pattern_cells = [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in pattern_local]
            pattern_colors = {int(g[r, c]) for r, c in pattern_cells}
            if len(pattern_colors) != 1:
                continue
            pattern_color = pattern_colors.pop()
            patch, pattern_bbox = _crop_pattern(pattern_cells, pattern_color, bg)

            if gap_r == r0:
                direction = "up"
            elif gap_r == r1:
                direction = "down"
            elif gap_c == c0:
                direction = "left"
            else:
                direction = "right"

            frame_infos.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "gap": (gap_r, gap_c),
                    "direction": direction,
                    "pattern_color": pattern_color,
                    "pattern_cells": pattern_cells,
                    "pattern_patch": patch,
                    "pattern_bbox": pattern_bbox,
                    "pattern_count": len(pattern_cells),
                }
            )

    assert frame_infos, "expected at least one gapped frame"

    max_frame_area = max((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in (f["bbox"] for f in frame_infos))
    canvas_candidates = [cand for cand in solid_candidates if cand["area"] > max_frame_area]
    assert canvas_candidates, "expected one solid canvas rectangle"
    canvas_candidates.sort(key=lambda item: (item["area"], item["bbox"][0], item["bbox"][1]), reverse=True)
    canvas = canvas_candidates[0]

    # Reject any frame whose bbox sits inside the chosen canvas.
    cr0, cc0, cr1, cc1 = canvas["bbox"]
    frame_infos = [
        frame
        for frame in frame_infos
        if not (
            cr0 <= frame["bbox"][0] <= frame["bbox"][2] <= cr1
            and cc0 <= frame["bbox"][1] <= frame["bbox"][3] <= cc1
        )
    ]
    assert frame_infos, "canvas detection swallowed all frames"

    return bg, canvas, frame_infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each interior pattern out through its frame gap, then copy all patterns
    onto the solid canvas ordered by increasing pixel count.
    """
    g = np.asarray(grid, dtype=int)
    bg, canvas, frames = _parse_scene(g)
    out = g.copy()

    cr0, cc0, cr1, cc1 = canvas["bbox"]
    canvas_color = int(canvas["color"])
    out[cr0 : cr1 + 1, cc0 : cc1 + 1] = canvas_color

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        patch = frame["pattern_patch"]
        ph, pw = patch.shape
        tr, tc = _pattern_push_top_left(frame)
        view = out[tr : tr + ph, tc : tc + pw]
        mask = patch != bg
        view[mask] = patch[mask]

    ordered = sorted(frames, key=lambda item: (item["pattern_count"], item["bbox"][0], item["bbox"][1]))
    packed = _pack_pattern_crops((cr1 - cr0 + 1, cc1 - cc0 + 1), ordered, bg, canvas_color)
    out[cr0 : cr1 + 1, cc0 : cc1 + 1] = packed
    return out


def _random_connected_mask(h: int, w: int, count: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(120):
        start = random.choice(cells)
        picked = {start}
        frontier = set()
        sr, sc = start
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = sr + dr, sc + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))
        while len(picked) < count and frontier:
            cell = random.choice(list(frontier))
            frontier.remove(cell)
            picked.add(cell)
            cr, cc = cell
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in picked:
                    frontier.add((nr, nc))
        if len(picked) != count:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in picked:
            mask[r, c] = True
        coords = list(picked)
        r0, c0, r1, c1 = _bbox(coords)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
        if cropped.shape[0] <= PATTERN_BBOX_MAX and cropped.shape[1] <= PATTERN_BBOX_MAX:
            return cropped
    return None


def _build_pattern_spec(count: int, color: int, bg: int) -> dict | None:
    dims = [
        (h, w)
        for h in range(PATTERN_BBOX_MIN, PATTERN_BBOX_MAX + 1)
        for w in range(PATTERN_BBOX_MIN, PATTERN_BBOX_MAX + 1)
        if h * w >= count
    ]
    random.shuffle(dims)
    for h, w in dims:
        mask = _random_connected_mask(h, w, count)
        if mask is None:
            continue
        ph, pw = mask.shape
        patch = np.full((ph, pw), int(bg), dtype=int)
        patch[mask] = int(color)
        return {
            "count": int(count),
            "color": int(color),
            "patch": patch,
            "height": ph,
            "width": pw,
        }
    return None


def _choose_counts(n: int) -> list[int]:
    values = list(range(PATTERN_COUNT_RANGE[0], PATTERN_COUNT_RANGE[1] + 1))
    random.shuffle(values)
    return sorted(values[:n])


def _balanced_directions(n: int) -> list[str]:
    directions: list[str] = []
    while len(directions) < n:
        chunk = list(DIRS)
        random.shuffle(chunk)
        directions.extend(chunk)
    return directions[:n]


def _choose_canvas_layout(patterns: list[dict], min_area: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    ordered = sorted(patterns, key=lambda item: item["count"])
    max_w = max(item["width"] for item in ordered)
    sum_w = sum(item["width"] for item in ordered)
    candidates = []
    for width in range(max_w, sum_w + 1):
        cursor_r = 0
        cursor_c = 0
        row_h = 0
        positions: list[tuple[int, int]] = []
        ok = True
        for item in ordered:
            ph = item["height"]
            pw = item["width"]
            if cursor_c > 0 and cursor_c + pw > width:
                cursor_r += row_h
                cursor_c = 0
                row_h = 0
            if pw > width:
                ok = False
                break
            positions.append((cursor_r, cursor_c))
            cursor_c += pw
            row_h = max(row_h, ph)
        if not ok:
            continue
        used_h = cursor_r + row_h
        wraps = used_h > max(item["height"] for item in ordered)
        area = width * used_h
        if area < min_area:
            continue
        candidates.append((width, used_h, positions, wraps))
    if not candidates:
        return None

    want_wrap = random.choice([False, True])
    filtered = [cand for cand in candidates if cand[3] == want_wrap]
    if filtered:
        candidates = filtered
    width, used_h, positions, _wraps = random.choice(candidates)
    width += random.randint(*CANVAS_SLACK_RANGE)
    height = used_h + random.randint(*CANVAS_SLACK_RANGE)
    return height, width, positions


def _paint_patch(dst: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    view = dst[top : top + h, left : left + w]
    mask = patch != bg
    view[mask] = patch[mask]


def _make_frame_spec(pattern: dict, direction: str) -> dict:
    inner_h = random.randint(max(FRAME_INNER_H_RANGE[0], pattern["height"]), max(FRAME_INNER_H_RANGE[1], pattern["height"]))
    inner_w = random.randint(max(FRAME_INNER_W_RANGE[0], pattern["width"]), max(FRAME_INNER_W_RANGE[1], pattern["width"]))
    off_r = random.randint(0, inner_h - pattern["height"])
    off_c = random.randint(0, inner_w - pattern["width"])
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    if direction in ("up", "down"):
        gap_pos = random.randint(1, frame_w - 2)
        gap = (0, gap_pos) if direction == "up" else (frame_h - 1, gap_pos)
    else:
        gap_pos = random.randint(1, frame_h - 2)
        gap = (gap_pos, 0) if direction == "left" else (gap_pos, frame_w - 1)

    pattern_bbox_rel = (1 + off_r, 1 + off_c, 1 + off_r + pattern["height"] - 1, 1 + off_c + pattern["width"] - 1)
    pr0, pc0, pr1, pc1 = pattern_bbox_rel
    if direction == "up":
        push_bbox_rel = (-pattern["height"], pc0, -1, pc1)
    elif direction == "down":
        push_bbox_rel = (frame_h, pc0, frame_h + pattern["height"] - 1, pc1)
    elif direction == "left":
        push_bbox_rel = (pr0, -pattern["width"], pr1, -1)
    else:
        push_bbox_rel = (pr0, frame_w, pr1, frame_w + pattern["width"] - 1)

    union_rel = _frame_union_rect((0, 0, frame_h - 1, frame_w - 1), push_bbox_rel)
    return {
        "direction": direction,
        "frame_h": frame_h,
        "frame_w": frame_w,
        "gap_rel": gap,
        "pattern": pattern,
        "pattern_bbox_rel": pattern_bbox_rel,
        "push_bbox_rel": push_bbox_rel,
        "union_rel": union_rel,
    }


def _place_rect(
    grid_h: int,
    grid_w: int,
    rel_rect: tuple[int, int, int, int],
    reserved: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    rr0, rc0, rr1, rc1 = rel_rect
    min_r = max(1 - rr0, 0 - rr0)
    max_r = min(grid_h - 2 - rr1, grid_h - 1 - rr1)
    min_c = max(1 - rc0, 0 - rc0)
    max_c = min(grid_w - 2 - rc1, grid_w - 1 - rc1)
    if min_r > max_r or min_c > max_c:
        return None

    for _ in range(PLACEMENT_TRIES):
        r0 = random.randint(min_r, max_r)
        c0 = random.randint(min_c, max_c)
        abs_rect = (r0 + rr0, c0 + rc0, r0 + rr1, c0 + rc1)
        if any(_rect_overlap(_expand_rect(abs_rect, 1), rect) for rect in reserved):
            continue
        return r0, c0
    return None


def _render_input_from_spec(
    grid_h: int,
    grid_w: int,
    bg: int,
    frame_color: int,
    canvas_color: int,
    canvas_bbox: tuple[int, int, int, int],
    placed_frames: list[dict],
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    cr0, cc0, cr1, cc1 = canvas_bbox
    grid[cr0 : cr1 + 1, cc0 : cc1 + 1] = int(canvas_color)

    for frame in placed_frames:
        r0, c0, r1, c1 = frame["bbox"]
        gap_r, gap_c = frame["gap"]
        grid[r0, c0 : c1 + 1] = int(frame_color)
        grid[r1, c0 : c1 + 1] = int(frame_color)
        grid[r0 : r1 + 1, c0] = int(frame_color)
        grid[r0 : r1 + 1, c1] = int(frame_color)
        grid[gap_r, gap_c] = int(bg)
        pr0, pc0, _pr1, _pc1 = frame["pattern_bbox"]
        _paint_patch(grid, frame["pattern_patch"], pr0, pc0, bg)

    return grid


def _render_expected_from_spec(
    inp: np.ndarray,
    bg: int,
    canvas_color: int,
    canvas_bbox: tuple[int, int, int, int],
    placed_frames: list[dict],
) -> np.ndarray:
    out = inp.copy()
    cr0, cc0, cr1, cc1 = canvas_bbox
    out[cr0 : cr1 + 1, cc0 : cc1 + 1] = int(canvas_color)

    for frame in placed_frames:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(bg)
        tr, tc = _pattern_push_top_left(frame)
        _paint_patch(out, frame["pattern_patch"], tr, tc, bg)

    ordered = sorted(placed_frames, key=lambda item: (item["pattern_count"], item["bbox"][0], item["bbox"][1]))
    packed = _pack_pattern_crops((cr1 - cr0 + 1, cc1 - cc0 + 1), ordered, bg, canvas_color)
    out[cr0 : cr1 + 1, cc0 : cc1 + 1] = packed
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    n_frames = random.choice(FRAME_COUNT_OPTIONS)
    palette = random.sample(list(range(1, 10)), 2 + n_frames)
    frame_color = palette[0]
    canvas_color = palette[1]
    pattern_colors = palette[2:]

    counts = _choose_counts(n_frames)
    directions = _balanced_directions(n_frames)
    patterns = []
    for count, color in zip(counts, pattern_colors, strict=False):
        pattern = _build_pattern_spec(count, color, bg)
        if pattern is None:
            return None
        patterns.append(pattern)

    frame_specs = [_make_frame_spec(pattern, direction) for pattern, direction in zip(patterns, directions, strict=False)]

    max_frame_area = max(spec["frame_h"] * spec["frame_w"] for spec in frame_specs)
    canvas_layout = _choose_canvas_layout(patterns, min_area=max_frame_area + 1)
    if canvas_layout is None:
        return None
    canvas_h, canvas_w, _positions = canvas_layout

    min_h = max(GRID_H_RANGE[0], canvas_h + 4, max(spec["frame_h"] + pattern["height"] + 3 for spec, pattern in zip(frame_specs, patterns, strict=False)))
    min_w = max(GRID_W_RANGE[0], canvas_w + 4, max(spec["frame_w"] + pattern["width"] + 3 for spec, pattern in zip(frame_specs, patterns, strict=False)))
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None
    grid_h = random.randint(min_h, GRID_H_RANGE[1])
    grid_w = random.randint(min_w, GRID_W_RANGE[1])

    reserved: list[tuple[int, int, int, int]] = []
    canvas_top_left = _place_rect(grid_h, grid_w, (0, 0, canvas_h - 1, canvas_w - 1), reserved)
    if canvas_top_left is None:
        return None
    cr0, cc0 = canvas_top_left
    canvas_bbox = (cr0, cc0, cr0 + canvas_h - 1, cc0 + canvas_w - 1)
    reserved.append(_expand_rect(canvas_bbox, 1))

    placed_frames: list[dict] = []
    shuffled_specs = list(frame_specs)
    random.shuffle(shuffled_specs)
    for spec in shuffled_specs:
        top_left = _place_rect(grid_h, grid_w, spec["union_rel"], reserved)
        if top_left is None:
            return None
        fr0, fc0 = top_left
        frame_bbox = (fr0, fc0, fr0 + spec["frame_h"] - 1, fc0 + spec["frame_w"] - 1)
        gap = (fr0 + spec["gap_rel"][0], fc0 + spec["gap_rel"][1])
        pr0, pc0, pr1, pc1 = spec["pattern_bbox_rel"]
        pattern_bbox = (fr0 + pr0, fc0 + pc0, fr0 + pr1, fc0 + pc1)
        ur0, uc0, ur1, uc1 = spec["union_rel"]
        reserved.append(_expand_rect((fr0 + ur0, fc0 + uc0, fr0 + ur1, fc0 + uc1), 1))
        placed_frames.append(
            {
                "bbox": frame_bbox,
                "gap": gap,
                "direction": spec["direction"],
                "pattern_patch": spec["pattern"]["patch"],
                "pattern_bbox": pattern_bbox,
                "pattern_color": spec["pattern"]["color"],
                "pattern_count": spec["pattern"]["count"],
                "color": frame_color,
            }
        )

    inp = _render_input_from_spec(grid_h, grid_w, bg, frame_color, canvas_color, canvas_bbox, placed_frames)
    expected = _render_expected_from_spec(inp, bg, canvas_color, canvas_bbox, placed_frames)

    if np.count_nonzero(inp == bg) <= (grid_h * grid_w) // 2:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
