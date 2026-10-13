from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


CoordArray = np.ndarray


def _bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _bbox(coords: CoordArray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(grid[sr, sc]) == int(bg):
                continue
            color = int(grid[sr, sc])
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            pts: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(pts, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _side_coords(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    if side == "right":
        return [(r, c1) for r in range(r0, r1 + 1)]
    raise ValueError(side)


def _detect_gap(grid: np.ndarray, bg: int, color: int, bbox: tuple[int, int, int, int]) -> dict | None:
    gaps: list[dict] = []
    for side in ("top", "bottom", "left", "right"):
        coords = _side_coords(bbox, side)
        missing = [i for i, (r, c) in enumerate(coords) if int(grid[r, c]) != int(color)]
        if not missing:
            continue
        if missing != list(range(missing[0], missing[-1] + 1)):
            return None
        if missing[0] == 0 or missing[-1] == len(coords) - 1:
            return None
        if any(int(grid[coords[i][0], coords[i][1]]) != int(bg) for i in missing):
            return None
        start = coords[missing[0]][1] if side in ("top", "bottom") else coords[missing[0]][0]
        end = coords[missing[-1]][1] if side in ("top", "bottom") else coords[missing[-1]][0]
        gaps.append({"side": side, "start": int(start), "end": int(end), "width": len(missing)})
    if len(gaps) != 1:
        return None
    return gaps[0]


def _find_frame(grid: np.ndarray, bg: int, comps: list[dict]) -> tuple[int, dict, dict]:
    candidates: list[tuple[int, int, dict]] = []
    for i, comp in enumerate(comps):
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        gap = _detect_gap(grid, bg, color, comp["bbox"])
        if gap is None:
            continue

        perim = np.zeros(grid.shape, dtype=bool)
        perim[r0, c0 : c1 + 1] = True
        perim[r1, c0 : c1 + 1] = True
        perim[r0 : r1 + 1, c0] = True
        perim[r0 : r1 + 1, c1] = True
        if int(((grid == color) & perim).sum()) != int(comp["area"]):
            continue
        if bool((grid[r0 + 1 : r1, c0 + 1 : c1] == color).any()):
            continue
        candidates.append((int(comp["area"]), i, gap))

    assert candidates, "expected one hollow rectangle frame with one gap"
    candidates.sort(reverse=True)
    area, idx, gap = candidates[0]
    if len(candidates) > 1:
        assert area > candidates[1][0], "frame must be the unique largest hollow rectangle"
    return idx, comps[idx], gap


def _direction(side: str) -> tuple[int, int]:
    return {
        "top": (1, 0),
        "bottom": (-1, 0),
        "left": (0, 1),
        "right": (0, -1),
    }[side]


def _admission(comp: dict, frame_bbox: tuple[int, int, int, int], gap: dict) -> tuple[bool, int]:
    fr0, fc0, fr1, fc1 = frame_bbox
    r0, c0, r1, c1 = comp["bbox"]
    side = gap["side"]
    start, end, gap_width = int(gap["start"]), int(gap["end"]), int(gap["width"])

    if side == "top":
        outside = r1 < fr0
        proj0, proj1 = c0, c1
        distance = fr0 - r1
    elif side == "bottom":
        outside = r0 > fr1
        proj0, proj1 = c0, c1
        distance = r0 - fr1
    elif side == "left":
        outside = c1 < fc0
        proj0, proj1 = r0, r1
        distance = fc0 - c1
    else:
        outside = c0 > fc1
        proj0, proj1 = r0, r1
        distance = c0 - fc1

    width = proj1 - proj0 + 1
    through_gap = start <= proj0 and proj1 <= end
    return bool(outside and width <= gap_width and through_gap), int(distance)


def _inside_frame(coords: CoordArray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(
        (coords[:, 0] > r0).all()
        and (coords[:, 0] < r1).all()
        and (coords[:, 1] > c0).all()
        and (coords[:, 1] < c1).all()
    )


def _slide_until_blocked(coords: CoordArray, direction: tuple[int, int], obstacle: np.ndarray) -> CoordArray:
    h, w = obstacle.shape
    step = np.array(direction, dtype=int)
    cur = coords.copy()
    while True:
        nxt = cur + step
        if not bool((0 <= nxt[:, 0]).all() and (nxt[:, 0] < h).all()):
            return cur
        if not bool((0 <= nxt[:, 1]).all() and (nxt[:, 1] < w).all()):
            return cur
        if bool(obstacle[nxt[:, 0], nxt[:, 1]].any()):
            return cur
        cur = nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide gap-admitted sprites into a hollow frame and stack them inside.

    Input structure:
    - The background is the most common color.
    - A large monochrome hollow rectangle sits near the center, and exactly one
      side has one contiguous background-colored gap.
    - Several monochrome connected sprites lie outside the frame on different
      sides; some align with the gap corridor and some do not.

    Rule:
    - The gap side gives the inward sliding direction, perpendicular to that side.
    - A sprite may enter only if it starts outside that same side, its bounding
      box projection along the gap side fits within the gap width, and that
      projection lies entirely inside the gap interval.
    - Admitted sprites are processed from nearest to farthest from the gap and
      slide inward until their next step would hit the opposite frame wall or an
      already parked sprite. Non-admitted sprites stay in place.
    - The frame pixels themselves are unchanged.

    Key insight:
    - The one missing run in the frame is simultaneously the width filter, the
      path corridor, and the queue entrance that determines the stacking order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg(g)
    comps = _components(g, bg)
    frame_idx, frame, gap = _find_frame(g, bg, comps)
    frame_color = int(frame["color"])
    frame_bbox = frame["bbox"]
    side = gap["side"]
    direction = _direction(side)

    admitted: list[tuple[int, int, dict]] = []
    for i, comp in enumerate(comps):
        if i == frame_idx:
            continue
        ok, distance = _admission(comp, frame_bbox, gap)
        if ok:
            admitted.append((distance, i, comp))

    admitted.sort(key=lambda item: (item[0], item[2]["bbox"]))

    out = g.copy()
    for _distance, _idx, comp in admitted:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    obstacle = g == frame_color
    for _distance, _idx, comp in admitted:
        final = _slide_until_blocked(comp["coords"], direction, obstacle)
        assert _inside_frame(final, frame_bbox), "admitted sprite must come to rest inside the frame"
        color = int(comp["color"])
        out[final[:, 0], final[:, 1]] = color
        obstacle[final[:, 0], final[:, 1]] = True

    return out


# ------------------------- generator -------------------------

SIDES = ("top", "left", "bottom", "right")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
MAX_ATTEMPTS = 400
_GEN_COUNTER = 0


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, gap: dict, bg: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    side, start, end = gap["side"], int(gap["start"]), int(gap["end"])
    if side == "top":
        grid[r0, start : end + 1] = bg
    elif side == "bottom":
        grid[r1, start : end + 1] = bg
    elif side == "left":
        grid[start : end + 1, c0] = bg
    else:
        grid[start : end + 1, c1] = bg


def _rect_for_side(
    side: str,
    frame_bbox: tuple[int, int, int, int],
    distance: int,
    depth: int,
    tangent_start: int,
    tangent_len: int,
) -> tuple[int, int, int, int]:
    fr0, fc0, fr1, fc1 = frame_bbox
    if side == "top":
        return fr0 - distance - depth + 1, tangent_start, depth, tangent_len
    if side == "bottom":
        return fr1 + distance, tangent_start, depth, tangent_len
    if side == "left":
        return tangent_start, fc0 - distance - depth + 1, tangent_len, depth
    if side == "right":
        return tangent_start, fc1 + distance, tangent_len, depth
    raise ValueError(side)


def _can_place(grid: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 0) -> bool:
    rows, cols = grid.shape
    if r < 0 or c < 0 or r + h > rows or c + w > cols:
        return False
    rr0, cc0 = max(0, r - pad), max(0, c - pad)
    rr1, cc1 = min(rows, r + h + pad), min(cols, c + w + pad)
    return bool((grid[rr0:rr1, cc0:cc1] == int(grid[0, 0])).all())


def _paint_rect(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> dict:
    r, c, h, w = rect
    grid[r : r + h, c : c + w] = color
    rr, cc = np.mgrid[r : r + h, c : c + w]
    coords = np.column_stack([rr.reshape(-1), cc.reshape(-1)]).astype(int)
    return {"color": int(color), "rect": rect, "coords": coords}


def _choose_misaligned_interval(
    tan_min: int, tan_max: int, gap_start: int, gap_end: int, gap_width: int
) -> tuple[int, int] | None:
    segments = []
    if gap_start - 1 >= tan_min:
        segments.append((tan_min, gap_start - 1))
    if gap_end + 1 <= tan_max:
        segments.append((gap_end + 1, tan_max))
    segments = [seg for seg in segments if seg[1] >= seg[0]]
    if not segments:
        return None
    lo, hi = random.choice(segments)
    max_len = min(gap_width, hi - lo + 1)
    width = random.randint(1, max_len)
    start = random.randint(lo, hi - width + 1)
    return start, width


def _wide_interval(tan_min: int, tan_max: int, gap_start: int, gap_end: int, gap_width: int) -> tuple[int, int]:
    max_extra = min(2, (gap_start - tan_min) + (tan_max - gap_end))
    extra = random.randint(1, max_extra)
    width = gap_width + extra
    left_spare = gap_start - tan_min
    right_spare = tan_max - gap_end
    lo = max(0, extra - right_spare)
    hi = min(extra, left_spare)
    left_extra = random.randint(lo, hi)
    return gap_start - left_extra, width


def _expected_from_meta(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = grid.copy()
    bg = int(meta["bg"])
    side = meta["side"]
    fr0, fc0, fr1, fc1 = meta["frame_bbox"]
    gap_start, gap_width = int(meta["gap_start"]), int(meta["gap_width"])

    for entry in meta["eligible"]:
        coords = entry["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    if side == "top":
        next_bottom = fr1 - 1
        for entry in meta["eligible"]:
            depth = int(entry["depth"])
            r = next_bottom - depth + 1
            out[r : r + depth, gap_start : gap_start + gap_width] = int(entry["color"])
            next_bottom = r - 1
    elif side == "bottom":
        next_top = fr0 + 1
        for entry in meta["eligible"]:
            depth = int(entry["depth"])
            out[next_top : next_top + depth, gap_start : gap_start + gap_width] = int(entry["color"])
            next_top += depth
    elif side == "left":
        next_right = fc1 - 1
        for entry in meta["eligible"]:
            depth = int(entry["depth"])
            c = next_right - depth + 1
            out[gap_start : gap_start + gap_width, c : c + depth] = int(entry["color"])
            next_right = c - 1
    else:
        next_left = fc0 + 1
        for entry in meta["eligible"]:
            depth = int(entry["depth"])
            out[gap_start : gap_start + gap_width, next_left : next_left + depth] = int(entry["color"])
            next_left += depth
    return out


def _make_case(preferred_side: str) -> tuple[np.ndarray, dict] | None:
    side = preferred_side
    bg = random.choice([0, 7, 8])
    palette = [c for c in range(10) if c != bg]
    frame_color = random.choice(palette)
    sprite_colors = [c for c in palette if c != frame_color]
    random.shuffle(sprite_colors)

    gap_width = random.randint(2, 5)
    interior_tangent = random.randint(max(gap_width + 4, 8), 15)
    interior_depth = random.randint(7, 10)

    eligible_n = random.choice([2, 2, 3])
    depths = [random.choice([1, 1, 2, 2, 3]) for _ in range(eligible_n)]
    for _ in range(40):
        if sum(depths) <= interior_depth - 1:
            break
        depths = [random.choice([1, 1, 2, 2, 3]) for _ in range(eligible_n)]
    if sum(depths) > interior_depth - 1:
        return None

    distances: list[int] = []
    dist = random.randint(1, 2)
    for depth in depths:
        distances.append(dist)
        dist += depth + random.randint(1, 2)
    too_depth = random.choice([1, 2])
    too_distance = dist + random.randint(1, 2)
    side_margin = too_distance + too_depth - 1 + random.randint(0, 2)
    opp_margin = random.randint(3, 5)
    cross_a = random.randint(3, 5)
    cross_b = random.randint(3, 5)

    if side in ("top", "bottom"):
        frame_h, frame_w = interior_depth + 2, interior_tangent + 2
        top_margin = side_margin if side == "top" else opp_margin
        bottom_margin = side_margin if side == "bottom" else opp_margin
        left_margin, right_margin = cross_a, cross_b
    else:
        frame_h, frame_w = interior_tangent + 2, interior_depth + 2
        left_margin = side_margin if side == "left" else opp_margin
        right_margin = side_margin if side == "right" else opp_margin
        top_margin, bottom_margin = cross_a, cross_b

    rows = top_margin + frame_h + bottom_margin
    cols = left_margin + frame_w + right_margin
    if rows > 30 or cols > 30:
        return None

    grid = np.full((rows, cols), bg, dtype=int)
    frame_bbox = (top_margin, left_margin, top_margin + frame_h - 1, left_margin + frame_w - 1)
    fr0, fc0, fr1, fc1 = frame_bbox
    if side in ("top", "bottom"):
        tan_min, tan_max = fc0 + 1, fc1 - 1
    else:
        tan_min, tan_max = fr0 + 1, fr1 - 1
    gap_offset = random.randint(1, interior_tangent - gap_width - 1)
    gap_start = tan_min + gap_offset
    gap = {"side": side, "start": gap_start, "end": gap_start + gap_width - 1, "width": gap_width}
    _draw_frame(grid, frame_bbox, frame_color, gap, bg)

    meta = {
        "bg": bg,
        "side": side,
        "frame_bbox": frame_bbox,
        "gap_start": gap_start,
        "gap_width": gap_width,
        "eligible": [],
        "ineligible": [],
    }

    color_idx = 0
    for distance, depth in zip(distances, depths):
        rect = _rect_for_side(side, frame_bbox, distance, depth, gap_start, gap_width)
        if not _can_place(grid, *rect, pad=0):
            return None
        color = sprite_colors[color_idx]
        color_idx += 1
        entry = _paint_rect(grid, rect, color)
        entry["distance"] = distance
        entry["depth"] = depth
        meta["eligible"].append(entry)

    wide_start, wide_len = _wide_interval(tan_min, tan_max, gap_start, gap["end"], gap_width)
    rect = _rect_for_side(side, frame_bbox, too_distance, too_depth, wide_start, wide_len)
    if not _can_place(grid, *rect, pad=0):
        return None
    entry = _paint_rect(grid, rect, sprite_colors[color_idx])
    color_idx += 1
    entry["reason"] = "too_wide"
    meta["ineligible"].append(entry)

    mis = _choose_misaligned_interval(tan_min, tan_max, gap_start, gap["end"], gap_width)
    if mis is None:
        return None
    mis_start, mis_len = mis
    mis_depth = random.choice([1, 2])
    mis_distance = random.randint(1, max(1, side_margin - mis_depth))
    rect = _rect_for_side(side, frame_bbox, mis_distance, mis_depth, mis_start, mis_len)
    if not _can_place(grid, *rect, pad=0):
        return None
    entry = _paint_rect(grid, rect, sprite_colors[color_idx])
    color_idx += 1
    entry["reason"] = "misaligned"
    meta["ineligible"].append(entry)

    wrong_side = OPPOSITE[side]
    wrong_depth = random.choice([1, 2])
    wrong_distance = random.randint(1, max(1, opp_margin - wrong_depth))
    rect = _rect_for_side(wrong_side, frame_bbox, wrong_distance, wrong_depth, gap_start, gap_width)
    if not _can_place(grid, *rect, pad=0):
        return None
    entry = _paint_rect(grid, rect, sprite_colors[color_idx])
    color_idx += 1
    entry["reason"] = "wrong_side"
    meta["ineligible"].append(entry)

    expected = _expected_from_meta(grid, meta)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    # Semantic checks aimed at common shortcuts: there must be real stacking,
    # all three reject classes are present, and the movement changes the grid.
    if len(meta["eligible"]) < 2:
        return None
    if sorted(e["distance"] for e in meta["eligible"]) != [e["distance"] for e in meta["eligible"]]:
        return None
    if len({e["distance"] for e in meta["eligible"]}) != len(meta["eligible"]):
        return None
    if {e["reason"] for e in meta["ineligible"]} != {"too_wide", "misaligned", "wrong_side"}:
        return None
    if np.array_equal(grid, expected):
        return None

    return grid, meta


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    preferred_side = SIDES[_GEN_COUNTER % len(SIDES)]
    _GEN_COUNTER += 1
    for attempt in range(MAX_ATTEMPTS):
        side = preferred_side if attempt < MAX_ATTEMPTS // 2 else random.choice(SIDES)
        case = _make_case(side)
        if case is not None:
            return case[0]
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")

