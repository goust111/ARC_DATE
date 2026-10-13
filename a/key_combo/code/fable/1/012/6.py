from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists():
    sys.path.append(str(_UTILS_DIR))

try:
    from arc_utils import bbox_coords, components_mask, get_bg
except Exception:  # pragma: no cover - fallback for direct loading without the shared toolbox
    def get_bg(grid: np.ndarray) -> int:
        vals, counts = np.unique(grid, return_counts=True)
        return int(vals[np.argmax(counts)])

    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        h, w = mask.shape
        seen = np.zeros((h, w), dtype=bool)
        if int(connectivity) == 4:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            dirs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
        out = []
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                comp = []
                while stack:
                    rr, cc = stack.pop()
                    comp.append((rr, cc))
                    for dr, dc in dirs:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                out.append(np.array(comp, dtype=int))
        return out


SIDE_TOP = "top"
SIDE_RIGHT = "right"
SIDE_BOTTOM = "bottom"
SIDE_LEFT = "left"
SIDES = [SIDE_TOP, SIDE_RIGHT, SIDE_BOTTOM, SIDE_LEFT]

MAX_GEN_ATTEMPTS = 500
MAX_SIDE_SEGMENT_ATTEMPTS = 120
MAX_BAR_PLACEMENT_ATTEMPTS = 120
FRAME_SIZE_RANGE = (10, 16)
GRID_SIZE_RANGE = (20, 30)
SLOT_LENGTH_RANGE = (2, 6)
PATTERNS = [
    [SIDE_TOP, SIDE_BOTTOM],
    [SIDE_LEFT, SIDE_RIGHT],
    [SIDE_TOP, SIDE_RIGHT],
    [SIDE_TOP, SIDE_BOTTOM, SIDE_RIGHT],
    [SIDE_LEFT, SIDE_RIGHT, SIDE_TOP],
    [SIDE_TOP, SIDE_RIGHT, SIDE_BOTTOM],
    [SIDE_TOP, SIDE_RIGHT, SIDE_BOTTOM, SIDE_LEFT],
    [SIDE_TOP, SIDE_TOP, SIDE_BOTTOM],
    [SIDE_LEFT, SIDE_RIGHT, SIDE_RIGHT],
    [SIDE_TOP, SIDE_BOTTOM, SIDE_LEFT, SIDE_RIGHT],
    [SIDE_TOP, SIDE_TOP, SIDE_RIGHT, SIDE_BOTTOM],
    [SIDE_LEFT, SIDE_TOP, SIDE_RIGHT, SIDE_RIGHT],
]
PATTERN_WEIGHTS = [6, 6, 3, 6, 6, 6, 8, 4, 4, 8, 4, 4]


def _all_colors_except(*excluded: int) -> list[int]:
    return [c for c in range(10) if c not in set(map(int, excluded))]


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _expand_bbox(b: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    return (
        max(0, int(b[0] - pad)),
        max(0, int(b[1] - pad)),
        min(h - 1, int(b[2] + pad)),
        min(w - 1, int(b[3] + pad)),
    )


def _inside_frame_interior(
    frame_bbox: tuple[int, int, int, int], r: int, c: int, h: int, w: int
) -> bool:
    r0, c0, r1, c1 = frame_bbox
    return r0 + 1 <= r and c0 + 1 <= c and r + h - 1 <= r1 - 1 and c + w - 1 <= c1 - 1


def _missing_segments(side_vals: np.ndarray, frame_color: int) -> list[tuple[int, int]]:
    segments = []
    i = 0
    n = int(len(side_vals))
    while i < n:
        if int(side_vals[i]) == int(frame_color):
            i += 1
            continue
        j = i
        while j + 1 < n and int(side_vals[j + 1]) != int(frame_color):
            j += 1
        segments.append((i, j))
        i = j + 1
    return segments


def _extract_slots(
    grid: np.ndarray, frame_color: int, frame_bbox: tuple[int, int, int, int]
) -> list[dict]:
    r0, c0, r1, c1 = frame_bbox
    slots = []

    top = _missing_segments(grid[r0, c0 + 1 : c1], frame_color)
    for s, e in top:
        slots.append(
            {
                "side": SIDE_TOP,
                "start": int(c0 + 1 + s),
                "end": int(c0 + 1 + e),
                "length": int(e - s + 1),
            }
        )

    right = _missing_segments(grid[r0 + 1 : r1, c1], frame_color)
    for s, e in right:
        slots.append(
            {
                "side": SIDE_RIGHT,
                "start": int(r0 + 1 + s),
                "end": int(r0 + 1 + e),
                "length": int(e - s + 1),
            }
        )

    bottom = _missing_segments(grid[r1, c0 + 1 : c1], frame_color)
    for s, e in bottom:
        slots.append(
            {
                "side": SIDE_BOTTOM,
                "start": int(c0 + 1 + s),
                "end": int(c0 + 1 + e),
                "length": int(e - s + 1),
            }
        )

    left = _missing_segments(grid[r0 + 1 : r1, c0], frame_color)
    for s, e in left:
        slots.append(
            {
                "side": SIDE_LEFT,
                "start": int(r0 + 1 + s),
                "end": int(r0 + 1 + e),
                "length": int(e - s + 1),
            }
        )

    return slots


def _slot_clockwise_key(slot: dict, frame_bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = frame_bbox
    if slot["side"] == SIDE_TOP:
        return int(slot["start"] - c0)
    if slot["side"] == SIDE_RIGHT:
        return int((c1 - c0) + (slot["start"] - r0))
    if slot["side"] == SIDE_BOTTOM:
        return int((c1 - c0) + (r1 - r0) + (c1 - slot["end"]))
    return int((c1 - c0) * 2 + (r1 - r0) + (r1 - slot["end"]))


def _detect_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], list[dict]]:
    best = None
    colors = [int(c) for c in np.unique(grid) if int(c) != int(bg)]
    for color in colors:
        coords = np.argwhere(grid == int(color))
        if coords.size == 0:
            continue
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 5 or w < 5:
            continue
        points = {(int(r), int(c)) for r, c in coords}
        if (r0, c0) not in points or (r0, c1) not in points or (r1, c0) not in points or (r1, c1) not in points:
            continue
        if any(r not in (r0, r1) and c not in (c0, c1) for r, c in points):
            continue
        slots = _extract_slots(grid, int(color), bbox)
        if not slots:
            continue
        if any(slot["length"] <= 0 for slot in slots):
            continue
        full_perimeter = 2 * (h + w) - 4
        if len(points) >= full_perimeter:
            continue
        score = (h * w, full_perimeter - len(points))
        if best is None or score > best[0]:
            best = (score, int(color), bbox, slots)
    if best is None:
        raise AssertionError("failed to detect a notched rectangular frame")
    _score, frame_color, frame_bbox, slots = best
    slots.sort(key=lambda s: _slot_clockwise_key(s, frame_bbox))
    lengths = [slot["length"] for slot in slots]
    assert len(lengths) == len(set(lengths)), "slot lengths must be unique"
    return frame_color, frame_bbox, slots


def _detect_bars(
    grid: np.ndarray, bg: int, frame_color: int, frame_bbox: tuple[int, int, int, int]
) -> list[dict]:
    bars = []
    for color in np.unique(grid):
        color = int(color)
        if color in (int(bg), int(frame_color)):
            continue
        comps = components_mask(grid == color, connectivity=4)
        for coords in comps:
            bbox = bbox_coords(coords)
            r0, c0, r1, c1 = bbox
            if _bbox_overlap(bbox, frame_bbox):
                raise AssertionError("external bars must stay outside the frame")
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if int(coords.shape[0]) != patch.size or not np.all(patch == color):
                raise AssertionError("each external object must be a solid rectangle")
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h == w:
                raise AssertionError("bars must have a unique long side")
            bars.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "h": h,
                    "w": w,
                    "length": max(h, w),
                    "thickness": min(h, w),
                }
            )
    lengths = [bar["length"] for bar in bars]
    assert len(lengths) == len(set(lengths)), "bar lengths must be unique"
    return bars


def _erase_bars(out: np.ndarray, bars: list[dict], bg: int) -> None:
    for bar in bars:
        r0, c0, r1, c1 = bar["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)


def _simulate_drop(
    out: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    slots: list[dict],
    bars_by_length: dict[int, dict],
) -> tuple[list[str], list[tuple[int, int, int, int, int]]]:
    occupied = np.zeros(out.shape, dtype=bool)
    stops = []
    placements = []
    r0, c0, r1, c1 = frame_bbox

    for slot in slots:
        bar = bars_by_length[int(slot["length"])]
        h = int(bar["h"])
        w = int(bar["w"])
        side = slot["side"]
        if side in (SIDE_TOP, SIDE_BOTTOM):
            assert w == int(slot["length"]) and h < w
        else:
            assert h == int(slot["length"]) and w < h

        if side == SIDE_TOP:
            cur_r, cur_c = int(r0 + 1), int(slot["start"])
            dr, dc = 1, 0
        elif side == SIDE_BOTTOM:
            cur_r, cur_c = int(r1 - h), int(slot["start"])
            dr, dc = -1, 0
        elif side == SIDE_LEFT:
            cur_r, cur_c = int(slot["start"]), int(c0 + 1)
            dr, dc = 0, 1
        else:
            cur_r, cur_c = int(slot["start"]), int(c1 - w)
            dr, dc = 0, -1

        if occupied[cur_r : cur_r + h, cur_c : cur_c + w].any():
            raise AssertionError("a bar cannot enter through an already blocked slot")

        stop_reason = "wall"
        while True:
            nr, nc = cur_r + dr, cur_c + dc
            if not _inside_frame_interior(frame_bbox, nr, nc, h, w):
                break
            if occupied[nr : nr + h, nc : nc + w].any():
                stop_reason = "bar"
                break
            cur_r, cur_c = nr, nc

        out[cur_r : cur_r + h, cur_c : cur_c + w] = int(bar["color"])
        occupied[cur_r : cur_r + h, cur_c : cur_c + w] = True
        if side == SIDE_TOP:
            out[r0, slot["start"] : slot["end"] + 1] = int(bar["color"])
        elif side == SIDE_BOTTOM:
            out[r1, slot["start"] : slot["end"] + 1] = int(bar["color"])
        elif side == SIDE_LEFT:
            out[slot["start"] : slot["end"] + 1, c0] = int(bar["color"])
        else:
            out[slot["start"] : slot["end"] + 1, c1] = int(bar["color"])
        stops.append(stop_reason)
        placements.append((cur_r, cur_c, h, w, int(bar["color"])))

    return stops, placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each external bar through the same-length notch, slide it inward, and recolor the notch to seal it.

    Input structure:
    - The grid has a dominant background, one large one-pixel-thick rectangular frame of a single color, and several missing notch segments on that frame.
    - Outside the frame sit several solid monochrome rectangular bars; each bar is axis-aligned and its long side length matches exactly one notch length.

    Rule:
    - Read the notches in clockwise order starting from the top-left corner of the frame.
    - For each notch, find the unique bar whose long side has the same length, translate it so that it enters through that notch, then slide it straight inward perpendicular to that frame side until the next step would hit the opposite wall or a previously parked bar.
    - Remove the original outside bars, keep every parked bar inside the frame, and recolor the missing notch cells with that bar's color.

    Key insight:
    - The notch length determines which bar uses which opening, while the clockwise processing order affects later collisions because earlier parked bars become new internal obstacles.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    frame_color, frame_bbox, slots = _detect_frame(g, bg)
    bars = _detect_bars(g, bg, frame_color, frame_bbox)
    bars_by_length = {int(bar["length"]): bar for bar in bars}
    assert {slot["length"] for slot in slots} == set(bars_by_length), "slots and bars must match by unique lengths"

    out = g.copy()
    _erase_bars(out, bars, bg)
    _simulate_drop(out, frame_bbox, slots, bars_by_length)
    return out


def _choose_frame_bbox(h: int, w: int) -> tuple[int, int, int, int]:
    frame_h = random.randint(FRAME_SIZE_RANGE[0], min(FRAME_SIZE_RANGE[1], h - 6))
    frame_w = random.randint(FRAME_SIZE_RANGE[0], min(FRAME_SIZE_RANGE[1], w - 6))
    top = random.randint(3, h - frame_h - 3)
    left = random.randint(3, w - frame_w - 3)
    return (top, left, top + frame_h - 1, left + frame_w - 1)


def _sample_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(int(cut - prev - 1))
        prev = cut
    return values


def _sample_side_segments(side_len: int, count: int, used_lengths: set[int]) -> list[tuple[int, int, int]] | None:
    if count == 0:
        return []
    low, high = SLOT_LENGTH_RANGE
    candidates = [v for v in range(low, min(high, side_len) + 1) if v not in used_lengths]
    if len(candidates) < count:
        return None
    for _ in range(MAX_SIDE_SEGMENT_ATTEMPTS):
        lengths = random.sample(candidates, count)
        if sum(lengths) + (count - 1) > side_len:
            continue
        random.shuffle(lengths)
        spare = side_len - sum(lengths) - (count - 1)
        gaps = _sample_composition(spare, count + 1)
        pos = gaps[0]
        out = []
        for i, length in enumerate(lengths):
            start = int(pos)
            end = int(start + length - 1)
            out.append((start, end, int(length)))
            pos = end + 1
            if i != count - 1:
                pos += 1 + gaps[i + 1]
        return out
    return None


def _pattern_counts(pattern: list[str]) -> Counter:
    return Counter(pattern)


def _build_slots(frame_bbox: tuple[int, int, int, int]) -> list[dict] | None:
    r0, c0, r1, c1 = frame_bbox
    side_lengths = {
        SIDE_TOP: int(c1 - c0 - 1),
        SIDE_BOTTOM: int(c1 - c0 - 1),
        SIDE_LEFT: int(r1 - r0 - 1),
        SIDE_RIGHT: int(r1 - r0 - 1),
    }

    for _ in range(MAX_SIDE_SEGMENT_ATTEMPTS):
        pattern = random.choices(PATTERNS, weights=PATTERN_WEIGHTS, k=1)[0]
        counts = _pattern_counts(pattern)
        used_lengths: set[int] = set()
        slots = []
        ok = True
        for side in SIDES:
            segs = _sample_side_segments(side_lengths[side], counts[side], used_lengths)
            if segs is None:
                ok = False
                break
            for start_rel, end_rel, length in segs:
                if side in (SIDE_TOP, SIDE_BOTTOM):
                    start = int(c0 + 1 + start_rel)
                    end = int(c0 + 1 + end_rel)
                else:
                    start = int(r0 + 1 + start_rel)
                    end = int(r0 + 1 + end_rel)
                slots.append({"side": side, "start": start, "end": end, "length": int(length)})
                used_lengths.add(int(length))
        if not ok:
            continue
        slots.sort(key=lambda s: _slot_clockwise_key(s, frame_bbox))
        return slots
    return None


def _make_bar_specs(
    slots: list[dict], frame_bbox: tuple[int, int, int, int], colors: list[int]
) -> dict[int, dict] | None:
    r0, c0, r1, c1 = frame_bbox
    interior_h = int(r1 - r0 - 1)
    interior_w = int(c1 - c0 - 1)
    bars = {}
    for slot, color in zip(slots, colors, strict=False):
        length = int(slot["length"])
        if slot["side"] in (SIDE_TOP, SIDE_BOTTOM):
            cross = interior_h
            candidates = [t for t in (1, 2, 3) if t < length and t <= cross]
            if not candidates:
                return None
            thickness = random.choices(
                candidates,
                weights=[5 if t == 1 else 3 if t == 2 else 1 for t in candidates],
                k=1,
            )[0]
            h, w = int(thickness), int(length)
        else:
            cross = interior_w
            candidates = [t for t in (1, 2, 3) if t < length and t <= cross]
            if not candidates:
                return None
            thickness = random.choices(
                candidates,
                weights=[5 if t == 1 else 3 if t == 2 else 1 for t in candidates],
                k=1,
            )[0]
            h, w = int(length), int(thickness)
        bars[length] = {"color": int(color), "length": length, "thickness": int(thickness), "h": h, "w": w}
    return bars


def _draw_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], frame_color: int, slots: list[dict], bg: int) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = int(frame_color)
    grid[r1, c0 : c1 + 1] = int(frame_color)
    grid[r0 : r1 + 1, c0] = int(frame_color)
    grid[r0 : r1 + 1, c1] = int(frame_color)
    for slot in slots:
        if slot["side"] == SIDE_TOP:
            grid[r0, slot["start"] : slot["end"] + 1] = int(bg)
        elif slot["side"] == SIDE_BOTTOM:
            grid[r1, slot["start"] : slot["end"] + 1] = int(bg)
        elif slot["side"] == SIDE_LEFT:
            grid[slot["start"] : slot["end"] + 1, c0] = int(bg)
        else:
            grid[slot["start"] : slot["end"] + 1, c1] = int(bg)


def _place_external_bars(
    grid_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    bars_by_length: dict[int, dict],
) -> list[dict] | None:
    h, w = grid_shape
    placed: list[dict] = []
    frame_keepout = _expand_bbox(frame_bbox, 1, grid_shape)
    bar_list = list(bars_by_length.values())
    random.shuffle(bar_list)

    def candidates_for(bar: dict, existing: list[dict]) -> list[tuple[int, int]]:
        bh, bw = int(bar["h"]), int(bar["w"])
        cands = []
        for r in range(0, h - bh + 1):
            for c in range(0, w - bw + 1):
                bbox = (r, c, r + bh - 1, c + bw - 1)
                if _bbox_overlap(bbox, frame_keepout):
                    continue
                if any(_bbox_overlap(bbox, _expand_bbox(item["bbox"], 1, grid_shape)) for item in existing):
                    continue
                cands.append((r, c))
        random.shuffle(cands)
        return cands

    def search(todo: list[dict], existing: list[dict]) -> list[dict] | None:
        if not todo:
            return existing
        best_i = None
        best_cands = None
        for i, bar in enumerate(todo):
            cands = candidates_for(bar, existing)
            if not cands:
                return None
            if best_cands is None or len(cands) < len(best_cands):
                best_i = i
                best_cands = cands
        assert best_i is not None and best_cands is not None
        bar = todo[best_i]
        rest = todo[:best_i] + todo[best_i + 1 :]
        for r, c in best_cands[:MAX_BAR_PLACEMENT_ATTEMPTS]:
            placed_bar = dict(bar)
            placed_bar["bbox"] = (int(r), int(c), int(r + bar["h"] - 1), int(c + bar["w"] - 1))
            result = search(rest, existing + [placed_bar])
            if result is not None:
                return result
        return None

    return search(bar_list, placed)


def _render_input(
    shape: tuple[int, int],
    bg: int,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
    slots: list[dict],
    placed_bars: list[dict],
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    _draw_frame(grid, frame_bbox, frame_color, slots, bg)
    for bar in placed_bars:
        r0, c0, r1, c1 = bar["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(bar["color"])
    return grid


def _accept_geometry(
    shape: tuple[int, int],
    bg: int,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
    slots: list[dict],
    bars_by_length: dict[int, dict],
    placed_bars: list[dict],
) -> np.ndarray | None:
    inp = _render_input(shape, bg, frame_color, frame_bbox, slots, placed_bars)
    out = inp.copy()
    _erase_bars(out, placed_bars, bg)
    try:
        stops, placements = _simulate_drop(out, frame_bbox, slots, bars_by_length)
    except AssertionError:
        return None

    if len(set(stops)) == 1 and stops[0] == "wall" and len(slots) >= 3 and random.random() < 0.7:
        return None

    r0, c0, r1, c1 = frame_bbox
    interior_area = max(1, (r1 - r0 - 1) * (c1 - c0 - 1))
    filled_area = sum(h * w for _r, _c, h, w, _color in placements)
    if filled_area > interior_area * 0.55:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        h = random.randint(*GRID_SIZE_RANGE)
        w = random.randint(*GRID_SIZE_RANGE)
        frame_bbox = _choose_frame_bbox(h, w)
        slots = _build_slots(frame_bbox)
        if slots is None:
            continue

        n_slots = len(slots)
        bg = random.randint(0, 9)
        frame_color = random.choice(_all_colors_except(bg))
        bar_colors = random.sample(_all_colors_except(bg, frame_color), n_slots)
        bars_by_length = _make_bar_specs(slots, frame_bbox, bar_colors)
        if bars_by_length is None:
            continue

        placed_bars = _place_external_bars((h, w), frame_bbox, bars_by_length)
        if placed_bars is None:
            continue

        grid = _accept_geometry((h, w), bg, frame_color, frame_bbox, slots, bars_by_length, placed_bars)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid keyhole-drop input")
