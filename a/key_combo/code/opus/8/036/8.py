from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
CROSS = 2
FRAME = 5
TARGET = 8
STRIPE_COLORS = (1, 3, 4, 6, 7, 9)
NOISE_COLORS = (1, 3, 4, 6, 7, 8, 9)
CORNERS = ("tl", "tr", "bl", "br")
GEN_ATTEMPTS = 500


def _rectangular_frame_bbox(g: np.ndarray, color: int = FRAME) -> tuple[int, int, int, int]:
    coords = np.argwhere(g == int(color))
    assert coords.size > 0, "expected a frame"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose an interior"

    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    mask = sub == int(color)
    perimeter = np.zeros_like(mask, dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True
    assert np.array_equal(mask, perimeter), "frame must be a one-pixel hollow rectangle"
    return r0, c0, r1, c1


def _perimeter_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = bbox
    out[r0, c0 : c1 + 1] = True
    out[r1, c0 : c1 + 1] = True
    out[r0 : r1 + 1, c0] = True
    out[r0 : r1 + 1, c1] = True
    return out


def _exterior_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    mask = np.ones((h, w), dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    return mask


def _build_segments(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    *,
    by_rows: bool,
) -> list[dict[str, int | str]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    segments: list[dict[str, int | str]] = []
    if by_rows:
        for r in range(h):
            if r0 <= r <= r1:
                segments.append({"axis": r, "kind": "left", "start": 0, "end": c0 - 1})
                segments.append({"axis": r, "kind": "right", "start": c1 + 1, "end": w - 1})
            else:
                segments.append({"axis": r, "kind": "full", "start": 0, "end": w - 1})
    else:
        for c in range(w):
            if c0 <= c <= c1:
                segments.append({"axis": c, "kind": "top", "start": 0, "end": r0 - 1})
                segments.append({"axis": c, "kind": "bottom", "start": r1 + 1, "end": h - 1})
            else:
                segments.append({"axis": c, "kind": "full", "start": 0, "end": h - 1})

    for seg in segments:
        assert int(seg["start"]) <= int(seg["end"])
    return segments


def _segment_values(g: np.ndarray, seg: dict[str, int | str], *, by_rows: bool) -> np.ndarray:
    axis = int(seg["axis"])
    start = int(seg["start"])
    end = int(seg["end"])
    if by_rows:
        return g[axis, start : end + 1]
    return g[start : end + 1, axis]


def _fill_segment(
    out: np.ndarray,
    seg: dict[str, int | str],
    color: int,
    *,
    by_rows: bool,
) -> None:
    axis = int(seg["axis"])
    start = int(seg["start"])
    end = int(seg["end"])
    if by_rows:
        out[axis, start : end + 1] = int(color)
    else:
        out[start : end + 1, axis] = int(color)


def _segment_key(seg: dict[str, int | str]) -> tuple[int, str]:
    return int(seg["axis"]), str(seg["kind"])


def _segment_mode_ignoring_cross(values: np.ndarray) -> tuple[int | None, int]:
    cleaned = [int(v) for v in values if int(v) != CROSS]
    if not cleaned:
        return None, 0
    counts = Counter(cleaned)
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, "each stripe segment must have a unique visible non-red mode"
    return int(winners[0]), int(best)


def _resolve_segment_colors(
    g: np.ndarray,
    segments: list[dict[str, int | str]],
    *,
    by_rows: bool,
) -> tuple[dict[tuple[int, str], int], int]:
    colors: dict[tuple[int, str], int] = {}
    score = 0
    unresolved: list[dict[str, int | str]] = []
    max_axis = g.shape[0] if by_rows else g.shape[1]

    for seg in segments:
        color, count = _segment_mode_ignoring_cross(_segment_values(g, seg, by_rows=by_rows))
        if color is None:
            unresolved.append(seg)
            continue
        colors[_segment_key(seg)] = int(color)
        score += int(count)

    for seg in unresolved:
        kind = str(seg["kind"])
        axis = int(seg["axis"])
        assert kind != "full", "only the cross-obscured side segments may need interpolation"
        resolved = None
        for dist in range(1, max_axis):
            prev = colors.get((axis - dist, kind))
            nxt = colors.get((axis + dist, kind))
            if prev is not None and nxt is not None:
                assert int(prev) == int(nxt), "obscured segment must agree with neighboring stripe band"
                resolved = int(prev)
                break
            if prev is not None:
                resolved = int(prev)
                break
            if nxt is not None:
                resolved = int(nxt)
                break
        assert resolved is not None, "failed to reconstruct an obscured stripe segment"
        colors[_segment_key(seg)] = int(resolved)

    return colors, score


def _find_cross(g: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(g == CROSS)
    assert coords.size > 0, "expected a red cross"
    row_counts = Counter(map(int, coords[:, 0]))
    col_counts = Counter(map(int, coords[:, 1]))
    cross_r, row_count = max(row_counts.items(), key=lambda kv: kv[1])
    cross_c, col_count = max(col_counts.items(), key=lambda kv: kv[1])
    assert row_count >= 3 and col_count >= 3
    assert all(int(r) == int(cross_r) or int(c) == int(cross_c) for r, c in coords), "red pixels must form one cross"
    return int(cross_r), int(cross_c)


def _corner_keys(
    bbox: tuple[int, int, int, int],
    *,
    by_rows: bool,
) -> dict[str, tuple[int, str]]:
    r0, c0, r1, c1 = bbox
    if by_rows:
        return {
            "tl": (r0, "left"),
            "tr": (r0, "right"),
            "bl": (r1, "left"),
            "br": (r1, "right"),
        }
    return {
        "tl": (c0, "top"),
        "tr": (c1, "top"),
        "bl": (c0, "bottom"),
        "br": (c1, "bottom"),
    }


def _corner_anchor(bbox: tuple[int, int, int, int], corner: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return {
        "tl": (r0, c0),
        "tr": (r0, c1),
        "bl": (r1, c0),
        "br": (r1, c1),
    }[corner]


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, bool, str]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bbox = _rectangular_frame_bbox(g, FRAME)
    r0, c0, r1, c1 = bbox
    cross_r, cross_c = _find_cross(g)
    assert r0 < cross_r < r1 and c0 < cross_c < c1, "input cross must pass through the strict frame interior"

    row_segments = _build_segments(g.shape, bbox, by_rows=True)
    col_segments = _build_segments(g.shape, bbox, by_rows=False)
    row_colors, row_score = _resolve_segment_colors(g, row_segments, by_rows=True)
    col_colors, col_score = _resolve_segment_colors(g, col_segments, by_rows=False)
    assert row_score != col_score, "stripe orientation must be uniquely determined"

    by_rows = bool(row_score > col_score)
    segments = row_segments if by_rows else col_segments
    seg_colors = row_colors if by_rows else col_colors

    corner_keys = _corner_keys(bbox, by_rows=by_rows)
    teal_corners = [corner for corner, key in corner_keys.items() if int(seg_colors[key]) == TARGET]
    assert len(teal_corners) == 1, "exactly one corner-adjacent stripe segment must denoise to teal"
    target_corner = teal_corners[0]
    anchor_r, anchor_c = _corner_anchor(bbox, target_corner)

    out = np.full_like(g, BG)
    for seg in segments:
        _fill_segment(out, seg, seg_colors[_segment_key(seg)], by_rows=by_rows)

    perimeter = _perimeter_mask(g.shape, bbox)
    out[perimeter] = FRAME
    out[anchor_r, :] = CROSS
    out[:, anchor_c] = CROSS
    out[perimeter] = FRAME
    return out, by_rows, target_corner


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise the outer stripe field to reveal the teal corner anchor, then move the red cross onto that frame corner.

    Input structure:
    - A grey one-pixel hollow rectangle sits near the center, with a black interior.
    - A red full-width/full-height cross passes through the strict interior of the frame, but the frame hides the overlaps on its border.
    - Outside the rectangle is a noisy stripe field whose true orientation is horizontal or vertical; because the frame blocks the middle, each stripe is read on the connected exterior row-segment or column-segment that remains visible.

    Rule:
    - Compare the exterior row-segment interpretation and column-segment interpretation by summing the visible non-red dominant-count of every segment; the larger score is the true stripe orientation.
    - Repaint every stripe segment in that orientation to its dominant non-red color, interpolating the one segment fully obscured by the red cross from its neighboring same-side stripe band.
    - Among the four corner-adjacent denoised stripe segments, exactly one is teal; erase the original cross and redraw it through that frame corner's row and column, with the grey frame restored on top.

    Key insight:
    - The stripe cleaning is not the answer itself: it is the only way to decode which frame corner is special, and that decoded corner becomes the destination anchor for the cross translation.
    """
    return _solve_details(grid)[0]


def _sample_band_sequence(length: int) -> list[int]:
    assert length >= 2
    seq: list[int] = []
    prev = None
    remaining = length
    while remaining > 0:
        choices = []
        for width in (2, 3, 4):
            if width > remaining:
                continue
            tail = remaining - width
            if tail in (0,) or tail >= 2:
                choices.append(width)
        assert choices
        width = random.choice(choices)
        color_choices = [color for color in STRIPE_COLORS if color != prev]
        color = random.choice(color_choices)
        seq.extend([int(color)] * int(width))
        prev = int(color)
        remaining -= int(width)
    return seq


def _horizontal_pattern(
    h: int,
    w: int,
    bbox: tuple[int, int, int, int],
    target_corner: str,
) -> tuple[np.ndarray, dict[tuple[int, str], int], int]:
    r0, c0, r1, c1 = bbox
    frame_h = r1 - r0 + 1
    top_seq = _sample_band_sequence(r0)
    left_seq = _sample_band_sequence(frame_h)
    right_seq = _sample_band_sequence(frame_h)
    bottom_seq = _sample_band_sequence(h - r1 - 1)

    if target_corner == "tl":
        left_seq[0] = TARGET
    elif target_corner == "tr":
        right_seq[0] = TARGET
    elif target_corner == "bl":
        left_seq[-1] = TARGET
    else:
        right_seq[-1] = TARGET

    candidates = []
    for idx in range(1, frame_h - 1):
        left_ok = left_seq[idx] == left_seq[idx - 1] or left_seq[idx] == left_seq[idx + 1]
        right_ok = right_seq[idx] == right_seq[idx - 1] or right_seq[idx] == right_seq[idx + 1]
        if left_ok and right_ok:
            candidates.append(idx)
    assert candidates, "need an interior row whose obscured left/right stripes can be reconstructed"
    cross_r = r0 + random.choice(candidates)

    clean = np.full((h, w), BG, dtype=int)
    for r in range(r0):
        clean[r, :] = int(top_seq[r])
    for offset, r in enumerate(range(r0, r1 + 1)):
        clean[r, :c0] = int(left_seq[offset])
        clean[r, c1 + 1 :] = int(right_seq[offset])
    for offset, r in enumerate(range(r1 + 1, h)):
        clean[r, :] = int(bottom_seq[offset])

    colors: dict[tuple[int, str], int] = {}
    for r in range(h):
        if r0 <= r <= r1:
            colors[(r, "left")] = int(left_seq[r - r0])
            colors[(r, "right")] = int(right_seq[r - r0])
        else:
            colors[(r, "full")] = int(top_seq[r] if r < r0 else bottom_seq[r - r1 - 1])
    return clean, colors, int(cross_r)


def _vertical_pattern(
    h: int,
    w: int,
    bbox: tuple[int, int, int, int],
    target_corner: str,
) -> tuple[np.ndarray, dict[tuple[int, str], int], int]:
    r0, c0, r1, c1 = bbox
    frame_w = c1 - c0 + 1
    left_seq = _sample_band_sequence(c0)
    top_seq = _sample_band_sequence(frame_w)
    bottom_seq = _sample_band_sequence(frame_w)
    right_seq = _sample_band_sequence(w - c1 - 1)

    if target_corner == "tl":
        top_seq[0] = TARGET
    elif target_corner == "tr":
        top_seq[-1] = TARGET
    elif target_corner == "bl":
        bottom_seq[0] = TARGET
    else:
        bottom_seq[-1] = TARGET

    candidates = []
    for idx in range(1, frame_w - 1):
        top_ok = top_seq[idx] == top_seq[idx - 1] or top_seq[idx] == top_seq[idx + 1]
        bottom_ok = bottom_seq[idx] == bottom_seq[idx - 1] or bottom_seq[idx] == bottom_seq[idx + 1]
        if top_ok and bottom_ok:
            candidates.append(idx)
    assert candidates, "need an interior column whose obscured top/bottom stripes can be reconstructed"
    cross_c = c0 + random.choice(candidates)

    clean = np.full((h, w), BG, dtype=int)
    for c in range(c0):
        clean[:, c] = int(left_seq[c])
    for offset, c in enumerate(range(c0, c1 + 1)):
        clean[:r0, c] = int(top_seq[offset])
        clean[r1 + 1 :, c] = int(bottom_seq[offset])
    for offset, c in enumerate(range(c1 + 1, w)):
        clean[:, c] = int(right_seq[offset])

    colors: dict[tuple[int, str], int] = {}
    for c in range(w):
        if c0 <= c <= c1:
            colors[(c, "top")] = int(top_seq[c - c0])
            colors[(c, "bottom")] = int(bottom_seq[c - c0])
        else:
            colors[(c, "full")] = int(left_seq[c] if c < c0 else right_seq[c - c1 - 1])
    return clean, colors, int(cross_c)


def _draw_frame_and_cross(
    clean: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    cross_r: int,
    cross_c: int,
) -> np.ndarray:
    out = clean.copy()
    r0, c0, r1, c1 = bbox
    out[cross_r, :] = CROSS
    out[:, cross_c] = CROSS
    out[r0, c0 : c1 + 1] = FRAME
    out[r1, c0 : c1 + 1] = FRAME
    out[r0 : r1 + 1, c0] = FRAME
    out[r0 : r1 + 1, c1] = FRAME
    return out


def _sample_frame_bbox(h: int, w: int) -> tuple[int, int, int, int]:
    frame_h = random.randint(7, min(11, h - 6))
    frame_w = random.randint(7, min(11, w - 6))
    r0 = random.randint(3, h - frame_h - 3)
    c0 = random.randint(3, w - frame_w - 3)
    return r0, c0, r0 + frame_h - 1, c0 + frame_w - 1


def _apply_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    by_rows: bool,
    target_corner: str,
    true_colors: dict[tuple[int, str], int],
) -> None:
    segments = _build_segments(inp.shape, bbox, by_rows=by_rows)
    corner_keys = _corner_keys(bbox, by_rows=by_rows)

    def candidates(seg: dict[str, int | str]) -> list[tuple[int, int]]:
        axis = int(seg["axis"])
        start = int(seg["start"])
        end = int(seg["end"])
        pts: list[tuple[int, int]] = []
        if by_rows:
            for c in range(start, end + 1):
                if inp[axis, c] != CROSS:
                    pts.append((axis, c))
        else:
            for r in range(start, end + 1):
                if inp[r, axis] != CROSS:
                    pts.append((r, axis))
        return pts

    used: set[tuple[int, int]] = set()
    for corner in CORNERS:
        if corner == target_corner:
            continue
        key = corner_keys[corner]
        seg = next(seg for seg in segments if _segment_key(seg) == key)
        pts = [pt for pt in candidates(seg) if pt not in used]
        if len(pts) >= 3:
            pt = random.choice(pts)
            inp[pt] = TARGET
            used.add(pt)

    for seg in segments:
        key = _segment_key(seg)
        dominant = int(true_colors[key])
        pts = [pt for pt in candidates(seg) if pt not in used]
        if not pts:
            continue
        length = len(pts)
        max_noise = min(2, max(0, length // 4))
        if max_noise <= 0:
            continue
        noise_n = random.randint(0, max_noise)
        if noise_n == 0:
            continue
        random.shuffle(pts)
        for pt in pts[:noise_n]:
            choices = [color for color in NOISE_COLORS if color != dominant]
            inp[pt] = random.choice(choices)
            used.add(pt)

    exterior = _exterior_mask(inp.shape, bbox)
    assert np.all(inp[~exterior & ~_perimeter_mask(inp.shape, bbox)] == BG) or np.any(inp == CROSS)
    assert np.all(clean[exterior] != BG)


def _count_cyan_corner_segments(
    inp: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    by_rows: bool,
) -> int:
    corner_keys = _corner_keys(bbox, by_rows=by_rows)
    segments = _build_segments(inp.shape, bbox, by_rows=by_rows)
    seg_map = {_segment_key(seg): seg for seg in segments}
    count = 0
    for key in corner_keys.values():
        vals = _segment_values(inp, seg_map[key], by_rows=by_rows)
        if np.any(vals == TARGET):
            count += 1
    return count


def _attempt() -> np.ndarray | None:
    h = random.randint(14, 19)
    w = random.randint(14, 19)
    bbox = _sample_frame_bbox(h, w)
    by_rows = bool(random.getrandbits(1))
    target_corner = random.choice(CORNERS)
    r0, c0, r1, c1 = bbox

    try:
        if by_rows:
            clean, true_colors, cross_r = _horizontal_pattern(h, w, bbox, target_corner)
            cross_c = random.randint(c0 + 1, c1 - 1)
        else:
            clean, true_colors, cross_c = _vertical_pattern(h, w, bbox, target_corner)
            cross_r = random.randint(r0 + 1, r1 - 1)
    except AssertionError:
        return None

    target_r, target_c = _corner_anchor(bbox, target_corner)
    expected = _draw_frame_and_cross(clean, bbox, cross_r=target_r, cross_c=target_c)
    inp = _draw_frame_and_cross(clean, bbox, cross_r=cross_r, cross_c=cross_c)
    _apply_noise(inp, clean, bbox, by_rows=by_rows, target_corner=target_corner, true_colors=true_colors)

    try:
        solved, solved_rows, solved_corner = _solve_details(inp)
    except AssertionError:
        return None

    if solved_rows != by_rows or solved_corner != target_corner:
        return None
    if not np.array_equal(solved, expected):
        return None

    row_score = _resolve_segment_colors(inp, _build_segments(inp.shape, bbox, by_rows=True), by_rows=True)[1]
    col_score = _resolve_segment_colors(inp, _build_segments(inp.shape, bbox, by_rows=False), by_rows=False)[1]
    if by_rows and row_score <= col_score:
        return None
    if (not by_rows) and col_score <= row_score:
        return None

    if _count_cyan_corner_segments(inp, bbox, by_rows=by_rows) < 2:
        return None
    if np.array_equal(inp, expected):
        return None
    if np.count_nonzero(inp == TARGET) < 4:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
