import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BG = 0
FRAME = 5
MARKER_COLORS = (1, 2, 3, 4, 7)
FOUR_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_ATTEMPTS = 500
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
FRAME_COUNT_RANGE = (3, 5)
FRAME_OUTER_MIN = 4
FRAME_OUTER_MAX = 6


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in FOUR_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(coords: Sequence[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _solid_square(mask: np.ndarray) -> bool:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not cells:
        return False
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return h == w and len(cells) == h * w


def _crosses_frame(segment: Dict[str, int], frame_bbox: Tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = frame_bbox
    if segment["orientation"] == "h":
        return r0 <= segment["fixed"] <= r1 and segment["lo"] < c0 and segment["hi"] > c1
    return c0 <= segment["fixed"] <= c1 and segment["lo"] < r0 and segment["hi"] > r1


def _extract_frames(grid: np.ndarray) -> List[Dict[str, object]]:
    frames: List[Dict[str, object]] = []
    for gray_cells in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(gray_cells)
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        gray_mask = crop == FRAME
        h, w = gray_mask.shape
        assert h >= 3 and w >= 3, "frame bbox must be at least 3x3"
        assert np.all(gray_mask[0, :]) and np.all(gray_mask[-1, :])
        assert np.all(gray_mask[:, 0]) and np.all(gray_mask[:, -1])

        interior = crop[1:-1, 1:-1]
        assert interior.size > 0
        assert np.all((interior == BG) | (interior == FRAME)), "markers must stay outside frames"
        hole_mask = interior == BG
        hole_components = _components(hole_mask)
        assert len(hole_components) == 1, "each frame must enclose exactly one black hole"
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gray_mask": gray_mask,
                "hole_mask": hole_mask,
            }
        )
    assert frames, "expected at least one gray frame"
    return frames


def _extract_segments(grid: np.ndarray) -> List[Dict[str, int]]:
    segments: List[Dict[str, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if v not in (BG, FRAME)):
        pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        assert len(pts) == 2, "each marker color must appear exactly twice"
        (r0, c0), (r1, c1) = pts
        if r0 == r1:
            lo, hi = sorted((c0, c1))
            segments.append(
                {
                    "color": color,
                    "orientation": "h",
                    "fixed": r0,
                    "lo": lo,
                    "hi": hi,
                }
            )
        else:
            assert c0 == c1, "same-color markers must share a row or column"
            lo, hi = sorted((r0, r1))
            segments.append(
                {
                    "color": color,
                    "orientation": "v",
                    "fixed": c0,
                    "lo": lo,
                    "hi": hi,
                }
            )
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """Route each marker-pair color through its crossed gray frame and fill only solid-square holes.

    Input structure
    - The grid has a black background, several gray framed objects, and one same-colored marker pair per frame.
    - Each gray object has a rectangular outer border and encloses exactly one connected black hole region.
    - Each marker pair shares a row or a column, and the straight segment between the pair fully crosses exactly one gray frame.

    Rule
    - Recover every horizontal or vertical segment implied by a same-colored marker pair.
    - Match each segment to the unique gray frame whose outer rectangle it crosses from one side to the opposite side.
    - For that frame, inspect the enclosed black hole: if the hole itself is a completely filled square bbox, recolor the hole with the segment color; otherwise leave the hole black.
    - Erase all markers and keep only the gray frame objects plus any accepted hole fills.

    Key insight
    - The marker pairs only transmit colors to frames; the hole geometry of each crossed frame is the gate that decides whether the transmitted color survives in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _extract_frames(g)
    segments = _extract_segments(g)
    assert len(frames) == len(segments), "expected one marker pair per frame"

    frame_to_color: Dict[int, int] = {}
    for segment in segments:
        matches = [idx for idx, frame in enumerate(frames) if _crosses_frame(segment, frame["bbox"])]
        assert len(matches) == 1, "each segment must cross exactly one frame"
        idx = matches[0]
        assert idx not in frame_to_color, "each frame must be matched by exactly one segment"
        frame_to_color[idx] = int(segment["color"])
    assert len(frame_to_color) == len(frames)

    out = np.zeros_like(g)
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        gray_mask = frame["gray_mask"]
        out[r0 : r1 + 1, c0 : c1 + 1][gray_mask] = FRAME
        hole_mask = frame["hole_mask"]
        if _solid_square(hole_mask):
            out[r0 + 1 : r1, c0 + 1 : c1][hole_mask] = frame_to_color[idx]
    return out


def _choose_dims(shape_kind: str) -> Tuple[int, int]:
    if shape_kind == "square":
        side = random.randint(FRAME_OUTER_MIN, FRAME_OUTER_MAX)
        return side, side
    rects = [
        (h, w)
        for h in range(FRAME_OUTER_MIN, FRAME_OUTER_MAX + 1)
        for w in range(FRAME_OUTER_MIN, FRAME_OUTER_MAX + 1)
        if h != w
    ]
    return random.choice(rects)


def _make_square_fill_hole(ih: int, iw: int) -> np.ndarray:
    side = random.randint(1, min(ih, iw))
    top = random.randint(0, ih - side)
    left = random.randint(0, iw - side)
    mask = np.zeros((ih, iw), dtype=bool)
    mask[top : top + side, left : left + side] = True
    return mask


def _make_rect_fill_hole(ih: int, iw: int) -> np.ndarray:
    choices = [(h, w) for h in range(1, ih + 1) for w in range(1, iw + 1) if h != w and h * w >= 2]
    rect_h, rect_w = random.choice(choices)
    top = random.randint(0, ih - rect_h)
    left = random.randint(0, iw - rect_w)
    mask = np.zeros((ih, iw), dtype=bool)
    mask[top : top + rect_h, left : left + rect_w] = True
    return mask


def _make_square_not_full_hole(ih: int, iw: int) -> np.ndarray:
    side = random.randint(2, min(ih, iw))
    top = random.randint(0, ih - side)
    left = random.randint(0, iw - side)
    mask = np.zeros((ih, iw), dtype=bool)
    mask[top : top + side, left : left + side] = True
    corners = [
        (top, left),
        (top, left + side - 1),
        (top + side - 1, left),
        (top + side - 1, left + side - 1),
    ]
    rr, cc = random.choice(corners)
    mask[rr, cc] = False
    return mask


def _build_frame(kind: str, outer_shape: str) -> Dict[str, object]:
    h, w = _choose_dims(outer_shape)
    ih, iw = h - 2, w - 2
    if kind == "square_fill":
        hole_mask = _make_square_fill_hole(ih, iw)
    elif kind == "rect_fill":
        hole_mask = _make_rect_fill_hole(ih, iw)
    else:
        hole_mask = _make_square_not_full_hole(ih, iw)

    crop = np.full((h, w), FRAME, dtype=int)
    crop[1:-1, 1:-1][hole_mask] = BG
    return {
        "crop": crop,
        "hole_mask": hole_mask,
        "fillable": _solid_square(hole_mask),
        "outer_square": h == w,
    }


def _frame_blueprint(count: int) -> List[Tuple[str, str]]:
    specs: List[Tuple[str, str]] = [
        ("square_fill", "rect"),
        (random.choice(("rect_fill", "square_not_full")), "square"),
        ("square_not_full", random.choice(("square", "rect"))),
    ]
    if count >= 4 and "rect_fill" not in [kind for kind, _ in specs]:
        specs.append(("rect_fill", random.choice(("square", "rect"))))
    while len(specs) < count:
        specs.append(
            (
                random.choice(("square_fill", "rect_fill", "square_not_full")),
                random.choice(("square", "rect")),
            )
        )
    random.shuffle(specs)
    return specs


def _boxes_separated(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_frames(
    frames: Sequence[Dict[str, object]], height: int, width: int
) -> List[Tuple[int, int]] | None:
    order = sorted(range(len(frames)), key=lambda i: int(np.prod(frames[i]["crop"].shape)), reverse=True)
    placements: List[Tuple[int, int] | None] = [None] * len(frames)
    boxes: List[Tuple[int, int, int, int] | None] = [None] * len(frames)

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        crop = frames[idx]["crop"]
        h, w = crop.shape
        all_positions = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)]
        random.shuffle(all_positions)
        for r0, c0 in all_positions:
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(other is None or _boxes_separated(box, other) for other in boxes):
                placements[idx] = (r0, c0)
                boxes[idx] = box
                if dfs(pos + 1):
                    return True
                placements[idx] = None
                boxes[idx] = None
        return False

    if not dfs(0):
        return None
    return [tuple(p) for p in placements]  # type: ignore[arg-type]


def _sample_values(lo: int, hi: int) -> List[int]:
    values = {lo, hi, (lo + hi) // 2}
    if hi - lo >= 2:
        values.add(lo + 1)
        values.add(hi - 1)
    return sorted(v for v in values if lo <= v <= hi)


def _segment_candidates(
    boxes: Sequence[Tuple[int, int, int, int]], idx: int, height: int, width: int
) -> List[Dict[str, object]]:
    r0, c0, r1, c1 = boxes[idx]
    candidates: List[Dict[str, object]] = []

    for row in range(r0 + 1, r1):
        left_block = -1
        right_block = width
        blocked = False
        for j, (or0, oc0, or1, oc1) in enumerate(boxes):
            if j == idx or not (or0 <= row <= or1):
                continue
            if oc1 < c0:
                left_block = max(left_block, oc1)
            elif oc0 > c1:
                right_block = min(right_block, oc0)
            else:
                blocked = True
                break
        if blocked:
            continue
        left_lo, left_hi = left_block + 1, c0 - 1
        right_lo, right_hi = c1 + 1, right_block - 1
        if left_lo > left_hi or right_lo > right_hi:
            continue
        for left in _sample_values(left_lo, left_hi):
            for right in _sample_values(right_lo, right_hi):
                candidates.append(
                    {
                        "orientation": "h",
                        "fixed": row,
                        "lo": left,
                        "hi": right,
                        "endpoints": ((row, left), (row, right)),
                        "frame_idx": idx,
                    }
                )

    for col in range(c0 + 1, c1):
        top_block = -1
        bottom_block = height
        blocked = False
        for j, (or0, oc0, or1, oc1) in enumerate(boxes):
            if j == idx or not (oc0 <= col <= oc1):
                continue
            if or1 < r0:
                top_block = max(top_block, or1)
            elif or0 > r1:
                bottom_block = min(bottom_block, or0)
            else:
                blocked = True
                break
        if blocked:
            continue
        top_lo, top_hi = top_block + 1, r0 - 1
        bot_lo, bot_hi = r1 + 1, bottom_block - 1
        if top_lo > top_hi or bot_lo > bot_hi:
            continue
        for top in _sample_values(top_lo, top_hi):
            for bottom in _sample_values(bot_lo, bot_hi):
                candidates.append(
                    {
                        "orientation": "v",
                        "fixed": col,
                        "lo": top,
                        "hi": bottom,
                        "endpoints": ((top, col), (bottom, col)),
                        "frame_idx": idx,
                    }
                )

    deduped: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for cand in candidates:
        key = (cand["orientation"], cand["fixed"], cand["lo"], cand["hi"])
        deduped[key] = cand
    result = list(deduped.values())
    random.shuffle(result)
    return result


def _segments_intersect(a: Dict[str, object], b: Dict[str, object]) -> bool:
    if a["orientation"] == b["orientation"] == "h":
        return a["fixed"] == b["fixed"] and not (a["hi"] < b["lo"] or b["hi"] < a["lo"])
    if a["orientation"] == b["orientation"] == "v":
        return a["fixed"] == b["fixed"] and not (a["hi"] < b["lo"] or b["hi"] < a["lo"])
    if a["orientation"] == "h":
        h_seg, v_seg = a, b
    else:
        h_seg, v_seg = b, a
    return h_seg["lo"] <= v_seg["fixed"] <= h_seg["hi"] and v_seg["lo"] <= h_seg["fixed"] <= v_seg["hi"]


def _assign_segments(
    candidate_lists: Sequence[List[Dict[str, object]]]
) -> List[Dict[str, object]] | None:
    order = sorted(range(len(candidate_lists)), key=lambda i: len(candidate_lists[i]))
    chosen: List[Dict[str, object]] = []
    answer: List[Dict[str, object]] = [None] * len(candidate_lists)  # type: ignore[assignment]

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        candidates = candidate_lists[idx][:]
        random.shuffle(candidates)
        for cand in candidates:
            if all(not _segments_intersect(cand, other) for other in chosen):
                chosen.append(cand)
                answer[idx] = cand
                if dfs(pos + 1):
                    return True
                chosen.pop()
                answer[idx] = None  # type: ignore[assignment]
        return False

    if not dfs(0):
        return None
    return answer


def _render_input(
    shape: Tuple[int, int],
    frames: Sequence[Dict[str, object]],
    placements: Sequence[Tuple[int, int]],
    segments: Sequence[Dict[str, object]],
    colors: Sequence[int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for frame, (r0, c0) in zip(frames, placements):
        crop = frame["crop"]
        h, w = crop.shape
        grid[r0 : r0 + h, c0 : c0 + w] = crop
    for seg, color in zip(segments, colors):
        for r, c in seg["endpoints"]:
            assert grid[r, c] == BG
            grid[r, c] = color
    return grid


def _expected_output(
    shape: Tuple[int, int],
    frames: Sequence[Dict[str, object]],
    placements: Sequence[Tuple[int, int]],
    colors: Sequence[int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for frame, (r0, c0), color in zip(frames, placements, colors):
        crop = frame["crop"]
        h, w = crop.shape
        out[r0 : r0 + h, c0 : c0 + w][crop == FRAME] = FRAME
        if frame["fillable"]:
            out[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1][frame["hole_mask"]] = color
    return out


def _boxes_from_placements(
    frames: Sequence[Dict[str, object]], placements: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int, int, int]]:
    boxes = []
    for frame, (r0, c0) in zip(frames, placements):
        h, w = frame["crop"].shape
        boxes.append((r0, c0, r0 + h - 1, c0 + w - 1))
    return boxes


def _scene_validity_checks(
    frames: Sequence[Dict[str, object]],
    placements: Sequence[Tuple[int, int]],
    segments: Sequence[Dict[str, object]],
) -> bool:
    boxes = _boxes_from_placements(frames, placements)
    fillables = [bool(frame["fillable"]) for frame in frames]
    outer_squares = [bool(frame["outer_square"]) for frame in frames]
    if not any(fillables) or all(fillables):
        return False
    if not any(fillables[i] and not outer_squares[i] for i in range(len(frames))):
        return False
    if not any((not fillables[i]) and outer_squares[i] for i in range(len(frames))):
        return False
    for idx, seg in enumerate(segments):
        matches = [j for j, box in enumerate(boxes) if _crosses_frame(seg, box)]
        if matches != [idx]:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    frame_count = random.randint(*FRAME_COUNT_RANGE)

    frames = [_build_frame(kind, outer_shape) for kind, outer_shape in _frame_blueprint(frame_count)]
    placements = _place_frames(frames, height, width)
    if placements is None:
        return None

    boxes = _boxes_from_placements(frames, placements)
    candidate_lists = [_segment_candidates(boxes, idx, height, width) for idx in range(frame_count)]
    if any(not candidates for candidates in candidate_lists):
        return None

    segments = _assign_segments(candidate_lists)
    if segments is None:
        return None

    colors = random.sample(MARKER_COLORS, frame_count)
    inp = _render_input((height, width), frames, placements, segments, colors)
    expected = _expected_output((height, width), frames, placements, colors)

    if not _scene_validity_checks(frames, placements, segments):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
