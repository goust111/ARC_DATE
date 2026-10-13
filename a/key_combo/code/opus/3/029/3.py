from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path(__file__).resolve().parents[6] / "new" / "solvers"
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components


BG = Color.BLACK
WALL = Color.GREY
EXTERNAL_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE)
INTERNAL_POOL = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)

FRAME_COUNT_CHOICES = (4, 5)
FRAME_HEIGHT_RANGE = (5, 7)
FRAME_WIDTH_RANGE = (6, 8)
FRAME_GAP = 2
OUTER_MARGIN_RANGE = (1, 3)
CENTER_BAND_RANGE = (6, 8)
RECT_HEIGHT_CHOICES = (2, 3)
RECT_WIDTH_CHOICES = (2, 3, 4)
RECT_PADDING = 1
MAX_GENERATE_ATTEMPTS = 500


def _border_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _side_cells(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    if side == "right":
        return [(r, c1) for r in range(r0 + 1, r1)]
    raise ValueError(f"unknown side: {side}")


def _gap_direction(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0:
        return -1, 0
    if r == r1:
        return 1, 0
    if c == c0:
        return 0, -1
    if c == c1:
        return 0, 1
    raise ValueError("gap is not on the frame border")


def _frame_bbox_mask(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _detect_frames(grid: np.ndarray) -> list[dict]:
    frames = find_components(
        grid,
        bg=BG,
        monochromatic=True,
        colors=[WALL],
        connectivity=4,
        return_format="dict",
    )
    parsed: list[dict] = []
    for comp in frames:
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        border = _border_cells(comp["bbox"])
        missing = [(r, c) for r, c in border if int(grid[r, c]) != WALL]
        if len(missing) != 1:
            continue
        gap = missing[0]
        if gap in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == WALL):
            continue
        if comp["area"] != len(border) - 1:
            continue
        if not all(r in (r0, r1) or c in (c0, c1) for r, c in comp["coords"]):
            continue

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        inner_colors = sorted(set(map(int, np.unique(inner))) - {BG, WALL})
        if len(inner_colors) != 2:
            continue

        parsed.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "step": _gap_direction(gap, comp["bbox"]),
                "inner_colors": inner_colors,
            }
        )

    parsed.sort(key=lambda frame: frame["bbox"])
    return parsed


def _detect_external_rectangles(grid: np.ndarray, frames: list[dict]) -> list[dict]:
    frame_mask = _frame_bbox_mask(grid.shape, frames)
    rectangles: list[dict] = []
    for color in EXTERNAL_COLORS:
        components = find_components(
            grid,
            bg=BG,
            monochromatic=True,
            colors=[color],
            connectivity=4,
            return_format="dict",
        )
        for comp in components:
            coords = comp["coords"]
            if frame_mask[coords[:, 0], coords[:, 1]].any():
                continue
            r0, c0, r1, c1 = comp["bbox"]
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(patch == color):
                continue
            rectangles.append(comp)
    rectangles.sort(key=lambda rect: (rect["bbox"], rect["color"]))
    return rectangles


def _dominant_external_color(rectangles: list[dict]) -> int:
    counts = Counter(int(rect["color"]) for rect in rectangles)
    if not counts:
        raise ValueError("no external rectangles")
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        raise ValueError("dominant external color is not unique")
    return int(ranked[0][0])


def _analyze_input(grid: np.ndarray) -> tuple[list[dict], list[dict], int]:
    frames = _detect_frames(grid)
    if not (4 <= len(frames) <= 5):
        raise ValueError("expected 4-5 rectangular frames")
    rectangles = _detect_external_rectangles(grid, frames)
    dominant = _dominant_external_color(rectangles)
    return frames, rectangles, dominant


def main(grid: np.ndarray) -> np.ndarray:
    """Use external rectangle frequency to choose either in-frame color swapping or gap-ray leakage.

    Input structure:
    - Black background contains 4-5 gray one-pixel-thick rectangular frames, each with exactly one non-corner border gap.
    - Every frame interior contains exactly two non-black, non-gray colors, possibly separated by black cells.
    - Outside the frames sit several solid monochrome rectangles whose colors are drawn from a visible palette.

    Rule:
    - Count the outside rectangles by object, not by area, and take the unique most frequent rectangle color as the global gate color.
    - For each frame, if its interior does not contain the gate color, swap its two interior colors and seal the gap with gray.
    - If its interior does contain the gate color, erase that color to black, keep the other interior color, and paint the gap plus the straight outward ray from that gap with the surviving color.

    Key insight:
    - The same local frame pattern splits into two different behaviors, and the split is controlled by a global frequency signal that comes from separate outside rectangles.
    """
    g = np.asarray(grid, dtype=int)
    frames, _rectangles, dominant = _analyze_input(g)

    out = g.copy()
    h, w = out.shape
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        gap_r, gap_c = frame["gap"]
        dr, dc = frame["step"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        a, b = frame["inner_colors"]

        if dominant in frame["inner_colors"]:
            survivor = b if a == dominant else a
            inner[inner == dominant] = BG
            out[gap_r, gap_c] = survivor
            rr, cc = gap_r + dr, gap_c + dc
            while 0 <= rr < h and 0 <= cc < w:
                if out[rr, cc] == BG:
                    out[rr, cc] = survivor
                rr += dr
                cc += dc
        else:
            mask_a = inner == a
            mask_b = inner == b
            inner[mask_a] = b
            inner[mask_b] = a
            out[gap_r, gap_c] = WALL

    return out


def _distribute_extra(extra: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(extra):
        parts[random.randrange(slots)] += 1
    return parts


def _axis_positions(total: int, lengths: list[int], margin_lo: int, margin_hi: int, gap: int) -> list[int] | None:
    required = margin_lo + margin_hi + sum(lengths) + gap * (len(lengths) - 1)
    if required > total:
        return None
    parts = _distribute_extra(total - required, len(lengths) + 1)
    pos = margin_lo + parts[0]
    starts: list[int] = []
    for idx, length in enumerate(lengths):
        starts.append(pos)
        pos += length
        if idx + 1 < len(lengths):
            pos += gap + parts[idx + 1]
    return starts


def _make_two_color_pattern(height: int, width: int, color_a: int, color_b: int) -> np.ndarray | None:
    canvas = np.zeros((height, width), dtype=int)
    for color in (color_a, color_b):
        placed = 0
        target_rects = random.randint(1, 2)
        for _ in range(80):
            rh = random.randint(1, height)
            rw = random.randint(1, width)
            if rh * rw < 2 and height * width >= 6:
                continue
            rr = random.randint(0, height - rh)
            cc = random.randint(0, width - rw)
            if np.any(canvas[rr : rr + rh, cc : cc + rw] != BG):
                continue
            canvas[rr : rr + rh, cc : cc + rw] = color
            placed += 1
            if placed >= target_rects:
                break
        if placed == 0:
            return None

        for _ in range(random.randint(0, 2)):
            coords = np.argwhere(canvas == color)
            random.shuffle(coords)
            expanded = False
            for base_r, base_c in coords:
                for dr, dc in random.sample([(1, 0), (-1, 0), (0, 1), (0, -1)], 4):
                    rr = int(base_r + dr)
                    cc = int(base_c + dc)
                    if 0 <= rr < height and 0 <= cc < width and canvas[rr, cc] == BG:
                        canvas[rr, cc] = color
                        expanded = True
                        break
                if expanded:
                    break

    if np.count_nonzero(canvas == color_a) < 2 or np.count_nonzero(canvas == color_b) < 2:
        return None

    if not np.any(canvas == BG):
        candidates = np.argwhere(canvas != BG)
        random.shuffle(candidates)
        for rr, cc in candidates:
            color = int(canvas[rr, cc])
            if np.count_nonzero(canvas == color) > 2:
                canvas[rr, cc] = BG
                break

    return canvas


def _mark_blocked(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _render_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
    color_a: int,
    color_b: int,
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = bbox
    pattern = _make_two_color_pattern(r1 - r0 - 1, c1 - c0 - 1, color_a, color_b)
    if pattern is None:
        return None

    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL
    grid[r0 + 1 : r1, c0 + 1 : c1] = pattern

    gap = random.choice(_side_cells(bbox, side))
    grid[gap] = BG
    return gap


def _row_layout(frame_count: int) -> tuple[int, int, list[dict], tuple[int, int, int, int]] | None:
    top_n = frame_count // 2
    if frame_count == 5 and random.random() < 0.5:
        top_n += 1
    bottom_n = frame_count - top_n

    top_heights = [random.randint(*FRAME_HEIGHT_RANGE) for _ in range(top_n)]
    bottom_heights = [random.randint(*FRAME_HEIGHT_RANGE) for _ in range(bottom_n)]
    top_widths = [random.randint(*FRAME_WIDTH_RANGE) for _ in range(top_n)]
    bottom_widths = [random.randint(*FRAME_WIDTH_RANGE) for _ in range(bottom_n)]

    top_margin = random.randint(*OUTER_MARGIN_RANGE)
    bottom_margin = random.randint(*OUTER_MARGIN_RANGE)
    middle_gap = random.randint(*CENTER_BAND_RANGE)
    left_margin = random.randint(*OUTER_MARGIN_RANGE)
    right_margin = random.randint(*OUTER_MARGIN_RANGE)

    h_min = max(top_margin + max(top_heights) + middle_gap + max(bottom_heights) + bottom_margin, 22)
    w_min = max(
        left_margin + right_margin + sum(top_widths) + FRAME_GAP * (top_n - 1),
        left_margin + right_margin + sum(bottom_widths) + FRAME_GAP * (bottom_n - 1),
        22,
    )
    if h_min > 30 or w_min > 30:
        return None

    height = random.randint(h_min, min(30, h_min + 3))
    width = random.randint(w_min, min(30, w_min + 3))
    top_cols = _axis_positions(width, top_widths, left_margin, right_margin, FRAME_GAP)
    bottom_cols = _axis_positions(width, bottom_widths, left_margin, right_margin, FRAME_GAP)
    if top_cols is None or bottom_cols is None:
        return None

    frames: list[dict] = []
    for h_box, w_box, c0 in zip(top_heights, top_widths, top_cols, strict=False):
        frames.append({"bbox": (top_margin, c0, top_margin + h_box - 1, c0 + w_box - 1), "side": "top"})
    for h_box, w_box, c0 in zip(bottom_heights, bottom_widths, bottom_cols, strict=False):
        r0 = height - bottom_margin - h_box
        frames.append({"bbox": (r0, c0, r0 + h_box - 1, c0 + w_box - 1), "side": "bottom"})

    top_bottom = max(frame["bbox"][2] for frame in frames if frame["side"] == "top")
    bottom_top = min(frame["bbox"][0] for frame in frames if frame["side"] == "bottom")
    zone = (top_bottom + 2, 1, bottom_top - 2, width - 2)
    if zone[0] > zone[2] or zone[1] > zone[3] or zone[2] - zone[0] + 1 < 3:
        return None
    return height, width, frames, zone


def _column_layout(frame_count: int) -> tuple[int, int, list[dict], tuple[int, int, int, int]] | None:
    left_n = frame_count // 2
    if frame_count == 5 and random.random() < 0.5:
        left_n += 1
    right_n = frame_count - left_n

    left_heights = [random.randint(*FRAME_HEIGHT_RANGE) for _ in range(left_n)]
    right_heights = [random.randint(*FRAME_HEIGHT_RANGE) for _ in range(right_n)]
    left_widths = [random.randint(*FRAME_WIDTH_RANGE) for _ in range(left_n)]
    right_widths = [random.randint(*FRAME_WIDTH_RANGE) for _ in range(right_n)]

    top_margin = random.randint(*OUTER_MARGIN_RANGE)
    bottom_margin = random.randint(*OUTER_MARGIN_RANGE)
    left_margin = random.randint(*OUTER_MARGIN_RANGE)
    right_margin = random.randint(*OUTER_MARGIN_RANGE)
    middle_gap = random.randint(*CENTER_BAND_RANGE)

    h_min = max(
        top_margin + bottom_margin + sum(left_heights) + FRAME_GAP * (left_n - 1),
        top_margin + bottom_margin + sum(right_heights) + FRAME_GAP * (right_n - 1),
        22,
    )
    left_band = max(left_widths)
    right_band = max(right_widths)
    w_min = max(left_margin + left_band + middle_gap + right_band + right_margin, 22)
    if h_min > 30 or w_min > 30:
        return None

    height = random.randint(h_min, min(30, h_min + 3))
    width = random.randint(w_min, min(30, w_min + 3))
    left_rows = _axis_positions(height, left_heights, top_margin, bottom_margin, FRAME_GAP)
    right_rows = _axis_positions(height, right_heights, top_margin, bottom_margin, FRAME_GAP)
    if left_rows is None or right_rows is None:
        return None

    frames: list[dict] = []
    for r0, h_box, w_box in zip(left_rows, left_heights, left_widths, strict=False):
        offset = random.randint(0, left_band - w_box)
        c0 = left_margin + offset
        frames.append({"bbox": (r0, c0, r0 + h_box - 1, c0 + w_box - 1), "side": "left"})
    right_band_c0 = width - right_margin - right_band
    for r0, h_box, w_box in zip(right_rows, right_heights, right_widths, strict=False):
        offset = random.randint(0, right_band - w_box)
        c0 = right_band_c0 + offset
        frames.append({"bbox": (r0, c0, r0 + h_box - 1, c0 + w_box - 1), "side": "right"})

    left_right = max(frame["bbox"][3] for frame in frames if frame["side"] == "left")
    right_left = min(frame["bbox"][1] for frame in frames if frame["side"] == "right")
    zone = (1, left_right + 2, height - 2, right_left - 2)
    if zone[0] > zone[2] or zone[1] > zone[3] or zone[3] - zone[1] + 1 < 4:
        return None
    return height, width, frames, zone


def _external_count_plan() -> dict[int, int]:
    dominant = random.choice(EXTERNAL_COLORS)
    others = random.sample([c for c in EXTERNAL_COLORS if c != dominant], random.randint(2, 4))
    dominant_count = random.randint(3, 5)
    counts = {dominant: dominant_count}
    total = dominant_count
    for color in others:
        counts[color] = random.randint(1, min(2, dominant_count - 1))
        total += counts[color]
    if total > 10:
        return _external_count_plan()
    return counts


def _place_external_rectangles(
    grid: np.ndarray,
    zone: tuple[int, int, int, int],
    counts: dict[int, int],
    frame_bboxes: list[tuple[int, int, int, int]],
) -> bool:
    blocked = np.zeros(grid.shape, dtype=bool)
    for bbox in frame_bboxes:
        _mark_blocked(blocked, bbox, RECT_PADDING)

    zr0, zc0, zr1, zc1 = zone
    rect_colors: list[int] = []
    for color, count in counts.items():
        rect_colors.extend([color] * count)
    random.shuffle(rect_colors)

    for color in rect_colors:
        rect_h = random.choice(RECT_HEIGHT_CHOICES)
        rect_w = random.choice(RECT_WIDTH_CHOICES)
        candidates: list[tuple[int, int]] = []
        for r0 in range(zr0, zr1 - rect_h + 2):
            for c0 in range(zc0, zc1 - rect_w + 2):
                if blocked[r0 : r0 + rect_h, c0 : c0 + rect_w].any():
                    continue
                rr0 = max(0, r0 - RECT_PADDING)
                cc0 = max(0, c0 - RECT_PADDING)
                rr1 = min(grid.shape[0], r0 + rect_h + RECT_PADDING)
                cc1 = min(grid.shape[1], c0 + rect_w + RECT_PADDING)
                if blocked[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
        _mark_blocked(blocked, (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1), RECT_PADDING)
    return True


def _scene_attempt() -> np.ndarray | None:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    layout = _row_layout(frame_count) if random.random() < 0.5 else _column_layout(frame_count)
    if layout is None:
        return None
    height, width, frames, zone = layout
    grid = np.full((height, width), BG, dtype=int)

    counts = _external_count_plan()
    dominant = max(counts, key=counts.get)

    contains_mask = [False] * frame_count
    dominant_frame_count = random.randint(1, frame_count - 1)
    for idx in random.sample(range(frame_count), dominant_frame_count):
        contains_mask[idx] = True

    random.shuffle(frames)
    for idx, frame in enumerate(frames):
        if contains_mask[idx]:
            other = random.choice([color for color in INTERNAL_POOL if color != dominant])
            pair = (dominant, other)
        else:
            choices = [color for color in INTERNAL_POOL if color != dominant]
            pair = tuple(random.sample(choices, 2))
        gap = _render_frame(grid, frame["bbox"], frame["side"], pair[0], pair[1])
        if gap is None:
            return None
        frame["gap"] = gap
        frame["colors"] = pair

    if not _place_external_rectangles(grid, zone, counts, [frame["bbox"] for frame in frames]):
        return None

    return grid


def _validate_scene(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30

    frames, rectangles, dominant = _analyze_input(grid)
    counts = Counter(int(rect["color"]) for rect in rectangles)
    assert counts[dominant] == max(counts.values())
    assert list(counts.values()).count(counts[dominant]) == 1

    contains = [dominant in frame["inner_colors"] for frame in frames]
    assert any(contains)
    assert not all(contains)

    for idx, frame in enumerate(frames):
        ar0, ac0, ar1, ac1 = frame["bbox"]
        for other in frames[idx + 1 :]:
            br0, bc0, br1, bc1 = other["bbox"]
            row_sep = ar1 < br0 - 1 or br1 < ar0 - 1
            col_sep = ac1 < bc0 - 1 or bc1 < ac0 - 1
            assert row_sep or col_sep

    for frame in frames:
        rr, cc = frame["gap"]
        dr, dc = frame["step"]
        while 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
            assert int(grid[rr, cc]) == BG
            rr += dr
            cc += dc

    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        color = int(rect["color"])
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)


def _validate_transformation(inp: np.ndarray, out: np.ndarray) -> None:
    assert inp.shape == out.shape
    frames, rectangles, dominant = _analyze_input(inp)

    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        assert np.array_equal(inp[r0 : r1 + 1, c0 : c1 + 1], out[r0 : r1 + 1, c0 : c1 + 1])

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        gap_r, gap_c = frame["gap"]
        dr, dc = frame["step"]
        inner_in = inp[r0 + 1 : r1, c0 + 1 : c1]
        inner_out = out[r0 + 1 : r1, c0 + 1 : c1]
        a, b = frame["inner_colors"]

        for border_r, border_c in _border_cells(frame["bbox"]):
            if (border_r, border_c) == frame["gap"]:
                continue
            assert int(out[border_r, border_c]) == WALL

        if dominant in frame["inner_colors"]:
            survivor = b if a == dominant else a
            expected_inner = inner_in.copy()
            expected_inner[expected_inner == dominant] = BG
            assert np.array_equal(inner_out, expected_inner)
            assert int(out[gap_r, gap_c]) == survivor
            rr, cc = gap_r + dr, gap_c + dc
            while 0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]:
                assert int(out[rr, cc]) == survivor
                rr += dr
                cc += dc
            assert not np.array_equal(inner_in, inner_out)
        else:
            expected_inner = inner_in.copy()
            expected_inner[inner_in == a] = b
            expected_inner[inner_in == b] = a
            assert np.array_equal(inner_out, expected_inner)
            assert int(out[gap_r, gap_c]) == WALL
            assert not np.array_equal(inner_in, inner_out)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _scene_attempt()
        if grid is None:
            continue
        try:
            _validate_scene(grid)
            out = main(grid)
            _validate_transformation(grid, out)
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid scene")
