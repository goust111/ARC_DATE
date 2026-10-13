from __future__ import annotations

import random
import sys
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, colors_present, find_components, get_bg


BLUE = Color.BLUE
YELLOW = Color.YELLOW
GREY = Color.GREY
BLACK = Color.BLACK
FRAME_COLORS = [Color.RED, Color.GREEN, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

GRID_RANGE = (18, 24)
LINE_PATTERNS = [
    (1, 1, False),
    (2, 1, True),
    (1, 2, False),
    (2, 0, True),
    (0, 2, False),
    (1, 1, True),
]
SIZE_OPTIONS = [(5, 5), (3, 5), (5, 3), (3, 3)]
MAX_GENERATION_ATTEMPTS = 600

_SEEN_INPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_OUTPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()
_GEN_CALL_COUNT = 0


def _bands(length: int, lines: list[int]) -> list[tuple[int, int]]:
    cuts = [-1, *lines, length]
    return [(cuts[i] + 1, cuts[i + 1] - 1) for i in range(len(cuts) - 1)]


def _region_id(line_rows: list[int], line_cols: list[int], r: int, c: int) -> tuple[int, int]:
    assert r not in line_rows and c not in line_cols
    return bisect_right(line_rows, r), bisect_right(line_cols, c)


def _parse_guides(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    line_rows = [r for r in range(1, h - 1) if int(grid[r, 0]) == BLUE and int(grid[r, w - 1]) == BLUE]
    line_cols = [c for c in range(1, w - 1) if int(grid[0, c]) == BLUE and int(grid[h - 1, c]) == BLUE]
    total = len(line_rows) + len(line_cols)
    assert 2 <= total <= 3

    expected = {(r, 0) for r in line_rows} | {(r, w - 1) for r in line_rows}
    expected |= {(0, c) for c in line_cols} | {(h - 1, c) for c in line_cols}
    actual = {(int(r), int(c)) for r, c in np.argwhere(grid == BLUE)}
    assert actual == expected
    return line_rows, line_cols


def _is_hollow_rect(grid: np.ndarray, comp: dict) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    if hh < 3 or ww < 3:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((hh, ww), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(patch[border] == color):
        return False
    if np.any(patch[~border] == color):
        return False
    return int(comp["area"]) == int(border.sum())


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    line_rows, line_cols = _parse_guides(g)

    frame_colors = colors_present(g, exclude={bg, BLUE, YELLOW, GREY})
    comps = find_components(g, bg=bg, colors=frame_colors, connectivity=4, monochromatic=True)
    frames: list[dict] = []
    for comp in comps:
        assert _is_hollow_rect(g, comp)
        r0, c0, r1, c1 = map(int, comp["bbox"])
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        assert center[0] not in line_rows and center[1] not in line_cols
        frames.append(
            {
                "color": int(comp["color"]),
                "bbox": (r0, c0, r1, c1),
                "center": center,
                "bbox_marker_count": int(np.count_nonzero(g[r0 : r1 + 1, c0 : c1 + 1] == YELLOW)),
            }
        )

    assert 3 <= len(frames) <= 4
    assert len({frame["color"] for frame in frames}) == len(frames)

    region_counts: Counter[tuple[int, int]] = Counter()
    for r, c in np.argwhere(g == YELLOW):
        rr = int(r)
        cc = int(c)
        assert rr not in line_rows and cc not in line_cols
        region_counts[_region_id(line_rows, line_cols, rr, cc)] += 1

    summaries = []
    for frame in frames:
        cr, cc = frame["center"]
        region = _region_id(line_rows, line_cols, cr, cc)
        count = int(region_counts[region])
        assert count > 0
        frame["region"] = region
        frame["count"] = count
        summaries.append((count, cr, int(frame["color"])))

    assert len({(count, row) for count, row, _ in summaries}) == len(summaries)
    return {
        "bg": bg,
        "line_rows": line_rows,
        "line_cols": line_cols,
        "frames": frames,
        "summaries": summaries,
        "region_counts": dict(region_counts),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Summarize hollow frame colors by the yellow-marker counts of guide-line partitions.

    Input structure:
    - Black background contains 2-3 blue border-marker pairs, several colored hollow rectangles, and scattered yellow singleton markers.
    - Matching blue markers on opposite borders define full horizontal or vertical guide lines, while each colored frame is identified by its hollow rectangular border.

    Rule:
    - Extend every blue border-marker pair into a full blue guide line across the grid.
    - Treat those guide lines as partition boundaries, count yellow markers inside each resulting rectangular region, and assign each frame the count of the region containing its bbox center pixel.
    - Build a new grid with one row per frame, sorted by ascending assigned count and then ascending center row; fill each row with the frame color for exactly that many cells and pad the rest with black.

    Key insight:
    - The blue markers are only controls for an invisible partition system: yellow markers and colored frames become related solely through the region containing each frame center.
    """
    scene = _analyze_scene(np.asarray(grid, dtype=int))
    summaries = sorted(scene["summaries"], key=lambda item: (item[0], item[1]))
    width = max(count for count, _row, _color in summaries)
    out = np.full((len(summaries), width), scene["bg"], dtype=int)
    for row, (count, _center_row, color) in enumerate(summaries):
        out[row, :count] = color
    return out


def _choose_lines(length: int, count: int, rng: random.Random, min_band: int = 5) -> list[int] | None:
    if count == 0:
        return []
    candidates = list(range(2, length - 2))
    if len(candidates) < count:
        return None
    for _ in range(200):
        lines = sorted(rng.sample(candidates, count))
        gaps = [b - a - 1 for a, b in zip([-1, *lines], [*lines, length])]
        if min(gaps) >= min_band:
            return lines
    return None


def _region_boxes(h: int, w: int, line_rows: list[int], line_cols: list[int]) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for ri, (r0, r1) in enumerate(_bands(h, line_rows)):
        for ci, (c0, c1) in enumerate(_bands(w, line_cols)):
            boxes[(ri, ci)] = (r0, r1, c0, c1)
    return boxes


def _reserve(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _placement_candidates(
    region_box: tuple[int, int, int, int],
    size: tuple[int, int],
    reserved: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    r0, r1, c0, c1 = region_box
    hh, ww = size
    rmin = r0 + 1
    rmax = r1 - hh
    cmin = c0 + 1
    cmax = c1 - ww
    candidates: list[tuple[int, int, int, int]] = []
    if rmin > rmax or cmin > cmax:
        return candidates
    for rr in range(rmin, rmax + 1):
        for cc in range(cmin, cmax + 1):
            bbox = (rr, cc, rr + hh - 1, cc + ww - 1)
            rr0 = max(0, rr - 1)
            cc0 = max(0, cc - 1)
            rr1 = min(reserved.shape[0] - 1, rr + hh)
            cc1 = min(reserved.shape[1] - 1, cc + ww)
            if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            candidates.append(bbox)
    return candidates


def _place_frames_in_region(
    region_box: tuple[int, int, int, int],
    count: int,
    reserved: np.ndarray,
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    for _ in range(120):
        work = reserved.copy()
        bboxes: list[tuple[int, int, int, int]] = []
        sizes = [rng.choice(SIZE_OPTIONS) for _ in range(count)]
        sizes.sort(key=lambda item: item[0] * item[1], reverse=True)
        ok = True
        for size in sizes:
            candidates = _placement_candidates(region_box, size, work)
            if not candidates:
                ok = False
                break
            bbox = rng.choice(candidates)
            bboxes.append(bbox)
            _reserve(work, bbox)
        if not ok:
            continue
        centers = [((r0 + r1) // 2, (c0 + c1) // 2) for r0, c0, r1, c1 in bboxes]
        if len({center[0] for center in centers}) != len(centers):
            continue
        return bboxes
    return None


def _marker_candidates(
    h: int,
    w: int,
    region_box: tuple[int, int, int, int],
    all_bboxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    blocked = np.zeros((h, w), dtype=bool)
    for bbox in all_bboxes:
        _reserve(blocked, bbox)

    r0, r1, c0, c1 = region_box
    cells = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (0, h - 1) or c in (0, w - 1):
                continue
            if blocked[r, c]:
                continue
            cells.append((r, c))
    return cells


def _expected_output(frames: list[dict], region_counts: dict[tuple[int, int], int]) -> np.ndarray:
    order = sorted(frames, key=lambda frame: (region_counts[frame["region"]], frame["center"][0]))
    width = max(region_counts[frame["region"]] for frame in frames)
    out = np.zeros((len(frames), width), dtype=int)
    for row, frame in enumerate(order):
        out[row, : region_counts[frame["region"]]] = int(frame["color"])
    return out


def _scene_is_meaningful(grid: np.ndarray, expected: np.ndarray, require_tie: bool) -> bool:
    try:
        scene = _analyze_scene(grid)
    except AssertionError:
        return False

    frames = scene["frames"]
    counts = [int(frame["count"]) for frame in frames]
    regions = [frame["region"] for frame in frames]
    center_order = sorted(range(len(frames)), key=lambda idx: frames[idx]["center"][0])
    rule_order = sorted(range(len(frames)), key=lambda idx: (frames[idx]["count"], frames[idx]["center"][0]))

    if len(set(regions)) < 2:
        return False
    if len(set(counts)) < 2:
        return False
    if center_order == rule_order:
        return False
    if not any(int(frame["bbox_marker_count"]) == 0 and int(frame["count"]) > 0 for frame in frames):
        return False
    if require_tie and len(set(counts)) == len(counts):
        return False
    if np.count_nonzero(grid == BLACK) <= max(np.count_nonzero(grid == color) for color in FRAME_COLORS + [BLUE, YELLOW]):
        return False
    return np.array_equal(main(grid), expected)


def _gen_attempt(mode: tuple[int, int, bool]) -> tuple[np.ndarray, np.ndarray] | None:
    rng = random.Random()
    h = rng.randint(*GRID_RANGE)
    w = rng.randint(*GRID_RANGE)
    h_lines, v_lines, require_tie = mode

    line_rows = _choose_lines(h, h_lines, rng)
    line_cols = _choose_lines(w, v_lines, rng)
    if line_rows is None or line_cols is None:
        return None

    region_boxes = _region_boxes(h, w, line_rows, line_cols)
    region_keys = list(region_boxes)
    region_total = len(region_keys)
    frame_count = 4 if region_total >= 4 and rng.random() < 0.6 else 3
    if not require_tie:
        frame_count = min(frame_count, region_total)
    else:
        frame_count = min(frame_count, region_total + 1)
    if frame_count < 3:
        return None

    if require_tie:
        loads = [2] + [1] * (frame_count - 2)
    else:
        loads = [1] * frame_count
    if len(loads) > region_total:
        return None

    for _ in range(160):
        chosen_keys = rng.sample(region_keys, len(loads))
        chosen_keys.sort(key=lambda key: (region_boxes[key][1] - region_boxes[key][0], region_boxes[key][3] - region_boxes[key][2]), reverse=True)
        loads_sorted = sorted(loads, reverse=True)
        assignments = list(zip(chosen_keys, loads_sorted))

        reserved = np.zeros((h, w), dtype=bool)
        placed: list[tuple[tuple[int, int], tuple[int, int, int, int]]] = []
        ok = True
        for region_key, load in assignments:
            bboxes = _place_frames_in_region(region_boxes[region_key], load, reserved, rng)
            if not bboxes:
                ok = False
                break
            for bbox in bboxes:
                placed.append((region_key, bbox))
                _reserve(reserved, bbox)
        if not ok:
            continue

        colors = rng.sample(FRAME_COLORS, len(placed))
        frames = []
        all_bboxes = [bbox for _region, bbox in placed]
        for color, (region_key, bbox) in zip(colors, placed):
            r0, c0, r1, c1 = bbox
            frames.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "center": ((r0 + r1) // 2, (c0 + c1) // 2),
                    "region": region_key,
                }
            )

        candidates_by_region = {
            key: _marker_candidates(h, w, box, all_bboxes) for key, box in region_boxes.items()
        }
        used_regions = sorted({frame["region"] for frame in frames})
        if any(len(candidates_by_region[key]) == 0 for key in used_regions):
            continue

        region_counts: dict[tuple[int, int], int] = {}
        for _ in range(200):
            for key in used_regions:
                cap = min(6, len(candidates_by_region[key]))
                if cap < 1:
                    break
                region_counts[key] = rng.randint(1, cap)
            frame_counts = [region_counts[frame["region"]] for frame in frames]
            if len(set(frame_counts)) < 2:
                continue
            if len({(region_counts[frame["region"]], frame["center"][0]) for frame in frames}) != len(frames):
                continue
            if require_tie and len(set(frame_counts)) == len(frame_counts):
                continue
            center_order = sorted(frames, key=lambda frame: frame["center"][0])
            rule_order = sorted(frames, key=lambda frame: (region_counts[frame["region"]], frame["center"][0]))
            if [frame["bbox"] for frame in center_order] == [frame["bbox"] for frame in rule_order]:
                continue
            break
        else:
            continue

        grid = np.zeros((h, w), dtype=int)
        for r in line_rows:
            grid[r, 0] = BLUE
            grid[r, w - 1] = BLUE
        for c in line_cols:
            grid[0, c] = BLUE
            grid[h - 1, c] = BLUE

        for frame in frames:
            color = int(frame["color"])
            r0, c0, r1, c1 = frame["bbox"]
            grid[r0, c0 : c1 + 1] = color
            grid[r1, c0 : c1 + 1] = color
            grid[r0 : r1 + 1, c0] = color
            grid[r0 : r1 + 1, c1] = color

        for key in used_regions:
            cells = candidates_by_region[key][:]
            rng.shuffle(cells)
            for r, c in cells[: region_counts[key]]:
                grid[r, c] = YELLOW

        spare_regions = [key for key in region_boxes if key not in used_regions and candidates_by_region[key]]
        rng.shuffle(spare_regions)
        for key in spare_regions[: rng.randint(0, min(2, len(spare_regions)))]:
            cells = candidates_by_region[key][:]
            rng.shuffle(cells)
            extra = rng.randint(1, min(3, len(cells)))
            for r, c in cells[:extra]:
                grid[r, c] = YELLOW

        expected = _expected_output(frames, region_counts)
        if not _scene_is_meaningful(grid, expected, require_tie):
            continue
        return grid, expected
    return None


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    for attempt in range(MAX_GENERATION_ATTEMPTS):
        mode = LINE_PATTERNS[(_GEN_CALL_COUNT + attempt) % len(LINE_PATTERNS)]
        result = _gen_attempt(mode)
        if result is None:
            continue
        grid, expected = result
        in_sig = (grid.shape, grid.tobytes())
        out_sig = (expected.shape, expected.tobytes())
        if in_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
            continue
        _SEEN_INPUT_SIGS.add(in_sig)
        _SEEN_OUTPUT_SIGS.add(out_sig)
        _GEN_CALL_COUNT += 1
        return grid

    raise RuntimeError("failed to generate a valid input")
