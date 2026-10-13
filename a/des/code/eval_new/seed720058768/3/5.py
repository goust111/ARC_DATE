import random
from collections import deque

import numpy as np


BG = 0
DIRS = ("up", "down", "left", "right")
GEN_RETRIES = 400
CANVAS_SIZE = 30


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _gap_run(side: np.ndarray) -> tuple[int, int] | None:
    missing = [idx for idx, val in enumerate(side.tolist()) if not val]
    if not missing:
        return None
    if missing[0] == 0 or missing[-1] == len(side) - 1:
        return None
    for i in range(len(missing) - 1):
        if missing[i + 1] != missing[i] + 1:
            return None
    return missing[0], len(missing)


def _parse_frame_component(coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True

    if mask[1:-1, 1:-1].any():
        return None

    sides = {
        "up": mask[0, :],
        "down": mask[-1, :],
        "left": mask[:, 0],
        "right": mask[:, -1],
    }
    gap_info: dict[str, tuple[int, int]] = {}
    for direction, side in sides.items():
        run = _gap_run(side)
        if run is None:
            if not side.all():
                return None
            continue
        start, gap_len = run
        if gap_len not in (2, 3):
            return None
        gap_center = start + (gap_len - 1) / 2.0
        side_center = (len(side) - 1) / 2.0
        if abs(gap_center - side_center) > 0.5:
            return None
        gap_info[direction] = run

    if len(gap_info) != 1:
        return None

    gap_dir = next(iter(gap_info))
    gap_start, gap_len = gap_info[gap_dir]
    expected = 2 * h + 2 * w - 4 - gap_len
    if int(mask.sum()) != expected:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "h": h,
        "w": w,
        "gap_dir": gap_dir,
        "gap_start": gap_start,
        "gap_len": gap_len,
    }


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        for comp in _components(grid == color):
            parsed = _parse_frame_component(comp)
            if parsed is None:
                continue
            parsed["color"] = color
            frames.append(parsed)
    assert frames, "expected at least one gapped frame"
    return frames


def _corner_distance(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return min(
        r0 + c0,
        r0 + (w - 1 - c1),
        (h - 1 - r1) + c0,
        (h - 1 - r1) + (w - 1 - c1),
    )


def _find_legend_mapping(grid: np.ndarray, frame_colors: set[int]) -> dict[int, int]:
    candidates: list[tuple[int, dict[int, int]]] = []
    for comp in _components(grid != BG):
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if sub.shape[0] != 2:
            continue
        if len(comp) != sub.size:
            continue
        if (sub == BG).any():
            continue
        top = [int(v) for v in sub[0]]
        bottom = [int(v) for v in sub[1]]
        if len(top) != len(frame_colors):
            continue
        if set(top) != frame_colors or len(set(top)) != len(top):
            continue
        mapping = {int(a): int(b) for a, b in zip(top, bottom)}
        candidates.append((_corner_distance(grid.shape, (r0, c0, r1, c1)), mapping))

    assert candidates, "expected a 2xK corner legend"
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _draw_frame(canvas: np.ndarray, frame: dict, top: int, left: int) -> None:
    h = int(frame["h"])
    w = int(frame["w"])
    color = int(frame["color"])
    r0, c0 = top, left
    r1, c1 = r0 + h - 1, c0 + w - 1

    canvas[r0, c0 : c1 + 1] = color
    canvas[r1, c0 : c1 + 1] = color
    canvas[r0 : r1 + 1, c0] = color
    canvas[r0 : r1 + 1, c1] = color

    gap_start = int(frame["gap_start"])
    gap_len = int(frame["gap_len"])
    if frame["gap_dir"] == "up":
        canvas[r0, c0 + gap_start : c0 + gap_start + gap_len] = BG
    elif frame["gap_dir"] == "down":
        canvas[r1, c0 + gap_start : c0 + gap_start + gap_len] = BG
    elif frame["gap_dir"] == "left":
        canvas[r0 + gap_start : r0 + gap_start + gap_len, c0] = BG
    else:
        canvas[r0 + gap_start : r0 + gap_start + gap_len, c1] = BG


def _nested_layout(frames: list[dict]) -> tuple[int, int, list[dict]]:
    ordered = sorted(frames, key=lambda f: int(f["h"]) * int(f["w"]), reverse=True)
    outer = ordered[0]
    out_h = int(outer["h"]) + 2
    out_w = int(outer["w"]) + 2
    placed: list[dict] = []
    for frame in ordered:
        top = 1 + (int(outer["h"]) - int(frame["h"])) // 2
        left = 1 + (int(outer["w"]) - int(frame["w"])) // 2
        item = dict(frame)
        item["placed_bbox"] = (top, left, top + int(frame["h"]) - 1, left + int(frame["w"]) - 1)
        placed.append(item)
    return out_h, out_w, placed


def _in_half(r: int, c: int, direction: str, row_sum: int, col_sum: int) -> bool:
    if direction == "up":
        return 2 * r <= row_sum
    if direction == "down":
        return 2 * r >= row_sum
    if direction == "left":
        return 2 * c <= col_sum
    return 2 * c >= col_sum


def _render_output(frames: list[dict], mapping: dict[int, int]) -> np.ndarray:
    out_h, out_w, placed = _nested_layout(frames)
    out = np.full((out_h, out_w), BG, dtype=int)

    inner = placed[-1]
    ir0, ic0, ir1, ic1 = inner["placed_bbox"]
    out[ir0 + 1 : ir1, ic0 + 1 : ic1] = int(mapping[int(inner["color"])])

    for idx in range(len(placed) - 2, -1, -1):
        outer = placed[idx]
        inner = placed[idx + 1]
        fill_color = int(mapping[int(outer["color"])])
        or0, oc0, or1, oc1 = outer["placed_bbox"]
        ir0, ic0, ir1, ic1 = inner["placed_bbox"]
        row_sum = ir0 + ir1
        col_sum = ic0 + ic1
        for r in range(or0 + 1, or1):
            for c in range(oc0 + 1, oc1):
                if ir0 <= r <= ir1 and ic0 <= c <= ic1:
                    continue
                if _in_half(r, c, inner["gap_dir"], row_sum, col_sum):
                    out[r, c] = fill_color

    for frame in placed:
        r0, c0, _, _ = frame["placed_bbox"]
        _draw_frame(out, frame, r0, c0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered gapped rectangular frames into a centered nest and color each ring from the inside out.

    Input structure:
    - bg = black. The grid contains several disconnected rectangular hollow frames with border width 1.
    - Each frame uses a unique border color and has exactly one centered 2-3 cell gap on one side.
    - A 2×K legend in a corner maps frame border colors on the first row to fill colors on the second row.

    Rule:
    - Detect every gapped frame and sort them by bounding-box area from largest to smallest.
    - Rebuild the frames concentrically, keeping each frame's size, border color, and gap direction, and place the largest frame with a 1-cell outer margin.
    - Fill the innermost frame interior with its mapped fill color.
    - For every other frame, use the next smaller inner frame's gap direction to choose which half of the ring between them is filled with the current frame's mapped fill color; the opposite half stays black.

    Key insight:
    - The legend provides the color mapping, but the crucial dependency is reversed: each inner frame's gap orientation controls how the immediately surrounding outer ring is filled.
    """
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    mapping = _find_legend_mapping(g, {int(frame["color"]) for frame in frames})
    return _render_output(frames, mapping)


def _parity_values(lo: int, hi: int, parity: int) -> list[int]:
    return [v for v in range(lo, hi + 1) if v % 2 == parity]


def _gap_length(frame: dict) -> int:
    axis_len = int(frame["w"]) if frame["gap_dir"] in ("up", "down") else int(frame["h"])
    return 2 if axis_len % 2 == 0 else 3


def _make_frame(h: int, w: int, color: int, gap_dir: str) -> dict:
    axis_len = w if gap_dir in ("up", "down") else h
    gap_len = 2 if axis_len % 2 == 0 else 3
    return {
        "h": h,
        "w": w,
        "color": int(color),
        "gap_dir": gap_dir,
        "gap_len": gap_len,
        "gap_start": (axis_len - gap_len) // 2,
    }


def _size_sequence(rng: random.Random, count: int, parity: int, axis: str) -> list[int] | None:
    min_inner = 4 if parity == 0 else 5
    if count == 2:
        inner_hi = 12 if axis == "w" else 11
        inner = rng.choice(_parity_values(min_inner, inner_hi, parity))
        choices = [d for d in (4, 6, 8) if inner + d <= (20 if parity == 0 else 21)]
        if not choices:
            return None
        return [inner, inner + rng.choice(choices)]

    limit = 16 if parity == 0 else 17
    inner_hi = 8 if parity == 0 else 9
    inner = rng.choice(_parity_values(min_inner, inner_hi, parity))
    sizes = [inner]
    cur = inner
    for step in range(2):
        remaining_steps = 1 - step
        min_future = 4 * remaining_steps
        choices = [d for d in (4, 6) if cur + d + min_future <= limit]
        if not choices:
            return None
        cur += rng.choice(choices)
        sizes.append(cur)
    return sizes


def _legend_bbox(corner: str, k: int) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, 1, k - 1
    if corner == "tr":
        return 0, CANVAS_SIZE - k, 1, CANVAS_SIZE - 1
    if corner == "bl":
        return CANVAS_SIZE - 2, 0, CANVAS_SIZE - 1, k - 1
    return CANVAS_SIZE - 2, CANVAS_SIZE - k, CANVAS_SIZE - 1, CANVAS_SIZE - 1


def _too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _layout_relative_boxes(rng: random.Random, frames: list[dict]) -> list[tuple[dict, int, int]]:
    count = len(frames)
    shuffled = list(frames)
    rng.shuffle(shuffled)
    if count == 2:
        first, second = shuffled
        gap = rng.randint(2, 4)
        if rng.random() < 0.5:
            rel = [(first, 0, 0), (second, 0, int(first["w"]) + gap)]
        else:
            rel = [(first, 0, 0), (second, int(first["h"]) + gap, 0)]
        return rel

    ordered = sorted(frames, key=lambda f: int(f["h"]) * int(f["w"]), reverse=True)
    largest, medium, smallest = ordered
    vgap = rng.randint(2, 4)
    hgap = rng.randint(2, 4)
    medium_top = int(largest["h"]) + vgap if rng.random() < 0.5 else -(int(medium["h"]) + vgap)
    small_left = int(largest["w"]) + hgap if rng.random() < 0.5 else -(int(smallest["w"]) + hgap)
    rel = [(largest, 0, 0), (medium, medium_top, 0), (smallest, 0, small_left)]
    min_r = min(top for _, top, _ in rel)
    min_c = min(left for _, _, left in rel)
    return [(frame, top - min_r, left - min_c) for frame, top, left in rel]


def _candidate_placements(rel_boxes: list[tuple[dict, int, int]], legend_k: int) -> list[tuple[int, int, str]]:
    block_h = max(top + int(frame["h"]) for frame, top, _ in rel_boxes)
    block_w = max(left + int(frame["w"]) for frame, _, left in rel_boxes)
    candidates: list[tuple[int, int, str]] = []
    for top in range(CANVAS_SIZE - block_h + 1):
        for left in range(CANVAS_SIZE - block_w + 1):
            abs_boxes = []
            for frame, rt, lc in rel_boxes:
                r0 = top + rt
                c0 = left + lc
                abs_boxes.append((r0, c0, r0 + int(frame["h"]) - 1, c0 + int(frame["w"]) - 1))
            for corner in ("tl", "tr", "bl", "br"):
                lb = _legend_bbox(corner, legend_k)
                if all(not _too_close(box, lb, pad=1) for box in abs_boxes):
                    candidates.append((top, left, corner))
    return candidates


def _draw_legend(canvas: np.ndarray, corner: str, border_colors: list[int], fill_colors: list[int]) -> None:
    k = len(border_colors)
    r0, c0, _, _ = _legend_bbox(corner, k)
    canvas[r0, c0 : c0 + k] = np.array(border_colors, dtype=int)
    canvas[r0 + 1, c0 : c0 + k] = np.array(fill_colors, dtype=int)


def _crop_with_corner(canvas: np.ndarray, corner: str, rng: random.Random) -> np.ndarray:
    pts = np.argwhere(canvas != BG)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    extra_a = rng.randint(0, 2)
    extra_b = rng.randint(0, 2)

    if corner == "tl":
        top = 0
        left = 0
        bottom = min(CANVAS_SIZE, int(r1) + 1 + extra_a)
        right = min(CANVAS_SIZE, int(c1) + 1 + extra_b)
    elif corner == "tr":
        top = 0
        right = CANVAS_SIZE
        bottom = min(CANVAS_SIZE, int(r1) + 1 + extra_a)
        left = max(0, int(c0) - extra_b)
    elif corner == "bl":
        bottom = CANVAS_SIZE
        left = 0
        top = max(0, int(r0) - extra_a)
        right = min(CANVAS_SIZE, int(c1) + 1 + extra_b)
    else:
        bottom = CANVAS_SIZE
        right = CANVAS_SIZE
        top = max(0, int(r0) - extra_a)
        left = max(0, int(c0) - extra_b)

    return canvas[top:bottom, left:right]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.choice([2, 3])
    parity_h = rng.choice([0, 1])
    parity_w = rng.choice([0, 1])

    heights = _size_sequence(rng, count, parity_h, "h")
    widths = _size_sequence(rng, count, parity_w, "w")
    if heights is None or widths is None:
        return None

    sizes = list(zip(heights, widths))
    colors = rng.sample(range(1, 10), 2 * count)
    border_colors = colors[:count]
    fill_colors = colors[count:]
    mapping = {int(border): int(fill) for border, fill in zip(border_colors, fill_colors)}

    frames: list[dict] = []
    for (h, w), color in zip(sizes, border_colors):
        gap_dir = rng.choice(DIRS)
        frames.append(_make_frame(int(h), int(w), int(color), gap_dir))

    expected = _render_output(frames, mapping)
    if max(expected.shape) > 30:
        return None

    rel_boxes = _layout_relative_boxes(rng, frames)
    placements = _candidate_placements(rel_boxes, count)
    if not placements:
        return None
    block_top, block_left, corner = rng.choice(placements)

    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE), BG, dtype=int)
    for frame, rel_top, rel_left in rel_boxes:
        _draw_frame(canvas, frame, block_top + rel_top, block_left + rel_left)

    legend_order = list(range(count))
    rng.shuffle(legend_order)
    legend_border = [int(border_colors[i]) for i in legend_order]
    legend_fill = [int(fill_colors[i]) for i in legend_order]
    _draw_legend(canvas, corner, legend_border, legend_fill)

    grid = _crop_with_corner(canvas, corner, rng)
    if max(grid.shape) > 30:
        return None

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
