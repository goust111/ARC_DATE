import random
from collections import Counter, defaultdict, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

BG = 0
FRAME = 5
ARROW_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
GRID_MIN = 20
GRID_MAX = 30
SAFE_MARGIN = 3
INNER_MARGIN = 1


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in dirs:
                    nr, nc = int(cr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _rect_border(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((int(r0), int(c)))
        pts.add((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        pts.add((int(r), int(c0)))
        pts.add((int(r), int(c1)))
    return pts


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        for comp in _components(grid == int(color), connectivity=4):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            border = _rect_border((r0, c0, r1, c1))
            pts = set(comp)
            if pts != border:
                continue
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0 or not bool(np.all(interior == int(bg))):
                continue
            frames.append(
                {
                    "color": int(color),
                    "box": (int(r0), int(c0), int(r1), int(c1)),
                    "border": border,
                }
            )
    frames.sort(key=lambda frame: frame["box"])
    return frames


def _parse_arrow(points: list[tuple[int, int]]) -> dict | None:
    pts = set((int(r), int(c)) for r, c in points)
    matches: list[dict] = []

    rows = sorted({r for r, _ in pts})
    for row in rows:
        cols = sorted(c for r, c in pts if r == row)
        if len(cols) < 2 or cols != list(range(cols[0], cols[-1] + 1)):
            continue
        shaft = {(int(row), int(c)) for c in range(cols[0], cols[-1] + 1)}
        right = shaft | {(int(row - 1), int(cols[-1] + 1)), (int(row + 1), int(cols[-1] + 1))}
        if right == pts:
            matches.append(
                {
                    "direction": (0, 1),
                    "length": len(shaft),
                    "front": (int(row), int(cols[-1] + 1)),
                    "cells": pts,
                }
            )
        left = shaft | {(int(row - 1), int(cols[0] - 1)), (int(row + 1), int(cols[0] - 1))}
        if left == pts:
            matches.append(
                {
                    "direction": (0, -1),
                    "length": len(shaft),
                    "front": (int(row), int(cols[0] - 1)),
                    "cells": pts,
                }
            )

    cols = sorted({c for _, c in pts})
    for col in cols:
        rows_here = sorted(r for r, c in pts if c == col)
        if len(rows_here) < 2 or rows_here != list(range(rows_here[0], rows_here[-1] + 1)):
            continue
        shaft = {(int(r), int(col)) for r in range(rows_here[0], rows_here[-1] + 1)}
        down = shaft | {(int(rows_here[-1] + 1), int(col - 1)), (int(rows_here[-1] + 1), int(col + 1))}
        if down == pts:
            matches.append(
                {
                    "direction": (1, 0),
                    "length": len(shaft),
                    "front": (int(rows_here[-1] + 1), int(col)),
                    "cells": pts,
                }
            )
        up = shaft | {(int(rows_here[0] - 1), int(col - 1)), (int(rows_here[0] - 1), int(col + 1))}
        if up == pts:
            matches.append(
                {
                    "direction": (-1, 0),
                    "length": len(shaft),
                    "front": (int(rows_here[0] - 1), int(col)),
                    "cells": pts,
                }
            )

    unique = {(m["direction"], m["length"], m["front"]) for m in matches}
    if len(unique) != 1:
        return None
    return matches[0]


def _trace_target(
    grid: np.ndarray,
    bg: int,
    border_to_frame: dict[tuple[int, int], int],
    front: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = grid.shape
    dr, dc = direction
    r, c = front
    gap = 0
    while 0 <= r < h and 0 <= c < w:
        if (int(r), int(c)) in border_to_frame:
            return border_to_frame[(int(r), int(c))], int(gap)
        if int(grid[r, c]) != int(bg):
            return None
        gap += 1
        r += dr
        c += dc
    return None


def _paint_single_ring(out: np.ndarray, top: int, left: int, bottom: int, right: int, color: int) -> None:
    out[top, left : right + 1] = int(color)
    out[bottom, left : right + 1] = int(color)
    out[top : bottom + 1, left] = int(color)
    out[top : bottom + 1, right] = int(color)


def _paint_layers(out: np.ndarray, box: tuple[int, int, int, int], layers: list[tuple[int, int]]) -> None:
    r0, c0, r1, c1 = box
    inset = 0
    for color, count in layers:
        for _ in range(int(count)):
            top = int(r0 + 1 + inset)
            left = int(c0 + 1 + inset)
            bottom = int(r1 - 1 - inset)
            right = int(c1 - 1 - inset)
            if top > bottom or left > right:
                return
            _paint_single_ring(out, top, left, bottom, right, int(color))
            inset += 1


def _layer_capacity(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    return (min(inner_h, inner_w) + 1) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase color-coded arrows and fill the pointed hollow rectangles with colored inner rings.

    Input structure:
    - A black background contains several grey 1-pixel-thick hollow rectangular frames.
    - Each non-grey foreground component is a monochrome arrow made from a straight 1-pixel shaft and two diagonal head pixels.
    - Every arrow points horizontally or vertically toward exactly one frame, and the first frame wall reached along that direction is its target.

    Rule:
    - For each arrow, identify its direction from the shaft-plus-head shape and count the shaft pixels only.
    - Follow the arrow direction from the tip axis to the first frame wall to determine the target frame, and use the arrow color as the fill color.
    - Fill the target frame interior with that many concentric inner rectangular rings; if several arrows target the same frame, apply them from smaller arrow-to-frame gap to larger gap so nearer arrows claim the outer layers first.
    - Remove all arrows from the output while keeping the original grey frame borders.

    Key insight:
    - Each arrow simultaneously encodes target, color, and ring count: geometry gives direction, color gives paint, and shaft length gives how many layers to consume.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(int(v) for v in g.ravel()).most_common(1)[0][0])
    frames = _find_frames(g, bg)

    border_to_frame: dict[tuple[int, int], int] = {}
    frame_cells: set[tuple[int, int]] = set()
    frame_signatures = {frozenset(frame["border"]) for frame in frames}
    for idx, frame in enumerate(frames):
        for cell in frame["border"]:
            border_to_frame[cell] = idx
        frame_cells |= frame["border"]

    arrows: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        for comp in _components(g == int(color), connectivity=8):
            pts = frozenset((int(r), int(c)) for r, c in comp)
            if pts in frame_signatures:
                continue
            parsed = _parse_arrow(comp)
            assert parsed is not None, "all non-frame components must be valid arrows"
            target = _trace_target(g, bg, border_to_frame, parsed["front"], parsed["direction"])
            assert target is not None, "every arrow must point to a frame"
            frame_idx, gap = target
            arrows.append(
                {
                    "color": int(color),
                    "length": int(parsed["length"]),
                    "cells": parsed["cells"],
                    "frame_idx": int(frame_idx),
                    "gap": int(gap),
                }
            )

    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = int(bg)

    layers_by_frame: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for arrow in arrows:
        layers_by_frame[int(arrow["frame_idx"])].append((int(arrow["gap"]), int(arrow["color"]), int(arrow["length"])))

    for idx, frame in enumerate(frames):
        layers = sorted(layers_by_frame.get(int(idx), []), key=lambda item: (item[0], item[1], item[2]))
        _paint_layers(out, frame["box"], [(color, length) for _gap, color, length in layers])

    return out


def _quadrant_ranges(h: int, w: int, quadrant: str, outer_h: int, outer_w: int) -> tuple[range, range] | None:
    mid_r = h // 2
    mid_c = w // 2
    if quadrant == "TL":
        r_lo, r_hi = SAFE_MARGIN, mid_r - INNER_MARGIN - outer_h
        c_lo, c_hi = SAFE_MARGIN, mid_c - INNER_MARGIN - outer_w
    elif quadrant == "TR":
        r_lo, r_hi = SAFE_MARGIN, mid_r - INNER_MARGIN - outer_h
        c_lo, c_hi = mid_c + INNER_MARGIN, w - SAFE_MARGIN - outer_w
    elif quadrant == "BL":
        r_lo, r_hi = mid_r + INNER_MARGIN, h - SAFE_MARGIN - outer_h
        c_lo, c_hi = SAFE_MARGIN, mid_c - INNER_MARGIN - outer_w
    else:
        r_lo, r_hi = mid_r + INNER_MARGIN, h - SAFE_MARGIN - outer_h
        c_lo, c_hi = mid_c + INNER_MARGIN, w - SAFE_MARGIN - outer_w
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return range(int(r_lo), int(r_hi + 1)), range(int(c_lo), int(c_hi + 1))


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _safe_directions(quadrant: str) -> list[str]:
    if quadrant == "TL":
        return ["down", "right"]
    if quadrant == "TR":
        return ["down", "left"]
    if quadrant == "BL":
        return ["up", "right"]
    return ["up", "left"]


def _max_gap_for_direction(box: tuple[int, int, int, int], h: int, w: int, direction: str) -> int:
    r0, c0, r1, c1 = box
    if direction == "down":
        margin = r0
    elif direction == "up":
        margin = h - 1 - r1
    elif direction == "right":
        margin = c0
    else:
        margin = w - 1 - c1
    return int(margin - 2)


def _build_arrow(
    box: tuple[int, int, int, int],
    direction: str,
    axis: int,
    gap: int,
    length: int,
) -> dict:
    r0, c0, r1, c1 = box
    shaft: list[tuple[int, int]]
    head: list[tuple[int, int]]
    front: tuple[int, int]

    if direction == "down":
        end_r = r0 - gap - 1
        start_r = end_r - length + 1
        shaft = [(int(r), int(axis)) for r in range(start_r, end_r + 1)]
        head = [(int(end_r + 1), int(axis - 1)), (int(end_r + 1), int(axis + 1))]
        front = (int(end_r + 1), int(axis))
    elif direction == "up":
        start_r = r1 + gap + 1
        end_r = start_r + length - 1
        shaft = [(int(r), int(axis)) for r in range(start_r, end_r + 1)]
        head = [(int(start_r - 1), int(axis - 1)), (int(start_r - 1), int(axis + 1))]
        front = (int(start_r - 1), int(axis))
    elif direction == "right":
        end_c = c0 - gap - 1
        start_c = end_c - length + 1
        shaft = [(int(axis), int(c)) for c in range(start_c, end_c + 1)]
        head = [(int(axis - 1), int(end_c + 1)), (int(axis + 1), int(end_c + 1))]
        front = (int(axis), int(end_c + 1))
    else:
        start_c = c1 + gap + 1
        end_c = start_c + length - 1
        shaft = [(int(axis), int(c)) for c in range(start_c, end_c + 1)]
        head = [(int(axis - 1), int(start_c - 1)), (int(axis + 1), int(start_c - 1))]
        front = (int(axis), int(start_c - 1))

    return {
        "cells": shaft + head,
        "shaft_len": int(length),
        "front": front,
        "direction": direction,
    }


def _can_place_arrow(
    grid: np.ndarray,
    arrow_cells: list[tuple[int, int]],
    target_border: set[tuple[int, int]],
    arrow_occupied: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for r, c in arrow_cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG or (int(r), int(c)) in arrow_occupied:
            return False
    for r, c in arrow_cells:
        for dr, dc in DIR8:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in arrow_occupied:
                return False
            if int(grid[nr, nc]) != BG and (nr, nc) not in target_border:
                return False
    return True


def _path_is_clear(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    direction: str,
    axis: int,
    gap: int,
) -> bool:
    r0, c0, r1, c1 = box
    if direction == "down":
        cells = [(int(r), int(axis)) for r in range(r0 - gap, r0)]
        wall = (int(r0), int(axis))
    elif direction == "up":
        cells = [(int(r), int(axis)) for r in range(r1 + 1, r1 + gap + 1)]
        wall = (int(r1), int(axis))
    elif direction == "right":
        cells = [(int(axis), int(c)) for c in range(c0 - gap, c0)]
        wall = (int(axis), int(c0))
    else:
        cells = [(int(axis), int(c)) for c in range(c1 + 1, c1 + gap + 1)]
        wall = (int(axis), int(c1))
    return all(int(grid[r, c]) == BG for r, c in cells) and int(grid[wall]) == FRAME


def _choose_frame_count() -> int:
    return random.choice([1, 2, 2, 3, 3, 4])


def _plan_arrow_counts(frames: list[dict]) -> list[int]:
    counts = [1 for _ in frames]
    candidates = [i for i, frame in enumerate(frames) if _layer_capacity(frame["box"]) >= 3]
    if not candidates or random.random() < 0.35:
        return counts
    extra_slots = min(len(candidates), random.choice([1, 1, 2]))
    for idx in random.sample(candidates, k=extra_slots):
        counts[idx] += 1
    return counts


def _place_frame(h: int, w: int, quadrant: str, prefer_large: bool) -> dict | None:
    for _ in range(50):
        inner_h = random.randint(3 if prefer_large else 2, 7)
        inner_w = random.randint(3 if prefer_large else 2, 7)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        ranges = _quadrant_ranges(h, w, quadrant, outer_h, outer_w)
        if ranges is None:
            continue
        row_range, col_range = ranges
        r0 = random.choice(list(row_range))
        c0 = random.choice(list(col_range))
        return {
            "quadrant": quadrant,
            "box": (int(r0), int(c0), int(r0 + outer_h - 1), int(c0 + outer_w - 1)),
        }
    return None


def _length_for_mode(max_len: int, cap: int, mode: str, reserve_outer: bool) -> int | None:
    lo = 2
    hi = max_len
    if hi < lo:
        return None
    if reserve_outer:
        hi = min(hi, max(2, cap - 1))
        if hi < lo:
            return None
        return random.randint(lo, hi)
    if mode == "fit":
        hi = min(hi, cap)
        if hi >= lo:
            return random.randint(lo, hi)
    if mode == "overflow":
        lo = max(lo, cap + 1)
        if hi >= lo:
            return random.randint(lo, hi)
    return random.randint(2, max_len)


def _place_one_arrow(
    grid: np.ndarray,
    frame: dict,
    desired_gap: int,
    color: int,
    arrow_occupied: set[tuple[int, int]],
    reserve_outer: bool,
) -> dict | None:
    h, w = grid.shape
    box = frame["box"]
    cap = _layer_capacity(box)
    sides = _safe_directions(frame["quadrant"])
    modes = ["fit", "overflow"] if random.random() < 0.5 else ["overflow", "fit"]

    for _ in range(200):
        direction = random.choice(sides)
        max_gap = _max_gap_for_direction(box, h, w, direction)
        if desired_gap > max_gap:
            continue
        r0, c0, r1, c1 = box
        if direction in ("down", "up"):
            axes = list(range(c0 + 1, c1))
        else:
            axes = list(range(r0 + 1, r1))
        random.shuffle(axes)
        if not axes:
            continue
        if direction == "down":
            max_len = r0 - desired_gap
        elif direction == "up":
            max_len = h - 1 - r1 - desired_gap
        elif direction == "right":
            max_len = c0 - desired_gap
        else:
            max_len = w - 1 - c1 - desired_gap
        if max_len < 2:
            continue
        mode = random.choice(modes)
        length = _length_for_mode(max_len, cap, mode, reserve_outer)
        if length is None:
            continue
        for axis in axes:
            if not _path_is_clear(grid, box, direction, axis, desired_gap):
                continue
            arrow = _build_arrow(box, direction, axis, desired_gap, length)
            if not _can_place_arrow(grid, arrow["cells"], _rect_border(box), arrow_occupied):
                continue
            for r, c in arrow["cells"]:
                grid[r, c] = int(color)
                arrow_occupied.add((int(r), int(c)))
            return {
                "cells": list(arrow["cells"]),
                "color": int(color),
                "gap": int(desired_gap),
                "length": int(length),
                "frame_box": box,
            }
    return None


def _build_expected(inp: np.ndarray, frames: list[dict], arrows: list[dict]) -> np.ndarray:
    out = inp.copy()
    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = BG
    by_box: dict[tuple[int, int, int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for arrow in arrows:
        by_box[arrow["frame_box"]].append((int(arrow["gap"]), int(arrow["color"]), int(arrow["length"])))
    for frame in frames:
        layers = sorted(by_box.get(frame["box"], []), key=lambda item: item[0])
        _paint_layers(out, frame["box"], [(color, length) for _gap, color, length in layers])
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)

    n_frames = _choose_frame_count()
    quadrants = random.sample(["TL", "TR", "BL", "BR"], k=n_frames)
    prefer_large_quadrants = set(random.sample(quadrants, k=min(len(quadrants), random.choice([1, 1, 2]))))

    frames: list[dict] = []
    for quadrant in quadrants:
        frame = _place_frame(h, w, quadrant, prefer_large=quadrant in prefer_large_quadrants)
        if frame is None:
            return None
        frames.append(frame)
        _draw_frame(grid, frame["box"])

    counts = _plan_arrow_counts(frames)
    total_arrows = sum(counts)
    if total_arrows > len(ARROW_PALETTE):
        return None

    colors = random.sample(ARROW_PALETTE, k=total_arrows)
    color_idx = 0
    arrows: list[dict] = []
    arrow_occupied: set[tuple[int, int]] = set()

    for frame, count in zip(frames, counts):
        cap = _layer_capacity(frame["box"])
        max_gap = max(_max_gap_for_direction(frame["box"], h, w, d) for d in _safe_directions(frame["quadrant"]))
        if count == 2 and (cap < 3 or max_gap < 2):
            return None
        gap_pool = list(range(1, min(4, max_gap) + 1))
        if len(gap_pool) < count:
            return None
        chosen_gaps = sorted(random.sample(gap_pool, k=count))
        for order, gap in enumerate(chosen_gaps):
            arrow = _place_one_arrow(
                grid=grid,
                frame=frame,
                desired_gap=gap,
                color=colors[color_idx],
                arrow_occupied=arrow_occupied,
                reserve_outer=(count == 2 and order == 0),
            )
            if arrow is None:
                return None
            arrows.append(arrow)
            color_idx += 1

    expected = _build_expected(grid, frames, arrows)
    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-frame ARC input")
