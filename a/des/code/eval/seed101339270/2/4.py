import numpy as np
from collections import Counter, defaultdict


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

ARROW_COLORS = (BLUE, RED, GREEN, YELLOW)
OP_COLORS = {BLUE, RED, GREEN, YELLOW}
BG_CHOICES = (0, 6, 7, 8, 9)

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
DIR_BY_NAME = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

RNG = np.random.default_rng()

GRID_H_RANGE = (20, 31)
GRID_W_RANGE = (20, 31)
FRAME_COUNT_RANGE = (2, 5)
FRAME_H_RANGE = (6, 11)
FRAME_W_RANGE = (6, 12)
ARROW_GAP_RANGE = (2, 8)
ARROW_SHAFT_RANGE = (3, 6)
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    steps = DIR8 if connectivity == 8 else DIR4
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _frame_lookup(frames: list[dict]) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for idx, frame in enumerate(frames):
        for cell in _rect_border_coords(frame["r0"], frame["c0"], frame["r1"], frame["c1"]):
            lookup[cell] = idx
    return lookup


def _build_arrow_cells(direction: str, top: int, left: int, shaft_len: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if direction == "right":
        width = shaft_len + 1
        cells.update({(top, left + width - 1), (top + 2, left + width - 1)})
        cells.update((top + 1, left + c) for c in range(width - 1))
    elif direction == "left":
        width = shaft_len + 1
        cells.update({(top, left), (top + 2, left)})
        cells.update((top + 1, left + c) for c in range(1, width))
    elif direction == "up":
        height = shaft_len + 1
        cells.update({(top, left), (top, left + 2)})
        cells.update((top + r, left + 1) for r in range(1, height))
    elif direction == "down":
        height = shaft_len + 1
        cells.update({(top + height - 1, left), (top + height - 1, left + 2)})
        cells.update((top + r, left + 1) for r in range(height - 1))
    return cells


def _parse_arrow(coords: list[tuple[int, int]]) -> dict | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    norm = {(r - r0, c - c0) for r, c in coords}
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    if h == 3 and 4 <= w <= 6:
        shaft_len = w - 1
        if norm == _build_arrow_cells("right", 0, 0, shaft_len):
            return {
                "direction": "right",
                "ray_start": (r0 + 1, c1 + 1),
                "cells": set(coords),
            }
        if norm == _build_arrow_cells("left", 0, 0, shaft_len):
            return {
                "direction": "left",
                "ray_start": (r0 + 1, c0 - 1),
                "cells": set(coords),
            }
    if w == 3 and 4 <= h <= 6:
        shaft_len = h - 1
        if norm == _build_arrow_cells("up", 0, 0, shaft_len):
            return {
                "direction": "up",
                "ray_start": (r0 - 1, c0 + 1),
                "cells": set(coords),
            }
        if norm == _build_arrow_cells("down", 0, 0, shaft_len):
            return {
                "direction": "down",
                "ray_start": (r1 + 1, c0 + 1),
                "cells": set(coords),
            }
    return None


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid == GREY, connectivity=4):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = set(_rect_border_coords(r0, c0, r1, c1))
        if set(comp) != border:
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not np.all(inner == bg):
            continue
        frames.append({"r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1)})
    frames.sort(key=lambda frame: (frame["r0"], frame["c0"]))
    return frames


def _detect_arrows(grid: np.ndarray, bg: int) -> list[dict]:
    arrows: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid)):
        if color in (bg, GREY) or color not in OP_COLORS:
            continue
        for comp in _components(grid == color, connectivity=8):
            parsed = _parse_arrow(comp)
            if parsed is None:
                continue
            parsed["color"] = color
            arrows.append(parsed)
    arrows.sort(key=lambda arrow: (arrow["ray_start"][0], arrow["ray_start"][1], arrow["color"]))
    return arrows


def _bresenham(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return points


def _apply_operation(out: np.ndarray, frame: dict, color: int) -> None:
    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    if color == RED:
        out[r0 + 1 : r1, c0 + 1 : c1] = RED
        return
    if color == BLUE:
        for r, c in _bresenham(r0 + 1, c0 + 1, r1 - 1, c1 - 1):
            if r0 < r < r1 and c0 < c < c1:
                out[r, c] = BLUE
        return
    if color == GREEN:
        mid_r = (r0 + r1) // 2
        mid_c = (c0 + c1) // 2
        out[mid_r, c0 + 1 : c1] = GREEN
        out[r0 + 1 : r1, mid_c] = GREEN
        return
    if color == YELLOW:
        top, bottom = r0 + 2, r1 - 2
        left, right = c0 + 2, c1 - 2
        if top > bottom or left > right:
            return
        out[top, left : right + 1] = YELLOW
        out[bottom, left : right + 1] = YELLOW
        out[top : bottom + 1, left] = YELLOW
        out[top : bottom + 1, right] = YELLOW


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _dominant_color(g)
    frames = _detect_frames(g, bg)
    border_lookup = _frame_lookup(frames)
    arrows = _detect_arrows(g, bg)

    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = bg

    hits: dict[int, list[tuple[int, int, dict]]] = defaultdict(list)
    for order, arrow in enumerate(arrows):
        dr, dc = DIR_BY_NAME[arrow["direction"]]
        r, c = arrow["ray_start"]
        dist = 0
        while 0 <= r < h and 0 <= c < w:
            dist += 1
            frame_idx = border_lookup.get((r, c))
            if frame_idx is not None:
                hits[frame_idx].append((dist, order, arrow))
                break
            r += dr
            c += dc

    for frame_idx, ops in hits.items():
        frame = frames[frame_idx]
        for _dist, _order, arrow in sorted(ops, key=lambda item: (-item[0], item[1])):
            _apply_operation(out, frame, arrow["color"])
    return out


def _can_place(cells: set[tuple[int, int]], blocked: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w and (r, c) not in blocked for r, c in cells)


def _add_object(cells: set[tuple[int, int]], color: int, grid: np.ndarray, occupied: set, blocked: set) -> None:
    h, w = grid.shape
    for r, c in cells:
        grid[r, c] = color
        occupied.add((r, c))
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    blocked.add((rr, cc))


def _balanced_arrow_colors(n: int) -> list[int]:
    colors: list[int] = []
    while len(colors) < n:
        block = list(ARROW_COLORS)
        RNG.shuffle(block)
        colors.extend(block)
    return colors[:n]


def _build_frame_cells(frame: dict) -> set[tuple[int, int]]:
    return set(_rect_border_coords(frame["r0"], frame["c0"], frame["r1"], frame["c1"]))


def _arrow_from_side(frame: dict, side: str, axis: int, gap: int, shaft_len: int) -> dict:
    if side == "left":
        c1 = frame["c0"] - gap - 1
        top = axis - 1
        left = c1 - shaft_len
        direction = "right"
    elif side == "right":
        top = axis - 1
        left = frame["c1"] + gap + 1
        direction = "left"
    elif side == "top":
        top = frame["r0"] - gap - shaft_len - 1
        left = axis - 1
        direction = "down"
    else:
        top = frame["r1"] + gap + 1
        left = axis - 1
        direction = "up"
    cells = _build_arrow_cells(direction, top, left, shaft_len)
    parsed = _parse_arrow(list(cells))
    assert parsed is not None
    parsed["cells"] = cells
    parsed["distance"] = gap + 1
    parsed["direction"] = direction
    return parsed


def _corridor_hits_target(
    ray_start: tuple[int, int],
    direction: str,
    frame_idx: int,
    lookup: dict[tuple[int, int], int],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    dr, dc = DIR_BY_NAME[direction]
    r, c = ray_start
    while 0 <= r < h and 0 <= c < w:
        hit = lookup.get((r, c))
        if hit is not None:
            return hit == frame_idx
        if (r, c) in occupied:
            return False
        r += dr
        c += dc
    return False


def _place_frames(grid: np.ndarray) -> list[dict] | None:
    h, w = grid.shape
    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    frames: list[dict] = []

    num_frames = int(RNG.integers(*FRAME_COUNT_RANGE))
    for _ in range(num_frames):
        placed = False
        for _attempt in range(250):
            fh = int(RNG.integers(*FRAME_H_RANGE))
            fw = int(RNG.integers(*FRAME_W_RANGE))
            if fh >= h or fw >= w:
                continue
            r0 = int(RNG.integers(0, h - fh + 1))
            c0 = int(RNG.integers(0, w - fw + 1))
            frame = {"r0": r0, "c0": c0, "r1": r0 + fh - 1, "c1": c0 + fw - 1}
            cells = _build_frame_cells(frame)
            if not _can_place(cells, blocked, h, w):
                continue
            _add_object(cells, GREY, grid, occupied, blocked)
            frames.append(frame)
            placed = True
            break
        if not placed:
            return None
    return frames


def _sample_arrow_targets(num_frames: int) -> list[int]:
    frames = list(range(num_frames))
    RNG.shuffle(frames)
    if num_frames == 1:
        target_count = 1
    else:
        target_count = int(RNG.integers(1, num_frames + 1))
        if target_count == num_frames and RNG.random() < 0.5:
            target_count -= 1
    target_frames = frames[:target_count]
    arrow_targets = target_frames.copy()
    if len(arrow_targets) < 2:
        arrow_targets.append(target_frames[0])
    multi = int(target_frames[int(RNG.integers(0, len(target_frames)))])
    if RNG.random() < 0.65:
        arrow_targets.append(multi)
    if RNG.random() < 0.3:
        arrow_targets.append(int(target_frames[int(RNG.integers(0, len(target_frames)))]))
    RNG.shuffle(arrow_targets)
    return arrow_targets


def _place_arrows(grid: np.ndarray, frames: list[dict]) -> list[dict] | None:
    h, w = grid.shape
    occupied = {(r, c) for r in range(h) for c in range(w) if grid[r, c] != _dominant_color(grid)}
    blocked = set()
    for r, c in occupied:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    blocked.add((rr, cc))

    lookup = _frame_lookup(frames)
    arrow_targets = _sample_arrow_targets(len(frames))
    colors = _balanced_arrow_colors(len(arrow_targets))
    RNG.shuffle(colors)
    arrows: list[dict] = []
    used_distances: dict[int, set[int]] = defaultdict(set)

    for frame_idx, color in zip(arrow_targets, colors):
        frame = frames[frame_idx]
        placed = False
        sides = ["left", "right", "top", "bottom"]
        RNG.shuffle(sides)
        for side in sides:
            if side in ("left", "right"):
                axes = list(range(frame["r0"] + 1, frame["r1"]))
            else:
                axes = list(range(frame["c0"] + 1, frame["c1"]))
            if not axes:
                continue
            RNG.shuffle(axes)
            gaps = list(range(*ARROW_GAP_RANGE))
            shafts = list(range(*ARROW_SHAFT_RANGE))
            RNG.shuffle(gaps)
            RNG.shuffle(shafts)
            for axis in axes:
                for gap in gaps:
                    if gap + 1 in used_distances[frame_idx]:
                        continue
                    for shaft_len in shafts:
                        arrow = _arrow_from_side(frame, side, axis, gap, shaft_len)
                        cells = arrow["cells"]
                        if not _can_place(cells, blocked, h, w):
                            continue
                        if not _corridor_hits_target(
                            arrow["ray_start"],
                            arrow["direction"],
                            frame_idx,
                            lookup,
                            occupied,
                            h,
                            w,
                        ):
                            continue
                        arrow["color"] = color
                        arrow["frame_idx"] = frame_idx
                        _add_object(cells, color, grid, occupied, blocked)
                        arrows.append(arrow)
                        used_distances[frame_idx].add(arrow["distance"])
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None
    return arrows


def _expected_from_case(case: dict) -> np.ndarray:
    out = case["grid"].copy()
    bg = case["bg"]
    for arrow in case["arrows"]:
        for r, c in arrow["cells"]:
            out[r, c] = bg
    grouped: dict[int, list[tuple[int, int, dict]]] = defaultdict(list)
    for order, arrow in enumerate(case["arrows"]):
        grouped[arrow["frame_idx"]].append((arrow["distance"], order, arrow))
    for frame_idx, ops in grouped.items():
        for _dist, _order, arrow in sorted(ops, key=lambda item: (-item[0], item[1])):
            _apply_operation(out, case["frames"][frame_idx], arrow["color"])
    return out


def _generate_case() -> dict | None:
    h = int(RNG.integers(*GRID_H_RANGE))
    w = int(RNG.integers(*GRID_W_RANGE))
    bg = int(RNG.choice(BG_CHOICES))
    grid = np.full((h, w), bg, dtype=int)

    frames = _place_frames(grid)
    if frames is None:
        return None
    arrows = _place_arrows(grid, frames)
    if arrows is None:
        return None

    case = {"grid": grid, "bg": bg, "frames": frames, "arrows": arrows}
    expected = _expected_from_case(case)

    if len(_detect_frames(grid, bg)) != len(frames):
        return None
    if len(_detect_arrows(grid, bg)) != len(arrows):
        return None
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return case


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case["grid"]
    raise RuntimeError("failed to generate a valid arrow-frame puzzle")
