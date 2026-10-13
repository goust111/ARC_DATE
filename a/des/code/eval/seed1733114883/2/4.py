from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GRID_H_RANGE = (16, 28)
GRID_W_RANGE = (16, 30)
MAX_DEPTH_RANGE = (1, 4)
GROUP_COUNT_RANGE = (1, 3)
INNER_SIZE_RANGE = (3, 7)
MARGIN_RANGE = (2, 4)
GEN_RETRIES = 400

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPEN_SIDES = ("top", "bottom", "left", "right")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _full_perimeter(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _strip_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    candidates: list[dict] = []
    for top in (True, False):
        for left in (True, False):
            c = 0 if left else w - 1
            dc = 1 if left else -1
            r = 0 if top else h - 1
            step = 1 if top else -1
            coords: list[tuple[int, int]] = []
            while 0 <= r < h and int(grid[r, c]) != bg:
                if 0 <= c + dc < w and int(grid[r, c + dc]) != bg:
                    coords = []
                    break
                coords.append((r, c))
                r += step
            if not coords:
                continue
            if 0 <= r < h and int(grid[r, c]) != bg:
                continue
            rows = [rr for rr, _ in coords]
            if len(set(rows)) != len(rows):
                continue
            coords.sort()
            candidates.append({"coords": coords, "corner": ("top" if top else "bottom", "left" if left else "right")})
    return candidates


def _find_color_strip(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    candidates = _strip_candidates(grid, bg)
    assert len(candidates) == 1, "expected exactly one corner color strip"
    return candidates[0]["coords"]


def _parse_frames(grid: np.ndarray, strip_cells: set[tuple[int, int]], bg: int) -> tuple[int, list[dict]]:
    working = grid.copy()
    for r, c in strip_cells:
        working[r, c] = bg

    colors = [int(v) for v in np.unique(working) if int(v) != bg]
    assert colors, "expected frame color after removing strip"
    counts = Counter(int(v) for v in working.ravel() if int(v) != bg)
    frame_color = counts.most_common(1)[0][0]

    frames: list[dict] = []
    for comp in _components(working == frame_color):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0 = min(rows)
        r1 = max(rows)
        c0 = min(cols)
        c1 = max(cols)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must be at least 3x3"

        cells = set(comp)
        perimeter = _full_perimeter(r0, c0, r1, c1)
        missing = perimeter - cells
        extra = cells - perimeter
        assert not extra, "frame pixels must lie on a rectangle perimeter"
        assert len(missing) == 1, "frame must have exactly one perimeter gap"
        gap = next(iter(missing))
        gr, gc = gap
        assert gap not in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}, "gap may not be a corner"
        if gr == r0:
            side = "top"
            ray_dir = (-1, 0)
        elif gr == r1:
            side = "bottom"
            ray_dir = (1, 0)
        elif gc == c0:
            side = "left"
            ray_dir = (0, -1)
        else:
            assert gc == c1
            side = "right"
            ray_dir = (0, 1)
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "side": side,
                "ray_dir": ray_dir,
            }
        )

    assert frames, "expected at least one frame"
    return frame_color, frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Encode nested-frame depth as fill color and emit outward rays from each opening.

    Input structure:
    - Background is black.
    - Several one-cell-thick rectangular frames share one outline color.
    - Each frame has exactly one single-cell gap on one side.
    - A vertical corner strip maps depth 1..N to colors from top to bottom.

    Rule:
    - Compute each frame's nesting depth from strict containment.
    - Fill every frame interior with the strip color for its depth; inner frames override outer ones.
    - From each frame gap, emit a straight ray in the opening direction with length equal to depth.
    - Rays paint only cells that were background in the input and stop at the first original non-background cell.
    - Frame outlines stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = BG

    strip = _find_color_strip(g, bg)
    strip_cells = set(strip)
    depth_colors = [int(g[r, c]) for r, c in strip]

    frame_color, frames = _parse_frames(g, strip_cells, bg)
    original_bg = g == bg
    out = g.copy()

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        depth = 1 + sum(
            1
            for other in frames
            if other is not frame
            and other["bbox"][0] < r0
            and other["bbox"][1] < c0
            and r1 < other["bbox"][2]
            and c1 < other["bbox"][3]
        )
        frame["depth"] = depth
        assert depth <= len(depth_colors), "strip must provide a color for every depth"

    for frame in sorted(frames, key=lambda item: item["depth"]):
        r0, c0, r1, c1 = frame["bbox"]
        fill = depth_colors[frame["depth"] - 1]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        mask = original_bg[r0 + 1 : r1, c0 + 1 : c1]
        interior[mask] = fill

    for frame in frames:
        fill = depth_colors[frame["depth"] - 1]
        gr, gc = frame["gap"]
        dr, dc = frame["ray_dir"]
        for step in range(frame["depth"]):
            rr = gr + dr * step
            cc = gc + dc * step
            if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                break
            if not original_bg[rr, cc]:
                break
            out[rr, cc] = fill

    unique_non_bg = {int(v) for v in np.unique(g) if int(v) != bg}
    assert frame_color in unique_non_bg
    return out


def _choose_colors(max_depth: int) -> tuple[int, list[int]]:
    palette = [c for c in range(10) if c != BG]
    frame_color = random.choice(palette)
    remaining = [c for c in palette if c != frame_color]
    depth_colors = random.sample(remaining, max_depth)
    return frame_color, depth_colors


def _random_bbox_chain(depth: int) -> list[dict]:
    inner_h = random.randint(*INNER_SIZE_RANGE)
    inner_w = random.randint(*INNER_SIZE_RANGE)
    boxes = [{"bbox": (0, 0, inner_h - 1, inner_w - 1)}]
    for _ in range(depth - 1):
        r0, c0, r1, c1 = boxes[0]["bbox"]
        top = random.randint(*MARGIN_RANGE)
        bottom = random.randint(*MARGIN_RANGE)
        left = random.randint(*MARGIN_RANGE)
        right = random.randint(*MARGIN_RANGE)
        boxes.insert(0, {"bbox": (r0 - top, c0 - left, r1 + bottom, c1 + right)})

    min_r = min(box["bbox"][0] for box in boxes)
    min_c = min(box["bbox"][1] for box in boxes)
    for box in boxes:
        r0, c0, r1, c1 = box["bbox"]
        box["bbox"] = (r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c)
        box["side"] = random.choice(OPEN_SIDES)
        rr0, cc0, rr1, cc1 = box["bbox"]
        if box["side"] in ("top", "bottom"):
            gap_c = random.randint(cc0 + 1, cc1 - 1)
            gap_r = rr0 if box["side"] == "top" else rr1
        else:
            gap_r = random.randint(rr0 + 1, rr1 - 1)
            gap_c = cc0 if box["side"] == "left" else cc1
        box["gap"] = (gap_r, gap_c)
    return boxes


def _group_size(chain: list[dict]) -> tuple[int, int]:
    r0, c0, r1, c1 = chain[0]["bbox"]
    return r1 - r0 + 1, c1 - c0 + 1


def _draw_chain(chain: list[dict], frame_color: int) -> np.ndarray:
    h, w = _group_size(chain)
    patch = np.full((h, w), BG, dtype=int)
    for box in chain:
        r0, c0, r1, c1 = box["bbox"]
        patch[r0, c0 : c1 + 1] = frame_color
        patch[r1, c0 : c1 + 1] = frame_color
        patch[r0 : r1 + 1, c0] = frame_color
        patch[r0 : r1 + 1, c1] = frame_color
        gr, gc = box["gap"]
        patch[gr, gc] = BG
    return patch


def _ray_cells_for_chain(chain: list[dict]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for depth, box in enumerate(chain, start=1):
        gr, gc = box["gap"]
        if box["side"] == "top":
            dr, dc = -1, 0
        elif box["side"] == "bottom":
            dr, dc = 1, 0
        elif box["side"] == "left":
            dr, dc = 0, -1
        else:
            dr, dc = 0, 1
        for step in range(depth):
            cells.add((gr + dr * step, gc + dc * step))
    return cells


def _place_groups(
    h: int,
    w: int,
    side: str,
    max_depth: int,
    groups: list[dict],
) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int]] = []
    reserved = np.zeros((h, w), dtype=bool)
    reserved[0, :] = True
    reserved[h - 1, :] = True
    reserved[:, 0] = True
    reserved[:, w - 1] = True
    if side == "left":
        reserved[:, 1] = True
    else:
        reserved[:, w - 2] = True

    order = sorted(range(len(groups)), key=lambda idx: _group_size(groups[idx]["chain"])[0] * _group_size(groups[idx]["chain"])[1], reverse=True)
    out_positions = [(-1, -1)] * len(groups)
    for idx in order:
        gh, gw = _group_size(groups[idx]["chain"])
        top_min = max_depth + 1
        top_max = h - max_depth - gh - 1
        left_min = max_depth + 1
        left_max = w - max_depth - gw - 1
        if side == "left":
            left_min = max(left_min, 2 + max_depth)
        else:
            left_max = min(left_max, w - gw - max_depth - 2)
        if top_max < top_min or left_max < left_min:
            return None

        candidates = [(r, c) for r in range(top_min, top_max + 1) for c in range(left_min, left_max + 1)]
        random.shuffle(candidates)
        placed = None
        for top, left in candidates[:200]:
            r0 = top - max_depth
            c0 = left - max_depth
            r1 = top + gh + max_depth - 1
            c1 = left + gw + max_depth - 1
            if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            placed = (top, left)
            reserved[r0 : r1 + 1, c0 : c1 + 1] = True
            break
        if placed is None:
            return None
        out_positions[idx] = placed
    return out_positions


def _add_strip(grid: np.ndarray, colors: list[int], corner_top: bool, side: str) -> None:
    h, w = grid.shape
    c = 0 if side == "left" else w - 1
    rows = list(range(len(colors))) if corner_top else list(range(h - len(colors), h))
    for r, color in zip(rows, colors):
        grid[r, c] = color


def _frame_data_from_chain(chain: list[dict], top: int, left: int) -> list[dict]:
    frames = []
    for depth, box in enumerate(chain, start=1):
        r0, c0, r1, c1 = box["bbox"]
        gr, gc = box["gap"]
        frames.append(
            {
                "depth": depth,
                "bbox": (top + r0, left + c0, top + r1, left + c1),
                "gap": (top + gr, left + gc),
                "side": box["side"],
            }
        )
    return frames


def _ray_paths_do_not_overlap(frames: list[dict], grid: np.ndarray) -> bool:
    occupied: dict[tuple[int, int], int] = {}
    h, w = grid.shape
    for frame in frames:
        gr, gc = frame["gap"]
        if frame["side"] == "top":
            dr, dc = -1, 0
        elif frame["side"] == "bottom":
            dr, dc = 1, 0
        elif frame["side"] == "left":
            dr, dc = 0, -1
        else:
            dr, dc = 0, 1
        for step in range(frame["depth"]):
            rr = gr + dr * step
            cc = gc + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                break
            if int(grid[rr, cc]) != BG:
                break
            prev = occupied.get((rr, cc))
            if prev is not None and prev != frame["depth"]:
                return False
            occupied[(rr, cc)] = frame["depth"]
    return True


def _generate_attempt() -> np.ndarray | None:
    max_depth = random.randint(*MAX_DEPTH_RANGE)
    group_count = random.randint(*GROUP_COUNT_RANGE)
    group_depths = [max_depth]
    group_depths.extend(random.randint(1, max_depth) for _ in range(group_count - 1))
    random.shuffle(group_depths)

    frame_color, depth_colors = _choose_colors(max_depth)
    side = random.choice(["left", "right"])
    corner_top = random.choice([True, False])

    groups = []
    for depth in group_depths:
        chain = _random_bbox_chain(depth)
        groups.append({"chain": chain, "patch": _draw_chain(chain, frame_color)})

    min_h = max(max_depth + 2, max(_group_size(group["chain"])[0] for group in groups) + 2 * max_depth + 2)
    min_w = max(
        8,
        max(_group_size(group["chain"])[1] for group in groups) + 2 * max_depth + 4,
    )
    min_h = max(min_h, len(depth_colors) + 2)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    h = random.randint(max(GRID_H_RANGE[0], min_h), GRID_H_RANGE[1])
    w = random.randint(max(GRID_W_RANGE[0], min_w), GRID_W_RANGE[1])
    positions = _place_groups(h, w, side, max_depth, groups)
    if positions is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    _add_strip(grid, depth_colors, corner_top, side)

    all_frames: list[dict] = []
    for group, (top, left) in zip(groups, positions):
        patch = group["patch"]
        ph, pw = patch.shape
        if np.any((grid[top : top + ph, left : left + pw] != BG) & (patch != BG)):
            return None
        grid[top : top + ph, left : left + pw][patch != BG] = frame_color
        all_frames.extend(_frame_data_from_chain(group["chain"], top, left))

    if not _ray_paths_do_not_overlap(all_frames, grid):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if len(np.unique(grid)) != max_depth + 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
