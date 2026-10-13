from __future__ import annotations

import random
from typing import Iterable

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


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
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


def _is_rectangular_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected_area = 2 * (h + w) - 4
    coord_set = set(coords)
    if len(coord_set) != expected_area:
        return False
    for r, c in coord_set:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in coord_set or (r1, c) not in coord_set:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in coord_set or (r, c1) not in coord_set:
            return False
    return True


def _extract_frames(grid: np.ndarray, bg: int) -> tuple[int, list[dict[str, object]]]:
    candidates: list[tuple[int, int, int, list[dict[str, object]]]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        comps = _components(grid == color)
        if not comps:
            continue
        frame_infos: list[dict[str, object]] = []
        total_area = 0
        valid = True
        for coords in comps:
            if not _is_rectangular_frame(coords):
                valid = False
                break
            box = _bbox(coords)
            frame_infos.append({"bbox": box, "coords": coords})
            total_area += len(coords)
        if valid:
            candidates.append((len(frame_infos), total_area, color, frame_infos))
    assert candidates, "expected one color made of rectangular frame components"
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    _count, _area, frame_color, frames = candidates[0]
    return int(frame_color), frames


def _extract_bar(grid: np.ndarray, bg: int, frame_color: int) -> tuple[list[int], list[tuple[int, int]]]:
    mask = (grid != bg) & (grid != frame_color)
    comps = _components(mask)
    assert len(comps) == 1, "expected a single color bar"
    coords = comps[0]
    coord_set = set(coords)
    h, w = grid.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    touched = [corner for corner in corners if corner in coord_set]
    assert len(touched) == 1, "color bar must occupy exactly one corner"
    corner = touched[0]
    r0, c0, r1, c1 = _bbox(coords)
    assert r0 == r1 or c0 == c1, "color bar must be a straight 1-cell line"
    if r0 == r1:
        ordered = sorted(coords, key=lambda pt: pt[1], reverse=corner[1] != 0)
    else:
        ordered = sorted(coords, key=lambda pt: pt[0], reverse=corner[0] != 0)
    for i in range(1, len(ordered)):
        prev_r, prev_c = ordered[i - 1]
        curr_r, curr_c = ordered[i]
        assert abs(prev_r - curr_r) + abs(prev_c - curr_c) == 1, "bar cells must be contiguous"
    colors = [int(grid[r, c]) for r, c in ordered]
    assert len(colors) == len(set(colors)), "bar colors should be distinct"
    return colors, ordered


def _depth(frame: dict[str, object], frames: list[dict[str, object]]) -> int:
    r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
    depth = 1
    for other in frames:
        if other is frame:
            continue
        or0, oc0, or1, oc1 = other["bbox"]  # type: ignore[index]
        if or0 < r0 and oc0 < c0 and or1 > r1 and oc1 > c1:
            depth += 1
    return depth


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested grey rectangular frames by containment depth.

    Input structure:
    - Black background.
    - Several grey hollow rectangular frames.
    - A 1-cell-thick color bar sits in one corner and lists colors from
      outermost depth to innermost depth.

    Rule:
    - The outermost frames have depth 1.
    - Each frame's depth is 1 plus the number of enclosing frames.
    - Replace grey pixels with the bar color for that depth.
    - Remove the color bar in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    frame_color, frames = _extract_frames(g, bg)
    bar_colors, bar_cells = _extract_bar(g, bg, frame_color)

    out = g.copy()
    for r, c in bar_cells:
        out[r, c] = bg

    for frame in frames:
        depth = _depth(frame, frames)
        assert 1 <= depth <= len(bar_colors), "frame depth exceeds bar length"
        new_color = int(bar_colors[depth - 1])
        for r, c in frame["coords"]:  # type: ignore[index]
            out[r, c] = new_color

    return out


BG = 0
FRAME_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DEPTH_RANGE = (2, 5)
LEAF_H_RANGE = (3, 6)
LEAF_W_RANGE = (3, 7)
MARGIN_CHOICES = (2, 3)
PLACEMENT_PAD = 2
GEN_ATTEMPTS = 400
MAX_SIDE = 40


def _build_chain(depth: int) -> list[tuple[int, int, int, int]]:
    inner_h = random.randint(*LEAF_H_RANGE)
    inner_w = random.randint(*LEAF_W_RANGE)
    boxes = [(0, 0, inner_h - 1, inner_w - 1)]
    for _ in range(depth - 1):
        r0, c0, r1, c1 = boxes[0]
        top = random.choice(MARGIN_CHOICES)
        bottom = random.choice(MARGIN_CHOICES)
        left = random.choice(MARGIN_CHOICES)
        right = random.choice(MARGIN_CHOICES)
        boxes.insert(0, (r0 - top, c0 - left, r1 + bottom, c1 + right))
    shift_r = -boxes[0][0]
    shift_c = -boxes[0][1]
    return [(r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c) for r0, c0, r1, c1 in boxes]


def _box_size(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _bar_cells(h: int, w: int, length: int, corner: str, orient: str) -> list[tuple[int, int]]:
    if orient == "horizontal":
        if corner == "tl":
            return [(0, i) for i in range(length)]
        if corner == "tr":
            return [(0, w - 1 - i) for i in range(length)]
        if corner == "bl":
            return [(h - 1, i) for i in range(length)]
        return [(h - 1, w - 1 - i) for i in range(length)]
    if corner == "tl":
        return [(i, 0) for i in range(length)]
    if corner == "tr":
        return [(i, w - 1) for i in range(length)]
    if corner == "bl":
        return [(h - 1 - i, 0) for i in range(length)]
    return [(h - 1 - i, w - 1) for i in range(length)]


def _choose_depths(max_depth: int) -> list[int]:
    if max_depth >= 5:
        chain_count = 1
    elif max_depth == 4:
        chain_count = random.randint(1, 2)
    else:
        chain_count = random.randint(1, 3)
    depths = [max_depth]
    while len(depths) < chain_count:
        depths.append(random.randint(1, max_depth))
    random.shuffle(depths)
    return depths


def _canvas_bounds(sizes: list[tuple[int, int]]) -> tuple[int, int]:
    max_h = max(h for h, _ in sizes)
    max_w = max(w for _, w in sizes)
    padded_area = sum((h + PLACEMENT_PAD * 2) * (w + PLACEMENT_PAD * 2) for h, w in sizes)
    side_hint = int(np.ceil(np.sqrt(padded_area))) + 4
    min_h = max(max_h + 5, side_hint)
    min_w = max(max_w + 5, side_hint)
    return min_h, min_w


def _place_outer_boxes(sizes: list[tuple[int, int]]) -> tuple[int, int, list[tuple[int, int]]]:
    min_h, min_w = _canvas_bounds(sizes)
    for _ in range(120):
        height = random.randint(min_h, min(MAX_SIDE, min_h + 8))
        width = random.randint(min_w, min(MAX_SIDE, min_w + 8))
        positions: list[tuple[int, int]] = []
        placed_boxes: list[tuple[int, int, int, int]] = []
        ok = True
        order = list(range(len(sizes)))
        random.shuffle(order)
        for idx in order:
            box_h, box_w = sizes[idx]
            max_r0 = height - box_h - 3
            max_c0 = width - box_w - 3
            if max_r0 < 2 or max_c0 < 2:
                ok = False
                break
            found = False
            for _ in range(200):
                r0 = random.randint(2, max_r0)
                c0 = random.randint(2, max_c0)
                candidate = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
                if any(_boxes_overlap(candidate, other, pad=PLACEMENT_PAD) for other in placed_boxes):
                    continue
                positions.append((r0, c0))
                placed_boxes.append(candidate)
                found = True
                break
            if not found:
                ok = False
                break
        if ok:
            ordered_positions = [(-1, -1)] * len(sizes)
            for idx, pos in zip(order, positions):
                ordered_positions[idx] = pos
            return height, width, ordered_positions
    raise RuntimeError("failed to place frame groups")


def _paint_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _generate_attempt() -> np.ndarray:
    max_depth = random.randint(*DEPTH_RANGE)
    depths = _choose_depths(max_depth)
    chains = [_build_chain(depth) for depth in depths]
    sizes = [_box_size(chain[0]) for chain in chains]
    height, width, positions = _place_outer_boxes(sizes)

    grid = np.full((height, width), BG, dtype=int)
    color_bar = random.sample(PALETTE, max_depth)
    corner = random.choice(["tl", "tr", "bl", "br"])
    orient = random.choice(["horizontal", "vertical"])
    for (r, c), color in zip(_bar_cells(height, width, max_depth, corner, orient), color_bar):
        grid[r, c] = int(color)

    for chain, (base_r, base_c) in zip(chains, positions):
        for box in chain:
            r0, c0, r1, c1 = box
            shifted = (r0 + base_r, c0 + base_c, r1 + base_r, c1 + base_c)
            _paint_frame(grid, shifted, FRAME_COLOR)

    # The same structural extractor used by main validates the generator output.
    frame_color, frames = _extract_frames(grid, BG)
    assert frame_color == FRAME_COLOR
    bar_colors, _bar = _extract_bar(grid, BG, FRAME_COLOR)
    assert bar_colors == color_bar
    depths_seen = [_depth(frame, frames) for frame in frames]
    assert max(depths_seen) == max_depth
    assert set(depths_seen) == set(range(1, max_depth + 1))
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            return _generate_attempt()
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid nested-frame puzzle")
