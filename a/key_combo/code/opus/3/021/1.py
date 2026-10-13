from __future__ import annotations

from collections import Counter
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9

    NOT_BLACK = [BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]


SIDES = ("top", "bottom", "left", "right")
HORIZONTAL_SIDES = {"top", "bottom"}
VERTICAL_SIDES = {"left", "right"}

GEN_GRID_MIN = 26
GEN_GRID_MAX = 29
GEN_INTERIOR_MIN = 4
GEN_INTERIOR_MAX = 6
GEN_FRAME_COUNT = 3
GEN_FRAME_PAD = 4
GEN_MAX_ATTEMPTS = 500
GEN_SHAPE_ATTEMPTS = 80
GEN_SHORTCUT_RETRY = 80


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(bbox: tuple[int, int, int, int]) -> int:
    r0, _c0, r1, _c1 = bbox
    return int(r1 - r0 + 1)


def bbox_w(bbox: tuple[int, int, int, int]) -> int:
    _r0, c0, _r1, c1 = bbox
    return int(c1 - c0 + 1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    assert connectivity in (4, 8)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in deltas:
                    nr = int(rr) + int(dr)
                    nc = int(cc) + int(dc)
                    if in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
) -> list[dict]:
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    components: list[dict] = []
    if monochromatic:
        for color in sorted(int(c) for c in np.unique(grid) if int(c) != bg):
            for coords in components_mask(grid == color, connectivity=connectivity):
                components.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
    else:
        for coords in components_mask(grid != bg, connectivity=connectivity):
            components.append(
                {
                    "color": None,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    components.sort(key=lambda comp: comp["bbox"])
    return components


def _coords_set(coords: np.ndarray | list[tuple[int, int]] | set[tuple[int, int]]) -> set[tuple[int, int]]:
    if isinstance(coords, set):
        return {(int(r), int(c)) for r, c in coords}
    if isinstance(coords, list):
        return {(int(r), int(c)) for r, c in coords}
    return {(int(r), int(c)) for r, c in np.asarray(coords, dtype=int)}


def _sorted_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    return np.array(sorted((int(r), int(c)) for r, c in coords), dtype=int)


def _frame_border_mask(h: int, w: int) -> np.ndarray:
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _frame_side_from_marker(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = map(int, marker)
    r0, c0, r1, c1 = map(int, bbox)
    candidates = []
    if r == r0 and c0 < c < c1:
        candidates.append("top")
    if r == r1 and c0 < c < c1:
        candidates.append("bottom")
    if c == c0 and r0 < r < r1:
        candidates.append("left")
    if c == c1 and r0 < r < r1:
        candidates.append("right")
    assert len(candidates) == 1, "marker must lie on exactly one non-corner side"
    return candidates[0]


def _marker_target(marker: tuple[int, int], side: str) -> tuple[int, int]:
    r, c = map(int, marker)
    if side == "top":
        return int(r + 1), int(c)
    if side == "bottom":
        return int(r - 1), int(c)
    if side == "left":
        return int(r), int(c + 1)
    assert side == "right"
    return int(r), int(c - 1)


def _reflect_coords_across_frame(
    coords: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    out: list[tuple[int, int]] = []
    if side in HORIZONTAL_SIDES:
        for r, c in coords:
            rr = int(round(2 * center_r - int(r)))
            out.append((rr, int(c)))
    else:
        for r, c in coords:
            cc = int(round(2 * center_c - int(c)))
            out.append((int(r), cc))
    return np.array(out, dtype=int)


def _reflect_relative(
    rel_coords: set[tuple[int, int]],
    axis: str,
) -> set[tuple[int, int]]:
    if axis == "horizontal":
        return {(-int(r), int(c)) for r, c in rel_coords}
    assert axis == "vertical"
    return {(int(r), -int(c)) for r, c in rel_coords}


def _normalize_relative(
    coords: np.ndarray,
    pivot: tuple[int, int],
) -> set[tuple[int, int]]:
    pr, pc = map(int, pivot)
    return {(int(r) - pr, int(c) - pc) for r, c in coords}


def _nearest_sprite_index(coords: np.ndarray, marker: tuple[int, int]) -> int:
    mr, mc = map(int, marker)
    l1 = np.abs(coords[:, 0] - mr) + np.abs(coords[:, 1] - mc)
    best_l1 = int(l1.min())
    idxs = np.flatnonzero(l1 == best_l1)
    if idxs.size == 1:
        return int(idxs[0])

    l2 = (coords[idxs, 0] - mr) ** 2 + (coords[idxs, 1] - mc) ** 2
    best_l2 = int(l2.min())
    refined = idxs[l2 == best_l2]
    if refined.size == 1:
        return int(refined[0])

    order = sorted(
        (int(coords[i, 0]), int(coords[i, 1]), int(i))
        for i in refined
    )
    return int(order[0][2])


def _parse_frame_component(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    coords = np.asarray(comp["coords"], dtype=int)
    bbox = comp["bbox"]
    r0, c0, r1, c1 = map(int, bbox)
    h = bbox_h(bbox)
    w = bbox_w(bbox)
    if h < 4 or w < 4:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = _frame_border_mask(h, w)
    if not np.array_equal(mask, border):
        return None
    if np.any(sub[1:-1, 1:-1] != bg):
        return None

    counts = Counter(int(grid[int(r), int(c)]) for r, c in coords)
    if len(counts) != 2:
        return None
    marker_candidates = [color for color, count in counts.items() if int(count) == 1]
    if len(marker_candidates) != 1:
        return None
    marker_color = int(marker_candidates[0])
    frame_color = int([color for color in counts if color != marker_color][0])
    marker_cells = coords[np.array([int(grid[int(r), int(c)]) == marker_color for r, c in coords], dtype=bool)]
    assert marker_cells.shape[0] == 1
    marker = tuple(map(int, marker_cells[0]))
    side = _frame_side_from_marker(marker, bbox)
    target = _marker_target(marker, side)
    return {
        "bbox": bbox,
        "frame_color": frame_color,
        "marker_color": marker_color,
        "marker": marker,
        "side": side,
        "target": target,
        "interior_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
    }


def _parse_sprite_component(grid: np.ndarray, comp: dict) -> dict | None:
    coords = np.asarray(comp["coords"], dtype=int)
    colors = {int(grid[int(r), int(c)]) for r, c in coords}
    if len(colors) != 1:
        return None
    return {
        "coords": coords,
        "color": int(next(iter(colors))),
        "bbox": comp["bbox"],
        "area": int(coords.shape[0]),
    }


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False)

    frames: list[dict] = []
    sprites: list[dict] = []
    for comp in comps:
        frame = _parse_frame_component(g, comp, bg)
        if frame is not None:
            frames.append(frame)
            continue
        sprite = _parse_sprite_component(g, comp)
        assert sprite is not None, "every non-frame component must be a monochrome sprite"
        sprites.append(sprite)

    assert frames, "expected at least one frame"
    assert len(frames) == len(sprites), "expected a matching sprite for each frame"

    frame_by_marker = {frame["marker_color"]: frame for frame in frames}
    assert len(frame_by_marker) == len(frames), "marker colors must be unique"
    sprite_by_color = {sprite["color"]: sprite for sprite in sprites}
    assert len(sprite_by_color) == len(sprites), "sprite colors must be unique"
    assert set(frame_by_marker) == set(sprite_by_color), "sprite colors must match marker colors"

    out = g.copy()
    for sprite in sprites:
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = bg

    for marker_color in sorted(frame_by_marker):
        frame = frame_by_marker[int(marker_color)]
        sprite = sprite_by_color[int(marker_color)]
        marker = frame["marker"]
        pivot_idx = _nearest_sprite_index(sprite["coords"], marker)
        pivot = tuple(map(int, sprite["coords"][pivot_idx]))

        reflected = _reflect_coords_across_frame(sprite["coords"], frame["bbox"], frame["side"])
        reflected_pivot = tuple(map(int, reflected[pivot_idx]))
        target = frame["target"]
        dr = int(target[0] - reflected_pivot[0])
        dc = int(target[1] - reflected_pivot[1])
        translated = reflected + np.array([[dr, dc]], dtype=int)

        ir0, ic0, ir1, ic1 = map(int, frame["interior_bbox"])
        for r, c in translated:
            rr = int(r)
            cc = int(c)
            assert ir0 <= rr <= ir1 and ic0 <= cc <= ic1, "stamp must remain inside frame interior"
            out[rr, cc] = int(frame["frame_color"])
        out[int(target[0]), int(target[1])] = int(frame["marker_color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each outside sprite to its colored frame marker, mirror it using the marker side, and stamp the recolored copy inside the frame.

    Input structure:
    - Black background grid with several hollow one-pixel-thick rectangular frames.
    - Each frame has one border marker pixel whose color is different from the frame color.
    - Outside the frames sit monochrome 4-connected sprites whose colors match the marker colors one-to-one.

    Rule:
    - Identify each hollow frame, its border marker, and the monochrome sprite sharing that marker color.
    - If the marker is on the top or bottom side, reflect the sprite across the horizontal center line of its frame; if the marker is on the left or right side, reflect it across the vertical center line.
    - Choose the sprite pixel nearest to the marker, then translate the reflected sprite so that this paired pixel lands in the cell immediately inside the frame along the marker's inward normal.
    - Recolor the stamped sprite to the frame color, except keep the aligned nearest pixel in the original marker color, erase the original outside sprite, and leave the frame border, marker, and untouched background unchanged.

    Key insight:
    - The marker color chooses the destination frame, and the marker side locally chooses which mirror axis the sprite must use before it can be stamped inside that frame.
    """
    return _solve(np.asarray(grid, dtype=int))


def _grow_connected_shape(allowed: set[tuple[int, int]], size: int) -> set[tuple[int, int]] | None:
    cells = {(0, 0)}
    while len(cells) < int(size):
        frontier = []
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                pos = (int(nr), int(nc))
                if pos in allowed and pos not in cells:
                    frontier.append(pos)
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _make_final_rel(side: str, interior_h: int, interior_w: int) -> set[tuple[int, int]] | None:
    for _ in range(GEN_SHAPE_ATTEMPTS):
        size = random.randint(3, 6)
        if side in HORIZONTAL_SIDES:
            depth = random.randint(1, min(3, interior_h - 1))
            left = random.randint(0, min(2, interior_w - 1))
            right = random.randint(0, min(2, interior_w - 1))
            allowed = {(r, c) for r in range(depth + 1) for c in range(-left, right + 1)}
            rel = _grow_connected_shape(allowed, min(size, len(allowed)))
            if rel is None or all(r == 0 for r, _c in rel):
                continue
            if side == "bottom":
                rel = {(-r, c) for r, c in rel}
        else:
            depth = random.randint(1, min(3, interior_w - 1))
            up = random.randint(0, min(2, interior_h - 1))
            down = random.randint(0, min(2, interior_h - 1))
            allowed = {(r, c) for r in range(-up, down + 1) for c in range(depth + 1)}
            rel = _grow_connected_shape(allowed, min(size, len(allowed)))
            if rel is None or all(c == 0 for _r, c in rel):
                continue
            if side == "right":
                rel = {(r, -c) for r, c in rel}
        return {(int(r), int(c)) for r, c in rel}
    return None


def _choose_target_offset(
    side: str,
    rel: set[tuple[int, int]],
    interior_h: int,
    interior_w: int,
) -> tuple[int, int] | None:
    rows = [int(r) for r, _c in rel]
    cols = [int(c) for _r, c in rel]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    if side == "top":
        lo = -min_c
        hi = interior_w - 1 - max_c
        if lo > hi:
            return None
        return 0, random.randint(int(lo), int(hi))
    if side == "bottom":
        lo = -min_c
        hi = interior_w - 1 - max_c
        if lo > hi:
            return None
        return interior_h - 1, random.randint(int(lo), int(hi))
    if side == "left":
        lo = -min_r
        hi = interior_h - 1 - max_r
        if lo > hi:
            return None
        return random.randint(int(lo), int(hi)), 0
    lo = -min_r
    hi = interior_h - 1 - max_r
    if lo > hi:
        return None
    return random.randint(int(lo), int(hi)), interior_w - 1


def _target_from_bbox_and_offset(
    bbox: tuple[int, int, int, int],
    target_offset: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, _r1, _c1 = map(int, bbox)
    return int(r0 + 1 + target_offset[0]), int(c0 + 1 + target_offset[1])


def _marker_from_target(target: tuple[int, int], side: str) -> tuple[int, int]:
    r, c = map(int, target)
    if side == "top":
        return int(r - 1), int(c)
    if side == "bottom":
        return int(r + 1), int(c)
    if side == "left":
        return int(r), int(c - 1)
    assert side == "right"
    return int(r), int(c + 1)


def _place_input_sprite(
    grid_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    side: str,
    marker: tuple[int, int],
    input_rel: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = map(int, grid_shape)
    r0, c0, r1, c1 = map(int, frame_bbox)
    rows = [int(r) for r, _c in input_rel]
    cols = [int(c) for _r, c in input_rel]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    mr, mc = map(int, marker)

    if side == "top":
        low = max(-min_r, r0 - 4)
        high = r0 - 2
        if low > high:
            return None
        return random.randint(int(low), int(high)), int(mc)
    if side == "bottom":
        low = r1 + 2
        high = min(h - 1 - max_r, r1 + 4)
        if low > high:
            return None
        return random.randint(int(low), int(high)), int(mc)
    if side == "left":
        low = max(-min_c, c0 - 4)
        high = c0 - 2
        if low > high:
            return None
        return int(mr), random.randint(int(low), int(high))
    low = c1 + 2
    high = min(w - 1 - max_c, c1 + 4)
    if low > high:
        return None
    return int(mr), random.randint(int(low), int(high))


def _abs_coords(pivot: tuple[int, int], rel: set[tuple[int, int]]) -> set[tuple[int, int]]:
    pr, pc = map(int, pivot)
    return {(int(pr + r), int(pc + c)) for r, c in rel}


def _all_in_bounds(shape: tuple[int, int], coords: set[tuple[int, int]]) -> bool:
    return all(in_bounds(shape, r, c) for r, c in coords)


def _expanded_coords(
    coords: set[tuple[int, int]],
    shape: tuple[int, int],
    connectivity: int = 8,
) -> set[tuple[int, int]]:
    expanded = set()
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    else:
        deltas = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 0),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    for r, c in coords:
        for dr, dc in deltas:
            nr = int(r) + int(dr)
            nc = int(c) + int(dc)
            if in_bounds(shape, nr, nc):
                expanded.add((int(nr), int(nc)))
    return expanded


def _draw_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_color: int,
    marker_color: int,
    marker: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = int(frame_color)
    grid[r1, c0 : c1 + 1] = int(frame_color)
    grid[r0 : r1 + 1, c0] = int(frame_color)
    grid[r0 : r1 + 1, c1] = int(frame_color)
    grid[int(marker[0]), int(marker[1])] = int(marker_color)


def _expected_output_from_spec(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for spec in specs:
        for r, c in spec["input_coords"]:
            out[int(r), int(c)] = Color.BLACK
    for spec in specs:
        for r, c in spec["output_coords"]:
            out[int(r), int(c)] = int(spec["frame_color"])
        tr, tc = spec["target"]
        out[int(tr), int(tc)] = int(spec["marker_color"])
    return out


def _same_unique_nearest_under_l1_l2(
    coords: set[tuple[int, int]],
    marker: tuple[int, int],
    pivot: tuple[int, int],
) -> bool:
    ordered = sorted((int(r), int(c)) for r, c in coords)
    mr, mc = map(int, marker)
    l1 = [abs(r - mr) + abs(c - mc) for r, c in ordered]
    l2 = [(r - mr) ** 2 + (c - mc) ** 2 for r, c in ordered]
    best_l1 = min(l1)
    best_l2 = min(l2)
    if l1.count(best_l1) != 1 or l2.count(best_l2) != 1:
        return False
    return ordered[l1.index(best_l1)] == ordered[l2.index(best_l2)] == tuple(map(int, pivot))


def _translated_rel(rel: set[tuple[int, int]], target: tuple[int, int]) -> set[tuple[int, int]]:
    tr, tc = map(int, target)
    return {(int(tr + r), int(tc + c)) for r, c in rel}


def _shortcut_distinct(spec: dict) -> bool:
    target = spec["target"]
    correct = spec["output_coords"]
    input_rel = spec["input_rel"]
    no_reflect = _translated_rel(input_rel, target)
    if no_reflect == correct:
        return False

    wrong_axis = "vertical" if spec["axis"] == "horizontal" else "horizontal"
    wrong_rel = _reflect_relative(input_rel, wrong_axis)
    wrong = _translated_rel(wrong_rel, target)
    if wrong == correct:
        return False

    return True


def _input_visual_components_ok(grid: np.ndarray) -> bool:
    bg = get_bg(grid)
    comps4 = find_components(grid, bg=bg, connectivity=4, monochromatic=False)
    comps8 = find_components(grid, bg=bg, connectivity=8, monochromatic=False)
    return len(comps4) == 2 * GEN_FRAME_COUNT and len(comps8) == 2 * GEN_FRAME_COUNT


def _candidate_quadrants(h: int, w: int) -> list[tuple[int, int, int, int]]:
    return [
        (GEN_FRAME_PAD, h // 2 - 2, GEN_FRAME_PAD, w // 2 - 2),
        (GEN_FRAME_PAD, h // 2 - 2, w // 2 + 1, w - GEN_FRAME_PAD - 1),
        (h // 2 + 1, h - GEN_FRAME_PAD - 1, GEN_FRAME_PAD, w // 2 - 2),
        (h // 2 + 1, h - GEN_FRAME_PAD - 1, w // 2 + 1, w - GEN_FRAME_PAD - 1),
    ]


def _pick_frame_bboxes(
    h: int,
    w: int,
) -> list[tuple[int, int, int, int, int, int]] | None:
    quadrants = random.sample(_candidate_quadrants(h, w), k=GEN_FRAME_COUNT)
    bboxes: list[tuple[int, int, int, int, int, int]] = []
    for q_r0, q_r1, q_c0, q_c1 in quadrants:
        interior_h = random.randint(GEN_INTERIOR_MIN, GEN_INTERIOR_MAX)
        interior_w = random.randint(GEN_INTERIOR_MIN, GEN_INTERIOR_MAX)
        outer_h = interior_h + 2
        outer_w = interior_w + 2
        max_r0 = q_r1 - outer_h + 1
        max_c0 = q_c1 - outer_w + 1
        if q_r0 > max_r0 or q_c0 > max_c0:
            return None
        r0 = random.randint(int(q_r0), int(max_r0))
        c0 = random.randint(int(q_c0), int(max_c0))
        bboxes.append((int(r0), int(c0), int(r0 + outer_h - 1), int(c0 + outer_w - 1), int(interior_h), int(interior_w)))
    return bboxes


def _generate_case() -> tuple[np.ndarray, list[dict]] | None:
    h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    bboxes = _pick_frame_bboxes(h, w)
    if bboxes is None:
        return None

    colors = random.sample(Color.NOT_BLACK, 2 * GEN_FRAME_COUNT)
    frame_colors = colors[:GEN_FRAME_COUNT]
    marker_colors = colors[GEN_FRAME_COUNT:]
    sides = random.sample(SIDES, k=GEN_FRAME_COUNT)
    if not (set(sides) & HORIZONTAL_SIDES and set(sides) & VERTICAL_SIDES):
        return None

    grid = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    specs: list[dict] = []

    for i, side in enumerate(sides):
        r0, c0, r1, c1, interior_h, interior_w = bboxes[i]
        bbox = (r0, c0, r1, c1)
        final_rel = _make_final_rel(side, interior_h, interior_w)
        if final_rel is None:
            return None
        target_offset = _choose_target_offset(side, final_rel, interior_h, interior_w)
        if target_offset is None:
            return None
        target = _target_from_bbox_and_offset(bbox, target_offset)
        marker = _marker_from_target(target, side)
        axis = "horizontal" if side in HORIZONTAL_SIDES else "vertical"
        input_rel = _reflect_relative(final_rel, axis)
        pivot = _place_input_sprite((h, w), bbox, side, marker, input_rel)
        if pivot is None:
            return None
        input_coords = _abs_coords(pivot, input_rel)
        output_coords = _abs_coords(target, final_rel)

        interior = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        ir0, ic0, ir1, ic1 = interior
        if not all(ir0 <= r <= ir1 and ic0 <= c <= ic1 for r, c in output_coords):
            return None
        if not _all_in_bounds((h, w), input_coords):
            return None

        frame_border = {
            (rr, cc)
            for rr in range(r0, r1 + 1)
            for cc in range(c0, c1 + 1)
            if rr in (r0, r1) or cc in (c0, c1)
        }
        occupied_guard = _expanded_coords(occupied, (h, w), connectivity=8)
        frame_guard = _expanded_coords(frame_border, (h, w), connectivity=8)
        if frame_border & occupied_guard:
            return None
        if input_coords & frame_border:
            return None
        if any((r0 <= r <= r1 and c0 <= c <= c1) for r, c in input_coords):
            return None
        if input_coords & occupied_guard:
            return None
        if input_coords & frame_guard:
            return None

        occupied |= frame_border
        occupied |= input_coords
        _draw_frame(grid, bbox, frame_colors[i], marker_colors[i], marker)
        for r, c in input_coords:
            grid[int(r), int(c)] = int(marker_colors[i])

        specs.append(
            {
                "bbox": bbox,
                "frame_color": int(frame_colors[i]),
                "marker_color": int(marker_colors[i]),
                "marker": marker,
                "target": target,
                "axis": axis,
                "side": side,
                "final_rel": final_rel,
                "input_rel": input_rel,
                "pivot": pivot,
                "input_coords": input_coords,
                "output_coords": output_coords,
            }
        )

    expected = _expected_output_from_spec(grid, specs)
    if not np.array_equal(main(grid), expected):
        return None
    if not _input_visual_components_ok(grid):
        return None

    for spec in specs:
        if not _same_unique_nearest_under_l1_l2(spec["input_coords"], spec["marker"], spec["pivot"]):
            return None
        if not _shortcut_distinct(spec):
            return None

    return grid, specs


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid reflection-stamp ARC input")
