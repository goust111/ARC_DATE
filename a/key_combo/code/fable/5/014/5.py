from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]  # inclusive (r0, c0, r1, c1)

_DIR4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_PALETTE = tuple(range(10))
_FRAME_SIZE = 7
_GEN_RETRIES = 500


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(shape: tuple[int, int], cell: Coord):
    h, w = shape
    r, c = cell
    for dr, dc in _DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _mono_components(grid: np.ndarray, bg: int) -> list[dict]:
    """Return monochromatic 4-components using the common (row, column) convention."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            cells: list[Coord] = []
            while stack:
                cell = stack.pop()
                cells.append(cell)
                for rr, cc in _neighbors4((h, w), cell):
                    if not seen[rr, cc] and int(grid[rr, cc]) == color:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            result.append(
                {
                    "color": color,
                    "cells": frozenset(cells),
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                }
            )
    return result


def _rect_border(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells.update((r1, c) for c in range(c0, c1 + 1))
    cells.update((r, c0) for r in range(r0 + 1, r1))
    cells.update((r, c1) for r in range(r0 + 1, r1))
    return cells


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    """Recognize exact monochromatic hollow rectangles, then use the shared size cue."""
    by_size: dict[tuple[int, int], list[dict]] = {}
    for comp in _mono_components(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        height, width = r1 - r0 + 1, c1 - c0 + 1
        if height < 4 or width < 4:
            continue
        if set(comp["cells"]) != _rect_border(comp["bbox"]):
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if not bool(np.any(interior != bg)):
            continue
        if bool(np.any(interior == int(comp["color"]))):
            continue
        frame = {
            "color": int(comp["color"]),
            "bbox": comp["bbox"],
            "border": comp["cells"],
            "mask": interior != bg,
        }
        by_size.setdefault((height, width), []).append(frame)

    eligible = {size: fs for size, fs in by_size.items() if len(fs) >= 2}
    assert eligible, "expected at least two same-size hollow frames"
    largest_count = max(len(fs) for fs in eligible.values())
    winners = [fs for fs in eligible.values() if len(fs) == largest_count]
    assert len(winners) == 1, "frame size must be uniquely determined by repetition"
    frames = winners[0]
    assert all(
        not _boxes_overlap(frames[i]["bbox"], frames[j]["bbox"])
        for i in range(len(frames))
        for j in range(i + 1, len(frames))
    ), "frames must not overlap"
    return sorted(frames, key=lambda f: f["bbox"])


def _path_pairs(grid: np.ndarray, bg: int, frames: list[dict]) -> list[tuple[int, int]]:
    h, w = grid.shape
    frame_at: dict[Coord, int] = {}
    covered: set[Coord] = set()
    for i, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        covered.update((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
        for cell in frame["border"]:
            frame_at[cell] = i

    external_components = []
    for comp in _mono_components(grid, bg):
        cells = set(comp["cells"])
        if cells.isdisjoint(covered):
            external_components.append(comp)
        else:
            assert cells <= covered, "a foreground component may not cross a frame boundary"

    assert external_components, "expected at least one colored path"
    path_colors = [int(comp["color"]) for comp in external_components]
    assert len(path_colors) == len(set(path_colors)), "each path must have a unique color"
    assert set(path_colors).isdisjoint({int(f["color"]) for f in frames}), (
        "path colors and frame colors must be distinct"
    )

    pairs: list[tuple[int, int]] = []
    used_frames: set[int] = set()
    for comp in external_components:
        cells = set(comp["cells"])
        degrees = {
            cell: sum(neighbor in cells for neighbor in _neighbors4((h, w), cell))
            for cell in cells
        }
        endpoints = [cell for cell, degree in degrees.items() if degree == 1]
        assert len(endpoints) == 2 and all(degree in (1, 2) for degree in degrees.values()), (
            "each path must be one unbranched 4-connected stroke"
        )

        endpoint_frames: list[int] = []
        for endpoint in endpoints:
            touched = {frame_at[n] for n in _neighbors4((h, w), endpoint) if n in frame_at}
            assert len(touched) == 1, "each path endpoint must touch exactly one frame"
            endpoint_frames.append(touched.pop())
        assert endpoint_frames[0] != endpoint_frames[1], "a path must join two different frames"

        for cell in cells - set(endpoints):
            assert not any(n in frame_at for n in _neighbors4((h, w), cell)), (
                "only path endpoints may touch frames"
            )
        assert not (set(endpoint_frames) & used_frames), "a frame may belong to only one swap"
        used_frames.update(endpoint_frames)
        pairs.append((endpoint_frames[0], endpoint_frames[1]))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the shapes inside path-paired frames, recolor them by destination frames, and empty isolated frames.

    Input structure:
    - The dominant color is the background; equal-size monochromatic hollow rectangular frames contain nonempty small patterns.
    - Every external foreground component is a uniquely colored, unbranched 4-connected path whose endpoints touch two different frame borders.
    - Path colors are distinct from frame colors, and no frame is touched by more than one path.

    Rule:
    - For every endpoint-paired frame pair P and Q, copy Q's interior mask to the same frame-relative cells of P and color it with P's border color; symmetrically copy P's mask to Q in Q's border color.
    - Clear the interiors of frames not incident to a path, while preserving every frame border and every path pixel.

    Key insight:
    - A path encodes only a pairing: its geometry and color do not move, the exchanged information is the endpoint frames' relative shape masks, and the destination border supplies the new color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30 in either dimension"
    bg = _background(g)
    frames = _find_frames(g, bg)
    pairs = _path_pairs(g, bg, frames)

    original_masks = [frame["mask"].copy() for frame in frames]
    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = bg

    for i, j in pairs:
        for destination, source in ((i, j), (j, i)):
            r0, c0, r1, c1 = frames[destination]["bbox"]
            interior = out[r0 + 1 : r1, c0 + 1 : c1]
            interior[original_masks[source]] = int(frames[destination]["color"])
    return out


def _frame_slots(height: int, width: int) -> list[Coord]:
    s = _FRAME_SIZE
    rows = (1, (height - s) // 2, height - s - 1)
    cols = (1, (width - s) // 2, width - s - 1)
    return [(r, c) for r in rows for c in cols]


def _frame_centers(boxes: list[BBox]) -> list[Coord]:
    return [((b[0] + b[2]) // 2, (b[1] + b[3]) // 2) for b in boxes]


def _choose_pairs(rng: random.Random, boxes: list[BBox], count: int) -> list[tuple[int, int]] | None:
    centers = _frame_centers(boxes)
    for _ in range(100):
        order = list(range(len(boxes)))
        rng.shuffle(order)
        pairs = [(order[2 * k], order[2 * k + 1]) for k in range(count)]
        path_is_essential = False
        for i, j in pairs:
            dij = abs(centers[i][0] - centers[j][0]) + abs(centers[i][1] - centers[j][1])
            if any(
                k not in (i, j)
                and abs(centers[i][0] - centers[k][0]) + abs(centers[i][1] - centers[k][1]) < dij
                for k in range(len(boxes))
            ):
                path_is_essential = True
        if path_is_essential:
            return pairs
    return None


def _random_connected_mask(rng: random.Random) -> np.ndarray:
    """Make a small connected shape in the central 3x3 of a 5x5 interior."""
    inner = np.zeros((_FRAME_SIZE - 2, _FRAME_SIZE - 2), dtype=bool)
    target = rng.randint(2, 7)
    cells = {(rng.randint(1, 3), rng.randint(1, 3))}
    while len(cells) < target:
        r, c = rng.choice(tuple(cells))
        candidates = [
            (r + dr, c + dc)
            for dr, dc in _DIR4
            if 1 <= r + dr <= 3 and 1 <= c + dc <= 3 and (r + dr, c + dc) not in cells
        ]
        if candidates:
            cells.add(rng.choice(candidates))
        elif len(cells) == 9:
            break
    for cell in cells:
        inner[cell] = True
    return inner


def _port_candidates(box: BBox, shape: tuple[int, int], all_boxes: list[BBox]) -> list[Coord]:
    h, w = shape
    r0, c0, r1, c1 = box
    raw = []
    for k in range(1, c1 - c0):
        raw.extend(((r0 - 1, c0 + k), (r1 + 1, c0 + k)))
    for k in range(1, r1 - r0):
        raw.extend(((r0 + k, c0 - 1), (r0 + k, c1 + 1)))

    result = []
    own_border = _rect_border(box)
    all_borders = [_rect_border(other) for other in all_boxes]
    for cell in raw:
        r, c = cell
        if not (0 <= r < h and 0 <= c < w):
            continue
        touched = {
            i for i, border in enumerate(all_borders)
            if any(neighbor in border for neighbor in _neighbors4(shape, cell))
        }
        own_index = all_boxes.index(box)
        if touched == {own_index} and any(n in own_border for n in _neighbors4(shape, cell)):
            result.append(cell)
    return result


def _shortest_path(
    rng: random.Random,
    shape: tuple[int, int],
    start: Coord,
    goal: Coord,
    blocked: set[Coord],
) -> list[Coord] | None:
    if start in blocked or goal in blocked:
        return None
    queue = deque([start])
    parent: dict[Coord, Coord | None] = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            break
        neighbors = list(_neighbors4(shape, cell))
        rng.shuffle(neighbors)
        for nxt in neighbors:
            if nxt not in blocked and nxt not in parent:
                parent[nxt] = cell
                queue.append(nxt)
    if goal not in parent:
        return None
    path = []
    cell: Coord | None = goal
    while cell is not None:
        path.append(cell)
        cell = parent[cell]
    return path[::-1]


def _turn_count(path: list[Coord]) -> int:
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return sum(steps[k] != steps[k - 1] for k in range(1, len(steps)))


def _route_pair(
    rng: random.Random,
    shape: tuple[int, int],
    boxes: list[BBox],
    pair: tuple[int, int],
    existing_paths: set[Coord],
) -> list[Coord] | None:
    h, w = shape
    frame_cells = {
        (r, c)
        for box in boxes
        for r in range(box[0], box[2] + 1)
        for c in range(box[1], box[3] + 1)
    }
    near_frames = set(frame_cells)
    for cell in frame_cells:
        near_frames.update(_neighbors4(shape, cell))

    path_clearance = set(existing_paths)
    for r, c in existing_paths:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    path_clearance.add((rr, cc))

    ports_a = _port_candidates(boxes[pair[0]], shape, boxes)
    ports_b = _port_candidates(boxes[pair[1]], shape, boxes)
    rng.shuffle(ports_a)
    rng.shuffle(ports_b)
    combinations = [(a, b) for a in ports_a for b in ports_b]
    rng.shuffle(combinations)
    for start, goal in combinations[:100]:
        blocked = near_frames | path_clearance
        blocked.discard(start)
        blocked.discard(goal)
        # Invisible random obstacles make route geometry varied without adding a new symbol.
        for r in range(h):
            for c in range(w):
                cell = (r, c)
                if cell not in blocked and cell not in (start, goal) and rng.random() < 0.085:
                    blocked.add(cell)
        path = _shortest_path(rng, shape, start, goal, blocked)
        if path is None or len(path) < 6 or _turn_count(path) < 2:
            continue
        cells = set(path)
        if any(sum(n in cells for n in _neighbors4(shape, p)) not in (1, 2) for p in cells):
            continue
        return path
    return None


def _render_expected(
    grid: np.ndarray,
    boxes: list[BBox],
    frame_colors: list[int],
    masks: list[np.ndarray],
    pairs: list[tuple[int, int]],
    bg: int,
) -> np.ndarray:
    out = grid.copy()
    for r0, c0, r1, c1 in boxes:
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
    for i, j in pairs:
        for destination, source in ((i, j), (j, i)):
            r0, c0, r1, c1 = boxes[destination]
            interior = out[r0 + 1 : r1, c0 + 1 : c1]
            interior[masks[source]] = frame_colors[destination]
    return out


def _valid_scene(
    grid: np.ndarray,
    expected: np.ndarray,
    boxes: list[BBox],
    frame_colors: list[int],
    masks: list[np.ndarray],
    pairs: list[tuple[int, int]],
    paths: list[list[Coord]],
    ink: int,
    bg: int,
) -> bool:
    try:
        actual = main(grid)
        found = _find_frames(grid, bg)
        found_pairs = {frozenset(p) for p in _path_pairs(grid, bg, found)}
    except (AssertionError, ValueError, IndexError):
        return False
    if not np.array_equal(actual, expected):
        return False
    found_boxes = [f["bbox"] for f in found]
    box_to_found = {box: i for i, box in enumerate(found_boxes)}
    if set(found_boxes) != set(boxes):
        return False
    planned_pairs = {
        frozenset((box_to_found[boxes[i]], box_to_found[boxes[j]])) for i, j in pairs
    }
    if found_pairs != planned_pairs or np.array_equal(grid, expected):
        return False

    paired = {i for pair in pairs for i in pair}
    if len(paired) == len(boxes):
        return False  # The isolated-frame erasure branch must be visible.
    if any(np.array_equal(masks[i], masks[j]) or int(masks[i].sum()) == int(masks[j].sum()) for i, j in pairs):
        return False  # Every exchange visibly changes both shape and occupied area.
    if any(_turn_count(path) < 2 for path in paths):
        return False
    if ink in frame_colors:
        return False

    changed_inside = np.zeros(grid.shape, dtype=bool)
    for box in boxes:
        changed_inside[box[0] + 1 : box[2], box[1] + 1 : box[3]] = True
    if not np.array_equal(grid[~changed_inside], expected[~changed_inside]):
        return False  # Borders and complete paths must be preserved exactly.
    for i in range(len(boxes)):
        r0, c0, r1, c1 = boxes[i]
        out_inner = expected[r0 + 1 : r1, c0 + 1 : c1]
        if i not in paired and bool(np.any(out_inner != bg)):
            return False
        if i in paired:
            vals = set(map(int, out_inner[out_inner != bg]))
            if vals != {frame_colors[i]}:
                return False
    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(27, 30)
    width = rng.randint(27, 30)
    frame_count = rng.choice((4, 5, 5))
    pair_count = 2 if frame_count == 5 and rng.random() < 0.72 else 1

    slots = _frame_slots(height, width)
    for _ in range(50):
        chosen = rng.sample(slots, frame_count)
        if len({r for r, _ in chosen}) >= 2 and len({c for _, c in chosen}) >= 2:
            break
    else:
        return None
    boxes = [(r, c, r + _FRAME_SIZE - 1, c + _FRAME_SIZE - 1) for r, c in chosen]
    pairs = _choose_pairs(rng, boxes, pair_count)
    if pairs is None:
        return None

    masks: list[np.ndarray] = []
    for _ in range(frame_count):
        for _ in range(80):
            mask = _random_connected_mask(rng)
            if not any(np.array_equal(mask, old) for old in masks):
                masks.append(mask)
                break
        else:
            return None
    if any(int(masks[i].sum()) == int(masks[j].sum()) for i, j in pairs):
        return None

    needed_colors = 1 + frame_count + pair_count + 1
    chosen_colors = rng.sample(_PALETTE, needed_colors)
    bg = chosen_colors[0]
    frame_colors = chosen_colors[1 : 1 + frame_count]
    path_colors = chosen_colors[1 + frame_count : 1 + frame_count + pair_count]
    ink = chosen_colors[-1]

    grid = np.full((height, width), bg, dtype=int)
    for box, color, mask in zip(boxes, frame_colors, masks):
        for cell in _rect_border(box):
            grid[cell] = color
        r0, c0, r1, c1 = box
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        interior[mask] = ink

    paths: list[list[Coord]] = []
    occupied_paths: set[Coord] = set()
    for pair, color in zip(pairs, path_colors):
        path = _route_pair(rng, grid.shape, boxes, pair, occupied_paths)
        if path is None:
            return None
        paths.append(path)
        occupied_paths.update(path)
        for cell in path:
            if int(grid[cell]) != bg:
                return None
            grid[cell] = color

    if _background(grid) != bg:
        return None
    expected = _render_expected(grid, boxes, frame_colors, masks, pairs, bg)
    if not _valid_scene(grid, expected, boxes, frame_colors, masks, pairs, paths, ink, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked instance with both swaps and isolated-frame erasure."""
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid path-pair wardrobe scene after {_GEN_RETRIES} attempts")
