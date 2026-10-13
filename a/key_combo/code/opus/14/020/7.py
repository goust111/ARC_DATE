from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


BACKGROUND = 0
GREY = 5
OUTPUT_COLOR_BY_DISTANCE = {
    "near": 3,   # D <= 2
    "mid": 2,    # 3 <= D <= 4
    "far": 1,    # D >= 5
}
INPUT_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_HEIGHTS = tuple(range(13, 18))
GEN_WIDTHS = tuple(range(13, 18))
GEN_SINGLETON_COUNTS = (5, 6)
GEN_DISTANCES = (1, 2, 3, 4, 5, 6)
GEN_RETRIES = 800


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def sign(x: int) -> int:
    return (x > 0) - (x < 0)


def chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def neighbors8(r: int, c: int) -> list[Coord]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def component_coords(mask: np.ndarray, connectivity: int = 4) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dirs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def bbox_of(coords: list[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def anchor_cells(anchor: BBox) -> list[Coord]:
    r0, c0, r1, c1 = anchor
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def frame_cells(anchor: BBox, shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    r0, c0, r1, c1 = anchor
    cells: list[Coord] = []
    for r in range(r0 - 1, r1 + 2):
        for c in range(c0 - 1, c1 + 2):
            if 0 <= r < h and 0 <= c < w and not (r0 <= r <= r1 and c0 <= c <= c1):
                cells.append((int(r), int(c)))
    return sorted(cells)


def frame_kind(anchor: BBox, cell: Coord) -> str:
    r0, c0, r1, c1 = anchor
    r, c = cell
    if r in (r0 - 1, r1 + 1) and c in (c0 - 1, c1 + 1):
        return "corner"
    return "edge"


def distance_to_block(point: Coord, anchor: BBox) -> int:
    r, c = point
    r0, c0, r1, c1 = anchor
    dr = max(r0 - r, 0, r - r1)
    dc = max(c0 - c, 0, c - c1)
    return max(dr, dc)


def nearest_anchor_index(point: Coord, anchors: list[BBox]) -> int | None:
    distances = [distance_to_block(point, anchor) for anchor in anchors]
    best = min(distances)
    if distances.count(best) != 1:
        return None
    return int(distances.index(best))


def target_frame_info(point: Coord, anchor: BBox, shape: tuple[int, int]) -> tuple[Coord, bool]:
    cells = frame_cells(anchor, shape)
    best_dist = min(chebyshev(point, cell) for cell in cells)
    tied = [cell for cell in cells if chebyshev(point, cell) == best_dist]
    target = min(tied, key=lambda cell: (cell[0], cell[1]))
    return target, len(tied) > 1


def king_path(start: Coord, target: Coord) -> list[Coord]:
    r, c = start
    tr, tc = target
    path = [(int(r), int(c))]
    while (r, c) != (tr, tc):
        r += sign(tr - r)
        c += sign(tc - c)
        path.append((int(r), int(c)))
    return path


def distance_color(distance: int) -> int:
    if distance <= 2:
        return OUTPUT_COLOR_BY_DISTANCE["near"]
    if distance <= 4:
        return OUTPUT_COLOR_BY_DISTANCE["mid"]
    return OUTPUT_COLOR_BY_DISTANCE["far"]


def detect_anchors(grid: np.ndarray, bg: int) -> list[BBox]:
    comps = component_coords(np.asarray(grid, dtype=int) == GREY, connectivity=4)
    anchors: list[BBox] = []
    for pts in comps:
        b = bbox_of(pts)
        r0, c0, r1, c1 = b
        if len(pts) != 4 or (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            raise AssertionError("grey components must be solid 2x2 anchors")
        anchors.append(b)
    anchors.sort()
    if len(anchors) != 2:
        raise AssertionError("expected exactly two grey anchors")
    return anchors


def singleton_points(grid: np.ndarray, bg: int) -> list[Coord]:
    mask = (grid != bg) & (grid != GREY)
    comps = component_coords(mask, connectivity=8)
    dots: list[Coord] = []
    for pts in comps:
        if len(pts) != 1:
            raise AssertionError("all non-grey colored objects must be singleton pixels")
        dots.append(pts[0])
    dots.sort()
    return dots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Snap each colored singleton onto the nearest grey anchor frame and repaint its full trail by move distance.

    Input structure:
    - Black is the background.
    - Exactly two grey 2x2 solid blocks act as anchors, each with a one-cell outer frame.
    - Five or six non-grey singleton pixels of distinct colors are scattered away from the anchors.

    Rule:
    - Keep both grey 2x2 anchors unchanged.
    - For each singleton, find the unique nearest anchor by Chebyshev distance to the nearest anchor cell.
    - Within that anchor's one-cell outer frame, choose the frame cell with minimum Chebyshev distance to the singleton; if several tie, pick the one with smallest row, then smallest column.
    - Let D be the Chebyshev distance from the singleton to that target frame cell. Recolor the singleton, every intermediate cell on the greedy shortest king-move path, and the target cell with green if D <= 2, red if 3 <= D <= 4, and blue if D >= 5.
    - The singleton's original color disappears completely.

    Key insight:
    - The same movement distance simultaneously decides which anchor frame cell is reached and which final color the whole trail receives, so source color is only a distractor.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    anchors = detect_anchors(g, bg)
    dots = singleton_points(g, bg)

    out = np.full_like(g, bg)
    for anchor in anchors:
        for cell in anchor_cells(anchor):
            out[cell] = GREY

    for dot in dots:
        anchor_idx = nearest_anchor_index(dot, anchors)
        if anchor_idx is None:
            raise AssertionError("each singleton must have a unique nearest anchor")
        target, _ = target_frame_info(dot, anchors[anchor_idx], g.shape)
        distance = chebyshev(dot, target)
        color = distance_color(distance)
        for r, c in king_path(dot, target):
            if g[r, c] == GREY:
                raise AssertionError("trail cannot pass through an anchor")
            out[r, c] = color
    return out


def bbox_chebyshev_gap(a: BBox, b: BBox) -> int:
    def axis_gap(a0: int, a1: int, b0: int, b1: int) -> int:
        if a1 < b0:
            return b0 - a1
        if b1 < a0:
            return a0 - b1
        return 0

    return max(axis_gap(a[0], a[2], b[0], b[2]), axis_gap(a[1], a[3], b[1], b[3]))


def choose_anchor_pair(rng: random.Random, h: int, w: int) -> list[BBox] | None:
    starts = [(r, c) for r in range(1, h - 2) for c in range(1, w - 2)]
    rng.shuffle(starts)
    for (r0, c0), (r1, c1) in combinations(starts, 2):
        anchors = [(r0, c0, r0 + 1, c0 + 1), (r1, c1, r1 + 1, c1 + 1)]
        if bbox_chebyshev_gap(anchors[0], anchors[1]) < 5:
            continue
        return sorted(anchors)
    return None


def move_mode(source: Coord, target: Coord) -> str:
    dr = abs(source[0] - target[0])
    dc = abs(source[1] - target[1])
    if dr == 0 or dc == 0:
        return "orth"
    if dr == dc:
        return "pure_diag"
    return "mixed"


def compute_candidate_options(shape: tuple[int, int], anchors: list[BBox]) -> dict[tuple[int, int], list[dict]]:
    h, w = shape
    anchor_set = {cell for anchor in anchors for cell in anchor_cells(anchor)}
    frame_set = {cell for anchor in anchors for cell in frame_cells(anchor, shape)}
    options: dict[tuple[int, int], list[dict]] = {}

    for r in range(h):
        for c in range(w):
            source = (r, c)
            if source in anchor_set or source in frame_set:
                continue
            anchor_idx = nearest_anchor_index(source, anchors)
            if anchor_idx is None:
                continue
            anchor = anchors[anchor_idx]
            target, tied = target_frame_info(source, anchor, shape)
            distance = chebyshev(source, target)
            if distance not in GEN_DISTANCES:
                continue
            path = king_path(source, target)
            if any(cell in anchor_set for cell in path):
                continue
            if any(cell in frame_set and cell != target for cell in path):
                continue
            option = {
                "source": source,
                "target": target,
                "anchor_idx": anchor_idx,
                "distance": distance,
                "block_distance": distance_to_block(source, anchor),
                "tie_target": tied,
                "target_kind": frame_kind(anchor, target),
                "move_mode": move_mode(source, target),
                "path": tuple(path),
            }
            options.setdefault((anchor_idx, distance), []).append(option)

    for group in options.values():
        group.sort(key=lambda opt: (opt["source"], opt["target"]))
    return options


def choose_distances(rng: random.Random, count: int) -> list[int]:
    while True:
        chosen = set()
        chosen.add(rng.choice([1, 2]))
        chosen.add(rng.choice([3, 4]))
        chosen.add(rng.choice([5, 6]))
        while len(chosen) < count:
            chosen.add(rng.choice(list(GEN_DISTANCES)))
        if len(chosen) == count:
            return sorted(chosen)


def anchor_count_splits(count: int) -> list[tuple[int, int]]:
    if count == 5:
        return [(2, 3), (3, 2)]
    return [(3, 3), (2, 4), (4, 2)]


def source_isolated(source: Coord, others: set[Coord]) -> bool:
    return all(chebyshev(source, other) >= 2 for other in others)


def selection_has_required_variety(chosen: list[dict]) -> bool:
    branches = {distance_color(int(opt["distance"])) for opt in chosen}
    if branches != {1, 2, 3}:
        return False
    if not any(bool(opt["tie_target"]) for opt in chosen):
        return False
    if not any(opt["target_kind"] == "corner" for opt in chosen):
        return False
    if not any(opt["target_kind"] == "edge" for opt in chosen):
        return False
    if not any(opt["move_mode"] == "mixed" for opt in chosen):
        return False
    return True


def choose_placements(
    rng: random.Random,
    shape: tuple[int, int],
    anchors: list[BBox],
    count: int,
) -> list[dict] | None:
    distances = choose_distances(rng, count)
    options = compute_candidate_options(shape, anchors)

    for split in anchor_count_splits(count):
        left_count, right_count = split
        assignments = list(combinations(distances, left_count))
        rng.shuffle(assignments)
        for left_distances in assignments:
            left_set = set(left_distances)
            slot_defs = [(0, dist) for dist in distances if dist in left_set] + [
                (1, dist) for dist in distances if dist not in left_set
            ]
            slot_options: list[tuple[tuple[int, int], list[dict]]] = []
            valid = True
            for slot in slot_defs:
                group = list(options.get(slot, []))
                rng.shuffle(group)
                if not group:
                    valid = False
                    break
                slot_options.append((slot, group))
            if not valid:
                continue

            slot_options.sort(key=lambda item: len(item[1]))
            chosen: list[dict] = []
            used_sources: set[Coord] = set()
            used_targets: set[Coord] = set()
            used_path_cells: set[Coord] = set()

            def backtrack(i: int) -> list[dict] | None:
                if i == len(slot_options):
                    if selection_has_required_variety(chosen):
                        return [dict(item) for item in chosen]
                    return None

                _slot, group = slot_options[i]
                for opt in group:
                    source = opt["source"]
                    target = opt["target"]
                    path_cells = set(opt["path"])
                    if source in used_sources or target in used_targets:
                        continue
                    if not source_isolated(source, used_sources):
                        continue
                    if path_cells & used_path_cells:
                        continue

                    chosen.append(opt)
                    used_sources.add(source)
                    used_targets.add(target)
                    used_path_cells.update(path_cells)

                    result = backtrack(i + 1)
                    if result is not None:
                        return result

                    chosen.pop()
                    used_sources.remove(source)
                    used_targets.remove(target)
                    for cell in path_cells:
                        if all(cell not in set(item["path"]) for item in chosen):
                            used_path_cells.discard(cell)
                return None

            found = backtrack(0)
            if found is not None:
                return found
    return None


def assign_input_colors(rng: random.Random, count: int) -> list[int]:
    mandatory = [1, 2, 3]
    rest = [color for color in INPUT_COLOR_POOL if color not in mandatory]
    colors = mandatory + rng.sample(rest, count - len(mandatory))
    rng.shuffle(colors)
    return colors


def build_grid(shape: tuple[int, int], anchors: list[BBox], placements: list[dict], colors: list[int]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for anchor in anchors:
        for r, c in anchor_cells(anchor):
            grid[r, c] = GREY
    for opt, color in zip(placements, colors):
        r, c = opt["source"]
        grid[r, c] = int(color)
    return grid


def extracted_singleton_info(grid: np.ndarray) -> list[dict]:
    bg = get_bg(grid)
    anchors = detect_anchors(grid, bg)
    dots = singleton_points(grid, bg)
    info: list[dict] = []
    for dot in dots:
        anchor_idx = nearest_anchor_index(dot, anchors)
        if anchor_idx is None:
            raise AssertionError("nearest anchor must be unique")
        target, tied = target_frame_info(dot, anchors[anchor_idx], grid.shape)
        distance = chebyshev(dot, target)
        info.append(
            {
                "source": dot,
                "anchor_idx": anchor_idx,
                "target": target,
                "distance": distance,
                "block_distance": distance_to_block(dot, anchors[anchor_idx]),
                "color": int(grid[dot]),
                "tie_target": tied,
                "target_kind": frame_kind(anchors[anchor_idx], target),
                "move_mode": move_mode(dot, target),
                "path": tuple(king_path(dot, target)),
            }
        )
    return info


def validate_generated_grid(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        raise AssertionError("grid must be a 2D ARC grid with size <= 30")

    bg = get_bg(g)
    if bg != BACKGROUND:
        raise AssertionError("generator must use black background")

    anchors = detect_anchors(g, bg)
    if any(len(frame_cells(anchor, g.shape)) != 12 for anchor in anchors):
        raise AssertionError("anchors must have a full 12-cell outer frame")
    if bbox_chebyshev_gap(anchors[0], anchors[1]) < 5:
        raise AssertionError("anchors are too close")

    info = extracted_singleton_info(g)
    if len(info) not in GEN_SINGLETON_COUNTS:
        raise AssertionError("singleton count must be 5 or 6")

    source_colors = [item["color"] for item in info]
    if len(set(source_colors)) != len(source_colors):
        raise AssertionError("singleton colors must be distinct")

    counts_by_anchor = Counter(int(item["anchor_idx"]) for item in info)
    if set(counts_by_anchor) != {0, 1} or min(counts_by_anchor.values()) < 2:
        raise AssertionError("both anchors must attract at least two singletons")

    block_distances = [int(item["block_distance"]) for item in info]
    if len(set(block_distances)) != len(block_distances):
        raise AssertionError("nearest anchor distances must all be distinct")

    if not any(item["distance"] <= 2 for item in info):
        raise AssertionError("missing short-distance branch")
    if not any(3 <= item["distance"] <= 4 for item in info):
        raise AssertionError("missing mid-distance branch")
    if not any(item["distance"] >= 5 for item in info):
        raise AssertionError("missing long-distance branch")

    if not selection_has_required_variety(info):
        raise AssertionError("generator failed diversity checks")

    source_set = {item["source"] for item in info}
    for source in source_set:
        if not source_isolated(source, source_set - {source}):
            raise AssertionError("singletons must be isolated from each other")

    anchor_set = {cell for anchor in anchors for cell in anchor_cells(anchor)}
    frame_set = {cell for anchor in anchors for cell in frame_cells(anchor, g.shape)}
    trail_union: set[Coord] = set()
    for item in info:
        path = set(item["path"])
        if path & anchor_set:
            raise AssertionError("trail may not cross an anchor")
        if any(cell in frame_set and cell != item["target"] for cell in path):
            raise AssertionError("trail may only enter a frame at its own target")
        if trail_union & path:
            raise AssertionError("trails must not overlap")
        trail_union |= path

    out = main(g)
    if not np.array_equal(out[tuple(zip(*list(anchor_set)))], g[tuple(zip(*list(anchor_set)))]):
        raise AssertionError("anchors must remain unchanged")
    if not set(map(int, np.unique(out))) <= {0, 1, 2, 3, 5}:
        raise AssertionError("output may only contain background, anchor grey, and trail colors")
    if len(trail_union) <= len(info):
        raise AssertionError("at least one trail must extend beyond its start cell")

    for item in info:
        expected_color = distance_color(int(item["distance"]))
        for cell in item["path"]:
            if int(out[cell]) != expected_color:
                raise AssertionError("trail color does not match distance bucket")


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        shape = (rng.choice(GEN_HEIGHTS), rng.choice(GEN_WIDTHS))
        anchors = choose_anchor_pair(rng, *shape)
        if anchors is None:
            continue
        count = rng.choice(GEN_SINGLETON_COUNTS)
        placements = choose_placements(rng, shape, anchors, count)
        if placements is None:
            continue
        colors = assign_input_colors(rng, count)
        grid = build_grid(shape, anchors, placements, colors)
        try:
            validate_generated_grid(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
