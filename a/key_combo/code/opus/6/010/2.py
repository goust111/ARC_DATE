from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists() and str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, get_bg, in_bounds  # type: ignore


BG = 0
CONTENT = 7
FRAME_COLORS = [1, 2, 3, 4, 5, 6, 8, 9]
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
QUADRANTS = ("UR", "UL", "DR", "DL")


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return _bbox_h(bbox) * _bbox_w(bbox)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _strictly_inside(bbox: tuple[int, int, int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 < r < r1 and c0 < c < c1


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    out = set()
    for c in range(c0, c1 + 1):
        out.add((r0, c))
        out.add((r1, c))
    for r in range(r0, r1 + 1):
        out.add((r, c0))
        out.add((r, c1))
    return out


def _interior_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _unique_topo_order(edges: dict[int, set[int]]) -> list[int]:
    indeg = {node: 0 for node in edges}
    for src, dsts in edges.items():
        for dst in dsts:
            indeg[dst] += 1

    order: list[int] = []
    zeros = sorted(node for node, deg in indeg.items() if deg == 0)
    while zeros:
        if len(zeros) != 1:
            raise AssertionError("occlusion order is ambiguous")
        node = zeros.pop()
        order.append(node)
        unlocked = []
        for dst in sorted(edges[node]):
            indeg[dst] -= 1
            if indeg[dst] == 0:
                unlocked.append(dst)
        zeros = sorted(unlocked + zeros)
    if len(order) != len(edges):
        raise AssertionError("occlusion graph must be acyclic")
    return order


def _tail_candidates(coords_set: set[tuple[int, int]]) -> list[tuple[str, tuple[int, int]]]:
    by_row: dict[int, list[int]] = defaultdict(list)
    by_col: dict[int, list[int]] = defaultdict(list)
    for r, c in coords_set:
        by_row[r].append(c)
        by_col[c].append(r)

    rows = sorted(by_row)
    cols = sorted(by_col)
    out: list[tuple[str, tuple[int, int]]] = []

    min_row = rows[0]
    max_row = rows[-1]
    min_col = cols[0]
    max_col = cols[-1]

    if len(by_row[min_row]) == 1:
        tail = (min_row, by_row[min_row][0])
        if (tail[0] + 1, tail[1]) in coords_set:
            out.append(("U", tail))
    if len(by_row[max_row]) == 1:
        tail = (max_row, by_row[max_row][0])
        if (tail[0] - 1, tail[1]) in coords_set:
            out.append(("D", tail))
    if len(by_col[min_col]) == 1:
        tail = (by_col[min_col][0], min_col)
        if (tail[0], tail[1] + 1) in coords_set:
            out.append(("L", tail))
    if len(by_col[max_col]) == 1:
        tail = (by_col[max_col][0], max_col)
        if (tail[0], tail[1] - 1) in coords_set:
            out.append(("R", tail))
    return out


def _detect_frame(grid: np.ndarray, color: int) -> dict:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == int(color))]
    if not coords:
        raise AssertionError("missing frame color")
    coords_set = set(coords)

    candidates = _tail_candidates(coords_set)
    if len(candidates) != 1:
        raise AssertionError("frame tail must be uniquely recoverable")
    direction, tail = candidates[0]

    border_coords = [cell for cell in coords if cell != tail]
    bbox = bbox_coords(np.array(border_coords, dtype=int))
    perimeter = _perimeter_cells(bbox)

    if not all(cell in perimeter for cell in border_coords):
        raise AssertionError("visible frame pixels must lie on the rectangle perimeter")

    r0, c0, r1, c1 = bbox
    if not any(r == r0 for r, _ in border_coords):
        raise AssertionError("top edge not visible")
    if not any(r == r1 for r, _ in border_coords):
        raise AssertionError("bottom edge not visible")
    if not any(c == c0 for _, c in border_coords):
        raise AssertionError("left edge not visible")
    if not any(c == c1 for _, c in border_coords):
        raise AssertionError("right edge not visible")

    return {
        "color": int(color),
        "bbox": bbox,
        "direction": direction,
        "tail": tail,
        "visible": coords_set,
        "perimeter": perimeter,
    }


def _parse_scene(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    frame_colors = [c for c in colors_present(g, exclude={bg, CONTENT}) if c in FRAME_COLORS]
    if not frame_colors:
        raise AssertionError("no frames found")

    frames = [_detect_frame(g, color) for color in frame_colors]
    by_color = {frame["color"]: frame for frame in frames}

    orange_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == CONTENT)]
    if len(orange_cells) != len(frames):
        raise AssertionError("must have exactly one orange content pixel per frame")

    assigned_colors: set[int] = set()
    for cell in orange_cells:
        owners = [frame for frame in frames if _strictly_inside(frame["bbox"], cell)]
        if len(owners) != 1:
            raise AssertionError("each orange cell must belong to exactly one frame")
        owner = owners[0]
        if owner["color"] in assigned_colors:
            raise AssertionError("each frame must own exactly one orange cell")
        owner["content"] = cell
        assigned_colors.add(owner["color"])

    edges = {frame["color"]: set() for frame in frames}
    for i, left in enumerate(frames):
        for right in frames[i + 1 :]:
            inter = left["perimeter"] & right["perimeter"]
            if not inter:
                continue
            visible_colors = {int(g[r, c]) for r, c in inter} & {left["color"], right["color"]}
            if len(visible_colors) != 1:
                raise AssertionError("pairwise occlusion relation must be visible and unique")
            top = visible_colors.pop()
            bottom = right["color"] if top == left["color"] else left["color"]
            edges[bottom].add(top)

    order = _unique_topo_order(edges)
    for rank, color in enumerate(order, start=1):
        by_color[color]["rank"] = int(rank)
    return [by_color[color] for color in order]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct overlapping hollow frames, rank them by occlusion depth, then eject each inner orange pixel by that rank along the frame tail.

    Input structure:
    - Black background with 3 or 4 colored hollow rectangular frames, each using a unique border color.
    - Frames can overlap at border-crossing pixels, and only the top frame color remains visible at each overlap.
    - Each frame has one one-pixel tail spike sticking outward from one side and one orange pixel strictly inside it.

    Rule:
    - For each colored frame, recover its full rectangle bbox from the visible perimeter pixels and detect the tail direction from the outward spike.
    - Compare every pair of reconstructed perimeters: if two perimeters geometrically overlap, the visible color at those shared cells tells which frame is above the other.
    - Topologically sort all pairwise occlusion relations from bottom rank 1 to top rank N.
    - Erase every original frame, tail, and inner orange pixel; then place a single orange pixel at the tail position shifted outward by exactly that frame's rank.

    Key insight:
    - Every frame always contains exactly one orange pixel, so the ejection distance cannot come from content count; it is encoded only by the hidden occlusion ordering reconstructed from overlapping borders.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    for frame in _parse_scene(g):
        dr, dc = DIRS[frame["direction"]]
        rr = frame["tail"][0] + dr * frame["rank"]
        cc = frame["tail"][1] + dc * frame["rank"]
        if in_bounds(out.shape, rr, cc):
            out[rr, cc] = CONTENT
    return out


def _relative_overlap_bbox(
    prev: tuple[int, int, int, int],
    h: int,
    w: int,
    quadrant: str,
    ov_r: int,
    ov_c: int,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = prev
    if quadrant == "UR":
        nr0 = r0 - (h - ov_r)
        nc0 = c1 - ov_c + 1
    elif quadrant == "UL":
        nr0 = r0 - (h - ov_r)
        nc0 = c0 - (w - ov_c)
    elif quadrant == "DR":
        nr0 = r1 - ov_r + 1
        nc0 = c1 - ov_c + 1
    elif quadrant == "DL":
        nr0 = r1 - ov_r + 1
        nc0 = c0 - (w - ov_c)
    else:
        raise ValueError(quadrant)
    return (int(nr0), int(nc0), int(nr0 + h - 1), int(nc0 + w - 1))


def _monotone(seq: list[int]) -> bool:
    return all(a <= b for a, b in zip(seq, seq[1:])) or all(a >= b for a, b in zip(seq, seq[1:]))


def _path_cells(tail: tuple[int, int], direction: str, rank: int) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    return [(tail[0] + dr * step, tail[1] + dc * step) for step in range(rank + 1)]


def _side_candidates(
    bbox: tuple[int, int, int, int], direction: str
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    out = []
    if direction == "U":
        for c in range(c0 + 1, c1):
            out.append(((r0, c), (r0 - 1, c)))
    elif direction == "D":
        for c in range(c0 + 1, c1):
            out.append(((r1, c), (r1 + 1, c)))
    elif direction == "L":
        for r in range(r0 + 1, r1):
            out.append(((r, c0), (r, c0 - 1)))
    else:
        for r in range(r0 + 1, r1):
            out.append(((r, c1), (r, c1 + 1)))
    return out


def _build_chain_bboxes(rng: random.Random, count: int) -> list[tuple[int, int, int, int]] | None:
    h0 = rng.randint(6, 9)
    w0 = rng.randint(6, 9)
    bboxes = [(0, 0, h0 - 1, w0 - 1)]

    for idx in range(1, count):
        prev = bboxes[-1]
        placed = None
        for _ in range(80):
            h = rng.randint(6, 9)
            w = rng.randint(6, 9)
            max_ov_r = min(_bbox_h(prev) - 2, h - 2, 4)
            max_ov_c = min(_bbox_w(prev) - 2, w - 2, 4)
            if max_ov_r < 2 or max_ov_c < 2:
                continue
            ov_r = rng.randint(2, max_ov_r)
            ov_c = rng.randint(2, max_ov_c)
            quadrant = rng.choice(QUADRANTS)
            cand = _relative_overlap_bbox(prev, h, w, quadrant, ov_r, ov_c)

            inter = _perimeter_cells(prev) & _perimeter_cells(cand)
            if len(inter) != 2:
                continue
            bad = False
            for old in bboxes[:-1]:
                if _boxes_overlap(old, cand) or (_perimeter_cells(old) & _perimeter_cells(cand)):
                    bad = True
                    break
            if bad:
                continue
            placed = cand
            break
        if placed is None:
            return None
        bboxes.append(placed)
    return bboxes


def _shift_bboxes(
    rng: random.Random,
    bboxes: list[tuple[int, int, int, int]],
    directions: list[str],
) -> tuple[int, int, list[tuple[int, int, int, int]]] | None:
    min_r = min(b[0] for b in bboxes)
    min_c = min(b[1] for b in bboxes)
    max_r = max(b[2] for b in bboxes)
    max_c = max(b[3] for b in bboxes)
    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1

    margins = {"U": 0, "D": 0, "L": 0, "R": 0}
    for rank, direction in enumerate(directions, start=1):
        margins[direction] = max(margins[direction], rank + 1)

    h_min = span_h + margins["U"] + margins["D"]
    w_min = span_w + margins["L"] + margins["R"]
    if h_min > 30 or w_min > 30:
        return None

    h_low = max(18, h_min)
    w_low = max(18, w_min)
    h_high = max(h_low, min(30, h_min + 5))
    w_high = max(w_low, min(30, w_min + 5))

    height = rng.randint(h_low, h_high)
    width = rng.randint(w_low, w_high)
    slack_h = height - h_min
    slack_w = width - w_min
    off_r = margins["U"] + rng.randint(0, slack_h) - min_r
    off_c = margins["L"] + rng.randint(0, slack_w) - min_c

    shifted = [(r0 + off_r, c0 + off_c, r1 + off_r, c1 + off_c) for r0, c0, r1, c1 in bboxes]
    return height, width, shifted


def _assign_tails(
    rng: random.Random, frames: list[dict], height: int, width: int
) -> bool:
    intersections_by_color = {frame["color"]: set() for frame in frames}
    bbox_cells = {
        frame["color"]: {
            (r, c)
            for r in range(frame["bbox"][0], frame["bbox"][2] + 1)
            for c in range(frame["bbox"][1], frame["bbox"][3] + 1)
        }
        for frame in frames
    }
    perimeter_by_color = {frame["color"]: _perimeter_cells(frame["bbox"]) for frame in frames}

    for i, left in enumerate(frames):
        for right in frames[i + 1 :]:
            inter = perimeter_by_color[left["color"]] & perimeter_by_color[right["color"]]
            intersections_by_color[left["color"]].update(inter)
            intersections_by_color[right["color"]].update(inter)

    used_paths: list[set[tuple[int, int]]] = []
    for rank, frame in enumerate(frames, start=1):
        candidates = _side_candidates(frame["bbox"], frame["direction"])
        rng.shuffle(candidates)
        chosen = None
        for attach, tail in candidates:
            if attach in intersections_by_color[frame["color"]]:
                continue
            path = _path_cells(tail, frame["direction"], rank)
            if not all(in_bounds((height, width), r, c) for r, c in path):
                continue

            blocked = False
            for r, c in path:
                for other in frames:
                    if other["color"] == frame["color"]:
                        continue
                    if (r, c) in bbox_cells[other["color"]]:
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                continue

            path_set = set(path)
            if any(path_set & prev for prev in used_paths):
                continue
            chosen = (attach, tail, path_set)
            break

        if chosen is None:
            return False
        frame["tail_attach"], frame["tail"], frame["path"] = chosen
        used_paths.append(chosen[2])
    return True


def _assign_contents(rng: random.Random, frames: list[dict]) -> bool:
    interiors = {frame["color"]: set(_interior_cells(frame["bbox"])) for frame in frames}
    for frame in frames:
        others = [other for other in frames if other["color"] != frame["color"]]
        exclusive = []
        for cell in interiors[frame["color"]]:
            if any(_strictly_inside(other["bbox"], cell) for other in others):
                continue
            exclusive.append(cell)
        if not exclusive:
            return False
        frame["content"] = rng.choice(exclusive)
    return True


def _render_input(height: int, width: int, frames: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for frame in frames:
        for r, c in _perimeter_cells(frame["bbox"]):
            grid[r, c] = frame["color"]
        tr, tc = frame["tail"]
        grid[tr, tc] = frame["color"]
    for frame in frames:
        cr, cc = frame["content"]
        grid[cr, cc] = CONTENT
    return grid


def _render_output(height: int, width: int, frames: list[dict]) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    for rank, frame in enumerate(frames, start=1):
        dr, dc = DIRS[frame["direction"]]
        rr = frame["tail"][0] + dr * rank
        cc = frame["tail"][1] + dc * rank
        out[rr, cc] = CONTENT
    return out


def _semantic_checks(inp: np.ndarray, frames: list[dict], expected: np.ndarray) -> bool:
    parsed = _parse_scene(inp)
    if [frame["color"] for frame in parsed] != [frame["color"] for frame in frames]:
        return False
    if not np.array_equal(main(inp), expected):
        return False

    output_points = [tuple(map(int, rc)) for rc in np.argwhere(expected == CONTENT)]
    if len(output_points) != len(frames) or len(set(output_points)) != len(frames):
        return False

    bbox_areas = [_bbox_area(frame["bbox"]) for frame in frames]
    visible_counts = [int(np.count_nonzero(inp == frame["color"])) for frame in frames]
    row_major = [frame["bbox"][0] * 100 + frame["bbox"][1] for frame in frames]
    if _monotone(bbox_areas) or _monotone(visible_counts) or _monotone(row_major):
        return False

    for i, left in enumerate(frames):
        for right in frames[i + 1 :]:
            inter = _perimeter_cells(left["bbox"]) & _perimeter_cells(right["bbox"])
            if right["color"] == left["color"]:
                return False
            if abs(left["rank"] - right["rank"]) == 1:
                if len(inter) != 2:
                    return False
            elif inter:
                return False

    orange_cells = [tuple(map(int, rc)) for rc in np.argwhere(inp == CONTENT)]
    for cell in orange_cells:
        owners = [frame for frame in frames if _strictly_inside(frame["bbox"], cell)]
        if len(owners) != 1:
            return False
    return True


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    count = rng.choice([3, 4])
    directions = rng.sample(list(DIRS), count)
    colors = rng.sample(FRAME_COLORS, count)

    bboxes = _build_chain_bboxes(rng, count)
    if bboxes is None:
        return None

    shifted = _shift_bboxes(rng, bboxes, directions)
    if shifted is None:
        return None
    height, width, shifted_boxes = shifted

    frames = []
    for rank, (bbox, color, direction) in enumerate(zip(shifted_boxes, colors, directions), start=1):
        frames.append(
            {
                "rank": int(rank),
                "bbox": bbox,
                "color": int(color),
                "direction": direction,
            }
        )

    if not _assign_tails(rng, frames, height, width):
        return None
    if not _assign_contents(rng, frames):
        return None

    inp = _render_input(height, width, frames)
    expected = _render_output(height, width, frames)
    if not _semantic_checks(inp, frames, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random(random.randrange(1 << 30))
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid occlusion-ejection puzzle")
