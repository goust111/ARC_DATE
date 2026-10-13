import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg


MARKER_COLORS = (Color.BLUE, Color.RED, Color.GREEN)
RETRY_LIMIT = 400


def _ring_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = map(int, box)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _box_center_twice(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, box)
    return r0 + r1, c0 + c1


def _marker_distance_twice(marker: dict, box: tuple[int, int, int, int]) -> int:
    cr2, cc2 = _box_center_twice(box)
    return abs(2 * int(marker["r"]) - cr2) + abs(2 * int(marker["c"]) - cc2)


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for comp in find_components(
        grid,
        bg=get_bg(grid),
        connectivity=4,
        monochromatic=True,
        colors=[Color.GREY],
    ):
        box = tuple(map(int, comp["bbox"]))
        if bbox_h(box) < 4 or bbox_w(box) < 4:
            continue
        actual = np.zeros(grid.shape, dtype=bool)
        coords = comp["coords"]
        actual[coords[:, 0], coords[:, 1]] = True
        expected = _ring_mask(grid.shape, box)
        if np.array_equal(actual, expected):
            r0, c0, r1, c1 = box
            frames.append(
                {
                    "bbox": box,
                    "interior_rows": tuple(range(r0 + 1, r1)),
                    "interior_cols": tuple(range(c0 + 1, c1)),
                    "interior_size": (r1 - r0 - 1) * (c1 - c0 - 1),
                }
            )
    assert 2 <= len(frames) <= 3, "expected 2-3 hollow grey rectangles"
    return sorted(frames, key=lambda frame: frame["bbox"])


def _edge_markers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    markers: list[dict] = []
    for c in range(1, w - 1):
        if int(g[0, c]) in MARKER_COLORS:
            markers.append({"r": 0, "c": c, "color": int(g[0, c]), "dr": 1, "dc": 0, "side": "top"})
        if int(g[h - 1, c]) in MARKER_COLORS:
            markers.append({"r": h - 1, "c": c, "color": int(g[h - 1, c]), "dr": -1, "dc": 0, "side": "bottom"})
    for r in range(1, h - 1):
        if int(g[r, 0]) in MARKER_COLORS:
            markers.append({"r": r, "c": 0, "color": int(g[r, 0]), "dr": 0, "dc": 1, "side": "left"})
        if int(g[r, w - 1]) in MARKER_COLORS:
            markers.append({"r": r, "c": w - 1, "color": int(g[r, w - 1]), "dr": 0, "dc": -1, "side": "right"})
    assert markers, "expected edge markers"
    return markers


def _ray_cells(shape: tuple[int, int], marker: dict, blockers: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    r = int(marker["r"]) + int(marker["dr"])
    c = int(marker["c"]) + int(marker["dc"])
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        if (r, c) in blockers:
            break
        cells.append((r, c))
        r += int(marker["dr"])
        c += int(marker["dc"])
    return cells


def _frame_votes(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    markers = _edge_markers(g)
    blockers = {tuple(rc) for rc in np.argwhere(g == Color.YELLOW)}
    rays = [(marker, _ray_cells(g.shape, marker, blockers)) for marker in markers]
    votes: list[dict] = []

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        covered: dict[int, set[tuple[int, int]]] = defaultdict(set)
        contributors: dict[int, list[dict]] = defaultdict(list)

        for marker, ray in rays:
            inside = [(r, c) for r, c in ray if r0 < r < r1 and c0 < c < c1]
            if inside:
                covered[int(marker["color"])].update(inside)
                contributors[int(marker["color"])].append(marker)

        counts = {color: len(cells) for color, cells in covered.items() if cells}
        assert counts, "each frame must receive at least one ray inside its interior"
        best_area = max(counts.values())
        candidate_colors = [color for color, area in counts.items() if area == best_area]
        if len(candidate_colors) == 1:
            winner = candidate_colors[0]
        else:
            distances = {
                color: min(_marker_distance_twice(marker, frame["bbox"]) for marker in contributors[color])
                for color in candidate_colors
            }
            best_dist = min(distances.values())
            dist_winners = [color for color, dist in distances.items() if dist == best_dist]
            assert len(dist_winners) == 1, "tie-break by marker distance must be unique"
            winner = dist_winners[0]

        votes.append(
            {
                "bbox": frame["bbox"],
                "winner": int(winner),
                "counts": counts,
                "contributors": contributors,
                "blockers": [(r, c) for r, c in blockers if r0 < r < r1 and c0 < c < c1],
            }
        )
    return votes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow rectangle interior with the color whose edge-emitted rays cover the most cells inside it.

    Input structure:
    - A black background grid contains 2-3 non-overlapping hollow grey rectangles with one-pixel borders.
    - Blue, red, and green single-pixel markers sit on the outer grid edges and emit straight inward rays.
    - Some rectangle interiors may contain a yellow blocker pixel that stops rays but stays visible.

    Rule:
    - Each edge marker emits a ray perpendicular to its edge, traveling along one row or one column.
    - Grey rectangle borders are transparent to rays, but yellow blocker pixels stop them before the blocker cell.
    - For every rectangle, count how many strict-interior cells are covered by rays of each color, using only the cells inside that rectangle.
    - Fill the whole strict interior with the color having the largest covered area; if areas tie, choose the color whose contributing marker is closer to the rectangle center by Manhattan distance.
    - Keep grey borders, keep the original edge markers, and keep yellow blockers yellow after the fill.

    Key insight:
    - The rectangles are not obstacles: they are sampling windows that measure per-color ray coverage inside their interiors and convert that vote into a solid fill color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for vote in _frame_votes(g):
        r0, c0, r1, c1 = vote["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(vote["winner"])
        for r, c in vote["blockers"]:
            out[r, c] = Color.YELLOW
    return out


def _random_composition(total: int, parts: int) -> list[int]:
    buckets = [0] * parts
    for _ in range(total):
        buckets[random.randrange(parts)] += 1
    return buckets


def _scenario_sequence(frame_count: int) -> list[str]:
    if frame_count == 2:
        scenarios = ["single", random.choice(["multi", "tie"])]
    else:
        scenarios = ["single", "multi", "tie"]
    random.shuffle(scenarios)
    return scenarios


def _sample_dimensions(scenarios: list[str]) -> tuple[list[int], list[int]]:
    heights: list[int] = []
    widths: list[int] = []
    for scenario in scenarios:
        if scenario == "tie":
            side = random.randint(4, 7)
            heights.append(side)
            widths.append(side)
        else:
            heights.append(random.randint(4, 7))
            widths.append(random.randint(4, 7))
    return heights, widths


def _place_boxes(scenarios: list[str]) -> tuple[tuple[int, int], list[dict]] | None:
    frame_count = len(scenarios)
    heights, widths = _sample_dimensions(scenarios)
    row_gaps = [random.randint(1, 3) for _ in range(frame_count - 1)]
    col_gaps = [random.randint(1, 3) for _ in range(frame_count - 1)]

    min_h = 2 + sum(heights) + sum(row_gaps)
    min_w = 2 + sum(widths) + sum(col_gaps)
    if min_h > 30 or min_w > 30:
        return None

    extra_h = random.randint(0, min(4, 30 - min_h))
    extra_w = random.randint(0, min(4, 30 - min_w))
    row_spaces = [1] + row_gaps + [1]
    col_spaces = [1] + col_gaps + [1]
    for idx, val in enumerate(_random_composition(extra_h, len(row_spaces))):
        row_spaces[idx] += val
    for idx, val in enumerate(_random_composition(extra_w, len(col_spaces))):
        col_spaces[idx] += val

    boxes: list[dict] = []
    r = row_spaces[0]
    c = col_spaces[0]
    for idx, scenario in enumerate(scenarios):
        h = heights[idx]
        w = widths[idx]
        box = (r, c, r + h - 1, c + w - 1)
        boxes.append({"bbox": box, "scenario": scenario})
        r = box[2] + 1 + row_spaces[idx + 1]
        c = box[3] + 1 + col_spaces[idx + 1]
    return (sum(row_spaces) + sum(heights), sum(col_spaces) + sum(widths)), boxes


def _edge_position(side: str, index: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    if side == "top":
        return 0, index
    if side == "bottom":
        return h - 1, index
    if side == "left":
        return index, 0
    if side == "right":
        return index, w - 1
    raise ValueError("invalid side")


def _make_marker(side: str, index: int, color: int, shape: tuple[int, int]) -> dict:
    r, c = _edge_position(side, index, shape)
    directions = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}
    dr, dc = directions[side]
    return {"r": r, "c": c, "color": int(color), "dr": dr, "dc": dc, "side": side}


def _choose_distinct_colors(count: int) -> list[int]:
    return random.sample(list(MARKER_COLORS), count)


def _candidate_horizontal(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[str, int]:
    r0, _c0, r1, _c1 = box
    row = random.randint(r0 + 1, r1 - 1)
    return random.choice(["left", "right"]), row


def _candidate_vertical(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[str, int]:
    _r0, c0, _r1, c1 = box
    col = random.randint(c0 + 1, c1 - 1)
    return random.choice(["top", "bottom"]), col


def _full_count(box: tuple[int, int, int, int], side: str) -> int:
    if side in {"left", "right"}:
        return bbox_w(box) - 2
    return bbox_h(box) - 2


def _count_from_blocker(box: tuple[int, int, int, int], side: str, blocker: tuple[int, int]) -> int:
    r0, c0, r1, c1 = map(int, box)
    br, bc = map(int, blocker)
    if side == "left":
        return bc - (c0 + 1)
    if side == "right":
        return (c1 - 1) - bc
    if side == "top":
        return br - (r0 + 1)
    if side == "bottom":
        return (r1 - 1) - br
    raise ValueError("invalid side")


def _build_single(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[list[dict], tuple[int, int] | None]:
    color = _choose_distinct_colors(1)[0]
    if random.random() < 0.5:
        side, row = _candidate_horizontal(box, shape)
        marker = _make_marker(side, row, color, shape)
    else:
        side, col = _candidate_vertical(box, shape)
        marker = _make_marker(side, col, color, shape)
    return [marker], None


def _build_multi(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[list[dict], tuple[int, int]]:
    h_color, v_color = _choose_distinct_colors(2)
    r0, c0, r1, c1 = map(int, box)
    for _ in range(80):
        h_side, row = _candidate_horizontal(box, shape)
        v_side, col = _candidate_vertical(box, shape)
        horizontal_full = _full_count(box, h_side)
        vertical_full = _full_count(box, v_side)

        if random.random() < 0.5:
            feasible: list[tuple[int, int]] = []
            for target_vertical in range(1, min(vertical_full - 1, horizontal_full - 1) + 1):
                blocker_row = r0 + 1 + target_vertical if v_side == "top" else r1 - 1 - target_vertical
                if blocker_row != row:
                    feasible.append((blocker_row, col))
        else:
            feasible = []
            for target_horizontal in range(1, min(horizontal_full - 1, vertical_full - 1) + 1):
                blocker_col = c0 + 1 + target_horizontal if h_side == "left" else c1 - 1 - target_horizontal
                if blocker_col != col:
                    feasible.append((row, blocker_col))

        if feasible:
            h_marker = _make_marker(h_side, row, h_color, shape)
            v_marker = _make_marker(v_side, col, v_color, shape)
            blocker = random.choice(feasible)
            assert r0 < blocker[0] < r1 and c0 < blocker[1] < c1
            return [h_marker, v_marker], blocker
    raise RuntimeError("failed to build non-degenerate multi-color frame")


def _build_tie(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[list[dict], tuple[int, int] | None]:
    h_color, v_color = _choose_distinct_colors(2)
    for _ in range(80):
        h_side, row = _candidate_horizontal(box, shape)
        v_side, col = _candidate_vertical(box, shape)
        h_marker = _make_marker(h_side, row, h_color, shape)
        v_marker = _make_marker(v_side, col, v_color, shape)
        if _marker_distance_twice(h_marker, box) != _marker_distance_twice(v_marker, box):
            return [h_marker, v_marker], None
    return None, None


def _paint_input(shape: tuple[int, int], boxes: list[dict], markers: list[dict], blockers: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, Color.BLACK, dtype=int)
    for frame in boxes:
        grid[_ring_mask(shape, frame["bbox"])] = Color.GREY
    for marker in markers:
        grid[int(marker["r"]), int(marker["c"])] = int(marker["color"])
    for r, c in blockers:
        grid[int(r), int(c)] = Color.YELLOW
    return grid


def _scenario_signature(vote: dict) -> str:
    counts = vote["counts"]
    if len(counts) == 1:
        return "single"
    best = max(counts.values())
    n_best = sum(area == best for area in counts.values())
    return "tie" if n_best > 1 else "multi"


def _validate_generated(inp: np.ndarray, scenarios: list[str]) -> bool:
    votes = _frame_votes(inp)
    if sorted(_scenario_signature(vote) for vote in votes) != sorted(scenarios):
        return False

    if not any(vote["blockers"] for vote in votes):
        return False

    out = main(inp)
    for vote in votes:
        box = vote["bbox"]
        interior_area = (bbox_h(box) - 2) * (bbox_w(box) - 2)
        if interior_area <= 0:
            return False
        interior = out[box[0] + 1 : box[2], box[1] + 1 : box[3]]
        non_blocker = interior != Color.YELLOW
        if np.any(non_blocker) and not np.all(interior[non_blocker] == vote["winner"]):
            return False

        counts = vote["counts"]
        if _scenario_signature(vote) == "single" and len(counts) != 1:
            return False
        if _scenario_signature(vote) == "multi" and len(counts) < 2:
            return False
        if _scenario_signature(vote) == "tie":
            best = max(counts.values())
            candidate_colors = [color for color, area in counts.items() if area == best]
            if len(candidate_colors) < 2:
                return False
            distances = {
                color: min(_marker_distance_twice(marker, box) for marker in vote["contributors"][color])
                for color in candidate_colors
            }
            if len({distances[color] for color in candidate_colors}) != len(candidate_colors):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choices([2, 3], weights=[2, 3], k=1)[0]
    scenarios = _scenario_sequence(frame_count)
    placed = _place_boxes(scenarios)
    if placed is None:
        return None
    shape, boxes = placed
    markers: list[dict] = []
    blockers: list[tuple[int, int]] = []

    for frame in boxes:
        box = frame["bbox"]
        scenario = frame["scenario"]
        try:
            if scenario == "single":
                new_markers, blocker = _build_single(box, shape)
            elif scenario == "multi":
                new_markers, blocker = _build_multi(box, shape)
            else:
                new_markers, blocker = _build_tie(box, shape)
                if new_markers is None:
                    return None
        except RuntimeError:
            return None
        markers.extend(new_markers)
        if blocker is not None:
            blockers.append(blocker)

    inp = _paint_input(shape, boxes, markers, blockers)
    if not _validate_generated(inp, scenarios):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-vote rectangle input")
