import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

MARKER_COLORS = (BLUE, RED, GREEN, YELLOW)
DIRECTIONS = ("up", "down", "left", "right")
DIR_TO_COLOR = {"up": RED, "down": BLUE, "left": GREEN, "right": YELLOW}
COLOR_TO_DIR = {value: key for key, value in DIR_TO_COLOR.items()}
DIR_STEP = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

GRID_MIN = 16
GRID_MAX = 30
FRAME_COLOR_CHOICES = (5, 6, 7, 8, 9)
FRAME_COUNT_OPTIONS = (2, 2, 3, 3, 4)
INNER_H_RANGE = (4, 8)
INNER_W_RANGE = (4, 8)
PLACEMENT_TRIES = 100
GEN_RETRIES = 400
MAX_MARKERS_PER_SIDE = 5


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    return mask.astype(int).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    total = int(prefix[r1, c1])
    if r0 > 0:
        total -= int(prefix[r0 - 1, c1])
    if c0 > 0:
        total -= int(prefix[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(prefix[r0 - 1, c0 - 1])
    return total


def _find_frames(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    frames: list[dict] = []
    candidate_colors = [
        int(color)
        for color in np.unique(grid)
        if int(color) not in (BG,) + MARKER_COLORS
    ]

    for frame_color in candidate_colors:
        positions = [(int(r), int(c)) for r, c in np.argwhere(grid == frame_color)]
        if not positions:
            continue
        pos_set = set(positions)
        prefix = _prefix_sum(grid == frame_color)
        seen_boxes: set[tuple[int, int, int, int]] = set()

        for r0, c0 in positions:
            if r0 >= h - 2 or c0 >= w - 2:
                continue
            for r1, c1 in positions:
                if r1 <= r0 + 1 or c1 <= c0 + 1:
                    continue
                box = (r0, c0, r1, c1)
                if box in seen_boxes:
                    continue
                if (r0, c1) not in pos_set or (r1, c0) not in pos_set:
                    continue
                if _rect_sum(prefix, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
                    continue

                border = set()
                for c in range(c0, c1 + 1):
                    border.add((r0, c))
                    border.add((r1, c))
                for r in range(r0 + 1, r1):
                    border.add((r, c0))
                    border.add((r, c1))

                if any(int(grid[r, c]) not in (BG, frame_color) for r, c in border):
                    continue

                side_cells = {
                    "up": [(r0, c) for c in range(c0 + 1, c1)],
                    "down": [(r1, c) for c in range(c0 + 1, c1)],
                    "left": [(r, c0) for r in range(r0 + 1, r1)],
                    "right": [(r, c1) for r in range(r0 + 1, r1)],
                }

                gaps: dict[str, tuple[int, int]] = {}
                valid = True
                for side, cells in side_cells.items():
                    gap_cells = [(r, c) for r, c in cells if int(grid[r, c]) == BG]
                    if len(gap_cells) > 1:
                        valid = False
                        break
                    if gap_cells:
                        gaps[side] = gap_cells[0]
                if not valid or not gaps:
                    continue

                frame_pixels_on_border = sum(int(grid[r, c] == frame_color) for r, c in border)
                if frame_pixels_on_border != len(border) - len(gaps):
                    continue

                inner = grid[r0 + 1 : r1, c0 + 1 : c1]
                inner_colors = {int(color) for color in np.unique(inner) if int(color) != BG}
                if not inner_colors.issubset(set(MARKER_COLORS)):
                    continue

                marker_counts = {color: int(np.sum(inner == color)) for color in MARKER_COLORS}
                if any(marker_counts[color] > 0 and COLOR_TO_DIR[color] not in gaps for color in MARKER_COLORS):
                    continue

                frames.append(
                    {
                        "color": frame_color,
                        "bbox": box,
                        "gaps": gaps,
                        "marker_counts": marker_counts,
                    }
                )
                seen_boxes.add(box)

    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow outward rays from each rectangular frame gap using counts of matching interior markers.

    Input structure:
    - The background is black.
    - The grid contains one or more hollow rectangular frames drawn in a non-marker color; each side may contain a single black one-cell gap.
    - Inside a frame are isolated marker pixels whose colors encode directions: red for the top gap, blue for the bottom gap, green for the left gap, and yellow for the right gap.

    Rule:
    - Keep the original frames and all interior marker pixels unchanged.
    - For each frame, count how many interior markers there are of each direction color.
    - If that frame has the corresponding side gap, draw a straight ray of that marker color starting at the gap cell and extending outward for exactly that many cells, clipped by the grid boundary.

    Key insight:
    - The marker color determines which side gap is activated, while the number of markers of that color determines the ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    out = g.copy()
    frames = _find_frames(g)
    assert frames, "expected at least one rectangular frame"

    for frame in frames:
        for side, gap in frame["gaps"].items():
            color = DIR_TO_COLOR[side]
            length = frame["marker_counts"][color]
            if length <= 0:
                continue
            dr, dc = DIR_STEP[side]
            rr, cc = gap
            for _ in range(length):
                if not (0 <= rr < h and 0 <= cc < w):
                    break
                out[rr, cc] = color
                rr += dr
                cc += dc

    return out


def _choose_side_sets(rng: random.Random, n_frames: int) -> list[tuple[str, ...]]:
    side_sets = [{rng.choice(DIRECTIONS)} for _ in range(n_frames)]
    for side in DIRECTIONS:
        side_sets[rng.randrange(n_frames)].add(side)
    for side_set in side_sets:
        for side in DIRECTIONS:
            if side not in side_set and rng.random() < 0.25:
                side_set.add(side)
    ordered = []
    for side_set in side_sets:
        ordered.append(tuple(side for side in DIRECTIONS if side in side_set))
    return ordered


def _boxes_separated(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _place_boxes(
    rng: random.Random, h: int, w: int, specs: list[dict]
) -> list[tuple[int, int, int, int]] | None:
    order = sorted(
        range(len(specs)),
        key=lambda idx: specs[idx]["outer_h"] * specs[idx]["outer_w"],
        reverse=True,
    )
    boxes: list[tuple[int, int, int, int] | None] = [None] * len(specs)

    for idx in order:
        outer_h = specs[idx]["outer_h"]
        outer_w = specs[idx]["outer_w"]
        if h - outer_h < 3 or w - outer_w < 3:
            return None

        row_choices = list(range(1, h - outer_h))
        col_choices = list(range(1, w - outer_w))
        rng.shuffle(row_choices)
        rng.shuffle(col_choices)

        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = rng.choice(row_choices)
            c0 = rng.choice(col_choices)
            box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if all(
                other is None or _boxes_separated(box, other, pad=1)
                for other in boxes
            ):
                boxes[idx] = box
                placed = True
                break
        if not placed:
            return None

    return [box for box in boxes if box is not None]


def _gap_cell(
    rng: random.Random, box: tuple[int, int, int, int], side: str
) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if side == "up":
        return r0, rng.randint(c0 + 1, c1 - 1)
    if side == "down":
        return r1, rng.randint(c0 + 1, c1 - 1)
    if side == "left":
        return rng.randint(r0 + 1, r1 - 1), c0
    return rng.randint(r0 + 1, r1 - 1), c1


def _marker_count_map(
    rng: random.Random,
    sides: tuple[str, ...],
    inner_area: int,
    spaces: dict[str, int],
    trunc_side: str | None,
) -> dict[str, int] | None:
    for _ in range(80):
        counts: dict[str, int] = {}
        total = 0
        ok = True
        for side in sides:
            if trunc_side == side:
                low = spaces[side] + 1
                high = min(MAX_MARKERS_PER_SIDE, spaces[side] + 2, inner_area)
            else:
                low = 1
                high = min(MAX_MARKERS_PER_SIDE, spaces[side] + 1, inner_area)
            if low > high:
                ok = False
                break
            value = rng.randint(low, high)
            counts[side] = value
            total += value
        if ok and total <= inner_area - 2:
            return counts
    return None


def _trace_ray(
    start: tuple[int, int], side: str, length: int, h: int, w: int
) -> list[tuple[int, int]]:
    dr, dc = DIR_STEP[side]
    rr, cc = start
    cells: list[tuple[int, int]] = []
    for _ in range(length):
        if not (0 <= rr < h and 0 <= cc < w):
            break
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.choice(FRAME_COUNT_OPTIONS)
    h = rng.randint(max(GRID_MIN, 12 + 2 * n_frames), GRID_MAX)
    w = rng.randint(max(GRID_MIN, 12 + 2 * n_frames), GRID_MAX)
    frame_color = rng.choice(FRAME_COLOR_CHOICES)

    side_sets = _choose_side_sets(rng, n_frames)
    specs: list[dict] = []
    for sides in side_sets:
        inner_h = rng.randint(*INNER_H_RANGE)
        inner_w = rng.randint(*INNER_W_RANGE)
        specs.append(
            {
                "sides": sides,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
            }
        )

    boxes = _place_boxes(rng, h, w, specs)
    if boxes is None or len(boxes) != n_frames:
        return None

    frame_data: list[dict] = []
    for spec, box in zip(specs, boxes):
        gaps = {side: _gap_cell(rng, box, side) for side in spec["sides"]}
        r0, c0, r1, c1 = box
        spaces = {
            "up": r0 + 1,
            "down": h - r1,
            "left": c0 + 1,
            "right": w - c1,
        }
        frame_data.append(
            {
                "bbox": box,
                "sides": spec["sides"],
                "inner_h": spec["inner_h"],
                "inner_w": spec["inner_w"],
                "gaps": gaps,
                "spaces": spaces,
            }
        )

    trunc_candidates = [
        (idx, side)
        for idx, frame in enumerate(frame_data)
        for side in frame["sides"]
        if frame["spaces"][side] < MAX_MARKERS_PER_SIDE
    ]
    chosen_trunc = rng.choice(trunc_candidates) if trunc_candidates and rng.random() < 0.45 else None

    for idx, frame in enumerate(frame_data):
        trunc_side = None
        if chosen_trunc is not None and chosen_trunc[0] == idx:
            trunc_side = chosen_trunc[1]
        counts = _marker_count_map(
            rng,
            frame["sides"],
            frame["inner_h"] * frame["inner_w"],
            frame["spaces"],
            trunc_side,
        )
        if counts is None:
            return None
        frame["counts"] = counts

    grid = np.full((h, w), BG, dtype=int)
    for frame in frame_data:
        r0, c0, r1, c1 = frame["bbox"]
        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color
        for gap in frame["gaps"].values():
            grid[gap] = BG

    for frame in frame_data:
        r0, c0, r1, c1 = frame["bbox"]
        cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        rng.shuffle(cells)
        total_markers = sum(frame["counts"].values())
        if total_markers > len(cells) - 2:
            return None
        cursor = 0
        for side in frame["sides"]:
            color = DIR_TO_COLOR[side]
            for _ in range(frame["counts"][side]):
                grid[cells[cursor]] = color
                cursor += 1

    occupied = {(int(r), int(c)) for r, c in np.argwhere(grid != BG)}
    expected = grid.copy()
    painted_cells: set[tuple[int, int]] = set()
    for frame in frame_data:
        for side in frame["sides"]:
            color = DIR_TO_COLOR[side]
            path = _trace_ray(frame["gaps"][side], side, frame["counts"][side], h, w)
            for cell in path:
                if cell in occupied or cell in painted_cells:
                    return None
                painted_cells.add(cell)
                expected[cell] = color

    if not painted_cells:
        return None
    if set().union(*(set(frame["sides"]) for frame in frame_data)) != set(DIRECTIONS):
        return None

    if len(_find_frames(grid)) != n_frames:
        return None
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
