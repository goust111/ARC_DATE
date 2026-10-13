from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
GRAY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_SIZE = 30
GEN_ATTEMPTS = 600


def _detect_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rectangles: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color in (BLACK, GRAY):
            continue
        coords = np.argwhere(g == color)
        if coords.shape[0] != 4:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if int(r1) - int(r0) < 2 or int(c1) - int(c0) < 2:
            continue
        corners = {
            (int(r0), int(c0)),
            (int(r0), int(c1)),
            (int(r1), int(c0)),
            (int(r1), int(c1)),
        }
        if {tuple(map(int, rc)) for rc in coords} != corners:
            continue
        interior = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
        rectangles.append(
            {
                "color": color,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "grey_count": int(np.count_nonzero(interior == GRAY)),
            }
        )
    return rectangles


def _winner_and_next(grid: np.ndarray) -> tuple[dict | None, np.ndarray]:
    rectangles = _detect_rectangles(grid)
    if not rectangles:
        return None, np.asarray(grid, dtype=int).copy()
    best = max(rect["grey_count"] for rect in rectangles)
    winners = [rect for rect in rectangles if rect["grey_count"] == best]
    assert len(winners) == 1, "grey-count winner must be unique"
    winner = winners[0]
    r0, c0, r1, c1 = winner["bbox"]
    return winner, np.asarray(grid, dtype=int)[r0 + 1 : r1, c0 + 1 : c1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeatedly elect the hidden same-color rectangle whose strict interior contains the most grey dots, then crop to that interior.

    Input structure:
    - The background is black.
    - Each non-black, non-grey color appears as exactly four isolated pixels that are the unseen corners of one axis-aligned rectangle.
    - Some such corner-defined rectangles are nested, and grey singleton dots are scattered in the canvas.

    Rule:
    - In the current canvas, detect every color whose four pixels are exactly the four corners of an axis-aligned rectangle.
    - Count grey dots strictly inside each reconstructed rectangle; the unique maximum-count rectangle wins.
    - Replace the current canvas by the winner's strict interior and repeat until no complete corner-defined rectangle remains.
    - Output that final cropped canvas, keeping grey dots grey and filling every other cell with the last winning corner color.

    Key insight:
    - The rectangles themselves are invisible; only their four corner generators exist, so the task is a recursive vote over reconstructed interiors rather than a one-shot crop.
    """
    current = np.asarray(grid, dtype=int).copy()
    last_color: int | None = None

    while True:
        winner, next_canvas = _winner_and_next(current)
        if winner is None:
            break
        last_color = int(winner["color"])
        current = next_canvas

    assert last_color is not None, "input must contain at least one hidden rectangle"
    out = np.full(current.shape, last_color, dtype=int)
    out[current == GRAY] = GRAY
    return out


def _paint_corners(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0] = color
    grid[r0, c1] = color
    grid[r1, c0] = color
    grid[r1, c1] = color


def _reserve(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0] - 1, r1 + pad)
    cc1 = min(blocked.shape[1] - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _enumerate_boxes(shape: tuple[int, int], blocked: np.ndarray) -> list[tuple[int, int, int, int]]:
    h_lim, w_lim = shape
    boxes: list[tuple[int, int, int, int]] = []
    for h in range(4, min(6, h_lim) + 1):
        for w in range(4, min(6, w_lim) + 1):
            for r0 in range(h_lim - h + 1):
                for c0 in range(w_lim - w + 1):
                    r1 = r0 + h - 1
                    c1 = c0 + w - 1
                    rr0 = max(0, r0 - 1)
                    cc0 = max(0, c0 - 1)
                    rr1 = min(h_lim - 1, r1 + 1)
                    cc1 = min(w_lim - 1, c1 + 1)
                    if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                        continue
                    boxes.append((r0, c0, r1, c1))
    random.shuffle(boxes)
    return boxes


def _interior_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _match_counts_to_boxes(
    counts: list[int],
    caps: list[int],
) -> list[int] | None:
    counts = sorted(counts, reverse=True)
    order = sorted(range(len(caps)), key=lambda idx: caps[idx], reverse=True)
    assigned = [0] * len(caps)

    def dfs(pos: int, remaining: list[int]) -> bool:
        if pos == len(order):
            return not remaining
        idx = order[pos]
        for count in list(remaining):
            if count > caps[idx]:
                continue
            assigned[idx] = count
            nxt = remaining.copy()
            nxt.remove(count)
            if dfs(pos + 1, nxt):
                return True
        return False

    if dfs(0, counts):
        return assigned
    return None


def _choose_counts(caps: list[int], winner_count: int, desired: int) -> list[int] | None:
    values = list(range(1, winner_count))
    if not values:
        return None
    desired = min(desired, len(caps), len(values))
    preferred: list[list[int]] = []
    if desired >= 2:
        preferred.append([winner_count - 1, max(1, winner_count - 2)])
    preferred.append(values[-desired:])
    for counts in preferred:
        if len(set(counts)) != len(counts):
            continue
        matched = _match_counts_to_boxes(counts, caps)
        if matched is not None:
            return matched

    combos = list(combinations(values, desired))
    random.shuffle(combos)
    for combo in combos:
        matched = _match_counts_to_boxes(list(combo), caps)
        if matched is not None:
            return matched
    return None


def _sample_cells(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    picks = cells.copy()
    random.shuffle(picks)
    return picks[:count]


def _make_final_canvas() -> np.ndarray:
    height = random.randint(3, 5)
    width = random.randint(3, 5)
    grid = np.full((height, width), BLACK, dtype=int)
    area = height * width
    grey_count = random.randint(3, min(area - 1, 6))
    cells = [(r, c) for r in range(height) for c in range(width)]
    for r, c in _sample_cells(cells, grey_count):
        grid[r, c] = GRAY
    return grid


def _build_layer(
    child: np.ndarray,
    winner_color: int,
    competitor_colors: list[int],
    orientation: str,
) -> np.ndarray | None:
    child = np.asarray(child, dtype=int)
    child_count = int(np.count_nonzero(child == GRAY))
    child_h, child_w = child.shape
    winner_h = child_h + 2
    winner_w = child_w + 2

    for _ in range(150):
        box_sizes = [(random.randint(4, 6), random.randint(4, 6)) for _ in competitor_colors]
        caps = [((h - 2) * (w - 2)) - 1 for h, w in box_sizes]
        if min(caps) < 1:
            continue
        counts = _choose_counts(caps, child_count, len(competitor_colors))
        if counts is None:
            continue

        if orientation == "bottom":
            top_pad = random.randint(0, 1)
            bottom_pad = random.randint(0, 1)
            gap = 1
            gaps = [random.randint(1, 2) for _ in range(max(0, len(box_sizes) - 1))]
            strip_h = max(h for h, _ in box_sizes)
            total_boxes_w = sum(w for _, w in box_sizes) + sum(gaps)
            width = max(winner_w + 2, total_boxes_w + 2)
            height = top_pad + winner_h + gap + strip_h + bottom_pad
        else:
            left_pad = random.randint(0, 1)
            right_pad = random.randint(0, 1)
            gap = 1
            gaps = [random.randint(1, 2) for _ in range(max(0, len(box_sizes) - 1))]
            strip_w = max(w for _, w in box_sizes)
            total_boxes_h = sum(h for h, _ in box_sizes) + sum(gaps)
            width = left_pad + winner_w + gap + strip_w + right_pad
            height = max(winner_h + 2, total_boxes_h + 2)
        if max(height, width) > MAX_SIZE:
            continue

        grid = np.full((height, width), BLACK, dtype=int)

        if orientation == "bottom":
            winner_r0 = top_pad
            winner_c0 = random.randint(0, width - winner_w)
            child_r0 = winner_r0 + 1
            child_c0 = winner_c0 + 1
            grid[child_r0 : child_r0 + child_h, child_c0 : child_c0 + child_w] = child
            winner_bbox = (winner_r0, winner_c0, winner_r0 + winner_h - 1, winner_c0 + winner_w - 1)

            strip_r0 = winner_r0 + winner_h + gap
            total_boxes_w = sum(w for _, w in box_sizes) + sum(gaps)
            start_c = random.randint(0, width - total_boxes_w)
            boxes = []
            cursor = start_c
            for i, (box_h, box_w) in enumerate(box_sizes):
                r0 = strip_r0 + random.randint(0, strip_h - box_h)
                c0 = cursor
                boxes.append((r0, c0, r0 + box_h - 1, c0 + box_w - 1))
                cursor += box_w + (gaps[i] if i < len(gaps) else 0)
        else:
            winner_c0 = left_pad
            winner_r0 = random.randint(0, height - winner_h)
            child_r0 = winner_r0 + 1
            child_c0 = winner_c0 + 1
            grid[child_r0 : child_r0 + child_h, child_c0 : child_c0 + child_w] = child
            winner_bbox = (winner_r0, winner_c0, winner_r0 + winner_h - 1, winner_c0 + winner_w - 1)

            strip_c0 = winner_c0 + winner_w + gap
            total_boxes_h = sum(h for h, _ in box_sizes) + sum(gaps)
            start_r = random.randint(0, height - total_boxes_h)
            boxes = []
            cursor = start_r
            for i, (box_h, box_w) in enumerate(box_sizes):
                r0 = cursor
                c0 = strip_c0 + random.randint(0, strip_w - box_w)
                boxes.append((r0, c0, r0 + box_h - 1, c0 + box_w - 1))
                cursor += box_h + (gaps[i] if i < len(gaps) else 0)

        _paint_corners(grid, winner_bbox, winner_color)
        blocked = np.zeros((height, width), dtype=bool)
        _reserve(blocked, winner_bbox, pad=0)
        for box, color, grey_count in zip(boxes, competitor_colors, counts):
            _paint_corners(grid, box, color)
            _reserve(blocked, box, pad=0)
            interior = _interior_cells(box)
            for r, c in _sample_cells(interior, grey_count):
                grid[r, c] = GRAY

        free_cells = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if grid[r, c] == BLACK and not blocked[r, c]
        ]
        extra = min(len(free_cells), random.randint(0, 2))
        for r, c in _sample_cells(free_cells, extra):
            grid[r, c] = GRAY
        if random.choice([False, True]):
            grid = np.flipud(grid)
        if random.choice([False, True]):
            grid = np.fliplr(grid)
        return grid
    return None


def _simulate(grid: np.ndarray) -> tuple[list[dict], list[np.ndarray]]:
    rounds: list[dict] = []
    canvases: list[np.ndarray] = [np.asarray(grid, dtype=int).copy()]
    current = canvases[0]
    while True:
        rectangles = _detect_rectangles(current)
        if not rectangles:
            break
        counts = [rect["grey_count"] for rect in rectangles]
        best = max(counts)
        winners = [rect for rect in rectangles if rect["grey_count"] == best]
        if len(winners) != 1:
            raise AssertionError("winner must be unique")
        winner = winners[0]
        rounds.append({"winner": winner, "rectangles": rectangles, "canvas": current.copy()})
        r0, c0, r1, c1 = winner["bbox"]
        current = current[r0 + 1 : r1, c0 + 1 : c1].copy()
        canvases.append(current)
    return rounds, canvases


def _semantic_validate(
    grid: np.ndarray,
    winner_colors: list[int],
    expected_final: np.ndarray,
) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > MAX_SIZE:
        return False

    colors = [int(c) for c in np.unique(g) if int(c) not in (BLACK, GRAY)]
    if any(int(np.count_nonzero(g == color)) != 4 for color in colors):
        return False

    try:
        rounds, canvases = _simulate(g)
    except AssertionError:
        return False

    if len(rounds) != len(winner_colors):
        return False
    if [int(item["winner"]["color"]) for item in rounds] != list(map(int, winner_colors)):
        return False
    if any(len(item["rectangles"]) < 2 for item in rounds):
        return False
    if len(set(map(int, winner_colors))) != len(winner_colors):
        return False
    if len(winner_colors) >= 2 and int(winner_colors[0]) == int(winner_colors[-1]):
        return False

    for i in range(len(rounds) - 1):
        this_count = int(rounds[i]["winner"]["grey_count"])
        next_count = int(rounds[i + 1]["winner"]["grey_count"])
        if this_count <= next_count:
            return False

    final_canvas = canvases[-1]
    if _detect_rectangles(final_canvas):
        return False
    if not np.any(final_canvas == GRAY):
        return False
    if not np.any(final_canvas == BLACK):
        return False

    produced = main(g)
    if not np.array_equal(produced, expected_final):
        return False
    if produced.shape != expected_final.shape:
        return False
    if not np.any(produced == GRAY):
        return False
    if not np.any(produced == int(winner_colors[-1])):
        return False
    return True


def _build_attempt() -> tuple[np.ndarray, list[int], np.ndarray] | None:
    depth = random.choice([2, 3])
    if depth == 2:
        competitor_plan = random.choice([(1, 1), (2, 1), (1, 2), (2, 2)])
        orientations = random.choice([["bottom", "right"], ["right", "bottom"]])
    else:
        competitor_plan = random.choice(
            [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1)]
        )
        orientations = random.choice(
            [["bottom", "right", "bottom"], ["right", "bottom", "right"]]
        )

    total_colors = depth + sum(competitor_plan)
    if total_colors > len(PALETTE):
        return None
    colors = random.sample(PALETTE, total_colors)

    winner_colors = colors[:depth]
    comp_cursor = depth

    current = _make_final_canvas()
    for layer in range(depth - 1, -1, -1):
        need = competitor_plan[layer]
        competitor_colors = colors[comp_cursor : comp_cursor + need]
        comp_cursor += need
        wrapped = _build_layer(current, winner_colors[layer], competitor_colors, orientations[layer])
        if wrapped is None:
            return None
        current = wrapped
        if max(current.shape) > MAX_SIZE:
            return None

    final_out = np.full(_simulate(current)[1][-1].shape, winner_colors[-1], dtype=int)
    final_canvas = _simulate(current)[1][-1]
    final_out[final_canvas == GRAY] = GRAY
    return current, winner_colors, final_out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        grid, winner_colors, expected_final = built
        if _semantic_validate(grid, winner_colors, expected_final):
            return grid
    raise RuntimeError("failed to generate a valid invisible-rectangle recursion sample")
