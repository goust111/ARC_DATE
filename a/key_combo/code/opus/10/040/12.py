from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GRAY = 5
ORANGE = 7
MAJORITY_COLORS = (BLUE, RED, GREEN)
RECT_SIZES = (2, 3)
GEN_RETRY_LIMIT = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_border_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _frame_corners(box: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = box
    return ((r0, c0), (r0, c1), (r1, c0), (r1, c1))


def _find_hollow_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == GRAY
    box = _bbox_from_mask(mask)
    assert box is not None, "expected a gray frame"
    r0, c0, r1, c1 = box
    assert r1 - r0 + 1 >= 6 and c1 - c0 + 1 >= 6, "frame is too small"
    expected = _frame_border_mask(grid.shape, box)
    assert np.array_equal(mask, expected), "expected exactly one 1-cell-thick hollow frame"
    return box


def _find_solid_rect(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == ORANGE
    box = _bbox_from_mask(mask)
    assert box is not None, "expected an orange rectangle"
    r0, c0, r1, c1 = box
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    assert side_h == side_w and side_h in RECT_SIZES, "expected a 2x2 or 3x3 orange square"
    assert int(mask.sum()) == side_h * side_w, "expected the orange object to be solid"
    return box


def _majority_color(grid: np.ndarray, frame_box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = frame_box
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    vals = [int(v) for v in interior.ravel() if int(v) != BLACK]
    assert vals, "expected colored pixels inside the frame"
    counts = Counter(vals)
    ranked = counts.most_common()
    assert ranked[0][0] in MAJORITY_COLORS, "majority color must be one of the interior colors"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique majority color"
    return int(ranked[0][0])


def _rect_corner_rays(rect_box: tuple[int, int, int, int]) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    r0, c0, r1, c1 = rect_box
    return (
        ((r0, c0), (-1, -1)),
        ((r0, c1), (-1, 1)),
        ((r1, c0), (1, -1)),
        ((r1, c1), (1, 1)),
    )


def _ray_first_hit(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[str, tuple[int, int] | None]:
    r, c = start
    dr, dc = direction
    rr = r + dr
    cc = c + dc
    while _in_bounds(grid.shape, rr, cc):
        value = int(grid[rr, cc])
        if value not in (BLACK, ORANGE):
            return "hit", (rr, cc)
        rr += dr
        cc += dc
    return "miss", None


def _layout_ray_statuses(
    shape: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    rect_box: tuple[int, int, int, int],
) -> list[dict[str, object]]:
    grid = np.zeros(shape, dtype=int)
    grid[_frame_border_mask(shape, frame_box)] = GRAY
    r0, c0, r1, c1 = rect_box
    grid[r0 : r1 + 1, c0 : c1 + 1] = ORANGE
    frame_corners = set(_frame_corners(frame_box))
    statuses: list[dict[str, object]] = []
    for corner, direction in _rect_corner_rays(rect_box):
        outcome, hit = _ray_first_hit(grid, corner, direction)
        if outcome == "miss":
            label = "miss"
        elif hit in frame_corners:
            label = "corner"
        else:
            label = "edge"
        statuses.append({"corner": corner, "dir": direction, "label": label, "hit": hit})
    return statuses


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the frame interior with its unique majority color and only extend diagonal rays that first reach a frame corner.

    Input structure:
    - The grid has a black background, one gray one-cell-thick hollow rectangular frame, and one orange solid 2x2 or 3x3 rectangle outside the frame.
    - The frame interior contains scattered non-black pixels chosen from blue, red, and green, with one color occurring strictly most often.
    - Some outward diagonals from orange rectangle corners may meet a gray frame corner, while others hit a non-corner frame edge or miss the frame entirely.

    Rule:
    - Detect the gray hollow frame and count the non-black pixels strictly inside it to obtain the unique majority color C.
    - Erase the interior scatter by filling the entire frame interior with C.
    - For each orange rectangle corner, trace the outward diagonal away from the rectangle center. If the first non-black non-orange pixel on that ray is a gray frame corner, keep tracing in the same direction to the grid boundary and recolor every background pixel on that continuation to C, without overwriting the gray frame or the orange rectangle.
    - Rays whose first non-black non-orange hit is a non-corner pixel, or that reach the boundary first, do nothing.

    Key insight:
    - The frame plays two roles at once: its interior votes for the fill/ray color, and its corner pixels gate which rectangle-corner diagonals are allowed to continue.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_box = _find_hollow_frame(g)
    rect_box = _find_solid_rect(g)
    majority = _majority_color(g, frame_box)
    frame_corners = set(_frame_corners(frame_box))

    r0, c0, r1, c1 = frame_box
    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = majority

    active = 0
    for corner, direction in _rect_corner_rays(rect_box):
        outcome, hit = _ray_first_hit(g, corner, direction)
        if outcome != "hit" or hit not in frame_corners:
            continue
        active += 1
        dr, dc = direction
        rr = hit[0] + dr
        cc = hit[1] + dc
        while _in_bounds(g.shape, rr, cc):
            if out[rr, cc] == BLACK:
                out[rr, cc] = majority
            rr += dr
            cc += dc

    assert active >= 1, "expected at least one active ray"
    return out


def _relation(frame_box: tuple[int, int, int, int], rect_box: tuple[int, int, int, int]) -> str:
    fr0, fc0, fr1, fc1 = frame_box
    rr0, cc0, rr1, cc1 = rect_box
    vertical = "mid"
    horizontal = "mid"
    if rr1 < fr0:
        vertical = "top"
    elif rr0 > fr1:
        vertical = "bottom"
    if cc1 < fc0:
        horizontal = "left"
    elif cc0 > fc1:
        horizontal = "right"
    if vertical == "mid":
        return horizontal
    if horizontal == "mid":
        return vertical
    return f"{vertical}-{horizontal}"


def _rect_intersects_frame(rect_box: tuple[int, int, int, int], frame_box: tuple[int, int, int, int]) -> bool:
    rr0, cc0, rr1, cc1 = rect_box
    fr0, fc0, fr1, fc1 = frame_box
    return not (rr1 < fr0 or rr0 > fr1 or cc1 < fc0 or cc0 > fc1)


def _candidate_rects(
    shape: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    side: int,
) -> list[dict[str, object]]:
    h, w = shape
    candidates: list[dict[str, object]] = []
    for rr0 in range(h - side + 1):
        for cc0 in range(w - side + 1):
            rect_box = (rr0, cc0, rr0 + side - 1, cc0 + side - 1)
            if _rect_intersects_frame(rect_box, frame_box):
                continue
            statuses = _layout_ray_statuses(shape, frame_box, rect_box)
            counts = Counter(status["label"] for status in statuses)
            if counts["corner"] == 0:
                continue
            relation = _relation(frame_box, rect_box)
            if relation == "mid":
                continue
            candidates.append(
                {
                    "rect_box": rect_box,
                    "statuses": statuses,
                    "counts": counts,
                    "relation": relation,
                }
            )
    return candidates


def _choose_candidate(
    shape: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    side: int,
) -> dict[str, object] | None:
    candidates = _candidate_rects(shape, frame_box, side)
    if not candidates:
        return None

    pattern_buckets = {
        "double": [c for c in candidates if c["counts"]["corner"] >= 2 and c["counts"]["miss"] >= 1],
        "edge_gate": [
            c
            for c in candidates
            if c["counts"]["corner"] == 1 and c["counts"]["edge"] >= 1 and c["counts"]["miss"] >= 1
        ],
        "single": [c for c in candidates if c["counts"]["corner"] == 1 and c["counts"]["miss"] >= 2],
    }
    preferred = random.choices(["edge_gate", "double", "single"], weights=[4, 3, 2], k=6)
    for name in preferred + ["edge_gate", "double", "single"]:
        bucket = pattern_buckets[name]
        if bucket:
            relation = random.choice(sorted({str(c["relation"]) for c in bucket}))
            relation_bucket = [c for c in bucket if c["relation"] == relation]
            return random.choice(relation_bucket)
    return None


def _split_counts(total: int, n_colors: int, majority_index: int) -> list[int] | None:
    others = n_colors - 1
    for _ in range(80):
        maj = random.randint(max(3, total // 2 + 1), total - others)
        remainder = total - maj
        if others == 0:
            return [maj]
        cuts = sorted(random.sample(range(1, remainder), others - 1)) if others > 1 else []
        pieces = []
        prev = 0
        for cut in cuts + [remainder]:
            pieces.append(cut - prev)
            prev = cut
        if any(piece <= 0 for piece in pieces):
            continue
        if any(piece >= maj for piece in pieces):
            continue
        counts = []
        piece_idx = 0
        for i in range(n_colors):
            if i == majority_index:
                counts.append(maj)
            else:
                counts.append(pieces[piece_idx])
                piece_idx += 1
        return counts
    return None


def _place_interior_scatter(
    grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
) -> tuple[int, dict[int, int]] | None:
    colors = list(random.sample(MAJORITY_COLORS, random.choice((2, 3))))
    majority_index = random.randrange(len(colors))
    total = random.randint(5, 10)
    counts = _split_counts(total, len(colors), majority_index)
    if counts is None:
        return None

    r0, c0, r1, c1 = frame_box
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    chosen = random.sample(cells, total)
    random.shuffle(chosen)

    color_counts: dict[int, int] = {}
    idx = 0
    for color, count in zip(colors, counts):
        color_counts[color] = count
        for _ in range(count):
            r, c = chosen[idx]
            idx += 1
            grid[r, c] = color
    majority = colors[majority_index]
    return majority, color_counts


def _render_expected(
    inp: np.ndarray,
    frame_box: tuple[int, int, int, int],
    majority: int,
    statuses: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    r0, c0, r1, c1 = frame_box
    out[r0 + 1 : r1, c0 + 1 : c1] = majority
    for status in statuses:
        if status["label"] != "corner":
            continue
        hit = status["hit"]
        dr, dc = status["dir"]
        rr = hit[0] + dr
        cc = hit[1] + dc
        while _in_bounds(out.shape, rr, cc):
            if out[rr, cc] == BLACK:
                out[rr, cc] = majority
            rr += dr
            cc += dc
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 20)
    w = random.randint(14, 20)
    frame_h = random.randint(6, 8)
    frame_w = random.randint(6, 8)
    if h - frame_h < 6 or w - frame_w < 6:
        return None

    fr0 = random.randint(3, h - frame_h - 3)
    fc0 = random.randint(3, w - frame_w - 3)
    frame_box = (fr0, fc0, fr0 + frame_h - 1, fc0 + frame_w - 1)

    rect_side = random.choice(RECT_SIZES)
    candidate = _choose_candidate((h, w), frame_box, rect_side)
    if candidate is None:
        return None

    rect_box = candidate["rect_box"]
    statuses = candidate["statuses"]
    counts = candidate["counts"]

    inp = np.zeros((h, w), dtype=int)
    inp[_frame_border_mask((h, w), frame_box)] = GRAY
    rr0, cc0, rr1, cc1 = rect_box
    inp[rr0 : rr1 + 1, cc0 : cc1 + 1] = ORANGE

    scatter = _place_interior_scatter(inp, frame_box)
    if scatter is None:
        return None
    majority, color_counts = scatter

    expected = _render_expected(inp, frame_box, majority, statuses)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    if counts["corner"] < 1 or counts["miss"] < 1:
        return None
    solved_majority = _majority_color(inp, frame_box)
    if solved_majority != majority:
        return None
    other_max = max((v for k, v in color_counts.items() if k != majority), default=-1)
    if color_counts[majority] <= other_max:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 12")
