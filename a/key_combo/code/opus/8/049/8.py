import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

FRAME_COLORS = (BLUE, GREEN, ORANGE, MAROON)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_TO_GAP_SIDE = {
    (-1, 0): "bottom",
    (1, 0): "top",
    (0, -1): "right",
    (0, 1): "left",
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _legend_direction_map(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    h, w = grid.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    starts = set()
    for c in range(w - 2):
        starts.add((0, c))
        starts.add((h - 3, c))
    for r in range(h - 2):
        starts.add((r, 0))
        starts.add((r, w - 3))

    for r0, c0 in starts:
        patch = grid[r0 : r0 + 3, c0 : c0 + 3]
        vals, counts = np.unique(patch, return_counts=True)
        colors = {int(v): int(k) for v, k in zip(vals, counts)}
        if colors.get(GREY) != 2:
            continue
        if int(patch[1, 1]) != GREY:
            continue
        grey_pos = {(int(r), int(c)) for r, c in np.argwhere(patch == GREY)}
        if grey_pos not in ({(1, 1), (0, 1)}, {(1, 1), (2, 1)}, {(1, 1), (1, 0)}, {(1, 1), (1, 2)}):
            continue
        fill_colors = [int(v) for v in vals if int(v) != GREY]
        if len(fill_colors) != 1:
            continue
        fill = fill_colors[0]
        if colors[fill] != 7:
            continue
        other = next(iter(grey_pos - {(1, 1)}))
        direction = (other[0] - 1, other[1] - 1)
        found[(r0, c0)] = (fill, direction)

    out: dict[int, tuple[int, int]] = {}
    for _, (fill, direction) in sorted(found.items()):
        out[fill] = direction
    return out


def _find_frames(grid: np.ndarray, legend_dirs: dict[int, tuple[int, int]]) -> list[dict]:
    h, w = grid.shape
    frames: list[dict] = []
    for color in sorted(legend_dirs):
        comps = _components(grid == color)
        good = []
        for comp in comps:
            if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh < 4 or ww < 4:
                continue
            sub = np.zeros((hh, ww), dtype=bool)
            sub[comp[:, 0] - r0, comp[:, 1] - c0] = True
            border = np.zeros_like(sub, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if np.any(sub & ~border):
                continue
            missing = np.argwhere(border & ~sub)
            if len(missing) != 1:
                continue
            mr, mc = map(int, missing[0])
            if (mr in (0, hh - 1)) and (mc in (0, ww - 1)):
                continue
            if mr == 0:
                side = "top"
            elif mr == hh - 1:
                side = "bottom"
            elif mc == 0:
                side = "left"
            else:
                side = "right"
            gap = (r0 + mr, c0 + mc)
            good.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "gap": gap,
                    "gap_side": side,
                    "direction": legend_dirs[color],
                    "border_cells": {(int(r), int(c)) for r, c in comp},
                }
            )
        assert len(good) == 1, f"expected exactly one frame for color {color}"
        frame = good[0]
        frame["line_cells"] = _line_cells(frame)
        frame["interior_cells"] = _interior_cells(frame)
        frames.append(frame)
    return sorted(frames, key=lambda fr: fr["color"])


def _line_cells(frame: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    gr, gc = frame["gap"]
    dr, dc = frame["direction"]
    if (dr, dc) == (1, 0):
        return [(r, gc) for r in range(r1 - 1, r0, -1)]
    if (dr, dc) == (-1, 0):
        return [(r, gc) for r in range(r0 + 1, r1)]
    if (dr, dc) == (0, 1):
        return [(gr, c) for c in range(c1 - 1, c0, -1)]
    return [(gr, c) for c in range(c0 + 1, c1)]


def _interior_cells(frame: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _nearest_frame(frames: list[dict], pos: tuple[int, int]) -> dict:
    r, c = pos
    best = None
    best_frame = None
    for frame in frames:
        dist = min(abs(r - br) + abs(c - bc) for br, bc in frame["border_cells"])
        key = (dist, frame["color"])
        if best is None or key < best:
            best = key
            best_frame = frame
    assert best_frame is not None
    return best_frame


def _path_clear_to_gap(grid: np.ndarray, start: tuple[int, int], frame: dict) -> bool:
    r, c = start
    gr, gc = frame["gap"]
    dr, dc = frame["direction"]
    if frame["gap_side"] != DIR_TO_GAP_SIDE[(dr, dc)]:
        return False

    if (dr, dc) == (1, 0):
        if c != gc or r >= gr:
            return False
        cells = [(rr, c) for rr in range(r + 1, gr)]
    elif (dr, dc) == (-1, 0):
        if c != gc or r <= gr:
            return False
        cells = [(rr, c) for rr in range(gr + 1, r)]
    elif (dr, dc) == (0, 1):
        if r != gr or c >= gc:
            return False
        cells = [(r, cc) for cc in range(c + 1, gc)]
    else:
        if r != gr or c <= gc:
            return False
        cells = [(r, cc) for cc in range(gc + 1, c)]

    for rr, cc in cells:
        if int(grid[rr, cc]) not in (BLACK, YELLOW):
            return False
    return True


def _fill_color(count: int) -> int:
    if count == 0:
        return BLACK
    if count <= 2:
        return GREEN
    if count <= 4:
        return RED
    return MAGENTA


def _simulate(grid: np.ndarray, return_details: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    legend_dirs = _legend_direction_map(g)
    assert legend_dirs, "expected border legends"
    frames = _find_frames(g, legend_dirs)
    assert len(frames) == len(legend_dirs), "legend/frame color pairing must be one-to-one"

    out = g.copy()
    out[out == YELLOW] = BLACK
    accepted: dict[int, list[tuple[int, int]]] = {frame["color"]: [] for frame in frames}
    assignments: list[dict] = []

    for r, c in map(tuple, np.argwhere(g == YELLOW)):
        pos = (int(r), int(c))
        frame = _nearest_frame(frames, pos)
        ok = _path_clear_to_gap(g, pos, frame)
        assignments.append({"pos": pos, "frame_color": frame["color"], "enters": ok})
        if ok:
            accepted[frame["color"]].append(pos)

    counts: dict[int, int] = {}
    final_positions: dict[int, list[tuple[int, int]]] = {}
    for frame in frames:
        line = frame["line_cells"]
        n = min(len(accepted[frame["color"]]), len(line))
        final = line[:n]
        final_positions[frame["color"]] = final
        counts[frame["color"]] = n
        fill = _fill_color(n)
        final_set = set(final)
        for cell in frame["interior_cells"]:
            out[cell] = fill
        for cell in final:
            out[cell] = YELLOW
        if n == 0:
            for cell in frame["interior_cells"]:
                out[cell] = BLACK
        else:
            for cell in final_set:
                out[cell] = YELLOW

    if return_details:
        return out, {
            "legend_dirs": legend_dirs,
            "frames": frames,
            "accepted": accepted,
            "assignments": assignments,
            "counts": counts,
            "final_positions": final_positions,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each frame's gravity from a matching border legend, drop yellow pixels into the frame through its gap, and recolor the interior by deposited count.

    Input structure:
    - The grid has a black background with 2-4 hollow rectangular frames in distinct border colors, each with a single one-cell gap on one side.
    - Matching 3x3 border legends sit on the outer boundary; each legend uses the frame color as its fill and two grey cells to encode one cardinal direction.
    - Yellow pixels are scattered outside the frames.

    Rule:
    - Match each frame to the border legend with the same fill color, and read the gravity direction from the grey center cell toward the adjacent grey indicator cell.
    - Assign every yellow pixel to its nearest frame by Manhattan distance to that frame's border cells, breaking ties by smaller frame color id; only pixels aligned with that frame's gap along the frame's gravity direction can enter, and all others are deleted.
    - For each frame, entered yellow pixels stack against the opposite inner wall along the gravity line; then every remaining interior cell is filled by the deposited count: 0 stays black, 1-2 become green, 3-4 become red, and 5+ become magenta.

    Key insight:
    - The legends do not name a global gravity: each frame has its own direction, and the final interior color depends on how many assigned yellow pixels successfully pass through that specific frame's gap.
    """
    return _simulate(grid, return_details=False)


def _draw_legend(grid: np.ndarray, r0: int, c0: int, color: int, direction: tuple[int, int]) -> None:
    patch = np.full((3, 3), color, dtype=int)
    patch[1, 1] = GREY
    patch[1 + direction[0], 1 + direction[1]] = GREY
    grid[r0 : r0 + 3, c0 : c0 + 3] = patch


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    color = frame["color"]
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    gr, gc = frame["gap"]
    grid[gr, gc] = BLACK


def _legend_candidates(h: int, w: int) -> list[tuple[int, int]]:
    starts = []
    for c in range(w - 2):
        starts.append((0, c))
        starts.append((h - 3, c))
    for r in range(1, h - 3):
        starts.append((r, 0))
        starts.append((r, w - 3))
    random.shuffle(starts)
    return starts


def _boxes_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _place_legends(grid: np.ndarray, colors: list[int], directions: dict[int, tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    placed: list[tuple[int, int, int, int]] = []
    for color in colors:
        ok = False
        for r0, c0 in _legend_candidates(*grid.shape):
            box = (r0, c0, r0 + 2, c0 + 2)
            if any(_boxes_touch(box, other, pad=1) for other in placed):
                continue
            _draw_legend(grid, r0, c0, color, directions[color])
            placed.append(box)
            ok = True
            break
        if not ok:
            raise RuntimeError("failed to place legends")
    return placed


def _sample_frame_specs(k: int) -> list[dict]:
    specs = []
    for _ in range(k):
        direction = random.choice(DIRS)
        h = random.choice((5, 6, 7))
        w = random.choice((5, 6, 7))
        if direction in ((1, 0), (-1, 0)):
            h = random.choice((5, 6, 7))
        else:
            w = random.choice((5, 6, 7))
        specs.append({"direction": direction, "size": (h, w)})
    specs.sort(key=lambda item: item["size"][0] * item["size"][1], reverse=True)
    return specs


def _frame_box(frame: dict) -> tuple[int, int, int, int]:
    r0, c0 = frame["origin"]
    h, w = frame["size"]
    return r0, c0, r0 + h - 1, c0 + w - 1


def _make_frame(origin: tuple[int, int], size: tuple[int, int], color: int, direction: tuple[int, int]) -> dict:
    r0, c0 = origin
    h, w = size
    gap_side = DIR_TO_GAP_SIDE[direction]
    if gap_side == "top":
        gap = (r0, c0 + random.randint(1, w - 2))
    elif gap_side == "bottom":
        gap = (r0 + h - 1, c0 + random.randint(1, w - 2))
    elif gap_side == "left":
        gap = (r0 + random.randint(1, h - 2), c0)
    else:
        gap = (r0 + random.randint(1, h - 2), c0 + w - 1)
    frame = {
        "origin": origin,
        "size": size,
        "color": color,
        "direction": direction,
        "gap_side": gap_side,
        "gap": gap,
        "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
    }
    r0, c0, r1, c1 = frame["bbox"]
    border_cells = set()
    for c in range(c0, c1 + 1):
        border_cells.add((r0, c))
        border_cells.add((r1, c))
    for r in range(r0, r1 + 1):
        border_cells.add((r, c0))
        border_cells.add((r, c1))
    border_cells.remove(gap)
    frame["border_cells"] = border_cells
    frame["line_cells"] = _line_cells(frame)
    frame["interior_cells"] = _interior_cells(frame)
    return frame


def _place_frames(
    grid: np.ndarray,
    colors: list[int],
    directions: dict[int, tuple[int, int]],
    legend_boxes: list[tuple[int, int, int, int]],
) -> list[dict] | None:
    h, w = grid.shape
    frames: list[dict] = []
    occupied = list(legend_boxes)
    for spec, color in zip(_sample_frame_specs(len(colors)), colors):
        fh, fw = spec["size"]
        candidates = []
        for r0 in range(1, h - fh):
            for c0 in range(1, w - fw):
                box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
                if any(_boxes_touch(box, other, pad=1) for other in occupied):
                    continue
                candidates.append((r0, c0))
        random.shuffle(candidates)
        placed = None
        for origin in candidates:
            frame = _make_frame(origin, spec["size"], color, directions[color])
            placed = frame
            break
        if placed is None:
            return None
        frames.append(placed)
        occupied.append(placed["bbox"])

    for frame in frames:
        _draw_frame(grid, frame)
    return sorted(frames, key=lambda fr: fr["color"])


def _aligned_candidates(grid: np.ndarray, frame: dict, frames: list[dict]) -> list[tuple[int, int]]:
    gr, gc = frame["gap"]
    dr, dc = frame["direction"]
    cur = (gr - dr, gc - dc)
    out = []
    while _in_bounds(grid.shape, cur[0], cur[1]):
        r, c = cur
        if int(grid[r, c]) != BLACK:
            break
        if _nearest_frame(frames, cur)["color"] == frame["color"] and _path_clear_to_gap(grid, cur, frame):
            out.append(cur)
        cur = (r - dr, c - dc)
    return out


def _misaligned_candidates(grid: np.ndarray, frame: dict, frames: list[dict]) -> list[tuple[int, int]]:
    out = []
    r0, c0, r1, c1 = frame["bbox"]
    for r in range(max(0, r0 - 4), min(grid.shape[0], r1 + 5)):
        for c in range(max(0, c0 - 4), min(grid.shape[1], c1 + 5)):
            if int(grid[r, c]) != BLACK:
                continue
            pos = (r, c)
            if _nearest_frame(frames, pos)["color"] != frame["color"]:
                continue
            if _path_clear_to_gap(grid, pos, frame):
                continue
            if r0 < r < r1 and c0 < c < c1:
                continue
            out.append(pos)
    random.shuffle(out)
    return out


def _count_categories(capacity: int) -> list[int]:
    cats = [0]
    if capacity >= 1:
        cats.append(1)
    if capacity >= 3:
        cats.append(2)
    if capacity >= 5:
        cats.append(3)
    return cats


def _sample_target_counts(frames: list[dict], capacities: dict[int, int]) -> dict[int, int] | None:
    possible = {frame["color"]: _count_categories(capacities[frame["color"]]) for frame in frames}
    desired: list[int] = []
    if any(3 in cats for cats in possible.values()) and random.random() < 0.6:
        desired.append(3)
    if any(2 in cats for cats in possible.values()):
        desired.append(2)
    desired.append(1)
    desired.append(0)
    desired = desired[: len(frames)]

    random.shuffle(desired)
    assignments: dict[int, int] = {}
    used = set()
    for cat in sorted(set(desired), reverse=True):
        candidates = [f for f in frames if cat in possible[f["color"]] and f["color"] not in used]
        if not candidates:
            continue
        frame = random.choice(candidates)
        used.add(frame["color"])
        if cat == 0:
            assignments[frame["color"]] = 0
        elif cat == 1:
            assignments[frame["color"]] = random.randint(1, min(2, capacities[frame["color"]]))
        elif cat == 2:
            assignments[frame["color"]] = random.randint(3, min(4, capacities[frame["color"]]))
        else:
            assignments[frame["color"]] = random.randint(5, capacities[frame["color"]])

    for frame in frames:
        color = frame["color"]
        if color in assignments:
            continue
        cat = random.choice(possible[color])
        if cat == 0:
            assignments[color] = 0
        elif cat == 1:
            assignments[color] = random.randint(1, min(2, capacities[color]))
        elif cat == 2:
            assignments[color] = random.randint(3, min(4, capacities[color]))
        else:
            assignments[color] = random.randint(5, capacities[color])

    if GREEN in assignments and assignments[GREEN] in (1, 2):
        alt = [n for n in range(capacities[GREEN] + 1) if n not in (1, 2)]
        if not alt:
            return None
        assignments[GREEN] = random.choice(alt)

    total = sum(assignments.values())
    if total < 3 or total > 8:
        return None
    categories = {_fill_color(v) for v in assignments.values()}
    if len(categories) < 2:
        return None
    if all(v == 0 for v in assignments.values()):
        return None
    return assignments


def _semantic_check(inp: np.ndarray, out: np.ndarray, target_counts: dict[int, int], distractors: set[tuple[int, int]]) -> bool:
    try:
        expected, details = _simulate(inp, return_details=True)
    except Exception:
        return False
    if not np.array_equal(expected, out):
        return False
    if details["counts"] != target_counts:
        return False
    assignments = {item["pos"]: item for item in details["assignments"]}
    for pos in distractors:
        if pos not in assignments or assignments[pos]["enters"]:
            return False
    fill_colors = {_fill_color(v) for v in target_counts.values()}
    if len(fill_colors) < 2:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(14, 18)
    w = random.randint(14, 18)
    grid = np.full((h, w), BLACK, dtype=int)

    k = random.choices([2, 3, 4], weights=[4, 4, 2])[0]
    colors = random.sample(list(FRAME_COLORS), k)
    directions = {color: random.choice(DIRS) for color in colors}

    try:
        legend_boxes = _place_legends(grid, colors, directions)
    except RuntimeError:
        return None

    frames = _place_frames(grid, colors, directions, legend_boxes)
    if frames is None:
        return None

    capacities = {}
    aligned: dict[int, list[tuple[int, int]]] = {}
    misaligned: dict[int, list[tuple[int, int]]] = {}
    for frame in frames:
        ac = _aligned_candidates(grid, frame, frames)
        mc = _misaligned_candidates(grid, frame, frames)
        cap = min(len(frame["line_cells"]), len(ac))
        capacities[frame["color"]] = cap
        aligned[frame["color"]] = ac
        misaligned[frame["color"]] = mc

    if max(capacities.values(), default=0) == 0:
        return None

    targets = _sample_target_counts(frames, capacities)
    if targets is None:
        return None

    if not any(misaligned[frame["color"]] for frame in frames):
        return None

    used = set()
    for frame in frames:
        color = frame["color"]
        need = targets[color]
        if need > capacities[color]:
            return None
        candidates = aligned[color][:]
        if len(candidates) < need:
            return None
        choices = random.sample(candidates, need)
        for r, c in choices:
            grid[r, c] = YELLOW
        used.update(choices)

    distractor_count = random.randint(1, min(10 - sum(targets.values()), 3))
    all_misaligned = []
    for frame in frames:
        for pos in misaligned[frame["color"]]:
            if pos not in used:
                all_misaligned.append(pos)
    random.shuffle(all_misaligned)
    if len(all_misaligned) < distractor_count:
        return None
    distractors = set(all_misaligned[:distractor_count])
    for r, c in distractors:
        grid[r, c] = YELLOW

    total_yellow = int((grid == YELLOW).sum())
    if not (4 <= total_yellow <= 10):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not _semantic_check(grid, out, targets, distractors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
