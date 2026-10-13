from __future__ import annotations

import math
import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
PINK = 6
ORANGE = 7

FRAME_COLOR_POOL = (BLUE, RED, PINK, ORANGE)
GEN_RETRIES = 400


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _children_between_gapped_pairs(mask: np.ndarray) -> np.ndarray:
    children = np.zeros(mask.shape, dtype=bool)
    children[1:-1] = mask[:-2] & ~mask[1:-1] & mask[2:]
    return children


def _find_lower_start(grid: np.ndarray) -> int:
    green_rows = np.flatnonzero(np.any(grid == GREEN, axis=1))
    assert len(green_rows) > 0, "expected a green lower region"
    start = int(green_rows[0])
    expected = np.arange(start, start + len(green_rows))
    assert np.array_equal(green_rows, expected), "green rows must form one contiguous lower block"
    return start


def _line_candidates(counts: np.ndarray) -> list[int]:
    positive = counts[counts > 0]
    assert positive.size > 0
    lo = int(positive.min())
    hi = int(positive.max())
    threshold = int(math.ceil((lo + hi) / 2.0))
    return [int(i) for i, v in enumerate(counts) if int(v) >= threshold]


def _reconstruct_walls(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask.copy()

    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    hrows = _line_candidates(row_counts)
    vcols = _line_candidates(col_counts)
    assert len(hrows) >= 2 and len(vcols) >= 2

    walls = np.zeros_like(mask, dtype=bool)

    for r in hrows:
        for c0, c1 in zip(vcols, vcols[1:]):
            seg = mask[r, c0 : c1 + 1]
            need = max(2, int(math.ceil(0.6 * seg.size)))
            if int(seg.sum()) >= need:
                walls[r, c0 : c1 + 1] = True

    for c in vcols:
        for r0, r1 in zip(hrows, hrows[1:]):
            seg = mask[r0 : r1 + 1, c]
            need = max(2, int(math.ceil(0.6 * seg.size)))
            if int(seg.sum()) >= need:
                walls[r0 : r1 + 1, c] = True

    return walls


def _count_enclosed_regions(walls: np.ndarray) -> int:
    h, w = walls.shape
    seen = np.zeros_like(walls, dtype=bool)
    total = 0
    for r in range(h):
        for c in range(w):
            if walls[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            touches_edge = r == 0 or r == h - 1 or c == 0 or c == w - 1
            while q:
                cr, cc = q.popleft()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if walls[nr, nc] or seen[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    if nr == 0 or nr == h - 1 or nc == 0 or nc == w - 1:
                        touches_edge = True
                    q.append((nr, nc))
            if not touches_edge:
                total += 1
    return total


def _room_count_for_color(upper: np.ndarray, color: int) -> int:
    box = _bbox(upper == color)
    assert box is not None
    mask = _crop(upper == color, box).astype(bool)
    walls = _reconstruct_walls(mask)
    return _count_enclosed_regions(walls)


def _growth_layers(seed_mask: np.ndarray) -> list[np.ndarray]:
    layers: list[np.ndarray] = []
    active = seed_mask.copy()
    while True:
        children = _children_between_gapped_pairs(active)
        if not np.any(children):
            break
        layers.append(children)
        active = children
    return layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count enclosed frame regions, sort frame colors by that count, and use the sorted colors as the cyclic palette for midpoint growth below.

    Input structure:
    - A black upper area contains 3-4 colored rectilinear partition frames and scattered grey distractor pixels.
    - Each real frame uses one distinct color from blue/red/pink/orange and encloses 1-4 black rooms.
    - A contiguous lower block has green background, with one orange seed row at the top of that block.

    Rule:
    - Split the puzzle at the first row of the green lower block.
    - In the upper area, ignore grey, reconstruct each frame's horizontal and vertical wall lines from dense row/column evidence inside its color bbox, and count its enclosed black 4-connected regions.
    - Sort the frame colors by enclosed-region count from small to large; generation guarantees all counts are distinct.
    - Copy the input, then grow downward from the lower seed row every two rows: each active-background-active triplet creates one child at its midpoint, and the generated rows cycle through the sorted frame colors.
    - Stop when a generation would create no children or when the next generation row would leave the grid.

    Key insight:
    - The top half is not decoration: its room counts define an ordered color cycle, and only that sorted cycle resolves the colors of the lower midpoint-growth pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    lower_start = _find_lower_start(g)
    upper = g[:lower_start, :]
    frame_colors = _colors_present(upper, exclude={BLACK, GREY})
    assert 3 <= len(frame_colors) <= 4

    counts = [(color, _room_count_for_color(upper, color)) for color in frame_colors]
    room_values = [rooms for _color, rooms in counts]
    assert len(set(room_values)) == len(room_values)
    cycle = [color for color, _rooms in sorted(counts, key=lambda item: item[1])]

    out = g.copy()
    active = out[lower_start, :] == ORANGE
    row = lower_start + 2
    step = 0
    while row < g.shape[0]:
        children = _children_between_gapped_pairs(active)
        if not np.any(children):
            break
        out[row, children] = cycle[step % len(cycle)]
        active = children
        step += 1
        row += 2
    return out


def _frame_layouts(room_count: int) -> list[tuple[int, int]]:
    return {
        1: [(1, 1)],
        2: [(1, 2), (2, 1)],
        3: [(1, 3), (3, 1)],
        4: [(1, 4), (4, 1), (2, 2)],
    }[room_count]


def _make_frame(room_count: int) -> dict | None:
    for _ in range(80):
        cell_rows, cell_cols = random.choice(_frame_layouts(room_count))
        row_gaps = [random.randint(2, 3) for _ in range(cell_rows)]
        col_gaps = [random.randint(2, 3) for _ in range(cell_cols)]
        h = 1 + sum(row_gaps)
        w = 1 + sum(col_gaps)
        if h > 13 or w > 13:
            continue

        row_lines = [0]
        for gap in row_gaps:
            row_lines.append(row_lines[-1] + gap)
        col_lines = [0]
        for gap in col_gaps:
            col_lines.append(col_lines[-1] + gap)

        walls = np.zeros((h, w), dtype=bool)
        for r in row_lines:
            walls[r, :] = True
        for c in col_lines:
            walls[:, c] = True

        if _count_enclosed_regions(walls) != room_count:
            continue

        return {
            "walls": walls,
            "row_lines": row_lines,
            "col_lines": col_lines,
            "rooms": room_count,
            "shape": walls.shape,
        }
    return None


def _damage_frame(grid: np.ndarray, frame: dict, top: int, left: int) -> None:
    hrows = set(frame["row_lines"])
    vcols = set(frame["col_lines"])
    candidates: list[tuple[int, int]] = []
    walls = frame["walls"]
    for r, c in np.argwhere(walls):
        rr = top + int(r)
        cc = left + int(c)
        if (rr in hrows) and (cc in vcols):
            continue
        candidates.append((rr, cc))
    if not candidates:
        return
    max_damage = min(3, len(candidates) // 8)
    if max_damage <= 0:
        return
    for rr, cc in random.sample(candidates, random.randint(0, max_damage)):
        grid[rr, cc] = GREY


def _place_frames(frames: list[dict]) -> tuple[int, list[dict]] | None:
    order = list(range(len(frames)))
    random.shuffle(order)

    widths = [frames[i]["shape"][1] for i in order]
    gaps = [random.randint(1, 3) for _ in range(len(order) - 1)]
    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)
    needed_width = left_margin + sum(widths) + sum(gaps) + right_margin
    if needed_width > 28:
        return None

    width = random.randint(max(18, needed_width), 28)
    upper_height = max(frame["shape"][0] for frame in frames) + random.randint(0, 2)
    if upper_height > 14:
        return None

    x = left_margin
    placements: list[dict] = []
    for pos, idx in enumerate(order):
        frame = frames[idx]
        fh, fw = frame["shape"]
        y = random.randint(0, upper_height - fh)
        placements.append(
            {
                "index": idx,
                "left": x,
                "top": y,
                "height": fh,
                "width": fw,
            }
        )
        if pos < len(gaps):
            x += fw + gaps[pos]

    placements.sort(key=lambda item: item["index"])
    return width, placements


def _random_seed_mask(width: int) -> np.ndarray | None:
    run_count_max = min(7, max(4, width // 2))
    run_count = random.randint(4, run_count_max)
    run_lengths = random.choices([1, 2, 3, 4], weights=[5, 3, 1, 1], k=run_count)
    span = sum(run_lengths) + (run_count - 1)
    if span > width:
        return None
    left_margin = random.randint(0, width - span)
    mask = np.zeros(width, dtype=bool)
    c = left_margin
    for length in run_lengths:
        mask[c : c + length] = True
        c += length + 1
    return mask


def _seed_plan(width: int, frame_count: int) -> tuple[np.ndarray, int, str] | None:
    for _ in range(250):
        seed = _random_seed_mask(width)
        if seed is None:
            continue
        layers = _growth_layers(seed)
        actual_levels = len(layers)
        if actual_levels < frame_count:
            continue

        stop_mode = random.choice(["children", "boundary"])
        if stop_mode == "children":
            visible_levels = actual_levels
            if visible_levels < frame_count or visible_levels > 6:
                continue
            return seed, visible_levels, stop_mode

        if actual_levels < frame_count + 1:
            continue
        visible_levels = random.randint(frame_count, min(6, actual_levels - 1))
        return seed, visible_levels, stop_mode
    return None


def _render_upper(
    upper_height: int,
    width: int,
    frames: list[dict],
    placements: list[dict],
    frame_colors: list[int],
) -> tuple[np.ndarray, list[dict]]:
    upper = np.zeros((upper_height, width), dtype=int)
    info: list[dict] = []
    for idx, frame in enumerate(frames):
        top = placements[idx]["top"]
        left = placements[idx]["left"]
        walls = frame["walls"]
        upper[top : top + walls.shape[0], left : left + walls.shape[1]][walls] = frame_colors[idx]
        info.append(
            {
                "color": frame_colors[idx],
                "rooms": frame["rooms"],
                "top": top,
                "left": left,
                "bottom": top + walls.shape[0] - 1,
                "right": left + walls.shape[1] - 1,
                "row_lines": [top + r for r in frame["row_lines"]],
                "col_lines": [left + c for c in frame["col_lines"]],
                "walls": walls,
            }
        )
    return upper, info


def _add_grey_noise(upper: np.ndarray, frame_info: list[dict]) -> None:
    for frame in frame_info:
        _damage_frame(upper, frame, frame["top"], frame["left"])

    h, w = upper.shape
    background_slots = [(r, c) for r in range(h) for c in range(w) if upper[r, c] == BLACK]
    if background_slots:
        n_noise = random.randint(3, min(10, len(background_slots)))
        for r, c in random.sample(background_slots, n_noise):
            upper[r, c] = GREY


def _expected_output(
    inp: np.ndarray,
    lower_start: int,
    seed_mask: np.ndarray,
    cycle: list[int],
) -> np.ndarray:
    out = inp.copy()
    active = seed_mask.copy()
    row = lower_start + 2
    step = 0
    while row < out.shape[0]:
        children = _children_between_gapped_pairs(active)
        if not np.any(children):
            break
        out[row, children] = cycle[step % len(cycle)]
        active = children
        step += 1
        row += 2
    return out


def _validate_semantics(
    inp: np.ndarray,
    expected: np.ndarray,
    frame_info: list[dict],
    seed_mask: np.ndarray,
    visible_levels: int,
    stop_mode: str,
) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False
    if not np.array_equal(out, expected):
        return False

    lower_start = _find_lower_start(inp)
    upper = inp[:lower_start, :]
    try:
        recovered = {frame["color"]: _room_count_for_color(upper, frame["color"]) for frame in frame_info}
    except Exception:
        return False
    intended = {frame["color"]: frame["rooms"] for frame in frame_info}
    if recovered != intended:
        return False

    cycle = [color for color, rooms in sorted(recovered.items(), key=lambda item: item[1])]
    left_to_right = [frame["color"] for frame in sorted(frame_info, key=lambda item: item["left"])]
    if cycle == left_to_right:
        return False
    if cycle == sorted(cycle) or cycle == sorted(cycle, reverse=True):
        return False

    layers = _growth_layers(seed_mask)
    if stop_mode == "children":
        if len(layers) != visible_levels:
            return False
        if inp.shape[0] - lower_start < 2 * visible_levels + 3:
            return False
    else:
        if len(layers) <= visible_levels:
            return False
        if inp.shape[0] - lower_start != 2 * visible_levels + 1:
            return False

    generated_colors = []
    for i in range(visible_levels):
        row = lower_start + 2 * (i + 1)
        vals = sorted(set(map(int, out[row, :])) - {GREEN})
        if len(vals) != 1:
            return False
        generated_colors.append(vals[0])
    if generated_colors[: len(cycle)] != cycle:
        return False
    if len(set(generated_colors[: len(cycle)])) != len(cycle):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice([3, 4])
    room_counts = [1, 2, 3, 4]
    if frame_count == 3:
        room_counts = random.sample(room_counts, 3)
    random.shuffle(room_counts)

    frame_colors = random.sample(list(FRAME_COLOR_POOL), frame_count)
    frames = []
    for rooms in room_counts:
        frame = _make_frame(rooms)
        if frame is None:
            return None
        frames.append(frame)

    placed = _place_frames(frames)
    if placed is None:
        return None
    width, placements = placed
    upper_height = max(item["top"] + item["height"] for item in placements)
    upper_height += random.randint(0, 1)

    seed_plan = _seed_plan(width, frame_count)
    if seed_plan is None:
        return None
    seed_mask, visible_levels, stop_mode = seed_plan

    lower_height = 2 * visible_levels + 1
    if stop_mode == "children":
        lower_height += random.randint(2, 4)
    separator = 2
    height = upper_height + separator + lower_height
    if height > 30:
        return None

    upper, frame_info = _render_upper(upper_height, width, frames, placements, frame_colors)
    _add_grey_noise(upper, frame_info)

    inp = np.zeros((height, width), dtype=int)
    inp[:upper_height, :] = upper
    lower_start = upper_height + separator
    inp[lower_start:, :] = GREEN
    inp[lower_start, seed_mask] = ORANGE

    cycle = [frame["color"] for frame in sorted(frame_info, key=lambda item: item["rooms"])]
    expected = _expected_output(inp, lower_start, seed_mask, cycle)

    if not _validate_semantics(inp, expected, frame_info, seed_mask, visible_levels, stop_mode):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
