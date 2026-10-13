from __future__ import annotations

import math
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
TEAL = 8
MASK_COLORS = [3, 4, 5, 6, 7, 9]

FRAME_SIZE = 5
INTERIOR_SIZE = FRAME_SIZE - 2
SLOT_ORDER = ["TL", "TM", "TR", "RM", "BR", "BM", "BL", "LM"]
GEN_ATTEMPTS = 400

NON_DOTTED_MASKS = [
    np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
]
CORNER_CELLS = [(0, 0), (0, 2), (2, 2), (2, 0)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        cells = []
        while stack:
            rr, cc = stack.pop()
            cells.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_frames(g: np.ndarray, bg: int) -> tuple[list[dict], list[np.ndarray]]:
    red_components = _components(g == RED)
    assert len(red_components) >= 4

    frames = []
    for coords in red_components:
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w and h >= 3

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert coords.shape[0] == int(border.sum())
        assert np.all(patch[border] == RED)

        interior = patch[1:-1, 1:-1]
        interior_colors = set(map(int, np.unique(interior))) - {int(bg)}
        mask_colors = interior_colors - {TEAL}
        assert len(mask_colors) == 1
        mask_color = next(iter(mask_colors))
        mask = interior == mask_color
        assert mask.any()

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                "color": mask_color,
                "mask": mask,
                "teal_count": int(np.count_nonzero(interior == TEAL)),
            }
        )

    return frames, red_components


def _blue_cycle_edges(g: np.ndarray, red_components: list[np.ndarray]) -> set[tuple[int, int]]:
    red_id = {}
    for i, coords in enumerate(red_components):
        for r, c in coords:
            red_id[(int(r), int(c))] = i

    edges = set()
    for coords in _components(g == BLUE):
        touched = set()
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            for nr, nc in _neighbors4(rr, cc):
                frame_id = red_id.get((nr, nc))
                if frame_id is not None:
                    touched.add(frame_id)
        assert len(touched) == 2
        a, b = sorted(touched)
        edges.add((a, b))

    assert len(edges) == len(red_components)
    for i in range(len(red_components)):
        assert sum(i in edge for edge in edges) == 2
    return edges


def _clockwise_order(frames: list[dict], edges: set[tuple[int, int]]) -> list[int]:
    mean_r = sum(frame["center"][0] for frame in frames) / len(frames)
    mean_c = sum(frame["center"][1] for frame in frames) / len(frames)
    angles = [math.atan2(frame["center"][0] - mean_r, frame["center"][1] - mean_c) for frame in frames]
    assert len({round(angle, 12) for angle in angles}) == len(angles)

    order = sorted(range(len(frames)), key=lambda i: angles[i])
    for i in range(len(order)):
        a = order[i]
        b = order[(i + 1) % len(order)]
        assert tuple(sorted((a, b))) in edges
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate framed interior mask colors clockwise along the blue ring by the number of teal counter dots.

    Input structure:
    - The grid contains several hollow red square frames on a black background.
    - Blue connector segments join the frames into a single cycle, and each frame touches exactly two blue segments.
    - Every frame interior contains background plus one single-color mask; exactly one frame also contains one or more isolated teal dots.

    Rule:
    - Sort the frames by their centers in clockwise order around the ring.
    - Count the teal dots in the unique dotted frame.
    - Keep every interior mask in place, but recolor it with the mask color from the frame that is that many steps clockwise ahead.
    - Remove all teal dots, while leaving the red frames, blue connectors, and background unchanged.

    Key insight:
    - The teal dots are not part of any mask; they act as a counter that sets how far the frame colors rotate around the connector cycle.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames, red_components = _extract_frames(g, bg)
    edges = _blue_cycle_edges(g, red_components)
    order = _clockwise_order(frames, edges)

    teal_frames = [frame for frame in frames if frame["teal_count"] > 0]
    assert len(teal_frames) == 1
    steps = teal_frames[0]["teal_count"]

    out = g.copy()
    out[out == TEAL] = bg
    for idx, frame_id in enumerate(order):
        src_id = order[(idx + steps) % len(order)]
        r0, c0, r1, c1 = frames[frame_id]["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[frames[frame_id]["mask"]] = frames[src_id]["color"]
    return out


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    if sr == er:
        step = 1 if ec >= sc else -1
        return [(sr, c) for c in range(sc, ec + step, step)]
    assert sc == ec
    step = 1 if er >= sr else -1
    return [(r, sc) for r in range(sr, er + step, step)]


def _slot_bbox(pos: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = pos
    return r0, c0, r0 + FRAME_SIZE - 1, c0 + FRAME_SIZE - 1


def _selected_slots() -> list[str]:
    extras = random.sample(["TM", "RM", "BM", "LM"], k=random.choice([0, 1, 2]))
    chosen = {"TL", "TR", "BR", "BL", *extras}
    return [slot for slot in SLOT_ORDER if slot in chosen]


def _mask_for_plain_frame() -> np.ndarray:
    return random.choice(NON_DOTTED_MASKS).copy()


def _dot_layout(frame_count: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    choices = [n for n in range(1, 5) if n % frame_count != 0]
    weights = [1 if n == 1 else 2 for n in choices]
    dot_count = random.choices(choices, weights=weights, k=1)[0]
    dot_cells = random.sample(CORNER_CELLS, dot_count)
    mask = np.zeros((INTERIOR_SIZE, INTERIOR_SIZE), dtype=bool)
    mask[1, 1] = True
    return mask, dot_cells


def _draw_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask_color: int,
    mask: np.ndarray,
    dots: list[tuple[int, int]] | None = None,
) -> None:
    r1 = r0 + FRAME_SIZE - 1
    c1 = c0 + FRAME_SIZE - 1
    grid[r0 : r1 + 1, c0] = RED
    grid[r0 : r1 + 1, c1] = RED
    grid[r0, c0 : c1 + 1] = RED
    grid[r1, c0 : c1 + 1] = RED

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.shape == (INTERIOR_SIZE, INTERIOR_SIZE)
    assert not np.any(mask & (interior != BG))
    interior[mask] = mask_color
    if dots is not None:
        for rr, cc in dots:
            assert not mask[rr, cc]
            assert interior[rr, cc] == BG
            interior[rr, cc] = TEAL


def _connector_cells(a_bbox: tuple[int, int, int, int], b_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    ar0, ac0, ar1, ac1 = a_bbox
    br0, bc0, br1, bc1 = b_bbox
    a_center_r = (ar0 + ar1) // 2
    a_center_c = (ac0 + ac1) // 2
    b_center_r = (br0 + br1) // 2
    b_center_c = (bc0 + bc1) // 2

    if a_center_r == b_center_r:
        if a_center_c < b_center_c:
            start = (a_center_r, ac1 + 1)
            end = (b_center_r, bc0 - 1)
        else:
            start = (a_center_r, ac0 - 1)
            end = (b_center_r, bc1 + 1)
        return _line_cells(start, end)

    assert a_center_c == b_center_c
    if a_center_r < b_center_r:
        start = (ar1 + 1, a_center_c)
        end = (br0 - 1, b_center_c)
    else:
        start = (ar0 - 1, a_center_c)
        end = (br1 + 1, b_center_c)
    return _line_cells(start, end)


def _build_layout() -> tuple[np.ndarray, dict[str, tuple[int, int]], list[str]]:
    top_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    gap_h1 = random.randint(2, 4)
    gap_h2 = random.randint(2, 4)
    gap_v1 = random.randint(2, 4)
    gap_v2 = random.randint(2, 4)

    top_r = top_margin
    mid_r = top_r + FRAME_SIZE + gap_v1
    bot_r = mid_r + FRAME_SIZE + gap_v2
    left_c = left_margin
    mid_c = left_c + FRAME_SIZE + gap_h1
    right_c = mid_c + FRAME_SIZE + gap_h2

    height = bot_r + FRAME_SIZE + bottom_margin
    width = right_c + FRAME_SIZE + right_margin
    assert height <= 30 and width <= 30

    positions = {
        "TL": (top_r, left_c),
        "TM": (top_r, mid_c),
        "TR": (top_r, right_c),
        "RM": (mid_r, right_c),
        "BR": (bot_r, right_c),
        "BM": (bot_r, mid_c),
        "BL": (bot_r, left_c),
        "LM": (mid_r, left_c),
    }
    return np.zeros((height, width), dtype=int), positions, _selected_slots()


def _expected_output(
    inp: np.ndarray,
    ordered_slots: list[str],
    positions: dict[str, tuple[int, int]],
    masks: dict[str, np.ndarray],
    slot_colors: dict[str, int],
    steps: int,
) -> np.ndarray:
    out = inp.copy()
    out[out == TEAL] = BG
    frame_count = len(ordered_slots)
    for idx, slot in enumerate(ordered_slots):
        src_slot = ordered_slots[(idx + steps) % frame_count]
        r0, c0 = positions[slot]
        patch = out[r0 + 1 : r0 + 1 + INTERIOR_SIZE, c0 + 1 : c0 + 1 + INTERIOR_SIZE]
        patch[masks[slot]] = slot_colors[src_slot]
    return out


def _gen_attempt() -> np.ndarray | None:
    grid, positions, ordered_slots = _build_layout()
    frame_count = len(ordered_slots)
    colors = random.sample(MASK_COLORS, frame_count)
    slot_colors = dict(zip(ordered_slots, colors))

    dotted_slot = random.choice(ordered_slots)
    dotted_mask, dots = _dot_layout(frame_count)

    masks: dict[str, np.ndarray] = {}
    for slot in ordered_slots:
        r0, c0 = positions[slot]
        if slot == dotted_slot:
            mask = dotted_mask.copy()
            masks[slot] = mask
            _draw_frame(grid, r0, c0, slot_colors[slot], mask, dots=dots)
        else:
            mask = _mask_for_plain_frame()
            masks[slot] = mask
            _draw_frame(grid, r0, c0, slot_colors[slot], mask)

    bboxes = {slot: _slot_bbox(positions[slot]) for slot in ordered_slots}
    for idx, slot in enumerate(ordered_slots):
        nxt = ordered_slots[(idx + 1) % frame_count]
        cells = _connector_cells(bboxes[slot], bboxes[nxt])
        if not cells:
            return None
        for r, c in cells:
            if grid[r, c] != BG:
                return None
        for r, c in cells:
            grid[r, c] = BLUE

    steps = len(dots)
    expected = _expected_output(grid, ordered_slots, positions, masks, slot_colors, steps)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 5")
