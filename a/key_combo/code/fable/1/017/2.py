from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

PALETTE = list(range(10))
MAX_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(g[r0, c0]) == bg:
                continue
            color = int(g[r0, c0])
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            cells: list[Coord] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def shift_box(box: BBox, dr: int, dc: int, steps: int = 1) -> BBox:
    r0, c0, r1, c1 = box
    return (
        int(r0 + dr * steps),
        int(c0 + dc * steps),
        int(r1 + dr * steps),
        int(c1 + dc * steps),
    )


def box_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def box_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def boxes_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def in_bounds_box(shape: tuple[int, int], box: BBox) -> bool:
    h, w = shape
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def render_boxes(shape: tuple[int, int], bg: int, rects: list[dict], boxes: list[BBox]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    occ = np.zeros(shape, dtype=bool)
    for rect, box in zip(rects, boxes):
        r0, c0, r1, c1 = box
        assert not occ[r0 : r1 + 1, c0 : c1 + 1].any(), "final rectangles must not overlap"
        out[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])
        occ[r0 : r1 + 1, c0 : c1 + 1] = True
    return out


def parse_scene(grid: np.ndarray) -> tuple[int, int, list[dict], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = get_bg(g)
    comps = find_components(g, bg)
    rects: list[dict] = []
    used_corner_marks: set[Coord] = set()
    control_colors: set[int] = set()

    for comp in comps:
        if int(comp["area"]) <= 1:
            continue
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        h = box_h(comp["bbox"])
        w = box_w(comp["bbox"])
        assert h >= 2 and w >= 2, "relay rectangles must be at least 2x2"
        assert int(comp["area"]) == h * w - 1, "each rectangle body must be a filled rectangle missing one corner"

        region = g[r0 : r1 + 1, c0 : c1 + 1]
        body_mask = region == color
        assert int(body_mask.sum()) == int(comp["area"]), "rectangle body color must stay inside its bbox"
        missing = np.argwhere(~body_mask)
        assert missing.shape == (1, 2), "each rectangle must have exactly one non-body cell in its bbox"
        mr, mc = map(int, missing[0])
        assert mr in (0, h - 1) and mc in (0, w - 1), "the replaced cell must be a rectangle corner"

        corner = (r0 + mr, c0 + mc)
        control_colors.add(int(g[corner]))
        used_corner_marks.add(corner)
        rects.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "dir": (-1 if mr == 0 else 1, -1 if mc == 0 else 1),
            }
        )

    assert rects, "expected at least one marked rectangle"
    assert len(control_colors) == 1, "all corner marks must use one shared control color"
    control = next(iter(control_colors))
    control_pixels = {tuple(map(int, p)) for p in np.argwhere(g == control)}
    assert used_corner_marks <= control_pixels, "every missing corner must contain the control color"
    start_marks = list(control_pixels - used_corner_marks)
    assert len(start_marks) == 1, "expected one isolated start marker"
    start = start_marks[0]

    owners = {
        idx
        for idx, rect in enumerate(rects)
        for nr, nc in neighbors4(*start)
        if rect["bbox"][0] <= nr <= rect["bbox"][2] and rect["bbox"][1] <= nc <= rect["bbox"][3]
    }
    assert len(owners) == 1, "start marker must side-select exactly one rectangle"
    start_idx = next(iter(owners))
    return bg, control, rects, start_idx


def simulate_chain(rects: list[dict], shape: tuple[int, int], start_idx: int) -> tuple[list[BBox], list[dict]]:
    boxes = [tuple(map(int, rect["bbox"])) for rect in rects]
    visited: set[int] = set()
    events: list[dict] = []
    current = int(start_idx)

    while True:
        assert current not in visited, "relay must visit each rectangle at most once"
        visited.add(current)
        dr, dc = rects[current]["dir"]
        moves = 0
        while True:
            trial = shift_box(boxes[current], dr, dc, 1)
            if not in_bounds_box(shape, trial):
                events.append({"idx": current, "reason": "boundary", "blocker": None, "moves": moves})
                break

            blockers = [j for j, other in enumerate(boxes) if j != current and boxes_overlap(trial, other)]
            if blockers:
                assert len(blockers) == 1, "each stop must have a unique blocker"
                blocker = blockers[0]
                events.append({"idx": current, "reason": "collision", "blocker": blocker, "moves": moves})
                current = blocker
                break

            boxes[current] = trial
            moves += 1

        if events[-1]["reason"] == "boundary":
            break

    assert len(visited) == len(rects), "relay chain must visit every rectangle exactly once"
    return boxes, events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the start-selected marked rectangle diagonally, handing motion to each collided rectangle until the last one reaches the boundary.

    Input structure:
    - The background contains several filled monochrome rectangles, each with one corner cell replaced by a shared control color.
    - One extra isolated control-color pixel sits next to exactly one rectangle side and marks the first moving rectangle.
    - The marked corner of each rectangle indicates both the missing cell to restore and that rectangle's diagonal motion direction.

    Rule:
    - Restore every marked rectangle to a full solid rectangle conceptually, then begin with the rectangle selected by the isolated start pixel.
    - The current rectangle slides one cell at a time along its own corner-indicated diagonal until the next step would either leave the grid or overlap another rectangle.
    - If it stops because of a rectangle, that unique blocker immediately becomes the new current rectangle; if it stops because of the boundary, the relay ends.
    - Draw all rectangles at their stopping positions, fill back their missing corners with the rectangle colors, and remove the isolated start pixel.

    Key insight:
    - The corner mark gives a rectangle's personal diagonal, but the final layout depends on the dynamic relay order because each rectangle stops beside the next one before that next rectangle moves away.
    """
    g = np.asarray(grid, dtype=int)
    bg, _control, rects, start_idx = parse_scene(g)
    final_boxes, events = simulate_chain(rects, g.shape, start_idx)
    assert events[-1]["reason"] == "boundary", "the relay must end at the boundary"
    return render_boxes(g.shape, bg, rects, final_boxes)


def collision_source_box(target_box: BBox, size: tuple[int, int], direction: Coord, overlap_step: int) -> BBox:
    h, w = size
    dr, dc = direction
    tr0, tc0, tr1, tc1 = target_box
    if dr == 1:
        r0 = tr0 - overlap_step - (h - 1)
    else:
        r0 = tr1 + overlap_step
    if dc == 1:
        c0 = tc0 - overlap_step - (w - 1)
    else:
        c0 = tc1 + overlap_step
    return (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))


def choose_palette(rect_count: int) -> tuple[int, int, list[int]]:
    colors = random.sample(PALETTE, rect_count + 2)
    bg = colors[0]
    control = colors[1]
    rect_colors = colors[2:]
    return bg, control, rect_colors


def sample_sizes(rect_count: int) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for i in range(rect_count):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        if i == 0 and h == 2 and w == 2:
            if random.random() < 0.5:
                h = 3
            else:
                w = 3
        sizes.append((h, w))
    return sizes


def build_chain() -> tuple[list[dict], list[BBox], list[dict], dict]:
    rect_count = random.randint(2, 4)
    bg, control, rect_colors = choose_palette(rect_count)
    sizes = sample_sizes(rect_count)

    mode = random.choice(("horizontal", "vertical"))
    rects: list[dict] = []
    plans: list[dict] = [None] * rect_count  # type: ignore[assignment]

    if mode == "horizontal":
        chain_dc = random.choice((-1, 1))
        directions = [(random.choice((-1, 1)), chain_dc) for _ in range(rect_count)]
    else:
        chain_dr = random.choice((-1, 1))
        directions = [(chain_dr, random.choice((-1, 1))) for _ in range(rect_count)]

    boxes: list[BBox] = [None] * rect_count  # type: ignore[assignment]
    last_h, last_w = sizes[-1]
    boxes[-1] = (0, 0, last_h - 1, last_w - 1)

    for i in range(rect_count - 2, -1, -1):
        moved_steps = random.randint(1, 3)
        overlap_step = moved_steps + 1
        boxes[i] = collision_source_box(boxes[i + 1], sizes[i], directions[i], overlap_step)
        plans[i] = {"moves": moved_steps, "reason": "collision", "blocker": i + 1}

    last_moves = random.randint(1, 3)
    plans[-1] = {"moves": last_moves, "reason": "boundary", "blocker": None}

    stop_boxes = [shift_box(boxes[i], directions[i][0], directions[i][1], plans[i]["moves"]) for i in range(rect_count)]
    all_boxes = boxes + stop_boxes
    min_r = min(box[0] for box in all_boxes)
    max_r = max(box[2] for box in all_boxes)
    min_c = min(box[1] for box in all_boxes)
    max_c = max(box[3] for box in all_boxes)

    if mode == "horizontal":
        top_pad = random.randint(1, 3)
        bottom_pad = random.randint(0, 2)
        shift_r = top_pad - min_r
        if directions[-1][1] == 1:
            left_pad = random.randint(1, 3)
            shift_c = left_pad - min_c
            shifted_stop_last = shift_box(stop_boxes[-1], shift_r, shift_c, 1)
            width = shifted_stop_last[3] + 1
        else:
            right_pad = random.randint(1, 3)
            shift_c = -stop_boxes[-1][1]
            shifted_boxes = [shift_box(box, shift_r, shift_c, 1) for box in all_boxes]
            width = max(box[3] for box in shifted_boxes) + 1 + right_pad
        shifted_boxes = [shift_box(box, shift_r, shift_c, 1) for box in all_boxes]
        height = max(box[2] for box in shifted_boxes) + 1 + bottom_pad
    else:
        left_pad = random.randint(1, 3)
        right_pad = random.randint(0, 2)
        shift_c = left_pad - min_c
        if directions[-1][0] == 1:
            top_pad = random.randint(1, 3)
            shift_r = top_pad - min_r
            shifted_stop_last = shift_box(stop_boxes[-1], shift_r, shift_c, 1)
            height = shifted_stop_last[2] + 1
        else:
            bottom_pad = random.randint(1, 3)
            shift_r = -stop_boxes[-1][0]
            shifted_boxes = [shift_box(box, shift_r, shift_c, 1) for box in all_boxes]
            height = max(box[2] for box in shifted_boxes) + 1 + bottom_pad
        shifted_boxes = [shift_box(box, shift_r, shift_c, 1) for box in all_boxes]
        width = max(box[3] for box in shifted_boxes) + 1 + right_pad

    if max(height, width) > 30:
        raise AssertionError("grid too large")

    shifted_initial = [shift_box(box, shift_r, shift_c, 1) for box in boxes]
    shifted_final = [shift_box(box, shift_r, shift_c, 1) for box in stop_boxes]

    for rect_idx, (box, size, color, direction) in enumerate(zip(shifted_initial, sizes, rect_colors, directions)):
        rects.append({"bbox": box, "color": color, "dir": direction, "size": size, "idx": rect_idx})

    meta = {"bg": bg, "control": control, "shape": (height, width), "plans": plans}
    return rects, shifted_final, plans, meta


def corner_cell(box: BBox, direction: Coord) -> Coord:
    r0, c0, r1, c1 = box
    dr, dc = direction
    return (r0 if dr == -1 else r1, c0 if dc == -1 else c1)


def choose_start_marker(shape: tuple[int, int], initial_boxes: list[BBox], first_box: BBox) -> Coord | None:
    h, w = shape
    r0, c0, r1, c1 = first_box
    candidates: list[Coord] = []
    if c1 - c0 + 1 >= 3:
        for c in range(c0 + 1, c1):
            candidates.extend([(r0 - 1, c), (r1 + 1, c)])
    if r1 - r0 + 1 >= 3:
        for r in range(r0 + 1, r1):
            candidates.extend([(r, c0 - 1), (r, c1 + 1)])

    good: list[Coord] = []
    for r, c in candidates:
        if not (0 <= r < h and 0 <= c < w):
            continue
        owners = {
            idx
            for idx, box in enumerate(initial_boxes)
            for nr, nc in neighbors4(r, c)
            if box[0] <= nr <= box[2] and box[1] <= nc <= box[3]
        }
        if owners == {0}:
            good.append((r, c))
    if not good:
        return None
    return random.choice(good)


def render_input(shape: tuple[int, int], bg: int, control: int, rects: list[dict], start_marker: Coord) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        assert not occupied[r0 : r1 + 1, c0 : c1 + 1].any(), "initial rectangles must not overlap"
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        cr, cc = corner_cell(rect["bbox"], rect["dir"])
        grid[cr, cc] = int(control)
    sr, sc = start_marker
    assert grid[sr, sc] == bg, "start marker must be isolated from rectangle cells"
    grid[sr, sc] = int(control)
    return grid


def validate_construction(rects: list[dict], final_boxes: list[BBox], plans: list[dict], shape: tuple[int, int]) -> bool:
    actual_boxes, events = simulate_chain(rects, shape, 0)
    if actual_boxes != final_boxes or len(events) != len(plans):
        return False
    for idx, (event, plan) in enumerate(zip(events, plans)):
        if event["idx"] != idx:
            return False
        if event["reason"] != plan["reason"] or event["blocker"] != plan["blocker"] or event["moves"] != plan["moves"]:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            rects, final_boxes, plans, meta = build_chain()
        except AssertionError:
            continue

        start_marker = choose_start_marker(meta["shape"], [rect["bbox"] for rect in rects], rects[0]["bbox"])
        if start_marker is None:
            continue

        inp = render_input(meta["shape"], meta["bg"], meta["control"], rects, start_marker)
        expected = render_boxes(meta["shape"], meta["bg"], rects, final_boxes)

        try:
            bg, control, parsed_rects, start_idx = parse_scene(inp)
            if bg != meta["bg"] or control != meta["control"] or start_idx != 0:
                continue
            if not validate_construction(rects, final_boxes, plans, meta["shape"]):
                continue
            out = main(inp)
        except AssertionError:
            continue

        if np.array_equal(out, expected):
            return inp

    raise RuntimeError("failed to generate a valid relay-sliding input")
