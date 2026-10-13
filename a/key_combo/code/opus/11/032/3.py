from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
RECT_COLORS = (BLUE, RED, GREEN, YELLOW)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rects: list[dict] = []
    for color in RECT_COLORS:
        for coords in _component_coords(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            view = g[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all((view == color) | (view == GREY)):
                raise AssertionError("expected solid rectangles optionally containing grey markers")
            markers = [(r0 + rr, c0 + cc) for rr, cc in np.argwhere(view == GREY)]
            rects.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "markers": markers,
                }
            )
    rects.sort(key=lambda rect: rect["bbox"])
    return rects


def _draw_split_lines(grid: np.ndarray, *, return_specs: bool = False):
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rects = _extract_rectangles(g)
    specs: list[dict] = []

    for rect in rects:
        if not rect["markers"]:
            continue
        if len(rect["markers"]) != 1:
            raise AssertionError("each marked rectangle must contain exactly one grey marker")

        r0, c0, r1, c1 = rect["bbox"]
        mr, mc = rect["markers"][0]
        h = r1 - r0 + 1
        w = c1 - c0 + 1

        if w >= h:
            out[r0 : r1 + 1, mc] = GREY
            specs.append(
                {
                    "orientation": "v",
                    "bbox": rect["bbox"],
                    "marker": (mr, mc),
                    "fixed": mc,
                    "neg_len": 0,
                    "pos_len": 0,
                }
            )
        else:
            out[mr, c0 : c1 + 1] = GREY
            specs.append(
                {
                    "orientation": "h",
                    "bbox": rect["bbox"],
                    "marker": (mr, mc),
                    "fixed": mr,
                    "neg_len": 0,
                    "pos_len": 0,
                }
            )

    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        if spec["orientation"] == "v":
            c = spec["fixed"]
            rr = r0 - 1
            while rr >= 0 and out[rr, c] == BLACK:
                out[rr, c] = GREY
                spec["neg_len"] += 1
                rr -= 1
            rr = r1 + 1
            while rr < out.shape[0] and out[rr, c] == BLACK:
                out[rr, c] = GREY
                spec["pos_len"] += 1
                rr += 1
        else:
            r = spec["fixed"]
            cc = c0 - 1
            while cc >= 0 and out[r, cc] == BLACK:
                out[r, cc] = GREY
                spec["neg_len"] += 1
                cc -= 1
            cc = c1 + 1
            while cc < out.shape[1] and out[r, cc] == BLACK:
                out[r, cc] = GREY
                spec["pos_len"] += 1
                cc += 1

    if return_specs:
        return out, rects, specs
    return out


def _flood_from_anchor(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    anchors = np.argwhere(out == ORANGE)
    if anchors.shape[0] != 1:
        raise AssertionError("expected exactly one orange anchor")
    ar, ac = map(int, anchors[0])

    q = deque([(ar, ac)])
    seen = {(ar, ac)}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in seen or not in_bounds(out.shape, nr, nc):
                continue
            seen.add((nr, nc))
            if out[nr, nc] == BLACK:
                out[nr, nc] = ORANGE
                q.append((nr, nc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend short-axis divider lines out of marked rectangles, then flood-fill the anchor-side background partition.

    Input structure:
    - A black background contains 3-4 non-overlapping solid colored rectangles, with 1-2 of them containing one grey marker pixel.
    - Rectangles use only the object colors; the grey marker lies strictly inside its rectangle, and one orange anchor pixel sits in a corner background cell.
    - Rectangles may touch a border, but the anchor is always in background and the rectangles remain separated by black gaps.

    Rule:
    - For each marked rectangle, draw a grey divider through the marker along the rectangle's short axis: vertical if width >= height, otherwise horizontal.
    - Extend each divider from the rectangle boundary in both directions through black background cells until the next non-black pixel or the grid edge.
    - After all dividers exist, treat the orange anchor as a seed and flood only 4-connected black cells reachable from it with orange.
    - Preserve all rectangles and all grey divider pixels; unreachable black regions remain black.

    Key insight:
    - The marker only decides the divider's orientation and anchor coordinate inside its own rectangle; the final flooded area is determined by how the extended divider interacts with the surrounding background channels and blockers.
    """
    g = np.asarray(grid, dtype=int)
    split = _draw_split_lines(g)
    return _flood_from_anchor(split)


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _can_place(box: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    h, w = shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    return all(not _boxes_too_close(box, other) for other in boxes)


def _paint_rect(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _sample_spaced_positions(rng: random.Random, values: list[int], count: int, gap: int) -> list[int] | None:
    vals = values[:]
    rng.shuffle(vals)
    chosen: list[int] = []
    for value in vals:
        if all(abs(value - other) >= gap for other in chosen):
            chosen.append(value)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _pick_span_covering_positions(
    rng: random.Random, axis_size: int, positions: list[int], *, margin: int = 1
) -> tuple[int, int]:
    start = max(margin, min(positions) - rng.randint(1, 2))
    end = min(axis_size - 1 - margin, max(positions) + rng.randint(1, 2))
    while end - start + 1 < 4:
        if start > margin:
            start -= 1
        elif end < axis_size - 1 - margin:
            end += 1
        else:
            break
    return int(start), int(end)


def _background_component_count(grid: np.ndarray) -> int:
    base = np.asarray(grid, dtype=int).copy()
    base[base == ORANGE] = BLACK
    return len(_component_coords(base == BLACK))


def _erase_marker(grid: np.ndarray, rect: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in rect["markers"]:
        out[r, c] = rect["color"]
    return out


def _valid_generated_instance(grid: np.ndarray) -> bool:
    try:
        rects = _extract_rectangles(grid)
    except AssertionError:
        return False

    if len(rects) not in (3, 4):
        return False
    if len(np.argwhere(grid == ORANGE)) != 1:
        return False

    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _boxes_too_close(rects[i]["bbox"], rects[j]["bbox"], pad=2):
                return False

    marked = [rect for rect in rects if rect["markers"]]
    if len(marked) not in (1, 2):
        return False

    for rect in marked:
        r0, c0, r1, c1 = rect["bbox"]
        mr, mc = rect["markers"][0]
        if mr in (r0, r1) or mc in (c0, c1):
            return False

    if _background_component_count(grid) != 1:
        return False

    split, _, specs = _draw_split_lines(grid, return_specs=True)
    if len(specs) != len(marked):
        return False
    if any(spec["neg_len"] == 0 or spec["pos_len"] == 0 for spec in specs):
        return False

    split_black = np.count_nonzero(split == BLACK)
    split_components = _background_component_count(split)
    if split_components < 2:
        return False

    out = main(grid)
    fill_added = np.count_nonzero((out == ORANGE) & (grid != ORANGE))
    if fill_added == 0:
        return False
    if np.count_nonzero(out == BLACK) == 0:
        return False

    rect_mask = np.zeros_like(grid, dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        rect_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    new_outside_grey = (split == GREY) & (grid != GREY) & ~rect_mask
    if int(new_outside_grey.sum()) < len(marked):
        return False

    baseline = _flood_from_anchor(grid)
    if np.array_equal(out, baseline):
        return False

    for rect in marked:
        if np.array_equal(main(_erase_marker(grid, rect)), out):
            return False

    if split_black == np.count_nonzero(out == BLACK):
        return False
    return True


def _render_example(shape: tuple[int, int], rects: list[dict], anchor: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect in rects:
        _paint_rect(grid, rect["bbox"], rect["color"])
        if rect["marker"] is not None:
            mr, mc = rect["marker"]
            grid[mr, mc] = GREY
    grid[anchor] = ORANGE
    return grid


def _vertical_rect_spec(rng: random.Random, shape: tuple[int, int], x: int, top_h: int, bottom_h: int) -> dict | None:
    h_grid, w_grid = shape
    for _ in range(30):
        kind = rng.choices(["wide", "square"], weights=[3, 2])[0]
        if kind == "square":
            h = w = rng.randint(4, 5)
        else:
            h = rng.randint(3, 4)
            w = rng.randint(max(4, h + 1), 6)

        local_col = rng.randint(1, w - 2)
        c0 = x - local_col
        c1 = c0 + w - 1
        if c0 < 1 or c1 > w_grid - 2:
            continue

        min_r0 = top_h + 2
        max_r0 = h_grid - bottom_h - h - 2
        if min_r0 > max_r0:
            continue
        r0 = rng.randint(min_r0, max_r0)
        r1 = r0 + h - 1
        marker = (rng.randint(r0 + 1, r1 - 1), x)
        return {"bbox": (r0, c0, r1, c1), "marker": marker}
    return None


def _horizontal_rect_spec(rng: random.Random, shape: tuple[int, int], y: int, left_w: int, right_w: int) -> dict | None:
    h_grid, w_grid = shape
    for _ in range(30):
        w = rng.randint(3, 4)
        h = rng.randint(max(4, w + 1), 6)
        local_row = rng.randint(1, h - 2)
        r0 = y - local_row
        r1 = r0 + h - 1
        if r0 < 1 or r1 > h_grid - 2:
            continue

        min_c0 = left_w + 2
        max_c0 = w_grid - right_w - w - 2
        if min_c0 > max_c0:
            continue
        c0 = rng.randint(min_c0, max_c0)
        c1 = c0 + w - 1
        marker = (y, rng.randint(c0 + 1, c1 - 1))
        return {"bbox": (r0, c0, r1, c1), "marker": marker}
    return None


def _place_distractors(
    rng: random.Random,
    shape: tuple[int, int],
    existing: list[dict],
    count: int,
    *,
    avoid_cols: list[int] | None = None,
    avoid_rows: list[int] | None = None,
) -> list[dict] | None:
    rects = existing[:]
    boxes = [rect["bbox"] for rect in rects]

    for _ in range(count):
        placed = False
        for _ in range(120):
            shape_choice = rng.choice([(3, 4), (4, 3), (4, 4), (3, 5), (5, 3), (4, 5), (5, 4)])
            h, w = shape_choice
            if h >= shape[0] - 2 or w >= shape[1] - 2:
                continue
            r0 = rng.randint(0, shape[0] - h)
            c0 = rng.randint(0, shape[1] - w)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if avoid_cols and any(box[1] <= col <= box[3] for col in avoid_cols):
                continue
            if avoid_rows and any(box[0] <= row <= box[2] for row in avoid_rows):
                continue
            if not _can_place(box, boxes, shape):
                continue
            rect = {"bbox": box, "marker": None}
            rects.append(rect)
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return rects


def _build_vertical_example(rng: random.Random) -> np.ndarray | None:
    shape = (rng.randint(15, 20), rng.randint(15, 20))
    h, w = shape
    marked_count = rng.randint(1, 2)
    xs = _sample_spaced_positions(rng, list(range(4, w - 4)), marked_count, gap=6)
    if xs is None:
        return None

    blocker_sides = rng.choices(
        [(), ("top",), ("bottom",), ("top", "bottom")],
        weights=[2, 3, 3, 4],
    )[0]
    top_h = rng.randint(3, 4) if "top" in blocker_sides else 0
    bottom_h = rng.randint(3, 4) if "bottom" in blocker_sides else 0

    rect_blueprints: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []

    if top_h:
        c0, c1 = _pick_span_covering_positions(rng, w, xs)
        top_box = (0, c0, top_h - 1, c1)
        rect_blueprints.append({"bbox": top_box, "marker": None})
        boxes.append(top_box)
    if bottom_h:
        c0, c1 = _pick_span_covering_positions(rng, w, xs)
        bottom_box = (h - bottom_h, c0, h - 1, c1)
        if not _can_place(bottom_box, boxes, shape):
            return None
        rect_blueprints.append({"bbox": bottom_box, "marker": None})
        boxes.append(bottom_box)

    for x in xs:
        spec = _vertical_rect_spec(rng, shape, x, top_h, bottom_h)
        if spec is None or not _can_place(spec["bbox"], boxes, shape):
            return None
        rect_blueprints.append(spec)
        boxes.append(spec["bbox"])

    base_count = len(rect_blueprints)
    distractor_options = [d for d in range(3) if 3 <= base_count + d <= 4]
    if not distractor_options:
        return None
    rect_blueprints = _place_distractors(
        rng,
        shape,
        rect_blueprints,
        rng.choice(distractor_options),
        avoid_cols=xs,
    )
    if rect_blueprints is None:
        return None

    colors = rng.sample(list(RECT_COLORS), len(rect_blueprints))
    rects = []
    marked_seen = 0
    for blueprint, color in zip(rect_blueprints, colors):
        marker = blueprint["marker"]
        if marker is not None:
            marked_seen += 1
        rects.append({"bbox": blueprint["bbox"], "marker": marker, "color": color})
    if marked_seen != marked_count:
        return None

    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    rng.shuffle(corners)
    for anchor in corners:
        grid = _render_example(shape, rects, anchor)
        if grid[anchor] != ORANGE:
            continue
        if _valid_generated_instance(grid):
            return grid
    return None


def _build_horizontal_example(rng: random.Random) -> np.ndarray | None:
    shape = (rng.randint(15, 20), rng.randint(15, 20))
    h, w = shape
    marked_count = rng.randint(1, 2)
    ys = _sample_spaced_positions(rng, list(range(4, h - 4)), marked_count, gap=6)
    if ys is None:
        return None

    blocker_sides = rng.choices(
        [(), ("left",), ("right",), ("left", "right")],
        weights=[2, 3, 3, 4],
    )[0]
    left_w = rng.randint(3, 4) if "left" in blocker_sides else 0
    right_w = rng.randint(3, 4) if "right" in blocker_sides else 0

    rect_blueprints: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []

    if left_w:
        r0, r1 = _pick_span_covering_positions(rng, h, ys)
        left_box = (r0, 0, r1, left_w - 1)
        rect_blueprints.append({"bbox": left_box, "marker": None})
        boxes.append(left_box)
    if right_w:
        r0, r1 = _pick_span_covering_positions(rng, h, ys)
        right_box = (r0, w - right_w, r1, w - 1)
        if not _can_place(right_box, boxes, shape):
            return None
        rect_blueprints.append({"bbox": right_box, "marker": None})
        boxes.append(right_box)

    for y in ys:
        spec = _horizontal_rect_spec(rng, shape, y, left_w, right_w)
        if spec is None or not _can_place(spec["bbox"], boxes, shape):
            return None
        rect_blueprints.append(spec)
        boxes.append(spec["bbox"])

    base_count = len(rect_blueprints)
    distractor_options = [d for d in range(3) if 3 <= base_count + d <= 4]
    if not distractor_options:
        return None
    rect_blueprints = _place_distractors(
        rng,
        shape,
        rect_blueprints,
        rng.choice(distractor_options),
        avoid_rows=ys,
    )
    if rect_blueprints is None:
        return None

    colors = rng.sample(list(RECT_COLORS), len(rect_blueprints))
    rects = []
    marked_seen = 0
    for blueprint, color in zip(rect_blueprints, colors):
        marker = blueprint["marker"]
        if marker is not None:
            marked_seen += 1
        rects.append({"bbox": blueprint["bbox"], "marker": marker, "color": color})
    if marked_seen != marked_count:
        return None

    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    rng.shuffle(corners)
    for anchor in corners:
        grid = _render_example(shape, rects, anchor)
        if _valid_generated_instance(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(600):
        builder = rng.choice([_build_vertical_example, _build_horizontal_example])
        grid = builder(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate input")
