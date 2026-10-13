from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def box_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def box_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def box_area(box: tuple[int, int, int, int]) -> int:
    return box_h(box) * box_w(box)


def shift_box(box: tuple[int, int, int, int], dr: int, dc: int, steps: int = 1) -> tuple[int, int, int, int]:
    return (
        int(box[0] + dr * steps),
        int(box[1] + dc * steps),
        int(box[2] + dr * steps),
        int(box[3] + dc * steps),
    )


def box_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    out[r0 : r1 + 1, c0 : c1 + 1] = True
    return out


def box_in_bounds(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def box_overlaps_mask(box: tuple[int, int, int, int], occupied: np.ndarray) -> bool:
    r0, c0, r1, c1 = box
    return bool(np.any(occupied[r0 : r1 + 1, c0 : c1 + 1]))


def paint_box(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        cells = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comp = np.zeros_like(mask, dtype=bool)
        for r, c in cells:
            comp[r, c] = True
        comps.append(comp)
    return comps


def _largest_solid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    b = bbox(mask)
    assert b is not None
    r0b, c0b, r1b, c1b = b
    best: list[tuple[int, int, int, int]] = []
    best_area = 0
    for r0 in range(r0b, r1b + 1):
        for c0 in range(c0b, c1b + 1):
            if not mask[r0, c0]:
                continue
            for r1 in range(r0, r1b + 1):
                for c1 in range(c0, c1b + 1):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area < best_area:
                        continue
                    if not np.all(mask[r0 : r1 + 1, c0 : c1 + 1]):
                        continue
                    if area > best_area:
                        best_area = area
                        best = []
                    best.append((r0, c0, r1, c1))
    assert best_area > 0
    assert len(best) == 1, "body must be the unique largest solid rectangle"
    return best[0]


def _parse_stem(stem_mask: np.ndarray, body_box: tuple[int, int, int, int]) -> tuple[str, int]:
    rows, cols = np.where(stem_mask)
    assert rows.size > 0, "every moving object must have a stem"
    r0, c0, r1, c1 = body_box

    unique_rows = sorted(set(map(int, rows)))
    unique_cols = sorted(set(map(int, cols)))
    if len(unique_rows) == 1:
        row = unique_rows[0]
        cols_sorted = sorted(map(int, cols))
        assert cols_sorted == list(range(cols_sorted[0], cols_sorted[-1] + 1))
        assert r0 <= row <= r1
        if cols_sorted[-1] == c0 - 1:
            return "left", len(cols_sorted)
        if cols_sorted[0] == c1 + 1:
            return "right", len(cols_sorted)
        raise AssertionError("horizontal stem must attach flush to the body")

    if len(unique_cols) == 1:
        col = unique_cols[0]
        rows_sorted = sorted(map(int, rows))
        assert rows_sorted == list(range(rows_sorted[0], rows_sorted[-1] + 1))
        assert c0 <= col <= c1
        if rows_sorted[-1] == r0 - 1:
            return "up", len(rows_sorted)
        if rows_sorted[0] == r1 + 1:
            return "down", len(rows_sorted)
        raise AssertionError("vertical stem must attach flush to the body")

    raise AssertionError("stem must be a straight one-cell-wide segment")


def _parse_scene(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], list[dict[str, int | str | tuple[int, int, int, int]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) >= 3, "expected one anchor and at least two moving objects"

    solid_colors = []
    masks = {color: g == color for color in fg_colors}
    for color, mask in masks.items():
        b = bbox(mask)
        assert b is not None
        if int(mask.sum()) == box_area(b):
            solid_colors.append(color)
    assert len(solid_colors) == 1, "expected exactly one solid rectangular anchor color"

    anchor_color = solid_colors[0]
    anchor_box = bbox(masks[anchor_color])
    assert anchor_box is not None

    objects = []
    for color in fg_colors:
        if color == anchor_color:
            continue
        mask = masks[color]
        assert len(components_mask(mask)) == 1, "each moving color must form one 4-connected object"
        body_box = _largest_solid_rectangle(mask)
        body_mask = box_mask(mask.shape, body_box)
        stem_mask = mask & ~body_mask
        direction, stem_len = _parse_stem(stem_mask, body_box)
        objects.append(
            {
                "color": int(color),
                "body_box": body_box,
                "direction": direction,
                "stem_len": int(stem_len),
            }
        )

    lengths = [int(obj["stem_len"]) for obj in objects]
    assert len(lengths) >= 2
    assert len(set(lengths)) == len(lengths), "stem lengths must be pairwise distinct"
    return int(bg), anchor_box, objects


def _slide_box(
    body_box: tuple[int, int, int, int],
    direction: str,
    occupied: np.ndarray,
    shape: tuple[int, int],
) -> tuple[tuple[int, int, int, int], int]:
    dr, dc = DIRS[direction]
    cur = body_box
    steps = 0
    while True:
        nxt = shift_box(cur, dr, dc, 1)
        if not box_in_bounds(nxt, shape):
            break
        if box_overlaps_mask(nxt, occupied):
            break
        cur = nxt
        steps += 1
    return cur, steps


def _render_stem_cells(obj: dict[str, int | str | tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = obj["body_box"]  # type: ignore[index]
    direction = obj["direction"]  # type: ignore[index]
    stem_len = int(obj["stem_len"])
    if direction == "left":
        row = (r0 + r1) // 2
        return [(row, c) for c in range(c0 - stem_len, c0)]
    if direction == "right":
        row = (r0 + r1) // 2
        return [(row, c) for c in range(c1 + 1, c1 + stem_len + 1)]
    if direction == "up":
        col = (c0 + c1) // 2
        return [(r, col) for r in range(r0 - stem_len, r0)]
    col = (c0 + c1) // 2
    return [(r, col) for r in range(r1 + 1, r1 + stem_len + 1)]


def _simulate_specs(
    shape: tuple[int, int],
    anchor_box: tuple[int, int, int, int],
    anchor_color: int,
    objects: list[dict[str, int | str | tuple[int, int, int, int]]],
    order: list[int],
    *,
    require_positive_move: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    out = np.full(shape, BLACK, dtype=int)
    occupied = box_mask(shape, anchor_box)
    paint_box(out, anchor_box, anchor_color)
    landed: list[tuple[int, int, int, int]] = [(-1, -1, -1, -1)] * len(objects)

    for idx in order:
        obj = objects[idx]
        final_box, moved = _slide_box(
            obj["body_box"],  # type: ignore[arg-type]
            obj["direction"],  # type: ignore[arg-type]
            occupied,
            shape,
        )
        if require_positive_move:
            assert moved > 0, "generator must not create already-blocked bodies"
        landed[idx] = final_box
        paint_box(out, final_box, int(obj["color"]))
        occupied |= box_mask(shape, final_box)
    return out, landed


def _travel_with_anchor_only(
    shape: tuple[int, int],
    anchor_box: tuple[int, int, int, int],
    obj: dict[str, int | str | tuple[int, int, int, int]],
) -> int:
    occupied = box_mask(shape, anchor_box)
    _final, steps = _slide_box(
        obj["body_box"],  # type: ignore[arg-type]
        obj["direction"],  # type: ignore[arg-type]
        occupied,
        shape,
    )
    return int(steps)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase every guide stem, then slide the rectangular bodies in increasing stem-length order until they touch the anchor, earlier parked bodies, or the grid edge.

    Input structure:
    - A black background contains one solid grey rectangular anchor block.
    - Every other foreground color is a distinct object made from one solid rectangular
      body plus a straight one-cell-wide stem attached to the middle of one body edge.
    - Different objects have different stem lengths, and at least one pair has
      conflicting slide paths.

    Rule:
    - Remove all stems while keeping the anchor fixed.
    - Sort the movable bodies by stem length from shortest to longest.
    - In that order, slide each body in the stem's pointing direction until the next
      step would hit the anchor, a previously parked body, or the grid boundary.
    - Output only the anchor and the final parked bodies.

    Key insight:
    - The stem length does not encode travel distance; it only decides move order, and
      earlier parked rectangles become blockers that change where later rectangles stop.
    """
    g = np.asarray(grid, dtype=int)
    bg, anchor_box, objects = _parse_scene(g)

    anchor_color = int(g[anchor_box[0], anchor_box[1]])
    out = np.full(g.shape, bg, dtype=int)
    paint_box(out, anchor_box, anchor_color)
    occupied = box_mask(g.shape, anchor_box)

    for obj in sorted(objects, key=lambda item: int(item["stem_len"])):
        final_box, _steps = _slide_box(
            obj["body_box"],  # type: ignore[arg-type]
            obj["direction"],  # type: ignore[arg-type]
            occupied,
            g.shape,
        )
        paint_box(out, final_box, int(obj["color"]))
        occupied |= box_mask(g.shape, final_box)
    return out


def _sample_horizontal_body() -> tuple[int, int]:
    return random.choice([3, 5]), random.randint(3, 5)


def _sample_vertical_body() -> tuple[int, int]:
    return random.randint(3, 5), random.choice([3, 5])


def _available_stem_cap(
    shape: tuple[int, int],
    body_box: tuple[int, int, int, int],
    direction: str,
    occupied: np.ndarray,
) -> int:
    dr, dc = DIRS[direction]
    cur = body_box
    cap = 0
    while True:
        nxt = shift_box(cur, dr, dc, 1)
        if not box_in_bounds(nxt, shape):
            break
        if box_overlaps_mask(nxt, occupied):
            break
        cur = nxt
        cap += 1
    return int(cap)


def _build_core_pair() -> tuple[
    tuple[int, int],
    tuple[int, int, int, int],
    list[dict[str, int | str | tuple[int, int, int, int]]],
    list[int],
]:
    for _ in range(300):
        h = random.randint(18, 26)
        w = random.randint(18, 26)
        ah = random.randint(4, 6)
        aw = random.randint(4, 6)

        min_anchor_r = 8
        max_anchor_r = h - ah - 7
        min_anchor_c = 8
        max_anchor_c = w - aw - 7
        if min_anchor_r > max_anchor_r or min_anchor_c > max_anchor_c:
            continue

        ar0 = random.randint(min_anchor_r, max_anchor_r)
        ac0 = random.randint(min_anchor_c, max_anchor_c)
        anchor_box = (ar0, ac0, ar0 + ah - 1, ac0 + aw - 1)

        a_h, a_w = _sample_horizontal_body()
        b_h, b_w = _sample_vertical_body()

        max_offset = min(2, a_h - 1, b_h - 1)
        if max_offset < 1:
            continue
        offset = random.randint(1, max_offset)
        a_r0 = ar0 - offset
        if a_r0 < 2:
            continue

        overlap_cols = random.randint(1, min(aw, b_w - 1))
        left_overhang = b_w - overlap_cols
        if left_overhang < 1:
            continue
        b_c0 = ac0 - left_overhang
        if b_c0 < 2 or b_c0 + b_w > w - 2:
            continue

        max_a_free = min(8, ac0 - a_w - 3)
        max_b_free = min(8, ar0 - b_h - 3)
        if max_a_free < 4 or max_b_free < 4:
            continue

        a_free = random.randint(4, max_a_free)
        b_free = random.randint(4, max_b_free)
        a_c0 = ac0 - a_w - a_free
        b_r0 = ar0 - b_h - b_free
        if a_c0 < 2 or b_r0 < 2:
            continue

        a_box = (a_r0, a_c0, a_r0 + a_h - 1, a_c0 + a_w - 1)
        b_box = (b_r0, b_c0, b_r0 + b_h - 1, b_c0 + b_w - 1)
        if boxes_overlap(a_box, b_box):
            continue

        objects = [
            {"color": 1, "body_box": a_box, "direction": "right", "stem_len": 0},
            {"color": 2, "body_box": b_box, "direction": "down", "stem_len": 0},
        ]

        free_a = _travel_with_anchor_only((h, w), anchor_box, objects[0])
        free_b = _travel_with_anchor_only((h, w), anchor_box, objects[1])
        if free_a == free_b:
            continue

        out_ab, _ = _simulate_specs((h, w), anchor_box, GREY, objects, [0, 1], require_positive_move=False)
        out_ba, _ = _simulate_specs((h, w), anchor_box, GREY, objects, [1, 0], require_positive_move=False)
        if np.array_equal(out_ab, out_ba):
            continue

        preferred = [0, 1] if free_a > free_b else [1, 0]
        return (h, w), anchor_box, objects, preferred
    raise RuntimeError("failed to construct conflict pair")


def _maybe_add_side_object(
    shape: tuple[int, int],
    anchor_box: tuple[int, int, int, int],
    objects: list[dict[str, int | str | tuple[int, int, int, int]]],
    side: str,
) -> bool:
    h, w = shape
    ar0, ac0, ar1, ac1 = anchor_box
    body_h, body_w = (_sample_horizontal_body() if side == "right" else _sample_vertical_body())

    if side == "right":
        max_free = min(8, w - (ac1 + 1) - body_w - 3)
        if max_free < 4:
            return False
        top_lo = max(1, ar0 - body_h + 1)
        top_hi = min(h - body_h - 2, ar1)
        if top_lo > top_hi:
            return False
        body_r0 = random.randint(top_lo, top_hi)
        free = random.randint(4, max_free)
        body_c0 = ac1 + 1 + free
        new_obj = {"color": 3, "body_box": (body_r0, body_c0, body_r0 + body_h - 1, body_c0 + body_w - 1), "direction": "left", "stem_len": 0}
    else:
        max_free = min(8, h - (ar1 + 1) - body_h - 3)
        if max_free < 4:
            return False
        left_lo = max(1, ac0 - body_w + 1)
        left_hi = min(w - body_w - 2, ac1)
        if left_lo > left_hi:
            return False
        body_c0 = random.randint(left_lo, left_hi)
        free = random.randint(4, max_free)
        body_r0 = ar1 + 1 + free
        new_obj = {"color": 4, "body_box": (body_r0, body_c0, body_r0 + body_h - 1, body_c0 + body_w - 1), "direction": "up", "stem_len": 0}

    for obj in objects:
        if boxes_overlap(new_obj["body_box"], obj["body_box"]):  # type: ignore[arg-type]
            return False
    if boxes_overlap(new_obj["body_box"], anchor_box):  # type: ignore[arg-type]
        return False

    objects.append(new_obj)
    return True


def _assign_lengths(
    shape: tuple[int, int],
    anchor_box: tuple[int, int, int, int],
    objects: list[dict[str, int | str | tuple[int, int, int, int]]],
    preferred_pair_order: list[int],
) -> bool:
    occupied = box_mask(shape, anchor_box)
    for obj in objects:
        occupied |= box_mask(shape, obj["body_box"])  # type: ignore[arg-type]

    caps = []
    for idx, obj in enumerate(objects):
        others = occupied.copy()
        r0, c0, r1, c1 = obj["body_box"]  # type: ignore[index]
        others[r0 : r1 + 1, c0 : c1 + 1] = False
        caps.append(_available_stem_cap(shape, obj["body_box"], obj["direction"], others))  # type: ignore[arg-type]

    base_lengths = [None] * len(objects)
    base_lengths[preferred_pair_order[0]] = 2
    base_lengths[preferred_pair_order[1]] = 3
    next_len = 4
    for idx in range(len(objects)):
        if base_lengths[idx] is None:
            base_lengths[idx] = next_len
            next_len += 1

    for idx, need in enumerate(base_lengths):
        if need is None or caps[idx] < need:
            return False
        objects[idx]["stem_len"] = int(need)
    return True


def _render_input(
    shape: tuple[int, int],
    anchor_box: tuple[int, int, int, int],
    objects: list[dict[str, int | str | tuple[int, int, int, int]]],
) -> np.ndarray | None:
    grid = np.full(shape, BLACK, dtype=int)
    paint_box(grid, anchor_box, GREY)

    used = grid != BLACK
    for obj in objects:
        body = obj["body_box"]  # type: ignore[assignment]
        if box_overlaps_mask(body, used):
            return None
        paint_box(grid, body, int(obj["color"]))
        used |= box_mask(shape, body)

        for r, c in _render_stem_cells(obj):
            if not (0 <= r < shape[0] and 0 <= c < shape[1]):
                return None
            if used[r, c]:
                return None
            grid[r, c] = int(obj["color"])
            used[r, c] = True
    return grid


def _heuristic_output(
    shape: tuple[int, int],
    anchor_box: tuple[int, int, int, int],
    objects: list[dict[str, int | str | tuple[int, int, int, int]]],
) -> np.ndarray:
    distances = [(_travel_with_anchor_only(shape, anchor_box, obj), idx) for idx, obj in enumerate(objects)]
    order = [idx for _dist, idx in sorted(distances)]
    out, _ = _simulate_specs(shape, anchor_box, GREY, objects, order, require_positive_move=False)
    return out


def _rotated_grid(grid: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(grid, k=k).copy()


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        bg, anchor_box, objects = _parse_scene(grid)
    except AssertionError:
        return False

    if bg != BLACK:
        return False

    anchor_color = int(grid[anchor_box[0], anchor_box[1]])
    try:
        output = main(grid)
        expected, _ = _simulate_specs(
            grid.shape,
            anchor_box,
            anchor_color,
            objects,
            [idx for idx, _obj in sorted(enumerate(objects), key=lambda item: int(item[1]["stem_len"]))],
        )
        canonical = [0, 1] + list(range(2, len(objects)))
        pair_swap = [1, 0] + list(range(2, len(objects)))
        swapped = _simulate_specs(
            grid.shape,
            anchor_box,
            anchor_color,
            objects,
            pair_swap,
            require_positive_move=False,
        )[0]
        heuristic = _heuristic_output(grid.shape, anchor_box, objects)
    except AssertionError:
        return False

    if output.shape != grid.shape:
        return False
    if not np.array_equal(output, expected):
        return False

    if np.array_equal(output, grid):
        return False

    if np.array_equal(
        _simulate_specs(grid.shape, anchor_box, anchor_color, objects, canonical, require_positive_move=False)[0],
        swapped,
    ):
        return False

    if np.array_equal(output, heuristic):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    shape, anchor_box, objects, preferred_pair_order = _build_core_pair()

    want_right = random.random() < 0.75
    want_bottom = random.random() < 0.75
    if want_right:
        _maybe_add_side_object(shape, anchor_box, objects, "right")
    if want_bottom:
        _maybe_add_side_object(shape, anchor_box, objects, "bottom")

    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], len(objects))
    for obj, color in zip(objects, colors):
        obj["color"] = int(color)

    if not _assign_lengths(shape, anchor_box, objects, preferred_pair_order):
        return None

    grid = _render_input(shape, anchor_box, objects)
    if grid is None:
        return None

    k = random.randrange(4)
    grid = _rotated_grid(grid, k)
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained-parking example")
