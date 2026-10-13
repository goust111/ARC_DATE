from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
SPRITE_COLOR = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # right, down, left, up
SHAPES = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
)
GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: tuple[tuple[int, int], ...] | np.ndarray) -> tuple[int, int, int, int]:
    if isinstance(coords, np.ndarray):
        rows = coords[:, 0]
        cols = coords[:, 1]
        return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_h(box: tuple[int, int, int, int]) -> int:
    return box[2] - box[0] + 1


def _bbox_w(box: tuple[int, int, int, int]) -> int:
    return box[3] - box[1] + 1


def _components_of_value(grid: np.ndarray, value: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != value:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == value:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _is_hollow_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    return expected == {tuple(map(int, rc)) for rc in coords}


def _frame_contains_box(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _frame_contains_coords(frame_box: tuple[int, int, int, int], coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = frame_box
    return bool(np.all(coords[:, 0] > r0) and np.all(coords[:, 0] < r1) and np.all(coords[:, 1] > c0) and np.all(coords[:, 1] < c1))


def _distance_to_target(frame_box: tuple[int, int, int, int], coords: np.ndarray, direction: tuple[int, int]) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return frame_box[3] - 1 - c1
    if (dr, dc) == (1, 0):
        return frame_box[2] - 1 - r1
    if (dr, dc) == (0, -1):
        return c0 - (frame_box[1] + 1)
    return r0 - (frame_box[0] + 1)


def _analyze_grid(grid: np.ndarray) -> tuple[list[dict], list[dict], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    frames: list[dict] = []
    frame_id = 0
    for color in sorted(set(map(int, np.unique(g))) - {bg, SPRITE_COLOR}):
        for comp in _components_of_value(g, color):
            assert _is_hollow_rectangle(comp), "expected every non-grey foreground component to be a hollow rectangle frame"
            box = _bbox(comp)
            frames.append(
                {
                    "id": frame_id,
                    "color": int(color),
                    "coords": comp,
                    "bbox": box,
                    "area": int(comp.shape[0]),
                }
            )
            frame_id += 1

    assert frames, "expected at least one frame"

    for frame in frames:
        frame["depth"] = sum(
            1 for other in frames if other["id"] != frame["id"] and _frame_contains_box(other["bbox"], frame["bbox"])
        )

    for frame in frames:
        parents = [
            other
            for other in frames
            if other["depth"] == frame["depth"] - 1 and _frame_contains_box(other["bbox"], frame["bbox"])
        ]
        frame["parent_id"] = parents[0]["id"] if len(parents) == 1 else None

    sprites: list[dict] = []
    for sprite_id, comp in enumerate(_components_of_value(g, SPRITE_COLOR)):
        assert 3 <= int(comp.shape[0]) <= 5, "expected each grey sprite to have area 3..5"
        containers = [frame for frame in frames if _frame_contains_coords(frame["bbox"], comp)]
        assert containers, "every sprite must lie inside at least one frame"
        max_depth = max(frame["depth"] for frame in containers)
        deepest = [frame for frame in containers if frame["depth"] == max_depth]
        assert len(deepest) == 1, "sprite containment must be unambiguous"
        frame = deepest[0]
        direction = DIRECTIONS[frame["depth"] % 4]
        box = _bbox(comp)
        sprites.append(
            {
                "id": sprite_id,
                "coords": comp,
                "bbox": box,
                "frame_id": frame["id"],
                "frame_box": frame["bbox"],
                "frame_depth": frame["depth"],
                "frame_color": frame["color"],
                "direction": direction,
                "distance": _distance_to_target(frame["bbox"], comp, direction),
            }
        )

    assert sprites, "expected at least one grey sprite"
    return frames, sprites, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each grey sprite to the side encoded by the depth of its containing nested frame.

    Input structure:
    - The background is black.
    - Foreground non-grey objects are one-cell-thick hollow rectangular frames, possibly nested into independent groups.
    - Grey 4-connected sprites sit inside the deepest frame that directly contains them.

    Rule:
    - Compute every frame's nesting depth: outermost frames have depth 0, frames inside one other frame have depth 1, and so on.
    - Map depth modulo 4 to a slide direction: 0=right, 1=down, 2=left, 3=up.
    - For each grey sprite, use the deepest frame that contains it, sort sprites by target-edge distance then bbox top-left, and slide them rigidly until the next step would hit a frame, a settled sprite, or the grid boundary.

    Key insight:
    - Frame colors are distractors; the true motion code is the geometric nesting depth of the containing frame.
    """
    g = np.asarray(grid, dtype=int)
    frames, sprites, bg = _analyze_grid(g)
    out = g.copy()
    out[out == SPRITE_COLOR] = bg
    occupied = out != bg
    h, w = g.shape

    ordered = sorted(sprites, key=lambda item: (item["distance"], item["bbox"][0], item["bbox"][1]))
    for sprite in ordered:
        coords = sprite["coords"].astype(int).copy()
        dr, dc = sprite["direction"]
        step = np.array([dr, dc], dtype=int)
        while True:
            nxt = coords + step
            if not (
                np.all(0 <= nxt[:, 0])
                and np.all(nxt[:, 0] < h)
                and np.all(0 <= nxt[:, 1])
                and np.all(nxt[:, 1] < w)
            ):
                break
            if np.any(occupied[nxt[:, 0], nxt[:, 1]]):
                break
            coords = nxt
        out[coords[:, 0], coords[:, 1]] = SPRITE_COLOR
        occupied[coords[:, 0], coords[:, 1]] = True

    return out


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(shape)
    return tuple(sorted((r - r0, c - c0) for r, c in shape))


def _rotate_shape(shape: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    out = tuple(shape)
    for _ in range(turns % 4):
        h = max(r for r, _ in out) + 1
        out = tuple((c, h - 1 - r) for r, c in out)
        out = _normalize_shape(out)
    return _normalize_shape(out)


def _random_shape() -> tuple[tuple[int, int], ...]:
    return _rotate_shape(random.choice(SHAPES), random.randrange(4))


def _translate(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r0 + r, c0 + c) for r, c in shape))


def _coords_array(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    return np.array(coords, dtype=int)


def _coords_with_margin(coords: tuple[tuple[int, int], ...], margin: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                out.add((r + dr, c + dc))
    return out


def _inside_margin(coords: tuple[tuple[int, int], ...], ih: int, iw: int, gap: int = 1) -> bool:
    return all(gap <= r <= ih - 1 - gap and gap <= c <= iw - 1 - gap for r, c in coords)


def _can_step(coords: tuple[tuple[int, int], ...], direction: tuple[int, int], ih: int, iw: int, occupied: set[tuple[int, int]]) -> bool:
    dr, dc = direction
    for r, c in coords:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < ih and 0 <= nc < iw):
            return False
        if (nr, nc) in occupied:
            return False
    return True


def _settled_candidates(
    shape: tuple[tuple[int, int], ...],
    ih: int,
    iw: int,
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> list[tuple[tuple[tuple[int, int], ...], int]]:
    r0, c0, r1, c1 = _bbox(shape)
    sh = r1 - r0 + 1
    sw = c1 - c0 + 1
    candidates: list[tuple[tuple[tuple[int, int], ...], int]] = []
    for r in range(ih - sh + 1):
        for c in range(iw - sw + 1):
            placed = _translate(shape, r, c)
            if any(cell in occupied for cell in placed):
                continue
            if _can_step(placed, direction, ih, iw, occupied):
                continue
            arr = _coords_array(placed)
            base = _distance_to_target((0, 0, ih + 1, iw + 1), arr + np.array([1, 1]), direction)
            candidates.append((placed, base))
    random.shuffle(candidates)
    return candidates


def _backshift_options(
    final_coords: tuple[tuple[int, int], ...],
    direction: tuple[int, int],
    ih: int,
    iw: int,
) -> list[tuple[int, tuple[tuple[int, int], ...], int]]:
    dr, dc = direction
    arr = _coords_array(final_coords)
    if (dr, dc) == (1, 0):
        max_back = min(r for r, _ in final_coords) - 1
        step = (-1, 0)
    elif (dr, dc) == (0, -1):
        max_back = iw - 2 - max(c for _, c in final_coords)
        step = (0, 1)
    elif (dr, dc) == (0, 1):
        max_back = min(c for _, c in final_coords) - 1
        step = (0, -1)
    else:
        max_back = ih - 2 - max(r for r, _ in final_coords)
        step = (1, 0)

    options: list[tuple[int, tuple[tuple[int, int], ...], int]] = []
    frame_box = (0, 0, ih + 1, iw + 1)
    for dist in range(1, max_back + 1):
        shifted = tuple(sorted((r + step[0] * dist, c + step[1] * dist) for r, c in final_coords))
        if not _inside_margin(shifted, ih, iw, gap=1):
            continue
        boundary_distance = _distance_to_target(frame_box, _coords_array(shifted) + np.array([1, 1]), direction)
        options.append((dist, shifted, boundary_distance))
    random.shuffle(options)
    return options


def _paint_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _build_group_frames(layers: int, inner_h: int, inner_w: int, gaps: list[int]) -> list[tuple[int, int, int, int]]:
    total_shrink = sum(gap + 1 for gap in gaps)
    outer_h = inner_h + 2 + 2 * total_shrink
    outer_w = inner_w + 2 + 2 * total_shrink
    boxes = [(0, 0, outer_h - 1, outer_w - 1)]
    for gap in gaps:
        r0, c0, r1, c1 = boxes[-1]
        d = gap + 1
        boxes.append((r0 + d, c0 + d, r1 - d, c1 - d))
    assert len(boxes) == layers
    return boxes


def _finalize_group(
    layers: int,
    colors: list[int],
    final_local: list[tuple[tuple[int, int], ...]],
    input_local: list[tuple[tuple[int, int], ...]],
    boxes: list[tuple[int, int, int, int]],
) -> dict:
    patch_h = boxes[0][2] + 1
    patch_w = boxes[0][3] + 1
    inp = np.zeros((patch_h, patch_w), dtype=int)
    out = np.zeros_like(inp)
    for box, color in zip(boxes, colors):
        _paint_frame(inp, box, color)
        _paint_frame(out, box, color)
    inner_box = boxes[-1]
    origin = (inner_box[0] + 1, inner_box[1] + 1)
    for coords in input_local:
        for r, c in coords:
            inp[origin[0] + r, origin[1] + c] = SPRITE_COLOR
    for coords in final_local:
        for r, c in coords:
            out[origin[0] + r, origin[1] + c] = SPRITE_COLOR
    return {
        "layers": layers,
        "direction": DIRECTIONS[(layers - 1) % 4],
        "colors": colors,
        "inner_color": colors[-1],
        "input_patch": inp,
        "output_patch": out,
        "boxes": boxes,
        "inner_box": inner_box,
        "input_local": input_local,
        "final_local": final_local,
    }


def _generate_tie_down_group(colors: list[int]) -> dict | None:
    layers = 2
    gaps = [1]
    inner_h = random.randint(5, 6)
    inner_w = 7
    boxes = _build_group_frames(layers, inner_h, inner_w, gaps)
    shape_a = ((0, 0), (1, 0), (1, 1))
    shape_b = ((0, 1), (1, 0), (1, 1))
    if random.random() < 0.5:
        shape_a, shape_b = shape_b, shape_a
    row = inner_h - 2
    first_final = _translate(shape_a, row, 1)
    second_final = _translate(shape_b, row, 4)
    max_shared = row - 1
    if max_shared < 1:
        return None
    dist = random.randint(1, max_shared)
    first_input = tuple(sorted((r - dist, c) for r, c in first_final))
    second_input = tuple(sorted((r - dist, c) for r, c in second_final))
    if not _inside_margin(first_input, inner_h, inner_w, 1):
        return None
    if not _inside_margin(second_input, inner_h, inner_w, 1):
        return None
    if _coords_with_margin(first_input, 1) & set(second_input):
        return None
    return _finalize_group(layers, colors, [first_final, second_final], [first_input, second_input], boxes)


def _generate_blocking_left_group(colors: list[int]) -> dict | None:
    layers = 3
    gaps = [1, 1]
    inner_h = 6
    inner_w = 7
    boxes = _build_group_frames(layers, inner_h, inner_w, gaps)
    shape_a = ((0, 0), (1, 0), (1, 1))
    shape_b = ((0, 1), (1, 0), (1, 1))
    row = random.randint(1, 3)
    if random.random() < 0.5:
        shape_a, shape_b = shape_b, shape_a
    first_final = _translate(shape_a, row, 0)
    second_final = _translate(shape_b, row, 2)
    first_input = tuple(sorted((r, c + 1) for r, c in first_final))
    second_input = tuple(sorted((r, c + 2) for r, c in second_final))
    if not _inside_margin(first_input, inner_h, inner_w, 1):
        return None
    if not _inside_margin(second_input, inner_h, inner_w, 1):
        return None
    if _coords_with_margin(first_input, 1) & set(second_input):
        return None
    return _finalize_group(layers, colors, [first_final, second_final], [first_input, second_input], boxes)


def _generate_generic_group(layers: int, colors: list[int]) -> dict | None:
    direction = DIRECTIONS[(layers - 1) % 4]
    sprite_count = random.choice((1, 2))
    if layers == 2:
        gaps = [random.choice((1, 1, 2))]
        inner_h = random.randint(5 if sprite_count == 2 else 4, 7)
        inner_w = random.randint(5 if sprite_count == 2 else 4, 7)
    else:
        gaps = [1, 1]
        inner_h = random.randint(5, 6)
        inner_w = random.randint(5, 6)
    boxes = _build_group_frames(layers, inner_h, inner_w, gaps)

    for _ in range(160):
        occupied: set[tuple[int, int]] = set()
        finals: list[tuple[tuple[int, int], ...]] = []
        bases: list[int] = []
        for _sprite_idx in range(sprite_count):
            candidates = _settled_candidates(_random_shape(), inner_h, inner_w, direction, occupied)
            if not candidates:
                break
            coords, base = random.choice(candidates)
            finals.append(coords)
            bases.append(base)
            occupied.update(coords)
        if len(finals) != sprite_count:
            continue

        input_local: list[tuple[tuple[int, int], ...]] = []
        used: set[tuple[int, int]] = set()
        prev_distance = -1
        ok = True
        for coords in finals:
            options = _backshift_options(coords, direction, inner_h, inner_w)
            options = [opt for opt in options if opt[2] > prev_distance]
            random.shuffle(options)
            picked = None
            for _d, shifted, dist in options:
                if _coords_with_margin(shifted, 1) & used:
                    continue
                picked = (shifted, dist)
                break
            if picked is None:
                ok = False
                break
            shifted, dist = picked
            input_local.append(shifted)
            used.update(shifted)
            prev_distance = dist
        if ok:
            return _finalize_group(layers, colors, finals, input_local, boxes)
    return None


def _pick_group_colors(layer_counts: list[int]) -> list[list[int]]:
    shared_inner = random.choice(FRAME_COLORS)
    groups: list[list[int]] = []
    shared_assigned = False
    for layers in layer_counts:
        pool = list(FRAME_COLORS)
        random.shuffle(pool)
        if not shared_assigned or (layers == 3 and all(len(g) == 2 for g in groups)):
            colors = [pool.pop() for _ in range(layers - 1)] + [shared_inner]
            while len(set(colors)) != len(colors):
                random.shuffle(pool)
                colors = [pool.pop() for _ in range(layers - 1)] + [shared_inner]
            shared_assigned = True
        else:
            colors = []
            while len(colors) < layers:
                color = random.choice(FRAME_COLORS)
                if color not in colors:
                    colors.append(color)
            if layers == 2 and random.random() < 0.5:
                colors[-1] = shared_inner
                if len(set(colors)) < len(colors):
                    colors[0] = next(c for c in FRAME_COLORS if c not in colors)
        groups.append(colors)
    if len(groups) >= 2:
        groups[0][-1] = shared_inner
        groups[1][-1] = shared_inner
        if len(set(groups[0])) < len(groups[0]):
            groups[0][0] = next(c for c in FRAME_COLORS if c != shared_inner and c not in groups[0][1:])
        if len(set(groups[1])) < len(groups[1]):
            groups[1][0] = next(c for c in FRAME_COLORS if c != shared_inner and c not in groups[1][1:])
    return groups


def _layout_groups(groups: list[dict]) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    patches_in = [group["input_patch"] for group in groups]
    patches_out = [group["output_patch"] for group in groups]
    dims = [(patch.shape[0], patch.shape[1]) for patch in patches_in]
    gap = 1

    placements: list[tuple[int, int]] = []
    if len(groups) == 2:
        options: list[tuple[str, int, int]] = []
        row_h = max(dims[0][0], dims[1][0])
        row_w = dims[0][1] + gap + dims[1][1]
        if max(row_h, row_w) <= 30:
            options.append(("row", row_h, row_w))
        col_h = dims[0][0] + gap + dims[1][0]
        col_w = max(dims[0][1], dims[1][1])
        if max(col_h, col_w) <= 30:
            options.append(("col", col_h, col_w))
        assert options, f"no feasible 2-group layout for {dims}"
        mode, h, w = random.choice(options)
        if mode == "row":
            extra_r0 = random.randint(0, row_h - dims[0][0])
            extra_r1 = random.randint(0, row_h - dims[1][0])
            slack = w - row_w
            left_pad = random.randint(0, slack) if slack > 0 else 0
            placements = [
                (extra_r0, left_pad),
                (extra_r1, left_pad + dims[0][1] + gap),
            ]
        else:
            extra_c0 = random.randint(0, col_w - dims[0][1])
            extra_c1 = random.randint(0, col_w - dims[1][1])
            slack = h - col_h
            top_pad = random.randint(0, slack) if slack > 0 else 0
            placements = [
                (top_pad, extra_c0),
                (top_pad + dims[0][0] + gap, extra_c1),
            ]
    else:
        pad_top = pad_left = pad_bottom = pad_right = 1
        order = list(range(3))
        random.shuffle(order)
        top = order[:2]
        bottom = order[2]
        top_h = max(dims[top[0]][0], dims[top[1]][0])
        top_w = dims[top[0]][1] + gap + dims[top[1]][1]
        bottom_h = dims[bottom][0]
        bottom_w = dims[bottom][1]
        width = max(top_w, bottom_w)
        height = top_h + gap + bottom_h
        h = height + pad_top + pad_bottom
        w = width + pad_left + pad_right
        positions: dict[int, tuple[int, int]] = {}
        left_top = random.randint(0, width - top_w)
        left_bottom = random.randint(0, width - bottom_w)
        positions[top[0]] = (
            pad_top + random.randint(0, top_h - dims[top[0]][0]),
            pad_left + left_top,
        )
        positions[top[1]] = (
            pad_top + random.randint(0, top_h - dims[top[1]][0]),
            pad_left + left_top + dims[top[0]][1] + gap,
        )
        positions[bottom] = (
            pad_top + top_h + gap,
            pad_left + left_bottom,
        )
        placements = [positions[i] for i in range(3)]

    assert max(h, w) <= 30, f"layout exceeds ARC limit: {(h, w)}"
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    for patch_in, patch_out, (r0, c0) in zip(patches_in, patches_out, placements):
        ph, pw = patch_in.shape
        inp[r0 : r0 + ph, c0 : c0 + pw] = np.maximum(inp[r0 : r0 + ph, c0 : c0 + pw], patch_in)
        out[r0 : r0 + ph, c0 : c0 + pw] = np.maximum(out[r0 : r0 + ph, c0 : c0 + pw], patch_out)
    return inp, out, placements


def _validate_generated(inp: np.ndarray, out: np.ndarray, groups: list[dict], placements: list[tuple[int, int]]) -> None:
    solved = main(inp)
    assert np.array_equal(solved, out), "solver output must match the constructed oracle output"

    frames, sprites, _bg = _analyze_grid(inp)
    roots = [frame for frame in frames if frame["depth"] == 0]
    assert 2 <= len(roots) <= 3, "expected 2 or 3 independent root groups"

    layer_by_root: dict[int, int] = {frame["id"]: 1 for frame in roots}
    root_of: dict[int, int] = {}
    for frame in frames:
        cur = frame
        while cur["parent_id"] is not None:
            cur = next(other for other in frames if other["id"] == cur["parent_id"])
        root_of[frame["id"]] = cur["id"]
        layer_by_root[cur["id"]] = max(layer_by_root[cur["id"]], frame["depth"] + 1)
    assert set(layer_by_root.values()) <= {2, 3}, "generator should only use 2- or 3-layer groups"
    assert 2 in layer_by_root.values() and 3 in layer_by_root.values(), "reachable depth branches 1 and 2 must both appear"

    deepest_depth_per_root = {
        root_id: max(frame["depth"] for frame in frames if root_of[frame["id"]] == root_id)
        for root_id in layer_by_root
    }
    sprites_per_root = Counter(root_of[sprite["frame_id"]] for sprite in sprites)
    for root_id, deepest in deepest_depth_per_root.items():
        deepest_frames = [frame for frame in frames if root_of[frame["id"]] == root_id and frame["depth"] == deepest]
        assert len(deepest_frames) == 1, "each generated group should be a single nesting chain"
        assert 1 <= sprites_per_root[root_id] <= 2, "each innermost frame should contain 1 or 2 sprites"

    direction_set = {sprite["direction"] for sprite in sprites}
    assert direction_set == {(1, 0), (0, -1)}, "strict 2-3 layer generation should realize down and left directions"

    color_depths: dict[int, set[int]] = {}
    for sprite in sprites:
        color_depths.setdefault(int(sprite["frame_color"]), set()).add(int(sprite["frame_depth"]))
    assert any(depths == {1, 2} for depths in color_depths.values()), "same frame color should appear at different depths"

    # At least one tied distance branch must be present in the input ordering.
    ordered = sorted(sprites, key=lambda item: (item["distance"], item["bbox"][0], item["bbox"][1]))
    assert any(ordered[i]["distance"] == ordered[i + 1]["distance"] for i in range(len(ordered) - 1)), "need a distance tie case"
    assert any(
        min(c for _, c in group["final_local"][1]) > 0
        for group in groups
        if group["direction"] == (0, -1) and len(group["final_local"]) == 2
    ), "one left-moving group should include a sprite blocked before the border"


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    n_groups = 2
    layer_counts = [2, 3]
    color_sets = _pick_group_colors(layer_counts)

    groups: list[dict] = []
    tie_group = _generate_tie_down_group(color_sets[0])
    if tie_group is None:
        raise RuntimeError("failed to construct tie group")
    groups.append(tie_group)

    block_group = _generate_blocking_left_group(color_sets[1])
    if block_group is None:
        raise RuntimeError("failed to construct blocking group")
    groups.append(block_group)

    inp, out, placements = _layout_groups(groups)
    _validate_generated(inp, out, groups, placements)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, _out = _generate_case()
        except Exception:
            continue
        return inp
    raise RuntimeError("failed to generate a valid nested-frame sliding input")
