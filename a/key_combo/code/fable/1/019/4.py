import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
WALL = 5
SAND = 4

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, *, exclude: Iterable[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= {int(v) for v in exclude}
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        stack = [(rr, cc)]
        coords = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.asarray(coords, dtype=int))
    return comps


def _box_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _parse_box_component(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    rel = np.zeros((h, w), dtype=bool)
    rel[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = _box_mask(h, w)
    if np.any(rel & ~border):
        return None

    missing = np.argwhere(border & ~rel)
    if missing.shape[0] != 1:
        return None
    mr, mc = map(int, missing[0])
    if (mr in (0, h - 1)) and (mc in (0, w - 1)):
        return None

    if mr == 0:
        side = "up"
    elif mr == h - 1:
        side = "down"
    elif mc == 0:
        side = "left"
    else:
        side = "right"

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": (r0 + mr, c0 + mc),
        "side": side,
        "interior": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
    }


def _extract_scene(grid: np.ndarray) -> tuple[int, int, int, list[dict]]:
    bg = get_bg(grid)
    non_bg = colors_present(grid, exclude={bg})
    assert len(non_bg) >= 2, "expected wall color and sand color"

    candidates: list[tuple[int, list[dict]]] = []
    for color in non_bg:
        comps = components_mask(grid == color)
        parsed = []
        valid = True
        for comp in comps:
            box = _parse_box_component(comp)
            if box is None:
                valid = False
                break
            parsed.append(box)
        if valid and parsed:
            candidates.append((color, parsed))

    assert len(candidates) == 1, "expected exactly one wall color whose components are boxes"
    wall_color, boxes = candidates[0]
    sand_colors = [color for color in non_bg if color != wall_color]
    assert len(sand_colors) == 1, "expected exactly one sand color"
    sand_color = sand_colors[0]

    boxes = sorted(boxes, key=lambda box: box["bbox"])
    return bg, int(wall_color), int(sand_color), boxes


def _box_owner(shape: tuple[int, int], boxes: list[dict]) -> np.ndarray:
    owners = np.full(shape, -1, dtype=int)
    for idx, box in enumerate(boxes):
        r0, c0, r1, c1 = box["interior"]
        if r0 > r1 or c0 > c1:
            continue
        view = owners[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(view == -1), "box interiors must not overlap"
        view[:] = idx
    return owners


def _line_key(box: dict, r: int, c: int) -> int:
    return int(r) if box["side"] in ("left", "right") else int(c)


def _lane_start(box: dict, key: int) -> tuple[int, int]:
    r0, c0, r1, c1 = box["bbox"]
    if box["side"] == "right":
        return int(key), int(c0 + 1)
    if box["side"] == "left":
        return int(key), int(c1 - 1)
    if box["side"] == "down":
        return int(r0 + 1), int(key)
    return int(r1 - 1), int(key)


def _trace_lane(start: tuple[int, int], direction: tuple[int, int], wall_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = wall_mask.shape
    dr, dc = direction
    r, c = start
    assert 0 <= r < h and 0 <= c < w and not wall_mask[r, c]
    cells = []
    while True:
        cells.append((int(r), int(c)))
        nr = r + dr
        nc = c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w or wall_mask[nr, nc]:
            break
        r, c = nr, nc
    return cells


def _settle_from_boxes(
    shape: tuple[int, int],
    boxes: list[dict],
    sand_counts: dict[int, dict[int, int]],
    *,
    bg: int,
    wall_color: int,
    sand_color: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    wall_mask = np.zeros(shape, dtype=bool)
    for box in boxes:
        r0, c0, r1, c1 = box["bbox"]
        out[r0, c0 : c1 + 1] = wall_color
        out[r1, c0 : c1 + 1] = wall_color
        out[r0 : r1 + 1, c0] = wall_color
        out[r0 : r1 + 1, c1] = wall_color
        gr, gc = box["gap"]
        out[gr, gc] = bg
        wall_mask[r0, c0 : c1 + 1] = True
        wall_mask[r1, c0 : c1 + 1] = True
        wall_mask[r0 : r1 + 1, c0] = True
        wall_mask[r0 : r1 + 1, c1] = True
        wall_mask[gr, gc] = False

    used: set[tuple[int, int]] = set()
    for idx, box in enumerate(boxes):
        direction = DIRS[box["side"]]
        for key, count in sorted(sand_counts.get(idx, {}).items()):
            path = _trace_lane(_lane_start(box, key), direction, wall_mask)
            assert count <= len(path), "lane cannot hold all grains"
            path_set = set(path)
            assert used.isdisjoint(path_set), "active lanes must be disjoint on valid inputs"
            used |= path_set
            for r, c in path[-count:]:
                out[r, c] = sand_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each box's sand toward its own gap side, letting only the gap-aligned lane leak straight through corridors and other box gaps.

    Input structure:
    - Black background with one wall color forming several one-cell-thick hollow rectangles, each missing exactly one non-corner wall pixel.
    - One sand color marks movable pixels, and every sand pixel starts strictly inside some box interior.
    - Different boxes may point in different directions because each box's missing wall cell sits on a different side.

    Rule:
    - Inside a box, every sand pixel moves only along the axis pointing toward that box's gap side.
    - Each row or column of motion behaves like a separate lane: grains keep their order and compact against the farthest reachable cells on that lane.
    - Only the lane aligned with the gap gets a longer track: it passes through the gap, continues straight across background cells, and if it enters another box through that box's gap it keeps the same straight line until a wall, a parked grain, or the grid edge stops it.
    - Keep every box wall and every gap unchanged, and repaint the sand at its final resting cells.

    Key insight:
    - The scene decomposes into independent 1D lanes. A box's gap defines both its private slide direction and the one special escape lane whose track can extend outside the box and relay into another box.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg, wall_color, sand_color, boxes = _extract_scene(g)
    owners = _box_owner(g.shape, boxes)
    sand_counts: dict[int, dict[int, int]] = {idx: {} for idx in range(len(boxes))}
    sand_coords = np.argwhere(g == sand_color)
    assert sand_coords.size > 0, "expected at least one sand pixel"
    for r, c in sand_coords:
        rr = int(r)
        cc = int(c)
        owner = int(owners[rr, cc])
        assert owner >= 0, "every sand pixel must lie inside exactly one box"
        key = _line_key(boxes[owner], rr, cc)
        sand_counts[owner][key] = sand_counts[owner].get(key, 0) + 1

    out = _settle_from_boxes(
        g.shape,
        boxes,
        sand_counts,
        bg=bg,
        wall_color=wall_color,
        sand_color=sand_color,
    )
    assert int(np.count_nonzero(out == sand_color)) == int(sand_coords.shape[0])
    return out


def _make_box(name: str, top: int, left: int, h: int, w: int, side: str, gap_offset: int) -> dict:
    r0 = int(top)
    c0 = int(left)
    r1 = r0 + int(h) - 1
    c1 = c0 + int(w) - 1
    if side == "up":
        gap = (r0, c0 + gap_offset)
    elif side == "down":
        gap = (r1, c0 + gap_offset)
    elif side == "left":
        gap = (r0 + gap_offset, c0)
    else:
        gap = (r0 + gap_offset, c1)
    return {
        "name": name,
        "bbox": (r0, c0, r1, c1),
        "interior": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
        "gap": gap,
        "side": side,
    }


def _interior_rows(box: dict) -> list[int]:
    r0, _c0, r1, _c1 = box["interior"]
    return list(range(r0, r1 + 1))


def _interior_cols(box: dict) -> list[int]:
    _r0, c0, _r1, c1 = box["interior"]
    return list(range(c0, c1 + 1))


def _populate_box(box: dict, counts: dict[int, int]) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    if box["side"] in ("left", "right"):
        cols = _interior_cols(box)
        for row, count in counts.items():
            sample = random.sample(cols, count)
            coords.update((int(row), int(col)) for col in sample)
    else:
        rows = _interior_rows(box)
        for col, count in counts.items():
            sample = random.sample(rows, count)
            coords.update((int(row), int(col)) for row in sample)
    return coords


def _scene_boxes() -> tuple[int, int, list[dict], dict[str, dict[int, int]], dict[str, object]]:
    h_a = random.randint(6, 8)
    w_a = random.randint(6, 8)
    h_b = random.randint(6, 8)
    w_b = random.randint(6, 8)
    h_c = random.randint(6, 8)
    w_c = random.randint(6, 7)

    gap_a = random.randint(1, h_a - 2)
    gap_b = random.randint(1, h_b - 2)
    gap_c = random.randint(1, w_c - 2)

    corridor_ab = random.randint(2, 5)
    gap_between_rows = random.randint(2, 4)
    bottom_escape = random.randint(2, 4)
    margin_top = random.randint(1, 3)
    margin_left = random.randint(1, 3)
    margin_right = random.randint(1, 3)

    a0 = _make_box("A", 0, 0, h_a, w_a, "right", gap_a)
    b0 = _make_box("B", gap_a - gap_b, w_a + corridor_ab, h_b, w_b, "left", gap_b)

    pair_top = min(a0["bbox"][0], b0["bbox"][0])
    pair_bottom = max(a0["bbox"][2], b0["bbox"][2])
    pair_right = max(a0["bbox"][3], b0["bbox"][3])

    c_left = random.randint(0, max(pair_right - w_c + 2, 0) + 2)
    c_top = pair_bottom + gap_between_rows
    c0 = _make_box("C", c_top, c_left, h_c, w_c, "down", gap_c)

    min_top = min(pair_top, c0["bbox"][0])
    max_bottom = max(pair_bottom, c0["bbox"][2])
    max_right = max(pair_right, c0["bbox"][3])
    shift_r = margin_top - min_top
    shift_c = margin_left

    boxes = []
    for box in (a0, b0, c0):
        r0, c0b, r1, c1 = box["bbox"]
        shifted = _make_box(
            box["name"],
            r0 + shift_r,
            c0b + shift_c,
            r1 - r0 + 1,
            c1 - c0b + 1,
            box["side"],
            box["gap"][0] - r0 if box["side"] in ("left", "right") else box["gap"][1] - c0b,
        )
        boxes.append(shifted)

    height = max_bottom + shift_r + 1 + bottom_escape
    width = max_right + shift_c + 1 + margin_right
    assert 12 <= height <= 30 and 12 <= width <= 30

    box_map = {box["name"]: box for box in boxes}

    a_gap_row = box_map["A"]["gap"][0]
    b_gap_row = box_map["B"]["gap"][0]
    c_gap_col = box_map["C"]["gap"][1]

    counts_a: dict[int, int] = {}
    max_relay = min(len(_interior_cols(box_map["A"])), box_map["B"]["bbox"][3] - box_map["B"]["bbox"][1])
    counts_a[a_gap_row] = random.randint(2, min(max_relay, 4))
    other_rows_a = [row for row in _interior_rows(box_map["A"]) if row != a_gap_row]
    for row in random.sample(other_rows_a, random.randint(1, min(3, len(other_rows_a)))):
        counts_a[row] = random.randint(1, min(3, len(_interior_cols(box_map["A"]))))

    counts_b: dict[int, int] = {}
    other_rows_b = [row for row in _interior_rows(box_map["B"]) if row != b_gap_row]
    chosen_b = random.sample(other_rows_b, random.randint(2, min(3, len(other_rows_b))))
    for row in chosen_b:
        counts_b[row] = random.randint(1, min(3, len(_interior_cols(box_map["B"]))))

    counts_c: dict[int, int] = {}
    min_gap_count = min(len(_interior_rows(box_map["C"])), bottom_escape + 2)
    counts_c[c_gap_col] = random.randint(2, max(2, min_gap_count))
    other_cols_c = [col for col in _interior_cols(box_map["C"]) if col != c_gap_col]
    for col in random.sample(other_cols_c, random.randint(1, min(2, len(other_cols_c)))):
        counts_c[col] = random.randint(1, min(3, len(_interior_rows(box_map["C"]))))

    sand_counts = {"A": counts_a, "B": counts_b, "C": counts_c}
    meta = {
        "relay_row": a_gap_row,
        "receiver_gap_row": b_gap_row,
        "bottom_escape": bottom_escape,
        "edge_box_gap_col": c_gap_col,
    }
    return height, width, boxes, sand_counts, meta


def _render_scene(
    shape: tuple[int, int],
    boxes: list[dict],
    sand_coords: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for box in boxes:
        r0, c0, r1, c1 = box["bbox"]
        grid[r0, c0 : c1 + 1] = WALL
        grid[r1, c0 : c1 + 1] = WALL
        grid[r0 : r1 + 1, c0] = WALL
        grid[r0 : r1 + 1, c1] = WALL
        gr, gc = box["gap"]
        grid[gr, gc] = BACKGROUND
    for r, c in sand_coords:
        grid[r, c] = SAND
    return grid


def _spec_counts(boxes: list[dict], sand_counts: dict[str, dict[int, int]]) -> dict[int, dict[int, int]]:
    name_to_index = {box["name"]: idx for idx, box in enumerate(boxes)}
    out: dict[int, dict[int, int]] = {}
    for name, counts in sand_counts.items():
        out[name_to_index[name]] = {int(k): int(v) for k, v in counts.items()}
    return out


def _spec_expected(shape: tuple[int, int], boxes: list[dict], sand_counts: dict[str, dict[int, int]]) -> np.ndarray:
    return _settle_from_boxes(
        shape,
        boxes,
        _spec_counts(boxes, sand_counts),
        bg=BACKGROUND,
        wall_color=WALL,
        sand_color=SAND,
    )


def _active_lane_paths(shape: tuple[int, int], boxes: list[dict], sand_counts: dict[str, dict[int, int]]) -> dict[tuple[str, int], list[tuple[int, int]]]:
    wall_mask = np.zeros(shape, dtype=bool)
    for box in boxes:
        r0, c0, r1, c1 = box["bbox"]
        wall_mask[r0, c0 : c1 + 1] = True
        wall_mask[r1, c0 : c1 + 1] = True
        wall_mask[r0 : r1 + 1, c0] = True
        wall_mask[r0 : r1 + 1, c1] = True
        wall_mask[box["gap"]] = False

    paths = {}
    for box in boxes:
        for key, count in sand_counts[box["name"]].items():
            if count <= 0:
                continue
            start = _lane_start(box, key)
            paths[(box["name"], int(key))] = _trace_lane(start, DIRS[box["side"]], wall_mask)
    return paths


def _validate_scene(
    inp: np.ndarray,
    expected: np.ndarray,
    boxes: list[dict],
    sand_counts: dict[str, dict[int, int]],
    meta: dict[str, object],
) -> None:
    assert inp.shape == expected.shape
    assert np.count_nonzero(inp == SAND) == np.count_nonzero(expected == SAND)
    assert not np.array_equal(inp, expected), "scene must require motion"

    owners = _box_owner(inp.shape, boxes)
    sand_in = np.argwhere(inp == SAND)
    assert sand_in.size > 0
    assert np.all(owners[sand_in[:, 0], sand_in[:, 1]] >= 0), "all input sand must start inside boxes"

    paths = _active_lane_paths(inp.shape, boxes, sand_counts)
    seen: set[tuple[int, int]] = set()
    for path in paths.values():
        assert seen.isdisjoint(path), "generated active lanes must be disjoint"
        seen.update(path)

    name_map = {box["name"]: box for box in boxes}
    relay_path = paths[("A", int(meta["relay_row"]))]
    b_box = name_map["B"]
    b_cells = {
        (r, c)
        for r in range(b_box["bbox"][0], b_box["bbox"][2] + 1)
        for c in range(b_box["bbox"][1], b_box["bbox"][3] + 1)
        if (r in (b_box["bbox"][0], b_box["bbox"][2]) or c in (b_box["bbox"][1], b_box["bbox"][3]))
        and (r, c) != b_box["gap"]
    }
    b_cells |= {
        (r, c)
        for r in range(b_box["interior"][0], b_box["interior"][2] + 1)
        for c in range(b_box["interior"][1], b_box["interior"][3] + 1)
    }
    assert any(cell in b_cells for cell in relay_path), "relay lane must enter box B"
    relay_final = set(relay_path[-sand_counts["A"][int(meta["relay_row"])] :])
    assert any(cell in b_cells for cell in relay_final), "some escaped grains must settle in box B"

    c_path = paths[("C", int(meta["edge_box_gap_col"]))]
    c_box = name_map["C"]
    outside_c = [
        cell
        for cell in c_path
        if cell[0] > c_box["bbox"][2] or cell[0] < c_box["bbox"][0] or cell[1] < c_box["bbox"][1] or cell[1] > c_box["bbox"][3]
    ]
    assert outside_c, "box C must leak outside its walls"
    final_sand = {
        (int(r), int(c))
        for r, c in np.argwhere(expected == SAND)
    }
    assert any(cell in final_sand for cell in outside_c), "some sand must finish outside every box"

    a_box = name_map["A"]
    a_non_gap_rows = [row for row in sand_counts["A"] if row != a_box["gap"][0]]
    assert a_non_gap_rows, "box A must also have non-escaping sand"

    solved = main(inp)
    assert np.array_equal(solved, expected), "solver must match independently rendered scene"


def _transform_pair(inp: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = random.randint(0, 3)
    inp = np.rot90(inp, k)
    expected = np.rot90(expected, k)
    if random.random() < 0.5:
        inp = np.fliplr(inp)
        expected = np.fliplr(expected)
    return inp.copy(), expected.copy()


def _generate_pair() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(300):
        shape_h, shape_w, boxes, sand_counts, meta = _scene_boxes()
        sand_coords: set[tuple[int, int]] = set()
        for box in boxes:
            sand_coords |= _populate_box(box, sand_counts[box["name"]])
        inp = _render_scene((shape_h, shape_w), boxes, sand_coords)
        expected = _spec_expected((shape_h, shape_w), boxes, sand_counts)
        _validate_scene(inp, expected, boxes, sand_counts, meta)
        inp, expected = _transform_pair(inp, expected)
        assert inp.ndim == 2 and expected.ndim == 2
        assert max(inp.shape) <= 30 and max(expected.shape) <= 30
        if not np.array_equal(inp, expected):
            return inp, expected
    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    return _generate_pair()[0]
