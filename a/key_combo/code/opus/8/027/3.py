import random
from collections import Counter

import numpy as np


FRAME = 5
BG = 0
PALETTE = (1, 2, 3, 4, 7)
GEN_ATTEMPTS = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:, :] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]


def _is_4_connected(coords: np.ndarray) -> bool:
    pts = {tuple(map(int, rc)) for rc in coords}
    if not pts:
        return False
    stack = [next(iter(pts))]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in _neighbors4(*cur):
            if nxt in pts and nxt not in seen:
                stack.append(nxt)
    return len(seen) == len(pts)


def _rows_to_shape(*rows: str) -> np.ndarray:
    coords = [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "#"]
    arr = np.array(coords, dtype=int)
    arr -= arr.min(axis=0)
    assert _is_4_connected(arr)
    return arr


SHAPES = {
    "outer_h": _rows_to_shape(
        "####",
        "...#",
        "...#",
    ),
    "outer_v": _rows_to_shape(
        "#..",
        "#..",
        "#..",
        "###",
    ),
    "outer_double": _rows_to_shape(
        "####",
        "...#",
        "...#",
        "...#",
    ),
    "dom_h": _rows_to_shape("##"),
    "dom_v": _rows_to_shape("#", "#"),
    "square": _rows_to_shape("##", "##"),
    "el": _rows_to_shape("#.", "##"),
    "line3_h": _rows_to_shape("###"),
    "line3_v": _rows_to_shape("#", "#", "#"),
    "zig5": _rows_to_shape("##.", ".##", "..#"),
}


def _shape_dims(shape: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox_coords(shape)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _normalize(coords: np.ndarray) -> np.ndarray:
    return coords - coords.min(axis=0)


def _shape_to_patch(shape: np.ndarray, color: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.zeros((h, w), dtype=int)
    patch[shape[:, 0], shape[:, 1]] = int(color)
    return patch


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        int(inner[0]) > int(outer[0])
        and int(inner[1]) > int(outer[1])
        and int(inner[2]) < int(outer[2])
        and int(inner[3]) < int(outer[3])
    )


def _blank_panel(height: int, width: int) -> np.ndarray:
    out = np.zeros((int(height), int(width)), dtype=int)
    out[0, :] = FRAME
    out[-1, :] = FRAME
    out[:, 0] = FRAME
    out[:, -1] = FRAME
    return out


def _panel_objects(panel: np.ndarray) -> dict[int, dict]:
    colors = sorted(set(map(int, panel.flatten())) - {BG, FRAME})
    objects = {}
    for color in colors:
        coords = np.argwhere(panel == int(color))
        assert coords.size > 0
        assert _is_4_connected(coords)
        bbox = _bbox_coords(coords)
        shape = _normalize(coords)
        crop = _crop_bbox(panel, bbox)
        patch = np.where(crop == int(color), int(color), BG)
        objects[int(color)] = {
            "color": int(color),
            "coords": coords,
            "bbox": bbox,
            "shape": shape,
            "patch": patch,
        }
    return objects


def _split_panels(grid: np.ndarray) -> tuple[int, list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    frame = int(g[0, 0])
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == frame)]
    assert len(sep_cols) == 4
    strides = [b - a for a, b in zip(sep_cols, sep_cols[1:])]
    assert len(set(strides)) == 1
    panels = [g[:, sep_cols[i] : sep_cols[i + 1] + 1].copy() for i in range(3)]
    return frame, panels


def _draw_shape(panel: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> None:
    patch = _shape_to_patch(shape, color)
    _paste(panel, patch, int(top), int(left), transparent=BG)


def main(grid: np.ndarray) -> np.ndarray:
    """Extrapolate all moving panel objects to the next frame, then erase those strictly bbox-contained by a different color.

    Input structure:
    - The grid is three equal-height framed panels laid out horizontally with shared vertical border columns.
    - Each panel has a grey border, black interior, and one 4-connected monochrome object for each foreground color.
    - The same color keeps the same shape across panels and moves by a constant per-frame offset in panel-local coordinates.

    Rule:
    - Parse the three input panels and match objects across frames by color.
    - For each color, infer the constant motion from panel 1->2 and 2->3 using the object's bbox top-left corner.
    - Build one standalone output panel with the same size and border, place every object at its extrapolated panel-4 position, then compute all pairwise strict bbox containments.
    - Erase every object whose bbox is strictly inside another object's bbox of a different color, applying the removals simultaneously.

    Key insight:
    - The decisive containment relation does not exist in the visible input frames; it appears only after independently extrapolating every object's motion to the unseen next frame.
    """
    _frame, panels = _split_panels(np.asarray(grid, dtype=int))
    height, width = panels[0].shape
    panel_objects = [_panel_objects(panel) for panel in panels]

    colors0 = set(panel_objects[0])
    assert colors0
    assert all(set(objs) == colors0 for objs in panel_objects[1:])

    out = _blank_panel(height, width)
    placed = []
    for color in sorted(colors0):
        obj1 = panel_objects[0][color]
        obj2 = panel_objects[1][color]
        obj3 = panel_objects[2][color]

        assert np.array_equal(obj1["shape"], obj2["shape"])
        assert np.array_equal(obj1["shape"], obj3["shape"])

        r1, c1, _, _ = obj1["bbox"]
        r2, c2, _, _ = obj2["bbox"]
        r3, c3, _, _ = obj3["bbox"]
        delta1 = (int(r2 - r1), int(c2 - c1))
        delta2 = (int(r3 - r2), int(c3 - c2))
        assert delta1 == delta2

        nr = int(r3 + delta1[0])
        nc = int(c3 + delta1[1])
        h, w = obj3["patch"].shape
        assert 1 <= nr and nr + h <= height - 1
        assert 1 <= nc and nc + w <= width - 1
        _paste(out, obj3["patch"], nr, nc, transparent=BG)
        coords = obj3["shape"] + np.array([[nr, nc]], dtype=int)
        placed.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": _bbox_coords(coords),
            }
        )

    removed = set()
    for obj in placed:
        for other in placed:
            if obj["color"] == other["color"]:
                continue
            if _strictly_contains(other["bbox"], obj["bbox"]):
                removed.add(int(obj["color"]))
                break

    for obj in placed:
        if int(obj["color"]) in removed:
            out[obj["coords"][:, 0], obj["coords"][:, 1]] = BG
    return out


def _pos_at(obj: dict, t: int) -> tuple[int, int]:
    dr, dc = obj["delta"]
    r4, c4 = obj["p4"]
    step = 4 - int(t)
    return int(r4 - step * dr), int(c4 - step * dc)


def _object_bbox(obj: dict, t: int) -> tuple[int, int, int, int]:
    top, left = _pos_at(obj, t)
    h, w = _shape_dims(obj["shape"])
    return int(top), int(left), int(top + h - 1), int(left + w - 1)


def _cells_at(obj: dict, t: int) -> list[tuple[int, int]]:
    top, left = _pos_at(obj, t)
    return [(int(top + r), int(left + c)) for r, c in obj["shape"]]


def _in_panel(shape: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = _shape_dims(shape)
    return 1 <= int(top) and int(top + h) <= int(height - 1) and 1 <= int(left) and int(left + w) <= int(width - 1)


def _frame_from_specs(specs: list[dict], height: int, width: int, t: int) -> np.ndarray | None:
    panel = _blank_panel(height, width)
    occupied = {}
    for obj in specs:
        top, left = _pos_at(obj, t)
        if not _in_panel(obj["shape"], top, left, height, width):
            return None
        for cell in _cells_at(obj, t):
            if cell in occupied:
                return None
            occupied[cell] = int(obj["color"])
        _draw_shape(panel, obj["shape"], top, left, int(obj["color"]))
    return panel


def _expected_output(specs: list[dict], height: int, width: int) -> tuple[np.ndarray, set[int]]:
    panel = _blank_panel(height, width)
    placed = []
    for obj in specs:
        top, left = _pos_at(obj, 4)
        _draw_shape(panel, obj["shape"], top, left, int(obj["color"]))
        coords = np.array(_cells_at(obj, 4), dtype=int)
        placed.append({"color": int(obj["color"]), "coords": coords, "bbox": _bbox_coords(coords)})

    removed = set()
    for obj in placed:
        for other in placed:
            if obj["color"] == other["color"]:
                continue
            if _strictly_contains(other["bbox"], obj["bbox"]):
                removed.add(int(obj["color"]))
                break

    for obj in placed:
        if int(obj["color"]) in removed:
            panel[obj["coords"][:, 0], obj["coords"][:, 1]] = BG
    return panel, removed


def _compose_input(frames: list[np.ndarray]) -> np.ndarray:
    panel_h, panel_w = frames[0].shape
    out = np.zeros((panel_h, 3 * (panel_w - 1) + 1), dtype=int)
    for i, panel in enumerate(frames):
        _paste(out, panel, 0, i * (panel_w - 1))
    return out


def _shape_spans_full_bbox(shape: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_coords(shape)
    rows = set(int(r) for r, _ in shape)
    cols = set(int(c) for _, c in shape)
    return 0 in rows and int(r1) in rows and 0 in cols and int(c1) in cols


def _semantic_ok(specs: list[dict], height: int, width: int) -> bool:
    if len({int(obj["color"]) for obj in specs}) != len(specs):
        return False
    for obj in specs:
        if not _is_4_connected(obj["shape"]):
            return False
        if not _shape_spans_full_bbox(obj["shape"]):
            return False

    frames = []
    containment_counts = []
    for t in (1, 2, 3, 4):
        panel = _frame_from_specs(specs, height, width, t)
        if panel is None:
            return False
        frames.append(panel)
        bboxes = {int(obj["color"]): _object_bbox(obj, t) for obj in specs}
        cnt = 0
        for a in specs:
            for b in specs:
                if a["color"] == b["color"]:
                    continue
                if _strictly_contains(bboxes[int(b["color"])], bboxes[int(a["color"])]):
                    cnt += 1
        containment_counts.append(cnt)

    if any(containment_counts[i] != 0 for i in range(3)):
        return False
    if containment_counts[3] == 0:
        return False

    expected, removed = _expected_output(specs, height, width)
    if not removed:
        return False
    inp = _compose_input(frames[:3])
    try:
        predicted = main(inp)
    except Exception:
        return False
    if not np.array_equal(predicted, expected):
        return False

    frame, panels = _split_panels(inp)
    if int(frame) != FRAME or len(panels) != 3:
        return False
    for color in {int(obj["color"]) for obj in specs}:
        sigs = []
        for panel in panels:
            coords = np.argwhere(panel == int(color))
            if coords.size == 0:
                return False
            sigs.append(_normalize(coords))
        if not (np.array_equal(sigs[0], sigs[1]) and np.array_equal(sigs[0], sigs[2])):
            return False
    return True


def _feasible_positions(shape: np.ndarray, delta: tuple[int, int], height: int, width: int) -> list[tuple[int, int]]:
    h, w = _shape_dims(shape)
    out = []
    for top in range(1, height - h):
        for left in range(1, width - w):
            if all(_in_panel(shape, top - (4 - t) * delta[0], left - (4 - t) * delta[1], height, width) for t in (1, 2, 3, 4)):
                out.append((int(top), int(left)))
    return out


def _pick_distractor(specs: list[dict], height: int, width: int, colors: list[int]) -> list[dict] | None:
    shape_names = ["el", "square", "line3_h", "line3_v", "zig5"]
    delta_choices = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (1, 1), (-1, 1), (1, -1)]
    for _ in range(120):
        shape = SHAPES[random.choice(shape_names)]
        delta = random.choice(delta_choices)
        pos_choices = _feasible_positions(shape, delta, height, width)
        if not pos_choices:
            continue
        color = random.choice(colors)
        obj = {
            "color": int(color),
            "shape": shape,
            "delta": (int(delta[0]), int(delta[1])),
            "p4": random.choice(pos_choices),
        }
        new_specs = specs + [obj]
        if _semantic_ok(new_specs, height, width):
            return new_specs
    return None


def _try_single_horizontal(height: int, width: int, colors: list[int]) -> list[dict] | None:
    shift_r = random.choice([-1, 0, 1])
    outer = {
        "color": int(colors[0]),
        "shape": SHAPES["outer_h"],
        "delta": (int(shift_r), 0),
    }
    inner = {
        "color": int(colors[1]),
        "shape": SHAPES["dom_h"],
        "delta": (int(shift_r), 1),
    }
    outer_positions = _feasible_positions(outer["shape"], outer["delta"], height, width)
    random.shuffle(outer_positions)
    for top, left in outer_positions:
        specs = [
            {**outer, "p4": (int(top), int(left))},
            {**inner, "p4": (int(top + 1), int(left + 1))},
        ]
        if len(colors) == 2:
            if _semantic_ok(specs, height, width):
                return specs
            continue
        with_distractor = _pick_distractor(specs, height, width, colors[2:])
        if with_distractor is not None:
            return with_distractor
    return None


def _try_single_vertical(height: int, width: int, colors: list[int]) -> list[dict] | None:
    shift_c = random.choice([-1, 0, 1])
    outer = {
        "color": int(colors[0]),
        "shape": SHAPES["outer_v"],
        "delta": (0, int(shift_c)),
    }
    inner = {
        "color": int(colors[1]),
        "shape": SHAPES["dom_v"],
        "delta": (1, int(shift_c)),
    }
    outer_positions = _feasible_positions(outer["shape"], outer["delta"], height, width)
    random.shuffle(outer_positions)
    for top, left in outer_positions:
        specs = [
            {**outer, "p4": (int(top), int(left))},
            {**inner, "p4": (int(top + 1), int(left + 1))},
        ]
        if len(colors) == 2:
            if _semantic_ok(specs, height, width):
                return specs
            continue
        with_distractor = _pick_distractor(specs, height, width, colors[2:])
        if with_distractor is not None:
            return with_distractor
    return None


def _try_double_horizontal(height: int, width: int, colors: list[int]) -> list[dict] | None:
    shift_r = random.choice([-1, 0, 1])
    outer = {
        "color": int(colors[0]),
        "shape": SHAPES["outer_double"],
        "delta": (int(shift_r), 0),
        "p4": None,
    }
    inner1 = {
        "color": int(colors[1]),
        "shape": SHAPES["dom_h"],
        "delta": (int(shift_r), 1),
        "p4": None,
    }
    inner2 = {
        "color": int(colors[2]),
        "shape": SHAPES["dom_h"],
        "delta": (int(shift_r), 1),
        "p4": None,
    }
    outer_positions = _feasible_positions(outer["shape"], outer["delta"], height, width)
    random.shuffle(outer_positions)
    for top, left in outer_positions:
        specs = [
            {**outer, "p4": (int(top), int(left))},
            {**inner1, "p4": (int(top + 1), int(left + 1))},
            {**inner2, "p4": (int(top + 2), int(left + 1))},
        ]
        if _semantic_ok(specs, height, width):
            return specs
    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(9, 10)
    width = random.randint(8, 10)
    scenario = random.choices(
        ["single_h_2", "single_h_3", "single_v_2", "single_v_3", "double_h"],
        weights=[18, 24, 18, 24, 16],
        k=1,
    )[0]
    if scenario == "double_h":
        colors = random.sample(PALETTE, 3)
        specs = _try_double_horizontal(height, width, colors)
    elif scenario == "single_h_2":
        colors = random.sample(PALETTE, 2)
        specs = _try_single_horizontal(height, width, colors)
    elif scenario == "single_v_2":
        colors = random.sample(PALETTE, 2)
        specs = _try_single_vertical(height, width, colors)
    elif scenario == "single_h_3":
        colors = random.sample(PALETTE, 3)
        specs = _try_single_horizontal(height, width, colors)
    else:
        colors = random.sample(PALETTE, 3)
        specs = _try_single_vertical(height, width, colors)
    if specs is None:
        return None

    frames = [_frame_from_specs(specs, height, width, t) for t in (1, 2, 3)]
    if any(frame is None for frame in frames):
        return None
    inp = _compose_input(frames)  # type: ignore[arg-type]
    if not _semantic_ok(specs, height, width):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
