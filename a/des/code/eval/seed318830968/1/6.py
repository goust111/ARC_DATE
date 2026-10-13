from __future__ import annotations

import random

import numpy as np


SHADOW_COLOR = 5
DEFAULT_BG = 8
OBJECT_COLORS = [0, 1, 2, 3, 4, 6, 7, 9]
DIRECTIONS = ("right", "left", "down", "up")

GEN_RETRIES = 400
PLACE_RETRIES = 300
HEIGHT_RANGE = (20, 26)
WIDTH_RANGE = (20, 26)


def _dominant_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _shadow_length_from_area(area: int) -> int:
    if area == 4:
        return 1
    if area == 6:
        return 2
    assert area >= 8, "objects must be 2x2 to 4x4 rectangles"
    return 3


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {
        (int(r), int(c))
        for r in range(int(r0), int(r0 + h))
        for c in range(int(c0), int(c0 + w))
    }


def _shadow_step_cells(
    r0: int, c0: int, h: int, w: int, direction: str, step: int
) -> list[tuple[int, int]]:
    if direction == "right":
        return [(int(r), int(c0 + w - 1 + step)) for r in range(int(r0), int(r0 + h))]
    if direction == "left":
        return [(int(r), int(c0 - step)) for r in range(int(r0), int(r0 + h))]
    if direction == "down":
        return [(int(r0 + h - 1 + step), int(c)) for c in range(int(c0), int(c0 + w))]
    assert direction == "up"
    return [(int(r0 - step), int(c)) for c in range(int(c0), int(c0 + w))]


def _shadow_cells(
    r0: int, c0: int, h: int, w: int, direction: str, length: int
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for step in range(1, int(length) + 1):
        cells.update(_shadow_step_cells(r0, c0, h, w, direction, step))
    return cells


def _footprint_cells(obj: dict, direction: str) -> set[tuple[int, int]]:
    cells = _rect_cells(obj["r0"], obj["c0"], obj["h"], obj["w"])
    cells.update(
        _shadow_cells(
            obj["r0"],
            obj["c0"],
            obj["h"],
            obj["w"],
            direction,
            obj["actual_len"],
        )
    )
    return cells


def _cells_fit(cells: set[tuple[int, int]], height: int, width: int) -> bool:
    for r, c in cells:
        if not (1 <= r < height - 1 and 1 <= c < width - 1):
            return False
    return True


def _can_place(
    cells: set[tuple[int, int]], reserved: np.ndarray, height: int, width: int
) -> bool:
    if not _cells_fit(cells, height, width):
        return False
    return all(not reserved[r, c] for r, c in cells)


def _mark_reserved(reserved: np.ndarray, cells: set[tuple[int, int]]) -> None:
    height, width = reserved.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < height and 0 <= cc < width:
                    reserved[rr, cc] = True


def _sample_dims(category: str) -> tuple[int, int]:
    if category == "small":
        return (2, 2)
    if category == "medium":
        return random.choice([(2, 3), (3, 2)])
    assert category == "large"
    return random.choice([(2, 4), (4, 2), (3, 3), (3, 4), (4, 3), (4, 4)])


def _sample_blocker_dims(direction: str, target_dims: tuple[int, int]) -> tuple[int, int]:
    h, w = target_dims
    if direction in ("right", "left"):
        return (h, random.randint(2, 4))
    return (random.randint(2, 4), w)


def _approx_footprint(spec: dict, direction: str) -> int:
    side = spec["h"] if direction in ("right", "left") else spec["w"]
    return int(spec["h"] * spec["w"] + spec["actual_len"] * side)


def _measure_shadow(g: np.ndarray, obj: dict, direction: str, grey: int) -> int:
    h, w = g.shape
    r0, c0, r1, c1 = obj["r0"], obj["c0"], obj["r1"], obj["c1"]
    length = 0
    if direction == "right":
        c = int(c1 + 1)
        while c < w and np.all(g[r0 : r1 + 1, c] == grey):
            length += 1
            c += 1
        return length
    if direction == "left":
        c = int(c0 - 1)
        while c >= 0 and np.all(g[r0 : r1 + 1, c] == grey):
            length += 1
            c -= 1
        return length
    if direction == "down":
        r = int(r1 + 1)
        while r < h and np.all(g[r, c0 : c1 + 1] == grey):
            length += 1
            r += 1
        return length
    assert direction == "up"
    r = int(r0 - 1)
    while r >= 0 and np.all(g[r, c0 : c1 + 1] == grey):
        length += 1
        r -= 1
    return length


def _extract_objects(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    objects = []
    for color in sorted(set(map(int, np.unique(g))) - {int(bg), int(grey)}):
        coords = np.argwhere(g == int(color))
        assert coords.size > 0, "object color must appear"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        assert area == int(coords.shape[0]), "each colored object must be a solid rectangle"
        objects.append(
            {
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
                "area": area,
            }
        )
    assert objects, "expected at least one colored rectangle"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the shared shadow direction from existing examples and add matching
    shadows to rectangles that do not have one yet.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    grey = SHADOW_COLOR
    objects = _extract_objects(g, bg, grey)

    detected: list[tuple[dict, str, int]] = []
    for obj in objects:
        hits = []
        for direction in DIRECTIONS:
            length = _measure_shadow(g, obj, direction, grey)
            if length > 0:
                hits.append((direction, length))
        assert len(hits) <= 1, "an object cannot have shadows on multiple sides"
        if hits:
            detected.append((obj, hits[0][0], hits[0][1]))

    assert detected, "expected at least one visible shadow"
    shadow_direction = detected[0][1]
    assert all(direction == shadow_direction for _obj, direction, _len in detected), (
        "all visible shadows must share one direction"
    )

    shadowed_colors = {obj["color"] for obj, _direction, _length in detected}
    out = g.copy()
    for obj in objects:
        if obj["color"] in shadowed_colors:
            continue
        full_len = _shadow_length_from_area(obj["area"])
        for step in range(1, full_len + 1):
            cells = _shadow_step_cells(
                obj["r0"], obj["c0"], obj["h"], obj["w"], shadow_direction, step
            )
            if any(
                not (0 <= r < out.shape[0] and 0 <= c < out.shape[1]) for r, c in cells
            ):
                break
            if any(out[r, c] not in (bg, grey) for r, c in cells):
                break
            for r, c in cells:
                out[r, c] = grey

    return out


def _place_single(
    spec: dict, direction: str, reserved: np.ndarray, height: int, width: int
) -> dict | None:
    for _ in range(PLACE_RETRIES):
        obj = spec.copy()
        obj["r0"] = random.randint(1, height - obj["h"] - 1)
        obj["c0"] = random.randint(1, width - obj["w"] - 1)
        cells = _footprint_cells(obj, direction)
        if not _can_place(cells, reserved, height, width):
            continue
        _mark_reserved(reserved, cells)
        return obj
    return None


def _place_blocked_pair(
    target_spec: dict,
    blocker_spec: dict,
    direction: str,
    reserved: np.ndarray,
    height: int,
    width: int,
) -> tuple[dict, dict] | None:
    for _ in range(PLACE_RETRIES):
        target = target_spec.copy()
        target["r0"] = random.randint(1, height - target["h"] - 1)
        target["c0"] = random.randint(1, width - target["w"] - 1)

        blocker = blocker_spec.copy()
        if direction == "right":
            blocker["r0"] = target["r0"]
            blocker["c0"] = target["c0"] + target["w"] + target["actual_len"]
        elif direction == "left":
            blocker["r0"] = target["r0"]
            blocker["c0"] = target["c0"] - target["actual_len"] - blocker["w"]
        elif direction == "down":
            blocker["r0"] = target["r0"] + target["h"] + target["actual_len"]
            blocker["c0"] = target["c0"]
        else:
            blocker["r0"] = target["r0"] - target["actual_len"] - blocker["h"]
            blocker["c0"] = target["c0"]

        target_cells = _footprint_cells(target, direction)
        blocker_cells = _footprint_cells(blocker, direction)
        combo = target_cells | blocker_cells
        if not _can_place(combo, reserved, height, width):
            continue
        _mark_reserved(reserved, combo)
        return target, blocker
    return None


def _render_scene(height: int, width: int, objects: list[dict], direction: str) -> np.ndarray:
    grid = np.full((height, width), DEFAULT_BG, dtype=int)
    for obj in objects:
        r0, c0 = obj["r0"], obj["c0"]
        grid[r0 : r0 + obj["h"], c0 : c0 + obj["w"]] = int(obj["color"])
    for obj in objects:
        if not obj["draw_shadow"]:
            continue
        for r, c in _shadow_cells(
            obj["r0"], obj["c0"], obj["h"], obj["w"], direction, obj["actual_len"]
        ):
            grid[r, c] = SHADOW_COLOR
    return grid


def _make_spec(
    category: str,
    color: int,
    draw_shadow: bool,
    *,
    actual_len: int | None = None,
    dims: tuple[int, int] | None = None,
) -> dict:
    h, w = dims if dims is not None else _sample_dims(category)
    area = int(h * w)
    return {
        "category": category,
        "color": int(color),
        "h": int(h),
        "w": int(w),
        "area": area,
        "actual_len": int(actual_len if actual_len is not None else _shadow_length_from_area(area)),
        "draw_shadow": bool(draw_shadow),
    }


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    direction = random.choice(DIRECTIONS)
    use_blocked = random.choice([False, True])
    extra_count = random.randint(0, 2)

    total_objects = 4 + extra_count + (1 if use_blocked else 0)
    colors = random.sample(OBJECT_COLORS, total_objects)
    color_i = 0

    exemplars = []
    for category in ("small", "medium", "large"):
        exemplars.append(_make_spec(category, colors[color_i], True))
        color_i += 1

    if use_blocked:
        target_category = random.choice(["medium", "large"])
    else:
        target_category = random.choice(["small", "medium", "large"])
    target_dims = _sample_dims(target_category)
    target_full = _shadow_length_from_area(target_dims[0] * target_dims[1])
    if use_blocked:
        target_len = random.randint(1, target_full - 1)
    else:
        target_len = target_full
    target = _make_spec(
        target_category,
        colors[color_i],
        False,
        actual_len=target_len,
        dims=target_dims,
    )
    color_i += 1

    blocker = None
    if use_blocked:
        blocker_dims = _sample_blocker_dims(direction, target_dims)
        blocker_area = blocker_dims[0] * blocker_dims[1]
        blocker_category = "small" if blocker_area == 4 else "medium" if blocker_area == 6 else "large"
        blocker = _make_spec(
            blocker_category,
            colors[color_i],
            random.choice([False, True]),
            dims=blocker_dims,
        )
        color_i += 1

    extras = []
    for _ in range(extra_count):
        extras.append(
            _make_spec(
                random.choice(["small", "medium", "large"]),
                colors[color_i],
                random.choice([False, True]),
            )
        )
        color_i += 1

    reserved = np.zeros((height, width), dtype=bool)
    placed: list[dict] = []

    if blocker is not None:
        pair = _place_blocked_pair(target, blocker, direction, reserved, height, width)
        if pair is None:
            return None
        placed.extend(pair)
    else:
        placed_target = _place_single(target, direction, reserved, height, width)
        if placed_target is None:
            return None
        placed.append(placed_target)

    singles = sorted(
        exemplars + extras,
        key=lambda spec: -_approx_footprint(spec, direction),
    )
    for spec in singles:
        obj = _place_single(spec, direction, reserved, height, width)
        if obj is None:
            return None
        placed.append(obj)

    grid = _render_scene(height, width, placed, direction)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out == SHADOW_COLOR)) <= int(np.count_nonzero(grid == SHADOW_COLOR)):
        return None

    exemplar_areas = sorted(
        obj["area"] for obj in placed if obj["draw_shadow"] and obj["actual_len"] in (1, 2, 3)
    )
    if not ({4, 6} <= set(exemplar_areas) and any(area >= 8 for area in exemplar_areas)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-completion puzzle")
