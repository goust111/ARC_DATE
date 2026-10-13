import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
STEP_BY_DIR = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

BG = 0
ARROW_COLOR = 5
TARGET_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (18, 27)
ARROW_COUNT_RANGE = (1, 5)
ARROW_BODY_LENGTHS = (3, 4, 5)
ARROW_THICKNESSES = (2, 3)
TARGET_SIZES = (2, 3)

ARROW_PLACE_RETRIES = 120
TARGET_PLACE_RETRIES = 160
GENERATE_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def normalize_cells(cells: set[tuple[int, int]], anchor: tuple[int, int] | None = None):
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    norm = frozenset((r - min_r, c - min_c) for r, c in cells)
    if anchor is None:
        return norm
    return norm, (int(anchor[0] - min_r), int(anchor[1] - min_c))


def build_arrow_shape(
    direction: str, body_length: int, thickness: int, variant: int = 0
) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    cells: set[tuple[int, int]] = set()

    if direction in {"left", "right"}:
        for r in range(thickness):
            for c in range(body_length):
                cells.add((r, c))

        if direction == "right":
            if thickness == 3:
                tip = {(0, body_length), (1, body_length), (2, body_length), (1, body_length + 1)}
                apex = (1, body_length + 1)
            elif variant == 0:
                tip = {(0, body_length), (1, body_length), (1, body_length + 1)}
                apex = (1, body_length + 1)
            else:
                tip = {(0, body_length), (0, body_length + 1), (1, body_length)}
                apex = (0, body_length + 1)
        else:
            if thickness == 3:
                tip = {(0, -1), (1, -1), (2, -1), (1, -2)}
                apex = (1, -2)
            elif variant == 0:
                tip = {(0, -1), (1, -1), (1, -2)}
                apex = (1, -2)
            else:
                tip = {(0, -1), (0, -2), (1, -1)}
                apex = (0, -2)
    else:
        for r in range(body_length):
            for c in range(thickness):
                cells.add((r, c))

        if direction == "down":
            if thickness == 3:
                tip = {(body_length, 0), (body_length, 1), (body_length, 2), (body_length + 1, 1)}
                apex = (body_length + 1, 1)
            elif variant == 0:
                tip = {(body_length, 0), (body_length, 1), (body_length + 1, 1)}
                apex = (body_length + 1, 1)
            else:
                tip = {(body_length, 0), (body_length, 1), (body_length + 1, 0)}
                apex = (body_length + 1, 0)
        else:
            if thickness == 3:
                tip = {(-1, 0), (-1, 1), (-1, 2), (-2, 1)}
                apex = (-2, 1)
            elif variant == 0:
                tip = {(-1, 0), (-1, 1), (-2, 1)}
                apex = (-2, 1)
            else:
                tip = {(-1, 0), (-1, 1), (-2, 0)}
                apex = (-2, 0)

    cells |= tip
    return normalize_cells(cells, apex)


ARROW_LIBRARY: list[dict[str, object]] = []
for _direction in STEP_BY_DIR:
    for _body_length in ARROW_BODY_LENGTHS:
        for _thickness in ARROW_THICKNESSES:
            n_variants = 1 if _thickness == 3 else 2
            for _variant in range(n_variants):
                _cells, _apex = build_arrow_shape(_direction, _body_length, _thickness, _variant)
                rows = [r for r, _c in _cells]
                cols = [c for _r, c in _cells]
                ARROW_LIBRARY.append(
                    {
                        "direction": _direction,
                        "body_length": _body_length,
                        "thickness": _thickness,
                        "variant": _variant,
                        "cells": _cells,
                        "apex": _apex,
                        "height": max(rows) + 1,
                        "width": max(cols) + 1,
                    }
                )


def match_arrow(coords: np.ndarray) -> dict[str, object] | None:
    cells = {tuple(map(int, rc)) for rc in coords}
    norm, apex = normalize_cells(cells, (0, 0))
    for spec in ARROW_LIBRARY:
        if norm != spec["cells"]:
            continue
        return {
            "direction": spec["direction"],
            "body_length": spec["body_length"],
            "thickness": spec["thickness"],
            "variant": spec["variant"],
            "apex_offset": spec["apex"],
        }
    return None


def find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "cells": frozenset(coords),
                    "area": int(len(coords)),
                    "bbox": (
                        int(arr[:, 0].min()),
                        int(arr[:, 1].min()),
                        int(arr[:, 0].max()),
                        int(arr[:, 1].max()),
                    ),
                }
            )
    return comps


def parse_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    arrows: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []

    for comp in find_components(g, bg):
        r0, c0, r1, c1 = comp["bbox"]
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        area = int(comp["area"])

        if height == width and height in TARGET_SIZES and area == height * width:
            targets.append(comp)
            continue

        arrow_info = match_arrow(comp["coords"] - np.array([[r0, c0]], dtype=int))
        assert arrow_info is not None, f"unrecognized component at bbox {comp['bbox']}"
        apex_offset = arrow_info["apex_offset"]
        arrows.append(
            {
                **comp,
                **arrow_info,
                "apex": (int(r0 + apex_offset[0]), int(c0 + apex_offset[1])),
            }
        )

    assert arrows, "expected at least one arrow"
    assert targets, "expected at least one target block"
    arrows.sort(key=lambda item: (item["apex"][0], item["apex"][1], item["direction"]))
    targets.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return bg, arrows, targets


def first_target_on_ray(
    shape: tuple[int, int], apex: tuple[int, int], direction: str, cell_to_target: dict[tuple[int, int], int]
) -> int | None:
    h, w = shape
    dr, dc = STEP_BY_DIR[direction]
    r, c = int(apex[0]), int(apex[1])
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        hit = cell_to_target.get((r, c))
        if hit is not None:
            return int(hit)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each arrow with the color of the first target block on its forward ray,
    then erase the consumed target block.
    """
    g = np.asarray(grid, dtype=int)
    bg, arrows, targets = parse_scene(g)

    cell_to_target: dict[tuple[int, int], int] = {}
    for idx, target in enumerate(targets):
        for cell in target["cells"]:
            cell_to_target[cell] = idx

    chosen: dict[frozenset[tuple[int, int]], int] = {}
    consumed: set[int] = set()
    for arrow in arrows:
        hit = first_target_on_ray(g.shape, arrow["apex"], arrow["direction"], cell_to_target)
        if hit is None:
            continue
        chosen[arrow["cells"]] = int(targets[hit]["color"])
        consumed.add(hit)

    out = g.copy()
    for hit in consumed:
        coords = targets[hit]["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
    for arrow in arrows:
        new_color = chosen.get(arrow["cells"])
        if new_color is None:
            continue
        coords = arrow["coords"]
        out[coords[:, 0], coords[:, 1]] = int(new_color)
    return out


def translate_cells(cells: frozenset[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(r + dr), int(c + dc)) for r, c in cells}


def expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int], margin: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - margin, r + margin + 1):
            for cc in range(c - margin, c + margin + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((int(rr), int(cc)))
    return out


def can_place(cells: set[tuple[int, int]], shape: tuple[int, int], forbidden: set[tuple[int, int]]) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in forbidden:
            return False
    return True


def steps_to_edge(apex: tuple[int, int], direction: str, shape: tuple[int, int]) -> int:
    h, w = shape
    r, c = apex
    if direction == "up":
        return int(r)
    if direction == "down":
        return int(h - 1 - r)
    if direction == "left":
        return int(c)
    return int(w - 1 - c)


def block_ray_distance(
    apex: tuple[int, int], direction: str, cells: set[tuple[int, int]]
) -> int | None:
    ar, ac = apex
    if direction == "right":
        cols = [c for r, c in cells if r == ar and c > ac]
        return None if not cols else int(min(cols) - ac)
    if direction == "left":
        cols = [c for r, c in cells if r == ar and c < ac]
        return None if not cols else int(ac - max(cols))
    if direction == "down":
        rows = [r for r, c in cells if c == ac and r > ar]
        return None if not rows else int(min(rows) - ar)
    rows = [r for r, c in cells if c == ac and r < ar]
    return None if not rows else int(ar - max(rows))


def build_target_cells(
    apex: tuple[int, int], direction: str, size: int, distance: int, offset: int
) -> set[tuple[int, int]]:
    ar, ac = apex
    if direction == "right":
        top = ar - offset
        left = ac + distance
    elif direction == "left":
        top = ar - offset
        left = ac - distance - size + 1
    elif direction == "down":
        top = ar + distance
        left = ac - offset
    else:
        top = ar - distance - size + 1
        left = ac - offset
    return {
        (int(top + dr), int(left + dc))
        for dr in range(size)
        for dc in range(size)
    }


def choose_arrow_spec(rng: np.random.Generator) -> dict[str, object]:
    spec = ARROW_LIBRARY[int(rng.integers(0, len(ARROW_LIBRARY)))]
    return {
        "direction": spec["direction"],
        "body_length": spec["body_length"],
        "thickness": spec["thickness"],
        "variant": spec["variant"],
        "shape_cells": spec["cells"],
        "shape_apex": spec["apex"],
        "height": spec["height"],
        "width": spec["width"],
    }


def place_arrow(
    rng: np.random.Generator,
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
    wants_hit: bool,
) -> dict[str, object] | None:
    h, w = shape
    spec = choose_arrow_spec(rng)
    min_room = 5 if wants_hit else 1

    for _ in range(ARROW_PLACE_RETRIES):
        top = int(rng.integers(0, h - int(spec["height"]) + 1))
        left = int(rng.integers(0, w - int(spec["width"]) + 1))
        cells = translate_cells(spec["shape_cells"], top, left)
        if not can_place(cells, shape, forbidden):
            continue
        apex = (int(top + spec["shape_apex"][0]), int(left + spec["shape_apex"][1]))
        if steps_to_edge(apex, spec["direction"], shape) < min_room:
            continue
        return {
            "direction": spec["direction"],
            "cells": cells,
            "apex": apex,
        }
    return None


def intersected_arrow_ids(
    block_cells: set[tuple[int, int]], arrows: list[dict[str, object]]
) -> list[tuple[int, int]]:
    hits: list[tuple[int, int]] = []
    for idx, arrow in enumerate(arrows):
        dist = block_ray_distance(arrow["apex"], arrow["direction"], block_cells)
        if dist is not None:
            hits.append((idx, dist))
    return hits


def place_target(
    rng: np.random.Generator,
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
    arrows: list[dict[str, object]],
    owner_idx: int | None,
    size: int,
    min_distance: int,
    max_distance: int,
) -> set[tuple[int, int]] | None:
    if min_distance > max_distance:
        return None

    for _ in range(TARGET_PLACE_RETRIES):
        if owner_idx is None:
            top = int(rng.integers(0, shape[0] - size + 1))
            left = int(rng.integers(0, shape[1] - size + 1))
            cells = {
                (int(top + dr), int(left + dc))
                for dr in range(size)
                for dc in range(size)
            }
        else:
            arrow = arrows[owner_idx]
            distance = int(rng.integers(min_distance, max_distance + 1))
            offset = int(rng.integers(0, size))
            cells = build_target_cells(arrow["apex"], arrow["direction"], size, distance, offset)

        if not can_place(cells, shape, forbidden):
            continue

        ray_hits = intersected_arrow_ids(cells, arrows)
        if owner_idx is None:
            if ray_hits:
                continue
        else:
            if sorted(idx for idx, _dist in ray_hits) != [owner_idx]:
                continue
        return cells
    return None


def write_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def validate_generated(
    grid: np.ndarray,
    expected_hit_color: dict[frozenset[tuple[int, int]], int | None],
) -> bool:
    try:
        _bg, arrows, targets = parse_scene(grid)
    except AssertionError:
        return False

    if len(arrows) != len(expected_hit_color):
        return False

    cell_to_target: dict[tuple[int, int], int] = {}
    for idx, target in enumerate(targets):
        for cell in target["cells"]:
            cell_to_target[cell] = idx

    for arrow in arrows:
        expected = expected_hit_color.get(arrow["cells"])
        if expected is None and arrow["cells"] not in expected_hit_color:
            return False
        hit = first_target_on_ray(grid.shape, arrow["apex"], arrow["direction"], cell_to_target)
        found = None if hit is None else int(targets[hit]["color"])
        if found != expected:
            return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    return isinstance(out, np.ndarray) and out.shape == grid.shape


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    shape = (h, w)
    grid = np.full(shape, BG, dtype=int)

    n_arrows = int(rng.integers(*ARROW_COUNT_RANGE))
    n_hits = int(rng.integers(0, n_arrows + 1))
    hit_flags = np.array([True] * n_hits + [False] * (n_arrows - n_hits), dtype=bool)
    rng.shuffle(hit_flags)

    arrows: list[dict[str, object]] = []
    forbidden: set[tuple[int, int]] = set()
    for wants_hit in hit_flags:
        arrow = place_arrow(rng, shape, forbidden, bool(wants_hit))
        if arrow is None:
            return None
        arrows.append(arrow)
        forbidden |= expand_cells(arrow["cells"], shape, margin=1)
        write_cells(grid, arrow["cells"], ARROW_COLOR)

    color_pool = list(TARGET_COLORS)
    rng.shuffle(color_pool)
    expected_hit_color: dict[frozenset[tuple[int, int]], int | None] = {
        frozenset(arrow["cells"]): None for arrow in arrows
    }

    first_hit_distance: dict[int, int] = {}
    target_count = 0

    hit_indices = [idx for idx, flag in enumerate(hit_flags) if flag]
    for idx in hit_indices:
        if not color_pool:
            return None
        arrow = arrows[idx]
        room = steps_to_edge(arrow["apex"], arrow["direction"], shape)
        size = int(rng.choice(TARGET_SIZES))
        max_distance = room - size + 1
        cells = place_target(
            rng,
            shape,
            forbidden,
            arrows,
            owner_idx=idx,
            size=size,
            min_distance=2,
            max_distance=max_distance,
        )
        if cells is None:
            return None
        color = color_pool.pop()
        write_cells(grid, cells, color)
        forbidden |= expand_cells(cells, shape, margin=1)
        expected_hit_color[frozenset(arrow["cells"])] = color
        first_hit_distance[idx] = int(block_ray_distance(arrow["apex"], arrow["direction"], cells))
        target_count += 1

    extra_same_ray_candidates = hit_indices.copy()
    rng.shuffle(extra_same_ray_candidates)
    for idx in extra_same_ray_candidates:
        if not color_pool or rng.random() >= 0.45:
            continue
        arrow = arrows[idx]
        room = steps_to_edge(arrow["apex"], arrow["direction"], shape)
        size = int(rng.choice(TARGET_SIZES))
        min_distance = first_hit_distance[idx] + size + 1
        max_distance = room - size + 1
        cells = place_target(
            rng,
            shape,
            forbidden,
            arrows,
            owner_idx=idx,
            size=size,
            min_distance=min_distance,
            max_distance=max_distance,
        )
        if cells is None:
            continue
        color = color_pool.pop()
        write_cells(grid, cells, color)
        forbidden |= expand_cells(cells, shape, margin=1)
        target_count += 1

    distractor_cap = min(3, len(color_pool))
    distractor_count = int(rng.integers(0, distractor_cap + 1))
    for _ in range(distractor_count):
        if not color_pool:
            break
        size = int(rng.choice(TARGET_SIZES))
        cells = place_target(
            rng,
            shape,
            forbidden,
            arrows,
            owner_idx=None,
            size=size,
            min_distance=0,
            max_distance=0,
        )
        if cells is None:
            continue
        color = color_pool.pop()
        write_cells(grid, cells, color)
        forbidden |= expand_cells(cells, shape, margin=1)
        target_count += 1

    if target_count == 0:
        return None
    if not validate_generated(grid, expected_hit_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATE_ATTEMPTS} attempts")
