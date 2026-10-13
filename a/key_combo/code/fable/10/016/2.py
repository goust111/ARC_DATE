from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

NON_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _fill_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _side_anchor(bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if direction == "left":
        return (r0 + r1) // 2, c0
    if direction == "right":
        return (r0 + r1) // 2, c1
    if direction == "up":
        return r0, (c0 + c1) // 2
    return r1, (c0 + c1) // 2


def _bbox_from_anchor(anchor: tuple[int, int], height: int, width: int, direction: str) -> tuple[int, int, int, int]:
    ar, ac = anchor
    if direction == "right":
        r0 = ar - height // 2
        c0 = ac - (width - 1)
    elif direction == "left":
        r0 = ar - height // 2
        c0 = ac
    elif direction == "down":
        r0 = ar - (height - 1)
        c0 = ac - width // 2
    else:
        r0 = ar
        c0 = ac - width // 2
    return r0, c0, r0 + height - 1, c0 + width - 1


def _next_bbox(current: tuple[int, int, int, int], direction: str) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = current
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    new_height = height - 2
    new_width = width - 2
    if new_height <= 0 or new_width <= 0:
        return None
    ar, ac = _side_anchor(current, direction)
    dr, dc = DIR_VECTORS[direction]
    new_anchor = (ar + dr, ac + dc)
    return _bbox_from_anchor(new_anchor, new_height, new_width, direction)


def _paint_trajectory(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: str,
    color: int,
    bg: int,
) -> None:
    current = bbox
    while current is not None:
        r0, c0, r1, c1 = current
        assert 0 <= r0 <= r1 < out.shape[0] and 0 <= c0 <= c1 < out.shape[1], "trajectory must stay in bounds"
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        clash = (patch != bg) & (patch != color)
        assert not np.any(clash), "different objects must not paint conflicting output cells"
        patch[:, :] = int(color)
        current = _next_bbox(current, direction)


def _parse_object(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], str, int, tuple[int, int]]:
    values = [int(grid[r, c]) for r, c in coords]
    counts = Counter(values)
    assert len(counts) == 2, "each component must contain a rectangle color and one tab color"
    tab_colors = [color for color, count in counts.items() if count == 1]
    assert len(tab_colors) == 1, "expected exactly one tab pixel"
    tab_color = tab_colors[0]
    rect_colors = [color for color in counts if color != tab_color]
    assert len(rect_colors) == 1
    rect_color = rect_colors[0]

    rect_coords = np.array([(int(r), int(c)) for r, c in coords if int(grid[r, c]) == rect_color], dtype=int)
    tab = next((int(r), int(c)) for r, c in coords if int(grid[r, c]) == tab_color)
    r0, c0, r1, c1 = _bbox(rect_coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    assert counts[rect_color] == height * width, "rectangle body must be solid"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == rect_color), "rectangle bbox must be fully filled"

    matches = []
    if height % 2 == 1 and tab == (r0 + height // 2, c0 - 1):
        matches.append("left")
    if height % 2 == 1 and tab == (r0 + height // 2, c1 + 1):
        matches.append("right")
    if width % 2 == 1 and tab == (r0 - 1, c0 + width // 2):
        matches.append("up")
    if width % 2 == 1 and tab == (r1 + 1, c0 + width // 2):
        matches.append("down")
    assert len(matches) == 1, "tab must sit one cell outside the center of exactly one side"

    return (r0, c0, r1, c1), matches[0], rect_color, tab


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each tabbed solid rectangle stepwise in the tab direction while peeling one outer layer per step.

    Input structure:
    - The grid has a black background with several separated solid single-color rectangles.
    - Each rectangle has exactly one differently colored one-pixel tab attached just outside the center of one side.
    - The tab indicates one of the four cardinal directions.

    Rule:
    - Erase the tab pixels and keep each original rectangle.
    - For each rectangle, repeatedly peel away one layer from all four sides, then place the smaller rectangle one step farther in the tab direction while preserving the same tab-facing side-center as the moving anchor.
    - Paint every stage in the rectangle's own color until the shape shrinks away.

    Key insight:
    - The tab does not encode a destination or collision stop: it selects the motion axis, while the rectangle's own thickness controls how many shrinking translated copies can exist.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    parts = _components(g != bg)
    assert parts, "expected at least one foreground component"

    out = np.full_like(g, bg)
    for coords in parts:
        bbox, direction, rect_color, _tab = _parse_object(g, coords, bg)
        _paint_trajectory(out, bbox, direction, rect_color, bg)
    return out


def _rect_coords_from_bbox(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _tab_coord(bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if direction == "left":
        return (r0 + r1) // 2, c0 - 1
    if direction == "right":
        return (r0 + r1) // 2, c1 + 1
    if direction == "up":
        return r0 - 1, (c0 + c1) // 2
    return r1 + 1, (c0 + c1) // 2


def _make_object(direction: str, height: int, width: int, rect_color: int, tab_color: int) -> dict:
    bbox = (0, 0, height - 1, width - 1)
    rect_coords = _rect_coords_from_bbox(bbox)
    tab = _tab_coord(bbox, direction)
    out_coords = set(rect_coords)
    steps = 0
    current = bbox
    while True:
        current = _next_bbox(current, direction)
        if current is None:
            break
        steps += 1
        out_coords.update(_rect_coords_from_bbox(current))

    all_coords = set(rect_coords)
    all_coords.add(tab)
    all_coords.update(out_coords)
    min_r = min(r for r, _ in all_coords)
    min_c = min(c for _, c in all_coords)
    shift_r = -min_r
    shift_c = -min_c

    def shift(coord: tuple[int, int]) -> tuple[int, int]:
        return coord[0] + shift_r, coord[1] + shift_c

    local_input = [shift(rc) for rc in rect_coords]
    local_tab = shift(tab)
    local_out = [shift(rc) for rc in sorted(out_coords)]
    local_union = sorted({*local_input, local_tab, *local_out})
    max_r = max(r for r, _ in local_union)
    max_c = max(c for _, c in local_union)
    return {
        "direction": direction,
        "rect_color": int(rect_color),
        "tab_color": int(tab_color),
        "input_coords": local_input,
        "tab_coord": local_tab,
        "output_coords": local_out,
        "union_coords": local_union,
        "shape": (max_r + 1, max_c + 1),
        "steps": steps,
        "dims": (height, width),
    }


def _halo(coords: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _translate_coords(coords: list[tuple[int, int]], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in coords]


def _paint_points(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _render_from_specs(shape: tuple[int, int], placed: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros(shape, dtype=int)
    out = np.zeros(shape, dtype=int)
    for spec in placed:
        _paint_points(inp, spec["input_coords"], spec["rect_color"])
        tr, tc = spec["tab_coord"]
        inp[tr, tc] = spec["tab_color"]
        _paint_points(out, spec["output_coords"], spec["rect_color"])
    return inp, out


def _choose_directions(count: int) -> list[str]:
    if count == 4:
        dirs = ["up", "down", "left", "right"]
        random.shuffle(dirs)
        return dirs
    if count == 3:
        while True:
            dirs = random.sample(["up", "down", "left", "right"], 3)
            if any(d in {"left", "right"} for d in dirs) and any(d in {"up", "down"} for d in dirs):
                return dirs
    dirs = [random.choice(["left", "right"]), random.choice(["up", "down"])]
    random.shuffle(dirs)
    return dirs


def _choose_dims(direction: str, role: str) -> tuple[int, int]:
    if direction in {"left", "right"}:
        if role == "deep":
            return random.choice([5, 7]), random.randint(5, 8)
        if role == "shallow":
            return 3, random.randint(5, 8)
        return random.choice([3, 5, 7]), random.randint(3, 8)
    if role == "deep":
        return random.randint(5, 8), random.choice([5, 7])
    if role == "shallow":
        return random.randint(5, 8), 3
    return random.randint(3, 8), random.choice([3, 5, 7])


def _specs_for_attempt() -> list[dict]:
    count = random.choices([2, 3, 4], weights=[2, 3, 4])[0]
    directions = _choose_directions(count)
    roles = ["deep", "shallow"] + ["mixed"] * max(0, count - 2)
    random.shuffle(roles)
    colors = random.sample(NON_BLACK, 2 * count)

    specs = []
    for i, direction in enumerate(directions):
        rect_color = colors[2 * i]
        tab_color = colors[2 * i + 1]
        height, width = _choose_dims(direction, roles[i])
        specs.append(_make_object(direction, height, width, rect_color, tab_color))
    specs.sort(key=lambda spec: len(spec["union_coords"]), reverse=True)
    return specs


def _place_specs(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    h, w = shape
    blocked: set[tuple[int, int]] = set()
    placed: list[dict] = []
    for spec in specs:
        sh, sw = spec["shape"]
        candidates = []
        for top in range(h - sh + 1):
            for left in range(w - sw + 1):
                union = _translate_coords(spec["union_coords"], top, left)
                if any(cell in blocked for cell in union):
                    continue
                candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        entry = {
            **spec,
            "input_coords": _translate_coords(spec["input_coords"], top, left),
            "tab_coord": (spec["tab_coord"][0] + top, spec["tab_coord"][1] + left),
            "output_coords": _translate_coords(spec["output_coords"], top, left),
            "union_coords": _translate_coords(spec["union_coords"], top, left),
        }
        blocked |= _halo(entry["union_coords"], shape)
        placed.append(entry)
    return placed


def _semantic_checks(inp: np.ndarray, out: np.ndarray, specs: list[dict]) -> bool:
    try:
        solved = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(solved, out):
        return False

    parsed = []
    for coords in _components(inp != 0):
        try:
            bbox, direction, color, tab = _parse_object(inp, coords, 0)
        except AssertionError:
            return False
        parsed.append((bbox, direction, color, tab))
    if len(parsed) != len(specs):
        return False

    without_tabs = inp.copy()
    for _, _, _, tab in parsed:
        without_tabs[tab] = 0
    if np.array_equal(without_tabs, out):
        return False

    if not any(spec["steps"] >= 2 for spec in specs):
        return False
    if not any(spec["steps"] == 1 for spec in specs):
        return False
    dirs = {spec["direction"] for spec in specs}
    if len(specs) >= 2 and not (dirs & {"left", "right"} and dirs & {"up", "down"}):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    specs = _specs_for_attempt()
    rows = random.randint(20, 30)
    cols = random.randint(20, 30)
    placed = _place_specs((rows, cols), specs)
    if placed is None:
        return None
    inp, out = _render_from_specs((rows, cols), placed)
    if not _semantic_checks(inp, out, placed):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tabbed-rectangle sliding input")
