import numpy as np


OBJECT_COLORS = (5, 6)
PALETTE_COLORS = (1, 2, 3, 4, 7, 8, 9)
DIRS = ("up", "down", "left", "right")
MAX_GENERATE_ATTEMPTS = 300
OBJECT_PLACE_TRIES = 120
EXTRA_PALETTE_TRIES = 120


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _interior_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _parse_object(grid: np.ndarray, comp: dict[str, object], bg: int) -> dict[str, object] | None:
    coords = list(comp["coords"])
    color = int(comp["color"])
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    if int(mask.sum()) != h * w - 1:
        return None

    missing = np.argwhere(~mask)
    if missing.shape != (1, 2):
        return None
    mr, mc = map(int, missing[0])
    if int(grid[r0 + mr, c0 + mc]) != bg:
        return None

    direction = None
    if mr == 0 and 0 < mc < w - 1:
        direction = "up"
    elif mr == h - 1 and 0 < mc < w - 1:
        direction = "down"
    elif mc == 0 and 0 < mr < h - 1:
        direction = "left"
    elif mc == w - 1 and 0 < mr < h - 1:
        direction = "right"
    if direction is None:
        return None

    if direction == "up":
        dr, dc = -h, 0
    elif direction == "down":
        dr, dc = h, 0
    elif direction == "left":
        dr, dc = 0, -w
    else:
        dr, dc = 0, w

    return {
        "color": color,
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "size": (h, w),
        "direction": direction,
        "gap": (r0 + mr, c0 + mc),
        "shift": (dr, dc),
    }


def _boundary_palette(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    palette: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if (r, c) in seen:
                continue
            seen.add((r, c))
            value = int(grid[r, c])
            if value != bg:
                palette.append((r, c, value))
    return palette


def _palette_color_for_object(
    palette: list[tuple[int, int, int]],
    grid_shape: tuple[int, int],
    direction: str,
    gap: tuple[int, int],
    fallback: int,
) -> int:
    h, w = grid_shape
    gr, gc = gap
    if direction == "up":
        candidates = [(abs(c - gc), c, color) for r, c, color in palette if r == 0]
    elif direction == "down":
        candidates = [(abs(c - gc), c, color) for r, c, color in palette if r == h - 1]
    elif direction == "left":
        candidates = [(abs(r - gr), r, color) for r, c, color in palette if c == 0]
    else:
        candidates = [(abs(r - gr), r, color) for r, c, color in palette if c == w - 1]
    if not candidates:
        return fallback
    return int(min(candidates)[-1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each notched rectangle outward and recolor the pushed copy from the pointed border palette.

    Input structure:
    - The background is black, and the border contains several isolated one-pixel palette markers.
    - Inside the grid are one or more monochrome rectangular objects; each rectangle is solid except for exactly one missing boundary pixel.
    - The missing pixel lies on exactly one side of its rectangle and acts as the object's directional notch.

    Rule:
    - Separate border palette pixels from interior objects, then locate each object's bbox and its unique missing boundary pixel.
    - The side containing the missing pixel sets the direction: top means up, bottom means down, left means left, right means right.
    - Look toward that border and choose the nearest palette pixel on that border, measured in the same column for up/down or the same row for left/right.
    - Copy the original filled shape, translate it by one full object height or width in the notch direction, clip any out-of-bounds cells, and paint the translated copy with the chosen palette color.

    Key insight:
    - The notch is not repaired; it simultaneously encodes which way to translate and which border palette should determine the copy's color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    palette = _boundary_palette(g, bg)
    objects: list[dict[str, object]] = []
    for comp in _interior_components(g, bg):
        parsed = _parse_object(g, comp, bg)
        if parsed is not None:
            objects.append(parsed)

    out = g.copy()
    h, w = g.shape
    objects.sort(key=lambda obj: (obj["bbox"], int(obj["color"])))
    for obj in objects:
        copy_color = _palette_color_for_object(
            palette, g.shape, str(obj["direction"]), tuple(obj["gap"]), int(obj["color"])
        )
        dr, dc = map(int, obj["shift"])
        for r, c in obj["coords"]:
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = copy_color
    return out


def _bbox_mask(h: int, w: int, r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(h - 1, r1)
    cc1 = min(w - 1, c1)
    if rr0 <= rr1 and cc0 <= cc1:
        mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return mask


def _expand_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    h, w = mask.shape
    expanded = mask.copy()
    for r, c in np.argwhere(mask):
        r0 = max(0, int(r) - radius)
        r1 = min(h - 1, int(r) + radius)
        c0 = max(0, int(c) - radius)
        c1 = min(w - 1, int(c) + radius)
        expanded[r0 : r1 + 1, c0 : c1 + 1] = True
    return expanded


def _projected_palette_pos(obj: dict[str, object], grid_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = grid_shape
    gr, gc = map(int, obj["gap"])
    direction = str(obj["direction"])
    if direction == "up":
        return (0, gc)
    if direction == "down":
        return (h - 1, gc)
    if direction == "left":
        return (gr, 0)
    return (gr, w - 1)


def _reserve_mask_for_object(obj: dict[str, object], grid_shape: tuple[int, int]) -> np.ndarray:
    h, w = grid_shape
    r0, c0, r1, c1 = map(int, obj["bbox"])
    mask = _bbox_mask(h, w, r0, c0, r1, c1)
    dr, dc = map(int, obj["shift"])
    mask |= _bbox_mask(h, w, r0 + dr, c0 + dc, r1 + dr, c1 + dc)
    return _expand_mask(mask, radius=1)


def _palette_is_clear(pos: tuple[int, int], used: list[tuple[int, int]]) -> bool:
    pr, pc = pos
    for ur, uc in used:
        if max(abs(pr - ur), abs(pc - uc)) <= 1:
            return False
    return True


def _sample_dimensions(rng: np.random.Generator, direction: str) -> tuple[int, int]:
    if direction in ("up", "down"):
        height = int(rng.integers(2, 7))
        width = int(rng.integers(3, 7))
    else:
        height = int(rng.integers(3, 7))
        width = int(rng.integers(2, 7))
    return height, width


def _sample_bbox(
    rng: np.random.Generator, grid_shape: tuple[int, int], direction: str, truncated: bool, height: int, width: int
) -> tuple[int, int] | None:
    h, w = grid_shape
    r0_min, r0_max = 1, h - height - 1
    c0_min, c0_max = 1, w - width - 1
    if r0_min > r0_max or c0_min > c0_max:
        return None

    if direction == "up":
        if truncated:
            r0_max = min(r0_max, height - 1)
        else:
            r0_min = max(r0_min, height)
    elif direction == "down":
        if truncated:
            r0_min = max(r0_min, h - 2 * height + 1)
        else:
            r0_max = min(r0_max, h - 2 * height)
    elif direction == "left":
        if truncated:
            c0_max = min(c0_max, width - 1)
        else:
            c0_min = max(c0_min, width)
    else:
        if truncated:
            c0_min = max(c0_min, w - 2 * width + 1)
        else:
            c0_max = min(c0_max, w - 2 * width)

    if r0_min > r0_max or c0_min > c0_max:
        return None
    r0 = int(rng.integers(r0_min, r0_max + 1))
    c0 = int(rng.integers(c0_min, c0_max + 1))
    return r0, c0


def _make_object(
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
    direction: str,
    truncated: bool,
    color: int,
    occupied_output: np.ndarray,
    used_palette: list[tuple[int, int]],
) -> dict[str, object] | None:
    for _ in range(OBJECT_PLACE_TRIES):
        height, width = _sample_dimensions(rng, direction)
        bbox_pos = _sample_bbox(rng, grid_shape, direction, truncated, height, width)
        if bbox_pos is None:
            continue
        r0, c0 = bbox_pos
        if direction == "up":
            gap = (r0, c0 + int(rng.integers(1, width - 1)))
            shift = (-height, 0)
        elif direction == "down":
            gap = (r0 + height - 1, c0 + int(rng.integers(1, width - 1)))
            shift = (height, 0)
        elif direction == "left":
            gap = (r0 + int(rng.integers(1, height - 1)), c0)
            shift = (0, -width)
        else:
            gap = (r0 + int(rng.integers(1, height - 1)), c0 + width - 1)
            shift = (0, width)

        coords = [(rr, cc) for rr in range(r0, r0 + height) for cc in range(c0, c0 + width) if (rr, cc) != gap]
        obj = {
            "color": color,
            "coords": coords,
            "bbox": (r0, c0, r0 + height - 1, c0 + width - 1),
            "size": (height, width),
            "direction": direction,
            "gap": gap,
            "shift": shift,
        }
        palette_pos = _projected_palette_pos(obj, grid_shape)
        if occupied_output[palette_pos]:
            continue
        if not _palette_is_clear(palette_pos, used_palette):
            continue
        reserve = _reserve_mask_for_object(obj, grid_shape)
        if np.any(reserve & occupied_output):
            continue
        obj["palette_pos"] = palette_pos
        obj["truncated"] = truncated
        obj["reserve_mask"] = reserve
        return obj
    return None


def _paint_object(grid: np.ndarray, obj: dict[str, object]) -> None:
    color = int(obj["color"])
    for r, c in obj["coords"]:
        grid[int(r), int(c)] = color


def _expected_output(inp: np.ndarray, objects: list[dict[str, object]], palette_colors: dict[tuple[int, int], int]) -> np.ndarray:
    out = inp.copy()
    h, w = out.shape
    for obj in objects:
        dr, dc = map(int, obj["shift"])
        palette_color = int(palette_colors[tuple(obj["palette_pos"])])
        for r, c in obj["coords"]:
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = palette_color
    return out


def _free_boundary_positions(
    grid_shape: tuple[int, int], occupied_output: np.ndarray, used_palette: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    h, w = grid_shape
    positions: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        pos = (0, c)
        if not occupied_output[pos] and _palette_is_clear(pos, used_palette):
            positions.append(pos)
    for c in range(1, w - 1):
        pos = (h - 1, c)
        if not occupied_output[pos] and _palette_is_clear(pos, used_palette):
            positions.append(pos)
    for r in range(1, h - 1):
        pos = (r, 0)
        if not occupied_output[pos] and _palette_is_clear(pos, used_palette):
            positions.append(pos)
    for r in range(1, h - 1):
        pos = (r, w - 1)
        if not occupied_output[pos] and _palette_is_clear(pos, used_palette):
            positions.append(pos)
    return positions


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = int(rng.integers(12, 21))
    grid_w = int(rng.integers(12, 21))
    grid_shape = (grid_h, grid_w)
    object_count = int(rng.integers(2, 5))

    directions = list(DIRS)
    rng.shuffle(directions)
    directions = directions[:object_count]

    truncated_flags = [bool(rng.integers(0, 2)) for _ in range(object_count)]
    if object_count >= 2 and all(flag == truncated_flags[0] for flag in truncated_flags):
        truncated_flags[int(rng.integers(0, object_count))] = not truncated_flags[0]

    object_colors = [OBJECT_COLORS[int(rng.integers(0, len(OBJECT_COLORS)))] for _ in range(object_count)]

    occupied_output = np.zeros(grid_shape, dtype=bool)
    used_palette: list[tuple[int, int]] = []
    objects: list[dict[str, object]] = []
    for direction, truncated, color in zip(directions, truncated_flags, object_colors):
        obj = _make_object(rng, grid_shape, direction, truncated, color, occupied_output, used_palette)
        if obj is None:
            return None
        occupied_output |= obj["reserve_mask"]
        used_palette.append(tuple(obj["palette_pos"]))
        objects.append(obj)

    remaining_palette_colors = [color for color in PALETTE_COLORS if color not in object_colors]
    rng.shuffle(remaining_palette_colors)
    min_extra = max(0, 3 - object_count)
    max_extra = min(2, len(remaining_palette_colors) - object_count)
    if max_extra < min_extra:
        return None
    extra_count = int(rng.integers(min_extra, max_extra + 1))

    needed_palette_colors = remaining_palette_colors[: object_count + extra_count]
    if len(needed_palette_colors) != object_count + extra_count:
        return None

    palette_positions = [tuple(obj["palette_pos"]) for obj in objects]
    free_positions = _free_boundary_positions(grid_shape, occupied_output, used_palette)
    rng.shuffle(free_positions)
    extras: list[tuple[int, int]] = []
    for pos in free_positions:
        if len(extras) >= extra_count:
            break
        if not _palette_is_clear(pos, used_palette):
            continue
        extras.append(pos)
        used_palette.append(pos)
    if len(extras) != extra_count:
        return None

    palette_assignments: dict[tuple[int, int], int] = {}
    for obj, color in zip(objects, needed_palette_colors[:object_count]):
        palette_assignments[tuple(obj["palette_pos"])] = int(color)
        obj["copy_color"] = int(color)
    for pos, color in zip(extras, needed_palette_colors[object_count:]):
        palette_assignments[pos] = int(color)

    grid = np.zeros(grid_shape, dtype=int)
    for obj in objects:
        _paint_object(grid, obj)
    for pos, color in palette_assignments.items():
        grid[pos] = int(color)

    expected = _expected_output(grid, objects, palette_assignments)

    # Keep generated scenes readable and ensure the solver sees exactly the intended objects.
    if len(_interior_components(grid, _bg_color(grid))) != len(objects):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        result = _gen_attempt(np.random.default_rng())
        if result is not None:
            return result[0]
    raise RuntimeError(f"generate_input failed after {MAX_GENERATE_ATTEMPTS} attempts")
