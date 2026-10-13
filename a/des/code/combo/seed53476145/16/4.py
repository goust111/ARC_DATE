from __future__ import annotations

import random

import numpy as np


CARDINALS = ("top", "right", "bottom", "left")
FOUR_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# The missing corner of the 2x2 L sprite encodes a cardinal direction.
# Rotating the patch clockwise advances the direction clockwise as well.
MISSING_BY_DIRECTION = {
    "top": (0, 0),
    "right": (0, 1),
    "bottom": (1, 1),
    "left": (1, 0),
}
DIR_BY_MISSING = {value: key for key, value in MISSING_BY_DIRECTION.items()}


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(counts.argmax())])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in FOUR_NEIGHBORS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((int(r0), int(c)))
        cells.add((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        cells.add((int(r), int(c0)))
        cells.add((int(r), int(c1)))
    return cells


def _gap_side(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0:
        return "top"
    if c == c1:
        return "right"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    raise AssertionError("gap must lie on the rectangle perimeter")


def _sprite_direction(patch: np.ndarray, bg: int) -> str:
    assert patch.shape == (2, 2), "sprite bbox must be 2x2"
    missing = np.argwhere(patch == int(bg))
    assert missing.shape == (1, 2), "sprite patch must have exactly one missing corner"
    assert int(np.count_nonzero(patch != int(bg))) == 3, "sprite patch must have exactly three foreground pixels"
    return DIR_BY_MISSING[(int(missing[0, 0]), int(missing[0, 1]))]


def _rotate_sprite_to_side(patch: np.ndarray, bg: int, target_side: str) -> np.ndarray:
    for k in range(4):
        candidate = np.rot90(patch, k)
        if _sprite_direction(candidate, bg) == target_side:
            return candidate
    raise AssertionError("some rotation must align the sprite with the requested side")


def _find_regions(grid: np.ndarray, bg: int) -> list[dict[str, tuple[int, int, int, int] | str]]:
    regions: list[dict[str, tuple[int, int, int, int] | str]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 4 or w < 4:
                continue

            comp_set = set(coords)
            perimeter = _perimeter_cells(r0, c0, r1, c1)
            if comp_set - perimeter:
                continue
            missing = list(perimeter - comp_set)
            if len(missing) != 1:
                continue
            gap = missing[0]
            if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
                continue
            if int(grid[gap]) != int(bg):
                continue
            if len(comp_set) != len(perimeter) - 1:
                continue

            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            sprite_components = _components(interior != int(bg))
            if len(sprite_components) != 1 or len(sprite_components[0]) != 3:
                continue

            sprite_local = sprite_components[0]
            sr0, sc0, sr1, sc1 = _bbox(sprite_local)
            if (sr1 - sr0 + 1, sc1 - sc0 + 1) != (2, 2):
                continue

            sprite_bbox = (r0 + 1 + sr0, c0 + 1 + sc0, r0 + 1 + sr1, c0 + 1 + sc1)
            patch = grid[sprite_bbox[0] : sprite_bbox[2] + 1, sprite_bbox[1] : sprite_bbox[3] + 1]
            _sprite_direction(patch, bg)

            regions.append({"gap_side": _gap_side(gap, (r0, c0, r1, c1)), "sprite_bbox": sprite_bbox})

    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each internal 3-cell L sprite so its missing-corner direction matches
    the side containing the single wall gap of its surrounding rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    regions = _find_regions(g, bg)
    assert regions, "expected at least one rectangular wall region"

    out = g.copy()
    for region in regions:
        sr0, sc0, sr1, sc1 = region["sprite_bbox"]
        patch = out[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = _rotate_sprite_to_side(patch, bg, str(region["gap_side"]))

    return out


# --------------- generator ---------------

GEN_BG = 0
GEN_WALL = 5
GEN_SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RECT_SIZES = (5, 7, 9)
GEN_GRID_MIN = 16
GEN_GRID_MAX = 28
GEN_BOX_PADDING = 2
GEN_MARGIN = 1
GEN_MAX_ATTEMPTS = 400
GEN_PLACEMENT_ATTEMPTS = 200


def _make_sprite_patch(color: int, direction: str, bg: int = GEN_BG) -> np.ndarray:
    patch = np.full((2, 2), int(color), dtype=int)
    mr, mc = MISSING_BY_DIRECTION[direction]
    patch[mr, mc] = int(bg)
    return patch


def _make_region_patch(height: int, width: int, gap_side: str, sprite_side: str, sprite_color: int) -> np.ndarray:
    patch = np.full((height, width), GEN_BG, dtype=int)
    patch[0, :] = GEN_WALL
    patch[-1, :] = GEN_WALL
    patch[:, 0] = GEN_WALL
    patch[:, -1] = GEN_WALL

    if gap_side == "top":
        patch[0, width // 2] = GEN_BG
    elif gap_side == "right":
        patch[height // 2, width - 1] = GEN_BG
    elif gap_side == "bottom":
        patch[height - 1, width // 2] = GEN_BG
    else:
        patch[height // 2, 0] = GEN_BG

    sr0 = height // 2 - 1
    sc0 = width // 2 - 1
    patch[sr0 : sr0 + 2, sc0 : sc0 + 2] = _make_sprite_patch(sprite_color, sprite_side)
    return patch


def _boxes_separated(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    padding: int,
) -> bool:
    return (
        first[2] + padding < second[0]
        or second[2] + padding < first[0]
        or first[3] + padding < second[1]
        or second[3] + padding < first[1]
    )


def _sample_target_sides(count: int) -> list[str]:
    dirs = list(CARDINALS)
    random.shuffle(dirs)
    if count <= len(dirs):
        return dirs[:count]
    extra = [random.choice(CARDINALS) for _ in range(count - len(dirs))]
    result = dirs + extra
    random.shuffle(result)
    return result


def _sample_alignment_flags(count: int) -> list[bool]:
    if count == 1:
        return [random.choice([True, False])]
    flags = [True, False] + [random.choice([True, False]) for _ in range(count - 2)]
    random.shuffle(flags)
    return flags


def _place_specs(
    shape: tuple[int, int],
    specs: list[dict[str, int | str]],
) -> list[dict[str, int | str]] | None:
    height, width = shape
    order = sorted(range(len(specs)), key=lambda i: specs[i]["height"] * specs[i]["width"], reverse=True)
    placed_boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        spec = specs[idx]
        rh = int(spec["height"])
        rw = int(spec["width"])
        r_min = GEN_MARGIN
        r_max = height - rh - GEN_MARGIN
        c_min = GEN_MARGIN
        c_max = width - rw - GEN_MARGIN
        if r_min > r_max or c_min > c_max:
            return None

        placed = False
        for _ in range(GEN_PLACEMENT_ATTEMPTS):
            r0 = random.randint(r_min, r_max)
            c0 = random.randint(c_min, c_max)
            box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if any(not _boxes_separated(box, other, GEN_BOX_PADDING) for other in placed_boxes):
                continue
            spec["top"] = r0
            spec["left"] = c0
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    return specs


def _render_from_specs(specs: list[dict[str, int | str]], use_target_orientation: bool) -> np.ndarray:
    total_height = max(int(spec["top"]) + int(spec["height"]) for spec in specs) + GEN_MARGIN
    total_width = max(int(spec["left"]) + int(spec["width"]) for spec in specs) + GEN_MARGIN
    grid = np.full((total_height, total_width), GEN_BG, dtype=int)

    for spec in specs:
        sprite_side = str(spec["target_side"] if use_target_orientation else spec["input_side"])
        patch = _make_region_patch(
            int(spec["height"]),
            int(spec["width"]),
            str(spec["target_side"]),
            sprite_side,
            int(spec["sprite_color"]),
        )
        r0 = int(spec["top"])
        c0 = int(spec["left"])
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch

    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    width = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    max_regions = 1 + int(min(height, width) >= 19) + int(min(height, width) >= 23) + int(min(height, width) >= 27)
    count = random.randint(1, max_regions)

    target_sides = _sample_target_sides(count)
    align_flags = _sample_alignment_flags(count)
    sprite_colors = random.sample(list(GEN_SPRITE_COLORS), count)
    specs: list[dict[str, int | str]] = []
    for i in range(count):
        target_side = target_sides[i]
        if align_flags[i]:
            input_side = target_side
        else:
            input_side = random.choice([side for side in CARDINALS if side != target_side])
        specs.append(
            {
                "height": random.choice(GEN_RECT_SIZES),
                "width": random.choice(GEN_RECT_SIZES),
                "target_side": target_side,
                "input_side": input_side,
                "sprite_color": sprite_colors[i],
            }
        )

    placed = _place_specs((height, width), specs)
    if placed is None:
        return None

    inp = np.full((height, width), GEN_BG, dtype=int)
    expected = np.full((height, width), GEN_BG, dtype=int)
    for spec in placed:
        r0 = int(spec["top"])
        c0 = int(spec["left"])
        input_patch = _make_region_patch(
            int(spec["height"]),
            int(spec["width"]),
            str(spec["target_side"]),
            str(spec["input_side"]),
            int(spec["sprite_color"]),
        )
        output_patch = _make_region_patch(
            int(spec["height"]),
            int(spec["width"]),
            str(spec["target_side"]),
            str(spec["target_side"]),
            int(spec["sprite_color"]),
        )
        inp[r0 : r0 + input_patch.shape[0], c0 : c0 + input_patch.shape[1]] = input_patch
        expected[r0 : r0 + output_patch.shape[0], c0 : c0 + output_patch.shape[1]] = output_patch

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
