import random

import numpy as np


BG = 0
OUTLINE = 5
SPRITE_COLORS = (1, 2, 3, 4, 7)
AREAS = tuple(range(3, 9))
GEN_ATTEMPTS = 400

FOUR_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EIGHT_DIRS = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
]

_OUTLINE_COUNT_BAG = []
_POLYOMINO_BANK = None
_OUTLINE_BANK = None


def _connected_components(mask: np.ndarray, connectivity: int) -> list[tuple[tuple[int, int], ...]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = FOUR_DIRS if connectivity == 4 else EIGHT_DIRS
    comps: list[tuple[tuple[int, int], ...]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(tuple(sorted(cells)))
    return comps


def _bbox(coords: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_coords(coords: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cells = set(coords)
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _transform_coords(coords: tuple[tuple[int, int], ...], variant: int) -> tuple[tuple[int, int], ...]:
    cells = list(coords)
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)

    transformed = []
    for r, c in cells:
        if variant == 0:
            rr, cc = r, c
        elif variant == 1:
            rr, cc = r, max_c - c
        elif variant == 2:
            rr, cc = max_r - r, c
        elif variant == 3:
            rr, cc = max_r - r, max_c - c
        elif variant == 4:
            rr, cc = c, r
        elif variant == 5:
            rr, cc = c, max_r - r
        elif variant == 6:
            rr, cc = max_c - c, r
        else:
            rr, cc = max_c - c, max_r - r
        transformed.append((rr, cc))
    return _normalize_coords(set(transformed))


def _coords_to_array(coords: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    arr = np.zeros((h, w), dtype=int)
    for r, c in coords:
        arr[r, c] = color
    return arr


def _build_polyomino_bank() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    bank: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(2, max(AREAS) + 1):
        shapes = set()
        for shape in bank[size - 1]:
            cells = set(shape)
            frontier = set()
            for r, c in cells:
                for dr, dc in FOUR_DIRS:
                    frontier.add((r + dr, c + dc))
            frontier -= cells
            for cell in frontier:
                shapes.add(_normalize_coords(cells | {cell}))
        bank[size] = shapes

    filtered: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for size in AREAS:
        keep = []
        for shape in bank[size]:
            h = max(r for r, _ in shape) + 1
            w = max(c for _, c in shape) + 1
            if max(h, w) <= 4:
                keep.append(shape)
        filtered[size] = sorted(keep)
    return filtered


def _polyomino_bank() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    global _POLYOMINO_BANK
    if _POLYOMINO_BANK is None:
        _POLYOMINO_BANK = _build_polyomino_bank()
    return _POLYOMINO_BANK


def _render_outline_profile(profile: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    height = len(profile) + 2
    width = max(right for _, right in profile) + 3
    arr = np.zeros((height, width), dtype=int)

    first_left, first_right = profile[0]
    last_left, last_right = profile[-1]
    arr[0, first_left : first_right + 3] = OUTLINE
    arr[-1, last_left : last_right + 3] = OUTLINE

    interior = []
    for idx, (left, right) in enumerate(profile, start=1):
        arr[idx, left] = OUTLINE
        arr[idx, right + 2] = OUTLINE
        for c in range(left + 1, right + 2):
            interior.append((idx, c))

    return arr, tuple(sorted(interior))


def _profile_is_irregular(profile: tuple[tuple[int, int], ...]) -> bool:
    widths = [right - left + 1 for left, right in profile]
    lefts = [left for left, _ in profile]
    rights = [right for _, right in profile]
    return len(profile) > 1 and (len(set(widths)) > 1 or len(set(lefts)) > 1 or len(set(rights)) > 1)


def _build_outline_bank() -> dict[int, list[dict]]:
    by_area: dict[int, dict[tuple[tuple[int, int], bytes], dict]] = {area: {} for area in AREAS}

    def visit(profile: list[tuple[int, int]], area: int) -> None:
        if area in by_area:
            norm_shift = min(left for left, _ in profile)
            normalized = tuple((left - norm_shift, right - norm_shift) for left, right in profile)
            arr, interior = _render_outline_profile(normalized)
            comps = _connected_components(arr == OUTLINE, connectivity=8)
            if len(comps) == 1 and len(interior) == area:
                key = (arr.shape, arr.tobytes())
                by_area[area][key] = {
                    "array": arr,
                    "interior": interior,
                    "profile": normalized,
                    "irregular": _profile_is_irregular(normalized),
                }

        if len(profile) >= 4 or area >= max(AREAS):
            return

        left, right = profile[-1]
        for next_left in range(left - 1, left + 2):
            for next_right in range(right - 1, right + 2):
                if next_right < next_left:
                    continue
                width = next_right - next_left + 1
                if not (1 <= width <= 5):
                    continue
                if area + width > max(AREAS):
                    continue
                visit(profile + [(next_left, next_right)], area + width)

    for width in range(1, 6):
        visit([(0, width - 1)], width)

    return {area: list(entries.values()) for area, entries in by_area.items()}


def _outline_bank() -> dict[int, list[dict]]:
    global _OUTLINE_BANK
    if _OUTLINE_BANK is None:
        _OUTLINE_BANK = _build_outline_bank()
    return _OUTLINE_BANK


def _outline_interior_cells(grid: np.ndarray, coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rows: dict[int, list[int]] = {}
    for r, c in coords:
        rows.setdefault(r, []).append(c)

    interior = []
    for r in sorted(rows):
        cols = sorted(rows[r])
        for c in range(cols[0] + 1, cols[-1]):
            if int(grid[r, c]) == BG:
                interior.append((r, c))
    return tuple(interior)


def _parse_outlines(grid: np.ndarray) -> list[dict]:
    outlines = []
    for coords in _connected_components(grid == OUTLINE, connectivity=8):
        interior = _outline_interior_cells(grid, coords)
        r0, c0, r1, c1 = _bbox(coords)
        outlines.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "interior": tuple(sorted(interior)),
                "area": len(interior),
            }
        )
    return outlines


def _parse_sprites(grid: np.ndarray) -> list[dict]:
    sprites = []
    mask = (grid != BG) & (grid != OUTLINE)
    for coords in _connected_components(mask, connectivity=4):
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "sprites must be monochromatic"
        color = colors.pop()
        r0, c0, r1, c1 = _bbox(coords)
        sprites.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "color": color,
                "count": len(coords),
                "scan": tuple(sorted(coords)),
            }
        )
    return sprites


def _solve(grid: np.ndarray) -> np.ndarray:
    outlines = _parse_outlines(grid)
    sprites = _parse_sprites(grid)

    assert 3 <= len(outlines) <= 4
    assert len(sprites) == len(outlines)
    assert len({outline["area"] for outline in outlines}) == len(outlines)
    assert len({sprite["count"] for sprite in sprites}) == len(sprites)
    assert len({sprite["color"] for sprite in sprites}) == len(sprites)

    outline_by_area = {outline["area"]: outline for outline in outlines}
    assert len(outline_by_area) == len(outlines)

    out = np.array(grid, dtype=int, copy=True)
    for sprite in sprites:
        for r, c in sprite["coords"]:
            out[r, c] = BG

    used_areas = set()
    for sprite in sprites:
        assert sprite["count"] in outline_by_area, "every sprite must match one outline"
        outline = outline_by_area[sprite["count"]]
        assert sprite["count"] not in used_areas
        used_areas.add(sprite["count"])

        target = tuple(sorted(outline["interior"]))
        assert len(target) == sprite["count"]
        for r, c in target:
            out[r, c] = sprite["color"]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move each monochrome sprite into the unique gray outline cavity whose row-span area matches its pixel count.

    Input structure:
    - The grid has black background, 3-4 gray 8-connected outline objects, and the same number of separated colored sprites.
    - Each gray outline is one-pixel thick and defines its cavity row by row: on every occupied row, black cells strictly between the leftmost and rightmost gray pixels belong to that row's interior.
    - Each sprite is a 4-connected monochrome shape outside the outlines, and sprite sizes are all distinct.

    Rule:
    - Compute every outline's total row-span interior area and every sprite's foreground pixel count.
    - Match each sprite to the unique outline whose interior area equals that sprite count.
    - Erase the sprite from its original location and fill the matched outline interior with the sprite color, preserving the gray borders.

    Key insight:
    - Color, position, and shape do not determine the destination; only the equality between sprite pixel count and outline cavity area gives the correct relocation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _solve(g)


def _pick_outline_count() -> int:
    global _OUTLINE_COUNT_BAG
    if not _OUTLINE_COUNT_BAG:
        _OUTLINE_COUNT_BAG = [3, 4, 3, 4]
        random.shuffle(_OUTLINE_COUNT_BAG)
    return _OUTLINE_COUNT_BAG.pop()


def _sample_outline(area: int) -> dict:
    choices = _outline_bank()[area]
    irregular = [choice for choice in choices if choice["irregular"]]
    pool = irregular if irregular and random.random() < 0.75 else choices
    choice = random.choice(pool)
    arr = choice["array"].copy()
    interior = choice["interior"]

    if random.random() < 0.5:
        arr = np.flipud(arr)
        h = arr.shape[0]
        interior = tuple(sorted((h - 1 - r, c) for r, c in interior))
    if random.random() < 0.5:
        arr = np.fliplr(arr)
        w = arr.shape[1]
        interior = tuple(sorted((r, w - 1 - c) for r, c in interior))

    return {
        "array": arr,
        "interior": tuple(sorted(interior)),
        "area": area,
        "irregular": choice["irregular"],
    }


def _sample_sprite(area: int, color: int) -> dict:
    shapes = _polyomino_bank()[area]
    complex_shapes = []
    for shape in shapes:
        h = max(r for r, _ in shape) + 1
        w = max(c for _, c in shape) + 1
        if h * w != area:
            complex_shapes.append(shape)

    pool = complex_shapes if complex_shapes and random.random() < 0.8 else shapes
    shape = _transform_coords(random.choice(pool), random.randrange(8))
    arr = _coords_to_array(shape, color)
    return {
        "array": arr,
        "count": area,
        "color": color,
        "complex": int(np.count_nonzero(arr)) < arr.size,
    }


def _place_rectangles(items: list[dict], height: int, width: int) -> list[tuple[int, int]] | None:
    blocked = np.zeros((height, width), dtype=bool)
    placements = []
    for item in items:
        obj = item["array"]
        h, w = obj.shape
        candidates = []
        for r in range(height - h + 1):
            for c in range(width - w + 1):
                r0 = max(0, r - 1)
                c0 = max(0, c - 1)
                r1 = min(height, r + h + 1)
                c1 = min(width, c + w + 1)
                if not blocked[r0:r1, c0:c1].any():
                    candidates.append((r, c))
        if not candidates:
            return None
        pos = random.choice(candidates)
        placements.append(pos)
        r, c = pos
        h, w = obj.shape
        blocked[max(0, r - 1) : min(height, r + h + 1), max(0, c - 1) : min(width, c + w + 1)] = True
    return placements


def _render_items(items: list[dict], placements: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for item, (r, c) in zip(items, placements):
        obj = item["array"]
        h, w = obj.shape
        patch = grid[r : r + h, c : c + w]
        mask = obj != BG
        patch[mask] = obj[mask]
    return grid


def _expected_from_layout(items: list[dict], placements: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    inp = _render_items(items, placements, shape)
    out = inp.copy()

    sprite_color_by_count = {}
    for item, (r, c) in zip(items, placements):
        if item["kind"] != "sprite":
            continue
        sprite_color_by_count[item["count"]] = item["color"]
        obj = item["array"]
        h, w = obj.shape
        mask = obj != BG
        out[r : r + h, c : c + w][mask] = BG

    for item, (r, c) in zip(items, placements):
        if item["kind"] != "outline":
            continue
        color = sprite_color_by_count[item["area"]]
        for rr, cc in item["interior"]:
            out[r + rr, c + cc] = color

    return out


def _instance_valid(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        outlines = _parse_outlines(inp)
        sprites = _parse_sprites(inp)
        if len(outlines) != len(sprites) or len(outlines) not in (3, 4):
            return False
        if len({outline["area"] for outline in outlines}) != len(outlines):
            return False
        if len({sprite["count"] for sprite in sprites}) != len(sprites):
            return False
        if len({sprite["color"] for sprite in sprites}) != len(sprites):
            return False
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _generate_attempt() -> np.ndarray | None:
    outline_count = _pick_outline_count()
    areas = random.sample(list(AREAS), outline_count)
    colors = random.sample(list(SPRITE_COLORS), outline_count)

    outlines = [_sample_outline(area) for area in areas]
    sprites = [_sample_sprite(area, color) for area, color in zip(areas, colors)]

    if not any(outline["irregular"] for outline in outlines):
        return None
    if not any(sprite["complex"] for sprite in sprites):
        return None

    items = [{"kind": "outline", **outline} for outline in outlines] + [{"kind": "sprite", **sprite} for sprite in sprites]
    random.shuffle(items)

    height = random.randint(16 + outline_count, 22 + outline_count)
    width = random.randint(20, 28)
    placements = _place_rectangles(items, height, width)
    if placements is None:
        return None

    inp = _render_items(items, placements, (height, width))
    expected = _expected_from_layout(items, placements, (height, width))
    if not _instance_valid(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
