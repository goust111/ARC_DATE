import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

SIDE_ORDER = ("top", "right", "bottom", "left")
SIDE_COLORS = {
    "top": BLUE,
    "right": RED,
    "bottom": GREEN,
    "left": YELLOW,
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    nbrs = _neighbors4 if connectivity == 4 else _neighbors8

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
                for nr, nc in nbrs(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _side_segment(line: np.ndarray) -> dict:
    idx = np.flatnonzero(line != BLACK)
    assert idx.size > 0, "expected one colored border segment"
    assert np.all(np.diff(idx) == 1), "border segment must be contiguous"
    color = int(line[idx[0]])
    assert color != BLACK
    assert np.all(line[idx] == color), "border segment must be monochromatic"
    return {
        "color": color,
        "length": int(idx.size),
        "start": int(idx[0]),
        "end": int(idx[-1]),
    }


def _enclosed_black_mask(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape
    black = crop == BLACK
    exterior = np.zeros((h, w), dtype=bool)
    stack = []

    for r in range(h):
        for c in (0, w - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                stack.append((nr, nc))

    return black & ~exterior


def _analyze_grid(grid: np.ndarray) -> tuple[dict[str, dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30
    assert h >= 12 and w >= 12
    assert int(g[0, 0]) == BLACK
    assert int(g[0, w - 1]) == BLACK
    assert int(g[h - 1, 0]) == BLACK
    assert int(g[h - 1, w - 1]) == BLACK

    sides = {
        "top": _side_segment(g[0, 1 : w - 1]),
        "right": _side_segment(g[1 : h - 1, w - 1]),
        "bottom": _side_segment(g[h - 1, 1 : w - 1]),
        "left": _side_segment(g[1 : h - 1, 0]),
    }

    inner_gray = g[1 : h - 1, 1 : w - 1] == GRAY
    comps = _components(inner_gray, connectivity=8)
    assert len(comps) == 4, "expected four gray outline objects"

    objects = []
    for comp in comps:
        coords = comp + 1
        bbox = _bbox_from_coords(coords)
        crop = _crop_bbox(g, bbox)
        cavity_mask = _enclosed_black_mask(crop)
        cavity_comps = _components(cavity_mask, connectivity=4)
        assert len(cavity_comps) == 1, "each outline must enclose one cavity"
        cavity_bbox = _bbox_from_coords(cavity_comps[0])
        cavity_bbox_area = (cavity_bbox[2] - cavity_bbox[0] + 1) * (cavity_bbox[3] - cavity_bbox[1] + 1)
        objects.append(
            {
                "bbox": bbox,
                "crop": crop,
                "cavity_mask": cavity_mask,
                "cavity_area": int(cavity_mask.sum()),
                "cavity_bbox_area": int(cavity_bbox_area),
            }
        )

    return sides, objects


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Measure each colored frame side and fill the gray cavity whose area matches that side length.

    Input structure:
    - Black background with a one-pixel outer border carrying four colored side segments; corners stay black.
    - The top, right, bottom, and left border segments use distinct visible colors and encode four lengths.
    - Inside the border are four separate gray single-pixel outline objects, each enclosing one black cavity.

    Rule:
    - Read the unique contiguous non-black segment on each of the four outer sides and record its color and pixel length.
    - Detect the four gray 8-connected outline components and compute each enclosed black cavity area.
    - Match every side to the unique outline whose cavity area equals that side length.
    - Fill that cavity with the matched side color while leaving the gray outlines, border, and other background pixels unchanged.

    Key insight:
    - The border segments are numeric instructions: their visible lengths select which cavity receives each side's color.
    """
    grid = np.asarray(input_grid, dtype=int)
    sides, objects = _analyze_grid(grid)

    lengths = [sides[name]["length"] for name in SIDE_ORDER]
    assert len(set(lengths)) == 4, "side lengths must be distinct"

    by_area = {}
    for obj in objects:
        area = obj["cavity_area"]
        assert area not in by_area, "cavity areas must be unique"
        by_area[area] = obj

    out = grid.copy()
    for side_name in SIDE_ORDER:
        side = sides[side_name]
        area = side["length"]
        assert area in by_area, "every side length must match a cavity area"
        obj = by_area[area]
        r0, c0, r1, c1 = obj["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1][obj["cavity_mask"]] = side["color"]

    return out


def _random_connected_shape(area: int, rng: np.random.Generator, force_irregular: bool) -> np.ndarray | None:
    candidates = []
    for h in range(2, 6):
        for w in range(2, 6):
            if h * w >= area and h * w <= max(area + 4, area):
                candidates.append((h, w))
    rng.shuffle(candidates)

    # Allow some exact rectangles, but make every puzzle include irregular counterexamples.
    if not force_irregular:
        rectangles = [(h, w) for h, w in candidates if h * w == area]
        if rectangles and float(rng.random()) < 0.35:
            h, w = rectangles[int(rng.integers(len(rectangles)))]
            return np.ones((h, w), dtype=bool)

    for h, w in candidates:
        for _ in range(50):
            start = (int(rng.integers(h)), int(rng.integers(w)))
            cells = {start}
            frontier = set()
            for nr, nc in _neighbors4(*start):
                if 0 <= nr < h and 0 <= nc < w:
                    frontier.add((nr, nc))

            while len(cells) < area and frontier:
                options = list(frontier)
                pick = options[int(rng.integers(len(options)))]
                frontier.remove(pick)
                cells.add(pick)
                for nr, nc in _neighbors4(*pick):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))

            if len(cells) != area:
                continue

            coords = np.array(sorted(cells), dtype=int)
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            if r1 == r0 or c1 == c0:
                continue
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in cells:
                mask[r - r0, c - c0] = True
            if force_irregular and int(mask.sum()) == mask.shape[0] * mask.shape[1]:
                continue
            return mask
    return None


def _outline_sprite(cavity: np.ndarray) -> np.ndarray:
    ch, cw = cavity.shape
    padded = np.zeros((ch + 2, cw + 2), dtype=bool)
    padded[1 : ch + 1, 1 : cw + 1] = cavity
    shell = np.zeros_like(padded)

    for r, c in np.argwhere(padded):
        for nr, nc in _neighbors8(int(r), int(c)):
            if 0 <= nr < padded.shape[0] and 0 <= nc < padded.shape[1] and not padded[nr, nc]:
                shell[nr, nc] = True

    sprite = np.zeros(shell.shape, dtype=int)
    sprite[shell] = GRAY
    return sprite


def _draw_side_segment(grid: np.ndarray, side: str, length: int, color: int, rng: np.random.Generator) -> None:
    h, w = grid.shape
    if side == "top":
        start = int(rng.integers(1, w - length))
        grid[0, start : start + length] = color
    elif side == "bottom":
        start = int(rng.integers(1, w - length))
        grid[h - 1, start : start + length] = color
    elif side == "left":
        start = int(rng.integers(1, h - length))
        grid[start : start + length, 0] = color
    else:
        start = int(rng.integers(1, h - length))
        grid[start : start + length, w - 1] = color


def _place_sprite(
    grid: np.ndarray,
    sprite: np.ndarray,
    zone: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> bool:
    r0, r1, c0, c1 = zone
    sh, sw = sprite.shape
    if sh > (r1 - r0 + 1) or sw > (c1 - c0 + 1):
        return False

    row_choices = list(range(r0, r1 - sh + 2))
    col_choices = list(range(c0, c1 - sw + 2))
    rng.shuffle(row_choices)
    rng.shuffle(col_choices)

    for rr in row_choices:
        for cc in col_choices:
            if np.any(grid[rr : rr + sh, cc : cc + sw] != BLACK):
                continue
            grid[rr : rr + sh, cc : cc + sw] = np.where(sprite != BLACK, sprite, grid[rr : rr + sh, cc : cc + sw])
            return True
    return False


def _validate_generated(grid: np.ndarray) -> None:
    sides, objects = _analyze_grid(grid)
    lengths = [sides[name]["length"] for name in SIDE_ORDER]
    assert len(set(lengths)) == 4
    assert [sides[name]["color"] for name in SIDE_ORDER] == [BLUE, RED, GREEN, YELLOW]

    areas = [obj["cavity_area"] for obj in objects]
    assert len(set(areas)) == 4
    assert sorted(areas) == sorted(lengths)
    assert any(obj["cavity_area"] != obj["cavity_bbox_area"] for obj in objects), "need irregular cavities"

    out = main(grid)
    assert out.shape == grid.shape
    assert np.array_equal(out == GRAY, grid == GRAY)
    assert np.array_equal(out[grid != BLACK], np.where(grid != BLACK, grid, out)[grid != BLACK])

    area_to_color = {sides[name]["length"]: sides[name]["color"] for name in SIDE_ORDER}
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        filled = out[r0 : r1 + 1, c0 : c1 + 1][obj["cavity_mask"]]
        assert np.all(filled == area_to_color[obj["cavity_area"]])


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(400):
        h = int(rng.integers(22, 31))
        w = int(rng.integers(22, 31))
        if h == w and h < 30 and float(rng.random()) < 0.7:
            h += 1

        lengths = list(map(int, rng.choice(np.arange(4, 13), size=4, replace=False)))
        rng.shuffle(lengths)
        side_lengths = {side: length for side, length in zip(SIDE_ORDER, lengths)}

        irregular_count = int(rng.integers(1, 4))
        irregular_areas = set(map(int, rng.choice(lengths, size=irregular_count, replace=False)))
        objects = []
        ok = True
        for area in lengths:
            cavity = _random_connected_shape(area, rng, force_irregular=area in irregular_areas)
            if cavity is None:
                ok = False
                break
            sprite = _outline_sprite(cavity)
            objects.append({"area": area, "sprite": sprite})
        if not ok:
            continue

        grid = np.zeros((h, w), dtype=int)
        for side in SIDE_ORDER:
            _draw_side_segment(grid, side, side_lengths[side], SIDE_COLORS[side], rng)
        grid[0, 0] = BLACK
        grid[0, w - 1] = BLACK
        grid[h - 1, 0] = BLACK
        grid[h - 1, w - 1] = BLACK

        mid_r = h // 2
        mid_c = w // 2
        zones = [
            (2, mid_r - 2, 2, mid_c - 2),
            (2, mid_r - 2, mid_c + 1, w - 3),
            (mid_r + 1, h - 3, 2, mid_c - 2),
            (mid_r + 1, h - 3, mid_c + 1, w - 3),
        ]

        zone_order = list(range(4))
        rng.shuffle(zone_order)
        rng.shuffle(objects)
        placed = True
        for zone_idx, obj in zip(zone_order, objects):
            if not _place_sprite(grid, obj["sprite"], zones[zone_idx], rng):
                placed = False
                break
        if not placed:
            continue

        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
