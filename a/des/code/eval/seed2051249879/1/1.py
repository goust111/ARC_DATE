import random
from typing import Dict, List, Tuple

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3)  # blue, red, green
QUADRANTS = ("TL", "TR", "BR", "BL")


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(pts)
    return parts


def _extract_legend_patterns(grid: np.ndarray) -> Dict[int, np.ndarray]:
    h, w = grid.shape
    candidates: Dict[int, List[np.ndarray]] = {}

    for r in range(h - 2):
        for c in range(w - 4):
            color = int(grid[r, c])
            if color in (BLACK, GREY):
                continue

            block = grid[r : r + 3, c : c + 2]
            if not np.all(block == color):
                continue

            if c > 0 and np.any(grid[r : r + 3, c - 1] == color):
                continue
            if r > 0 and np.any(grid[r - 1, c : c + 2] == color):
                continue
            if r + 3 < h and np.any(grid[r + 3, c : c + 2] == color):
                continue

            sample = grid[r : r + 3, c + 2 : c + 5]
            values = set(map(int, np.unique(sample)))
            if not values <= {BLACK, color} or color not in values:
                continue

            candidates.setdefault(color, []).append(sample == color)

    patterns: Dict[int, np.ndarray] = {}
    for color, items in candidates.items():
        unique: List[np.ndarray] = []
        for item in items:
            if not any(np.array_equal(item, prev) for prev in unique):
                unique.append(item)
        if len(unique) == 1:
            patterns[color] = unique[0]

    assert len(patterns) == 3, "expected exactly three legend entries"
    return patterns


def _find_regions(grid: np.ndarray, patterns: Dict[int, np.ndarray]) -> List[Tuple[int, int, int, int, int, int, int]]:
    regions: List[Tuple[int, int, int, int, int, int, int]] = []
    for pts in _components(grid == GREY):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        oh = r1 - r0 + 1
        ow = c1 - c0 + 1
        if oh < 3 or ow < 3:
            continue

        expected = 2 * (oh + ow) - 4
        if len(pts) != expected:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == GREY):
            continue
        if oh > 2 and ow > 2 and np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == GREY):
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        seeds = np.argwhere(interior != BLACK)
        if seeds.shape[0] != 1:
            continue

        sr, sc = map(int, seeds[0])
        color = int(interior[sr, sc])
        if color not in patterns:
            continue
        regions.append((r0, r1, c0, c1, sr, sc, color))

    assert regions, "expected at least one framed region"
    return regions


def _rotation_k(local_r: int, local_c: int, height: int, width: int) -> int:
    top = 2 * local_r < height
    left = 2 * local_c < width
    if top and left:
        return 0
    if top and not left:
        return 3
    if not top and not left:
        return 2
    return 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed interior with the legend texture for its seed color.
    The seed quadrant selects a clockwise rotation: TL=0, TR=90, BR=180, BL=270.
    """
    g = np.asarray(grid, dtype=int)
    patterns = _extract_legend_patterns(g)
    regions = _find_regions(g, patterns)

    out = g.copy()
    for r0, r1, c0, c1, sr, sc, color in regions:
        interior_h = r1 - r0 - 1
        interior_w = c1 - c0 - 1
        k = _rotation_k(sr, sc, interior_h, interior_w)
        texture = np.rot90(patterns[color], k=k)
        tiled = np.tile(texture, ((interior_h + 2) // 3, (interior_w + 2) // 3))[:interior_h, :interior_w]
        out[r0 + 1 : r1, c0 + 1 : c1] = np.where(tiled, color, BLACK)

    return out


# --------------- generator ---------------


def _texture_key(mask: np.ndarray) -> Tuple[int, ...]:
    return tuple(mask.astype(int).ravel())


def _random_texture(rng: random.Random, used: set[Tuple[int, ...]]) -> np.ndarray:
    while True:
        count = rng.randint(3, 6)
        mask = np.zeros((3, 3), dtype=bool)
        for idx in rng.sample(range(9), count):
            mask[idx // 3, idx % 3] = True

        if np.all(mask[:, 0]):
            continue

        rotations = [_texture_key(np.rot90(mask, k=k)) for k in range(4)]
        if len(set(rotations)) != 4:
            continue

        canonical = min(rotations)
        if canonical in used:
            continue

        used.add(canonical)
        return mask


def _fits(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    r1 = min(h, top + height + pad)
    c0 = max(0, left - pad)
    c1 = min(w, left + width + pad)
    return not occupied[r0:r1, c0:c1].any()


def _occupy(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    r1 = min(h, top + height + pad)
    c0 = max(0, left - pad)
    c1 = min(w, left + width + pad)
    occupied[r0:r1, c0:c1] = True


def _pick_spot(
    occupied: np.ndarray,
    rng: random.Random,
    height: int,
    width: int,
    tries: int = 200,
) -> Tuple[int, int] | None:
    h, w = occupied.shape
    if h - height < 2 or w - width < 2:
        return None
    for _ in range(tries):
        top = rng.randint(1, h - height - 1)
        left = rng.randint(1, w - width - 1)
        if _fits(occupied, top, left, height, width):
            return top, left
    return None


def _place_legend(
    grid: np.ndarray,
    occupied: np.ndarray,
    rng: random.Random,
    patterns: Dict[int, np.ndarray],
) -> None:
    h, w = grid.shape
    entry_h, entry_w = 3, 5
    gap = rng.choice((1, 2))
    arrangement = rng.choice(("horizontal", "vertical"))
    colors = list(SEED_COLORS)
    rng.shuffle(colors)

    if arrangement == "horizontal":
        area_h = entry_h
        area_w = 3 * entry_w + 2 * gap
        top = rng.choice((1, h - area_h - 1))
        left = rng.randint(1, w - area_w - 1)
        offsets = [(0, i * (entry_w + gap)) for i in range(3)]
    else:
        area_h = 3 * entry_h + 2 * gap
        area_w = entry_w
        top = rng.randint(1, h - area_h - 1)
        left = rng.choice((1, w - area_w - 1))
        offsets = [(i * (entry_h + gap), 0) for i in range(3)]

    assert _fits(occupied, top, left, area_h, area_w)
    _occupy(occupied, top, left, area_h, area_w)

    for color, (dr, dc) in zip(colors, offsets):
        r = top + dr
        c = left + dc
        grid[r : r + 3, c : c + 2] = color
        sample = patterns[color]
        grid[r : r + 3, c + 2 : c + 5][sample] = color


def _seed_position(rng: random.Random, quadrant: str, height: int, width: int) -> Tuple[int, int]:
    half_h = height // 2
    half_w = width // 2
    if quadrant == "TL":
        rs = range(0, half_h)
        cs = range(0, half_w)
    elif quadrant == "TR":
        rs = range(0, half_h)
        cs = range(half_w, width)
    elif quadrant == "BR":
        rs = range(half_h, height)
        cs = range(half_w, width)
    else:
        rs = range(half_h, height)
        cs = range(0, half_w)
    return rng.choice(tuple(rs)), rng.choice(tuple(cs))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(24, 32)
    width = rng.randint(24, 34)
    grid = np.full((height, width), BLACK, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    used_textures: set[Tuple[int, ...]] = set()
    patterns = {color: _random_texture(rng, used_textures) for color in SEED_COLORS}

    try:
        _place_legend(grid, occupied, rng, patterns)
    except Exception:
        return None

    region_count = rng.randint(4, 6)
    region_colors = list(SEED_COLORS) + [rng.choice(SEED_COLORS) for _ in range(region_count - 3)]
    region_quadrants = list(QUADRANTS) + [rng.choice(QUADRANTS) for _ in range(region_count - 4)]
    rng.shuffle(region_colors)
    rng.shuffle(region_quadrants)

    if region_count == 4:
        size_choices = (6, 8, 10)
    elif region_count == 5:
        size_choices = (4, 6, 8, 10)
    else:
        size_choices = (4, 6, 8)

    specs = []
    for color, quadrant in zip(region_colors, region_quadrants):
        inner_h = rng.choice(size_choices)
        inner_w = rng.choice(size_choices)
        specs.append((inner_h * inner_w, inner_h, inner_w, color, quadrant))
    specs.sort(reverse=True)

    for _, inner_h, inner_w, color, quadrant in specs:
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        spot = _pick_spot(occupied, rng, outer_h, outer_w)
        if spot is None:
            return None
        top, left = spot
        _occupy(occupied, top, left, outer_h, outer_w)

        grid[top, left : left + outer_w] = GREY
        grid[top + outer_h - 1, left : left + outer_w] = GREY
        grid[top : top + outer_h, left] = GREY
        grid[top : top + outer_h, left + outer_w - 1] = GREY

        sr, sc = _seed_position(rng, quadrant, inner_h, inner_w)
        grid[top + 1 + sr, left + 1 + sc] = color

    try:
        parsed = _extract_legend_patterns(grid)
        regions = _find_regions(grid, parsed)
    except Exception:
        return None

    if len(regions) != region_count:
        return None

    out = main(grid)
    if out.shape != grid.shape or not np.any(out != grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
