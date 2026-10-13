from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("left", "right", "top", "bottom")
LEGEND_BLOCK = 4
LEGEND_TEMPLATE = 3
LEGEND_GAP = 1
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _enclosed_regions(grid: np.ndarray) -> list[tuple[list[tuple[int, int]], tuple[int, int, int, int], int]]:
    h, w = grid.shape
    passable = grid != WALL
    exterior = np.zeros_like(passable, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if passable[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    interior_mask = passable & ~exterior
    regions = []
    for comp in _components(interior_mask):
        r0, c0, r1, c1 = _bbox(comp)
        colors = {
            int(grid[r, c])
            for r, c in comp
            if int(grid[r, c]) not in (BG, WALL)
        }
        assert len(colors) == 1, "each enclosed region must contain exactly one seed color"
        seed_color = next(iter(colors))
        seed_count = sum(int(grid[r, c]) == seed_color for r, c in comp)
        assert seed_count == 1, "each enclosed region must contain exactly one seed pixel"
        regions.append((comp, (r0, c0, r1, c1), seed_color))

    assert regions, "expected at least one enclosed region"
    return regions


def _ring_is_background(grid: np.ndarray, r0: int, c0: int, size: int) -> bool:
    h, w = grid.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + size + 1)
    cc1 = min(w, c0 + size + 1)
    for r in range(rr0, rr1):
        for c in range(cc0, cc1):
            inside = r0 <= r < r0 + size and c0 <= c < c0 + size
            if inside:
                continue
            if int(grid[r, c]) != BG:
                return False
    return True


def _extract_legend_template(patch: np.ndarray, size: int) -> tuple[int, np.ndarray] | None:
    key = int(patch[0, 0])
    if key in (BG, WALL):
        return None

    if size == 4:
        if np.any(patch[0, 1:] != BG) or np.any(patch[1:, 0] != BG):
            return None
        template = patch[1:, 1:]
    else:
        template = patch

    values = {int(v) for v in np.unique(template)}
    if not values.issubset({BG, key}) or BG not in values or key not in values:
        return None
    if int(np.count_nonzero(template == key)) < 2:
        return None
    return key, template.copy()


def _edge_starts(shape: tuple[int, int], side: str, size: int):
    h, w = shape
    if side == "left":
        if w < size:
            return
        for r0 in range(h - size + 1):
            yield r0, 0
    elif side == "right":
        if w < size:
            return
        c0 = w - size
        for r0 in range(h - size + 1):
            yield r0, c0
    elif side == "top":
        if h < size:
            return
        for c0 in range(w - size + 1):
            yield 0, c0
    else:
        if h < size:
            return
        r0 = h - size
        for c0 in range(w - size + 1):
            yield r0, c0


def _legend_side_mapping(
    grid: np.ndarray,
    needed_colors: set[int],
    side: str,
    size: int,
) -> dict[int, np.ndarray] | None:
    found: dict[int, list[np.ndarray]] = {}
    for r0, c0 in _edge_starts(grid.shape, side, size):
        if not _ring_is_background(grid, r0, c0, size):
            continue
        patch = grid[r0 : r0 + size, c0 : c0 + size]
        extracted = _extract_legend_template(patch, size)
        if extracted is None:
            continue
        key, template = extracted
        if key not in needed_colors:
            continue
        found.setdefault(key, []).append(template)

    if set(found) != needed_colors:
        return None
    if any(len(items) != 1 for items in found.values()):
        return None
    return {color: items[0] for color, items in found.items()}


def _legend_templates(grid: np.ndarray, needed_colors: set[int]) -> dict[int, np.ndarray]:
    candidates: list[dict[int, np.ndarray]] = []
    for size in (4, 3):
        candidates.clear()
        for side in SIDES:
            mapping = _legend_side_mapping(grid, needed_colors, side, size)
            if mapping is not None:
                candidates.append(mapping)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            same = all(
                all(np.array_equal(candidates[0][color], cand[color]) for color in needed_colors)
                for cand in candidates[1:]
            )
            if same:
                return candidates[0]
            raise AssertionError("legend detection is ambiguous across edges")
    raise AssertionError("failed to find a valid legend mapping on any edge")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed grey-walled region with the tiled 3x3 texture indexed by its seed color.

    Input structure:
    - black background
    - one or more closed rectangular regions bordered by grey walls
    - exactly one non-black, non-grey seed pixel inside each region
    - a legend on one grid edge, where each block's top-left pixel gives the seed color and the
      associated 3x3 two-color template is tiled into matching regions
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    regions = _enclosed_regions(g)
    needed_colors = {seed_color for _coords, _bbox_coords, seed_color in regions}
    legend = _legend_templates(g, needed_colors)

    out = g.copy()
    for coords, (r0, c0, _r1, _c1), seed_color in regions:
        template = legend[seed_color]
        for r, c in coords:
            out[r, c] = int(template[(r - r0) % LEGEND_TEMPLATE, (c - c0) % LEGEND_TEMPLATE])
    return out


def _split_total(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    random.shuffle(values)
    return values


def _legend_starts(length: int, count: int) -> list[int] | None:
    needed = LEGEND_BLOCK * count + LEGEND_GAP * (count - 1)
    if needed > length:
        return None
    extra = length - needed
    slack = _split_total(extra, count + 1)
    starts = []
    pos = slack[0]
    for idx in range(count):
        starts.append(pos)
        pos += LEGEND_BLOCK
        if idx + 1 < count:
            pos += LEGEND_GAP + slack[idx + 1]
    return starts


def _random_template(color: int, used: set[tuple[tuple[int, ...], ...]]) -> np.ndarray | None:
    for _ in range(200):
        mask = np.random.random((3, 3)) < random.uniform(0.35, 0.7)
        mask[0, 0] = False
        colored = int(mask.sum())
        if not 3 <= colored <= 6:
            continue
        if int(np.count_nonzero(mask.any(axis=0))) < 2:
            continue
        if int(np.count_nonzero(mask.any(axis=1))) < 2:
            continue
        signature = tuple(tuple(int(v) for v in row) for row in mask.astype(int))
        if signature in used:
            continue
        used.add(signature)
        template = np.zeros((3, 3), dtype=int)
        template[mask] = color
        return template
    return None


def _legend_patch(color: int, template: np.ndarray) -> np.ndarray:
    patch = np.zeros((4, 4), dtype=int)
    patch[0, 0] = color
    patch[1:, 1:] = template
    return patch


def _boxes_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (
        a[2] + 1 < b[0]
        or b[2] + 1 < a[0]
        or a[3] + 1 < b[1]
        or b[3] + 1 < a[1]
    )


def _play_bounds(side: str, h: int, w: int) -> tuple[int, int, int, int]:
    if side == "left":
        return 1, h - 2, 5, w - 2
    if side == "right":
        return 1, h - 2, 1, w - 6
    if side == "top":
        return 5, h - 2, 1, w - 2
    return 1, h - 6, 1, w - 2


def _place_legend(grid: np.ndarray, side: str, colors: list[int], templates: dict[int, np.ndarray]) -> bool:
    axis = grid.shape[0] if side in ("left", "right") else grid.shape[1]
    starts = _legend_starts(axis, len(colors))
    if starts is None:
        return False
    order = colors[:]
    random.shuffle(order)

    for color, start in zip(order, starts, strict=True):
        patch = _legend_patch(color, templates[color])
        if side == "left":
            grid[start : start + 4, 0:4] = patch
        elif side == "right":
            grid[start : start + 4, grid.shape[1] - 4 : grid.shape[1]] = patch
        elif side == "top":
            grid[0:4, start : start + 4] = patch
        else:
            grid[grid.shape[0] - 4 : grid.shape[0], start : start + 4] = patch
    return True


def _region_specs(count: int, play_h: int, play_w: int, colors: list[int]) -> list[tuple[int, int, int]]:
    size_hi = 7 if count <= 3 else 6 if count == 4 else 5
    size_hi = min(size_hi, play_h - 2, play_w - 2)
    if size_hi < 3:
        return []

    region_colors = colors[:] + [random.choice(colors) for _ in range(count - len(colors))]
    random.shuffle(region_colors)
    specs = []
    for color in region_colors:
        inner_h = random.randint(3, size_hi)
        inner_w = random.randint(3, size_hi)
        specs.append((inner_h, inner_w, color))
    specs.sort(key=lambda item: item[0] * item[1], reverse=True)
    return specs


def _place_regions(
    grid: np.ndarray,
    side: str,
    count: int,
    colors: list[int],
) -> list[tuple[tuple[int, int, int, int], int]] | None:
    min_r, max_r, min_c, max_c = _play_bounds(side, *grid.shape)
    play_h = max_r - min_r + 1
    play_w = max_c - min_c + 1
    specs = _region_specs(count, play_h, play_w, colors)
    if len(specs) != count:
        return None

    placed: list[tuple[tuple[int, int, int, int], int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for inner_h, inner_w, color in specs:
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        if outer_h > play_h or outer_w > play_w:
            return None
        for _ in range(300):
            r0 = random.randint(min_r, max_r - outer_h + 1)
            c0 = random.randint(min_c, max_c - outer_w + 1)
            box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(_boxes_touch(box, other) for other in boxes):
                continue
            boxes.append(box)
            placed.append((box, color))
            break
        else:
            return None
    return placed


def _draw_regions(grid: np.ndarray, regions: list[tuple[tuple[int, int, int, int], int]]) -> None:
    for (r0, c0, r1, c1), color in regions:
        grid[r0 : r1 + 1, c0] = WALL
        grid[r0 : r1 + 1, c1] = WALL
        grid[r0, c0 : c1 + 1] = WALL
        grid[r1, c0 : c1 + 1] = WALL
        seed_r = random.randint(r0 + 1, r1 - 1)
        seed_c = random.randint(c0 + 1, c1 - 1)
        grid[seed_r, seed_c] = color


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        regions = _enclosed_regions(grid)
        legend = _legend_templates(grid, {color for _coords, _bbox_coords, color in regions})
        out = main(grid)
    except AssertionError:
        return False

    changed = False
    for coords, (r0, c0, _r1, _c1), color in regions:
        template = legend[color]
        for r, c in coords:
            expected = int(template[(r - r0) % 3, (c - c0) % 3])
            if int(out[r, c]) != expected:
                return False
            if int(grid[r, c]) != expected:
                changed = True
    return changed


def _generate_attempt() -> np.ndarray | None:
    side = random.choice(SIDES)
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    grid = np.zeros((h, w), dtype=int)

    axis = h if side in ("left", "right") else w
    max_colors = min(4, (axis + 1) // (LEGEND_BLOCK + LEGEND_GAP))
    if max_colors < 2:
        return None

    color_count = random.randint(2, max_colors)
    colors = random.sample(COLOR_POOL, color_count)

    templates: dict[int, np.ndarray] = {}
    used_signatures: set[tuple[tuple[int, ...], ...]] = set()
    for color in colors:
        template = _random_template(color, used_signatures)
        if template is None:
            return None
        templates[color] = template

    if not _place_legend(grid, side, colors, templates):
        return None

    region_count = random.randint(color_count, min(5, color_count + 2))
    regions = _place_regions(grid, side, region_count, colors)
    if regions is None:
        return None
    _draw_regions(grid, regions)

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
