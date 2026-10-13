from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_MIN = 18
GRID_MAX = 30
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        dirs = DIR4
    else:
        dirs = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(points: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _is_rectangular_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    point_set = set(points)
    expected = 2 * (height + width) - 4
    if len(point_set) != expected:
        return False
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            present = (r, c) in point_set
            if on_border != present:
                return False
    return True


def _internal_bg_regions(
    grid: np.ndarray, bg: int
) -> list[dict[str, object]]:
    h, w = grid.shape
    regions: list[dict[str, object]] = []
    for comp in _components(grid == bg, connectivity=4):
        if _touches_border(comp, h, w):
            continue
        boundary_colors: set[int] = set()
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != bg:
                    boundary_colors.add(int(grid[nr, nc]))
        regions.append(
            {
                "points": comp,
                "bbox": _bbox(comp),
                "boundary_colors": boundary_colors,
            }
        )
    return regions


def _structure_color(grid: np.ndarray, bg: int, regions: list[dict[str, object]]) -> int:
    color_hits: Counter[int] = Counter()
    for region in regions:
        boundary_colors = region["boundary_colors"]
        if len(boundary_colors) == 1:
            color_hits[next(iter(boundary_colors))] += 1

    non_bg = [int(v) for v in np.unique(grid) if int(v) != bg]
    frame_candidates: list[tuple[int, int, int]] = []
    for color in non_bg:
        comps = _components(grid == color, connectivity=4)
        if comps and all(_is_rectangular_frame(comp) for comp in comps):
            total_pixels = sum(len(comp) for comp in comps)
            frame_candidates.append((color_hits[color], total_pixels, color))

    if frame_candidates:
        frame_candidates.sort(reverse=True)
        return int(frame_candidates[0][2])

    if color_hits:
        return int(max(color_hits.items(), key=lambda item: item[1])[0])
    raise AssertionError("could not identify structure color")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each nested-frame region with the external sprite color matched by
    descending sprite area: largest sprite -> outermost region, smallest -> innermost.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    regions = _internal_bg_regions(g, bg)
    assert regions, "expected enclosed background regions"

    structure = _structure_color(g, bg, regions)
    target_regions = [region for region in regions if region["boundary_colors"] == {structure}]
    assert target_regions, "expected structure-bounded regions"

    target_regions.sort(
        key=lambda region: (
            -(
                (region["bbox"][2] - region["bbox"][0] + 1)
                * (region["bbox"][3] - region["bbox"][1] + 1)
            ),
            region["bbox"][0],
            region["bbox"][1],
        )
    )

    sprite_mask = (g != bg) & (g != structure)
    sprite_components = _components(sprite_mask, connectivity=4)
    sprites: list[tuple[int, int]] = []
    for comp in sprite_components:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each sprite must be monochrome"
        sprites.append((len(comp), next(iter(colors))))

    assert len(sprites) == len(target_regions), "sprite count must match region count"
    sprites.sort(key=lambda item: (-item[0], item[1]))
    assert len({area for area, _ in sprites}) == len(sprites), "sprite areas must be distinct"

    out = g.copy()
    for region, (_area, color) in zip(target_regions, sprites):
        for r, c in region["points"]:
            out[r, c] = color
    return out


def _normalize(points: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    shifted = {(r - min_r, c - min_c) for r, c in points}
    max_r = max(r for r, _ in shifted)
    max_c = max(c for _, c in shifted)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in shifted:
        mask[r, c] = True
    return mask


def _transform_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = mask.copy()
    if rng.random() < 0.5:
        out = np.flipud(out)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    for _ in range(rng.randrange(4)):
        out = np.rot90(out)
    return out.copy()


def _random_polyomino(area: int, rng: random.Random) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.add(nxt)
        cells.add(rng.choice(list(frontier)))
    return _transform_mask(_normalize(cells), rng)


def _sample_frames(layer_count: int, rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    for _ in range(200):
        core_h = rng.randint(2, 6)
        core_w = rng.randint(2, 6)
        frames = [(0, 0, core_h + 1, core_w + 1)]  # inner -> outer
        top, left, bottom, right = frames[0]
        for _ in range(layer_count - 1):
            gap_t = rng.randint(1, 3)
            gap_b = rng.randint(1, 3)
            gap_l = rng.randint(1, 3)
            gap_r = rng.randint(1, 3)
            top -= gap_t + 1
            left -= gap_l + 1
            bottom += gap_b + 1
            right += gap_r + 1
            frames.append((top, left, bottom, right))

        min_r = min(r0 for r0, _, _, _ in frames)
        min_c = min(c0 for _, c0, _, _ in frames)
        shifted = [
            (r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c) for r0, c0, r1, c1 in frames
        ]
        shifted.sort(
            key=lambda box: -((box[2] - box[0] + 1) * (box[3] - box[1] + 1))
        )
        total_h = shifted[0][2] + 1
        total_w = shifted[0][3] + 1
        if total_h <= GRID_MAX - 6 and total_w <= GRID_MAX - 6:
            return shifted
    return None


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _region_masks_from_frames(
    frames: list[tuple[int, int, int, int]], shape: tuple[int, int]
) -> list[np.ndarray]:
    h, w = shape
    regions: list[np.ndarray] = []
    for i, (r0, c0, r1, c1) in enumerate(frames):
        region = np.zeros((h, w), dtype=bool)
        region[r0 + 1 : r1, c0 + 1 : c1] = True
        if i + 1 < len(frames):
            nr0, nc0, nr1, nc1 = frames[i + 1]
            region[nr0 : nr1 + 1, nc0 : nc1 + 1] = False
        regions.append(region)
    return regions


def _dilate_into(mask: np.ndarray, points: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in points:
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                mask[nr, nc] = True


def _place_sprites(
    grid: np.ndarray,
    outer_box: tuple[int, int, int, int],
    sprites: list[tuple[int, int, np.ndarray]],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = outer_box
    blocked = np.zeros((h, w), dtype=bool)
    blocked[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True

    for _area, color, mask in sorted(sprites, key=lambda item: -item[0]):
        ph, pw = mask.shape
        placed = False
        for _ in range(300):
            rr = rng.randint(0, h - ph)
            cc = rng.randint(0, w - pw)
            coords = [(rr + dr, cc + dc) for dr, dc in np.argwhere(mask)]
            if any(blocked[r, c] for r, c in coords):
                continue
            for r, c in coords:
                grid[r, c] = color
            _dilate_into(blocked, coords)
            placed = True
            break
        if not placed:
            return False
    return True


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    layer_count = rng.randint(2, 5)
    frames = _sample_frames(layer_count, rng)
    if frames is None:
        return None

    struct_h = frames[0][2] + 1
    struct_w = frames[0][3] + 1
    min_h = max(GRID_MIN, struct_h + 6)
    min_w = max(GRID_MIN, struct_w + 6)
    max_h = min(GRID_MAX, struct_h + 14)
    max_w = min(GRID_MAX, struct_w + 14)
    if min_h > max_h or min_w > max_w:
        return None

    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)
    top = rng.randint(1, h - struct_h - 1)
    left = rng.randint(1, w - struct_w - 1)
    abs_frames = [
        (r0 + top, c0 + left, r1 + top, c1 + left) for r0, c0, r1, c1 in frames
    ]

    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    structure = rng.choice(colors)
    colors.remove(structure)

    grid = np.full((h, w), bg, dtype=int)
    for box in abs_frames:
        _draw_frame(grid, box, structure)

    area_choices = rng.sample(range(1, 10), layer_count)
    sprite_colors = rng.sample(colors, layer_count)
    sprite_specs = [
        (area, color, _random_polyomino(area, rng))
        for area, color in zip(area_choices, sprite_colors)
    ]
    if not _place_sprites(grid, abs_frames[0], sprite_specs, rng):
        return None

    counts = Counter(int(v) for v in grid.ravel())
    if counts[bg] <= max(counts[c] for c in range(10) if c != bg):
        return None
    if counts[structure] <= max(area for area, _color, _mask in sprite_specs):
        return None

    out = grid.copy()
    sorted_sprites = sorted(sprite_specs, key=lambda item: -item[0])
    for region_mask, (_area, color, _mask) in zip(_region_masks_from_frames(abs_frames, grid.shape), sorted_sprites):
        out[region_mask] = color

    actual = main(grid)
    if not np.array_equal(actual, out):
        return None
    return grid, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
