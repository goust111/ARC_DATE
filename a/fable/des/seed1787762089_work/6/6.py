import random
from collections import defaultdict
from itertools import combinations

import numpy as np


BACKGROUND = 0
COLOR_POOL = tuple(range(1, 10))
GRID_MIN = 18
GRID_MAX = 24
GENERATION_RETRIES = 600
PLACEMENT_RETRIES = 300

SHAPES = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _stamp_shape(
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    top: int,
    left: int,
    color: int,
) -> None:
    for dr, dc in shape:
        grid[top + dr, left + dc] = color


def _reserve_rect(
    blocked: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    margin: int,
) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _rect_is_free(
    blocked: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    margin: int,
) -> bool:
    h, w = blocked.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    return not bool(np.any(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1]))


def _legend_candidates(grid: np.ndarray, bg: int) -> list[tuple[dict[int, int], np.ndarray]]:
    h, w = grid.shape
    candidates: list[tuple[dict[int, int], np.ndarray]] = []
    for rows in (2, 3):
        if h < rows or w < 2:
            continue
        boxes = [
            (0, 0),
            (0, w - 2),
            (h - rows, 0),
            (h - rows, w - 2),
        ]
        for r0, c0 in boxes:
            r1 = r0 + rows - 1
            c1 = c0 + 1
            block = grid[r0 : r1 + 1, c0 : c1 + 1]
            if block.shape != (rows, 2) or np.any(block == bg):
                continue
            left = [int(v) for v in block[:, 0]]
            right = [int(v) for v in block[:, 1]]
            if len(set(left)) != rows or any(a == b for a, b in zip(left, right)):
                continue

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h - 1, r1 + 1)
            cc1 = min(w - 1, c1 + 1)
            patch = grid[rr0 : rr1 + 1, cc0 : cc1 + 1]
            keep = np.zeros_like(patch, dtype=bool)
            keep[r0 - rr0 : r1 - rr0 + 1, c0 - cc0 : c1 - cc0 + 1] = True
            if np.any((patch != bg) & ~keep):
                continue

            mask = np.zeros_like(grid, dtype=bool)
            mask[r0 : r1 + 1, c0 : c1 + 1] = True
            candidates.append((dict(zip(left, right)), mask))
    return candidates


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    candidates = _legend_candidates(grid, bg)
    assert len(candidates) == 1, "expected a unique isolated corner legend"
    return candidates[0]


def _find_components(
    grid: np.ndarray,
    bg: int,
    forbidden: np.ndarray,
) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = forbidden.copy()
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or forbidden[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            shape = tuple(sorted((int(rr - r0), int(cc - c0)) for rr, cc in arr))
            components.append(
                {
                    "id": len(components),
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "shape": shape,
                }
            )
    return components


def _connection_region(
    grid: np.ndarray,
    bg: int,
    forbidden: np.ndarray,
    a: dict[str, object],
    b: dict[str, object],
) -> tuple[str, tuple[slice, slice]] | None:
    if a["shape"] != b["shape"]:
        return None

    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    if ar0 == br0 and ar1 == br1:
        if ac1 < bc0:
            left, right = a, b
        elif bc1 < ac0:
            left, right = b, a
        else:
            return None
        r0 = ar0
        r1 = ar1
        c0 = left["bbox"][3] + 1
        c1 = right["bbox"][1] - 1
        if c1 - c0 + 1 < 3:
            return None
        orientation = "h"
    elif ac0 == bc0 and ac1 == bc1:
        if ar1 < br0:
            top, bottom = a, b
        elif br1 < ar0:
            top, bottom = b, a
        else:
            return None
        r0 = top["bbox"][2] + 1
        r1 = bottom["bbox"][0] - 1
        c0 = ac0
        c1 = ac1
        if r1 - r0 + 1 < 3:
            return None
        orientation = "v"
    else:
        return None

    region = (slice(r0, r1 + 1), slice(c0, c1 + 1))
    if np.any(forbidden[region]) or np.any(grid[region] != bg):
        return None
    return orientation, region


def _qualifying_pairs(
    grid: np.ndarray,
    bg: int,
    mapping: dict[int, int],
    forbidden: np.ndarray,
) -> list[tuple[int, tuple[slice, slice]]]:
    components = _find_components(grid, bg, forbidden)
    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for comp in components:
        by_color[int(comp["color"])].append(comp)

    pairs: list[tuple[int, tuple[slice, slice]]] = []
    used_ids: set[int] = set()
    for color, comps in by_color.items():
        for a, b in combinations(comps, 2):
            region_info = _connection_region(grid, bg, forbidden, a, b)
            if region_info is None:
                continue
            assert color in mapping, "qualifying pair color must appear in the legend"
            assert int(a["id"]) not in used_ids and int(b["id"]) not in used_ids, "ambiguous pairing"
            used_ids.add(int(a["id"]))
            used_ids.add(int(b["id"]))
            pairs.append((color, region_info[1]))

    assert {color for color, _ in pairs} == set(mapping), "legend colors must match the transform pairs"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color, same-shape aligned object pair using the legend's color map.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    mapping, legend_mask = _find_legend(g, bg)
    pairs = _qualifying_pairs(g, bg, mapping, legend_mask)

    out = g.copy()
    for color, region in pairs:
        assert np.all(out[region] == bg)
        out[region] = mapping[color]
    return out


def _choice_position(limit: int) -> int:
    if limit <= 2:
        return 0
    return random.randint(1, limit - 2)


def _place_free_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    color: int,
    banned_row_band: tuple[int, int] | None = None,
    banned_col_band: tuple[int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    sh, sw = _shape_dims(shape)
    max_r = h - sh
    max_c = w - sw
    for _ in range(PLACEMENT_RETRIES):
        top = _choice_position(max_r)
        left = _choice_position(max_c)
        bbox = (top, left, top + sh - 1, left + sw - 1)
        if banned_row_band is not None and bbox[0] == banned_row_band[0] and bbox[2] == banned_row_band[1]:
            continue
        if banned_col_band is not None and bbox[1] == banned_col_band[0] and bbox[3] == banned_col_band[1]:
            continue
        if not _rect_is_free(blocked, *bbox, margin=1):
            continue
        _stamp_shape(grid, shape, top, left, color)
        _reserve_rect(blocked, *bbox, margin=1)
        return bbox
    return None


def _place_pair(
    grid: np.ndarray,
    blocked: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    color: int,
    orientation: str,
) -> bool:
    h, w = grid.shape
    sh, sw = _shape_dims(shape)

    for _ in range(PLACEMENT_RETRIES):
        if orientation == "h":
            if w < 2 * sw + 5 or h < sh + 2:
                return False
            top = random.randint(1, h - sh - 1)
            left = random.randint(1, w - (2 * sw + 3) - 1)
            max_gap = min(8, w - left - 2 * sw - 1)
            if max_gap < 3:
                continue
            gap = random.randint(3, max_gap)
            right = left + sw + gap
            strip = (top, left, top + sh - 1, right + sw - 1)
            if not _rect_is_free(blocked, *strip, margin=1):
                continue
            _stamp_shape(grid, shape, top, left, color)
            _stamp_shape(grid, shape, top, right, color)
            _reserve_rect(blocked, *strip, margin=1)
            return True

        if h < 2 * sh + 5 or w < sw + 2:
            return False
        left = random.randint(1, w - sw - 1)
        top = random.randint(1, h - (2 * sh + 3) - 1)
        max_gap = min(8, h - top - 2 * sh - 1)
        if max_gap < 3:
            continue
        gap = random.randint(3, max_gap)
        bottom = top + sh + gap
        strip = (top, left, bottom + sh - 1, left + sw - 1)
        if not _rect_is_free(blocked, *strip, margin=1):
            continue
        _stamp_shape(grid, shape, top, left, color)
        _stamp_shape(grid, shape, bottom, left, color)
        _reserve_rect(blocked, *strip, margin=1)
        return True

    return False


def _place_legend(
    grid: np.ndarray,
    blocked: np.ndarray,
    source_colors: list[int],
    line_colors: list[int],
    corner: str,
) -> None:
    h, w = grid.shape
    rows = len(source_colors)
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - 2
    elif corner == "bl":
        r0, c0 = h - rows, 0
    else:
        r0, c0 = h - rows, w - 2

    for i, (src, dst) in enumerate(zip(source_colors, line_colors)):
        grid[r0 + i, c0] = src
        grid[r0 + i, c0 + 1] = dst
    _reserve_rect(blocked, r0, c0, r0 + rows - 1, c0 + 1, margin=2)


def _choose_distractor_specs(remaining_objects: int) -> list[str]:
    specs: list[str] = []
    while remaining_objects > 0:
        options = ["singleton"]
        if remaining_objects >= 2:
            options.extend(["same_shape_pair", "different_shape_pair"])
        kind = random.choice(options)
        specs.append(kind)
        remaining_objects -= 1 if kind == "singleton" else 2
    return specs


def _build_sample() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BACKGROUND, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    pair_count = random.randint(2, 3)
    remaining_objects = random.randint(max(1, 6 - 2 * pair_count), min(3, 9 - 2 * pair_count))
    distractor_specs = _choose_distractor_specs(remaining_objects)
    total_group_colors = len(distractor_specs)

    chosen = random.sample(COLOR_POOL, 2 * pair_count + total_group_colors)
    source_colors = chosen[:pair_count]
    line_colors = chosen[pair_count : 2 * pair_count]
    distractor_colors = chosen[2 * pair_count :]
    random.shuffle(source_colors)
    random.shuffle(line_colors)

    _place_legend(grid, blocked, source_colors, line_colors, random.choice(["tl", "tr", "bl", "br"]))

    orientations = ["h", "v"]
    if pair_count == 3:
        orientations.append(random.choice(["h", "v"]))
    random.shuffle(orientations)

    for color, orientation in zip(source_colors, orientations):
        shape = random.choice(SHAPES)
        if not _place_pair(grid, blocked, shape, color, orientation):
            return None

    for color, spec in zip(distractor_colors, distractor_specs):
        if spec == "singleton":
            if _place_free_object(grid, blocked, random.choice(SHAPES), color) is None:
                return None
            continue

        if spec == "same_shape_pair":
            shape = random.choice(SHAPES)
            bbox = _place_free_object(grid, blocked, shape, color)
            if bbox is None:
                return None
            if _place_free_object(
                grid,
                blocked,
                shape,
                color,
                banned_row_band=(bbox[0], bbox[2]),
                banned_col_band=(bbox[1], bbox[3]),
            ) is None:
                return None
            continue

        shape_a, shape_b = random.sample(SHAPES, 2)
        if _place_free_object(grid, blocked, shape_a, color) is None:
            return None
        if _place_free_object(grid, blocked, shape_b, color) is None:
            return None

    try:
        out = main(grid)
        bg = _get_bg(grid)
        mapping, legend_mask = _find_legend(grid, bg)
        pairs = _qualifying_pairs(grid, bg, mapping, legend_mask)
    except AssertionError:
        return None

    if len(pairs) != pair_count or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _build_sample()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
