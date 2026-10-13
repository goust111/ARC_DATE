from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
AZURE = 8
PALETTE = [1, 2, 3, 4, 6, 7, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MAX_GENERATION_ATTEMPTS = 300
NUM_REGIONS_OPTIONS = (2, 3)
AREA_RANGE = (8, 20)
BODY_H_RANGE = (12, 22)
BODY_W_RANGE = (16, 28)
FORCE_TIE_PROB = 0.5


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _parse_reference_segment(segment: np.ndarray) -> list[int] | None:
    seg = [int(v) for v in segment]
    if len(seg) < 3:
        return None
    for starts_with_grey in (True, False):
        colors: list[int] = []
        ok = True
        for idx, value in enumerate(seg):
            want_grey = (idx % 2 == 0) if starts_with_grey else (idx % 2 == 1)
            if want_grey:
                if value != GREY:
                    ok = False
                    break
            else:
                if value in (BG, GREY):
                    ok = False
                    break
                colors.append(value)
        if ok and len(colors) >= 2 and len(colors) == len(set(colors)):
            return colors
    return None


def _extract_reference_strip(grid: np.ndarray) -> tuple[int, list[int]]:
    bg = _background_color(grid)
    h, _w = grid.shape
    for r in range(h - 1, -1, -1):
        row = grid[r]
        non_bg = np.flatnonzero(row != bg)
        if non_bg.size < 3:
            continue
        left, right = int(non_bg[0]), int(non_bg[-1])
        segment = row[left : right + 1]
        if np.any(segment == bg):
            continue
        colors = _parse_reference_segment(segment)
        if colors is not None:
            return r, colors
    raise AssertionError("failed to locate reference strip")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor enclosed regions by sorting them on their internal azure-marker counts and applying the bottom reference strip colors.

    Input structure:
    - Black background surrounds several grey-outlined closed regions in the main body.
    - Each region interior is one connected non-grey area with a single fill color plus 0-4 azure marker pixels.
    - A bottom reference strip contains singleton color swatches separated by grey pixels.

    Rule:
    - Locate the reference strip and read its colors from left to right.
    - Find every enclosed region interior above the strip and count how many azure pixels it contains.
    - Sort regions by marker count ascending, breaking ties by region position from left to right and then top to bottom.
    - Repaint the i-th sorted region with the i-th strip color, replacing azure markers with that new fill color while leaving grey outlines and the strip unchanged.

    Key insight: The azure pixels are not output objects; they encode a hidden sort key that tells the solver how to assign the visible colors from the bottom strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    strip_row, strip_colors = _extract_reference_strip(g)
    body = g[:strip_row]
    bg = _background_color(g)

    region_mask = (body != bg) & (body != GREY)
    components = _components_mask(region_mask)
    assert len(components) == len(strip_colors), "region count must match reference strip"

    regions: list[tuple[int, int, int, np.ndarray]] = []
    for coords in components:
        values = body[coords[:, 0], coords[:, 1]]
        marker_count = int(np.count_nonzero(values == AZURE))
        top = int(coords[:, 0].min())
        left = int(coords[:, 1].min())
        regions.append((marker_count, left, top, coords))

    regions.sort(key=lambda item: (item[0], item[1], item[2]))

    out = g.copy()
    for new_color, (_count, _left, _top, coords) in zip(strip_colors, regions):
        out[coords[:, 0], coords[:, 1]] = int(new_color)

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _shape_has_hole(cells: set[tuple[int, int]]) -> bool:
    cells = _normalize_cells(cells)
    h = max(r for r, _c in cells) + 1
    w = max(c for _r, c in cells) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    for comp in _components_mask(~mask):
        if not np.any(
            (comp[:, 0] == 0)
            | (comp[:, 0] == h - 1)
            | (comp[:, 1] == 0)
            | (comp[:, 1] == w - 1)
        ):
            return True
    return False


def _grow_shape(area: int, rng: random.Random) -> set[tuple[int, int]] | None:
    for _ in range(80):
        h = rng.randint(3, 6)
        w = rng.randint(3, 7)
        if h * w < area:
            continue
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != area:
            continue
        cells = _normalize_cells(cells)
        box_h = max(r for r, _c in cells) + 1
        box_w = max(c for _r, c in cells) + 1
        if len(cells) == box_h * box_w:
            continue
        if _shape_has_hole(cells):
            continue
        return cells
    return None


def _render_region_patch(
    cells: set[tuple[int, int]],
    fill_color: int,
    marker_count: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    norm = _normalize_cells(cells)
    shifted = {(r + 1, c + 1) for r, c in norm}
    h = max(r for r, _c in shifted) + 2
    w = max(c for _r, c in shifted) + 2

    patch = np.full((h, w), BG, dtype=int)
    for r, c in shifted:
        patch[r, c] = int(fill_color)

    boundary = set()
    for r, c in shifted:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in shifted:
                boundary.add((nr, nc))
    for r, c in boundary:
        patch[r, c] = GREY

    shifted_list = sorted(shifted)
    markers = rng.sample(shifted_list, marker_count)
    for r, c in markers:
        patch[r, c] = AZURE

    interior = np.array(sorted(shifted), dtype=int)
    return patch, interior


def _sample_marker_counts(num_regions: int, rng: random.Random) -> list[int]:
    if rng.random() >= FORCE_TIE_PROB:
        return rng.sample(range(5), num_regions)

    counts = [rng.randint(0, 4) for _ in range(num_regions)]
    if num_regions == 2:
        shared = rng.randint(0, 4)
        return [shared, shared]

    shared = rng.randint(0, 4)
    counts[0] = shared
    counts[1] = shared
    if rng.random() < 0.5:
        counts[2] = shared
    return counts


def _place_patch(
    canvas: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = canvas.shape
    ph, pw = patch.shape
    if ph + 2 > h or pw + 2 > w:
        return None

    positions = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            r1, c1 = r0 + ph, c0 + pw
            if blocked[r0 - 1 : r1 + 1, c0 - 1 : c1 + 1].any():
                continue
            positions.append((r0, c0))
    if not positions:
        return None

    r0, c0 = rng.choice(positions)
    mask = patch != BG
    canvas[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
    blocked[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1] = True
    return r0, c0


def _build_reference_row(width: int, colors: list[int], rng: random.Random) -> np.ndarray:
    row = np.full(width, BG, dtype=int)
    segment = [GREY]
    for color in colors:
        segment.extend([int(color), GREY])
    seg = np.array(segment, dtype=int)
    start = rng.randint(0, width - len(seg))
    row[start : start + len(seg)] = seg
    return row


def _expected_output(
    grid: np.ndarray,
    region_infos: list[dict[str, object]],
    strip_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    ordered = sorted(
        region_infos,
        key=lambda info: (
            int(info["marker_count"]),
            int(info["left"]),
            int(info["top"]),
        ),
    )
    for new_color, info in zip(strip_colors, ordered):
        coords = info["coords"]
        assert isinstance(coords, np.ndarray)
        out[coords[:, 0], coords[:, 1]] = int(new_color)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_regions = rng.choice(NUM_REGIONS_OPTIONS)
    chosen_colors = rng.sample(PALETTE, 2 * num_regions)
    fill_colors = chosen_colors[:num_regions]
    strip_colors = chosen_colors[num_regions:]
    marker_counts = _sample_marker_counts(num_regions, rng)

    regions = []
    for fill_color, marker_count in zip(fill_colors, marker_counts):
        area = rng.randint(*AREA_RANGE)
        cells = _grow_shape(area, rng)
        if cells is None or marker_count > len(cells):
            return None
        patch, local_interior = _render_region_patch(cells, fill_color, marker_count, rng)
        regions.append(
            {
                "patch": patch,
                "local_interior": local_interior,
                "marker_count": marker_count,
            }
        )

    min_h = max(int(region["patch"].shape[0]) for region in regions) + 2
    min_w = max(
        max(int(region["patch"].shape[1]) for region in regions),
        2 * num_regions + 1,
    ) + 2
    body_h = rng.randint(max(BODY_H_RANGE[0], min_h), BODY_H_RANGE[1])
    body_w = rng.randint(max(BODY_W_RANGE[0], min_w), BODY_W_RANGE[1])

    body = np.full((body_h, body_w), BG, dtype=int)
    blocked = np.zeros((body_h, body_w), dtype=bool)
    rng.shuffle(regions)

    region_infos: list[dict[str, object]] = []
    for region in regions:
        patch = region["patch"]
        assert isinstance(patch, np.ndarray)
        placed = _place_patch(body, blocked, patch, rng)
        if placed is None:
            return None
        r0, c0 = placed
        local_interior = region["local_interior"]
        assert isinstance(local_interior, np.ndarray)
        global_coords = local_interior + np.array([r0, c0], dtype=int)
        region_infos.append(
            {
                "coords": global_coords,
                "marker_count": int(region["marker_count"]),
                "top": int(global_coords[:, 0].min()),
                "left": int(global_coords[:, 1].min()),
            }
        )

    grid = np.full((body_h + 2, body_w), BG, dtype=int)
    grid[:body_h] = body
    grid[-1] = _build_reference_row(body_w, strip_colors, rng)

    expected = _expected_output(grid, region_infos, strip_colors)

    counts = Counter(map(int, grid.ravel()))
    if counts[BG] <= max((n for c, n in counts.items() if c != BG), default=0):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
