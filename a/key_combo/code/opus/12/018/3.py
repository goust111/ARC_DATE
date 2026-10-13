from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
WALL = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GEN_ATTEMPTS = 400
HEIGHT_RANGE = (18, 24)
WIDTH_RANGE = (18, 24)
MIN_REGION_HEIGHT = 5
MIN_REGION_WIDTH = 4


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            idx = 0
            while idx < len(queue):
                rr, cc = queue[idx]
                idx += 1
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boundary_distance(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    dist = np.full((h, w), -1, dtype=int)
    queue: list[tuple[int, int]] = []

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        boundary = False
        for dr, dc in DIR8:
            nr = rr + dr
            nc = cc + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w or not mask[nr, nc]:
                boundary = True
                break
        if boundary:
            dist[rr, cc] = 0
            queue.append((rr, cc))

    idx = 0
    while idx < len(queue):
        r, c = queue[idx]
        idx += 1
        nd = dist[r, c] + 1
        for dr, dc in DIR8:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = nd
                queue.append((nr, nc))
    return dist


def _template_roles(template: np.ndarray) -> tuple[int, int]:
    vals = [int(v) for v in template]
    assert all(v not in (BG, WALL) for v in vals), "template row must be fully colored"
    first = vals[0]
    other = next((v for v in vals if v != first), None)
    assert other is not None, "template row must contain two colors"
    expected = np.asarray([first if i % 2 == 0 else other for i in range(len(vals))], dtype=int)
    assert np.array_equal(template, expected), "template row must alternate exactly two colors"
    return int(first), int(other)


def _decode_palette(template: np.ndarray, prefix: np.ndarray) -> tuple[tuple[int, int], int]:
    a, b = _template_roles(template)
    filled = np.flatnonzero(prefix != BG)
    assert len(filled) >= 2, "prefix must show both template roles"
    prefix_len = int(filled[-1]) + 1
    assert prefix_len < len(prefix), "prefix row must leave background to the right"
    assert np.array_equal(filled, np.arange(prefix_len)), "prefix must be contiguous from the left edge"
    assert np.all(prefix[prefix_len:] == BG), "only the left prefix may be colored"

    mapping: dict[int, int] = {}
    for c in range(prefix_len):
        role = int(template[c])
        color = int(prefix[c])
        assert color not in (BG, WALL), "prefix colors must be visible non-wall colors"
        if role in mapping:
            assert mapping[role] == color, "template role mapping must be consistent"
        else:
            mapping[role] = color

    assert set(mapping) == {a, b}, "prefix must reveal both template colors"
    return (int(mapping[a]), int(mapping[b])), prefix_len


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a two-color palette from each region's template-prefix pair and fill the lower area with concentric bands.

    Input structure:
    - A black background grid is partitioned into rectangular regions by grey full-row or full-column walls.
    - Inside each region, the top row is a full-width alternating two-color template strip.
    - The next row is a left-aligned colored prefix whose columnwise pairing with the template decodes that region's two output colors.

    Rule:
    - For every region, match each colored prefix cell with the template cell directly above it to recover the mapping from the two template colors to two palette colors.
    - Treat the region below those two rows as the fillable subregion, compute every fill cell's Chebyshev distance to that subregion's boundary, and color it with palette[distance mod 2].
    - Keep the template strip, prefix strip, and all grey walls unchanged.

    Key insight:
    - The template row gives symbolic roles, the prefix row decodes those roles into actual colors, and only that decoded palette tells you how to paint the concentric bands in each separate region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for coords in _components(g != WALL):
        r0, c0, r1, c1 = _bbox(coords)
        assert r1 - r0 + 1 >= 3, "region must have at least template, prefix, and fill rows"
        template = g[r0, c0 : c1 + 1]
        prefix = g[r0 + 1, c0 : c1 + 1]
        palette, _ = _decode_palette(template, prefix)

        region_mask = np.zeros_like(g, dtype=bool)
        region_mask[coords[:, 0], coords[:, 1]] = True
        fill_mask = region_mask.copy()
        fill_mask[: r0 + 2, :] = False
        if not np.any(fill_mask):
            continue

        dist = _boundary_distance(fill_mask)
        for r, c in np.argwhere(fill_mask & (g == BG)):
            out[int(r), int(c)] = int(palette[int(dist[int(r), int(c)]) % 2])
    return out


def _split_sizes(total: int, parts: int, minimum: int, rng: random.Random) -> list[int]:
    sizes = [minimum] * parts
    extra = total - parts * minimum
    assert extra >= 0
    for _ in range(extra):
        sizes[rng.randrange(parts)] += 1
    return sizes


def _build_regions(
    branch: str, height: int, width: int, rng: random.Random
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    grid = np.full((height, width), BG, dtype=int)
    regions: list[tuple[int, int, int, int]] = []

    if branch == "quad":
        hs = _split_sizes(height - 1, 2, MIN_REGION_HEIGHT, rng)
        ws = _split_sizes(width - 1, 2, MIN_REGION_WIDTH, rng)
        wr = hs[0]
        wc = ws[0]
        grid[wr, :] = WALL
        grid[:, wc] = WALL
        regions = [
            (0, 0, hs[0] - 1, ws[0] - 1),
            (0, wc + 1, hs[0] - 1, width - 1),
            (wr + 1, 0, height - 1, ws[0] - 1),
            (wr + 1, wc + 1, height - 1, width - 1),
        ]
    elif branch == "rows":
        hs = _split_sizes(height - 2, 3, MIN_REGION_HEIGHT, rng)
        wall1 = hs[0]
        wall2 = hs[0] + 1 + hs[1]
        grid[wall1, :] = WALL
        grid[wall2, :] = WALL
        regions = [
            (0, 0, hs[0] - 1, width - 1),
            (wall1 + 1, 0, wall1 + hs[1], width - 1),
            (wall2 + 1, 0, height - 1, width - 1),
        ]
    else:
        ws = _split_sizes(width - 2, 3, MIN_REGION_WIDTH, rng)
        wall1 = ws[0]
        wall2 = ws[0] + 1 + ws[1]
        grid[:, wall1] = WALL
        grid[:, wall2] = WALL
        regions = [
            (0, 0, height - 1, ws[0] - 1),
            (0, wall1 + 1, height - 1, wall1 + ws[1]),
            (0, wall2 + 1, height - 1, width - 1),
        ]
    return grid, regions


def _choose_palette_pair(
    rng: random.Random,
    template_pair: tuple[int, int],
    forbidden: set[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    allowed = [c for c in COLORS if c not in template_pair]
    options = list(itertools.permutations(allowed, 2))
    if forbidden is not None:
        options = [pair for pair in options if pair not in forbidden]
    assert options
    return tuple(options[rng.randrange(len(options))])


def _render_region(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    template_pair: tuple[int, int],
    phase: int,
    palette: tuple[int, int],
    prefix_len: int,
) -> None:
    r0, c0, r1, c1 = rect
    width = c1 - c0 + 1
    first = int(template_pair[phase])
    second = int(template_pair[1 - phase])
    template = np.asarray([first if i % 2 == 0 else second for i in range(width)], dtype=int)
    prefix = np.zeros(width, dtype=int)
    for c in range(prefix_len):
        prefix[c] = palette[0] if template[c] == first else palette[1]

    grid[r0, c0 : c1 + 1] = template
    grid[r0 + 1, c0 : c1 + 1] = prefix


def _expected_from_specs(
    grid: np.ndarray,
    region_specs: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    for spec in region_specs:
        r0, c0, r1, c1 = spec["rect"]
        x, y = spec["palette"]
        fill_top = r0 + 2
        for r in range(fill_top, r1 + 1):
            for c in range(c0, c1 + 1):
                if out[r, c] != BG:
                    continue
                d = min(r - fill_top, r1 - r, c - c0, c1 - c)
                out[r, c] = x if d % 2 == 0 else y
    return out


def _validate_case(grid: np.ndarray, region_specs: list[dict[str, object]]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not np.any(grid == WALL):
        return False

    comps = _components(grid != WALL)
    if len(comps) != len(region_specs) or len(comps) not in (3, 4):
        return False

    shared_counterexample = False
    decoded: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 + 1 < MIN_REGION_HEIGHT or c1 - c0 + 1 < MIN_REGION_WIDTH:
            return False
        template = grid[r0, c0 : c1 + 1]
        prefix = grid[r0 + 1, c0 : c1 + 1]
        palette, prefix_len = _decode_palette(template, prefix)
        if prefix_len < 2 or prefix_len > min(4, len(template) - 1):
            return False
        if r1 - (r0 + 2) + 1 < 3:
            return False
        decoded.append((tuple(sorted(set(int(v) for v in template))), palette))

    for i in range(len(decoded)):
        for j in range(i + 1, len(decoded)):
            same_template = decoded[i][0] == decoded[j][0]
            different_palette = decoded[i][1] != decoded[j][1]
            if same_template and different_palette:
                shared_counterexample = True
                break
        if shared_counterexample:
            break
    if not shared_counterexample:
        return False

    expected = _expected_from_specs(grid, region_specs)
    solved = main(grid)
    if not np.array_equal(expected, solved):
        return False
    if np.array_equal(expected, grid):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    branch = rng.choice(("quad", "rows", "cols"))
    height = rng.randint(*HEIGHT_RANGE)
    width = rng.randint(*WIDTH_RANGE)
    grid, regions = _build_regions(branch, height, width, rng)
    count = len(regions)

    shared_template = tuple(sorted(rng.sample(COLORS, 2)))
    shared_indices = rng.sample(range(count), 2)

    if rng.random() < 0.5:
        base_two = rng.sample([c for c in COLORS if c not in shared_template], 2)
        shared_palettes = [tuple(base_two), (base_two[1], base_two[0])]
    else:
        first_palette = _choose_palette_pair(rng, shared_template)
        second_palette = _choose_palette_pair(rng, shared_template, forbidden={first_palette})
        shared_palettes = [first_palette, second_palette]

    region_specs: list[dict[str, object]] = []
    shared_phase = rng.randint(0, 1)
    for idx, rect in enumerate(regions):
        r0, c0, r1, c1 = rect
        width_i = c1 - c0 + 1

        if idx == shared_indices[0]:
            template_pair = shared_template
            palette = shared_palettes[0]
            phase = shared_phase
        elif idx == shared_indices[1]:
            template_pair = shared_template
            palette = shared_palettes[1]
            phase = 1 - shared_phase
        else:
            template_pair = tuple(sorted(rng.sample(COLORS, 2)))
            palette = _choose_palette_pair(rng, template_pair)
            phase = rng.randint(0, 1)

        prefix_len = rng.randint(2, min(4, width_i - 1))
        _render_region(grid, rect, template_pair, phase, palette, prefix_len)
        region_specs.append(
            {
                "rect": rect,
                "template_pair": template_pair,
                "palette": palette,
                "phase": phase,
                "prefix_len": prefix_len,
            }
        )

    if not _validate_case(grid, region_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
