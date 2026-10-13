import random
from typing import Iterable

import numpy as np


CELL_SIZE = 4
TEXTURE_SIZE = 3
MAX_GENERATION_ATTEMPTS = 400
TEXTURE_LIBRARY = (
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [0, 1, 1]], dtype=bool),
)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    point_set = set(points)
    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    return point_set == expected


def _panel_spans(line_indices: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for start, stop in zip(line_indices, line_indices[1:]):
        if stop - start > 1:
            spans.append((start + 1, stop - 1))
    return spans


def _find_panel(
    grid: np.ndarray, bg: int
) -> tuple[int, tuple[int, int, int, int], dict[int, np.ndarray]]:
    candidates: list[tuple[int, tuple[int, int, int, int], dict[int, np.ndarray]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            touches_top = r0 == 0
            touches_left = c0 == 0
            if touches_top == touches_left:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            full_rows = [idx for idx in range(sub.shape[0]) if np.all(sub[idx, :] == color)]
            full_cols = [idx for idx in range(sub.shape[1]) if np.all(sub[:, idx] == color)]
            row_spans = _panel_spans(full_rows)
            col_spans = _panel_spans(full_cols)
            if not row_spans or not col_spans:
                continue
            if any(re - rs + 1 != CELL_SIZE for rs, re in row_spans):
                continue
            if any(ce - cs + 1 != CELL_SIZE for cs, ce in col_spans):
                continue

            mapping: dict[int, np.ndarray] = {}
            ok = True
            for rs, re in row_spans:
                for cs, ce in col_spans:
                    cell = sub[rs : re + 1, cs : ce + 1]
                    marker = int(cell[0, 0])
                    if marker in (bg, color):
                        ok = False
                        break
                    if marker in mapping:
                        ok = False
                        break
                    if np.any(cell[0, 1:] != bg) or np.any(cell[1:, 0] != bg):
                        ok = False
                        break
                    texture = cell[1:, 1:]
                    if texture.shape != (TEXTURE_SIZE, TEXTURE_SIZE):
                        ok = False
                        break
                    if not np.all((texture == bg) | (texture == marker)):
                        ok = False
                        break
                    if np.all(texture == bg) or np.all(texture == marker):
                        ok = False
                        break
                    mapping[marker] = texture == marker
                if not ok:
                    break
            if ok and mapping:
                candidates.append((color, (r0, c0, r1, c1), mapping))
    assert len(candidates) == 1, "expected a unique legend panel"
    return candidates[0]


def _find_regions(
    grid: np.ndarray, bg: int, panel_bbox: tuple[int, int, int, int], panel_color: int
) -> tuple[int, list[tuple[int, int, int, int]]]:
    outside_panel = np.ones_like(grid, dtype=bool)
    pr0, pc0, pr1, pc1 = panel_bbox
    outside_panel[pr0 : pr1 + 1, pc0 : pc1 + 1] = False

    border_color = None
    regions: list[tuple[int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, panel_color):
            continue
        for comp in _components((grid == color) & outside_panel):
            if not _is_rect_frame(comp):
                continue
            if border_color is None:
                border_color = color
            assert border_color == color, "all region borders must share one color"
            regions.append(_bbox(comp))
    assert border_color is not None and regions, "expected at least one closed region"
    return border_color, regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed rectangle with the tiled legend texture selected by its interior seed color.

    Input structure:
    - The grid has a dominant background color.
    - A rectangular legend panel sits on the top edge or left edge and is divided into equal cells; each cell has a one-pixel color marker in its top-left corner and a 3x3 two-color texture made from that marker color and the background.
    - Elsewhere, several one-pixel-thick rectangular borders of one shared color enclose background interiors, and each interior contains exactly one non-background seed pixel whose color matches one legend marker.

    Rule:
    - Keep the legend panel and every rectangular border unchanged.
    - Read the legend as a mapping from seed color to the 3x3 texture shown in that cell.
    - For each bordered rectangle, use its seed color to select the matching legend texture, then tile that 3x3 pattern across the whole interior starting from the interior's top-left corner, overwriting the seed pixel.

    Key insight:
    - The seed pixel is only an index into the legend; the visible fill comes from first decoding the color-to-texture mapping and then repeating the chosen texture inside the matching enclosure.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    panel_color, panel_bbox, mapping = _find_panel(g, bg)
    border_color, regions = _find_regions(g, bg, panel_bbox, panel_color)

    out = g.copy()
    for r0, c0, r1, c1 in regions:
        ir0, ic0 = r0 + 1, c0 + 1
        ir1, ic1 = r1 - 1, c1 - 1
        interior = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
        seed_points = np.argwhere((interior != bg) & (interior != border_color))
        assert len(seed_points) == 1, "each region must contain exactly one seed"
        sr, sc = seed_points[0]
        seed_color = int(interior[sr, sc])
        assert seed_color in mapping, "seed color must appear in the legend"
        texture = mapping[seed_color]

        for r in range(ir0, ir1 + 1):
            for c in range(ic0, ic1 + 1):
                out[r, c] = seed_color if texture[(r - ir0) % TEXTURE_SIZE, (c - ic0) % TEXTURE_SIZE] else bg
    return out


def _draw_panel(
    grid: np.ndarray,
    origin: tuple[int, int],
    orientation: str,
    panel_color: int,
    marker_colors: list[int],
    textures: list[np.ndarray],
    bg: int,
) -> tuple[int, int, int, int]:
    count = len(marker_colors)
    rows = 1 if orientation == "top" else count
    cols = count if orientation == "top" else 1
    panel_h = rows * CELL_SIZE + rows + 1
    panel_w = cols * CELL_SIZE + cols + 1
    r0, c0 = origin
    sub = grid[r0 : r0 + panel_h, c0 : c0 + panel_w]

    for rr in range(0, panel_h, CELL_SIZE + 1):
        sub[rr, :] = panel_color
    for cc in range(0, panel_w, CELL_SIZE + 1):
        sub[:, cc] = panel_color

    for idx, (marker, texture) in enumerate(zip(marker_colors, textures)):
        cell_r = 1 + (0 if orientation == "top" else idx) * (CELL_SIZE + 1)
        cell_c = 1 + (idx if orientation == "top" else 0) * (CELL_SIZE + 1)
        cell = sub[cell_r : cell_r + CELL_SIZE, cell_c : cell_c + CELL_SIZE]
        cell[:, :] = bg
        cell[0, 0] = marker
        cell[1:, 1:] = np.where(texture, marker, bg)

    return r0, c0, r0 + panel_h - 1, c0 + panel_w - 1


def _sample_region_colors(rng: random.Random, marker_colors: list[int], count: int) -> list[int]:
    if count >= len(marker_colors):
        colors = marker_colors[:]
        while len(colors) < count:
            colors.append(rng.choice(marker_colors))
        rng.shuffle(colors)
        return colors
    chosen = rng.sample(marker_colors, count)
    rng.shuffle(chosen)
    return chosen


def _place_regions(
    rng: random.Random,
    grid: np.ndarray,
    occupied: np.ndarray,
    border_color: int,
    region_colors: list[int],
    bg: int,
) -> bool:
    h, w = grid.shape
    for seed_color in region_colors:
        placed = False
        for _ in range(250):
            inner_h = rng.randint(4, 8)
            inner_w = rng.randint(4, 8)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if h - outer_h - 1 < 1 or w - outer_w - 1 < 1:
                return False
            r0 = rng.randint(1, h - outer_h - 1)
            c0 = rng.randint(1, w - outer_w - 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            mr0 = max(0, r0 - 1)
            mc0 = max(0, c0 - 1)
            mr1 = min(h, r1 + 2)
            mc1 = min(w, c1 + 2)
            if occupied[mr0:mr1, mc0:mc1].any():
                continue

            grid[r0 : r1 + 1, c0] = border_color
            grid[r0 : r1 + 1, c1] = border_color
            grid[r0, c0 : c1 + 1] = border_color
            grid[r1, c0 : c1 + 1] = border_color

            seed_r = rng.randint(r0 + 1, r1 - 1)
            seed_c = rng.randint(c0 + 1, c1 - 1)
            grid[seed_r, seed_c] = seed_color
            occupied[mr0:mr1, mc0:mc1] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    orientation = rng.choice(("top", "left"))
    legend_count = rng.randint(2, 4)
    region_count = rng.randint(2, 4)

    bg = rng.randrange(10)
    remaining = [color for color in range(10) if color != bg]
    rng.shuffle(remaining)
    panel_color = remaining.pop()
    border_color = remaining.pop()
    marker_colors = [remaining.pop() for _ in range(legend_count)]
    textures = [TEXTURE_LIBRARY[idx].copy() for idx in rng.sample(range(len(TEXTURE_LIBRARY)), legend_count)]

    rows = 1 if orientation == "top" else legend_count
    cols = legend_count if orientation == "top" else 1
    panel_h = rows * CELL_SIZE + rows + 1
    panel_w = cols * CELL_SIZE + cols + 1

    min_h = max(18, panel_h + 10)
    min_w = max(18, panel_w + 10)
    if min_h > 30 or min_w > 30:
        return None
    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)
    grid = np.full((h, w), bg, dtype=int)

    if orientation == "top":
        if w - panel_w - 1 < 1:
            return None
        panel_origin = (0, rng.randint(1, w - panel_w - 1))
    else:
        if h - panel_h - 1 < 1:
            return None
        panel_origin = (rng.randint(1, h - panel_h - 1), 0)

    panel_bbox = _draw_panel(grid, panel_origin, orientation, panel_color, marker_colors, textures, bg)
    occupied = np.zeros_like(grid, dtype=bool)
    pr0, pc0, pr1, pc1 = panel_bbox
    occupied[max(0, pr0 - 1) : min(h, pr1 + 2), max(0, pc0 - 1) : min(w, pc1 + 2)] = True

    region_colors = _sample_region_colors(rng, marker_colors, region_count)
    if len(set(region_colors)) == 1:
        return None
    if not _place_regions(rng, grid, occupied, border_color, region_colors, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
