from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

SOURCE_COLORS = (BLUE, RED, GREEN)
TARGET_COLORS = (YELLOW, MAGENTA, ORANGE)

GRID_ROWS_RANGE = (15, 20)
GRID_COLS_RANGE = (18, 28)
OBJECT_COUNT_OPTIONS = (2, 3)
USABLE_BOTTOM_MARGIN = 2
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 400


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _find_components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(sorted(coords), dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _transform_cells(cells: set[tuple[int, int]], rotate_k: int, flip_lr: bool) -> set[tuple[int, int]]:
    arr = np.array(sorted(cells), dtype=int)
    h = int(arr[:, 0].max()) + 1
    w = int(arr[:, 1].max()) + 1
    patch = np.zeros((h, w), dtype=int)
    patch[arr[:, 0], arr[:, 1]] = 1
    patch = np.rot90(patch, rotate_k)
    if flip_lr:
        patch = np.fliplr(patch)
    out = {(int(r), int(c)) for r, c in np.argwhere(patch == 1)}
    return _normalize(out)


def _cells_to_patch(cells: set[tuple[int, int]], color: int) -> np.ndarray:
    coords = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = int(color)
    return patch


def _row_span_cavity(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    occupied = {(int(r), int(c)) for r, c in coords}
    rows = sorted({int(r) for r, _ in occupied})
    cells: list[tuple[int, int]] = []
    for r in rows:
        cols = sorted(c for rr, c in occupied if rr == r)
        if len(cols) < 2:
            continue
        for c in range(cols[0] + 1, cols[-1]):
            if (r, c) not in occupied:
                cells.append((r, c))
    return tuple(cells)


def _parse_legend(row: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    c = 0
    while c < row.size:
        if int(row[c]) == BLACK:
            c += 1
            continue
        start = c
        while c < row.size and int(row[c]) != BLACK:
            c += 1
        runs.append((start, c - 1))

    pairs: list[tuple[int, int]] = []
    for start, end in runs:
        assert end - start + 1 == 2, "legend runs must be adjacent pixel pairs"
        src = int(row[start])
        dst = int(row[start + 1])
        assert src in SOURCE_COLORS, "legend source color must be blue/red/green"
        assert dst in TARGET_COLORS, "legend target color must be yellow/magenta/orange"
        pairs.append((src, dst))

    assert 2 <= len(pairs) <= 3, "expected 2 or 3 legend pairs"
    assert len({src for src, _ in pairs}) == len(pairs), "legend sources must be unique"
    return pairs


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4 and g.shape[1] >= 8
    body = g[:-1]
    legend_pairs = _parse_legend(g[-1])
    mapping = dict(legend_pairs)

    body_colors = set(map(int, np.unique(body))) - {BLACK}
    assert body_colors <= {GREY, *mapping.keys()}, "body may only contain outlines and source blocks"

    outline_comps = _find_components(body == GREY, connectivity=8)
    assert 2 <= len(outline_comps) <= 3, "expected 2 or 3 grey outline objects"

    cavities = []
    for comp in outline_comps:
        cavity = _row_span_cavity(comp)
        assert 3 <= len(cavity) <= 8, "cavity area out of range"
        for r, c in cavity:
            assert int(body[r, c]) == BLACK, "cavity cells must be black in the input"
        cavities.append(
            {
                "coords": comp,
                "bbox": _bbox(comp),
                "cavity": cavity,
                "area": len(cavity),
            }
        )
    cavity_areas = [item["area"] for item in cavities]
    assert len(set(cavity_areas)) == len(cavity_areas), "cavity areas must be distinct"

    blocks = []
    for src in mapping:
        comps = _find_components(body == int(src), connectivity=4)
        assert len(comps) == 1, "each legend source color must have exactly one body block"
        comp = comps[0]
        blocks.append(
            {
                "color": int(src),
                "coords": comp,
                "bbox": _bbox(comp),
                "area": int(comp.shape[0]),
            }
        )
    assert len(blocks) == len(cavities), "outline count and block count must match"

    block_areas = [item["area"] for item in blocks]
    assert len(set(block_areas)) == len(block_areas), "block sizes must be distinct"
    assert set(block_areas) == set(cavity_areas), "block sizes must match cavity areas one-to-one"

    cavity_by_area = {item["area"]: item for item in cavities}
    block_by_area = {item["area"]: item for item in blocks}
    return {
        "mapping": mapping,
        "legend_pairs": legend_pairs,
        "cavities": cavities,
        "blocks": blocks,
        "cavity_by_area": cavity_by_area,
        "block_by_area": block_by_area,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey outline cavity with the legend-mapped color of the size-matched source block.

    Input structure:
    - The grid has a black background, with 2 or 3 grey 8-connected outline objects in the main area.
    - Each grey object has a row-span cavity: on each occupied row, black cells strictly between the leftmost and rightmost grey pixels belong to that cavity.
    - The main area also contains 2 or 3 separate monochrome source blocks in blue/red/green, and the bottom row is a legend of adjacent source→target color pairs.

    Rule:
    - Ignore the bottom legend row when locating objects in the main area.
    - Compute every grey outline's row-span cavity area, and compute every source block's pixel count.
    - Match each source block to the unique cavity with the same area, erase the original block, and fill that cavity with the block color translated through the bottom-row legend.
    - Keep the grey outlines and preserve the bottom-row legend unchanged.

    Key insight:
    - The source block size decides which cavity it goes to, but the source block color is only an intermediate key that must be converted through the bottom legend before filling.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_input(g)
    out = g.copy()

    for block in info["blocks"]:
        rr, cc = block["coords"][:, 0], block["coords"][:, 1]
        out[rr, cc] = BLACK

    for area, cavity in info["cavity_by_area"].items():
        src_color = info["block_by_area"][area]["color"]
        fill_color = info["mapping"][src_color]
        for r, c in cavity["cavity"]:
            out[r, c] = int(fill_color)

    return out


def _u_outline(area: int) -> set[tuple[int, int]] | None:
    options = [(gap, rows) for gap in range(1, 8) for rows in range(1, 5) if gap * rows == area]
    if not options:
        return None
    gap, rows = random.choice(options)
    cells = {(r, 0) for r in range(rows)}
    cells |= {(r, gap + 1) for r in range(rows)}
    cells |= {(rows, c) for c in range(gap + 2)}
    if random.random() < 0.5:
        max_r = max(r for r, _ in cells)
        cells = {(max_r - r, c) for r, c in cells}
    return _normalize(cells)


def _ring_outline(area: int) -> set[tuple[int, int]] | None:
    options = [(inner_h, inner_w) for inner_h in range(1, 8) for inner_w in range(1, 8) if inner_h * inner_w == area]
    if not options:
        return None
    inner_h, inner_w = min(options, key=lambda hw: max(hw[0] + 2, hw[1] + 2))
    if random.random() < 0.5:
        inner_h, inner_w = inner_w, inner_h
    h = inner_h + 2
    w = inner_w + 2
    cells = set()
    for c in range(w):
        cells.add((0, c))
        cells.add((h - 1, c))
    for r in range(1, h - 1):
        cells.add((r, 0))
        cells.add((r, w - 1))
    return cells


def _outline_for_area(area: int) -> np.ndarray | None:
    builders = [_u_outline, _ring_outline]
    random.shuffle(builders)
    for builder in builders:
        cells = builder(area)
        if cells is None:
            continue
        cells = _transform_cells(cells, rotate_k=random.randrange(4), flip_lr=bool(random.randrange(2)))
        patch = _cells_to_patch(cells, GREY)
        cavity = _row_span_cavity(np.argwhere(patch == GREY))
        if len(cavity) != area:
            continue
        if max(patch.shape) > 10:
            continue
        return patch
    return None


def _random_polyomino(size: int) -> np.ndarray:
    for _ in range(120):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = {
                nb
                for r, c in cells
                for nb in _neighbors(r, c, 4)
                if nb not in cells
            }
            chosen = random.choice(sorted(frontier))
            cells.add(chosen)
        cells = _normalize(cells)
        cells = _transform_cells(cells, rotate_k=random.randrange(4), flip_lr=bool(random.randrange(2)))
        coords = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h > 5 or w > 5:
            continue
        if h * w > max(2 * size + 2, size + 6):
            continue
        patch = _cells_to_patch(cells, 1)
        if len(_find_components(patch == 1, connectivity=4)) != 1:
            continue
        return patch
    raise RuntimeError("failed to build polyomino")


def _bbox_clear(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int) -> bool:
    for r0, c0, r1, c1 in others:
        if not (box[2] + gap < r0 or r1 + gap < box[0] or box[3] + gap < c0 or c1 + gap < box[1]):
            return False
    return True


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    color: int,
    boxes: list[tuple[int, int, int, int]],
    row_limit: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    h, w = patch.shape
    positions = [
        (r, c)
        for r in range(0, row_limit - h + 1)
        for c in range(0, grid.shape[1] - w + 1)
    ]
    random.shuffle(positions)
    for r, c in positions:
        box = (r, c, r + h - 1, c + w - 1)
        if not _bbox_clear(box, boxes, PLACEMENT_GAP):
            continue
        target = grid[r : r + h, c : c + w]
        mask = patch != BLACK
        if np.any(target[mask] != BLACK):
            continue
        colored = np.zeros_like(patch)
        colored[mask] = int(color)
        return colored, box
    return None


def _legend_positions(width: int, count: int) -> list[int]:
    starts = [combo for combo in itertools.combinations(range(width - 1), count) if all(b - a >= 3 for a, b in zip(combo, combo[1:]))]
    assert starts, "legend width too small"
    return list(random.choice(starts))


def _row_major_key(item: dict) -> tuple[int, int, int, int]:
    return item["bbox"]


def _build_row_major_shortcut_output(grid: np.ndarray, info: dict) -> np.ndarray:
    out = grid.copy()
    for block in info["blocks"]:
        rr, cc = block["coords"][:, 0], block["coords"][:, 1]
        out[rr, cc] = BLACK
    cavities = sorted(info["cavities"], key=_row_major_key)
    blocks = sorted(info["blocks"], key=_row_major_key)
    for cavity, block in zip(cavities, blocks):
        fill = info["mapping"][block["color"]]
        for r, c in cavity["cavity"]:
            out[r, c] = int(fill)
    return out


def _build_legend_order_shortcut_output(grid: np.ndarray, info: dict) -> np.ndarray:
    out = grid.copy()
    for block in info["blocks"]:
        rr, cc = block["coords"][:, 0], block["coords"][:, 1]
        out[rr, cc] = BLACK
    cavities = sorted(info["cavities"], key=_row_major_key)
    legend_targets = [dst for _, dst in info["legend_pairs"]]
    for cavity, dst in zip(cavities, legend_targets):
        for r, c in cavity["cavity"]:
            out[r, c] = int(dst)
    return out


def _build_direct_fill_output(grid: np.ndarray, info: dict) -> np.ndarray:
    out = grid.copy()
    for block in info["blocks"]:
        rr, cc = block["coords"][:, 0], block["coords"][:, 1]
        out[rr, cc] = BLACK
    for area, cavity in info["cavity_by_area"].items():
        src = info["block_by_area"][area]["color"]
        for r, c in cavity["cavity"]:
            out[r, c] = int(src)
    return out


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        info = _analyze_input(grid)
        out = main(grid)
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, _build_direct_fill_output(grid, info)):
        return False
    if np.array_equal(out, _build_row_major_shortcut_output(grid, info)):
        return False
    if np.array_equal(out, _build_legend_order_shortcut_output(grid, info)):
        return False

    used_targets = {int(v) for v in np.unique(out)} & set(TARGET_COLORS)
    if len(used_targets) != len(info["blocks"]):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        n_objects = random.choice(OBJECT_COUNT_OPTIONS)
        areas = random.sample(range(3, 9), n_objects)
        outline_patches = []
        ok = True
        for area in areas:
            patch = _outline_for_area(area)
            if patch is None:
                ok = False
                break
            outline_patches.append((area, patch))
        if not ok:
            continue

        min_h = max(GRID_ROWS_RANGE[0], max(p.shape[0] for _, p in outline_patches) + USABLE_BOTTOM_MARGIN + 1)
        min_w = max(
            GRID_COLS_RANGE[0],
            max(p.shape[1] for _, p in outline_patches) + 2,
            3 * n_objects + 2,
        )
        h = random.randint(min_h, GRID_ROWS_RANGE[1])
        w = random.randint(min_w, GRID_COLS_RANGE[1])
        grid = np.zeros((h, w), dtype=int)
        boxes: list[tuple[int, int, int, int]] = []
        row_limit = h - USABLE_BOTTOM_MARGIN

        random.shuffle(outline_patches)
        for _area, patch in outline_patches:
            placed = _place_patch(grid, patch, GREY, boxes, row_limit)
            if placed is None:
                ok = False
                break
            colored, box = placed
            r0, c0, r1, c1 = box
            mask = colored != BLACK
            grid[r0 : r1 + 1, c0 : c1 + 1][mask] = colored[mask]
            boxes.append(box)
        if not ok:
            continue

        source_colors = random.sample(list(SOURCE_COLORS), n_objects)
        target_colors = random.sample(list(TARGET_COLORS), n_objects)
        area_to_src = dict(zip(random.sample(areas, len(areas)), source_colors))
        mapping = dict(zip(source_colors, target_colors))

        for area in areas:
            block_patch = _random_polyomino(area)
            placed = _place_patch(grid, block_patch, area_to_src[area], boxes, row_limit)
            if placed is None:
                ok = False
                break
            colored, box = placed
            r0, c0, r1, c1 = box
            mask = colored != BLACK
            grid[r0 : r1 + 1, c0 : c1 + 1][mask] = colored[mask]
            boxes.append(box)
        if not ok:
            continue

        legend_starts = _legend_positions(w, n_objects)
        legend_pairs = list(mapping.items())
        random.shuffle(legend_pairs)
        bottom = np.zeros(w, dtype=int)
        for start, (src, dst) in zip(legend_starts, legend_pairs):
            bottom[start] = int(src)
            bottom[start + 1] = int(dst)
        grid[-1] = bottom

        if _validate_instance(grid):
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
