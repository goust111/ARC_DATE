from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BG = 0
LEGEND_PANEL = 8
ANCHOR_MARKER = 7
PATTERN_BG = 9
BLOCK_SIZE = 5
LEGEND_COLORS = (1, 2, 3, 4)
PANEL_SLOTS = ((1, 1), (1, 4), (4, 1), (4, 4))
MARKER_CORNERS = ((0, 0), (0, 4), (4, 4), (4, 0))


def _neighbors(connectivity: int) -> tuple[tuple[int, int], ...]:
    if int(connectivity) == 4:
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    return (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], copy=True)


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int,
    monochromatic: bool,
    colors: tuple[int, ...] | list[int] | None = None,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    visited = np.zeros((H, W), dtype=bool)
    nbrs = _neighbors(connectivity)
    if colors is None:
        allowed = None
    else:
        allowed = {int(c) for c in colors if int(c) != int(bg)}

    def allowed_cell(r: int, c: int) -> bool:
        val = int(g[r, c])
        if val == int(bg):
            return False
        if allowed is None:
            return True
        return val in allowed

    comps: list[dict] = []
    for r in range(H):
        for c in range(W):
            if visited[r, c] or not allowed_cell(r, c):
                continue
            base = int(g[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if visited[nr, nc] or not allowed_cell(nr, nc):
                        continue
                    if monochromatic and int(g[nr, nc]) != base:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            comps.append(
                {
                    "color": base if monochromatic else None,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _normalize(coords: np.ndarray | list[tuple[int, int]] | set[tuple[int, int]]):
    arr = np.asarray(list(coords), dtype=int)
    assert arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0
    r0, c0, r1, c1 = _bbox_coords(arr)
    shifted = arr - np.array([r0, c0], dtype=int)
    shape = frozenset((int(r), int(c)) for r, c in shifted)
    return shape, (int(r1 - r0 + 1), int(c1 - c0 + 1))


def _d4_variants(shape: frozenset[tuple[int, int]], dims: tuple[int, int]):
    h, w = map(int, dims)
    transforms = (
        lambda r, c: (r, c),
        lambda r, c: (c, h - 1 - r),
        lambda r, c: (h - 1 - r, w - 1 - c),
        lambda r, c: (w - 1 - c, r),
        lambda r, c: (r, w - 1 - c),
        lambda r, c: (h - 1 - r, c),
        lambda r, c: (c, r),
        lambda r, c: (w - 1 - c, h - 1 - r),
    )
    out: list[tuple[frozenset[tuple[int, int]], tuple[int, int]]] = []
    seen = set()
    for transform in transforms:
        norm = _normalize([transform(r, c) for r, c in shape])
        key = (tuple(sorted(norm[0])), norm[1])
        if key not in seen:
            seen.add(key)
            out.append(norm)
    return out


def _shape_mask(shape: frozenset[tuple[int, int]], dims: tuple[int, int]) -> np.ndarray:
    h, w = map(int, dims)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[int(r), int(c)] = True
    return mask


def _centered_variant(shape: frozenset[tuple[int, int]], dims: tuple[int, int]):
    best = None
    for var_shape, var_dims in _d4_variants(shape, dims):
        h, w = map(int, var_dims)
        if h > 5 or w > 5:
            continue
        r_off = (5 - h) // 2
        c_off = (5 - w) // 2
        placed = tuple(sorted((int(r_off + r), int(c_off + c)) for r, c in var_shape))
        key = (placed[0], placed)
        if best is None or key < best[0]:
            best = (key, var_shape, var_dims, placed)
    assert best is not None
    return best[1], best[2], best[3]


def _render_pattern(shape: frozenset[tuple[int, int]], dims: tuple[int, int], color: int) -> np.ndarray:
    patch = np.full((BLOCK_SIZE, BLOCK_SIZE), PATTERN_BG, dtype=int)
    _shape, _dims, placed = _centered_variant(shape, dims)
    for r, c in placed:
        patch[int(r), int(c)] = int(color)
    return patch


def _panel_box(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    teal = _find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[LEGEND_PANEL],
    )
    candidates = [
        comp
        for comp in teal
        if (comp["bbox"][2] - comp["bbox"][0] + 1) >= 7
        and (comp["bbox"][3] - comp["bbox"][1] + 1) >= 7
        and comp["area"] >= 30
    ]
    assert len(candidates) == 1
    return candidates[0]["bbox"]


def _extract_legend(grid: np.ndarray, box: tuple[int, int, int, int]) -> dict[int, tuple[frozenset[tuple[int, int]], tuple[int, int]]]:
    r0, c0, r1, c1 = map(int, box)
    region = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    legend: dict[int, tuple[frozenset[tuple[int, int]], tuple[int, int]]] = {}
    for color in sorted(set(map(int, region.ravel())) - {BG, LEGEND_PANEL}):
        coords = np.argwhere(region == int(color))
        comps = _find_components(
            (region == int(color)).astype(int),
            bg=0,
            connectivity=8,
            monochromatic=True,
            colors=[1],
        )
        assert len(comps) == 1
        shape, dims = _normalize(coords)
        assert dims[0] <= 3 and dims[1] <= 3
        legend[int(color)] = (shape, dims)
    assert len(legend) in (3, 4)
    return legend


def _extract_blocks(grid: np.ndarray, panel_box: tuple[int, int, int, int], bg: int) -> list[dict]:
    work = np.array(grid, copy=True)
    pr0, pc0, pr1, pc1 = map(int, panel_box)
    work[pr0 : pr1 + 1, pc0 : pc1 + 1] = int(bg)
    comps = _find_components(work, bg=bg, connectivity=4, monochromatic=False)
    blocks = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if (r1 - r0 + 1, c1 - c0 + 1) != (BLOCK_SIZE, BLOCK_SIZE):
            continue
        block = _crop_bbox(grid, comp["bbox"])
        colors = set(map(int, block.ravel()))
        marker_cells = np.argwhere(block == ANCHOR_MARKER)
        if marker_cells.shape[0] > 1:
            raise AssertionError("expected at most one anchor marker")
        base_colors = colors - {ANCHOR_MARKER}
        assert len(base_colors) == 1
        base_color = int(next(iter(base_colors)))
        blocks.append(
            {
                "bbox": comp["bbox"],
                "r0": r0,
                "c0": c0,
                "color": base_color,
                "marker_local": tuple(map(int, marker_cells[0])) if marker_cells.size else None,
            }
        )
    assert len(blocks) in (3, 4)
    return sorted(blocks, key=lambda d: (d["r0"], d["c0"]))


def _marker_rotation(marker_local: tuple[int, int]) -> int:
    for k, corner in enumerate(MARKER_CORNERS):
        if tuple(map(int, marker_local)) == corner:
            return int(k)
    raise AssertionError("anchor marker must sit on a 5x5 corner")


def _render_from_matrix(pattern_matrix: np.ndarray, rot_k: int) -> np.ndarray:
    k = int(rot_k) % 4
    rotated_positions = np.rot90(pattern_matrix, k)
    out_h = rotated_positions.shape[0] * BLOCK_SIZE
    out_w = rotated_positions.shape[1] * BLOCK_SIZE
    out = np.full((out_h, out_w), PATTERN_BG, dtype=int)
    for rr in range(rotated_positions.shape[0]):
        for cc in range(rotated_positions.shape[1]):
            patch = rotated_positions[rr, cc]
            assert isinstance(patch, np.ndarray)
            out[
                rr * BLOCK_SIZE : (rr + 1) * BLOCK_SIZE,
                cc * BLOCK_SIZE : (cc + 1) * BLOCK_SIZE,
            ] = np.rot90(np.asarray(patch, dtype=int), k)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert color swatches into centered legend patterns, then rotate and assemble them by the anchor block.

    Input structure:
    - Black background with one large teal legend panel on the left and 3 or 4 separate 5x5 solid-color blocks on the right.
    - The panel contains one small colored 8-connected shape per block color; each shape defines that color's legend entry.
    - Exactly one outside block has a single orange pixel on one of its four corners, marking the anchor block and the input orientation.

    Rule:
    - Read the teal panel as a color-to-shape dictionary.
    - Replace each outside solid block by a 5x5 maroon pattern whose centered marker shape is the legend shape for that block's color; choose the D4 variant whose centered occupied cells start as far toward the top-left as possible.
    - Treat the scattered 5x5 blocks as a sparse rectangular block layout, rotate that layout and every generated pattern by the anchor corner's quarter-turn so the anchor becomes the output top-left block, then remove the gaps.

    Key insight:
    - The legend panel determines block contents, but the orange corner marker determines the only global orientation that turns the sparse block layout into the final tight rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    panel_box = _panel_box(g, bg)
    legend = _extract_legend(g, panel_box)
    blocks = _extract_blocks(g, panel_box, bg)

    anchors = [block for block in blocks if block["marker_local"] is not None]
    assert len(anchors) == 1
    rot_k = _marker_rotation(anchors[0]["marker_local"])

    rows = sorted({int(block["r0"]) for block in blocks})
    cols = sorted({int(block["c0"]) for block in blocks})
    assert len(rows) * len(cols) == len(blocks)
    row_index = {value: i for i, value in enumerate(rows)}
    col_index = {value: i for i, value in enumerate(cols)}

    pattern_matrix = np.empty((len(rows), len(cols)), dtype=object)
    pattern_matrix[:] = None
    for block in blocks:
        r_idx = row_index[int(block["r0"])]
        c_idx = col_index[int(block["c0"])]
        assert pattern_matrix[r_idx, c_idx] is None
        assert block["color"] in legend
        shape, dims = legend[int(block["color"])]
        pattern_matrix[r_idx, c_idx] = _render_pattern(shape, dims, int(block["color"]))

    assert all(patch is not None for patch in pattern_matrix.ravel())
    return _render_from_matrix(pattern_matrix, rot_k)


def _shape_catalog():
    cache = getattr(_shape_catalog, "_cache", None)
    if cache is not None:
        return cache

    catalog = []
    seen = set()
    cells = list(range(9))
    for area in range(3, 6):
        for combo in combinations(cells, area):
            coords = np.array([(idx // 3, idx % 3) for idx in combo], dtype=int)
            mask = np.zeros((3, 3), dtype=bool)
            mask[coords[:, 0], coords[:, 1]] = True
            comps = _find_components(mask.astype(int), bg=0, connectivity=8, monochromatic=True, colors=[1])
            if len(comps) != 1:
                continue
            shape, dims = _normalize(coords)
            if dims[0] > 3 or dims[1] > 3 or (dims[0] % 2 == 0) or (dims[1] % 2 == 0):
                continue
            orbit = _d4_variants(shape, dims)
            key = min((tuple(sorted(variant_shape)), variant_dims) for variant_shape, variant_dims in orbit)
            if key in seen:
                continue
            seen.add(key)
            catalog.append(
                {
                    "orbit": orbit,
                    "area": area,
                    "rot_asym": any(
                        tuple(sorted(var_shape)) != tuple(sorted(shape))
                        or var_dims != dims
                        for var_shape, var_dims in orbit[1:]
                    ),
                }
            )
    assert len(catalog) >= 8
    _shape_catalog._cache = catalog
    return catalog


def _random_oriented_shape(entry: dict):
    return random.choice(entry["orbit"])


def _canonical_tile_shape(count: int) -> tuple[int, int]:
    return (1, 3) if int(count) == 3 else (2, 2)


def _build_expected(tile_in: np.ndarray, legend: dict[int, tuple[frozenset[tuple[int, int]], tuple[int, int]]], rot_k: int) -> np.ndarray:
    matrix = np.empty(tile_in.shape, dtype=object)
    for r in range(tile_in.shape[0]):
        for c in range(tile_in.shape[1]):
            color = int(tile_in[r, c])
            shape, dims = legend[color]
            matrix[r, c] = _render_pattern(shape, dims, color)
    return _render_from_matrix(matrix, rot_k)


def _place_panel(grid: np.ndarray, panel_r0: int, panel_c0: int, legend: dict[int, tuple[frozenset[tuple[int, int]], tuple[int, int]]], slot_order: list[tuple[int, int]]) -> None:
    grid[panel_r0 : panel_r0 + 8, panel_c0 : panel_c0 + 8] = LEGEND_PANEL
    for (color, (shape, dims)), (sr, sc) in zip(legend.items(), slot_order):
        h, w = map(int, dims)
        local_r = panel_r0 + sr + random.randint(0, 3 - h)
        local_c = panel_c0 + sc + random.randint(0, 3 - w)
        for r, c in shape:
            grid[local_r + int(r), local_c + int(c)] = int(color)


def _place_blocks(
    grid: np.ndarray,
    blocks_r0: int,
    blocks_c0: int,
    tile_in: np.ndarray,
    anchor_color: int,
    rot_k: int,
) -> None:
    row_gap = 1 if tile_in.shape[0] > 1 else 0
    col_gap = 1 if tile_in.shape[1] > 1 else 0
    anchor_cell = tuple(map(int, np.argwhere(tile_in == int(anchor_color))[0]))
    for r in range(tile_in.shape[0]):
        for c in range(tile_in.shape[1]):
            rr = blocks_r0 + r * (BLOCK_SIZE + row_gap)
            cc = blocks_c0 + c * (BLOCK_SIZE + col_gap)
            grid[rr : rr + BLOCK_SIZE, cc : cc + BLOCK_SIZE] = int(tile_in[r, c])
            if (r, c) == anchor_cell:
                mr, mc = MARKER_CORNERS[int(rot_k) % 4]
                grid[rr + mr, cc + mc] = ANCHOR_MARKER


def _wrong_rotations_unique(tile_in: np.ndarray, legend: dict[int, tuple[frozenset[tuple[int, int]], tuple[int, int]]], true_k: int, expected: np.ndarray) -> bool:
    seen = 0
    for k in range(4):
        out = _build_expected(tile_in, legend, k)
        if np.array_equal(out, expected):
            if k != int(true_k):
                return False
            seen += 1
    return seen == 1


def generate_input() -> np.ndarray:
    for _ in range(400):
        count = random.choice((3, 4))
        colors = random.sample(list(LEGEND_COLORS), count)

        catalog = _shape_catalog()
        while True:
            chosen = random.sample(catalog, count)
            if any(entry["rot_asym"] for entry in chosen):
                break

        legend = {
            int(color): _random_oriented_shape(entry)
            for color, entry in zip(colors, chosen)
        }

        out_rows, out_cols = _canonical_tile_shape(count)
        tile_out = np.zeros((out_rows, out_cols), dtype=int)
        perm = random.sample(colors, count)
        anchor_color = int(perm[0])
        tile_out[0, 0] = anchor_color
        idx = 1
        for r in range(out_rows):
            for c in range(out_cols):
                if (r, c) == (0, 0):
                    continue
                tile_out[r, c] = int(perm[idx])
                idx += 1

        rot_k = random.randrange(4)
        tile_in = np.rot90(tile_out, -rot_k)

        row_gap = 1 if tile_in.shape[0] > 1 else 0
        col_gap = 1 if tile_in.shape[1] > 1 else 0
        block_h = tile_in.shape[0] * BLOCK_SIZE + (tile_in.shape[0] - 1) * row_gap
        block_w = tile_in.shape[1] * BLOCK_SIZE + (tile_in.shape[1] - 1) * col_gap

        panel_h = panel_w = 8
        panel_c0 = random.choice((0, 1))
        gap = random.choice((1, 2))
        blocks_c0 = panel_c0 + panel_w + gap
        min_w = blocks_c0 + block_w
        if min_w > 30:
            continue
        W = random.randint(max(18, min_w), min(30, max(18, min_w) + 4))

        min_h = max(panel_h, block_h)
        if min_h > 30:
            continue
        H = random.randint(max(18, min_h), min(30, max(18, min_h) + 5))

        panel_r0 = random.randint(0, H - panel_h)
        blocks_r0 = random.randint(0, H - block_h)

        grid = np.full((H, W), BG, dtype=int)
        slot_order = random.sample(list(PANEL_SLOTS), count)
        _place_panel(grid, panel_r0, panel_c0, legend, slot_order)
        _place_blocks(grid, blocks_r0, blocks_c0, tile_in, anchor_color, rot_k)

        expected = _build_expected(tile_in, legend, rot_k)
        if expected.shape[0] > 30 or expected.shape[1] > 30:
            continue
        if not _wrong_rotations_unique(tile_in, legend, rot_k, expected):
            continue

        solved = main(grid)
        if not np.array_equal(solved, expected):
            continue

        if len(_extract_legend(grid, _panel_box(grid, BG))) != count:
            continue
        if len(_extract_blocks(grid, _panel_box(grid, BG), BG)) != count:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
