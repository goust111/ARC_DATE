from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_for_036_8",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
bbox = _ARC_UTILS.bbox
find_components = _ARC_UTILS.find_components
neighbors4 = _ARC_UTILS.neighbors4
neighbors8 = _ARC_UTILS.neighbors8

BG = Color.BLACK
SEP = Color.GREY
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GEN_RETRIES = 400


_BASE_TEMPLATES = [
    np.array(
        [
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 1],
            [3, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 1],
            [2, 2],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 1],
            [3, 1],
            [1, 3],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 1],
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 2],
        ],
        dtype=int,
    ),
]


def _segments_from_full_lines(flags: list[bool]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_sep in enumerate(flags):
        if not is_sep and start is None:
            start = idx
        elif is_sep and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(flags)))
    return segments


def _diagonal_only_offsets(panel: np.ndarray, base_color: int, motif_color: int) -> set[tuple[int, int]]:
    h, w = panel.shape
    offsets: set[tuple[int, int]] = set()
    for r, c in np.argwhere(panel == motif_color):
        has_base_in_8 = False
        has_base_in_4 = False
        for rr, cc in neighbors8(int(r), int(c)):
            if 0 <= rr < h and 0 <= cc < w and int(panel[rr, cc]) == int(base_color):
                has_base_in_8 = True
                break
        if not has_base_in_8:
            continue
        for rr, cc in neighbors4(int(r), int(c)):
            if 0 <= rr < h and 0 <= cc < w and int(panel[rr, cc]) == int(base_color):
                has_base_in_4 = True
                break
        if not has_base_in_4:
            offsets.add((int(r), int(c)))
    return offsets


def _analyze_panel_patch(panel: np.ndarray) -> dict:
    colors = [int(c) for c in np.unique(panel)]
    assert BG not in colors and SEP not in colors
    assert len(colors) == 2
    counts = {color: int(np.sum(panel == color)) for color in colors}
    base_color, motif_color = sorted(colors, key=lambda color: (-counts[color], color))
    assert counts[base_color] > counts[motif_color]

    motif_components = find_components(
        panel,
        bg=base_color,
        connectivity=4,
        monochromatic=True,
        colors=[motif_color],
        return_format="dict",
    )
    assert len(motif_components) == 1
    motif_offsets = {tuple(map(int, coord)) for coord in motif_components[0]["coords"]}
    diag_offsets = _diagonal_only_offsets(panel, base_color, motif_color)
    non_diag_offsets = motif_offsets - diag_offsets
    return {
        "panel": panel.copy(),
        "base_color": int(base_color),
        "motif_color": int(motif_color),
        "motif_offsets": motif_offsets,
        "diag_offsets": diag_offsets,
        "non_diag_offsets": non_diag_offsets,
    }


def _extract_panel_from_cell(cell: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    mask = (cell != BG) & (cell != SEP)
    panel_bbox = bbox(mask)
    assert panel_bbox is not None
    r0, c0, r1, c1 = map(int, panel_bbox)
    panel = cell[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert np.all(panel != BG) and np.all(panel != SEP)
    outer = cell.copy()
    outer[r0 : r1 + 1, c0 : c1 + 1] = BG
    assert np.all(outer == BG)
    return panel, (r0, c0, r1, c1)


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sep_rows = [bool(np.all(g[r, :] == SEP)) for r in range(g.shape[0])]
    sep_cols = [bool(np.all(g[:, c] == SEP)) for c in range(g.shape[1])]
    row_segments = _segments_from_full_lines(sep_rows)
    col_segments = _segments_from_full_lines(sep_cols)
    assert len(col_segments) == 2
    assert len(row_segments) >= 1

    bands = []
    for rs, re in row_segments:
        left_cs, left_ce = col_segments[0]
        right_cs, right_ce = col_segments[1]
        left_cell = g[rs:re, left_cs:left_ce]
        right_cell = g[rs:re, right_cs:right_ce]
        left_panel, left_bbox = _extract_panel_from_cell(left_cell)
        right_panel, right_bbox = _extract_panel_from_cell(right_cell)
        assert left_panel.shape == right_panel.shape

        left_info = _analyze_panel_patch(left_panel)
        right_info = _analyze_panel_patch(right_panel)
        left_info["panel_bbox_abs"] = (
            rs + left_bbox[0],
            left_cs + left_bbox[1],
            rs + left_bbox[2],
            left_cs + left_bbox[3],
        )
        right_info["panel_bbox_abs"] = (
            rs + right_bbox[0],
            right_cs + right_bbox[1],
            rs + right_bbox[2],
            right_cs + right_bbox[3],
        )
        bands.append({"left": left_info, "right": right_info})

    return {
        "grid": g,
        "row_segments": row_segments,
        "col_segments": col_segments,
        "bands": bands,
    }


def _apply_pair_swap(left_info: dict, right_info: dict) -> tuple[np.ndarray, np.ndarray]:
    left_panel = left_info["panel"].copy()
    right_panel = right_info["panel"].copy()

    for r, c in left_info["diag_offsets"]:
        left_panel[r, c] = left_info["base_color"]
    for r, c in right_info["diag_offsets"]:
        right_panel[r, c] = right_info["base_color"]

    for r, c in right_info["diag_offsets"]:
        left_panel[r, c] = right_info["base_color"]
    for r, c in left_info["diag_offsets"]:
        right_panel[r, c] = left_info["base_color"]

    return left_panel, right_panel


def _apply_full_motif_swap(case: dict) -> np.ndarray:
    out = case["grid"].copy()
    for band in case["bands"]:
        left_info = band["left"]
        right_info = band["right"]
        left_panel = left_info["panel"].copy()
        right_panel = right_info["panel"].copy()

        for r, c in left_info["motif_offsets"]:
            left_panel[r, c] = left_info["base_color"]
        for r, c in right_info["motif_offsets"]:
            right_panel[r, c] = right_info["base_color"]
        for r, c in right_info["motif_offsets"]:
            left_panel[r, c] = right_info["base_color"]
        for r, c in left_info["motif_offsets"]:
            right_panel[r, c] = left_info["base_color"]

        lr0, lc0, lr1, lc1 = left_info["panel_bbox_abs"]
        rr0, rc0, rr1, rc1 = right_info["panel_bbox_abs"]
        out[lr0 : lr1 + 1, lc0 : lc1 + 1] = left_panel
        out[rr0 : rr1 + 1, rc0 : rc1 + 1] = right_panel
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Swap only the diagonally base-touching motif pixels between paired panels.

    Input structure:
    - Background is black, while gray full rows and one gray full column split the grid into row bands of two cells.
    - Each cell contains exactly one inset rectangular panel filled mostly by a base color and one 4-connected motif color.
    - The left and right panels within a band have the same size, but their motif shapes and inset positions may differ.

    Rule:
    - For each panel, identify motif pixels whose 8-neighborhood contains base color but whose 4-neighborhood contains no base color.
    - Within each band, swap only those selected offsets between the left and right panels, preserving the offsets relative to each panel's own top-left corner.
    - A selected source pixel is cleared back to its own panel base color, while the copied target pixel is painted with the other panel's base color.
    - All non-selected motif pixels stay unchanged, and incoming swapped pixels overwrite any target motif pixels already at those offsets.

    Key insight:
    - The exchanged subset is not the whole motif or its whole border: it is the strict subset of motif pixels that touch the panel base only diagonally.
    """
    case = _parse_grid(np.asarray(input_grid, dtype=int))
    out = case["grid"].copy()
    for band in case["bands"]:
        left_panel, right_panel = _apply_pair_swap(band["left"], band["right"])
        lr0, lc0, lr1, lc1 = band["left"]["panel_bbox_abs"]
        rr0, rc0, rr1, rc1 = band["right"]["panel_bbox_abs"]
        out[lr0 : lr1 + 1, lc0 : lc1 + 1] = left_panel
        out[rr0 : rr1 + 1, rc0 : rc1 + 1] = right_panel
    return out


def _transform_template(template: np.ndarray, rotate_k: int, flip_lr: bool) -> np.ndarray:
    h = int(template[:, 0].max()) + 1
    w = int(template[:, 1].max()) + 1
    patch = np.zeros((h, w), dtype=int)
    patch[template[:, 0], template[:, 1]] = 1
    if flip_lr:
        patch = np.fliplr(patch)
    if rotate_k:
        patch = np.rot90(patch, k=int(rotate_k))
    return np.argwhere(patch == 1).astype(int)


def _build_single_panel(panel_h: int, panel_w: int, base_color: int, motif_color: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    candidates = []
    for template in _BASE_TEMPLATES:
        for rotate_k in range(4):
            for flip_lr in (False, True):
                coords = _transform_template(template, rotate_k, flip_lr)
                h = int(coords[:, 0].max()) + 1
                w = int(coords[:, 1].max()) + 1
                if h <= panel_h - 2 and w <= panel_w - 2:
                    candidates.append(coords)
    assert candidates

    for _ in range(200):
        coords = candidates[int(rng.integers(0, len(candidates)))]
        h = int(coords[:, 0].max()) + 1
        w = int(coords[:, 1].max()) + 1
        top = int(rng.integers(1, panel_h - h))
        left = int(rng.integers(1, panel_w - w))
        placed = coords + np.array([top, left], dtype=int)

        panel = np.full((panel_h, panel_w), int(base_color), dtype=int)
        panel[placed[:, 0], placed[:, 1]] = int(motif_color)
        info = _analyze_panel_patch(panel)

        motif_bbox = bbox(panel == motif_color)
        assert motif_bbox is not None
        mr0, mc0, mr1, mc1 = motif_bbox
        motif_h = mr1 - mr0 + 1
        motif_w = mc1 - mc0 + 1
        if motif_h * motif_w == len(info["motif_offsets"]):
            continue
        if not (1 <= len(info["diag_offsets"]) <= len(info["motif_offsets"]) // 2):
            continue
        if len(info["motif_offsets"]) < 5 or len(info["motif_offsets"]) > 8:
            continue
        return panel, info
    raise RuntimeError("failed to build a valid panel")


def _build_panel_pair(panel_h: int, panel_w: int, colors: np.ndarray, force_overlap: bool, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    left_base, left_motif, right_base, right_motif = [int(x) for x in colors]
    for _ in range(300):
        left_panel, left_info = _build_single_panel(panel_h, panel_w, left_base, left_motif, rng)
        right_panel, right_info = _build_single_panel(panel_h, panel_w, right_base, right_motif, rng)
        if left_info["diag_offsets"] == right_info["diag_offsets"] and left_info["motif_offsets"] == right_info["motif_offsets"]:
            continue
        overlap = bool(
            (left_info["diag_offsets"] & right_info["non_diag_offsets"])
            or (right_info["diag_offsets"] & left_info["non_diag_offsets"])
        )
        if force_overlap and not overlap:
            continue
        if not force_overlap and rng.random() < 0.4 and overlap:
            continue
        return left_panel, right_panel
    raise RuntimeError("failed to build a valid panel pair")


def _validate_case(grid: np.ndarray) -> bool:
    try:
        case = _parse_grid(grid)
        out = main(grid)
        wrong_out = _apply_full_motif_swap(case)

        assert grid.shape == out.shape
        assert 1 <= len(case["bands"]) <= 3
        assert np.any(grid == BG)
        assert np.any(grid == SEP)
        assert not np.array_equal(out, grid)
        assert not np.array_equal(out, wrong_out)

        changed_bands = 0
        for band in case["bands"]:
            left = band["left"]
            right = band["right"]
            assert left["panel"].shape == right["panel"].shape
            assert left["base_color"] != right["base_color"]
            assert len(left["motif_offsets"]) >= 5 and len(right["motif_offsets"]) >= 5
            assert 1 <= len(left["diag_offsets"]) <= len(left["motif_offsets"]) // 2
            assert 1 <= len(right["diag_offsets"]) <= len(right["motif_offsets"]) // 2
            assert left["non_diag_offsets"] and right["non_diag_offsets"]
            assert left["diag_offsets"] != left["motif_offsets"]
            assert right["diag_offsets"] != right["motif_offsets"]

            lr0, lc0, lr1, lc1 = left["panel_bbox_abs"]
            rr0, rc0, rr1, rc1 = right["panel_bbox_abs"]
            if not np.array_equal(grid[lr0 : lr1 + 1, lc0 : lc1 + 1], out[lr0 : lr1 + 1, lc0 : lc1 + 1]):
                changed_bands += 1
            if not np.array_equal(grid[rr0 : rr1 + 1, rc0 : rc1 + 1], out[rr0 : rr1 + 1, rc0 : rc1 + 1]):
                changed_bands += 1

        assert changed_bands >= 2
        return True
    except Exception:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_bands = int(rng.integers(1, 4))
    cell_w = int(rng.integers(7, 11))
    cell_heights = [int(rng.integers(7, 10)) for _ in range(n_bands)]
    total_h = sum(cell_heights) + (n_bands - 1)
    total_w = cell_w * 2 + 1
    if total_h > 30 or total_w > 30:
        return None

    grid = np.full((total_h, total_w), BG, dtype=int)
    grid[:, cell_w] = SEP

    row = 0
    for band_idx, cell_h in enumerate(cell_heights):
        panel_h = int(rng.integers(5, cell_h + 1))
        panel_w = int(rng.integers(5, cell_w + 1))
        if band_idx == 0 and panel_h == cell_h and panel_w == cell_w:
            panel_h -= 1

        colors = rng.choice(np.array(PALETTE, dtype=int), size=4, replace=False)
        force_overlap = bool(rng.integers(0, 2))
        try:
            left_panel, right_panel = _build_panel_pair(panel_h, panel_w, colors, force_overlap, rng)
        except RuntimeError:
            return None

        left_cell = np.full((cell_h, cell_w), BG, dtype=int)
        right_cell = np.full((cell_h, cell_w), BG, dtype=int)

        left_r = int(rng.integers(0, cell_h - panel_h + 1))
        left_c = int(rng.integers(0, cell_w - panel_w + 1))
        right_r = int(rng.integers(0, cell_h - panel_h + 1))
        right_c = int(rng.integers(0, cell_w - panel_w + 1))

        left_cell[left_r : left_r + panel_h, left_c : left_c + panel_w] = left_panel
        right_cell[right_r : right_r + panel_h, right_c : right_c + panel_w] = right_panel

        grid[row : row + cell_h, 0:cell_w] = left_cell
        grid[row : row + cell_h, cell_w + 1 :] = right_cell

        row += cell_h
        if band_idx != n_bands - 1:
            grid[row, :] = SEP
            row += 1

    return grid if _validate_case(grid) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
