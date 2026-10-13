from __future__ import annotations

import importlib.util
import random
from collections import Counter
from pathlib import Path

import numpy as np


def _load_arc_utils():
    this_file = Path(__file__).resolve()
    for base in (this_file, *this_file.parents):
        candidate = base / "new" / "solvers" / "arc_utils.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("arc_utils_local", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("Failed to locate arc_utils.py from this task file path.")


_arc_utils = _load_arc_utils()
get_bg = _arc_utils.get_bg
bbox = _arc_utils.bbox
bbox_coords = _arc_utils.bbox_coords
components_mask = _arc_utils.components_mask


def _mirror_top_left(top_left: tuple[int, int], panel_box: tuple[int, int, int, int]) -> tuple[int, int]:
    r, c = top_left
    r0, c0, r1, c1 = panel_box
    return r0 + r1 - (r + 1), c0 + c1 - (c + 1)


def _block_quadrant(top_left: tuple[int, int], panel_box: tuple[int, int, int, int]) -> tuple[int, int]:
    r, c = top_left
    r0, c0, r1, c1 = panel_box
    mid_r = (r0 + r1 + 1) / 2
    mid_c = (c0 + c1 + 1) / 2
    return (0 if r + 0.5 < mid_r else 1, 0 if c + 0.5 < mid_c else 1)


def _find_panel(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    h, w = grid.shape
    colors = list(map(int, np.unique(grid)))
    best = None
    best_meta: tuple[bool, int, float] | None = None

    for color in colors:
        for comp in components_mask(grid == color, connectivity=4):
            comp = np.asarray(comp, dtype=int)
            if comp.shape[0] < 8:
                continue
            comp_box = bbox_coords(comp)
            if comp_box is None:
                continue
            r0, c0, r1, c1 = comp_box
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if area < 16:
                continue
            fill_ratio = len(comp) / area
            if fill_ratio < 0.55:
                continue
            interior = r0 > 0 and c0 > 0 and r1 < h - 1 and c1 < w - 1
            score = (interior, len(comp), fill_ratio)
            if best is None or score > best_meta:
                best = (int(color), comp, r0, c0, r1, c1)
                best_meta = score

    if best is None:
        raise AssertionError("main: no rectangular-ish interior component found for panel")

    panel_color, panel_cells, r0, c0, r1, c1 = best
    panel_box = (r0, c0, r1, c1)
    inside = grid[r0 : r1 + 1, c0 : c1 + 1]
    counts = Counter(int(v) for v in inside.ravel())
    if len(counts) < 2:
        raise AssertionError("main: empty panel area after background removal")
    if panel_color not in counts:
        raise AssertionError("main: selected panel color not in panel bbox")
    if counts[panel_color] < 8:
        raise AssertionError("main: panel area is too small or malformed")
    return int(panel_color), r0, c0, r1, c1


def _find_thumbnail(grid: np.ndarray, bg: int, panel_box: tuple[int, int, int, int]) -> tuple[int, int]:
    h, w = grid.shape
    corners = [(0, 0), (0, w - 2), (h - 2, 0), (h - 2, w - 2)]
    candidates: list[tuple[int, int]] = []
    pr0, pc0, pr1, pc1 = panel_box

    for sr, sc in corners:
        if sr < 0 or sc < 0 or sr + 1 >= h or sc + 1 >= w:
            continue
        if sr >= pr0 and sr + 1 <= pr1 and sc >= pc0 and sc + 1 <= pc1:
            continue
        block = grid[sr : sr + 2, sc : sc + 2]
        if block.shape != (2, 2):
            continue
        if int(np.all(block == bg)):
            continue
        non_bg = int(np.count_nonzero(block != bg))
        if non_bg == 0:
            continue
        candidates.append((sr, sc))

    if len(candidates) != 1:
        raise AssertionError("main: expected exactly one corner thumbnail")
    return candidates[0]


def _extract_markers(
    grid: np.ndarray,
    panel_color: int,
    panel_box: tuple[int, int, int, int],
) -> tuple[dict[int, tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = panel_box
    panel_area = np.zeros_like(grid, dtype=bool)
    panel_area[r0 : r1 + 1, c0 : c1 + 1] = True
    components = components_mask((grid != panel_color) & panel_area, connectivity=4)

    by_color: dict[int, tuple[int, int]] = {}
    by_quad: dict[tuple[int, int], tuple[int, int]] = {}

    for comp in components:
        comp = np.asarray(comp, dtype=int)
        if comp.shape != (4, 2):
            continue

        r_min, c_min = comp.min(axis=0)
        r_max, c_max = comp.max(axis=0)
        if r_max - r_min != 1 or c_max - c_min != 1:
            continue
        color = int(grid[int(r_min), int(c_min)])
        if not np.all(grid[r_min : r_max + 1, c_min : c_max + 1] == color):
            continue

        block_top_left = (int(r_min), int(c_min))
        quad = _block_quadrant(block_top_left, panel_box)
        by_color[color] = block_top_left
        by_quad[quad] = block_top_left

    if not by_color:
        raise AssertionError("main: no 2x2 marker blocks found in panel")
    return by_color, by_quad


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the missing quadrant marker by reading a 2×2 thumbnail spec and placing it by central symmetry.

    Input structure
    - The grid is one solid rectangular panel color with several 2×2 solid marker blocks, and a 2×2 thumbnail in one corner.
    - The thumbnail cells are aligned to panel quadrants (top-left, top-right, bottom-left, bottom-right).
    - Exactly one declaration in the thumbnail has no matching marker in the panel, while the opposite quadrant does contain one marker block.

    Rule
    - Find the panel rectangle and its background panel color, then decode quadrant-by-quadrant declarations from the thumbnail.
    - Identify which declared color is missing from the existing panel markers.
    - Take the existing marker in the declaration's opposite quadrant, reflect its 2×2 top-left corner through the panel center, and paint a 2×2 block of the missing color there.
    - Keep all other cells unchanged and return the full grid.

    Key insight
    - The thumbnail is a spec, not the answer: it tells what is missing and which quadrant to recover, while only central-symmetric geometry from the opposite quadrant gives the unique placement.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("main: expected a 2D grid")

    bg = get_bg(g)
    panel_color, pr0, pc0, pr1, pc1 = _find_panel(g)
    panel_box = (pr0, pc0, pr1, pc1)
    th_r, th_c = _find_thumbnail(g, bg, panel_box)

    declarations: dict[tuple[int, int], int] = {}
    for qr in (0, 1):
        for qc in (0, 1):
            v = int(g[th_r + qr, th_c + qc])
            if v != bg:
                declarations[(qr, qc)] = v
    if not declarations:
        raise AssertionError("main: no non-background declaration in thumbnail")
    if len(set(declarations.values())) != len(declarations):
        raise AssertionError("main: duplicate declaration colors are ambiguous")

    marker_by_color, marker_by_quad = _extract_markers(g, panel_color, panel_box)

    missing = [(q, c) for q, c in declarations.items() if c not in marker_by_color]
    if len(missing) != 1:
        raise AssertionError("main: declaration-panel difference must be exactly one marker")
    missing_quad, missing_color = missing[0]

    opposite_quad = (1 - missing_quad[0], 1 - missing_quad[1])
    if opposite_quad not in marker_by_quad:
        raise AssertionError("main: opposite-quadrant anchor marker missing")
    anchor_r, anchor_c = marker_by_quad[opposite_quad]
    dest_r, dest_c = _mirror_top_left((anchor_r, anchor_c), panel_box)

    if not (pr0 <= dest_r <= pr1 - 1 and pc0 <= dest_c <= pc1 - 1):
        raise AssertionError("main: mirrored position falls outside panel")

    out = g.copy()
    out[dest_r : dest_r + 2, dest_c : dest_c + 2] = missing_color
    return out


def _candidate_marker_pos(
    panel_box: tuple[int, int, int, int], quadrant: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = panel_box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError("main: panel must have even dimensions for strict quadrants")

    half_h = h // 2
    half_w = w // 2
    qr, qc = quadrant

    if qr == 0:
        row_min = r0
        row_max = r0 + half_h - 2
    else:
        row_min = r0 + half_h
        row_max = r1 - 1

    if qc == 0:
        col_min = c0
        col_max = c0 + half_w - 2
    else:
        col_min = c0 + half_w
        col_max = c1 - 1

    if row_min > row_max or col_min > col_max:
        raise ValueError("main: panel too small for quadrant marker placement")

    return random.randint(row_min, row_max), random.randint(col_min, col_max)


def _build_once() -> np.ndarray:
    palette = list(range(10))

    bg = random.choice(palette)
    panel = random.choice([c for c in palette if c != bg])
    marker_candidates = [c for c in palette if c not in (bg, panel)]
    random.shuffle(marker_candidates)
    marker_colors = marker_candidates[:4]

    max_try = 200
    for _ in range(max_try):
        panel_h = random.randrange(8, 20, 2)
        panel_w = random.randrange(8, 20, 2)

        top = random.randint(2, 7)
        left = random.randint(2, 7)
        bottom = random.randint(2, 7)
        right = random.randint(2, 7)
        H = panel_h + top + bottom
        W = panel_w + left + right
        if H > 30 or W > 30:
            continue

        grid = np.full((H, W), int(bg), dtype=int)
        pr0, pc0 = top, left
        pr1, pc1 = pr0 + panel_h - 1, pc0 + panel_w - 1

        # Draw the panel.
        grid[pr0 : pr1 + 1, pc0 : pc1 + 1] = int(panel)

        panel_box = (pr0, pc0, pr1, pc1)
        quadrants = [(0, 0), (0, 1), (1, 0), (1, 1)]
        declarations = {q: marker_colors[i] for i, q in enumerate(quadrants)}

        # Spec branch: usually declare all four quadrants, sometimes hide one with background.
        full_spec = random.random() < 0.8
        hidden = None
        if not full_spec:
            hidden = random.choice(quadrants)
            declarations[hidden] = int(bg)

        declaration_list = {q: c for q, c in declarations.items() if c != bg}
        missing_candidates = [
            q
            for q in declaration_list
            if (1 - q[0], 1 - q[1]) in declaration_list
        ]
        if not missing_candidates:
            continue

        missing_quad = random.choice(missing_candidates)
        # Place all declarations except the missing one.
        for q, c in declarations.items():
            if q == missing_quad or c == bg:
                continue
            r, c0 = _candidate_marker_pos(panel_box, q)
            grid[r : r + 2, c0 : c0 + 2] = int(c)

        # Put the spec in one corner; each non-bg cell declares its quadrant color.
        corner = random.choice([(0, 0), (0, W - 2), (H - 2, 0), (H - 2, W - 2)])
        sr, sc = corner
        for q, c in declarations.items():
            grid[sr + q[0], sc + q[1]] = int(c)

        # Avoid collisions with panel (kept by outer margin).
        try:
            _ = main(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("generator: failed to build a valid sample")


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            return _build_once()
        except RuntimeError:
            continue
    raise RuntimeError("generate_input: unable to produce a valid input")
