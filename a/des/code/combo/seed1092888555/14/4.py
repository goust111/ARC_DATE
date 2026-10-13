import random
from typing import Iterable

import numpy as np


BLACK = 0
GREEN = 3
TEMPLATE_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREEN):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        if len(comp) != 2 * (h + w) - 4:
            continue
        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if on_border != (grid[r, c] == GREEN):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            frames.append((r0, c0, r1, c1))
    assert frames, "expected at least one green frame"
    return frames


def _find_templates(grid: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    mask = (grid != BLACK) & (grid != GREEN)
    templates: dict[tuple[int, int], np.ndarray] = {}
    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        patch_mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch_mask), "template rectangles must be fully filled"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        size = tuple(map(int, patch.shape))
        assert size not in templates, "template sizes must be unique"
        templates[size] = patch
    assert templates, "expected at least one template"
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each green rectangular frame with the corner template whose size matches the frame interior.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    templates = _find_templates(g)

    out[(g != BLACK) & (g != GREEN)] = BLACK
    for r0, c0, r1, c1 in frames:
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        key = (inner_h, inner_w)
        assert key in templates, "every frame interior must have a size-matching template"
        out[r0 + 1 : r1, c0 + 1 : c1] = templates[key]
    return out


def _make_template(rng: random.Random, h: int, w: int) -> np.ndarray:
    palette = list(rng.sample(TEMPLATE_COLORS, rng.randint(2, min(4, len(TEMPLATE_COLORS)))))
    style = rng.choice(("checker", "rows", "cols", "bands"))
    patch = np.zeros((h, w), dtype=int)

    if style == "checker":
        for r in range(h):
            for c in range(w):
                patch[r, c] = palette[(r + c) % len(palette)]
    elif style == "rows":
        row_colors = [rng.choice(palette) for _ in range(h)]
        for r in range(h):
            patch[r, :] = row_colors[r]
    elif style == "cols":
        col_colors = [rng.choice(palette) for _ in range(w)]
        for c in range(w):
            patch[:, c] = col_colors[c]
    else:
        split_r = rng.randint(1, h - 1) if h > 1 else 1
        split_c = rng.randint(1, w - 1) if w > 1 else 1
        patch[:split_r, :split_c] = palette[0]
        patch[:split_r, split_c:] = palette[min(1, len(palette) - 1)]
        patch[split_r:, :split_c] = palette[min(2, len(palette) - 1)]
        patch[split_r:, split_c:] = palette[min(3, len(palette) - 1)]

    if len(np.unique(patch)) == 1:
        rr = rng.randrange(h)
        cc = rng.randrange(w)
        choices = [c for c in palette if c != int(patch[rr, cc])]
        if choices:
            patch[rr, cc] = rng.choice(choices)
    return patch


def _cluster_bbox(layout: list[tuple[int, int, np.ndarray]]) -> tuple[int, int]:
    max_r = max(r + patch.shape[0] for r, c, patch in layout)
    max_c = max(c + patch.shape[1] for r, c, patch in layout)
    return max_r, max_c


def _build_corner_layout(
    rng: random.Random, shapes: list[tuple[int, int]]
) -> list[tuple[int, int, np.ndarray]]:
    mode = rng.choice(("row", "col"))
    gap = rng.randint(1, 2)
    layout: list[tuple[int, int, np.ndarray]] = []

    if mode == "row":
        cursor = 0
        for h, w in shapes:
            patch = _make_template(rng, h, w)
            layout.append((0, cursor, patch))
            cursor += w + gap
    else:
        cursor = 0
        for h, w in shapes:
            patch = _make_template(rng, h, w)
            layout.append((cursor, 0, patch))
            cursor += h + gap
    return layout


def _place_layout(
    grid: np.ndarray,
    layout: list[tuple[int, int, np.ndarray]],
    corner: str,
) -> tuple[int, int, int, int]:
    cluster_h, cluster_w = _cluster_bbox(layout)
    h, w = grid.shape
    if corner == "tl":
        base_r, base_c = 0, 0
    elif corner == "tr":
        base_r, base_c = 0, w - cluster_w
    elif corner == "bl":
        base_r, base_c = h - cluster_h, 0
    else:
        base_r, base_c = h - cluster_h, w - cluster_w

    for dr, dc, patch in layout:
        ph, pw = patch.shape
        grid[base_r + dr : base_r + dr + ph, base_c + dc : base_c + dc + pw] = patch
    return base_r, base_c, base_r + cluster_h - 1, base_c + cluster_w - 1


def _can_place_box(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    hh, ww = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not np.any(occupied[rr0:rr1, cc0:cc1])


def _draw_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = GREEN
    grid[r1, c0 : c1 + 1] = GREEN
    grid[r0 : r1 + 1, c0] = GREEN
    grid[r0 : r1 + 1, c1] = GREEN


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_templates = rng.randint(2, 4)
    shapes_pool = [(h, w) for h in range(2, 6) for w in range(2, 6)]
    shapes = rng.sample(shapes_pool, n_templates)
    rng.shuffle(shapes)

    layout = _build_corner_layout(rng, shapes)
    cluster_h, cluster_w = _cluster_bbox(layout)

    grid_h = rng.randint(max(18, cluster_h + 8), max(19, cluster_h + 18))
    grid_w = rng.randint(max(18, cluster_w + 8), max(19, cluster_w + 18))
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    corner = rng.choice(("tl", "tr", "bl", "br"))
    tr0, tc0, tr1, tc1 = _place_layout(grid, layout, corner)

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    rr0 = max(0, tr0 - 2)
    cc0 = max(0, tc0 - 2)
    rr1 = min(grid_h, tr1 + 3)
    cc1 = min(grid_w, tc1 + 3)
    occupied[rr0:rr1, cc0:cc1] = True

    for inner_h, inner_w in shapes:
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        candidates: list[tuple[int, int]] = []
        for r0 in range(grid_h - outer_h + 1):
            for c0 in range(grid_w - outer_w + 1):
                if _can_place_box(occupied, r0, c0, outer_h, outer_w, pad=1):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        _draw_frame(grid, r0, c0, inner_h, inner_w)
        occupied[max(0, r0 - 1) : min(grid_h, r0 + outer_h + 1), max(0, c0 - 1) : min(grid_w, c0 + outer_w + 1)] = True

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        out = main(grid)
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
