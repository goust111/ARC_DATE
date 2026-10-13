import numpy as np


BG = 0
MARKER = 4
SEPARATOR = 9
PALETTE = (1, 2, 3, 5, 6, 7, 8)
SHAPES = (
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
)
MAX_GEN_ATTEMPTS = 300


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    seen = {tuple(map(int, pts[0]))}
    stack = [tuple(map(int, pts[0]))]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == len(pts)


def _separator_runs(mask: np.ndarray) -> list[int]:
    idx = np.flatnonzero(mask).tolist()
    assert idx, "expected separator lines"
    assert idx[0] == 0 and idx[-1] == len(mask) - 1, "separator frame must include borders"
    for a, b in zip(idx, idx[1:]):
        assert b > a, "separator indices must increase"
    return idx


def _extract_panels(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_full = np.all(g == g[:, :1], axis=1) & (g[:, 0] != BG)
    col_full = np.all(g == g[:1, :], axis=0) & (g[0, :] != BG)
    row_ids = _separator_runs(row_full)
    col_ids = _separator_runs(col_full)

    sep_values = {int(g[r, 0]) for r in row_ids} | {int(g[0, c]) for c in col_ids}
    assert len(sep_values) == 1, "all separators must share one color"
    sep_color = next(iter(sep_values))
    assert np.all(g[row_ids, :] == sep_color)
    assert np.all(g[:, col_ids] == sep_color)

    panel_hs = [b - a - 1 for a, b in zip(row_ids, row_ids[1:])]
    panel_ws = [b - a - 1 for a, b in zip(col_ids, col_ids[1:])]
    assert panel_hs and panel_ws
    assert min(panel_hs) > 0 and min(panel_ws) > 0
    assert len(set(panel_hs)) == 1 and len(set(panel_ws)) == 1, "panels must be equal-sized"

    panels = []
    for r0, r1 in zip(row_ids, row_ids[1:]):
        for c0, c1 in zip(col_ids, col_ids[1:]):
            panels.append((r0 + 1, r1, c0 + 1, c1))
    return panels, panel_hs[0], panel_ws[0]


def _crop_to_bbox(panel: np.ndarray) -> np.ndarray:
    pts = np.argwhere(panel != BG)
    assert pts.size > 0, "expected a foreground object"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return panel[r0 : r1 + 1, c0 : c1 + 1]


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    assert factor >= 1
    return np.repeat(np.repeat(sprite, factor, axis=0), factor, axis=1)


def _center_paste(dst: np.ndarray, sprite: np.ndarray) -> None:
    h, w = dst.shape
    sh, sw = sprite.shape
    top = (h - sh) // 2
    left = (w - sw) // 2
    for r in range(sh):
        rr = top + r
        if not (0 <= rr < h):
            continue
        for c in range(sw):
            cc = left + c
            if 0 <= cc < w and sprite[r, c] != BG:
                dst[rr, cc] = sprite[r, c]


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each separator-defined panel, the yellow marker's distance from the panel's
    top edge encodes the scale factor. Scale the non-marker object by that factor,
    clear the panel, and place the scaled result centered inside the panel, clipped
    to the panel bounds.
    """
    g = np.asarray(grid, dtype=int)
    panels, panel_h, panel_w = _extract_panels(g)
    out = g.copy()

    for rs, re, cs, ce in panels:
        panel = g[rs:re, cs:ce]
        assert panel.shape == (panel_h, panel_w)

        marker_pts = np.argwhere(panel == MARKER)
        assert marker_pts.shape == (1, 2), "each panel must contain one yellow marker"
        mr, mc = map(int, marker_pts[0])
        factor = mr + 1
        assert 1 <= factor <= 3, "marker distance must encode 1..3"

        sprite_panel = panel.copy()
        sprite_panel[sprite_panel == MARKER] = BG
        assert np.any(sprite_panel != BG), "panel must contain a non-marker object"
        assert _is_connected(sprite_panel != BG), "panel object must be connected"

        sprite = _crop_to_bbox(sprite_panel)
        scaled = _scale_sprite(sprite, factor)

        new_panel = np.full_like(panel, BG)
        _center_paste(new_panel, scaled)
        out[rs:re, cs:ce] = new_panel

    return out


def _make_colored_shape(rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    shape = np.zeros_like(mask, dtype=int)
    colors = rng.choice(PALETTE, size=int(mask.sum()), replace=True)
    if rng.random() < 0.7:
        colors[:] = int(rng.choice(PALETTE))
    idx = 0
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c]:
                shape[r, c] = int(colors[idx])
                idx += 1
    return shape


def _pick_marker_column(
    rng: np.random.Generator, occupied: np.ndarray, marker_row: int
) -> int | None:
    cols = [c for c in range(occupied.shape[1]) if not occupied[marker_row, c]]
    if not cols:
        return None
    detached = []
    for c in cols:
        ok = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = marker_row + dr, c + dc
            if 0 <= rr < occupied.shape[0] and 0 <= cc < occupied.shape[1] and occupied[rr, cc]:
                ok = False
                break
        if ok:
            detached.append(c)
    choice_pool = detached if detached else cols
    return int(choice_pool[int(rng.integers(len(choice_pool)))])


def _build_panel(
    rng: np.random.Generator, panel_h: int, panel_w: int, force_big: bool
) -> np.ndarray | None:
    panel = np.full((panel_h, panel_w), BG, dtype=int)
    factor = int(rng.integers(1, 4))

    candidates = [s for s in SHAPES if s.shape[0] <= min(3, panel_h) and s.shape[1] <= min(3, panel_w)]
    if force_big:
        big = [s for s in candidates if s.shape[0] == 3 or s.shape[1] == 3]
        if big:
            candidates = big
    mask = candidates[int(rng.integers(len(candidates)))]
    shape = _make_colored_shape(rng, mask)

    sh, sw = shape.shape
    for _ in range(40):
        top = int(rng.integers(0, panel_h - sh + 1))
        left = int(rng.integers(0, panel_w - sw + 1))
        occupied = np.zeros((panel_h, panel_w), dtype=bool)
        occupied[top : top + sh, left : left + sw] = shape != BG
        marker_row = factor - 1
        marker_col = _pick_marker_column(rng, occupied, marker_row)
        if marker_col is None:
            continue
        panel.fill(BG)
        sl = panel[top : top + sh, left : left + sw]
        sl[shape != BG] = shape[shape != BG]
        panel[marker_row, marker_col] = MARKER

        non_marker = panel.copy()
        non_marker[non_marker == MARKER] = BG
        if not _is_connected(non_marker != BG):
            continue
        return panel
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    panel_rows = int(rng.integers(2, 4))
    panel_cols = int(rng.integers(2, 4))
    panel_h = int(rng.integers(6, 10))
    panel_w = int(rng.integers(6, 10))

    grid_h = panel_rows * panel_h + panel_rows + 1
    grid_w = panel_cols * panel_w + panel_cols + 1
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[:: panel_h + 1, :] = SEPARATOR
    grid[:, :: panel_w + 1] = SEPARATOR

    force_big_panel = bool(rng.random() < 0.5)
    big_target = None
    if force_big_panel:
        big_target = (
            int(rng.integers(panel_rows)),
            int(rng.integers(panel_cols)),
        )

    for pr in range(panel_rows):
        for pc in range(panel_cols):
            force_big = big_target == (pr, pc)
            panel = _build_panel(rng, panel_h, panel_w, force_big=force_big)
            if panel is None:
                return None
            rs = 1 + pr * (panel_h + 1)
            cs = 1 + pc * (panel_w + 1)
            grid[rs : rs + panel_h, cs : cs + panel_w] = panel

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
