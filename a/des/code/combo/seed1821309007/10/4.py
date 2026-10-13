import random

import numpy as np


BG = 0
SEP = 5
MAX_GENERATION_ATTEMPTS = 300


def _solid_bands(mask: np.ndarray, axis: int) -> list[tuple[int, int]]:
    flags = np.all(mask, axis=axis)
    bands: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(flags.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            bands.append((start, idx))
            start = None
    if start is not None:
        bands.append((start, len(flags)))
    return bands


def _gaps(size: int, bands: list[tuple[int, int]]) -> list[tuple[int, int]]:
    parts: list[tuple[int, int]] = []
    pos = 0
    for start, end in bands:
        if pos < start:
            parts.append((pos, start))
        pos = end
    if pos < size:
        parts.append((pos, size))
    return parts


def _parse_panels(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    grey = grid == SEP
    row_bands = _solid_bands(grey, axis=1)
    col_bands = _solid_bands(grey, axis=0)
    row_parts = _gaps(grid.shape[0], row_bands)
    col_parts = _gaps(grid.shape[1], col_bands)
    assert row_parts and col_parts
    panel_hs = {r1 - r0 for r0, r1 in row_parts}
    panel_ws = {c1 - c0 for c0, c1 in col_parts}
    assert len(panel_hs) == 1 and len(panel_ws) == 1
    panel_h = next(iter(panel_hs))
    panel_w = next(iter(panel_ws))
    assert panel_h > 0 and panel_w > 0
    for r0, r1 in row_parts:
        for c0, c1 in col_parts:
            panel = grid[r0:r1, c0:c1]
            assert panel.shape == (panel_h, panel_w)
            assert not np.any(panel == SEP)
    return row_parts, col_parts


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_parts, col_parts = _parse_panels(g)
    restored_rows: list[np.ndarray] = []
    panel_h = row_parts[0][1] - row_parts[0][0]
    panel_w = col_parts[0][1] - col_parts[0][0]

    for r0, r1 in row_parts:
        merged = np.full((panel_h, panel_w), BG, dtype=int)
        for c0, c1 in col_parts:
            panel = g[r0:r1, c0:c1]
            fill = (merged == BG) & (panel != BG)
            merged[fill] = panel[fill]
        assert np.all(merged != BG)
        restored_rows.append(merged)

    return np.vstack(restored_rows)


def _random_master(height: int, width: int, colors: list[int]) -> np.ndarray:
    master = np.full((height, width), colors[0], dtype=int)

    for _ in range(random.randint(2, 4)):
        color = random.choice(colors)
        branch = random.randint(0, 3)
        if branch == 0:
            r0 = random.randint(0, height - 1)
            r1 = random.randint(r0, height - 1)
            c0 = random.randint(0, width - 1)
            c1 = random.randint(c0, width - 1)
            master[r0 : r1 + 1, c0 : c1 + 1] = color
        elif branch == 1:
            row = random.randint(0, height - 1)
            step = random.randint(1, 3)
            master[row::step, :] = color
        elif branch == 2:
            col = random.randint(0, width - 1)
            step = random.randint(1, 3)
            master[:, col::step] = color
        else:
            offset = random.randint(0, max(height, width) - 1)
            for r in range(height):
                c = (r + offset) % width
                master[r, c] = color
                if width >= 3 and random.random() < 0.5:
                    master[r, (c + 1) % width] = color

    for _ in range(random.randint(1, 3)):
        color = random.choice(colors)
        h2 = random.randint(1, max(1, height // 2))
        w2 = random.randint(1, max(1, width // 2))
        r0 = random.randint(0, height - h2)
        c0 = random.randint(0, width - w2)
        master[r0 : r0 + h2, c0 : c0 + w2] = color

    assert np.all(master != BG)
    return master


def _erase_panels(master: np.ndarray, panel_cols: int) -> list[np.ndarray] | None:
    height, width = master.shape
    vis = np.zeros((panel_cols, height, width), dtype=bool)
    order = list(np.ndindex(height, width))
    random.shuffle(order)
    for r, c in order:
        owner_count = random.randint(1, panel_cols)
        owners = random.sample(range(panel_cols), owner_count)
        for owner in owners:
            vis[owner, r, c] = True

    for idx in range(panel_cols):
        if vis[idx].all():
            rr = random.randint(0, height - 1)
            cc = random.randint(0, width - 1)
            donors = [j for j in range(panel_cols) if j != idx and vis[j, rr, cc]]
            if not donors:
                return None
            vis[idx, rr, cc] = False

    if not np.all(vis.any(axis=0)):
        return None
    if not all(np.any(~vis[idx]) for idx in range(panel_cols)):
        return None

    panels: list[np.ndarray] = []
    for idx in range(panel_cols):
        panel = np.full_like(master, BG)
        panel[vis[idx]] = master[vis[idx]]
        panels.append(panel)
    return panels


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    panel_rows = random.randint(1, 3)
    panel_cols = random.randint(2, 4)
    panel_h = random.randint(4, 8)
    panel_w = random.randint(4, 8)
    sep_h = random.randint(1, 2)
    sep_w = random.randint(1, 2)

    palette = [c for c in range(1, 10) if c != SEP]
    random.shuffle(palette)
    row_masters: list[np.ndarray] = []
    row_panels: list[list[np.ndarray]] = []

    for _ in range(panel_rows):
        color_count = random.randint(2, min(5, len(palette)))
        master = _random_master(panel_h, panel_w, palette[:color_count])
        panels = _erase_panels(master, panel_cols)
        if panels is None:
            return None
        row_masters.append(master)
        row_panels.append(panels)

    grid_h = panel_rows * panel_h + (panel_rows - 1) * sep_h
    grid_w = panel_cols * panel_w + (panel_cols - 1) * sep_w
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    for rr in range(panel_rows):
        r0 = rr * (panel_h + sep_h)
        for cc in range(panel_cols):
            c0 = cc * (panel_w + sep_w)
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = row_panels[rr][cc]

    for rr in range(1, panel_rows):
        r0 = rr * panel_h + (rr - 1) * sep_h
        grid[r0 : r0 + sep_h, :] = SEP
    for cc in range(1, panel_cols):
        c0 = cc * panel_w + (cc - 1) * sep_w
        grid[:, c0 : c0 + sep_w] = SEP

    out = np.vstack(row_masters)
    if not np.array_equal(main(grid), out):
        return None
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _build_case()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate valid input")
