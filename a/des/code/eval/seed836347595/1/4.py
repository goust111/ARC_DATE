import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREY = 5
ORANGE = 7
MAROON = 9

TEXTURES = ("solid", "hstripe", "vstripe", "checker")
TICK_COLORS = (1, 3, 4, 6, 8)

SHAPES = {
    "hline": frozenset({(0, -1), (0, 0), (0, 1)}),
    "vline": frozenset({(-1, 0), (0, 0), (1, 0)}),
    "corner_dr": frozenset({(0, 0), (1, 0), (0, 1)}),
    "corner_ul": frozenset({(0, 0), (-1, 0), (0, -1)}),
}

PANEL_CELL_H = 3
PANEL_SAMPLE_W = 3
PANEL_SHAPE_W = 3
PANEL_COUNT = 4
PANEL_INNER_H = PANEL_COUNT * PANEL_CELL_H + (PANEL_COUNT - 1)
PANEL_INNER_W = PANEL_SAMPLE_W + 1 + PANEL_SHAPE_W
PANEL_H = PANEL_INNER_H + 2
PANEL_W = PANEL_INNER_W + 2

GEN_RETRIES = 300


def _bands(separators: list[int], size: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        if start < sep:
            spans.append((start, sep))
        start = sep + 1
    if start < size:
        spans.append((start, size))
    return spans


def _find_panel_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == ORANGE)
    assert len(pts) > 0, "expected an orange legend panel"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    panel = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert panel.shape[0] >= 5 and panel.shape[1] >= 5, "legend panel is too small"
    assert np.all(panel[0, :] == ORANGE) and np.all(panel[-1, :] == ORANGE)
    assert np.all(panel[:, 0] == ORANGE) and np.all(panel[:, -1] == ORANGE)
    return int(r0), int(c0), int(r1), int(c1)


def _component_bbox(passable: np.ndarray, start: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = passable.shape
    sr, sc = start
    assert passable[sr, sc], "start must be inside a passable component"
    seen = np.zeros_like(passable, dtype=bool)
    q = deque([(sr, sc)])
    seen[sr, sc] = True
    r0 = r1 = sr
    c0 = c1 = sc
    while q:
        r, c = q.popleft()
        r0 = min(r0, r)
        r1 = max(r1, r)
        c0 = min(c0, c)
        c1 = max(c1, c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return r0, c0, r1, c1


def _render_texture(kind: str, h: int, w: int, phase: int) -> np.ndarray:
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    if kind == "solid":
        mask = np.ones((h, w), dtype=bool)
    elif kind == "hstripe":
        mask = ((rows + phase) % 2) == 0
        mask = np.repeat(mask, w, axis=1)
    elif kind == "vstripe":
        mask = ((cols + phase) % 2) == 0
        mask = np.repeat(mask, h, axis=0)
    elif kind == "checker":
        mask = ((rows + cols + phase) % 2) == 0
    else:
        raise ValueError(kind)
    out = np.full((h, w), BG, dtype=int)
    out[mask] = MAROON
    return out


def _classify_texture(mask: np.ndarray) -> str:
    h, w = mask.shape
    if np.all(mask):
        return "solid"
    for phase in (0, 1):
        if np.array_equal(mask, np.repeat((((np.arange(h)[:, None] + phase) % 2) == 0), w, axis=1)):
            return "hstripe"
        if np.array_equal(mask, np.repeat((((np.arange(w)[None, :] + phase) % 2) == 0), h, axis=0)):
            return "vstripe"
        if np.array_equal(mask, ((np.add.outer(np.arange(h), np.arange(w)) + phase) % 2) == 0):
            return "checker"
    raise AssertionError("unrecognized texture")


def _shape_offsets(shape_cell: np.ndarray) -> frozenset[tuple[int, int]]:
    pts = np.argwhere(shape_cell == RED)
    assert len(pts) >= 2, "legend shape must contain red pixels"
    cr = shape_cell.shape[0] // 2
    cc = shape_cell.shape[1] // 2
    return frozenset((int(r - cr), int(c - cc)) for r, c in pts)


def _parse_legend(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, frozenset[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    row_seps = [r for r in range(inner.shape[0]) if np.all(inner[r, :] == GREY)]
    col_seps = [c for c in range(inner.shape[1]) if np.all(inner[:, c] == GREY)]
    row_bands = _bands(row_seps, inner.shape[0])
    col_bands = _bands(col_seps, inner.shape[1])
    assert len(row_bands) == PANEL_COUNT, "legend must expose four entries"
    assert len(col_bands) == 2, "legend must have sample and shape columns"

    (s0, s1), (m0, m1) = col_bands
    mapping: dict[str, frozenset[tuple[int, int]]] = {}
    for rb0, rb1 in row_bands:
        sample = inner[rb0:rb1, s0:s1]
        shape_cell = inner[rb0:rb1, m0:m1]
        kind = _classify_texture(sample == MAROON)
        mapping[kind] = _shape_offsets(shape_cell)
    assert len(mapping) == len(TEXTURES), "legend must cover each texture exactly once"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    top_ticks = {int(g[0, c]): int(c) for c in range(1, w) if int(g[0, c]) != BG}
    left_ticks = {int(g[r, 0]): int(r) for r in range(1, h) if int(g[r, 0]) != BG}
    colors = sorted(set(top_ticks) & set(left_ticks))
    assert colors, "expected paired tick colors on the top and left edges"

    panel_bbox = _find_panel_bbox(g)
    legend = _parse_legend(g, panel_bbox)

    out = g.copy()
    pr0, pc0, pr1, pc1 = panel_bbox
    out[pr0 : pr1 + 1, pc0 : pc1 + 1] = BG

    passable = (g != GREY) & (g != ORANGE)
    for color in colors:
        r = left_ticks[color]
        c = top_ticks[color]
        region_bbox = _component_bbox(passable, (r, c))
        r0, c0, r1, c1 = region_bbox
        kind = _classify_texture(g[r0 : r1 + 1, c0 : c1 + 1] == MAROON)
        for dr, dc in legend[kind]:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color
    return out


def _build_panel(rng: random.Random, mapping: dict[str, frozenset[tuple[int, int]]]) -> np.ndarray:
    panel = np.full((PANEL_H, PANEL_W), BG, dtype=int)
    panel[0, :] = ORANGE
    panel[-1, :] = ORANGE
    panel[:, 0] = ORANGE
    panel[:, -1] = ORANGE

    inner = panel[1:-1, 1:-1]
    inner[:, PANEL_SAMPLE_W] = GREY

    order = list(TEXTURES)
    rng.shuffle(order)
    row = 0
    for idx, kind in enumerate(order):
        phase = rng.randint(0, 1)
        inner[row : row + PANEL_CELL_H, :PANEL_SAMPLE_W] = _render_texture(
            kind, PANEL_CELL_H, PANEL_SAMPLE_W, phase
        )
        shape_cell = np.full((PANEL_CELL_H, PANEL_SHAPE_W), BG, dtype=int)
        cr = PANEL_CELL_H // 2
        cc = PANEL_SHAPE_W // 2
        for dr, dc in mapping[kind]:
            shape_cell[cr + dr, cc + dc] = RED
        inner[row : row + PANEL_CELL_H, PANEL_SAMPLE_W + 1 :] = shape_cell
        row += PANEL_CELL_H
        if idx != len(order) - 1:
            inner[row, :] = GREY
            row += 1
    return panel


def _choose_targets(
    rng: random.Random,
    regions: list[dict[str, int | str]],
    k: int,
) -> list[tuple[int, int, str]]:
    candidates: list[tuple[int, int, str]] = []
    by_texture: dict[str, list[tuple[int, int, str]]] = {kind: [] for kind in TEXTURES}
    for reg in regions:
        kind = str(reg["texture"])
        for r in range(int(reg["r0"]) + 1, int(reg["r1"])):
            for c in range(int(reg["c0"]) + 1, int(reg["c1"])):
                pt = (r, c, kind)
                candidates.append(pt)
                by_texture[kind].append(pt)

    preferred = list(TEXTURES)
    rng.shuffle(preferred)

    for _ in range(200):
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        chosen: list[tuple[int, int, str]] = []
        ok = True

        for kind in preferred:
            opts = [pt for pt in by_texture[kind] if pt[0] not in used_rows and pt[1] not in used_cols]
            if not opts:
                ok = False
                break
            pick = rng.choice(opts)
            chosen.append(pick)
            used_rows.add(pick[0])
            used_cols.add(pick[1])

        if not ok:
            continue

        rest = candidates[:]
        rng.shuffle(rest)
        for pt in rest:
            if len(chosen) >= k:
                break
            if pt[0] in used_rows or pt[1] in used_cols:
                continue
            chosen.append(pt)
            used_rows.add(pt[0])
            used_cols.add(pt[1])

        if len(chosen) == k:
            return chosen
    raise RuntimeError("failed to pick non-overlapping targets")


def _gen_attempt(rng: random.Random) -> np.ndarray:
    n_region_rows = rng.randint(2, 3)
    n_region_cols = rng.randint(2, 3)
    region_heights = [rng.randint(5, 7) for _ in range(n_region_rows)]
    region_widths = [rng.randint(5, 7) for _ in range(n_region_cols)]

    slab_h = sum(region_heights) + n_region_rows + 1
    slab_w = sum(region_widths) + n_region_cols + 1

    gap = rng.randint(2, 4)
    right_margin = rng.randint(0, 2)
    bottom_margin = rng.randint(0, 2)
    h = 1 + max(slab_h, PANEL_H) + bottom_margin
    w = 1 + slab_w + gap + PANEL_W + right_margin

    grid = np.full((h, w), BG, dtype=int)

    row_lines = [1]
    cursor = 1
    for rh in region_heights:
        cursor += rh + 1
        row_lines.append(cursor)

    col_lines = [1]
    cursor = 1
    for cw in region_widths:
        cursor += cw + 1
        col_lines.append(cursor)

    for rr in row_lines:
        grid[rr, 1 : slab_w + 1] = GREY
    for cc in col_lines:
        grid[1 : slab_h + 1, cc] = GREY

    textures = list(TEXTURES)
    extra = n_region_rows * n_region_cols - len(TEXTURES)
    textures.extend(rng.choice(TEXTURES) for _ in range(extra))
    rng.shuffle(textures)

    regions: list[dict[str, int | str]] = []
    tidx = 0
    row_cursor = 1
    for rh in region_heights:
        r0 = row_cursor + 1
        r1 = r0 + rh - 1
        col_cursor = 1
        for cw in region_widths:
            c0 = col_cursor + 1
            c1 = c0 + cw - 1
            texture = textures[tidx]
            tidx += 1
            phase = rng.randint(0, 1)
            grid[r0 : r1 + 1, c0 : c1 + 1] = _render_texture(texture, rh, cw, phase)
            regions.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1, "texture": texture})
            col_cursor = c1 + 1
        row_cursor = r1 + 1

    shape_names = list(SHAPES)
    rng.shuffle(shape_names)
    legend_mapping = {kind: SHAPES[name] for kind, name in zip(TEXTURES, shape_names)}

    k = rng.randint(4, len(TICK_COLORS))
    targets = _choose_targets(rng, regions, k)
    colors = list(TICK_COLORS)
    rng.shuffle(colors)
    colors = colors[:k]

    for color, (r, c, _kind) in zip(colors, targets):
        grid[0, c] = color
        grid[r, 0] = color

    panel = _build_panel(rng, legend_mapping)
    panel_r0 = h - PANEL_H - bottom_margin
    panel_c0 = w - PANEL_W - right_margin
    grid[panel_r0 : panel_r0 + PANEL_H, panel_c0 : panel_c0 + PANEL_W] = panel

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            return _gen_attempt(rng)
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
