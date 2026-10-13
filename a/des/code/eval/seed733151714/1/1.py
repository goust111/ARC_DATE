import random

import numpy as np


BG = 0
RED = 2
GREY = 5
DIRS = ("up", "down", "left", "right")
NON_SPECIAL_COLORS = [1, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 300


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


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _bands(separators: list[int]) -> list[tuple[int, int]]:
    bands = []
    for a, b in zip(separators, separators[1:]):
        if b - a > 1:
            bands.append((a + 1, b - 1))
    return bands


def _direction_from_open_box(region: np.ndarray) -> str:
    mask = region != BG
    pts = np.argwhere(mask)
    assert len(pts) > 0, "each region must contain an open box"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    box = mask[r0 : r1 + 1, c0 : c1 + 1]
    counts = {
        "up": int(box[0, :].sum()),
        "down": int(box[-1, :].sum()),
        "left": int(box[:, 0].sum()),
        "right": int(box[:, -1].sum()),
    }
    best = min(counts, key=counts.get)
    best_value = counts[best]
    assert list(counts.values()).count(best_value) == 1, "open-box direction must be unique"
    return best


def _parse_panel(panel: np.ndarray) -> tuple[str, np.ndarray]:
    assert panel.shape[0] >= 5 and panel.shape[1] >= 5
    assert np.all(panel[0, :] == GREY) and np.all(panel[-1, :] == GREY)
    assert np.all(panel[:, 0] == GREY) and np.all(panel[:, -1] == GREY)
    inner = panel[1:-1, 1:-1]

    matches = []
    if np.all(inner[0, :] == RED):
        matches.append("up")
    if np.all(inner[-1, :] == RED):
        matches.append("down")
    if np.all(inner[:, 0] == RED):
        matches.append("left")
    if np.all(inner[:, -1] == RED):
        matches.append("right")
    assert len(matches) == 1, "panel must encode exactly one direction"

    direction = matches[0]
    if direction == "up":
        texture = inner[1:, :].copy()
    elif direction == "down":
        texture = inner[:-1, :].copy()
    elif direction == "left":
        texture = inner[:, 1:].copy()
    else:
        texture = inner[:, :-1].copy()
    assert texture.size > 0 and not np.any(texture == RED)
    return direction, texture


def _tile_texture(height: int, width: int, texture: np.ndarray) -> np.ndarray:
    th, tw = texture.shape
    rr = np.arange(height)[:, None] % th
    cc = np.arange(width)[None, :] % tw
    return texture[rr, cc]


def _render_region(height: int, width: int, texture: np.ndarray) -> np.ndarray:
    out = np.full((height + 2, width + 2), GREY, dtype=int)
    out[1:-1, 1:-1] = _tile_texture(height, width, texture)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Choose textures from bottom direction panels using each region's open side,
    then reorder the regions from smallest to largest area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    grey_comps = _components(g == GREY)
    assert len(grey_comps) >= 5, "expected one lattice and four panel frames"
    lattice_comp = max(grey_comps, key=len)
    lr0, lc0, lr1, lc1 = _bbox(lattice_comp)
    lattice = g[lr0 : lr1 + 1, lc0 : lc1 + 1]

    sep_rows = [r for r in range(lattice.shape[0]) if np.all(lattice[r, :] == GREY)]
    sep_cols = [c for c in range(lattice.shape[1]) if np.all(lattice[:, c] == GREY)]
    row_bands = _bands(sep_rows)
    col_bands = _bands(sep_cols)
    assert row_bands and col_bands, "lattice must define rectangular regions"

    panel_map: dict[str, np.ndarray] = {}
    for comp in grey_comps:
        if comp is lattice_comp:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        panel = g[r0 : r1 + 1, c0 : c1 + 1]
        direction, texture = _parse_panel(panel)
        panel_map[direction] = texture
    assert set(panel_map) == set(DIRS), "must contain four unique direction panels"

    regions = []
    for rb in row_bands:
        for cb in col_bands:
            r0, r1 = rb
            c0, c1 = cb
            region = lattice[r0 : r1 + 1, c0 : c1 + 1]
            direction = _direction_from_open_box(region)
            regions.append(
                {
                    "height": region.shape[0],
                    "width": region.shape[1],
                    "area": int(region.shape[0] * region.shape[1]),
                    "order_key": (lr0 + r0, lc0 + c0),
                    "texture": panel_map[direction],
                }
            )

    regions.sort(key=lambda item: (item["area"], item["order_key"][0], item["order_key"][1]))
    rendered = [_render_region(item["height"], item["width"], item["texture"]) for item in regions]
    out_h = max(tile.shape[0] for tile in rendered)
    out_w = sum(tile.shape[1] for tile in rendered) + (len(rendered) - 1)
    out = np.full((out_h, out_w), BG, dtype=int)

    x = 0
    for tile in rendered:
        h, w = tile.shape
        out[:h, x : x + w] = tile
        x += w + 1
    return out


def _build_open_box(height: int, width: int, color: int, direction: str) -> np.ndarray:
    box = np.zeros((height, width), dtype=int)
    box[0, :] = color
    box[-1, :] = color
    box[:, 0] = color
    box[:, -1] = color
    if direction == "up":
        box[0, 1:-1] = BG
    elif direction == "down":
        box[-1, 1:-1] = BG
    elif direction == "left":
        box[1:-1, 0] = BG
    else:
        box[1:-1, -1] = BG
    return box


def _make_texture(rng: random.Random, existing: list[np.ndarray]) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        color_count = rng.randint(2, min(4, len(NON_SPECIAL_COLORS)))
        palette = rng.sample(NON_SPECIAL_COLORS, color_count)
        tex = np.array([[rng.choice(palette) for _ in range(w)] for _ in range(h)], dtype=int)
        if len(set(map(int, tex.ravel()))) < 2:
            continue
        if any(np.array_equal(tex, other) for other in existing):
            continue
        return tex
    raise RuntimeError("failed to build a unique texture")


def _panel_from_texture(texture: np.ndarray, direction: str) -> np.ndarray:
    th, tw = texture.shape
    if direction == "up":
        inner = np.vstack([np.full((1, tw), RED, dtype=int), texture])
    elif direction == "down":
        inner = np.vstack([texture, np.full((1, tw), RED, dtype=int)])
    elif direction == "left":
        inner = np.hstack([np.full((th, 1), RED, dtype=int), texture])
    else:
        inner = np.hstack([texture, np.full((th, 1), RED, dtype=int)])

    panel = np.full((inner.shape[0] + 2, inner.shape[1] + 2), BG, dtype=int)
    panel[0, :] = GREY
    panel[-1, :] = GREY
    panel[:, 0] = GREY
    panel[:, -1] = GREY
    panel[1:-1, 1:-1] = inner
    return panel


def _unique_areas(rng: random.Random, n_rows: int, n_cols: int) -> tuple[list[int], list[int]] | None:
    for _ in range(200):
        heights = [rng.randint(6, 10) for _ in range(n_rows)]
        widths = [rng.randint(6, 10) for _ in range(n_cols)]
        areas = [h * w for h in heights for w in widths]
        if len(set(areas)) == len(areas):
            return heights, widths
    return None


def _layout_panels(
    rng: random.Random, panels: list[np.ndarray]
) -> tuple[list[tuple[int, np.ndarray]], int]:
    order = list(range(len(panels)))
    rng.shuffle(order)
    placed: list[tuple[int, np.ndarray]] = []
    x = 0
    for idx in order:
        if placed:
            x += rng.randint(1, 2)
        placed.append((x, panels[idx]))
        x += panels[idx].shape[1]
    return placed, x


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rows = rng.choice([2, 3])
    n_cols = rng.choice([2, 3, 4])
    if n_rows * n_cols < 4:
        return None

    dims = _unique_areas(rng, n_rows, n_cols)
    if dims is None:
        return None
    heights, widths = dims

    sep_rows = [0]
    for h in heights:
        sep_rows.append(sep_rows[-1] + h + 1)
    sep_cols = [0]
    for w in widths:
        sep_cols.append(sep_cols[-1] + w + 1)
    lattice_h = sep_rows[-1] + 1
    lattice_w = sep_cols[-1] + 1

    lattice = np.full((lattice_h, lattice_w), BG, dtype=int)
    lattice[sep_rows, :] = GREY
    lattice[:, sep_cols] = GREY

    directions = list(DIRS)
    rng.shuffle(directions)
    while len(directions) < n_rows * n_cols:
        extra = list(DIRS)
        rng.shuffle(extra)
        directions.extend(extra)
    directions = directions[: n_rows * n_cols]
    rng.shuffle(directions)

    idx = 0
    for ri, h in enumerate(heights):
        for ci, w in enumerate(widths):
            direction = directions[idx]
            idx += 1
            region = lattice[
                sep_rows[ri] + 1 : sep_rows[ri] + 1 + h,
                sep_cols[ci] + 1 : sep_cols[ci] + 1 + w,
            ]
            box_h = rng.randint(4, h - 2)
            box_w = rng.randint(4, w - 2)
            top = rng.randint(1, h - box_h - 1)
            left = rng.randint(1, w - box_w - 1)
            color = rng.choice(NON_SPECIAL_COLORS)
            box = _build_open_box(box_h, box_w, color, direction)
            region[top : top + box_h, left : left + box_w] = np.maximum(
                region[top : top + box_h, left : left + box_w], box
            )

    textures: list[np.ndarray] = []
    panels: list[np.ndarray] = []
    for direction in DIRS:
        texture = _make_texture(rng, textures)
        textures.append(texture)
        panels.append(_panel_from_texture(texture, direction))

    placed_panels, panel_span = _layout_panels(rng, panels)
    max_panel_h = max(panel.shape[0] for _, panel in placed_panels)
    gap = rng.randint(2, 4)
    top_margin = rng.randint(1, 2)
    bottom_margin = rng.randint(1, 2)
    total_h = top_margin + lattice_h + gap + max_panel_h + bottom_margin
    total_w = max(lattice_w, panel_span) + rng.randint(4, 8)
    grid = np.full((total_h, total_w), BG, dtype=int)

    lattice_x = rng.randint(1, total_w - lattice_w - 1)
    lattice_y = top_margin
    grid[lattice_y : lattice_y + lattice_h, lattice_x : lattice_x + lattice_w] = lattice

    panel_x_offset = rng.randint(1, total_w - panel_span - 1)
    panel_y = lattice_y + lattice_h + gap
    for x, panel in placed_panels:
        h, w = panel.shape
        grid[panel_y : panel_y + h, panel_x_offset + x : panel_x_offset + x + w] = panel

    try:
        out = main(grid)
    except Exception:
        return None
    if out.size == 0 or np.any(out == RED):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
