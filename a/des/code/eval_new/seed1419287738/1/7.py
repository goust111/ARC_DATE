from __future__ import annotations

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
FENCE_COLOR = 9
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]
MAX_GRID_SIZE = 30
MAX_GEN_ATTEMPTS = 300
MARGIN = 1
GAP = 1

_RNG = np.random.default_rng()
_GEN_COUNTER = 0
_DIRS = ("top", "bottom", "left", "right")
_OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}

_TEMPLATE_LIBRARY = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[0, 1], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _dilate8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0 = max(0, dr)
            r1 = min(h, h + dr)
            c0 = max(0, dc)
            c1 = min(w, w + dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _enclosed_region(blocked: np.ndarray) -> np.ndarray:
    blocked = np.asarray(blocked, dtype=bool)
    h, w = blocked.shape
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if not blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return (~blocked) & (~seen)


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    rows = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != BG and np.all(grid[r, :] == grid[r, 0])
    ]
    assert len(rows) == 1, "expected exactly one full-width separator row"
    sep_r = int(rows[0])
    return sep_r, int(grid[sep_r, 0])


def _marker_direction(marker_rc: tuple[int, int], interior: np.ndarray) -> str:
    r, c = marker_rc
    h, w = interior.shape
    hits: list[str] = []
    if r + 1 < h and interior[r + 1, c]:
        hits.append("top")
    if r - 1 >= 0 and interior[r - 1, c]:
        hits.append("bottom")
    if c + 1 < w and interior[r, c + 1]:
        hits.append("left")
    if c - 1 >= 0 and interior[r, c - 1]:
        hits.append("right")
    assert len(hits) == 1, "marker must face exactly one interior direction"
    return hits[0]


def _tile_from_side(interior: np.ndarray, template: np.ndarray, side: str) -> np.ndarray:
    interior = np.asarray(interior, dtype=bool)
    template = np.asarray(template, dtype=bool)
    coords = np.argwhere(interior)
    assert coords.size > 0, "fence interior must be non-empty"
    ir0, ic0, ir1, ic1 = _bbox(coords)
    th, tw = map(int, template.shape)
    anchor_r = int(ir0)
    anchor_c = int(ic0)
    if side == "bottom":
        anchor_r = int(ir1 - th + 1)
    elif side == "right":
        anchor_c = int(ic1 - tw + 1)

    rows = np.arange(interior.shape[0], dtype=int)[:, None]
    cols = np.arange(interior.shape[1], dtype=int)[None, :]
    pattern = template[(rows - anchor_r) % th, (cols - anchor_c) % tw]
    return interior & pattern


def _parse_templates(panel: np.ndarray) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}
    for coords in _components(panel != BG):
        r0, c0, r1, c1 = _bbox(coords)
        sub = panel[r0 : r1 + 1, c0 : c1 + 1]
        colors = [int(v) for v in np.unique(sub) if int(v) != BG]
        assert len(colors) == 1, "each template must be monochrome"
        color = colors[0]
        assert color not in templates, "template colors must be unique"
        templates[color] = sub == color
    assert templates, "bottom panel must contain templates"
    return templates


def _parse_fences(panel: np.ndarray) -> list[dict]:
    fences: list[dict] = []
    for coords in _components(panel != BG):
        r0, c0, r1, c1 = _bbox(coords)
        sub = panel[r0 : r1 + 1, c0 : c1 + 1]
        vals, counts = np.unique(sub[sub != BG], return_counts=True)
        assert len(vals) == 2, "each fence must contain a border color and one marker color"
        border_color = int(vals[int(np.argmax(counts))])
        marker_colors = [int(v) for v in vals if int(v) != border_color]
        assert len(marker_colors) == 1
        marker_color = marker_colors[0]
        assert int(np.sum(sub == marker_color)) == 1, "marker must be a single pixel"

        interior = _enclosed_region(sub != BG)
        assert interior.any(), "fence must enclose an interior"
        assert len(_components(interior)) == 1, "fence interior must be one 4-connected region"
        marker_rc = tuple(map(int, np.argwhere(sub == marker_color)[0]))
        direction = _marker_direction(marker_rc, interior)

        fences.append(
            {
                "bbox": (r0, c0, r1, c1),
                "border_color": border_color,
                "marker_color": marker_color,
                "direction": direction,
                "interior": interior,
            }
        )
    assert fences, "top panel must contain fences"
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match lower templates to upper marked fences and tile each fence interior from the marker-indicated side.

    Input structure:
    - Black background with one full-width grey separator row splitting the grid into an upper fence panel and a lower template panel.
    - Upper panel: several closed maroon fence components; each fence border contains exactly one non-maroon colored marker pixel replacing one border cell.
    - Lower panel: the same number of small monochrome connected templates, each color-matched to exactly one fence marker.
    - Fence interiors are black in the input.

    Rule:
    - Pair each lower template with the upper fence whose marker has the same color.
    - Read the marker side from the fence geometry: marker on the top border means tile downward, bottom means upward, left means rightward, and right means leftward.
    - Repeat the unrotated template on its own bounding-box lattice, anchored to the indicated interior side, and keep only the template pixels that fall inside the enclosed fence interior.
    - Preserve the fence border, marker, and grey separator, and clear the whole lower panel to black.

    Key insight:
    - Each marker encodes both which template to use and which side fixes the tiling phase, so solving the puzzle requires combining color matching with border-side direction decoding.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_r, _sep_color = _find_separator(g)
    top = g[:sep_r, :]
    bottom = g[sep_r + 1 :, :]
    assert top.size and bottom.size, "both panels must be non-empty"

    templates = _parse_templates(bottom)
    fences = _parse_fences(top)
    assert {f["marker_color"] for f in fences} == set(templates), "template colors must match marker colors"

    out = g.copy()
    out[sep_r + 1 :, :] = BG

    for fence in fences:
        r0, c0, r1, c1 = fence["bbox"]
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        fill = _tile_from_side(fence["interior"], templates[fence["marker_color"]], fence["direction"])
        sub[fill] = fence["marker_color"]

    return out


def _rows_vary(mask: np.ndarray) -> bool:
    return len({tuple(map(int, row)) for row in mask.astype(int)}) > 1


def _cols_vary(mask: np.ndarray) -> bool:
    return len({tuple(map(int, col)) for col in mask.T.astype(int)}) > 1


def _peek_directions(n: int) -> list[str]:
    start = _GEN_COUNTER % 4
    return [_DIRS[(start + i) % 4] for i in range(n)]


def _distribute(total: int, buckets: int) -> list[int]:
    if buckets == 1:
        return [int(total)]
    if total <= 0:
        return [0] * buckets
    return list(map(int, _RNG.multinomial(total, np.full(buckets, 1.0 / buckets))))


def _template_candidates(direction: str, used: set[int]) -> list[int]:
    candidates: list[int] = []
    for idx, mask in enumerate(_TEMPLATE_LIBRARY):
        if idx in used:
            continue
        if direction in ("top", "bottom"):
            if mask.shape[0] < 2 or not _rows_vary(mask):
                continue
        else:
            if mask.shape[1] < 2 or not _cols_vary(mask):
                continue
        candidates.append(idx)
    return candidates


def _make_fence_asset(marker_color: int, direction: str, template: np.ndarray) -> np.ndarray | None:
    th, tw = map(int, template.shape)
    for _ in range(120):
        inner_h = int(_RNG.integers(max(3, th), 8))
        inner_w = int(_RNG.integers(max(3, tw), 8))

        inner = np.ones((inner_h, inner_w), dtype=bool)
        if inner_h >= 4 and inner_w >= 4 and _RNG.random() < 0.65:
            bite_h = int(_RNG.integers(1, inner_h - 1))
            bite_w = int(_RNG.integers(1, inner_w - 1))
            corner = int(_RNG.integers(0, 4))
            if corner == 0:
                inner[:bite_h, :bite_w] = False
            elif corner == 1:
                inner[:bite_h, inner_w - bite_w :] = False
            elif corner == 2:
                inner[inner_h - bite_h :, :bite_w] = False
            else:
                inner[inner_h - bite_h :, inner_w - bite_w :] = False

        if len(_components(inner)) != 1:
            continue

        padded_inner = np.zeros((inner_h + 2, inner_w + 2), dtype=bool)
        padded_inner[1:-1, 1:-1] = inner
        border = _dilate8(padded_inner) & (~padded_inner)
        if len(_components(border)) != 1:
            continue

        candidates: list[tuple[int, int]] = []
        for r, c in np.argwhere(border):
            hits: list[str] = []
            if r + 1 < padded_inner.shape[0] and padded_inner[r + 1, c]:
                hits.append("top")
            if r - 1 >= 0 and padded_inner[r - 1, c]:
                hits.append("bottom")
            if c + 1 < padded_inner.shape[1] and padded_inner[r, c + 1]:
                hits.append("left")
            if c - 1 >= 0 and padded_inner[r, c - 1]:
                hits.append("right")
            if hits == [direction]:
                candidates.append((int(r), int(c)))
        if not candidates:
            continue

        if direction == "top":
            extreme = min(r for r, _ in candidates)
            candidates = [rc for rc in candidates if rc[0] == extreme]
        elif direction == "bottom":
            extreme = max(r for r, _ in candidates)
            candidates = [rc for rc in candidates if rc[0] == extreme]
        elif direction == "left":
            extreme = min(c for _, c in candidates)
            candidates = [rc for rc in candidates if rc[1] == extreme]
        else:
            extreme = max(c for _, c in candidates)
            candidates = [rc for rc in candidates if rc[1] == extreme]

        fill = _tile_from_side(padded_inner, template, direction)
        other = _tile_from_side(padded_inner, template, _OPPOSITE[direction])
        if np.array_equal(fill, other):
            continue

        asset = np.zeros(border.shape, dtype=int)
        asset[border] = FENCE_COLOR
        marker_r, marker_c = candidates[int(_RNG.integers(len(candidates)))]
        asset[marker_r, marker_c] = marker_color
        return asset
    return None


def _colored_template(mask: np.ndarray, color: int) -> np.ndarray:
    asset = np.zeros(mask.shape, dtype=int)
    asset[mask] = color
    return asset


def _sample_row_groups(n: int) -> list[list[int]]:
    options = {
        2: [[2], [1, 1]],
        3: [[3], [2, 1], [1, 2]],
        4: [[4], [2, 2], [3, 1], [1, 3]],
    }[n]
    counts = options[int(_RNG.integers(len(options)))]
    order = list(range(n))
    _RNG.shuffle(order)
    rows: list[list[int]] = []
    idx = 0
    for cnt in counts:
        rows.append(order[idx : idx + cnt])
        idx += cnt
    return rows


def _panel_min_size(assets: list[np.ndarray], rows: list[list[int]]) -> tuple[int, int]:
    row_heights = [max(int(assets[i].shape[0]) for i in row) for row in rows]
    row_widths = [
        sum(int(assets[i].shape[1]) for i in row) + GAP * (len(row) - 1)
        for row in rows
    ]
    panel_h = sum(row_heights) + GAP * (len(rows) - 1) + 2 * MARGIN
    panel_w = max(row_widths) + 2 * MARGIN
    return int(panel_h), int(panel_w)


def _paste(canvas: np.ndarray, asset: np.ndarray, top: int, left: int) -> None:
    h, w = asset.shape
    sub = canvas[top : top + h, left : left + w]
    mask = asset != BG
    assert not np.any(sub[mask] != BG), "assets must not overlap"
    sub[mask] = asset[mask]


def _render_panel(assets: list[np.ndarray], rows: list[list[int]], panel_h: int, width: int) -> np.ndarray:
    panel = np.zeros((panel_h, width), dtype=int)
    row_heights = [max(int(assets[i].shape[0]) for i in row) for row in rows]

    base_h = sum(row_heights) + GAP * (len(rows) - 1) + 2 * MARGIN
    extra_rows = _distribute(panel_h - base_h, len(rows) + 1)

    y = MARGIN + extra_rows[0]
    for row_idx, row in enumerate(rows):
        row_h = row_heights[row_idx]
        row_width = sum(int(assets[i].shape[1]) for i in row) + GAP * (len(row) - 1)
        extra_cols = _distribute(width - (row_width + 2 * MARGIN), len(row) + 1)

        x = MARGIN + extra_cols[0]
        for item_idx, asset_idx in enumerate(row):
            asset = assets[asset_idx]
            y_off = int(_RNG.integers(0, row_h - asset.shape[0] + 1))
            _paste(panel, asset, y + y_off, x)
            x += int(asset.shape[1])
            if item_idx < len(row) - 1:
                x += GAP + extra_cols[item_idx + 1]

        y += row_h
        if row_idx < len(rows) - 1:
            y += GAP + extra_rows[row_idx + 1]
    return panel


def _build_attempt(n: int, directions: list[str]) -> np.ndarray | None:
    directions = list(directions)
    _RNG.shuffle(directions)
    colors = list(map(int, _RNG.choice(COLOR_POOL, size=n, replace=False)))

    top_assets: list[np.ndarray] = []
    bottom_assets: list[np.ndarray] = []
    used_templates: set[int] = set()

    for color, direction in zip(colors, directions, strict=False):
        candidates = _template_candidates(direction, used_templates)
        if not candidates:
            return None
        template_idx = int(candidates[int(_RNG.integers(len(candidates)))])
        used_templates.add(template_idx)
        template = _TEMPLATE_LIBRARY[template_idx]
        fence_asset = _make_fence_asset(color, direction, template)
        if fence_asset is None:
            return None
        top_assets.append(fence_asset)
        bottom_assets.append(_colored_template(template, color))

    top_rows = _sample_row_groups(n)
    bottom_rows = _sample_row_groups(n)

    top_h_min, top_w_min = _panel_min_size(top_assets, top_rows)
    bottom_h_min, bottom_w_min = _panel_min_size(bottom_assets, bottom_rows)
    min_w = max(top_w_min, bottom_w_min)
    min_h = top_h_min + 1 + bottom_h_min
    if min_w > MAX_GRID_SIZE or min_h > MAX_GRID_SIZE:
        return None

    extra_w = int(_RNG.integers(0, min(7, MAX_GRID_SIZE - min_w) + 1))
    width = min_w + extra_w

    height_budget = MAX_GRID_SIZE - min_h
    extra_top = int(_RNG.integers(0, min(4, height_budget) + 1))
    height_budget -= extra_top
    extra_bottom = int(_RNG.integers(0, min(4, height_budget) + 1))
    top_h = top_h_min + extra_top
    bottom_h = bottom_h_min + extra_bottom

    top_panel = _render_panel(top_assets, top_rows, top_h, width)
    bottom_panel = _render_panel(bottom_assets, bottom_rows, bottom_h, width)

    if len(_components(top_panel != BG)) != n or len(_components(bottom_panel != BG)) != n:
        return None

    grid = np.zeros((top_h + 1 + bottom_h, width), dtype=int)
    grid[:top_h, :] = top_panel
    grid[top_h, :] = SEPARATOR_COLOR
    grid[top_h + 1 :, :] = bottom_panel

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    for _ in range(MAX_GEN_ATTEMPTS):
        n = int(_RNG.integers(2, 5))
        directions = _peek_directions(n)
        grid = _build_attempt(n, directions)
        if grid is not None:
            _GEN_COUNTER += n
            return grid
    raise RuntimeError("failed to generate a valid input")
