# Combo variant of e3721c99 (index 16)
# combo_spec: D1/3+D10/0
# Name: 外接矩形宽高比匹配+大型图例

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste
from scipy.ndimage import binary_dilation, generate_binary_structure


# Design analysis for combo D1/3 + D10/0
# 1. Concrete behavior:
#    - Original task: each colored legend sprite was small; solver counted legend holes,
#      counted grey-sprite holes, and recolored exact matches.
#    - Fused task: legends become large edge-panel structures with many holes, but the
#      holes are now a decoy. The real key is each component's bbox width:height ratio.
#      Example: a tall 12x5 legend, a square 8x8 legend, and a wide 6x11 legend define
#      three target ratios. A grey sprite with bbox 7x3 is recolored by the tall legend
#      because 3/7 is closest to 5/12, even if its hole count disagrees completely.
# 2. Complexity calibration:
#    - Ratios are continuous, so the generator restricts them to three clearly separated
#      families (~0.42, 1.0, ~1.8). This keeps matching visible to humans and prevents
#      ambiguous nearest-neighbor ties.
#    - Large legends retain multiple enclosed holes so D10/0 is visually active rather
#      than collapsing to plain rectangles.
# 3. Broken original assumptions:
#    - Original `_legend_mapping()` assumed hole counts are unique keys. That is invalid:
#      the fused solver must ignore hole count and instead read bbox ratio.
#    - Original grey-sprite handling erased unlisted hole counts. That is also invalid:
#      under nearest-ratio matching every grey sprite maps to some legend.
#    - Original small-legend thresholds are too weak; the fused solver now expects large
#      non-border-touching colored components in the divider-separated panel.
# 4. Placement legality:
#    - Legends sit in a reserved top/bottom panel separated by a full divider row.
#    - Grey sprites are placed only in the workspace rectangle, with an 8-neighbor moat
#      so they neither overlap nor visually merge.
# 5. Visual observability:
#    - Humans can see three repeated sprite families whose overall proportions match the
#      three large legend proportions, while legend hole counts are larger and misleading.
#    - Because the same family appears multiple times per sample, color inference is not
#      based on a single fragile example.
# 6. Tidiness:
#    - Only one legend panel is used, so legends and sprites never compete for the same
#      region. Legends use distinct colors; sprites stay grey until recolored.
# 7. Non-degeneracy:
#    - The legends are built with distinct hole counts, but sprite holes are not aligned
#      to those values. Generation rejects any sample whose aspect-ratio output equals the
#      old hole-count output, so the combo cannot silently fall back to the original task.


BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
PANEL_H = 14
LEGEND_COLORS = [2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 250
PLACE_RETRIES = 80
STRUCT8 = generate_binary_structure(2, 2)

LEGEND_FAMILIES = {
    "tall": {
        "legend_options": [(12, 5)],
        "sprite_options": [(6, 3), (7, 3), (8, 4)],
    },
    "square": {
        "legend_options": [(8, 8), (9, 9)],
        "sprite_options": [(5, 5), (6, 6), (7, 7)],
    },
    "wide": {
        "legend_options": [(6, 11), (7, 12)],
        "sprite_options": [(4, 7), (5, 9), (6, 10)],
    },
}


def _hole_count(mask: np.ndarray) -> int:
    """Count 4-connected holes in a boolean component mask."""
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            touches = False
            while st:
                r, c = st.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                holes += 1
    return holes


def _mask_from_points(pts: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(pts)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[r - r0, c - c0] = True
    return mask


def _bbox_ratio_points(pts: np.ndarray) -> float:
    r0, c0, r1, c1 = bbox_coords(pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return w / h


def _legend_entries(g: np.ndarray, bg: int, grey: int) -> list[tuple[float, int]]:
    """
    Find one large legend component per legend color and return (bbox_ratio, color).
    Divider color is ignored because it touches the border.
    """
    h, w = g.shape
    entries: list[tuple[float, int]] = []
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey, DIV)]
    for color in colors:
        comps = components_mask(g == color, connectivity=4)
        candidates = []
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            if hh < 5 or ww < 5:
                continue
            candidates.append(pts)
        assert len(candidates) == 1, f"color {color} must have exactly one large legend component"
        entries.append((_bbox_ratio_points(candidates[0]), color))
    assert len(entries) >= 2, "grid must contain at least two legend entries"
    return entries


def _nearest_legend_color(ratio: float, legends: list[tuple[float, int]]) -> int:
    return min(legends, key=lambda item: (abs(item[0] - ratio), item[0]))[1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey sprites by comparing their bounding-box aspect ratios to a large color legend.

    Input structure:
    - The grid has a full divider row that separates a legend panel from the workspace.
    - Inside the panel are several large, hole-rich colored legend components; grey (5) sprites are
      scattered in the workspace and are much smaller than the legends.

    Rule:
    - Treat each large colored legend component as one reference whose key feature is its bbox
      width:height ratio.
    - For every grey connected component, compute its own bbox ratio and recolor it with the color of
      the legend whose ratio is closest.

    Key insight:
    - The legend interiors are deliberately complicated, but that topology is irrelevant here; the
      useful signal is the overall proportion of each component's bounding rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = GREY

    legends = _legend_entries(g, bg, grey)
    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        ratio = _bbox_ratio_points(pts)
        new_color = _nearest_legend_color(ratio, legends)
        for r, c in pts:
            out[r, c] = new_color
    return out


def _build_tall_legend(h: int, w: int, color: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    for rr in range(2, h - 2, 2):
        patch[rr, 1:-1] = color
    return patch


def _build_square_legend(h: int, w: int, color: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[h // 2, 1:-1] = color
    patch[1:-1, w // 2] = color
    return patch


def _build_wide_legend(h: int, w: int, color: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    cols = [w // 3, (2 * w) // 3]
    for cc in cols:
        patch[1:-1, cc] = color
    return patch


def _legend_patch(family: str, h: int, w: int, color: int) -> np.ndarray:
    if family == "tall":
        return _build_tall_legend(h, w, color)
    if family == "square":
        return _build_square_legend(h, w, color)
    return _build_wide_legend(h, w, color)


def _choose_legends() -> list[dict]:
    families = ["tall", "square", "wide"]
    colors = random.sample(LEGEND_COLORS, len(families))
    random.shuffle(families)
    legends = []
    for family, color in zip(families, colors):
        h, w = random.choice(LEGEND_FAMILIES[family]["legend_options"])
        legends.append({
            "family": family,
            "color": color,
            "h": h,
            "w": w,
            "ratio": w / h,
        })
    return legends


def _place_legends_and_divider(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    legends = _choose_legends()
    total_w = sum(item["w"] for item in legends) + (len(legends) - 1)
    c0 = random.randint(1, GRID_SIZE - 1 - total_w)
    top = random.choice([True, False])

    if top:
        div_r = PANEL_H
        grid[div_r, :] = DIV
        blocked[: div_r + 1, :] = True
    else:
        div_r = GRID_SIZE - 1 - PANEL_H
        grid[div_r, :] = DIV
        blocked[div_r:, :] = True

    c = c0
    for item in legends:
        if top:
            r = random.randint(1, PANEL_H - 1 - item["h"])
        else:
            panel_top = div_r + 1
            r = random.randint(panel_top + 1, GRID_SIZE - 1 - item["h"])
        item["r"] = r
        item["c"] = c
        patch = _legend_patch(item["family"], item["h"], item["w"], item["color"])
        paste(grid, patch, r, c)
        c += item["w"] + 1
    return legends


def _paint_path(mask: np.ndarray, start: tuple[int, int], target: tuple[int, int]) -> None:
    r, c = start
    tr, tc = target
    mask[r, c] = True
    while (r, c) != (tr, tc):
        moves = []
        if r < tr:
            moves.append((r + 1, c))
        elif r > tr:
            moves.append((r - 1, c))
        if c < tc:
            moves.append((r, c + 1))
        elif c > tc:
            moves.append((r, c - 1))
        r, c = random.choice(moves)
        mask[r, c] = True


def _make_blob_sprite(h: int, w: int) -> np.ndarray:
    for _ in range(PLACE_RETRIES):
        mask = np.zeros((h, w), dtype=bool)
        center = (random.randint(h // 3, (2 * h) // 3), random.randint(w // 3, (2 * w) // 3))
        anchors = [
            (0, random.randint(0, w - 1)),
            (h - 1, random.randint(0, w - 1)),
            (random.randint(0, h - 1), 0),
            (random.randint(0, h - 1), w - 1),
        ]
        mask[center] = True
        for anchor in anchors:
            _paint_path(mask, anchor, center)

        min_area = max(int(np.ceil(0.42 * h * w)), int(mask.sum()))
        max_area = max(min_area, int(np.floor(0.68 * h * w)))
        target_area = random.randint(min_area, max_area)

        frontier = set()
        for rr, cc in np.argwhere(mask):
            for nr, nc in neighbors4(int(rr), int(cc)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))

        while int(mask.sum()) < target_area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            rr, cc = cell
            if mask[rr, cc]:
                continue
            mask[rr, cc] = True
            for nr, nc in neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))

        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        return np.where(mask, GREY, BG).astype(int)
    raise RuntimeError(f"failed to build sprite with bbox {h}x{w}")


def _sprite_plan(legends: list[dict]) -> list[tuple[np.ndarray, int]]:
    family_to_color = {item["family"]: item["color"] for item in legends}
    items: list[tuple[np.ndarray, int]] = []
    for family in ("tall", "square", "wide"):
        count = random.randint(2, 3)
        for _ in range(count):
            h, w = random.choice(LEGEND_FAMILIES[family]["sprite_options"])
            items.append((_make_blob_sprite(h, w), family_to_color[family]))
    random.shuffle(items)
    items.sort(key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True)
    return items


def _workspace_rows(blocked: np.ndarray) -> tuple[int, int]:
    blocked_rows = np.any(blocked, axis=1)
    if blocked_rows[0]:
        first_free = int(np.argmax(~blocked_rows))
        return first_free, GRID_SIZE - 1
    last_free = GRID_SIZE - 1 - int(np.argmax((~blocked_rows)[::-1]))
    return 0, last_free


def _place_sprites(grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[dict]) -> bool:
    items = _sprite_plan(legends)
    occupied = np.zeros_like(grid, dtype=bool)
    r_lo, r_hi = _workspace_rows(blocked)
    c_lo, c_hi = 1, GRID_SIZE - 2

    for sprite, color in items:
        h, w = sprite.shape
        sprite_mask = sprite == GREY
        forbidden = binary_dilation(occupied | blocked, structure=STRUCT8)
        candidates = [(r, c) for r in range(r_lo, r_hi - h + 2) for c in range(c_lo, c_hi - w + 2)]
        random.shuffle(candidates)
        placed = False
        for r, c in candidates:
            if np.any(forbidden[r : r + h, c : c + w] & sprite_mask):
                continue
            paste(grid, sprite, r, c, transparent=BG)
            occupied[r : r + h, c : c + w] |= sprite_mask
            out_patch = np.where(sprite_mask, color, BG)
            paste(expected, out_patch, r, c, transparent=BG)
            placed = True
            break
        if not placed:
            return False
    return True


def _hole_based_output(grid: np.ndarray) -> np.ndarray:
    """Reference old-style behavior used only to reject degenerate generated samples."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    mapping: dict[int, int] = {}
    for color in [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, GREY, DIV)]:
        comps = components_mask(g == color, connectivity=4)
        candidates = []
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 == 0 or c0 == 0 or r1 == GRID_SIZE - 1 or c1 == GRID_SIZE - 1:
                continue
            candidates.append(pts)
        assert len(candidates) == 1
        hc = _hole_count(_mask_from_points(candidates[0]))
        assert hc not in mapping
        mapping[hc] = color

    out = g.copy()
    for pts in components_mask(g == GREY, connectivity=4):
        hc = _hole_count(_mask_from_points(pts))
        new_color = mapping.get(hc, bg)
        for r, c in pts:
            out[r, c] = new_color
    return out


def _fusion_gap_metric(grid: np.ndarray) -> int:
    return int(np.sum(main(grid) != _hole_based_output(grid)))


def _min_ratio_margin(grid: np.ndarray) -> float:
    g = np.asarray(grid, dtype=int)
    legends = _legend_entries(g, get_bg(g), GREY)
    min_margin = 999.0
    for pts in components_mask(g == GREY, connectivity=4):
        ratio = _bbox_ratio_points(pts)
        diffs = sorted(abs(lr - ratio) for lr, _ in legends)
        min_margin = min(min_margin, diffs[1] - diffs[0])
    return float(min_margin)


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    expected = np.zeros_like(grid)
    blocked = np.zeros_like(grid, dtype=bool)

    legends = _place_legends_and_divider(grid, blocked)
    expected[:, :] = grid
    if not _place_sprites(grid, expected, blocked, legends):
        return None
    if _min_ratio_margin(grid) < 0.18:
        return None
    if _fusion_gap_metric(grid) <= 0:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e3721c99 combo D1/3+D10/0 input")
