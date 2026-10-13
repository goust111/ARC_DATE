from __future__ import annotations

import numpy as np


BG_COLOR = 7
SEPARATOR_COLOR = 3
FOREGROUND_COLORS = [1, 2, 4, 5, 6, 8, 9]
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- solver ---------------


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _anchor(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    idx = np.lexsort((coords[:, 1], coords[:, 0]))[0]
    return int(coords[idx, 0]), int(coords[idx, 1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge complementary same-color masks across a horizontal separator.

    Input structure
    - Background is orange.
    - A single green full-width separator row splits the grid into equal top and bottom panels.
    - Each panel contains one foreground object per color, and the color sets match across the two panels.

    Rule
    - For each foreground color, take the union of its top-panel mask and bottom-panel mask using panel-relative coordinates.
    - Draw those unions on a fresh single-panel canvas with the background color.
    - Process colors by decreasing top-panel area; later colors overwrite earlier ones.

    Key insight
    - The two panels are complementary views of the same color-keyed objects, so the output is the overlay of matched masks rather than either panel alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 3

    h, _w = g.shape
    assert h % 2 == 1, "grid must split into two equal-height panels"
    mid = h // 2

    bg = _get_bg(g)
    sep = int(g[mid, 0])
    assert sep != bg
    assert np.all(g[mid, :] == sep), "separator row must be monochrome and full width"

    top = g[:mid, :]
    bottom = g[mid + 1 :, :]
    top_colors = {int(color) for color in np.unique(top) if color not in {bg, sep}}
    bottom_colors = {int(color) for color in np.unique(bottom) if color not in {bg, sep}}
    assert top_colors and top_colors == bottom_colors

    ordered = []
    for color in top_colors:
        top_mask = top == color
        bottom_mask = bottom == color
        assert top_mask.any() and bottom_mask.any()
        ordered.append(
            {
                "color": int(color),
                "area": int(top_mask.sum()),
                "anchor": _anchor(top_mask),
                "union": top_mask | bottom_mask,
            }
        )

    ordered.sort(key=lambda item: (-item["area"], item["anchor"][0], item["anchor"][1]))

    out = np.full(top.shape, bg, dtype=int)
    for item in ordered:
        out[item["union"]] = item["color"]
    return out


# --------------- generator ---------------

PANEL_H_RANGE = (8, 13)
PANEL_W_RANGE = (9, 15)
COLOR_COUNT_RANGE = (2, 5)
TOP_AREA_MIN = 3
TOP_AREA_MAX = 10
BOTTOM_AREA_MIN = 2
BOTTOM_AREA_MAX = 10
SHAPE_RETRIES = 80
PLACEMENT_RETRIES = 80
GENERATION_RETRIES = 300


def _feasible_dims(area: int, panel_h: int, panel_w: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for h in range(1, min(panel_h, area + 1) + 1):
        for w in range(1, min(panel_w, area + 1) + 1):
            if area > h * w:
                continue
            if h * w > area + 4:
                continue
            if h > 1 and w > 1 and area < max(h, w):
                continue
            dims.append((h, w))
    return dims


def _random_connected_mask(
    rng: np.random.Generator, area: int, panel_h: int, panel_w: int
) -> np.ndarray | None:
    dims = _feasible_dims(area, panel_h, panel_w)
    if not dims:
        return None

    for _ in range(SHAPE_RETRIES):
        h, w = dims[int(rng.integers(len(dims)))]
        mask = np.zeros((h, w), dtype=bool)
        seed = (int(rng.integers(h)), int(rng.integers(w)))
        mask[seed] = True
        frontier: set[tuple[int, int]] = set()

        def add_frontier(r: int, c: int) -> None:
            for dr, dc in DIRS4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                    frontier.add((rr, cc))

        add_frontier(*seed)
        while int(mask.sum()) < area:
            if not frontier:
                break
            candidates = tuple(frontier)
            rr, cc = candidates[int(rng.integers(len(candidates)))]
            frontier.remove((rr, cc))
            mask[rr, cc] = True
            add_frontier(rr, cc)

        if int(mask.sum()) != area:
            continue

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        mask = mask[rows][:, cols]
        if area >= 5 and int(mask.sum()) == mask.size and rng.random() < 0.65:
            continue
        return mask
    return None


def _mark_blocked(blocked: np.ndarray, full_mask: np.ndarray) -> None:
    h, w = blocked.shape
    for r_raw, c_raw in np.argwhere(full_mask):
        r = int(r_raw)
        c = int(c_raw)
        blocked[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True


def _place_mask(
    rng: np.random.Generator,
    panel: np.ndarray,
    blocked: np.ndarray,
    color: int,
    area: int,
    required_overlap: np.ndarray | None = None,
    forbidden_overlap: np.ndarray | None = None,
    forbid_full_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    panel_h, panel_w = panel.shape
    for _ in range(PLACEMENT_RETRIES):
        local = _random_connected_mask(rng, area, panel_h, panel_w)
        if local is None:
            return None
        h, w = local.shape
        positions = []
        for r0 in range(panel_h - h + 1):
            for c0 in range(panel_w - w + 1):
                if np.any(blocked[r0 : r0 + h, c0 : c0 + w] & local):
                    continue
                if required_overlap is not None and not np.any(
                    required_overlap[r0 : r0 + h, c0 : c0 + w] & local
                ):
                    continue
                if forbidden_overlap is not None and np.any(
                    forbidden_overlap[r0 : r0 + h, c0 : c0 + w] & local
                ):
                    continue
                positions.append((r0, c0))

        if not positions:
            continue

        for idx in rng.permutation(len(positions)):
            r0, c0 = positions[int(idx)]
            full_mask = np.zeros_like(panel, dtype=bool)
            full_mask[r0 : r0 + h, c0 : c0 + w] = local
            if forbid_full_mask is not None and np.array_equal(full_mask, forbid_full_mask):
                continue
            panel[full_mask] = color
            _mark_blocked(blocked, full_mask)
            return full_mask
    return None


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = [(r, c)]
            seen[r, c] = True
            cells = []
            while queue:
                cr, cc = queue.pop()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _render_expected(
    panel_shape: tuple[int, int], specs: list[dict], top_masks: dict[int, np.ndarray], bottom_masks: dict[int, np.ndarray]
) -> np.ndarray:
    out = np.full(panel_shape, BG_COLOR, dtype=int)
    ordered = sorted(specs, key=lambda item: (-item["top_area"],))
    for spec in ordered:
        color = int(spec["color"])
        out[top_masks[color] | bottom_masks[color]] = color
    return out


def _pairwise_overlap_pairs(unions: dict[int, np.ndarray]) -> set[tuple[int, int]]:
    colors = list(unions)
    pairs: set[tuple[int, int]] = set()
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            if np.any(unions[color_a] & unions[color_b]):
                pairs.add(tuple(sorted((int(color_a), int(color_b)))))
    return pairs


def _validate_generated(
    grid: np.ndarray,
    specs: list[dict],
    top_masks: dict[int, np.ndarray],
    bottom_masks: dict[int, np.ndarray],
    overlap_mode: bool,
    overlap_src: int | None,
    overlap_dst: int | None,
) -> bool:
    if _get_bg(grid) != BG_COLOR:
        return False

    mid = grid.shape[0] // 2
    if not np.all(grid[mid, :] == SEPARATOR_COLOR):
        return False

    top = grid[:mid, :]
    bottom = grid[mid + 1 :, :]
    colors = {int(spec["color"]) for spec in specs}
    if {int(color) for color in np.unique(top) if color not in {BG_COLOR, SEPARATOR_COLOR}} != colors:
        return False
    if {int(color) for color in np.unique(bottom) if color not in {BG_COLOR, SEPARATOR_COLOR}} != colors:
        return False

    areas = [int(spec["top_area"]) for spec in specs]
    if len(set(areas)) != len(areas):
        return False

    for color in colors:
        top_mask = top_masks[color]
        bottom_mask = bottom_masks[color]
        if np.array_equal(top_mask, bottom_mask):
            return False
        if len(_components(top_mask)) != 1 or len(_components(bottom_mask)) != 1:
            return False

    expected = _render_expected(top.shape, specs, top_masks, bottom_masks)
    if np.array_equal(expected, top) or np.array_equal(expected, bottom):
        return False

    if any(not np.any(expected == color) for color in colors):
        return False

    unions = {color: top_masks[color] | bottom_masks[color] for color in colors}
    overlap_pairs = _pairwise_overlap_pairs(unions)
    if overlap_mode:
        if overlap_src is None or overlap_dst is None:
            return False
        if overlap_pairs != {tuple(sorted((overlap_src, overlap_dst)))}:
            return False
    elif overlap_pairs:
        return False

    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    panel_h = int(rng.integers(*PANEL_H_RANGE))
    panel_w = int(rng.integers(*PANEL_W_RANGE))
    color_count = int(rng.integers(*COLOR_COUNT_RANGE))
    if color_count > len(FOREGROUND_COLORS):
        return None

    panel_area = panel_h * panel_w
    max_top_area = min(TOP_AREA_MAX, max(TOP_AREA_MIN + color_count - 1, panel_area // 7))
    if max_top_area < TOP_AREA_MIN + color_count - 1:
        return None

    top_area_pool = list(range(TOP_AREA_MIN, max_top_area + 1))
    if len(top_area_pool) < color_count:
        return None

    colors = rng.choice(np.array(FOREGROUND_COLORS, dtype=int), size=color_count, replace=False).tolist()
    top_areas = rng.choice(np.array(top_area_pool, dtype=int), size=color_count, replace=False).tolist()
    top_areas.sort(reverse=True)
    bottom_areas = rng.integers(BOTTOM_AREA_MIN, BOTTOM_AREA_MAX + 1, size=color_count).tolist()
    overlap_mode = bool(rng.integers(0, 2))

    specs = []
    for idx, color in enumerate(colors):
        specs.append(
            {
                "color": int(color),
                "top_area": int(top_areas[idx]),
                "bottom_area": int(bottom_areas[idx]),
            }
        )

    top_panel = np.full((panel_h, panel_w), BG_COLOR, dtype=int)
    top_blocked = np.zeros((panel_h, panel_w), dtype=bool)
    top_masks: dict[int, np.ndarray] = {}
    for spec in sorted(specs, key=lambda item: (-item["top_area"], item["color"])):
        mask = _place_mask(
            rng=rng,
            panel=top_panel,
            blocked=top_blocked,
            color=int(spec["color"]),
            area=int(spec["top_area"]),
        )
        if mask is None:
            return None
        top_masks[int(spec["color"])] = mask

    overlap_src = None
    overlap_dst = None
    ranked = sorted(specs, key=lambda item: -item["top_area"])
    if overlap_mode:
        src_index = int(rng.integers(0, len(ranked) - 1))
        dst_index = int(rng.integers(src_index + 1, len(ranked)))
        overlap_src = int(ranked[src_index]["color"])
        overlap_dst = int(ranked[dst_index]["color"])

    all_top = np.zeros((panel_h, panel_w), dtype=bool)
    for mask in top_masks.values():
        all_top |= mask

    bottom_panel = np.full((panel_h, panel_w), BG_COLOR, dtype=int)
    bottom_blocked = np.zeros((panel_h, panel_w), dtype=bool)
    bottom_masks: dict[int, np.ndarray] = {}
    bottom_order = sorted(specs, key=lambda item: (-item["bottom_area"], item["color"]))
    if overlap_mode and overlap_dst is not None:
        bottom_order.sort(key=lambda item: item["color"] != overlap_dst)

    for spec in bottom_order:
        color = int(spec["color"])
        required_overlap = None
        forbidden_overlap = all_top & ~top_masks[color]
        if overlap_mode and color == overlap_dst and overlap_src is not None:
            required_overlap = top_masks[overlap_src]
            forbidden_overlap = np.zeros((panel_h, panel_w), dtype=bool)
            for other_color, other_mask in top_masks.items():
                if other_color not in {color, overlap_src}:
                    forbidden_overlap |= other_mask

        mask = _place_mask(
            rng=rng,
            panel=bottom_panel,
            blocked=bottom_blocked,
            color=color,
            area=int(spec["bottom_area"]),
            required_overlap=required_overlap,
            forbidden_overlap=forbidden_overlap,
            forbid_full_mask=top_masks[color],
        )
        if mask is None:
            return None
        bottom_masks[color] = mask

    grid = np.full((panel_h * 2 + 1, panel_w), BG_COLOR, dtype=int)
    grid[:panel_h, :] = top_panel
    grid[panel_h, :] = SEPARATOR_COLOR
    grid[panel_h + 1 :, :] = bottom_panel

    if not _validate_generated(
        grid=grid,
        specs=specs,
        top_masks=top_masks,
        bottom_masks=bottom_masks,
        overlap_mode=overlap_mode,
        overlap_src=overlap_src,
        overlap_dst=overlap_dst,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-panel input")
