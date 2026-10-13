import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
SHAPES = ("rectangle", "L", "T")
FILL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

FIELD_H_RANGE = (14, 20)
FIELD_W_RANGE = (22, 32)
EXTRA_REGION_CHOICES = (0, 1, 2)
PLACEMENT_RETRIES = 300
GENERATION_RETRIES = 400


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_edge(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _dilate8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    pts = np.argwhere(mask)
    for r, c in pts:
        for dr, dc in DIR8 + [(0, 0)]:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _local_enclosed_mask(local_grey: np.ndarray) -> np.ndarray:
    local_grey = np.asarray(local_grey, dtype=bool)
    h, w = local_grey.shape
    open_mask = ~local_grey
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if open_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if open_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    enclosed = open_mask & (~outside)
    comps = _components(enclosed, connectivity=4)
    assert len(comps) == 1, "legend thumbnail must enclose exactly one region"
    return _coords_to_mask(comps[0])


def _is_l_oriented(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 2 or w < 2:
        return False
    for bar_h in range(1, h):
        for bar_w in range(1, w):
            cand = np.zeros((h, w), dtype=bool)
            cand[:bar_h, :] = True
            cand[:, :bar_w] = True
            if np.array_equal(mask, cand):
                return True
    return False


def _is_t_oriented(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 2 or w < 3:
        return False
    for bar_h in range(1, h):
        for stem_w in range(1, w - 1):
            for stem_left in range(1, w - stem_w):
                cand = np.zeros((h, w), dtype=bool)
                cand[:bar_h, :] = True
                cand[:, stem_left : stem_left + stem_w] = True
                if np.array_equal(mask, cand):
                    return True
    return False


def _classify_shape(mask: np.ndarray) -> str:
    mask = _trim(np.asarray(mask, dtype=bool))
    assert mask.any(), "empty shape"
    if bool(mask.all()):
        return "rectangle"
    for rot in range(4):
        cur = np.rot90(mask, rot)
        if _is_l_oriented(cur):
            return "L"
        if _is_t_oriented(cur):
            return "T"
    raise AssertionError("unrecognized enclosed shape")


def _shape_thumbnail_base(shape: str) -> np.ndarray:
    if shape == "rectangle":
        return np.ones((1, 1), dtype=bool)
    if shape == "L":
        return np.array([[1, 1], [1, 0]], dtype=bool)
    if shape == "T":
        return np.array([[1, 1, 1], [0, 1, 0]], dtype=bool)
    raise ValueError(shape)


def _render_outline_patch(interior: np.ndarray) -> np.ndarray:
    h, w = interior.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1 : 1 + h, 1 : 1 + w] = interior
    border = _dilate8(padded) & (~padded)
    patch = np.zeros((h + 2, w + 2), dtype=int)
    patch[border] = GREY
    return patch


def _render_legend_group(shape: str, color: int, rng: random.Random) -> np.ndarray:
    interior = np.rot90(_shape_thumbnail_base(shape), rng.randrange(4))
    thumb = _render_outline_patch(interior)
    th, tw = thumb.shape
    group = np.zeros((th, tw + 1), dtype=int)
    group[:, :tw] = thumb
    block_top = (th - 2) // 2
    group[block_top : block_top + 2, tw] = color
    return group


def _parse_legend(grid: np.ndarray) -> tuple[dict[str, int], list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    grey_comps = _components(g == GREY, connectivity=8)
    grey_id = -np.ones((h, w), dtype=int)
    for idx, comp in enumerate(grey_comps):
        grey_id[comp[:, 0], comp[:, 1]] = idx

    color_comps = _components((g != BG) & (g != GREY), connectivity=4)
    mapping: dict[str, int] = {}
    thumb_bboxes: list[tuple[int, int, int, int]] = []
    used_thumb_ids: set[int] = set()
    for comp in color_comps:
        vals = np.unique(g[comp[:, 0], comp[:, 1]])
        assert vals.size == 1, "legend color block must be monochrome"
        r0, c0, r1, c1 = _bbox(comp)
        assert int(comp.shape[0]) == 2 and r1 == r0 + 1 and c0 == c1, "legend block must be 2x1"

        adj_ids: set[int] = set()
        for r, c in comp:
            for dr, dc in DIR8:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == GREY:
                    adj_ids.add(int(grey_id[nr, nc]))
        assert len(adj_ids) == 1, "each legend block must touch exactly one grey thumbnail"
        thumb_id = adj_ids.pop()
        assert thumb_id not in used_thumb_ids, "legend thumbnails must be distinct"
        used_thumb_ids.add(thumb_id)

        thumb = grey_comps[thumb_id]
        tr0, tc0, tr1, tc1 = _bbox(thumb)
        local = np.zeros((tr1 - tr0 + 1, tc1 - tc0 + 1), dtype=bool)
        local[thumb[:, 0] - tr0, thumb[:, 1] - tc0] = True
        interior = _local_enclosed_mask(local)
        shape = _classify_shape(interior)
        mapping[shape] = int(vals[0])
        thumb_bboxes.append((tr0, tc0, tr1, tc1))

    assert set(mapping) == set(SHAPES), "legend must define rectangle, L, and T"
    return mapping, thumb_bboxes


def _bbox_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] <= inner[0] <= inner[2] <= outer[2]
        and outer[1] <= inner[1] <= inner[3] <= outer[3]
    )


def _collect_regions(
    grid: np.ndarray, legend_bboxes: list[tuple[int, int, int, int]]
) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    regions: list[dict[str, object]] = []
    for comp in _components(g == BG, connectivity=4):
        if _touches_edge(comp, h, w):
            continue
        bbox = _bbox(comp)
        if any(_bbox_inside(bbox, lb) for lb in legend_bboxes):
            continue
        mask = _coords_to_mask(comp)
        regions.append(
            {
                "coords": comp,
                "bbox": bbox,
                "shape": _classify_shape(mask),
                "area": int(comp.shape[0]),
            }
        )
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region using the shape->color legend at the bottom.

    Regions are classified from their enclosed black interior masks. The bottom
    legend is read by pairing each grey thumbnail with the adjacent 2x1 color
    block on its right. Regions are then painted from larger area to smaller
    area so later (smaller) regions would overwrite earlier ones if any 1-pixel
    overlap exists.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_bboxes = _parse_legend(g)
    regions = _collect_regions(g, legend_bboxes)

    out = g.copy()
    regions.sort(key=lambda item: (-int(item["area"]), tuple(int(x) for x in item["bbox"])))
    for region in regions:
        color = mapping[str(region["shape"])]
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _make_rectangle_mask(rng: random.Random) -> np.ndarray:
    h = rng.randint(2, 6)
    w = rng.randint(2, 7)
    return np.ones((h, w), dtype=bool)


def _make_l_mask(rng: random.Random) -> np.ndarray:
    for _ in range(80):
        h = rng.randint(3, 7)
        w = rng.randint(3, 7)
        bar_h = rng.randint(1, h - 1)
        bar_w = rng.randint(1, w - 1)
        mask = np.zeros((h, w), dtype=bool)
        mask[:bar_h, :] = True
        mask[:, :bar_w] = True
        if 4 <= int(mask.sum()) < h * w:
            return np.rot90(mask, rng.randrange(4))
    raise RuntimeError("failed to sample L mask")


def _make_t_mask(rng: random.Random) -> np.ndarray:
    for _ in range(80):
        h = rng.randint(3, 7)
        w = rng.randint(3, 8)
        bar_h = rng.randint(1, h - 1)
        stem_w = rng.randint(1, w - 2)
        stem_left = (w - stem_w) // 2
        if stem_left == 0 or stem_left + stem_w == w:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[:bar_h, :] = True
        mask[:, stem_left : stem_left + stem_w] = True
        if 4 <= int(mask.sum()) < h * w:
            return np.rot90(mask, rng.randrange(4))
    raise RuntimeError("failed to sample T mask")


def _make_region_mask(shape: str, rng: random.Random) -> np.ndarray:
    if shape == "rectangle":
        return _make_rectangle_mask(rng)
    if shape == "L":
        return _make_l_mask(rng)
    if shape == "T":
        return _make_t_mask(rng)
    raise ValueError(shape)


def _overlay_mask(field_shape: tuple[int, int], mask: np.ndarray, top: int, left: int) -> np.ndarray:
    full = np.zeros(field_shape, dtype=bool)
    h, w = mask.shape
    full[top : top + h, left : left + w] = mask
    return full


def _straight_segment(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) < 2:
        return False
    if len(np.unique(coords[:, 0])) == 1:
        xs = np.sort(coords[:, 1])
        return bool(np.all(np.diff(xs) == 1))
    if len(np.unique(coords[:, 1])) == 1:
        xs = np.sort(coords[:, 0])
        return bool(np.all(np.diff(xs) == 1))
    return False


def _valid_positions(
    mask: np.ndarray,
    occupied: np.ndarray,
    require_shared: bool,
    forbid_shared: bool,
) -> list[tuple[int, int, int]]:
    field_h, field_w = occupied.shape
    h, w = mask.shape
    prev_border = _dilate8(occupied) & (~occupied)
    occupied_halo = _dilate8(occupied)
    positions: list[tuple[int, int, int]] = []
    for top in range(2, field_h - h - 1):
        for left in range(2, field_w - w - 1):
            placed = _overlay_mask((field_h, field_w), mask, top, left)
            if np.any(placed & occupied):
                continue
            if np.any(placed & occupied_halo):
                continue
            overlap = prev_border & (_dilate8(placed) & (~placed))
            if require_shared:
                if not _straight_segment(overlap):
                    continue
            elif forbid_shared and overlap.any():
                continue
            positions.append((top, left, int(overlap.sum())))
    return positions


def _place_specs(
    specs: list[dict[str, object]], field_h: int, field_w: int, want_shared: bool, rng: random.Random
) -> list[dict[str, object]] | None:
    occupied = np.zeros((field_h, field_w), dtype=bool)
    ordered = sorted(specs, key=lambda item: -int(item["area"]))
    placed: list[dict[str, object]] = []
    for idx, spec in enumerate(ordered):
        require_shared = want_shared and idx == 1
        forbid_shared = not require_shared
        positions = _valid_positions(
            np.asarray(spec["mask"], dtype=bool),
            occupied,
            require_shared=require_shared,
            forbid_shared=forbid_shared,
        )
        if not positions:
            return None
        if require_shared:
            best = max(score for _, _, score in positions)
            pool = [(r, c) for r, c, score in positions if score >= best - 1]
            top, left = rng.choice(pool)
        else:
            top, left, _ = rng.choice(positions)
        placed_mask = _overlay_mask((field_h, field_w), np.asarray(spec["mask"], dtype=bool), top, left)
        occupied |= placed_mask
        coords = np.argwhere(placed_mask)
        placed.append(
            {
                "shape": spec["shape"],
                "mask": spec["mask"],
                "area": spec["area"],
                "top": top,
                "left": left,
                "coords": coords,
            }
        )
    return placed


def _sample_specs(rng: random.Random) -> list[dict[str, object]] | None:
    shapes = list(SHAPES) + [rng.choice(SHAPES) for _ in range(rng.choice(EXTRA_REGION_CHOICES))]
    rng.shuffle(shapes)
    used_areas: set[int] = set()
    specs: list[dict[str, object]] = []
    for shape in shapes:
        picked = None
        for _ in range(100):
            mask = _make_region_mask(shape, rng)
            area = int(mask.sum())
            if area in used_areas:
                continue
            picked = {"shape": shape, "mask": mask, "area": area}
            used_areas.add(area)
            break
        if picked is None:
            return None
        specs.append(picked)
    return specs


def _build_field(field_h: int, field_w: int, placed: list[dict[str, object]]) -> np.ndarray:
    interior = np.zeros((field_h, field_w), dtype=bool)
    for item in placed:
        coords = item["coords"]
        interior[coords[:, 0], coords[:, 1]] = True
    border = _dilate8(interior) & (~interior)
    field = np.zeros((field_h, field_w), dtype=int)
    field[border] = GREY
    return field


def _build_grid(
    field: np.ndarray, color_map: dict[str, int], rng: random.Random
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    groups = []
    order = list(SHAPES)
    rng.shuffle(order)
    for shape in order:
        groups.append(_render_legend_group(shape, color_map[shape], rng))

    legend_h = max(group.shape[0] for group in groups)
    gap = rng.randint(2, 4)
    need_w = sum(group.shape[1] for group in groups) + gap * (len(groups) - 1) + 2

    field_h, field_w = field.shape
    total_w = max(field_w, need_w)
    total_h = field_h + 1 + legend_h
    grid = np.zeros((total_h, total_w), dtype=int)
    grid[:field_h, :field_w] = field

    left = rng.randint(1, total_w - need_w + 1)
    thumb_bboxes: list[tuple[int, int, int, int]] = []
    cursor = left
    for group in groups:
        gh, gw = group.shape
        top = total_h - gh
        grid[top : top + gh, cursor : cursor + gw] = np.where(group != 0, group, grid[top : top + gh, cursor : cursor + gw])
        grey_pts = np.argwhere(group == GREY)
        if len(grey_pts):
            r0, c0 = grey_pts.min(axis=0)
            r1, c1 = grey_pts.max(axis=0)
            thumb_bboxes.append((int(top + r0), int(cursor + c0), int(top + r1), int(cursor + c1)))
        cursor += gw + gap
    return grid, thumb_bboxes


def _validate_generated(
    grid: np.ndarray, placed: list[dict[str, object]], color_map: dict[str, int]
) -> bool:
    mapping, legend_bboxes = _parse_legend(grid)
    if mapping != color_map:
        return False
    regions = _collect_regions(grid, legend_bboxes)
    if len(regions) != len(placed):
        return False
    by_bbox = {tuple(int(x) for x in region["bbox"]): region for region in regions}
    for item in placed:
        bbox = _bbox(item["coords"])
        if bbox not in by_bbox:
            return False
        if by_bbox[bbox]["shape"] != item["shape"]:
            return False

    out = main(grid)
    if not np.all(out[grid == GREY] == GREY):
        return False
    for item in placed:
        coords = item["coords"]
        color = color_map[str(item["shape"])]
        if not np.all(out[coords[:, 0], coords[:, 1]] == color):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    field_h = rng.randint(*FIELD_H_RANGE)
    field_w = rng.randint(*FIELD_W_RANGE)
    specs = _sample_specs(rng)
    if specs is None:
        return None

    want_shared = len(specs) >= 2 and (rng.random() < 0.5)
    placed = _place_specs(specs, field_h, field_w, want_shared=want_shared, rng=rng)
    if placed is None:
        return None

    colors = rng.sample(FILL_COLORS, k=3)
    color_map = {"rectangle": colors[0], "L": colors[1], "T": colors[2]}
    field = _build_field(field_h, field_w, placed)
    grid, _ = _build_grid(field, color_map, rng)

    if not _validate_generated(grid, placed, color_map):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid inputs for task 7")
