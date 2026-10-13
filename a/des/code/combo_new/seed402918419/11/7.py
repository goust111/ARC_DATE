from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
STAMP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _component_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    arr = np.asarray(coords, dtype=int)
    r0, c0, r1, c1 = _bbox(arr)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out, (r0, c0, r1, c1)


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    pts = np.argwhere(mask)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _find_matches(pattern_mask: np.ndarray, stamp_mask: np.ndarray) -> list[tuple[int, int]]:
    ph, pw = pattern_mask.shape
    sh, sw = stamp_mask.shape
    matches: list[tuple[int, int]] = []
    if sh > ph or sw > pw:
        return matches
    for r in range(ph - sh + 1):
        for c in range(pw - sw + 1):
            if np.array_equal(pattern_mask[r : r + sh, c : c + sw], stamp_mask):
                matches.append((r, c))
    return matches


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...], bytes]] = set()
    for base in (mask, np.fliplr(mask)):
        for k in range(4):
            cur = _normalize_mask(np.rot90(base, k))
            key = (cur.shape, cur.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(cur)
    return variants


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the matching substructures inside the central grey pattern using scattered monochrome stamps.

    Input structure:
    - Background is Black.
    - One large central pattern is drawn in Grey and contains several local substructures.
    - Around that pattern sit smaller single-color 4-connected stamp objects in distinct colors.
    - Each stamp has exactly the same shape as one or more Grey subregions inside the large pattern, without rotation or reflection.

    Rule:
    - Crop the overall Grey pattern to its tight bounding box and treat it as the output canvas.
    - For each colored stamp, slide its shape over the Grey pattern and collect every exact local match where Grey pixels inside the stamp bbox equal the stamp mask.
    - If a stamp shape matches in several places, choose the match whose position relative to the Grey pattern center is closest to the stamp's own relative position.
    - Recolor only the matched Grey pixels with the stamp color; all other Grey pixels and internal background gaps stay unchanged.

    Key insight:
    - The small objects are both templates and paint sources: shape matching finds the target subregion, and the stamp's offset around the big figure resolves repeated matches.
    """
    g = np.asarray(grid, dtype=int)
    grey_pts = np.argwhere(g == GREY)
    assert grey_pts.size > 0, "expected a central grey pattern"
    gr0, gc0, gr1, gc1 = _bbox(grey_pts)
    pattern = g[gr0 : gr1 + 1, gc0 : gc1 + 1].copy()
    pattern_mask = pattern == GREY
    pattern_center = ((gr0 + gr1) / 2.0, (gc0 + gc1) / 2.0)

    stamps: list[dict[str, object]] = []
    for color in np.unique(g):
        color = int(color)
        if color in (BG, GREY):
            continue
        for comp in _components(g == color):
            mask, (r0, c0, _r1, _c1) = _component_mask(comp)
            ctr_r, ctr_c = _centroid(mask)
            stamps.append(
                {
                    "color": color,
                    "mask": mask,
                    "stamp_center": (r0 + ctr_r, c0 + ctr_c),
                    "shape_center": (ctr_r, ctr_c),
                    "bbox": (r0, c0),
                }
            )

    stamps.sort(key=lambda item: tuple(item["bbox"]))  # deterministic only

    for stamp in stamps:
        mask = stamp["mask"]
        sh, sw = mask.shape
        matches = _find_matches(pattern_mask, mask)
        assert matches, "every stamp must match at least one grey subregion"
        stamp_center = stamp["stamp_center"]
        shape_center = stamp["shape_center"]
        stamp_vec = (
            float(stamp_center[0] - pattern_center[0]),
            float(stamp_center[1] - pattern_center[1]),
        )

        def match_key(pos: tuple[int, int]) -> tuple[float, float, int, int]:
            mr, mc = pos
            cand_center = (gr0 + mr + shape_center[0], gc0 + mc + shape_center[1])
            cand_vec = (
                float(cand_center[0] - pattern_center[0]),
                float(cand_center[1] - pattern_center[1]),
            )
            dr = cand_vec[0] - stamp_vec[0]
            dc = cand_vec[1] - stamp_vec[1]
            return (dr * dr + dc * dc, abs(dr) + abs(dc), mr, mc)

        mr, mc = min(matches, key=match_key)
        pattern[mr : mr + sh, mc : mc + sw][mask] = int(stamp["color"])

    return pattern


_BASE_MASKS = [
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=bool),
]


def _build_mask_pool() -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for base in _BASE_MASKS:
        for var in _dihedral_variants(base):
            key = (var.shape, var.tobytes())
            if key in seen:
                continue
            seen.add(key)
            pool.append(var)
    return pool


MASK_POOL = _build_mask_pool()


def _choose_masks(rng: np.random.Generator, count: int) -> list[np.ndarray]:
    allow_dupes = count >= 3 and rng.random() < 0.5
    extra_dupes = 0
    if allow_dupes:
        extra_dupes = int(rng.integers(1, min(3, count)))
    base_count = count - extra_dupes
    base_ids = rng.choice(len(MASK_POOL), size=base_count, replace=False)
    chosen = [MASK_POOL[int(i)] for i in base_ids]
    ids = list(range(base_count))
    for _ in range(extra_dupes):
        ids.append(int(rng.integers(0, base_count)))
    rng.shuffle(ids)
    return [chosen[i] for i in ids]


def _slot_defs(hub_h: int, hub_w: int) -> list[dict[str, int | str]]:
    return [
        {"side": "top", "coord": 1},
        {"side": "top", "coord": hub_w - 2},
        {"side": "bottom", "coord": 1},
        {"side": "bottom", "coord": hub_w - 2},
        {"side": "left", "coord": 1},
        {"side": "left", "coord": hub_h - 2},
        {"side": "right", "coord": 1},
        {"side": "right", "coord": hub_h - 2},
    ]


def _anchor(mask: np.ndarray, side: str) -> tuple[int, int]:
    pts = np.argwhere(mask)
    if side == "top":
        edge = pts[pts[:, 0] == mask.shape[0] - 1]
        idx = len(edge) // 2
        return int(edge[idx, 0]), int(edge[idx, 1])
    if side == "bottom":
        edge = pts[pts[:, 0] == 0]
        idx = len(edge) // 2
        return int(edge[idx, 0]), int(edge[idx, 1])
    if side == "left":
        edge = pts[pts[:, 1] == mask.shape[1] - 1]
        idx = len(edge) // 2
        return int(edge[idx, 0]), int(edge[idx, 1])
    edge = pts[pts[:, 1] == 0]
    idx = len(edge) // 2
    return int(edge[idx, 0]), int(edge[idx, 1])


def _place_lobe(mask: np.ndarray, slot: dict[str, int | str], hub_h: int, hub_w: int) -> dict[str, object]:
    side = str(slot["side"])
    coord = int(slot["coord"])
    h, w = mask.shape
    ar, ac = _anchor(mask, side)
    if side == "top":
        top, left = -h - 1, coord - ac
        bridge = {(-1, coord)}
    elif side == "bottom":
        top, left = hub_h + 1, coord - ac
        bridge = {(hub_h, coord)}
    elif side == "left":
        top, left = coord - ar, -w - 1
        bridge = {(coord, -1)}
    else:
        top, left = coord - ar, hub_w + 1
        bridge = {(coord, hub_w)}

    coords = {(top + r, left + c) for r, c in np.argwhere(mask)}
    ctr_r, ctr_c = _centroid(mask)
    return {
        "side": side,
        "mask": mask,
        "top_left": (top, left),
        "coords": coords,
        "bridge": bridge,
        "center": (top + ctr_r, left + ctr_c),
    }


def _coords_to_grid(coords: set[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    arr = np.asarray(sorted(coords), dtype=int)
    r0, c0, r1, c1 = _bbox(arr)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out, (r0, c0, r1, c1)


def _can_place_isolated(occ: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        if occ[rr, cc]:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < occ.shape[0] and 0 <= nc < occ.shape[1] and occ[nr, nc]:
                    return False
    return True


def _stamp_positions(
    occ: np.ndarray,
    mask: np.ndarray,
    side: str,
    target_center: tuple[float, float],
    pattern_bbox: tuple[int, int, int, int],
) -> list[tuple[float, int, int]]:
    h, w = mask.shape
    H, W = occ.shape
    pr0, pc0, pr1, pc1 = pattern_bbox
    ctr_r, ctr_c = _centroid(mask)
    rows: range
    cols: range
    if side == "top":
        max_r = pr0 - h - 1
        if max_r < 0:
            return []
        rows = range(0, max_r + 1)
        cols = range(0, W - w + 1)
    elif side == "bottom":
        min_r = pr1 + 2
        if min_r + h > H:
            return []
        rows = range(min_r, H - h + 1)
        cols = range(0, W - w + 1)
    elif side == "left":
        max_c = pc0 - w - 1
        if max_c < 0:
            return []
        rows = range(0, H - h + 1)
        cols = range(0, max_c + 1)
    else:
        min_c = pc1 + 2
        if min_c + w > W:
            return []
        rows = range(0, H - h + 1)
        cols = range(min_c, W - w + 1)

    out: list[tuple[float, int, int]] = []
    for top in rows:
        for left in cols:
            if not _can_place_isolated(occ, mask, top, left):
                continue
            sc_r = top + ctr_r
            sc_c = left + ctr_c
            if side in ("top", "bottom"):
                cross = abs(sc_c - target_center[1])
                gap = pr0 - (top + h - 1) if side == "top" else top - pr1
            else:
                cross = abs(sc_r - target_center[0])
                gap = pc0 - (left + w - 1) if side == "left" else left - pc1
            full = (sc_r - target_center[0]) ** 2 + (sc_c - target_center[1]) ** 2
            out.append((cross * 1000.0 + gap * 10.0 + full, top, left))
    return out


def _paint(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = color


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    hub_h = int(rng.integers(5, 8))
    hub_w = int(rng.integers(6, 9))
    slot_defs = _slot_defs(hub_h, hub_w)
    stamp_count = int(rng.integers(3, 7))
    slot_ids = rng.choice(len(slot_defs), size=stamp_count, replace=False)
    masks = _choose_masks(rng, stamp_count)
    colors = [int(c) for c in rng.choice(STAMP_COLORS, size=stamp_count, replace=False)]

    hub_cells = {(r, c) for r in range(hub_h) for c in range(hub_w)}
    pattern_cells = set(hub_cells)
    placements: list[dict[str, object]] = []
    for idx, slot_id in enumerate(slot_ids):
        placement = _place_lobe(masks[idx], slot_defs[int(slot_id)], hub_h, hub_w)
        new_cells = set(placement["coords"]) | set(placement["bridge"])
        if pattern_cells & new_cells:
            return None
        pattern_cells |= new_cells
        placement["color"] = colors[idx]
        placements.append(placement)

    pattern_mask, local_bbox = _coords_to_grid(pattern_cells)
    lr0, lc0, lr1, lc1 = local_bbox
    pattern_h, pattern_w = pattern_mask.shape
    if pattern_h > 18 or pattern_w > 18:
        return None

    expected = np.full((pattern_h, pattern_w), BG, dtype=int)
    expected[pattern_mask] = GREY
    for placement in placements:
        top = int(placement["top_left"][0] - lr0)
        left = int(placement["top_left"][1] - lc0)
        placement["crop_top_left"] = (top, left)
        placement["crop_center"] = (
            float(placement["center"][0] - lr0),
            float(placement["center"][1] - lc0),
        )
        _paint(expected, placement["mask"], top, left, int(placement["color"]))

    top_margin = max((mask.shape[0] for mask, p in ((p["mask"], p) for p in placements) if p["side"] == "top"), default=0)
    bot_margin = max((mask.shape[0] for mask, p in ((p["mask"], p) for p in placements) if p["side"] == "bottom"), default=0)
    left_margin = max((mask.shape[1] for mask, p in ((p["mask"], p) for p in placements) if p["side"] == "left"), default=0)
    right_margin = max((mask.shape[1] for mask, p in ((p["mask"], p) for p in placements) if p["side"] == "right"), default=0)
    top_margin = top_margin + 2 if top_margin else 1
    bot_margin = bot_margin + 2 if bot_margin else 1
    left_margin = left_margin + 2 if left_margin else 1
    right_margin = right_margin + 2 if right_margin else 1

    min_h = pattern_h + top_margin + bot_margin
    min_w = pattern_w + left_margin + right_margin
    if min_h > 30 or min_w > 30:
        return None
    H = int(rng.integers(min_h, 31))
    W = int(rng.integers(min_w, 31))

    r_lo = top_margin
    r_hi = H - bot_margin - pattern_h
    c_lo = left_margin
    c_hi = W - right_margin - pattern_w
    if r_lo > r_hi or c_lo > c_hi:
        return None
    r_mid = max(r_lo, min(r_hi, (H - pattern_h) // 2 + int(rng.integers(-1, 2))))
    c_mid = max(c_lo, min(c_hi, (W - pattern_w) // 2 + int(rng.integers(-1, 2))))

    grid = np.zeros((H, W), dtype=int)
    grid[r_mid : r_mid + pattern_h, c_mid : c_mid + pattern_w][pattern_mask] = GREY
    pattern_bbox = (r_mid, c_mid, r_mid + pattern_h - 1, c_mid + pattern_w - 1)

    occ = grid != BG
    side_axis = {
        "top": 1,
        "bottom": 1,
        "left": 0,
        "right": 0,
    }
    for side in ("top", "bottom", "left", "right"):
        side_items = [p for p in placements if p["side"] == side]
        side_items.sort(key=lambda item: item["crop_center"][side_axis[side]])
        for placement in side_items:
            target_center = (
                float(r_mid + placement["crop_center"][0]),
                float(c_mid + placement["crop_center"][1]),
            )
            options = _stamp_positions(occ, placement["mask"], side, target_center, pattern_bbox)
            if not options:
                return None
            _, top, left = min(options)
            _paint(grid, placement["mask"], top, left, int(placement["color"]))
            occ = grid != BG

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_RETRIES} attempts")
