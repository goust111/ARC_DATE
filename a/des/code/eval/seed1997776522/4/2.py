import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
COLOR_SEQUENCE = [2, 1, 3, 4, 6]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_MIN = 20
GRID_MAX = 28
LOGICAL_H_CHOICES = (3, 4, 5)
LOGICAL_W_CHOICES = (4, 5, 6)
REGION_COUNT_CHOICES = (3, 4, 5)
GEN_RETRIES = 500


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _orientations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen = set()
    for base in (mask, np.fliplr(mask)):
        cur = base
        for _ in range(4):
            cropped = _crop_mask(cur)
            key = (cropped.shape[0], cropped.shape[1], cropped.astype(np.uint8).tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(cropped.copy())
            cur = np.rot90(cur)
    return variants


def _canonical_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(
        (ori.shape[0], ori.shape[1], ori.astype(np.uint8).tobytes())
        for ori in _orientations(mask)
    )


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _extract_regions(template_crop: np.ndarray, bg: int) -> list[dict]:
    h, w = template_crop.shape
    regions = []
    for comp in _components(template_crop == bg):
        if _touches_border(comp, h, w):
            continue
        mask = _mask_from_coords(comp)
        regions.append(
            {
                "coords": comp,
                "mask": mask,
                "key": _canonical_key(mask),
            }
        )
    return regions


def _extract_pieces(grid: np.ndarray, bg: int) -> list[dict]:
    pieces = []
    for color in sorted(int(v) for v in np.unique(grid)):
        if color in (bg, GREY):
            continue
        for comp in _components(grid == color):
            pieces.append(
                {
                    "color": color,
                    "coords": comp,
                    "mask": _mask_from_coords(comp),
                    "key": _canonical_key(_mask_from_coords(comp)),
                    "bbox": _bbox(comp),
                }
            )
    return pieces


def _piece_graph(pieces: list[dict], h: int, w: int) -> list[set[int]]:
    owner = np.full((h, w), -1, dtype=int)
    for idx, piece in enumerate(pieces):
        owner[piece["coords"][:, 0], piece["coords"][:, 1]] = idx

    adj = [set() for _ in pieces]
    for idx, piece in enumerate(pieces):
        for r, c in piece["coords"]:
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    other = int(owner[nr, nc])
                    if other >= 0 and other != idx:
                        adj[idx].add(other)
    return adj


def _chain_order(pieces: list[dict], h: int, w: int) -> list[int]:
    if len(pieces) == 1:
        return [0]

    adj = _piece_graph(pieces, h, w)
    degrees = [len(nei) for nei in adj]
    endpoints = [i for i, deg in enumerate(degrees) if deg == 1]
    assert len(endpoints) == 2, "piece contacts must form a simple chain"

    border_endpoints = [i for i in endpoints if _touches_border(pieces[i]["coords"], h, w)]
    if len(border_endpoints) == 1:
        start = border_endpoints[0]
    else:
        start = min(endpoints, key=lambda i: pieces[i]["bbox"])

    order = []
    prev = -1
    cur = start
    while True:
        order.append(cur)
        nxt = [idx for idx in adj[cur] if idx != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "piece contacts must not branch"
        prev, cur = cur, nxt[0]
        assert len(order) <= len(pieces), "piece traversal looped unexpectedly"

    assert len(order) == len(pieces), "piece chain must be connected"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match outside pieces to template cavities by shape, then recolor cavities by the
    order of the outside contact chain.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)

    grey_components = _components(g == GREY)
    assert grey_components, "missing grey template"
    template_component = max(grey_components, key=len)
    r0, c0, r1, c1 = _bbox(template_component)
    template_crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()

    regions = _extract_regions(template_crop, bg)
    pieces = _extract_pieces(g, bg)
    assert regions and len(regions) == len(pieces), "piece/region count mismatch"
    assert len(regions) <= len(COLOR_SEQUENCE), "color sequence too short"

    region_by_key: dict[tuple[int, int, bytes], int] = {}
    for idx, region in enumerate(regions):
        assert region["key"] not in region_by_key, "region shapes must be unique"
        region_by_key[region["key"]] = idx

    piece_to_region: dict[int, int] = {}
    used_regions = set()
    for idx, piece in enumerate(pieces):
        assert piece["key"] in region_by_key, "piece shape has no matching region"
        region_idx = region_by_key[piece["key"]]
        assert region_idx not in used_regions, "piece matching must be one-to-one"
        used_regions.add(region_idx)
        piece_to_region[idx] = region_idx

    out = template_crop.copy()
    for step, piece_idx in enumerate(_chain_order(pieces, *g.shape)):
        region = regions[piece_to_region[piece_idx]]
        out[region["coords"][:, 0], region["coords"][:, 1]] = COLOR_SEQUENCE[step]
    return out


def _logical_has_hole(mask: np.ndarray) -> bool:
    inv = ~mask
    h, w = mask.shape
    for comp in _components(inv):
        if not _touches_border(comp, h, w):
            return True
    return False


def _random_outer_mask(region_count: int) -> np.ndarray | None:
    for _ in range(120):
        h = random.choice(LOGICAL_H_CHOICES)
        w = random.choice(LOGICAL_W_CHOICES)
        mask = np.ones((h, w), dtype=bool)

        max_remove = min(4, int(mask.sum()) - 2 * region_count)
        target_remove = random.randint(0, max_remove) if max_remove > 0 else 0
        border_cells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if r in (0, h - 1) or c in (0, w - 1)
        ]
        random.shuffle(border_cells)

        removed = 0
        for r, c in border_cells:
            if removed >= target_remove:
                break
            trial = mask.copy()
            trial[r, c] = False
            if int(trial.sum()) < 2 * region_count:
                continue
            if not trial[0].any() or not trial[-1].any() or not trial[:, 0].any() or not trial[:, -1].any():
                continue
            if len(_components(trial)) != 1:
                continue
            if _logical_has_hole(trial):
                continue
            mask = trial
            removed += 1

        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randint(0, 3))
        if int(mask.sum()) >= 2 * region_count and not _logical_has_hole(mask):
            return mask
    return None


def _frontier(labels: np.ndarray, mask: np.ndarray, region_cells: list[list[tuple[int, int]]], region_id: int, unassigned: set[tuple[int, int]]) -> list[tuple[int, int]]:
    frontier = set()
    for r, c in region_cells[region_id]:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) in unassigned:
                frontier.add((nr, nc))
    return list(frontier)


def _partition_mask(mask: np.ndarray, region_count: int) -> np.ndarray | None:
    cells = [tuple(pt) for pt in np.argwhere(mask)]
    if len(cells) < 2 * region_count:
        return None

    for _ in range(120):
        labels = np.full(mask.shape, -1, dtype=int)
        random.shuffle(cells)

        seeds: list[tuple[int, int]] = []
        for cell in cells:
            if all(abs(cell[0] - sr) + abs(cell[1] - sc) >= 2 for sr, sc in seeds):
                seeds.append(cell)
            if len(seeds) == region_count:
                break
        if len(seeds) < region_count:
            continue

        region_cells: list[list[tuple[int, int]]] = [[] for _ in range(region_count)]
        unassigned = set(cells)
        for idx, seed in enumerate(seeds):
            labels[seed] = idx
            region_cells[idx].append(seed)
            unassigned.remove(seed)

        ok = True
        while unassigned:
            choices = []
            weights = []
            for region_id in range(region_count):
                fr = _frontier(labels, mask, region_cells, region_id, unassigned)
                if not fr:
                    continue
                choices.append((region_id, fr))
                weights.append((len(fr) + 1.0) / (len(region_cells[region_id]) ** 1.2))
            if not choices:
                ok = False
                break
            pick = random.choices(range(len(choices)), weights=weights, k=1)[0]
            region_id, fr = choices[pick]
            cell = random.choice(fr)
            labels[cell] = region_id
            region_cells[region_id].append(cell)
            unassigned.remove(cell)

        if not ok:
            continue

        sizes = [len(region) for region in region_cells]
        if min(sizes) < 2 or max(sizes) > 8:
            continue
        return labels
    return None


def _render_template(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    crop = np.full((2 * h + 1, 2 * w + 1), BG, dtype=int)

    for r, c in np.argwhere(mask):
        rr, cc = 2 * int(r), 2 * int(c)
        crop[rr : rr + 3, cc : cc + 3] = GREY
        crop[rr + 1, cc + 1] = BG

    for r, c in np.argwhere(mask):
        region_id = int(labels[r, c])
        if c + 1 < w and mask[r, c + 1] and int(labels[r, c + 1]) == region_id:
            crop[2 * int(r) + 1, 2 * int(c) + 2] = BG
        if r + 1 < h and mask[r + 1, c] and int(labels[r + 1, c]) == region_id:
            crop[2 * int(r) + 2, 2 * int(c) + 1] = BG

    return crop


def _free_positions_on_side(mask: np.ndarray, h: int, w: int, side: str) -> list[tuple[int, int]]:
    mh, mw = mask.shape
    positions = []
    if side in ("top", "bottom"):
        c_values = list(range(max(0, 1), max(1, w - mw)))
        if not c_values:
            c_values = list(range(0, w - mw + 1))
        row = 0 if side == "top" else h - mh
        positions = [(row, c0) for c0 in c_values if 0 <= row <= h - mh]
    else:
        r_values = list(range(max(0, 1), max(1, h - mh)))
        if not r_values:
            r_values = list(range(0, h - mh + 1))
        col = 0 if side == "left" else w - mw
        positions = [(r0, col) for r0 in r_values if 0 <= col <= w - mw]
    random.shuffle(positions)
    return positions


def _adjacent_translations(prev_coords: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    base = np.argwhere(mask)
    positions = set()
    for ar, ac in prev_coords:
        for dr, dc in DIR4:
            tr, tc = int(ar + dr), int(ac + dc)
            for br, bc in base:
                positions.add((tr - int(br), tc - int(bc)))
    out = list(positions)
    random.shuffle(out)
    return out


def _can_place(coords: np.ndarray, owner: np.ndarray, blocked: np.ndarray, prev_idx: int | None, allow_border: bool) -> bool:
    h, w = owner.shape
    if np.any(coords[:, 0] < 0) or np.any(coords[:, 0] >= h) or np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= w):
        return False
    if np.any(blocked[coords[:, 0], coords[:, 1]]):
        return False
    if np.any(owner[coords[:, 0], coords[:, 1]] >= 0):
        return False
    if not allow_border and _touches_border(coords, h, w):
        return False

    touches_prev = prev_idx is None
    for r, c in coords:
        for dr, dc in DIR4:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                other = int(owner[nr, nc])
                if other == -1:
                    continue
                if prev_idx is not None and other == prev_idx:
                    touches_prev = True
                else:
                    return False
    return touches_prev


def _choose_template_position(h: int, w: int, th: int, tw: int, side: str) -> tuple[int, int] | None:
    if h - th < 4 or w - tw < 4:
        return None

    if side == "top":
        r_lo, r_hi = max(2, h // 2 - th // 2), h - th - 2
        c_lo, c_hi = 2, w - tw - 2
    elif side == "bottom":
        r_lo, r_hi = 2, min(h - th - 2, h // 2 - th // 2)
        c_lo, c_hi = 2, w - tw - 2
    elif side == "left":
        r_lo, r_hi = 2, h - th - 2
        c_lo, c_hi = max(2, w // 2 - tw // 2), w - tw - 2
    else:
        r_lo, r_hi = 2, h - th - 2
        c_lo, c_hi = 2, min(w - tw - 2, w // 2 - tw // 2)

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _place_chain(piece_masks: list[np.ndarray], side: str, blocked: np.ndarray) -> list[dict] | None:
    h, w = blocked.shape
    owner = np.full((h, w), -1, dtype=int)
    placements: list[dict] = []

    for idx, piece_mask in enumerate(piece_masks):
        placed = False
        variants = _orientations(piece_mask)
        random.shuffle(variants)
        for variant in variants:
            base = np.argwhere(variant)
            if idx == 0:
                translations = _free_positions_on_side(variant, h, w, side)
            else:
                translations = _adjacent_translations(placements[-1]["coords"], variant)
            for r0, c0 in translations:
                coords = base + np.array([r0, c0], dtype=int)
                if not _can_place(coords, owner, blocked, idx - 1 if idx > 0 else None, idx == 0):
                    continue
                owner[coords[:, 0], coords[:, 1]] = idx
                placements.append({"coords": coords.copy(), "mask": variant.copy()})
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return placements


def _expected_output(template_crop: np.ndarray, regions: list[dict], region_order: list[int]) -> np.ndarray:
    out = template_crop.copy()
    for step, region_idx in enumerate(region_order):
        out[regions[region_idx]["coords"][:, 0], regions[region_idx]["coords"][:, 1]] = COLOR_SEQUENCE[step]
    return out


def _gen_attempt() -> np.ndarray | None:
    region_count = random.choice(REGION_COUNT_CHOICES)
    outer_mask = _random_outer_mask(region_count)
    if outer_mask is None:
        return None

    labels = _partition_mask(outer_mask, region_count)
    if labels is None:
        return None

    template_crop = _render_template(labels, outer_mask)
    regions = _extract_regions(template_crop, BG)
    if len(regions) != region_count:
        return None
    if len({region["key"] for region in regions}) != region_count:
        return None

    region_order = list(range(region_count))
    random.shuffle(region_order)
    piece_masks = [regions[idx]["mask"] for idx in region_order]
    side = random.choice(("top", "bottom", "left", "right"))

    th, tw = template_crop.shape
    for _ in range(100):
        h = random.randint(max(GRID_MIN, th + 6), GRID_MAX)
        w = random.randint(max(GRID_MIN, tw + 6), GRID_MAX)
        pos = _choose_template_position(h, w, th, tw, side)
        if pos is None:
            continue
        tr, tc = pos

        blocked = np.zeros((h, w), dtype=bool)
        blocked[max(0, tr - 1) : min(h, tr + th + 1), max(0, tc - 1) : min(w, tc + tw + 1)] = True

        placements = _place_chain(piece_masks, side, blocked)
        if placements is None:
            continue

        grid = np.full((h, w), BG, dtype=int)
        grid[tr : tr + th, tc : tc + tw] = template_crop

        piece_colors = random.sample([c for c in range(1, 10) if c != GREY], region_count)
        for idx, placement in enumerate(placements):
            coords = placement["coords"]
            grid[coords[:, 0], coords[:, 1]] = piece_colors[idx]

        expected = _expected_output(template_crop, regions, region_order)
        if np.array_equal(main(grid), expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
