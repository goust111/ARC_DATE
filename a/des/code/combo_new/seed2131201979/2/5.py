import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 0
GEN_RETRIES = 300
PATCH_RETRIES = 80
PLACE_RETRIES = 120


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield int(r + dr), int(c + dc)


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comp_ids = np.full((h, w), -1, dtype=int)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not bool(seen[nr, nc])
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            cid = len(comps)
            arr = np.array(pts, dtype=int)
            comp_ids[arr[:, 0], arr[:, 1]] = int(cid)
            comps.append(
                {
                    "id": int(cid),
                    "color": int(color),
                    "coords": arr,
                    "coord_list": pts,
                    "area": int(len(pts)),
                }
            )
    return comps, comp_ids


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        (coords[:, 0] == 0).any()
        or (coords[:, 0] == h - 1).any()
        or (coords[:, 1] == 0).any()
        or (coords[:, 1] == w - 1).any()
    )


def _find_nested_pairs(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict], dict[int, dict]]:
    h, w = grid.shape
    comps, comp_ids = _monochrome_components(grid, bg)
    candidates: list[dict] = []

    for outer in comps:
        free = np.ones((h, w), dtype=bool)
        free[outer["coords"][:, 0], outer["coords"][:, 1]] = False
        for region in _mask_components(free):
            if _touches_border(region, h, w):
                continue
            inner_ids: set[int] = set()
            gap: set[tuple[int, int]] = set()
            for r, c in region:
                rr, cc = int(r), int(c)
                if int(grid[rr, cc]) == bg:
                    gap.add((int(rr), int(cc)))
                else:
                    cid = int(comp_ids[rr, cc])
                    if cid != int(outer["id"]):
                        inner_ids.add(int(cid))
            if len(inner_ids) != 1 or not gap:
                continue

            keep: set[tuple[int, int]] = set()
            for rr, cc in outer["coord_list"]:
                for nr, nc in _neighbors4(rr, cc):
                    if (int(nr), int(nc)) in gap:
                        keep.add((int(rr), int(cc)))
                        break
            if not keep:
                continue

            inner_id = next(iter(inner_ids))
            candidates.append(
                {
                    "outer_id": int(outer["id"]),
                    "inner_id": int(inner_id),
                    "region_area": int(len(region)),
                    "gap": gap,
                    "keep": keep,
                }
            )

    best_by_inner: dict[int, dict] = {}
    for cand in candidates:
        inner_id = int(cand["inner_id"])
        prev = best_by_inner.get(inner_id)
        if prev is None or int(cand["region_area"]) < int(prev["region_area"]):
            best_by_inner[inner_id] = cand

    merged: dict[int, dict] = {}
    for cand in best_by_inner.values():
        outer_id = int(cand["outer_id"])
        if outer_id not in merged:
            merged[outer_id] = {
                "outer_id": int(outer_id),
                "inner_ids": [],
                "keep": set(),
            }
        merged[outer_id]["inner_ids"].append(int(cand["inner_id"]))
        merged[outer_id]["keep"].update(cand["keep"])

    return comps, list(merged.values()), best_by_inner


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reduce each enclosing region to the ring of pixels facing its enclosed partner.

    Input structure:
    - The background is black.
    - The grid contains several non-overlapping nested pairs of colored 4-connected monochrome regions.
    - In each pair, one region encloses the other, with a black gap band between them.
    - Outer and inner regions use different colors, and different pairs use disjoint colors.

    Rule:
    - Identify which region in each pair is the outer enclosure by checking spatial containment, not area.
    - Keep every inner region unchanged.
    - For each outer region, keep only the pixels that are 4-adjacent to the black gap band separating it from its inner region.
    - Remove the other outer pixels to black; the gap band stays black.

    Key insight:
    - The critical step is to detect enclosure through connected empty-space regions, then preserve only the outer region's inward-facing contour rather than its whole boundary or a generic erosion.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    comps, pairs, _ = _find_nested_pairs(g, bg)

    out = g.copy()
    for pair in pairs:
        outer = comps[int(pair["outer_id"])]
        out[outer["coords"][:, 0], outer["coords"][:, 1]] = int(bg)
        for r, c in pair["keep"]:
            out[int(r), int(c)] = int(outer["color"])
    return out


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate_n(mask: np.ndarray, steps: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(int(steps)):
        out = _dilate4(out)
    return out


def _random_connected_mask(height: int, width: int, min_fill: float, max_fill: float) -> np.ndarray:
    total = int(height * width)
    lo = max(3, int(np.ceil(total * float(min_fill))))
    hi = max(lo, int(np.floor(total * float(max_fill))))
    area = random.randint(int(lo), int(hi))

    start = (random.randint(0, height - 1), random.randint(0, width - 1))
    cells = {start}
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((int(nr), int(nc)))

    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < height and 0 <= nc < width and (int(nr), int(nc)) not in cells:
                frontier.add((int(nr), int(nc)))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return _trim_mask(mask)


def _transform_masks(outer_mask: np.ndarray, inner_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labeled = np.zeros(outer_mask.shape, dtype=int)
    labeled[outer_mask] = 1
    labeled[inner_mask] = 2

    for _ in range(random.randint(0, 3)):
        labeled = np.rot90(labeled)
    if random.random() < 0.5:
        labeled = np.fliplr(labeled)
    if random.random() < 0.5:
        labeled = np.flipud(labeled)

    union = labeled != 0
    union = _trim_mask(union)
    rows, cols = np.where(union)
    r0, c0 = int(rows.min()), int(cols.min())
    h = int(rows.max() - r0 + 1)
    w = int(cols.max() - c0 + 1)
    labeled = labeled[r0 : r0 + h, c0 : c0 + w]
    return labeled == 1, labeled == 2


def _build_frame_pair() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PATCH_RETRIES):
        thickness = random.choice([1, 1, 2])
        cavity_h = random.randint(5, 9)
        cavity_w = random.randint(5, 9)
        if cavity_h <= 3 or cavity_w <= 3:
            continue

        inner_h = random.randint(max(2, cavity_h - 4), cavity_h - 2)
        inner_w = random.randint(max(2, cavity_w - 4), cavity_w - 2)
        inner = _random_connected_mask(inner_h, inner_w, 0.55, 0.95)

        slack_r = cavity_h - inner.shape[0]
        slack_c = cavity_w - inner.shape[1]
        if slack_r < 2 or slack_c < 2:
            continue
        inner_r0 = random.randint(1, slack_r - 1)
        inner_c0 = random.randint(1, slack_c - 1)

        outer_h = cavity_h + 2 * thickness
        outer_w = cavity_w + 2 * thickness
        outer = np.ones((outer_h, outer_w), dtype=bool)
        outer[thickness : thickness + cavity_h, thickness : thickness + cavity_w] = False

        inner_mask = np.zeros((outer_h, outer_w), dtype=bool)
        rr = thickness + inner_r0
        cc = thickness + inner_c0
        inner_mask[rr : rr + inner.shape[0], cc : cc + inner.shape[1]] = inner

        outer, inner_mask = _transform_masks(outer, inner_mask)
        if len(_mask_components(outer)) != 1 or len(_mask_components(inner_mask)) != 1:
            continue
        if int(outer.sum()) > int(inner_mask.sum()):
            continue
        return outer, inner_mask
    return None


def _build_organic_pair() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PATCH_RETRIES):
        gap = random.choice([1, 2])
        thickness = random.choice([2, 2, 3])
        seed_h = random.randint(2, 5)
        seed_w = random.randint(2, 5)
        inner_seed = _random_connected_mask(seed_h, seed_w, 0.45, 0.85)

        pad = gap + thickness + 2
        canvas = np.zeros((inner_seed.shape[0] + 2 * pad, inner_seed.shape[1] + 2 * pad), dtype=bool)
        canvas[pad : pad + inner_seed.shape[0], pad : pad + inner_seed.shape[1]] = inner_seed

        outer_limit = _dilate_n(canvas, gap + thickness)
        gap_limit = _dilate_n(canvas, gap)
        outer = outer_limit & (~gap_limit)
        inner = canvas.copy()

        outer, inner = _transform_masks(outer, inner)
        if len(_mask_components(outer)) != 1 or len(_mask_components(inner)) != 1:
            continue
        if int(outer.sum()) < int(inner.sum()):
            continue
        return outer, inner
    return None


def _paint_pair(outer_mask: np.ndarray, inner_mask: np.ndarray, outer_color: int, inner_color: int) -> np.ndarray:
    patch = np.zeros(outer_mask.shape, dtype=int)
    patch[outer_mask] = int(outer_color)
    patch[inner_mask] = int(inner_color)
    return patch


def _try_place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph > h or pw > w:
        return False

    row_lo = 1 if h - ph >= 2 else 0
    row_hi = h - ph - (1 if h - ph >= 2 else 0)
    col_lo = 1 if w - pw >= 2 else 0
    col_hi = w - pw - (1 if w - pw >= 2 else 0)

    for _ in range(PLACE_RETRIES):
        r0 = random.randint(int(row_lo), int(row_hi))
        c0 = random.randint(int(col_lo), int(col_hi))
        r1 = r0 + ph
        c1 = c0 + pw
        br0 = max(0, r0 - 1)
        bc0 = max(0, c0 - 1)
        br1 = min(h, r1 + 1)
        bc1 = min(w, c1 + 1)
        if bool(blocked[br0:br1, bc0:bc1].any()):
            continue
        grid[r0:r1, c0:c1][patch != BG] = patch[patch != BG]
        blocked[br0:br1, bc0:bc1] = True
        return True
    return False


def _group_modes(n_groups: int) -> list[str]:
    if n_groups == 1:
        return [random.choice(["frame", "organic"])]
    if n_groups == 2:
        modes = ["frame", "organic"]
        random.shuffle(modes)
        return modes
    modes = ["frame", "organic", random.choice(["frame", "organic"])]
    random.shuffle(modes)
    return modes


def _valid_generated_grid(grid: np.ndarray, expected_groups: int) -> bool:
    comps, pairs, best_by_inner = _find_nested_pairs(grid, BG)
    outer_ids = {int(pair["outer_id"]) for pair in pairs}
    inner_ids = {int(inner_id) for inner_id in best_by_inner}
    all_ids = {int(comp["id"]) for comp in comps}

    if len(pairs) != expected_groups:
        return False
    if len(comps) != 2 * expected_groups:
        return False
    if outer_ids & inner_ids:
        return False
    if outer_ids | inner_ids != all_ids:
        return False
    if any(len(pair["inner_ids"]) != 1 for pair in pairs):
        return False
    if np.array_equal(main(grid), grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(16, 30)
        w = random.randint(16, 30)
        n_groups = random.choice([1, 2, 2, 3])
        colors = random.sample(range(1, 10), 2 * n_groups)
        modes = _group_modes(n_groups)

        patches: list[np.ndarray] = []
        ok = True
        for idx, mode in enumerate(modes):
            pair = _build_frame_pair() if mode == "frame" else _build_organic_pair()
            if pair is None:
                ok = False
                break
            outer_mask, inner_mask = pair
            patch = _paint_pair(outer_mask, inner_mask, colors[2 * idx], colors[2 * idx + 1])
            patches.append(patch)
        if not ok:
            continue

        grid = np.zeros((h, w), dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        random.shuffle(patches)
        for patch in patches:
            if not _try_place_patch(grid, blocked, patch):
                ok = False
                break
        if not ok:
            continue

        if _valid_generated_grid(grid, n_groups):
            return grid

    raise RuntimeError("failed to generate a valid ARC input")
