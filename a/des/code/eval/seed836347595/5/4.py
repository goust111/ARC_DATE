from __future__ import annotations

import random

import numpy as np


BLACK = 0
REFERENCE_COLOR = 5
GEN_ATTEMPTS = 400
PAIR_PARTITION_ATTEMPTS = 120
PAIR_RECONSTRUCT_LIMIT = 3


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _component_masks(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            bbox = _bbox_from_coords(coords)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox,
                    "mask": _mask_from_coords(coords),
                }
            )
    return comps


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _hole_pixels(mask: np.ndarray) -> int:
    return int(mask.size - int(mask.sum()))


def _reference_candidates(grid: np.ndarray) -> tuple[int, dict[int, int]]:
    colors = [int(c) for c in np.unique(grid) if int(c) != BLACK]
    candidates = []
    for color in colors:
        comps = _component_masks(grid, color)
        if len(comps) != 4:
            continue
        holes = []
        ok = True
        for comp in comps:
            mask = comp["mask"]
            if not (mask[0, :].all() and mask[-1, :].all() and mask[:, 0].all() and mask[:, -1].all()):
                ok = False
                break
            holes.append(_hole_pixels(mask))
        if ok and sorted(holes) == [0, 1, 2, 3]:
            ordered = sorted(comps, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
            mapping = {_hole_pixels(comp["mask"]): idx for idx, comp in enumerate(ordered)}
            candidates.append((color, mapping, ordered))
    if not candidates:
        raise AssertionError("reference column not found")
    candidates.sort(key=lambda item: (item[2][0]["bbox"][1], item[2][0]["bbox"][0]))
    color, mapping, _ordered = candidates[0]
    return int(color), mapping


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs = []
    for h in range(1, int(area**0.5) + 1):
        if area % h == 0:
            w = area // h
            pairs.append((h, w))
            if h != w:
                pairs.append((w, h))
    return pairs


def _valid_reconstruction(mask1: np.ndarray, mask2: np.ndarray, H: int, W: int) -> list[np.ndarray]:
    h1, w1 = mask1.shape
    h2, w2 = mask2.shape
    out: dict[tuple[bytes, tuple[int, int]], np.ndarray] = {}
    for r1 in range(H - h1 + 1):
        for c1 in range(W - w1 + 1):
            occ1 = np.zeros((H, W), dtype=bool)
            occ1[r1 : r1 + h1, c1 : c1 + w1] = mask1
            for r2 in range(H - h2 + 1):
                for c2 in range(W - w2 + 1):
                    occ2 = np.zeros((H, W), dtype=bool)
                    occ2[r2 : r2 + h2, c2 : c2 + w2] = mask2
                    if bool((occ1 & occ2).any()):
                        continue
                    union = occ1 | occ2
                    if not union[0, :].all() or not union[-1, :].all() or not union[:, 0].all() or not union[:, -1].all():
                        continue
                    missing = int(union.size - int(union.sum()))
                    if missing > PAIR_RECONSTRUCT_LIMIT:
                        continue
                    key = (union.tobytes(), tuple(map(int, union.shape)))
                    out[key] = union
    return list(out.values())


def _reconstruct_pair(mask1: np.ndarray, mask2: np.ndarray, color: int) -> tuple[np.ndarray, int]:
    total = int(mask1.sum()) + int(mask2.sum())
    candidates: dict[tuple[bytes, tuple[int, int]], tuple[np.ndarray, int]] = {}
    min_h = max(mask1.shape[0], mask2.shape[0])
    min_w = max(mask1.shape[1], mask2.shape[1])
    for holes in range(PAIR_RECONSTRUCT_LIMIT + 1):
        area = total + holes
        for H, W in _factor_pairs(area):
            if H < min_h or W < min_w:
                continue
            for union in _valid_reconstruction(mask1, mask2, H, W):
                hole_pixels = int(union.size - int(union.sum()))
                if hole_pixels != holes:
                    continue
                patch = np.full((H, W), int(color), dtype=int)
                patch[~union] = BLACK
                key = (patch.tobytes(), tuple(map(int, patch.shape)))
                candidates[key] = (patch, hole_pixels)
    if not candidates:
        raise AssertionError("pair reconstruction failed")
    chosen = sorted(
        candidates.values(),
        key=lambda item: (item[1], item[0].shape[0] * item[0].shape[1], item[0].shape, item[0].tobytes()),
    )[0]
    return chosen


def _pack_output(rects: list[dict], hole_to_col: dict[int, int]) -> np.ndarray:
    grouped: dict[int, list[dict]] = {hole: [] for hole in sorted(hole_to_col)}
    for rect in rects:
        grouped[int(rect["holes"])].append(rect)
    for hole in grouped:
        grouped[hole].sort(key=lambda rect: (rect["sort_key"][0], rect["sort_key"][1], rect["patch"].shape[0], rect["patch"].shape[1]))

    col_order = [hole for hole, _ in sorted(hole_to_col.items(), key=lambda kv: kv[1])]
    col_widths = []
    col_heights = []
    for hole in col_order:
        items = grouped[hole]
        if items:
            col_widths.append(max(int(item["patch"].shape[1]) for item in items))
            col_heights.append(sum(int(item["patch"].shape[0]) for item in items) + len(items) - 1)
        else:
            col_widths.append(1)
            col_heights.append(0)
    out_h = max(col_heights) if col_heights else 0
    out_w = sum(col_widths) + max(0, len(col_widths) - 1)
    out = np.zeros((out_h, out_w), dtype=int)

    c0 = 0
    for hole, col_w in zip(col_order, col_widths):
        r0 = 0
        for item in grouped[hole]:
            patch = item["patch"]
            h, w = patch.shape
            region = out[r0 : r0 + h, c0 : c0 + w]
            region[patch != BLACK] = patch[patch != BLACK]
            r0 += h + 1
        c0 += col_w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored fragments, reconstruct their complementary rectangles,
    classify them by black-hole count, and stack each class into the legend-defined output column.
    """
    g = np.asarray(grid, dtype=int)
    ref_color, hole_to_col = _reference_candidates(g)

    rects = []
    for color in [int(c) for c in np.unique(g) if int(c) not in (BLACK, ref_color)]:
        comps = _component_masks(g, color)
        if len(comps) != 2:
            raise AssertionError("each fragment color must appear in exactly two components")
        patch, holes = _reconstruct_pair(comps[0]["mask"], comps[1]["mask"], color)
        top_row = min(int(comps[0]["bbox"][0]), int(comps[1]["bbox"][0]))
        left_col = min(int(comps[0]["bbox"][1]), int(comps[1]["bbox"][1]))
        rects.append({"patch": patch, "holes": holes, "sort_key": (top_row, left_col)})

    return _pack_output(rects, hole_to_col)


def _reference_patch(holes: int, color: int) -> np.ndarray:
    width = 2 * holes + 3
    patch = np.full((3, width), int(color), dtype=int)
    for idx in range(holes):
        patch[1, 1 + 2 * idx] = BLACK
    return patch


def _random_hole_positions(H: int, W: int, holes: int) -> list[tuple[int, int]] | None:
    if holes == 0:
        return []
    choices = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1)]
    random.shuffle(choices)
    picked: list[tuple[int, int]] = []
    for cell in choices:
        if all(abs(cell[0] - pr) + abs(cell[1] - pc) > 1 for pr, pc in picked):
            picked.append(cell)
            if len(picked) == holes:
                return picked
    return None


def _grow_partition(shape: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    cells = [tuple(map(int, cell)) for cell in np.argwhere(shape)]
    total = len(cells)
    if total < 8:
        return None

    seed_a, seed_b = random.sample(cells, 2)
    label: dict[tuple[int, int], int] = {seed_a: 1, seed_b: 2}
    target_a = random.randint(max(3, total // 3), min(total - 3, (2 * total) // 3))
    size_a = 1
    size_b = 1
    frontier_a = {cell for nr, nc in _neighbors4(*seed_a) if (cell := (nr, nc)) in cells and cell not in label}
    frontier_b = {cell for nr, nc in _neighbors4(*seed_b) if (cell := (nr, nc)) in cells and cell not in label}

    while len(label) < total:
        choices = []
        if frontier_a:
            choices.append(1)
        if frontier_b:
            choices.append(2)
        if not choices:
            return None
        preferred = []
        if size_a < target_a and frontier_a:
            preferred.append(1)
        if size_b < total - target_a and frontier_b:
            preferred.append(2)
        region = random.choice(preferred or choices)
        frontier = frontier_a if region == 1 else frontier_b
        pick = random.choice(tuple(frontier))
        frontier.discard(pick)
        if pick in label:
            continue
        label[pick] = region
        if region == 1:
            size_a += 1
        else:
            size_b += 1
        for nr, nc in _neighbors4(*pick):
            nxt = (nr, nc)
            if nxt in cells and nxt not in label:
                frontier_a.add(nxt) if region == 1 else frontier_b.add(nxt)

    a = np.zeros_like(shape, dtype=bool)
    b = np.zeros_like(shape, dtype=bool)
    for (r, c), region in label.items():
        if region == 1:
            a[r, c] = True
        else:
            b[r, c] = True
    if not _connected(a) or not _connected(b):
        return None
    if a.sum() < 3 or b.sum() < 3:
        return None
    return a, b


def _make_pair_patch(color: int, holes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    for _ in range(PAIR_PARTITION_ATTEMPTS):
        H = random.randint(4, 7)
        W = random.randint(4, 8)
        if (H - 2) * (W - 2) < max(holes, 1):
            continue
        shape = np.ones((H, W), dtype=bool)
        hole_positions = _random_hole_positions(H, W, holes)
        if hole_positions is None:
            continue
        for r, c in hole_positions:
            shape[r, c] = False
        parts = _grow_partition(shape)
        if parts is None:
            continue
        mask_a, mask_b = parts
        recon, recon_holes = _reconstruct_pair(_mask_from_coords([(r, c) for r, c in np.argwhere(mask_a)]), _mask_from_coords([(r, c) for r, c in np.argwhere(mask_b)]), color)
        if recon_holes != holes or recon.shape != (H, W) or not np.array_equal(recon != BLACK, shape):
            continue
        patch_a = np.full((H, W), BLACK, dtype=int)
        patch_b = np.full((H, W), BLACK, dtype=int)
        patch_a[mask_a] = int(color)
        patch_b[mask_b] = int(color)
        return recon, patch_a, patch_b
    return None


def _crop_nonzero(patch: np.ndarray) -> np.ndarray:
    pts = np.argwhere(patch != BLACK)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _can_place(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 1) -> bool:
    H, W = occupied.shape
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    cells = np.argwhere(patch != BLACK)
    for dr, dc in cells:
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - padding)
        r2 = min(H, rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(W, cc + padding + 1)
        if occupied[r1:r2, c1:c2].any():
            return False
    return True


def _paste_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    nz = patch != BLACK
    h, w = patch.shape
    region = grid[r0 : r0 + h, c0 : c0 + w]
    region[nz] = patch[nz]
    occ_region = occupied[r0 : r0 + h, c0 : c0 + w]
    occ_region[nz] = True


def _build_expected(objects: list[dict]) -> np.ndarray:
    hole_to_col = {hole: hole for hole in range(4)}
    rects = [
        {
            "patch": obj["recon"],
            "holes": obj["holes"],
            "sort_key": obj["sort_key"],
        }
        for obj in objects
    ]
    return _pack_output(rects, hole_to_col)


def _gen_attempt() -> np.ndarray | None:
    counts = {holes: 1 + random.randint(0, 1) for holes in range(4)}
    total_pairs = sum(counts.values())
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], total_pairs)

    objects = []
    for holes in range(4):
        for _ in range(counts[holes]):
            color = colors.pop()
            made = _make_pair_patch(color, holes)
            if made is None:
                return None
            recon, part_a, part_b = made
            objects.append(
                {
                    "color": color,
                    "holes": holes,
                    "recon": recon,
                    "parts": [_crop_nonzero(part_a), _crop_nonzero(part_b)],
                }
            )

    ref_patches = [_reference_patch(holes, REFERENCE_COLOR) for holes in range(4)]
    ref_h = sum(p.shape[0] for p in ref_patches) + 3
    ref_w = max(p.shape[1] for p in ref_patches)

    H = random.randint(max(24, ref_h + 2), 30)
    W = random.randint(max(24, ref_w + 14), 32)
    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    legend_left = bool(random.randint(0, 1))
    top_margin = random.randint(0, max(0, H - ref_h))
    col0 = 0 if legend_left else W - ref_w
    rr = top_margin
    for patch in ref_patches:
        cc = col0 if legend_left else W - patch.shape[1]
        _paste_patch(grid, occupied, patch, rr, cc)
        rr += patch.shape[0] + 1

    free_c0 = 2 if legend_left else 0
    free_c1 = W - 2 if legend_left else W - 2
    if legend_left:
        free_c0 = ref_w + 2
    else:
        free_c1 = W - ref_w - 2
    if free_c1 - free_c0 < 8:
        return None

    random.shuffle(objects)
    for obj in objects:
        for idx, patch in enumerate(obj["parts"]):
            placed = False
            positions = [(r, c) for r in range(H - patch.shape[0] + 1) for c in range(free_c0, free_c1 - patch.shape[1] + 1)]
            random.shuffle(positions)
            for r0, c0 in positions:
                if _can_place(occupied, patch, r0, c0, padding=1):
                    _paste_patch(grid, occupied, patch, r0, c0)
                    obj[f"part_{idx}_bbox"] = (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1)
                    placed = True
                    break
            if not placed:
                return None
        obj["sort_key"] = (
            min(obj["part_0_bbox"][0], obj["part_1_bbox"][0]),
            min(obj["part_0_bbox"][1], obj["part_1_bbox"][1]),
        )

    expected = _build_expected(objects)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GEN_ATTEMPTS} attempts")
