from __future__ import annotations

import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _label_components(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[list[tuple[int, int]]] = []
    dirs = DIR4 if connectivity == 4 else DIR8
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(labels[r, c]) >= 0:
                continue
            stack = [(r, c)]
            labels[r, c] = comp_id
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in dirs:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and int(labels[nr, nc]) < 0:
                        labels[nr, nc] = comp_id
                        stack.append((nr, nc))
            comps.append(coords)
            comp_id += 1
    return labels, comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _tight_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if bool(cell) else "0" for cell in row) for row in mask)


def _shape_signature(mask: np.ndarray) -> tuple[str, ...]:
    base = _tight_bool(mask)
    variants = []
    for k in range(4):
        rot = np.rot90(base, k)
        variants.append(_mask_key(_tight_bool(rot)))
        variants.append(_mask_key(_tight_bool(np.fliplr(rot))))
    return min(variants)


def _random_dihedral(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.astype(bool).copy()


def _dilate_box(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    padded = np.pad(mask, radius, constant_values=False)
    out = np.zeros((h, w), dtype=bool)
    span = 2 * radius + 1
    for dr in range(span):
        for dc in range(span):
            out |= padded[dr : dr + h, dc : dc + w]
    return out


def _mask_has_hole(mask: np.ndarray) -> bool:
    bg = ~mask.astype(bool)
    if not bool(bg.any()):
        return False
    h, w = bg.shape
    seen = np.zeros_like(bg, dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if bool(bg[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bool(bg[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and bool(bg[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                stack.append((nr, nc))
    return bool((bg & ~seen).any())


def _expand_cells(base: np.ndarray, seeds: list[tuple[int, int]], steps: int, bg: int) -> list[tuple[int, int]]:
    if steps <= 0:
        return []
    h, w = base.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int, int]] = deque()
    for r, c in seeds:
        if not bool(seen[r, c]):
            seen[r, c] = True
            q.append((int(r), int(c), 0))
    grown: list[tuple[int, int]] = []
    while q:
        r, c, dist = q.popleft()
        if dist >= steps:
            continue
        for dr, dc in DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if bool(seen[nr, nc]) or int(base[nr, nc]) != int(bg):
                continue
            seen[nr, nc] = True
            grown.append((nr, nc))
            q.append((nr, nc, dist + 1))
    return grown


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b

    def gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return gap1(ar0, ar1, br0, br1) + gap1(ac0, ac1, bc0, bc1)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    non_bg = g[g != int(bg)]
    assert non_bg.size > 0, "expected at least one non-background color"
    vals, cnts = np.unique(non_bg, return_counts=True)
    main_color = int(vals[cnts.argmax()])

    main_labels, main_comps = _label_components(g == int(main_color), connectivity=4)
    assert main_comps, "expected at least one enclosing region"

    sprite_mask = (g != int(bg)) & (g != int(main_color))
    _sprite_labels, sprite_comps = _label_components(sprite_mask, connectivity=4)
    shape_to_color: dict[tuple[str, ...], int] = {}
    sprite_cells: list[tuple[int, int]] = []
    for comp in sprite_comps:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "every sprite must be monochrome"
        color = int(next(iter(colors)))
        sig = _shape_signature(_tight_mask_from_coords(comp))
        if sig in shape_to_color:
            assert int(shape_to_color[sig]) == int(color), "same shape cannot map to multiple colors"
        shape_to_color[sig] = color
        sprite_cells.extend(comp)

    out = g.copy()
    fill_colors_by_region: list[set[int]] = [set() for _ in main_comps]
    _bg_labels, bg_comps = _label_components(g == int(bg), connectivity=4)
    h, w = g.shape
    for comp in bg_comps:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        adjacent_regions = set()
        for r, c in comp:
            for dr, dc in DIR4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and int(main_labels[nr, nc]) >= 0:
                    adjacent_regions.add(int(main_labels[nr, nc]))
        if len(adjacent_regions) != 1:
            continue
        region_id = int(next(iter(adjacent_regions)))
        sig = _shape_signature(_tight_mask_from_coords(comp))
        assert sig in shape_to_color, "every hole must match exactly one sprite shape"
        fill_color = int(shape_to_color[sig])
        fill_colors_by_region[region_id].add(fill_color)
        for r, c in comp:
            out[r, c] = fill_color

    filled = out.copy()
    grow_mask = np.zeros_like(g, dtype=bool)
    for region_id, comp in enumerate(main_comps):
        k = len(fill_colors_by_region[region_id])
        for r, c in _expand_cells(filled, comp, k, bg):
            grow_mask[r, c] = True
    out[grow_mask] = int(main_color)

    for r, c in sprite_cells:
        out[r, c] = int(bg)
    return out


GEN_MAX_ATTEMPTS = 400
GEN_REGION_RETRIES = 120
GEN_GRID_H = (24, 36)
GEN_GRID_W = (24, 38)
GEN_REGION_COUNT = (1, 3)
GEN_HOLES_PER_REGION = (1, 3)
GEN_REGION_BORDER = 4
GEN_REGION_PADDING = 2
GEN_SPRITE_PADDING = 1
GEN_SPRITE_REGION_GAP = 5
GEN_SHAPE_RETRIES = 300


def _random_connected_mask() -> np.ndarray | None:
    for _ in range(80):
        area = random.choices([1, 2, 3, 4, 5], weights=[1, 2, 3, 3, 2])[0]
        h = random.randint(1, 4)
        w = random.randint(1, 4)
        if h * w < area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        r0 = random.randrange(h)
        c0 = random.randrange(w)
        mask[r0, c0] = True
        while int(mask.sum()) < area:
            frontier = set()
            for r, c in np.argwhere(mask):
                for dr, dc in DIR4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(list(frontier))
            mask[nr, nc] = True
        if int(mask.sum()) != area:
            continue
        mask = _tight_bool(mask)
        if _mask_has_hole(mask):
            continue
        return mask
    return None


def _unique_shapes(n: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    seen: set[tuple[str, ...]] = set()
    for _ in range(GEN_SHAPE_RETRIES):
        mask = _random_connected_mask()
        if mask is None:
            continue
        sig = _shape_signature(mask)
        if sig in seen:
            continue
        seen.add(sig)
        shapes.append(mask)
        if len(shapes) == n:
            return shapes
    return None


def _place_holes_in_core(core_h: int, core_w: int, masks: list[np.ndarray]) -> list[tuple[int, int, np.ndarray]] | None:
    order = list(range(len(masks)))
    random.shuffle(order)
    occupied = np.zeros((core_h, core_w), dtype=bool)
    reserved = np.zeros((core_h, core_w), dtype=bool)
    placed: dict[int, tuple[int, int, np.ndarray]] = {}
    for idx in order:
        mask = masks[idx]
        h, w = mask.shape
        if core_h - h < 2 or core_w - w < 2:
            return None
        options = [(r, c) for r in range(1, core_h - h) for c in range(1, core_w - w)]
        random.shuffle(options)
        choice = None
        for r, c in options:
            full = np.zeros((core_h, core_w), dtype=bool)
            full[r : r + h, c : c + w] = mask
            if bool((full & reserved).any()):
                continue
            choice = (r, c, full)
            break
        if choice is None:
            return None
        r, c, full = choice
        occupied |= full
        reserved = _dilate_box(occupied, radius=1)
        placed[idx] = (int(r), int(c), mask)
    return [placed[i] for i in range(len(masks))]


def _random_span(length: int) -> tuple[int, int]:
    if length <= 2:
        return 0, length
    lo = max(2, length // 3)
    hi = length if random.random() < 0.35 else max(lo, length - 1)
    span = random.randint(lo, hi)
    start = random.randint(0, length - span)
    return start, span


def _build_region_patch(
    bg: int,
    main_color: int,
    hole_shapes: list[np.ndarray],
    hole_colors: list[int],
) -> dict[str, object] | None:
    hole_count = len(hole_shapes)
    for _ in range(GEN_REGION_RETRIES):
        oriented = [_random_dihedral(shape) for shape in hole_shapes]
        max_h = max(mask.shape[0] for mask in oriented)
        max_w = max(mask.shape[1] for mask in oriented)
        core_h = random.randint(max(7, max_h + 4), max(9, max_h + 4 + 2 * hole_count))
        core_w = random.randint(max(7, max_w + 4), max(10, max_w + 5 + 2 * hole_count))
        placements = _place_holes_in_core(core_h, core_w, oriented)
        if placements is None:
            continue

        top = random.randint(0, 2)
        bottom = random.randint(0, 2)
        left = random.randint(0, 2)
        right = random.randint(0, 2)
        if top == bottom == left == right == 0:
            side = random.choice(["top", "bottom", "left", "right"])
            if side == "top":
                top = random.randint(1, 2)
            elif side == "bottom":
                bottom = random.randint(1, 2)
            elif side == "left":
                left = random.randint(1, 2)
            else:
                right = random.randint(1, 2)

        patch_h = core_h + top + bottom
        patch_w = core_w + left + right
        patch = np.full((patch_h, patch_w), int(bg), dtype=int)
        base_r = top
        base_c = left
        patch[base_r : base_r + core_h, base_c : base_c + core_w] = int(main_color)

        bulges = 0
        if top > 0 and (random.random() < 0.8 or bulges == 0):
            start, span = _random_span(core_w)
            patch[0:top, base_c + start : base_c + start + span] = int(main_color)
            bulges += 1
        if bottom > 0 and (random.random() < 0.8 or bulges == 0):
            start, span = _random_span(core_w)
            patch[base_r + core_h : base_r + core_h + bottom, base_c + start : base_c + start + span] = int(main_color)
            bulges += 1
        if left > 0 and (random.random() < 0.8 or bulges == 0):
            start, span = _random_span(core_h)
            patch[base_r + start : base_r + start + span, 0:left] = int(main_color)
            bulges += 1
        if right > 0 and (random.random() < 0.8 or bulges == 0):
            start, span = _random_span(core_h)
            patch[base_r + start : base_r + start + span, base_c + core_w : base_c + core_w + right] = int(main_color)
            bulges += 1

        hole_infos = []
        for i, (r, c, mask) in enumerate(placements):
            rr = int(base_r + r)
            cc = int(base_c + c)
            window = patch[rr : rr + mask.shape[0], cc : cc + mask.shape[1]]
            window[mask] = int(bg)
            coords = [(int(rr + dr), int(cc + dc)) for dr, dc in np.argwhere(mask)]
            hole_infos.append(
                {
                    "coords": coords,
                    "color": int(hole_colors[i]),
                    "signature": _shape_signature(mask),
                    "mask": mask.copy(),
                }
            )

        region_coords = [(int(r), int(c)) for r, c in np.argwhere(patch == int(main_color))]
        return {
            "patch": patch,
            "region_coords": region_coords,
            "hole_infos": hole_infos,
            "k": len(set(int(x) for x in hole_colors)),
        }
    return None


def _can_place_patch(
    canvas: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    *,
    padding: int,
    border: int,
) -> bool:
    h, w = patch.shape
    canvas_h, canvas_w = canvas.shape
    if r0 < border or c0 < border or r0 + h > canvas_h - border or c0 + w > canvas_w - border:
        return False
    occ = canvas != int(bg)
    for dr, dc in np.argwhere(patch != int(bg)):
        r = int(r0 + dr)
        c = int(c0 + dc)
        r1 = max(0, r - padding)
        r2 = min(canvas_h, r + padding + 1)
        c1 = max(0, c - padding)
        c2 = min(canvas_w, c + padding + 1)
        if bool(occ[r1:r2, c1:c2].any()):
            return False
    return True


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    view = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(bg)
    view[mask] = patch[mask]


def _build_expected(
    inp: np.ndarray,
    *,
    bg: int,
    main_color: int,
    regions: list[dict[str, object]],
    sprites: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    for region in regions:
        for hole in region["holes"]:
            for r, c in hole["coords"]:
                out[r, c] = int(hole["color"])
    filled = out.copy()
    grow_mask = np.zeros_like(out, dtype=bool)
    for region in regions:
        for r, c in _expand_cells(filled, region["coords"], int(region["k"]), bg):
            grow_mask[r, c] = True
    out[grow_mask] = int(main_color)
    for sprite in sprites:
        for r, c in sprite["coords"]:
            out[r, c] = int(bg)
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    main_color = random.choice([c for c in range(10) if c != bg])
    fill_pool = [c for c in range(10) if c not in (bg, main_color)]

    region_count = random.randint(*GEN_REGION_COUNT)
    hole_counts = [random.randint(*GEN_HOLES_PER_REGION) for _ in range(region_count)]
    total_holes = sum(hole_counts)
    shapes = _unique_shapes(total_holes)
    if shapes is None:
        return None

    shape_cursor = 0
    region_specs = []
    for hole_count in hole_counts:
        target_k = random.randint(1, min(3, hole_count))
        palette = random.sample(fill_pool, target_k)
        hole_colors = palette[:] + [random.choice(palette) for _ in range(hole_count - target_k)]
        random.shuffle(hole_colors)
        region_shapes = shapes[shape_cursor : shape_cursor + hole_count]
        shape_cursor += hole_count
        spec = _build_region_patch(bg, main_color, region_shapes, hole_colors)
        if spec is None:
            return None
        region_specs.append(spec)

    min_h = min(GEN_GRID_H[1], GEN_GRID_H[0] + 2 * (region_count - 1) + total_holes)
    min_w = min(GEN_GRID_W[1], GEN_GRID_W[0] + 2 * (region_count - 1) + total_holes)
    h = random.randint(min_h, GEN_GRID_H[1])
    w = random.randint(min_w, GEN_GRID_W[1])
    inp = np.full((h, w), int(bg), dtype=int)

    placed_regions: list[dict[str, object]] = []
    for spec in region_specs:
        patch = spec["patch"]
        ph, pw = patch.shape
        placed = False
        for _ in range(300):
            r0 = random.randint(0, h - ph)
            c0 = random.randint(0, w - pw)
            if not _can_place_patch(inp, patch, r0, c0, bg, padding=GEN_REGION_PADDING, border=GEN_REGION_BORDER):
                continue
            cand_bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            okay = True
            for prev in placed_regions:
                if _bbox_gap(cand_bbox, prev["bbox"]) < int(spec["k"]) + int(prev["k"]) + 2:
                    okay = False
                    break
            if not okay:
                continue
            _paste_patch(inp, patch, r0, c0, bg)
            holes = []
            for hole in spec["hole_infos"]:
                holes.append(
                    {
                        "coords": [(int(r0 + r), int(c0 + c)) for r, c in hole["coords"]],
                        "color": int(hole["color"]),
                        "signature": hole["signature"],
                    }
                )
            placed_regions.append(
                {
                    "coords": [(int(r0 + r), int(c0 + c)) for r, c in spec["region_coords"]],
                    "bbox": cand_bbox,
                    "k": int(spec["k"]),
                    "holes": holes,
                }
            )
            placed = True
            break
        if not placed:
            return None

    sprites: list[dict[str, object]] = []
    for region in placed_regions:
        for hole in region["holes"]:
            # Rebuild the exact mask from the signature strings.
            sig_rows = list(hole["signature"])
            shape = np.array([[ch == "1" for ch in row] for row in sig_rows], dtype=bool)
            shape = _random_dihedral(shape)
            sprite_patch = np.full(shape.shape, int(bg), dtype=int)
            sprite_patch[shape] = int(hole["color"])
            sh, sw = sprite_patch.shape
            placed = False
            for _ in range(300):
                r0 = random.randint(0, h - sh)
                c0 = random.randint(0, w - sw)
                if not _can_place_patch(inp, sprite_patch, r0, c0, bg, padding=GEN_SPRITE_PADDING, border=1):
                    continue
                cand_bbox = (r0, c0, r0 + sh - 1, c0 + sw - 1)
                if any(_bbox_gap(cand_bbox, region_info["bbox"]) < int(region_info["k"]) + GEN_SPRITE_REGION_GAP for region_info in placed_regions):
                    continue
                if any(_bbox_gap(cand_bbox, sprite["bbox"]) < 2 for sprite in sprites):
                    continue
                _paste_patch(inp, sprite_patch, r0, c0, bg)
                coords = [(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(shape)]
                sprites.append({"coords": coords, "bbox": cand_bbox})
                placed = True
                break
            if not placed:
                return None

    expected = _build_expected(inp, bg=bg, main_color=main_color, regions=placed_regions, sprites=sprites)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
