from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
RED = 2
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[top : top + h, left : left + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _neighbors(connectivity: int) -> list[tuple[int, int]]:
    if int(connectivity) == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    neigh = _neighbors(connectivity)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in neigh:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _touches(mask: np.ndarray, coords: np.ndarray) -> bool:
    h, w = mask.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    return True
    return False


def _catalog_background(grid: np.ndarray) -> tuple[int, np.ndarray]:
    comps: list[tuple[int, int, np.ndarray]] = []
    for color in sorted(set(map(int, grid.flat)) - {BLACK}):
        for coords in components_mask(grid == color, connectivity=4):
            comps.append((len(coords), int(color), coords))
    assert comps, "expected a non-black catalog background"
    comps.sort(key=lambda item: item[0], reverse=True)
    assert len(comps) == 1 or comps[0][0] > comps[1][0], "catalog background must be uniquely largest"

    _area, color, coords = comps[0]
    mask = np.zeros_like(grid, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    assert np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[:, -1])
    return color, mask


def _extract_templates(grid: np.ndarray, catalog_color: int, catalog_mask: np.ndarray) -> list[np.ndarray]:
    visible = (grid != BLACK) & (grid != int(catalog_color))
    templates: list[np.ndarray] = []
    for coords in components_mask(visible, connectivity=8):
        if not _touches(catalog_mask, coords):
            continue
        box = bbox_coords(coords)
        raw = crop_bbox(grid, box)
        sprite = np.where(raw == int(catalog_color), BLACK, raw).astype(int)
        assert np.any(sprite != BLACK), "empty template"
        templates.append(sprite)
    assert templates, "expected catalog templates"
    return templates


def _workspace_parts(grid: np.ndarray, catalog_mask: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    visible = (grid != BLACK) & (~catalog_mask)
    parts: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for coords in components_mask(visible, connectivity=8):
        if _touches(catalog_mask, coords):
            continue
        box = bbox_coords(coords)
        parts.append((box, crop_bbox(grid, box)))
    return parts


def _transform(sprite: np.ndarray, label: int) -> np.ndarray:
    k = int(label) % 4
    arr = np.rot90(sprite, k)
    if int(label) >= 4:
        arr = np.fliplr(arr)
    return arr.copy()


def _unique_variants(sprite: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for label in range(8):
        arr = _transform(sprite, label)
        key = (tuple(map(int, arr.shape)), arr.tobytes())
        if key in seen:
            continue
        seen.add(key)
        variants.append(arr)
    return variants


def _match_offsets(part: np.ndarray, sprite: np.ndarray) -> list[tuple[int, int]]:
    ph, pw = part.shape
    sh, sw = sprite.shape
    if ph > sh or pw > sw:
        return []
    part_mask = part != BLACK
    hits: list[tuple[int, int]] = []
    for off_r in range(sh - ph + 1):
        for off_c in range(sw - pw + 1):
            sub = sprite[off_r : off_r + ph, off_c : off_c + pw]
            if np.array_equal(sub[part_mask], part[part_mask]):
                hits.append((off_r, off_c))
    return hits


def _find_unique_match(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    catalog_color, catalog_mask = _catalog_background(g)
    templates = _extract_templates(g, catalog_color, catalog_mask)
    parts = _workspace_parts(g, catalog_mask)
    assert len(parts) == 1, "expected exactly one workspace sprite"

    (r0, c0, _r1, _c1), part = parts[0]
    matches: list[tuple[int, int, bytes]] = []
    chosen: tuple[tuple[int, int], np.ndarray] | None = None
    for template in templates:
        for variant in _unique_variants(template):
            vh, vw = variant.shape
            for off_r, off_c in _match_offsets(part, variant):
                anchor = (r0 - off_r, c0 - off_c)
                ar, ac = anchor
                if ar < 0 or ac < 0 or ar + vh > g.shape[0] or ac + vw > g.shape[1]:
                    continue
                key = (int(ar), int(ac), variant.tobytes())
                if key in matches:
                    continue
                matches.append(key)
                chosen = (anchor, variant)

    assert len(matches) == 1, "partial sprite must have a unique template pose match"
    assert chosen is not None
    return chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the workspace remnant by matching it to the only compatible catalog template pose and paint the missing cells red.

    Input structure:
    - One side of the grid is a non-black catalog strip containing several complete multicolor template sprites.
    - The rest of the grid is a black workspace with exactly one multicolor remnant sprite.
    - Red does not appear in the input; the remnant is a partial view of one template under some rotation or mirror pose.

    Rule:
    - Enumerate every catalog template under all rotations and mirror poses.
    - Find the unique transformed template whose colored pixels can cover every visible remnant pixel with exact color agreement after alignment.
    - Keep the catalog and visible remnant unchanged, and fill every other pixel of that matched transformed template in the workspace with red.

    Key insight:
    - The remnant does not complete itself by local symmetry; the correct completion is determined by a unique whole-template match against the catalog after trying hidden poses.
    """
    g = np.asarray(grid, dtype=int)
    anchor, variant = _find_unique_match(g)
    out = g.copy()
    ar, ac = anchor
    for rr, cc in np.argwhere(variant != BLACK):
        gr = ar + int(rr)
        gc = ac + int(cc)
        if out[gr, gc] == BLACK:
            out[gr, gc] = RED
    return out


def _connected_subset(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    return coords.size > 0 and len(components_mask(mask, connectivity=4)) == 1


def _occupancy_symmetry(mask: np.ndarray) -> bool:
    return np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask))


def _grow_mask(height: int, width: int, target_area: int) -> np.ndarray | None:
    target_area = int(target_area)
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = {start}
    while len(cells) < target_area and frontier:
        r, c = random.choice(tuple(frontier))
        options = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                options.append((nr, nc))
        if not options:
            frontier.remove((r, c))
            continue
        cell = random.choice(options)
        cells.add(cell)
        frontier.add(cell)
    if len(cells) != target_area:
        return None

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if not (np.any(mask[0, :]) and np.any(mask[-1, :]) and np.any(mask[:, 0]) and np.any(mask[:, -1])):
        return None
    return mask


def _colorize_mask(mask: np.ndarray, colors: tuple[int, int]) -> np.ndarray | None:
    coords = np.argwhere(mask)
    if coords.shape[0] < 6:
        return None
    seed_a = tuple(coords[random.randrange(len(coords))])
    farthest = max(coords, key=lambda rc: abs(int(rc[0]) - int(seed_a[0])) + abs(int(rc[1]) - int(seed_a[1])))
    seed_b = (int(farthest[0]), int(farthest[1]))
    if seed_a == seed_b:
        return None

    sprite = np.zeros(mask.shape, dtype=int)
    count = Counter()
    for r, c in coords:
        da = abs(int(r) - seed_a[0]) + abs(int(c) - seed_a[1])
        db = abs(int(r) - seed_b[0]) + abs(int(c) - seed_b[1])
        if da == db:
            color = colors[(int(r) + int(c)) % 2]
        else:
            color = colors[0] if da < db else colors[1]
        sprite[int(r), int(c)] = int(color)
        count[int(color)] += 1
    if min(count.values()) < 2:
        return None
    return sprite


def _make_template(size_range: tuple[int, int], colors: tuple[int, int]) -> np.ndarray | None:
    min_size, max_size = map(int, size_range)
    height = random.randint(min_size, max_size)
    width = random.randint(min_size, max_size)
    area_lo = max(height + width - 1, int(np.ceil(height * width * 0.45)))
    area_hi = int(np.floor(height * width * 0.75))
    if area_lo > area_hi:
        return None
    for _ in range(80):
        mask = _grow_mask(height, width, random.randint(area_lo, area_hi))
        if mask is None or not _connected_subset(mask):
            continue
        sprite = _colorize_mask(mask, colors)
        if sprite is None:
            continue
        if len(_unique_variants(sprite)) < 4:
            continue
        return sprite
    return None


def _embed_catalog(grid: np.ndarray, thickness: int, templates: list[np.ndarray], edge: str) -> bool:
    h, w = grid.shape
    if edge in ("top", "bottom"):
        total_w = sum(t.shape[1] for t in templates) + 2 * (len(templates) - 1)
        if total_w + 2 > w:
            return False
        if any(t.shape[0] > thickness for t in templates):
            return False
        gap_slots = [0] * (len(templates) + 1)
        spare = w - 2 - total_w
        for _ in range(spare):
            gap_slots[random.randrange(len(gap_slots))] += 1
        c = 1 + gap_slots[0]
        y0 = 0 if edge == "top" else h - thickness
        for idx, template in enumerate(templates):
            th, tw = template.shape
            top = y0 + random.randint(0, max(0, thickness - th))
            paste(grid, template, top, c, transparent=BLACK)
            c += tw + 2 + gap_slots[idx + 1]
        return True

    total_h = sum(t.shape[0] for t in templates) + 2 * (len(templates) - 1)
    if total_h + 2 > h:
        return False
    if any(t.shape[1] > thickness for t in templates):
        return False
    gap_slots = [0] * (len(templates) + 1)
    spare = h - 2 - total_h
    for _ in range(spare):
        gap_slots[random.randrange(len(gap_slots))] += 1
    r = 1 + gap_slots[0]
    x0 = 0 if edge == "left" else w - thickness
    for idx, template in enumerate(templates):
        th, tw = template.shape
        left = x0 + random.randint(0, max(0, thickness - tw))
        paste(grid, template, r, left, transparent=BLACK)
        r += th + 2 + gap_slots[idx + 1]
    return True


def _catalog_strip(shape: tuple[int, int], edge: str, thickness: int, color: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    mask = np.zeros((h, w), dtype=bool)
    if edge == "top":
        grid[:thickness, :] = int(color)
        mask[:thickness, :] = True
    elif edge == "bottom":
        grid[h - thickness :, :] = int(color)
        mask[h - thickness :, :] = True
    elif edge == "left":
        grid[:, :thickness] = int(color)
        mask[:, :thickness] = True
    else:
        grid[:, w - thickness :] = int(color)
        mask[:, w - thickness :] = True
    return grid, mask


def _random_symmetric_stencil(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    kind = random.choice(("rect", "cross", "diamond", "mirror"))
    keep = np.zeros((h, w), dtype=bool)
    if kind == "rect":
        top = random.randint(0, max(0, h // 2 - 1))
        bottom = random.randint(top + 1, h - 1)
        left = random.randint(0, max(0, w // 2 - 1))
        right = random.randint(left + 1, w - 1)
        top = min(top, h - 1 - bottom)
        left = min(left, w - 1 - right)
        keep[top : h - top, left : w - left] = True
        return keep
    if kind == "cross":
        row_band = random.randint(1, max(1, min(2, h)))
        col_band = random.randint(1, max(1, min(2, w)))
        r0 = (h - row_band) // 2
        c0 = (w - col_band) // 2
        keep[r0 : r0 + row_band, :] = True
        keep[:, c0 : c0 + col_band] = True
        return keep
    if kind == "diamond":
        cr = (h - 1) / 2.0
        cc = (w - 1) / 2.0
        radius = random.randint(1, max(1, int((h + w) / 4)))
        for r in range(h):
            for c in range(w):
                if abs(r - cr) + abs(c - cc) <= radius + 0.1:
                    keep[r, c] = True
        return keep

    axis = random.choice(("h", "v"))
    if axis == "h":
        left_w = (w + 1) // 2
        half = np.random.rand(h, left_w) < random.uniform(0.45, 0.8)
        for r in range(h):
            for c in range(left_w):
                keep[r, c] = bool(half[r, c])
                keep[r, w - 1 - c] = bool(half[r, c])
    else:
        top_h = (h + 1) // 2
        half = np.random.rand(top_h, w) < random.uniform(0.45, 0.8)
        for r in range(top_h):
            for c in range(w):
                keep[r, c] = bool(half[r, c])
                keep[h - 1 - r, c] = bool(half[r, c])
    return keep


def _random_connected_subset(sprite: np.ndarray, target: int) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(sprite != BLACK)]
    start = random.choice(coords)
    chosen = {start}
    frontier = {start}
    while len(chosen) < target and frontier:
        r, c = random.choice(tuple(frontier))
        options = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < sprite.shape[0] and 0 <= nc < sprite.shape[1] and sprite[nr, nc] != BLACK and (nr, nc) not in chosen:
                options.append((nr, nc))
        if not options:
            frontier.remove((r, c))
            continue
        cell = random.choice(options)
        chosen.add(cell)
        frontier.add(cell)

    subset = np.zeros_like(sprite)
    for r, c in chosen:
        subset[r, c] = sprite[r, c]
    return subset


def _select_partial(
    full_sprite: np.ndarray,
    all_templates: list[np.ndarray],
    catalog_color: int,
    *,
    prefer_symmetric: bool,
) -> np.ndarray | None:
    full_area = int(np.count_nonzero(full_sprite))
    for attempt in range(220):
        if prefer_symmetric and attempt < 140:
            stencil = _random_symmetric_stencil(full_sprite.shape)
            candidate = np.where(stencil, full_sprite, BLACK)
        else:
            target = random.randint(max(4, int(np.ceil(full_area * 0.4))), max(4, int(np.floor(full_area * 0.72))))
            candidate = _random_connected_subset(full_sprite, target)
        coords = np.argwhere(candidate != BLACK)
        if coords.size == 0 or coords.shape[0] >= full_area:
            continue
        box = bbox_coords(coords)
        part = crop_bbox(candidate, box)
        occ = part != BLACK
        if len(set(map(int, part[occ].flat))) < 2:
            continue
        if not _connected_subset(occ):
            continue
        if prefer_symmetric and attempt < 140 and not _occupancy_symmetry(occ):
            continue
        catalog, _mask = _catalog_strip((28, 28), "top", 8, catalog_color)
        grid = catalog.copy()
        x = 1
        for template in all_templates:
            paste(grid, template, 1, x, transparent=BLACK)
            x += template.shape[1] + 2
        paste(grid, part, 10, 10, transparent=BLACK)
        try:
            _find_unique_match(grid)
        except Exception:
            continue
        return part
    return None


def _place_workspace(grid: np.ndarray, catalog_mask: np.ndarray, part: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = part.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h - ph + 1):
        for c in range(w - pw + 1):
            if np.any(catalog_mask[max(0, r - 1) : min(h, r + ph + 1), max(0, c - 1) : min(w, c + pw + 1)]):
                continue
            region = grid[r : r + ph, c : c + pw]
            if np.all(region[part != BLACK] == BLACK):
                candidates.append((r, c))
    if not candidates:
        return False
    top, left = random.choice(candidates)
    paste(grid, part, top, left, transparent=BLACK)
    return True


def _manual_output(inp: np.ndarray) -> np.ndarray:
    anchor, variant = _find_unique_match(inp)
    out = inp.copy()
    ar, ac = anchor
    for rr, cc in np.argwhere(variant != BLACK):
        gr = ar + int(rr)
        gc = ac + int(cc)
        if out[gr, gc] == BLACK:
            out[gr, gc] = RED
    return out


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(20, 30)
    width = random.randint(20, 30)
    edge = random.choice(("top", "bottom", "left", "right"))
    thickness = random.randint(6, 8)
    if edge in ("top", "bottom") and height - thickness < 7:
        return None
    if edge in ("left", "right") and width - thickness < 7:
        return None

    n_templates = random.randint(2, 3)
    color_pool = list(PALETTE)
    random.shuffle(color_pool)
    catalog_color = color_pool.pop()

    templates: list[np.ndarray] = []
    for _ in range(n_templates):
        if len(color_pool) < 2:
            return None
        colors = (color_pool.pop(), color_pool.pop())
        template = _make_template((4, 7), colors)
        if template is None:
            return None
        templates.append(template)

    if edge in ("top", "bottom"):
        needed = sum(t.shape[1] for t in templates) + 2 * (len(templates) - 1) + 2
        if needed > width:
            return None
    else:
        needed = sum(t.shape[0] for t in templates) + 2 * (len(templates) - 1) + 2
        if needed > height:
            return None

    grid, catalog_mask = _catalog_strip((height, width), edge, thickness, catalog_color)
    if not _embed_catalog(grid, thickness, templates, edge):
        return None

    target_idx = random.randrange(len(templates))
    target_pose = random.randrange(8)
    full_sprite = _transform(templates[target_idx], target_pose)
    prefer_symmetric = random.random() < 0.7
    part = _select_partial(full_sprite, templates, catalog_color, prefer_symmetric=prefer_symmetric)
    if part is None:
        return None
    if not _place_workspace(grid, catalog_mask, part):
        return None

    try:
        oracle = _manual_output(grid)
        solved = main(grid)
    except Exception:
        return None
    if np.array_equal(oracle, solved) and np.any(solved == RED):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _attempt_generate()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
