import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    start_r, start_c = map(int, coords[0])
    stack = [(start_r, start_c)]
    seen[start_r, start_c] = True
    count = 0
    while stack:
        r, c = stack.pop()
        count += 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return count == int(mask.sum())


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _fragment_mask(zone_h: int, zone_w: int, require_nonfull: bool) -> np.ndarray | None:
    h_choices = [size for size in range(1, zone_h + 1) if (zone_h - size) % 2 == 0]
    w_choices = [size for size in range(1, zone_w + 1) if (zone_w - size) % 2 == 0]
    random.shuffle(h_choices)
    random.shuffle(w_choices)

    for fh in h_choices:
        for fw in w_choices:
            if fh == 1 and fw == 1 and require_nonfull:
                continue
            mask = np.ones((fh, fw), dtype=bool)
            max_remove = fh * fw - 1
            if require_nonfull and max_remove == 0:
                continue
            min_remove = 1 if require_nonfull else 0
            max_target = max(min_remove, (fh * fw) // 2)
            target_remove = random.randint(min_remove, max_target) if max_target > 0 else 0

            cells = [(r, c) for r in range(fh) for c in range(fw)]
            random.shuffle(cells)
            removed = 0
            progress = True
            while progress and removed < target_remove:
                progress = False
                for r, c in cells:
                    if not mask[r, c]:
                        continue
                    mask[r, c] = False
                    if not _touches_all_edges(mask) or not _mask_connected(mask):
                        mask[r, c] = True
                        continue
                    removed += 1
                    progress = True
                    if removed >= target_remove:
                        break

            if require_nonfull and mask.all():
                continue
            if not _mask_connected(mask) or not _touches_all_edges(mask):
                continue
            return mask
    return None


def _split_template(height: int, width: int, count: int, min_size: int) -> list[tuple[int, int, int, int]] | None:
    rects = [(0, 0, height, width)]
    while len(rects) < count:
        splittable: list[tuple[int, int]] = []
        for idx, (_, _, h, w) in enumerate(rects):
            if h >= 2 * min_size:
                splittable.append((idx, 0))
            if w >= 2 * min_size:
                splittable.append((idx, 1))
        if not splittable:
            return None
        idx, axis = random.choice(splittable)
        r0, c0, h, w = rects.pop(idx)
        if axis == 0:
            cut = random.randint(min_size, h - min_size)
            rects.extend([(r0, c0, cut, w), (r0 + cut, c0, h - cut, w)])
        else:
            cut = random.randint(min_size, w - min_size)
            rects.extend([(r0, c0, h, cut), (r0, c0 + cut, h, w - cut)])
    return rects


def _boxes_far_apart(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return (
        ar + ah + gap <= br
        or br + bh + gap <= ar
        or ac + aw + gap <= bc
        or bc + bw + gap <= ac
    )


def _place_boxes(
    grid_h: int,
    grid_w: int,
    template_box: tuple[int, int, int, int],
    fragment_shapes: list[np.ndarray],
    gap: int,
) -> list[tuple[int, int, int, int]] | None:
    placed = [template_box]
    result: list[tuple[int, int, int, int]] = []
    order = list(range(len(fragment_shapes)))
    order.sort(key=lambda idx: fragment_shapes[idx].shape[0] * fragment_shapes[idx].shape[1], reverse=True)
    ordered_boxes: dict[int, tuple[int, int, int, int]] = {}

    for idx in order:
        fh, fw = map(int, fragment_shapes[idx].shape)
        candidates = [
            (r, c, fh, fw)
            for r in range(1, grid_h - fh)
            for c in range(1, grid_w - fw)
        ]
        random.shuffle(candidates)
        chosen = None
        for box in candidates:
            if all(_boxes_far_apart(box, other, gap) for other in placed):
                chosen = box
                break
        if chosen is None:
            return None
        ordered_boxes[idx] = chosen
        placed.append(chosen)

    for idx in range(len(fragment_shapes)):
        result.append(ordered_boxes[idx])
    return result


def _expected_output(
    grid: np.ndarray,
    bg: int,
    zones: list[dict],
    fragments: dict[int, dict],
) -> np.ndarray:
    out = grid.copy()

    for frag in fragments.values():
        fr, fc = frag["pos"]
        mask = frag["mask"]
        fh, fw = mask.shape
        patch = out[fr : fr + fh, fc : fc + fw]
        patch[mask] = bg

    for zone in zones:
        color = zone["color"]
        if color not in fragments:
            continue
        zr, zc, zh, zw = zone["rect"]
        frag = fragments[color]
        mask = frag["mask"]
        fh, fw = mask.shape
        top = zr + (zh - fh) // 2
        left = zc + (zw - fw) // 2
        out[zr : zr + zh, zc : zc + zw] = bg
        patch = out[top : top + fh, left : left + fw]
        patch[mask] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Clip template color-rectangles by centering same-color fragment masks into them."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    comps = _components(g != bg)
    assert comps

    infos = []
    for coords in comps:
        colors = {int(g[r, c]) for r, c in coords}
        infos.append(
            {
                "coords": coords,
                "colors": colors,
                "area": len(coords),
                "bbox": _bbox(coords),
            }
        )

    template_candidates = [info for info in infos if len(info["colors"]) >= 2]
    assert len(template_candidates) == 1
    template = template_candidates[0]

    template_mask = np.zeros_like(g, dtype=bool)
    for r, c in template["coords"]:
        template_mask[r, c] = True

    zones = {}
    for color in template["colors"]:
        coords = [(r, c) for r, c in template["coords"] if int(g[r, c]) == color]
        assert coords
        r0, c0, r1, c1 = _bbox(coords)
        box = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(box == color)
        assert len(coords) == box.size
        zones[int(color)] = (r0, c0, r1, c1)

    fragments = {}
    for info in infos:
        if info is template:
            continue
        assert len(info["colors"]) == 1
        color = int(next(iter(info["colors"])))
        assert color in zones
        assert color not in fragments
        fragments[color] = info

    out = g.copy()
    for frag in fragments.values():
        for r, c in frag["coords"]:
            out[r, c] = bg

    for color, zone_bbox in zones.items():
        if color not in fragments:
            continue
        zr0, zc0, zr1, zc1 = zone_bbox
        frag = fragments[color]
        fr0, fc0, fr1, fc1 = frag["bbox"]
        dr = (zr0 + zr1 - fr0 - fr1) // 2
        dc = (zc0 + zc1 - fc0 - fc1) // 2
        out[zr0 : zr1 + 1, zc0 : zc1 + 1] = bg
        for r, c in frag["coords"]:
            nr = r + dr
            nc = c + dc
            assert zr0 <= nr <= zr1 and zc0 <= nc <= zc1
            out[nr, nc] = color

    return out


MIN_ZONE_SIZE = 3
ZONE_COUNT_RANGE = (3, 6)
TEMPLATE_H_RANGE = (8, 14)
TEMPLATE_W_RANGE = (8, 14)
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
PLACEMENT_GAP = 1
GEN_ATTEMPTS = 400


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)
    colors = [color for color in range(10) if color != bg]

    zone_count = random.randint(*ZONE_COUNT_RANGE)
    template_h = random.randint(*TEMPLATE_H_RANGE)
    template_w = random.randint(*TEMPLATE_W_RANGE)
    rects = _split_template(template_h, template_w, zone_count, MIN_ZONE_SIZE)
    if rects is None:
        return None

    zone_colors = random.sample(colors, zone_count)
    zones = []
    for color, rect in zip(zone_colors, rects):
        zones.append({"color": int(color), "rect": rect})

    visible_count = random.randint(1, zone_count)
    visible_colors = set(random.sample(zone_colors, visible_count))
    must_clip_color = random.choice(list(visible_colors))

    fragments: dict[int, dict] = {}
    nonfull_seen = False
    for zone in zones:
        color = zone["color"]
        if color not in visible_colors:
            continue
        _, _, zh, zw = zone["rect"]
        require_nonfull = color == must_clip_color
        mask = _fragment_mask(zh, zw, require_nonfull=require_nonfull)
        if mask is None:
            return None
        nonfull_seen |= not bool(mask.all())
        fragments[color] = {"mask": mask}
    if not nonfull_seen:
        return None

    grid_h = random.randint(max(GRID_H_RANGE[0], template_h + 4), GRID_H_RANGE[1])
    grid_w = random.randint(max(GRID_W_RANGE[0], template_w + 4), GRID_W_RANGE[1])
    if grid_h <= template_h + 2 or grid_w <= template_w + 2:
        return None

    tr = random.randint(1, grid_h - template_h - 1)
    tc = random.randint(1, grid_w - template_w - 1)
    template_box = (tr, tc, template_h, template_w)

    fragment_masks = [fragments[color]["mask"] for color in fragments]
    fragment_boxes = _place_boxes(grid_h, grid_w, template_box, fragment_masks, PLACEMENT_GAP)
    if fragment_boxes is None:
        return None

    for color, box in zip(fragments, fragment_boxes):
        fragments[color]["pos"] = (box[0], box[1])

    grid = np.full((grid_h, grid_w), bg, dtype=int)

    for zone in zones:
        color = zone["color"]
        zr, zc, zh, zw = zone["rect"]
        grid[tr + zr : tr + zr + zh, tc + zc : tc + zc + zw] = color
        zone["rect"] = (tr + zr, tc + zc, zh, zw)

    for color, frag in fragments.items():
        fr, fc = frag["pos"]
        mask = frag["mask"]
        fh, fw = mask.shape
        patch = grid[fr : fr + fh, fc : fc + fw]
        patch[mask] = color

    if _get_bg(grid) != bg:
        return None

    expected = _expected_output(grid, bg, zones, fragments)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, actual):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
