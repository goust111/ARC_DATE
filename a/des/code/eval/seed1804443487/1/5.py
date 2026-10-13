from __future__ import annotations

import random

import numpy as np


BG = 0
GENERATION_RETRIES = 400
STAMP_PLACEMENT_RETRIES = 200

_BASE_SHAPES = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
]


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(grid: np.ndarray, bg: int = BG) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            colors: set[int] = set()
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                colors.add(int(g[rr, cc]))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(g[nr, nc]) == int(bg):
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            arr = np.array(sorted(coords), dtype=int)
            comps.append(
                {
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "colors": tuple(sorted(colors)),
                }
            )

    return comps


def _mask_from_component(comp: dict[str, object]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel: list[tuple[int, int]] = []
    for r, c in comp["coords"]:
        rr = int(r) - r0
        cc = int(c) - c0
        mask[rr, cc] = True
        rel.append((rr, cc))
    rel.sort()
    return mask, rel


def _extract_scene(grid: np.ndarray) -> tuple[dict[str, object], np.ndarray, list[dict[str, object]]]:
    comps = _components(grid, bg=BG)
    assert len(comps) >= 3, "expected one canvas and at least two stamps"
    comps.sort(key=lambda d: (-int(d["area"]), int(d["bbox"][0]), int(d["bbox"][1])))

    canvas = comps[0]
    assert len(canvas["colors"]) >= 2, "canvas must be multicolor"

    g = np.asarray(grid, dtype=int)
    canvas_mask = np.zeros_like(g, dtype=bool)
    for r, c in canvas["coords"]:
        canvas_mask[int(r), int(c)] = True

    stamps: list[dict[str, object]] = []
    for comp in comps[1:]:
        assert len(comp["colors"]) == 1, "stamps must be monochrome"
        mask, rel = _mask_from_component(comp)
        stamps.append(
            {
                "bbox": comp["bbox"],
                "mask": mask,
                "rel": rel,
                "area": int(comp["area"]),
                "color": int(comp["colors"][0]),
            }
        )

    stamps.sort(
        key=lambda s: (
            -int(s["area"]),
            int(s["bbox"][0]),
            int(s["bbox"][1]),
            int(s["bbox"][2]),
            int(s["bbox"][3]),
        )
    )
    return canvas, canvas_mask, stamps


def _match_positions(
    grid: np.ndarray, canvas_mask: np.ndarray, stamp: dict[str, object]
) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rel = stamp["rel"]
    mask_h, mask_w = stamp["mask"].shape
    matches: list[dict[str, object]] = []

    for top in range(h - mask_h + 1):
        for left in range(w - mask_w + 1):
            first_color = None
            ok = True
            coords: list[tuple[int, int]] = []
            for dr, dc in rel:
                rr = top + int(dr)
                cc = left + int(dc)
                if not canvas_mask[rr, cc]:
                    ok = False
                    break
                color = int(g[rr, cc])
                if color == BG:
                    ok = False
                    break
                if first_color is None:
                    first_color = color
                elif color != first_color:
                    ok = False
                    break
                coords.append((rr, cc))
            if ok:
                matches.append({"anchor": (int(top), int(left)), "coords": coords, "src_color": int(first_color)})

    return matches


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]], dict[tuple[int, int], list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    canvas, canvas_mask, stamps = _extract_scene(g)
    out = g.copy()
    best_area = np.full(g.shape, -1, dtype=int)
    coverers: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for stamp_id, stamp in enumerate(stamps):
        matches = _match_positions(g, canvas_mask, stamp)
        stamp["matches"] = matches
        for match in matches:
            for rr, cc in match["coords"]:
                cell = (int(rr), int(cc))
                coverers.setdefault(cell, []).append((int(stamp_id), int(stamp["area"])))
                if int(stamp["area"]) > int(best_area[rr, cc]):
                    best_area[rr, cc] = int(stamp["area"])
                    out[rr, cc] = int(stamp["color"])

    return out, canvas, stamps, coverers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each small monochrome stamp as a shape detector on the large multicolor canvas.

    For every stamp:
    - extract its binary shape mask from its bbox,
    - slide that mask over the canvas,
    - match whenever all mask-covered canvas pixels are the same non-background color,
    - recolor the matched pixels to the stamp's color.

    If multiple stamps want the same pixel, the stamp with larger mask area wins.
    """
    out, _canvas, _stamps, _coverers = _simulate(np.asarray(grid, dtype=int))
    return out


def _unique_orientations(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for rot in range(4):
        cur = np.rot90(mask, rot)
        for variant in (cur, np.fliplr(cur)):
            key = (tuple(map(int, variant.shape)), variant.tobytes())
            if key not in seen:
                seen.add(key)
                out.append(variant.copy())
    return out


def _sample_stamp_shapes(num_stamps: int, force_conflict: bool) -> list[np.ndarray] | None:
    for _ in range(60):
        idxs = random.sample(range(len(_BASE_SHAPES)), num_stamps)
        masks = [random.choice(_unique_orientations(_BASE_SHAPES[idx])) for idx in idxs]
        if not force_conflict or len({int(mask.sum()) for mask in masks}) >= 2:
            return masks
    return None


def _base_canvas(h: int, w: int, palette: list[int]) -> np.ndarray:
    bias = random.randrange(len(palette))
    canvas = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            canvas[r, c] = int(palette[(2 * r + 3 * c + bias) % len(palette)])
    if random.random() < 0.5:
        canvas = np.flipud(canvas)
    if random.random() < 0.5:
        canvas = np.fliplr(canvas)
    return canvas


def _coords_from_mask(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {
        (int(top + r), int(left + c))
        for r, c in np.argwhere(mask)
    }


def _box_from_anchor(mask: np.ndarray, top: int, left: int) -> tuple[int, int, int, int]:
    h, w = map(int, mask.shape)
    return int(top), int(left), int(top + h - 1), int(left + w - 1)


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _anchor_options(mask: np.ndarray, canvas_h: int, canvas_w: int) -> list[tuple[tuple[int, int], set[tuple[int, int]], tuple[int, int, int, int]]]:
    h, w = map(int, mask.shape)
    out: list[tuple[tuple[int, int], set[tuple[int, int]], tuple[int, int, int, int]]] = []
    for top in range(canvas_h - h + 1):
        for left in range(canvas_w - w + 1):
            coords = _coords_from_mask(mask, top, left)
            out.append(((int(top), int(left)), coords, _box_from_anchor(mask, top, left)))
    random.shuffle(out)
    return out


def _pick_disjoint_option(
    options: list[tuple[tuple[int, int], set[tuple[int, int]], tuple[int, int, int, int]]],
    blocked_cells: set[tuple[int, int]],
    blocked_boxes: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int], set[tuple[int, int]], tuple[int, int, int, int]] | None:
    for anchor, coords, box in options:
        if not coords.isdisjoint(blocked_cells):
            continue
        if any(not _boxes_separated(box, other, gap=1) for other in blocked_boxes):
            continue
        return anchor, coords, box
    return None


def _find_conflict_pair(
    mask_a: np.ndarray, mask_b: np.ndarray, canvas_h: int, canvas_w: int
) -> tuple[
    tuple[int, int],
    set[tuple[int, int]],
    tuple[int, int, int, int],
    tuple[int, int],
    set[tuple[int, int]],
    tuple[int, int, int, int],
] | None:
    opts_a = _anchor_options(mask_a, canvas_h, canvas_w)
    opts_b = _anchor_options(mask_b, canvas_h, canvas_w)
    if not opts_a or not opts_b:
        return None

    for _ in range(180):
        anchor_a, coords_a, box_a = random.choice(opts_a)
        anchor_b, coords_b, box_b = random.choice(opts_b)
        overlap = coords_a & coords_b
        if not overlap:
            continue
        if len(overlap) >= min(len(coords_a), len(coords_b)):
            continue
        return anchor_a, coords_a, box_a, anchor_b, coords_b, box_b
    return None


def _place_stamps_outside(
    grid: np.ndarray,
    canvas_bbox: tuple[int, int, int, int],
    stamp_arrays: list[np.ndarray],
) -> bool:
    h, w = grid.shape
    placed_boxes = [canvas_bbox]
    order = list(range(len(stamp_arrays)))
    random.shuffle(order)

    for idx in order:
        patch = stamp_arrays[idx]
        ph, pw = map(int, patch.shape)
        candidates = [
            (top, left)
            for top in range(h - ph + 1)
            for left in range(w - pw + 1)
        ]
        random.shuffle(candidates)
        placed = False
        for top, left in candidates:
            box = (int(top), int(left), int(top + ph - 1), int(left + pw - 1))
            if any(not _boxes_separated(box, other, gap=1) for other in placed_boxes):
                continue
            if np.any(grid[top : top + ph, left : left + pw] != BG):
                continue
            stamp_mask = patch != BG
            grid_slice = grid[top : top + ph, left : left + pw]
            if np.any(grid_slice[stamp_mask] != BG):
                continue
            grid_slice[stamp_mask] = patch[stamp_mask]
            grid[top : top + ph, left : left + pw] = grid_slice
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return False

    return True


def _build_attempt() -> np.ndarray | None:
    num_stamps = random.randint(2, 4)
    force_conflict = bool(random.getrandbits(1))
    masks = _sample_stamp_shapes(num_stamps, force_conflict)
    if masks is None:
        return None

    palette_size = 5 if num_stamps <= 3 and random.random() < 0.5 else 4
    if num_stamps + palette_size > 9:
        palette_size = 9 - num_stamps
    colors = random.sample(range(1, 10), num_stamps + palette_size)
    stamp_colors = colors[:num_stamps]
    canvas_palette = colors[num_stamps:]

    stamp_arrays = [np.where(mask, int(color), BG) for mask, color in zip(masks, stamp_colors, strict=False)]

    canvas_h = random.randint(9, 14)
    canvas_w = random.randint(9, 14)
    grid_h = canvas_h + random.randint(7, 11)
    grid_w = canvas_w + random.randint(7, 11)
    if grid_h < canvas_h + 5 or grid_w < canvas_w + 5:
        return None

    canvas_top = random.randint(2, grid_h - canvas_h - 2)
    canvas_left = random.randint(2, grid_w - canvas_w - 2)
    canvas_bbox = (canvas_top, canvas_left, canvas_top + canvas_h - 1, canvas_left + canvas_w - 1)

    placements: list[list[tuple[tuple[int, int], int]]] = [[] for _ in range(num_stamps)]
    blocked_cells: set[tuple[int, int]] = set()
    blocked_boxes: list[tuple[int, int, int, int]] = []

    if force_conflict:
        pair_choices = [
            (i, j)
            for i in range(num_stamps)
            for j in range(i + 1, num_stamps)
            if int(masks[i].sum()) != int(masks[j].sum())
        ]
        if not pair_choices:
            return None
        i, j = random.choice(pair_choices)
        found = _find_conflict_pair(masks[i], masks[j], canvas_h, canvas_w)
        if found is None:
            return None
        anchor_i, coords_i, box_i, anchor_j, coords_j, box_j = found
        shared_color = random.choice(canvas_palette)
        placements[i].append((anchor_i, int(shared_color)))
        placements[j].append((anchor_j, int(shared_color)))
        blocked_cells |= coords_i | coords_j
        blocked_boxes.extend([box_i, box_j])

    for idx, mask in enumerate(masks):
        target_count = random.choice([1, 2])
        if placements[idx]:
            target_count = 1
        options = _anchor_options(mask, canvas_h, canvas_w)
        for _ in range(target_count):
            picked = _pick_disjoint_option(options, blocked_cells, blocked_boxes)
            if picked is None:
                if not placements[idx]:
                    return None
                break
            anchor, coords, box = picked
            placements[idx].append((anchor, int(random.choice(canvas_palette))))
            blocked_cells |= coords
            blocked_boxes.append(box)

    canvas = _base_canvas(canvas_h, canvas_w, canvas_palette)
    for idx, mask in enumerate(masks):
        rel_cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        for (top, left), color in placements[idx]:
            for dr, dc in rel_cells:
                canvas[top + dr, left + dc] = int(color)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[canvas_top : canvas_top + canvas_h, canvas_left : canvas_left + canvas_w] = canvas

    if not _place_stamps_outside(grid, canvas_bbox, stamp_arrays):
        return None

    out, actual_canvas, stamps, coverers = _simulate(grid)
    if int(actual_canvas["area"]) != int(canvas_h * canvas_w):
        return None
    if len(stamps) != num_stamps:
        return None
    if any(len(stamp["matches"]) == 0 for stamp in stamps):
        return None

    changed = np.argwhere(out != grid)
    if changed.size == 0:
        return None
    if int(changed.shape[0]) < num_stamps + 2:
        return None
    if int(changed.shape[0]) > int(0.45 * canvas_h * canvas_w):
        return None

    total_matches = sum(len(stamp["matches"]) for stamp in stamps)
    if total_matches > num_stamps + 4:
        return None

    stamp_color_hits = {int(stamp["color"]): 0 for stamp in stamps}
    for r, c in changed:
        color = int(out[int(r), int(c)])
        if color in stamp_color_hits:
            stamp_color_hits[color] += 1
    if any(count == 0 for count in stamp_color_hits.values()):
        return None

    distinct_overlap = 0
    priority_overlap = 0
    for entries in coverers.values():
        stamp_ids = {stamp_id for stamp_id, _area in entries}
        if len(stamp_ids) >= 2:
            distinct_overlap += 1
            areas = {area for _stamp_id, area in entries}
            if len(areas) >= 2:
                priority_overlap += 1
    if force_conflict:
        if priority_overlap == 0:
            return None
    else:
        if distinct_overlap != 0:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated retries")
