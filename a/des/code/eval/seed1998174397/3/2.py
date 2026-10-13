import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300
MASK_RETRIES = 200


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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_and_mask(coords: list[tuple[int, int]]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    pts = np.array(coords, dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[pts[:, 0] - r0, pts[:, 1] - c0] = True
    return (int(r0), int(c0), int(r1), int(c1)), mask


def _rotate_pos(shape: tuple[int, int], pos: tuple[int, int], k: int) -> tuple[int, int]:
    marker = np.zeros(shape, dtype=int)
    marker[pos] = 1
    rr, cc = np.argwhere(np.rot90(marker, k) == 1)[0]
    return int(rr), int(cc)


def _marker_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        sides = 0
        if r == 0 and 0 < c < w - 1:
            sides += 1
        if r == h - 1 and 0 < c < w - 1:
            sides += 1
        if c == 0 and 0 < r < h - 1:
            sides += 1
        if c == w - 1 and 0 < r < h - 1:
            sides += 1
        if sides == 1:
            candidates.append((int(r), int(c)))
    return candidates


def _find_template(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros(sub.shape, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == color):
            continue
        if not np.any(sub != color):
            continue
        area = int(sub.shape[0] * sub.shape[1])
        candidates.append((area, int(coords.shape[0]), color, (int(r0), int(c0), int(r1), int(c1))))
    assert candidates, "template rectangle not found"
    candidates.sort(reverse=True)
    _area, _count, color, bbox = candidates[0]
    return color, bbox


def _extract_cavities(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    fill_color: int,
    bg: int,
) -> list[dict]:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    comps = _components(sub != fill_color)
    cavities: list[dict] = []
    for comp in comps:
        if any(rr == 0 or cc == 0 or rr == sub.shape[0] - 1 or cc == sub.shape[1] - 1 for rr, cc in comp):
            continue
        local_bbox, mask = _bbox_and_mask(comp)
        cr0, cc0, cr1, cc1 = local_bbox
        marker_pixels = [(rr, cc) for rr, cc in comp if int(sub[rr, cc]) != bg]
        assert len(marker_pixels) == 1, "each cavity must contain exactly one marker"
        mr, mc = marker_pixels[0]
        marker_pos = (int(mr - cr0), int(mc - cc0))
        cavities.append(
            {
                "bbox": (r0 + cr0, c0 + cc0, r0 + cr1, c0 + cc1),
                "mask": mask,
                "marker_color": int(sub[mr, mc]),
                "marker_pos": marker_pos,
            }
        )
    return cavities


def _extract_fragments(
    grid: np.ndarray,
    template_bbox: tuple[int, int, int, int],
    bg: int,
) -> list[dict]:
    mask = grid != bg
    r0, c0, r1, c1 = template_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = _components(mask)
    fragments: list[dict] = []
    for comp in comps:
        bbox, shape_mask = _bbox_and_mask(comp)
        fr0, fc0, _fr1, _fc1 = bbox
        colors = Counter(int(grid[rr, cc]) for rr, cc in comp)
        marker_colors = [color for color, count in colors.items() if count == 1]
        assert len(colors) == 2 and len(marker_colors) == 1, "fragment must have one marker pixel"
        marker_color = int(marker_colors[0])
        body_color = next(int(color) for color in colors if color != marker_color)
        marker_global = next((rr, cc) for rr, cc in comp if int(grid[rr, cc]) == marker_color)
        fragments.append(
            {
                "coords": comp,
                "bbox": bbox,
                "mask": shape_mask,
                "marker_color": marker_color,
                "marker_pos": (int(marker_global[0] - fr0), int(marker_global[1] - fc0)),
                "body_color": body_color,
            }
        )
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered fragments into the holed template using the marker orientation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    template_color, template_bbox = _find_template(g, bg)
    cavities = _extract_cavities(g, template_bbox, template_color, bg)
    fragments = _extract_fragments(g, template_bbox, bg)

    assert len(cavities) == len(fragments) and cavities, "fragment/cavity count mismatch"

    out = g.copy()
    for fragment in fragments:
        for r, c in fragment["coords"]:
            out[r, c] = bg

    used: set[int] = set()
    for cavity in cavities:
        matches: list[int] = []
        for idx, fragment in enumerate(fragments):
            if idx in used or fragment["marker_color"] != cavity["marker_color"]:
                continue
            for k in range(4):
                rot_mask = np.rot90(fragment["mask"], k)
                if rot_mask.shape != cavity["mask"].shape:
                    continue
                if not np.array_equal(rot_mask, cavity["mask"]):
                    continue
                rot_marker = _rotate_pos(fragment["mask"].shape, fragment["marker_pos"], k)
                if rot_marker == cavity["marker_pos"]:
                    matches.append(idx)
                    break
        assert len(matches) == 1, "expected a unique fragment match"
        match = fragments[matches[0]]
        used.add(matches[0])
        r0, c0, r1, c1 = cavity["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[cavity["mask"]] = match["body_color"]

    return out


def _random_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int], rng: random.Random) -> None:
    r, c = start
    tr, tc = end
    mask[r, c] = True
    while (r, c) != (tr, tc):
        choices = []
        if r < tr:
            choices.append((1, 0))
        elif r > tr:
            choices.append((-1, 0))
        if c < tc:
            choices.append((0, 1))
        elif c > tc:
            choices.append((0, -1))
        dr, dc = rng.choice(choices)
        r += dr
        c += dc
        mask[r, c] = True


def _random_shape(rng: random.Random) -> np.ndarray | None:
    for _ in range(MASK_RETRIES):
        h = rng.randint(3, 6)
        w = rng.randint(3, 6)
        center = (rng.randrange(1, h - 1), rng.randrange(1, w - 1))
        mask = np.zeros((h, w), dtype=bool)
        targets = [
            (0, rng.randrange(w)),
            (h - 1, rng.randrange(w)),
            (rng.randrange(h), 0),
            (rng.randrange(h), w - 1),
        ]
        for target in targets:
            _random_path(mask, center, target, rng)

        min_area = max(5, int(mask.sum()))
        max_area = min(h * w - 1, max(min_area, int(np.ceil(0.72 * h * w))))
        if min_area > max_area:
            continue
        target_area = rng.randint(min_area, max_area)

        while int(mask.sum()) < target_area:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                for dr, dc in DIR4:
                    nr, nc = int(r) + dr, int(c) + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            mask[rng.choice(tuple(frontier))] = True

        if int(mask.sum()) >= h * w:
            continue
        if not _marker_candidates(mask):
            continue
        return mask
    return None


def _find_positions(
    occupied: np.ndarray,
    shape: tuple[int, int],
    row_min: int,
    row_max_inclusive: int,
    col_min: int,
    col_max_inclusive: int,
) -> list[tuple[int, int]]:
    h, w = shape
    H, W = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(row_min, row_max_inclusive + 1):
        for c0 in range(col_min, col_max_inclusive + 1):
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(H, r0 + h + 1)
            cc1 = min(W, c0 + w + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            positions.append((r0, c0))
    return positions


def _place_patch(
    grid: np.ndarray,
    top_left: tuple[int, int],
    mask: np.ndarray,
    body_color: int,
    marker_pos: tuple[int, int] | None = None,
    marker_color: int | None = None,
) -> None:
    r0, c0 = top_left
    patch = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    patch[mask] = body_color
    if marker_pos is not None and marker_color is not None:
        patch[marker_pos] = marker_color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_parts = rng.choice([2, 3, 4])
    bg = rng.randrange(10)
    remaining = [color for color in range(10) if color != bg]
    rng.shuffle(remaining)
    template_color = remaining.pop()
    marker_colors = [remaining.pop() for _ in range(n_parts)]
    body_colors = [remaining.pop() for _ in range(n_parts)]

    specs: list[dict] = []
    base_shapes: list[np.ndarray] = []
    for idx in range(n_parts):
        if base_shapes and rng.random() < 0.35:
            base_mask = base_shapes[rng.randrange(len(base_shapes))].copy()
        else:
            base_mask = _random_shape(rng)
            if base_mask is None:
                return None
            base_shapes.append(base_mask.copy())
        candidates = _marker_candidates(base_mask)
        if not candidates:
            return None
        base_marker = rng.choice(candidates)
        hole_rot = rng.randrange(4)
        frag_rot = rng.randrange(4)
        hole_mask = np.rot90(base_mask, hole_rot)
        frag_mask = np.rot90(base_mask, frag_rot)
        specs.append(
            {
                "idx": idx,
                "marker_color": marker_colors[idx],
                "body_color": body_colors[idx],
                "hole_mask": hole_mask,
                "hole_marker": _rotate_pos(base_mask.shape, base_marker, hole_rot),
                "frag_mask": frag_mask,
                "frag_marker": _rotate_pos(base_mask.shape, base_marker, frag_rot),
            }
        )

    max_hole_h = max(spec["hole_mask"].shape[0] for spec in specs)
    max_hole_w = max(spec["hole_mask"].shape[1] for spec in specs)
    min_template_h = max(8, max_hole_h + 4)
    min_template_w = max(10, max_hole_w + 4)

    hole_box_area = sum((spec["hole_mask"].shape[0] + 2) * (spec["hole_mask"].shape[1] + 2) for spec in specs)
    frag_box_area = sum((spec["frag_mask"].shape[0] + 2) * (spec["frag_mask"].shape[1] + 2) for spec in specs)

    H = rng.randint(20, 30)
    W = rng.randint(20, 30)
    max_template_h = min(17, H - 3)
    max_template_w = min(19, W - 3)
    if min_template_h > max_template_h or min_template_w > max_template_w:
        return None

    th = rng.randint(min_template_h, max_template_h)
    tw = rng.randint(min_template_w, max_template_w)
    if th * tw < hole_box_area + 8:
        return None
    if H * W - th * tw < frag_box_area + 16:
        return None

    tr0 = rng.randint(1, H - th - 1)
    tc0 = rng.randint(1, W - tw - 1)

    hole_occ = np.zeros((th, tw), dtype=bool)
    order = sorted(range(n_parts), key=lambda i: specs[i]["hole_mask"].shape[0] * specs[i]["hole_mask"].shape[1], reverse=True)
    for idx in order:
        h, w = specs[idx]["hole_mask"].shape
        positions = _find_positions(hole_occ, (h, w), 1, th - h - 1, 1, tw - w - 1)
        if not positions:
            return None
        hr, hc = rng.choice(positions)
        specs[idx]["hole_pos"] = (hr, hc)
        hole_occ[max(0, hr - 1) : min(th, hr + h + 1), max(0, hc - 1) : min(tw, hc + w + 1)] = True

    occ = np.zeros((H, W), dtype=bool)
    occ[max(0, tr0 - 1) : min(H, tr0 + th + 1), max(0, tc0 - 1) : min(W, tc0 + tw + 1)] = True
    frag_order = sorted(range(n_parts), key=lambda i: specs[i]["frag_mask"].shape[0] * specs[i]["frag_mask"].shape[1], reverse=True)
    for idx in frag_order:
        h, w = specs[idx]["frag_mask"].shape
        positions = _find_positions(occ, (h, w), 1, H - h - 1, 1, W - w - 1)
        if not positions:
            return None
        fr, fc = rng.choice(positions)
        specs[idx]["frag_pos"] = (fr, fc)
        occ[max(0, fr - 1) : min(H, fr + h + 1), max(0, fc - 1) : min(W, fc + w + 1)] = True

    target = np.full((H, W), bg, dtype=int)
    target[tr0 : tr0 + th, tc0 : tc0 + tw] = template_color
    for spec in specs:
        hr, hc = spec["hole_pos"]
        patch = target[tr0 + hr : tr0 + hr + spec["hole_mask"].shape[0], tc0 + hc : tc0 + hc + spec["hole_mask"].shape[1]]
        patch[spec["hole_mask"]] = spec["body_color"]

    inp = target.copy()
    for spec in specs:
        hr, hc = spec["hole_pos"]
        patch = inp[tr0 + hr : tr0 + hr + spec["hole_mask"].shape[0], tc0 + hc : tc0 + hc + spec["hole_mask"].shape[1]]
        patch[spec["hole_mask"]] = bg
        patch[spec["hole_marker"]] = spec["marker_color"]

    for spec in specs:
        _place_patch(inp, spec["frag_pos"], spec["frag_mask"], spec["body_color"], spec["frag_marker"], spec["marker_color"])

    if _get_bg(inp) != bg:
        return None
    if not np.array_equal(main(inp), target):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
