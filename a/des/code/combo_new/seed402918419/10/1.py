from __future__ import annotations

import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 300

FRAG_TEMPLATES = {
    1: (
        np.array([[1]], dtype=bool),
    ),
    2: (
        np.array([[1, 1]], dtype=bool),
        np.array([[1], [1]], dtype=bool),
    ),
    3: (
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1], [1], [1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[0, 1], [1, 1]], dtype=bool),
    ),
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _edge_mask(shape: tuple[int, int], side: str) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        mask[0, :] = True
    elif side == "bottom":
        mask[-1, :] = True
    elif side == "left":
        mask[:, 0] = True
    else:
        mask[:, -1] = True
    return mask


def _fill_cavity(out: np.ndarray, cavity_bbox: tuple[int, int, int, int], side: str, color: int, count: int) -> None:
    r0, c0, r1, c1 = cavity_bbox
    limit = min(int(count), int((r1 - r0 + 1) * (c1 - c0 + 1)))
    filled = 0
    if side == "top":
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if filled >= limit:
                    return
                out[r, c] = int(color)
                filled += 1
    elif side == "bottom":
        for r in range(r1, r0 - 1, -1):
            for c in range(c0, c1 + 1):
                if filled >= limit:
                    return
                out[r, c] = int(color)
                filled += 1
    elif side == "left":
        for c in range(c0, c1 + 1):
            for r in range(r0, r1 + 1):
                if filled >= limit:
                    return
                out[r, c] = int(color)
                filled += 1
    else:
        for c in range(c1, c0 - 1, -1):
            for r in range(r0, r1 + 1):
                if filled >= limit:
                    return
                out[r, c] = int(color)
                filled += 1


def _find_anchors(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    anchors: list[dict] = []
    anchor_mask = np.zeros_like(grid, dtype=bool)
    for coords in _components(grid != 0):
        r0, c0, r1, c1 = _bbox(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        occ = np.zeros((h, w), dtype=bool)
        occ[coords[:, 0] - r0, coords[:, 1] - c0] = True

        if not (bool(occ[0, :].all()) and bool(occ[-1, :].all()) and bool(occ[:, 0].all()) and bool(occ[:, -1].all())):
            continue

        hole = ~occ
        if not bool(hole.any()):
            continue
        hole_coords = np.argwhere(hole)
        hr0 = int(hole_coords[:, 0].min())
        hc0 = int(hole_coords[:, 1].min())
        hr1 = int(hole_coords[:, 0].max())
        hc1 = int(hole_coords[:, 1].max())
        if hr0 == 0 or hc0 == 0 or hr1 == h - 1 or hc1 == w - 1:
            continue
        if int(hole.sum()) != int((hr1 - hr0 + 1) * (hc1 - hc0 + 1)):
            continue

        colors, counts = np.unique(sub[occ], return_counts=True)
        if len(colors) != 2:
            continue
        order = np.argsort(-counts)
        border_color = int(colors[int(order[0])])
        stripe_color = int(colors[int(order[1])])

        stripe_sides = [side for side in SIDES if bool(np.all(sub[_edge_mask((h, w), side)] == stripe_color))]
        if len(stripe_sides) != 1:
            continue
        side = stripe_sides[0]
        stripe_mask = _edge_mask((h, w), side)
        if not bool(np.all(sub[stripe_mask] == stripe_color)):
            continue
        if not bool(np.all(sub[occ & ~stripe_mask] == border_color)):
            continue

        anchors.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "coords": coords,
                "border_color": int(border_color),
                "stripe_color": int(stripe_color),
                "side": side,
                "cavity_bbox": (int(r0 + hr0), int(c0 + hc0), int(r0 + hr1), int(c0 + hc1)),
            }
        )
        anchor_mask[coords[:, 0], coords[:, 1]] = True

    anchors.sort(key=lambda anchor: (anchor["bbox"][0], anchor["bbox"][1]))
    return anchors, anchor_mask


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    anchors, anchor_mask = _find_anchors(g)
    assert anchors, "main: no anchors detected"

    out = g.copy()
    external = (g != 0) & ~anchor_mask
    out[external] = 0

    fragment_counts: dict[int, int] = {}
    if bool(external.any()):
        colors, counts = np.unique(g[external], return_counts=True)
        fragment_counts = {int(color): int(count) for color, count in zip(colors, counts)}

    for anchor in anchors:
        cavity_bbox = anchor["cavity_bbox"]
        r0, c0, r1, c1 = cavity_bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = 0
        _fill_cavity(
            out,
            cavity_bbox,
            anchor["side"],
            anchor["border_color"],
            fragment_counts.get(int(anchor["border_color"]), 0),
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collect same-color fragments into hollow anchor blocks, using the border stripe to choose the stacking direction.

    Input structure:
    - The background is black.
    - Several multicolor rectangular anchor frames appear; each anchor has one main border color, one full-edge stripe color on exactly one side, and a rectangular black cavity inside.
    - Small 1-3-cell fragments of the border colors are scattered outside the anchors.

    Rule:
    - Detect each anchor frame and read the side carrying the stripe as the fill direction.
    - Count how many external fragment pixels share that anchor's main border color, then erase all external fragments.
    - Refill the anchor's cavity with that many border-color pixels, packed from the striped side row-by-row for top/bottom stripes or column-by-column for left/right stripes.

    Key insight:
    - The stripe does not change the target color; it only encodes which side of the cavity acts as the starting edge for stacking the absorbed fragment pixels.
    """
    return _solve(np.asarray(grid, dtype=int))


def _build_anchor_patch(
    cavity_h: int, cavity_w: int, thickness: int, border_color: int, stripe_color: int, side: str
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h = int(cavity_h + 2 * thickness)
    w = int(cavity_w + 2 * thickness)
    patch = np.full((h, w), int(border_color), dtype=int)
    patch[thickness : h - thickness, thickness : w - thickness] = 0
    if side == "top":
        patch[0, :] = int(stripe_color)
    elif side == "bottom":
        patch[-1, :] = int(stripe_color)
    elif side == "left":
        patch[:, 0] = int(stripe_color)
    else:
        patch[:, -1] = int(stripe_color)
    return patch, (int(thickness), int(thickness), int(h - thickness - 1), int(w - thickness - 1))


def _partition_total(total: int, rng: random.Random) -> list[int]:
    min_parts = int((total + 2) // 3)
    max_parts = int(min(total, 6))
    num_parts = int(rng.randint(min_parts, max_parts))
    parts = [1] * num_parts
    remaining = int(total - num_parts)
    while remaining > 0:
        choices = [idx for idx, value in enumerate(parts) if value < 3]
        idx = int(rng.choice(choices))
        parts[idx] += 1
        remaining -= 1
    rng.shuffle(parts)
    return parts


def _random_fragment_shape(area: int, rng: random.Random) -> np.ndarray:
    return np.array(rng.choice(FRAG_TEMPLATES[int(area)]), dtype=bool)


def _can_place(reserved: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = reserved.shape
    if top < 1 or left < 1 or top + height > h - 1 or left + width > w - 1:
        return False
    r0 = max(0, int(top - pad))
    c0 = max(0, int(left - pad))
    r1 = min(h, int(top + height + pad))
    c1 = min(w, int(left + width + pad))
    return not bool(reserved[r0:r1, c0:c1].any())


def _random_position(reserved: np.ndarray, shape: tuple[int, int], rng: random.Random) -> tuple[int, int] | None:
    height, width = shape
    h, w = reserved.shape
    positions: list[tuple[int, int]] = []
    for top in range(1, h - height):
        for left in range(1, w - width):
            if _can_place(reserved, int(top), int(left), int(height), int(width)):
                positions.append((int(top), int(left)))
    if not positions:
        return None
    return rng.choice(positions)


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    mask = patch != 0
    region[mask] = patch[mask]


def _build_expected(grid: np.ndarray, anchors: list[dict], fragments: list[dict]) -> np.ndarray:
    out = grid.copy()
    for fragment in fragments:
        top = fragment["top"]
        left = fragment["left"]
        shape = fragment["shape"]
        out[top : top + shape.shape[0], left : left + shape.shape[1]][shape] = 0
    for anchor in anchors:
        r0, c0, r1, c1 = anchor["cavity_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = 0
        _fill_cavity(out, anchor["cavity_bbox"], anchor["side"], anchor["border_color"], anchor["fill_count"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_anchors = int(rng.randint(2, 4))
    palette = list(range(1, 10))
    rng.shuffle(palette)
    border_colors = palette[:num_anchors]
    stripe_colors = palette[num_anchors : 2 * num_anchors]
    sides = rng.sample(list(SIDES), k=num_anchors)

    anchors: list[dict] = []
    load = 0
    for idx in range(num_anchors):
        thickness = int(rng.choice((1, 2)))
        cavity_h = int(rng.randint(1 if rng.random() < 0.25 else 2, 5))
        cavity_w = int(rng.randint(1 if rng.random() < 0.25 else 2, 5))
        patch, cavity_rel = _build_anchor_patch(
            cavity_h,
            cavity_w,
            thickness,
            int(border_colors[idx]),
            int(stripe_colors[idx]),
            sides[idx],
        )
        cavity_area = int(cavity_h * cavity_w)
        fill_cap = min(cavity_area, 15)
        fill_total = int(fill_cap if rng.random() < 0.2 else rng.randint(1, fill_cap))
        fragment_shapes = [_random_fragment_shape(area, rng) for area in _partition_total(fill_total, rng)]
        anchors.append(
            {
                "patch": patch,
                "cavity_rel": cavity_rel,
                "border_color": int(border_colors[idx]),
                "stripe_color": int(stripe_colors[idx]),
                "side": sides[idx],
                "fill_count": int(fill_total),
                "fragment_shapes": fragment_shapes,
            }
        )
        load += int(patch.shape[0] * patch.shape[1])
        load += sum(int(shape.shape[0] * shape.shape[1]) for shape in fragment_shapes)

    min_dim = max(14, int(np.ceil(np.sqrt(load * 2.4))))
    if min_dim > 30:
        return None
    grid_h = int(rng.randint(min_dim, 30))
    grid_w = int(rng.randint(min_dim, 30))
    grid = np.zeros((grid_h, grid_w), dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    for idx in sorted(range(num_anchors), key=lambda i: anchors[i]["patch"].shape[0] * anchors[i]["patch"].shape[1], reverse=True):
        anchor = anchors[idx]
        patch = anchor["patch"]
        pos = _random_position(reserved, patch.shape, rng)
        if pos is None:
            return None
        top, left = pos
        _paint_patch(grid, patch, top, left)
        reserved[top : top + patch.shape[0], left : left + patch.shape[1]] = True
        cr0, cc0, cr1, cc1 = anchor["cavity_rel"]
        anchor["bbox"] = (int(top), int(left), int(top + patch.shape[0] - 1), int(left + patch.shape[1] - 1))
        anchor["cavity_bbox"] = (int(top + cr0), int(left + cc0), int(top + cr1), int(left + cc1))

    fragments: list[dict] = []
    placement_order: list[tuple[int, np.ndarray]] = []
    for anchor_idx, anchor in enumerate(anchors):
        for shape in anchor["fragment_shapes"]:
            placement_order.append((int(anchor_idx), np.array(shape, dtype=bool)))
    rng.shuffle(placement_order)

    for anchor_idx, shape in placement_order:
        pos = _random_position(reserved, shape.shape, rng)
        if pos is None:
            return None
        top, left = pos
        color = int(anchors[anchor_idx]["border_color"])
        patch = np.where(shape, color, 0).astype(int)
        _paint_patch(grid, patch, top, left)
        reserved[top : top + shape.shape[0], left : left + shape.shape[1]] = True
        fragments.append(
            {
                "anchor_idx": int(anchor_idx),
                "top": int(top),
                "left": int(left),
                "shape": shape,
            }
        )

    expected = _build_expected(grid, anchors, fragments)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
