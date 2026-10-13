import random
from collections import Counter

import numpy as np


BG = 0
TEMPLATE = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _coords_to_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[int, ...]:
    flat = tuple(int(v) for v in mask.astype(int).ravel())
    return (int(mask.shape[0]), int(mask.shape[1]), *flat)


def _canonical_signature(mask: np.ndarray) -> tuple[int, ...]:
    return min(_mask_signature(np.rot90(mask, k)) for k in range(4))


def _touches_bbox_edge(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _border_background(g: np.ndarray) -> int:
    border = np.concatenate([g[0], g[-1], g[1:-1, 0], g[1:-1, -1]])
    return int(Counter(border.tolist()).most_common(1)[0][0])


def _side_run(mask: np.ndarray, side: str) -> int:
    if side == "top":
        line = mask[0, :]
    elif side == "bottom":
        line = mask[-1, :]
    elif side == "left":
        line = mask[:, 0]
    else:
        line = mask[:, -1]

    best = cur = 0
    for value in line.tolist():
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _marker_sides(mask: np.ndarray) -> list[str]:
    sides = ["top", "bottom", "left", "right"]
    runs = {side: _side_run(mask, side) for side in sides}
    good = [side for side, run in runs.items() if run >= 2]
    if good:
        return good
    return [side for side, run in runs.items() if run >= 1]


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _place_patch(
    grid: np.ndarray,
    occ: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w][patch != BG] = patch[patch != BG]
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(grid.shape[0], top + h + 1)
    c1 = min(grid.shape[1], left + w + 1)
    occ[r0:r1, c0:c1] = True


SHAPES = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit scattered fragments into template holes by shape, then recolor each filled hole from its border marker.

    Input structure:
    - The background is black-like and the largest foreground object is a grey rectangular template near the center.
    - Inside the template are one or more enclosed background holes; on one side of each hole, part of the grey border is replaced by a unique non-grey marker color.
    - Outside the template are separate 4-connected colored fragments whose shapes match the holes up to 0/90/180/270 degree rotation.

    Rule:
    - Identify the template, collect its enclosed background holes, and read each hole's fill color from the unique non-grey border segment adjacent to that hole.
    - Find the external fragments and match them to holes by rotation-invariant shape, without using fragment color as the output color.
    - Erase all external fragments to background, then fill each hole with its own border-marker color while keeping the grey template and colored border markers unchanged.

    Key insight:
    - The fragment-to-hole correspondence is determined only by shape under rotation, but the final fill color comes from the hole's border marker rather than from the fragment itself.
    """
    g = np.asarray(grid, dtype=int)
    bg = _border_background(g)

    comps = _components(g != bg)
    assert comps, "expected at least one foreground component"
    template_coords = max(comps, key=lambda comp: (len(comp), (_bbox(comp)[2] - _bbox(comp)[0] + 1) * (_bbox(comp)[3] - _bbox(comp)[1] + 1)))
    template_mask = np.zeros_like(g, dtype=bool)
    for r, c in template_coords:
        template_mask[r, c] = True

    template_color = int(Counter(g[template_mask].tolist()).most_common(1)[0][0])
    tr0, tc0, tr1, tc1 = _bbox(template_coords)
    sub = g[tr0 : tr1 + 1, tc0 : tc1 + 1]

    hole_infos = []
    for comp in _components(sub == bg):
        if _touches_bbox_edge(comp, sub.shape):
            continue
        hole_mask = _coords_to_mask(comp)
        marker_colors = Counter()
        for hr, hc in comp:
            for nr, nc in _neighbors4(hr, hc):
                if not (0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1]):
                    continue
                val = int(sub[nr, nc])
                if val != bg and val != template_color:
                    marker_colors[val] += 1
        assert marker_colors, "each hole must have a non-grey border marker"
        marker_color = int(marker_colors.most_common(1)[0][0])
        hole_infos.append(
            {
                "coords": [(tr0 + r, tc0 + c) for r, c in comp],
                "mask": hole_mask,
                "sig": _canonical_signature(hole_mask),
                "marker": marker_color,
            }
        )

    fragment_mask = (g != bg) & ~template_mask
    fragment_sigs = Counter(_canonical_signature(_coords_to_mask(comp)) for comp in _components(fragment_mask))
    hole_sigs = Counter(info["sig"] for info in hole_infos)
    assert fragment_sigs == hole_sigs, "external fragments must match hole shapes up to rotation"

    out = g.copy()
    out[fragment_mask] = bg
    for info in hole_infos:
        for r, c in info["coords"]:
            out[r, c] = int(info["marker"])
    return out


def _draw_template(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
) -> None:
    grid[top : top + height, left : left + width] = TEMPLATE


def _paint_hole(
    grid: np.ndarray,
    hole_mask: np.ndarray,
    top: int,
    left: int,
    marker_color: int,
    side: str,
) -> list[tuple[int, int]]:
    h, w = hole_mask.shape
    carved = []
    for r in range(h):
        for c in range(w):
            if hole_mask[r, c]:
                grid[top + r, left + c] = BG
                carved.append((top + r, left + c))

    if side == "top":
        border_cells = [(top - 1, left + c) for c in range(w) if hole_mask[0, c]]
    elif side == "bottom":
        border_cells = [(top + h, left + c) for c in range(w) if hole_mask[h - 1, c]]
    elif side == "left":
        border_cells = [(top + r, left - 1) for r in range(h) if hole_mask[r, 0]]
    else:
        border_cells = [(top + r, left + w) for r in range(h) if hole_mask[r, w - 1]]

    for r, c in border_cells:
        grid[r, c] = int(marker_color)
    return carved


def _expected_output(
    grid: np.ndarray,
    holes: list[tuple[list[tuple[int, int]], int]],
    fragments: list[tuple[np.ndarray, int, int]],
) -> np.ndarray:
    out = grid.copy()
    for patch, top, left in fragments:
        h, w = patch.shape
        out[top : top + h, left : left + w][patch != BG] = BG
    for coords, marker in holes:
        for r, c in coords:
            out[r, c] = int(marker)
    return out


def _gen_attempt() -> np.ndarray | None:
    num_holes = random.choice([2, 3, 4])
    shape_indices = random.sample(range(len(SHAPES)), num_holes)
    hole_masks = [SHAPES[idx].copy() for idx in shape_indices]

    layout_options = {2: [(1, 2), (2, 1)], 3: [(1, 3), (3, 1), (2, 2)], 4: [(2, 2)]}
    rows, cols = random.choice(layout_options[num_holes])

    max_h = max(mask.shape[0] for mask in hole_masks)
    max_w = max(mask.shape[1] for mask in hole_masks)
    slot_h = max_h + random.choice([3, 4])
    slot_w = max_w + random.choice([3, 4])
    gap = random.choice([1, 2])
    template_h = 2 + rows * slot_h + (rows - 1) * gap
    template_w = 2 + cols * slot_w + (cols - 1) * gap

    min_h = max(18, template_h + 8)
    min_w = max(18, template_w + 8)
    if min_h > 30 or min_w > 30:
        return None

    max_h_total = min(30, template_h + 12)
    max_w_total = min(30, template_w + 12)
    H = random.randint(min_h, max_h_total)
    W = random.randint(min_w, max_w_total)

    grid = np.full((H, W), BG, dtype=int)
    top = random.randint(2, H - template_h - 2)
    left = random.randint(2, W - template_w - 2)
    _draw_template(grid, top, left, template_h, template_w)

    colors = random.sample(COLOR_POOL, 2 * num_holes)
    marker_colors = colors[:num_holes]
    piece_colors = colors[num_holes:]

    slots = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(slots)
    slots = slots[:num_holes]
    random.shuffle(hole_masks)

    hole_infos: list[tuple[list[tuple[int, int]], int]] = []
    piece_shapes: list[np.ndarray] = []

    for idx, hole_mask in enumerate(hole_masks):
        slot_r, slot_c = slots[idx]
        cell_top = top + 1 + slot_r * (slot_h + gap)
        cell_left = left + 1 + slot_c * (slot_w + gap)
        h, w = hole_mask.shape

        hole_top = cell_top + random.randint(1, slot_h - h - 1)
        hole_left = cell_left + random.randint(1, slot_w - w - 1)
        marker_side = random.choice(_marker_sides(hole_mask))
        carved = _paint_hole(grid, hole_mask, hole_top, hole_left, marker_colors[idx], marker_side)
        hole_infos.append((carved, marker_colors[idx]))
        piece_shapes.append(hole_mask)

    occ = np.zeros((H, W), dtype=bool)
    occ[max(0, top - 1) : min(H, top + template_h + 1), max(0, left - 1) : min(W, left + template_w + 1)] = True

    fragment_infos: list[tuple[np.ndarray, int, int]] = []
    for idx, hole_mask in enumerate(piece_shapes):
        rot = random.randrange(4)
        piece_mask = np.rot90(hole_mask, rot)
        patch = _mask_to_patch(piece_mask, piece_colors[idx])
        ph, pw = patch.shape
        placed = False
        for _ in range(250):
            pr = random.randint(1, H - ph - 2)
            pc = random.randint(1, W - pw - 2)
            if occ[pr - 1 : pr + ph + 1, pc - 1 : pc + pw + 1].any():
                continue
            _place_patch(grid, occ, patch, pr, pc)
            fragment_infos.append((patch, pr, pc))
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, hole_infos, fragment_infos)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
