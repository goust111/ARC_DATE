from __future__ import annotations

import random

import numpy as np


BG = 0
GEN_TEMPLATE_COLOR = 5
GEN_FRAGMENT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_MIN_PIECES = 2
GEN_MAX_PIECES = 4
GEN_SHAPE_RETRIES = 200
GEN_SAMPLE_RETRIES = 400
GEN_PLACEMENT_RETRIES = 200


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coord arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    """Normalize coords to a tight top-left boolean mask."""
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _mask_signature(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask.astype(bool))


def _canonical_signature(mask: np.ndarray) -> tuple[str, ...]:
    return min(_mask_signature(np.rot90(mask, k)) for k in range(4))


def _rotation_signature_count(mask: np.ndarray) -> int:
    return len({_mask_signature(np.rot90(mask, k)) for k in range(4)})


def _dominant_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values.astype(int), return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Half-open box overlap."""
    return max(int(a[0]), int(b[0])) < min(int(a[2]), int(b[2])) and max(int(a[1]), int(b[1])) < min(int(a[3]), int(b[3]))


def _anchor_candidates(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Return candidate anchor relations as (cell_r, cell_c, dr, dc).

    The candidate anchor position sits one step outside the mask and is adjacent
    to exactly one shape cell, which keeps the clue visually unambiguous.
    """
    h, w = mask.shape
    out: list[tuple[int, int, int, int]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr, dc in DIRS4:
            ar = r + dr
            ac = c + dc
            if 0 <= ar < h and 0 <= ac < w and bool(mask[ar, ac]):
                continue
            count = 0
            for dr2, dc2 in DIRS4:
                hr = ar + dr2
                hc = ac + dc2
                if 0 <= hr < h and 0 <= hc < w and bool(mask[hr, hc]):
                    count += 1
            if count == 1:
                out.append((r, c, int(dr), int(dc)))
    return out


def _random_shape() -> np.ndarray | None:
    """Generate a small asymmetric connected polyomino."""
    for _ in range(GEN_SHAPE_RETRIES):
        box_h = random.randint(2, 4)
        box_w = random.randint(2, 4)
        max_area = min(7, box_h * box_w - 1)
        if max_area < 4:
            continue
        area = random.randint(4, max_area)

        start = (random.randrange(box_h), random.randrange(box_w))
        cells = {start}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue

        raw = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            raw[r, c] = True
        mask = _mask_from_coords(np.argwhere(raw))
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if bool(mask.all()):
            continue
        if _rotation_signature_count(mask) < 4:
            continue
        if not _anchor_candidates(mask):
            continue
        return mask
    return None


def _unique_shapes(count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    seen: set[tuple[str, ...]] = set()
    for _ in range(GEN_SHAPE_RETRIES * count * 2):
        mask = _random_shape()
        if mask is None:
            continue
        sig = _canonical_signature(mask)
        if sig in seen:
            continue
        seen.add(sig)
        shapes.append(mask)
        if len(shapes) == count:
            return shapes
    return None


def _place_holes(
    template_h: int,
    template_w: int,
    hole_masks: list[np.ndarray],
    colors: list[int],
) -> tuple[np.ndarray, list[dict]] | None:
    template = np.full((template_h, template_w), GEN_TEMPLATE_COLOR, dtype=int)
    reserved: list[tuple[int, int, int, int]] = []
    holes: list[dict] = []

    order = sorted(range(len(hole_masks)), key=lambda i: int(hole_masks[i].sum()), reverse=True)
    for idx in order:
        mask = hole_masks[idx]
        color = int(colors[idx])
        hh, ww = mask.shape
        candidates = _anchor_candidates(mask)
        placed = False

        for _ in range(GEN_PLACEMENT_RETRIES):
            if template_h - hh - 2 < 0 or template_w - ww - 2 < 0:
                return None
            r0 = random.randint(1, template_h - hh - 1)
            c0 = random.randint(1, template_w - ww - 1)
            padded = (r0 - 1, c0 - 1, r0 + hh + 1, c0 + ww + 1)
            if any(_boxes_overlap(padded, other) for other in reserved):
                continue

            valid_anchors: list[tuple[int, int]] = []
            for cell_r, cell_c, dr, dc in candidates:
                ar = int(r0 + cell_r + dr)
                ac = int(c0 + cell_c + dc)
                if 0 <= ar < template_h and 0 <= ac < template_w:
                    valid_anchors.append((ar, ac))
            if not valid_anchors:
                continue

            abs_coords = np.argwhere(mask) + np.array([r0, c0], dtype=int)
            template[abs_coords[:, 0], abs_coords[:, 1]] = BG
            anchor = random.choice(valid_anchors)
            holes.append({"color": color, "coords": abs_coords, "anchor": anchor, "mask": mask})
            reserved.append(padded)
            placed = True
            break

        if not placed:
            return None

    for hole in holes:
        ar, ac = hole["anchor"]
        template[int(ar), int(ac)] = int(hole["color"])

    return template, holes


def _draw_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(mask)
    grid[top + coords[:, 0], left + coords[:, 1]] = int(color)


def _place_fragments(
    grid: np.ndarray,
    start_col: int,
    fragment_masks: list[np.ndarray],
    colors: list[int],
) -> list[dict] | None:
    h, w = grid.shape
    reserved: list[tuple[int, int, int, int]] = []
    placements: list[dict] = []

    order = sorted(range(len(fragment_masks)), key=lambda i: int(fragment_masks[i].sum()), reverse=True)
    for idx in order:
        mask = fragment_masks[idx]
        color = int(colors[idx])
        ph, pw = mask.shape
        if start_col >= w or pw > w - start_col:
            return None
        placed = False

        for _ in range(GEN_PLACEMENT_RETRIES):
            r0 = random.randint(0, h - ph)
            c0 = random.randint(start_col, w - pw)
            box = (max(0, r0 - 1), max(0, c0 - 1), min(h, r0 + ph + 1), min(w, c0 + pw + 1))
            if any(_boxes_overlap(box, other) for other in reserved):
                continue
            if np.any(grid[r0 : r0 + ph, c0 : c0 + pw][mask] != BG):
                continue
            _draw_mask(grid, mask, r0, c0, color)
            placements.append({"color": color, "mask": mask, "top": r0, "left": c0})
            reserved.append(box)
            placed = True
            break

        if not placed:
            return None

    return placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each scattered fragment into the matching template hole indicated by its same-color anchor.

    Input structure:
    - The background is black.
    - One large connected template occupies part of the grid; it is mostly one base color and may include a few non-base anchor pixels on its boundary.
    - Inside the template are black holes that are fully enclosed by the template's tight bounding box.
    - Elsewhere, each remaining non-black component is a monochrome fragment whose color matches exactly one anchor pixel on the template.

    Rule:
    - Identify the largest non-black connected component as the template and infer its base color as the dominant color inside that component.
    - Find every enclosed black hole inside the template bounding box; for each hole, read the unique non-base neighbor color on its boundary as that hole's anchor color.
    - Match each hole to the fragment of the same color, rotate the fragment by 0/90/180/270 degrees until its shape matches the hole, then fill the hole with that fragment color.
    - Erase the original scattered fragments to black, while keeping the template body and anchor pixels unchanged.

    Key insight:
    - The anchor color is the pairing key: it identifies which fragment belongs to which enclosed hole, and the fragment's correct rotation is the one whose occupied shape exactly coincides with that hole.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    non_bg = g != BG
    template_components = _components(non_bg)
    assert template_components, "expected at least one foreground component"

    template_coords = max(template_components, key=len)
    template_mask = np.zeros_like(non_bg, dtype=bool)
    template_mask[template_coords[:, 0], template_coords[:, 1]] = True
    template_color = _dominant_color(g[template_mask])

    tr0, tc0, tr1, tc1 = _bbox(template_coords)
    sub_grid = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    sub_template = template_mask[tr0 : tr1 + 1, tc0 : tc1 + 1]
    hole_candidates = (sub_grid == BG) & (~sub_template)

    holes: dict[int, dict] = {}
    for rel_coords in _components(hole_candidates):
        if _touches_boundary(rel_coords, hole_candidates.shape):
            continue
        abs_coords = rel_coords + np.array([tr0, tc0], dtype=int)
        boundary_anchors: dict[tuple[int, int], int] = {}
        for r, c in abs_coords:
            r = int(r)
            c = int(c)
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and bool(template_mask[nr, nc]):
                    color = int(g[nr, nc])
                    if color != template_color:
                        boundary_anchors[(nr, nc)] = color

        assert boundary_anchors, "each hole needs an anchor"
        anchor_colors = set(boundary_anchors.values())
        assert len(anchor_colors) == 1, "each hole must have exactly one anchor color"
        assert len(boundary_anchors) == 1, "each hole must have exactly one anchor pixel"
        anchor_color = next(iter(anchor_colors))
        hole_mask = _mask_from_coords(abs_coords)
        holes[anchor_color] = {"coords": abs_coords, "mask": hole_mask}

    fragment_mask = (g != BG) & (~template_mask)
    fragments: dict[int, np.ndarray] = {}
    for coords in _components(fragment_mask):
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "fragments must be monochrome"
        color = next(iter(colors))
        assert color != template_color, "fragment color must differ from the template color"
        assert color not in fragments, "each fragment color must be unique"
        fragments[color] = _mask_from_coords(coords)

    assert holes, "expected at least one enclosed hole"
    assert set(holes) == set(fragments), "anchor colors and fragment colors must match one-to-one"

    out = g.copy()
    out[fragment_mask] = BG

    for color, hole in holes.items():
        hole_mask = hole["mask"]
        piece_mask = fragments[color]
        matches = [
            np.rot90(piece_mask, k)
            for k in range(4)
            if np.rot90(piece_mask, k).shape == hole_mask.shape and np.array_equal(np.rot90(piece_mask, k), hole_mask)
        ]
        assert matches, "fragment must match its hole under some rotation"
        coords = hole["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)

    return out


def _gen_attempt() -> np.ndarray | None:
    piece_count = random.randint(GEN_MIN_PIECES, GEN_MAX_PIECES)
    colors = random.sample(list(GEN_FRAGMENT_COLORS), piece_count)

    base_shapes = _unique_shapes(piece_count)
    if base_shapes is None:
        return None

    hole_rots = [random.randrange(4) for _ in range(piece_count)]
    piece_rots = [random.randrange(4) for _ in range(piece_count)]
    needed_rots = [int((hole_rots[i] - piece_rots[i]) % 4) for i in range(piece_count)]
    if len(set(needed_rots)) < 2:
        return None

    hole_masks = [np.rot90(base_shapes[i], hole_rots[i]) for i in range(piece_count)]
    fragment_masks = [np.rot90(base_shapes[i], piece_rots[i]) for i in range(piece_count)]

    max_hole_h = max(mask.shape[0] for mask in hole_masks)
    max_hole_w = max(mask.shape[1] for mask in hole_masks)
    template_h = random.randint(max(9, max_hole_h + 4), 15)
    template_w = random.randint(max(9, max_hole_w + 4), 15)

    placed = _place_holes(template_h, template_w, hole_masks, colors)
    if placed is None:
        return None
    template_local, holes = placed

    template_top = random.randint(1, 2)
    template_left = random.randint(1, 2)
    gap = random.randint(3, 4)

    max_frag_h = max(mask.shape[0] for mask in fragment_masks)
    min_grid_h = max(template_top + template_h + 1, max_frag_h + 2, 14)
    max_grid_h = min(24, min_grid_h + 5)
    if min_grid_h > max_grid_h:
        return None
    grid_h = random.randint(min_grid_h, max_grid_h)

    min_grid_w = max(template_left + template_w + gap + 7, 20)
    max_grid_w = min(30, min_grid_w + 6)
    if min_grid_w > max_grid_w:
        return None
    grid_w = random.randint(min_grid_w, max_grid_w)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[template_top : template_top + template_h, template_left : template_left + template_w] = template_local

    fragment_start_col = template_left + template_w + gap
    placements = _place_fragments(grid, fragment_start_col, fragment_masks, colors)
    if placements is None:
        return None

    expected = grid.copy()
    for placed_fragment in placements:
        mask = placed_fragment["mask"]
        top = int(placed_fragment["top"])
        left = int(placed_fragment["left"])
        coords = np.argwhere(mask)
        expected[top + coords[:, 0], left + coords[:, 1]] = BG

    for hole in holes:
        coords = hole["coords"]
        expected[template_top + coords[:, 0], template_left + coords[:, 1]] = int(hole["color"])

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_SAMPLE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
