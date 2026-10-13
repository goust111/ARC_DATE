from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

_RNG = np.random.default_rng()

# Generator configuration.
GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
SLOT_COUNT_CHOICES = (2, 3, 4)
TEMPLATE_H_RANGE = (9, 17)
TEMPLATE_W_RANGE = (10, 18)
SHAPE_SIDE_RANGE = (2, 6)
SHAPE_AREA_RANGE = (3, 8)
SHAPE_BUILD_RETRIES = 120
SHAPE_SAMPLE_RETRIES = 400
TEMPLATE_RETRIES = 120
FRAGMENT_RETRIES = 160
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def find_components(mask: np.ndarray) -> list[dict]:
    """Return 4-connected components of a boolean mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return comps


def component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    trimmed = mask[r0 : r1 + 1, c0 : c1 + 1]
    return (int(trimmed.shape[0]), int(trimmed.shape[1]), trimmed.astype(np.uint8).tobytes())


def dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    for rr, cc in np.argwhere(mask):
        grid[int(r0 + rr), int(c0 + cc)] = int(color)


def patch_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert each scattered fragment into the matching cavity of the grey template after a horizontal mirror.

    Input structure:
    - The background is black.
    - A large central template is one connected non-black object dominated by grey pixels, with several black cavities inside it.
    - Each cavity has exactly one non-grey marker pixel on its grey boundary, and separate monochrome fragments lie outside the template.

    Rule:
    - Treat the largest non-black component as the template and find every enclosed black cavity inside it.
    - For each outside fragment, horizontally mirror its shape and match it to the unique cavity with the same shape.
    - Fill the matched cavity with that cavity's marker color, keep the marker pixel unchanged, and erase the outside fragment.

    Key insight:
    - Shape decides the destination only after a horizontal mirror, while the cavity-edge marker decides the final color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    comps = find_components(g != bg)
    assert comps

    template = max(comps, key=lambda comp: comp["area"])
    template_mask = np.zeros((h, w), dtype=bool)
    template_mask[template["coords"][:, 0], template["coords"][:, 1]] = True

    template_colors = [int(g[int(r), int(c)]) for r, c in template["coords"]]
    grey = int(Counter(template_colors).most_common(1)[0][0])

    tr0, tc0, tr1, tc1 = template["bbox"]
    template_local = template_mask[tr0 : tr1 + 1, tc0 : tc1 + 1]
    holes: dict[tuple[int, int, bytes], dict] = {}

    for comp in find_components(~template_local):
        coords = comp["coords"]
        if (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == template_local.shape[0] - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == template_local.shape[1] - 1)
        ):
            continue

        abs_coords = coords + np.array([tr0, tc0], dtype=int)
        marker_colors = set()
        for rr, cc in abs_coords:
            for dr, dc in NEIGHBORS4:
                nr, nc = int(rr) + dr, int(cc) + dc
                if 0 <= nr < h and 0 <= nc < w and template_mask[nr, nc]:
                    col = int(g[nr, nc])
                    if col != grey:
                        marker_colors.add(col)

        assert len(marker_colors) == 1
        key = shape_key(component_mask(comp))
        assert key not in holes
        holes[key] = {"coords": abs_coords, "marker": int(next(iter(marker_colors)))}

    fragments = [comp for comp in comps if comp is not template]
    assert len(fragments) == len(holes)

    out = g.copy()
    for frag in fragments:
        out[frag["coords"][:, 0], frag["coords"][:, 1]] = int(bg)

    used_keys: set[tuple[int, int, bytes]] = set()
    for frag in fragments:
        frag_mask = component_mask(frag)
        key = shape_key(np.fliplr(frag_mask))
        assert key in holes and key not in used_keys
        used_keys.add(key)
        hole = holes[key]
        out[hole["coords"][:, 0], hole["coords"][:, 1]] = int(hole["marker"])

    return out


def _random_shape() -> np.ndarray | None:
    for _ in range(SHAPE_BUILD_RETRIES):
        sh = int(_RNG.integers(*SHAPE_SIDE_RANGE))
        sw = int(_RNG.integers(*SHAPE_SIDE_RANGE))
        lo = max(SHAPE_AREA_RANGE[0], max(sh, sw))
        hi = min(SHAPE_AREA_RANGE[1], sh * sw - 1)
        if lo > hi:
            continue
        area = int(_RNG.integers(lo, hi + 1))

        start = (int(_RNG.integers(sh)), int(_RNG.integers(sw)))
        cells = {start}
        while len(cells) < area:
            frontier = {
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in NEIGHBORS4
                if 0 <= r + dr < sh and 0 <= c + dc < sw and (r + dr, c + dc) not in cells
            }
            if not frontier:
                break
            frontier_list = list(frontier)
            pick = frontier_list[int(_RNG.integers(len(frontier_list)))]
            cells.add(pick)
        if len(cells) != area:
            continue

        rs = [r for r, _c in cells]
        cs = [c for _r, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if r1 - r0 + 1 != sh or c1 - c0 + 1 != sw:
            continue

        mask = np.zeros((sh, sw), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    return None


def _sample_fragment_shapes(count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    originals: set[tuple[int, int, bytes]] = set()
    mirrored: set[tuple[int, int, bytes]] = set()

    for _ in range(SHAPE_SAMPLE_RETRIES):
        if len(shapes) == count:
            return shapes
        mask = _random_shape()
        if mask is None:
            continue
        orig_key = shape_key(mask)
        mirr_key = shape_key(np.fliplr(mask))
        if orig_key == mirr_key:
            continue
        if (
            orig_key in originals
            or orig_key in mirrored
            or mirr_key in originals
            or mirr_key in mirrored
        ):
            continue
        shapes.append(mask)
        originals.add(orig_key)
        mirrored.add(mirr_key)

    return None


def _place_slots(slot_masks: list[np.ndarray]) -> tuple[int, int, list[tuple[int, int]]] | None:
    order = list(range(len(slot_masks)))
    order.sort(key=lambda i: int(slot_masks[i].sum()), reverse=True)
    max_h = max(mask.shape[0] for mask in slot_masks)
    max_w = max(mask.shape[1] for mask in slot_masks)

    for _ in range(TEMPLATE_RETRIES):
        th_lo = max(TEMPLATE_H_RANGE[0], int(max_h + 3))
        tw_lo = max(TEMPLATE_W_RANGE[0], int(max_w + 3))
        if th_lo >= TEMPLATE_H_RANGE[1] or tw_lo >= TEMPLATE_W_RANGE[1]:
            return None

        th = int(_RNG.integers(th_lo, TEMPLATE_H_RANGE[1]))
        tw = int(_RNG.integers(tw_lo, TEMPLATE_W_RANGE[1]))
        occupied = np.zeros((th, tw), dtype=bool)
        placements: list[tuple[int, int] | None] = [None] * len(slot_masks)

        ok = True
        for idx in order:
            mask = slot_masks[idx]
            mh, mw = mask.shape
            dilated = dilate8(occupied)
            candidates: list[tuple[int, int]] = []
            for r0 in range(1, th - mh):
                for c0 in range(1, tw - mw):
                    if np.any(dilated[r0 : r0 + mh, c0 : c0 + mw] & mask):
                        continue
                    candidates.append((r0, c0))
            if not candidates:
                ok = False
                break
            placements[idx] = candidates[int(_RNG.integers(len(candidates)))]
            r0, c0 = placements[idx]
            occupied[r0 : r0 + mh, c0 : c0 + mw] |= mask
        if ok:
            return th, tw, [(int(r), int(c)) for r, c in placements]
    return None


def _hole_boundary_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    cells = set()
    mh, mw = mask.shape
    for rr, cc in np.argwhere(mask):
        ar, ac = int(top + rr), int(left + cc)
        for dr, dc in NEIGHBORS4:
            nr, nc = ar + dr, ac + dc
            inside = top <= nr < top + mh and left <= nc < left + mw and mask[nr - top, nc - left]
            if not inside:
                cells.add((nr, nc))
    return list(cells)


def _center_with_jitter(total: int, size: int) -> int:
    base = (total - size) // 2
    lo = max(1, base - 2)
    hi = min(total - size - 1, base + 2)
    if lo > hi:
        return int(max(0, min(base, total - size)))
    return int(_RNG.integers(lo, hi + 1))


def _place_fragments(grid: np.ndarray, reserved: np.ndarray, masks: list[np.ndarray], colors: list[int]) -> bool:
    order = list(range(len(masks)))
    order.sort(key=lambda i: int(masks[i].sum()), reverse=True)

    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        dilated = dilate8(reserved)
        candidates: list[tuple[int, int]] = []
        for r0 in range(0, grid.shape[0] - mh + 1):
            for c0 in range(0, grid.shape[1] - mw + 1):
                if np.any(dilated[r0 : r0 + mh, c0 : c0 + mw] & mask):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = candidates[int(_RNG.integers(len(candidates)))]
        paint_mask(grid, mask, r0, c0, colors[idx])
        reserved[r0 : r0 + mh, c0 : c0 + mw] |= mask
    return True


def _build_template(
    slot_masks: list[np.ndarray], marker_colors: list[int]
) -> tuple[np.ndarray, np.ndarray] | None:
    placed = _place_slots(slot_masks)
    if placed is None:
        return None
    th, tw, positions = placed

    template_in = np.full((th, tw), GREY, dtype=int)
    template_out = np.full((th, tw), GREY, dtype=int)
    used_markers: set[tuple[int, int]] = set()
    boundary_sets: list[set[tuple[int, int]]] = []
    for mask, (r0, c0) in zip(slot_masks, positions):
        boundary_sets.append(
            {
                (r, c)
                for r, c in _hole_boundary_cells(mask, r0, c0)
                if 0 <= r < th and 0 <= c < tw
            }
        )

    for mask, color, (r0, c0) in zip(slot_masks, marker_colors, positions):
        view_in = template_in[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        view_out = template_out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        view_in[mask] = BG
        view_out[mask] = int(color)

    for i, (color, boundary) in enumerate(zip(marker_colors, boundary_sets)):
        other_boundary = set().union(*(b for j, b in enumerate(boundary_sets) if j != i))
        boundary = [
            (r, c)
            for r, c in boundary
            if (r, c) not in other_boundary
            and (r, c) not in used_markers
            and template_in[r, c] == GREY
        ]
        if not boundary:
            return None
        mr, mc = boundary[int(_RNG.integers(len(boundary)))]
        used_markers.add((mr, mc))
        template_in[mr, mc] = int(color)
        template_out[mr, mc] = int(color)

    return template_in, template_out


def _gen_attempt() -> np.ndarray | None:
    slot_count = int(_RNG.choice(SLOT_COUNT_CHOICES))
    shapes = _sample_fragment_shapes(slot_count)
    if shapes is None or len(shapes) != slot_count:
        return None

    colors = list(PALETTE)
    _RNG.shuffle(colors)
    marker_colors = [int(c) for c in colors[:slot_count]]
    fragment_colors = [int(c) for c in colors[slot_count : 2 * slot_count]]
    slot_masks = [np.fliplr(mask) for mask in shapes]

    built = _build_template(slot_masks, marker_colors)
    if built is None:
        return None
    template_in, template_out = built
    th, tw = template_in.shape

    h_lo = max(GRID_H_RANGE[0], int(th + 6))
    w_lo = max(GRID_W_RANGE[0], int(tw + 8))
    if h_lo >= GRID_H_RANGE[1] or w_lo >= GRID_W_RANGE[1]:
        return None
    h = int(_RNG.integers(h_lo, GRID_H_RANGE[1]))
    w = int(_RNG.integers(w_lo, GRID_W_RANGE[1]))

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    tr0 = _center_with_jitter(h, th)
    tc0 = _center_with_jitter(w, tw)
    inp[tr0 : tr0 + th, tc0 : tc0 + tw] = template_in
    out[tr0 : tr0 + th, tc0 : tc0 + tw] = template_out

    reserved = np.zeros((h, w), dtype=bool)
    reserved[tr0 : tr0 + th, tc0 : tc0 + tw] = True
    if not _place_fragments(inp, reserved, shapes, fragment_colors):
        return None

    if get_bg(inp) != BG:
        return None
    if len(find_components(inp != BG)) != slot_count + 1:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-slot sample")
