import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

HOLE_COUNT_RANGE = (2, 4)
HOLE_DIM_RANGE = (3, 6)
TEMPLATE_SIDE_MIN = 10
TEMPLATE_SIDE_MAX = 20
GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 30

SHAPE_RETRIES = 160
PACK_RETRIES = 120
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _relative_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _has_internal_holes(mask: np.ndarray) -> bool:
    empty = ~mask
    for comp in _components(empty):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == mask.shape[0] - 1):
            continue
        if np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == mask.shape[1] - 1):
            continue
        return True
    return False


def _bounded_walk(
    length: int, limit: int, start: int, end: int, rng: random.Random
) -> list[int] | None:
    if abs(int(end) - int(start)) > int(length) - 1:
        return None
    seq = [int(start)]
    pos = int(start)
    for i in range(1, int(length)):
        remaining = int(length) - 1 - i
        candidates: list[int] = []
        weights: list[int] = []
        for delta in (-1, 0, 1):
            nxt = pos + delta
            if not (0 <= nxt < limit):
                continue
            if abs(int(end) - nxt) > remaining:
                continue
            candidates.append(nxt)
            weight = 1
            if delta == 0:
                weight += 1
            if (end > pos and delta == 1) or (end < pos and delta == -1):
                weight += 2
            weights.append(weight)
        if not candidates:
            return None
        pos = int(rng.choices(candidates, weights=weights, k=1)[0])
        seq.append(pos)
    return seq


def _random_piece_mask(height: int, width: int, rng: random.Random) -> np.ndarray | None:
    area = int(height) * int(width)
    for _ in range(SHAPE_RETRIES):
        tb_start = rng.randrange(width)
        tb_end = rng.choice([c for c in range(width) if abs(c - tb_start) <= height - 1])
        tb_cols = _bounded_walk(height, width, tb_start, tb_end, rng)
        if tb_cols is None:
            continue

        anchor_row = rng.randrange(height)
        anchor_col = int(tb_cols[anchor_row])

        left_starts = [r for r in range(height) if abs(r - anchor_row) <= anchor_col]
        right_steps = width - 1 - anchor_col
        right_ends = [r for r in range(height) if abs(r - anchor_row) <= right_steps]
        if not left_starts or not right_ends:
            continue
        lr_start = rng.choice(left_starts)
        lr_end = rng.choice(right_ends)
        left_rows = _bounded_walk(anchor_col + 1, height, lr_start, anchor_row, rng)
        right_rows = _bounded_walk(width - anchor_col, height, anchor_row, lr_end, rng)
        if left_rows is None or right_rows is None:
            continue
        lr_rows = left_rows + right_rows[1:]

        mask = np.zeros((height, width), dtype=bool)
        for r, c in enumerate(tb_cols):
            mask[r, c] = True
        for c, r in enumerate(lr_rows):
            mask[r, c] = True

        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue

        current = int(mask.sum())
        min_area = max(current + 1, int(np.ceil(0.45 * area)))
        max_area = min(area - 1, int(np.floor(0.8 * area)))
        if min_area > max_area:
            continue
        target = rng.randint(min_area, max_area)

        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((int(nr), int(nc)))

        while int(mask.sum()) < target and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            r, c = cell
            mask[r, c] = True
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((int(nr), int(nc)))

        if int(mask.sum()) != target:
            continue
        if len(_components(mask)) != 1:
            continue
        if _has_internal_holes(mask):
            continue
        if np.all(mask):
            continue
        return mask
    return None


def _candidate_anchor_sites(mask: np.ndarray) -> dict[str, list[int]]:
    sites = {
        "top": [int(c) for c in np.flatnonzero(mask[0])],
        "bottom": [int(c) for c in np.flatnonzero(mask[-1])],
        "left": [int(r) for r in np.flatnonzero(mask[:, 0])],
        "right": [int(r) for r in np.flatnonzero(mask[:, -1])],
    }
    return {side: vals for side, vals in sites.items() if vals}


def _place_holes(
    template_h: int, template_w: int, dims: list[tuple[int, int]], rng: random.Random
) -> dict[tuple[int, int], tuple[int, int]] | None:
    placements: dict[tuple[int, int], tuple[int, int]] = {}
    reserved = np.zeros((template_h, template_w), dtype=bool)
    for dim in sorted(dims, key=lambda d: d[0] * d[1], reverse=True):
        h, w = dim
        candidates = []
        for r0 in range(1, template_h - h):
            for c0 in range(1, template_w - w):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(template_h, r0 + h + 1)
                cc1 = min(template_w, c0 + w + 1)
                if reserved[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        placements[dim] = (r0, c0)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(template_h, r0 + h + 1)
        cc1 = min(template_w, c0 + w + 1)
        reserved[rr0:rr1, cc0:cc1] = True
    return placements


def _paint_piece(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    grid[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = int(color)


def _place_pieces_on_canvas(
    grid_h: int,
    grid_w: int,
    template_box: tuple[int, int, int, int],
    pieces: list[dict],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    tr0, tc0, tr1, tc1 = template_box
    reserved[max(0, tr0 - 1) : min(grid_h, tr1 + 2), max(0, tc0 - 1) : min(grid_w, tc1 + 2)] = True

    placements: list[tuple[int, int]] = [(-1, -1)] * len(pieces)
    order = sorted(range(len(pieces)), key=lambda i: pieces[i]["mask"].size, reverse=True)
    for idx in order:
        mask = pieces[idx]["mask"]
        h, w = mask.shape
        candidates = []
        for r0 in range(grid_h - h + 1):
            for c0 in range(grid_w - w + 1):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(grid_h, r0 + h + 1)
                cc1 = min(grid_w, c0 + w + 1)
                if reserved[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        placements[idx] = (r0, c0)
        reserved[max(0, r0 - 1) : min(grid_h, r0 + h + 1), max(0, c0 - 1) : min(grid_w, c0 + w + 1)] = True
    return placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert the scattered fragments into the template holes and recolor each filled hole by its anchor pixel.

    Input structure:
    - Background is black.
    - One large connected template object is mostly grey and may include a few non-grey anchor pixels on hole borders.
    - Inside the template bbox are several enclosed black hole components, each with exactly one adjacent non-black, non-grey anchor color.
    - Elsewhere are separate monochrome fragments whose bounding-box sizes uniquely match the holes.

    Rule:
    - Identify the largest non-black connected component as the template and crop to its bounding box.
    - Find every enclosed black hole inside that crop, record its bbox and the unique anchor color adjacent to it.
    - Match each external fragment to the unique hole with the same bounding-box size.
    - Translate the fragment so its bbox aligns with the hole bbox, repaint the fragment cells with the hole's anchor color, and return only the cropped template region.

    Key insight:
    - The holes are paired with fragments by spatial size, but the filled color comes from the tiny anchor pixel on the template border rather than from the fragment itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _components(g != BLACK)
    assert comps, "grid must contain at least one non-black component"
    template_idx = max(range(len(comps)), key=lambda i: len(comps[i]))
    template_bbox = _bbox(comps[template_idx])
    tr0, tc0, tr1, tc1 = template_bbox
    crop = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
    out = crop.copy()

    holes_by_dim: dict[tuple[int, int], dict[str, object]] = {}
    for pts in _components(crop == BLACK):
        if np.any(pts[:, 0] == 0) or np.any(pts[:, 0] == crop.shape[0] - 1):
            continue
        if np.any(pts[:, 1] == 0) or np.any(pts[:, 1] == crop.shape[1] - 1):
            continue

        box = _bbox(pts)
        mask = _relative_mask(pts, box)
        dim = mask.shape
        assert dim not in holes_by_dim, "hole bbox sizes must be unique"

        anchor_colors: set[int] = set()
        for r, c in pts:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < crop.shape[0] and 0 <= nc < crop.shape[1]:
                    value = int(crop[nr, nc])
                    if value not in (BLACK, GREY):
                        anchor_colors.add(value)
        assert len(anchor_colors) == 1, "each hole must have exactly one anchor color"
        holes_by_dim[dim] = {
            "bbox": box,
            "mask": mask,
            "anchor": int(next(iter(anchor_colors))),
        }

    fragments = []
    for idx, comp in enumerate(comps):
        if idx == template_idx:
            continue
        box = _bbox(comp)
        mask = _relative_mask(comp, box)
        colors = set(int(g[r, c]) for r, c in comp)
        assert len(colors) == 1, "each fragment must be monochrome"
        fragments.append({"mask": mask, "dim": mask.shape})

    assert len(fragments) == len(holes_by_dim), "fragment and hole counts must match"
    for frag in fragments:
        dim = frag["dim"]
        assert dim in holes_by_dim, "each fragment must match a hole bbox size"
        hole = holes_by_dim[dim]
        assert np.array_equal(frag["mask"], hole["mask"]), "fragment shape must match hole shape"
        r0, c0, r1, c1 = hole["bbox"]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[frag["mask"]] = int(hole["anchor"])

    return out


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    hole_count = rng.randint(*HOLE_COUNT_RANGE)

    dims: list[tuple[int, int]] = []
    masks_by_dim: dict[tuple[int, int], np.ndarray] = {}
    for _ in range(PACK_RETRIES):
        dims.clear()
        masks_by_dim.clear()
        while len(dims) < hole_count:
            dim = (rng.randint(*HOLE_DIM_RANGE), rng.randint(*HOLE_DIM_RANGE))
            if dim in masks_by_dim:
                continue
            mask = _random_piece_mask(dim[0], dim[1], rng)
            if mask is None:
                break
            dims.append(dim)
            masks_by_dim[dim] = mask
        if len(dims) == hole_count:
            break
    if len(dims) != hole_count:
        return None

    available = COLOR_POOL[:]
    rng.shuffle(available)
    anchor_colors = available[:hole_count]
    piece_colors = available[hole_count : 2 * hole_count]

    placements = None
    template_h = template_w = -1
    for _ in range(PACK_RETRIES):
        template_h = rng.randint(max(TEMPLATE_SIDE_MIN, max(h for h, _ in dims) + 2), TEMPLATE_SIDE_MAX)
        template_w = rng.randint(max(TEMPLATE_SIDE_MIN, max(w for _, w in dims) + 2), TEMPLATE_SIDE_MAX)
        placements = _place_holes(template_h, template_w, dims, rng)
        if placements is not None:
            break
    if placements is None:
        return None

    template = np.full((template_h, template_w), GREY, dtype=int)
    expected = template.copy()
    pieces: list[dict] = []
    dim_to_index = {dim: i for i, dim in enumerate(dims)}
    used_sides = []

    for dim in dims:
        mask = masks_by_dim[dim]
        r0, c0 = placements[dim]
        h, w = dim
        template[r0 : r0 + h, c0 : c0 + w][mask] = BLACK
        expected[r0 : r0 + h, c0 : c0 + w][mask] = int(anchor_colors[dim_to_index[dim]])

        sites = _candidate_anchor_sites(mask)
        if not sites:
            return None
        sides = list(sites)
        rng.shuffle(sides)
        side = sides[0]
        used_sides.append(side)
        if side == "top":
            col = rng.choice(sites[side])
            template[r0 - 1, c0 + col] = int(anchor_colors[dim_to_index[dim]])
            expected[r0 - 1, c0 + col] = int(anchor_colors[dim_to_index[dim]])
        elif side == "bottom":
            col = rng.choice(sites[side])
            template[r0 + h, c0 + col] = int(anchor_colors[dim_to_index[dim]])
            expected[r0 + h, c0 + col] = int(anchor_colors[dim_to_index[dim]])
        elif side == "left":
            row = rng.choice(sites[side])
            template[r0 + row, c0 - 1] = int(anchor_colors[dim_to_index[dim]])
            expected[r0 + row, c0 - 1] = int(anchor_colors[dim_to_index[dim]])
        else:
            row = rng.choice(sites[side])
            template[r0 + row, c0 + w] = int(anchor_colors[dim_to_index[dim]])
            expected[r0 + row, c0 + w] = int(anchor_colors[dim_to_index[dim]])

        pieces.append(
            {
                "mask": mask.copy(),
                "color": int(piece_colors[dim_to_index[dim]]),
            }
        )

    if len(set(used_sides)) == 1 and hole_count > 2 and rng.random() < 0.5:
        return None

    grid = None
    for _ in range(PACK_RETRIES):
        grid_h = rng.randint(max(GRID_SIDE_MIN, template_h + 2), GRID_SIDE_MAX)
        grid_w = rng.randint(max(GRID_SIDE_MIN, template_w + 2), GRID_SIDE_MAX)
        if grid_h * grid_w <= template_h * template_w + sum(p["mask"].size for p in pieces):
            continue

        template_top = rng.randint(0, grid_h - template_h)
        template_left = rng.randint(0, grid_w - template_w)
        template_box = (
            template_top,
            template_left,
            template_top + template_h - 1,
            template_left + template_w - 1,
        )
        piece_places = _place_pieces_on_canvas(grid_h, grid_w, template_box, pieces, rng)
        if piece_places is None:
            continue

        grid = np.full((grid_h, grid_w), BLACK, dtype=int)
        grid[template_top : template_top + template_h, template_left : template_left + template_w] = template
        for piece, (pr0, pc0) in zip(pieces, piece_places):
            _paint_piece(grid, pr0, pc0, piece["mask"], piece["color"])
        break
    if grid is None:
        return None

    if len(_components(grid != BLACK)) != hole_count + 1:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        built = _build_case(rng)
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate valid input")
