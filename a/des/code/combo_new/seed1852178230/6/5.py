import random
from collections import Counter

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _canonical_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append((int(rot.shape[0]), int(rot.shape[1]), rot.tobytes()))
        flip = np.fliplr(rot)
        variants.append((int(flip.shape[0]), int(flip.shape[1]), flip.tobytes()))
    return min(variants)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    out = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for variant in (rot, np.fliplr(rot)):
            key = (int(variant.shape[0]), int(variant.shape[1]), variant.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(variant.copy())
    return out


def _dilate(mask: np.ndarray, connectivity: int = 8, steps: int = 1) -> np.ndarray:
    out = mask.copy()
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    for _ in range(steps):
        src = out.copy()
        expanded = src.copy()
        for dr, dc in neighbors:
            if dr >= 0:
                src_r = slice(0, src.shape[0] - dr)
                dst_r = slice(dr, src.shape[0])
            else:
                src_r = slice(-dr, src.shape[0])
                dst_r = slice(0, src.shape[0] + dr)
            if dc >= 0:
                src_c = slice(0, src.shape[1] - dc)
                dst_c = slice(dc, src.shape[1])
            else:
                src_c = slice(-dc, src.shape[1])
                dst_c = slice(0, src.shape[1] + dc)
            expanded[dst_r, dst_c] |= src[src_r, src_c]
        out = expanded
    return out


def _find_positions(
    occupied: np.ndarray,
    mask: np.ndarray,
    *,
    border: int,
    padding: int,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    mh, mw = mask.shape
    if mh + 2 * border > h or mw + 2 * border > w:
        return []
    forbidden = _dilate(occupied, connectivity=8, steps=padding) if padding else occupied
    positions = []
    for r0 in range(border, h - mh - border + 1):
        for c0 in range(border, w - mw - border + 1):
            if not (forbidden[r0 : r0 + mh, c0 : c0 + mw] & mask).any():
                positions.append((int(r0), int(c0)))
    return positions


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    sub = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    sub[mask] = color


def _hole_data(subgrid: np.ndarray, bg: int, base_r: int, base_c: int) -> list[dict]:
    holes = []
    for coords in _components(subgrid == bg, connectivity=4):
        if any(
            r == 0 or c == 0 or r == subgrid.shape[0] - 1 or c == subgrid.shape[1] - 1
            for r, c in coords
        ):
            continue
        mask = _tight_mask(coords)
        rr0, cc0, rr1, cc1 = _bbox(coords)
        abs_coords = coords + np.array([base_r, base_c])
        holes.append(
            {
                "coords": abs_coords,
                "bbox": (base_r + rr0, base_c + cc0, base_r + rr1, base_c + cc1),
                "area": int(mask.sum()),
                "mask": mask,
                "key": _canonical_shape_key(mask),
            }
        )
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit scattered pieces into the carved holes of a solid rectangle, then keep only the final cascade color.

    Input structure:
    - The background is the most common color.
    - One large monochrome rectangular block contains several 4-connected background holes fully enclosed by its border.
    - Elsewhere on the canvas, there are several smaller monochrome 4-connected pieces, one per hole, each matching a hole shape up to rotation or reflection.

    Rule:
    - Detect the holed rectangle and extract each enclosed hole as a binary shape.
    - Match every outside piece to a hole by shape, allowing 90-degree rotations and mirror flips.
    - Order the matched pieces by decreasing area; if two areas tie, use the reading order of the source pieces.
    - Filling is cumulative: whenever a later piece is placed, it recolors all earlier filled holes, so after the last placement every hole has the same final color.
    - Clear the scattered source pieces from their original positions and fill all holes with that last piece's color.

    Key insight:
    - The explicit puzzle layer is dihedral shape matching, but the hidden layer is that the sequential recoloring collapses every fitted hole to a single final color determined by the last piece in the fill order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    fg_components = _components(g != bg, connectivity=4)
    assert fg_components, "expected at least one foreground component"

    target = None
    for idx, coords in enumerate(fg_components):
        colors = {int(g[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        r0, c0, r1, c1 = _bbox(coords)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all((sub == color) | (sub == bg)):
            continue
        if not (
            np.all(sub[0, :] == color)
            and np.all(sub[-1, :] == color)
            and np.all(sub[:, 0] == color)
            and np.all(sub[:, -1] == color)
        ):
            continue
        holes = _hole_data(sub, bg, r0, c0)
        if not holes:
            continue
        candidate = {
            "index": idx,
            "color": color,
            "coords": coords,
            "bbox": (r0, c0, r1, c1),
            "holes": holes,
            "area": int(coords.shape[0]),
        }
        if target is None or candidate["area"] > target["area"]:
            target = candidate

    assert target is not None, "failed to locate the holed rectangle"

    pieces = []
    for idx, coords in enumerate(fg_components):
        if idx == target["index"]:
            continue
        colors = {int(g[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        mask = _tight_mask(coords)
        pieces.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "area": int(mask.sum()),
                "color": color,
                "mask": mask,
                "key": _canonical_shape_key(mask),
            }
        )

    holes = target["holes"]
    assert len(pieces) == len(holes), "piece count must match hole count"

    hole_groups: dict[tuple[int, int, bytes], list[dict]] = {}
    for hole in holes:
        hole_groups.setdefault(hole["key"], []).append(hole)

    piece_groups: dict[tuple[int, int, bytes], list[dict]] = {}
    for piece in pieces:
        piece_groups.setdefault(piece["key"], []).append(piece)

    matched_pieces = []
    for key, hole_list in hole_groups.items():
        piece_list = piece_groups.get(key, [])
        assert len(piece_list) == len(hole_list), "shape matching must be one-to-one"
        hole_list.sort(key=lambda h: (h["bbox"][0], h["bbox"][1]))
        piece_list.sort(key=lambda p: (p["bbox"][0], p["bbox"][1]))
        matched_pieces.extend(piece_list)

    matched_pieces.sort(key=lambda p: (-p["area"], p["bbox"][0], p["bbox"][1]))
    final_color = int(matched_pieces[-1]["color"])

    out = g.copy()
    for piece in pieces:
        out[piece["coords"][:, 0], piece["coords"][:, 1]] = bg
    for hole in holes:
        out[hole["coords"][:, 0], hole["coords"][:, 1]] = final_color
    return out


BG_COLOR = 0
RECT_COLOR = 5
GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
RECT_H_RANGE = (9, 16)
RECT_W_RANGE = (10, 19)
HOLE_COUNT_CHOICES = [2, 3, 4]
HOLE_AREA_POOL = list(range(3, 10))
SHAPE_RETRIES = 60
PLACE_RETRIES = 80
GEN_RETRIES = 400


def _random_connected_mask(area: int, rng: random.Random) -> np.ndarray | None:
    dims = [
        (h, w)
        for h in range(2, 6)
        for w in range(2, 6)
        if area <= h * w and area >= max(h, w) - 1
    ]
    rng.shuffle(dims)
    for h, w in dims:
        for _ in range(SHAPE_RETRIES):
            start = (rng.randrange(h), rng.randrange(w))
            cells = {start}
            while len(cells) < area:
                frontier = []
                for r, c in cells:
                    for dr, dc in NEIGHBORS4:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                            frontier.append((nr, nc))
                if not frontier:
                    break
                cells.add(rng.choice(frontier))
            if len(cells) != area:
                continue
            pts = np.array(sorted(cells), dtype=int)
            mask = _tight_mask(pts)
            if int(mask.sum()) != area:
                continue
            if mask.shape[0] == 1 or mask.shape[1] == 1:
                continue
            if area == mask.size and area > 4 and rng.random() < 0.75:
                continue
            return mask
    return None


def _generate_shapes(count: int, rng: random.Random) -> list[dict] | None:
    areas = rng.sample(HOLE_AREA_POOL, count)
    rng.shuffle(areas)
    shapes = []
    used_keys = set()
    for area in areas:
        for _ in range(SHAPE_RETRIES):
            mask = _random_connected_mask(area, rng)
            if mask is None:
                continue
            key = _canonical_shape_key(mask)
            if key in used_keys:
                continue
            used_keys.add(key)
            shapes.append({"area": int(area), "mask": mask, "key": key})
            break
        else:
            return None
    return shapes


def _place_holes(inner_h: int, inner_w: int, shapes: list[dict], rng: random.Random) -> list[dict] | None:
    occupied = np.zeros((inner_h, inner_w), dtype=bool)
    placed = []
    order = sorted(shapes, key=lambda s: -s["area"])
    for shape in order:
        mask = shape["mask"]
        positions = _find_positions(occupied, mask, border=0, padding=1)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        occupied[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]] |= mask
        placed.append({**shape, "inner_r0": int(r0), "inner_c0": int(c0)})
    rng.shuffle(placed)
    return placed


def _scatter_pieces(
    canvas_shape: tuple[int, int],
    rect_bbox: tuple[int, int, int, int],
    pieces: list[dict],
    rng: random.Random,
) -> list[dict] | None:
    h, w = canvas_shape
    occupied = np.zeros((h, w), dtype=bool)
    rr0, rc0, rr1, rc1 = rect_bbox
    occupied[rr0 : rr1 + 1, rc0 : rc1 + 1] = True

    placed = []
    order = sorted(pieces, key=lambda p: -p["area"])
    for piece in order:
        variants = _dihedral_variants(piece["mask"])
        rng.shuffle(variants)
        success = None
        for variant in variants:
            positions = _find_positions(occupied, variant, border=1, padding=1)
            if not positions:
                continue
            r0, c0 = rng.choice(positions)
            success = (variant, int(r0), int(c0))
            break
        if success is None:
            return None
        variant, r0, c0 = success
        occupied[r0 : r0 + variant.shape[0], c0 : c0 + variant.shape[1]] |= variant
        placed.append({**piece, "placed_mask": variant, "r0": r0, "c0": c0})
    return placed


def _expected_output(inp: np.ndarray, holes: list[dict], placed_pieces: list[dict]) -> np.ndarray:
    out = inp.copy()
    final_color = min(placed_pieces, key=lambda p: (p["area"], p["r0"], p["c0"]))["color"]
    for piece in placed_pieces:
        coords = np.argwhere(piece["placed_mask"])
        out[piece["r0"] + coords[:, 0], piece["c0"] + coords[:, 1]] = BG_COLOR
    for hole in holes:
        out[hole["abs_coords"][:, 0], hole["abs_coords"][:, 1]] = final_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randrange(*GRID_H_RANGE)
    w = rng.randrange(*GRID_W_RANGE)
    rect_h = rng.randrange(RECT_H_RANGE[0], min(RECT_H_RANGE[1], h - 2) + 1)
    rect_w = rng.randrange(RECT_W_RANGE[0], min(RECT_W_RANGE[1], w - 2) + 1)
    rect_r0 = rng.randrange(1, h - rect_h)
    rect_c0 = rng.randrange(1, w - rect_w)
    rect_bbox = (rect_r0, rect_c0, rect_r0 + rect_h - 1, rect_c0 + rect_w - 1)

    hole_count = rng.choice(HOLE_COUNT_CHOICES)
    shapes = _generate_shapes(hole_count, rng)
    if shapes is None:
        return None

    inner_h = rect_h - 2
    inner_w = rect_w - 2
    if inner_h < 2 or inner_w < 2:
        return None

    holes = _place_holes(inner_h, inner_w, shapes, rng)
    if holes is None:
        return None

    piece_colors = [c for c in range(10) if c not in (BG_COLOR, RECT_COLOR)]
    if len(piece_colors) < hole_count:
        return None
    rng.shuffle(piece_colors)

    pieces = []
    for hole, color in zip(holes, piece_colors[:hole_count]):
        pieces.append({"area": hole["area"], "mask": hole["mask"], "key": hole["key"], "color": int(color)})

    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[rect_r0 : rect_r0 + rect_h, rect_c0 : rect_c0 + rect_w] = RECT_COLOR

    hole_infos = []
    for hole in holes:
        ar0 = rect_r0 + 1 + hole["inner_r0"]
        ac0 = rect_c0 + 1 + hole["inner_c0"]
        _paint_mask(grid, hole["mask"], ar0, ac0, BG_COLOR)
        coords = np.argwhere(hole["mask"]) + np.array([ar0, ac0])
        hole_infos.append({**hole, "abs_coords": coords})

    placed_pieces = _scatter_pieces(grid.shape, rect_bbox, pieces, rng)
    if placed_pieces is None:
        return None

    for piece in placed_pieces:
        _paint_mask(grid, piece["placed_mask"], piece["r0"], piece["c0"], piece["color"])

    vals, counts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(counts)]) != BG_COLOR:
        return None

    rect_area = int(np.sum(grid == RECT_COLOR))
    piece_areas = [int(piece["area"]) for piece in placed_pieces]
    if rect_area <= max(piece_areas):
        return None

    expected = _expected_output(grid, hole_infos, placed_pieces)
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
