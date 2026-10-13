import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BG = 8
TEMPLATE = 5
MARKER = 1
FILL_COLORS = [0, 2, 3, 4, 6, 7, 9]
DIRS: Tuple[Tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))

GEN_RETRIES = 300
SHAPE_RETRIES = 120
PLACEMENT_RETRIES = 200
GRID_MIN = 20
GRID_MAX = 28
TEMPLATE_MIN = 11
TEMPLATE_MAX = 17
PIECE_COUNT_RANGE = (2, 4)
SHAPE_AREA_RANGE = (3, 7)
SHAPE_BBOX_MAX = 5


def _boundary_mode(grid: np.ndarray) -> int:
    boundary = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    values, counts = np.unique(boundary, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: List[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> Tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _trim(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _coords_to_mask(coords: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0)


def _mask_signature(mask: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    variants = []
    for k in range(4):
        rot = _trim(np.rot90(mask, k))
        variants.append(tuple(map(tuple, np.argwhere(rot).tolist())))
    return min(variants)


def _mode(values: Sequence[int]) -> int:
    arr = np.asarray(values, dtype=int)
    uniq, counts = np.unique(arr, return_counts=True)
    return int(uniq[np.argmax(counts)])


def _rotate_dir(direction: int, k: int) -> int:
    return (direction - k) % 4


def _rotate_coord(coord: Tuple[int, int], shape: Tuple[int, int], k: int) -> Tuple[int, int]:
    r, c = coord
    h, w = shape
    for _ in range(k % 4):
        r, c = w - 1 - c, r
        h, w = w, h
    return int(r), int(c)


def _directional_cells(mask: np.ndarray) -> Dict[int, List[Tuple[int, int]]]:
    h, w = mask.shape
    out = {d: [] for d in range(4)}
    for r, c in np.argwhere(mask):
        exposed = []
        for d, (dr, dc) in enumerate(DIRS):
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                exposed.append(d)
        if len(exposed) == 1:
            out[exposed[0]].append((int(r), int(c)))
    return out


def _hole_marker_direction(
    subgrid: np.ndarray,
    hole_coords: np.ndarray,
    bg: int,
    template_color: int,
) -> Tuple[int, Tuple[int, int]]:
    hole_set = {tuple(map(int, coord)) for coord in hole_coords}
    markers = set()
    for r, c in hole_set:
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if subgrid[nr, nc] != bg and subgrid[nr, nc] != template_color:
                markers.add((nr, nc))
    assert len(markers) == 1, "each hole must have exactly one boundary marker"
    mr, mc = next(iter(markers))

    adjacent_hole_dirs = []
    for d, (dr, dc) in enumerate(DIRS):
        if (mr + dr, mc + dc) in hole_set:
            adjacent_hole_dirs.append(d)
    assert len(adjacent_hole_dirs) == 1, "hole marker direction must be unambiguous"
    return (adjacent_hole_dirs[0] + 2) % 4, (mr, mc)


def _piece_fill_and_marker(grid: np.ndarray, coords: np.ndarray) -> Tuple[int, Tuple[int, int]]:
    colors = grid[coords[:, 0], coords[:, 1]]
    fill = _mode(colors.tolist())
    marker_coords = coords[colors != fill]
    assert len(marker_coords) == 1, "each piece must contain exactly one orientation marker pixel"
    marker = tuple(map(int, marker_coords[0]))
    return fill, marker


def _piece_marker_direction(mask: np.ndarray, marker_local: Tuple[int, int]) -> int:
    r, c = marker_local
    h, w = mask.shape
    exposed = []
    for d, (dr, dc) in enumerate(DIRS):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
            exposed.append(d)
    assert len(exposed) == 1, "piece marker direction must be unambiguous"
    return exposed[0]


def _find_aligned_rotation(piece_mask: np.ndarray, piece_dir: int, hole_mask: np.ndarray, hole_dir: int) -> int:
    for k in range(4):
        if np.array_equal(_trim(np.rot90(piece_mask, k)), hole_mask) and _rotate_dir(piece_dir, k) == hole_dir:
            return k
    raise AssertionError("piece cannot be rotated to match the hole shape and marker direction")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each template hole with the matching scattered piece, using the marker
    direction to validate the intended rotation before inserting the piece.
    """
    g = np.asarray(grid, dtype=int)
    bg = _boundary_mode(g)

    comps = _components(g != bg)
    assert len(comps) >= 2, "expected one template and at least one scattered piece"

    template_index = max(range(len(comps)), key=lambda idx: len(comps[idx]))
    template_coords = comps[template_index]
    tr0, tc0, tr1, tc1 = _bbox(template_coords)
    template_patch = g[tr0 : tr1 + 1, tc0 : tc1 + 1]

    template_color = _mode(template_patch[template_patch != bg].tolist())
    assert template_color != bg, "template color must differ from background"

    holes = []
    for hole_coords in _components(template_patch == bg):
        if np.any(
            (hole_coords[:, 0] == 0)
            | (hole_coords[:, 0] == template_patch.shape[0] - 1)
            | (hole_coords[:, 1] == 0)
            | (hole_coords[:, 1] == template_patch.shape[1] - 1)
        ):
            continue
        hole_mask, (hr0, hc0) = _coords_to_mask(hole_coords)
        hole_dir, marker_pos = _hole_marker_direction(template_patch, hole_coords, bg, template_color)
        holes.append(
            {
                "signature": _mask_signature(hole_mask),
                "mask": hole_mask,
                "coords_abs": hole_coords + np.array([tr0, tc0], dtype=int),
                "dir": hole_dir,
                "marker_abs": (marker_pos[0] + tr0, marker_pos[1] + tc0),
                "sort_key": (int(np.min(hole_coords[:, 0])), int(np.min(hole_coords[:, 1]))),
            }
        )
    assert holes, "template must contain at least one enclosed hole"

    pieces = []
    for idx, comp in enumerate(comps):
        if idx == template_index:
            continue
        fill, marker_abs = _piece_fill_and_marker(g, comp)
        piece_mask, (pr0, pc0) = _coords_to_mask(comp)
        marker_local = (marker_abs[0] - pr0, marker_abs[1] - pc0)
        piece_dir = _piece_marker_direction(piece_mask, marker_local)
        pieces.append(
            {
                "signature": _mask_signature(piece_mask),
                "mask": piece_mask,
                "fill": fill,
                "dir": piece_dir,
                "coords_abs": comp,
                "sort_key": (pr0, pc0),
            }
        )
    assert len(pieces) == len(holes), "hole count and piece count must match"

    holes_by_sig: Dict[Tuple[Tuple[int, int], ...], List[dict]] = {}
    for hole in holes:
        holes_by_sig.setdefault(hole["signature"], []).append(hole)

    pieces_by_sig: Dict[Tuple[Tuple[int, int], ...], List[dict]] = {}
    for piece in pieces:
        pieces_by_sig.setdefault(piece["signature"], []).append(piece)

    assert set(holes_by_sig) == set(pieces_by_sig), "piece shapes must match hole shapes"

    out = g.copy()
    for idx, comp in enumerate(comps):
        if idx != template_index:
            out[comp[:, 0], comp[:, 1]] = bg

    marker_mask = (template_patch != bg) & (template_patch != template_color)
    out[tr0 : tr1 + 1, tc0 : tc1 + 1][marker_mask] = template_color

    for signature, hole_group in holes_by_sig.items():
        piece_group = pieces_by_sig[signature]
        assert len(hole_group) == len(piece_group), "matched shape multiplicities must agree"
        hole_group = sorted(hole_group, key=lambda item: item["sort_key"])
        piece_group = sorted(piece_group, key=lambda item: item["sort_key"])
        for hole, piece in zip(hole_group, piece_group):
            _find_aligned_rotation(piece["mask"], piece["dir"], hole["mask"], hole["dir"])
            coords = hole["coords_abs"]
            out[coords[:, 0], coords[:, 1]] = int(piece["fill"])

    return out


def _random_shape(rng: random.Random, used_signatures: set) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        target_area = rng.randint(*SHAPE_AREA_RANGE)
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS:
                    nbr = (r + dr, c + dc)
                    if nbr not in cells:
                        frontier.append(nbr)
            rng.shuffle(frontier)
            added = False
            for nbr in frontier:
                trial = cells | {nbr}
                rows = [r for r, _ in trial]
                cols = [c for _, c in trial]
                if max(rows) - min(rows) + 1 > SHAPE_BBOX_MAX or max(cols) - min(cols) + 1 > SHAPE_BBOX_MAX:
                    continue
                cells = trial
                added = True
                break
            if not added:
                break
        if len(cells) != target_area:
            continue

        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm_cells = sorted((r - min_r, c - min_c) for r, c in cells)
        h = max(r for r, _ in norm_cells) + 1
        w = max(c for _, c in norm_cells) + 1
        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm_cells:
            mask[r, c] = True

        if not any(_directional_cells(mask).values()):
            continue
        signature = _mask_signature(mask)
        if signature in used_signatures:
            continue
        used_signatures.add(signature)
        return mask
    return None


def _can_place_hole(mask: np.ndarray, top: int, left: int, used: np.ndarray) -> bool:
    for r, c in np.argwhere(mask):
        ar, ac = top + int(r), left + int(c)
        r0 = max(0, ar - 1)
        c0 = max(0, ac - 1)
        r1 = min(used.shape[0] - 1, ar + 1)
        c1 = min(used.shape[1] - 1, ac + 1)
        if np.any(used[r0 : r1 + 1, c0 : c1 + 1]):
            return False
    return True


def _place_holes_in_template(
    rng: random.Random,
    shapes: Sequence[np.ndarray],
    template_h: int,
    template_w: int,
) -> List[dict] | None:
    used = np.zeros((template_h, template_w), dtype=bool)
    holes = []
    for base_mask in shapes:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            rot_k = rng.randrange(4)
            mask = _trim(np.rot90(base_mask, rot_k))
            mh, mw = mask.shape
            if template_h - mh < 5 or template_w - mw < 5:
                return None
            top = rng.randint(2, template_h - mh - 3)
            left = rng.randint(2, template_w - mw - 3)
            if not _can_place_hole(mask, top, left, used):
                continue

            directional = _directional_cells(mask)
            available_dirs = [d for d, cells in directional.items() if cells]
            if not available_dirs:
                continue
            hole_dir = rng.choice(available_dirs)
            marker_cell = rng.choice(directional[hole_dir])
            marker_pos = (top + marker_cell[0] + DIRS[hole_dir][0], left + marker_cell[1] + DIRS[hole_dir][1])

            coords = np.argwhere(mask) + np.array([top, left], dtype=int)
            used[coords[:, 0], coords[:, 1]] = True
            holes.append(
                {
                    "mask": mask,
                    "top": top,
                    "left": left,
                    "dir": hole_dir,
                    "marker_local": marker_cell,
                    "marker_pos": marker_pos,
                    "coords_local": coords,
                }
            )
            placed = True
            break
        if not placed:
            return None
    return holes


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0] - 1, r1 + pad)
    cc1 = min(blocked.shape[1] - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_piece_bbox(
    rng: random.Random,
    blocked: np.ndarray,
    piece_shape: Tuple[int, int],
) -> Tuple[int, int] | None:
    ph, pw = piece_shape
    if blocked.shape[0] - ph < 2 or blocked.shape[1] - pw < 2:
        return None
    for _ in range(PLACEMENT_RETRIES * 2):
        top = rng.randint(1, blocked.shape[0] - ph - 1)
        left = rng.randint(1, blocked.shape[1] - pw - 1)
        if np.any(blocked[top : top + ph, left : left + pw]):
            continue
        return top, left
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    piece_count = rng.randint(*PIECE_COUNT_RANGE)
    if piece_count > len(FILL_COLORS):
        return None

    used_signatures = set()
    shapes = []
    for _ in range(piece_count):
        shape = _random_shape(rng, used_signatures)
        if shape is None:
            return None
        shapes.append(shape)

    template_h = rng.randint(max(TEMPLATE_MIN, 7 + piece_count), TEMPLATE_MAX)
    template_w = rng.randint(max(TEMPLATE_MIN, 7 + piece_count), TEMPLATE_MAX)
    holes = _place_holes_in_template(rng, shapes, template_h, template_w)
    if holes is None:
        return None

    grid_h = rng.randint(max(GRID_MIN, template_h + 8), GRID_MAX)
    grid_w = rng.randint(max(GRID_MIN, template_w + 8), GRID_MAX)
    if grid_h - template_h < 4 or grid_w - template_w < 4:
        return None

    template_top = rng.randint(2, grid_h - template_h - 2)
    template_left = rng.randint(2, grid_w - template_w - 2)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[template_top : template_top + template_h, template_left : template_left + template_w] = TEMPLATE

    for hole in holes:
        abs_coords = hole["coords_local"] + np.array([template_top, template_left], dtype=int)
        grid[abs_coords[:, 0], abs_coords[:, 1]] = BG
        mr, mc = hole["marker_pos"]
        grid[template_top + mr, template_left + mc] = MARKER
        hole["coords_abs"] = abs_coords

    blocked = np.zeros_like(grid, dtype=bool)
    _mark_blocked(
        blocked,
        template_top,
        template_left,
        template_top + template_h - 1,
        template_left + template_w - 1,
        pad=1,
    )

    fill_colors = rng.sample(FILL_COLORS, piece_count)
    rng.shuffle(holes)
    for hole, fill_color in zip(holes, fill_colors):
        scatter_k = rng.randrange(4)
        piece_mask = _trim(np.rot90(hole["mask"], scatter_k))
        marker_local = _rotate_coord(hole["marker_local"], hole["mask"].shape, scatter_k)
        placement = _place_piece_bbox(rng, blocked, piece_mask.shape)
        if placement is None:
            return None
        top, left = placement
        coords = np.argwhere(piece_mask) + np.array([top, left], dtype=int)
        grid[coords[:, 0], coords[:, 1]] = fill_color
        grid[top + marker_local[0], left + marker_local[1]] = MARKER
        _mark_blocked(blocked, top, left, top + piece_mask.shape[0] - 1, left + piece_mask.shape[1] - 1, pad=1)

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except AssertionError:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
