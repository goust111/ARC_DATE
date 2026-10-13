import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DIR_TO_COLOR = {
    "up": RED,
    "down": BLUE,
    "left": GREEN,
    "right": YELLOW,
}

FRAGMENT_COLORS = [6, 7, 8, 9]

MAX_GENERATION_ATTEMPTS = 400
PIECE_COUNT_CHOICES = (4, 5, 6)
PIECE_AREA_MIN = 3
PIECE_AREA_MAX = 6
PARTITION_BBOX_MAX = 8
SIDE_BAND_MIN = 6
SIDE_BAND_MAX = 9
TEMPLATE_GAP = 2
PLACEMENT_TRIES = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, forbidden: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in forbidden or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    cur = mask.astype(bool)
    for k in range(4):
        rot = np.rot90(cur, k)
        key = (rot.shape, rot.tobytes())
        if key not in seen:
            seen.add(key)
            out.append(rot)
    return out


def _template_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _direction_from_piece(coords: np.ndarray, template_bbox: tuple[int, int, int, int]) -> str:
    pr, pc = coords.mean(axis=0)
    tr, tc = _template_center(template_bbox)
    dr = float(pr - tr)
    dc = float(pc - tc)
    if abs(dc) > abs(dr):
        return "left" if dc < 0 else "right"
    return "up" if dr < 0 else "down"


def _enclosed_black(sub: np.ndarray) -> np.ndarray:
    h, w = sub.shape
    black = sub == BLACK
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if black[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if black[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    return black & ~seen


def _build_exact_cover(cavity_mask: np.ndarray, piece_masks: list[np.ndarray]):
    cell_coords = [tuple(x) for x in np.argwhere(cavity_mask)]
    index_of = {coord: i for i, coord in enumerate(cell_coords)}
    placements_by_piece: list[list[int]] = []
    coverers: list[list[tuple[int, int]]] = [[] for _ in range(len(cell_coords))]
    ch, cw = cavity_mask.shape

    for piece_idx, piece_mask in enumerate(piece_masks):
        placements: list[int] = []
        seen_masks: set[int] = set()
        for rot in _unique_rotations(piece_mask):
            ph, pw = rot.shape
            rot_cells = np.argwhere(rot)
            for top in range(ch - ph + 1):
                for left in range(cw - pw + 1):
                    if not np.all(cavity_mask[top : top + ph, left : left + pw][rot]):
                        continue
                    bits = 0
                    for rr, cc in rot_cells:
                        bits |= 1 << index_of[(top + int(rr), left + int(cc))]
                    if bits in seen_masks:
                        continue
                    seen_masks.add(bits)
                    placements.append(bits)
        placements_by_piece.append(placements)
        for bits in placements:
            mask_bits = bits
            while mask_bits:
                lsb = mask_bits & -mask_bits
                cell_idx = lsb.bit_length() - 1
                coverers[cell_idx].append((piece_idx, bits))
                mask_bits ^= lsb

    return cell_coords, placements_by_piece, coverers


def _solve_exact_cover(
    cavity_mask: np.ndarray,
    piece_masks: list[np.ndarray],
    limit: int = 1,
) -> tuple[int, list[int] | None]:
    cell_coords, placements_by_piece, coverers = _build_exact_cover(cavity_mask, piece_masks)
    cell_count = len(cell_coords)
    piece_areas = [int(mask.sum()) for mask in piece_masks]
    if sum(piece_areas) != cell_count:
        return 0, None
    if any(not placements for placements in placements_by_piece):
        return 0, None

    full_mask = (1 << cell_count) - 1
    unused_all = (1 << len(piece_masks)) - 1
    first_solution: list[int] | None = None
    solution_count = 0
    partial = [0] * len(piece_masks)

    def dfs(remaining_mask: int, unused_mask: int):
        nonlocal first_solution, solution_count
        if solution_count >= limit:
            return
        if remaining_mask == 0:
            if unused_mask == 0:
                solution_count += 1
                if first_solution is None:
                    first_solution = partial.copy()
            return
        if unused_mask == 0:
            return

        area_left = remaining_mask.bit_count()
        unused_area = 0
        for piece_idx, area in enumerate(piece_areas):
            if (unused_mask >> piece_idx) & 1:
                unused_area += area
        if area_left != unused_area:
            return

        best_options: list[tuple[int, int]] | None = None
        bit_scan = remaining_mask
        while bit_scan:
            lsb = bit_scan & -bit_scan
            cell_idx = lsb.bit_length() - 1
            options = []
            for piece_idx, placement_bits in coverers[cell_idx]:
                if not ((unused_mask >> piece_idx) & 1):
                    continue
                if placement_bits & ~remaining_mask:
                    continue
                options.append((piece_idx, placement_bits))
            if not options:
                return
            if best_options is None or len(options) < len(best_options):
                best_options = options
                if len(best_options) == 1:
                    break
            bit_scan ^= lsb

        assert best_options is not None
        best_options.sort(key=lambda item: (-piece_areas[item[0]], item[0]))
        for piece_idx, placement_bits in best_options:
            partial[piece_idx] = placement_bits
            dfs(remaining_mask & ~placement_bits, unused_mask & ~(1 << piece_idx))
            partial[piece_idx] = 0
            if solution_count >= limit:
                return

    dfs(full_mask, unused_all)
    return solution_count, first_solution


def _render_solution(
    template_crop: np.ndarray,
    cavity_mask: np.ndarray,
    solution_bits: list[int],
    directions: list[str],
) -> np.ndarray:
    out = np.where(template_crop == GREY, GREY, BLACK).astype(int)
    cavity_coords = [tuple(x) for x in np.argwhere(cavity_mask)]
    for piece_idx, bits in enumerate(solution_bits):
        color = DIR_TO_COLOR[directions[piece_idx]]
        mask_bits = int(bits)
        while mask_bits:
            lsb = mask_bits & -mask_bits
            cell_idx = lsb.bit_length() - 1
            rr, cc = cavity_coords[cell_idx]
            out[rr, cc] = color
            mask_bits ^= lsb
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered fragments into the template cavity, then recolor them by source direction.

    Input structure:
    - The background is black. Near the center sits one large grey closed outline made of a single-pixel border, enclosing a black cavity.
    - Outside the outline are several disconnected monochrome fragments. Each fragment is a connected shape and may appear in any rotation.
    - The cavity can be partitioned exactly into those fragment shapes, using only 90° rotations.

    Rule:
    - Find the unique grey template, crop to its bounding box, and extract the enclosed black cavity.
    - Match every outside fragment to one placement inside the cavity so that all fragments together tile the cavity exactly.
    - Recolor the placed fragment by the fragment's source side relative to the template center: above -> red, below -> blue, left -> green, right -> yellow.
    - Remove all outside fragments and output only the cropped template region with the grey border and recolored interior.

    Key insight:
    - Shape matching determines where each fragment goes, but the final color is determined independently by the fragment's original direction from the template.
    """
    g = np.asarray(grid, dtype=int)
    grey_components = _components_from_mask(g == GREY)
    assert grey_components, "expected a grey template border"
    border_coords = max(grey_components, key=len)
    template_bbox = _bbox(border_coords)
    r0, c0, r1, c1 = template_bbox

    template_crop = g[r0 : r1 + 1, c0 : c1 + 1]
    cavity_mask = _enclosed_black(template_crop)
    cavity_components = _components_from_mask(cavity_mask)
    assert len(cavity_components) == 1, "expected one enclosed cavity"

    fragments = _monochrome_components(g, forbidden={BLACK, GREY})
    assert fragments, "expected outside fragments"
    piece_masks = [_mask_from_coords(comp["coords"]) for comp in fragments]
    directions = [_direction_from_piece(comp["coords"], template_bbox) for comp in fragments]

    solution_count, solution_bits = _solve_exact_cover(cavity_mask, piece_masks, limit=2)
    assert solution_count == 1 and solution_bits is not None, "expected a unique tiling"

    return _render_solution(template_crop, cavity_mask, solution_bits, directions)


def _mask_has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    pad = np.pad(~mask, 1, constant_values=True)
    seen = np.zeros_like(pad, dtype=bool)
    stack = [(0, 0)]
    seen[0, 0] = True
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and pad[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return np.any(pad & ~seen)


def _cells_to_mask(cells: list[tuple[int, int]]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    arr = np.array(cells, dtype=int)
    r0, c0, r1, c1 = _bbox(arr)
    rel = [(int(r - r0), int(c - c0)) for r, c in cells]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for rr, cc in rel:
        mask[rr, cc] = True
    return mask, rel


def _generate_partition(rng: np.random.Generator):
    piece_count = int(rng.choice(PIECE_COUNT_CHOICES))
    target_areas = [int(rng.integers(PIECE_AREA_MIN, PIECE_AREA_MAX + 1)) for _ in range(piece_count)]

    pieces: list[set[tuple[int, int]]] = [{(0, 0)}]
    occupied = {(0, 0)}
    for _ in range(1, piece_count):
        frontier = {
            nbr
            for cell in occupied
            for nbr in _neighbors4(*cell)
            if nbr not in occupied
        }
        if not frontier:
            return None
        frontier_list = list(frontier)
        seed = tuple(frontier_list[int(rng.integers(len(frontier_list)))])
        pieces.append({seed})
        occupied.add(seed)

    total_steps = sum(target_areas) - piece_count
    for _ in range(total_steps):
        growable = [idx for idx in range(piece_count) if len(pieces[idx]) < target_areas[idx]]
        growable = [int(x) for x in rng.permutation(growable)]
        placed = False
        for piece_idx in growable:
            frontier = []
            for cell in pieces[piece_idx]:
                for nbr in _neighbors4(*cell):
                    if nbr in occupied:
                        continue
                    new_occupied = occupied | {nbr}
                    rows = [r for r, _ in new_occupied]
                    cols = [c for _, c in new_occupied]
                    if max(rows) - min(rows) + 1 > PARTITION_BBOX_MAX:
                        continue
                    if max(cols) - min(cols) + 1 > PARTITION_BBOX_MAX:
                        continue
                    frontier.append(nbr)
            if not frontier:
                continue
            frontier.sort(key=lambda cell: (-sum((nn in occupied) for nn in _neighbors4(*cell)), cell))
            choice = frontier[int(rng.integers(len(frontier)))]
            pieces[piece_idx].add(choice)
            occupied.add(choice)
            placed = True
            break
        if not placed:
            return None

    all_cells = np.array(sorted(occupied), dtype=int)
    shift = np.array([all_cells[:, 0].min(), all_cells[:, 1].min()])
    all_cells -= shift
    cavity_mask = np.zeros((int(all_cells[:, 0].max()) + 1, int(all_cells[:, 1].max()) + 1), dtype=bool)
    cavity_mask[all_cells[:, 0], all_cells[:, 1]] = True
    if _mask_has_hole(cavity_mask):
        return None

    normalized_pieces: list[np.ndarray] = []
    cavity_regions: list[np.ndarray] = []
    for piece in pieces:
        coords = np.array(sorted(piece), dtype=int) - shift
        piece_mask = _mask_from_coords(coords)
        normalized_pieces.append(piece_mask)
        cavity_regions.append(coords)

    if cavity_mask.sum() < 14:
        return None
    return cavity_mask, normalized_pieces, cavity_regions


def _shell_from_cavity(cavity_mask: np.ndarray) -> np.ndarray:
    ch, cw = cavity_mask.shape
    cavity_in_patch = np.zeros((ch + 2, cw + 2), dtype=bool)
    cavity_in_patch[1 : 1 + ch, 1 : 1 + cw] = cavity_mask
    shell = np.zeros_like(cavity_in_patch)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rs = 1 + dr
            cs = 1 + dc
            shell[rs : rs + ch, cs : cs + cw] |= cavity_mask
    shell &= ~cavity_in_patch
    template = np.zeros_like(cavity_in_patch, dtype=int)
    template[shell] = GREY
    return template


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray):
    h, w = blocked.shape
    for r, c in coords:
        r0 = max(0, int(r) - 1)
        r1 = min(h, int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(w, int(c) + 2)
        blocked[r0:r1, c0:c1] = True


def _place_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> np.ndarray:
    coords = np.argwhere(mask) + np.array([top, left])
    grid[coords[:, 0], coords[:, 1]] = int(color)
    return coords


def _balanced_directions(rng: np.random.Generator, count: int) -> list[str]:
    base = ["up", "down", "left", "right"]
    base = [base[i] for i in rng.permutation(len(base))]
    out = base.copy()
    extra = count - 4
    if extra > 0:
        extra_dirs = base.copy()
        extra_dirs = [extra_dirs[i] for i in rng.permutation(len(extra_dirs))]
        out.extend(extra_dirs[:extra])
    out = [out[i] for i in rng.permutation(len(out))]
    return out


def _try_place_fragment(
    blocked: np.ndarray,
    mask: np.ndarray,
    direction: str,
    template_bbox: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    h, w = blocked.shape
    ph, pw = mask.shape
    local = np.argwhere(mask).astype(float)
    local_center_r = float(local[:, 0].mean())
    local_center_c = float(local[:, 1].mean())
    tr, tc = _template_center(template_bbox)
    r0, c0, r1, c1 = template_bbox

    for _ in range(PLACEMENT_TRIES):
        if direction == "up":
            max_top = r0 - TEMPLATE_GAP - ph
            if max_top < 0:
                return None
            top = int(rng.integers(0, max_top + 1))
            piece_center_r = top + local_center_r
            max_dc = int(np.floor(abs(piece_center_r - tr)))
            col_min = int(np.ceil(tc - max_dc - local_center_c))
            col_max = int(np.floor(tc + max_dc - local_center_c))
            col_min = max(0, col_min)
            col_max = min(w - pw, col_max)
            if col_min > col_max:
                continue
            left = int(rng.integers(col_min, col_max + 1))
        elif direction == "down":
            min_top = r1 + TEMPLATE_GAP + 1
            max_top = h - ph
            if min_top > max_top:
                return None
            top = int(rng.integers(min_top, max_top + 1))
            piece_center_r = top + local_center_r
            max_dc = int(np.floor(abs(piece_center_r - tr)))
            col_min = int(np.ceil(tc - max_dc - local_center_c))
            col_max = int(np.floor(tc + max_dc - local_center_c))
            col_min = max(0, col_min)
            col_max = min(w - pw, col_max)
            if col_min > col_max:
                continue
            left = int(rng.integers(col_min, col_max + 1))
        elif direction == "left":
            max_left = c0 - TEMPLATE_GAP - pw
            if max_left < 0:
                return None
            left = int(rng.integers(0, max_left + 1))
            piece_center_c = left + local_center_c
            max_dr = int(np.floor(abs(piece_center_c - tc)))
            top_min = int(np.ceil(tr - max_dr - local_center_r))
            top_max = int(np.floor(tr + max_dr - local_center_r))
            top_min = max(0, top_min)
            top_max = min(h - ph, top_max)
            if top_min > top_max:
                continue
            top = int(rng.integers(top_min, top_max + 1))
        else:
            min_left = c1 + TEMPLATE_GAP + 1
            max_left = w - pw
            if min_left > max_left:
                return None
            left = int(rng.integers(min_left, max_left + 1))
            piece_center_c = left + local_center_c
            max_dr = int(np.floor(abs(piece_center_c - tc)))
            top_min = int(np.ceil(tr - max_dr - local_center_r))
            top_max = int(np.floor(tr + max_dr - local_center_r))
            top_min = max(0, top_min)
            top_max = min(h - ph, top_max)
            if top_min > top_max:
                continue
            top = int(rng.integers(top_min, top_max + 1))

        coords = np.argwhere(mask) + np.array([top, left])
        if np.any(blocked[coords[:, 0], coords[:, 1]]):
            continue
        if _direction_from_piece(coords, template_bbox) != direction:
            continue
        return top, left
    return None


def _expected_output(template_patch: np.ndarray, cavity_regions: list[np.ndarray], directions: list[str]) -> np.ndarray:
    out = np.where(template_patch == GREY, GREY, BLACK).astype(int)
    for region, direction in zip(cavity_regions, directions):
        color = DIR_TO_COLOR[direction]
        region_shifted = region + np.array([1, 1])
        out[region_shifted[:, 0], region_shifted[:, 1]] = color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    partition = _generate_partition(rng)
    if partition is None:
        return None
    cavity_mask, piece_masks, cavity_regions = partition

    solution_count, _ = _solve_exact_cover(cavity_mask, piece_masks, limit=2)
    if solution_count != 1:
        return None

    template_patch = _shell_from_cavity(cavity_mask)
    th, tw = template_patch.shape

    top_band = int(rng.integers(SIDE_BAND_MIN, SIDE_BAND_MAX))
    bottom_band = int(rng.integers(SIDE_BAND_MIN, SIDE_BAND_MAX))
    left_band = int(rng.integers(SIDE_BAND_MIN, SIDE_BAND_MAX))
    right_band = int(rng.integers(SIDE_BAND_MIN, SIDE_BAND_MAX))

    grid_h = top_band + th + bottom_band
    grid_w = left_band + tw + right_band
    if max(grid_h, grid_w) > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    ttop = top_band
    tleft = left_band
    grid[ttop : ttop + th, tleft : tleft + tw] = template_patch
    template_bbox = (ttop, tleft, ttop + th - 1, tleft + tw - 1)

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[max(0, ttop - 1) : min(grid_h, ttop + th + 1), max(0, tleft - 1) : min(grid_w, tleft + tw + 1)] = True

    directions = _balanced_directions(rng, len(piece_masks))
    colors = [FRAGMENT_COLORS[i % len(FRAGMENT_COLORS)] for i in range(len(piece_masks))]
    colors = [colors[i] for i in rng.permutation(len(colors))]

    placed_directions: list[str] = []
    for piece_idx, piece_mask in enumerate(piece_masks):
        rotations = [np.rot90(piece_mask, int(k)) for k in rng.permutation(4)]
        placed = False
        for rot in rotations:
            placement = _try_place_fragment(blocked, rot, directions[piece_idx], template_bbox, rng)
            if placement is None:
                continue
            top, left = placement
            coords = _place_mask(grid, rot, top, left, colors[piece_idx % len(colors)])
            _mark_blocked(blocked, coords)
            actual_direction = _direction_from_piece(coords, template_bbox)
            if actual_direction != directions[piece_idx]:
                return None
            placed_directions.append(actual_direction)
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(template_patch, cavity_regions, placed_directions)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
