import random
from collections import deque

import numpy as np


BG = 0
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(mask.shape), mask.astype(np.uint8).tobytes()


def _array_key(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(arr.shape), arr.tobytes()


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True

            components.append(
                {
                    "color": color,
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )

    return components


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for k in range(4):
        rotated = np.rot90(mask, k).astype(bool)
        key = _mask_key(rotated)
        if key in seen:
            continue
        seen.add(key)
        variants.append(rotated)
    return variants


def _mask_coords(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple((int(r), int(c)) for r, c in np.argwhere(mask))


def _candidate_placements(
    piece: dict,
    occupied: set[tuple[int, int]],
    min_contacts: int = 1,
) -> list[tuple[tuple[int, int], ...]]:
    touch_targets = set()
    for r, c in occupied:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in occupied:
                touch_targets.add((nr, nc))

    placements = []
    seen = set()
    for coords in piece["orient_coords"]:
        for tr, tc in touch_targets:
            for pr, pc in coords:
                dr = tr - pr
                dc = tc - pc
                placed = tuple(sorted((rr + dr, cc + dc) for rr, cc in coords))
                if placed in seen:
                    continue
                seen.add(placed)
                if any(cell in occupied for cell in placed):
                    continue
                if _contact_count(placed, occupied) < min_contacts:
                    continue
                placements.append(placed)
    return placements


def _render_solution(
    pieces: list[dict],
    placements: list[tuple[tuple[int, int], ...] | None],
    bg: int,
) -> np.ndarray:
    all_cells = [cell for placed in placements if placed is not None for cell in placed]
    rows = [r for r, _ in all_cells]
    cols = [c for _, c in all_cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for piece, placed in zip(pieces, placements):
        assert placed is not None
        for r, c in placed:
            patch[r - r0, c - c0] = int(piece["color"])

    return _canonicalize_patch(patch, bg)


def _canonicalize_patch(patch: np.ndarray, bg: int) -> np.ndarray:
    mask = patch != bg
    best_key = None
    best_variants = {}

    for k in range(4):
        rotated_mask = np.rot90(mask, k)
        key = _mask_key(rotated_mask)
        rotated_patch = np.rot90(patch, k)
        if best_key is None or key < best_key:
            best_key = key
            best_variants = {_array_key(rotated_patch): rotated_patch}
        elif key == best_key:
            best_variants.setdefault(_array_key(rotated_patch), rotated_patch)

    assert len(best_variants) == 1
    return next(iter(best_variants.values()))


def _solve_output(pieces: list[dict], bg: int) -> np.ndarray:
    ordered = sorted(
        pieces,
        key=lambda piece: (-piece["area"], -len(piece["orient_coords"]), -piece["mask"].shape[0] * piece["mask"].shape[1]),
    )
    count = len(ordered)
    placements: list[tuple[tuple[int, int], ...] | None] = [None] * count
    best_rank = None
    best_patch = None
    nodes = 0

    def record_solution():
        nonlocal best_rank, best_patch
        patch = _render_solution(ordered, placements, bg)
        rank = _mask_key(patch != bg)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_patch = patch

    def search(placed_mask: int, occupied: set[tuple[int, int]]):
        nonlocal nodes
        nodes += 1
        assert nodes <= SEARCH_NODE_LIMIT
        if placed_mask == (1 << count) - 1:
            record_solution()
            return

        next_idx = None
        next_candidates = None
        for idx in range(count):
            if placed_mask & (1 << idx):
                continue
            candidates = _candidate_placements(ordered[idx], occupied, min_contacts=MIN_CONTACTS)
            if not candidates:
                continue
            if next_candidates is None or len(candidates) < len(next_candidates):
                next_idx = idx
                next_candidates = candidates

        if next_idx is None:
            return

        for placed in next_candidates:
            placements[next_idx] = placed
            search(placed_mask | (1 << next_idx), occupied | set(placed))
        placements[next_idx] = None

    for anchor in ordered[0]["orient_coords"]:
        placements[0] = tuple(sorted(anchor))
        search(1, set(placements[0]))
    placements[0] = None

    assert best_patch is not None
    return best_patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered rotated monochrome fragments into one common template.

    Input structure:
    - Background is black.
    - There are 3 to 4 non-black monochrome connected components, each in a different color.
    - Every component is a connected subset of the same template after a quarter-turn rotation, cropped to its own tight bounding box.

    Rule:
    - Treat each component as a rotatable fragment and try all 0/90/180/270 degree orientations.
    - Place the counter-rotated fragments on one shared canvas so they are pairwise disjoint and form a single connected template.
    - Tight-crop the completed template, rotate it to its canonical binary-mask orientation, and keep every fragment's original color.

    Key insight:
    - The objects are not translated cutouts from one layout; they are differently rotated partial masks that must be jointly oriented before they can be overlaid into the full picture.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    pieces = _find_components(g, bg)
    assert 3 <= len(pieces) <= 4

    for piece in pieces:
        piece["orient_coords"] = [_mask_coords(variant) for variant in _rotation_variants(piece["mask"])]
        assert piece["orient_coords"]

    return _solve_output(pieces, bg)


MIN_PIECES = 3
MAX_PIECES = 4
MIN_PIECE_AREA = 4
WORK_H_RANGE = (5, 8)
WORK_W_RANGE = (5, 8)
TEMPLATE_AREA_MAX = 18
GRID_SIDE_MIN = 13
GRID_SIDE_MAX = 22
PLACEMENT_PADDING = 1
REGION_ATTEMPTS = 80
DESIGN_ATTEMPTS = 200
GEN_ATTEMPTS = 400
SEARCH_NODE_LIMIT = 800
MIN_CONTACTS = 3


def _cells_to_mask(cells: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _is_connected_cells(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(cells)


def _has_hole(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    empty = ~mask
    seen = np.zeros_like(empty, dtype=bool)
    queue = deque()

    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))

    return bool((empty & ~seen).any())


def _rotation_count(mask: np.ndarray) -> int:
    return len(_rotation_variants(mask))


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_key(variant) for variant in _rotation_variants(mask))


def _contact_count(cells, occupied: set[tuple[int, int]]) -> int:
    total = 0
    for r, c in cells:
        for nbr in _neighbors4(r, c):
            if nbr in occupied:
                total += 1
    return total


def _sample_piece_sizes(total_area: int, piece_count: int) -> list[int]:
    sizes = [MIN_PIECE_AREA] * piece_count
    for _ in range(total_area - MIN_PIECE_AREA * piece_count):
        sizes[random.randrange(piece_count)] += 1
    sizes.sort(reverse=True)
    return sizes


def _crop_patch(patch: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(patch != bg)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _grow_region(
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
    size: int,
    must_touch_union: bool,
) -> set[tuple[int, int]] | None:
    empty = {(r, c) for r in range(h) for c in range(w) if (r, c) not in occupied}
    if must_touch_union:
        seeds = [
            cell
            for cell in empty
            if any(nbr in occupied for nbr in _neighbors4(*cell))
        ]
    else:
        seeds = list(empty)
    if not seeds:
        return None

    for _ in range(REGION_ATTEMPTS):
        seed = random.choice(seeds)
        region = {seed}
        frontier = {
            nbr
            for nbr in _neighbors4(*seed)
            if 0 <= nbr[0] < h and 0 <= nbr[1] < w and nbr in empty and nbr not in region
        }

        while len(region) < size and frontier:
            ranked = list(frontier)
            ranked.sort(
                key=lambda cell: sum(
                    0 <= nbr[0] < h and 0 <= nbr[1] < w and nbr in empty and nbr not in region
                    for nbr in _neighbors4(*cell)
                ),
                reverse=True,
            )
            pick = random.choice(ranked[: min(6, len(ranked))])
            region.add(pick)
            frontier.discard(pick)
            for nbr in _neighbors4(*pick):
                if 0 <= nbr[0] < h and 0 <= nbr[1] < w and nbr in empty and nbr not in region:
                    frontier.add(nbr)

        if len(region) == size and (not must_touch_union or _contact_count(region, occupied) >= MIN_CONTACTS):
            return region
    return None


def _build_output(piece_count: int) -> np.ndarray | None:
    for _ in range(DESIGN_ATTEMPTS):
        h = random.randint(*WORK_H_RANGE)
        w = random.randint(*WORK_W_RANGE)
        min_area = max(MIN_PIECE_AREA * piece_count, 14)
        max_area = min(TEMPLATE_AREA_MAX, h * w - 1)
        if min_area > max_area:
            continue

        total_area = random.randint(min_area, max_area)
        sizes = _sample_piece_sizes(total_area, piece_count)

        occupied: set[tuple[int, int]] = set()
        regions: list[set[tuple[int, int]]] = []
        for idx, size in enumerate(sizes):
            region = _grow_region(occupied, h, w, size, must_touch_union=(idx > 0))
            if region is None:
                break
            regions.append(region)
            occupied |= region
        else:
            if not _is_connected_cells(occupied):
                continue
            union_mask = _crop_mask(_cells_to_mask(occupied, (h, w)))
            if _has_hole(union_mask):
                continue
            if _rotation_count(union_mask) != 4:
                continue

            colors = random.sample(range(1, 10), piece_count)
            patch = np.zeros((h, w), dtype=int)
            valid_regions = True
            signatures = set()
            for color, region in zip(colors, regions):
                region_mask = _crop_mask(_cells_to_mask(region, (h, w)))
                if _rotation_count(region_mask) != 4:
                    valid_regions = False
                    break
                sig = _shape_signature(region_mask)
                if sig in signatures:
                    valid_regions = False
                    break
                signatures.add(sig)
                patch[_cells_to_mask(region, (h, w))] = color
            if not valid_regions:
                continue
            return _canonicalize_patch(_crop_patch(patch, BG), BG)

    return None


def _extract_color_fragments(output: np.ndarray) -> list[tuple[int, np.ndarray]]:
    fragments = []
    for color in sorted(int(v) for v in np.unique(output) if int(v) != BG):
        mask = output == color
        fragments.append((color, _crop_mask(mask)))
    return fragments


def _choose_rotations(piece_count: int) -> list[int]:
    return random.sample(range(4), piece_count)


def _place_fragments(fragments: list[tuple[int, np.ndarray]]) -> np.ndarray | None:
    total_box_area = sum(mask.shape[0] * mask.shape[1] for _, mask in fragments)
    base = max(GRID_SIDE_MIN, int(np.ceil(np.sqrt(total_box_area))) + 5)
    h = random.randint(base, min(GRID_SIDE_MAX, base + 4))
    w = random.randint(base, min(GRID_SIDE_MAX, base + 4))
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    order = list(range(len(fragments)))
    order.sort(key=lambda idx: fragments[idx][1].shape[0] * fragments[idx][1].shape[1], reverse=True)

    for idx in order:
        color, mask = fragments[idx]
        ph, pw = map(int, mask.shape)
        choices = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(choices)
        placed = False
        for r0, c0 in choices:
            rr0 = max(0, r0 - PLACEMENT_PADDING)
            cc0 = max(0, c0 - PLACEMENT_PADDING)
            rr1 = min(h, r0 + ph + PLACEMENT_PADDING)
            cc1 = min(w, c0 + pw + PLACEMENT_PADDING)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw][mask] = color
            reserved[r0:r0 + ph, c0:c0 + pw] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    piece_count = random.choice([MIN_PIECES, MAX_PIECES])
    output = _build_output(piece_count)
    if output is None:
        return None

    fragments = _extract_color_fragments(output)
    rotations = _choose_rotations(piece_count)
    if len(set(rotations)) < 2 or all(rotation == 0 for rotation in rotations):
        return None

    rotated = []
    for (color, mask), rotation in zip(fragments, rotations):
        rotated.append((color, np.rot90(mask, rotation).astype(bool)))

    grid = _place_fragments(rotated)
    if grid is None:
        return None

    try:
        _ = main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
