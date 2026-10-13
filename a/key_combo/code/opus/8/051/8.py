from __future__ import annotations

import itertools
import math
import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAROON = 9
PIECE_COLORS = (BLUE, RED, GREEN, MAROON)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int = BLACK) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    view[mask] = patch[mask]


def _patch_from_coords(coords: np.ndarray, color: int, grey_cells: set[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        cell = (int(r), int(c))
        patch[int(r) - r0, int(c) - c0] = GREY if cell in grey_cells else int(color)
    return patch


def _shape_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _rotate_clockwise(patch: np.ndarray, quarter_turns: int) -> np.ndarray:
    k = int(quarter_turns) % 4
    return np.rot90(patch, -k).copy()


def _normalize_patch(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BLACK)
    if coords.size == 0:
        return patch.copy()
    r0, c0, r1, c1 = _bbox_coords(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _unique_rotations_count(patch: np.ndarray) -> int:
    seen = set()
    for k in range(4):
        rot = _normalize_patch(np.rot90(patch, -k))
        seen.add((rot.shape, rot.tobytes()))
    return len(seen)


def _extract_fragments(grid: np.ndarray) -> list[dict]:
    fragment_mask = (grid != BLACK) & (grid != YELLOW)
    comps = _components_mask(fragment_mask)
    fragments = []
    for coords in comps:
        bbox = _bbox_coords(coords)
        patch = _crop_bbox(grid, bbox)
        mask = patch != BLACK
        colors = set(map(int, np.unique(patch[mask])))
        base_colors = colors - {GREY}
        assert len(base_colors) == 1, "each fragment must have exactly one non-grey base color"
        fragments.append(
            {
                "coords": coords,
                "bbox": bbox,
                "patch": patch,
                "base_color": int(next(iter(base_colors))),
                "grey_count": int(np.sum(patch == GREY)),
                "area": int(np.sum(mask)),
            }
        )
    return fragments


def _target_interior(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]]:
    yellow = np.argwhere(grid == YELLOW)
    assert yellow.shape == (4, 2), "expected exactly four yellow markers"
    rows = yellow[:, 0]
    cols = yellow[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    expected = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    actual = {tuple(map(int, rc)) for rc in yellow}
    assert actual == expected, "yellow markers must sit on the rectangle corners"
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "target rectangle must have a non-empty interior"
    return (r0 + 1, c0 + 1), (r1 - r0 - 1, c1 - c0 - 1), expected


def _pack_pieces(region_shape: tuple[int, int], pieces: list[np.ndarray]) -> np.ndarray | None:
    h, w = region_shape
    board = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    for patch in pieces:
        ph, pw = patch.shape
        mask = patch != BLACK
        placed = False
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                if np.any(occupied[r0 : r0 + ph, c0 : c0 + pw] & mask):
                    continue
                _paste(board, patch, r0, c0, transparent=BLACK)
                occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    if not np.all(occupied):
        return None
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count grey markers inside each fragment to determine its rotation, then embed the rotated fragments into the yellow-marked frame.

    Input structure:
    - A black background contains 3-4 separated multicolor 4-connected fragments in the upper part of the grid.
    - Each fragment uses exactly one base color from {blue, red, green, maroon} plus 0-3 grey cells that belong to the fragment itself.
    - Four yellow pixels mark the corners of an outer rectangle whose black interior is the target packing region.

    Rule:
    - For each fragment, count its grey cells k and rotate the whole fragment clockwise by k quarter-turns.
    - Sort fragments by the top-left corner of their input bounding boxes in row-major order.
    - Inside the yellow-defined interior, place the rotated fragments in that order; for each fragment, scan top-left to bottom-right and use the first position where its full rotated shape fits on still-black cells.
    - Remove the original upper fragments, keep the yellow corner markers, and leave every placed base-color and grey cell in the target region.

    Key insight:
    - The grey cells are both the rotation code and part of the geometry, so counting them and rotating the full fragment are both necessary for the final packing to work.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    target_origin, target_shape, marker_cells = _target_interior(g)
    fragments = _extract_fragments(g)
    assert 3 <= len(fragments) <= 4, "expected 3-4 fragments"

    ordered = sorted(fragments, key=lambda frag: (frag["bbox"][0], frag["bbox"][1]))
    rotated = [_rotate_clockwise(frag["patch"], frag["grey_count"]) for frag in ordered]
    packed = _pack_pieces(target_shape, rotated)
    assert packed is not None, "rotated fragments must tile the target interior"

    out = np.zeros_like(g)
    for r, c in marker_cells:
        out[r, c] = YELLOW
    _paste(out, packed, target_origin[0], target_origin[1], transparent=BLACK)
    return out


TARGET_DIMS = ((5, 3), (6, 3), (5, 4), (7, 3), (6, 4))
MAX_GENERATION_ATTEMPTS = 500


def _compact_shape(area: int) -> tuple[int, int]:
    pairs = [(d, area // d) for d in range(1, int(math.isqrt(area)) + 1) if area % d == 0]
    return max(pairs, key=lambda hw: hw[0])


def _size_compositions(total: int, count: int) -> list[tuple[int, ...]]:
    out = []
    for combo in itertools.product(range(4, 8), repeat=count):
        if sum(combo) == total:
            out.append(combo)
    return out


def _region_frontier(region: set[tuple[int, int]], remaining: set[tuple[int, int]]) -> list[tuple[int, int]]:
    frontier = set()
    for r, c in region:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in remaining:
                frontier.add((nr, nc))
    return list(frontier)


def _random_partition(h: int, w: int, sizes: tuple[int, ...]) -> list[np.ndarray] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(160):
        seeds = random.sample(cells, len(sizes))
        regions = [{seed} for seed in seeds]
        capacities = [size - 1 for size in sizes]
        remaining = set(cells) - set(seeds)
        while remaining:
            choices = []
            for idx, region in enumerate(regions):
                if capacities[idx] <= 0:
                    continue
                frontier = _region_frontier(region, remaining)
                if frontier:
                    choices.append((idx, frontier))
            if not choices:
                break
            idx, frontier = random.choice(choices)
            frontier.sort(key=lambda cell: (sum((nr, nc) in remaining for nr, nc in _neighbors4(*cell)), cell[0], cell[1]))
            if random.random() < 0.65:
                cell = frontier[0]
            else:
                cell = random.choice(frontier)
            regions[idx].add(cell)
            remaining.remove(cell)
            capacities[idx] -= 1
        if remaining:
            continue
        if any(len(region) != size for region, size in zip(regions, sizes)):
            continue
        return [np.array(sorted(region), dtype=int) for region in regions]
    return None


def _choose_grey_cells(coords: np.ndarray, k: int, color: int) -> np.ndarray | None:
    cells = [tuple(map(int, rc)) for rc in coords]
    if k == 0:
        patch = _patch_from_coords(coords, color, set())
        return patch if _unique_rotations_count(patch) == 4 else None
    for _ in range(120):
        grey_cells = set(random.sample(cells, k))
        patch = _patch_from_coords(coords, color, grey_cells)
        if _unique_rotations_count(patch) == 4:
            return patch
    return None


def _build_piece_set(region_shape: tuple[int, int], count: int) -> tuple[list[np.ndarray], list[int]] | None:
    area = region_shape[0] * region_shape[1]
    size_options = _size_compositions(area, count)
    random.shuffle(size_options)
    for sizes in size_options[:40]:
        regions = _random_partition(region_shape[0], region_shape[1], sizes)
        if regions is None:
            continue
        k_templates = ([0, 1, 2, 3], [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3])[0 if count == 4 else random.randint(1, 4)]
        if count == 3:
            k_values = list(k_templates)
        else:
            k_values = [0, 1, 2, 3]
        random.shuffle(k_values)

        region_order = list(range(count))
        random.shuffle(region_order)
        pieces: list[np.ndarray] = [None] * count  # type: ignore[list-item]
        colors = random.sample(PIECE_COLORS, count)
        ok = True
        for piece_idx, region_idx in enumerate(region_order):
            patch = _choose_grey_cells(regions[region_idx], k_values[piece_idx], colors[piece_idx])
            if patch is None:
                ok = False
                break
            pieces[piece_idx] = patch
        if not ok:
            continue
        if max(np.count_nonzero(patch) for patch in pieces) > 7:
            continue
        return pieces, k_values
    return None


def _all_orders(pieces: list[np.ndarray]) -> list[tuple[int, ...]]:
    perms = list(itertools.permutations(range(len(pieces))))
    random.shuffle(perms)
    return perms


def _wrong_rotation_changes(region_shape: tuple[int, int], pieces: list[np.ndarray], k_values: list[int]) -> bool:
    correct = _pack_pieces(region_shape, pieces)
    assert correct is not None
    for idx, k in enumerate(k_values):
        if k == 0:
            continue
        wrong = pieces.copy()
        wrong[idx] = _rotate_clockwise(pieces[idx], -k)
        alt = _pack_pieces(region_shape, wrong)
        if alt is None or not np.array_equal(alt, correct):
            return True
    return False


def _alternative_order_changes(region_shape: tuple[int, int], pieces: list[np.ndarray]) -> bool:
    correct = _pack_pieces(region_shape, pieces)
    assert correct is not None
    n = len(pieces)
    for i in range(n - 1):
        alt_order = list(range(n))
        alt_order[i], alt_order[i + 1] = alt_order[i + 1], alt_order[i]
        alt = _pack_pieces(region_shape, [pieces[j] for j in alt_order])
        if alt is None or not np.array_equal(alt, correct):
            return True
    return False


def _layout_rows(patches: list[np.ndarray]) -> list[list[int]]:
    n = len(patches)
    options = {
        3: [[3], [1, 2], [2, 1], [1, 1, 1]],
        4: [[4], [2, 2], [1, 3], [3, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]],
    }[n]
    return [layout for layout in options if sum(layout) == n]


def _distribute_slack(slack: int, slots: int) -> list[int]:
    if slots <= 0:
        return []
    cuts = [0] + sorted(random.sample(range(slack + slots), slots - 1)) + [slack + slots - 1]
    vals = []
    for a, b in zip(cuts, cuts[1:]):
        vals.append(b - a)
    if len(vals) < slots:
        vals.append(0)
    return vals[:slots]


def _place_input_patches(input_patches: list[np.ndarray], target_dims: tuple[int, int]) -> np.ndarray | None:
    layouts = _layout_rows(input_patches)
    random.shuffle(layouts)
    outer_h = target_dims[0] + 2
    outer_w = target_dims[1] + 2
    for layout in layouts:
        row_groups = []
        cursor = 0
        gap_choices = []
        row_heights = []
        row_widths = []
        for row_count in layout:
            idxs = list(range(cursor, cursor + row_count))
            cursor += row_count
            widths = [input_patches[i].shape[1] for i in idxs]
            heights = [input_patches[i].shape[0] for i in idxs]
            gaps = [random.randint(2, 4) for _ in range(max(0, row_count - 1))]
            row_groups.append((idxs, widths, heights, gaps))
            row_heights.append(max(heights))
            row_widths.append(sum(widths) + sum(gaps))
            gap_choices.append(gaps)

        top_margin = random.randint(1, 2)
        side_margin = random.randint(1, 2)
        row_gap = random.randint(2, 3)
        top_height = top_margin + sum(row_heights) + row_gap * (len(layout) - 1)
        frame_top = max(top_height + random.randint(2, 3), outer_h + 1, 8)
        bottom_margin = random.randint(1, 2)
        h = frame_top + outer_h + bottom_margin
        w_min = max(max(row_widths) + side_margin * 2, outer_w + 2 * side_margin, 16)
        w = w_min + random.randint(0, 4)
        if h > 30 or w > 30:
            continue

        if frame_top < h // 2:
            continue

        grid = np.zeros((h, w), dtype=int)
        frame_left = random.randint(1, w - outer_w - 1)
        marker_cells = (
            (frame_top, frame_left),
            (frame_top, frame_left + outer_w - 1),
            (frame_top + outer_h - 1, frame_left),
            (frame_top + outer_h - 1, frame_left + outer_w - 1),
        )
        for r, c in marker_cells:
            grid[r, c] = YELLOW

        rr = top_margin
        ok = True
        for idxs, widths, heights, gaps in row_groups:
            row_w = sum(widths) + sum(gaps)
            start_c = random.randint(side_margin, w - side_margin - row_w)
            cc = start_c
            for local_idx, piece_idx in enumerate(idxs):
                patch = input_patches[piece_idx]
                _paste(grid, patch, rr, cc, transparent=BLACK)
                cc += patch.shape[1]
                if local_idx < len(gaps):
                    cc += gaps[local_idx]
            rr += max(heights) + row_gap

        upper_mask = (grid != BLACK) & (grid != YELLOW)
        comps = _components_mask(upper_mask)
        if len(comps) != len(input_patches):
            ok = False
        if ok:
            ordered_bboxes = sorted((_bbox_coords(comp) for comp in comps), key=lambda b: (b[0], b[1]))
            actual_order = []
            for bbox in ordered_bboxes:
                actual_order.append(bbox)
            for bbox in actual_order:
                if bbox[0] >= frame_top - 1:
                    ok = False
                    break
        if ok:
            return grid
    return None


def _semantic_checks(inp: np.ndarray, out: np.ndarray, target_dims: tuple[int, int]) -> bool:
    if not np.array_equal(main(inp), out):
        return False

    fragments = _extract_fragments(inp)
    if len(fragments) not in (3, 4):
        return False
    if len({frag["base_color"] for frag in fragments}) != len(fragments):
        return False
    if not any(frag["grey_count"] > 0 for frag in fragments):
        return False
    if len({frag["grey_count"] for frag in fragments}) < 2:
        return False
    if any(not (4 <= frag["area"] <= 7) for frag in fragments):
        return False

    yellow = np.argwhere(inp == YELLOW)
    rows = yellow[:, 0]
    cols = yellow[:, 1]
    if (int(rows.max()) - int(rows.min()) - 1, int(cols.max()) - int(cols.min()) - 1) != target_dims:
        return False

    compact = _compact_shape(target_dims[0] * target_dims[1])
    if target_dims == compact:
        return False
    return True


def _generate_instance() -> tuple[np.ndarray, np.ndarray] | None:
    target_dims = random.choice(TARGET_DIMS)
    piece_count = random.choice((3, 4))
    built = _build_piece_set(target_dims, piece_count)
    if built is None:
        return None
    base_pieces, base_k_values = built

    accepted_order = None
    accepted_k = None
    packed = None
    for order in _all_orders(base_pieces):
        pieces = [base_pieces[i] for i in order]
        k_values = [base_k_values[i] for i in order]
        packed = _pack_pieces(target_dims, pieces)
        if packed is None:
            continue
        if not _wrong_rotation_changes(target_dims, pieces, k_values):
            continue
        if not _alternative_order_changes(target_dims, pieces):
            continue
        accepted_order = pieces
        accepted_k = k_values
        break
    if accepted_order is None or accepted_k is None or packed is None:
        return None

    input_patches = [np.rot90(piece, k).copy() for piece, k in zip(accepted_order, accepted_k)]
    inp = _place_input_patches(input_patches, target_dims)
    if inp is None:
        return None

    out = np.zeros_like(inp)
    target_origin, _, marker_cells = _target_interior(inp)
    for r, c in marker_cells:
        out[r, c] = YELLOW
    _paste(out, packed, target_origin[0], target_origin[1], transparent=BLACK)
    if not _semantic_checks(inp, out, target_dims):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _generate_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input for puzzle 8")
