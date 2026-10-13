from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

MARKER_TO_TURNS = {
    RED: 0,
    GREEN: 1,
    YELLOW: 2,
    BLUE: 3,
}
TURNS_TO_MARKER = {turns: color for color, turns in MARKER_TO_TURNS.items()}
BASE_COLORS = (6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GENERATION_RETRIES = 500
PARTITION_RETRIES = 120
PATCH_PLACEMENT_RETRIES = 120


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    if isinstance(coords, np.ndarray):
        rows = coords[:, 0]
        cols = coords[:, 1]
        return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int = GREY) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            sub[r - r0, c - c0] = True
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.array_equal(sub, border):
            candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, f"expected one grey hollow rectangle, found {len(candidates)}"
    return candidates[0]


def _extract_pieces(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], bg: int) -> list[dict[str, object]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    outside = np.ones(grid.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    piece_mask = (grid != bg) & (grid != GREY) & outside
    pieces: list[dict[str, object]] = []
    for comp in _components(piece_mask):
        colors = [int(grid[r, c]) for r, c in comp]
        marker_cells = [(r, c, col) for (r, c), col in zip(comp, colors) if col in MARKER_TO_TURNS]
        base_colors = sorted(set(colors) - set(MARKER_TO_TURNS))
        assert len(marker_cells) == 1, "each piece must contain exactly one marker pixel"
        assert len(base_colors) == 1, "each piece must be monochrome apart from its marker"
        marker_r, marker_c, marker_color = marker_cells[0]
        base_color = int(base_colors[0])
        turns = MARKER_TO_TURNS[int(marker_color)]
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True
        rotated = np.rot90(mask, -turns)
        pieces.append(
            {
                "color": base_color,
                "turns": turns,
                "size": int(mask.sum()),
                "input_mask": mask,
                "marker_pos": (marker_r - r0, marker_c - c0),
                "mask": rotated.astype(bool),
            }
        )
    pieces.sort(key=lambda piece: (int(piece["color"]), int(piece["size"])))
    return pieces


def _enumerate_placements(mask: np.ndarray, height: int, width: int) -> tuple[list[dict[str, object]], dict[int, list[int]]]:
    h, w = mask.shape
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    placements: list[dict[str, object]] = []
    cover_map = {cell: [] for cell in range(height * width)}
    for r0 in range(height - h + 1):
        for c0 in range(width - w + 1):
            bits = 0
            cells: list[tuple[int, int]] = []
            for dr, dc in coords:
                rr, cc = r0 + dr, c0 + dc
                bits |= 1 << (rr * width + cc)
                cells.append((rr, cc))
            idx = len(placements)
            placements.append({"bits": bits, "coords": tuple(cells), "top": r0, "left": c0})
            for rr, cc in cells:
                cover_map[rr * width + cc].append(idx)
    return placements, cover_map


def _solve_tiling(
    pieces: list[dict[str, object]], height: int, width: int, limit: int = 2
) -> tuple[list[tuple[int, ...]], list[tuple[list[dict[str, object]], dict[int, list[int]]]]]:
    total_area = sum(int(piece["size"]) for piece in pieces)
    assert total_area == height * width, "piece areas must exactly match the frame interior"
    placement_data = [_enumerate_placements(np.asarray(piece["mask"], dtype=bool), height, width) for piece in pieces]
    full_mask = (1 << (height * width)) - 1
    placement_counts = [len(data[0]) for data in placement_data]
    solutions: list[tuple[int, ...]] = []

    def search(occupied: int, used_mask: int, assignment: list[int]) -> None:
        if len(solutions) >= limit:
            return
        if occupied == full_mask:
            solutions.append(tuple(assignment))
            return
        remaining = full_mask & ~occupied
        next_bit = remaining & -remaining
        cell = next_bit.bit_length() - 1
        options: list[tuple[int, int]] = []
        for piece_idx in range(len(pieces)):
            if used_mask & (1 << piece_idx):
                continue
            placements, cover_map = placement_data[piece_idx]
            for placement_idx in cover_map[cell]:
                if placements[placement_idx]["bits"] & occupied:
                    continue
                options.append((piece_idx, placement_idx))
        options.sort(key=lambda item: (placement_counts[item[0]], -int(pieces[item[0]]["size"]), int(pieces[item[0]]["color"])))
        for piece_idx, placement_idx in options:
            assignment[piece_idx] = placement_idx
            bits = int(placement_data[piece_idx][0][placement_idx]["bits"])
            search(occupied | bits, used_mask | (1 << piece_idx), assignment)
            assignment[piece_idx] = -1
            if len(solutions) >= limit:
                return

    search(0, 0, [-1] * len(pieces))
    return solutions, placement_data


def _render_interior(
    height: int,
    width: int,
    pieces: list[dict[str, object]],
    placement_data: list[tuple[list[dict[str, object]], dict[int, list[int]]]],
    assignment: tuple[int, ...],
    bg: int,
) -> np.ndarray:
    interior = np.full((height, width), bg, dtype=int)
    for piece_idx, placement_idx in enumerate(assignment):
        placement = placement_data[piece_idx][0][placement_idx]
        color = int(pieces[piece_idx]["color"])
        for r, c in placement["coords"]:
            interior[int(r), int(c)] = color
    return interior


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered marked fragments into a central grey frame so they exactly tile its interior.

    Input structure:
    - The background is black.
    - A single one-pixel-thick grey hollow rectangle sits near the center; its interior is empty.
    - Outside the frame are several small 4-connected fragments, each mostly one base color but with exactly one marker pixel in red, green, yellow, or blue.
    - Marker colors encode clockwise rotations: red=0 degrees, green=90 degrees, yellow=180 degrees, blue=270 degrees.

    Rule:
    - Find the grey frame and its interior rectangle.
    - For each fragment, read its marker color, restore that marker pixel to the fragment's base color, and rotate the whole fragment by the encoded amount.
    - Translate the rotated fragments inside the frame so they form the unique non-overlapping tiling that exactly fills the frame interior.
    - Keep the grey frame, clear everything outside it to black, and draw the assembled colored fragments inside.

    Key insight:
    - The special marker pixel is not part of the final picture: it is a rotation instruction that fixes each fragment's orientation before the remaining task becomes a translation-only exact packing problem.
    """
    g = np.asarray(grid, dtype=int)
    bg = _majority_color(g)
    frame_bbox = _find_frame(g)
    fr0, fc0, fr1, fc1 = frame_bbox
    interior_h = fr1 - fr0 - 1
    interior_w = fc1 - fc0 - 1
    assert interior_h > 0 and interior_w > 0, "frame must have a non-empty interior"
    pieces = _extract_pieces(g, frame_bbox, bg)
    assert pieces, "expected at least one fragment outside the frame"
    solutions, placement_data = _solve_tiling(pieces, interior_h, interior_w, limit=2)
    assert len(solutions) == 1, f"expected a unique tiling solution, found {len(solutions)}"
    out = np.full_like(g, bg)
    _draw_frame(out, frame_bbox, GREY)
    interior = _render_interior(interior_h, interior_w, pieces, placement_data, solutions[0], bg)
    out[fr0 + 1 : fr1, fc0 + 1 : fc1] = interior
    return out


def _size_compositions(total: int, parts: int, lo: int = 2, hi: int = 6) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)] if lo <= total <= hi else []
    out: list[tuple[int, ...]] = []
    for first in range(lo, hi + 1):
        rest = total - first
        if rest < lo * (parts - 1) or rest > hi * (parts - 1):
            continue
        for tail in _size_compositions(rest, parts - 1, lo=lo, hi=hi):
            out.append((first,) + tail)
    return out


_LAYOUT_OPTIONS = {
    piece_count: [
        (h, w, sizes)
        for h in range(2, 5)
        for w in range(3, 7)
        for sizes in _size_compositions(h * w, piece_count)
    ]
    for piece_count in range(2, 5)
}


def _region_frontier(region: set[tuple[int, int]], unassigned: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    frontier = set()
    for r, c in region:
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in unassigned:
                frontier.add((nr, nc))
    return list(frontier)


def _random_connected_partition(h: int, w: int, sizes: tuple[int, ...]) -> list[set[tuple[int, int]]] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    all_cells = set(cells)
    for _ in range(PARTITION_RETRIES):
        seeds = random.sample(cells, len(sizes))
        regions = [{seed} for seed in seeds]
        remaining = [size - 1 for size in sizes]
        unassigned = all_cells - set(seeds)
        ok = True
        while unassigned:
            frontiers = []
            for idx, region in enumerate(regions):
                if remaining[idx] <= 0:
                    frontiers.append([])
                    continue
                frontier = _region_frontier(region, unassigned, h, w)
                if not frontier:
                    ok = False
                    break
                frontiers.append(frontier)
            if not ok:
                break
            choices = [idx for idx in range(len(sizes)) if remaining[idx] > 0 and frontiers[idx]]
            if not choices:
                ok = False
                break
            choices.sort(key=lambda idx: (len(frontiers[idx]), -remaining[idx], random.random()))
            pool = choices[: min(3, len(choices))]
            region_idx = pool[0] if random.random() < 0.65 else random.choice(pool)
            cell = random.choice(frontiers[region_idx])
            regions[region_idx].add(cell)
            remaining[region_idx] -= 1
            unassigned.remove(cell)
        if ok and not unassigned and all(left == 0 for left in remaining):
            return regions
    return None


def _mask_is_nonrectangular(mask: np.ndarray) -> bool:
    return int(mask.sum()) != mask.shape[0] * mask.shape[1]


def _sample_turns(masks: list[np.ndarray]) -> list[int] | None:
    allowed_per_mask: list[list[int]] = []
    for mask in masks:
        allowed = [0]
        for turns in (1, 2, 3):
            if not np.array_equal(mask, np.rot90(mask, turns)):
                allowed.append(turns)
        allowed_per_mask.append(allowed)
    if not any(any(turns != 0 for turns in allowed) for allowed in allowed_per_mask):
        return None
    for _ in range(80):
        turns = [random.choice(allowed) for allowed in allowed_per_mask]
        if not any(turn != 0 for turn in turns):
            continue
        if len(set(turns)) < 2:
            continue
        return turns
    return None


def _choose_marker_cell(mask: np.ndarray) -> tuple[int, int]:
    h, w = mask.shape
    coords = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    preferred: list[tuple[int, int]] = []
    for r, c in coords:
        degree = sum((r + dr, c + dc) in coords for dr, dc in NEIGHBORS4)
        if degree <= 2 or (r in (0, h - 1) and c in (0, w - 1)):
            preferred.append((r, c))
    candidates = preferred if preferred else sorted(coords)
    return random.choice(candidates)


def _make_input_patch(output_mask: np.ndarray, color: int, turns: int) -> np.ndarray:
    input_mask = np.rot90(output_mask, turns)
    patch = np.full(input_mask.shape, BLACK, dtype=int)
    patch[input_mask] = color
    marker_r, marker_c = _choose_marker_cell(input_mask)
    patch[marker_r, marker_c] = TURNS_TO_MARKER[turns]
    return patch


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _blocked_bbox(bbox: tuple[int, int, int, int], pad: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _place_patches(
    canvas_h: int,
    canvas_w: int,
    frame_bbox: tuple[int, int, int, int],
    patches: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    blocked = [_blocked_bbox(frame_bbox, pad=1)]
    placements: list[tuple[int, int]] = []
    for patch in patches:
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(canvas_h - ph + 1):
            for c0 in range(canvas_w - pw + 1):
                bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if any(_boxes_overlap(bbox, other) for other in blocked):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        spot = random.choice(candidates)
        placements.append(spot)
        blocked.append(_blocked_bbox((spot[0], spot[1], spot[0] + ph - 1, spot[1] + pw - 1), pad=1))
    return placements


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    piece_count = random.randint(2, 4)
    h, w, sizes = random.choice(_LAYOUT_OPTIONS[piece_count])
    regions = _random_connected_partition(h, w, sizes)
    if regions is None:
        return None

    masks = [_tight_mask(region) for region in regions]
    if not any(_mask_is_nonrectangular(mask) for mask in masks):
        return None
    if not any(mask.shape[0] > 1 and mask.shape[1] > 1 for mask in masks):
        return None

    turns = _sample_turns(masks)
    if turns is None:
        return None

    colors = random.sample(BASE_COLORS, piece_count)
    expected_interior = np.full((h, w), BLACK, dtype=int)
    pieces: list[dict[str, object]] = []
    for region, mask, color in zip(regions, masks, colors):
        for r, c in region:
            expected_interior[r, c] = color
        pieces.append({"color": color, "mask": mask, "size": int(mask.sum())})

    solutions, placement_data = _solve_tiling(pieces, h, w, limit=2)
    if len(solutions) != 1:
        return None
    solved_interior = _render_interior(h, w, pieces, placement_data, solutions[0], BLACK)
    if not np.array_equal(solved_interior, expected_interior):
        return None

    frame_h, frame_w = h + 2, w + 2
    canvas_h = random.randint(max(frame_h + 8, 14), min(30, frame_h + 14))
    canvas_w = random.randint(max(frame_w + 8, 15), min(30, frame_w + 14))
    ideal_r = (canvas_h - frame_h) // 2
    ideal_c = (canvas_w - frame_w) // 2
    fr0 = min(max(1, ideal_r + random.randint(-2, 2)), canvas_h - frame_h - 1)
    fc0 = min(max(1, ideal_c + random.randint(-2, 2)), canvas_w - frame_w - 1)
    frame_bbox = (fr0, fc0, fr0 + frame_h - 1, fc0 + frame_w - 1)

    patches = [_make_input_patch(mask, color, turn) for mask, color, turn in zip(masks, colors, turns)]
    for patch, mask, turn in zip(patches, masks, turns):
        if turn != 0 and np.array_equal(patch != BLACK, mask):
            return None

    for _ in range(PATCH_PLACEMENT_RETRIES):
        order = list(range(piece_count))
        random.shuffle(order)
        shuffled_patches = [patches[idx] for idx in order]
        placements = _place_patches(canvas_h, canvas_w, frame_bbox, shuffled_patches)
        if placements is None:
            continue
        inp = np.full((canvas_h, canvas_w), BLACK, dtype=int)
        _draw_frame(inp, frame_bbox, GREY)
        for idx, (r0, c0) in zip(order, placements):
            patch = patches[idx]
            ph, pw = patch.shape
            region = patch != BLACK
            target = inp[r0 : r0 + ph, c0 : c0 + pw]
            target[region] = patch[region]
        out = np.full_like(inp, BLACK)
        _draw_frame(out, frame_bbox, GREY)
        out[fr0 + 1 : fr0 + 1 + h, fc0 + 1 : fc0 + 1 + w] = expected_interior
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, out):
            return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
