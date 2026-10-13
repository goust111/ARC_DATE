from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np


RNG = np.random.default_rng()

ARC_GRID_MAX = 30
GEN_RETRIES = 400
PARTITION_RETRIES = 120
PLACEMENT_RETRIES = 400

GROUP_COUNT_CHOICES = (3, 4)
PIECE_COUNT_CHOICES = (2, 3, 4)
RECT_SIDE_MIN = 2
RECT_SIDE_MAX = 6
MAX_RECT_AREA = 24
INPUT_SIDE_MIN = 22

COLOR_VALUES = tuple(range(10))
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    exclude = exclude or set()
    return [int(c) for c in np.unique(grid) if int(c) not in exclude]


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    shifted = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return tuple(shifted)


def _piece_shape(piece: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    h = 1 + max(r for r, _ in piece)
    w = 1 + max(c for _, c in piece)
    return int(h), int(w)


def _placements_for_piece(
    piece: tuple[tuple[int, int], ...], rect_h: int, rect_w: int
) -> tuple[int, ...] | None:
    piece_h, piece_w = _piece_shape(piece)
    if piece_h > rect_h or piece_w > rect_w:
        return None

    placements: set[int] = set()
    for dr in range(rect_h - piece_h + 1):
        for dc in range(rect_w - piece_w + 1):
            mask = 0
            for r, c in piece:
                idx = (r + dr) * rect_w + (c + dc)
                mask |= 1 << idx
            placements.add(mask)
    return tuple(sorted(placements))


def _can_tile_rectangle(
    pieces: tuple[tuple[tuple[int, int], ...], ...], rect_h: int, rect_w: int
) -> bool:
    placements_by_piece = []
    for piece in pieces:
        placements = _placements_for_piece(piece, rect_h, rect_w)
        if placements is None:
            return False
        placements_by_piece.append(placements)

    order = sorted(range(len(pieces)), key=lambda i: (len(placements_by_piece[i]), -len(pieces[i])))
    placements_by_piece = [placements_by_piece[i] for i in order]

    cover_lookup: list[dict[int, tuple[int, ...]]] = []
    for placements in placements_by_piece:
        cell_to_masks: dict[int, list[int]] = {}
        for mask in placements:
            bits = mask
            while bits:
                lsb = bits & -bits
                idx = lsb.bit_length() - 1
                cell_to_masks.setdefault(idx, []).append(mask)
                bits ^= lsb
        cover_lookup.append({idx: tuple(masks) for idx, masks in cell_to_masks.items()})

    full_mask = (1 << (rect_h * rect_w)) - 1
    all_piece_bits = (1 << len(placements_by_piece)) - 1

    @lru_cache(maxsize=None)
    def search(used_mask: int, remaining_piece_bits: int) -> bool:
        if remaining_piece_bits == 0:
            return used_mask == full_mask

        remaining_cells = full_mask & ~used_mask
        if remaining_cells == 0:
            return False

        target_cell = (remaining_cells & -remaining_cells).bit_length() - 1
        options = []
        bits = remaining_piece_bits
        while bits:
            piece_bit = bits & -bits
            piece_idx = piece_bit.bit_length() - 1
            masks = cover_lookup[piece_idx].get(target_cell, ())
            if masks:
                options.append((len(masks), piece_idx, piece_bit, masks))
            bits ^= piece_bit
        options.sort(key=lambda item: item[0])

        for _count, _piece_idx, piece_bit, masks in options:
            for mask in masks:
                if mask & used_mask:
                    continue
                if search(used_mask | mask, remaining_piece_bits ^ piece_bit):
                    return True
        return False

    return search(0, all_piece_bits)


def _infer_rectangle_dims(
    pieces: list[tuple[tuple[int, int], ...]]
) -> tuple[int, int]:
    norm_pieces = tuple(sorted((tuple(sorted(piece)) for piece in pieces), key=lambda p: (-len(p), p)))
    total_area = sum(len(piece) for piece in norm_pieces)
    candidates: set[tuple[int, int]] = set()

    for h in range(1, int(total_area ** 0.5) + 1):
        if total_area % h != 0:
            continue
        w = total_area // h
        dims = [(h, w)] if h == w else [(h, w), (w, h)]
        for rect_h, rect_w in dims:
            if _can_tile_rectangle(norm_pieces, int(rect_h), int(rect_w)):
                candidates.add((int(rect_h), int(rect_w)))

    assert len(candidates) == 1, f"expected one tilable rectangle, got {sorted(candidates)}"
    return next(iter(candidates))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each color's scattered fragments into its hidden rectangle, then stack the rectangles by area.

    Input structure:
    - A single dominant background color fills most of the grid.
    - Each foreground color appears in 2 to 4 separate 4-connected fragments.
    - All fragments of one color can be translated, without rotation or flipping, to tile one solid axis-aligned rectangle.
    - Different color groups correspond to different hidden rectangle sizes.

    Rule:
    - For each foreground color, collect its fragments and normalize their shapes.
    - Infer the unique rectangle height and width whose cells can be tiled exactly by translated copies of those fragments.
    - Build a new background canvas whose width is the largest inferred rectangle width and whose height is the sum of all inferred rectangle heights plus one separator row between consecutive groups.
    - Sort the rectangles by area from smallest to largest and draw each as a left-aligned solid block in its own color.

    Key insight:
    - The visible fragment positions are distractions; the real signal is the per-color set of shapes, which jointly determines a hidden rectangle that then controls the ordered output layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    bg = get_bg(g)
    groups = []
    for color in colors_present(g, exclude={bg}):
        comps = _component_coords(g == color)
        assert 2 <= len(comps) <= 4, "each color should have 2 to 4 fragments"
        pieces = [_normalize_coords(coords) for coords in comps]
        rect_h, rect_w = _infer_rectangle_dims(pieces)
        groups.append(
            {
                "color": int(color),
                "height": int(rect_h),
                "width": int(rect_w),
                "area": int(rect_h * rect_w),
            }
        )

    assert groups, "expected at least one foreground group"
    groups.sort(key=lambda item: (item["area"], item["height"], item["width"]))

    out_h = sum(item["height"] for item in groups) + len(groups) - 1
    out_w = max(item["width"] for item in groups)
    assert out_h <= ARC_GRID_MAX and out_w <= ARC_GRID_MAX

    out = np.full((out_h, out_w), bg, dtype=int)
    r0 = 0
    for item in groups:
        h = item["height"]
        w = item["width"]
        out[r0 : r0 + h, :w] = item["color"]
        r0 += h + 1
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    start = tuple(int(x) for x in coords[0])
    q = deque([start])
    seen[start] = True
    count = 0
    while q:
        r, c = q.popleft()
        count += 1
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return count == int(mask.sum())


def _initial_partition(rect_h: int, rect_w: int, piece_count: int) -> np.ndarray | None:
    rects = [(0, rect_h, 0, rect_w)]
    while len(rects) < piece_count:
        candidates = [i for i, (r0, r1, c0, c1) in enumerate(rects) if (r1 - r0) > 1 or (c1 - c0) > 1]
        if not candidates:
            return None
        idx = int(RNG.choice(candidates))
        r0, r1, c0, c1 = rects.pop(idx)
        h = r1 - r0
        w = c1 - c0
        orientations = []
        if h > 1:
            orientations.append("h")
        if w > 1:
            orientations.append("v")
        orient = str(RNG.choice(orientations))
        if orient == "h":
            cut = int(RNG.integers(r0 + 1, r1))
            rects.extend([(r0, cut, c0, c1), (cut, r1, c0, c1)])
        else:
            cut = int(RNG.integers(c0 + 1, c1))
            rects.extend([(r0, r1, c0, cut), (r0, r1, cut, c1)])

    labels = np.full((rect_h, rect_w), -1, dtype=int)
    for label, (r0, r1, c0, c1) in enumerate(rects):
        labels[r0:r1, c0:c1] = label
    return labels


def _jiggle_partition(labels: np.ndarray, steps: int) -> np.ndarray:
    out = labels.copy()
    rect_h, rect_w = out.shape
    for _ in range(steps):
        r = int(RNG.integers(0, rect_h))
        c = int(RNG.integers(0, rect_w))
        src = int(out[r, c])
        nbr_labels = {
            int(out[r + dr, c + dc])
            for dr, dc in NEIGHBORS4
            if 0 <= r + dr < rect_h and 0 <= c + dc < rect_w and int(out[r + dr, c + dc]) != src
        }
        if not nbr_labels:
            continue
        dst = int(RNG.choice(sorted(nbr_labels)))
        trial = out.copy()
        trial[r, c] = dst
        src_mask = trial == src
        if src_mask.sum() == 0 or not _is_connected(src_mask):
            continue
        out = trial
    return out


def _partition_rectangle(
    rect_h: int, rect_w: int, piece_count: int
) -> list[tuple[tuple[int, int], ...]] | None:
    for _ in range(PARTITION_RETRIES):
        labels = _initial_partition(rect_h, rect_w, piece_count)
        if labels is None:
            continue
        labels = _jiggle_partition(labels, steps=rect_h * rect_w * 3)
        pieces = []
        valid = True
        for label in range(piece_count):
            coords = np.argwhere(labels == label)
            if len(coords) == 0:
                valid = False
                break
            if not _is_connected(labels == label):
                valid = False
                break
            pieces.append(_normalize_coords(coords))
        if not valid:
            continue
        try:
            inferred = _infer_rectangle_dims(pieces)
        except AssertionError:
            continue
        if inferred == (rect_h, rect_w):
            return pieces
    return None


def _build_expected_output(groups: list[dict[str, object]], bg: int) -> np.ndarray:
    ordered = sorted(groups, key=lambda item: (item["area"], item["height"], item["width"]))
    out_h = sum(int(item["height"]) for item in ordered) + len(ordered) - 1
    out_w = max(int(item["width"]) for item in ordered)
    out = np.full((out_h, out_w), bg, dtype=int)
    r0 = 0
    for item in ordered:
        h = int(item["height"])
        w = int(item["width"])
        out[r0 : r0 + h, :w] = int(item["color"])
        r0 += h + 1
    return out


def _candidate_dims(piece_count: int) -> list[tuple[int, int]]:
    dims = []
    for rect_h in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1):
        for rect_w in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1):
            area = rect_h * rect_w
            if area > MAX_RECT_AREA or area < piece_count * 2:
                continue
            dims.append((rect_h, rect_w))
    RNG.shuffle(dims)
    return dims


def _sample_groups(bg: int) -> list[dict[str, object]] | None:
    group_count = int(RNG.choice(GROUP_COUNT_CHOICES))
    piece_counts = [int(RNG.choice(PIECE_COUNT_CHOICES)) for _ in range(group_count)]
    if sum(piece_counts) > 12:
        return None

    colors = [c for c in COLOR_VALUES if c != bg]
    chosen_colors = [int(c) for c in RNG.choice(colors, size=group_count, replace=False)]

    used_areas: set[int] = set()
    groups = []
    for color, piece_count in zip(chosen_colors, piece_counts):
        picked = None
        for rect_h, rect_w in _candidate_dims(piece_count):
            area = rect_h * rect_w
            if area in used_areas:
                continue
            pieces = _partition_rectangle(rect_h, rect_w, piece_count)
            if pieces is None:
                continue
            picked = {
                "color": int(color),
                "height": int(rect_h),
                "width": int(rect_w),
                "area": int(area),
                "pieces": pieces,
            }
            used_areas.add(area)
            break
        if picked is None:
            return None
        groups.append(picked)
    return groups


def _place_piece(
    grid: np.ndarray,
    blocked: np.ndarray,
    piece: tuple[tuple[int, int], ...],
    color: int,
) -> bool:
    piece_h, piece_w = _piece_shape(piece)
    grid_h, grid_w = grid.shape
    positions = [
        (r, c)
        for r in range(grid_h - piece_h + 1)
        for c in range(grid_w - piece_w + 1)
    ]
    RNG.shuffle(positions)

    for r0, c0 in positions[:PLACEMENT_RETRIES]:
        rr0 = max(0, r0 - 1)
        rr1 = min(grid_h, r0 + piece_h + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(grid_w, c0 + piece_w + 1)
        if blocked[rr0:rr1, cc0:cc1].any():
            continue
        for dr, dc in piece:
            grid[r0 + dr, c0 + dc] = color
        blocked[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    bg = int(RNG.integers(0, 10))
    groups = _sample_groups(bg)
    if groups is None:
        return None

    total_pieces = sum(len(group["pieces"]) for group in groups)
    min_side = max(INPUT_SIDE_MIN, 18 + total_pieces // 2)
    if min_side > ARC_GRID_MAX:
        return None
    grid_h = int(RNG.integers(min_side, ARC_GRID_MAX + 1))
    grid_w = int(RNG.integers(min_side, ARC_GRID_MAX + 1))

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    fragments: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for group in groups:
        color = int(group["color"])
        for piece in group["pieces"]:
            fragments.append((color, piece))
    RNG.shuffle(fragments)

    for color, piece in fragments:
        if not _place_piece(grid, blocked, piece, color):
            return None

    for group in groups:
        color = int(group["color"])
        comps = _component_coords(grid == color)
        if len(comps) != len(group["pieces"]):
            return None
        inferred = _infer_rectangle_dims([_normalize_coords(coords) for coords in comps])
        if inferred != (int(group["height"]), int(group["width"])):
            return None

    expected = _build_expected_output(groups, bg)
    if expected.shape[0] > ARC_GRID_MAX or expected.shape[1] > ARC_GRID_MAX:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
