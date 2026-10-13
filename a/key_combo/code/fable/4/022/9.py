from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300

PIECE_COUNT_CHOICES = (2, 3, 4)
AMBIGUOUS_PAIR_PROB = 0.75
SHAPE_HEIGHT_RANGE = (2, 4)
SHAPE_WIDTH_RANGE = (2, 4)
SHAPE_CELL_RANGE = (4, 7)
SLAB_INNER_H_RANGE = (6, 11)
SLAB_INNER_W_RANGE = (12, 22)
TOP_EXTRA_ROWS_MAX = 4
GRID_EXTRA_WIDTH_RANGE = (2, 6)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    components: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for coords in _components_from_mask(grid == int(color)):
            bbox = _bbox_from_coords(coords)
            components.append(
                {
                    "color": int(color),
                    "coords": np.asarray(coords, dtype=int),
                    "bbox": bbox,
                    "area": len(coords),
                }
            )
    return components


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    norm = _normalize_mask(mask)
    return int(norm.shape[0]), int(norm.shape[1]), norm.astype(np.uint8).tobytes()


def _has_hole(mask: np.ndarray) -> bool:
    bg_components = _components_from_mask(~mask)
    h, w = mask.shape
    for comp in bg_components:
        if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in comp):
            return True
    return False


def _rotate_for_rank(mask: np.ndarray, rank: int) -> np.ndarray:
    k = 3 if rank % 2 == 0 else 1
    return _normalize_mask(np.rot90(mask, k=k))


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    view = grid[top : top + h, left : left + w]
    view[mask] = int(color)


def _can_place_mask(forbidden: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    return not bool(np.any(forbidden[top : top + h, left : left + w] & mask))


def _mark_forbidden(forbidden: np.ndarray, mask: np.ndarray, top: int, left: int, padding: int) -> None:
    rows, cols = forbidden.shape
    for dr, dc in np.argwhere(mask):
        rr = int(top + dr)
        cc = int(left + dc)
        r0 = max(0, rr - padding)
        r1 = min(rows, rr + padding + 1)
        c0 = max(0, cc - padding)
        c1 = min(cols, cc + padding + 1)
        forbidden[r0:r1, c0:c1] = True


def _place_masks(
    canvas_h: int,
    canvas_w: int,
    masks: list[np.ndarray],
    *,
    padding: int,
    fixed_tops: list[int] | None = None,
) -> list[tuple[int, int]] | None:
    order = sorted(
        range(len(masks)),
        key=lambda i: (
            -int(masks[i].sum()),
            -(int(masks[i].shape[0]) * int(masks[i].shape[1])),
        ),
    )
    positions: list[tuple[int, int] | None] = [None] * len(masks)
    forbidden = np.zeros((canvas_h, canvas_w), dtype=bool)

    def search(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        mask = masks[idx]
        h, w = map(int, mask.shape)
        if h > canvas_h or w > canvas_w:
            return False

        if fixed_tops is None:
            candidates = [(r, c) for r in range(canvas_h - h + 1) for c in range(canvas_w - w + 1)]
        else:
            top = int(fixed_tops[idx])
            if top < 0 or top + h > canvas_h:
                return False
            candidates = [(top, c) for c in range(canvas_w - w + 1)]
        random.shuffle(candidates)

        for top, left in candidates:
            if not _can_place_mask(forbidden, mask, top, left):
                continue
            snapshot = forbidden.copy()
            _mark_forbidden(forbidden, mask, top, left, padding)
            positions[idx] = (int(top), int(left))
            if search(pos + 1):
                return True
            forbidden[:] = snapshot
            positions[idx] = None
        return False

    if not search(0):
        return None
    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _random_polyomino(*, require_parity_sensitive: bool) -> np.ndarray:
    for _ in range(200):
        raw_h = random.randint(*SHAPE_HEIGHT_RANGE)
        raw_w = random.randint(*SHAPE_WIDTH_RANGE)
        max_cells = min(SHAPE_CELL_RANGE[1], raw_h * raw_w - 1)
        if max_cells < SHAPE_CELL_RANGE[0]:
            continue
        n_cells = random.randint(SHAPE_CELL_RANGE[0], max_cells)

        start = (random.randrange(raw_h), random.randrange(raw_w))
        cells = {start}
        while len(cells) < n_cells:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < raw_h and 0 <= nc < raw_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != n_cells:
            continue

        mask = np.zeros((raw_h, raw_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _normalize_mask(mask)
        if min(mask.shape) < 2:
            continue
        if bool(mask.all()):
            continue
        if _has_hole(mask):
            continue
        if require_parity_sensitive:
            if _mask_signature(np.rot90(mask, k=3)) == _mask_signature(np.rot90(mask, k=1)):
                continue
        return mask
    raise RuntimeError("failed to sample a valid polyomino")


def _sample_positive_gaps(total: int, parts: int) -> list[int]:
    assert total >= parts >= 1
    gaps = [1] * parts
    for _ in range(total - parts):
        gaps[random.randrange(parts)] += 1
    return gaps


def _choose_piece_masks(n_pieces: int) -> list[np.ndarray] | None:
    pieces: list[np.ndarray | None] = [None] * n_pieces
    used_cavity_sigs: set[tuple[int, int, bytes]] = set()

    odd_ranks = [i for i in range(n_pieces) if i % 2 == 0]
    even_ranks = [i for i in range(n_pieces) if i % 2 == 1]
    if even_ranks and random.random() < AMBIGUOUS_PAIR_PROB:
        odd_rank = random.choice(odd_ranks)
        even_rank = random.choice(even_ranks)
        for _ in range(80):
            mask = _random_polyomino(require_parity_sensitive=True)
            odd_sig = _mask_signature(_rotate_for_rank(mask, odd_rank))
            even_sig = _mask_signature(_rotate_for_rank(mask, even_rank))
            if odd_sig == even_sig:
                continue
            pieces[odd_rank] = mask.copy()
            pieces[even_rank] = mask.copy()
            used_cavity_sigs.add(odd_sig)
            used_cavity_sigs.add(even_sig)
            break
        else:
            return None

    for rank in range(n_pieces):
        if pieces[rank] is not None:
            continue
        for _ in range(120):
            mask = _random_polyomino(require_parity_sensitive=False)
            sig = _mask_signature(_rotate_for_rank(mask, rank))
            if sig in used_cavity_sigs:
                continue
            pieces[rank] = mask
            used_cavity_sigs.add(sig)
            break
        else:
            return None

    return [mask.copy() for mask in pieces if mask is not None]


def _choose_top_rows(piece_masks: list[np.ndarray], slab_h: int) -> tuple[int, list[int]] | None:
    heights = [int(mask.shape[0]) for mask in piece_masks]
    min_top_h = sum(heights) + len(piece_masks) + 1
    extra_cap = min(TOP_EXTRA_ROWS_MAX, 30 - slab_h - min_top_h)
    if extra_cap < 0:
        return None
    top_h = min_top_h + random.randint(0, extra_cap)
    gaps = _sample_positive_gaps(top_h - sum(heights), len(piece_masks) + 1)
    top_rows: list[int] = []
    row = gaps[0]
    for height, gap_after in zip(heights, gaps[1:]):
        top_rows.append(int(row))
        row += int(height) + int(gap_after)
    return int(top_h), top_rows


def _build_expected_output(
    shape: tuple[int, int],
    slab_color: int,
    slab_top: int,
    slab_left: int,
    slab_h: int,
    slab_w: int,
    cavity_masks: list[np.ndarray],
    cavity_positions: list[tuple[int, int]],
    piece_colors: list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[slab_top : slab_top + slab_h, slab_left : slab_left + slab_w] = int(slab_color)
    for color, mask, (inner_r, inner_c) in zip(piece_colors, cavity_masks, cavity_positions):
        top = slab_top + 1 + int(inner_r)
        left = slab_left + 1 + int(inner_c)
        _paint_mask(out, mask, top, left, int(color))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate upper pieces by alternating parity and fill the matching slab cavities.

    Input structure:
    - The dominant background surrounds a large monochrome slab component in the lower part of the grid.
    - Background-colored 4-connected cavities lie strictly inside that slab.
    - Above the slab sit separated solid monochrome pieces whose bbox top rows are all distinct.

    Rule:
    - Sort the upper pieces by top bbox row.
    - Rotate the 1st, 3rd, 5th, ... pieces clockwise by 90 degrees and the 2nd, 4th, 6th, ... pieces counterclockwise by 90 degrees.
    - Match each rotated piece to the cavity with the same normalized shape, erase the source piece, and fill that cavity with the piece color.
    - Leave the rest of the slab unchanged.

    Key insight:
    - The target cavity is determined only after applying the odd/even rotation schedule; same-shape pieces in opposite parities can map to different cavities.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    components = _find_monochrome_components(g, bg)
    assert components, "expected a slab and at least one source piece"

    slab = max(components, key=lambda comp: int(comp["area"]))
    sr0, sc0, sr1, sc1 = map(int, slab["bbox"])
    slab_mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
    slab_coords = slab["coords"]
    slab_mask[slab_coords[:, 0] - sr0, slab_coords[:, 1] - sc0] = True

    cavity_lookup: dict[tuple[int, int, bytes], list[dict]] = defaultdict(list)
    local_bg = ~slab_mask
    for coords in _components_from_mask(local_bg):
        if any(r in (0, local_bg.shape[0] - 1) or c in (0, local_bg.shape[1] - 1) for r, c in coords):
            continue
        cr0, cc0, cr1, cc1 = _bbox_from_coords(coords)
        cavity_mask = np.zeros((cr1 - cr0 + 1, cc1 - cc0 + 1), dtype=bool)
        for r, c in coords:
            cavity_mask[r - cr0, c - cc0] = True
        cavity_lookup[_mask_signature(cavity_mask)].append(
            {
                "bbox": (sr0 + cr0, sc0 + cc0, sr0 + cr1, sc0 + cc1),
                "mask": cavity_mask,
            }
        )

    pieces = [comp for comp in components if comp is not slab]
    assert pieces, "expected source pieces above the slab"
    assert all(int(comp["bbox"][2]) < sr0 for comp in pieces), "source pieces must sit above the slab"

    top_rows = [int(comp["bbox"][0]) for comp in pieces]
    assert len(set(top_rows)) == len(top_rows), "piece order must be uniquely determined by top rows"
    ordered = sorted(pieces, key=lambda comp: int(comp["bbox"][0]))

    out = g.copy()
    for comp in ordered:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for rank, comp in enumerate(ordered):
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = g[pr0 : pr1 + 1, pc0 : pc1 + 1]
        piece_mask = patch == int(comp["color"])
        rotated = _rotate_for_rank(piece_mask, rank)
        sig = _mask_signature(rotated)
        assert cavity_lookup[sig], "each rotated piece must match one cavity"
        cavity = cavity_lookup[sig].pop()
        cr0, cc0, cr1, cc1 = map(int, cavity["bbox"])
        out[cr0 : cr1 + 1, cc0 : cc1 + 1][cavity["mask"]] = int(comp["color"])

    assert all(not remaining for remaining in cavity_lookup.values()), "every cavity must be filled exactly once"
    return out


def _generate_attempt() -> np.ndarray | None:
    n_pieces = random.choice(PIECE_COUNT_CHOICES)
    piece_masks = _choose_piece_masks(n_pieces)
    if piece_masks is None:
        return None

    cavity_masks = [_rotate_for_rank(mask, rank) for rank, mask in enumerate(piece_masks)]
    cavity_sigs = [_mask_signature(mask) for mask in cavity_masks]
    if len(set(cavity_sigs)) != len(cavity_sigs):
        return None

    for _ in range(80):
        inner_h = random.randint(
            max(SLAB_INNER_H_RANGE[0], max(int(mask.shape[0]) for mask in cavity_masks)),
            SLAB_INNER_H_RANGE[1],
        )
        inner_w = random.randint(
            max(SLAB_INNER_W_RANGE[0], max(int(mask.shape[1]) for mask in cavity_masks)),
            SLAB_INNER_W_RANGE[1],
        )
        cavity_positions = _place_masks(inner_h, inner_w, cavity_masks, padding=1)
        if cavity_positions is not None:
            break
    else:
        return None

    slab_h = inner_h + 2
    slab_w = inner_w + 2
    top_layout = _choose_top_rows(piece_masks, slab_h)
    if top_layout is None:
        return None
    top_h, top_rows = top_layout

    width_min = max(slab_w, max(int(mask.shape[1]) for mask in piece_masks) + 2)
    width = width_min + random.randint(*GRID_EXTRA_WIDTH_RANGE)
    if width > 30:
        return None
    slab_left = random.randint(0, width - slab_w)
    source_positions = _place_masks(top_h, width, piece_masks, padding=1, fixed_tops=top_rows)
    if source_positions is None:
        return None

    height = top_h + slab_h
    if height > 30:
        return None

    available_colors = [c for c in PALETTE if c != BG]
    slab_color = random.choice(available_colors)
    piece_colors = random.sample([c for c in available_colors if c != slab_color], n_pieces)

    grid = _build_expected_output(
        (height, width),
        slab_color,
        top_h,
        slab_left,
        slab_h,
        slab_w,
        [],
        [],
        [],
    )
    for mask, (inner_r, inner_c) in zip(cavity_masks, cavity_positions):
        top = top_h + 1 + int(inner_r)
        left = slab_left + 1 + int(inner_c)
        _paint_mask(grid, mask, top, left, BG)
    for color, mask, (top, left) in zip(piece_colors, piece_masks, source_positions):
        _paint_mask(grid, mask, int(top), int(left), int(color))

    expected = _build_expected_output(
        (height, width),
        slab_color,
        top_h,
        slab_left,
        slab_h,
        slab_w,
        cavity_masks,
        cavity_positions,
        piece_colors,
    )

    if _get_bg(grid) != BG:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
