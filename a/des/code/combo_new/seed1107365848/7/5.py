from __future__ import annotations

from collections import deque
import random

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 1  # blue
CLUE_COLOR = 2   # red
PIECE_COLORS = (3, 4, 5, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 300


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _cropped_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _outline_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if len(coords) != 2 * (h + w) - 4:
        return None
    pts = set(coords)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if ((r, c) in pts) != on_border:
                return None
    return r0, c0, r1, c1


def _find_outline_bboxes(grid: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == color):
        bbox = _outline_bbox(comp)
        if bbox is not None:
            out.append(bbox)
    return out


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _frame_clue_adjacent(
    frame_bbox: tuple[int, int, int, int],
    clue_bbox: tuple[int, int, int, int],
) -> bool:
    fr0, fc0, fr1, fc1 = frame_bbox
    rr0, rc0, rr1, rc1 = clue_bbox
    if 0 <= fr0 - rr1 - 1 <= 1 and _interval_overlap(fc0, fc1, rc0, rc1) > 0:
        return True
    if 0 <= rr0 - fr1 - 1 <= 1 and _interval_overlap(fc0, fc1, rc0, rc1) > 0:
        return True
    if 0 <= fc0 - rc1 - 1 <= 1 and _interval_overlap(fr0, fr1, rr0, rr1) > 0:
        return True
    if 0 <= rc0 - fc1 - 1 <= 1 and _interval_overlap(fr0, fr1, rr0, rr1) > 0:
        return True
    return False


def _pair_clues(
    frame_bboxes: list[tuple[int, int, int, int]],
    clue_bboxes: list[tuple[int, int, int, int]],
) -> dict[int, int]:
    assert len(frame_bboxes) == len(clue_bboxes) and frame_bboxes
    candidates: dict[int, list[int]] = {}
    for i, frame_bbox in enumerate(frame_bboxes):
        matches = [j for j, clue_bbox in enumerate(clue_bboxes) if _frame_clue_adjacent(frame_bbox, clue_bbox)]
        assert matches, "every blue frame must have an adjacent red clue"
        candidates[i] = matches

    order = sorted(range(len(frame_bboxes)), key=lambda i: (len(candidates[i]), frame_bboxes[i]))
    solutions: list[dict[int, int]] = []

    def dfs(pos: int, used: set[int], mapping: dict[int, int]) -> None:
        if len(solutions) >= 2:
            return
        if pos == len(order):
            solutions.append(mapping.copy())
            return
        frame_idx = order[pos]
        for clue_idx in candidates[frame_idx]:
            if clue_idx in used:
                continue
            used.add(clue_idx)
            mapping[frame_idx] = clue_idx
            dfs(pos + 1, used, mapping)
            del mapping[frame_idx]
            used.remove(clue_idx)

    dfs(0, set(), {})
    assert len(solutions) == 1, "frame-to-clue matching must be unique"
    return solutions[0]


def _enumerate_placements(mask: np.ndarray, target_shape: tuple[int, int]) -> list[int]:
    target_h, target_w = target_shape
    piece_h, piece_w = mask.shape
    if piece_h > target_h or piece_w > target_w:
        return []
    coords = np.argwhere(mask)
    placements: list[int] = []
    for r0 in range(target_h - piece_h + 1):
        for c0 in range(target_w - piece_w + 1):
            bits = 0
            for dr, dc in coords:
                idx = (r0 + int(dr)) * target_w + (c0 + int(dc))
                bits |= 1 << idx
            placements.append(bits)
    return placements


def _count_tilings(piece_masks: list[np.ndarray], target_shape: tuple[int, int], limit: int) -> int:
    target_h, target_w = target_shape
    total_cells = target_h * target_w
    if sum(int(mask.sum()) for mask in piece_masks) != total_cells:
        return 0
    placements = [_enumerate_placements(mask, target_shape) for mask in piece_masks]
    if any(not options for options in placements):
        return 0

    order = sorted(range(len(piece_masks)), key=lambda i: (len(placements[i]), -int(piece_masks[i].sum())))
    ordered_placements = [placements[i] for i in order]
    full_mask = (1 << total_cells) - 1
    count = 0

    def dfs(occupied: int, unused: tuple[int, ...]) -> None:
        nonlocal count
        if count >= limit:
            return
        if occupied == full_mask:
            if not unused:
                count += 1
            return
        if not unused:
            return

        remaining = full_mask & ~occupied
        lowbit = remaining & -remaining
        best_piece = None
        best_options = None
        for piece_idx in unused:
            options = [p for p in ordered_placements[piece_idx] if (p & lowbit) and not (p & occupied)]
            if not options:
                continue
            if best_options is None or len(options) < len(best_options):
                best_piece = piece_idx
                best_options = options
        if best_piece is None or best_options is None:
            return

        next_unused = tuple(i for i in unused if i != best_piece)
        for placement in best_options:
            dfs(occupied | placement, next_unused)
            if count >= limit:
                return

    dfs(0, tuple(range(len(ordered_placements))))
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each framed set of fragments into the solid rectangle indicated by its nearby clue frame.

    Input structure:
    - The background is black.
    - One or more blue one-pixel-thick hollow rectangles enclose work areas.
    - Inside each blue frame, several disconnected same-color fragments sit on black cells.
    - A small red hollow rectangle sits just outside each blue frame and encodes the target rectangle size.

    Rule:
    - For every blue frame, pair it with its adjacent red rectangle clue.
    - Read the red clue's height and width as the dimensions of the rectangle that the interior fragments must tile using translation only.
    - Verify that the fragments can fill that rectangle without overlap, then clear their original positions.
    - Draw the resulting solid rectangle in the fragments' color, anchored to the blue frame's inner top-left corner.
    - Keep the blue frames, red clues, and the rest of the grid unchanged.

    Key insight:
    The red outline gives the target rectangle dimensions, but the colored scraps still have to be interpreted as translatable puzzle pieces that jointly tile that rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_bboxes = _find_outline_bboxes(g, FRAME_COLOR)
    clue_bboxes = _find_outline_bboxes(g, CLUE_COLOR)
    pairing = _pair_clues(frame_bboxes, clue_bboxes)

    out = g.copy()
    for frame_idx, clue_idx in pairing.items():
        fr0, fc0, fr1, fc1 = frame_bboxes[frame_idx]
        rr0, rc0, rr1, rc1 = clue_bboxes[clue_idx]
        target_h = rr1 - rr0 + 1
        target_w = rc1 - rc0 + 1
        inner = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
        inner_h, inner_w = inner.shape
        assert target_h <= inner_h and target_w <= inner_w

        colors = sorted(set(map(int, np.unique(inner))) - {BG_COLOR, FRAME_COLOR, CLUE_COLOR})
        assert len(colors) == 1, "each frame must contain fragments of a single color"
        piece_color = colors[0]

        piece_mask = inner == piece_color
        piece_components = _components(piece_mask)
        assert len(piece_components) >= 2, "fragments must be disconnected pieces"
        piece_masks = [_cropped_mask(comp) for comp in piece_components]
        assert _count_tilings(piece_masks, (target_h, target_w), limit=1) > 0

        out_inner = out[fr0 + 1 : fr1, fc0 + 1 : fc1]
        out_inner[out_inner == piece_color] = BG_COLOR
        out_inner[:target_h, :target_w] = piece_color

    return out


# --------------- generator (jigsaw-rect) ---------------


def _draw_outline(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _random_composition(total: int, parts: int, min_each: int) -> list[int]:
    remaining = total
    out: list[int] = []
    for i in range(parts - 1):
        lo = min_each
        hi = remaining - min_each * (parts - i - 1)
        out.append(random.randint(lo, hi))
        remaining -= out[-1]
    out.append(remaining)
    random.shuffle(out)
    return out


def _grow_partition(h: int, w: int, targets: list[int]) -> np.ndarray | None:
    k = len(targets)
    labels = np.full((h, w), -1, dtype=int)
    seeds = random.sample([(r, c) for r in range(h) for c in range(w)], k)
    frontiers = [set() for _ in range(k)]
    sizes = [1] * k

    def add_frontier(label: int, r: int, c: int) -> None:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1:
                frontiers[label].add((int(nr), int(nc)))

    for label, (r, c) in enumerate(seeds):
        labels[r, c] = label
    for label, (r, c) in enumerate(seeds):
        add_frontier(label, r, c)

    remaining = h * w - k
    while remaining > 0:
        eligible = [i for i in range(k) if frontiers[i] and sizes[i] < targets[i]]
        if not eligible:
            eligible = [i for i in range(k) if frontiers[i]]
        if not eligible:
            return None
        weights = [max(1, targets[i] - sizes[i]) for i in eligible]
        label = random.choices(eligible, weights=weights, k=1)[0]
        cell = random.choice(tuple(frontiers[label]))
        frontiers[label].remove(cell)
        r, c = cell
        if labels[r, c] != -1:
            continue
        labels[r, c] = label
        sizes[label] += 1
        remaining -= 1
        for other in range(k):
            frontiers[other].discard(cell)
        add_frontier(label, r, c)

    return labels


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return mask.shape, mask.astype(np.uint8).tobytes()


def _generate_piece_masks(target_shape: tuple[int, int], piece_count: int) -> list[np.ndarray] | None:
    target_h, target_w = target_shape
    area = target_h * target_w
    if area < 2 * piece_count:
        return None
    require_irregular = random.random() < 0.75
    for _ in range(250):
        targets = _random_composition(area, piece_count, 2)
        labels = _grow_partition(target_h, target_w, targets)
        if labels is None:
            continue
        masks: list[np.ndarray] = []
        for label in range(piece_count):
            coords = [(int(r), int(c)) for r, c in np.argwhere(labels == label)]
            if len(coords) < 2:
                break
            masks.append(_cropped_mask(coords))
        if len(masks) != piece_count:
            continue
        if len({_mask_signature(mask) for mask in masks}) != piece_count:
            continue
        if require_irregular and all(bool(mask.all()) for mask in masks):
            continue
        if _count_tilings(masks, target_shape, limit=2) != 1:
            continue
        return masks
    return None


def _scatter_masks(
    masks: list[np.ndarray],
    inner_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    inner_h, inner_w = inner_shape
    target_h, target_w = target_shape
    coords_by_piece = [np.argwhere(mask) for mask in masks]
    order = sorted(range(len(masks)), key=lambda i: (-int(masks[i].sum()), masks[i].shape))
    candidates: dict[int, list[tuple[int, int, list[tuple[int, int]]]]] = {}
    for i in order:
        piece_h, piece_w = masks[i].shape
        options: list[tuple[int, int, list[tuple[int, int]]]] = []
        for r0 in range(inner_h - piece_h + 1):
            for c0 in range(inner_w - piece_w + 1):
                cells = [(r0 + int(dr), c0 + int(dc)) for dr, dc in coords_by_piece[i]]
                options.append((r0, c0, cells))
        random.shuffle(options)
        candidates[i] = options

    occupied = np.zeros((inner_h, inner_w), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}

    def can_place(cells: list[tuple[int, int]]) -> bool:
        for r, c in cells:
            if occupied[r, c]:
                return False
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < inner_h and 0 <= nc < inner_w and occupied[nr, nc]:
                    return False
        return True

    def fill(cells: list[tuple[int, int]], value: bool) -> None:
        for r, c in cells:
            occupied[r, c] = value

    def dfs(pos: int) -> bool:
        if pos == len(order):
            rect = np.zeros_like(occupied)
            rect[:target_h, :target_w] = True
            return not np.array_equal(occupied, rect)
        piece_idx = order[pos]
        for r0, c0, cells in candidates[piece_idx]:
            if not can_place(cells):
                continue
            fill(cells, True)
            placements[piece_idx] = (r0, c0)
            if dfs(pos + 1):
                return True
            del placements[piece_idx]
            fill(cells, False)
        return False

    if not dfs(0):
        return None
    return [placements[i] for i in range(len(masks))]


def _build_region_block(
    target_shape: tuple[int, int],
    piece_masks: list[np.ndarray],
    inner_shape: tuple[int, int],
    piece_color: int,
    clue_side: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    target_h, target_w = target_shape
    inner_h, inner_w = inner_shape
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    if clue_side in ("top", "bottom"):
        block_h = frame_h + 1 + target_h
        block_w = frame_w
        frame_r = target_h + 1 if clue_side == "top" else 0
        frame_c = 0
        clue_r = 0 if clue_side == "top" else frame_h + 1
        clue_c = random.randint(0, frame_w - target_w)
    else:
        block_h = frame_h
        block_w = frame_w + 1 + target_w
        frame_r = 0
        frame_c = target_w + 1 if clue_side == "left" else 0
        clue_r = random.randint(0, frame_h - target_h)
        clue_c = 0 if clue_side == "left" else frame_w + 1

    placements = _scatter_masks(piece_masks, inner_shape, target_shape)
    if placements is None:
        return None

    inp = np.zeros((block_h, block_w), dtype=int)
    out = np.zeros((block_h, block_w), dtype=int)
    _draw_outline(inp, frame_r, frame_c, frame_h, frame_w, FRAME_COLOR)
    _draw_outline(out, frame_r, frame_c, frame_h, frame_w, FRAME_COLOR)
    _draw_outline(inp, clue_r, clue_c, target_h, target_w, CLUE_COLOR)
    _draw_outline(out, clue_r, clue_c, target_h, target_w, CLUE_COLOR)

    inner_r = frame_r + 1
    inner_c = frame_c + 1
    for mask, (r0, c0) in zip(piece_masks, placements):
        for dr, dc in np.argwhere(mask):
            inp[inner_r + r0 + int(dr), inner_c + c0 + int(dc)] = piece_color

    out[inner_r : inner_r + target_h, inner_c : inner_c + target_w] = piece_color
    return inp, out


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    return a[2] + margin < b[0] or b[2] + margin < a[0] or a[3] + margin < b[1] or b[3] + margin < a[1]


def _place_blocks(blocks: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    canvas_in = np.zeros((30, 30), dtype=int)
    canvas_out = np.zeros((30, 30), dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []

    for inp_block, out_block in blocks:
        bh, bw = inp_block.shape
        found = False
        for _ in range(400):
            r0 = random.randint(1, 29 - bh)
            c0 = random.randint(1, 29 - bw)
            box = (r0, c0, r0 + bh - 1, c0 + bw - 1)
            if all(_boxes_separated(box, other, margin=2) for other in placed_boxes):
                canvas_in[r0 : r0 + bh, c0 : c0 + bw] = inp_block
                canvas_out[r0 : r0 + bh, c0 : c0 + bw] = out_block
                placed_boxes.append(box)
                found = True
                break
        if not found:
            return None

    coords = np.argwhere(canvas_in != BG_COLOR)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    top_margin = random.randint(0, min(3, r0))
    left_margin = random.randint(0, min(3, c0))
    bottom_margin = random.randint(0, min(3, 29 - r1))
    right_margin = random.randint(0, min(3, 29 - c1))
    rs = r0 - top_margin
    cs = c0 - left_margin
    re = r1 + bottom_margin + 1
    ce = c1 + right_margin + 1
    return canvas_in[rs:re, cs:ce], canvas_out[rs:re, cs:ce]


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    region_count = random.choice((1, 2, 3))
    piece_colors = random.sample(PIECE_COLORS, region_count)
    blocks: list[tuple[np.ndarray, np.ndarray]] = []

    for idx in range(region_count):
        piece_color = piece_colors[idx]
        max_dim = 4 if region_count == 3 else 5
        for _ in range(120):
            target_h = random.randint(3, max_dim)
            target_w = random.randint(3, max_dim)
            area = target_h * target_w
            max_pieces = min(4, area // 2)
            if max_pieces < 2:
                continue
            piece_count = random.choice(tuple(range(2, max_pieces + 1)))
            piece_masks = _generate_piece_masks((target_h, target_w), piece_count)
            if piece_masks is None:
                continue

            base_extra = 2 if region_count == 3 else 3
            extra_h = base_extra + random.randint(0, 2) + (1 if piece_count == 4 and random.random() < 0.5 else 0)
            extra_w = base_extra + random.randint(0, 2) + (1 if piece_count == 4 and random.random() < 0.5 else 0)
            inner_h = target_h + extra_h
            inner_w = target_w + extra_w
            if inner_h + 2 >= 30 or inner_w + 2 >= 30:
                continue

            clue_side = random.choice(("top", "bottom", "left", "right"))
            block = _build_region_block((target_h, target_w), piece_masks, (inner_h, inner_w), piece_color, clue_side)
            if block is not None:
                blocks.append(block)
                break
        if len(blocks) != idx + 1:
            return None

    placed = _place_blocks(blocks)
    if placed is None:
        return None
    return placed


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        generated = _gen_attempt()
        if generated is None:
            continue
        inp, out = generated
        try:
            result = main(inp)
        except Exception:
            continue
        if np.array_equal(result, out):
            return inp
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
