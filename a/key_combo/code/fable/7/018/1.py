from __future__ import annotations

import random
from collections import deque
from itertools import combinations

import numpy as np


BACKGROUND = 0
GRAY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 1200


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive bounding box, using the common ARC (row, column) convention."""
    points = np.argwhere(mask)
    if points.size == 0:
        raise ValueError("empty mask has no bounding box")
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(array: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        mask = array != BACKGROUND
    r0, c0, r1, c1 = _bbox(mask)
    return array[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as coordinate arrays."""
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    height, width = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        cells: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _normalized_mask(mask: np.ndarray) -> np.ndarray:
    return _crop(mask.astype(bool), mask)


def _packing_options(a: np.ndarray, b: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """All non-overlapping relative placements, deduplicated after bbox normalization."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    ha, wa = a.shape
    hb, wb = b.shape
    ar, ac = np.nonzero(a)
    br, bc = np.nonzero(b)
    a_cells = set(zip(map(int, ar), map(int, ac)))
    options: dict[bytes, tuple[int, np.ndarray]] = {}

    # An area-minimal placement must have overlapping or touching bounding-box
    # projections; any larger separation can be translated closer.
    for dr in range(-hb, ha + 1):
        for dc in range(-wb, wa + 1):
            b_cells = {(int(r) + dr, int(c) + dc) for r, c in zip(br, bc)}
            if not a_cells.isdisjoint(b_cells):
                continue
            union = a_cells | b_cells
            r0 = min(r for r, _ in union)
            c0 = min(c for _, c in union)
            r1 = max(r for r, _ in union)
            c1 = max(c for _, c in union)
            assembly = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.int8)
            for r, c in a_cells:
                assembly[r - r0, c - c0] = 1
            for r, c in b_cells:
                assembly[r - r0, c - c0] = 2
            holes = int(assembly.size - np.count_nonzero(assembly))
            options.setdefault(assembly.tobytes() + repr(assembly.shape).encode(), (holes, assembly))
    return list(options.values())


def _unique_best_packing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    options = _packing_options(a, b)
    if not options:
        raise AssertionError("the two pieces have no legal placement")
    best_score = min(score for score, _ in options)
    best = [assembly for score, assembly in options if score == best_score]
    assert len(best) == 1, "expected a unique densest minimum-bbox packing"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the gray sprite keyed by the negative space of two uniquely densest-packed pieces.

    Input structure:
    - A black grid contains two separated, differently colored large monochromatic pieces.
    - It also contains three or four separated gray candidate sprites of equal area.
    - Every object uses ordinary grid coordinates; pieces may only be translated, never rotated or reflected.

    Rule:
    - Try every non-overlapping relative translation of the two large pieces and choose the unique
      placement whose bounding box contains the fewest black cells.
    - Treat every black cell inside that assembled bounding box as one negative-space mask.
    - Return the bounding-box crop of the unique gray candidate translation-congruent to that mask.

    Key insight:
    - No visible property of a candidate identifies the answer: the key is the unpainted cavity
      created only after the two colored pieces are assembled in their unique densest placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    piece_colors = [int(color) for color in np.unique(g) if int(color) not in (BACKGROUND, GRAY)]
    assert len(piece_colors) == 2, "expected exactly two non-gray foreground colors"
    piece_masks: list[np.ndarray] = []
    for color in piece_colors:
        mask = g == color
        assert len(_components(mask)) == 1, "each large color must form one 4-connected piece"
        piece_masks.append(_normalized_mask(mask))

    gray_components = _components(g == GRAY)
    assert len(gray_components) in (3, 4), "expected three or four gray candidates"
    candidate_masks: list[np.ndarray] = []
    candidate_crops: list[np.ndarray] = []
    for coords in gray_components:
        mask = np.zeros(g.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        norm = _normalized_mask(mask)
        candidate_masks.append(norm)
        crop = np.zeros(norm.shape, dtype=int)
        crop[norm] = GRAY
        candidate_crops.append(crop)
    areas = {int(np.count_nonzero(mask)) for mask in candidate_masks}
    assert len(areas) == 1, "gray candidates must have equal area"

    assembly = _unique_best_packing(piece_masks[0], piece_masks[1])
    cavity = assembly == 0
    assert cavity.any(), "the densest assembly must leave a negative-space key"
    cavity = _normalized_mask(cavity)
    matches = [i for i, mask in enumerate(candidate_masks) if np.array_equal(mask, cavity)]
    assert len(matches) == 1, "the cavity must match exactly one gray candidate"
    return candidate_crops[matches[0]]


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


_POLYOMINO_CACHE: dict[tuple[int, int, int], list[np.ndarray]] = {}


def _tight_polyominoes(h: int, w: int, area: int) -> list[np.ndarray]:
    """Enumerate this small finite family once, avoiding probabilistic stalls."""
    key = (h, w, area)
    if key not in _POLYOMINO_CACHE:
        shapes: list[np.ndarray] = []
        for cells in combinations(range(h * w), area):
            mask = np.zeros((h, w), dtype=bool)
            mask.flat[list(cells)] = True
            if _bbox(mask) == (0, 0, h - 1, w - 1) and _is_connected(mask):
                shapes.append(mask)
        _POLYOMINO_CACHE[key] = shapes
    return _POLYOMINO_CACHE[key]


def _make_candidate_set(count: int) -> list[np.ndarray] | None:
    h, w = random.choice(((3, 3), (3, 4), (4, 3)))
    low = max(h, w, 5)
    high = min(h * w - 2, 9)
    if low > high:
        return None
    area = random.randint(low, high)
    options = _tight_polyominoes(h, w, area)
    if len(options) < count:
        return None
    return [mask.copy() for mask in random.sample(options, count)]


def _split_around_cavity(cavity_shape: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Embed a cavity and partition the remaining rectangle into two connected pieces."""
    ch, cw = cavity_shape.shape
    height = random.randint(max(6, ch + 3), 8)
    width = random.randint(max(6, cw + 3), 9)
    r0 = random.randint(1, height - ch - 1)
    c0 = random.randint(1, width - cw - 1)
    cavity = np.zeros((height, width), dtype=bool)
    cavity[r0 : r0 + ch, c0 : c0 + cw] = cavity_shape

    # A gently varying seam makes both pieces visibly irregular. The cavity is
    # allowed to cross the seam, so both pieces affect its final boundary.
    for _ in range(160):
        vertical = random.random() < 0.5
        if vertical:
            seam = [random.randint(2, width - 2)]
            for _r in range(1, height):
                seam.append(max(2, min(width - 2, seam[-1] + random.choice((-1, 0, 0, 1)))))
            a = np.fromfunction(lambda r, c: c < np.asarray(seam, dtype=int)[r.astype(int)],
                                (height, width), dtype=int).astype(bool)
        else:
            seam = [random.randint(2, height - 2)]
            for _c in range(1, width):
                seam.append(max(2, min(height - 2, seam[-1] + random.choice((-1, 0, 0, 1)))))
            a = np.fromfunction(lambda r, c: r < np.asarray(seam, dtype=int)[c.astype(int)],
                                (height, width), dtype=int).astype(bool)
        b = ~a
        a &= ~cavity
        b &= ~cavity
        if min(int(a.sum()), int(b.sum())) <= int(cavity_shape.sum()):
            continue
        if _is_connected(a) and _is_connected(b):
            return a, b, cavity
    return None


def _place_sprites(sprites: list[np.ndarray], height: int, width: int) -> np.ndarray | None:
    """Place bbox sprites with a one-cell moat, so every visual role stays separate."""
    order = sorted(range(len(sprites)), key=lambda i: sprites[i].size, reverse=True)
    placements: dict[int, tuple[int, int]] | None = None
    # Bounded randomized restarts avoid an unlucky exponential backtracking tree.
    for _ in range(40):
        trial: dict[int, tuple[int, int]] = {}
        boxes: list[tuple[int, int, int, int]] = []
        for i in order:
            sh, sw = sprites[i].shape
            positions = [(r, c) for r in range(height - sh + 1) for c in range(width - sw + 1)]
            random.shuffle(positions)
            chosen = None
            for r, c in positions:
                box = (r, c, r + sh - 1, c + sw - 1)
                if all(box[2] + 1 < q[0] or q[2] + 1 < box[0] or
                       box[3] + 1 < q[1] or q[3] + 1 < box[1] for q in boxes):
                    chosen = (r, c, box)
                    break
            if chosen is None:
                break
            r, c, box = chosen
            trial[i] = (r, c)
            boxes.append(box)
        if len(trial) == len(sprites):
            placements = trial
            break
    if placements is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for i, sprite in enumerate(sprites):
        r, c = placements[i]
        view = grid[r : r + sprite.shape[0], c : c + sprite.shape[1]]
        mask = sprite != BACKGROUND
        view[mask] = sprite[mask]
    return grid


def _generation_attempt() -> np.ndarray | None:
    candidate_count = random.choice((3, 4))
    candidates = _make_candidate_set(candidate_count)
    if candidates is None:
        return None
    random.shuffle(candidates)
    target_index = random.randrange(candidate_count)
    target = candidates[target_index]

    split = _split_around_cavity(target)
    if split is None:
        return None
    a_full, b_full, _ = split
    a = _normalized_mask(a_full)
    b = _normalized_mask(b_full)
    try:
        assembly = _unique_best_packing(a, b)
    except AssertionError:
        return None
    cavity = assembly == 0
    if not cavity.any() or not _is_connected(cavity):
        return None
    cavity_norm = _normalized_mask(cavity)
    if not np.array_equal(cavity_norm, target):
        return None
    if sum(np.array_equal(cavity_norm, mask) for mask in candidates) != 1:
        return None

    color_a, color_b = random.sample(PALETTE, 2)
    piece_a = np.zeros(a.shape, dtype=int)
    piece_b = np.zeros(b.shape, dtype=int)
    piece_a[a] = color_a
    piece_b[b] = color_b
    candidate_sprites = []
    for mask in candidates:
        sprite = np.zeros(mask.shape, dtype=int)
        sprite[mask] = GRAY
        candidate_sprites.append(sprite)

    # Random sprite order and random target location defeat ordering/position shortcuts.
    sprites = [piece_a, piece_b] + candidate_sprites
    canvas_h = random.randint(26, 30)
    canvas_w = random.randint(26, 30)
    grid = _place_sprites(sprites, canvas_h, canvas_w)
    if grid is None:
        return None

    expected = candidate_sprites[target_index]
    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate diverse, independently validated negative-space selection examples."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_GENERATION_ATTEMPTS} attempts")
