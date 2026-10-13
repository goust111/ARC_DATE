from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]


def _background(grid: np.ndarray) -> int:
    """Return the unique most frequent color."""
    counts = Counter(map(int, grid.ravel()))
    ordered = counts.most_common()
    assert len(ordered) == 1 or ordered[0][1] > ordered[1][1], "background must be unique"
    return int(ordered[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (row, column) coordinate arrays."""
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    height, width = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        coords: list[Coord] = []
        while queue:
            r, c = queue.popleft()
            coords.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(np.asarray(coords, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _dihedral(mask: np.ndarray) -> list[np.ndarray]:
    """All distinct tight D4 orientations, in a stable order."""
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        for candidate in (rotated, np.fliplr(rotated)):
            coords = np.argwhere(candidate)
            candidate = _tight_mask(coords)
            key = (candidate.shape, candidate.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(candidate.copy())
    return variants


def _foreground_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        for coords in _components(grid == color):
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(len(coords)),
                    "mask": _tight_mask(coords),
                }
            )
    return objects


def _analyze(grid: np.ndarray) -> tuple[int, dict, np.ndarray, dict, np.ndarray]:
    """Extract the base, its bbox-relative notch, and the unique oriented match."""
    bg = _background(grid)
    objects = _foreground_objects(grid, bg)
    assert len(objects) >= 3, "expected a base, one matching piece, and distractors"

    largest = max(obj["area"] for obj in objects)
    bases = [obj for obj in objects if obj["area"] == largest]
    assert len(bases) == 1, "base must be the unique largest object"
    base = bases[0]
    r0, c0, r1, c1 = base["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((sub == bg) | (sub == base["color"])), "base bbox must contain only base and notch"

    notch = sub == bg
    notch_parts = _components(notch)
    assert len(notch_parts) == 1, "the base must have exactly one connected notch"
    notch = _tight_mask(notch_parts[0])
    nr0, nc0, nr1, nc1 = _bbox(notch_parts[0])
    touches = (nr0 == 0, nr1 == sub.shape[0] - 1, nc0 == 0, nc1 == sub.shape[1] - 1)
    assert sum(touches) == 1, "notch must open through exactly one base edge"
    assert not bool(notch.all()), "notch must be non-rectangular"

    matches: list[tuple[dict, np.ndarray]] = []
    for piece in objects:
        if piece is base:
            continue
        for oriented in _dihedral(piece["mask"]):
            if oriented.shape == notch.shape and np.array_equal(oriented, notch):
                matches.append((piece, oriented))
    assert len(matches) == 1, "exactly one piece in exactly one orientation must complement the notch"
    piece, oriented = matches[0]
    return bg, base, notch, piece, oriented


def main(grid: np.ndarray) -> np.ndarray:
    """Dock the uniquely complementary piece into a base's edge notch, discard clutter, and tight-crop.

    Input structure:
    - The unique largest 4-connected monochrome object is a rectangular base with
      one non-rectangular background notch opening through exactly one bbox edge.
    - Separate smaller monochrome components are equal-bbox-sized pieces; exactly
      one piece matches the bbox-relative notch under one of the eight D4 orientations.
    - Other pieces are near-matching distractors, and the dominant color is background.

    Rule:
    - Compare every tight piece mask with the notch mask under four rotations and
      their reflections, using coordinates relative to the base bbox and notch bbox.
    - Transform the unique match into the matching orientation, place it in the
      notch in its own color, delete all external pieces, and crop to the base bbox.

    Key insight:
    - The destination is negative space cut from the base: only exact positive-to-
      negative D4 shape equality identifies both the correct piece and its orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a nonempty 2D grid"
    bg, base, notch, piece, oriented = _analyze(g)
    r0, c0, r1, c1 = base["bbox"]
    out = g[r0 : r1 + 1, c0 : c1 + 1].copy()

    notch_coords = np.argwhere(out == bg)
    nr0, nc0, nr1, nc1 = _bbox(notch_coords)
    target = out[nr0 : nr1 + 1, nc0 : nc1 + 1]
    assert oriented.shape == target.shape and np.array_equal(oriented, target == bg)
    target[oriented] = int(piece["color"])
    return out


def _connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return mask.shape, mask.tobytes()


def _random_notch() -> np.ndarray | None:
    """Grow a small asymmetric, edge-spanning, non-rectangular shape."""
    height = random.randint(3, 5)
    width = random.randint(3, 5)
    area = random.randint(max(height, width) + 1, min(height * width - 2, 11))
    mask = np.zeros((height, width), dtype=bool)
    mask[random.randrange(height), random.randrange(width)] = True
    while int(mask.sum()) < area:
        frontier = {
            (nr, nc)
            for r, c in np.argwhere(mask)
            for nr, nc in ((int(r) - 1, int(c)), (int(r) + 1, int(c)), (int(r), int(c) - 1), (int(r), int(c) + 1))
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]
        }
        if not frontier:
            return None
        mask[random.choice(tuple(frontier))] = True
    coords = np.argwhere(mask)
    if _bbox(coords) != (0, 0, height - 1, width - 1):
        return None
    if mask.all() or len(_dihedral(mask)) != 8:
        return None
    return mask


def _near_misses(correct: np.ndarray) -> list[np.ndarray]:
    """Enumerate connected same-bbox masks differing in exactly one or two cells."""
    candidates: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    flat_cells = list(itertools.product(range(correct.shape[0]), range(correct.shape[1])))
    notch_keys = {_shape_key(v) for v in _dihedral(correct)}
    for difference in (1, 2):
        for toggles in itertools.combinations(flat_cells, difference):
            mask = correct.copy()
            for r, c in toggles:
                mask[r, c] = not mask[r, c]
            if not _connected(mask):
                continue
            if _bbox(np.argwhere(mask)) != (0, 0, mask.shape[0] - 1, mask.shape[1] - 1):
                continue
            if any(_shape_key(v) in notch_keys for v in _dihedral(mask)):
                continue
            key = _shape_key(mask)
            if key not in seen:
                seen.add(key)
                candidates.append(mask)
    return candidates


def _place_rect(blocked: np.ndarray, height: int, width: int, gap: int = 1) -> Coord | None:
    choices: list[Coord] = []
    H, W = blocked.shape
    for r in range(H - height + 1):
        for c in range(W - width + 1):
            rr0, cc0 = max(0, r - gap), max(0, c - gap)
            rr1, cc1 = min(H, r + height + gap), min(W, c + width + gap)
            if not blocked[rr0:rr1, cc0:cc1].any():
                choices.append((r, c))
    return random.choice(choices) if choices else None


def _gen_attempt() -> np.ndarray | None:
    base_h, base_w = random.randint(7, 11), random.randint(8, 14)
    notch = _random_notch()
    if notch is None or int(notch.sum()) >= base_h * base_w // 3:
        return None
    nh, nw = notch.shape
    if nh > base_h - 2 or nw > base_w - 2:
        return None

    side = random.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        nr = 0 if side == "top" else base_h - nh
        nc = random.randint(1, base_w - nw - 1)
    else:
        nr = random.randint(1, base_h - nh - 1)
        nc = 0 if side == "left" else base_w - nw
    base_mask = np.ones((base_h, base_w), dtype=bool)
    base_mask[nr : nr + nh, nc : nc + nw][notch] = False
    if not _connected(base_mask):
        return None

    # Use every non-identity D4 branch; asymmetry makes the inverse match unique.
    transformed_options = _dihedral(notch)[1:]
    correct = random.choice(transformed_options)
    near = _near_misses(correct)
    distractor_count = random.randint(2, min(4, len(near))) if len(near) >= 2 else 0
    if distractor_count < 2:
        return None
    distractors = random.sample(near, distractor_count)

    canvas_h = random.randint(max(18, base_h + correct.shape[0] + 5), 26)
    canvas_w = random.randint(max(20, base_w + correct.shape[1] + 5), 30)
    palette = list(range(10))
    random.shuffle(palette)
    bg = palette.pop()
    base_color = palette.pop()
    piece_colors = [palette.pop() for _ in range(distractor_count + 1)]
    grid = np.full((canvas_h, canvas_w), bg, dtype=int)
    blocked = np.zeros(grid.shape, dtype=bool)

    base_pos = _place_rect(blocked, base_h, base_w, gap=1)
    if base_pos is None:
        return None
    br, bc = base_pos
    base_view = grid[br : br + base_h, bc : bc + base_w]
    base_view[base_mask] = base_color
    blocked[br : br + base_h, bc : bc + base_w] = True

    pieces = [(correct, piece_colors[0])] + list(zip(distractors, piece_colors[1:]))
    random.shuffle(pieces)
    for mask, color in pieces:
        ph, pw = mask.shape
        pos = _place_rect(blocked, ph, pw, gap=1)
        if pos is None:
            return None
        pr, pc = pos
        grid[pr : pr + ph, pc : pc + pw][mask] = color
        blocked[pr : pr + ph, pc : pc + pw] = True

    if _background(grid) != bg:
        return None
    expected = np.full((base_h, base_w), base_color, dtype=int)
    expected[nr : nr + nh, nc : nc + nw][notch] = piece_colors[0]
    try:
        solved = main(grid)
        analysis = _analyze(grid)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(solved, expected):
        return None
    if analysis[3]["color"] != piece_colors[0]:
        return None
    # Semantic checks: the input truly contains close nonmatches and a nontrivial transform.
    if any(min(np.count_nonzero(mask ^ correct), 99) not in (1, 2) for mask in distractors):
        return None
    if np.array_equal(correct, notch):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse valid scene with a unique exact complement and close distractors."""
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid notch-docking input")
