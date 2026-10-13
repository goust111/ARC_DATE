from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


AZURE = 8
ARC_COLORS = tuple(range(10))
PIECE_COLORS = tuple(c for c in ARC_COLORS if c != AZURE)
MAX_GRID_SIZE = 30


@dataclass(frozen=True)
class Piece:
    color: int
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    order: int


@dataclass(frozen=True)
class Assembly:
    patch: np.ndarray
    first_mask: np.ndarray
    pair: tuple[int, int]
    area: int
    height: int
    width: int
    order_key: tuple[int, int, int, int]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(pts)
    return comps


def _mask_from_points(points: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _extract_pieces(g: np.ndarray) -> list[Piece]:
    pieces: list[Piece] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != AZURE):
        comps = _components(g == color)
        assert len(comps) == 1, "each foreground color must form exactly one 4-connected object"
        mask, bbox = _mask_from_points(comps[0])
        pieces.append(Piece(color=int(color), mask=mask, bbox=bbox, order=-1))

    pieces.sort(key=lambda p: (p.bbox[1], p.bbox[0], p.bbox[3], p.bbox[2]))
    return [
        Piece(color=p.color, mask=p.mask, bbox=p.bbox, order=i)
        for i, p in enumerate(pieces)
    ]


def _transform(mask: np.ndarray, variant: int) -> np.ndarray:
    out = np.rot90(mask, int(variant) // 2)
    if int(variant) % 2:
        out = np.fliplr(out)
    return np.asarray(out, dtype=bool)


def _orientations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for v in range(8):
        m = _transform(mask, v)
        key = (tuple(map(int, m.shape)), m.tobytes())
        if key not in seen:
            seen.add(key)
            variants.append(m)
    return variants


def _placed_cells(mask: np.ndarray, dr: int = 0, dc: int = 0) -> set[tuple[int, int]]:
    return {
        (int(r) + int(dr), int(c) + int(dc))
        for r, c in np.argwhere(mask)
    }


def _assembly_key(assembly: Assembly) -> tuple:
    return (
        int(assembly.height),
        int(assembly.width),
        tuple(map(int, assembly.first_mask.shape)),
        assembly.first_mask.astype(np.uint8).tobytes(),
        assembly.order_key,
    )


def _ordered_assemblies(fixed: Piece, moving: Piece) -> list[Assembly]:
    fixed_cells = _placed_cells(fixed.mask)
    fixed_area = len(fixed_cells)
    assemblies: list[Assembly] = []
    fh, fw = map(int, fixed.mask.shape)

    for moved_mask in _orientations(moving.mask):
        mh, mw = map(int, moved_mask.shape)
        moved_base = _placed_cells(moved_mask)
        for dr in range(-mh, fh + 1):
            for dc in range(-mw, fw + 1):
                moved_cells = {(r + dr, c + dc) for r, c in moved_base}
                if fixed_cells & moved_cells:
                    continue
                union = fixed_cells | moved_cells
                rows = [r for r, _c in union]
                cols = [c for _r, c in union]
                r0, r1 = min(rows), max(rows)
                c0, c1 = min(cols), max(cols)
                height = r1 - r0 + 1
                width = c1 - c0 + 1
                if height * width != len(union):
                    continue

                patch = np.full((height, width), AZURE, dtype=int)
                first_mask = np.zeros((height, width), dtype=bool)
                for r, c in fixed_cells:
                    rr, cc = r - r0, c - c0
                    patch[rr, cc] = int(fixed.color)
                    if fixed.order < moving.order:
                        first_mask[rr, cc] = True
                for r, c in moved_cells:
                    rr, cc = r - r0, c - c0
                    patch[rr, cc] = int(moving.color)
                    if moving.order < fixed.order:
                        first_mask[rr, cc] = True

                assert not np.any(patch == AZURE)
                area = fixed_area + len(moved_cells)
                order_key = (
                    min(fixed.bbox[0], moving.bbox[0]),
                    min(fixed.bbox[1], moving.bbox[1]),
                    min(fixed.order, moving.order),
                    max(fixed.order, moving.order),
                )
                assemblies.append(
                    Assembly(
                        patch=patch,
                        first_mask=first_mask,
                        pair=(min(fixed.order, moving.order), max(fixed.order, moving.order)),
                        area=int(area),
                        height=int(height),
                        width=int(width),
                        order_key=order_key,
                    )
                )
    return assemblies


def _canonical_pair_assembly(a: Piece, b: Piece) -> Assembly | None:
    choices = _ordered_assemblies(a, b) + _ordered_assemblies(b, a)
    if not choices:
        return None
    choices.sort(key=_assembly_key)
    return choices[0]


def _find_matchings(pieces: list[Piece], limit: int | None = None) -> list[list[Assembly]]:
    candidates: dict[tuple[int, int], Assembly] = {}
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            assembly = _canonical_pair_assembly(pieces[i], pieces[j])
            if assembly is not None:
                candidates[(i, j)] = assembly

    def rec(unused: tuple[int, ...]) -> list[list[Assembly]]:
        if not unused:
            return [[]]
        first = unused[0]
        found: list[list[Assembly]] = []
        for pos in range(1, len(unused)):
            second = unused[pos]
            key = (min(first, second), max(first, second))
            assembly = candidates.get(key)
            if assembly is None:
                continue
            rest = tuple(x for x in unused if x not in key)
            for tail in rec(rest):
                found.append([assembly] + tail)
                if limit is not None and len(found) >= limit:
                    return found
        return found

    return rec(tuple(range(len(pieces))))


def _pack_rectangles(rectangles: list[Assembly]) -> np.ndarray:
    rectangles = sorted(
        rectangles,
        key=lambda a: (int(a.area), int(a.height), int(a.width), a.order_key),
    )
    assert len({a.area for a in rectangles}) == len(rectangles), "rectangle areas must be unique"
    out_h = max(int(a.height) for a in rectangles)
    out_w = sum(int(a.width) for a in rectangles) + len(rectangles) - 1
    assert out_h <= MAX_GRID_SIZE and out_w <= MAX_GRID_SIZE, "output must fit ARC bounds"

    out = np.full((out_h, out_w), AZURE, dtype=int)
    col = 0
    for assembly in rectangles:
        h, w = int(assembly.height), int(assembly.width)
        row = out_h - h
        out[row : row + h, col : col + w] = assembly.patch
        col += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Pair complementary fragments into rectangles and pack them by increasing area.

    Input structure:
    - The background is azure.
    - The grid contains 4, 6, or 8 scattered monochrome 4-connected foreground fragments.
    - Every fragment has a distinct color, and each valid pair can be joined into a
      solid rectangle after rotating or flipping one fragment relative to the other.

    Rule:
    - Extract each foreground fragment as a binary shape, ignoring its scattered position.
    - Find the perfect matching where each pair has some rotation/flip and translation
      that makes the two shapes fill one rectangle with no overlap and no holes.
    - Repaint each joined rectangle with the two original fragment colors, using a
      shape-canonical valid orientation when the whole rectangle has symmetric variants.
    - Sort the joined rectangles by area from small to large, then place them left to
      right on azure with one azure separator column and their bottoms aligned.

    Key insight:
    - The colors identify fragments but do not identify the pairs; the pairing comes
      from shape complementation under rotation and reflection, and only after that
      are the completed rectangles sorted by their visible area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= MAX_GRID_SIZE, "ARC grids must be at most 30x30"
    assert np.any(g == AZURE), "expected azure background"

    pieces = _extract_pieces(g)
    assert len(pieces) in (4, 6, 8), "expected 4, 6, or 8 fragments"

    matchings = _find_matchings(pieces, limit=2)
    assert len(matchings) == 1, "fragment pairing must be unambiguous"
    return _pack_rectangles(matchings[0])


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _stair_split(height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(80):
        cuts = [random.randint(1, width - 1)]
        for _r in range(1, height):
            cuts.append(max(1, min(width - 1, cuts[-1] + random.choice([-1, 0, 1]))))
        if len(set(cuts)) == 1:
            continue

        a = np.zeros((height, width), dtype=bool)
        for r, cut in enumerate(cuts):
            a[r, :cut] = True
        b = ~a
        if a.sum() < 3 or b.sum() < 3:
            continue
        if _is_connected(a) and _is_connected(b):
            return a, b
    return None


def _random_nontrivial_transform(mask: np.ndarray) -> np.ndarray:
    variants = []
    original = (tuple(map(int, mask.shape)), mask.tobytes())
    for v in range(1, 8):
        m = _transform(mask, v)
        if (tuple(map(int, m.shape)), m.tobytes()) != original:
            variants.append(m)
    if not variants:
        variants = [_transform(mask, random.randint(1, 7))]
    return random.choice(variants)


def _colored_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, AZURE, dtype=int)
    patch[mask] = int(color)
    return patch


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = map(int, mask.shape)
    if r0 < 0 or c0 < 0 or r0 + h > occupied.shape[0] or c0 + w > occupied.shape[1]:
        return False
    for rr, cc in np.argwhere(mask):
        gr, gc = int(r0 + rr), int(c0 + cc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < occupied.shape[0] and 0 <= nc < occupied.shape[1] and occupied[nr, nc]:
                    return False
    return True


def _paste_piece(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != AZURE
    h, w = map(int, patch.shape)
    grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
    occupied[r0 : r0 + h, c0 : c0 + w] |= mask


def _place_pieces(patches: list[np.ndarray], total_area: int) -> np.ndarray | None:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    lo_h = max(12, max_h + 2)
    lo_w = max(12, max_w + 2)
    if lo_h > MAX_GRID_SIZE or lo_w > MAX_GRID_SIZE:
        return None

    height = random.randint(lo_h, MAX_GRID_SIZE)
    width = random.randint(lo_w, MAX_GRID_SIZE)
    if height * width <= 2 * total_area:
        return None

    grid = np.full((height, width), AZURE, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda i: int((patches[i] != AZURE).sum()), reverse=True)

    for idx in order:
        patch = patches[idx]
        mask = patch != AZURE
        ph, pw = map(int, patch.shape)
        spots = [(r, c) for r in range(height - ph + 1) for c in range(width - pw + 1)]
        random.shuffle(spots)
        placed = False
        for r0, c0 in spots[:250]:
            if _can_place(occupied, mask, r0, c0):
                _paste_piece(grid, occupied, patch, r0, c0)
                placed = True
                break
        if not placed:
            return None
    return grid


def _sample_rect_dims(pair_count: int) -> list[tuple[int, int]] | None:
    dims: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    for _ in range(120):
        dims.clear()
        used_areas.clear()
        for _i in range(pair_count):
            for _j in range(80):
                h = random.randint(3, 7)
                w = random.randint(3, 8)
                area = h * w
                if area in used_areas:
                    continue
                split = _stair_split(h, w)
                if split is None:
                    continue
                dims.append((h, w))
                used_areas.add(area)
                break
            else:
                break
        if len(dims) != pair_count:
            continue
        if sum(max(h, w) for h, w in dims) + pair_count - 1 <= MAX_GRID_SIZE:
            return dims[:]
    return None


def _make_pair_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    split = _stair_split(height, width)
    if split is None:
        return None
    a, b = split
    base_variant = random.randint(0, 7)
    return _transform(a, base_variant), _transform(b, base_variant)


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice([2, 3, 4])
    dims = _sample_rect_dims(pair_count)
    if dims is None:
        return None

    colors = random.sample(PIECE_COLORS, 2 * pair_count)
    patches: list[np.ndarray] = []
    intended_pairs: set[frozenset[int]] = set()
    total_area = 0
    color_index = 0

    for h, w in dims:
        pair_masks = _make_pair_masks(h, w)
        if pair_masks is None:
            return None
        a, b = pair_masks
        if random.random() < 0.5:
            a = _random_nontrivial_transform(a)
        else:
            b = _random_nontrivial_transform(b)

        ca = int(colors[color_index])
        cb = int(colors[color_index + 1])
        color_index += 2
        patches.append(_colored_patch(a, ca))
        patches.append(_colored_patch(b, cb))
        intended_pairs.add(frozenset((ca, cb)))
        total_area += int(a.sum() + b.sum())

    random.shuffle(patches)
    grid = _place_pieces(patches, total_area)
    if grid is None:
        return None

    try:
        pieces = _extract_pieces(grid)
        matchings = _find_matchings(pieces, limit=2)
        if len(matchings) != 1:
            return None
        actual_pairs = {
            frozenset((pieces[assembly.pair[0]].color, pieces[assembly.pair[1]].color))
            for assembly in matchings[0]
        }
        if actual_pairs != intended_pairs:
            return None
        out = main(grid)
    except Exception:
        return None

    if max(grid.shape) > MAX_GRID_SIZE or max(out.shape) > MAX_GRID_SIZE:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid complementary-fragment input")
