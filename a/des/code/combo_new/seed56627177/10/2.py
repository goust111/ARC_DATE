from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
ARC_MAX = 30
OUTPUT_GAP = 1
PLACEMENT_PADDING = 1
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            if color == BLACK or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r1, c1 = min(rows), min(cols)
            r2, c2 = max(rows), max(cols)
            mask = np.zeros((r2 - r1 + 1, c2 - c1 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r1, cc - c1] = True
            comps.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": (r1, c1, r2, c2),
                    "mask": mask,
                    "area": len(coords),
                }
            )

    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _binary_signature(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mask.astype(np.uint8).ravel())


def _partition_signature(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, ...]:
    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())
    if area_a < area_b:
        ref = mask_a
    elif area_b < area_a:
        ref = mask_b
    else:
        ref = mask_a if _binary_signature(mask_a) <= _binary_signature(mask_b) else mask_b
    return _binary_signature(ref)


def _pair_sort_key(assembly: dict) -> tuple:
    return (
        int(assembly["area"]),
        int(assembly["height"]),
        int(assembly["width"]),
        assembly["partition_signature"],
    )


def _valid_rectangles(area_total: int, min_h: int, min_w: int) -> list[tuple[int, int]]:
    rects: list[tuple[int, int]] = []
    for h in range(1, area_total + 1):
        if area_total % h != 0:
            continue
        w = area_total // h
        if h >= min_h and w >= min_w:
            rects.append((h, w))
    rects.sort()
    return rects


def _pair_assemblies(comp_a: dict, comp_b: dict) -> list[dict]:
    mask_a = comp_a["mask"]
    mask_b = comp_b["mask"]
    ha, wa = map(int, mask_a.shape)
    hb, wb = map(int, mask_b.shape)
    area_a = int(comp_a["area"])
    area_b = int(comp_b["area"])
    total = area_a + area_b

    unique: dict[tuple, dict] = {}
    for out_h, out_w in _valid_rectangles(total, max(ha, hb), max(wa, wb)):
        placed_a: list[np.ndarray] = []
        for ra in range(out_h - ha + 1):
            for ca in range(out_w - wa + 1):
                canvas = np.zeros((out_h, out_w), dtype=bool)
                canvas[ra : ra + ha, ca : ca + wa] = mask_a
                placed_a.append(canvas)

        placed_b: list[np.ndarray] = []
        for rb in range(out_h - hb + 1):
            for cb in range(out_w - wb + 1):
                canvas = np.zeros((out_h, out_w), dtype=bool)
                canvas[rb : rb + hb, cb : cb + wb] = mask_b
                placed_b.append(canvas)

        for canvas_a in placed_a:
            for canvas_b in placed_b:
                if np.any(canvas_a & canvas_b):
                    continue
                patch = np.where(canvas_a, int(comp_a["color"]), int(comp_b["color"])).astype(int)
                key = (out_h, out_w, _partition_signature(canvas_a, canvas_b))
                unique.setdefault(
                    key,
                    {
                        "patch": patch,
                        "height": out_h,
                        "width": out_w,
                        "area": total,
                        "partition_signature": key[2],
                    },
                )

    return sorted(unique.values(), key=_pair_sort_key)


def _solve_pairing(components: list[dict]) -> tuple[list[dict], int]:
    n = len(components)
    edge_to_assembly: dict[tuple[int, int], dict] = {}
    adjacency = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            assemblies = _pair_assemblies(components[i], components[j])
            if not assemblies:
                continue
            edge_to_assembly[(i, j)] = assemblies[0]
            adjacency[i].append(j)
            adjacency[j].append(i)

    solutions: list[list[dict]] = []

    def backtrack(used: set[int], current: list[dict]) -> None:
        if len(solutions) >= 2:
            return
        if len(used) == n:
            solutions.append(current.copy())
            return

        candidates = []
        for i in range(n):
            if i in used:
                continue
            partners = [j for j in adjacency[i] if j not in used]
            candidates.append((len(partners), i, partners))

        candidates.sort()
        _, i, partners = candidates[0]
        if not partners:
            return

        for j in partners:
            key = (i, j) if i < j else (j, i)
            current.append(edge_to_assembly[key])
            used.add(i)
            used.add(j)
            backtrack(used, current)
            used.remove(i)
            used.remove(j)
            current.pop()

    backtrack(set(), [])
    return (solutions[0] if solutions else []), len(solutions)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered monochrome fragments by complementary shape and pack the completed rectangles by area.

    Input structure:
    - Black is the background.
    - Several separated 4-connected fragments are scattered across the grid.
    - Each fragment is monochromatic, and the number of fragments is even.
    - Every fragment has exactly one partner such that the two translated shapes can tile a solid rectangle with no overlap.

    Rule:
    - Extract every fragment as its own connected component.
    - Find the unique complementary pairing: for each valid pair, there exists a rectangle whose area equals the sum of the two fragment areas, and the two cropped shapes can be translated inside that rectangle to fill it exactly.
    - Reconstruct each full rectangle while preserving the original fragment colors on their respective cells.
    - Sort the completed rectangles by increasing area and place them left-to-right with a one-column black gap, aligned to the top row.

    Key insight:
    - The important relation is not position or color but shape complementarity under translation: two seemingly unrelated pieces belong together only when they exactly partition one rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g)
    assert components and len(components) % 2 == 0, "expected an even number of fragments"

    assemblies, solution_count = _solve_pairing(components)
    assert assemblies and solution_count == 1, "expected a unique complementary pairing"

    assemblies.sort(key=_pair_sort_key)
    out_h = max(int(a["height"]) for a in assemblies)
    out_w = sum(int(a["width"]) for a in assemblies) + OUTPUT_GAP * (len(assemblies) - 1)
    out = np.full((out_h, out_w), BLACK, dtype=int)

    c = 0
    for assembly in assemblies:
        h = int(assembly["height"])
        w = int(assembly["width"])
        out[:h, c : c + w] = assembly["patch"]
        c += w + OUTPUT_GAP

    return out


RECT_CANDIDATES = tuple(
    (h, w)
    for h in range(3, 7)
    for w in range(3, 7)
    if h * w >= 9
)
PAIR_COUNT_OPTIONS = (2, 3)


def _random_partition(rect_h: int, rect_w: int) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        cuts = [random.randint(1, rect_w - 1) for _ in range(rect_h)]
        while len(set(cuts)) == 1:
            cuts[random.randrange(rect_h)] = random.randint(1, rect_w - 1)
        mask = np.zeros((rect_h, rect_w), dtype=bool)
        for r, cut in enumerate(cuts):
            mask[r, :cut] = True
    else:
        cuts = [random.randint(1, rect_h - 1) for _ in range(rect_w)]
        while len(set(cuts)) == 1:
            cuts[random.randrange(rect_w)] = random.randint(1, rect_h - 1)
        mask = np.zeros((rect_h, rect_w), dtype=bool)
        for c, cut in enumerate(cuts):
            mask[:cut, c] = True

    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.rot90(mask, k=random.randrange(4))

    comp = ~mask
    return mask.astype(bool), comp.astype(bool)


def _mask_to_piece(mask: np.ndarray, color: int) -> np.ndarray:
    cropped = _crop_mask(mask)
    piece = np.zeros(cropped.shape, dtype=int)
    piece[cropped] = int(color)
    return piece


def _is_irregular(mask: np.ndarray) -> bool:
    cropped = _crop_mask(mask)
    h, w = map(int, cropped.shape)
    return h > 1 and w > 1 and int(cropped.sum()) < h * w


def _generate_pair_specs() -> list[dict] | None:
    n_pairs = random.choice(PAIR_COUNT_OPTIONS)
    dims = random.sample(list(RECT_CANDIDATES), n_pairs)
    areas = [h * w for h, w in dims]
    if len(set(areas)) != n_pairs:
        return None

    colors = random.sample(PALETTE, 2 * n_pairs)
    specs: list[dict] = []
    seen_signatures: set[tuple[int, ...]] = set()
    for idx, (rect_h, rect_w) in enumerate(dims):
        mask_a, mask_b = _random_partition(rect_h, rect_w)
        if int(mask_a.sum()) == 0 or int(mask_b.sum()) == 0:
            return None
        if not _is_irregular(mask_a) or not _is_irregular(mask_b):
            return None
        color_a = colors[2 * idx]
        color_b = colors[2 * idx + 1]
        patch = np.where(mask_a, color_a, color_b).astype(int)
        signature = _partition_signature(mask_a, mask_b)
        if signature in seen_signatures:
            return None
        seen_signatures.add(signature)
        specs.append(
            {
                "mask_a": mask_a,
                "mask_b": mask_b,
                "patch": patch,
                "height": patch.shape[0],
                "width": patch.shape[1],
                "area": patch.shape[0] * patch.shape[1],
                "partition_signature": signature,
                "piece_a": _mask_to_piece(mask_a, color_a),
                "piece_b": _mask_to_piece(mask_b, color_b),
            }
        )

    specs.sort(key=_pair_sort_key)
    out_w = sum(spec["width"] for spec in specs) + OUTPUT_GAP * (len(specs) - 1)
    out_h = max(spec["height"] for spec in specs)
    if out_w > ARC_MAX or out_h > ARC_MAX:
        return None
    return specs


def _render_expected(specs: list[dict]) -> np.ndarray:
    out_h = max(spec["height"] for spec in specs)
    out_w = sum(spec["width"] for spec in specs) + OUTPUT_GAP * (len(specs) - 1)
    out = np.full((out_h, out_w), BLACK, dtype=int)
    c = 0
    for spec in specs:
        h = int(spec["height"])
        w = int(spec["width"])
        out[:h, c : c + w] = spec["patch"]
        c += w + OUTPUT_GAP
    return out


def _canvas_shape(pieces: list[np.ndarray]) -> tuple[int, int]:
    max_h = max(piece.shape[0] for piece in pieces)
    max_w = max(piece.shape[1] for piece in pieces)
    padded_area = sum((piece.shape[0] + 2 * PLACEMENT_PADDING) * (piece.shape[1] + 2 * PLACEMENT_PADDING) for piece in pieces)

    min_h = max(max_h + 2, 12)
    min_w = max(max_w + 2, 12)
    for _ in range(50):
        h = random.randint(min_h, ARC_MAX)
        w = random.randint(min_w, ARC_MAX)
        if h * w >= padded_area + 30:
            return h, w
    return ARC_MAX, ARC_MAX


def _place_pieces(pieces: list[np.ndarray]) -> np.ndarray | None:
    pieces = sorted(pieces, key=lambda piece: (piece.shape[0] * piece.shape[1], piece.shape[0], piece.shape[1]), reverse=True)
    h, w = _canvas_shape(pieces)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    for piece in pieces:
        ph, pw = map(int, piece.shape)
        positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions[:250]:
            rr0 = max(0, r0 - PLACEMENT_PADDING)
            cc0 = max(0, c0 - PLACEMENT_PADDING)
            rr1 = min(h, r0 + ph + PLACEMENT_PADDING)
            cc1 = min(w, c0 + pw + PLACEMENT_PADDING)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw][piece != BLACK] = piece[piece != BLACK]
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    specs = _generate_pair_specs()
    if specs is None:
        return None

    expected = _render_expected(specs)
    pieces: list[np.ndarray] = []
    for spec in specs:
        pieces.append(spec["piece_a"])
        pieces.append(spec["piece_b"])

    random.shuffle(pieces)
    grid = _place_pieces(pieces)
    if grid is None:
        return None

    assemblies, solution_count = _solve_pairing(_find_components(grid))
    if solution_count != 1 or not assemblies:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment puzzle")
