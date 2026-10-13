from __future__ import annotations

import random
from collections import deque

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
PALETTE = [1, 2, 3, 4, 5]
GEN_ATTEMPTS = 400


SOLID_BASES = [
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 0), (0, 1), (1, 1), (1, 2)},
    {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)},
]

CHAIN_BASES = [
    {(0, 0), (1, 1), (2, 2)},
    {(0, 0), (0, 1), (1, 2)},
    {(0, 0), (0, 1), (1, 2), (1, 3)},
    {(0, 0), (1, 0), (1, 1), (2, 2)},
    {(0, 0), (1, 1), (1, 2), (2, 3)},
    {(0, 0), (0, 1), (1, 2), (2, 3), (2, 4)},
    {(0, 0), (1, 1), (2, 2), (2, 3), (3, 4)},
]


def _normalize(coords: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _transform(coords: set[tuple[int, int]], rot: int, flip: bool) -> frozenset[tuple[int, int]]:
    pts = {(int(r), int(c)) for r, c in coords}
    if flip:
        pts = {(r, -c) for r, c in pts}
    for _ in range(rot % 4):
        pts = {(c, -r) for r, c in pts}
    return _normalize(pts)


def _variants(base_shapes: list[set[tuple[int, int]]]) -> list[frozenset[tuple[int, int]]]:
    out: set[frozenset[tuple[int, int]]] = set()
    for shape in base_shapes:
        for rot in range(4):
            for flip in (False, True):
                out.add(_transform(shape, rot, flip))
    return sorted(out, key=lambda shape: (len(shape), tuple(sorted(shape))))


SOLID_SHAPES = _variants(SOLID_BASES)
CHAIN_SHAPES = _variants(CHAIN_BASES)


def _component_coords(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    assert connectivity in (4, 8)
    deltas = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in deltas:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _color_regions(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            region_mask = np.zeros((h, w), dtype=bool)
            for rr, cc in coords:
                region_mask[rr, cc] = True
            lobes = _component_coords(region_mask, connectivity=4)
            regions.append(
                {
                    "color": color,
                    "coords": coords,
                    "size": len(coords),
                    "lobe_count": len(lobes),
                    "kind": "chain" if len(lobes) > 1 else "solid",
                }
            )
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored region by choosing a block rule from its 8-connected / 4-connected topology.

    Input structure:
    - The grid has black background cells and 3-5 non-black monochrome regions.
    - Each colored region is 8-connected, and different regions use different colors.
    - Some regions split into multiple 4-connected lobes joined only by diagonal contacts; others are a single 4-connected blob.

    Rule:
    - Classify every non-black 8-connected region by its internal 4-connectivity.
    - If a region has multiple 4-connected lobes, treat it as a diagonal-chain region; each pixel of that region expands to an input-sized block containing only that color's mask from the input.
    - If a region is one 4-connected blob, treat it as a solid region; each pixel of that region expands to an input-sized block filled entirely with the region color.
    - Black input pixels expand to all-black blocks, and the blocks are tiled in the original row-major order.

    Key insight:
    - The output mixes two familiar expansion rules in one canvas, and the only cue for which rule to use is whether each 8-connected colored region stays connected under 4-neighborhood or falls apart into diagonal-only lobes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    h, w = g.shape
    out = np.zeros((h * h, w * w), dtype=int)
    regions = _color_regions(g)

    region_kind = np.zeros((h, w), dtype=np.int8)
    for region in regions:
        marker = 1 if region["kind"] == "chain" else 2
        for r, c in region["coords"]:
            region_kind[r, c] = marker

    color_masks = {
        color: np.where(g == color, color, 0)
        for color in map(int, np.unique(g))
        if color != 0
    }

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            br0, bc0 = r * h, c * w
            if color == 0:
                continue
            if region_kind[r, c] == 1:
                out[br0 : br0 + h, bc0 : bc0 + w] = color_masks[color]
            else:
                out[br0 : br0 + h, bc0 : bc0 + w] = color
    return out


def _shape_bbox(shape: frozenset[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _translate(shape: frozenset[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r + r0, c + c0) for r, c in shape]


def _fits(coords: list[tuple[int, int]], occupied: set[tuple[int, int]], forbidden: set[tuple[int, int]]) -> bool:
    for cell in coords:
        if cell in occupied or cell in forbidden:
            return False
    return True


def _expand_forbidden(coords: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    banned: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr, dc in NEIGHBORS8 + [(0, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                banned.add((nr, nc))
    return banned


def _bounded_shapes(
    shapes: list[frozenset[tuple[int, int]]],
    *,
    sizes: tuple[int, ...],
    max_h: int,
    max_w: int,
) -> list[frozenset[tuple[int, int]]]:
    picked = []
    for shape in shapes:
        sh, sw = _shape_bbox(shape)
        if len(shape) in sizes and sh <= max_h and sw <= max_w:
            picked.append(shape)
    return picked


def _choose_specs(rng: random.Random, h: int, w: int, num_regions: int) -> list[dict]:
    if num_regions == 4:
        overlap_size = 3
        chain_pool = _bounded_shapes(CHAIN_SHAPES, sizes=(overlap_size,), max_h=h, max_w=w)
        solid_pool = _bounded_shapes(SOLID_SHAPES, sizes=(overlap_size,), max_h=h, max_w=w)
        specs = [
            {"kind": "chain", "shape": rng.choice(chain_pool)},
            {"kind": "chain", "shape": rng.choice(chain_pool)},
            {"kind": "solid", "shape": rng.choice(solid_pool)},
            {"kind": "solid", "shape": rng.choice(solid_pool)},
        ]
        rng.shuffle(specs)
        return specs

    allowed_sizes = (3, 4) if min(h, w) == 4 else (3, 4, 5)
    overlap_size = rng.choice(allowed_sizes)
    chain_pool = _bounded_shapes(CHAIN_SHAPES, sizes=(overlap_size,), max_h=h, max_w=w)
    solid_pool = _bounded_shapes(SOLID_SHAPES, sizes=(overlap_size,), max_h=h, max_w=w)
    specs = [
        {"kind": "chain", "shape": rng.choice(chain_pool)},
        {"kind": "solid", "shape": rng.choice(solid_pool)},
    ]
    chain_count = rng.choice([1, 2])
    solid_count = num_regions - chain_count
    all_chain = _bounded_shapes(CHAIN_SHAPES, sizes=allowed_sizes, max_h=h, max_w=w)
    all_solid = _bounded_shapes(SOLID_SHAPES, sizes=allowed_sizes, max_h=h, max_w=w)
    while sum(spec["kind"] == "chain" for spec in specs) < chain_count:
        specs.append({"kind": "chain", "shape": rng.choice(all_chain)})
    while sum(spec["kind"] == "solid" for spec in specs) < solid_count:
        specs.append({"kind": "solid", "shape": rng.choice(all_solid)})
    rng.shuffle(specs)
    return specs


def _place_specs(h: int, w: int, specs: list[dict], rng: random.Random) -> list[dict] | None:
    order = list(range(len(specs)))
    order.sort(key=lambda idx: (-len(specs[idx]["shape"]), -(_shape_bbox(specs[idx]["shape"])[0] * _shape_bbox(specs[idx]["shape"])[1])))
    placed: list[dict | None] = [None] * len(specs)

    def backtrack(step: int, occupied: set[tuple[int, int]], forbidden: set[tuple[int, int]]) -> bool:
        if step == len(order):
            return True
        idx = order[step]
        shape = specs[idx]["shape"]
        sh, sw = _shape_bbox(shape)
        candidates = [(r0, c0) for r0 in range(h - sh + 1) for c0 in range(w - sw + 1)]
        rng.shuffle(candidates)
        for r0, c0 in candidates:
            coords = _translate(shape, r0, c0)
            if not _fits(coords, occupied, forbidden):
                continue
            new_occupied = occupied | set(coords)
            new_forbidden = forbidden | _expand_forbidden(coords, h, w)
            placed[idx] = {"kind": specs[idx]["kind"], "shape": shape, "coords": coords}
            if backtrack(step + 1, new_occupied, new_forbidden):
                return True
            placed[idx] = None
        return False

    if not backtrack(0, set(), set()):
        return None
    return [item for item in placed if item is not None]


def _independent_expected(inp: np.ndarray) -> np.ndarray:
    h, w = inp.shape
    regions = _color_regions(inp)
    coord_to_kind: dict[tuple[int, int], str] = {}
    for region in regions:
        for cell in region["coords"]:
            coord_to_kind[cell] = region["kind"]
    color_masks = {
        color: np.where(inp == color, color, 0)
        for color in map(int, np.unique(inp))
        if color != 0
    }
    out = np.zeros((h * h, w * w), dtype=int)
    for r in range(h):
        for c in range(w):
            color = int(inp[r, c])
            if color == 0:
                continue
            if coord_to_kind[(r, c)] == "chain":
                block = color_masks[color]
            else:
                block = np.full((h, w), color, dtype=int)
            out[r * h : (r + 1) * h, c * w : (c + 1) * w] = block
    return out


def _semantic_checks(inp: np.ndarray) -> bool:
    if inp.ndim != 2 or min(inp.shape) < 4 or max(inp.shape) > 5:
        return False
    h, w = inp.shape
    if h * h > 30 or w * w > 30:
        return False

    regions = _color_regions(inp)
    if len(regions) not in (3, 4):
        return False

    colors = [region["color"] for region in regions]
    if len(colors) != len(set(colors)):
        return False

    kinds = {region["kind"] for region in regions}
    if kinds != {"chain", "solid"}:
        return False

    sizes = [region["size"] for region in regions]
    if any(size < 3 or size > 6 for size in sizes):
        return False

    chain_sizes = {region["size"] for region in regions if region["kind"] == "chain"}
    solid_sizes = {region["size"] for region in regions if region["kind"] == "solid"}
    if not (chain_sizes & solid_sizes):
        return False

    if np.count_nonzero(inp) < 9:
        return False

    expected = _independent_expected(inp)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    kind_by_cell = {
        cell: region["kind"]
        for region in regions
        for cell in region["coords"]
    }
    for r in range(h):
        for c in range(w):
            block = actual[r * h : (r + 1) * h, c * w : (c + 1) * w]
            color = int(inp[r, c])
            if color == 0:
                if np.any(block != 0):
                    return False
                continue
            kind = kind_by_cell[(r, c)]
            if kind == "solid":
                if not np.all(block == color):
                    return False
            else:
                if not np.array_equal(block, np.where(inp == color, color, 0)):
                    return False
                if np.all(block == color):
                    return False
    return True


def _attempt_generate_profile(rng: random.Random, h: int, w: int, num_regions: int) -> np.ndarray | None:
    for _ in range(40):
        specs = _choose_specs(rng, h, w, num_regions)
        placed = _place_specs(h, w, specs, rng)
        if placed is None:
            continue

        colors = rng.sample(PALETTE, k=len(placed))
        grid = np.zeros((h, w), dtype=int)
        for spec, color in zip(placed, colors, strict=False):
            for r, c in spec["coords"]:
                grid[r, c] = color

        if _semantic_checks(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    profiles = [((5, 5), 3), ((5, 5), 4), ((5, 4), 3), ((4, 5), 3)]
    weights = [4, 3, 2, 2]
    for _ in range(GEN_ATTEMPTS):
        (h, w), num_regions = rng.choices(profiles, weights=weights, k=1)[0]
        grid = _attempt_generate_profile(rng, h, w, num_regions)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
