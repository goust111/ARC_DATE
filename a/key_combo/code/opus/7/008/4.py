from __future__ import annotations

import hashlib
import random

import numpy as np


BG = 5
CENTER = 2
# The prompt names five colors, but 13x13 needs six distinct foreground colors.
PALETTE = (1, 3, 4, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 2000

_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _extract_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, CENTER)):
        color_comps = _components_mask(grid == color)
        assert len(color_comps) == 1, "each foreground color must form exactly one component"
        coords = color_comps[0]
        top_left = coords[np.lexsort((coords[:, 1], coords[:, 0]))][0]
        comps.append(
            {
                "color": color,
                "coords": coords,
                "area": int(coords.shape[0]),
                "top_left": (int(top_left[0]), int(top_left[1])),
            }
        )
    return comps


def _ring_output(side: int, ordered_colors: list[int] | tuple[int, ...]) -> np.ndarray:
    center = side // 2
    rr, cc = np.indices((side, side))
    dist = np.maximum(np.abs(rr - center), np.abs(cc - center))
    out = np.full((side, side), BG, dtype=int)
    out[center, center] = CENTER
    for d, color in enumerate(ordered_colors, start=1):
        out[dist == d] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the whole canvas into concentric square rings whose palette comes from component sizes.

    Input structure:
    - The grid is an odd square with a grey background and a single red center pixel.
    - Every other foreground color forms one small 4-connected monochromatic component.
    - Those component colors are all distinct and their physical positions are arbitrary.

    Rule:
    - Find every non-grey, non-red 4-connected single-color component.
    - Sort those components by area in ascending order.
    - Repaint the full output as center-aligned concentric square rings, where Chebyshev distance d from the red center uses the color ranked d by component size.
    - Keep the center pixel red and overwrite the original scattered components and background.

    Key insight:
    - Component position does not matter at all; only area ranking determines the color sequence that the concentric rings use.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected an odd square grid"
    side = int(g.shape[0])
    assert side % 2 == 1 and side <= 30

    center = side // 2
    assert int(g[center, center]) == CENTER, "missing red center marker"
    assert int(np.sum(g == CENTER)) == 1, "center marker must be unique"

    comps = _extract_components(g)
    assert len(comps) == center, "number of components must equal the ring count"

    areas = [comp["area"] for comp in comps]
    assert len(set(areas)) == len(areas), "component areas must be unique"

    ordered = sorted(comps, key=lambda comp: comp["area"])
    ordered_colors = [int(comp["color"]) for comp in ordered]
    return _ring_output(side, ordered_colors)


def _signature(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def _normalize_patch(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_polyomino(area: int) -> np.ndarray:
    cells = {(0, 0)}
    frontier = {(0, 1), (1, 0), (-1, 0), (0, -1)}
    while len(cells) < area:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nbr in _neighbors4(*cell):
            if nbr not in cells:
                frontier.add(nbr)

    arr = np.array(sorted(cells), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    patch = np.zeros((int(arr[:, 0].max()) + 1, int(arr[:, 1].max()) + 1), dtype=bool)
    patch[arr[:, 0], arr[:, 1]] = True

    if random.random() < 0.5:
        patch = np.fliplr(patch)
    patch = np.rot90(patch, k=random.randrange(4))
    return _normalize_patch(patch)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _place_patch(
    grid: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    color: int,
) -> np.ndarray | None:
    h, w = patch.shape
    gh, gw = grid.shape
    positions = [(r0, c0) for r0 in range(gh - h + 1) for c0 in range(gw - w + 1)]
    random.shuffle(positions)
    for r0, c0 in positions:
        region = blocked[r0 : r0 + h, c0 : c0 + w]
        if bool(np.any(region & patch)):
            continue
        abs_coords = np.argwhere(patch) + np.array([[r0, c0]])
        for rr, cc in abs_coords:
            grid[int(rr), int(cc)] = int(color)
        placed = np.zeros_like(blocked)
        placed[abs_coords[:, 0], abs_coords[:, 1]] = True
        blocked |= _dilate8(placed)
        return abs_coords
    return None


def _sample_areas(k: int) -> list[int]:
    upper = 7 if k == 5 else 8
    limit = 24 if k == 5 else 30
    for _ in range(200):
        areas = sorted(random.sample(range(1, upper + 1), k))
        if sum(areas) <= limit and max(areas) >= k:
            return areas
    return list(range(1, k + 1))


def _describe_components(grid: np.ndarray) -> list[dict]:
    center = grid.shape[0] // 2
    desc = []
    for comp in _extract_components(grid):
        coords = comp["coords"]
        cheb = np.max(np.abs(coords - np.array([center, center])), axis=1)
        mean_r = float(coords[:, 0].mean())
        mean_c = float(coords[:, 1].mean())
        desc.append(
            {
                **comp,
                "nearest_center": int(cheb.min()),
                "farthest_center": int(cheb.max()),
                "centroid": (mean_r, mean_c),
            }
        )
    return desc


def _sequence_from(desc: list[dict], key_fn) -> tuple[int, ...]:
    ordered = sorted(desc, key=key_fn)
    return tuple(int(comp["color"]) for comp in ordered)


def _semantic_checks(grid: np.ndarray) -> bool:
    side = int(grid.shape[0])
    center = side // 2
    desc = _describe_components(grid)
    if len(desc) != center:
        return False

    areas = [int(comp["area"]) for comp in desc]
    if len(set(areas)) != len(areas):
        return False

    correct = _sequence_from(desc, lambda comp: (comp["area"], comp["top_left"][0], comp["top_left"][1]))
    if correct == tuple(sorted(correct)) or correct == tuple(sorted(correct, reverse=True)):
        return False

    quadrants = set()
    for r, c in np.argwhere(grid != BG):
        rr = int(r)
        cc = int(c)
        if (rr, cc) == (center, center) or rr == center or cc == center:
            continue
        quadrants.add((rr < center, cc < center))
    if len(quadrants) < 3:
        return False

    correct_out = _ring_output(side, correct)
    wrong_sequences = {
        _sequence_from(desc, lambda comp: comp["top_left"]),
        _sequence_from(desc, lambda comp: (comp["top_left"][1], comp["top_left"][0])),
        _sequence_from(desc, lambda comp: (comp["nearest_center"], comp["top_left"][0], comp["top_left"][1])),
        _sequence_from(desc, lambda comp: (-comp["farthest_center"], comp["top_left"][0], comp["top_left"][1])),
        _sequence_from(desc, lambda comp: (-comp["area"], comp["top_left"][0], comp["top_left"][1])),
        tuple(sorted(correct)),
        tuple(sorted(correct, reverse=True)),
    }
    for seq in wrong_sequences:
        if seq == correct:
            return False
        if np.array_equal(_ring_output(side, seq), correct_out):
            return False
    return True


def _generate_once() -> np.ndarray | None:
    side = random.choice((11, 13))
    ring_count = side // 2
    colors = random.sample(PALETTE, ring_count)
    ring_colors = random.sample(colors, len(colors))
    areas = _sample_areas(ring_count)

    grid = np.full((side, side), BG, dtype=int)
    center = side // 2
    grid[center, center] = CENTER

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[center, center] = True
    blocked = _dilate8(blocked)

    specs = [{"color": color, "area": area} for color, area in zip(ring_colors, areas)]
    random.shuffle(specs)
    specs.sort(key=lambda spec: spec["area"], reverse=True)

    for spec in specs:
        patch = _random_polyomino(int(spec["area"]))
        if spec["area"] >= 3 and patch.shape[0] * patch.shape[1] == spec["area"] and random.random() < 0.6:
            patch = _random_polyomino(int(spec["area"]))
        placed = _place_patch(grid, blocked, patch, int(spec["color"]))
        if placed is None:
            return None

    expected = _ring_output(side, ring_colors)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_checks(grid):
        return None

    in_sig = _signature(grid)
    out_sig = _signature(actual)
    if in_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
        return None

    _SEEN_INPUT_SIGS.add(in_sig)
    _SEEN_OUTPUT_SIGS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate input")
