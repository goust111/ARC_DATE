from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS) not in sys.path:
    sys.path.append(str(_UTILS))

from arc_utils import Color, components_mask, find_components, get_bg, neighbors4, paste


PAIR_COLOR = {int(Color.GREEN): int(Color.PINK), int(Color.PINK): int(Color.GREEN)}
SEED_DIMS = ((2, 4), (4, 2), (2, 5), (5, 2), (3, 4), (4, 3), (3, 5), (5, 3))
MARKER_PALETTE = (int(Color.BLUE), int(Color.RED), int(Color.YELLOW))

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
MIN_CLOSED_CHAMBERS = 3
MAX_CLOSED_CHAMBERS = 8
MIN_CHAMBER_AREA = 12
MAX_GENERATION_ATTEMPTS = 500


def _stamp_solid(out: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = map(int, patch.shape)
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(out.shape[0], int(r0) + h)
    cc1 = min(out.shape[1], int(c0) + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return

    color = int(patch[0, 0])
    window = out[rr0:rr1, cc0:cc1]
    assert np.all((window == int(bg)) | (window == color)), "tiling bars must not conflict"
    paste(out, patch, int(r0), int(c0), clip=True)


def _build_tiling(
    shape: tuple[int, int],
    seed_top_left: tuple[int, int],
    h: int,
    w: int,
    seed_color: int,
    *,
    include_rotations: bool = True,
    lattice_mode: str = "full",
) -> np.ndarray:
    """Render the bar network on a 2-D diagonal lattice."""
    H, W = map(int, shape)
    r0, c0 = map(int, seed_top_left)
    bg = int(Color.BLACK)
    other = int(PAIR_COLOR[int(seed_color)])
    original = np.full((int(h), int(w)), int(seed_color), dtype=int)
    rotated = np.full((int(w), int(h)), int(other), dtype=int)
    out = np.full((H, W), bg, dtype=int)

    period = int(h + w)
    limit = (H + W) // period + 4
    if lattice_mode == "full":
        index_pairs = (
            (a, b) for a in range(-limit, limit + 1) for b in range(-limit, limit + 1)
        )
    elif lattice_mode == "one_chain":
        index_pairs = ((n, 0) for n in range(-limit, limit + 1))
    else:
        raise ValueError(f"unknown lattice mode: {lattice_mode}")

    for a, b in index_pairs:
        base_r = int(r0 + (a + b) * period)
        base_c = int(c0 + (a - b) * period)
        _stamp_solid(out, original, base_r, base_c, bg)
        if include_rotations:
            _stamp_solid(out, rotated, base_r + h, base_c + w, bg)
            _stamp_solid(out, rotated, base_r + h, base_c - h, bg)

    return out


def _is_solid_seed(component: dict) -> bool:
    coords = component["coords"]
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    return h >= 2 and w >= 2 and h != w and int(component["area"]) == h * w and np.all(
        coords.shape[0] == h * w
    )


def _find_seed(grid: np.ndarray) -> dict:
    bg = int(get_bg(grid))
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[Color.GREEN, Color.PINK],
        return_format="dict",
    )
    candidates = [component for component in components if _is_solid_seed(component)]
    assert len(candidates) == 1, "expected exactly one solid non-square green/pink rectangle seed"
    return candidates[0]


def _closed_chambers(tiled: np.ndarray) -> list[np.ndarray]:
    H, W = map(int, tiled.shape)
    chambers = []
    for coords in components_mask(tiled == int(Color.BLACK), connectivity=4):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == H - 1):
            continue
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == W - 1):
            continue
        chambers.append(coords)
    return chambers


def _extract_markers(grid: np.ndarray, seed: dict) -> dict[tuple[int, int], int]:
    markers: dict[tuple[int, int], int] = {}
    seed_cells = {tuple(map(int, rc)) for rc in seed["coords"]}
    for r, c in np.argwhere(grid != int(Color.BLACK)):
        pos = (int(r), int(c))
        if pos in seed_cells:
            continue
        color = int(grid[pos])
        assert color in MARKER_PALETTE, "markers must use the dedicated vote colors"
        markers[pos] = color
    return markers


def _adjacent_marker_votes(
    chamber: np.ndarray, markers: dict[tuple[int, int], int], shape: tuple[int, int]
) -> Counter[int]:
    H, W = map(int, shape)
    touching: dict[tuple[int, int], int] = {}
    for r0, c0 in chamber:
        for r, c in neighbors4(int(r0), int(c0)):
            if 0 <= r < H and 0 <= c < W and (int(r), int(c)) in markers:
                touching[(int(r), int(c))] = int(markers[(int(r), int(c))])
    return Counter(touching.values())


def _render_output_from_seed(
    shape: tuple[int, int],
    seed_top_left: tuple[int, int],
    h: int,
    w: int,
    seed_color: int,
    markers: dict[tuple[int, int], int],
) -> np.ndarray:
    tiled = _build_tiling(shape, seed_top_left, h, w, seed_color)
    for r, c in markers:
        assert int(tiled[r, c]) == int(Color.BLACK), "markers must stay off the stripe network"

    out = tiled.copy()
    for chamber in _closed_chambers(tiled):
        counts = _adjacent_marker_votes(chamber, markers, shape)
        if not counts:
            continue
        top = max(counts.values())
        winners = [color for color, count in counts.items() if count == top]
        assert len(winners) == 1, "each closed chamber must have a unique marker majority"
        out[chamber[:, 0], chamber[:, 1]] = int(winners[0])

    for (r, c), color in markers.items():
        out[int(r), int(c)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the seed bar into a diagonal zigzag stripe lattice, then fill each enclosed chamber with the majority color of its touching marker pixels.

    Input structure
    - The background is black.
    - Exactly one foreground object is a solid non-square green or pink rectangle; it is the tiling seed.
    - Additional isolated single-pixel markers use non-bar colors and lie on background cells that the completed stripe lattice does not cover.

    Rule
    - Read the seed rectangle's height h, width w, color, and top-left coordinate.
    - Extend that bar on the 2-D diagonal lattice with period h+w: original-orientation copies repeat at all diagonal-lattice seed positions, and between neighboring originals on each diagonal branch place a 90-degree rotated copy in the paired green/pink color.
    - Clip every stamped bar to the canvas boundary.
    - In the completed stripe lattice, find every 4-connected black region that is fully enclosed and does not touch the outer border.
    - For each such chamber, collect the distinct marker pixels orthogonally adjacent to it, choose the unique majority marker color, and fill the chamber with that color.
    - Preserve all stripe pixels and all marker pixels unchanged; chambers without touching markers stay black.

    Key insight
    - The seed bar does not describe chambers directly; its diagonal lattice expansion creates the walls, and only after reconstructing that emergent enclosure structure can the marker-majority fill be applied correctly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(get_bg(g)) == int(Color.BLACK), "background must be black"

    seed = _find_seed(g)
    r0, c0, r1, c1 = map(int, seed["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    seed_color = int(seed["color"])
    markers = _extract_markers(g, seed)
    return _render_output_from_seed(g.shape, (r0, c0), h, w, seed_color, markers)


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    H, W = map(int, shape)
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == H - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == W - 1)
    )


def _sample_marker_positions(
    chamber: np.ndarray,
    count: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    candidates = [tuple(map(int, rc)) for rc in chamber]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for pos in candidates:
        if pos in occupied:
            continue
        if any(abs(pos[0] - r) + abs(pos[1] - c) <= 1 for r, c in chosen):
            continue
        chosen.append(pos)
        if len(chosen) == count:
            return chosen
    return None


def _vote_pattern(majority: int, palette: list[int], marker_count: int) -> list[int]:
    others = [color for color in palette if color != majority]
    if marker_count == 1:
        return [majority]
    if marker_count == 2:
        return [majority, majority]
    if marker_count == 3:
        return [majority, majority, random.choice(others)]
    if len(others) >= 2 and random.random() < 0.5:
        a, b = random.sample(others, 2)
        return [majority, majority, a, b]
    return [majority, majority, majority, random.choice(others)]


def _semantic_variants_differ(
    shape: tuple[int, int],
    seed_top_left: tuple[int, int],
    h: int,
    w: int,
    seed_color: int,
    markers: dict[tuple[int, int], int],
    expected: np.ndarray,
) -> bool:
    no_rot = _build_tiling(shape, seed_top_left, h, w, seed_color, include_rotations=False)
    one_chain = _build_tiling(shape, seed_top_left, h, w, seed_color, lattice_mode="one_chain")
    alt_no_rot = no_rot.copy()
    alt_one_chain = one_chain.copy()
    for (r, c), color in markers.items():
        if int(no_rot[r, c]) == int(Color.BLACK):
            alt_no_rot[r, c] = int(color)
        if int(one_chain[r, c]) == int(Color.BLACK):
            alt_one_chain[r, c] = int(color)
    return not np.array_equal(expected, alt_no_rot) and not np.array_equal(expected, alt_one_chain)


def _attempt_generate() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    seed_color = random.choice((int(Color.GREEN), int(Color.PINK)))
    palette_size = random.choice((2, 3))
    marker_palette = random.sample(list(MARKER_PALETTE), palette_size)

    dims = list(SEED_DIMS)
    random.shuffle(dims)
    for h, w in dims:
        if h > H or w > W:
            continue
        positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
        random.shuffle(positions)
        for r0, c0 in positions[: min(len(positions), 80)]:
            tiled = _build_tiling((H, W), (r0, c0), h, w, seed_color)
            chambers = [
                chamber
                for chamber in _closed_chambers(tiled)
                if int(chamber.shape[0]) >= MIN_CHAMBER_AREA and not _touches_border(chamber, (H, W))
            ]
            if not (MIN_CLOSED_CHAMBERS <= len(chambers) <= MAX_CLOSED_CHAMBERS):
                continue

            majorities = [random.choice(marker_palette) for _ in chambers]
            if len(set(majorities)) == 1 and len(marker_palette) > 1:
                pivot = random.randrange(len(majorities))
                choices = [color for color in marker_palette if color != majorities[pivot]]
                majorities[pivot] = random.choice(choices)

            markers: dict[tuple[int, int], int] = {}
            occupied: set[tuple[int, int]] = set()
            ok = True
            multi_color_chambers = 0
            for chamber, majority in zip(chambers, majorities):
                marker_count = random.choice((1, 2, 3, 4))
                if marker_count == 2 and random.random() < 0.65:
                    marker_count = 3
                colors = _vote_pattern(int(majority), marker_palette, int(marker_count))
                if len(set(colors)) > 1:
                    multi_color_chambers += 1
                positions_for_chamber = _sample_marker_positions(chamber, len(colors), occupied)
                if positions_for_chamber is None:
                    ok = False
                    break
                for pos, color in zip(positions_for_chamber, colors):
                    markers[pos] = int(color)
                    occupied.add(pos)

            if not ok or multi_color_chambers == 0:
                continue

            expected = _render_output_from_seed((H, W), (r0, c0), h, w, seed_color, markers)
            if not _semantic_variants_differ((H, W), (r0, c0), h, w, seed_color, markers, expected):
                continue

            chamber_votes = [
                _adjacent_marker_votes(chamber, markers, (H, W)) for chamber in _closed_chambers(tiled)
            ]
            if any(not counts for counts in chamber_votes):
                continue
            if any(sum(1 for v in counts.values() if v == max(counts.values())) != 1 for counts in chamber_votes):
                continue

            inp = np.full((H, W), int(Color.BLACK), dtype=int)
            inp[r0 : r0 + h, c0 : c0 + w] = int(seed_color)
            for (r, c), color in markers.items():
                if int(tiled[r, c]) != int(Color.BLACK):
                    ok = False
                    break
                inp[r, c] = int(color)
            if not ok:
                continue

            actual = main(inp)
            if np.array_equal(actual, expected):
                return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chamber-vote tiling puzzle")
