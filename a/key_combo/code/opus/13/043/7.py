from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREY = 5
PROTOTYPE_SIZE = 3
MATCH_RECOLOR = RED
INPUT_COLORS = (1, 3, 4, 7)
GRID_HEIGHT_RANGE = (17, 25)
GRID_WIDTH_RANGE = (11, 18)
BAND_HEIGHT_CHOICES = (3, 4)
MAX_ATTEMPTS = 500


def _repeat_rows(tile: np.ndarray, anchor_row: int, height: int) -> np.ndarray:
    row_ids = (np.arange(height) - int(anchor_row)) % tile.shape[0]
    return tile[row_ids].copy()


def _extract_band_and_prototype(grid: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= PROTOTYPE_SIZE and g.shape[1] >= PROTOTYPE_SIZE

    prototype_patch = g[-PROTOTYPE_SIZE:, -PROTOTYPE_SIZE:]
    assert set(map(int, prototype_patch.flat)).issubset({BG, GREY}), "prototype must be black/grey only"
    prototype_mask = prototype_patch == GREY
    assert prototype_mask.any(), "prototype must contain grey foreground"
    assert not prototype_mask.all(), "prototype cannot be solid grey"

    band_rows = np.flatnonzero(np.any((g != BG) & (g != GREY), axis=1))
    assert band_rows.size in BAND_HEIGHT_CHOICES, "expected a 3-4 row foreground band"
    r0 = int(band_rows[0])
    r1 = int(band_rows[-1])
    assert np.array_equal(band_rows, np.arange(r0, r1 + 1)), "band rows must be contiguous"

    tile = g[r0 : r1 + 1, :].copy()
    assert not np.any(tile == GREY), "band must not contain grey"
    assert np.all(np.any(tile != BG, axis=1)), "every band row must contain foreground"
    return tile, r0, prototype_mask


def _match_positions(tiled: np.ndarray, prototype_mask: np.ndarray) -> list[tuple[int, int]]:
    occ = tiled != BG
    height, width = tiled.shape
    matches: list[tuple[int, int]] = []
    for r in range(height - PROTOTYPE_SIZE + 1):
        for c in range(width - PROTOTYPE_SIZE + 1):
            if np.array_equal(occ[r : r + PROTOTYPE_SIZE, c : c + PROTOTYPE_SIZE], prototype_mask):
                matches.append((r, c))
    return matches


def _recolor_matches(tiled: np.ndarray, matches: list[tuple[int, int]]) -> np.ndarray:
    out = tiled.copy()
    for r, c in matches:
        window = out[r : r + PROTOTYPE_SIZE, c : c + PROTOTYPE_SIZE]
        window[window != BG] = MATCH_RECOLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Vertically tile the full-width row band, then recolor every 3x3 texture patch whose occupancy matches the lower-right prototype.

    Input structure:
    - Black background fills the canvas except for one contiguous full-width row band of height 3 or 4.
    - The band uses 2 or 3 non-background colors and contains both foreground and background cells.
    - A separate 3x3 prototype made of grey foreground cells on black sits in the lower-right corner.

    Rule:
    - Use the full-width row band as a vertical tile, repeating it upward and downward across the original canvas while keeping the band's first row aligned to its input row.
    - On the tiled result, slide a 3x3 window with unit stride.
    - A window matches when its non-black cell positions exactly equal the grey-cell positions in the prototype, ignoring the window's actual colors.
    - Recolor every non-black cell in every matching window red; leave all other tiled cells unchanged.

    Key insight:
    - The lower-right prototype is only a binary occupancy template, and the target patches appear only after the sparse row band is expanded into a repeated vertical texture.
    """
    g = np.asarray(grid, dtype=int)
    tile, anchor_row, prototype_mask = _extract_band_and_prototype(g)
    tiled = _repeat_rows(tile, anchor_row, g.shape[0])
    matches = _match_positions(tiled, prototype_mask)
    return _recolor_matches(tiled, matches)


def _window_crosses_band_boundary(top_row: int, anchor_row: int, band_height: int) -> bool:
    start_phase = (int(top_row) - int(anchor_row)) % int(band_height)
    return start_phase > int(band_height) - PROTOTYPE_SIZE


def _positions_overlap(positions: list[tuple[int, int]]) -> bool:
    for i, (r0, c0) in enumerate(positions):
        for r1, c1 in positions[i + 1 :]:
            if abs(r0 - r1) < PROTOTYPE_SIZE and abs(c0 - c1) < PROTOTYPE_SIZE:
                return True
    return False


def _paint_prototype(mask: np.ndarray) -> np.ndarray:
    patch = np.zeros((PROTOTYPE_SIZE, PROTOTYPE_SIZE), dtype=int)
    patch[mask] = GREY
    return patch


def _make_periodic_tile(height: int, width: int, colors: tuple[int, ...]) -> np.ndarray:
    for _ in range(120):
        period = random.randint(3, min(6, width))
        motif = np.zeros((height, period), dtype=int)
        for r in range(height):
            for c in range(period):
                if random.random() < random.uniform(0.33, 0.52):
                    motif[r, c] = BG
                else:
                    motif[r, c] = random.choice(colors)
        tile = motif[:, np.arange(width) % period]
        fg = tile != BG
        density = float(fg.mean())
        if not (0.33 <= density <= 0.72):
            continue
        if not np.all(np.any(fg, axis=1)):
            continue
        present = tuple(sorted(set(map(int, tile[fg]))))
        if len(present) != len(colors):
            continue
        if len({tile[r].tobytes() for r in range(height)}) < 2:
            continue
        return tile
    raise RuntimeError("failed to build a periodic row band")


def _candidate_groups(tiled: np.ndarray) -> list[dict]:
    occ = tiled != BG
    groups: dict[bytes, dict] = {}
    for r in range(tiled.shape[0] - PROTOTYPE_SIZE + 1):
        for c in range(tiled.shape[1] - PROTOTYPE_SIZE + 1):
            mask = occ[r : r + PROTOTYPE_SIZE, c : c + PROTOTYPE_SIZE]
            key = mask.tobytes()
            if key not in groups:
                groups[key] = {"mask": mask.copy(), "positions": []}
            groups[key]["positions"].append((r, c))
    return list(groups.values())


def _choose_prototype_mask(tiled: np.ndarray, anchor_row: int, band_height: int) -> np.ndarray | None:
    groups = _candidate_groups(tiled)
    want_overlap = random.random() < 0.5
    total_fg = int(np.count_nonzero(tiled))
    if total_fg == 0:
        return None

    viable: list[dict] = []
    for group in groups:
        mask = group["mask"]
        positions = group["positions"]
        fg_count = int(mask.sum())
        if fg_count < 2 or fg_count > 7 or fg_count == 9:
            continue
        if len(positions) < 2:
            continue
        if len(set(r for r, _c in positions)) < 2:
            continue
        if not any(
            (r < anchor_row or r + PROTOTYPE_SIZE - 1 >= anchor_row + band_height)
            for r, _c in positions
        ):
            continue
        if not any(_window_crosses_band_boundary(r, anchor_row, band_height) for r, _c in positions):
            continue

        coverage = np.zeros(tiled.shape, dtype=bool)
        for r, c in positions:
            coverage[r : r + PROTOTYPE_SIZE, c : c + PROTOTYPE_SIZE] |= tiled[
                r : r + PROTOTYPE_SIZE, c : c + PROTOTYPE_SIZE
            ] != BG
        coverage_ratio = float(coverage.sum()) / float(total_fg)
        if not (0.10 <= coverage_ratio <= 0.65):
            continue

        same_area_decoy = any(
            int(other["mask"].sum()) == fg_count and not np.array_equal(other["mask"], mask) for other in groups
        )
        if not same_area_decoy:
            continue

        group_overlap = _positions_overlap(positions)
        viable.append(
            {
                "mask": mask,
                "positions": positions,
                "overlap": group_overlap,
                "coverage_ratio": coverage_ratio,
            }
        )

    preferred = [cand for cand in viable if cand["overlap"] == want_overlap]
    pool = preferred or viable
    if not pool:
        return None
    return random.choice(pool)["mask"].copy()


def _build_input_and_output() -> tuple[np.ndarray, np.ndarray]:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    if height <= width:
        height = width + random.randint(1, 6)
        height = min(height, 24)
    band_height = random.choice(BAND_HEIGHT_CHOICES)
    band_top = random.randint(2, height - band_height - 5)
    color_count = random.choice((2, 3))
    colors = tuple(random.sample(INPUT_COLORS, color_count))

    tile = _make_periodic_tile(band_height, width, colors)
    tiled = _repeat_rows(tile, band_top, height)
    prototype_mask = _choose_prototype_mask(tiled, band_top, band_height)
    if prototype_mask is None:
        raise RuntimeError("no semantic prototype candidate found")

    inp = np.zeros((height, width), dtype=int)
    inp[band_top : band_top + band_height, :] = tile
    inp[-PROTOTYPE_SIZE:, -PROTOTYPE_SIZE:] = _paint_prototype(prototype_mask)

    matches = _match_positions(tiled, prototype_mask)
    out = _recolor_matches(tiled, matches)
    return inp, out


def _semantic_signature(inp: np.ndarray) -> tuple[int, int, int, int, int]:
    tile, anchor_row, prototype_mask = _extract_band_and_prototype(inp)
    tiled = _repeat_rows(tile, anchor_row, inp.shape[0])
    matches = _match_positions(tiled, prototype_mask)
    return (
        int(inp.shape[0]),
        int(inp.shape[1]),
        int(tile.shape[0]),
        int(prototype_mask.sum()),
        int(len(matches)),
    )


def _validate_sample(inp: np.ndarray, out: np.ndarray) -> bool:
    if not np.array_equal(main(inp), out):
        return False

    tile, anchor_row, prototype_mask = _extract_band_and_prototype(inp)
    tiled = _repeat_rows(tile, anchor_row, inp.shape[0])
    matches = _match_positions(tiled, prototype_mask)
    if len(matches) < 2:
        return False
    if not any(_window_crosses_band_boundary(r, anchor_row, tile.shape[0]) for r, _c in matches):
        return False
    if not any((r < anchor_row or r + PROTOTYPE_SIZE - 1 >= anchor_row + tile.shape[0]) for r, _c in matches):
        return False

    red_cells = int(np.count_nonzero(out == RED))
    total_fg = int(np.count_nonzero(tiled))
    if red_cells <= 0 or red_cells >= total_fg:
        return False

    groups = _candidate_groups(tiled)
    proto_fg = int(prototype_mask.sum())
    if not any(
        int(group["mask"].sum()) == proto_fg and not np.array_equal(group["mask"], prototype_mask) for group in groups
    ):
        return False

    band_colors = sorted(set(map(int, tile[tile != BG])))
    if len(band_colors) not in (2, 3):
        return False
    return True


def generate_input() -> np.ndarray:
    seen_signatures: set[tuple[int, int, int, int, int]] = set()
    for _ in range(MAX_ATTEMPTS):
        try:
            inp, out = _build_input_and_output()
        except RuntimeError:
            continue
        if not _validate_sample(inp, out):
            continue
        seen_signatures.add(_semantic_signature(inp))
        return inp
    raise RuntimeError(f"failed to generate valid puzzle after {MAX_ATTEMPTS} attempts; seen={len(seen_signatures)}")
