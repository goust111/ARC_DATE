from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator configuration.
PAIR_COUNT_RANGE = (2, 4)
RECT_SIDE_RANGE = (4, 7)
GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
MAX_LEVELS = 4
PLACEMENT_PADDING = 1
SHAPE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 200
GEN_ATTEMPTS = 400


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert rows.size and cols.size
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _component_from_coords(coords: list[tuple[int, int]]) -> dict[str, object]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel_coords: list[tuple[int, int]] = []
    for r, c in coords:
        rr = r - r0
        cc = c - c0
        mask[rr, cc] = True
        rel_coords.append((rr, cc))
    return {
        "bbox": (r0, c0, r1, c1),
        "shape": mask,
        "coords": tuple(sorted(rel_coords)),
        "area": len(coords),
    }


def _extract_components(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    comps = _connected_components(grid == color)
    return [_component_from_coords(comp) for comp in comps]


def _full_row(mask: set[tuple[int, int]], row: int, width: int) -> bool:
    return all((row, c) in mask for c in range(width))


def _full_col(mask: set[tuple[int, int]], col: int, height: int) -> bool:
    return all((r, col) in mask for r in range(height))


def _enumerate_pair_tilings(
    comp_a: dict[str, object], comp_b: dict[str, object]
) -> list[dict[str, object]]:
    coords_a = list(comp_a["coords"])
    coords_b = list(comp_b["coords"])
    ha, wa = comp_a["shape"].shape
    hb, wb = comp_b["shape"].shape
    total_area = int(comp_a["area"]) + int(comp_b["area"])

    solutions: list[dict[str, object]] = []
    for height in range(max(ha, hb), total_area + 1):
        if total_area % height != 0:
            continue
        width = total_area // height
        if width < max(wa, wb):
            continue

        for r0a in range(height - ha + 1):
            for c0a in range(width - wa + 1):
                placed_a = {(r0a + r, c0a + c) for r, c in coords_a}
                for r0b in range(height - hb + 1):
                    for c0b in range(width - wb + 1):
                        placed_b = {(r0b + r, c0b + c) for r, c in coords_b}
                        if placed_a & placed_b:
                            continue
                        if len(placed_a) + len(placed_b) != total_area:
                            continue
                        union = placed_a | placed_b
                        if len(union) != total_area:
                            continue
                        if (0, 0) in placed_a:
                            top_idx = 0
                            top_mask = placed_a
                        elif (0, 0) in placed_b:
                            top_idx = 1
                            top_mask = placed_b
                        else:
                            continue

                        solutions.append(
                            {
                                "height": height,
                                "width": width,
                                "top_idx": top_idx,
                                "top_has_full_row": _full_row(top_mask, 0, width),
                                "top_has_full_col": _full_col(top_mask, 0, height),
                                "offsets": ((r0a, c0a), (r0b, c0b)),
                            }
                        )
    return solutions


def _reconstruct_rectangle(
    comp_a: dict[str, object], comp_b: dict[str, object]
) -> dict[str, int]:
    solutions = _enumerate_pair_tilings(comp_a, comp_b)
    assert solutions, "same-color fragments must tile a rectangle"

    preferred = [
        sol
        for sol in solutions
        if bool(sol["top_has_full_row"]) and bool(sol["top_has_full_col"])
    ]
    if preferred:
        solutions = preferred

    best = min(
        solutions,
        key=lambda sol: (
            int(sol["height"]) * int(sol["width"]),
            int(sol["height"]) + int(sol["width"]),
            int(sol["height"]),
            int(sol["width"]),
            tuple(sol["offsets"][0]) + tuple(sol["offsets"][1]),
        ),
    )
    return {
        "height": int(best["height"]),
        "width": int(best["width"]),
        "top_idx": int(best["top_idx"]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair equal-color fragments, reconstruct each hidden rectangle, sort by area,
    then stack the full rectangles from top to bottom.

    Area is the only stated ordering rule. When areas tie, this implementation
    uses rectangle dimensions and the pair's input position as deterministic
    visible-structure fallback keys.
    """
    g = np.asarray(grid, dtype=int)
    colors = [int(v) for v in np.unique(g) if int(v) != BG]
    assert colors

    rects: list[dict[str, int]] = []
    for color in colors:
        comps = _extract_components(g, color)
        assert len(comps) == 2, "each non-background color must appear in two components"
        rect = _reconstruct_rectangle(comps[0], comps[1])
        pair_pos = min(
            (int(comp["bbox"][0]), int(comp["bbox"][1]))
            for comp in comps
        )
        rects.append(
            {
                "color": color,
                "height": rect["height"],
                "width": rect["width"],
                "area": rect["height"] * rect["width"],
                "row_key": pair_pos[0],
                "col_key": pair_pos[1],
            }
        )

    rects.sort(
        key=lambda item: (
            int(item["area"]),
            int(item["height"]),
            int(item["width"]),
            int(item["row_key"]),
            int(item["col_key"]),
        )
    )

    out_h = sum(int(item["height"]) for item in rects)
    out_w = max(int(item["width"]) for item in rects)
    out = np.full((out_h, out_w), BG, dtype=int)

    row = 0
    for item in rects:
        h = int(item["height"])
        w = int(item["width"])
        out[row : row + h, :w] = int(item["color"])
        row += h
    return out


def _positive_parts(total: int, count: int) -> list[int]:
    assert total >= count >= 1
    if count == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), count - 1))
    prev = 0
    parts: list[int] = []
    for cut in cuts + [total]:
        parts.append(cut - prev)
        prev = cut
    return parts


def _mask_is_connected(mask: np.ndarray) -> bool:
    comps = _connected_components(mask)
    return len(comps) == 1


def _build_pair_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    max_levels = min(MAX_LEVELS, height - 1, width - 1)
    if max_levels < 2:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        levels = random.randint(2, max_levels)
        lengths = sorted(random.sample(range(1, width), levels), reverse=True)
        if lengths[0] > width - 1 or lengths[-1] > width - 2:
            continue

        top_full_rows_max = height - levels
        if top_full_rows_max < 1:
            continue
        top_full_rows = random.randint(1, top_full_rows_max)
        band_sizes = _positive_parts(height - top_full_rows, levels)
        row_lengths = [width] * top_full_rows
        for length, band in zip(lengths, band_sizes):
            row_lengths.extend([length] * band)

        cut_rows = [length for length in row_lengths if length < width]
        if len(cut_rows) < 2 or len(set(cut_rows)) < 2:
            continue

        top_mask = np.zeros((height, width), dtype=bool)
        for r, length in enumerate(row_lengths):
            top_mask[r, :length] = True

        bottom_mask = _trim_mask(~top_mask)
        if bottom_mask.shape[0] < 2 or bottom_mask.shape[1] < 2:
            continue
        if bottom_mask.all():
            continue

        top_area = int(top_mask.sum())
        bottom_area = int(bottom_mask.sum())
        if min(top_area, bottom_area) < 4:
            continue
        if not _mask_is_connected(top_mask) or not _mask_is_connected(bottom_mask):
            continue
        return top_mask, bottom_mask
    return None


def _render_piece(mask: np.ndarray, color: int) -> np.ndarray:
    piece = np.full(mask.shape, BG, dtype=int)
    piece[mask] = color
    return piece


def _expected_output(specs: list[dict[str, object]]) -> np.ndarray:
    ordered = sorted(specs, key=lambda item: int(item["area"]))
    out_h = sum(int(item["height"]) for item in ordered)
    out_w = max(int(item["width"]) for item in ordered)
    out = np.full((out_h, out_w), BG, dtype=int)
    row = 0
    for item in ordered:
        h = int(item["height"])
        w = int(item["width"])
        out[row : row + h, :w] = int(item["color"])
        row += h
    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_piece(
    grid: np.ndarray, occupied: np.ndarray, piece: np.ndarray
) -> bool:
    piece_mask = piece != BG
    ph, pw = piece.shape
    gh, gw = grid.shape
    dilated = _dilate(occupied)

    if ph > gh or pw > gw:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = random.randint(0, gh - ph)
        c0 = random.randint(0, gw - pw)
        window = dilated[r0 : r0 + ph, c0 : c0 + pw]
        if bool((window & piece_mask).any()):
            continue

        target = grid[r0 : r0 + ph, c0 : c0 + pw]
        target[piece_mask] = piece[piece_mask]
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= piece_mask
        return True
    return False


def _sample_specs() -> list[dict[str, object]] | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(PALETTE, pair_count)
    specs: list[dict[str, object]] = []
    used_areas: set[int] = set()

    for color in colors:
        made = False
        for _ in range(SHAPE_ATTEMPTS):
            height = random.randint(*RECT_SIDE_RANGE)
            width = random.randint(*RECT_SIDE_RANGE)
            area = height * width
            if area in used_areas:
                continue
            masks = _build_pair_masks(height, width)
            if masks is None:
                continue
            top_mask, bottom_mask = masks
            specs.append(
                {
                    "color": color,
                    "height": height,
                    "width": width,
                    "area": area,
                    "pieces": [
                        _render_piece(top_mask, color),
                        _render_piece(bottom_mask, color),
                    ],
                }
            )
            used_areas.add(area)
            made = True
            break
        if not made:
            return None

    if sum(int(item["height"]) for item in specs) > 30:
        return None
    return specs


def _build_input_from_specs(specs: list[dict[str, object]]) -> np.ndarray | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        grid_h = random.randint(*GRID_HEIGHT_RANGE)
        grid_w = random.randint(*GRID_WIDTH_RANGE)
        grid = np.full((grid_h, grid_w), BG, dtype=int)
        occupied = np.zeros((grid_h, grid_w), dtype=bool)

        pieces: list[np.ndarray] = []
        for item in specs:
            pieces.extend([piece.copy() for piece in item["pieces"]])
        random.shuffle(pieces)
        pieces.sort(key=lambda piece: (piece.shape[0] * piece.shape[1], piece.shape[0]), reverse=True)

        ok = True
        for piece in pieces:
            if not _place_piece(grid, occupied, piece):
                ok = False
                break
        if not ok:
            continue

        if not occupied.any():
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        specs = _sample_specs()
        if specs is None:
            continue
        grid = _build_input_from_specs(specs)
        if grid is None:
            continue
        expected = _expected_output(specs)
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
