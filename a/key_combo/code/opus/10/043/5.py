from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
KEY_ORDER = (RED, BLUE, GREEN)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    out: list[np.ndarray] = []
    neighbors = DIR4 if connectivity == 4 else [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey = np.argwhere(grid == GREY)
    assert grey.size > 0, "legend border must exist"
    bbox = bbox_coords(grey)
    r0, c0, r1, c1 = bbox
    patch = crop_bbox(grid, bbox)
    assert patch.shape[0] >= 5 and patch.shape[1] >= 7, "legend box too small"
    assert np.all(patch[0, :] == GREY) and np.all(patch[-1, :] == GREY)
    assert np.all(patch[:, 0] == GREY) and np.all(patch[:, -1] == GREY)
    assert not np.any(patch[1:-1, 1:-1] == GREY), "legend interior must stay non-grey"
    return bbox


def _key_rows_and_colors(grid: np.ndarray, legend_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = legend_bbox
    box = crop_bbox(grid, legend_bbox)
    key_col = None
    for cc in range(1, box.shape[1] - 1):
        vals = [int(v) for v in box[1:-1, cc] if int(v) not in (BG, GREY)]
        if vals:
            key_col = cc
            break
    assert key_col is not None, "legend key column missing"
    rows = []
    for rr in range(1, box.shape[0] - 1):
        color = int(box[rr, key_col])
        if color not in (BG, GREY):
            rows.append((r0 + rr, color))
    assert [color for _rr, color in rows] == list(KEY_ORDER), "legend key order must be red, blue, green"
    return rows


def _piece_patch(grid: np.ndarray, coords: np.ndarray, color: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(color)
    return patch


def _piece_records(grid: np.ndarray, legend_bbox: tuple[int, int, int, int]) -> dict[int, list[dict]]:
    r0, c0, r1, c1 = legend_bbox
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    pieces: dict[int, list[dict]] = {}
    for color in KEY_ORDER:
        comps = components_mask((grid == color) & outside, connectivity=4)
        records = []
        for comp in comps:
            b = bbox_coords(comp)
            records.append(
                {
                    "bbox": b,
                    "patch": _piece_patch(grid, comp, color),
                    "area": int(comp.shape[0]),
                    "height": int(b[2] - b[0] + 1),
                }
            )
        records.sort(key=lambda rec: (rec["bbox"][0], rec["bbox"][1]))
        pieces[color] = records
    return pieces


def _place_band_piece(occ: np.ndarray, patch: np.ndarray) -> tuple[int, int]:
    H, W = occ.shape
    h, w = patch.shape
    mask = patch != BG
    for c0 in range(W - w + 1):
        for r0 in range(H - h + 1):
            if not np.any(occ[r0 : r0 + h, c0 : c0 + w] & mask):
                occ[r0 : r0 + h, c0 : c0 + w] |= mask
                return r0, c0
    raise AssertionError("piece cannot be assembled into its band")


def _assemble_band(color: int, band_h: int, band_w: int, pieces: list[dict]) -> np.ndarray:
    occ = np.zeros((band_h, band_w), dtype=bool)
    out = np.zeros((band_h, band_w), dtype=int)
    for piece in pieces:
        patch = piece["patch"]
        r0, c0 = _place_band_piece(occ, patch)
        mask = patch != BG
        view = out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        view[mask] = int(color)
    assert occ.all(), "assembled band must be gap-free"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count off-box color fragments, encode their tallies in the legend blueprint, and assemble them into ordered color bands.

    Input structure:
    - Black background with one grey rectangular legend frame on the left.
    - The legend interior is black except for one vertical key column whose rows are red, blue, then green.
    - Outside the legend, each key color appears as several scattered 4-connected monochrome fragments.

    Rule:
    - Read the key-color order from top to bottom inside the legend.
    - For each key color, count its outside fragments; that count is both the legend tally length and the width of that color's output band.
    - Let the output height be the maximum bounding-box height among all fragments. For each color in legend order, greedily place its fragments into an H x count band by scanning left-to-right and as high as possible without overlap, then concatenate the bands.

    Key insight:
    - The legend is not just a counter: the number of tally dots also prescribes each color band's width, so counting and assembly are coupled through one shared blueprint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_bbox = _find_legend_bbox(g)
    keys = _key_rows_and_colors(g, legend_bbox)
    pieces_by_color = _piece_records(g, legend_bbox)

    all_heights = [piece["height"] for color in KEY_ORDER for piece in pieces_by_color[color]]
    assert all_heights, "at least one fragment is required"
    band_h = max(all_heights)

    bands = []
    for _rr, color in keys:
        pieces = pieces_by_color[color]
        band_w = len(pieces)
        assert 2 <= band_w <= 5, "each key color must have 2..5 fragments"
        bands.append(_assemble_band(color, band_h, band_w, pieces))
    return np.concatenate(bands, axis=1)


def _normalize_shape(shape: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    min_r = min(r for r, _c in shape)
    min_c = min(c for _r, c in shape)
    return frozenset((r - min_r, c - min_c) for r, c in shape)


def _anchored_polyominoes(max_size: int = 5) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    frontier: set[frozenset[tuple[int, int]]] = {frozenset({(0, 0)})}
    by_size: dict[int, set[tuple[tuple[int, int], ...]]] = defaultdict(set)
    for _size in range(2, max_size + 1):
        nxt: set[frozenset[tuple[int, int]]] = set()
        for shape in frontier:
            cells = set(shape)
            boundary = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nb = (r + dr, c + dc)
                    if nb not in cells:
                        boundary.add(nb)
            for nb in boundary:
                grown = _normalize_shape(cells | {nb})
                if (0, 0) in grown:
                    nxt.add(grown)
        frontier = nxt
        for shape in frontier:
            by_size[len(shape)].add(tuple(sorted(shape)))
    return {size: sorted(shapes) for size, shapes in by_size.items()}


POLYOMINOES = _anchored_polyominoes(5)


def _first_empty_col_major(filled: np.ndarray) -> tuple[int, int] | None:
    H, W = filled.shape
    for c in range(W):
        for r in range(H):
            if not filled[r, c]:
                return r, c
    return None


def _remaining_components_ok(filled: np.ndarray) -> bool:
    remaining = ~filled
    for comp in components_mask(remaining, connectivity=4):
        if comp.shape[0] == 1:
            return False
    return True


def _tile_band_pieces(band_h: int, band_w: int, rng: random.Random) -> list[np.ndarray] | None:
    total_pieces = band_w
    filled = np.zeros((band_h, band_w), dtype=bool)
    chosen: list[np.ndarray] = []

    def backtrack(index: int) -> bool:
        anchor = _first_empty_col_major(filled)
        if anchor is None:
            return index == total_pieces

        rem_cells = int((~filled).sum())
        rem_pieces = total_pieces - index
        if not (2 * rem_pieces <= rem_cells <= 5 * rem_pieces):
            return False

        ar, ac = anchor
        sizes = [2, 3, 4, 5]
        rng.shuffle(sizes)
        candidates: list[np.ndarray] = []
        for size in sizes:
            rem_after = rem_cells - size
            if rem_pieces == 1 and rem_after != 0:
                continue
            if rem_pieces > 1 and not (2 * (rem_pieces - 1) <= rem_after <= 5 * (rem_pieces - 1)):
                continue
            patterns = POLYOMINOES[size][:]
            rng.shuffle(patterns)
            for pattern in patterns:
                cells = np.array([(ar + dr, ac + dc) for dr, dc in pattern], dtype=int)
                if np.any(cells[:, 0] >= band_h) or np.any(cells[:, 1] >= band_w):
                    continue
                if np.any(filled[cells[:, 0], cells[:, 1]]):
                    continue
                candidates.append(cells)
        rng.shuffle(candidates)
        for cells in candidates:
            filled[cells[:, 0], cells[:, 1]] = True
            if _remaining_components_ok(filled):
                chosen.append(cells.copy())
                if backtrack(index + 1):
                    return True
                chosen.pop()
            filled[cells[:, 0], cells[:, 1]] = False
        return False

    if backtrack(0):
        return chosen
    return None


def _cells_to_patch(cells: np.ndarray, color: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(cells)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        patch[int(r) - r0, int(c) - c0] = int(color)
    return patch


def _generate_color_pieces(color: int, band_h: int, count: int, rng: random.Random) -> list[np.ndarray] | None:
    for _ in range(200):
        cells_list = _tile_band_pieces(band_h, count, rng)
        if cells_list is None:
            continue
        patches = [_cells_to_patch(cells, color) for cells in cells_list]
        if max(p.shape[0] for p in patches) != band_h:
            continue
        if not any(np.count_nonzero(p != BG) != p.shape[0] * p.shape[1] for p in patches):
            if rng.random() < 0.7:
                continue
        return patches
    return None


def _build_legend(counts: dict[int, int], rng: random.Random) -> np.ndarray:
    max_count = max(counts.values())
    width = 2 * max_count + 3 + rng.randint(0, 2)
    legend = np.zeros((5, width), dtype=int)
    legend[0, :] = GREY
    legend[-1, :] = GREY
    legend[:, 0] = GREY
    legend[:, -1] = GREY
    for i, color in enumerate(KEY_ORDER, start=1):
        legend[i, 1] = int(color)
    return legend


def _expected_output(counts: dict[int, int], band_h: int) -> np.ndarray:
    return np.concatenate(
        [np.full((band_h, counts[color]), int(color), dtype=int) for color in KEY_ORDER],
        axis=1,
    )


def _slot_positions(grid_h: int, grid_w: int, legend_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    slots = []
    slot_pitch = 7
    slot_size = 5
    r0, c0, r1, c1 = legend_bbox
    blocked_r0 = max(0, r0 - 1)
    blocked_c0 = max(0, c0 - 1)
    blocked_r1 = min(grid_h - 1, r1 + 1)
    blocked_c1 = min(grid_w - 1, c1 + 1)
    for rr in range(1, grid_h - slot_size + 1, slot_pitch):
        for cc in range(1, grid_w - slot_size + 1, slot_pitch):
            if not (rr + slot_size - 1 < blocked_r0 or rr > blocked_r1 or cc + slot_size - 1 < blocked_c0 or cc > blocked_c1):
                continue
            slots.append((rr, cc))
    return slots


def _place_pieces_on_grid(
    grid: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
    patches_by_color: dict[int, list[np.ndarray]],
    rng: random.Random,
) -> None:
    grid_h, grid_w = grid.shape
    slots = _slot_positions(grid_h, grid_w, legend_bbox)
    total_needed = sum(len(v) for v in patches_by_color.values())
    if len(slots) < total_needed:
        raise ValueError("not enough slots for all pieces")

    available = slots[:]
    for color in KEY_ORDER:
        count = len(patches_by_color[color])
        chosen_slots = sorted(rng.sample(available, count))
        for slot in chosen_slots:
            available.remove(slot)
        for patch, (sr, sc) in zip(patches_by_color[color], chosen_slots):
            h, w = patch.shape
            jitter_r = rng.randint(0, 5 - h)
            jitter_c = rng.randint(0, 5 - w)
            r0 = sr + jitter_r
            c0 = sc + jitter_c
            view = grid[r0 : r0 + h, c0 : c0 + w]
            mask = patch != BG
            assert not np.any(view[mask] != BG)
            view[mask] = patch[mask]


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, counts: dict[int, int], band_h: int) -> None:
    legend_bbox = _find_legend_bbox(inp)
    keys = _key_rows_and_colors(inp, legend_bbox)
    assert [color for _rr, color in keys] == list(KEY_ORDER)

    pieces_by_color = _piece_records(inp, legend_bbox)
    heights = []
    for color in KEY_ORDER:
        assert len(pieces_by_color[color]) == counts[color]
        area = sum(piece["area"] for piece in pieces_by_color[color])
        assert area == counts[color] * band_h
        heights.extend(piece["height"] for piece in pieces_by_color[color])
        band = _assemble_band(color, band_h, counts[color], pieces_by_color[color])
        assert np.array_equal(band, np.full((band_h, counts[color]), color, dtype=int))
    assert max(heights) == band_h
    assert np.array_equal(main(inp), expected)


def generate_input() -> np.ndarray:
    for attempt in range(500):
        rng = random.Random(random.randrange(1 << 30) + attempt)
        band_h = rng.randint(3, 5)
        counts = {color: count for color, count in zip(KEY_ORDER, rng.sample([2, 3, 4, 5], 3))}

        patches_by_color: dict[int, list[np.ndarray]] = {}
        ok = True
        for color in KEY_ORDER:
            patches = _generate_color_pieces(color, band_h, counts[color], rng)
            if patches is None:
                ok = False
                break
            patches_by_color[color] = patches
        if not ok:
            continue

        if max(max(p.shape[0] for p in patches_by_color[color]) for color in KEY_ORDER) != band_h:
            continue

        legend = _build_legend(counts, rng)
        grid_h = rng.randint(27, 30)
        grid_w = rng.randint(27, 30)
        if legend.shape[1] + 2 > grid_w:
            continue
        grid = np.zeros((grid_h, grid_w), dtype=int)
        legend_r = rng.randint(1, grid_h - legend.shape[0] - 1)
        legend_c = rng.randint(1, 2)
        grid[legend_r : legend_r + legend.shape[0], legend_c : legend_c + legend.shape[1]] = legend
        legend_bbox = (legend_r, legend_c, legend_r + legend.shape[0] - 1, legend_c + legend.shape[1] - 1)

        try:
            _place_pieces_on_grid(grid, legend_bbox, patches_by_color, rng)
        except ValueError:
            continue

        expected = _expected_output(counts, band_h)
        try:
            _validate_semantics(grid, expected, counts, band_h)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input after many attempts")
