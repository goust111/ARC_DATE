import numpy as np
from collections import defaultdict


BG = 0
BLUE = 1
FRAME = 5
BAR_POOL = np.array([2, 3, 4, 6, 7, 8, 9], dtype=int)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()

GEN_RETRIES = 300
SHAPE_RETRIES = 200
PLACEMENT_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _trim(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[np.ix_(rows, cols)]


def _mask_from_points(points: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        out[r - r0, c - c0] = True
    return out


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    trimmed = _trim(mask)
    return tuple(tuple(int(v) for v in row) for row in trimmed.astype(int))


def _canonical_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    trimmed = _trim(mask)
    return min(_shape_signature(np.rot90(trimmed, k)) for k in range(4))


def _extract_frame_and_holes(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[dict]]:
    frame_components = _components(grid == FRAME)
    assert frame_components, "missing grey frame"
    frame_points = max(frame_components, key=len)
    r0, c0, r1, c1 = _bbox(frame_points)

    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes: list[dict] = []
    for pts in _components(crop == BG):
        if any(rr == 0 or rr == crop.shape[0] - 1 or cc == 0 or cc == crop.shape[1] - 1 for rr, cc in pts):
            continue
        local_mask = _mask_from_points(pts)
        abs_cells = [(r0 + rr, c0 + cc) for rr, cc in pts]
        holes.append(
            {
                "cells": abs_cells,
                "bbox": _bbox(abs_cells),
                "area": int(len(pts)),
                "sig": _canonical_signature(local_mask),
            }
        )
    assert holes, "frame must contain holes"
    return (r0, c0, r1, c1), holes


def _extract_fragments(grid: np.ndarray) -> list[dict]:
    fragments: list[dict] = []
    for pts in _components(grid == BLUE):
        mask = _mask_from_points(pts)
        fragments.append(
            {
                "cells": pts,
                "bbox": _bbox(pts),
                "area": int(len(pts)),
                "sig": _canonical_signature(mask),
            }
        )
    assert fragments, "missing blue fragments"
    return fragments


def _extract_bar(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    legend_mask = (grid != BG) & (grid != BLUE) & (grid != FRAME)
    pts = np.argwhere(legend_mask)
    assert len(pts) > 0, "missing color bar"
    cols = np.unique(pts[:, 1])
    assert len(cols) == 1, "color bar must be a single column"
    col = int(cols[0])
    rows = np.sort(pts[:, 0])
    assert int(rows[-1] - rows[0] + 1) == len(rows), "color bar must be contiguous"
    colors = [int(grid[int(r), col]) for r in rows]
    return colors, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _, holes = _extract_frame_and_holes(g)
    fragments = _extract_fragments(g)
    bar_colors, legend_mask = _extract_bar(g)
    assert len(bar_colors) == len(fragments) == len(holes)

    ranked_fragments = sorted(
        enumerate(fragments),
        key=lambda item: (-item[1]["area"], item[1]["bbox"][0], item[1]["bbox"][1]),
    )
    color_by_fragment: dict[int, int] = {
        frag_index: int(bar_colors[rank]) for rank, (frag_index, _frag) in enumerate(ranked_fragments)
    }

    holes_by_sig: dict[tuple[tuple[int, ...], ...], list[dict]] = defaultdict(list)
    frags_by_sig: dict[tuple[tuple[int, ...], ...], list[tuple[int, dict]]] = defaultdict(list)
    for hole in holes:
        holes_by_sig[hole["sig"]].append(hole)
    for frag_index, frag in enumerate(fragments):
        frags_by_sig[frag["sig"]].append((frag_index, frag))

    assert set(holes_by_sig) == set(frags_by_sig), "hole and fragment shapes must match"

    out = g.copy()
    out[g == BLUE] = BG
    out[legend_mask] = BG

    for sig, hole_group in holes_by_sig.items():
        frag_group = frags_by_sig[sig]
        assert len(hole_group) == len(frag_group), "shape multiplicities must match"
        hole_group = sorted(hole_group, key=lambda item: (item["bbox"][0], item["bbox"][1]))
        frag_group = sorted(frag_group, key=lambda item: (item[1]["bbox"][0], item[1]["bbox"][1]))
        for hole, (frag_index, _frag) in zip(hole_group, frag_group):
            fill_color = color_by_fragment[frag_index]
            for r, c in hole["cells"]:
                out[r, c] = fill_color

    return out


def _rotate(mask: np.ndarray, k: int) -> np.ndarray:
    return _trim(np.rot90(mask, k))


def _random_polyomino(area: int) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells and nxt not in frontier:
                        frontier.append(nxt)
            weights = []
            for r, c in frontier:
                neighbors = sum((r + dr, c + dc) in cells for dr, dc in DIR4)
                weights.append(neighbors * neighbors)
            weights_arr = np.asarray(weights, dtype=float)
            weights_arr /= weights_arr.sum()
            cells.add(frontier[int(RNG.choice(len(frontier), p=weights_arr))])
        mask = _trim(_mask_from_points(list(cells)))
        h, w = map(int, mask.shape)
        if h < 2 or w < 2:
            continue
        if h > 5 or w > 5:
            continue
        if int(mask.sum()) != area:
            continue
        if area == h * w:
            continue
        return mask
    raise RuntimeError("failed to sample polyomino")


def _can_place(used: np.ndarray, mask: np.ndarray, top: int, left: int, pad: int = 1) -> bool:
    h, w = map(int, mask.shape)
    uh, uw = map(int, used.shape)
    if top < 0 or left < 0 or top + h > uh or left + w > uw:
        return False
    for rr, cc in np.argwhere(mask):
        r = top + int(rr)
        c = left + int(cc)
        r0 = max(0, r - pad)
        r1 = min(uh, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(uw, c + pad + 1)
        if np.any(used[r0:r1, c0:c1]):
            return False
    return True


def _stamp(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for rr, cc in np.argwhere(mask):
        r = top + int(rr)
        c = left + int(cc)
        grid[r, c] = color
        cells.append((r, c))
    return cells


def _mark(used: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    for rr, cc in np.argwhere(mask):
        used[top + int(rr), left + int(cc)] = True


def _place_holes(inner_h: int, inner_w: int, pieces: list[dict]) -> bool:
    used = np.zeros((inner_h, inner_w), dtype=bool)
    for piece in sorted(pieces, key=lambda item: -item["area"]):
        mask = piece["hole_mask"]
        h, w = map(int, mask.shape)
        choices: list[tuple[int, int]] = []
        for top in range(inner_h - h + 1):
            for left in range(inner_w - w + 1):
                if _can_place(used, mask, top, left, pad=1):
                    choices.append((top, left))
        if not choices:
            return False
        top, left = choices[int(RNG.integers(len(choices)))]
        piece["hole_pos"] = (top, left)
        _mark(used, mask, top, left)
    return True


def _build_expected(grid: np.ndarray, pieces: list[dict], bar_colors: list[int]) -> np.ndarray:
    expected = np.array(grid, copy=True)
    expected[expected == BLUE] = BG
    legend_mask = (expected != BG) & (expected != FRAME)
    expected[legend_mask] = BG

    ranked = sorted(pieces, key=lambda item: (-item["area"], item["fragment_bbox"][0], item["fragment_bbox"][1]))
    for color, piece in zip(bar_colors, ranked):
        for r, c in piece["hole_cells"]:
            expected[r, c] = int(color)
    return expected


def _gen_attempt() -> np.ndarray:
    count = int(RNG.integers(2, 6))
    areas = list(map(int, RNG.choice(np.arange(5, 11), size=count, replace=False)))
    pieces: list[dict] = []
    seen_sigs: set[tuple[tuple[int, ...], ...]] = set()
    for area in areas:
        for _ in range(SHAPE_RETRIES):
            base = _random_polyomino(area)
            sig = _canonical_signature(base)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            pieces.append(
                {
                    "area": area,
                    "base_mask": base,
                    "sig": sig,
                    "hole_mask": _rotate(base, int(RNG.integers(4))),
                    "fragment_mask": _rotate(base, int(RNG.integers(4))),
                }
            )
            break
        else:
            raise RuntimeError("failed to find unique shapes")

    total_area = sum(piece["area"] for piece in pieces)
    max_h = max(piece["hole_mask"].shape[0] for piece in pieces)
    max_w = max(piece["hole_mask"].shape[1] for piece in pieces)
    inner_h = max(max_h + 4, int(np.ceil(np.sqrt(total_area * 3.0))) + 3) + int(RNG.integers(0, 3))
    inner_w = max(max_w + 5, int(np.ceil(np.sqrt(total_area * 3.2))) + 3) + int(RNG.integers(0, 4))

    if not _place_holes(inner_h, inner_w, pieces):
        raise RuntimeError("failed to place holes")

    frame_h = inner_h + 2
    frame_w = inner_w + 2
    min_h = max(20, frame_h + 7)
    min_w = max(24, frame_w + 10)
    max_h_grid = min(30, min_h + 5)
    max_w_grid = min(30, min_w + 5)
    if min_h > 30 or min_w > 30:
        raise RuntimeError("instance too large")

    grid_h = int(RNG.integers(min_h, max_h_grid + 1))
    grid_w = int(RNG.integers(min_w, max_w_grid + 1))
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    bar_colors = list(map(int, RNG.choice(BAR_POOL, size=count, replace=False)))
    bar_col = grid_w - 1 - int(RNG.integers(0, 2))
    bar_top = grid_h - count - int(RNG.integers(0, 2))
    if bar_top < 1:
        raise RuntimeError("color bar does not fit")
    for idx, color in enumerate(bar_colors):
        grid[bar_top + idx, bar_col] = color

    frame_top_limit = min(max(1, grid_h // 3), grid_h - frame_h - 2)
    frame_left_limit = min(max(1, grid_w // 3), grid_w - frame_w - 4)
    if frame_top_limit < 1 or frame_left_limit < 1:
        raise RuntimeError("frame does not fit")
    frame_top = int(RNG.integers(1, frame_top_limit + 1))
    frame_left = int(RNG.integers(1, frame_left_limit + 1))

    if frame_top + frame_h + 2 >= bar_top and frame_left + frame_w + 2 >= bar_col:
        raise RuntimeError("frame too close to color bar")

    grid[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = FRAME
    for piece in pieces:
        hole_top, hole_left = piece["hole_pos"]
        cells = _stamp(grid, piece["hole_mask"], frame_top + 1 + hole_top, frame_left + 1 + hole_left, BG)
        piece["hole_cells"] = cells

    used = np.zeros_like(grid, dtype=bool)
    used[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = True
    for idx in range(count):
        used[bar_top + idx, bar_col] = True

    for piece in sorted(pieces, key=lambda item: -max(item["fragment_mask"].shape)):
        mask = piece["fragment_mask"]
        h, w = map(int, mask.shape)
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = int(RNG.integers(1, grid_h - h))
            left = int(RNG.integers(1, grid_w - w))
            if not _can_place(used, mask, top, left, pad=1):
                continue
            cells = _stamp(grid, mask, top, left, BLUE)
            _mark(used, mask, top, left)
            piece["fragment_bbox"] = _bbox(cells)
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place fragments")

    _, holes = _extract_frame_and_holes(grid)
    fragments = _extract_fragments(grid)
    colors, legend_mask = _extract_bar(grid)
    assert len(holes) == count
    assert len(fragments) == count
    assert len(colors) == count
    assert len({frag["area"] for frag in fragments}) == count
    assert sorted(hole["sig"] for hole in holes) == sorted(frag["sig"] for frag in fragments)
    assert int(np.sum(legend_mask)) == count

    expected = _build_expected(grid, pieces, bar_colors)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid instance")
