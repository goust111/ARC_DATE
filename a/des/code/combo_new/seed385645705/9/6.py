from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
VETO = 2
VOTE_COLORS = (1, 3, 4, 6, 7, 8, 9)

CANVAS_SIZE = 30
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200
PADDING = 1


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
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
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        coords.add((r, c0))
        coords.add((r, c1))
    for c in range(c0, c1 + 1):
        coords.add((r0, c))
        coords.add((r1, c))
    return coords


def _side_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0 + 1, c1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _find_rect_frames(grid: np.ndarray) -> list[dict[str, object]]:
    border_mask = (grid == FRAME) | (grid == VETO)
    frames: list[dict[str, object]] = []

    for pts in _components(border_mask):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        border = set(pts)
        perimeter = _perimeter_coords(r0, c0, r1, c1)
        if border != perimeter:
            continue

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "border": border,
                "has_veto": any(int(grid[r, c]) == VETO for r, c in border),
            }
        )

    frames.sort(key=lambda frame: tuple(frame["bbox"]))  # type: ignore[arg-type]
    return frames


def _vote_colors(interior: np.ndarray) -> tuple[int, int] | None:
    values = [
        int(v)
        for v in interior.ravel()
        if int(v) not in (BG, FRAME, VETO)
    ]
    counts = Counter(values)
    if len(counts) != 2:
        return None
    ranked = counts.most_common()
    if ranked[0][1] == ranked[1][1]:
        return None
    return int(ranked[0][0]), int(ranked[1][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed rectangle by its vote majority unless a red border veto flips it to the minority color.

    Input structure:
    - Black background with several separate 1-pixel-thick rectangular frames.
    - Frames are grey, except that some frames replace one border pixel with a red veto marker.
    - Inside each frame are a few scattered vote pixels using exactly two non-black colors with unequal counts.

    Rule:
    - Detect every rectangular border component made from grey and optional red border pixels.
    - For each frame, count the two vote colors strictly inside the border to get a majority color and a minority color.
    - Fill all black interior cells with the majority color when the frame has no red veto marker.
    - If the frame has a red veto marker, fill all black interior cells with the minority color instead.
    - Preserve the original interior vote pixels, and recolor any red veto marker back to grey.

    Key insight:
    - The dominant pattern is majority-color filling, but a tiny red anomaly on the frame border reverses that decision for just that region.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for frame in _find_rect_frames(g):
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        votes = _vote_colors(interior)
        if votes is None:
            continue

        majority, minority = votes
        fill_color = minority if frame["has_veto"] else majority  # type: ignore[index]

        filled = out[r0 + 1 : r1, c0 + 1 : c1].copy()
        filled[filled == BG] = fill_color
        out[r0 + 1 : r1, c0 + 1 : c1] = filled

        if frame["has_veto"]:  # type: ignore[index]
            for br, bc in frame["border"]:  # type: ignore[index]
                if out[br, bc] == VETO:
                    out[br, bc] = FRAME

    return out


def _count_choices(interior_area: int) -> list[tuple[int, int]]:
    max_markers = min(5, interior_area - 1)
    return [
        pair
        for pair in ((2, 1), (3, 1), (3, 2), (4, 1))
        if sum(pair) <= max_markers
    ]


def _place_region(
    canvas: np.ndarray,
    occupied: np.ndarray,
    has_veto: bool,
    rng: random.Random,
) -> bool:
    for _ in range(PLACEMENT_RETRIES):
        inner_h = rng.randint(2, 6)
        inner_w = rng.randint(2, 7)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        count_choices = _count_choices(inner_h * inner_w)
        if not count_choices:
            continue

        r0 = rng.randint(PADDING, CANVAS_SIZE - outer_h - PADDING)
        c0 = rng.randint(PADDING, CANVAS_SIZE - outer_w - PADDING)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        rr0 = r0 - PADDING
        rr1 = r1 + PADDING
        cc0 = c0 - PADDING
        cc1 = c1 + PADDING
        if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue

        canvas[r0 : r1 + 1, c0] = FRAME
        canvas[r0 : r1 + 1, c1] = FRAME
        canvas[r0, c0 : c1 + 1] = FRAME
        canvas[r1, c0 : c1 + 1] = FRAME

        majority_color, minority_color = rng.sample(list(VOTE_COLORS), 2)
        majority_count, minority_count = rng.choice(count_choices)

        cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        rng.shuffle(cells)
        vote_cells = cells[: majority_count + minority_count]
        for vr, vc in vote_cells[:majority_count]:
            canvas[vr, vc] = majority_color
        for vr, vc in vote_cells[majority_count:]:
            canvas[vr, vc] = minority_color

        if has_veto:
            mr, mc = rng.choice(_side_coords(r0, c0, r1, c1))
            canvas[mr, mc] = VETO

        occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        return True

    return False


def _crop_canvas(canvas: np.ndarray, rng: random.Random) -> np.ndarray:
    pts = np.argwhere(canvas != BG)
    rmin, cmin = pts.min(axis=0)
    rmax, cmax = pts.max(axis=0)

    top = rng.randint(1, min(3, int(rmin)))
    left = rng.randint(1, min(3, int(cmin)))
    bottom = rng.randint(1, min(3, int(CANVAS_SIZE - 1 - rmax)))
    right = rng.randint(1, min(3, int(CANVAS_SIZE - 1 - cmax)))

    return canvas[rmin - top : rmax + bottom + 1, cmin - left : cmax + right + 1].copy()


def _generated_grid_is_valid(grid: np.ndarray, expected_regions: int) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    frames = _find_rect_frames(grid)
    if len(frames) != expected_regions:
        return False

    veto_flags: list[bool] = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            return False
        if np.count_nonzero(interior == BG) == 0:
            return False
        if _vote_colors(interior) is None:
            return False
        if np.any((interior == FRAME) | (interior == VETO)):
            return False
        veto_flags.append(bool(frame["has_veto"]))  # type: ignore[index]

    if not any(veto_flags) or all(veto_flags):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.any(out == VETO):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    occupied = np.zeros_like(canvas, dtype=bool)

    n_regions = rng.randint(2, 4)
    veto_count = rng.choice((n_regions // 2, (n_regions + 1) // 2))
    veto_flags = [True] * veto_count + [False] * (n_regions - veto_count)
    rng.shuffle(veto_flags)

    for has_veto in veto_flags:
        if not _place_region(canvas, occupied, has_veto, rng):
            return None

    cropped = _crop_canvas(canvas, rng)
    if not _generated_grid_is_valid(cropped, n_regions):
        return None
    return cropped


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to generate a valid grid")
