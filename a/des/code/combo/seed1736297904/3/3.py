from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
PALETTE = tuple(color for color in range(10) if color not in (BG, RED))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_patch(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BG)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotate_cw(arr: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(arr, -int(turns)).copy()


def _find_target(grid: np.ndarray) -> dict[str, int]:
    h, w = grid.shape
    best: tuple[tuple[float, int], dict[str, int]] | None = None

    for side in range(4, min(h, w) + 1):
        for top in range(0, h - side + 1):
            bottom = top + side - 1
            for left in range(0, w - side + 1):
                right = left + side - 1

                border = np.zeros((side, side), dtype=bool)
                border[0, :] = True
                border[-1, :] = True
                border[:, 0] = True
                border[:, -1] = True
                sub = grid[top : bottom + 1, left : right + 1]
                if not np.all(sub[border] == RED):
                    continue
                if not np.all(sub[~border] == BG):
                    continue

                row_hits = []
                for r in range(top + 1, bottom):
                    if np.all(grid[r, : left + 1] == RED) and np.all(grid[r, right:] == RED):
                        row_hits.append(r)
                col_hits = []
                for c in range(left + 1, right):
                    if np.all(grid[: top + 1, c] == RED) and np.all(grid[bottom:, c] == RED):
                        col_hits.append(c)

                if len(row_hits) != 1 or len(col_hits) != 1:
                    continue

                cross_r = int(row_hits[0])
                cross_c = int(col_hits[0])
                if cross_r <= top or cross_r >= bottom or cross_c <= left or cross_c >= right:
                    continue
                if cross_r == top + 1 or cross_r == bottom or cross_c == left + 1 or cross_c == right:
                    continue

                info = {
                    "top": int(top),
                    "left": int(left),
                    "bottom": int(bottom),
                    "right": int(right),
                    "cross_r": cross_r,
                    "cross_c": cross_c,
                }
                center_score = abs((top + bottom) - (h - 1)) + abs((left + right) - (w - 1))
                key = (float(center_score), -side)
                if best is None or key < best[0]:
                    best = (key, info)

    assert best is not None, "failed to locate central target square"
    return best[1]


def _source_quadrant(coords: np.ndarray, cross_r: int, cross_c: int) -> str:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 < cross_r and c1 < cross_c:
        return "ul"
    if r1 < cross_r and c0 > cross_c:
        return "ur"
    if r0 > cross_r and c0 > cross_c:
        return "lr"
    if r0 > cross_r and c1 < cross_c:
        return "ll"
    raise AssertionError("foreground object does not lie in a unique quadrant")


def _target_start(info: dict[str, int], quadrant: str, patch: np.ndarray) -> tuple[int, int]:
    h, w = patch.shape
    top = info["top"]
    left = info["left"]
    bottom = info["bottom"]
    right = info["right"]
    cross_r = info["cross_r"]
    cross_c = info["cross_c"]

    if quadrant == "ul":
        assert h <= cross_r - (top + 1)
        assert w <= cross_c - (left + 1)
        return top + 1, left + 1
    if quadrant == "ur":
        assert h <= cross_r - (top + 1)
        assert w <= right - cross_c
        return top + 1, right - w
    if quadrant == "lr":
        assert h <= bottom - cross_r
        assert w <= right - cross_c
        return bottom - h, right - w
    assert quadrant == "ll"
    assert h <= bottom - cross_r
    assert w <= cross_c - (left + 1)
    return bottom - h, left + 1


def _paste_patch(out: np.ndarray, patch: np.ndarray, start_r: int, start_c: int) -> None:
    mask = patch != BG
    h, w = patch.shape
    dest = out[start_r : start_r + h, start_c : start_c + w]
    assert dest.shape == patch.shape
    assert np.all(dest[mask] == BG)
    dest[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each quadrant object by a quadrant-dependent angle and move it into the
    matching quadrant of the central target box.

    Input structure:
    - Even-sized square grid with black background.
    - Red horizontal and vertical divider arms form four quadrants.
    - A red hollow square near the crossing creates a blank central target area.
    - Each outer quadrant contains exactly one non-red, non-black monochrome 4-connected object.

    Rule:
    - Keep the red cross structure unchanged.
    - Move the upper-left object with 0-degree rotation into the target's upper-left quadrant.
    - Move the upper-right object after a 90-degree clockwise rotation into the target's upper-right quadrant.
    - Move the lower-right object after a 180-degree clockwise rotation into the target's lower-right quadrant.
    - Move the lower-left object after a 270-degree clockwise rotation into the target's lower-left quadrant.
    - Remove the original outer objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1] and g.shape[0] % 2 == 0

    info = _find_target(g)
    fg_mask = (g != BG) & (g != RED)
    comps = _components(fg_mask)
    assert len(comps) == 4, "expected exactly four source objects"

    turns_by_quad = {"ul": 0, "ur": 1, "lr": 2, "ll": 3}
    out = g.copy()

    for coords in comps:
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1
        out[coords[:, 0], coords[:, 1]] = BG

    for coords in comps:
        quadrant = _source_quadrant(coords, info["cross_r"], info["cross_c"])
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch = _crop_patch(patch)
        patch = _rotate_cw(patch, turns_by_quad[quadrant])
        start_r, start_c = _target_start(info, quadrant, patch)
        _paste_patch(out, patch, start_r, start_c)

    return out


GRID_SIZES = (20, 22, 24)
INNER_SIZES = (6, 8, 10)
GEN_ATTEMPTS = 400


def _build_structure(side: int, inner: int) -> tuple[np.ndarray, dict[str, int]]:
    grid = np.full((side, side), BG, dtype=int)
    cross_r = cross_c = side // 2
    top = cross_r - inner // 2 - 1
    left = cross_c - inner // 2 - 1
    bottom = top + inner + 1
    right = left + inner + 1

    grid[top, left : right + 1] = RED
    grid[bottom, left : right + 1] = RED
    grid[top : bottom + 1, left] = RED
    grid[top : bottom + 1, right] = RED
    grid[cross_r, : left + 1] = RED
    grid[cross_r, right:] = RED
    grid[: top + 1, cross_c] = RED
    grid[bottom:, cross_c] = RED

    info = {
        "top": int(top),
        "left": int(left),
        "bottom": int(bottom),
        "right": int(right),
        "cross_r": int(cross_r),
        "cross_c": int(cross_c),
    }
    return grid, info


def _shape_invariant(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    mask = _crop_mask(mask)
    return tuple(tuple(int(x) for x in row) for row in mask.astype(int))


def _is_asymmetric(mask: np.ndarray) -> bool:
    base = _shape_invariant(mask)
    for turns in (1, 2, 3):
        if _shape_invariant(np.rot90(mask, turns)) == base:
            return False
    return True


def _random_shape(limit: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(200):
        h = rng.randint(2, limit)
        w = rng.randint(2, limit)
        max_area = min(h * w - 1, limit * limit - 1)
        if max_area < 3:
            continue
        area = rng.randint(3, max_area)

        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        frontier = set()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))

        while len(cells) < area and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)

        if mask.shape[0] > limit or mask.shape[1] > limit:
            continue
        if not _is_asymmetric(mask):
            continue

        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        mask = np.rot90(mask, rng.randrange(4)).copy()
        return _crop_mask(mask)
    return None


def _quadrant_fits(quadrant: str, r0: int, c0: int, h: int, w: int, cross_r: int, cross_c: int) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if quadrant == "ul":
        return r1 < cross_r and c1 < cross_c
    if quadrant == "ur":
        return r1 < cross_r and c0 > cross_c
    if quadrant == "ll":
        return r0 > cross_r and c1 < cross_c
    return r0 > cross_r and c0 > cross_c


def _place_source_object(
    canvas: np.ndarray,
    mask: np.ndarray,
    color: int,
    quadrant: str,
    info: dict[str, int],
    rng: random.Random,
) -> np.ndarray | None:
    h, w = mask.shape
    side = canvas.shape[0]
    candidates: list[tuple[int, int]] = []
    hole_top = info["top"] + 1
    hole_left = info["left"] + 1
    hole_bottom = info["bottom"] - 1
    hole_right = info["right"] - 1
    for r0 in range(1, side - h):
        for c0 in range(1, side - w):
            if not _quadrant_fits(quadrant, r0, c0, h, w, info["cross_r"], info["cross_c"]):
                continue
            r1 = r0 + h - 1
            c1 = c0 + w - 1
            if not (r1 < hole_top or r0 > hole_bottom or c1 < hole_left or c0 > hole_right):
                continue
            if np.any(canvas[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] != BG):
                continue
            candidates.append((r0, c0))

    if not candidates:
        return None

    r0, c0 = rng.choice(candidates)
    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = color
    canvas[r0 : r0 + h, c0 : c0 + w][mask] = color
    return patch


def _expected_output(structure: np.ndarray, sources: dict[str, np.ndarray], info: dict[str, int]) -> np.ndarray:
    turns_by_quad = {"ul": 0, "ur": 1, "lr": 2, "ll": 3}
    out = structure.copy()
    for quadrant, patch in sources.items():
        rotated = _rotate_cw(patch, turns_by_quad[quadrant])
        start_r, start_c = _target_start(info, quadrant, rotated)
        _paste_patch(out, rotated, start_r, start_c)
    return out


def _gen_case(rng: random.Random) -> np.ndarray | None:
    side = rng.choice(GRID_SIZES)
    valid_inner = [inner for inner in INNER_SIZES if inner + 8 <= side]
    inner = rng.choice(valid_inner)
    structure, info = _build_structure(side, inner)
    canvas = structure.copy()

    quadrant_span = inner // 2
    shape_limit = max(2, quadrant_span - 1)
    colors = rng.sample(PALETTE, 4)

    sources: dict[str, np.ndarray] = {}
    for quadrant, color in zip(("ul", "ur", "lr", "ll"), colors, strict=False):
        mask = _random_shape(shape_limit, rng)
        if mask is None:
            return None
        patch = _place_source_object(canvas, mask, color, quadrant, info, rng)
        if patch is None:
            return None
        sources[quadrant] = patch

    expected = _expected_output(structure, sources, info)
    if not np.array_equal(main(canvas), expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
