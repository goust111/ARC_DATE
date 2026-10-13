from __future__ import annotations

from collections import Counter
import random

import numpy as np


GEN_BG = 8
GEN_FRAME = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 9)
SIDES = ("top", "right", "bottom", "left")
ROTATION_K = {"top": 0, "right": 3, "bottom": 2, "left": 1}
NEIGHBORS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS_8 = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _connected_components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neighbors = NEIGHBORS_4 if connectivity == 4 else NEIGHBORS_8
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_patch(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch, (r0, c0)


def _border_coords(r0: int, c0: int, size: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c0 + size):
        coords.append((r0, c))
        coords.append((r0 + size - 1, c))
    for r in range(r0 + 1, r0 + size - 1):
        coords.append((r, c0))
        coords.append((r, c0 + size - 1))
    return coords


def _marker_side(r0: int, c0: int, size: int, mr: int, mc: int) -> str | None:
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    if mr == r0 and c0 < mc < c1:
        return "top"
    if mr == r1 and c0 < mc < c1:
        return "bottom"
    if mc == c0 and r0 < mr < r1:
        return "left"
    if mc == c1 and r0 < mr < r1:
        return "right"
    return None


def _canvas_target(canvas: dict) -> tuple[int, int]:
    r0 = int(canvas["r0"])
    c0 = int(canvas["c0"])
    size = int(canvas["size"])
    mr, mc = canvas["marker"]
    side = str(canvas["side"])
    if side == "top":
        return r0 + 1, mc
    if side == "right":
        return mr, c0 + size - 2
    if side == "bottom":
        return r0 + size - 2, mc
    return mr, c0 + 1


def _find_canvases(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    candidates: list[dict] = []
    for size in range(5, min(7, h, w) + 1):
        for r0 in range(0, h - size + 1):
            for c0 in range(0, w - size + 1):
                interior = grid[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1]
                if interior.size == 0 or not np.all(interior == bg):
                    continue
                border = _border_coords(r0, c0, size)
                vals = [int(grid[r, c]) for r, c in border]
                if bg in vals:
                    continue
                counts = Counter(vals)
                if len(counts) != 2:
                    continue
                (frame_color, frame_count), (marker_color, marker_count) = counts.most_common(2)
                if frame_count != len(border) - 1 or marker_count != 1:
                    continue
                marker = next((r, c) for r, c in border if int(grid[r, c]) == marker_color)
                side = _marker_side(r0, c0, size, marker[0], marker[1])
                if side is None:
                    continue
                candidates.append(
                    {
                        "r0": r0,
                        "c0": c0,
                        "r1": r0 + size - 1,
                        "c1": c0 + size - 1,
                        "size": size,
                        "frame_color": frame_color,
                        "marker_color": marker_color,
                        "marker": marker,
                        "side": side,
                    }
                )

    chosen: list[dict] = []
    used_colors: set[int] = set()
    for cand in sorted(candidates, key=lambda item: (item["r0"], item["c0"], item["size"])):
        if int(cand["marker_color"]) in used_colors:
            continue
        if any(
            not (
                cand["r1"] < prev["r0"]
                or prev["r1"] < cand["r0"]
                or cand["c1"] < prev["c0"]
                or prev["c1"] < cand["c0"]
            )
            for prev in chosen
        ):
            continue
        cand["target"] = _canvas_target(cand)
        chosen.append(cand)
        used_colors.add(int(cand["marker_color"]))
    return chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each colored stamp by the border-marker direction and stamp it into the matching framed canvas.

    Input structure:
    - The background is azure and dominates the grid.
    - Several hollow square canvases are outlined by a single border color; each canvas has exactly one non-border marker pixel embedded on one side of that border, and the interior is empty background.
    - Separate multicolor stamp sprites lie outside the canvases; each stamp contains exactly one anchor pixel whose color matches one canvas marker color.

    Rule:
    - Pair every outside stamp with the canvas whose border marker has the same color as the stamp's unique anchor pixel.
    - Read the marker side on that canvas to choose the stamp rotation: top -> 0 degrees, right -> clockwise 90 degrees, bottom -> 180 degrees, left -> counterclockwise 90 degrees.
    - Erase the outside stamps, keep every canvas border unchanged, rotate the paired stamp, and place it inside the canvas so the anchor pixel lands one cell inward from the border marker.

    Key insight:
    - The marker color identifies which canvas receives a stamp, while the marker side independently encodes the rotation that must be applied before stamping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    canvases = _find_canvases(g, bg)
    assert canvases, "expected at least one valid canvas"

    by_color = {int(canvas["marker_color"]): canvas for canvas in canvases}
    preserve = np.zeros(g.shape, dtype=bool)
    for canvas in canvases:
        for r, c in _border_coords(int(canvas["r0"]), int(canvas["c0"]), int(canvas["size"])):
            preserve[r, c] = True

    out = g.copy()
    out[(g != bg) & (~preserve)] = bg

    stamp_mask = (g != bg) & (~preserve)
    stamp_components = _connected_components(stamp_mask, connectivity=8)
    assert len(stamp_components) == len(canvases), "expected one outside stamp per canvas"

    matched_colors: set[int] = set()
    for comp in stamp_components:
        anchors = [(r, c, int(g[r, c])) for r, c in comp if int(g[r, c]) in by_color]
        assert len(anchors) == 1, "each stamp must contain exactly one canvas-marker color"
        anchor_r, anchor_c, anchor_color = anchors[0]
        assert anchor_color not in matched_colors, "marker colors should match stamps uniquely"
        matched_colors.add(anchor_color)

        patch, _origin = _extract_patch(g, comp, bg)
        rotated = np.rot90(patch, ROTATION_K[str(by_color[anchor_color]["side"])])
        anchor_locs = np.argwhere(rotated == anchor_color)
        assert anchor_locs.shape[0] == 1, "rotated stamp must keep a unique anchor pixel"
        local_anchor_r, local_anchor_c = map(int, anchor_locs[0])
        target_r, target_c = by_color[anchor_color]["target"]
        canvas = by_color[anchor_color]

        for rr, cc in np.argwhere(rotated != bg):
            gr = int(target_r + int(rr) - local_anchor_r)
            gc = int(target_c + int(cc) - local_anchor_c)
            assert canvas["r0"] < gr < canvas["r1"], "stamped row must stay inside the canvas"
            assert canvas["c0"] < gc < canvas["c1"], "stamped col must stay inside the canvas"
            out[gr, gc] = int(rotated[rr, cc])

    assert matched_colors == set(by_color), "every canvas marker should be used exactly once"
    return out


def _area_is_free(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0], r0 + h + pad)
    cc1 = min(mask.shape[1], c0 + w + pad)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _mark_area(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    mask[r0 : r0 + h, c0 : c0 + w] = True


def _draw_canvas(grid: np.ndarray, r0: int, c0: int, size: int, marker: tuple[int, int], marker_color: int) -> None:
    grid[r0, c0 : c0 + size] = GEN_FRAME
    grid[r0 + size - 1, c0 : c0 + size] = GEN_FRAME
    grid[r0 : r0 + size, c0] = GEN_FRAME
    grid[r0 : r0 + size, c0 + size - 1] = GEN_FRAME
    grid[marker] = marker_color


def _target_from_spec(spec: dict) -> tuple[int, int]:
    return _canvas_target(spec)


def _grow_stamp_patch(
    inner: int,
    anchor: tuple[int, int],
    anchor_color: int,
    filler_colors: tuple[int, ...],
    rng: random.Random,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    max_cells = min(6, inner * inner - 1)
    if max_cells < 3:
        return None
    for _ in range(200):
        want = rng.randint(3, max_cells)
        cells = {anchor}
        while len(cells) < want:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in NEIGHBORS_8:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < inner and 0 <= nc < inner and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if max(rows) - min(rows) + 1 >= inner or max(cols) - min(cols) + 1 >= inner:
            continue

        r0 = min(rows)
        c0 = min(cols)
        h = max(rows) - r0 + 1
        w = max(cols) - c0 + 1
        patch = np.full((h, w), GEN_BG, dtype=int)
        for r, c in cells:
            if (r, c) == anchor:
                patch[r - r0, c - c0] = anchor_color
            else:
                patch[r - r0, c - c0] = rng.choice(filler_colors)
        return patch, (anchor[0] - r0, anchor[1] - c0)
    return None


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != GEN_BG
    target = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    target[mask] = patch[mask]


def _paint_inside_canvas(
    grid: np.ndarray,
    patch: np.ndarray,
    anchor_local: tuple[int, int],
    target: tuple[int, int],
) -> None:
    ar, ac = anchor_local
    tr, tc = target
    for r, c in np.argwhere(patch != GEN_BG):
        grid[tr + int(r) - ar, tc + int(c) - ac] = int(patch[r, c])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    canvas_count = rng.randint(2, 4)
    min_side = {2: 18, 3: 22, 4: 24}[canvas_count]
    h = rng.randint(min_side, 30)
    w = rng.randint(min_side, 30)

    grid = np.full((h, w), GEN_BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    sides = list(SIDES)
    rng.shuffle(sides)
    sides = sides[:canvas_count]
    rng.shuffle(sides)
    marker_colors = rng.sample(MARKER_COLORS, canvas_count)
    filler_colors = tuple(color for color in range(10) if color not in {GEN_BG, GEN_FRAME, *marker_colors})
    if len(filler_colors) < 2:
        return None

    specs: list[dict] = []
    for side, marker_color in zip(sides, marker_colors):
        size = rng.randint(5, 7)
        placed = False
        for _ in range(200):
            r0 = rng.randint(0, h - size)
            c0 = rng.randint(0, w - size)
            if not _area_is_free(occupied, r0, c0, size, size, pad=2):
                continue
            if side in ("top", "bottom"):
                marker_col = rng.randint(c0 + 1, c0 + size - 2)
                marker = (r0 if side == "top" else r0 + size - 1, marker_col)
            else:
                marker_row = rng.randint(r0 + 1, r0 + size - 2)
                marker = (marker_row, c0 if side == "left" else c0 + size - 1)
            specs.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "r1": r0 + size - 1,
                    "c1": c0 + size - 1,
                    "size": size,
                    "side": side,
                    "marker": marker,
                    "marker_color": marker_color,
                }
            )
            _mark_area(occupied, r0, c0, size, size)
            placed = True
            break
        if not placed:
            return None

    for spec in specs:
        _draw_canvas(grid, int(spec["r0"]), int(spec["c0"]), int(spec["size"]), spec["marker"], int(spec["marker_color"]))
        spec["target"] = _target_from_spec(spec)

    expected = grid.copy()

    for spec in specs:
        size = int(spec["size"])
        inner = size - 2
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        target_r, target_c = spec["target"]
        anchor_inner = (target_r - (r0 + 1), target_c - (c0 + 1))

        final_info = _grow_stamp_patch(inner, anchor_inner, int(spec["marker_color"]), filler_colors, rng)
        if final_info is None:
            return None
        final_patch, final_anchor = final_info

        source_patch = np.rot90(final_patch, (-ROTATION_K[str(spec["side"])]) % 4)
        source_h, source_w = source_patch.shape
        if source_h >= inner or source_w >= inner:
            return None

        placed = False
        for _ in range(300):
            sr = rng.randint(0, h - source_h)
            sc = rng.randint(0, w - source_w)
            if not _area_is_free(occupied, sr, sc, source_h, source_w, pad=1):
                continue
            _place_patch(grid, source_patch, sr, sc)
            _mark_area(occupied, sr, sc, source_h, source_w)
            placed = True
            break
        if not placed:
            return None

        _paint_inside_canvas(expected, final_patch, final_anchor, (target_r, target_c))

    out = main(grid)
    if np.array_equal(out, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-to-canvas ARC input")
