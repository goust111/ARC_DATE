from __future__ import annotations

from pathlib import Path
import random
import sys
from collections import Counter
from typing import Iterator, Sequence

import numpy as np


# Ensure arc_utils can be imported when this file is loaded by SourceFileLoader.
for _p in Path(__file__).resolve().parents:
    _utils_candidate = _p / "new" / "solvers" / "arc_utils.py"
    if _utils_candidate.exists():
        _utils_dir = str(_utils_candidate.parent)
        if _utils_dir not in sys.path:
            sys.path.append(_utils_dir)
        break
else:  # pragma: no cover
    _utils_dir = None

try:  # pragma: no branch
    from arc_utils import Color, bbox_coords, bbox, components_mask, crop_bbox, get_bg, scale_sprite
except Exception:  # pragma: no cover
    # Minimal fallback implementations, used only if arc_utils cannot be imported.
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9
        NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])

    def bbox(mask: np.ndarray):
        cells = np.argwhere(mask)
        if cells.size == 0:
            return None
        r0, c0 = cells.min(axis=0)
        r1, c1 = cells.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def bbox_coords(coords: np.ndarray):
        coords = np.asarray(coords)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, bb: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = bb
        return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], copy=True)

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        if mask.dtype != bool:
            mask = mask.astype(bool)
        h, w = mask.shape
        visited = np.zeros((h, w), dtype=bool)
        if connectivity == 4:
            steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
        else:
            steps = tuple(
                (dr, dc)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if not (dr == 0 and dc == 0)
            )

        comps: list[np.ndarray] = []
        for r0 in range(h):
            for c0 in range(w):
                if not mask[r0, c0] or visited[r0, c0]:
                    continue
                stack = [(r0, c0)]
                visited[r0, c0] = True
                comp: list[tuple[int, int]] = []
                while stack:
                    r, c = stack.pop()
                    comp.append((r, c))
                    for dr, dc in steps:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(np.asarray(comp, dtype=int))
        return comps

    def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
        k = int(factor)
        if k <= 0:
            raise ValueError("scale factor must be >=1")
        return np.kron(np.asarray(sprite), np.ones((k, k), dtype=np.asarray(sprite).dtype))


def _neighbors4(coord: tuple[int, int]) -> Iterator[tuple[int, int]]:
    r, c = coord
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _count_green_full_blocks(grid: np.ndarray) -> int:
    out = 0
    for comp in components_mask(grid == Color.GREEN, connectivity=4):
        if comp.shape[0] != 4:
            continue
        b = bbox_coords(comp)
        if b is None:
            continue
        r0, c0, r1, c1 = b
        if r1 - r0 == 1 and c1 - c0 == 1:
            out += 1
    return out


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int, np.ndarray]:
    marker_coords = np.argwhere(grid == Color.YELLOW)
    assert marker_coords.shape[0] == 4, "expected exactly four yellow markers"

    r0 = int(np.min(marker_coords[:, 0]))
    r1 = int(np.max(marker_coords[:, 0]))
    c0 = int(np.min(marker_coords[:, 1]))
    c1 = int(np.max(marker_coords[:, 1]))

    frame = grid[r0 : r1 + 1, c0 : c1 + 1]
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    for r, c in marker_coords:
        corners.discard((int(r), int(c)))
    assert len(corners) == 0, "yellow pixels must be exactly at the frame corners"

    return r0, c0, r1, c1, np.array(frame, copy=True)


def _extract_sprite(grid: np.ndarray, frame_box: tuple[int, int, int, int]) -> np.ndarray:
    bg = get_bg(grid)
    r0, c0, r1, c1 = frame_box
    # Keep outside-frame components only and ignore green objects used for scale counting.
    outside_mask = (grid != bg) & (grid != Color.YELLOW) & (grid != Color.GREEN)
    outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False

    comps = components_mask(outside_mask, connectivity=4)
    multicolor = []
    for comp in comps:
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) >= 2:
            multicolor.append((len(colors), len(comp), comp))

    assert multicolor, "no multicolor sprite found outside the frame"
    multicolor.sort(key=lambda t: (t[1], t[0]))
    _, _, sprite_comp = multicolor[0]
    sprite_box = bbox_coords(sprite_comp)
    assert sprite_box is not None
    return crop_bbox(grid, sprite_box)


def _evidence_cells(frame: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    H, W = frame.shape
    evidence: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(frame[r, c])
            if v != bg and v != Color.YELLOW:
                evidence.append((r, c, v))
    return evidence


def _scale_and_place(sprite: np.ndarray, scale: int, frame: np.ndarray, bg: int) -> np.ndarray:
    assert scale >= 1, "scale must be >=1"
    big = scale_sprite(sprite, scale)
    Hf, Wf = frame.shape
    Hb, Wb = big.shape
    assert Hb <= Hf and Wb <= Wf, "scaled sprite does not fit the target frame"

    evidence = _evidence_cells(frame, bg)
    assert evidence, "frame must contain at least one color evidence pixel"

    solutions: list[tuple[int, int]] = []
    for r0 in range(Hf - Hb + 1):
        for c0 in range(Wf - Wb + 1):
            ok = True
            # Every visible evidence cell in the frame must match the scaled sprite.
            for er, ec, color in evidence:
                sr = er - r0
                sc = ec - c0
                if not (0 <= sr < Hb and 0 <= sc < Wb):
                    ok = False
                    break
                if int(big[sr, sc]) != color:
                    ok = False
                    break
            if not ok:
                continue

            # Keep yellow markers untouched.
            for sr in range(Hb):
                for sc in range(Wb):
                    fr = r0 + sr
                    fc = c0 + sc
                    if int(big[sr, sc]) != bg and int(frame[fr, fc]) == Color.YELLOW:
                        ok = False
                        break
                if not ok:
                    break

            if ok:
                solutions.append((r0, c0))

    assert len(solutions) == 1, f"placement not unique ({len(solutions)} candidates)"
    top, left = solutions[0]

    out = frame.copy()
    for sr in range(Hb):
        for sc in range(Wb):
            v = int(big[sr, sc])
            if v != bg:
                out[top + sr, left + sc] = v
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Scale and place the outside multi-color sprite into the yellow-cornered frame.

    Input structure:
    - The grid has one small multicolor sprite located outside the yellow-cornered target frame.
    - Four yellow singleton pixels mark the frame corners.
    - The frame interior contains sparse color evidence cells from a scaled copy of the sprite.
    - Elsewhere are green objects, including complete and incomplete 2x2 blocks.

    Rule:
    - Count all full green 2x2 connected blocks in the whole input grid and use the count k.
    - Extract the outside multicolor sprite and scale every pixel by k.
    - Find the unique placement in the yellow frame where every non-background, non-yellow
      evidence cell matches the scaled sprite color at the aligned position.
    - Paste the scaled sprite with background transparency so yellow markers remain unchanged.

    Key insight:
    - The visible frame evidence is purposely under-determined across scales; only the external count
      of complete green 2x2 squares provides the correct scaling factor, then matching fixes the
      placement uniquely.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] <= 30 and g.shape[1] <= 30, "grid must be <=30x30"

    bg = get_bg(g)
    frame_r0, frame_c0, frame_r1, frame_c1, frame = _find_frame(g)

    sprite = _extract_sprite(g, (frame_r0, frame_c0, frame_r1, frame_c1))
    scale = _count_green_full_blocks(g)
    assert scale >= 1, "expected at least one full green 2x2 block"

    out = _scale_and_place(sprite, scale, frame, bg)
    return out


# ---------------- generation helpers ----------------

def _rand_connected_mask(
    h: int,
    w: int,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    assert n >= 1
    start_r = int(rng.integers(0, h))
    start_c = int(rng.integers(0, w))
    pts = {(start_r, start_c)}

    while len(pts) < n:
        frontier = list(pts)
        random.shuffle(frontier)
        added = False
        for r, c in frontier:
            candidates = []
            for nr, nc in _neighbors4((r, c)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in pts:
                    candidates.append((nr, nc))
            if candidates:
                pts.add(random.choice(candidates))
                added = True
                break
        if not added:
            break

    # If growth got stuck, restart from another seed and keep only what we have.
    if len(pts) < n:
        candidates = [(r, c) for r in range(h) for c in range(w)]
        random.shuffle(candidates)
        for p in candidates:
            if len(pts) >= n:
                break
            if p in pts:
                continue
            pts.add(p)

    return _coords_to_mask(pts, h, w)


def _coords_to_mask(points: Sequence[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in points:
        mask[int(r), int(c)] = True
    return mask


def _make_multicolor_sprite(rng: np.random.Generator, color_pool: Sequence[int]) -> np.ndarray:
    h = int(rng.integers(3, 6))
    w = int(rng.integers(3, 6))
    n_cells = int(rng.integers(5, min(h * w, 11) + 1))

    mask = _rand_connected_mask(h, w, n_cells, rng)
    ys, xs = np.where(mask)
    coords = list(zip(ys.tolist(), xs.tolist()))

    num_colors = int(rng.integers(2, 4))
    chosen_colors = random.sample(list(color_pool), k=min(num_colors, len(color_pool)))
    colors = [int(c) for c in chosen_colors]

    if not coords:
        raise RuntimeError("failed to generate sprite cells")

    canvas = np.full((h, w), Color.BLACK, dtype=int)
    rng.shuffle(coords)
    # ensure each chosen color appears at least once when possible
    for i, coord in enumerate(coords[: len(colors)]):
        canvas[coord[0], coord[1]] = colors[i % len(colors)]
    for coord in coords[len(colors) :]:
        canvas[coord[0], coord[1]] = int(rng.choice(colors))

    bb = bbox(mask)
    assert bb is not None
    return crop_bbox(canvas, bb)


def _can_place(patch: np.ndarray, top: int, left: int, occupancy: np.ndarray, bg: int) -> bool:
    h, w = patch.shape
    if top < 0 or left < 0 or top + h > occupancy.shape[0] or left + w > occupancy.shape[1]:
        return False
    occ_slice = occupancy[top : top + h, left : left + w]
    fg_mask = patch != bg
    return not np.any(occ_slice & fg_mask)


def _paint_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    for r in range(h):
        for c in range(w):
            v = int(patch[r, c])
            if v != bg:
                canvas[top + r, left + c] = v


def _collect_cells_for_block(top: int, left: int, size: int = 2) -> list[tuple[int, int]]:
    return [
        (top + dr, left + dc)
        for dr in range(size)
        for dc in range(size)
    ]


def _free_with_margin(
    occupancy: np.ndarray,
    cells: Sequence[tuple[int, int]],
    margin: int = 1,
) -> bool:
    if not cells:
        return False
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = max(0, min(rows) - margin)
    r1 = min(occupancy.shape[0], max(rows) + margin + 1)
    c0 = max(0, min(cols) - margin)
    c1 = min(occupancy.shape[1], max(cols) + margin + 1)

    return not np.any(occupancy[r0:r1, c0:c1])


def _find_free_locations(
    patch: np.ndarray,
    occupancy: np.ndarray,
    bg: int,
    *,
    forbidden: tuple[int, int, int, int] | None = None,
) -> list[tuple[int, int]]:
    h, w = patch.shape
    H, W = occupancy.shape
    out = []
    for top in range(H - h + 1):
        for left in range(W - w + 1):
            if forbidden is not None:
                fr0, fc0, fr1, fc1 = forbidden
                if fr0 <= top < fr1 and fc0 <= left < fc1:
                    continue
            if _can_place(patch, top, left, occupancy, bg):
                out.append((top, left))
    return out

def _paste_sprite_sparse_evidence(
    frame: np.ndarray,
    sprite: np.ndarray,
    rng: np.random.Generator,
    scale: int,
    bg: int,
) -> np.ndarray:
    scaled = scale_sprite(sprite, scale)
    Hf, Wf = frame.shape
    Hs, Ws = scaled.shape

    max_off_r = Hf - Hs
    max_off_c = Wf - Ws
    assert max_off_r >= 3 and max_off_c >= 3, "frame must allow non-corner placement"

    top = int(rng.integers(1, max_off_r))
    left = int(rng.integers(1, max_off_c))

    out = frame.copy()
    for r in range(Hs):
        for c in range(Ws):
            v = int(scaled[r, c])
            if v != bg:
                out[top + r, left + c] = v

    # Sparse evidence: keep only part of the sprite cells.
    keep = 0.35 + float(rng.random()) * 0.45 if rng.random() >= 0.45 else 0.18
    kept = 0
    for r in range(Hs):
        for c in range(Ws):
            rr, cc = top + r, left + c
            v = int(out[rr, cc])
            if v == bg:
                continue
            if rng.random() > keep:
                out[rr, cc] = bg
            else:
                kept += 1

    if kept == 0:
        rr = int(rng.integers(0, Hs))
        cc = int(rng.integers(0, Ws))
        if int(scaled[rr, cc]) != bg:
            out[top + rr, left + cc] = int(scaled[rr, cc])
    return out


def _place_full_block_grid(
    grid: np.ndarray,
    occupancy: np.ndarray,
    top: int,
    left: int,
) -> None:
    cells = _collect_cells_for_block(top, left, size=2)
    for r, c in cells:
        grid[r, c] = Color.GREEN
        occupancy[r, c] = True


def _place_partial_block_grid(grid: np.ndarray, occupancy: np.ndarray, color: int) -> None:
    H, W = grid.shape
    if H < 2 or W < 2:
        return

    anchors = []
    for r in range(H - 1):
        for c in range(W - 1):
            cells = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
            if _free_with_margin(occupancy, cells, margin=1):
                anchors.append((r, c))

    if not anchors:
        return

    r0, c0 = random.choice(anchors)
    base = [(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)]
    missing = random.choice([0, 1, 2])
    for idx, (r, c) in enumerate(base):
        if idx == missing:
            continue
        if int(grid[r, c]) == Color.BLACK:
            grid[r, c] = color
            occupancy[r, c] = True


def _generate_once(rng: np.random.Generator) -> np.ndarray:
    BG = Color.BLACK
    grid_size = 30
    target = 1
    grid = np.full((grid_size, grid_size), BG, dtype=int)

    scale = int(rng.integers(1, 5))
    sprite_palette = [c for c in Color.NOT_BLACK if c not in {Color.GREEN, Color.YELLOW}]

    sprite = _make_multicolor_sprite(rng, sprite_palette)
    scaled_sprite = scale_sprite(sprite, scale)

    scaled_h, scaled_w = scaled_sprite.shape
    frame_h = int(scaled_h + rng.integers(2, 5))
    frame_w = int(scaled_w + rng.integers(2, 5))
    frame_h = min(30, frame_h)
    frame_w = min(30, frame_w)
    if frame_h > 26 or frame_w > 26:
        return _generate_once(rng)

    # Reserve enough inner area so sprite can be placed away from markers and still fit.
    frame_h = max(frame_h, scaled_h + 3)
    frame_w = max(frame_w, scaled_w + 3)

    frame_r0 = int(rng.integers(0, grid_size - frame_h + 1))
    frame_c0 = int(rng.integers(0, grid_size - frame_w + 1))
    frame_r1 = frame_r0 + frame_h - 1
    frame_c1 = frame_c0 + frame_w - 1

    frame = np.full((frame_h, frame_w), BG, dtype=int)
    frame[0, 0] = Color.YELLOW
    frame[0, -1] = Color.YELLOW
    frame[-1, 0] = Color.YELLOW
    frame[-1, -1] = Color.YELLOW

    frame = _paste_sprite_sparse_evidence(frame, sprite, rng, scale, BG)

    occupancy = np.zeros((grid_size, grid_size), dtype=bool)
    occupancy[frame_r0 : frame_r1 + 1, frame_c0 : frame_c1 + 1] = True

    # Place the source sprite outside the frame.
    locations = _find_free_locations(sprite, occupancy, BG)
    if not locations:
        return _generate_once(rng)
    sprite_r, sprite_c = random.choice(locations)
    _paint_patch(grid, sprite, sprite_r, sprite_c, BG)
    for r in range(sprite.shape[0]):
        for c in range(sprite.shape[1]):
            if sprite[r, c] != BG:
                occupancy[sprite_r + r, sprite_c + c] = True

    # Place full green 2x2 blocks that define scale.
    for _ in range(scale):
        full_places: list[tuple[int, int]] = []
        for r in range(0, grid_size - 1):
            for c in range(0, grid_size - 1):
                cells = _collect_cells_for_block(r, c)
                if any(occupancy[rr, cc] for rr, cc in cells):
                    continue
                full_places.append((r, c))
        if not full_places:
            return _generate_once(rng)

        top, left = random.choice(full_places)
        _place_full_block_grid(grid, occupancy, top, left)

    # Add distractor green objects.
    # Single pixels.
    singles = int(rng.integers(1, 5))
    for _ in range(singles):
        candidates = [
            (r, c)
            for r in range(grid_size)
            for c in range(grid_size)
            if not occupancy[r, c]
            and not (frame_r0 <= r <= frame_r1 and frame_c0 <= c <= frame_c1)
            and _free_with_margin(occupancy, [(r, c)], margin=1)
        ]
        if not candidates:
            break
        r, c = random.choice(candidates)
        grid[r, c] = Color.GREEN
        occupancy[r, c] = True

    # Incomplete blocks.
    if rng.random() < 0.7:
        _place_partial_block_grid(grid, occupancy, Color.GREEN)

    # Merge frame into final grid.
    grid[frame_r0 : frame_r1 + 1, frame_c0 : frame_c1 + 1] = frame
    return grid


def _validate_variant(grid: np.ndarray) -> np.ndarray:
    out = main(grid)
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        try:
            grid = _generate_once(rng)
            out = main(grid)
            assert isinstance(out, np.ndarray)
            assert out.ndim == 2
            assert out.shape[0] <= 30 and out.shape[1] <= 30
            assert grid.shape[0] == 30 and grid.shape[1] == 30
            assert _count_green_full_blocks(grid) >= 1
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate valid instance")


# Keep module self-contained and deterministic with numpy/pure Python helpers.
