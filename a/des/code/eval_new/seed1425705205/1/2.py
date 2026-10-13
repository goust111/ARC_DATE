from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500

_AREA_TO_DIMS: dict[int, list[tuple[int, int]]] = {}
for _h in range(1, 5):
    for _w in range(1, 7):
        _area = _h * _w
        if _area <= 15:
            _AREA_TO_DIMS.setdefault(_area, []).append((_h, _w))
AREA_CHOICES = tuple(sorted(area for area in _AREA_TO_DIMS if area >= 2))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    pts = {tuple(map(int, rc)) for rc in coords.tolist()}
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in {r0, r1} or c in {c0, c1}
    }
    return pts == expected


def _extract_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    comps = _components_mask(grid == GREY)
    frames = [_bbox(comp) for comp in comps if _is_rect_frame(comp)]
    assert len(frames) >= 2, "expected at least two nested frames"
    frames.sort(key=lambda box: (-_bbox_area(box), box[0], box[1]))
    for outer, inner in zip(frames, frames[1:], strict=False):
        assert outer[0] < inner[0] < inner[2] < outer[2], "frames must be strictly nested by rows"
        assert outer[1] < inner[1] < inner[3] < outer[3], "frames must be strictly nested by cols"
    return frames


def _extract_fragments(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    comps = _components_mask((grid != BLACK) & (grid != GREY))
    frags: list[dict[str, int | tuple[int, int, int, int]]] = []
    for comp in comps:
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        colors = {int(grid[r, c]) for r, c in comp.tolist()}
        assert len(colors) == 1, "each fragment must be monochrome"
        color = int(next(iter(colors)))
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(sub == color), "each fragment must be a solid rectangle"
        frags.append({"bbox": box, "color": color, "area": int(comp.shape[0])})
    areas = [int(frag["area"]) for frag in frags]
    assert len(set(areas)) == len(areas), "fragment areas must be distinct"
    return frags


def _paint_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int = GREY) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _shift_box(box: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return int(r0 + dr), int(c0 + dc), int(r1 + dr), int(c1 + dc)


def _sample_frame_boxes(rng: random.Random, levels: int) -> list[tuple[int, int, int, int]]:
    gap_hi = 2 if levels >= 4 else 3
    core_h = rng.randint(1, 4)
    core_w = rng.randint(1, 4)
    boxes = [(0, 0, core_h + 1, core_w + 1)]  # innermost bbox
    for _ in range(levels - 1):
        gap_r = rng.randint(1, gap_hi)
        gap_c = rng.randint(1, gap_hi)
        r0, c0, r1, c1 = boxes[0]
        boxes.insert(0, (r0 - gap_r - 1, c0 - gap_c - 1, r1 + gap_r + 1, c1 + gap_c + 1))
    min_r = min(box[0] for box in boxes)
    min_c = min(box[1] for box in boxes)
    return [_shift_box(box, -min_r, -min_c) for box in boxes]


def _sample_fragments(
    rng: random.Random, levels: int
) -> list[dict[str, int | tuple[int, int]]]:
    areas = rng.sample(list(AREA_CHOICES), levels)
    colors = rng.sample(list(COLOR_POOL), levels)
    frags = []
    for area, color in zip(areas, colors, strict=False):
        h, w = rng.choice(_AREA_TO_DIMS[int(area)])
        frags.append({"area": int(area), "color": int(color), "dims": (int(h), int(w))})
    rng.shuffle(frags)
    return frags


def _place_rectangles(
    rng: random.Random, zone_h: int, zone_w: int, dims: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    occ = np.zeros((zone_h, zone_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(dims)
    order = list(range(len(dims)))
    rng.shuffle(order)

    for idx in order:
        ph, pw = dims[idx]
        candidates: list[tuple[int, int]] = []
        for r0 in range(zone_h - ph + 1):
            for c0 in range(zone_w - pw + 1):
                rr0 = max(0, r0 - 1)
                rr1 = min(zone_h, r0 + ph + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(zone_w, c0 + pw + 1)
                if occ[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        occ[r0 : r0 + ph, c0 : c0 + pw] = True
        placements[idx] = (r0, c0)

    if any(pos is None for pos in placements):
        return None
    return [tuple(map(int, pos)) for pos in placements if pos is not None]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill nested rectangular rings using side fragments ranked by fragment area.

    Input structure:
    - Background is black.
    - One grey structure consists of 2-4 strictly nested 1-pixel-thick rectangular frames.
    - Elsewhere on the grid are the same number of detached solid-color rectangular fragments, each with a distinct area.

    Rule:
    - Order the grey frames from the outermost rectangle to the innermost rectangle.
    - Order the detached fragments by area from largest to smallest, then remove those fragments from the canvas.
    - Fill the black ring between consecutive frame levels with the correspondingly ranked fragment color.
    - Fill the black interior of the innermost frame with the smallest fragment color, while leaving every grey frame pixel unchanged.

    Key insight: Fragment size encodes frame depth: larger fragments map to outer rings, and the smallest fragment colors the innermost cavity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _extract_frames(g)
    fragments = _extract_fragments(g)
    assert len(frames) == len(fragments), "frame count and fragment count must match"

    ranked = sorted(fragments, key=lambda frag: -int(frag["area"]))
    out = np.where(g == GREY, GREY, BLACK)

    for idx, frag in enumerate(ranked):
        color = int(frag["color"])
        r0, c0, r1, c1 = frames[idx]
        out[r0 + 1 : r1, c0 + 1 : c1] = color
        if idx + 1 < len(frames):
            ir0, ic0, ir1, ic1 = frames[idx + 1]
            out[ir0 : ir1 + 1, ic0 : ic1 + 1] = np.where(
                g[ir0 : ir1 + 1, ic0 : ic1 + 1] == GREY,
                GREY,
                out[ir0 : ir1 + 1, ic0 : ic1 + 1],
            )
            out[ir0 + 1 : ir1, ic0 + 1 : ic1] = BLACK

    for box in frames:
        _paint_frame(out, box, GREY)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    levels = rng.randint(2, 4)
    frame_boxes = _sample_frame_boxes(rng, levels)
    outer = frame_boxes[0]
    frame_h = outer[2] + 1
    frame_w = outer[3] + 1

    fragments = _sample_fragments(rng, levels)
    dims = [tuple(map(int, frag["dims"])) for frag in fragments]
    piece_heights = [dims_i[0] for dims_i in dims]
    piece_widths = [dims_i[1] for dims_i in dims]
    side = rng.choice(("top", "bottom", "left", "right"))

    top_margin = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    gap = rng.randint(2, 4)

    if side in {"left", "right"}:
        usable_h = max(
            frame_h + rng.randint(0, 4),
            sum(piece_heights) + len(fragments) + rng.randint(0, 4),
        )
        zone_h = usable_h
        zone_w = max(piece_widths) + rng.randint(2, 5)
        h = top_margin + usable_h + bottom_margin
        if side == "right":
            w = left_margin + frame_w + gap + zone_w + right_margin
            frame_r0 = top_margin + rng.randint(0, usable_h - frame_h)
            frame_c0 = left_margin
            zone_r0 = top_margin
            zone_c0 = left_margin + frame_w + gap
        else:
            w = left_margin + zone_w + gap + frame_w + right_margin
            frame_r0 = top_margin + rng.randint(0, usable_h - frame_h)
            frame_c0 = left_margin + zone_w + gap
            zone_r0 = top_margin
            zone_c0 = left_margin
    else:
        usable_w = max(
            frame_w + rng.randint(0, 4),
            sum(piece_widths) + len(fragments) + rng.randint(0, 4),
        )
        zone_h = max(piece_heights) + rng.randint(2, 5)
        zone_w = usable_w
        w = left_margin + usable_w + right_margin
        if side == "bottom":
            h = top_margin + frame_h + gap + zone_h + bottom_margin
            frame_r0 = top_margin
            frame_c0 = left_margin + rng.randint(0, usable_w - frame_w)
            zone_r0 = top_margin + frame_h + gap
            zone_c0 = left_margin
        else:
            h = top_margin + zone_h + gap + frame_h + bottom_margin
            frame_r0 = top_margin + zone_h + gap
            frame_c0 = left_margin + rng.randint(0, usable_w - frame_w)
            zone_r0 = top_margin
            zone_c0 = left_margin

    if h > 30 or w > 30:
        return None

    placements = _place_rectangles(rng, zone_h, zone_w, dims)
    if placements is None or len(placements) != len(fragments):
        return None

    grid = np.zeros((h, w), dtype=int)
    for box in frame_boxes:
        _paint_frame(grid, _shift_box(box, frame_r0, frame_c0), GREY)

    for frag, (pr, pc) in zip(fragments, placements, strict=False):
        ph, pw = map(int, frag["dims"])
        color = int(frag["color"])
        rr0 = zone_r0 + pr
        cc0 = zone_c0 + pc
        grid[rr0 : rr0 + ph, cc0 : cc0 + pw] = color

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
