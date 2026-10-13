from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300

# All templates are 4-connected, use 3-5 cells, and include (0, 0).
SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1)),
)

QUAD_TO_SIGNS = {
    "TL": (1, 1),
    "TR": (1, -1),
    "BL": (-1, 1),
    "BR": (-1, -1),
}


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
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


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _point_in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _relative_quadrant(r: int, c: int, inner_h: int, inner_w: int) -> str | None:
    top = 2 * r + 1 < inner_h
    bottom = 2 * r + 1 > inner_h
    left = 2 * c + 1 < inner_w
    right = 2 * c + 1 > inner_w
    if not (top or bottom) or not (left or right):
        return None
    if top and left:
        return "TL"
    if top and right:
        return "TR"
    if bottom and left:
        return "BL"
    if bottom and right:
        return "BR"
    return None


def _shape_size(shape: Iterable[tuple[int, int]]) -> tuple[int, int]:
    shape_list = list(shape)
    return 1 + max(r for r, _ in shape_list), 1 + max(c for _, c in shape_list)


def _count_copies(
    inner_h: int,
    inner_w: int,
    seed_rel: tuple[int, int],
    shape: tuple[tuple[int, int], ...],
    signs: tuple[int, int],
) -> int:
    seed_r, seed_c = seed_rel
    step_h, step_w = _shape_size(shape)
    dr, dc = signs
    copies = 0
    while True:
        ok = True
        for sr, sc in shape:
            rr = seed_r + dr * (copies * step_h + sr)
            cc = seed_c + dc * (copies * step_w + sc)
            if not (0 <= rr < inner_h and 0 <= cc < inner_w):
                ok = False
                break
        if not ok:
            return copies
        copies += 1


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for pts in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border_count = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
        if len(pts) != border_count:
            continue
        window = grid[r0 : r1 + 1, c0 : c1 + 1]
        border_mask = np.zeros_like(window, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        if not np.all(window[border_mask] == FRAME):
            continue
        if np.any(window[~border_mask] == FRAME):
            continue
        frames.append(
            {
                "outer": (r0, c0, r1, c1),
                "inner": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
            }
        )
    frames.sort(key=lambda item: tuple(item["outer"]))  # deterministic
    return frames


def _extract_templates(
    grid: np.ndarray,
    frames: list[dict[str, object]],
    seed_colors: list[int],
) -> dict[int, tuple[tuple[int, int], ...]]:
    inner_boxes = [frame["inner"] for frame in frames]
    templates: dict[int, tuple[tuple[int, int], ...]] = {}
    for color in seed_colors:
        comps = _components(grid == color)
        outside_shapes = []
        for pts in comps:
            inside_some_frame = all(any(_point_in_bbox(pt, box) for box in inner_boxes) for pt in pts)
            if inside_some_frame:
                continue
            outside_shapes.append(pts)
        assert len(outside_shapes) == 1, "expected one external template per seed color"
        tpl = outside_shapes[0]
        r0, c0, _, _ = _bbox(tpl)
        norm = tuple(sorted((r - r0, c - c0) for r, c in tpl))
        assert 3 <= len(norm) <= 5, "templates must use 3-5 pixels"
        assert (0, 0) in norm, "template anchor pixel missing"
        templates[color] = norm
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the color-matched external template inside each frame toward the seed's opposite corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    assert frames, "expected at least one grey rectangular frame"

    seed_colors: list[int] = []
    for frame in frames:
        ir0, ic0, ir1, ic1 = frame["inner"]
        inner = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
        nz = np.argwhere((inner != BG) & (inner != FRAME))
        assert len(nz) == 1, "each frame interior must contain exactly one colored seed"
        rr, cc = map(int, nz[0])
        color = int(inner[rr, cc])
        assert np.all((inner == BG) | (inner == color)), "frame interior must be blank except for the seed"
        frame["seed"] = (ir0 + rr, ic0 + cc)
        frame["color"] = color
        seed_colors.append(color)

    assert len(set(seed_colors)) == len(seed_colors), "seed colors must be unique"
    templates = _extract_templates(g, frames, seed_colors)

    out = g.copy()
    for frame in frames:
        ir0, ic0, ir1, ic1 = frame["inner"]
        seed_r, seed_c = frame["seed"]
        color = int(frame["color"])
        inner_h = ir1 - ir0 + 1
        inner_w = ic1 - ic0 + 1
        rel_r = seed_r - ir0
        rel_c = seed_c - ic0
        quadrant = _relative_quadrant(rel_r, rel_c, inner_h, inner_w)
        assert quadrant is not None, "seed cannot lie on the frame center lines"
        signs = QUAD_TO_SIGNS[quadrant]
        shape = templates[color]
        step_h, step_w = _shape_size(shape)

        copy_idx = 0
        while True:
            pts: list[tuple[int, int]] = []
            for sr, sc in shape:
                rr = seed_r + signs[0] * (copy_idx * step_h + sr)
                cc = seed_c + signs[1] * (copy_idx * step_w + sc)
                if not (ir0 <= rr <= ir1 and ic0 <= cc <= ic1):
                    pts = []
                    break
                pts.append((rr, cc))
            if not pts:
                break
            for rr, cc in pts:
                out[rr, cc] = color
            copy_idx += 1

    return out


def _place_rect(
    rng: random.Random,
    occupied: np.ndarray,
    rect_h: int,
    rect_w: int,
    pad: int,
    tries: int = 200,
) -> tuple[int, int] | None:
    h, w = occupied.shape
    if rect_h > h or rect_w > w:
        return None
    for _ in range(tries):
        r0 = rng.randint(0, h - rect_h)
        c0 = rng.randint(0, w - rect_w)
        rs = max(0, r0 - pad)
        re = min(h, r0 + rect_h + pad)
        cs = max(0, c0 - pad)
        ce = min(w, c0 + rect_w + pad)
        if occupied[rs:re, cs:ce].any():
            continue
        occupied[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
        return r0, c0
    return None


def _build_spec(
    rng: random.Random,
    color: int,
    quadrant: str,
    shape: tuple[tuple[int, int], ...],
) -> dict[str, object] | None:
    shape_h, shape_w = _shape_size(shape)
    inner_h = rng.randint(max(5, 2 * shape_h), max(7, 2 * shape_h + 4))
    inner_w = rng.randint(max(5, 2 * shape_w), max(7, 2 * shape_w + 4))

    candidates = []
    for rr in range(inner_h):
        for cc in range(inner_w):
            if _relative_quadrant(rr, cc, inner_h, inner_w) != quadrant:
                continue
            copies = _count_copies(inner_h, inner_w, (rr, cc), shape, QUAD_TO_SIGNS[quadrant])
            if copies >= 2:
                candidates.append((rr, cc, copies))
    if not candidates:
        return None

    seed_r, seed_c, copies = rng.choice(candidates)
    return {
        "color": color,
        "quadrant": quadrant,
        "shape": shape,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "frame_h": inner_h + 2,
        "frame_w": inner_w + 2,
        "seed_rel": (seed_r, seed_c),
        "copies": copies,
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.choice((2, 3, 4))
    colors = rng.sample(COLOR_POOL, n_frames)
    quadrants = rng.sample(tuple(QUAD_TO_SIGNS), n_frames)
    shapes = rng.sample(SHAPES, n_frames)

    specs: list[dict[str, object]] = []
    for color, quadrant, shape in zip(colors, quadrants, shapes):
        spec = _build_spec(rng, color, quadrant, shape)
        if spec is None:
            return None
        specs.append(spec)

    tallest = max(int(spec["frame_h"]) for spec in specs)
    widest = max(int(spec["frame_w"]) for spec in specs)
    grid_h_lo = max(20, tallest + 8, 8 + 4 * n_frames)
    grid_w_lo = max(20, widest + 8, 8 + 4 * n_frames)
    grid_h = rng.randint(grid_h_lo, min(30, grid_h_lo + 6))
    grid_w = rng.randint(grid_w_lo, min(30, grid_w_lo + 6))

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    for spec in sorted(specs, key=lambda item: int(item["frame_h"]) * int(item["frame_w"]), reverse=True):
        pos = _place_rect(rng, occupied, int(spec["frame_h"]), int(spec["frame_w"]), pad=2)
        if pos is None:
            return None
        spec["frame_pos"] = pos

    for spec in sorted(specs, key=lambda item: _shape_size(item["shape"])[0] * _shape_size(item["shape"])[1], reverse=True):
        shape_h, shape_w = _shape_size(spec["shape"])
        pos = _place_rect(rng, occupied, shape_h, shape_w, pad=1)
        if pos is None:
            return None
        spec["template_pos"] = pos

    for spec in specs:
        fr0, fc0 = spec["frame_pos"]
        frame_h = int(spec["frame_h"])
        frame_w = int(spec["frame_w"])
        grid[fr0, fc0 : fc0 + frame_w] = FRAME
        grid[fr0 + frame_h - 1, fc0 : fc0 + frame_w] = FRAME
        grid[fr0 : fr0 + frame_h, fc0] = FRAME
        grid[fr0 : fr0 + frame_h, fc0 + frame_w - 1] = FRAME

        seed_rel_r, seed_rel_c = spec["seed_rel"]
        grid[fr0 + 1 + seed_rel_r, fc0 + 1 + seed_rel_c] = int(spec["color"])

        tr0, tc0 = spec["template_pos"]
        for dr, dc in spec["shape"]:
            grid[tr0 + dr, tc0 + dc] = int(spec["color"])

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if any(np.count_nonzero(out == int(spec["color"])) <= np.count_nonzero(grid == int(spec["color"])) for spec in specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
