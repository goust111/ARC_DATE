from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
YELLOW = 4
ORANGE = 7
FRAME_COLOR_POOL = (1, 2, 3, 5, 6, 8, 9)
MIN_LAYERS = 2
MAX_LAYERS = 4
MAX_GEN_ATTEMPTS = 500


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
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _frame_components(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BLACK):
        for comp in _components(grid == color):
            if not comp:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            expected = 2 * (h + w) - 4
            if len(comp) != expected:
                continue
            ok = True
            comp_set = set(comp)
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    on_border = rr in (r0, r1) or cc in (c0, c1)
                    if on_border != ((rr, cc) in comp_set):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                frames.append({"color": color, "bbox": (r0, c0, r1, c1)})
    frames.sort(key=lambda item: (item["bbox"][2] - item["bbox"][0] + 1) * (item["bbox"][3] - item["bbox"][1] + 1))
    return frames


def _extract_structure(grid: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray]:
    frames = _frame_components(grid)
    assert len(frames) >= 2, "expected at least two nested frames"
    for idx in range(1, len(frames)):
        ir0, ic0, ir1, ic1 = frames[idx - 1]["bbox"]
        or0, oc0, or1, oc1 = frames[idx]["bbox"]
        assert or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1, "frames must be strictly nested"

    ir0, ic0, ir1, ic1 = frames[0]["bbox"]
    inner_area = grid[ir0 + 1 : ir1, ic0 + 1 : ic1]
    points = np.argwhere(inner_area != BLACK)
    assert len(points) > 0, "innermost frame must contain a seed pattern"
    pr0, pc0 = points.min(axis=0)
    pr1, pc1 = points.max(axis=0)
    seed = inner_area[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
    assert (seed == YELLOW).any(), "seed must contain yellow"
    assert (seed == ORANGE).any(), "seed must contain orange"
    return frames, seed


def _tile_ring(
    out: np.ndarray,
    seed: np.ndarray,
    outer_bbox: tuple[int, int, int, int],
    inner_bbox: tuple[int, int, int, int],
    frame_color: int,
) -> None:
    r0, c0, r1, c1 = outer_bbox
    ir0, ic0, ir1, ic1 = inner_bbox
    sh, sw = seed.shape
    for rr in range(r0 + 1, r1):
        for cc in range(c0 + 1, c1):
            if ir0 <= rr <= ir1 and ic0 <= cc <= ic1:
                continue
            value = int(seed[(rr - (r0 + 1)) % sh, (cc - (c0 + 1)) % sw])
            if value == YELLOW:
                out[rr, cc] = frame_color
            else:
                out[rr, cc] = value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the innermost template outward across nested container rings with one color substitution.

    Input structure:
    - The background is black.
    - Several 1-pixel-thick rectangular frames of different colors are strictly nested, with black gaps between neighboring frames.
    - The innermost frame contains a small yellow/orange seed pattern; any unused cells inside the innermost frame are black.
    - Every outer frame contains only black in the ring between its border and the next inner frame.

    Rule:
    - Detect all rectangular frame containers and order them from innermost to outermost.
    - Crop the non-black seed pattern from inside the innermost frame.
    - For each outer frame, tile that seed pattern across the whole interior rectangle starting at the interior top-left corner, but only keep the cells that lie in the ring between this frame and the next inner frame.
    - During tiling, replace yellow seed pixels with the current frame color, keep orange unchanged, and leave any black seed cells black.

    Key insight:
    - The nested frames encode a containment hierarchy: the innermost content is the reusable template, and each larger ring reveals the same template again with only the yellow pixels recolored by that ring's border.
    """
    g = np.asarray(grid, dtype=int)
    frames, seed = _extract_structure(g)
    out = g.copy()
    for idx in range(1, len(frames)):
        _tile_ring(out, seed, frames[idx]["bbox"], frames[idx - 1]["bbox"], int(frames[idx]["color"]))
    return out


def _sample_seed(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        seed = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                roll = rng.random()
                if roll < 0.42:
                    seed[r, c] = YELLOW
                elif roll < 0.84:
                    seed[r, c] = ORANGE
        if not (seed == YELLOW).any() or not (seed == ORANGE).any():
            continue
        points = np.argwhere(seed != BLACK)
        if len(points) < 3:
            continue
        pr0, pc0 = points.min(axis=0)
        pr1, pc1 = points.max(axis=0)
        if (pr0, pc0, pr1, pc1) != (0, 0, h - 1, w - 1):
            continue
        probe = np.zeros((h + 2, w + 2), dtype=int)
        probe[1:-1, 1:-1] = seed
        if _frame_components(probe):
            continue
        return seed
    raise RuntimeError("failed to sample a valid seed pattern")


def _render_case(
    grid_shape: tuple[int, int],
    frames: list[dict[str, object]],
    seed: np.ndarray,
    seed_origin: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros(grid_shape, dtype=int)
    for frame in reversed(frames):
        _paint_frame(inp, frame["bbox"], int(frame["color"]))

    sr0, sc0 = seed_origin
    sh, sw = seed.shape
    inp[sr0 : sr0 + sh, sc0 : sc0 + sw] = seed

    out = inp.copy()
    for idx in range(1, len(frames)):
        _tile_ring(out, seed, frames[idx]["bbox"], frames[idx - 1]["bbox"], int(frames[idx]["color"]))
    return inp, out


def _build_spec(rng: random.Random) -> tuple[tuple[int, int], list[dict[str, object]], np.ndarray, tuple[int, int]] | None:
    layer_count = rng.randint(MIN_LAYERS, MAX_LAYERS)
    colors = rng.sample(FRAME_COLOR_POOL, layer_count)
    seed = _sample_seed(rng)

    if rng.random() < 0.5:
        pad_t = pad_b = pad_l = pad_r = 0
    else:
        pad_t, pad_b, pad_l, pad_r = [rng.randint(0, 1) for _ in range(4)]
        if pad_t + pad_b + pad_l + pad_r == 0:
            pad_t = 1

    inner_h = seed.shape[0] + pad_t + pad_b + 2
    inner_w = seed.shape[1] + pad_l + pad_r + 2

    expansions: list[tuple[int, int, int, int]] = []
    cur_h, cur_w = inner_h, inner_w
    for _ in range(layer_count - 1):
        gap_t, gap_b, gap_l, gap_r = [rng.randint(1, 2) for _ in range(4)]
        expansions.append((gap_t, gap_b, gap_l, gap_r))
        cur_h += gap_t + gap_b + 2
        cur_w += gap_l + gap_r + 2

    if cur_h > 26 or cur_w > 26:
        return None

    margin_t = rng.randint(1, min(2, 30 - cur_h - 1))
    margin_b = rng.randint(1, min(2, 30 - cur_h - margin_t))
    margin_l = rng.randint(1, min(2, 30 - cur_w - 1))
    margin_r = rng.randint(1, min(2, 30 - cur_w - margin_l))
    grid_shape = (cur_h + margin_t + margin_b, cur_w + margin_l + margin_r)
    if max(grid_shape) > 30:
        return None

    bboxes: list[tuple[int, int, int, int]] = [None] * layer_count  # type: ignore[assignment]
    bboxes[-1] = (margin_t, margin_l, margin_t + cur_h - 1, margin_l + cur_w - 1)
    for idx in range(layer_count - 2, -1, -1):
        gap_t, gap_b, gap_l, gap_r = expansions[idx]
        pr0, pc0, pr1, pc1 = bboxes[idx + 1]
        bboxes[idx] = (pr0 + 1 + gap_t, pc0 + 1 + gap_l, pr1 - 1 - gap_b, pc1 - 1 - gap_r)

    frames = [{"color": int(color), "bbox": bbox} for color, bbox in zip(colors, bboxes)]
    inner_bbox = bboxes[0]
    seed_origin = (inner_bbox[0] + 1 + pad_t, inner_bbox[1] + 1 + pad_l)
    return grid_shape, frames, seed, seed_origin


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        rng = random.Random()
        spec = _build_spec(rng)
        if spec is None:
            continue
        grid_shape, frames, seed, seed_origin = spec
        inp, expected = _render_case(grid_shape, frames, seed, seed_origin)
        found = {(int(frame["color"]), tuple(frame["bbox"])) for frame in _frame_components(inp)}
        want = {(int(frame["color"]), tuple(frame["bbox"])) for frame in frames}
        if found != want:
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid nested-container puzzle")
