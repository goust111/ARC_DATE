from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

GRID_MIN = 14
GRID_MAX = 30
LAYER_MIN = 2
LAYER_MAX = 5
INNER_MIN = 1
INNER_MAX = 6
SEED_MIN = 2
SEED_MAX = 5
GEN_ATTEMPTS = 500
SELF_CHECK_VARIANTS = 60


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _distance_to_bbox(r: int, c: int, bbox: BBox) -> int:
    r0, c0, r1, c1 = bbox
    dr = max(r0 - r, 0, r - r1)
    dc = max(c0 - c, 0, c - c1)
    return int(max(dr, dc))


def _make_depth_grid(height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    return np.minimum(
        np.minimum(rows, height - 1 - rows),
        np.minimum(cols, width - 1 - cols),
    )


def _make_nested_patch(palette: list[int], inner_h: int, inner_w: int) -> np.ndarray:
    layers = len(palette)
    height = inner_h + 2 * (layers - 1)
    width = inner_w + 2 * (layers - 1)
    depth = _make_depth_grid(height, width)
    return np.asarray(palette, dtype=int)[np.minimum(depth, layers - 1)]


def _extract_rectangle(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(patch == bg):
        return None
    if coords.shape[0] != patch.size:
        return None

    depth = _make_depth_grid(*patch.shape)
    max_depth = int(depth.max())
    palette: list[int] = []
    seen: set[int] = set()
    deepest_repeated = False
    for d in range(max_depth + 1):
        values = np.unique(patch[depth == d])
        if values.size != 1:
            return None
        color = int(values[0])
        if color == bg:
            return None
        if not palette or color != palette[-1]:
            if deepest_repeated or color in seen:
                return None
            seen.add(color)
            palette.append(color)
        else:
            deepest_repeated = True

    return {
        "bbox": bbox,
        "coords": coords,
        "palette": palette,
        "patch": patch.copy(),
    }


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"
    bg = _get_bg(g)

    comps = _component_coords(g != bg)
    assert comps, "expected one nested rectangle and at least one seed"

    rectangle_candidates = []
    seed_coords: list[Coord] = []
    for coords in comps:
        if coords.shape[0] == 1:
            seed_coords.append((int(coords[0, 0]), int(coords[0, 1])))
            continue
        rect = _extract_rectangle(g, coords, bg)
        if rect is not None:
            rectangle_candidates.append(rect)

    assert len(rectangle_candidates) == 1, "expected exactly one nested rectangle component"
    rect = rectangle_candidates[0]
    palette = rect["palette"]
    palette_index = {color: idx for idx, color in enumerate(palette)}
    assert len(palette_index) == len(palette), "layer colors must be distinct"
    assert seed_coords, "expected at least one seed pixel"

    seeds = []
    for r, c in seed_coords:
        color = int(g[r, c])
        assert color in palette_index, "every seed color must come from the rectangle palette"
        seeds.append((r, c, palette_index[color]))

    assert len(comps) == 1 + len(seeds), "all non-rectangle foreground components must be singleton seeds"
    return {
        "bg": bg,
        "rectangle": rect,
        "palette": palette,
        "seeds": seeds,
    }


def _render_output_from_analysis(shape: tuple[int, int], info: dict) -> np.ndarray:
    out = np.full(shape, int(info["bg"]), dtype=int)
    rect = info["rectangle"]
    r0, c0, r1, c1 = rect["bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = rect["patch"]

    palette = info["palette"]
    height, width = shape
    for r, c, k in info["seeds"]:
        for rr in range(max(0, r - k), min(height - 1, r + k) + 1):
            for cc in range(max(0, c - k), min(width - 1, c + k) + 1):
                d = max(abs(rr - r), abs(cc - c))
                if d <= k:
                    out[rr, cc] = palette[k - d]
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Expand each seed into square rings by reading outward colors from the nested-rectangle palette.

    Input structure:
    - The grid has one background color, one filled axis-aligned nested rectangle,
      and several isolated seed pixels elsewhere.
    - The rectangle is made of one-cell-thick concentric rectangular layers whose
      colors are all distinct from one another and from the background.
    - Every seed color matches exactly one layer color, and seed growth regions do
      not overlap the rectangle or one another.

    Rule:
    - Keep the nested rectangle unchanged.
    - Treat its layer order from outermost to innermost as a palette c0, c1, ..., cm.
    - For a seed whose color is ck, fill every cell at Chebyshev distance j from
      that seed with c(k-j), for j = 0..k, clipping rings at the canvas boundary.

    Key insight:
    - The rectangle is not the object to transform; it is a visible lookup table
      whose layer depth indexes a color sequence, and each seed color selects the
      phase at which its outward peeling process starts.
    """
    info = _analyze_grid(np.asarray(input_grid, dtype=int))
    return _render_output_from_analysis(tuple(map(int, input_grid.shape)), info)


def _oracle_from_spec(spec: dict) -> np.ndarray:
    patch = _make_nested_patch(spec["palette"], spec["inner_h"], spec["inner_w"])
    info = {
        "bg": spec["bg"],
        "rectangle": {
            "bbox": spec["rect_bbox"],
            "patch": patch,
        },
        "palette": spec["palette"],
        "seeds": spec["seeds"],
    }
    return _render_output_from_analysis((spec["height"], spec["width"]), info)


def _render_input_from_spec(spec: dict) -> np.ndarray:
    grid = np.full((spec["height"], spec["width"]), spec["bg"], dtype=int)
    patch = _make_nested_patch(spec["palette"], spec["inner_h"], spec["inner_w"])
    r0, c0, r1, c1 = spec["rect_bbox"]
    assert patch.shape == (r1 - r0 + 1, c1 - c0 + 1)
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch
    for r, c, k in spec["seeds"]:
        grid[r, c] = spec["palette"][k]
    return grid


def _sample_radii(layer_count: int, seed_count: int) -> list[int]:
    required = [0, layer_count - 1]
    radii = required[:seed_count]
    while len(radii) < seed_count:
        radii.append(random.randrange(layer_count))
    random.shuffle(radii)
    if all(k == 0 for k in radii):
        radii[0] = layer_count - 1
    return radii


def _place_seeds(height: int, width: int, rect_bbox: BBox, radii: list[int], force_clip: bool) -> list[Coord] | None:
    placed: list[tuple[int, int, int]] = []
    order = sorted(
        range(len(radii)),
        key=lambda idx: (idx == 0 and force_clip, radii[idx]),
        reverse=True,
    )
    result: list[Coord | None] = [None] * len(radii)
    clip_done = False

    for idx in order:
        k = radii[idx]
        require_clip = force_clip and not clip_done and k > 0
        candidates: list[Coord] = []
        for r in range(height):
            for c in range(width):
                if _distance_to_bbox(r, c, rect_bbox) < k + 2:
                    continue
                if any(max(abs(r - pr), abs(c - pc)) < k + pk + 2 for pr, pc, pk in placed):
                    continue
                clipped = min(r, c, height - 1 - r, width - 1 - c) < k
                if require_clip and not clipped:
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        chosen = random.choice(candidates)
        clipped = min(chosen[0], chosen[1], height - 1 - chosen[0], width - 1 - chosen[1]) < k
        clip_done = clip_done or clipped
        placed.append((chosen[0], chosen[1], k))
        result[idx] = chosen

    if force_clip and not clip_done:
        return None
    return [coord for coord in result if coord is not None]


def _is_semantically_valid(spec: dict, inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if len(np.unique(inp)) < 3:
        return False

    info = _analyze_grid(inp)
    if info["palette"] != spec["palette"]:
        return False
    if info["rectangle"]["bbox"] != spec["rect_bbox"]:
        return False
    inferred = sorted(info["seeds"])
    expected = sorted(spec["seeds"])
    if inferred != expected:
        return False
    if not np.array_equal(out, _oracle_from_spec(spec)):
        return False

    if len({k for _r, _c, k in spec["seeds"]}) < 2:
        return False
    if not any(k > 0 for _r, _c, k in spec["seeds"]):
        return False
    if np.array_equal(inp, out):
        return False
    return True


def _sample_instance() -> tuple[np.ndarray, dict]:
    for _ in range(GEN_ATTEMPTS):
        bg = random.randrange(10)
        colors = [c for c in range(10) if c != bg]

        layer_count = random.randint(LAYER_MIN, LAYER_MAX)
        palette = random.sample(colors, layer_count)
        inner_h = random.randint(INNER_MIN, INNER_MAX)
        inner_w = random.randint(INNER_MIN, INNER_MAX)
        patch_h = inner_h + 2 * (layer_count - 1)
        patch_w = inner_w + 2 * (layer_count - 1)

        seed_count = random.randint(SEED_MIN, SEED_MAX)
        radii = _sample_radii(layer_count, seed_count)
        max_k = max(radii)

        min_h = max(GRID_MIN, patch_h + 4, 2 * max_k + 5)
        min_w = max(GRID_MIN, patch_w + 4, 2 * max_k + 5)
        if min_h > GRID_MAX or min_w > GRID_MAX:
            continue

        height = random.randint(min_h, GRID_MAX)
        width = random.randint(min_w, GRID_MAX)
        r0 = random.randint(0, height - patch_h)
        c0 = random.randint(0, width - patch_w)
        rect_bbox = (r0, c0, r0 + patch_h - 1, c0 + patch_w - 1)

        force_clip = random.random() < 0.55 and max_k > 0
        seed_positions = _place_seeds(height, width, rect_bbox, radii, force_clip)
        if seed_positions is None:
            continue

        seeds = [(r, c, k) for (r, c), k in zip(seed_positions, radii)]
        spec = {
            "bg": bg,
            "palette": palette,
            "inner_h": inner_h,
            "inner_w": inner_w,
            "height": height,
            "width": width,
            "rect_bbox": rect_bbox,
            "seeds": seeds,
        }
        inp = _render_input_from_spec(spec)
        out = _oracle_from_spec(spec)
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(actual, out):
            continue
        if not _is_semantically_valid(spec, inp, out):
            continue
        return inp, spec
    raise RuntimeError("failed to generate a valid input")


def generate_input() -> np.ndarray:
    return _sample_instance()[0]
