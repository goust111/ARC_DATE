from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


BG = 0
ARC_MAX = 30
COLOR_POOL = tuple(range(1, 10))
PAIR_COUNT_CHOICES = (2, 3, 4)
RECT_DIMS_BY_AREA = {
    9: [(3, 3)],
    12: [(3, 4), (4, 3)],
    15: [(3, 5), (5, 3)],
    16: [(4, 4)],
    18: [(3, 6), (6, 3)],
    20: [(4, 5), (5, 4)],
    24: [(4, 6), (6, 4)],
}
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    order = 0
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "coords": np.argwhere(mask),
                    "area": int(mask.sum()),
                    "order": order,
                }
            )
            order += 1
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["order"]))
    for i, comp in enumerate(comps):
        comp["order"] = i
    return comps


@lru_cache(maxsize=None)
def _factor_pairs(area: int) -> tuple[tuple[int, int], ...]:
    pairs = []
    for h in range(1, int(area**0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        pairs.append((h, w))
        if h != w:
            pairs.append((w, h))
    pairs.sort()
    return tuple(pairs)


def _placement_signature(shape: tuple[int, int], coords: np.ndarray, offset: tuple[int, int]) -> bytes:
    h, w = shape
    r0, c0 = offset
    canvas = np.zeros((h, w), dtype=bool)
    canvas[r0 + coords[:, 0], c0 + coords[:, 1]] = True
    return canvas.tobytes()


def _pair_options(comp_a: dict, comp_b: dict) -> list[dict]:
    area = int(comp_a["area"] + comp_b["area"])
    ha, wa = comp_a["mask"].shape
    hb, wb = comp_b["mask"].shape
    coords_a = comp_a["coords"]
    coords_b = comp_b["coords"]

    options = []
    seen = set()
    for out_h, out_w in _factor_pairs(area):
        if out_h < max(ha, hb) or out_w < max(wa, wb):
            continue
        for ar in range(out_h - ha + 1):
            for ac in range(out_w - wa + 1):
                placed_a = {(int(ar + r), int(ac + c)) for r, c in coords_a}
                for br in range(out_h - hb + 1):
                    for bc in range(out_w - wb + 1):
                        placed_b = {(int(br + r), int(bc + c)) for r, c in coords_b}
                        if placed_a & placed_b:
                            continue
                        sig = (
                            out_h,
                            out_w,
                            _placement_signature((out_h, out_w), coords_a, (ar, ac)),
                            _placement_signature((out_h, out_w), coords_b, (br, bc)),
                        )
                        if sig in seen:
                            continue
                        seen.add(sig)

                        patch = np.full((out_h, out_w), BG, dtype=int)
                        for rr, cc in placed_a:
                            patch[rr, cc] = int(comp_a["color"])
                        for rr, cc in placed_b:
                            patch[rr, cc] = int(comp_b["color"])
                        options.append(
                            {
                                "i": int(comp_a["order"]),
                                "j": int(comp_b["order"]),
                                "patch": patch,
                                "shape_key": (
                                    area,
                                    out_h,
                                    out_w,
                                    _placement_signature((out_h, out_w), coords_a, (ar, ac)),
                                ),
                            }
                        )
    options.sort(key=lambda item: item["shape_key"])
    return options


def _stack_key(item: dict) -> tuple:
    patch = item["patch"]
    return (
        int(patch.size),
        int(patch.shape[0]),
        int(patch.shape[1]),
        item["shape_key"][3],
        int(item["i"]),
        int(item["j"]),
    )


def _collect_solutions(comps: list[dict], pair_map: dict[tuple[int, int], list[dict]], limit: int = 24) -> list[list[dict]]:
    all_indices = tuple(range(len(comps)))
    solutions: list[list[dict]] = []

    def rec(remaining: tuple[int, ...], chosen: list[dict]) -> None:
        if len(solutions) >= limit:
            return
        if not remaining:
            solutions.append(chosen.copy())
            return
        i = remaining[0]
        rest = set(remaining[1:])
        for j in remaining[1:]:
            options = pair_map.get((i, j), [])
            if not options:
                continue
            next_remaining = tuple(k for k in remaining if k not in {i, j})
            for option in options:
                chosen.append(option)
                rec(next_remaining, chosen)
                chosen.pop()
                if len(solutions) >= limit:
                    return

    rec(all_indices, [])
    return solutions


def _build_output(rectangles: list[np.ndarray]) -> np.ndarray:
    ordered = sorted(rectangles, key=lambda patch: (int(patch.size), int(patch.shape[0]), int(patch.shape[1]), patch.tobytes()))
    out_h = sum(int(patch.shape[0]) for patch in ordered) + len(ordered) - 1
    out_w = max(int(patch.shape[1]) for patch in ordered)
    out = np.full((out_h, out_w), BG, dtype=int)
    row = 0
    for patch in ordered:
        h, w = patch.shape
        out[row : row + h, :w] = patch
        row += h + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble complementary colored fragments into rectangles, then stack the completed rectangles by area.

    Input structure:
    - The background is black.
    - Several differently colored 4-connected components are scattered apart from one another.
    - The components come in pairs: after translation, each pair exactly tiles one solid axis-aligned rectangle.

    Rule:
    - Extract every monochrome 4-connected component and normalize it to its own bounding box.
    - Pair components whose translated shapes can occupy a common rectangle with no overlap and no gaps.
    - Reconstruct each filled rectangle, preserving the original fragment color at every covered cell.
    - Sort the completed rectangles by increasing area and stack them top-to-bottom, left-aligned, with one black separator row between consecutive rectangles.

    Key insight:
    The fragments are matched by geometric complementarity rather than by color or by current position: two distant pieces belong together only if rigid translation lets them tile one rectangle exactly, after which the finished rectangles are simply repacked by area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    comps = _find_components(g)
    assert comps and len(comps) % 2 == 0

    pair_map: dict[tuple[int, int], list[dict]] = {}
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            options = _pair_options(comps[i], comps[j])
            if options:
                pair_map[(i, j)] = options

    solutions = _collect_solutions(comps, pair_map)
    assert solutions, "expected at least one complete pairing"

    def solution_key(sol: list[dict]) -> tuple:
        return tuple(sorted((_stack_key(item) for item in sol)))

    best = min(solutions, key=solution_key)
    return _build_output([item["patch"] for item in best])


def _split_mask(h: int, w: int) -> np.ndarray:
    vertical = random.random() < 0.5
    if vertical:
        cuts = [random.randint(1, w - 1) for _ in range(h)]
        if w >= 4 and len(set(cuts)) == 1:
            idx = random.randrange(h)
            alternatives = [c for c in range(1, w) if c != cuts[idx]]
            cuts[idx] = random.choice(alternatives)
        mask = np.zeros((h, w), dtype=bool)
        for r, cut in enumerate(cuts):
            mask[r, :cut] = True
    else:
        cuts = [random.randint(1, h - 1) for _ in range(w)]
        if h >= 4 and len(set(cuts)) == 1:
            idx = random.randrange(w)
            alternatives = [c for c in range(1, h) if c != cuts[idx]]
            cuts[idx] = random.choice(alternatives)
        mask = np.zeros((h, w), dtype=bool)
        for c, cut in enumerate(cuts):
            mask[:cut, c] = True

    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return mask.copy()


def _crop_patch(mask: np.ndarray, color: int) -> np.ndarray:
    rows, cols = np.where(mask)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    patch[rows - r0, cols - c0] = int(color)
    return patch


def _intended_patch(mask: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    patch = np.full(mask.shape, int(color_b), dtype=int)
    patch[mask] = int(color_a)
    return patch


def _component_from_patch(patch: np.ndarray, order: int) -> dict:
    mask = patch != BG
    colors = sorted(set(map(int, patch[mask].ravel())))
    assert len(colors) == 1
    return {
        "color": int(colors[0]),
        "bbox": (0, 0, patch.shape[0] - 1, patch.shape[1] - 1),
        "mask": mask,
        "coords": np.argwhere(mask),
        "area": int(mask.sum()),
        "order": int(order),
    }


def _dilate8(mask: np.ndarray) -> np.ndarray:
    dilated = mask.copy()
    dilated[1:, :] |= mask[:-1, :]
    dilated[:-1, :] |= mask[1:, :]
    dilated[:, 1:] |= mask[:, :-1]
    dilated[:, :-1] |= mask[:, 1:]
    dilated[1:, 1:] |= mask[:-1, :-1]
    dilated[1:, :-1] |= mask[:-1, 1:]
    dilated[:-1, 1:] |= mask[1:, :-1]
    dilated[:-1, :-1] |= mask[1:, 1:]
    return dilated


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    fg = sum(int(np.count_nonzero(patch)) for patch in patches)
    min_side = max(max(max(patch.shape) for patch in patches), int(np.ceil(np.sqrt(fg * 3.0))))
    min_side = min(min_side, ARC_MAX)
    if min_side > ARC_MAX:
        return None

    for _ in range(40):
        h = random.randint(min_side, ARC_MAX)
        w = random.randint(min_side, ARC_MAX)
        grid = np.full((h, w), BG, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        order = list(range(len(patches)))
        order.sort(key=lambda idx: int(np.count_nonzero(patches[idx] != BG)), reverse=True)

        ok = True
        for idx in order:
            patch = patches[idx]
            mask = patch != BG
            ph, pw = patch.shape
            forbidden = _dilate8(occupied)
            candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
            random.shuffle(candidates)
            placed = False
            for r0, c0 in candidates:
                if np.any(forbidden[r0 : r0 + ph, c0 : c0 + pw] & mask):
                    continue
                grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
                occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    return None


def _sample_rectangles() -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    areas = random.sample(sorted(RECT_DIMS_BY_AREA), pair_count)
    colors = random.sample(COLOR_POOL, 2 * pair_count)

    full_rectangles = []
    component_patches = []
    total_output_h = pair_count - 1
    for i, area in enumerate(areas):
        h, w = random.choice(RECT_DIMS_BY_AREA[area])
        total_output_h += h
        if total_output_h > ARC_MAX:
            return None

        color_a = colors[2 * i]
        color_b = colors[2 * i + 1]
        for _ in range(80):
            mask = _split_mask(h, w)
            comp_a = _crop_patch(mask, color_a)
            comp_b = _crop_patch(~mask, color_b)
            options = _pair_options(_component_from_patch(comp_a, 0), _component_from_patch(comp_b, 1))
            intended = _intended_patch(mask, color_a, color_b)
            if len(options) == 1 and np.array_equal(options[0]["patch"], intended):
                full_rectangles.append(intended)
                component_patches.extend([comp_a, comp_b])
                break
        else:
            return None

    if max(int(rect.shape[1]) for rect in full_rectangles) > ARC_MAX:
        return None
    return full_rectangles, component_patches


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_rectangles()
    if sampled is None:
        return None
    full_rectangles, component_patches = sampled
    inp = _place_patches(component_patches)
    if inp is None:
        return None
    expected = _build_output(full_rectangles)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
