from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 600
GRID_MIN = 15
GRID_MAX = 22


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _components_by_color(grid: np.ndarray, colors: set[int], *, monochromatic: bool) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in colors or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc]:
                        continue
                    nxt = int(grid[nr, nc])
                    if nxt not in colors:
                        continue
                    if monochromatic and nxt != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _split_component(out: np.ndarray, grid: np.ndarray, coords: np.ndarray, rare: int, medium: int) -> None:
    vals = grid[coords[:, 0], coords[:, 1]]
    crack = coords[vals == rare]
    body = coords[vals == medium]
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h != w, "medium object bbox must be non-square"
    assert crack.size > 0 and body.size > 0

    if w > h:
        crack_cols = {int(c) for c in crack[:, 1]}
        crack_rows = {int(r) for r in crack[:, 0]}
        assert crack_rows == set(range(r0, r1 + 1)), "wide object needs a full vertical crack"
        assert len(crack_cols) == 1, "wide object crack must be one column"
        crack_c = next(iter(crack_cols))
        for r, c in body:
            rr = int(r)
            cc = int(c)
            shifted = (rr, cc - 1 if cc < crack_c else cc + 1)
            if _in_bounds(out.shape, shifted[0], shifted[1]):
                out[shifted] = medium
    else:
        crack_rows = {int(r) for r in crack[:, 0]}
        crack_cols = {int(c) for c in crack[:, 1]}
        assert crack_cols == set(range(c0, c1 + 1)), "tall object needs a full horizontal crack"
        assert len(crack_rows) == 1, "tall object crack must be one row"
        crack_r = next(iter(crack_rows))
        for r, c in body:
            rr = int(r)
            cc = int(c)
            shifted = (rr - 1 if rr < crack_r else rr + 1, cc)
            if _in_bounds(out.shape, shifted[0], shifted[1]):
                out[shifted] = medium


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assign color roles by global frequency, erase the rare crack color, split the medium objects, and recolor the most common objects.

    Input structure:
    - A black-background grid contains exactly three non-background colors with unique global pixel counts.
    - The rarest color appears only as a full crack line inside 2-3 mixed objects made jointly with the medium-frequency color.
    - The most common foreground color forms 2-3 separate plain objects with no internal second color.

    Rule:
    - Count the three foreground colors globally to determine rare, medium, and common roles.
    - Erase every rare-color pixel to background.
    - For each mixed medium object, treat the erased rare line as a crack across the bbox short axis and shift the two medium halves one cell away from that crack.
    - Recolor every pixel of the most common-color objects to the rare color; background stays unchanged and out-of-bounds shifted pixels are dropped.

    Key insight:
    - The same visible colors can play different transformation roles across examples; only the global frequency ranking tells you which color is erased, which one splits, and which one gets recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    fg = sorted({int(x) for x in g.ravel() if int(x) != bg})
    assert len(fg) == 3, "expected exactly three foreground colors"

    counts = {color: int(np.count_nonzero(g == color)) for color in fg}
    assert len(set(counts.values())) == 3, "foreground roles must be uniquely ordered by frequency"
    rare, medium, common = sorted(fg, key=lambda color: counts[color])

    out = np.full_like(g, bg)
    out[g == common] = rare

    mixed_mask = (g == rare) | (g == medium)
    mixed_components = _components_from_mask(mixed_mask)
    assert 2 <= len(mixed_components) <= 3, "expected 2-3 medium objects"
    for coords in mixed_components:
        _split_component(out, g, coords, rare, medium)

    return out


CM_DIMS = [
    (2, 4),
    (2, 5),
    (2, 6),
    (2, 7),
    (3, 4),
    (4, 2),
    (5, 2),
    (6, 2),
    (7, 2),
    (4, 3),
]

CF_DIMS = [
    (2, 5),
    (2, 6),
    (2, 7),
    (2, 8),
    (2, 9),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 3),
    (4, 4),
    (5, 2),
    (5, 3),
    (6, 2),
]


def _clip_bbox(bbox: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = bbox
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(h - 1, r1)
    c1 = min(w - 1, c1)
    if r0 > r1 or c0 > c1:
        return None
    return (r0, c0, r1, c1)


def _pad_bbox(bbox: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int] | None:
    return _clip_bbox((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), h, w)


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _cm_output_bbox(r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if w > h:
        return (r0, c0 - 1, r1, c1 + 1)
    return (r0 - 1, c0, r1 + 1, c1)


def _reservation_bbox(kind: str, r0: int, c0: int, h: int, w: int, gh: int, gw: int) -> tuple[int, int, int, int]:
    base = (r0, c0, r0 + h - 1, c0 + w - 1)
    union = base
    if kind == "cm":
        obox = _cm_output_bbox(r0, c0, h, w)
        union = (min(base[0], obox[0]), min(base[1], obox[1]), max(base[2], obox[2]), max(base[3], obox[3]))
    clipped = _clip_bbox(union, gh, gw)
    assert clipped is not None
    halo = _pad_bbox(clipped, 1, gh, gw)
    assert halo is not None
    return halo


def _choose_cm_specs() -> list[dict]:
    n_cm = random.randint(2, 3)
    dims = [random.choice(CM_DIMS) for _ in range(n_cm)]
    if n_cm == 2 and random.random() < 0.7:
        has_wide = any(w > h for h, w in dims)
        has_tall = any(h > w for h, w in dims)
        if not has_wide:
            dims[random.randrange(n_cm)] = random.choice([d for d in CM_DIMS if d[1] > d[0]])
        if not has_tall:
            dims[random.randrange(n_cm)] = random.choice([d for d in CM_DIMS if d[0] > d[1]])

    specs = []
    for h, w in dims:
        if w > h:
            crack = random.randint(1, w - 2)
        else:
            crack = random.randint(1, h - 2)
        specs.append({"kind": "cm", "h": h, "w": w, "crack": crack, "area": h * w})
    return specs


def _choose_cf_specs(cm_total: int) -> list[dict] | None:
    counts = [2, 3]
    random.shuffle(counts)
    for n_cf in counts:
        tries = 80
        while tries > 0:
            tries -= 1
            dims = [random.choice(CF_DIMS) for _ in range(n_cf)]
            total = sum(h * w for h, w in dims)
            if total < cm_total + 5:
                continue
            return [{"kind": "cf", "h": h, "w": w, "area": h * w} for h, w in dims]
    return None


def _place_specs(specs: list[dict], gh: int, gw: int) -> list[dict] | None:
    reserved: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []
    order = sorted(specs, key=lambda s: (s["h"] * s["w"], s["kind"] == "cm"), reverse=True)
    for spec in order:
        positions = [(r, c) for r in range(gh - spec["h"] + 1) for c in range(gw - spec["w"] + 1)]
        random.shuffle(positions)
        if spec["kind"] == "cm" and random.random() < 0.35:
            positions.sort(key=lambda rc: min(rc[0], rc[1], gh - spec["h"] - rc[0], gw - spec["w"] - rc[1]))
        chosen = None
        for r0, c0 in positions:
            halo = _reservation_bbox(spec["kind"], r0, c0, spec["h"], spec["w"], gh, gw)
            if any(_bbox_overlap(halo, old) for old in reserved):
                continue
            chosen = (r0, c0, halo)
            break
        if chosen is None:
            return None
        r0, c0, halo = chosen
        reserved.append(halo)
        placed.append({**spec, "r0": r0, "c0": c0, "halo": halo})
    return placed


def _draw_cm(inp: np.ndarray, expected: np.ndarray, spec: dict, rare: int, medium: int) -> None:
    r0 = spec["r0"]
    c0 = spec["c0"]
    h = spec["h"]
    w = spec["w"]
    crack = spec["crack"]
    inp[r0 : r0 + h, c0 : c0 + w] = medium
    if w > h:
        inp[r0 : r0 + h, c0 + crack] = rare
        for rr in range(r0, r0 + h):
            for cc in range(c0, c0 + w):
                if cc == c0 + crack:
                    continue
                nc = cc - 1 if cc < c0 + crack else cc + 1
                if _in_bounds(expected.shape, rr, nc):
                    expected[rr, nc] = medium
    else:
        inp[r0 + crack, c0 : c0 + w] = rare
        for rr in range(r0, r0 + h):
            for cc in range(c0, c0 + w):
                if rr == r0 + crack:
                    continue
                nr = rr - 1 if rr < r0 + crack else rr + 1
                if _in_bounds(expected.shape, nr, cc):
                    expected[nr, cc] = medium


def _draw_cf(inp: np.ndarray, expected: np.ndarray, spec: dict, rare: int, common: int) -> None:
    r0 = spec["r0"]
    c0 = spec["c0"]
    h = spec["h"]
    w = spec["w"]
    inp[r0 : r0 + h, c0 : c0 + w] = common
    expected[r0 : r0 + h, c0 : c0 + w] = rare


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    gh = random.randint(GRID_MIN, GRID_MAX)
    gw = random.randint(GRID_MIN, GRID_MAX)
    rare, medium, common = random.sample(PALETTE, 3)

    cm_specs = _choose_cm_specs()
    rare_total = sum(min(spec["h"], spec["w"]) for spec in cm_specs)
    medium_total = sum(spec["area"] - min(spec["h"], spec["w"]) for spec in cm_specs)
    if medium_total - rare_total < 5:
        return None

    cf_specs = _choose_cf_specs(medium_total)
    if cf_specs is None:
        return None
    common_total = sum(spec["area"] for spec in cf_specs)
    if common_total - medium_total < 5:
        return None

    all_specs = cm_specs + cf_specs
    placed = _place_specs(all_specs, gh, gw)
    if placed is None:
        return None

    inp = np.zeros((gh, gw), dtype=int)
    expected = np.zeros((gh, gw), dtype=int)
    for spec in placed:
        if spec["kind"] == "cm":
            _draw_cm(inp, expected, spec, rare, medium)
        else:
            _draw_cf(inp, expected, spec, rare, common)

    return inp, expected


def _validate_case(inp: np.ndarray, expected: np.ndarray) -> bool:
    fg = sorted({int(x) for x in inp.ravel() if int(x) != BG})
    if len(fg) != 3:
        return False
    counts = {color: int(np.count_nonzero(inp == color)) for color in fg}
    ordered_counts = sorted(counts.values())
    if len(set(ordered_counts)) != 3:
        return False
    if ordered_counts[1] - ordered_counts[0] < 5 or ordered_counts[2] - ordered_counts[1] < 5:
        return False
    rare, medium, common = sorted(fg, key=lambda color: counts[color])

    mixed = _components_from_mask((inp == rare) | (inp == medium))
    if not (2 <= len(mixed) <= 3):
        return False
    seen_rare = 0
    for coords in mixed:
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h == w:
            return False
        if h * w < 8 or h * w > 14:
            return False
        if coords.shape[0] != h * w:
            return False
        vals = inp[coords[:, 0], coords[:, 1]]
        crack = coords[vals == rare]
        seen_rare += int(crack.shape[0])
        if w > h:
            if crack.shape[0] != h:
                return False
            if {int(r) for r in crack[:, 0]} != set(range(r0, r1 + 1)):
                return False
            crack_cols = {int(c) for c in crack[:, 1]}
            if len(crack_cols) != 1:
                return False
            crack_c = next(iter(crack_cols))
            if crack_c in (c0, c1):
                return False
        else:
            if crack.shape[0] != w:
                return False
            if {int(c) for c in crack[:, 1]} != set(range(c0, c1 + 1)):
                return False
            crack_rows = {int(r) for r in crack[:, 0]}
            if len(crack_rows) != 1:
                return False
            crack_r = next(iter(crack_rows))
            if crack_r in (r0, r1):
                return False
    if seen_rare != counts[rare]:
        return False

    common_comps = _components_by_color(inp, {common}, monochromatic=True)
    if not (2 <= len(common_comps) <= 3):
        return False
    if any(not (10 <= comp["area"] <= 18) for comp in common_comps):
        return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if np.any(actual == common):
        return False
    if not np.any(actual == rare) or not np.any(actual == medium):
        return False
    if np.array_equal(actual, inp):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_case()
        if built is None:
            continue
        inp, expected = built
        if _validate_case(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid input after many attempts")
