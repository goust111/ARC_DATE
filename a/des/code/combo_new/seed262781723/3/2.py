from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], bytes]:
    mask = _mask_from_coords(coords)
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    coords = np.argwhere(mask)
    cropped = np.zeros((1, 1), dtype=bool) if coords.size == 0 else _mask_from_coords(coords)
    return tuple(map(int, cropped.shape)), cropped.astype(np.uint8).tobytes()


def _touches_boundary(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _find_holes(grid: np.ndarray, template_mask: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    holes: list[np.ndarray] = []
    for comp in _components(grid == BG):
        if _touches_boundary(comp, h, w):
            continue
        valid = True
        touches_template = False
        for r, c in comp:
            for dr, dc in DIR4:
                nr = int(r + dr)
                nc = int(c + dc)
                if grid[nr, nc] == BG:
                    continue
                if not template_mask[nr, nc]:
                    valid = False
                    break
                touches_template = True
            if not valid:
                break
        if valid and touches_template:
            holes.append(comp)
    return holes


def _select_template(grid: np.ndarray) -> tuple[int, np.ndarray, list[np.ndarray]]:
    best: tuple[int, int, int, np.ndarray, list[np.ndarray]] | None = None
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        for comp in _components(grid == int(color)):
            template_mask = np.zeros_like(grid, dtype=bool)
            template_mask[comp[:, 0], comp[:, 1]] = True
            holes = _find_holes(grid, template_mask)
            score = (len(holes), len(comp))
            if best is None or score > (best[0], best[1]):
                best = (int(score[0]), int(score[1]), int(color), template_mask, holes)
    assert best is not None and best[0] > 0, "expected one perforated template component"
    return int(best[2]), best[3], best[4]


def _extract_fragments(grid: np.ndarray, template_color: int) -> list[dict[str, object]]:
    fragments: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, template_color)):
        for comp in _components(grid == int(color)):
            fragments.append(
                {
                    "color": int(color),
                    "coords": comp,
                    "key": _shape_key(comp),
                }
            )
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each outside fragment to the grey template hole with the same exact shape and use its color to fill that hole.

    Input structure:
    - Black background surrounds one large grey 4-connected template component.
    - The grey template contains several black holes fully enclosed by grey, and those hole shapes are all different.
    - Outside the template, each colored fragment is a 4-connected single-color component on the black background.

    Rule:
    - Find the perforated grey template and extract every enclosed black hole as a shape normalized by translation only.
    - Extract every outside colored fragment and normalize each fragment shape in the same way.
    - Pair each hole with the unique fragment whose shape matches exactly, without rotation or reflection.
    - Fill each hole with the matched fragment color and remove all outside fragments to black while leaving the grey template unchanged.

    Key insight:
    - Color is only the payload; the correspondence is determined entirely by exact shape equality between holes and scattered fragments.
    """
    g = np.asarray(grid, dtype=int)
    template_color, template_mask, holes = _select_template(g)
    fragments = _extract_fragments(g, template_color)

    hole_by_key: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for hole in holes:
        key = _shape_key(hole)
        assert key not in hole_by_key, "hole shapes must be unique"
        hole_by_key[key] = hole

    frag_by_key: dict[tuple[tuple[int, int], bytes], dict[str, object]] = {}
    for frag in fragments:
        key = frag["key"]
        assert key not in frag_by_key, "fragment shapes must be unique"
        frag_by_key[key] = frag

    assert hole_by_key.keys() == frag_by_key.keys(), "hole/fragment shape sets must match exactly"

    out = g.copy()
    for frag in fragments:
        coords = frag["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for key, hole in hole_by_key.items():
        color = int(frag_by_key[key]["color"])
        out[hole[:, 0], hole[:, 1]] = color

    return out


SHAPE_LIBRARY = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [0, 1, 0, 0]], dtype=bool),
]

MAX_RETRIES = 400
HOLE_COUNT_RANGE = (2, 6)
TEMPLATE_H_RANGE = (11, 19)
TEMPLATE_W_RANGE = (11, 20)
GRID_SIZE_MIN = 18


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    comp = _components(mask)
    return len(comp) == 1 and len(comp[0]) == len(coords)


def _all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    variants: list[np.ndarray] = []
    for flip in (False, True):
        cur = np.fliplr(mask) if flip else mask
        for k in range(4):
            rot = np.rot90(cur, k)
            key = _mask_key(rot)
            if key not in seen:
                seen.add(key)
                variants.append(rot.copy())
    return variants


def _sample_shapes(count: int) -> list[np.ndarray] | None:
    keys: set[tuple[tuple[int, int], bytes]] = set()
    shapes: list[np.ndarray] = []
    for _ in range(200):
        if len(shapes) == count:
            return shapes
        base = SHAPE_LIBRARY[int(RNG.integers(0, len(SHAPE_LIBRARY)))]
        options = _all_orientations(base)
        shape = options[int(RNG.integers(0, len(options)))]
        key = _mask_key(shape)
        if key in keys:
            continue
        keys.add(key)
        shapes.append(shape)
    return None


def _mark_halo(blocked: np.ndarray, shape: np.ndarray, r0: int, c0: int) -> None:
    h, w = blocked.shape
    for dr, dc in np.argwhere(shape):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        blocked[max(0, rr - 1) : min(h, rr + 2), max(0, cc - 1) : min(w, cc + 2)] = True


def _place_holes(th: int, tw: int, shapes: list[np.ndarray]) -> list[dict[str, object]] | None:
    blocked = np.zeros((th, tw), dtype=bool)
    holes: list[dict[str, object]] = []
    for shape in sorted(shapes, key=lambda m: int(m.sum()), reverse=True):
        sh, sw = map(int, shape.shape)
        if th - sh < 2 or tw - sw < 2:
            return None
        candidates = [
            (r0, c0)
            for r0 in range(1, th - sh)
            for c0 in range(1, tw - sw)
            if not bool((blocked[r0 : r0 + sh, c0 : c0 + sw] & shape).any())
        ]
        if not candidates:
            return None
        order = RNG.permutation(len(candidates))
        r0, c0 = candidates[int(order[0])]
        holes.append({"shape": shape, "r0": int(r0), "c0": int(c0)})
        _mark_halo(blocked, shape, int(r0), int(c0))
    return holes


def _holes_mask(th: int, tw: int, holes: list[dict[str, object]]) -> np.ndarray:
    mask = np.zeros((th, tw), dtype=bool)
    for hole in holes:
        shape = hole["shape"]
        r0 = int(hole["r0"])
        c0 = int(hole["c0"])
        sh, sw = shape.shape
        mask[r0 : r0 + sh, c0 : c0 + sw] |= shape
    return mask


def _make_outer_mask(th: int, tw: int, holes_mask: np.ndarray) -> np.ndarray | None:
    outer = np.ones((th, tw), dtype=bool)
    protected = np.zeros((th, tw), dtype=bool)
    _mark_halo(protected, holes_mask, 0, 0)

    target_notches = int(RNG.integers(1, 5))
    accepted = 0
    for _ in range(80):
        if accepted >= target_notches:
            break
        side = int(RNG.integers(0, 4))
        if side in (0, 1):
            max_depth = max(1, th // 3)
            max_width = max(2, tw - 2)
            depth = int(RNG.integers(1, max_depth + 1))
            width = int(RNG.integers(1, max_width + 1))
            if width >= tw:
                width = tw - 1
            start = int(RNG.integers(0, tw - width + 1))
            rs = slice(0, depth) if side == 0 else slice(th - depth, th)
            cs = slice(start, start + width)
        else:
            max_depth = max(1, tw // 3)
            max_height = max(2, th - 2)
            depth = int(RNG.integers(1, max_depth + 1))
            height = int(RNG.integers(1, max_height + 1))
            if height >= th:
                height = th - 1
            start = int(RNG.integers(0, th - height + 1))
            rs = slice(start, start + height)
            cs = slice(0, depth) if side == 2 else slice(tw - depth, tw)

        if protected[rs, cs].any():
            continue
        candidate = outer.copy()
        candidate[rs, cs] = False
        template = candidate & ~holes_mask
        if not _connected(template):
            continue
        outer = candidate
        accepted += 1

    if accepted == 0:
        return None
    return outer


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    coords = np.argwhere(mask)
    grid[r0 + coords[:, 0], c0 + coords[:, 1]] = int(color)


def _expanded_rect_mask(h: int, w: int, r0: int, c0: int, th: int, tw: int, pad: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + th + pad)
    cc1 = min(w, c0 + tw + pad)
    mask[rr0:rr1, cc0:cc1] = True
    return mask


def _place_fragment(
    grid: np.ndarray,
    occupied: np.ndarray,
    forbidden: np.ndarray,
    shape: np.ndarray,
    color: int,
) -> bool:
    h, w = grid.shape
    sh, sw = shape.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            ok = True
            for dr, dc in np.argwhere(shape):
                rr = int(r0 + dr)
                cc = int(c0 + dc)
                if occupied[rr, cc] or forbidden[rr, cc]:
                    ok = False
                    break
            if ok:
                candidates.append((int(r0), int(c0)))
    if not candidates:
        return False
    r0, c0 = candidates[int(RNG.integers(0, len(candidates)))]
    _paint_mask(grid, shape, int(r0), int(c0), int(color))
    _mark_halo(occupied, shape, int(r0), int(c0))
    return True


def _gen_attempt() -> np.ndarray | None:
    hole_count = int(RNG.integers(*HOLE_COUNT_RANGE))
    shapes = _sample_shapes(hole_count)
    if shapes is None:
        return None

    th = int(RNG.integers(*TEMPLATE_H_RANGE))
    tw = int(RNG.integers(*TEMPLATE_W_RANGE))
    holes = _place_holes(th, tw, shapes)
    if holes is None:
        return None

    hole_mask = _holes_mask(th, tw, holes)
    outer_mask = _make_outer_mask(th, tw, hole_mask)
    if outer_mask is None:
        return None
    template_mask = outer_mask & ~hole_mask
    if not _connected(template_mask):
        return None

    min_h = max(GRID_SIZE_MIN, th + 6)
    min_w = max(GRID_SIZE_MIN, tw + 6)
    if min_h > 30 or min_w > 30:
        return None
    h = int(RNG.integers(min_h, 31))
    w = int(RNG.integers(min_w, 31))

    margin_r = h - th
    margin_c = w - tw
    if margin_r < 2 or margin_c < 2:
        return None
    template_r0 = int(RNG.integers(1, margin_r))
    template_c0 = int(RNG.integers(1, margin_c))

    colors = RNG.choice([1, 2, 3, 4, 6, 7, 8, 9], size=hole_count, replace=False)
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    _paint_mask(out, template_mask, template_r0, template_c0, GREY)
    _paint_mask(inp, template_mask, template_r0, template_c0, GREY)

    hole_entries: list[dict[str, object]] = []
    for hole, color in zip(holes, colors):
        shape = hole["shape"]
        hr = template_r0 + int(hole["r0"])
        hc = template_c0 + int(hole["c0"])
        _paint_mask(out, shape, hr, hc, int(color))
        hole_entries.append({"shape": shape, "color": int(color), "r0": int(hr), "c0": int(hc)})

    occupied = np.zeros((h, w), dtype=bool)
    _mark_halo(occupied, template_mask, template_r0, template_c0)
    forbidden = _expanded_rect_mask(h, w, template_r0, template_c0, th, tw, pad=1)

    for item in sorted(hole_entries, key=lambda d: int(np.sum(d["shape"])), reverse=True):
        if not _place_fragment(inp, occupied, forbidden, item["shape"], int(item["color"])):
            return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-fragment matching sample")
