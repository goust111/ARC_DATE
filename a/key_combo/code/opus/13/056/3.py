from __future__ import annotations

import random

import numpy as np


BLACK = 0
ORANGE = 7
BASE_COLORS = [1, 2, 3, 4, 5, 6, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_MIN = 20
GRID_MAX = 28
RECT_MIN = 4
RECT_MAX = 7
GEN_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    H, W = mask.shape
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rects = []
    for coords in _components_mask(np.asarray(grid) != BLACK):
        bbox = _bbox_coords(coords)
        sub = _crop_bbox(grid, bbox)
        if not np.all(sub != BLACK):
            raise AssertionError("foreground components must be dense rectangles")
        colors = sorted(set(map(int, sub.reshape(-1))))
        base_colors = [c for c in colors if c != ORANGE]
        if len(base_colors) != 1:
            raise AssertionError("each rectangle must have exactly one non-orange base color")
        markers = np.argwhere(sub == ORANGE)
        rects.append(
            {
                "bbox": bbox,
                "sub": sub,
                "h": int(sub.shape[0]),
                "w": int(sub.shape[1]),
                "base_color": int(base_colors[0]),
                "markers": [tuple(map(int, rc)) for rc in markers],
            }
        )
    return rects


def _equivalence_kind(source: dict, target: dict) -> str | None:
    if (source["h"], source["w"]) == (target["h"], target["w"]):
        return "same"
    if (source["h"], source["w"]) == (target["w"], target["h"]):
        return "swapped"
    return None


def _map_markers(source: dict, target_shape: tuple[int, int], kind: str) -> set[tuple[int, int]]:
    if kind == "same":
        return set(source["markers"])
    if kind != "swapped":
        raise ValueError(f"unknown mapping kind: {kind}")
    mapped = set()
    for r, c in source["markers"]:
        mapped.add((c, source["h"] - 1 - r))
    H, W = target_shape
    if not all(0 <= r < H and 0 <= c < W for r, c in mapped):
        raise AssertionError("rotated markers must stay inside the swapped target")
    return mapped


def _transpose_shortcut(source: dict, target_shape: tuple[int, int]) -> set[tuple[int, int]]:
    H, W = target_shape
    out = {(c, r) for r, c in source["markers"]}
    if not all(0 <= r < H and 0 <= c < W for r, c in out):
        raise AssertionError("transpose shortcut should stay in bounds for generated samples")
    return out


def _source_order_key(rect: dict) -> tuple[int, int]:
    r0, c0, _, _ = rect["bbox"]
    return int(c0), int(r0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer orange interior marker patterns only into plain rectangles whose outer rectangle is shape-equivalent to a marked one.

    Input structure:
    - The grid is a black background with 4-6 separated dense axis-aligned rectangles in distinct base colors.
    - Some rectangles are plain single-color blocks, while 1-2 rectangles also contain orange interior marker pixels.
    - Every foreground component fully fills its bounding box, so the outer shape is determined only by the rectangle height and width.

    Rule:
    - Detect every rectangle and split them into marked sources and plain targets.
    - A plain target is gated on only when its size matches a source exactly or after swapping height and width.
    - For an exact match, copy each orange marker with the same edge-relative position; for a swapped-height/width match, rotate the marker pattern 90 degrees clockwise so the source row distances become target column distances and source column distances become target row distances.
    - Apply all matching marked sources to each plain target in source order sorted by left-to-right, then top-to-bottom; non-matching plain rectangles and the marked source rectangles themselves stay unchanged.

    Key insight:
    - The orange pattern is not copied globally: rectangle-size equivalence is the gate, and width-height swaps require a corresponding rotation of the edge-relative marker layout.
    """
    g = np.asarray(grid, dtype=int)
    rects = _extract_rectangles(g)
    sources = sorted([rect for rect in rects if rect["markers"]], key=_source_order_key)
    targets = [rect for rect in rects if not rect["markers"]]

    out = g.copy()
    for target in targets:
        tr0, tc0, _, _ = target["bbox"]
        for source in sources:
            kind = _equivalence_kind(source, target)
            if kind is None:
                continue
            for rr, cc in _map_markers(source, (target["h"], target["w"]), kind):
                out[tr0 + rr, tc0 + cc] = ORANGE
    return out


def _pick_dims(
    *,
    exclude_equiv: set[tuple[int, int]] | None = None,
    force_nonsquare: bool = False,
) -> tuple[int, int]:
    blocked = set() if exclude_equiv is None else set(exclude_equiv)
    for _ in range(200):
        h = random.randint(RECT_MIN, RECT_MAX)
        w = random.randint(RECT_MIN, RECT_MAX)
        if force_nonsquare and h == w:
            continue
        if (h, w) in blocked or (w, h) in blocked:
            continue
        return h, w
    raise RuntimeError("failed to pick rectangle dimensions")


def _pattern_candidates(h: int, w: int) -> list[tuple[int, int]]:
    limit = min(h, w) - 2
    cands = []
    for r in range(1, limit + 1):
        for c in range(1, limit + 1):
            rr, cc = h - 1 - r, w - 1 - c
            if (r, c) == (rr, cc):
                continue
            if (r, c) > (rr, cc):
                continue
            cands.append((r, c))
    return cands


def _build_pattern(h: int, w: int, seeds: list[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set()
    for r, c in seeds:
        out.add((int(r), int(c)))
        out.add((int(h - 1 - r), int(w - 1 - c)))
    return out


def _sample_pattern(h: int, w: int) -> set[tuple[int, int]]:
    cands = _pattern_candidates(h, w)
    if len(cands) < 2:
        raise RuntimeError("not enough marker candidates")
    for _ in range(200):
        count = random.randint(1, min(3, len(cands)))
        seeds = random.sample(cands, count)
        pattern = _build_pattern(h, w, seeds)
        rotated = {(c, h - 1 - r) for r, c in pattern}
        if pattern == rotated:
            continue
        return pattern
    raise RuntimeError("failed to sample a non-degenerate marker pattern")


def _render_rectangles(rect_specs: list[dict], grid_shape: tuple[int, int]) -> np.ndarray | None:
    H, W = grid_shape
    grid = np.zeros((H, W), dtype=int)
    blocked = np.zeros((H, W), dtype=bool)
    order = list(range(len(rect_specs)))
    random.shuffle(order)

    for idx in order:
        spec = rect_specs[idx]
        h, w = spec["h"], spec["w"]
        placed = False
        row_lo = 1
        row_hi = H - h - 1
        col_lo = 1
        col_hi = W - w - 1
        if row_hi < row_lo or col_hi < col_lo:
            return None
        for _ in range(250):
            r0 = random.randint(row_lo, row_hi)
            c0 = random.randint(col_lo, col_hi)
            r1 = r0 + h - 1
            c1 = c0 + w - 1
            br0 = max(0, r0 - 1)
            bc0 = max(0, c0 - 1)
            br1 = min(H - 1, r1 + 1)
            bc1 = min(W - 1, c1 + 1)
            if blocked[br0 : br1 + 1, bc0 : bc1 + 1].any():
                continue
            patch = np.full((h, w), int(spec["base_color"]), dtype=int)
            for rr, cc in spec["markers"]:
                patch[rr, cc] = ORANGE
            grid[r0 : r1 + 1, c0 : c1 + 1] = patch
            blocked[br0 : br1 + 1, bc0 : bc1 + 1] = True
            spec["bbox"] = (r0, c0, r1, c1)
            placed = True
            break
        if not placed:
            return None
    return grid


def _random_unmatched_dims(forbidden: list[tuple[int, int]]) -> tuple[int, int]:
    return _pick_dims(exclude_equiv=set(forbidden), force_nonsquare=False)


def _single_source_scenario() -> list[dict]:
    source_dims = _pick_dims(force_nonsquare=True)
    h, w = source_dims
    pattern = _sample_pattern(h, w)

    rects = [
        {"kind": "source", "h": h, "w": w, "markers": pattern},
        {"kind": "plain", "h": h, "w": w, "markers": set()},
        {"kind": "plain", "h": w, "w": h, "markers": set()},
    ]

    uh, uw = _random_unmatched_dims([source_dims])
    rects.append({"kind": "plain", "h": uh, "w": uw, "markers": set()})

    if random.random() < 0.6:
        extra_same = random.choice([(h, w), (w, h)])
        rects.append({"kind": "plain", "h": extra_same[0], "w": extra_same[1], "markers": set()})
    elif random.random() < 0.5:
        eh, ew = _random_unmatched_dims([source_dims, (uh, uw)])
        rects.append({"kind": "plain", "h": eh, "w": ew, "markers": set()})

    return rects


def _two_source_disjoint_scenario() -> list[dict]:
    dims1 = _pick_dims(force_nonsquare=True)
    dims2 = _pick_dims(exclude_equiv={dims1}, force_nonsquare=True)
    p1 = _sample_pattern(*dims1)
    p2 = _sample_pattern(*dims2)

    rects = [
        {"kind": "source", "h": dims1[0], "w": dims1[1], "markers": p1},
        {"kind": "source", "h": dims2[0], "w": dims2[1], "markers": p2},
    ]

    t1 = random.choice([dims1, dims1[::-1]])
    t2 = random.choice([dims2, dims2[::-1]])
    rects.append({"kind": "plain", "h": t1[0], "w": t1[1], "markers": set()})
    rects.append({"kind": "plain", "h": t2[0], "w": t2[1], "markers": set()})

    uh, uw = _random_unmatched_dims([dims1, dims2])
    rects.append({"kind": "plain", "h": uh, "w": uw, "markers": set()})

    if random.random() < 0.55:
        extra = random.choice([dims1, dims1[::-1], dims2, dims2[::-1]])
        rects.append({"kind": "plain", "h": extra[0], "w": extra[1], "markers": set()})

    return rects


def _two_source_overlap_scenario() -> list[dict]:
    dims1 = _pick_dims(force_nonsquare=True)
    dims2 = dims1[::-1]
    for _ in range(200):
        p1 = _sample_pattern(*dims1)
        p2 = _sample_pattern(*dims2)
        shared_dims = random.choice([dims1, dims2])
        source1 = {"h": dims1[0], "w": dims1[1], "markers": p1}
        source2 = {"h": dims2[0], "w": dims2[1], "markers": p2}
        k1 = _equivalence_kind(source1, {"h": shared_dims[0], "w": shared_dims[1]})
        k2 = _equivalence_kind(source2, {"h": shared_dims[0], "w": shared_dims[1]})
        m1 = _map_markers(source1, shared_dims, k1)
        m2 = _map_markers(source2, shared_dims, k2)
        union = m1 | m2
        if union != m1 and union != m2:
            break
    else:
        raise RuntimeError("failed to build overlapping two-source case")

    rects = [
        {"kind": "source", "h": dims1[0], "w": dims1[1], "markers": p1},
        {"kind": "source", "h": dims2[0], "w": dims2[1], "markers": p2},
        {"kind": "plain", "h": shared_dims[0], "w": shared_dims[1], "markers": set()},
    ]

    if random.random() < 0.7:
        other = dims2 if shared_dims == dims1 else dims1
        rects.append({"kind": "plain", "h": other[0], "w": other[1], "markers": set()})

    uh, uw = _random_unmatched_dims([dims1])
    rects.append({"kind": "plain", "h": uh, "w": uw, "markers": set()})
    return rects


def _assign_colors(rect_specs: list[dict]) -> None:
    palette = random.sample(BASE_COLORS, len(rect_specs))
    for spec, color in zip(rect_specs, palette):
        spec["base_color"] = int(color)


def _make_scenario() -> list[dict]:
    mode = random.choices(
        ["single", "disjoint", "overlap"],
        weights=[0.4, 0.35, 0.25],
        k=1,
    )[0]
    if mode == "single":
        rects = _single_source_scenario()
    elif mode == "disjoint":
        rects = _two_source_disjoint_scenario()
    else:
        rects = _two_source_overlap_scenario()
    _assign_colors(rects)
    return rects


def _orange_set_in_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    local = np.argwhere(grid[r0 : r1 + 1, c0 : c1 + 1] == ORANGE)
    return {tuple(map(int, rc)) for rc in local}


def _validate_semantics(inp: np.ndarray) -> bool:
    try:
        rects = _extract_rectangles(inp)
        if not (4 <= len(rects) <= 6):
            return False
        sources = sorted([rect for rect in rects if rect["markers"]], key=_source_order_key)
        targets = [rect for rect in rects if not rect["markers"]]
        if not (1 <= len(sources) <= 2):
            return False
        if len(set(rect["base_color"] for rect in rects)) != len(rects):
            return False
        if any((rect["h"] < 3 or rect["w"] < 3) for rect in rects):
            return False

        out = main(inp)

        matched_targets = []
        unmatched_targets = []
        swapped_targets = []
        multi_source_targets = []
        for target in targets:
            matched = []
            final_markers = set()
            shortcut_markers = set()
            for source in sources:
                kind = _equivalence_kind(source, target)
                if kind is None:
                    continue
                matched.append((source, kind))
                final_markers |= _map_markers(source, (target["h"], target["w"]), kind)
                if kind == "same":
                    shortcut_markers |= set(source["markers"])
                else:
                    shortcut_markers |= _transpose_shortcut(source, (target["h"], target["w"]))
            painted = _orange_set_in_bbox(out, target["bbox"])
            if painted != final_markers:
                return False
            if matched:
                matched_targets.append(target)
            else:
                unmatched_targets.append(target)
                if painted:
                    return False
            if any(kind == "swapped" for _, kind in matched):
                swapped_targets.append((target, final_markers, shortcut_markers))
            if len(matched) >= 2:
                multi_source_targets.append((target, matched, final_markers))

        if not matched_targets or not unmatched_targets:
            return False

        if not swapped_targets:
            return False
        if not any(final != shortcut for _, final, shortcut in swapped_targets):
            return False

        if len(sources) == 2:
            has_distinguishing_target = False
            for target, matched, final_markers in multi_source_targets:
                per_source = [
                    _map_markers(source, (target["h"], target["w"]), kind)
                    for source, kind in matched
                ]
                if all(final_markers != markers for markers in per_source):
                    has_distinguishing_target = True
                    break
            if multi_source_targets and not has_distinguishing_target:
                return False

        if np.array_equal(inp, out):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rect_specs = _make_scenario()
        H = random.randint(GRID_MIN, GRID_MAX)
        W = random.randint(GRID_MIN, GRID_MAX)
        grid = _render_rectangles(rect_specs, (H, W))
        if grid is None:
            continue
        if _validate_semantics(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
