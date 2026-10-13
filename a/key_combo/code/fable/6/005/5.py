from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]

BG = 0
MAX_GEN_ATTEMPTS = 400
PARTITION_ATTEMPTS = 240
DECOY_ATTEMPTS = 240


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox_coords(coords: list[Coord] | tuple[Coord, ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_coords(coords: list[Coord] | tuple[Coord, ...] | set[Coord]) -> tuple[Coord, ...]:
    if not coords:
        return ()
    r0, c0, _, _ = _bbox_coords(list(coords))
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_dims(coords: list[Coord] | tuple[Coord, ...]) -> tuple[int, int]:
    norm = _normalize_coords(coords)
    if not norm:
        return 0, 0
    rows = [r for r, _ in norm]
    cols = [c for _, c in norm]
    return max(rows) + 1, max(cols) + 1


def _coords_to_mask(coords: list[Coord] | tuple[Coord, ...] | set[Coord]) -> np.ndarray:
    norm = _normalize_coords(coords)
    h, w = _shape_dims(norm)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _mask_to_coords(mask: np.ndarray) -> tuple[Coord, ...]:
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


def _components_mask(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != int(bg)):
        for coords in _components_mask(grid == color):
            comps.append(
                {
                    "color": int(color),
                    "coords": tuple(coords),
                    "area": len(coords),
                    "bbox": _bbox_coords(coords),
                }
            )
    return comps


def _all_transforms(coords: list[Coord] | tuple[Coord, ...]) -> list[tuple[Coord, ...]]:
    mask = _coords_to_mask(coords)
    variants: list[tuple[Coord, ...]] = []
    seen: set[tuple[Coord, ...]] = set()
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            out = np.rot90(base, k)
            norm = _normalize_coords(_mask_to_coords(out))
            if norm not in seen:
                seen.add(norm)
                variants.append(norm)
    return variants


def _shape_equal_under_d4(a: list[Coord] | tuple[Coord, ...], b: list[Coord] | tuple[Coord, ...]) -> bool:
    nb = _normalize_coords(b)
    return any(t == nb for t in _all_transforms(a))


def _is_connected_shape(coords: list[Coord] | tuple[Coord, ...]) -> bool:
    mask = _coords_to_mask(coords)
    return len(_components_mask(mask)) == 1


def _is_rectangle_shape(coords: list[Coord] | tuple[Coord, ...]) -> bool:
    h, w = _shape_dims(coords)
    return len(coords) == h * w


def _touches_all_edges(coords: list[Coord] | tuple[Coord, ...], h: int, w: int) -> bool:
    pts = set(coords)
    return (
        any(r == 0 for r, _ in pts)
        and any(r == h - 1 for r, _ in pts)
        and any(c == 0 for _, c in pts)
        and any(c == w - 1 for _, c in pts)
    )


def _complements_to_rectangle(hole: list[Coord] | tuple[Coord, ...], fragment: list[Coord] | tuple[Coord, ...]) -> bool:
    hole_norm = _normalize_coords(hole)
    hole_h, hole_w = _shape_dims(hole_norm)
    hole_set = set(hole_norm)
    frag_variants = [set(v) for v in _all_transforms(fragment)]
    total_area = len(hole_norm) + len(fragment)
    seen_rects: set[tuple[int, int]] = set()

    for rect_h in range(1, total_area + 1):
        if total_area % rect_h != 0:
            continue
        rect_w = total_area // rect_h
        for rh, rw in ((rect_h, rect_w), (rect_w, rect_h)):
            if (rh, rw) in seen_rects:
                continue
            seen_rects.add((rh, rw))
            if rh < hole_h or rw < hole_w:
                continue
            full = {(r, c) for r in range(rh) for c in range(rw)}
            for frag_norm in frag_variants:
                fh, fw = _shape_dims(tuple(frag_norm))
                if rh < fh or rw < fw:
                    continue
                for r0 in range(rh - hole_h + 1):
                    for c0 in range(rw - hole_w + 1):
                        placed_hole = {(r + r0, c + c0) for r, c in hole_set}
                        complement = full - placed_hole
                        if _normalize_coords(complement) == tuple(sorted(frag_norm)):
                            return True
    return False


def _hole_components(grid: np.ndarray, bg: int, body_color: int) -> list[dict]:
    holes: list[dict] = []
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            touches_border = False
            boundary_colors: set[int] = set()
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if grid[nr, nc] == bg and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
                    elif grid[nr, nc] != bg:
                        boundary_colors.add(int(grid[nr, nc]))
            if not touches_border and boundary_colors == {int(body_color)}:
                holes.append(
                    {
                        "coords": tuple(comp),
                        "bbox": _bbox_coords(comp),
                        "area": len(comp),
                    }
                )
    holes.sort(key=lambda x: (x["bbox"][0], x["bbox"][1], x["area"]))
    return holes


def _detect_structure(grid: np.ndarray) -> tuple[int, dict, list[dict], list[dict]]:
    bg = int(Counter(map(int, grid.ravel())).most_common(1)[0][0])
    comps = _find_monochrome_components(grid, bg)
    assert comps, "expected foreground components"
    areas = [c["area"] for c in comps]
    largest = max(areas)
    body_candidates = [c for c in comps if c["area"] == largest]
    assert len(body_candidates) == 1, "expected one largest body component"
    body = body_candidates[0]
    holes = _hole_components(grid, bg, int(body["color"]))
    assert holes, "expected enclosed holes"
    body_coords = set(body["coords"])
    fragments = [c for c in comps if set(c["coords"]) != body_coords]
    assert len(fragments) >= len(holes), "expected at least as many fragments as holes"
    return bg, body, holes, fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed hole in the main body with the uniquely matching fragment color that completes it to a solid rectangle.

    Input structure:
    - The background is the most common color.
    - One largest monochrome component is the main body; inside it are one or more enclosed background holes.
    - Outside the body are several monochrome fragment sprites in distinct colors, including possible decoys.

    Rule:
    - For each hole, find the unique fragment whose shape can be rotated, flipped, and translated so that hole plus fragment exactly tiles a solid rectangle.
    - Recolor every cell of that hole with the fragment's color.
    - Erase each used fragment from its original location, while leaving unmatched decoy fragments unchanged.

    Key insight:
    - The fragment is not equal to the hole; it is the complementary missing part of some rectangle, so matching must test rectangle completion under D4 transforms.
    """
    g = np.asarray(grid, dtype=int)
    bg, _body, holes, fragments = _detect_structure(g)

    matches: list[int] = []
    for hole in holes:
        fit_indices = [
            idx
            for idx, frag in enumerate(fragments)
            if _complements_to_rectangle(hole["coords"], frag["coords"])
        ]
        assert len(fit_indices) == 1, "each hole must have exactly one matching fragment"
        matches.append(fit_indices[0])

    assert len(set(matches)) == len(matches), "matched fragments must be unique"

    out = g.copy()
    used = set(matches)
    for hole, frag_idx in zip(holes, matches):
        color = int(fragments[frag_idx]["color"])
        for r, c in hole["coords"]:
            out[r, c] = color
    for frag_idx in used:
        for r, c in fragments[frag_idx]["coords"]:
            out[r, c] = bg
    return out


def _grow_connected_shape(h: int, w: int, target_area: int) -> tuple[Coord, ...] | None:
    seed = (random.randrange(h), random.randrange(w))
    cells = {seed}
    frontier = {p for p in _neighbors4(*seed) if 0 <= p[0] < h and 0 <= p[1] < w}
    while len(cells) < target_area and frontier:
        front = list(frontier)
        random.shuffle(front)
        chosen = None
        best_score = None
        for cand in front[: min(len(front), 10)]:
            rr, cc = cand
            score = 0
            if rr in (0, h - 1):
                score += 1
            if cc in (0, w - 1):
                score += 1
            score += random.random()
            if best_score is None or score > best_score:
                best_score = score
                chosen = cand
        assert chosen is not None
        frontier.remove(chosen)
        cells.add(chosen)
        for nb in _neighbors4(*chosen):
            if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in cells:
                frontier.add(nb)
    if len(cells) != target_area:
        return None
    return tuple(sorted(cells))


def _gen_partition() -> tuple[int, int, tuple[Coord, ...], tuple[Coord, ...]] | None:
    size_choices = [(3, 4), (4, 3), (4, 4), (4, 5), (5, 4), (5, 5)]
    for _ in range(PARTITION_ATTEMPTS):
        h, w = random.choice(size_choices)
        area = h * w
        lo = max(4, area // 3)
        hi = min(area - 4, (2 * area) // 3)
        if lo > hi:
            continue
        target = random.randint(lo, hi)
        hole = _grow_connected_shape(h, w, target)
        if hole is None:
            continue
        if not _touches_all_edges(hole, h, w):
            continue
        if _is_rectangle_shape(hole):
            continue
        fragment = tuple(sorted((r, c) for r in range(h) for c in range(w) if (r, c) not in set(hole)))
        if not _is_connected_shape(fragment):
            continue
        if _is_rectangle_shape(fragment):
            continue
        if _shape_equal_under_d4(hole, fragment):
            continue
        if len(_all_transforms(fragment)) < 2:
            continue
        return h, w, hole, fragment
    return None


def _can_place_rect(existing: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int, gap: int) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for er0, ec0, er1, ec1 in existing:
        if not (r1 + gap < er0 or er1 + gap < r0 or c1 + gap < ec0 or ec1 + gap < c0):
            return False
    return True


def _place_holes_in_body(body_h: int, body_w: int, rect_sizes: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    placed: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for h, w in rect_sizes:
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, body_h - h):
            for c0 in range(1, body_w - w):
                if _can_place_rect(boxes, r0, c0, h, w, gap=1):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        pos = random.choice(candidates)
        placed.append(pos)
        boxes.append((pos[0], pos[1], pos[0] + h - 1, pos[1] + w - 1))
    return placed


def _shape_to_patch(shape: tuple[Coord, ...], color: int) -> np.ndarray:
    mask = _coords_to_mask(shape)
    out = np.full(mask.shape, BG, dtype=int)
    out[mask] = color
    return out


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 1) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    cells = [(r0 + r, c0 + c) for r in range(h) for c in range(w) if patch[r, c] != BG]
    if not cells:
        return False
    for rr, cc in cells:
        if grid[rr, cc] != BG:
            return False
    for rr, cc in cells:
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                nr, nc = rr + dr, cc + dc
                if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] != BG and (nr, nc) not in cells:
                    return False
    return True


def _place_patch_random(grid: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for r0 in range(grid.shape[0] - patch.shape[0] + 1):
        for c0 in range(grid.shape[1] - patch.shape[1] + 1):
            if _can_place_patch(grid, patch, r0, c0, padding=1):
                candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _direct_match_any(hole_shapes: list[tuple[Coord, ...]], shape: tuple[Coord, ...]) -> bool:
    return any(_shape_equal_under_d4(hole, shape) for hole in hole_shapes)


def _random_decoy_shape(target_area: int | None = None) -> tuple[Coord, ...] | None:
    size_choices = [(3, 3), (3, 4), (4, 3), (4, 4), (4, 5), (5, 4)]
    for _ in range(DECOY_ATTEMPTS):
        h, w = random.choice(size_choices)
        area = h * w
        lo = 3
        hi = min(area - 1, area - 2)
        if lo > hi:
            continue
        target = target_area if target_area is not None else random.randint(lo, hi)
        if not (2 <= target <= area - 1):
            continue
        shape = _grow_connected_shape(h, w, target)
        if shape is None:
            continue
        if _is_rectangle_shape(shape):
            continue
        return _normalize_coords(shape)
    return None


def _build_decoys(hole_shapes: list[tuple[Coord, ...]], matched_fragments: list[tuple[Coord, ...]]) -> list[tuple[Coord, ...]]:
    decoys: list[tuple[Coord, ...]] = []

    hole_based = list(hole_shapes)
    random.shuffle(hole_based)
    for hole in hole_based:
        variant_choices = _all_transforms(hole)
        random.shuffle(variant_choices)
        for cand in variant_choices:
            if any(_complements_to_rectangle(h, cand) for h in hole_shapes):
                continue
            decoys.append(cand)
            break
        if decoys:
            break

    area_target = len(random.choice(matched_fragments))
    for _ in range(DECOY_ATTEMPTS):
        cand = _random_decoy_shape(area_target)
        if cand is None:
            continue
        if any(_complements_to_rectangle(h, cand) for h in hole_shapes):
            continue
        decoys.append(cand)
        break

    return decoys


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        bg, _body, holes, fragments = _detect_structure(inp)
    except Exception:
        return False

    if len(fragments) <= len(holes):
        return False

    hole_shapes = [_normalize_coords(h["coords"]) for h in holes]
    frag_shapes = [_normalize_coords(f["coords"]) for f in fragments]

    match_map: list[list[int]] = []
    for hole in hole_shapes:
        fit = [i for i, frag in enumerate(frag_shapes) if _complements_to_rectangle(hole, frag)]
        if len(fit) != 1:
            return False
        match_map.append(fit)

    used = {fit[0] for fit in match_map}
    if len(used) != len(holes):
        return False
    if any(i not in used and any(_complements_to_rectangle(hole, frag_shapes[i]) for hole in hole_shapes) for i in range(len(fragments))):
        return False

    if not any(_direct_match_any(hole_shapes, frag_shapes[i]) for i in range(len(fragments)) if i not in used):
        return False

    if not any(len(frag_shapes[i]) == len(frag_shapes[j]) for i in used for j in range(len(fragments)) if i != j):
        return False

    if any(any(_shape_equal_under_d4(hole, frag_shapes[i]) for i in used) for hole in hole_shapes):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected) and int(bg) == BG


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(22, 30)
    grid_w = random.randint(22, 30)
    body_h = random.randint(11, min(18, grid_h - 6))
    body_w = random.randint(11, min(18, grid_w - 6))
    holes_n = random.choice([1, 2, 2, 3])
    decoy_target = 2

    parts: list[tuple[int, int, tuple[Coord, ...], tuple[Coord, ...]]] = []
    for _ in range(holes_n):
        part = _gen_partition()
        if part is None:
            return None
        parts.append(part)

    rect_sizes = [(h, w) for h, w, _hole, _frag in parts]
    placements = _place_holes_in_body(body_h, body_w, rect_sizes)
    if placements is None:
        return None

    if body_h + 4 > grid_h or body_w + 4 > grid_w:
        return None
    body_r0 = random.randint(2, grid_h - body_h - 2)
    body_c0 = random.randint(2, grid_w - body_w - 2)

    palette = random.sample(range(1, 10), holes_n + decoy_target + 1)
    body_color = palette[0]
    fragment_colors = palette[1:]

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    inp[body_r0 : body_r0 + body_h, body_c0 : body_c0 + body_w] = body_color

    expected = inp.copy()
    hole_shapes: list[tuple[Coord, ...]] = []
    matched_fragments: list[tuple[Coord, ...]] = []

    for idx, ((rh, rw, hole, frag), (local_r0, local_c0)) in enumerate(zip(parts, placements)):
        hole_shapes.append(_normalize_coords(hole))
        matched_fragments.append(_normalize_coords(frag))
        color = fragment_colors[idx]
        for r, c in hole:
            inp[body_r0 + local_r0 + r, body_c0 + local_c0 + c] = BG
            expected[body_r0 + local_r0 + r, body_c0 + local_c0 + c] = color

    decoys = _build_decoys(hole_shapes, matched_fragments)
    if len(decoys) < decoy_target:
        return None

    all_fragment_shapes: list[tuple[Coord, ...]] = []
    for idx, frag in enumerate(matched_fragments):
        variants = _all_transforms(frag)
        original = _normalize_coords(frag)
        nontrivial = [v for v in variants if v != original]
        if idx == 0 and nontrivial:
            chosen = random.choice(nontrivial)
        else:
            chosen = random.choice(variants)
        all_fragment_shapes.append(chosen)
    all_fragment_shapes.extend(decoys[:decoy_target])

    for idx, shape in enumerate(all_fragment_shapes):
        color = fragment_colors[idx]
        patch = _shape_to_patch(shape, color)
        pos = _place_patch_random(inp, patch)
        if pos is None:
            return None
        r0, c0 = pos
        for r in range(patch.shape[0]):
            for c in range(patch.shape[1]):
                if patch[r, c] != BG:
                    inp[r0 + r, c0 + c] = color
                    if idx < holes_n:
                        expected[r0 + r, c0 + c] = BG
                    else:
                        expected[r0 + r, c0 + c] = color

    if not _semantic_checks(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
