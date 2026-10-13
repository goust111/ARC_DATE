from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import math
import random

import numpy as np


BG = 0
OUTER_COLOR = 1
SPRITE_COLOR = 2
GUIDE_COLOR = 3
MIDDLE_COLOR = 5
INNER_COLOR = 9

ORTHO4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ORTHO8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GEN_RETRIES = 600


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _contains_rect(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 <= ir0 and oc0 <= ic0 and ir1 <= or1 and ic1 <= oc1


def _inset_rect(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _side_matches(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return int(a[0] == b[0]) + int(a[1] == b[1]) + int(a[2] == b[2]) + int(a[3] == b[3])


def _rect_contains_point(rect: tuple[int, int, int, int], r: float, c: float) -> bool:
    r0, c0, r1, c1 = rect
    return r0 <= r <= r1 and c0 <= c <= c1


def _frame_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _paint_frame(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    for r, c in _frame_cells(rect):
        grid[r, c] = int(color)


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    dirs = ORTHO4 if int(connectivity) == 4 else ORTHO8
    seen = np.zeros(mask.shape, dtype=bool)
    parts: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _ordered_layers_from_grid(grid: np.ndarray) -> list[dict]:
    colors = sorted(set(map(int, grid.flat)) - {BG, SPRITE_COLOR, GUIDE_COLOR})
    layers = []
    for color in colors:
        coords = np.argwhere(grid == int(color))
        rect = _bbox_from_coords(coords)
        layers.append({"color": int(color), "raw": rect, "rect": rect, "area": _rect_area(rect)})
    layers.sort(key=lambda item: int(item["area"]), reverse=True)

    completed: list[dict] = []
    for layer in layers:
        raw = layer["raw"]
        rect = raw
        if completed:
            parents = [
                parent
                for parent in completed
                if _contains_rect(parent["rect"], raw) and parent["rect"] != raw
            ]
            if parents:
                parent = min(parents, key=lambda item: _rect_area(item["rect"]))
                inset = _inset_rect(parent["rect"])
                if inset[0] <= inset[2] and inset[1] <= inset[3] and _contains_rect(inset, raw):
                    if _side_matches(raw, inset) >= 3:
                        rect = inset
        layer["rect"] = rect
        completed.append(layer)
    return layers


def _deepest_layer_color(layers: list[dict], r: float, c: float) -> int | None:
    for layer in sorted(layers, key=lambda item: _rect_area(item["rect"])):
        if _rect_contains_point(layer["rect"], r, c):
            return int(layer["color"])
    return None


def _infer_vector_candidates(
    sprite_coords: np.ndarray, guide_coords: np.ndarray
) -> list[tuple[int, int]]:
    sprite = [tuple(map(int, p)) for p in sprite_coords]
    guide = {tuple(map(int, p)) for p in guide_coords}
    candidates = set()
    for sr, sc in sprite:
        for gr, gc in guide:
            dr, dc = gr - sr, gc - sc
            shifted = {(r + dr, c + dc) for r, c in sprite}
            if guide <= shifted:
                candidates.add((int(dr), int(dc)))
    return sorted(candidates)


def _infer_guide_vector(sprite_coords: np.ndarray, guide_coords: np.ndarray) -> tuple[int, int]:
    candidates = _infer_vector_candidates(sprite_coords, guide_coords)
    if not candidates:
        raise AssertionError("guide is not a subset of any translated central sprite")
    best_norm = max(dr * dr + dc * dc for dr, dc in candidates)
    best = [vec for vec in candidates if vec[0] * vec[0] + vec[1] * vec[1] == best_norm]
    best.sort()
    return best[-1]


def _projection_sequences(
    shape: tuple[int, int], layers: list[dict], sprite_coords: np.ndarray, vector: tuple[int, int]
) -> list[int]:
    center_r = float(np.mean(sprite_coords[:, 0]))
    center_c = float(np.mean(sprite_coords[:, 1]))
    dr, dc = vector
    colors = []
    for k in range(1, max(shape) + 1):
        color = _deepest_layer_color(layers, center_r + k * dr, center_c + k * dc)
        if color is None:
            break
        colors.append(int(color))
    return colors


def _render_output_from_solution(
    shape: tuple[int, int], layers: list[dict], sprite_coords: np.ndarray, vectors: list[tuple[int, int]]
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for layer in layers:
        _paint_frame(out, layer["rect"], int(layer["color"]))

    overlay = np.full(shape, BG, dtype=int)
    center_r = float(np.mean(sprite_coords[:, 0]))
    center_c = float(np.mean(sprite_coords[:, 1]))
    ordered = sorted(
        [(math.atan2(dr, dc) % (2 * math.pi), dr, dc) for dr, dc in vectors],
        key=lambda item: item[0],
    )

    for _ang, dr, dc in ordered:
        for k in range(1, max(shape) + 1):
            color = _deepest_layer_color(layers, center_r + k * dr, center_c + k * dc)
            if color is None:
                break
            for r, c in sprite_coords:
                rr, cc = int(r + k * dr), int(c + k * dc)
                if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                    overlay[rr, cc] = int(color)

    out[overlay != BG] = overlay[overlay != BG]
    for layer in layers:
        _paint_frame(out, layer["rect"], int(layer["color"]))
    out[sprite_coords[:, 0], sprite_coords[:, 1]] = SPRITE_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete sparse nested rectangle borders, then project the central sprite along each guide vector with layer-dependent recoloring.

    Input structure:
    - Black is the background.
    - Blue/grey/maroon sparse pixels indicate 2-3 strictly nested axis-aligned rectangle borders.
    - A single red 3x3 irregular sprite sits inside the innermost rectangle.
    - One to three green partial copies of that sprite sit outside the outer border and act as direction guides.

    Rule:
    - Complete each sparse border to its full rectangle frame; when a child bbox matches at least three sides of its parent's one-pixel inset, expand it to that inset.
    - For each green guide, find the largest translation that makes the red sprite cover the guide subset.
    - Repeatedly translate the red sprite by that vector; each translated copy is recolored to the deepest completed rectangle whose region contains the copy's geometric center.
    - Stop that direction as soon as the copy center leaves every completed rectangle; erase the original green guides; keep the completed borders and the original red sprite unchanged.
    - If different directions overlap, later directions overwrite earlier ones after sorting vectors by clockwise angle from the rightward direction.

    Key insight:
    - The sparse rectangles do not just need completion: once completed, they define the layer regions that decide the color of every projected sprite copy.
    """
    g = np.asarray(grid, dtype=int)
    layers = _ordered_layers_from_grid(g)

    red_components = _components(g == SPRITE_COLOR, connectivity=8)
    assert len(red_components) == 1, "expected exactly one central red sprite"
    sprite_coords = red_components[0]

    guide_components = _components(g == GUIDE_COLOR, connectivity=8)
    assert 1 <= len(guide_components) <= 3, "expected one to three green guide copies"
    vectors = [_infer_guide_vector(sprite_coords, guide) for guide in guide_components]

    return _render_output_from_solution(g.shape, layers, sprite_coords, vectors)


def _random_sprite_mask() -> np.ndarray:
    banks = [
        ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
        ((0, 1), (0, 2), (1, 1), (2, 1), (2, 2)),
        ((0, 1), (0, 2), (1, 1), (2, 0), (2, 1)),
        ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 2), (1, 0), (1, 1), (1, 2), (2, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)),
        ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ]
    cells = random.choice(banks)
    mask = np.zeros((3, 3), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _choose_rectangles(shape: tuple[int, int]) -> tuple[list[dict], list[tuple[int, int, int, int]], list[dict]]:
    h, w = shape
    layer_count = 2 if random.random() < 0.35 else 3
    colors = [OUTER_COLOR, INNER_COLOR] if layer_count == 2 else [OUTER_COLOR, MIDDLE_COLOR, INNER_COLOR]

    outer_h = random.randint(max(12, h - 10), h - 4)
    outer_w = random.randint(max(12, w - 10), w - 4)
    outer_r0 = random.randint(1, h - outer_h - 1)
    outer_c0 = random.randint(1, w - outer_w - 1)
    rects = [(outer_r0, outer_c0, outer_r0 + outer_h - 1, outer_c0 + outer_w - 1)]
    raw_specs = [{"mode": "exact"}]

    for i in range(1, layer_count):
        parent = rects[-1]
        pr0, pc0, pr1, pc1 = parent
        remaining = layer_count - i - 1
        min_side = 5 + 2 * remaining
        can_inset = (pr1 - pr0 + 1) >= min_side + 2 and (pc1 - pc0 + 1) >= min_side + 2

        if can_inset and random.random() < 0.45:
            child = (pr0 + 1, pc0 + 1, pr1 - 1, pc1 - 1)
            rects.append(child)
            raw_specs.append({"mode": "inset", "omit_side": random.choice([None, "top", "bottom", "left", "right"])})
            continue

        for _ in range(120):
            top = random.randint(2, 4)
            bottom = random.randint(2, 4)
            left = random.randint(2, 4)
            right = random.randint(2, 4)
            child = (pr0 + top, pc0 + left, pr1 - bottom, pc1 - right)
            if child[2] - child[0] + 1 < min_side or child[3] - child[1] + 1 < min_side:
                continue
            rects.append(child)
            raw_specs.append({"mode": "exact"})
            break
        else:
            raise RuntimeError("failed to place nested rectangles")

    layers = [
        {"color": int(color), "rect": rect, "area": _rect_area(rect)}
        for color, rect in zip(colors, rects)
    ]
    return layers, rects, raw_specs


def _side_positions(rect: tuple[int, int, int, int], side: str, omit_side: str | None) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    if side == "top":
        if omit_side == "top":
            return []
        cols = list(range(c0, c1 + 1))
        if omit_side == "left":
            cols = [c for c in cols if c != c0]
        if omit_side == "right":
            cols = [c for c in cols if c != c1]
        return [(r0, c) for c in cols]
    if side == "bottom":
        if omit_side == "bottom":
            return []
        cols = list(range(c0, c1 + 1))
        if omit_side == "left":
            cols = [c for c in cols if c != c0]
        if omit_side == "right":
            cols = [c for c in cols if c != c1]
        return [(r1, c) for c in cols]
    if side == "left":
        if omit_side == "left":
            return []
        rows = list(range(r0, r1 + 1))
        if omit_side == "top":
            rows = [r for r in rows if r != r0]
        if omit_side == "bottom":
            rows = [r for r in rows if r != r1]
        return [(r, c0) for r in rows]
    if omit_side == "right":
        return []
    rows = list(range(r0, r1 + 1))
    if omit_side == "top":
        rows = [r for r in rows if r != r0]
    if omit_side == "bottom":
        rows = [r for r in rows if r != r1]
    return [(r, c1) for r in rows]


def _sparse_frame_samples(
    rect: tuple[int, int, int, int], omit_side: str | None
) -> set[tuple[int, int]]:
    samples: set[tuple[int, int]] = set()
    for side in ("top", "bottom", "left", "right"):
        positions = _side_positions(rect, side, omit_side)
        if not positions:
            continue
        anchors = random.sample(positions, k=min(2, len(positions)))
        samples.update(anchors)
        extra_pool = [p for p in positions if p not in samples]
        random.shuffle(extra_pool)
        extra_count = random.randint(0, max(0, len(extra_pool) // 2))
        samples.update(extra_pool[:extra_count])
    return samples


def _sprite_coords_from_mask(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask)
    coords = coords + np.array([top, left], dtype=int)
    return coords.astype(int)


def _connected_subsets(coords: list[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    out = []
    for size in (3, 2):
        for subset in combinations(coords, size):
            patch = np.zeros((7, 7), dtype=bool)
            base_r = min(r for r, _ in subset)
            base_c = min(c for _, c in subset)
            for r, c in subset:
                patch[r - base_r, c - base_c] = True
            if len(_components(patch, connectivity=8)) == 1:
                out.append(tuple(sorted(subset)))
    return out


def _vector_signature(dr: int, dc: int) -> tuple[int, int]:
    g = math.gcd(abs(int(dr)), abs(int(dc)))
    if g == 0:
        return 0, 0
    return int(dr // g), int(dc // g)


def _build_candidate_guides(
    shape: tuple[int, int],
    layers: list[dict],
    sprite_coords: np.ndarray,
) -> list[dict]:
    h, w = shape
    guide_anchor = layers[-1]["rect"]
    center_r = float(np.mean(sprite_coords[:, 0]))
    center_c = float(np.mean(sprite_coords[:, 1]))
    sprite_list = [tuple(map(int, p)) for p in sprite_coords]
    candidates = []
    for dr, dc in SEARCH_VECTORS:
        shifted = [(r + dr, c + dc) for r, c in sprite_list]
        if not all(0 <= r < h and 0 <= c < w for r, c in shifted):
            continue
        seq = _projection_sequences(shape, layers, sprite_coords, (dr, dc))
        if not seq:
            continue
        outside = [(r, c) for r, c in shifted if not _rect_contains_point(guide_anchor, r, c)]
        if len(outside) < 2:
            continue
        subsets = _connected_subsets(outside)
        if not subsets:
            continue
        for subset in subsets:
            guide_coords = np.array(subset, dtype=int)
            inferred = _infer_guide_vector(sprite_coords, guide_coords)
            if inferred != (dr, dc):
                continue
            candidates.append(
                {
                    "vector": (int(dr), int(dc)),
                    "guide": guide_coords,
                    "seq": seq,
                    "sig": _vector_signature(dr, dc),
                    "angle": math.atan2(dr, dc) % (2 * math.pi),
                    "first_color": int(seq[0]),
                    "multi_color": len(set(seq)) >= 2,
                    "k1_center": (center_r + dr, center_c + dc),
                }
            )
    candidates.sort(key=lambda item: (item["angle"], len(item["seq"])))
    return candidates


def _guides_far_apart(selected: list[dict]) -> bool:
    for a, b in combinations(selected, 2):
        a_cells = {tuple(map(int, p)) for p in a["guide"]}
        b_cells = {tuple(map(int, p)) for p in b["guide"]}
        for ar, ac in a_cells:
            for br, bc in b_cells:
                if max(abs(ar - br), abs(ac - bc)) <= 1:
                    return False
    return True


def _choose_guides(candidates: list[dict], layer_count: int) -> list[dict] | None:
    del layer_count
    desired_count = random.randint(1, min(3, max(1, len(candidates))))
    combos = []
    for count in range(desired_count, 0, -1):
        for combo in combinations(candidates, count):
            sigs = {item["sig"] for item in combo}
            if len(sigs) != len(combo):
                continue
            if not _guides_far_apart(list(combo)):
                continue
            combos.append(list(combo))
        if combos:
            break
    return random.choice(combos) if combos else None


def _manual_expected(
    shape: tuple[int, int], layers: list[dict], sprite_coords: np.ndarray, selected: list[dict]
) -> np.ndarray:
    vectors = [tuple(item["vector"]) for item in selected]
    return _render_output_from_solution(shape, layers, sprite_coords, vectors)


def _place_sprite(rect: tuple[int, int, int, int], sprite_mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = rect
    ideal_r = (r0 + r1) // 2 - 1
    ideal_c = (c0 + c1) // 2 - 1
    row_lo = max(r0 + 1, ideal_r - 1)
    row_hi = min(r1 - 3, ideal_r + 1)
    col_lo = max(c0 + 1, ideal_c - 1)
    col_hi = min(c1 - 3, ideal_c + 1)
    top = random.randint(row_lo, row_hi)
    left = random.randint(col_lo, col_hi)
    return _sprite_coords_from_mask(sprite_mask, top, left)


def _make_input_grid(
    shape: tuple[int, int], layers: list[dict], raw_specs: list[dict], sprite_coords: np.ndarray, guides: list[dict]
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for layer, spec in zip(layers, raw_specs):
        samples = _sparse_frame_samples(layer["rect"], spec.get("omit_side"))
        for r, c in samples:
            grid[r, c] = int(layer["color"])
    grid[sprite_coords[:, 0], sprite_coords[:, 1]] = SPRITE_COLOR
    for guide in guides:
        coords = guide["guide"]
        grid[coords[:, 0], coords[:, 1]] = GUIDE_COLOR
    return grid


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    layers: list[dict],
    sprite_coords: np.ndarray,
    guides: list[dict],
) -> bool:
    anchor = layers[-1]["rect"]
    for guide in guides:
        coords = guide["guide"]
        if coords.shape[0] not in (2, 3):
            return False
        if all(_rect_contains_point(anchor, int(r), int(c)) for r, c in coords):
            return False

    if len(_components(inp == SPRITE_COLOR, connectivity=8)) != 1:
        return False
    if len(_components(inp == GUIDE_COLOR, connectivity=8)) != len(guides):
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    output_colors = set(map(int, expected.flat)) - {BG}
    if len(output_colors) < 2:
        return False

    projected = expected.copy()
    for layer in layers:
        for r, c in _frame_cells(layer["rect"]):
            projected[r, c] = BG
    projected[sprite_coords[:, 0], sprite_coords[:, 1]] = BG
    proj_colors = set(map(int, projected.flat)) - {BG}
    if not proj_colors:
        return False
    return True


GUIDE_TEMPLATES = {
    "top": [(-6, 0), (-6, -1), (-6, 1), (-5, 0)],
    "bottom": [(6, 0), (6, -1), (6, 1), (5, 0)],
    "left": [(0, -6), (-1, -6), (1, -6), (0, -5)],
    "right": [(0, 6), (-1, 6), (1, 6), (0, 5)],
}

SEARCH_VECTORS = [
    (-6, 0),
    (-6, -1),
    (-6, 1),
    (-4, -4),
    (-4, 4),
    (0, -6),
    (-1, -6),
    (1, -6),
    (0, 6),
    (-1, 6),
    (1, 6),
    (4, -4),
    (4, 4),
    (6, 0),
    (6, -1),
    (6, 1),
]


def _extreme_subset_key(subset: tuple[tuple[int, int], ...], side: str) -> tuple[int, int, int]:
    rows = [r for r, _ in subset]
    cols = [c for _, c in subset]
    if side == "top":
        return (max(rows), sum(rows), sum(cols))
    if side == "bottom":
        return (-min(rows), -sum(rows), sum(cols))
    if side == "left":
        return (max(cols), sum(cols), sum(rows))
    return (-min(cols), -sum(cols), sum(rows))


def _guide_subset_for_vector(
    sprite_coords: np.ndarray, vector: tuple[int, int], side: str
) -> np.ndarray | None:
    dr, dc = vector
    shifted = [(int(r + dr), int(c + dc)) for r, c in sprite_coords]
    centroid_r = float(np.mean([r for r, _ in shifted]))
    centroid_c = float(np.mean([c for _, c in shifted]))
    subsets = _connected_subsets(shifted)
    if not subsets:
        return None
    ordered = sorted(subsets, key=lambda subset: _extreme_subset_key(subset, side))
    for subset in ordered:
        if side == "top" and not all(r < centroid_r for r, _ in subset):
            continue
        if side == "bottom" and not all(r > centroid_r for r, _ in subset):
            continue
        if side == "left" and not all(c < centroid_c for _, c in subset):
            continue
        if side == "right" and not all(c > centroid_c for _, c in subset):
            continue
        rows = [r for r, _ in subset]
        cols = [c for _, c in subset]
        if side == "top" and max(rows) + 1 > math.floor(centroid_r):
            continue
        if side == "bottom" and math.ceil(centroid_r) > min(rows) - 1:
            continue
        if side == "left" and max(cols) + 1 > math.floor(centroid_c):
            continue
        if side == "right" and math.ceil(centroid_c) > min(cols) - 1:
            continue
        guide = np.array(subset, dtype=int)
        if _infer_guide_vector(sprite_coords, guide) != vector:
            continue
        return guide
    return None


def _choose_outer_from_guides(
    shape: tuple[int, int], sprite_coords: np.ndarray, guides: list[dict]
) -> tuple[int, int, int, int] | None:
    h, w = shape
    sbox = _bbox_from_coords(sprite_coords)
    r0_lo, r0_hi = 1, sbox[0] - 2
    c0_lo, c0_hi = 1, sbox[1] - 2
    r1_lo, r1_hi = sbox[2] + 2, h - 2
    c1_lo, c1_hi = sbox[3] + 2, w - 2

    for guide in guides:
        rows = guide["guide"][:, 0]
        cols = guide["guide"][:, 1]
        centroid_r, centroid_c = guide["centroid"]
        side = guide["side"]
        if side == "top":
            r0_lo = max(r0_lo, int(rows.max()) + 1)
            r0_hi = min(r0_hi, int(math.floor(centroid_r)))
        elif side == "bottom":
            r1_lo = max(r1_lo, int(math.ceil(centroid_r)))
            r1_hi = min(r1_hi, int(rows.min()) - 1)
        elif side == "left":
            c0_lo = max(c0_lo, int(cols.max()) + 1)
            c0_hi = min(c0_hi, int(math.floor(centroid_c)))
        else:
            c1_lo = max(c1_lo, int(math.ceil(centroid_c)))
            c1_hi = min(c1_hi, int(cols.min()) - 1)

    if r0_lo > r0_hi or c0_lo > c0_hi or r1_lo > r1_hi or c1_lo > c1_hi:
        return None

    for _ in range(120):
        r0 = random.randint(r0_lo, r0_hi)
        c0 = random.randint(c0_lo, c0_hi)
        r1 = random.randint(max(r1_lo, r0 + 8), r1_hi)
        c1 = random.randint(max(c1_lo, c0 + 8), c1_hi)
        outer = (r0, c0, r1, c1)
        if not _contains_rect(outer, sbox):
            continue
        if any(any(_rect_contains_point(outer, int(r), int(c)) for r, c in guide["guide"]) for guide in guides):
            continue
        return outer
    return None


def _random_inner_rect(
    container: tuple[int, int, int, int],
    sprite_box: tuple[int, int, int, int],
    min_margin_to_container: int,
) -> tuple[int, int, int, int] | None:
    cr0, cc0, cr1, cc1 = container
    sr0, sc0, sr1, sc1 = sprite_box
    r0_min = max(cr0 + min_margin_to_container, sr0 - 3)
    r0_max = min(sr0 - 1, cr1 - min_margin_to_container - 4)
    c0_min = max(cc0 + min_margin_to_container, sc0 - 3)
    c0_max = min(sc0 - 1, cc1 - min_margin_to_container - 4)
    r1_min = max(sr1 + 1, cr0 + min_margin_to_container + 4)
    r1_max = min(cc := cr1 - min_margin_to_container, sr1 + 3)
    c1_min = max(sc1 + 1, cc0 + min_margin_to_container + 4)
    c1_max = min(cc1 - min_margin_to_container, sc1 + 3)
    if r0_min > r0_max or c0_min > c0_max or r1_min > r1_max or c1_min > c1_max:
        return None
    for _ in range(80):
        r0 = random.randint(r0_min, r0_max)
        c0 = random.randint(c0_min, c0_max)
        r1 = random.randint(max(r1_min, r0 + 4), r1_max)
        c1 = random.randint(max(c1_min, c0 + 4), c1_max)
        rect = (r0, c0, r1, c1)
        if _contains_rect(rect, sprite_box) and rect[2] - rect[0] + 1 >= 5 and rect[3] - rect[1] + 1 >= 5:
            return rect
    return None


def _construct_layers_from_outer(
    outer: tuple[int, int, int, int], sprite_coords: np.ndarray
) -> tuple[list[dict], list[dict]] | None:
    sprite_box = _bbox_from_coords(sprite_coords)
    layer_count = 2 if random.random() < 0.35 else 3
    colors = [OUTER_COLOR, INNER_COLOR] if layer_count == 2 else [OUTER_COLOR, MIDDLE_COLOR, INNER_COLOR]

    if layer_count == 2:
        options = []
        inset = _inset_rect(outer)
        if _contains_rect(inset, sprite_box) and inset[2] - inset[0] + 1 >= 5 and inset[3] - inset[1] + 1 >= 5:
            options.append((inset, {"mode": "inset", "omit_side": random.choice([None, "top", "bottom", "left", "right"])}))
        inner = _random_inner_rect(outer, sprite_box, min_margin_to_container=2)
        if inner is not None:
            options.append((inner, {"mode": "exact"}))
        if not options:
            return None
        inner_rect, inner_spec = random.choice(options)
        rects = [outer, inner_rect]
        specs = [{"mode": "exact"}, inner_spec]
    else:
        middle_options = []
        mid_inset = _inset_rect(outer)
        if _contains_rect(mid_inset, sprite_box) and mid_inset[2] - mid_inset[0] + 1 >= 7 and mid_inset[3] - mid_inset[1] + 1 >= 7:
            middle_options.append((mid_inset, {"mode": "inset", "omit_side": random.choice([None, "top", "bottom", "left", "right"])}))
        middle_rand = _random_inner_rect(outer, sprite_box, min_margin_to_container=2)
        if middle_rand is not None and middle_rand[2] - middle_rand[0] + 1 >= 7 and middle_rand[3] - middle_rand[1] + 1 >= 7:
            middle_options.append((middle_rand, {"mode": "exact"}))
        if not middle_options:
            return None

        middle_rect, middle_spec = random.choice(middle_options)
        inner_options = []
        inner_inset = _inset_rect(middle_rect)
        if _contains_rect(inner_inset, sprite_box) and inner_inset[2] - inner_inset[0] + 1 >= 5 and inner_inset[3] - inner_inset[1] + 1 >= 5:
            inner_options.append((inner_inset, {"mode": "inset", "omit_side": random.choice([None, "top", "bottom", "left", "right"])}))
        inner_rand = _random_inner_rect(middle_rect, sprite_box, min_margin_to_container=2)
        if inner_rand is not None:
            inner_options.append((inner_rand, {"mode": "exact"}))
        if not inner_options:
            return None

        inner_rect, inner_spec = random.choice(inner_options)
        rects = [outer, middle_rect, inner_rect]
        specs = [{"mode": "exact"}, middle_spec, inner_spec]

    layers = [{"color": int(color), "rect": rect, "area": _rect_area(rect)} for color, rect in zip(colors, rects)]
    return layers, specs


def _constructive_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    sprite_mask = _random_sprite_mask()
    sprite_top = random.randint(6, h - 10)
    sprite_left = random.randint(6, w - 10)
    sprite_coords = _sprite_coords_from_mask(sprite_mask, sprite_top, sprite_left)

    supported: dict[str, dict] = {}
    for side, vectors in GUIDE_TEMPLATES.items():
        shuffled = vectors[:]
        random.shuffle(shuffled)
        for vector in shuffled:
            shifted = sprite_coords + np.array(vector, dtype=int)
            if shifted[:, 0].min() < 0 or shifted[:, 0].max() >= h or shifted[:, 1].min() < 0 or shifted[:, 1].max() >= w:
                continue
            guide_subset = _guide_subset_for_vector(sprite_coords, vector, side)
            if guide_subset is None:
                continue
            supported[side] = {
                "side": side,
                "vector": vector,
                "guide": guide_subset,
                "centroid": (
                    float(np.mean(sprite_coords[:, 0] + vector[0])),
                    float(np.mean(sprite_coords[:, 1] + vector[1])),
                ),
            }
            break

    if not supported:
        return None

    side_count = random.randint(1, min(3, len(supported)))
    chosen_sides = random.sample(list(supported), k=side_count)
    guides = []
    for side in chosen_sides:
        guides.append(supported[side])

    if not _guides_far_apart(guides):
        return None

    outer = _choose_outer_from_guides((h, w), sprite_coords, guides)
    if outer is None:
        return None

    constructed = _construct_layers_from_outer(outer, sprite_coords)
    if constructed is None:
        return None
    layers, raw_specs = constructed
    inp = _make_input_grid((h, w), layers, raw_specs, sprite_coords, guides)
    expected = _manual_expected((h, w), layers, sprite_coords, guides)
    if not _validate_instance(inp, expected, layers, sprite_coords, guides):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(18, 24)
        w = random.randint(18, 24)
        try:
            layers, _rects, raw_specs = _choose_rectangles((h, w))
        except RuntimeError:
            continue

        sprite_mask = _random_sprite_mask()
        try:
            sprite_coords = _place_sprite(layers[-1]["rect"], sprite_mask)
        except ValueError:
            continue

        candidates = _build_candidate_guides((h, w), layers, sprite_coords)
        if not candidates:
            continue
        selected = _choose_guides(candidates, len(layers))
        if not selected:
            continue

        inp = _make_input_grid((h, w), layers, raw_specs, sprite_coords, selected)
        expected = _manual_expected((h, w), layers, sprite_coords, selected)
        if _validate_instance(inp, expected, layers, sprite_coords, selected):
            return inp

    raise RuntimeError("failed to generate a valid layered ray-projection ARC input")
