from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


_PROTO_LIBRARY = [
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
]

_GRID_MIN = 16
_GRID_MAX = 21
_GEN_TRIES = 2000


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _scale_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((scale, scale), dtype=int)).astype(bool)


def _paint(shape: tuple[int, int], sprites: list[dict], order: list[int], protos: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=int)
    color_to_sprite = {int(s["color"]): s for s in sprites}
    for color in order:
        sprite = color_to_sprite[int(color)]
        mask = _scale_mask(protos[int(sprite["proto"])], int(sprite["scale"]))
        top = int(sprite["top"])
        left = int(sprite["left"])
        h, w = mask.shape
        region = canvas[top : top + h, left : left + w]
        region[mask] = int(color)
        canvas[top : top + h, left : left + w] = region
    return canvas


def _full_mask(shape: tuple[int, int], sprite: dict, proto: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    patch = _scale_mask(proto, int(sprite["scale"]))
    top = int(sprite["top"])
    left = int(sprite["left"])
    h, w = patch.shape
    mask[top : top + h, left : left + w] = patch
    return mask


def _unique_topo(nodes: list[int], edges: set[tuple[int, int]]) -> list[int] | None:
    above = {int(n): set() for n in nodes}
    indeg = {int(n): 0 for n in nodes}
    for lo, hi in edges:
        lo = int(lo)
        hi = int(hi)
        if lo == hi:
            return None
        if hi not in above[lo]:
            above[lo].add(hi)
            indeg[hi] += 1
    order = []
    placed = set()
    while len(order) < len(nodes):
        avail = [n for n in nodes if n not in placed and indeg[n] == 0]
        if len(avail) != 1:
            return None
        cur = avail[0]
        order.append(cur)
        placed.add(cur)
        for nxt in above[cur]:
            indeg[nxt] -= 1
    return order


def _observable_edges(canvas: np.ndarray, masks: dict[int, np.ndarray], colors: list[int]) -> set[tuple[int, int]] | None:
    edges: set[tuple[int, int]] = set()
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            overlap = masks[a] & masks[b]
            if not np.any(overlap):
                continue
            visible = {int(v) for v in np.unique(canvas[overlap]) if int(v) in (int(a), int(b))}
            if len(visible) > 1:
                return None
            if len(visible) == 1:
                top = next(iter(visible))
                bot = int(b) if top == int(a) else int(a)
                edges.add((bot, top))
    return edges


def _overlap_pairs(masks: dict[int, np.ndarray], colors: list[int]) -> set[frozenset[int]]:
    pairs: set[frozenset[int]] = set()
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            if np.any(masks[a] & masks[b]):
                pairs.add(frozenset((int(a), int(b))))
    return pairs


def _candidate_solutions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    if len(colors) < 3:
        return []

    sprites = []
    for color in colors:
        vis = g == int(color)
        box = _bbox(vis)
        if box is None:
            return []
        r0, c0, r1, c1 = box
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w or h % 3 != 0:
            return []
        scale = h // 3
        if scale < 1 or scale > 3:
            return []
        sprites.append(
            {
                "color": int(color),
                "top": int(r0),
                "left": int(c0),
                "scale": int(scale),
                "visible": _crop(vis, box).copy(),
                "area": int(h * w),
            }
        )

    min_area = min(int(s["area"]) for s in sprites)
    small = [s for s in sprites if int(s["area"]) == min_area]
    if len(small) < 2 or min_area != 9:
        return []

    solutions = []
    lib_pairs = list(itertools.combinations(range(len(_PROTO_LIBRARY)), 2))
    for ia, ib in lib_pairs:
        protos = (_PROTO_LIBRARY[int(ia)], _PROTO_LIBRARY[int(ib)])
        options = []
        for sprite in sprites:
            cands = []
            for proto_idx in (0, 1):
                scaled = _scale_mask(protos[proto_idx], int(sprite["scale"]))
                if scaled.shape != sprite["visible"].shape:
                    continue
                if np.all(sprite["visible"] <= scaled):
                    cands.append(proto_idx)
            if not cands:
                break
            options.append(cands)
        else:
            small_types = set()
            for sprite, cands in zip(sprites, options, strict=False):
                if int(sprite["area"]) == 9:
                    small_types.update(int(v) for v in cands)
            if small_types != {0, 1}:
                continue

            for assignment in itertools.product(*options):
                if len(set(int(v) for v in assignment)) != 2:
                    continue
                placed = []
                for sprite, proto_idx in zip(sprites, assignment, strict=False):
                    rec = dict(sprite)
                    rec["proto"] = int(proto_idx)
                    placed.append(rec)
                for perm in itertools.permutations([int(s["color"]) for s in placed]):
                    rendered = _paint(g.shape, placed, list(perm), protos)
                    if np.array_equal(rendered, g):
                        solutions.append({"sprites": placed, "order": list(perm), "protos": protos})
    return solutions


def _solve_scene(grid: np.ndarray) -> dict:
    sols = _candidate_solutions(grid)
    if len(sols) != 1:
        raise AssertionError(f"expected exactly one scene solution, found {len(sols)}")
    return sols[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every scaled sprite to the opposite prototype, then output the unique bottom-to-top color order.

    Input structure:
    - The background is black.
    - Every non-black color belongs to exactly one monochrome sprite with a 3x3 prototype scaled by a factor of 1..3.
    - There are exactly two prototype shapes, and the visible input is produced by drawing the sprites bottom-to-top so higher sprites occlude lower ones.

    Rule:
    - Determine each sprite's bounding box, scale, color, and which of the two prototypes it uses.
    - Toggle every sprite to the other prototype while keeping the same scale and top-left anchor.
    - Redraw the toggled sprites in the same layer order, infer the observable pairwise above/below constraints from the new occlusions, and return the unique bottom-to-top color order as an N x 1 column.

    Key insight:
    - The colors identify sprite instances, but the layer order becomes readable only after reconstructing the hidden full sprites and toggling their prototypes, because the overlap graph changes after the swap.
    """
    solution = _solve_scene(np.asarray(grid, dtype=int))
    sprites = [dict(s) for s in solution["sprites"]]
    protos = solution["protos"]
    order = [int(c) for c in solution["order"]]

    swapped_sprites = []
    for sprite in sprites:
        rec = dict(sprite)
        rec["proto"] = 1 - int(sprite["proto"])
        swapped_sprites.append(rec)

    swapped_canvas = _paint(np.asarray(grid).shape, swapped_sprites, order, protos)
    colors = [int(s["color"]) for s in swapped_sprites]
    masks = {int(s["color"]): _full_mask(swapped_canvas.shape, s, protos[int(s["proto"])]) for s in swapped_sprites}
    edges = _observable_edges(swapped_canvas, masks, colors)
    assert edges is not None, "swapped scene must have consistent observable occlusions"
    inferred = _unique_topo(colors, edges)
    assert inferred is not None, "swapped scene must determine a unique layer order"
    return np.array(inferred, dtype=int).reshape(-1, 1)


def _random_proto_pair() -> tuple[np.ndarray, np.ndarray]:
    ia, ib = random.sample(range(len(_PROTO_LIBRARY)), 2)
    return _PROTO_LIBRARY[int(ia)], _PROTO_LIBRARY[int(ib)]


def _coords_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, pt)) for pt in np.argwhere(mask)]


def _mask_cells(shape: tuple[int, int], top: int, left: int, patch: np.ndarray) -> set[tuple[int, int]]:
    cells = set()
    for rr, cc in _coords_from_mask(patch):
        r = int(top) + int(rr)
        c = int(left) + int(cc)
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            cells.add((int(r), int(c)))
    return cells


def _boundary_witnesses(top: int, left: int, patch: np.ndarray) -> set[tuple[int, int]]:
    pts = _coords_from_mask(patch)
    by_top = [p for p in pts if p[0] == 0]
    by_bottom = [p for p in pts if p[0] == patch.shape[0] - 1]
    by_left = [p for p in pts if p[1] == 0]
    by_right = [p for p in pts if p[1] == patch.shape[1] - 1]
    picks = [
        random.choice(by_top),
        random.choice(by_bottom),
        random.choice(by_left),
        random.choice(by_right),
    ]
    return {(int(top) + int(r), int(left) + int(c)) for r, c in picks}


def _position_score(shape: tuple[int, int], top: int, left: int, size: int) -> float:
    center_r = (shape[0] - size) / 2.0
    center_c = (shape[1] - size) / 2.0
    return abs(top - center_r) + abs(left - center_c) + random.random() * 0.5


def _construct_scene(shape: tuple[int, int], protos: tuple[np.ndarray, np.ndarray]) -> tuple[list[dict], list[int]] | None:
    h, w = shape
    n = random.choice([3, 4])
    if n == 3:
        scales = [random.choice([2, 3]), 1, 1]
    else:
        scales = [random.choice([2, 3]), random.choice([2, 3]), 1, 1]

    types = [random.randint(0, 1) for _ in range(n)]
    types[-2] = 0
    types[-1] = 1
    if 0 not in types or 1 not in types:
        types[0] = 0
        types[1] = 1

    colors = random.sample(list(range(1, 10)), n)
    specs = []
    for i in range(n):
        specs.append(
            {
                "color": int(colors[i]),
                "scale": int(scales[i]),
                "proto": int(types[i]),
            }
        )

    scaled_patches = [_scale_mask(protos[int(spec["proto"])], int(spec["scale"])) for spec in specs]
    masks: list[set[tuple[int, int]]] = []
    sprites: list[dict] = []
    protected: set[tuple[int, int]] = set()

    def rec(idx: int) -> bool:
        if idx == n:
            return True
        patch = scaled_patches[idx]
        size = patch.shape[0]
        all_positions = [(top, left) for top in range(h - size + 1) for left in range(w - size + 1)]
        all_positions.sort(key=lambda pos: _position_score(shape, pos[0], pos[1], size))

        if idx > 0:
            prev_cells = masks[idx - 1]
            scored = []
            for top, left in all_positions:
                cells = _mask_cells(shape, top, left, patch)
                overlap = len(cells & prev_cells)
                if overlap == 0:
                    continue
                scored.append((-overlap, random.random(), top, left))
            all_positions = [(top, left) for _ov, _noise, top, left in sorted(scored)]

        for top, left in all_positions[:180]:
            cells = _mask_cells(shape, top, left, patch)
            if cells & protected:
                continue

            new_protected = set(_boundary_witnesses(top, left, patch))
            if idx > 0:
                overlap_cells = list(cells & masks[idx - 1])
                if not overlap_cells:
                    continue
                witness_choices = [cell for cell in overlap_cells if cell not in protected]
                if not witness_choices:
                    continue
                new_protected.add(random.choice(witness_choices))

            sprite = dict(specs[idx])
            sprite["top"] = int(top)
            sprite["left"] = int(left)
            sprites.append(sprite)
            masks.append(cells)
            old_protected = protected.copy()
            protected.update(new_protected)
            if rec(idx + 1):
                return True
            protected.clear()
            protected.update(old_protected)
            sprites.pop()
            masks.pop()
        return False

    if not rec(0):
        return None
    order = [int(spec["color"]) for spec in specs]
    return sprites, order


def _visible_bboxes_preserved(grid: np.ndarray, sprites: list[dict], protos: tuple[np.ndarray, np.ndarray]) -> bool:
    for sprite in sprites:
        color = int(sprite["color"])
        vis = grid == color
        box = _bbox(vis)
        if box is None:
            return False
        size = 3 * int(sprite["scale"])
        expected = (int(sprite["top"]), int(sprite["left"]), int(sprite["top"]) + size - 1, int(sprite["left"]) + size - 1)
        if box != expected:
            return False
    return True


def _ground_truth_ok(shape: tuple[int, int], sprites: list[dict], order: list[int], protos: tuple[np.ndarray, np.ndarray]) -> bool:
    colors = [int(s["color"]) for s in sprites]
    orig_canvas = _paint(shape, sprites, order, protos)
    if not _visible_bboxes_preserved(orig_canvas, sprites, protos):
        return False

    orig_masks = {int(s["color"]): _full_mask(shape, s, protos[int(s["proto"])]) for s in sprites}
    orig_edges = _observable_edges(orig_canvas, orig_masks, colors)
    if orig_edges is None:
        return False
    if _unique_topo(colors, orig_edges) != order:
        return False

    swapped = []
    for sprite in sprites:
        rec = dict(sprite)
        rec["proto"] = 1 - int(rec["proto"])
        swapped.append(rec)
    swapped_canvas = _paint(shape, swapped, order, protos)
    swapped_masks = {int(s["color"]): _full_mask(shape, s, protos[int(s["proto"])]) for s in swapped}
    swapped_edges = _observable_edges(swapped_canvas, swapped_masks, colors)
    if swapped_edges is None:
        return False
    if _unique_topo(colors, swapped_edges) != order:
        return False

    if _overlap_pairs(orig_masks, colors) == _overlap_pairs(swapped_masks, colors):
        return False

    solution_count = len(_candidate_solutions(orig_canvas))
    if solution_count != 1:
        return False

    expected = np.array(order, dtype=int).reshape(-1, 1)
    try:
        produced = main(orig_canvas)
    except Exception:
        return False
    return np.array_equal(produced, expected)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(_GRID_MIN, _GRID_MAX)
    w = random.randint(_GRID_MIN, _GRID_MAX)
    protos = _random_proto_pair()
    built = _construct_scene((h, w), protos)
    if built is None:
        return None
    sprites, order = built
    if not _ground_truth_ok((h, w), sprites, order, protos):
        return None
    return _paint((h, w), sprites, order, protos)


def generate_input() -> np.ndarray:
    for _ in range(_GEN_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
