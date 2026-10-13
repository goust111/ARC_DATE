from __future__ import annotations

from collections import Counter, deque
import itertools
import random

import numpy as np


BG = 0
COLORS = list(range(1, 10))
GEN_ATTEMPTS = 400
PLACEMENT_PADDING = 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_true(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    total = int(mask.sum())
    return total > 0 and len(_components(mask)) == 1


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    return ~exterior


def _hole_masks(mask: np.ndarray) -> list[np.ndarray]:
    holes = _fill_holes(mask) & ~mask
    return [_crop_to_true(_coords_to_mask(coords, holes.shape)) for coords in _components(holes)]


def _coords_to_mask(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if coords.size:
        out[coords[:, 0], coords[:, 1]] = True
    return out


def _scale_mask(mask: np.ndarray, factor: int = 2) -> np.ndarray:
    return np.kron(mask.astype(np.uint8), np.ones((factor, factor), dtype=np.uint8)).astype(bool)


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    cropped = _crop_to_true(mask)
    coords = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(cropped)))
    return tuple(map(int, cropped.shape)), coords


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]:
                cells.append((r, c))
    return cells


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects: list[dict] = []
    for coords in _components(g != bg):
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        colors = {int(v) for v in patch[local]}
        assert len(colors) == 1, "each object must be monochrome"
        silhouette = _fill_holes(local)
        holes = _hole_masks(local)
        assert 1 <= len(holes) <= 4, "each object must contain one to four holes"
        objects.append(
            {
                "color": next(iter(colors)),
                "mask": local,
                "silhouette": silhouette,
                "holes": holes,
            }
        )
    assert len(objects) >= 2, "expected multiple objects"
    return objects


def _chain_order(objects: list[dict]) -> list[int]:
    n = len(objects)
    signatures: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], int] = {}
    for idx, obj in enumerate(objects):
        sig = _mask_signature(obj["silhouette"])
        assert sig not in signatures, "filled silhouettes must be unique"
        signatures[sig] = idx
    outgoing: dict[int, int] = {}
    incoming = [0] * n
    for idx, obj in enumerate(objects):
        hits = set()
        for hole in obj["holes"]:
            target = signatures.get(_mask_signature(_scale_mask(hole, 2)))
            if target is not None and target != idx:
                hits.add(int(target))
        assert len(hits) <= 1, "each object may encode at most one keyhole target"
        if hits:
            target = hits.pop()
            outgoing[idx] = target
            incoming[target] += 1
    starts = [idx for idx, deg in enumerate(incoming) if deg == 0]
    ends = [idx for idx in range(n) if idx not in outgoing]
    assert len(starts) == 1, "expected one outermost object"
    assert len(ends) == 1, "expected one innermost object"
    assert all(deg <= 1 for deg in incoming), "every object may be matched by at most one keyhole"

    chain: list[int] = []
    seen = set()
    cur = starts[0]
    while True:
        assert cur not in seen, "chain must be acyclic"
        chain.append(cur)
        seen.add(cur)
        if cur not in outgoing:
            break
        cur = outgoing[cur]
    assert len(chain) == n, "keyhole links must cover every object"
    return chain


def _render_output(objects: list[dict], chain: list[int]) -> np.ndarray:
    inner = objects[chain[-1]]
    size = len(inner["holes"])
    out = np.full((size, size), int(inner["color"]), dtype=int)
    for idx in range(len(chain) - 2, -1, -1):
        obj = objects[chain[idx]]
        thickness = len(obj["holes"])
        nxt = np.full((out.shape[0] + 2 * thickness, out.shape[1] + 2 * thickness), int(obj["color"]), dtype=int)
        nxt[thickness : thickness + out.shape[0], thickness : thickness + out.shape[1]] = out
        out = nxt
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn a chain of keyhole-coded holed blobs into concentric square layers.

    Input structure:
    - A black background contains several separated monochrome 4-connected objects.
    - Every object has one to four enclosed black holes; all non-key holes are 1x1.
    - For each non-innermost object, one hole is a keyhole whose shape scaled by 2 equals another object's filled outer silhouette.

    Rule:
    - Fill each object's holes to recover its outer silhouette, then match every scaled keyhole to the unique next object in the containment chain.
    - The unique object unmatched by any keyhole is the outermost layer; the unique object with no outgoing keyhole is the innermost layer.
    - Convert the innermost object into a solid square whose side length is its hole count, then wrap outward with one square frame per remaining object, using that object's color and total hole count as frame thickness.

    Key insight:
    - The same holes are read twice: once topologically to reconstruct the nesting order via exact 1:2 shape matching, and once arithmetically to set each concentric layer's thickness.
    """
    objects = _extract_objects(np.asarray(grid, dtype=int))
    chain = _chain_order(objects)
    return _render_output(objects, chain)


def _make_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    return _mask_signature(mask)


def _build_hole_library() -> list[np.ndarray]:
    seen = set()
    shapes: list[np.ndarray] = []
    cells = list(itertools.product(range(4), range(4)))
    for area in range(4, 7):
        for combo in itertools.combinations(cells, area):
            mask = np.zeros((4, 4), dtype=bool)
            for r, c in combo:
                mask[r, c] = True
            if not _is_connected(mask):
                continue
            cropped = _crop_to_true(mask)
            if min(cropped.shape) < 2:
                continue
            if len(_interior_cells(_scale_mask(cropped, 2))) == 0:
                continue
            key = _make_key(cropped)
            if key in seen:
                continue
            seen.add(key)
            shapes.append(cropped)
    shapes.sort(key=lambda item: (item.shape[0] * item.shape[1], int(item.sum()), item.shape[0], item.shape[1]))
    return shapes


HOLE_LIBRARY = _build_hole_library()
DEEP_HOLES = [mask for mask in HOLE_LIBRARY if len(_interior_cells(_scale_mask(mask, 2))) >= 4]


def _random_outer_silhouette() -> np.ndarray:
    for _ in range(80):
        h = random.randint(8, 11)
        w = random.randint(8, 11)
        mask = np.ones((h, w), dtype=bool)
        notch_count = random.randint(0, 2)
        for _ in range(notch_count):
            side = random.choice(["top", "bottom", "left", "right"])
            if side in {"top", "bottom"} and w >= 6:
                depth = random.randint(1, min(2, h - 4))
                span = random.randint(2, w - 2)
                start = random.randint(1, w - span - 1)
                rows = slice(0, depth) if side == "top" else slice(h - depth, h)
                mask[rows, start : start + span] = False
            if side in {"left", "right"} and h >= 6:
                depth = random.randint(1, min(2, w - 4))
                span = random.randint(2, h - 2)
                start = random.randint(1, h - span - 1)
                cols = slice(0, depth) if side == "left" else slice(w - depth, w)
                mask[start : start + span, cols] = False
        if _is_connected(mask) and np.array_equal(_fill_holes(mask), mask) and len(_interior_cells(mask)) >= 6:
            return mask
    raise RuntimeError("failed to build an outer silhouette")


def _hole_bboxes(mask: np.ndarray) -> list[np.ndarray]:
    return _hole_masks(mask)


def _valid_object(mask: np.ndarray, silhouette: np.ndarray, keyhole_shape: np.ndarray | None, expected_holes: int) -> bool:
    if not _is_connected(mask):
        return False
    if not np.array_equal(_fill_holes(mask), silhouette):
        return False
    holes = _hole_bboxes(mask)
    if len(holes) != expected_holes:
        return False
    if keyhole_shape is None:
        return all(int(h.sum()) == 1 for h in holes)
    seen_keyhole = 0
    target_key = _make_key(keyhole_shape)
    for hole in holes:
        if _make_key(hole) == target_key:
            seen_keyhole += 1
        elif int(hole.sum()) != 1:
            return False
    return seen_keyhole == 1


def _pick_singletons(base: np.ndarray, silhouette: np.ndarray, count: int, blocked: set[tuple[int, int]]) -> np.ndarray | None:
    if count == 0:
        return base if _valid_object(base, silhouette, None, 0) else None

    candidates = [(r, c) for r, c in _interior_cells(base) if (r, c) not in blocked]
    for _ in range(60):
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for r, c in candidates:
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in chosen):
                continue
            chosen.append((r, c))
            if len(chosen) == count:
                break
        if len(chosen) != count:
            continue
        trial = base.copy()
        for r, c in chosen:
            trial[r, c] = False
        if _valid_object(trial, silhouette, None, count):
            return trial
    return None


def _make_innermost_object(silhouette: np.ndarray, hole_count: int) -> np.ndarray | None:
    if hole_count < 1:
        return None
    return _pick_singletons(silhouette.copy(), silhouette, hole_count, set())


def _make_object_with_keyhole(silhouette: np.ndarray, keyhole_shape: np.ndarray, small_holes: int) -> np.ndarray | None:
    sh, sw = silhouette.shape
    kh, kw = keyhole_shape.shape
    positions = [(r, c) for r in range(sh - kh + 1) for c in range(sw - kw + 1)]
    random.shuffle(positions)
    target_holes = small_holes + 1
    for top, left in positions[:80]:
        window = silhouette[top : top + kh, left : left + kw]
        if not np.all(window[keyhole_shape]):
            continue
        base = silhouette.copy()
        hole_cells = [(top + r, left + c) for r, c in np.argwhere(keyhole_shape)]
        for r, c in hole_cells:
            base[r, c] = False
        if not np.array_equal(_fill_holes(base), silhouette):
            continue
        blocked = set()
        for r, c in hole_cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    blocked.add((r + dr, c + dc))
        if small_holes == 0:
            if _valid_object(base, silhouette, keyhole_shape, target_holes):
                return base
            continue
        candidates = [(r, c) for r, c in _interior_cells(base) if (r, c) not in blocked]
        for _ in range(60):
            random.shuffle(candidates)
            chosen: list[tuple[int, int]] = []
            for r, c in candidates:
                if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in chosen):
                    continue
                chosen.append((r, c))
                if len(chosen) == small_holes:
                    break
            if len(chosen) != small_holes:
                continue
            trial = base.copy()
            for r, c in chosen:
                trial[r, c] = False
            if _valid_object(trial, silhouette, keyhole_shape, target_holes):
                return trial
    return None


def _build_chain(levels: int, silhouette: np.ndarray) -> list[tuple[np.ndarray, int]] | None:
    if levels == 1:
        counts = [1, 2, 3, 4]
        random.shuffle(counts)
        for count in counts:
            obj = _make_innermost_object(silhouette, count)
            if obj is not None:
                return [(obj, count)]
        return None

    counts = [1, 2, 3, 4]
    random.shuffle(counts)
    pool = DEEP_HOLES if levels > 2 else HOLE_LIBRARY
    hole_shapes = random.sample(pool, k=min(len(pool), 40))
    for count in counts:
        small_holes = count - 1
        for keyhole_shape in hole_shapes:
            obj = _make_object_with_keyhole(silhouette, keyhole_shape, small_holes)
            if obj is None:
                continue
            rest = _build_chain(levels - 1, _scale_mask(keyhole_shape, 2))
            if rest is not None:
                return [(obj, count)] + rest
    return None


def _paint_object(dst: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    view = dst[top : top + h, left : left + w]
    view[mask] = int(color)


def _render_scene(objects: list[np.ndarray], colors: list[int]) -> np.ndarray | None:
    sizes = [mask.shape for mask in objects]
    total_box_area = sum((h + 2 * PLACEMENT_PADDING) * (w + 2 * PLACEMENT_PADDING) for h, w in sizes)
    base_side = max(max(h for h, _ in sizes), max(w for _, w in sizes), int(np.ceil(np.sqrt(total_box_area))) + 1)
    min_side = max(16, base_side + 1)
    max_side = min(30, min_side + 8)
    for _ in range(80):
        h = random.randint(min_side, max_side)
        w = random.randint(min_side, max_side)
        grid = np.full((h, w), BG, dtype=int)
        reserved = np.zeros((h, w), dtype=bool)
        placements: list[tuple[int, int]] = [(-1, -1)] * len(objects)
        order = list(range(len(objects)))
        random.shuffle(order)
        ok = True
        for idx in sorted(order, key=lambda item: objects[item].shape[0] * objects[item].shape[1], reverse=True):
            mask = objects[idx]
            mh, mw = mask.shape
            choices: list[tuple[int, int]] = []
            for top in range(0, h - mh + 1):
                for left in range(0, w - mw + 1):
                    r0 = max(0, top - PLACEMENT_PADDING)
                    c0 = max(0, left - PLACEMENT_PADDING)
                    r1 = min(h, top + mh + PLACEMENT_PADDING)
                    c1 = min(w, left + mw + PLACEMENT_PADDING)
                    if reserved[r0:r1, c0:c1].any():
                        continue
                    choices.append((top, left))
            if not choices:
                ok = False
                break
            top, left = random.choice(choices)
            placements[idx] = (top, left)
            r0 = max(0, top - PLACEMENT_PADDING)
            c0 = max(0, left - PLACEMENT_PADDING)
            r1 = min(h, top + mh + PLACEMENT_PADDING)
            c1 = min(w, left + mw + PLACEMENT_PADDING)
            reserved[r0:r1, c0:c1] = True
        if not ok:
            continue
        for idx, mask in enumerate(objects):
            top, left = placements[idx]
            _paint_object(grid, mask, colors[idx], top, left)
        return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    levels = random.choice([2, 3])
    chain = _build_chain(levels, _random_outer_silhouette())
    if chain is None:
        return None
    object_masks = [mask for mask, _count in chain]
    colors = random.sample(COLORS, levels)
    grid = _render_scene(object_masks, colors)
    if grid is None:
        return None
    expected_objects = [{"color": color, "holes": [None] * count} for color, (_mask, count) in zip(colors, chain)]
    expected = _render_output(expected_objects, list(range(levels)))
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid keyhole-chain input")
