import random
from collections import deque

import numpy as np


BG = 0
GEN_RETRIES = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _make_tabbed_shape(body_h: int, body_w: int, side: int, offset: int) -> dict:
    if side == 0:
        mask = np.zeros((body_h + 1, body_w), dtype=bool)
        mask[1:, :] = True
        mask[0, offset] = True
        body_origin = (1, 0)
        tab = (0, offset)
    elif side == 1:
        mask = np.zeros((body_h, body_w + 1), dtype=bool)
        mask[:, :body_w] = True
        mask[offset, body_w] = True
        body_origin = (0, 0)
        tab = (offset, body_w)
    elif side == 2:
        mask = np.zeros((body_h + 1, body_w), dtype=bool)
        mask[:body_h, :] = True
        mask[body_h, offset] = True
        body_origin = (0, 0)
        tab = (body_h, offset)
    else:
        mask = np.zeros((body_h, body_w + 1), dtype=bool)
        mask[:, 1:] = True
        mask[offset, 0] = True
        body_origin = (0, 1)
        tab = (offset, 0)
    return {
        "mask": mask,
        "body_h": int(body_h),
        "body_w": int(body_w),
        "body_origin": tuple(map(int, body_origin)),
        "tab": tuple(map(int, tab)),
    }


def _random_tabbed_shape(
    rng: random.Random,
    min_body_h: int,
    max_body_h: int,
    min_body_w: int,
    max_body_w: int,
) -> dict:
    body_h = rng.randint(min_body_h, max_body_h)
    body_w = rng.randint(min_body_w, max_body_w)
    side = rng.randrange(4)
    span = body_w if side in (0, 2) else body_h
    offset = rng.randint(1, span - 2)
    return _make_tabbed_shape(body_h, body_w, side, offset)


def _pack_shapes(body_h: int, body_w: int, shapes: list[dict], rng: random.Random):
    if not shapes:
        return []
    order = sorted(
        range(len(shapes)),
        key=lambda i: shapes[i]["mask"].shape[0] * shapes[i]["mask"].shape[1],
        reverse=True,
    )
    blocked = np.zeros((body_h, body_w), dtype=bool)
    pos: list[tuple[int, int] | None] = [None] * len(shapes)
    for idx in order:
        mask = shapes[idx]["mask"]
        hh, ww = map(int, mask.shape)
        candidates = []
        if body_h - hh < 2 or body_w - ww < 2:
            return None
        for r in range(1, body_h - hh):
            for c in range(1, body_w - ww):
                if blocked[r - 1 : r + hh + 1, c - 1 : c + ww + 1].any():
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        pos[idx] = rng.choice(candidates)
        r, c = pos[idx]
        blocked[r - 1 : r + hh + 1, c - 1 : c + ww + 1] = True
    return [(int(r), int(c)) for r, c in pos]


def _build_piece(
    rng: random.Random,
    color: int,
    children: list[dict],
    min_body_h: int,
    max_body_h: int,
    min_body_w: int,
    max_body_w: int,
) -> dict | None:
    child_shapes = [child["shape"] for child in children]
    for _ in range(300):
        outer = _random_tabbed_shape(rng, min_body_h, max_body_h, min_body_w, max_body_w)
        positions = _pack_shapes(outer["body_h"], outer["body_w"], child_shapes, rng)
        if positions is None:
            continue
        br, bc = outer["body_origin"]
        holes = []
        for child, (r, c) in zip(children, positions):
            holes.append(
                {
                    "child": child,
                    "child_color": int(child["color"]),
                    "shape": child["shape"],
                    "pos": (int(br + r), int(bc + c)),
                }
            )
        return {"color": int(color), "shape": outer, "holes": holes}
    return None


def _render_unassembled(piece: dict) -> np.ndarray:
    mask = piece["shape"]["mask"]
    out = np.full(mask.shape, BG, dtype=int)
    out[mask] = int(piece["color"])
    for hole in piece["holes"]:
        hm = hole["shape"]["mask"]
        hr, hc = hole["pos"]
        sub = out[hr : hr + hm.shape[0], hc : hc + hm.shape[1]]
        sub[hm] = BG
        mr, mc = hole["shape"]["tab"]
        sub[mr, mc] = int(hole["child_color"])
    return out


def _render_assembled(piece: dict) -> np.ndarray:
    mask = piece["shape"]["mask"]
    out = np.full(mask.shape, BG, dtype=int)
    out[mask] = int(piece["color"])
    for hole in piece["holes"]:
        child_patch = _render_assembled(hole["child"])
        hm = hole["shape"]["mask"]
        assert child_patch.shape == hm.shape
        assert np.array_equal(child_patch != BG, hm)
        hr, hc = hole["pos"]
        sub = out[hr : hr + hm.shape[0], hc : hc + hm.shape[1]]
        occ = child_patch != BG
        sub[occ] = child_patch[occ]
    return out


def _find_objects(grid: np.ndarray) -> dict[int, dict]:
    objects: dict[int, dict] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        comps = _components(grid == color)
        if not comps:
            continue
        main_comp = max(comps, key=lambda arr: int(arr.shape[0]))
        if int(main_comp.shape[0]) <= 1:
            continue
        r0, c0, r1, c1 = _bbox(main_comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[main_comp[:, 0] - r0, main_comp[:, 1] - c0] = True
        objects[color] = {
            "color": int(color),
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "patch": patch,
            "mask": mask,
        }
    assert objects, "expected at least one real piece"
    return objects


def _extract_holes(piece: dict) -> list[dict]:
    patch = piece["patch"]
    mask = piece["mask"]
    holes = []
    for comp in _components(~mask):
        if any(
            int(r) == 0 or int(c) == 0 or int(r) == patch.shape[0] - 1 or int(c) == patch.shape[1] - 1
            for r, c in comp
        ):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        hole_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        hole_mask[comp[:, 0] - r0, comp[:, 1] - c0] = True
        vals = patch[r0 : r1 + 1, c0 : c1 + 1][hole_mask]
        marker_colors = {int(v) for v in vals if int(v) != BG}
        assert len(marker_colors) == 1, "each hole must contain exactly one marker color"
        child_color = next(iter(marker_colors))
        assert int(np.count_nonzero(vals == child_color)) == 1, "marker must be a single pixel"
        holes.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": hole_mask,
                "child_color": int(child_color),
            }
        )
    holes.sort(key=lambda h: (h["bbox"][0], h["bbox"][1]))
    return holes


def _rotate_match(child_patch: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
    matches = []
    seen = set()
    for k in range(4):
        rotated = np.rot90(child_patch, k)
        if rotated.shape != hole_mask.shape:
            continue
        if not np.array_equal(rotated != BG, hole_mask):
            continue
        key = rotated.tobytes()
        if key in seen:
            continue
        seen.add(key)
        matches.append(rotated)
    assert len(matches) == 1, "piece-to-hole rotation must be uniquely determined"
    return matches[0]


def _assemble(color: int, objects: dict[int, dict], cache: dict[int, np.ndarray]) -> np.ndarray:
    if color in cache:
        return cache[color]
    piece = objects[color]
    out = np.full(piece["patch"].shape, BG, dtype=int)
    out[piece["mask"]] = int(color)
    for hole in piece["holes"]:
        child_patch = _assemble(hole["child_color"], objects, cache)
        fitted = _rotate_match(child_patch, hole["mask"])
        r0, c0, r1, c1 = hole["bbox"]
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        occ = fitted != BG
        sub[occ] = fitted[occ]
    cache[color] = out
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    objects = _find_objects(g)
    for piece in objects.values():
        piece["holes"] = _extract_holes(piece)
    children = {hole["child_color"] for piece in objects.values() for hole in piece["holes"]}
    roots = sorted(set(objects) - children)
    assert len(roots) == 1, "there must be exactly one root frame"
    return _assemble(roots[0], objects, {})


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble color-linked nested pieces by matching each marked hole with the uniquely
    colored child piece whose outer contour fits after rotation.
    """
    return _solve(grid)


def _scatter_patches(rng: random.Random, patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    area = sum(int(p.shape[0] * p.shape[1]) for p in patches)
    min_side = max(max_h + 4, max_w + 4, int(np.ceil(np.sqrt(area * 2.2))))
    for _ in range(150):
        h = rng.randint(min_side, max(min_side, min_side + 6))
        w = rng.randint(min_side, max(min_side, min_side + 6))
        blocked = np.zeros((h, w), dtype=bool)
        grid = np.full((h, w), BG, dtype=int)
        ok = True
        for patch in sorted(patches, key=lambda p: int(p.shape[0] * p.shape[1]), reverse=True):
            ph, pw = map(int, patch.shape)
            if ph + 2 > h or pw + 2 > w:
                ok = False
                break
            candidates = []
            for r in range(1, h - ph):
                for c in range(1, w - pw):
                    if blocked[r - 1 : r + ph + 1, c - 1 : c + pw + 1].any():
                        continue
                    candidates.append((r, c))
            if not candidates:
                ok = False
                break
            r0, c0 = rng.choice(candidates)
            sub = grid[r0 : r0 + ph, c0 : c0 + pw]
            fg = patch != BG
            sub[fg] = patch[fg]
            blocked[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1] = True
        if ok:
            return grid
    return None


def _hole_patterns(count: int) -> list[tuple[int, ...]]:
    patterns = []
    ranges = [0, 1, 2]
    def rec(cur):
        if len(cur) == count:
            total = sum(cur)
            if 1 <= total <= 4:
                patterns.append(tuple(cur))
            return
        for v in ranges:
            rec(cur + [v])
    rec([])
    return patterns


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    medium_count = rng.randint(1, 3)
    hole_counts = rng.choice(_hole_patterns(medium_count))
    total_small = sum(hole_counts)
    total_pieces = 1 + medium_count + total_small
    if total_pieces > 9:
        return None

    colors = rng.sample(list(range(1, 10)), total_pieces)
    idx = 0

    small_pieces = []
    for _ in range(total_small):
        piece = _build_piece(rng, colors[idx], [], 3, 5, 3, 5)
        idx += 1
        if piece is None:
            return None
        small_pieces.append(piece)

    mediums = []
    cursor = 0
    for holes in hole_counts:
        children = small_pieces[cursor : cursor + holes]
        cursor += holes
        piece = _build_piece(rng, colors[idx], children, 6, 10, 6, 10)
        idx += 1
        if piece is None:
            return None
        mediums.append(piece)

    root = _build_piece(rng, colors[idx], mediums, 12, 17, 12, 17)
    if root is None:
        return None

    canonical_output = _render_assembled(root)
    root_rot = rng.randrange(4)
    expected = np.rot90(canonical_output, root_rot)

    patches = [np.rot90(_render_unassembled(root), root_rot)]
    for piece in mediums + small_pieces:
        patches.append(np.rot90(_render_unassembled(piece), rng.randrange(4)))
    grid = _scatter_patches(rng, patches)
    if grid is None:
        return None

    solved = _solve(grid)
    if not np.array_equal(solved, expected):
        return None

    if len(_find_objects(grid)) != total_pieces:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
