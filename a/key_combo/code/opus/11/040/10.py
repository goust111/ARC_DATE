from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
BAR = 5
SPRITE_COLORS = (1, 2, 3, 4)
SIDES = ("top", "bottom", "left", "right")
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _extract_patch(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    bbox = _bbox_from_coords(coords)
    patch = _crop_bbox(grid, bbox)
    rel = coords - np.array([bbox[0], bbox[1]])
    out = np.zeros_like(patch)
    out[rel[:, 0], rel[:, 1]] = patch[rel[:, 0], rel[:, 1]]
    return out


def _unique_rotations(patch: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    cur = np.array(patch, dtype=int)
    for _ in range(4):
        key = (cur.tobytes(), tuple(map(int, cur.shape)))
        if key not in seen:
            seen.add(key)
            out.append(cur.copy())
        cur = np.rot90(cur)
    return out


def _connected_mask(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    return len(_component_coords(mask)) == 1


def _bar_component(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    comps = _component_coords(grid == BAR)
    assert comps, "expected a grey bar"
    comps.sort(key=len, reverse=True)
    bar = comps[0]
    return bar, _bbox_from_coords(bar)


def _extract_notches(grid: np.ndarray, bar_bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = bar_bbox
    local = _crop_bbox(grid, bar_bbox) == BG
    notches = []
    for coords in _component_coords(local):
        touches = {
            side
            for side, active in (
                ("top", np.any(coords[:, 0] == 0)),
                ("bottom", np.any(coords[:, 0] == local.shape[0] - 1)),
                ("left", np.any(coords[:, 1] == 0)),
                ("right", np.any(coords[:, 1] == local.shape[1] - 1)),
            )
            if active
        }
        if not touches:
            continue
        assert len(touches) == 1, f"notch must open to exactly one side, got {touches}"
        side = next(iter(touches))
        bb = _bbox_from_coords(coords)
        gbb = (r0 + bb[0], c0 + bb[1], r0 + bb[2], c0 + bb[3])
        notches.append(
            {
                "side": side,
                "bbox": gbb,
                "shape": (int(bb[2] - bb[0] + 1), int(bb[3] - bb[1] + 1)),
            }
        )
    notches.sort(key=lambda d: (d["bbox"][0], d["bbox"][1], d["side"]))
    assert 1 <= len(notches) <= 2, "expected 1-2 notches"
    return notches


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    sprites = []
    fg_mask = (grid != BG) & (grid != BAR)
    for coords in _component_coords(fg_mask):
        colors = {int(v) for v in grid[coords[:, 0], coords[:, 1]]}
        assert len(colors) == 1, "sprites must be monochrome"
        color = next(iter(colors))
        sprites.append({"color": color, "patch": _extract_patch(grid, coords)})
    sprites.sort(key=lambda d: (d["patch"].shape[0], d["patch"].shape[1], d["color"]))
    return sprites


def _match_notches_to_sprites(notches: list[dict], sprites: list[dict]) -> list[tuple[int, np.ndarray]]:
    options: list[list[tuple[int, np.ndarray]]] = []
    for notch in notches:
        nh, nw = notch["shape"]
        cur = []
        for si, sprite in enumerate(sprites):
            for rot in _unique_rotations(sprite["patch"]):
                if rot.shape == (nh, nw):
                    cur.append((si, rot))
        assert cur, f"no sprite matches notch {notch}"
        options.append(cur)

    solutions: list[list[tuple[int, np.ndarray]]] = []

    def rec(i: int, used: set[int], chosen: list[tuple[int, np.ndarray]]) -> None:
        if i == len(notches):
            solutions.append([(si, rot.copy()) for si, rot in chosen])
            return
        for si, rot in options[i]:
            if si in used:
                continue
            used.add(si)
            chosen.append((si, rot))
            rec(i + 1, used, chosen)
            chosen.pop()
            used.remove(si)

    rec(0, set(), [])
    assert len(solutions) == 1, f"sprite matching must be unique, got {len(solutions)}"
    return solutions[0]


def _stretch_patch(root: np.ndarray, side: str, k: int) -> np.ndarray:
    assert k >= 1
    if side == "down":
        return np.vstack([root, np.repeat(root[-1:, :], k, axis=0)])
    if side == "up":
        return np.vstack([np.repeat(root[:1, :], k, axis=0), root])
    if side == "right":
        return np.hstack([root, np.repeat(root[:, -1:], k, axis=1)])
    if side == "left":
        return np.hstack([np.repeat(root[:, :1], k, axis=1), root])
    raise ValueError(side)


def _notch_growth_side(notch_side: str) -> str:
    return {"top": "up", "bottom": "down", "left": "left", "right": "right"}[notch_side]


def _paint_patch_clipped(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> tuple[bool, bool]:
    h, w = patch.shape
    H, W = out.shape
    r0 = max(0, top)
    c0 = max(0, left)
    r1 = min(H, top + h)
    c1 = min(W, left + w)
    if r0 >= r1 or c0 >= c1:
        return False, False
    pr0 = r0 - top
    pc0 = c0 - left
    pr1 = pr0 + (r1 - r0)
    pc1 = pc0 + (c1 - c0)
    clip = patch[pr0:pr1, pc0:pc1]
    mask = clip != BG
    out[r0:r1, c0:c1][mask] = clip[mask]
    fully_inside = top >= 0 and left >= 0 and top + h <= H and left + w <= W
    return True, fully_inside


def _grow_chain(out: np.ndarray, root: np.ndarray, notch_bbox: tuple[int, int, int, int], notch_side: str) -> None:
    r0, c0, r1, c1 = notch_bbox
    _paint_patch_clipped(out, root, r0, c0)
    growth = _notch_growth_side(notch_side)
    prev_top, prev_left = r0, c0
    prev_h, prev_w = root.shape
    k = 1
    while True:
        patch = _stretch_patch(root, growth, k)
        if growth == "down":
            top, left = prev_top + prev_h, c0
        elif growth == "up":
            top, left = prev_top - patch.shape[0], c0
        elif growth == "right":
            top, left = r0, prev_left + prev_w
        else:
            top, left = r0, prev_left - patch.shape[1]
        painted, fully_inside = _paint_patch_clipped(out, patch, top, left)
        if not painted:
            break
        prev_top, prev_left = top, left
        prev_h, prev_w = patch.shape
        if not fully_inside:
            break
        k += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered sprites to bar notches, then grow stretched chains outward in the notch directions.

    Input structure:
    - Black background with one large grey rectangular bar near the center.
    - The bar contains 1-2 rectangular boundary notches cut out as black cells.
    - Several scattered monochrome sprites appear away from the bar; some match notch bbox sizes up to rotation and others are decoys.

    Rule:
    - Detect each notch and find the unique scattered sprite whose rotated bbox shape matches that notch.
    - Rotate that sprite to the notch orientation and place it into the notch as ring 0, keeping its color.
    - Use the notch opening direction as the chain direction; for ring k>=1, stretch the ring-0 sprite by k cells along that direction by repeating the outward-facing border row/column, then place the new ring immediately outside the previous one.
    - Continue until a ring would exceed the grid; paint any in-bounds clipped part of that last ring. Remove all unmatched decoy sprites and keep the grey bar unchanged.

    Key insight:
    - The notch determines both which sprite is selected and which side of that sprite becomes the outward-growing stretched chain.
    """
    g = np.asarray(grid, dtype=int)
    out = np.where(g == BAR, BAR, BG)
    _bar_coords, bar_bbox = _bar_component(g)
    notches = _extract_notches(g, bar_bbox)
    sprites = _extract_sprites(g)
    matches = _match_notches_to_sprites(notches, sprites)
    for notch, (_si, root) in zip(notches, matches):
        _grow_chain(out, root, notch["bbox"], notch["side"])
    return out


def _actual_notch_shape(size_class: tuple[int, int], side: str) -> tuple[int, int]:
    a, b = size_class
    if side in ("top", "bottom"):
        return a, b
    return b, a


def _chain_requirement(actual_shape: tuple[int, int], side: str) -> int:
    dim = actual_shape[0] if side in ("top", "bottom") else actual_shape[1]
    return sum(dim + k for k in range(1, 4))


def _random_mask_exact_bbox(h: int, w: int, rng: random.Random, irregular_only: bool) -> np.ndarray | None:
    if h == 1 or w == 1:
        return np.ones((h, w), dtype=bool)
    area_min = max(3, (h * w + 1) // 2)
    for _ in range(120):
        area = rng.randint(area_min, h * w)
        cells = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(cells)
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells[:area]:
            mask[r, c] = True
        if not _connected_mask(mask):
            continue
        rr, cc = np.argwhere(mask).T
        if rr.min() != 0 or rr.max() != h - 1 or cc.min() != 0 or cc.max() != w - 1:
            continue
        if irregular_only and mask.all():
            continue
        return mask
    return None


def _build_output_from_chains(
    H: int,
    W: int,
    bar_bbox: tuple[int, int, int, int],
    notch_specs: list[dict],
) -> np.ndarray | None:
    out = np.zeros((H, W), dtype=int)
    r0, c0, r1, c1 = bar_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = BAR
    for spec in notch_specs:
        nr0, nc0, nr1, nc1 = spec["bbox"]
        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = BG
    base = out.copy()

    occupied = set()
    for spec in notch_specs:
        nr0, nc0, nr1, nc1 = spec["bbox"]
        root = _mask_to_patch(spec["oriented_mask"], spec["color"])
        temp = np.zeros_like(out)
        _grow_chain(temp, root, spec["bbox"], spec["side"])
        chain_cells = {(int(r), int(c)) for r, c in np.argwhere(temp != BG)}
        if any(base[r, c] == BAR and (r < nr0 or r > nr1 or c < nc0 or c > nc1) for r, c in chain_cells):
            return None
        if chain_cells & occupied:
            return None
        occupied |= chain_cells
        out[temp != BG] = temp[temp != BG]
    return out


def _pick_bar_geometry(
    rng: random.Random,
    notch_sides: list[str],
    notch_classes: list[tuple[int, int]],
) -> tuple[int, int, tuple[int, int, int, int], list[dict]] | None:
    req = {side: 1 for side in SIDES}
    actual_shapes = []
    for side, size_class in zip(notch_sides, notch_classes):
        actual = _actual_notch_shape(size_class, side)
        actual_shapes.append(actual)
        req[side] = max(req[side], _chain_requirement(actual, side))

    if any(side in ("left", "right") and actual[0] == 3 for side, actual in zip(notch_sides, actual_shapes)):
        short_min = 5
    elif any(side in ("left", "right") and actual[0] == 2 for side, actual in zip(notch_sides, actual_shapes)):
        short_min = 4
    elif any(side in ("top", "bottom") and actual[1] == 3 for side, actual in zip(notch_sides, actual_shapes)):
        short_min = 5
    else:
        short_min = 4

    orientation = rng.choice(["horizontal", "vertical"])
    if orientation == "horizontal":
        bar_h = rng.randint(short_min, 5)
        bar_w = rng.randint(9, 12)
    else:
        bar_h = rng.randint(9, 12)
        bar_w = rng.randint(short_min, 5)

    H_min = max(22, req["top"] + bar_h + req["bottom"])
    W_min = max(22, req["left"] + bar_w + req["right"])
    if H_min > 30 or W_min > 30:
        return None
    H = rng.randint(H_min, 30)
    W = rng.randint(W_min, 30)

    r0_lo, r0_hi = req["top"], H - bar_h - req["bottom"]
    c0_lo, c0_hi = req["left"], W - bar_w - req["right"]
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None
    r0 = rng.randint(r0_lo, r0_hi)
    c0 = rng.randint(c0_lo, c0_hi)
    bar_bbox = (r0, c0, r0 + bar_h - 1, c0 + bar_w - 1)

    notch_specs = []
    for side, size_class, actual in zip(notch_sides, notch_classes, actual_shapes):
        nh, nw = actual
        if side in ("top", "bottom"):
            if bar_w - nw - 2 < 0:
                return None
            rel_c = rng.randint(1, bar_w - nw - 1)
            nr0 = r0 if side == "top" else r0 + bar_h - nh
            nc0 = c0 + rel_c
        else:
            if bar_h - nh - 2 < 0:
                return None
            rel_r = rng.randint(1, bar_h - nh - 1)
            nr0 = r0 + rel_r
            nc0 = c0 if side == "left" else c0 + bar_w - nw
        notch_specs.append(
            {
                "side": side,
                "bbox": (nr0, nc0, nr0 + nh - 1, nc0 + nw - 1),
                "size_class": size_class,
                "shape": actual,
            }
        )
    return H, W, bar_bbox, notch_specs


def _random_root_masks(
    rng: random.Random,
    notch_specs: list[dict],
) -> bool:
    irregular_present = False
    for spec in notch_specs:
        h, w = spec["shape"]
        irregular_needed = min(h, w) > 1
        mask = _random_mask_exact_bbox(h, w, rng, irregular_only=irregular_needed)
        if mask is None:
            return False
        irregular_present |= not mask.all()
        spec["oriented_mask"] = mask
    return irregular_present


def _random_color_order(rng: random.Random, count: int) -> list[int]:
    colors = list(SPRITE_COLORS)
    rng.shuffle(colors)
    return colors[:count]


def _decoy_shape(
    rng: random.Random,
    forbidden_classes: set[tuple[int, int]],
) -> np.ndarray | None:
    candidates = [(2, 4), (3, 3), (4, 2), (2, 5), (5, 2)]
    rng.shuffle(candidates)
    for h, w in candidates:
        cls = tuple(sorted((h, w)))
        if cls in forbidden_classes:
            continue
        mask = _random_mask_exact_bbox(h, w, rng, irregular_only=False)
        if mask is not None and 4 <= int(mask.sum()) <= 8:
            return mask
    return None


def _can_place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    margin: int = 1,
) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    for dr, dc in np.argwhere(patch != BG):
        rr = top + int(dr)
        cc = left + int(dc)
        r0 = max(0, rr - margin)
        c0 = max(0, cc - margin)
        r1 = min(H, rr + margin + 1)
        c1 = min(W, cc + margin + 1)
        if np.any(grid[r0:r1, c0:c1] != BG):
            return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    target = grid[top : top + h, left : left + w]
    mask = patch != BG
    target[mask] = patch[mask]


def _scatter_sprites(
    rng: random.Random,
    H: int,
    W: int,
    bar_bbox: tuple[int, int, int, int],
    roots: list[dict],
    decoys: list[dict],
) -> np.ndarray | None:
    grid = np.zeros((H, W), dtype=int)
    r0, c0, r1, c1 = bar_bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = BAR
    for root in roots:
        nr0, nc0, nr1, nc1 = root["bbox"]
        grid[nr0 : nr1 + 1, nc0 : nc1 + 1] = BG

    sprites = roots + decoys
    rng.shuffle(sprites)
    for sprite in sprites:
        rotations = _unique_rotations(_mask_to_patch(sprite["mask"], sprite["color"]))
        patch = rng.choice(rotations).copy()
        spots = [
            (rr, cc)
            for rr in range(H - patch.shape[0] + 1)
            for cc in range(W - patch.shape[1] + 1)
            if not (
                rr <= r1 + 1
                and rr + patch.shape[0] - 1 >= r0 - 1
                and cc <= c1 + 1
                and cc + patch.shape[1] - 1 >= c0 - 1
            )
        ]
        rng.shuffle(spots)
        placed = False
        for top, left in spots:
            if _can_place_patch(grid, patch, top, left, margin=1):
                _place_patch(grid, patch, top, left)
                placed = True
                break
        if not placed:
            return None
    return grid


def _validate_generation(inp: np.ndarray, expected: np.ndarray, notch_count: int) -> None:
    out = main(inp)
    assert np.array_equal(out, expected), "solver mismatch"
    notches = _extract_notches(inp, _bar_component(inp)[1])
    sprites = _extract_sprites(inp)
    matches = _match_notches_to_sprites(notches, sprites)
    assert len(matches) == notch_count
    assert len(sprites) > notch_count, "must include decoy sprites"
    assert any(np.any(rot == BG) for _si, rot in matches if min(rot.shape) > 1), "need a non-full matched root"
    for notch, (_si, rot) in zip(notches, matches):
        growth = _notch_growth_side(notch["side"])
        if growth in ("up", "down"):
            dim = rot.shape[0]
        else:
            dim = rot.shape[1]
        assert sum(dim + k for k in range(1, 4)) >= 9


def generate_input() -> np.ndarray:
    rng = random.Random()
    size_pool = [(2, 2), (2, 3), (1, 2), (1, 3)]
    for _ in range(GENERATION_ATTEMPTS):
        notch_count = rng.choice([1, 2])
        notch_sides = rng.sample(list(SIDES), notch_count)
        if notch_count == 1:
            notch_classes = [rng.choice([(2, 2), (2, 3)])]
        else:
            first = rng.choice([(2, 2), (2, 3)])
            rest = [s for s in size_pool if tuple(sorted(s)) != tuple(sorted(first))]
            second = rng.choice(rest)
            notch_classes = [first, second]
            rng.shuffle(notch_classes)

        geom = _pick_bar_geometry(rng, notch_sides, notch_classes)
        if geom is None:
            continue
        H, W, bar_bbox, notch_specs = geom
        if not _random_root_masks(rng, notch_specs):
            continue

        total_sprites = notch_count + rng.choice([1, 2] if notch_count == 1 else [1, 2])
        colors = _random_color_order(rng, total_sprites)
        roots = []
        for spec, color in zip(notch_specs, colors[:notch_count]):
            roots.append(
                {
                    "bbox": spec["bbox"],
                    "side": spec["side"],
                    "mask": spec["oriented_mask"],
                    "color": color,
                }
            )
            spec["color"] = color

        used_classes = {tuple(sorted(spec["shape"])) for spec in notch_specs}
        decoys = []
        ok = True
        for color in colors[notch_count:]:
            mask = _decoy_shape(rng, used_classes)
            if mask is None:
                ok = False
                break
            used_classes.add(tuple(sorted(mask.shape)))
            decoys.append({"mask": mask, "color": color})
        if not ok:
            continue

        expected = _build_output_from_chains(H, W, bar_bbox, notch_specs)
        if expected is None:
            continue
        inp = _scatter_sprites(rng, H, W, bar_bbox, roots, decoys)
        if inp is None:
            continue
        try:
            _validate_generation(inp, expected, notch_count)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
