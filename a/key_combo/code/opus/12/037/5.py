from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BAR = 5
MARKER = 7
BODY_COLORS = [1, 2, 3, 4]
OPENING_TO_INWARD = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
CANONICAL_RECT_PAIRS = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
MARKER_CORNERS_BY_INWARD = {
    "top": ["tl", "tr"],
    "bottom": ["bl", "br"],
    "left": ["tl", "bl"],
    "right": ["tr", "br"],
}


def _neighbors(r: int, c: int, shape: tuple[int, int], connectivity: int) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if connectivity == 4 and abs(dr) + abs(dc) != 1:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        pts = []
        while q:
            rr, cc = q.popleft()
            pts.append((rr, cc))
            for nr, nc in _neighbors(rr, cc, mask.shape, connectivity):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _marker_sides(marker_rc: tuple[int, int], shape: tuple[int, int]) -> set[str]:
    r, c = marker_rc
    h, w = shape
    sides = set()
    if r == 0:
        sides.add("top")
    if r == h - 1:
        sides.add("bottom")
    if c == 0:
        sides.add("left")
    if c == w - 1:
        sides.add("right")
    return sides


def _find_bar_and_notches(grid: np.ndarray) -> tuple[dict, list[dict]]:
    grey_comps = _components(grid == BAR, connectivity=4)
    assert grey_comps, "expected a central grey bar"
    bar_coords = max(grey_comps, key=len)
    r0, c0, r1, c1 = _bbox(bar_coords)
    bar_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    bar_mask[bar_coords[:, 0] - r0, bar_coords[:, 1] - c0] = True

    notches = []
    for pts in _components(~bar_mask, connectivity=4):
        sides = []
        if np.any(pts[:, 0] == 0):
            sides.append("top")
        if np.any(pts[:, 0] == bar_mask.shape[0] - 1):
            sides.append("bottom")
        if np.any(pts[:, 1] == 0):
            sides.append("left")
        if np.any(pts[:, 1] == bar_mask.shape[1] - 1):
            sides.append("right")
        assert len(sides) == 1, "each hole must open to exactly one bar side"
        local_box = _bbox(pts)
        local_mask = _mask_from_coords(pts)
        global_cells = [(r0 + int(rr), c0 + int(cc)) for rr, cc in pts]
        notches.append(
            {
                "opening": sides[0],
                "inward": OPENING_TO_INWARD[sides[0]],
                "bbox": (r0 + local_box[0], c0 + local_box[1], r0 + local_box[2], c0 + local_box[3]),
                "mask": local_mask,
                "cells": global_cells,
                "size_key": tuple(sorted(local_mask.shape)),
            }
        )
    notches.sort(key=lambda d: d["bbox"])
    return {"coords": bar_coords, "bbox": (r0, c0, r1, c1), "mask": bar_mask}, notches


def _find_sprites(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG) & (grid != BAR)
    sprites = []
    for coords in _components(mask, connectivity=8):
        vals = [int(grid[r, c]) for r, c in coords]
        assert vals.count(MARKER) == 1, "each sprite must have exactly one orange marker"
        body_colors = {v for v in vals if v != MARKER}
        assert len(body_colors) == 1, "each sprite body must be monochrome"
        body_color = next(iter(body_colors))
        box = _bbox(coords)
        sprites.append(
            {
                "coords": [(int(r), int(c)) for r, c in coords],
                "bbox": box,
                "patch": _crop(grid, box),
                "body_color": int(body_color),
            }
        )
    sprites.sort(key=lambda d: d["bbox"])
    return sprites


def _body_crop(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    return _crop(mask, _bbox(pts))


def _candidates_for_sprite(sprite: dict, notches: list[dict]) -> list[dict]:
    out = []
    for rot in range(4):
        patch = np.rot90(sprite["patch"], rot)
        marker_pts = np.argwhere(patch == MARKER)
        if len(marker_pts) != 1:
            continue
        marker_rc = tuple(int(x) for x in marker_pts[0])
        body_mask = patch == sprite["body_color"]
        if not np.any(body_mask):
            continue
        body_crop = _body_crop(body_mask)
        for notch_idx, notch in enumerate(notches):
            if body_crop.shape != notch["mask"].shape:
                continue
            if not np.array_equal(body_crop, notch["mask"]):
                continue
            out.append(
                {
                    "notch_idx": notch_idx,
                    "rot": rot,
                    "marker_ok": notch["inward"] in _marker_sides(marker_rc, patch.shape),
                }
            )
    return out


def _solve(grid: np.ndarray, *, strict: bool) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    bar, notches = _find_bar_and_notches(g)
    sprites = _find_sprites(g)
    if strict:
        assert len(notches) == len(sprites), "sprite/notch counts must match"

    sprite_matches = []
    used_notches = set()
    for sprite in sprites:
        candidates = _candidates_for_sprite(sprite, notches)
        valid = [c for c in candidates if c["marker_ok"]]
        if strict:
            assert len(valid) == 1, "marker side must uniquely choose one rotation"
            assert valid[0]["notch_idx"] not in used_notches, "each notch must be used once"
        if not valid:
            continue
        chosen = valid[0]
        used_notches.add(chosen["notch_idx"])
        sprite_matches.append((sprite, chosen, candidates))

    out = g.copy()
    for sprite, chosen, _cands in sprite_matches:
        for r, c in sprite["coords"]:
            out[r, c] = BG
        for r, c in notches[chosen["notch_idx"]]["cells"]:
            out[r, c] = sprite["body_color"]

    return out, {"bar": bar, "notches": notches, "sprites": sprites, "matches": sprite_matches}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate marked sprites so their orange corner points inward, then plug each one into its matching notch on the central bar.

    Input structure:
    - Black background with one large grey rectangular bar whose boundary contains 2-4 black notches.
    - Around the bar are equally many small monochrome sprites in colors 1-4, one color per sprite.
    - Each sprite also contains exactly one orange corner marker pixel that is not part of the fill shape.

    Rule:
    - Identify the grey bar and extract every boundary notch, including which bar side each notch opens toward.
    - For each scattered sprite, separate the orange marker from the functional body shape.
    - Match the body shape to exactly one notch by trying all 90-degree rotations while ignoring the marker.
    - If multiple rotations still fit the same notch, keep the unique one whose rotated orange marker lies on the side facing the bar interior, opposite the notch opening.
    - Erase the original sprite and fill the matched notch with the sprite body color, leaving the rest of the bar unchanged.

    Key insight:
    - Shape matching chooses the target notch, but symmetric shapes still admit multiple fitting rotations; the orange corner breaks that tie because only one rotated corner points inward toward the bar.
    """
    out, _info = _solve(grid, strict=True)
    return out


def _rectangle_mask(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _build_sprite_patch(body_mask: np.ndarray, body_color: int, marker_corner: str) -> np.ndarray:
    h, w = body_mask.shape
    patch = np.zeros((h + 1, w + 1), dtype=int)
    if marker_corner == "tl":
        pr, pc = 1, 1
        mr, mc = 0, 0
    elif marker_corner == "tr":
        pr, pc = 1, 0
        mr, mc = 0, w
    elif marker_corner == "bl":
        pr, pc = 0, 1
        mr, mc = h, 0
    elif marker_corner == "br":
        pr, pc = 0, 0
        mr, mc = h, w
    else:
        raise ValueError(marker_corner)
    patch[pr : pr + h, pc : pc + w][body_mask] = int(body_color)
    patch[mr, mc] = MARKER
    return patch


def _randint(lo: int, hi: int) -> int:
    return random.randint(int(lo), int(hi))


def _interval_ok(existing: list[tuple[int, int]], start: int, end: int) -> bool:
    for a, b in existing:
        if not (end < a - 1 or start > b + 1):
            return False
    return True


def _touches4(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        if (r - 1, c) in b or (r + 1, c) in b or (r, c - 1) in b or (r, c + 1) in b:
            return True
    return False


def _choose_notches(bar_h: int, bar_w: int, n_notches: int) -> list[dict] | None:
    pairs = random.sample(CANONICAL_RECT_PAIRS, n_notches)
    intervals = {"top": [], "bottom": [], "left": [], "right": []}
    used_cells: set[tuple[int, int]] = set()
    out = []
    for depth, width in pairs:
        placed = False
        side_order = random.sample(["top", "bottom", "left", "right"], 4)
        for side in side_order:
            if side in ("top", "bottom"):
                edge_len = bar_w
                cell_h, cell_w = depth, width
                if width > edge_len - 2 or depth >= bar_h:
                    continue
                starts = list(range(1, edge_len - width))
                random.shuffle(starts)
                for start in starts:
                    end = start + width - 1
                    if not _interval_ok(intervals[side], start, end):
                        continue
                    if side == "top":
                        cells = {(r, c) for r in range(depth) for c in range(start, start + width)}
                    else:
                        cells = {(bar_h - depth + r, c) for r in range(depth) for c in range(start, start + width)}
                    if cells & used_cells or _touches4(cells, used_cells):
                        continue
                    intervals[side].append((start, end))
                    used_cells |= cells
                    out.append({"opening": side, "depth": depth, "width": width, "cells": cells})
                    placed = True
                    break
            else:
                edge_len = bar_h
                cell_h, cell_w = width, depth
                if width > edge_len - 2 or depth >= bar_w:
                    continue
                starts = list(range(1, edge_len - width))
                random.shuffle(starts)
                for start in starts:
                    end = start + width - 1
                    if not _interval_ok(intervals[side], start, end):
                        continue
                    if side == "left":
                        cells = {(r, c) for r in range(start, start + width) for c in range(depth)}
                    else:
                        cells = {(r, bar_w - depth + c) for r in range(start, start + width) for c in range(depth)}
                    if cells & used_cells or _touches4(cells, used_cells):
                        continue
                    intervals[side].append((start, end))
                    used_cells |= cells
                    out.append({"opening": side, "depth": depth, "width": width, "cells": cells})
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None
    return out


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, padding: int = 1) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if top < 0 or left < 0 or top + h > gh or left + w > gw:
        return False
    coords = np.argwhere(patch != BG)
    for rr, cc in coords:
        gr = top + int(rr)
        gc = left + int(cc)
        if grid[gr, gc] != BG:
            return False
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < gh and 0 <= nc < gw and grid[nr, nc] != BG:
                    return False
    return True


def _place_patch_random(grid: np.ndarray, patch: np.ndarray) -> bool:
    gh, gw = grid.shape
    h, w = patch.shape
    spots = [(r, c) for r in range(gh - h + 1) for c in range(gw - w + 1)]
    random.shuffle(spots)
    for top, left in spots:
        if _can_place_patch(grid, patch, top, left, padding=1):
            mask = patch != BG
            grid[top : top + h, left : left + w][mask] = patch[mask]
            return True
    return False


def _semantic_ok(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        solved, info = _solve(inp, strict=True)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False
    if len(info["sprites"]) < 2:
        return False
    for _sprite, chosen, candidates in info["matches"]:
        shape_only = [c for c in candidates if c["notch_idx"] == chosen["notch_idx"]]
        if len(shape_only) != 2:
            return False
        if sum(int(c["marker_ok"]) for c in shape_only) != 1:
            return False
    return True


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    orientation = random.choice(["horizontal", "vertical"])
    H = _randint(20, 26)
    W = _randint(20, 26)

    if orientation == "horizontal":
        bar_h = _randint(5, 6)
        bar_w = _randint(12, min(16, W - 4))
        r0 = _randint(5, H - bar_h - 5)
        c0 = _randint(2, W - bar_w - 2)
    else:
        bar_w = _randint(5, 6)
        bar_h = _randint(12, min(16, H - 4))
        r0 = _randint(2, H - bar_h - 2)
        c0 = _randint(5, W - bar_w - 5)

    n_notches = random.choice([2, 3, 4])
    notch_specs = _choose_notches(bar_h, bar_w, n_notches)
    if notch_specs is None:
        return None

    body_colors = random.sample(BODY_COLORS, n_notches)
    inp = np.zeros((H, W), dtype=int)
    expected = np.zeros((H, W), dtype=int)
    inp[r0 : r0 + bar_h, c0 : c0 + bar_w] = BAR
    expected[r0 : r0 + bar_h, c0 : c0 + bar_w] = BAR

    scatter_patches = []
    seen_pairs = set()
    seen_sides = set()

    for spec, color in zip(notch_specs, body_colors):
        local_cells = spec["cells"]
        coords = np.array(sorted(local_cells), dtype=int)
        local_mask = _mask_from_coords(coords)
        size_key = tuple(sorted(local_mask.shape))
        if size_key in seen_pairs:
            return None
        seen_pairs.add(size_key)
        seen_sides.add(spec["opening"])
        for rr, cc in local_cells:
            inp[r0 + rr, c0 + cc] = BG
            expected[r0 + rr, c0 + cc] = int(color)

        inward = OPENING_TO_INWARD[spec["opening"]]
        marker_corner = random.choice(MARKER_CORNERS_BY_INWARD[inward])
        assembled_patch = _build_sprite_patch(local_mask, int(color), marker_corner)
        rot = random.choices([0, 1, 2, 3], weights=[1, 3, 3, 3])[0]
        scatter_patches.append(np.rot90(assembled_patch, rot))

    if len(seen_pairs) != n_notches:
        return None

    for patch in sorted(scatter_patches, key=lambda x: -int(np.sum(x != BG))):
        if not _place_patch_random(inp, patch):
            return None

    if not _semantic_ok(inp, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        out = _generate_attempt()
        if out is not None:
            return out[0]
    raise RuntimeError("failed to generate a valid input")
