from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, find_components, get_bg, neighbors4


SIDES = ("left", "right", "above", "below")
CONCAVE_ROWS = [
    (".#", "#.", ".#"),
    ("##", "#.", ".#"),
    (".#", "#.", "##"),
    ("#.#", ".#.", "..#"),
    ("###", ".#.", "..#"),
]
CONVEX_ROWS = [
    ("###",),
    ("####",),
    ("#####",),
    ("###", "###"),
    ("##", "##", "##"),
]
MAX_GEN_ATTEMPTS = 400


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _is_solid_2x2(comp: dict, grid: np.ndarray) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return (
        int(comp["area"]) == 4
        and int(r1 - r0 + 1) == 2
        and int(c1 - c0 + 1) == 2
        and bool((grid[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])).all())
    )


def _relation(sprite_bbox: tuple[int, int, int, int], anchor_bbox: tuple[int, int, int, int]):
    sr0, sc0, sr1, sc1 = map(int, sprite_bbox)
    ar0, ac0, ar1, ac1 = map(int, anchor_bbox)
    candidates: list[tuple[str, int, int]] = []
    if sc1 < ac0 and _intervals_overlap(sr0, sr1, ar0, ar1):
        candidates.append(("left", 0, ac0 - sc1 - 1))
    if sc0 > ac1 and _intervals_overlap(sr0, sr1, ar0, ar1):
        candidates.append(("right", 0, ac1 + 1 - sc0))
    if sr1 < ar0 and _intervals_overlap(sc0, sc1, ac0, ac1):
        candidates.append(("above", ar0 - sr1 - 1, 0))
    if sr0 > ar1 and _intervals_overlap(sc0, sc1, ac0, ac1):
        candidates.append(("below", ar1 + 1 - sr0, 0))
    if len(candidates) != 1:
        return None
    return candidates[0]


def _classify_objects(grid: np.ndarray):
    bg = int(get_bg(grid))
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=True)
    anchors = []
    sprites = []
    seeds = []
    for comp in comps:
        area = int(comp["area"])
        if area == 1:
            seeds.append(comp)
        elif _is_solid_2x2(comp, grid):
            anchors.append(comp)
        else:
            sprites.append(comp)
    assert 3 <= len(sprites) <= 4
    assert len(anchors) == len(sprites)
    assert 2 <= len(seeds) <= 3
    return bg, sprites, anchors, seeds


def _unique_pairing(sprites: list[dict], anchors: list[dict]):
    candidates: list[list[tuple[int, tuple[str, int, int]]]] = []
    for sprite in sprites:
        cur = []
        for anchor_idx, anchor in enumerate(anchors):
            rel = _relation(sprite["bbox"], anchor["bbox"])
            if rel is not None:
                cur.append((int(anchor_idx), rel))
        assert cur, "every sprite must have at least one anchor candidate"
        candidates.append(cur)

    order = sorted(range(len(sprites)), key=lambda idx: len(candidates[idx]))
    solutions: list[list[tuple[int, tuple[str, int, int]] | None]] = []
    current: list[tuple[int, tuple[str, int, int]] | None] = [None] * len(sprites)
    used: set[int] = set()

    def dfs(pos: int) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            solutions.append(current.copy())
            return
        sprite_idx = order[pos]
        for anchor_idx, rel in candidates[sprite_idx]:
            if anchor_idx in used:
                continue
            used.add(anchor_idx)
            current[sprite_idx] = (anchor_idx, rel)
            dfs(pos + 1)
            used.remove(anchor_idx)
            current[sprite_idx] = None

    dfs(0)
    assert len(solutions) == 1, "sprite-anchor matching must be unique"
    return solutions[0]


def _transform_coords(sprite: dict, anchor: dict, relation: tuple[str, int, int], shape: tuple[int, int]):
    side, dr, dc = relation
    H, W = map(int, shape)
    ac0 = int(anchor["bbox"][1])
    ac1 = int(anchor["bbox"][3])
    ar0 = int(anchor["bbox"][0])
    ar1 = int(anchor["bbox"][2])
    moved: list[tuple[int, int]] = []
    mirrored: list[tuple[int, int]] = []
    for r, c in sprite["coords"]:
        rr = int(r) + int(dr)
        cc = int(c) + int(dc)
        moved.append((rr, cc))
        if side == "left":
            mirrored.append((rr, 2 * ac0 - 1 - cc))
        elif side == "right":
            mirrored.append((rr, 2 * ac1 + 1 - cc))
        elif side == "above":
            mirrored.append((2 * ar0 - 1 - rr, cc))
        else:
            mirrored.append((2 * ar1 + 1 - rr, cc))
    assert all(0 <= r < H and 0 <= c < W for r, c in moved)
    assert all(0 <= r < H and 0 <= c < W for r, c in mirrored)
    assert not (set(moved) & set(mirrored))
    return moved, mirrored


def _cavities_from_fg_mask(fg_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = map(int, fg_mask.shape)
    seen = np.zeros((H, W), dtype=bool)
    cavities: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if bool(fg_mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            region: list[tuple[int, int]] = []
            touches_border = False
            while stack:
                rr, cc = stack.pop()
                region.append((rr, cc))
                if rr == 0 or cc == 0 or rr == H - 1 or cc == W - 1:
                    touches_border = True
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not fg_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            if not touches_border:
                cavities.append(region)
    return cavities


def _bbox_distance_to_point(bbox: tuple[int, int, int, int], point: tuple[int, int]) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    r, c = map(int, point)
    if r < r0:
        dr = r0 - r
    elif r > r1:
        dr = r - r1
    else:
        dr = 0
    if c < c0:
        dc = c0 - c
    elif c > c1:
        dc = c - c1
    else:
        dc = 0
    return int(dr + dc)


def _new_cavities(moved_mask: np.ndarray, full_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    moved_sets = {frozenset(region) for region in _cavities_from_fg_mask(moved_mask)}
    out = []
    for region in _cavities_from_fg_mask(full_mask):
        if frozenset(region) not in moved_sets:
            out.append(region)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each sprite to its 2x2 anchor, mirror it across the touching anchor edge, then fill the newly enclosed cavities with the nearest seed color.

    Input structure:
    - The background is black.
    - There are 3 to 4 small monochrome 8-connected sprites, the same number of solid 2x2 anchor blocks, and 2 to 3 single-pixel seeds.
    - Each sprite and its anchor have a unique cardinal-facing relation with overlapping row-band or column-band span.

    Rule:
    - For every sprite-anchor pair, slide the sprite straight toward the anchor until its front edge touches the anchor's near side.
    - Reflect the moved sprite across that touching anchor edge and draw the mirrored copy in the anchor color; discard all original sprite and anchor positions.
    - After all pairs are transformed, find 4-connected background cavities enclosed by the transformed sprite+mirror geometry but not already enclosed by the moved sprites alone.
    - For each such cavity, compute the Manhattan distance from its bounding box to every seed pixel, choose the nearest seed, and flood the cavity with that seed's color.
    - Preserve the seed pixels themselves unchanged.

    Key insight:
    - The mirror step creates new topology: only after sliding and reflecting do concave sprites produce closed background pockets, and those pockets are colored by spatial proximity to the seed pixels.
    """
    g = np.asarray(grid, dtype=int)
    bg, sprites, anchors, seeds = _classify_objects(g)
    pairing = _unique_pairing(sprites, anchors)

    out = np.full_like(g, bg)
    moved_only_mask = np.zeros(g.shape, dtype=bool)
    full_mask = np.zeros(g.shape, dtype=bool)

    for sprite, pair in zip(sprites, pairing):
        assert pair is not None
        anchor_idx, relation = pair
        anchor = anchors[int(anchor_idx)]
        moved, mirrored = _transform_coords(sprite, anchor, relation, g.shape)
        for r, c in moved:
            out[r, c] = int(sprite["color"])
            moved_only_mask[r, c] = True
            full_mask[r, c] = True
        for r, c in mirrored:
            out[r, c] = int(anchor["color"])
            full_mask[r, c] = True

    seed_points = []
    for seed in seeds:
        (r, c) = map(int, seed["coords"][0])
        out[r, c] = int(seed["color"])
        seed_points.append((r, c, int(seed["color"])))

    for region in _new_cavities(moved_only_mask, full_mask):
        bbox = bbox_coords(np.array(region, dtype=int))
        seed_r, seed_c, seed_color = min(
            seed_points,
            key=lambda item: (_bbox_distance_to_point(bbox, (item[0], item[1])), item[0], item[1], item[2]),
        )
        _ = (seed_r, seed_c)
        for r, c in region:
            out[r, c] = int(seed_color)

    return out


def _rows_to_mask(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[1 if ch == "#" else 0 for ch in row] for row in rows], dtype=bool)


CONCAVE_MASKS = [_rows_to_mask(rows) for rows in CONCAVE_ROWS]
CONVEX_MASKS = [_rows_to_mask(rows) for rows in CONVEX_ROWS]


def _orient_mask(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return mask.copy()
    if side == "right":
        return np.fliplr(mask)
    if side == "above":
        return np.rot90(mask, 3)
    assert side == "below"
    return np.rot90(mask, 1)


def _pair_layout(mask: np.ndarray, side: str, pr: int, pc: int, gap: int):
    h, w = map(int, mask.shape)
    if side == "left":
        moved_r, moved_c = pr, pc
        mirror_r, mirror_c = pr, pc + w
        input_r, input_c = pr, pc - gap
        union_bbox = (pr, pc, pr + h - 1, pc + 2 * w - 1)
    elif side == "right":
        moved_r, moved_c = pr, pc + w
        mirror_r, mirror_c = pr, pc
        input_r, input_c = pr, moved_c + gap
        union_bbox = (pr, pc, pr + h - 1, pc + 2 * w - 1)
    elif side == "above":
        moved_r, moved_c = pr, pc
        mirror_r, mirror_c = pr + h, pc
        input_r, input_c = pr - gap, pc
        union_bbox = (pr, pc, pr + 2 * h - 1, pc + w - 1)
    else:
        moved_r, moved_c = pr + h, pc
        mirror_r, mirror_c = pr, pc
        input_r, input_c = moved_r + gap, pc
        union_bbox = (pr, pc, pr + 2 * h - 1, pc + w - 1)
    return {
        "moved_pos": (int(moved_r), int(moved_c)),
        "mirror_pos": (int(mirror_r), int(mirror_c)),
        "input_pos": (int(input_r), int(input_c)),
        "union_bbox": tuple(map(int, union_bbox)),
    }


def _mask_coords(mask: np.ndarray, origin: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, origin)
    coords = []
    for r, c in np.argwhere(mask):
        coords.append((int(r0 + r), int(c0 + c)))
    return coords


def _anchor_bbox_for_layout(mask_shape: tuple[int, int], side: str, moved_pos: tuple[int, int], offset: int):
    h, w = map(int, mask_shape)
    mr, mc = map(int, moved_pos)
    if side == "left":
        return (mr + offset, mc + w, mr + offset + 1, mc + w + 1)
    if side == "right":
        return (mr + offset, mc - 2, mr + offset + 1, mc - 1)
    if side == "above":
        return (mr + h, mc + offset, mr + h + 1, mc + offset + 1)
    return (mr - 2, mc + offset, mr - 1, mc + offset + 1)


def _expanded_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1):
    H, W = map(int, shape)
    r0, c0, r1, c1 = map(int, bbox)
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(H - 1, r1 + pad),
        min(W - 1, c1 + pad),
    )


def _reserve_clear(reserved: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = map(int, bbox)
    return not bool(reserved[r0 : r1 + 1, c0 : c1 + 1].any())


def _paint_cells(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _rect_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    return [(int(r), int(c)) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _swept_path_cells(start: list[tuple[int, int]], end: list[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for (r0, c0), (r1, c1) in zip(sorted(start), sorted(end)):
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            for c in range(c0, c1 + step, step):
                out.add((int(r0), int(c)))
        else:
            step = 1 if r1 >= r0 else -1
            for r in range(r0, r1 + step, step):
                out.add((int(r), int(c0)))
    return out


def _pair_local_cavity_count(mask: np.ndarray, side: str) -> int:
    oriented = _orient_mask(mask, side)
    h, w = map(int, oriented.shape)
    if side in {"left", "right"}:
        fg = np.zeros((h, 2 * w), dtype=bool)
        if side == "left":
            fg[:, :w] |= oriented
            fg[:, w:] |= np.fliplr(oriented)
        else:
            fg[:, :w] |= np.fliplr(oriented)
            fg[:, w:] |= oriented
    else:
        fg = np.zeros((2 * h, w), dtype=bool)
        if side == "above":
            fg[:h, :] |= oriented
            fg[h:, :] |= np.flipud(oriented)
        else:
            fg[:h, :] |= np.flipud(oriented)
            fg[h:, :] |= oriented
    return len(_cavities_from_fg_mask(fg))


def _seed_candidates(
    shape: tuple[int, int],
    free_mask: np.ndarray,
    path_cells: set[tuple[int, int]],
    cavity_cells: set[tuple[int, int]],
    anchors: list[tuple[int, int, int, int]],
    target_idx: int,
) -> list[tuple[int, int]]:
    H, W = map(int, shape)
    target_bbox = anchors[int(target_idx)]
    candidates = []
    for r in range(H):
        for c in range(W):
            if not bool(free_mask[r, c]):
                continue
            if (r, c) in path_cells or (r, c) in cavity_cells:
                continue
            d_target = _bbox_distance_to_point(target_bbox, (r, c))
            if d_target < 1 or d_target > 2:
                continue
            dists = [_bbox_distance_to_point(bbox, (r, c)) for bbox in anchors]
            if sum(int(d == d_target) for d in dists) != 1 or int(dists[target_idx]) != d_target:
                continue
            candidates.append((int(r), int(c)))
    random.shuffle(candidates)
    return candidates


def _build_expected_output(shape: tuple[int, int], bg: int, records: list[dict], seeds: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    moved_only_mask = np.zeros(shape, dtype=bool)
    full_mask = np.zeros(shape, dtype=bool)
    for rec in records:
        _paint_cells(out, rec["moved_coords"], rec["sprite_color"])
        for r, c in rec["moved_coords"]:
            moved_only_mask[r, c] = True
            full_mask[r, c] = True
        _paint_cells(out, rec["mirror_coords"], rec["anchor_color"])
        for r, c in rec["mirror_coords"]:
            full_mask[r, c] = True
    for seed in seeds:
        out[int(seed["pos"][0]), int(seed["pos"][1])] = int(seed["color"])
    for region in _new_cavities(moved_only_mask, full_mask):
        bbox = bbox_coords(np.array(region, dtype=int))
        seed = min(
            seeds,
            key=lambda item: (
                _bbox_distance_to_point(bbox, item["pos"]),
                int(item["pos"][0]),
                int(item["pos"][1]),
                int(item["color"]),
            ),
        )
        for r, c in region:
            out[r, c] = int(seed["color"])
    return out


def _validate_scene(inp: np.ndarray, expected: np.ndarray, records: list[dict], seeds: list[dict]) -> bool:
    try:
        solved = main(inp)
        _bg, sprites, anchors, parsed_seeds = _classify_objects(inp)
        pairing = _unique_pairing(sprites, anchors)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    if len(parsed_seeds) != len(seeds) or len(sprites) != len(records):
        return False
    if sum(int(rec["concave"]) for rec in records) < 2:
        return False
    if sum(int(not rec["concave"]) for rec in records) < 1:
        return False

    anchor_bboxes = [tuple(map(int, rec["anchor_bbox"])) for rec in records]
    seed_positions = [tuple(map(int, seed["pos"])) for seed in seeds]
    anchor_hits = []
    for seed_pos in seed_positions:
        dists = [_bbox_distance_to_point(bbox, seed_pos) for bbox in anchor_bboxes]
        best = min(dists)
        if best > 2 or sum(int(d == best) for d in dists) != 1:
            return False
        anchor_hits.append(int(dists.index(best)))
    if len(set(anchor_hits)) != len(anchor_hits):
        return False

    moved_only_mask = np.zeros(inp.shape, dtype=bool)
    full_mask = np.zeros(inp.shape, dtype=bool)
    expected_cavity_count = 0
    for rec in records:
        count = len(rec["pair_cavities"])
        if rec["concave"] and count < 1:
            return False
        if (not rec["concave"]) and count != 0:
            return False
        expected_cavity_count += count
        for r, c in rec["moved_coords"]:
            moved_only_mask[r, c] = True
            full_mask[r, c] = True
        for r, c in rec["mirror_coords"]:
            full_mask[r, c] = True
    new_cavities = _new_cavities(moved_only_mask, full_mask)
    if len(new_cavities) != expected_cavity_count:
        return False
    for region in new_cavities:
        bbox = bbox_coords(np.array(region, dtype=int))
        dists = [_bbox_distance_to_point(bbox, seed_pos) for seed_pos in seed_positions]
        if sum(int(d == min(dists)) for d in dists) != 1:
            return False

    for sprite, pair in zip(sprites, pairing):
        if pair is None:
            return False
        anchor_idx, relation = pair
        moved, mirrored = _transform_coords(sprite, anchors[int(anchor_idx)], relation, inp.shape)
        if not moved or not mirrored:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(20, 28)
    W = random.randint(20, 28)
    n_pairs = random.randint(3, 4)
    n_concave = random.randint(2, min(3, n_pairs - 1))
    n_seeds = n_concave

    palette = list(range(1, 10))
    random.shuffle(palette)
    sprite_colors = palette[:n_pairs]
    anchor_palette = palette[n_pairs : n_pairs + 2]
    seed_colors = palette[n_pairs + 2 : n_pairs + 2 + n_seeds]
    if len(seed_colors) != n_seeds:
        return None

    shape = (H, W)
    reserved = np.zeros(shape, dtype=bool)
    records = []

    kinds = [True] * n_concave + [False] * (n_pairs - n_concave)
    random.shuffle(kinds)

    for pair_idx, concave in enumerate(kinds):
        templates = CONCAVE_MASKS if concave else CONVEX_MASKS
        placed = False
        for _ in range(200):
            side = random.choice(SIDES)
            base = random.choice(templates)
            mask = _orient_mask(base, side)
            h, w = map(int, mask.shape)
            gap = random.randint(2, 5)

            if side == "left":
                r_lo, r_hi = 1, H - h - 2
                c_lo, c_hi = gap + 1, W - 2 * w - 2
            elif side == "right":
                r_lo, r_hi = 1, H - h - 2
                c_lo, c_hi = 1, W - 2 * w - gap - 2
            elif side == "above":
                r_lo, r_hi = gap + 1, H - 2 * h - 2
                c_lo, c_hi = 1, W - w - 2
            else:
                r_lo, r_hi = 1, H - 2 * h - gap - 2
                c_lo, c_hi = 1, W - w - 2
            if r_lo > r_hi or c_lo > c_hi:
                continue

            pr = random.randint(r_lo, r_hi)
            pc = random.randint(c_lo, c_hi)
            layout = _pair_layout(mask, side, pr, pc, gap)
            moved_pos = layout["moved_pos"]
            mirror_pos = layout["mirror_pos"]
            input_pos = layout["input_pos"]
            moved_coords = _mask_coords(mask, moved_pos)
            mirror_mask = np.fliplr(mask) if side in {"left", "right"} else np.flipud(mask)
            mirror_coords = _mask_coords(mirror_mask, mirror_pos)
            input_coords = _mask_coords(mask, input_pos)
            offset_max = (h - 2) if side in {"left", "right"} else (w - 2)
            if offset_max < 0:
                continue
            offset = random.randint(0, offset_max)
            anchor_bbox = _anchor_bbox_for_layout(mask.shape, side, moved_pos, offset)

            all_coords = set(moved_coords) | set(mirror_coords) | set(input_coords) | set(_rect_coords(anchor_bbox))
            if any(not (0 <= r < H and 0 <= c < W) for r, c in all_coords):
                continue
            reserve_bbox = bbox_coords(np.array(list(all_coords), dtype=int))
            reserve_bbox = _expanded_bbox(reserve_bbox, shape, pad=1)
            if not _reserve_clear(reserved, reserve_bbox):
                continue

            local_cavity_count = _pair_local_cavity_count(base, side)
            if concave and local_cavity_count < 1:
                continue
            if (not concave) and local_cavity_count != 0:
                continue

            pair_fg = np.zeros(shape, dtype=bool)
            for r, c in moved_coords + mirror_coords:
                pair_fg[r, c] = True
            pair_cavities = _cavities_from_fg_mask(pair_fg)
            if concave and not pair_cavities:
                continue
            if (not concave) and pair_cavities:
                continue

            reserved[reserve_bbox[0] : reserve_bbox[2] + 1, reserve_bbox[1] : reserve_bbox[3] + 1] = True
            records.append(
                {
                    "pair_idx": int(pair_idx),
                    "concave": bool(concave),
                    "side": side,
                    "sprite_color": int(sprite_colors[pair_idx]),
                    "anchor_color": int(anchor_palette[pair_idx % len(anchor_palette)]),
                    "input_coords": input_coords,
                    "moved_coords": moved_coords,
                    "mirror_coords": mirror_coords,
                    "anchor_bbox": tuple(map(int, anchor_bbox)),
                    "pair_cavities": pair_cavities,
                    "path_cells": _swept_path_cells(input_coords, moved_coords),
                }
            )
            placed = True
            break
        if not placed:
            return None

    anchors = [tuple(map(int, rec["anchor_bbox"])) for rec in records]
    occupied = np.zeros(shape, dtype=bool)
    cavity_cells: set[tuple[int, int]] = set()
    path_cells: set[tuple[int, int]] = set()
    for rec in records:
        for r, c in rec["input_coords"] + rec["moved_coords"] + rec["mirror_coords"]:
            occupied[r, c] = True
        for r, c in _rect_coords(rec["anchor_bbox"]):
            occupied[r, c] = True
        for region in rec["pair_cavities"]:
            cavity_cells.update(region)
        path_cells |= rec["path_cells"]
    free_mask = ~occupied

    concave_indices = [idx for idx, rec in enumerate(records) if rec["concave"]]
    random.shuffle(concave_indices)
    seed_targets = concave_indices[:n_seeds]
    seeds = []
    used_seed_cells: set[tuple[int, int]] = set()
    for seed_color, target_idx in zip(seed_colors, seed_targets):
        candidates = _seed_candidates(shape, free_mask, path_cells, cavity_cells, anchors, target_idx)
        chosen = None
        for cand in candidates:
            if cand in used_seed_cells:
                continue
            chosen = cand
            break
        if chosen is None:
            return None
        used_seed_cells.add(chosen)
        free_mask[chosen[0], chosen[1]] = False
        seeds.append({"pos": chosen, "color": int(seed_color)})

    inp = np.zeros(shape, dtype=int)
    for rec in records:
        _paint_cells(inp, rec["input_coords"], rec["sprite_color"])
        _paint_cells(inp, _rect_coords(rec["anchor_bbox"]), rec["anchor_color"])
    for seed in seeds:
        inp[int(seed["pos"][0]), int(seed["pos"][1])] = int(seed["color"])

    try:
        _bg, sprites, anchors, _seeds = _classify_objects(inp)
        _unique_pairing(sprites, anchors)
    except Exception:
        return None

    expected = _build_expected_output(shape, 0, records, seeds)
    if not _validate_scene(inp, expected, records, seeds):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-cavity ARC input")
