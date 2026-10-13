from __future__ import annotations

import random
from collections import Counter, deque
from itertools import product

import numpy as np


BG = 0
FRAME = 1
MARK = 2
DEPTH_COLORS = (2, 3, 4)

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

GEN_ATTEMPTS = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_of(coords: list[Coord] | np.ndarray) -> BBox:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def border_cells(bbox: BBox) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    cells: list[Coord] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def components_from_mask(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def flood_background(grid: np.ndarray, start: Coord) -> np.ndarray:
    h, w = grid.shape
    reach = np.zeros((h, w), dtype=bool)
    sr, sc = start
    if not (0 <= sr < h and 0 <= sc < w) or int(grid[sr, sc]) != BG:
        return reach
    q = deque([start])
    reach[sr, sc] = True
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reach[nr, nc] and int(grid[nr, nc]) == BG:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def contains_strict(bbox: BBox, cell: Coord) -> bool:
    r0, c0, r1, c1 = bbox
    r, c = cell
    return r0 < r < r1 and c0 < c < c1


def gap_to_sprite_blue_offsets(gap: dict) -> list[Coord]:
    side = gap["side"]
    length = gap["length"]
    marker_at_end = gap["marker_at_end"]
    offsets: list[Coord] = []
    if side == "top":
        if marker_at_end == "start":
            offsets = [(1, dc) for dc in range(length)]
        else:
            offsets = [(1, -dc) for dc in range(length)]
    elif side == "bottom":
        if marker_at_end == "start":
            offsets = [(-1, dc) for dc in range(length)]
        else:
            offsets = [(-1, -dc) for dc in range(length)]
    elif side == "left":
        if marker_at_end == "start":
            offsets = [(dr, 1) for dr in range(length)]
        else:
            offsets = [(-dr, 1) for dr in range(length)]
    else:
        if marker_at_end == "start":
            offsets = [(dr, -1) for dr in range(length)]
        else:
            offsets = [(-dr, -1) for dr in range(length)]
    return offsets


def sprite_patch_from_offsets(blue_offsets: list[Coord]) -> dict:
    all_offsets = [(0, 0)] + list(blue_offsets)
    rs = [r for r, _ in all_offsets]
    cs = [c for _, c in all_offsets]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    patch = np.zeros((h, w), dtype=int)
    red_local = (-min_r, -min_c)
    patch[red_local] = MARK
    blue_cells_local: list[Coord] = []
    for dr, dc in blue_offsets:
        rr = dr - min_r
        cc = dc - min_c
        patch[rr, cc] = FRAME
        blue_cells_local.append((rr, cc))
    return {
        "patch": patch,
        "h": h,
        "w": w,
        "red_local": red_local,
        "blue_offsets": list(blue_offsets),
        "all_local": [(r - min_r, c - min_c) for r, c in all_offsets],
        "blue_local": blue_cells_local,
    }


def classify_objects(grid: np.ndarray) -> tuple[list[dict], list[Coord]]:
    comps = components_from_mask(grid != BG)
    sprites: list[dict] = []
    targets: list[Coord] = []
    for comp in comps:
        colors = [int(grid[r, c]) for r, c in comp]
        count_red = sum(v == MARK for v in colors)
        count_blue = sum(v == FRAME for v in colors)
        if count_red == 1 and count_blue >= 2 and count_red + count_blue == len(comp) and len(comp) <= 5:
            red = next((r, c) for r, c in comp if int(grid[r, c]) == MARK)
            blue_offsets = sorted((r - red[0], c - red[1]) for r, c in comp if int(grid[r, c]) == FRAME)
            sprites.append(
                {
                    "coords": list(comp),
                    "red": red,
                    "blue_offsets": blue_offsets,
                }
            )
        elif len(comp) == 1 and int(grid[comp[0][0], comp[0][1]]) == MARK:
            targets.append(comp[0])
    sprites.sort(key=lambda s: s["red"])
    targets.sort()
    return sprites, targets


def assign_sprites(sprites: list[dict], targets: list[Coord]) -> list[Coord]:
    assignments: list[Coord] = []
    used: set[Coord] = set()
    for sprite in sprites:
        dists = sorted(
            ((abs(sprite["red"][0] - tr) + abs(sprite["red"][1] - tc), (tr, tc)) for tr, tc in targets),
            key=lambda x: (x[0], x[1][0], x[1][1]),
        )
        assert dists, "missing target markers"
        if len(dists) > 1:
            assert dists[0][0] < dists[1][0], "nearest target marker must be unique"
        target = dists[0][1]
        assert target not in used, "each target marker must match exactly one repair sprite"
        used.add(target)
        assignments.append(target)
    assert len(assignments) == len(targets), "sprite count and target count must match"
    return assignments


def apply_repairs(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sprites, targets = classify_objects(g)
    assert sprites and len(sprites) == len(targets)
    assignments = assign_sprites(sprites, targets)

    repaired = g.copy()
    for sprite in sprites:
        for r, c in sprite["coords"]:
            repaired[r, c] = BG

    for sprite, target in zip(sprites, assignments):
        tr, tc = target
        repaired[tr, tc] = MARK
        for dr, dc in sprite["blue_offsets"]:
            rr, cc = tr + dr, tc + dc
            assert 0 <= rr < repaired.shape[0] and 0 <= cc < repaired.shape[1]
            assert int(repaired[rr, cc]) in (BG, FRAME)
            repaired[rr, cc] = FRAME
    return repaired


def detect_frames(repaired: np.ndarray) -> list[BBox]:
    comps = components_from_mask(repaired == FRAME)
    frames: list[BBox] = []
    for comp in comps:
        b = bbox_of(comp)
        if set(comp) == set(border_cells(b)):
            frames.append(b)
    frames.sort(key=lambda b: ((b[2] - b[0]) * (b[3] - b[1])), reverse=True)
    assert 2 <= len(frames) <= 3, "expected 2 or 3 repaired frame layers"
    return frames


def color_enclosed_regions(repaired: np.ndarray) -> np.ndarray:
    frames = detect_frames(repaired)
    reach = flood_background(repaired, (0, 0))
    out = repaired.copy()
    h, w = out.shape
    for r in range(h):
        for c in range(w):
            if int(out[r, c]) != BG or reach[r, c]:
                continue
            depth = sum(contains_strict(frame, (r, c)) for frame in frames)
            assert depth >= 1, "every unreachable background cell must be inside a frame"
            out[r, c] = DEPTH_COLORS[(depth - 1) % len(DEPTH_COLORS)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair gapped nested rectangular frames by marker-aligned sprite overlay, then color enclosed regions by nesting depth.

    Input structure:
    - The background is black. There are 2 or 3 nested blue 1-pixel rectangular frames, each with 1 or 2 missing edge runs of length 2 or 3.
    - Each gap has one isolated red marker on its exterior side. Outside the outer frame there are scattered small repair sprites made of blue cells plus exactly one red anchor.
    - Every repair sprite is a 4-connected red-blue component whose blue shape is the complement of one gap when the sprite is translated so its red anchor lands on that gap marker.

    Rule:
    - Find each repair sprite, then align it to the unique nearest isolated red gap marker by Manhattan distance between the sprite's red anchor and the marker.
    - Move the sprite there: erase it from its original location, keep the red anchor on the target marker, and overlay only its blue cells to close the matching frame gap.
    - After all gaps are repaired, flood-fill background from the top-left corner to mark the exterior. Any remaining background cell is enclosed; color depth 1 regions red, depth 2 green, depth 3 yellow, cycling every 3 depths. Blue frame cells stay blue and red markers stay red.

    Key insight:
    - The scattered sprites are not output objects themselves; their translated blue pixels change frame connectivity, and that repaired topology determines the nesting depth used for the final recoloring.
    """
    repaired = apply_repairs(np.asarray(grid, dtype=int))
    return color_enclosed_regions(repaired)


def choose_gap_positions(rng: random.Random, bbox: BBox) -> list[dict]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    gap_count = 2 if rng.random() < 0.58 else 1
    sides = rng.sample(["top", "bottom", "left", "right"], gap_count)
    gaps: list[dict] = []
    for side in sides:
        if side in ("top", "bottom"):
            length = rng.randint(2, 3)
            start = rng.randint(1, w - length - 1)
            marker_at_end = rng.choice(["start", "end"])
            cells = [(r0 if side == "top" else r1, c0 + start + dc) for dc in range(length)]
            if marker_at_end == "start":
                edge_c = c0 + start
            else:
                edge_c = c0 + start + length - 1
            marker = (r0 - 1, edge_c) if side == "top" else (r1 + 1, edge_c)
        else:
            length = rng.randint(2, 3)
            start = rng.randint(1, h - length - 1)
            marker_at_end = rng.choice(["start", "end"])
            cells = [(r0 + start + dr, c0 if side == "left" else c1) for dr in range(length)]
            if marker_at_end == "start":
                edge_r = r0 + start
            else:
                edge_r = r0 + start + length - 1
            marker = (edge_r, c0 - 1) if side == "left" else (edge_r, c1 + 1)
        gaps.append(
            {
                "bbox": bbox,
                "side": side,
                "length": length,
                "cells": cells,
                "marker": marker,
                "marker_at_end": marker_at_end,
            }
        )
    return gaps


def choose_frame_layout(rng: random.Random) -> tuple[int, int, list[BBox]]:
    for _ in range(200):
        n_frames = rng.choice([2, 3])
        outer_h = rng.randint(16, 22) if n_frames == 2 else rng.randint(19, 22)
        outer_w = rng.randint(16, 22) if n_frames == 2 else rng.randint(19, 22)
        margins = [rng.randint(3, 4) for _ in range(4)]
        h = outer_h + margins[0] + margins[2]
        w = outer_w + margins[1] + margins[3]
        if h > 30 or w > 30:
            continue
        frames = [
            (
                margins[0],
                margins[1],
                margins[0] + outer_h - 1,
                margins[1] + outer_w - 1,
            )
        ]
        ok = True
        for _layer in range(1, n_frames):
            prev = frames[-1]
            dt, dl, db, dr = [rng.randint(3, 4) for _ in range(4)]
            nxt = (prev[0] + dt, prev[1] + dl, prev[2] - db, prev[3] - dr)
            if nxt[2] - nxt[0] + 1 < 7 or nxt[3] - nxt[1] + 1 < 7:
                ok = False
                break
            frames.append(nxt)
        if ok:
            return h, w, frames
    raise RuntimeError("failed to choose nested frame layout")


def build_repaired_grid(h: int, w: int, frames: list[BBox], gaps: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for bbox in frames:
        for r, c in border_cells(bbox):
            grid[r, c] = FRAME
    for gap in gaps:
        grid[gap["marker"]] = MARK
    return grid


def build_input_base(h: int, w: int, frames: list[BBox], gaps: list[dict]) -> np.ndarray:
    grid = build_repaired_grid(h, w, frames, gaps)
    for gap in gaps:
        for r, c in gap["cells"]:
            grid[r, c] = BG
    return grid


def sprite_candidates(
    h: int,
    w: int,
    outer_bbox: BBox,
    target_markers: list[Coord],
    target_idx: int,
    patch_info: dict,
    occupied: set[Coord],
) -> list[tuple[int, int]]:
    ph = patch_info["h"]
    pw = patch_info["w"]
    red_local = patch_info["red_local"]
    target = target_markers[target_idx]
    out: list[tuple[int, int]] = []
    for top in range(h - ph + 1):
        for left in range(w - pw + 1):
            abs_cells = [(top + rr, left + cc) for rr, cc in patch_info["all_local"]]
            if any(outer_bbox[0] <= r <= outer_bbox[2] and outer_bbox[1] <= c <= outer_bbox[3] for r, c in abs_cells):
                continue
            if any((r, c) in occupied for r, c in abs_cells):
                continue
            new_set = set(abs_cells)
            too_close = False
            for r, c in abs_cells:
                for dr, dc in product((-1, 0, 1), repeat=2):
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if (rr, cc) in occupied and (rr, cc) not in new_set:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                continue
            red_abs = (top + red_local[0], left + red_local[1])
            dists = sorted(
                ((abs(red_abs[0] - mr) + abs(red_abs[1] - mc), idx) for idx, (mr, mc) in enumerate(target_markers)),
                key=lambda x: (x[0], x[1]),
            )
            if len(dists) > 1 and dists[0][0] == dists[1][0]:
                continue
            if dists[0][1] != target_idx:
                continue
            out.append((top, left))
    return out


def place_sprites(
    rng: random.Random,
    input_grid: np.ndarray,
    outer_bbox: BBox,
    gaps: list[dict],
) -> np.ndarray | None:
    target_markers = [gap["marker"] for gap in gaps]
    patch_infos = [sprite_patch_from_offsets(gap_to_sprite_blue_offsets(gap)) for gap in gaps]
    occupied = {(r, c) for r, c in zip(*np.nonzero(input_grid != BG))}

    options = []
    for idx, patch_info in enumerate(patch_infos):
        cands = sprite_candidates(
            input_grid.shape[0],
            input_grid.shape[1],
            outer_bbox,
            target_markers,
            idx,
            patch_info,
            occupied,
        )
        if not cands:
            return None
        options.append((idx, patch_info, cands))

    options.sort(key=lambda item: len(item[2]))
    out = input_grid.copy()

    def backtrack(pos: int, occ: set[Coord]) -> bool:
        if pos == len(options):
            return True
        gap_idx, patch_info, all_cands = options[pos]
        cands = list(all_cands)
        rng.shuffle(cands)
        for top, left in cands:
            abs_cells = [(top + rr, left + cc) for rr, cc in patch_info["all_local"]]
            cell_set = set(abs_cells)
            if any(cell in occ for cell in cell_set):
                continue
            too_close = False
            for r, c in abs_cells:
                for dr, dc in product((-1, 0, 1), repeat=2):
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]):
                        continue
                    if (rr, cc) in occ and (rr, cc) not in cell_set:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                continue
            changed: list[Coord] = []
            for rr, cc in patch_info["blue_local"]:
                r_abs, c_abs = top + rr, left + cc
                out[r_abs, c_abs] = FRAME
                changed.append((r_abs, c_abs))
            red_abs = (top + patch_info["red_local"][0], left + patch_info["red_local"][1])
            out[red_abs] = MARK
            changed.append(red_abs)
            occ2 = occ | cell_set
            if backtrack(pos + 1, occ2):
                return True
            for cell in changed:
                out[cell] = BG
        return False

    if backtrack(0, occupied):
        return out
    return None


def expected_output_from_truth(repaired: np.ndarray, frames: list[BBox]) -> np.ndarray:
    reach = flood_background(repaired, (0, 0))
    out = repaired.copy()
    h, w = out.shape
    for r in range(h):
        for c in range(w):
            if int(out[r, c]) != BG or reach[r, c]:
                continue
            depth = sum(contains_strict(frame, (r, c)) for frame in frames)
            if depth:
                out[r, c] = DEPTH_COLORS[(depth - 1) % len(DEPTH_COLORS)]
    return out


def validate_semantics(inp: np.ndarray, repaired: np.ndarray, frames: list[BBox], gaps: list[dict]) -> bool:
    if int(inp[0, 0]) != BG:
        return False

    pre_reach = flood_background(inp, (0, 0))
    for frame in frames:
        center = ((frame[0] + frame[2]) // 2, (frame[1] + frame[3]) // 2)
        if not pre_reach[center]:
            return False

    repaired_frames = detect_frames(repaired)
    if repaired_frames != frames:
        return False

    post_reach = flood_background(repaired, (0, 0))
    depths_seen: set[int] = set()
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            if int(repaired[r, c]) != BG or post_reach[r, c]:
                continue
            depth = sum(contains_strict(frame, (r, c)) for frame in frames)
            if depth == 0:
                return False
            depths_seen.add(depth)
    if depths_seen != set(range(1, len(frames) + 1)):
        return False

    sprites, targets = classify_objects(inp)
    if len(sprites) != len(gaps) or len(targets) != len(gaps):
        return False
    if assign_sprites(sprites, targets) != [gap["marker"] for gap in sorted(gaps, key=lambda g: g["marker"])]:
        # Rebuild expected target order to match classify_objects sorting.
        mapping = {gap["marker"]: sorted(gap_to_sprite_blue_offsets(gap)) for gap in gaps}
        for sprite, target in zip(sprites, assign_sprites(sprites, targets)):
            if sorted(sprite["blue_offsets"]) != mapping[target]:
                return False
    return np.array_equal(main(inp), expected_output_from_truth(repaired, frames))


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        try:
            h, w, frames = choose_frame_layout(rng)
            gaps: list[dict] = []
            for bbox in frames:
                gaps.extend(choose_gap_positions(rng, bbox))
            repaired = build_repaired_grid(h, w, frames, gaps)
            inp = build_input_base(h, w, frames, gaps)
            inp = place_sprites(rng, inp, frames[0], gaps)
            if inp is None:
                continue
            if validate_semantics(inp, repaired, frames, gaps):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate valid input")

