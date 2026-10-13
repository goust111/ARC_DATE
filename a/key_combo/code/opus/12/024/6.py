from __future__ import annotations

from collections import Counter
import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
ADJACENT_PAIRS = {
    frozenset(("top", "left")),
    frozenset(("top", "right")),
    frozenset(("bottom", "left")),
    frozenset(("bottom", "right")),
}
GUIDE_COLORS = (1, 5, 6, 7)
BODY_COLORS = (2, 3, 4, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray, bg: int = 0) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"coords": coords, "bbox": _bbox(coords), "area": len(coords)})
    return comps


def _paste_nonoverlap(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = dst.shape
    ph, pw = patch.shape
    for pr in range(ph):
        rr = r0 + pr
        if rr < 0 or rr >= h:
            continue
        for pc in range(pw):
            cc = c0 + pc
            if cc < 0 or cc >= w:
                continue
            val = int(patch[pr, pc])
            if val != 0 and int(dst[rr, cc]) == 0:
                dst[rr, cc] = val


def _transform(patch: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(patch, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(patch: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(patch) if flip else patch
    return np.rot90(out, -int(k))


def _side_values(patch: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return patch[0, :]
    if side == "bottom":
        return patch[-1, :]
    if side == "left":
        return patch[:, 0]
    return patch[:, -1]


def _detect_controls(grid: np.ndarray) -> dict[str, int]:
    h, w = grid.shape
    samples = {
        "top": grid[0, 1 : w - 1],
        "bottom": grid[h - 1, 1 : w - 1],
        "left": grid[1 : h - 1, 0],
        "right": grid[1 : h - 1, w - 1],
    }
    controls: dict[str, int] = {}
    for side, vals in samples.items():
        uniq = {int(v) for v in vals}
        if len(uniq) == 1:
            color = next(iter(uniq))
            if color != 0:
                controls[side] = color
    if not controls:
        raise ValueError("no control bars detected")
    if len(set(controls.values())) != len(controls):
        raise ValueError("control colors must be unique")
    return controls


def _erase_controls(grid: np.ndarray, controls: dict[str, int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    if "top" in controls:
        out[0, :] = 0
    if "bottom" in controls:
        out[h - 1, :] = 0
    if "left" in controls:
        out[:, 0] = 0
    if "right" in controls:
        out[:, w - 1] = 0
    return out


def _body_color(patch: np.ndarray) -> int:
    vals = [int(v) for v in patch.flat if int(v) != 0]
    if not vals:
        raise ValueError("empty object patch")
    return Counter(vals).most_common(1)[0][0]


def _detect_guides(patch: np.ndarray, body_color: int) -> dict[str, int]:
    guides: dict[str, int] = {}
    for side in SIDES:
        edge = [int(v) for v in _side_values(patch, side)]
        non_body = [v for v in edge if v != body_color]
        if not non_body:
            continue
        counts = Counter(non_body)
        color, count = counts.most_common(1)[0]
        if color == 0:
            continue
        if count >= len(edge) - 1 and all(v in (body_color, 0, color) for v in edge):
            guides[side] = color
    return guides


def _cuts_from_guides(guides: dict[str, int]) -> tuple[int, int, int, int]:
    top = 1 if "top" in guides else 0
    bottom = 1 if "bottom" in guides else 0
    left = 1 if "left" in guides else 0
    right = 1 if "right" in guides else 0
    return top, bottom, left, right


def _tile_positions(seed_r: int, seed_c: int, h: int, w: int, direction: str) -> list[tuple[int, int]]:
    positions = [(seed_r, seed_c)]
    if direction == "top":
        r = seed_r - h
        while r + h - 1 >= 0:
            positions.append((r, seed_c))
            r -= h
    elif direction == "bottom":
        r = seed_r + h
        while r < 10**9:
            positions.append((r, seed_c))
            r += h
            if len(positions) > 64:
                break
    elif direction == "left":
        c = seed_c - w
        while c + w - 1 >= 0:
            positions.append((seed_r, c))
            c -= w
    else:
        c = seed_c + w
        while c < 10**9:
            positions.append((seed_r, c))
            c += w
            if len(positions) > 64:
                break
    return positions


def _copies_to_boundary(seed_r: int, seed_c: int, h: int, w: int, direction: str, grid_shape: tuple[int, int]) -> int:
    gh, gw = grid_shape
    if direction == "top":
        if seed_r % h != 0:
            return 0
        return seed_r // h + 1
    if direction == "bottom":
        gap = gh - (seed_r + h)
        if gap < 0 or gap % h != 0:
            return 0
        return gap // h + 1
    if direction == "left":
        if seed_c % w != 0:
            return 0
        return seed_c // w + 1
    gap = gw - (seed_c + w)
    if gap < 0 or gap % w != 0:
        return 0
    return gap // w + 1


def _match_variants(patch: np.ndarray, color_to_side: dict[int, str]) -> list[dict]:
    body = _body_color(patch)
    matches: list[dict] = []
    for k in range(4):
        base = np.rot90(patch, k)
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            guides = _detect_guides(transformed, body)
            if len(guides) != 2 or frozenset(guides) not in ADJACENT_PAIRS:
                continue
            if not all(color_to_side.get(color) == side for side, color in guides.items()):
                continue
            top, bottom, left, right = _cuts_from_guides(guides)
            pure = transformed[top : transformed.shape[0] - bottom, left : transformed.shape[1] - right].copy()
            matches.append(
                {
                    "k": k,
                    "flip": flip,
                    "aligned": transformed,
                    "guides": guides,
                    "pure": pure,
                    "cuts": (top, bottom, left, right),
                }
            )
    return matches


def _analyze_grid(grid: np.ndarray) -> tuple[dict[str, int], list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    gh, gw = g.shape
    controls = _detect_controls(g)
    color_to_side = {color: side for side, color in controls.items()}
    work = _erase_controls(g, controls)
    comps = _find_components(work, bg=0)
    objects: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if r0 == 0 or c0 == 0 or r1 == gh - 1 or c1 == gw - 1:
            raise ValueError("objects must stay off the border")
        patch = _crop(work, comp["bbox"])
        matches = _match_variants(patch, color_to_side)
        if len(matches) != 1:
            raise ValueError("object transform is ambiguous or missing")
        match = matches[0]
        top, bottom, left, right = match["cuts"]
        seed_r = r0 + top
        seed_c = c0 + left
        ah, aw = match["aligned"].shape
        distances = {}
        for side in match["guides"]:
            if side == "top":
                distances[side] = r0
            elif side == "bottom":
                distances[side] = gh - (r0 + ah)
            elif side == "left":
                distances[side] = c0
            else:
                distances[side] = gw - (c0 + aw)
        order = sorted(distances.items(), key=lambda item: (item[1], SIDES.index(item[0])))
        if len(order) != 2:
            raise ValueError("each object must have two guide distances")
        direction = order[0][0]
        copy_count = _copies_to_boundary(seed_r, seed_c, match["pure"].shape[0], match["pure"].shape[1], direction, g.shape)
        objects.append(
            {
                "bbox": comp["bbox"],
                "coords": comp["coords"],
                "patch": patch,
                "aligned": match["aligned"],
                "pure": match["pure"],
                "guides": match["guides"],
                "direction": direction,
                "seed": (seed_r, seed_c),
                "copy_count": copy_count,
                "distances": distances,
            }
        )
    if not objects:
        raise ValueError("no objects found")
    return controls, objects, work


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each guided sprite so its colored edges face the matching border controls, crop the guides, then tile the cropped body toward the nearer matched control.

    Input structure:
    - Black background with 2-3 interior multicolor sprites and up to four 1-pixel control bars on the grid border.
    - Each sprite is a solid-color rectangle with exactly two adjacent 1-pixel guide edges of distinct colors.
    - Each guide color matches exactly one border control bar color, and every control bar spans an entire side.

    Rule:
    - For each sprite, find the unique rotation/reflection that makes each guide edge face the border side carrying the same color.
    - Remove the two aligned guide strips, leaving only the solid body at the transformed in-place location.
    - Compare the distances from that aligned sprite to its two matched control bars, choose the nearer bar's direction, and tile the cropped body from its original body position toward that side with no gaps.
    - When a tile would overlap an existing non-black pixel, leave the overlapping cells unchanged, and erase all control bars in the final output.

    Key insight:
    - The same guide colors jointly determine both the orientation of each sprite and the border direction that controls its repetition, so the tiling axis can only be resolved after the orientation and cropping steps.
    """
    g = np.asarray(grid, dtype=int)
    controls, objects, _work = _analyze_grid(g)
    out = _erase_controls(g, controls)
    for obj in objects:
        for r, c in obj["coords"]:
            out[r, c] = 0
    for obj in objects:
        seed_r, seed_c = obj["seed"]
        pure = obj["pure"]
        ph, pw = pure.shape
        direction = obj["direction"]
        if direction == "top":
            r = seed_r
            while r + ph - 1 >= 0:
                _paste_nonoverlap(out, pure, r, seed_c)
                r -= ph
        elif direction == "bottom":
            r = seed_r
            while r < g.shape[0]:
                _paste_nonoverlap(out, pure, r, seed_c)
                r += ph
        elif direction == "left":
            c = seed_c
            while c + pw - 1 >= 0:
                _paste_nonoverlap(out, pure, seed_r, c)
                c -= pw
        else:
            c = seed_c
            while c < g.shape[1]:
                _paste_nonoverlap(out, pure, seed_r, c)
                c += pw
    return out


def _make_aligned_patch(
    body_h: int,
    body_w: int,
    body_color: int,
    horiz_side: str,
    vert_side: str,
    side_to_color: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    top = 1 if horiz_side == "top" else 0
    bottom = 1 if horiz_side == "bottom" else 0
    left = 1 if vert_side == "left" else 0
    right = 1 if vert_side == "right" else 0
    full = np.zeros((body_h + top + bottom, body_w + left + right), dtype=int)
    full[top : top + body_h, left : left + body_w] = int(body_color)
    if top:
        full[0, left : left + body_w] = int(side_to_color["top"])
    if bottom:
        full[-1, left : left + body_w] = int(side_to_color["bottom"])
    if left:
        full[top : top + body_h, 0] = int(side_to_color["left"])
    if right:
        full[top : top + body_h, -1] = int(side_to_color["right"])
    if top and left:
        full[0, 0] = 0
    if top and right:
        full[0, -1] = 0
    if bottom and left:
        full[-1, 0] = 0
    if bottom and right:
        full[-1, -1] = 0
    pure = np.full((body_h, body_w), int(body_color), dtype=int)
    return full, pure


def _copy_count_for_candidate(
    r0: int,
    c0: int,
    pure_shape: tuple[int, int],
    cuts: tuple[int, int, int, int],
    direction: str,
    grid_shape: tuple[int, int],
) -> int:
    top, _bottom, left, _right = cuts
    seed_r = r0 + top
    seed_c = c0 + left
    return _copies_to_boundary(seed_r, seed_c, pure_shape[0], pure_shape[1], direction, grid_shape)


def _tile_cell_set(
    r0: int,
    c0: int,
    pure_shape: tuple[int, int],
    cuts: tuple[int, int, int, int],
    direction: str,
    grid_shape: tuple[int, int],
) -> tuple[set[tuple[int, int]], int]:
    gh, gw = grid_shape
    top, _bottom, left, _right = cuts
    seed_r = r0 + top
    seed_c = c0 + left
    ph, pw = pure_shape
    count = _copies_to_boundary(seed_r, seed_c, ph, pw, direction, grid_shape)
    if count < 2:
        return set(), 0
    cells: set[tuple[int, int]] = set()
    for rr, cc in _tile_positions(seed_r, seed_c, ph, pw, direction):
        if direction == "bottom" and rr >= gh:
            break
        if direction == "right" and cc >= gw:
            break
        if direction == "top" and rr < 0:
            break
        if direction == "left" and cc < 0:
            break
        for dr in range(ph):
            for dc in range(pw):
                ar = rr + dr
                ac = cc + dc
                if 0 <= ar < gh and 0 <= ac < gw:
                    cells.add((ar, ac))
    return cells, count


def _box_with_pad(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _enumerate_positions(
    grid_shape: tuple[int, int],
    input_shape: tuple[int, int],
    aligned_shape: tuple[int, int],
    pure_shape: tuple[int, int],
    cuts: tuple[int, int, int, int],
    guide_sides: tuple[str, str],
    chosen_direction: str,
) -> list[tuple[int, int, set[tuple[int, int]], int]]:
    gh, gw = grid_shape
    ih, iw = input_shape
    ah, aw = aligned_shape
    top, bottom, left, right = cuts
    horiz_side = "top" if "top" in guide_sides else "bottom"
    vert_side = "left" if "left" in guide_sides else "right"

    out: list[tuple[int, int, set[tuple[int, int]], int]] = []
    r_limit = min(gh - ih - 1, gh - ah - 1)
    c_limit = min(gw - iw - 1, gw - aw - 1)
    if r_limit < 1 or c_limit < 1:
        return out
    for r0 in range(1, r_limit + 1):
        for c0 in range(1, c_limit + 1):
            distances = {
                horiz_side: r0 if horiz_side == "top" else gh - (r0 + ah),
                vert_side: c0 if vert_side == "left" else gw - (c0 + aw),
            }
            if any(v < 0 for v in distances.values()):
                continue
            other_side = vert_side if chosen_direction == horiz_side else horiz_side
            if not distances[chosen_direction] < distances[other_side]:
                continue
            copy_count = _copy_count_for_candidate(r0, c0, pure_shape, cuts, chosen_direction, grid_shape)
            if copy_count < 2:
                continue
            cells, count = _tile_cell_set(r0, c0, pure_shape, cuts, chosen_direction, grid_shape)
            if count < 2:
                continue
            out.append((r0, c0, cells, count))
    random.shuffle(out)
    return out


def _pick_active_sides() -> tuple[list[str], list[str]]:
    horiz_count = random.choice((1, 2))
    vert_count = random.choice((1, 2))
    horiz = random.sample(["top", "bottom"], horiz_count)
    vert = random.sample(["left", "right"], vert_count)
    return horiz, vert


def _choose_pairs(horiz: list[str], vert: list[str], n_objects: int) -> list[tuple[str, str]]:
    pairs = [(h, v) for h in horiz for v in vert]
    if len(pairs) == 1:
        return [pairs[0]] * n_objects
    chosen = [random.choice(pairs)]
    if n_objects >= 2:
        alt = [p for p in pairs if p != chosen[0]]
        chosen.append(random.choice(alt))
    while len(chosen) < n_objects:
        chosen.append(random.choice(pairs))
    random.shuffle(chosen)
    return chosen


def _choose_directions(pairs: list[tuple[str, str]]) -> list[str]:
    directions: list[str] = []
    if len(pairs) == 1:
        return [random.choice(pairs[0])]
    want_horiz = random.choice((True, False))
    used_other_axis = False
    for idx, pair in enumerate(pairs):
        if idx == 0:
            direction = pair[0] if want_horiz else pair[1]
        elif idx == 1:
            direction = pair[1] if want_horiz else pair[0]
            used_other_axis = True
        else:
            direction = random.choice(pair)
        directions.append(direction)
    if len(pairs) >= 2 and not used_other_axis:
        directions[1] = pairs[1][1] if directions[0] == pairs[0][0] else pairs[1][0]
    return directions


def _build_object_plan(
    grid_shape: tuple[int, int],
    side_to_color: dict[str, int],
    horiz_side: str,
    vert_side: str,
    chosen_direction: str,
    body_color: int,
) -> dict | None:
    body_h = random.randint(3, 5)
    body_w = random.randint(4, 6)
    aligned, pure = _make_aligned_patch(body_h, body_w, body_color, horiz_side, vert_side, side_to_color)
    guide_sides = {horiz_side: side_to_color[horiz_side], vert_side: side_to_color[vert_side]}
    variants = [(k, flip) for k in range(4) for flip in (False, True)]
    random.shuffle(variants)
    for k, flip in variants:
        if k == 0 and not flip and random.random() < 0.8:
            continue
        input_patch = _inverse_transform(aligned, k, flip)
        matches = _match_variants(input_patch, {v: k2 for k2, v in side_to_color.items()})
        if len(matches) != 1:
            continue
        match = matches[0]
        if not np.array_equal(match["aligned"], aligned):
            continue
        positions = _enumerate_positions(
            grid_shape,
            input_patch.shape,
            aligned.shape,
            pure.shape,
            match["cuts"],
            (horiz_side, vert_side),
            chosen_direction,
        )
        if positions:
            return {
                "input_patch": input_patch,
                "aligned": aligned,
                "pure": pure,
                "horiz_side": horiz_side,
                "vert_side": vert_side,
                "chosen_direction": chosen_direction,
                "guide_colors": guide_sides,
                "body_color": body_color,
                "cuts": match["cuts"],
                "positions": positions,
            }
    return None


def _draw_controls(grid: np.ndarray, side_to_color: dict[str, int]) -> None:
    h, w = grid.shape
    if "top" in side_to_color:
        grid[0, :] = int(side_to_color["top"])
    if "bottom" in side_to_color:
        grid[h - 1, :] = int(side_to_color["bottom"])
    if "left" in side_to_color:
        grid[:, 0] = int(side_to_color["left"])
    if "right" in side_to_color:
        grid[:, w - 1] = int(side_to_color["right"])


def _attempt_generate() -> np.ndarray | None:
    gh = random.randint(18, 24)
    gw = random.randint(18, 24)
    horiz_sides, vert_sides = _pick_active_sides()
    active_sides = horiz_sides + vert_sides
    shuffled_colors = random.sample(list(GUIDE_COLORS), len(active_sides))
    side_to_color = {side: color for side, color in zip(active_sides, shuffled_colors)}

    n_objects = random.choice((2, 3))
    pairs = _choose_pairs(horiz_sides, vert_sides, n_objects)
    directions = _choose_directions(pairs)
    body_palette = random.sample(list(BODY_COLORS), n_objects)

    plans: list[dict] = []
    for (horiz_side, vert_side), chosen_direction, body_color in zip(pairs, directions, body_palette):
        plan = _build_object_plan((gh, gw), side_to_color, horiz_side, vert_side, chosen_direction, body_color)
        if plan is None:
            return None
        plans.append(plan)

    placements: list[dict] = []
    input_boxes: list[tuple[int, int, int, int]] = []
    output_cells: set[tuple[int, int]] = set()
    for plan in sorted(plans, key=lambda item: len(item["positions"])):
        placed = None
        for r0, c0, cells, count in plan["positions"]:
            input_box = (r0, c0, r0 + plan["input_patch"].shape[0] - 1, c0 + plan["input_patch"].shape[1] - 1)
            union_box = (
                r0,
                c0,
                r0 + max(plan["input_patch"].shape[0], plan["aligned"].shape[0]) - 1,
                c0 + max(plan["input_patch"].shape[1], plan["aligned"].shape[1]) - 1,
            )
            if any(_boxes_overlap(_box_with_pad(union_box, 1), _box_with_pad(prev, 1)) for prev in input_boxes):
                continue
            if cells & output_cells:
                continue
            placed = {
                **plan,
                "r0": r0,
                "c0": c0,
                "input_box": input_box,
                "union_box": union_box,
                "output_cells": cells,
                "copy_count": count,
            }
            break
        if placed is None:
            return None
        placements.append(placed)
        input_boxes.append(placed["union_box"])
        output_cells |= placed["output_cells"]

    grid = np.zeros((gh, gw), dtype=int)
    _draw_controls(grid, side_to_color)
    for placed in placements:
        patch = placed["input_patch"]
        r0 = placed["r0"]
        c0 = placed["c0"]
        for pr in range(patch.shape[0]):
            for pc in range(patch.shape[1]):
                val = int(patch[pr, pc])
                if val != 0:
                    grid[r0 + pr, c0 + pc] = val

    controls, objects, _ = _analyze_grid(grid)
    if len(objects) != n_objects:
        return None
    if controls != {side: side_to_color[side] for side in active_sides}:
        return None
    if any(obj["copy_count"] < 2 for obj in objects):
        return None
    out = main(grid)
    if out.shape != grid.shape:
        return None
    if any(int(out[0, c]) in GUIDE_COLORS for c in range(gw)):
        return None
    if any(int(out[gh - 1, c]) in GUIDE_COLORS for c in range(gw)):
        return None
    if any(int(out[r, 0]) in GUIDE_COLORS for r in range(gh)):
        return None
    if any(int(out[r, gw - 1]) in GUIDE_COLORS for r in range(gh)):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
