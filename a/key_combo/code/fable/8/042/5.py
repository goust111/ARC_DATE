from __future__ import annotations

import hashlib
import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500
_SEEN_OUTPUTS: set[bytes] = set()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return (
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    box = _bbox(np.argwhere(mask))
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), box


def _normalize_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_components(mask)) == 1


def _count_color_neighbors(grid: np.ndarray, r: int, c: int, color: int) -> tuple[int, bool, bool]:
    count = 0
    vertical = False
    horizontal = False
    h, w = grid.shape
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == color:
            count += 1
            if nr != r:
                vertical = True
            if nc != c:
                horizontal = True
    return count, vertical, horizontal


def _candidate_holes(sprite: np.ndarray, color: int, bg: int) -> list[tuple[int, int]]:
    h, w = sprite.shape
    holes: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(sprite[r, c]) != bg:
                continue
            count, vertical, horizontal = _count_color_neighbors(sprite, r, c, color)
            if count >= 3 and vertical and horizontal:
                holes.append((r, c))
    return holes


def _is_rectangular_frame(comp: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    coords = {tuple(map(int, rc)) for rc in comp}
    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    if coords != expected:
        return None
    return r0, c0, r1, c1


def _parse_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    bg = _get_bg(g)

    color_components: dict[int, list[np.ndarray]] = {}
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        comps = _components(g == color)
        if comps:
            color_components[int(color)] = comps

    frame = None
    frame_color = None
    frame_comp = None
    for color, comps in color_components.items():
        for comp in comps:
            box = _is_rectangular_frame(comp)
            if box is not None:
                if frame is not None:
                    raise AssertionError("expected exactly one hollow rectangular frame")
                frame = box
                frame_color = color
                frame_comp = comp
    if frame is None or frame_color is None or frame_comp is None:
        raise AssertionError("missing hollow frame")

    fr0, fc0, fr1, fc1 = frame
    interior = (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1)
    ir0, ic0, ir1, ic1 = interior
    if ir0 > ir1 or ic0 > ic1:
        raise AssertionError("frame interior must be non-empty")

    sockets: dict[int, tuple[int, int]] = {}
    sprites: dict[int, dict[str, object]] = {}
    for color, comps in color_components.items():
        if color == frame_color:
            if len(comps) != 1:
                raise AssertionError("frame color must only appear on the frame")
            continue
        for comp in comps:
            coords = [tuple(map(int, rc)) for rc in comp]
            inside = all(ir0 <= r <= ir1 and ic0 <= c <= ic1 for r, c in coords)
            outside = all(not (fr0 <= r <= fr1 and fc0 <= c <= fc1) for r, c in coords)
            if inside:
                if len(coords) != 1:
                    raise AssertionError("each socket must be a single isolated pixel")
                if color in sockets:
                    raise AssertionError("socket colors must be unique")
                sockets[color] = coords[0]
            elif outside:
                if color in sprites:
                    raise AssertionError("each sprite color must form one outside component")
                mask = np.zeros_like(g, dtype=bool)
                mask[comp[:, 0], comp[:, 1]] = True
                crop, box = _crop_mask(mask)
                sprite = np.full(crop.shape, bg, dtype=int)
                sprite[crop] = color
                sprites[color] = {
                    "sprite": sprite,
                    "bbox": box,
                    "coords": comp.copy(),
                }
            else:
                raise AssertionError("non-frame foreground must be either a socket or an outside sprite")

    if not sockets or set(sockets) != set(sprites):
        raise AssertionError("socket and sprite colors must match bijectively")

    for color_a, (ra, ca) in sockets.items():
        for color_b, (rb, cb) in sockets.items():
            if color_a >= color_b:
                continue
            if max(abs(ra - rb), abs(ca - cb)) <= 1:
                raise AssertionError("sockets must stay visually isolated")

    return {
        "grid": g,
        "bg": bg,
        "frame": frame,
        "frame_color": frame_color,
        "interior": interior,
        "sockets": sockets,
        "sprites": sprites,
    }


def _placement_cells(sprite: np.ndarray, top: int, left: int, color: int) -> list[tuple[int, int]]:
    coords = np.argwhere(sprite == color)
    return [(int(top + r), int(left + c)) for r, c in coords]


def _solve(parsed: dict[str, object], *, require_unique: bool = True) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    g = parsed["grid"]
    bg = int(parsed["bg"])
    ir0, ic0, ir1, ic1 = parsed["interior"]
    interior_area = (ir1 - ir0 + 1) * (ic1 - ic0 + 1)
    sockets: dict[int, tuple[int, int]] = parsed["sockets"]  # type: ignore[assignment]
    sprites: dict[int, dict[str, object]] = parsed["sprites"]  # type: ignore[assignment]

    options: list[dict[str, object]] = []
    for color in sorted(sockets):
        sprite = np.asarray(sprites[color]["sprite"], dtype=int)
        holes = _candidate_holes(sprite, color, bg)
        if not holes:
            raise AssertionError("each sprite must expose at least one plausible missing cell")
        socket_r, socket_c = sockets[color]
        placements = []
        for hr, hc in holes:
            top = socket_r - hr
            left = socket_c - hc
            cells = _placement_cells(sprite, top, left, color)
            in_bounds = all(ir0 <= r <= ir1 and ic0 <= c <= ic1 for r, c in cells)
            placements.append(
                {
                    "hole": (hr, hc),
                    "top": int(top),
                    "left": int(left),
                    "cells": cells,
                    "in_bounds": bool(in_bounds),
                }
            )
        feasible = [p for p in placements if p["in_bounds"]]
        if not feasible:
            raise AssertionError("at least one candidate placement must fit inside the frame")
        options.append(
            {
                "color": color,
                "sprite": sprite,
                "socket": sockets[color],
                "placements": feasible,
                "all_placements": placements,
            }
        )

    options.sort(key=lambda item: len(item["placements"]))  # prune early
    socket_cells = set(sockets.values())
    solutions: list[list[dict[str, object]]] = []

    def backtrack(idx: int, occupied: set[tuple[int, int]], chosen: list[dict[str, object]]) -> None:
        if require_unique and len(solutions) > 1:
            return
        if idx == len(options):
            if len(occupied) + len(socket_cells) == interior_area:
                solutions.append([dict(item) for item in chosen])
            return
        item = options[idx]
        for placement in item["placements"]:
            cells = placement["cells"]
            if any(cell in occupied or cell in socket_cells for cell in cells):
                continue
            cell_set = set(cells)
            if len(cell_set) != len(cells):
                continue
            new_occupied = occupied | cell_set
            chosen.append(
                {
                    "color": item["color"],
                    "hole": placement["hole"],
                    "top": placement["top"],
                    "left": placement["left"],
                    "cells": cells,
                }
            )
            backtrack(idx + 1, new_occupied, chosen)
            chosen.pop()

    backtrack(0, set(), [])
    if not solutions:
        raise AssertionError("no exact tiling solution found")
    if require_unique and len(solutions) != 1:
        raise AssertionError("expected a unique exact tiling solution")

    choice = solutions[0]
    out = g.copy()
    for sprite_info in sprites.values():
        coords = np.asarray(sprite_info["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = bg
    for picked in choice:
        color = int(picked["color"])
        for r, c in picked["cells"]:
            out[r, c] = color
    if np.any(out[ir0 : ir1 + 1, ic0 : ic1 + 1] == bg):
        raise AssertionError("assembled sprites and sockets must completely fill the frame interior")
    return out, options, choice


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each notched monochrome sprite so its missing cell docks onto the same-color socket inside the frame.

    Input structure:
    - The grid contains one large one-cell-thick hollow rectangular frame on the background.
    - Inside the frame are several isolated single-pixel sockets of distinct colors.
    - Outside the frame are same-color monochrome sprites, one per socket, each missing one internal cell and not rotated.

    Rule:
    - For each sprite, consider the plausible internal gap positions in its cropped shape.
    - Translate the sprite so one plausible gap lands on the same-color socket, then keep only the unique combination where all translated sprites stay inside the frame interior and, together with the sockets, exactly tile that interior without overlaps or gaps.
    - Erase the sprites from their original outside positions and leave the frame unchanged.

    Key insight:
    - The registration mark is negative space: the absent cell inside each sprite must be aligned to its matching socket, and the correct local alignment is confirmed by the global exact-tiling constraint.
    """
    parsed = _parse_grid(np.asarray(grid, dtype=int))
    out, _options, _choice = _solve(parsed, require_unique=True)
    return out


def _grow_partition(h: int, w: int, count: int) -> np.ndarray | None:
    min_h = 3
    min_w = 3
    rects = [(0, 0, h - 1, w - 1)]
    while len(rects) < count:
        candidates: list[tuple[int, str]] = []
        for idx, (r0, c0, r1, c1) in enumerate(rects):
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh >= 2 * min_h:
                candidates.append((idx, "h"))
            if ww >= 2 * min_w:
                candidates.append((idx, "v"))
        if not candidates:
            return None
        idx, orient = random.choice(candidates)
        r0, c0, r1, c1 = rects.pop(idx)
        if orient == "h":
            cut = random.randint(r0 + min_h - 1, r1 - min_h)
            rects.append((r0, c0, cut, c1))
            rects.append((cut + 1, c0, r1, c1))
        else:
            cut = random.randint(c0 + min_w - 1, c1 - min_w)
            rects.append((r0, c0, r1, cut))
            rects.append((r0, cut + 1, r1, c1))

    assign = np.full((h, w), -1, dtype=int)
    for rid, (r0, c0, r1, c1) in enumerate(rects):
        assign[r0 : r1 + 1, c0 : c1 + 1] = rid
    return assign


def _region_mask(assign: np.ndarray, region_id: int) -> np.ndarray:
    return assign == region_id


def _mutate_partition(assign: np.ndarray, steps: int) -> None:
    h, w = assign.shape
    region_ids = sorted(set(map(int, assign.ravel())))
    sizes = {rid: int(np.sum(assign == rid)) for rid in region_ids}
    min_keep = max(6, min(sizes.values()) // 2)
    for _ in range(steps):
        boundary = []
        for r in range(h):
            for c in range(w):
                src = int(assign[r, c])
                nbrs = {int(assign[nr, nc]) for nr, nc in _neighbors4(r, c) if 0 <= nr < h and 0 <= nc < w and int(assign[nr, nc]) != src}
                if nbrs:
                    boundary.append((r, c, src, random.choice(sorted(nbrs))))
        if not boundary:
            return
        r, c, src, dst = random.choice(boundary)
        if sizes[src] <= min_keep:
            continue
        src_mask = assign == src
        src_mask[r, c] = False
        if not _is_connected(src_mask):
            continue
        assign[r, c] = dst
        sizes[src] -= 1
        sizes[dst] += 1


def _removable_options(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    options: list[tuple[int, int]] = []
    for r, c in coords:
        r = int(r)
        c = int(c)
        if r in (r0, r1) or c in (c0, c1):
            continue
        count = 0
        vertical = False
        horizontal = False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                count += 1
                if nr != r:
                    vertical = True
                if nc != c:
                    horizontal = True
        if count < 3 or not vertical or not horizontal:
            continue
        shrunk = mask.copy()
        shrunk[r, c] = False
        if not _is_connected(shrunk):
            continue
        options.append((r, c))
    return options


def _build_output_partition() -> tuple[np.ndarray, list[np.ndarray], list[int], list[tuple[int, int]]] | None:
    for _ in range(120):
        interior_h = random.randint(8, 14)
        interior_w = random.randint(9, 16)
        count = random.choice((2, 3, 3, 4))
        assign = _grow_partition(interior_h, interior_w, count)
        if assign is None:
            continue
        _mutate_partition(assign, steps=random.randint(20, 45))

        regions = [_region_mask(assign, rid) for rid in range(count)]
        if any(np.sum(mask) < 8 for mask in regions):
            continue
        if any(not _is_connected(mask) for mask in regions):
            continue
        if len({_normalize_shape(np.argwhere(mask)) for mask in regions}) != count:
            continue

        hole_choices = [_removable_options(mask) for mask in regions]
        if any(not opts for opts in hole_choices):
            continue

        order = list(range(count))
        random.shuffle(order)
        chosen: list[tuple[int, int] | None] = [None] * count

        def pick(idx: int) -> bool:
            if idx == count:
                return True
            rid = order[idx]
            opts = hole_choices[rid][:]
            random.shuffle(opts)
            for cell in opts:
                if any(prev is not None and max(abs(cell[0] - prev[0]), abs(cell[1] - prev[1])) <= 1 for prev in chosen):
                    continue
                chosen[rid] = cell
                if pick(idx + 1):
                    return True
            chosen[rid] = None
            return False

        if not pick(0):
            continue

        colors = random.sample(PALETTE, count + 1)
        frame_color = colors[0]
        shape_colors = colors[1:]

        output = np.full((interior_h + 2, interior_w + 2), BG, dtype=int)
        output[0, :] = frame_color
        output[-1, :] = frame_color
        output[:, 0] = frame_color
        output[:, -1] = frame_color
        for rid, mask in enumerate(regions):
            output[1:-1, 1:-1][mask] = shape_colors[rid]

        hole_list = [(int(cell[0]), int(cell[1])) for cell in chosen if cell is not None]
        if len(hole_list) != count:
            continue
        return output, regions, shape_colors, hole_list
    return None


def _sprite_from_region(mask: np.ndarray, color: int, hole: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]], tuple[int, int]]:
    shrunk = mask.copy()
    hr, hc = hole
    shrunk[hr, hc] = False
    coords = np.argwhere(shrunk)
    r0, c0, r1, c1 = _bbox(coords)
    sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    shifted = [(int(r - r0), int(c - c0)) for r, c in coords]
    for r, c in shifted:
        sprite[r, c] = color
    return sprite, (r0, c0), shifted, (hr - r0, hc - c0)


def _place_bboxes(canvas_h: int, canvas_w: int, frame_box: tuple[int, int, int, int], sizes: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    blocked = np.zeros((canvas_h, canvas_w), dtype=bool)
    fr0, fc0, fr1, fc1 = frame_box
    r0 = max(0, fr0 - 1)
    c0 = max(0, fc0 - 1)
    r1 = min(canvas_h - 1, fr1 + 1)
    c1 = min(canvas_w - 1, fc1 + 1)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    placements: list[tuple[int, int]] = []
    for sh, sw in sizes:
        top = left = None
        for _ in range(200):
            cand_top = random.randint(1, canvas_h - sh - 1)
            cand_left = random.randint(1, canvas_w - sw - 1)
            rr0 = max(0, cand_top - 1)
            cc0 = max(0, cand_left - 1)
            rr1 = min(canvas_h, cand_top + sh + 1)
            cc1 = min(canvas_w, cand_left + sw + 1)
            if not blocked[rr0:rr1, cc0:cc1].any():
                top, left = cand_top, cand_left
                break
        if top is None or left is None:
            candidates = []
            for cand_top in range(1, canvas_h - sh):
                for cand_left in range(1, canvas_w - sw):
                    rr0 = max(0, cand_top - 1)
                    cc0 = max(0, cand_left - 1)
                    rr1 = min(canvas_h, cand_top + sh + 1)
                    cc1 = min(canvas_w, cand_left + sw + 1)
                    if not blocked[rr0:rr1, cc0:cc1].any():
                        candidates.append((cand_top, cand_left))
            if not candidates:
                return None
            top, left = random.choice(candidates)
        placements.append((int(top), int(left)))
        blocked[max(0, top - 1) : min(canvas_h, top + sh + 1), max(0, left - 1) : min(canvas_w, left + sw + 1)] = True
    return placements


def _construct_candidate() -> tuple[np.ndarray, np.ndarray, dict[str, object]] | None:
    base = _build_output_partition()
    if base is None:
        return None
    frame_sprite, regions, shape_colors, holes = base
    inner_h = frame_sprite.shape[0] - 2
    inner_w = frame_sprite.shape[1] - 2
    count = len(shape_colors)

    sprite_specs = []
    for rid, color in enumerate(shape_colors):
        sprite, region_origin, shifted_coords, true_hole = _sprite_from_region(regions[rid], color, holes[rid])
        candidates = _candidate_holes(sprite, color, BG)
        if true_hole not in candidates:
            return None
        sprite_specs.append(
            {
                "color": color,
                "sprite": sprite,
                "region_origin": region_origin,
                "coords": shifted_coords,
                "true_hole": true_hole,
                "candidates": candidates,
                "socket": holes[rid],
            }
        )

    locally_ambiguous = False
    for spec in sprite_specs:
        hr_true, hc_true = spec["true_hole"]
        socket_r, socket_c = spec["socket"]
        for hr, hc in spec["candidates"]:
            if (hr, hc) == (hr_true, hc_true):
                continue
            top = socket_r - hr
            left = socket_c - hc
            cells = [(top + r, left + c) for r, c in spec["coords"]]
            if all(0 <= r < inner_h and 0 <= c < inner_w for r, c in cells):
                locally_ambiguous = True
                break
        if locally_ambiguous:
            break
    if not locally_ambiguous:
        return None

    sprite_sizes = [spec["sprite"].shape for spec in sprite_specs]
    max_sprite_h = max(h for h, _w in sprite_sizes)
    max_sprite_w = max(w for _h, w in sprite_sizes)
    frame_h, frame_w = frame_sprite.shape

    min_canvas_h = max(frame_h + max_sprite_h + 4, 20)
    min_canvas_w = max(frame_w + max_sprite_w + 4, 20)
    if min_canvas_h > 30 or min_canvas_w > 30:
        return None

    for _ in range(80):
        canvas_h = random.randint(min_canvas_h, 30)
        canvas_w = random.randint(min_canvas_w, 30)
        frame_top = random.randint(2, canvas_h - frame_h - 2)
        frame_left = random.randint(2, canvas_w - frame_w - 2)
        frame_box = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)
        placements = _place_bboxes(canvas_h, canvas_w, frame_box, sprite_sizes)
        if placements is None:
            continue

        grid = np.full((canvas_h, canvas_w), BG, dtype=int)
        grid[frame_top, frame_left : frame_left + frame_w] = frame_sprite[0, :]
        grid[frame_top + frame_h - 1, frame_left : frame_left + frame_w] = frame_sprite[-1, :]
        grid[frame_top : frame_top + frame_h, frame_left] = frame_sprite[:, 0]
        grid[frame_top : frame_top + frame_h, frame_left + frame_w - 1] = frame_sprite[:, -1]
        for spec in sprite_specs:
            socket_r, socket_c = spec["socket"]
            grid[frame_top + 1 + socket_r, frame_left + 1 + socket_c] = spec["color"]
        for spec, (top, left) in zip(sprite_specs, placements):
            sprite = spec["sprite"]
            sh, sw = sprite.shape
            view = grid[top : top + sh, left : left + sw]
            if np.any((view != BG) & (sprite != BG)):
                break
            view[sprite != BG] = sprite[sprite != BG]
        else:
            expected = np.full_like(grid, BG)
            expected[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = frame_sprite
            for rid, color in enumerate(shape_colors):
                expected[frame_top + 1 : frame_top + 1 + inner_h, frame_left + 1 : frame_left + 1 + inner_w][regions[rid]] = color
            spec = {
                "frame_box": frame_box,
                "frame_sprite": frame_sprite,
                "regions": regions,
                "shape_colors": shape_colors,
                "holes": holes,
                "sprite_specs": sprite_specs,
                "placements": placements,
            }
            return grid, expected, spec
    return None


def _validate_candidate(grid: np.ndarray, expected: np.ndarray, spec: dict[str, object]) -> bool:
    try:
        parsed = _parse_grid(grid)
        actual, options, choice = _solve(parsed, require_unique=True)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    if len(choice) != len(spec["shape_colors"]):
        return False

    chosen_by_color = {int(item["color"]): tuple(item["hole"]) for item in choice}
    ambiguous_sprite = False
    locally_ambiguous = False
    for sprite_spec in spec["sprite_specs"]:
        color = int(sprite_spec["color"])
        true_hole = tuple(sprite_spec["true_hole"])
        if chosen_by_color.get(color) != true_hole:
            return False

        option = next(item for item in options if int(item["color"]) == color)
        holes = [tuple(p["hole"]) for p in option["placements"]]
        if true_hole not in holes:
            return False
        if len(holes) > 1:
            ambiguous_sprite = True
        socket_r, socket_c = sprite_spec["socket"]
        for hole in holes:
            if hole == true_hole:
                continue
            top = socket_r - hole[0]
            left = socket_c - hole[1]
            cells = [(top + r, left + c) for r, c in sprite_spec["coords"]]
            if all(0 <= r < expected.shape[0] and 0 <= c < expected.shape[1] for r, c in cells):
                locally_ambiguous = True
                break
    if not ambiguous_sprite or not locally_ambiguous:
        return False

    out_hash = hashlib.md5(expected.tobytes() + str(expected.shape).encode()).digest()
    if out_hash in _SEEN_OUTPUTS:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _construct_candidate()
        if candidate is None:
            continue
        grid, expected, spec = candidate
        if not _validate_candidate(grid, expected, spec):
            continue
        out_hash = hashlib.md5(expected.tobytes() + str(expected.shape).encode()).digest()
        _SEEN_OUTPUTS.add(out_hash)
        return grid
    raise RuntimeError("failed to generate a valid dock-and-tile puzzle")
