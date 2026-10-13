from __future__ import annotations

from functools import lru_cache
import random

import numpy as np


BG = 0
GEN_RETRIES = 400
INPUT_MIN_SIDE = 14
INPUT_MAX_SIDE = 20
INPUT_PADDING = 2
COLOR_POOL = tuple(range(1, 10))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_nonzero(arr: np.ndarray) -> np.ndarray:
    coords = np.argwhere(arr != BG)
    assert coords.size > 0, "expected a nonempty sprite"
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max()) + 1
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max()) + 1
    return arr[r0:r1, c0:c1].copy()


def _mask_from_coords(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        out[r - r0, c - c0] = 1
    return out


def _swap_colors(arr: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = arr.copy()
    out[arr == int(color_a)] = int(color_b)
    out[arr == int(color_b)] = int(color_a)
    return out


def _border_cycle(height: int, width: int) -> tuple[tuple[int, int], ...]:
    coords: list[tuple[int, int]] = []
    for c in range(width):
        coords.append((0, c))
    for r in range(1, height):
        coords.append((r, width - 1))
    for c in range(width - 2, -1, -1):
        coords.append((height - 1, c))
    for r in range(height - 2, 0, -1):
        coords.append((r, 0))
    return tuple(coords)


def _border_set(height: int, width: int) -> frozenset[tuple[int, int]]:
    return frozenset(
        (r, c)
        for r in range(height)
        for c in range(width)
        if r in (0, height - 1) or c in (0, width - 1)
    )


def _shape_info(coords: tuple[tuple[int, int], ...]) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    rel = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "h": r1 - r0 + 1,
        "w": c1 - c0 + 1,
        "rel": rel,
    }


def _balanced_frame_dims(total_pixels: int) -> tuple[int, int]:
    assert total_pixels >= 8 and total_pixels % 2 == 0
    summed = total_pixels // 2 + 2
    best: tuple[int, int, int] | None = None
    for height in range(3, summed + 1):
        width = summed - height
        if width < height or width < 3:
            continue
        diff = width - height
        if best is None or diff < best[0]:
            best = (diff, height, width)
    assert best is not None, "perimeter must define a valid rectangular frame"
    return best[1], best[2]


@lru_cache(maxsize=None)
def _compositions(total: int, parts: int, lo: int = 1, hi: int = 7) -> tuple[tuple[int, ...], ...]:
    out: list[tuple[int, ...]] = []

    def rec(remaining: int, slots: int, prefix: list[int]) -> None:
        if slots == 1:
            if lo <= remaining <= hi:
                out.append(tuple(prefix + [remaining]))
            return
        low = max(lo, remaining - hi * (slots - 1))
        high = min(hi, remaining - lo * (slots - 1))
        for value in range(low, high + 1):
            rec(remaining - value, slots - 1, prefix + [value])

    rec(total, parts, [])
    return tuple(out)


def _bbox_diagonal_adjacent(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> bool:
    ar0, ac0, ar1, ac1 = bbox_a
    br0, bc0, br1, bc1 = bbox_b
    return (
        (br0 - ar1, bc0 - ac1) == (1, 1)
        or (ar0 - br1, ac0 - bc1) == (1, 1)
        or (br0 - ar1, bc1 - ac0) == (1, -1)
        or (ar0 - br1, ac1 - bc0) == (1, -1)
    )


def _stack_options(upper_mask: np.ndarray, lower_mask: np.ndarray) -> list[dict]:
    upper_h, upper_w = upper_mask.shape
    lower_h, lower_w = lower_mask.shape
    total_h = upper_h + lower_h
    if upper_h != lower_h or not (4 <= total_h <= 8):
        return []

    upper_bottom = set(np.flatnonzero(upper_mask[-1]))
    lower_top = set(np.flatnonzero(lower_mask[0]))
    options: list[dict] = []

    for upper_x in range(-4, 5):
        for lower_x in range(-4, 5):
            min_x = min(upper_x, lower_x)
            max_x = max(upper_x + upper_w, lower_x + lower_w)
            width = max_x - min_x
            if not (2 <= width <= 4):
                continue

            upper_col = upper_x - min_x
            lower_col = lower_x - min_x
            shifted_upper = {upper_col + c for c in upper_bottom}
            shifted_lower = {lower_col + c for c in lower_top}
            if not (shifted_upper & shifted_lower):
                continue

            options.append(
                {
                    "upper_x": upper_col,
                    "lower_x": lower_col,
                    "width": width,
                    "height": total_h,
                }
            )
    return options


def _stacked_mask(upper_mask: np.ndarray, lower_mask: np.ndarray, option: dict) -> np.ndarray:
    out = np.zeros((int(option["height"]), int(option["width"])), dtype=int)
    upper_x = int(option["upper_x"])
    lower_x = int(option["lower_x"])
    upper_h, upper_w = upper_mask.shape
    lower_h, lower_w = lower_mask.shape

    upper_view = out[:upper_h, upper_x : upper_x + upper_w]
    lower_view = out[upper_h : upper_h + lower_h, lower_x : lower_x + lower_w]
    upper_view[upper_mask != 0] = 1
    lower_view[lower_mask != 0] = 1
    return out


def _colorize_object(mask: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=int)
    mode = random.choice(("checker", "rows", "cols", "hybrid"))
    phase_a = random.randint(0, 1)
    phase_b = random.randint(0, 1)
    split = mask.shape[0] // 2

    for r, c in np.argwhere(mask != 0):
        rr = int(r)
        cc = int(c)
        if mode == "checker":
            bit = (rr + cc + phase_a) % 2
        elif mode == "rows":
            bit = (rr + phase_a) % 2
        elif mode == "cols":
            bit = (cc + phase_a) % 2
        else:
            bit = (cc + phase_a) % 2 if rr < split else (rr + phase_b) % 2
        out[rr, cc] = color_a if bit == 0 else color_b

    fg = np.argwhere(mask != 0)
    values = {int(out[r, c]) for r, c in fg}
    if len(values) == 1 and fg.size > 0:
        rr, cc = map(int, fg[len(fg) // 2])
        out[rr, cc] = color_b if int(out[rr, cc]) == int(color_a) else color_a

    for rows in (slice(0, split), slice(split, mask.shape[0])):
        coords = np.argwhere(mask[rows] != 0)
        if coords.shape[0] < 2:
            continue
        vals = {int(out[rows][r, c]) for r, c in coords}
        if len(vals) == 1 and random.random() < 0.75:
            rr, cc = map(int, coords[-1])
            target = out[rows][rr, cc]
            out_rows = out[rows].copy()
            out_rows[rr, cc] = color_b if int(target) == int(color_a) else color_a
            out[rows] = out_rows

    return out


def _render_on_canvas(
    canvas: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
) -> None:
    sub = canvas[top : top + patch.shape[0], left : left + patch.shape[1]]
    sub[patch != BG] = patch[patch != BG]


def _can_place_patch(
    occupied: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    padding: int,
) -> bool:
    h, w = patch.shape
    if top < 0 or left < 0 or top + h > occupied.shape[0] or left + w > occupied.shape[1]:
        return False

    coords = np.argwhere(patch != BG)
    if coords.size == 0:
        return False
    r0 = max(0, top + int(coords[:, 0].min()) - padding)
    r1 = min(occupied.shape[0], top + int(coords[:, 0].max()) + padding + 1)
    c0 = max(0, left + int(coords[:, 1].min()) - padding)
    c1 = min(occupied.shape[1], left + int(coords[:, 1].max()) + padding + 1)
    return not occupied[r0:r1, c0:c1].any()


@lru_cache(maxsize=None)
def _generator_plans() -> tuple[dict, ...]:
    plans: list[dict] = []
    pairings = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))

    for perimeter in range(12, 25, 2):
        height, width = _balanced_frame_dims(perimeter)
        cycle = _border_cycle(height, width)
        for start in range(perimeter):
            for lengths in _compositions(perimeter, 4):
                pos = start
                arcs: list[dict] = []
                ok = True
                for length in lengths:
                    coords = tuple(cycle[(pos + i) % perimeter] for i in range(length))
                    pos += length
                    arc = _shape_info(coords)
                    if arc["h"] > 4 or arc["w"] > 4:
                        ok = False
                        break
                    arcs.append(arc)
                if not ok:
                    continue

                for pairing in pairings:
                    pair_data: list[dict] = []
                    for a, b in pairing:
                        if not _bbox_diagonal_adjacent(arcs[a]["bbox"], arcs[b]["bbox"]):
                            break

                        mask_a = _mask_from_coords(arcs[a]["coords"])
                        mask_b = _mask_from_coords(arcs[b]["coords"])
                        ab = _stack_options(mask_a, mask_b)
                        ba = _stack_options(mask_b, mask_a)
                        if ab:
                            pair_data.append(
                                {
                                    "arc_indices": (a, b),
                                    "upper_arc": a,
                                    "lower_arc": b,
                                    "options": ab,
                                }
                            )
                        elif ba:
                            pair_data.append(
                                {
                                    "arc_indices": (a, b),
                                    "upper_arc": b,
                                    "lower_arc": a,
                                    "options": ba,
                                }
                            )
                        else:
                            break
                    else:
                        plans.append(
                            {
                                "frame": (height, width),
                                "lengths": lengths,
                                "start": start,
                                "arcs": tuple(arcs),
                                "pairs": tuple(pair_data),
                            }
                        )
    assert plans, "expected at least one valid generator plan"
    return tuple(plans)


def _extract_pairs(grid: np.ndarray) -> tuple[list[dict], tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    fg_colors = sorted(int(v) for v in np.unique(g) if int(v) != BG)
    assert len(fg_colors) == 2, "expected exactly two foreground colors"
    color_a, color_b = fg_colors

    comps = _components(g != BG)
    assert 2 <= len(comps) <= 4, "expected two to four separated foreground objects"
    comps.sort(key=lambda coords: _bbox(coords))

    pairs: list[dict] = []
    total_pixels = 0
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert 4 <= height <= 8 and height % 2 == 0, "object bbox height must be even and between 4 and 8"
        assert 2 <= width <= 4, "object bbox width must be between 2 and 4"

        split_row = r0 + height // 2
        upper_cells = [(r, c) for r, c in coords if r < split_row]
        lower_cells = [(r, c) for r, c in coords if r >= split_row]
        assert upper_cells and lower_cells, "both split halves must be nonempty"

        upper_arr = np.zeros((split_row - r0, width), dtype=int)
        lower_arr = np.zeros((r1 - split_row + 1, width), dtype=int)
        for r, c in upper_cells:
            upper_arr[r - r0, c - c0] = int(g[r, c])
        for r, c in lower_cells:
            lower_arr[r - split_row, c - c0] = int(g[r, c])

        upper_sprite = _crop_nonzero(upper_arr)
        lower_sprite = _crop_nonzero(lower_arr)
        upper_swapped = _swap_colors(upper_sprite, color_a, color_b)
        lower_swapped = _swap_colors(lower_sprite, color_a, color_b)
        total_pixels += int(np.count_nonzero(upper_swapped)) + int(np.count_nonzero(lower_swapped))

        pairs.append({"upper": upper_swapped, "lower": lower_swapped})

    return pairs, (color_a, color_b), total_pixels


@lru_cache(maxsize=None)
def _sprite_placements(shape_key: tuple[int, int, tuple[tuple[int, int], ...]], frame_h: int, frame_w: int) -> tuple[dict, ...]:
    sprite_h, sprite_w, rel = shape_key
    border = _border_set(frame_h, frame_w)
    placements: list[dict] = []

    for top in range(frame_h - sprite_h + 1):
        for left in range(frame_w - sprite_w + 1):
            coords = frozenset((top + r, left + c) for r, c in rel)
            if coords <= border:
                placements.append(
                    {
                        "top": top,
                        "left": left,
                        "cells": coords,
                        "bbox": (top, left, top + sprite_h - 1, left + sprite_w - 1),
                    }
                )
    return tuple(placements)


def _shape_key(arr: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    rel = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(arr != BG)))
    return arr.shape[0], arr.shape[1], rel


def _pair_candidate_placements(
    upper: np.ndarray,
    lower: np.ndarray,
    frame_h: int,
    frame_w: int,
) -> list[dict]:
    upper_places = _sprite_placements(_shape_key(upper), frame_h, frame_w)
    lower_places = _sprite_placements(_shape_key(lower), frame_h, frame_w)
    candidates: list[dict] = []

    upper_pixels = tuple((int(r), int(c), int(upper[r, c])) for r, c in np.argwhere(upper != BG))
    lower_pixels = tuple((int(r), int(c), int(lower[r, c])) for r, c in np.argwhere(lower != BG))

    for up in upper_places:
        for lp in lower_places:
            if up["cells"] & lp["cells"]:
                continue
            if not _bbox_diagonal_adjacent(up["bbox"], lp["bbox"]):
                continue
            paint: list[tuple[int, int, int]] = []
            for r, c, color in upper_pixels:
                paint.append((up["top"] + r, up["left"] + c, color))
            for r, c, color in lower_pixels:
                paint.append((lp["top"] + r, lp["left"] + c, color))
            candidates.append({"cells": up["cells"] | lp["cells"], "paint": tuple(paint)})
    return candidates


def _solve_outputs(grid: np.ndarray) -> list[np.ndarray]:
    pairs, _colors, total_pixels = _extract_pairs(grid)
    frame_h, frame_w = _balanced_frame_dims(total_pixels)
    assert 2 * (frame_h + frame_w) - 4 == total_pixels
    border = _border_set(frame_h, frame_w)

    pair_options = [
        _pair_candidate_placements(pair["upper"], pair["lower"], frame_h, frame_w)
        for pair in pairs
    ]
    assert all(pair_options), "every source pair must have at least one frame placement"

    order = sorted(range(len(pair_options)), key=lambda idx: len(pair_options[idx]))
    results: dict[tuple[int, ...], np.ndarray] = {}

    def search(k: int, used: frozenset[tuple[int, int]], placed: list[dict]) -> None:
        if k == len(order):
            if used != border:
                return
            out = np.zeros((frame_h, frame_w), dtype=int)
            for item in placed:
                for r, c, color in item["paint"]:
                    assert out[r, c] == BG
                    out[r, c] = color
            results[tuple(map(int, out.ravel()))] = out
            return

        idx = order[k]
        remaining = len(order) - k - 1
        for option in pair_options[idx]:
            if option["cells"] & used:
                continue
            new_used = used | option["cells"]
            if len(border) - len(new_used) < remaining * 2:
                continue
            search(k + 1, new_used, placed + [option])

    search(0, frozenset(), [])
    assert results, "expected at least one valid frame assembly"
    return list(results.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each shared-color bicolor object, swap the two colors in both halves, and translate the halves into a complete rectangular frame.

    Input structure:
    - The background is black.
    - The foreground contains separated 4-connected bicolor objects, all using the same two non-black colors.
    - Each object has an even-height 2..4-column bounding box; splitting its bbox at the horizontal midpoint yields one upper and one lower sprite.

    Rule:
    - For every object, split its foreground cells into upper and lower halves at the bbox midpoint.
    - Swap the two foreground colors inside each half.
    - Let P be the total number of foreground pixels across all swapped halves, choose the unique most-balanced frame size whose one-pixel border has P cells, and place every half by translation only onto that frame border.
    - Cover every border pixel exactly once, keep the frame interior black, and require the two halves from the same source object to occupy diagonally adjacent tight-bbox corners in the final placement.

    Key insight:
    The output is not arranged near the input objects: the hidden perimeter length gives the frame size, while the split halves become independent sprites whose only legal global assembly is the balanced frame cover that also preserves their source-pair diagonal relation.
    """
    outputs = _solve_outputs(np.asarray(grid, dtype=int))
    assert len(outputs) == 1, "the frame assembly must determine a unique output"
    return outputs[0]


def _place_generated_objects(objects: list[np.ndarray], canvas_h: int, canvas_w: int) -> np.ndarray | None:
    canvas = np.zeros((canvas_h, canvas_w), dtype=int)
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)
    order = list(range(len(objects)))
    random.shuffle(order)

    for idx in order:
        patch = objects[idx]
        locations = [
            (r, c)
            for r in range(canvas_h - patch.shape[0] + 1)
            for c in range(canvas_w - patch.shape[1] + 1)
            if _can_place_patch(occupied, patch, r, c, INPUT_PADDING)
        ]
        if not locations:
            return None
        top, left = random.choice(locations)
        _render_on_canvas(canvas, patch, top, left)
        occ_view = occupied[top : top + patch.shape[0], left : left + patch.shape[1]]
        occ_view[patch != BG] = True

    return canvas


def _build_generated_case() -> tuple[np.ndarray, np.ndarray] | None:
    plan = random.choice(_generator_plans())
    frame_h, frame_w = map(int, plan["frame"])
    color_a, color_b = random.sample(COLOR_POOL, 2)

    objects: list[np.ndarray] = []
    arc_outputs: dict[int, np.ndarray] = {}

    for pair in random.sample(list(plan["pairs"]), len(plan["pairs"])):
        upper_arc = int(pair["upper_arc"])
        lower_arc = int(pair["lower_arc"])
        option = random.choice(list(pair["options"]))

        upper_mask = _mask_from_coords(plan["arcs"][upper_arc]["coords"])
        lower_mask = _mask_from_coords(plan["arcs"][lower_arc]["coords"])
        stacked_mask = _stacked_mask(upper_mask, lower_mask, option)
        colored_object = _colorize_object(stacked_mask, color_a, color_b)

        split = upper_mask.shape[0]
        upper_in = _crop_nonzero(colored_object[:split])
        lower_in = _crop_nonzero(colored_object[split:])

        assert np.array_equal((upper_in != BG).astype(int), upper_mask), "upper split must recover the planned sprite"
        assert np.array_equal((lower_in != BG).astype(int), lower_mask), "lower split must recover the planned sprite"

        arc_outputs[upper_arc] = _swap_colors(upper_in, color_a, color_b)
        arc_outputs[lower_arc] = _swap_colors(lower_in, color_a, color_b)
        objects.append(colored_object)

    canvas_h = random.randint(INPUT_MIN_SIDE, INPUT_MAX_SIDE)
    canvas_w = random.randint(INPUT_MIN_SIDE, INPUT_MAX_SIDE)
    canvas = _place_generated_objects(objects, canvas_h, canvas_w)
    if canvas is None:
        return None

    expected = np.zeros((frame_h, frame_w), dtype=int)
    border = _border_set(frame_h, frame_w)
    covered: set[tuple[int, int]] = set()
    for arc_idx, sprite in arc_outputs.items():
        r0, c0, _r1, _c1 = plan["arcs"][arc_idx]["bbox"]
        for r, c in np.argwhere(sprite != BG):
            rr = int(r0 + int(r))
            cc = int(c0 + int(c))
            assert (rr, cc) in border, "planned sprite placement must stay on the frame border"
            assert (rr, cc) not in covered, "planned sprites must not overlap"
            covered.add((rr, cc))
            expected[rr, cc] = int(sprite[r, c])
    assert covered == set(border), "planned sprites must cover the frame border exactly once"

    outputs = _solve_outputs(canvas)
    if len(outputs) != 1 or not np.array_equal(outputs[0], expected):
        return None
    return canvas, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_generated_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input for puzzle 8")
