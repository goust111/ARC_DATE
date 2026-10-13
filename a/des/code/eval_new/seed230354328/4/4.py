from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    mask = np.asarray(mask, dtype=bool)
    variants = []
    for rot in range(4):
        turned = np.rot90(mask, rot)
        variants.append(turned)
        variants.append(np.fliplr(turned))
    best = None
    for variant in variants:
        key = (tuple(map(int, variant.shape)), variant.tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _paste(out: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    target = out[r0 : r0 + h, c0 : c0 + w]
    colored = patch != int(bg)
    assert target.shape == patch.shape
    assert not bool(np.any((target != int(bg)) & colored))
    target[colored] = patch[colored]


def _component_records(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    records = []

    for start_r, start_c in np.argwhere(g != int(bg)):
        sr = int(start_r)
        sc = int(start_c)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((int(r), int(c)))
            for dr, dc in _NEIGHBORS4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] != int(bg):
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        arr = np.array(coords, dtype=int)
        r0, c0 = map(int, arr.min(axis=0))
        r1, c1 = map(int, arr.max(axis=0))
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        patch = np.full(mask.shape, int(bg), dtype=int)
        for r, c in coords:
            rr = int(r - r0)
            cc = int(c - c0)
            mask[rr, cc] = True
            patch[rr, cc] = int(g[r, c])

        rows_used = len({int(r) for r, _ in coords})
        cols_used = len({int(c) for _, c in coords})
        records.append(
            {
                "coords": tuple(sorted((int(r), int(c)) for r, c in coords)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "patch": patch,
                "shape_key": _canonical_shape_key(mask),
                "center": ((int(r0) + int(r1)) / 2.0, (int(c0) + int(c1)) / 2.0),
                "row_span": int(rows_used),
                "col_span": int(cols_used),
            }
        )

    records.sort(key=lambda rec: rec["bbox"])
    return records


def _single_axis(record: dict) -> str | None:
    if int(record["row_span"]) > int(record["col_span"]):
        return "v"
    if int(record["col_span"]) > int(record["row_span"]):
        return "h"
    return None


def _pair_axis(first: dict, second: dict) -> str:
    axis_a = _single_axis(first)
    axis_b = _single_axis(second)
    if axis_a == axis_b and axis_a is not None:
        return axis_a

    row_strength = max(int(first["row_span"]), int(second["row_span"]))
    col_strength = max(int(first["col_span"]), int(second["col_span"]))
    if row_strength > col_strength:
        return "v"
    if col_strength > row_strength:
        return "h"

    dr = abs(float(first["center"][0]) - float(second["center"][0]))
    dc = abs(float(first["center"][1]) - float(second["center"][1]))
    return "v" if dr >= dc else "h"


def _choose_primary(first: dict, second: dict, axis: str) -> tuple[dict, dict]:
    if axis == "v":
        key_a = (float(first["center"][0]), float(first["center"][1]))
        key_b = (float(second["center"][0]), float(second["center"][1]))
    else:
        key_a = (float(first["center"][1]), float(first["center"][0]))
        key_b = (float(second["center"][1]), float(second["center"][0]))
    return (first, second) if key_a <= key_b else (second, first)


def _adjacent(coords: tuple[tuple[int, int], ...], anchor: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        for dr, dc in _NEIGHBORS4:
            if (int(r + dr), int(c + dc)) in anchor:
                return True
    return False


def _find_translation(
    mover: dict,
    anchor: dict,
    axis: str,
    occupied_without_mover: np.ndarray,
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = map(int, shape)
    anchor_set = set(anchor["coords"])
    start_coords = tuple(mover["coords"])
    if _adjacent(start_coords, anchor_set):
        return (0, 0)

    if axis == "v":
        dr = -1 if float(mover["center"][0]) > float(anchor["center"][0]) else 1
        dc = 0
    else:
        dr = 0
        dc = -1 if float(mover["center"][1]) > float(anchor["center"][1]) else 1

    for step in range(1, h + w + 1):
        shifted = []
        for r, c in start_coords:
            nr = int(r + dr * step)
            nc = int(c + dc * step)
            if not (0 <= nr < h and 0 <= nc < w):
                return None
            if bool(occupied_without_mover[nr, nc]):
                return None
            shifted.append((nr, nc))
        shifted_t = tuple(shifted)
        if _adjacent(shifted_t, anchor_set):
            return (int(dr * step), int(dc * step))
    return None


def _solve(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    comps = _component_records(g, bg)
    assert comps, "expected at least one foreground object"

    groups: dict[tuple[tuple[int, int], bytes], list[dict]] = defaultdict(list)
    for comp in comps:
        groups[comp["shape_key"]].append(comp)
    assert all(len(group) == 2 for group in groups.values()), "each shape must appear exactly twice"

    occupied = g != int(bg)
    proposals = []
    meta_pairs = []
    shape_items = sorted(groups.items(), key=lambda item: item[0])
    for shape_key, pair in shape_items:
        first, second = pair
        axis = _pair_axis(first, second)
        primary, secondary = _choose_primary(first, second, axis)
        occ = occupied.copy()
        for r, c in secondary["coords"]:
            occ[int(r), int(c)] = False
        delta = _find_translation(secondary, primary, axis, occ, (h, w))
        moved = delta is not None and delta != (0, 0)
        blocked = delta is None
        meta_pairs.append(
            {
                "shape_key": shape_key,
                "axis": axis,
                "primary_bbox": primary["bbox"],
                "secondary_bbox": secondary["bbox"],
                "moved": bool(moved),
                "blocked": bool(blocked),
            }
        )
        if moved:
            proposals.append({"secondary": secondary, "delta": delta})

    conflict_ids = set()
    coord_to_ids: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, proposal in enumerate(proposals):
        dr, dc = map(int, proposal["delta"])
        for r, c in proposal["secondary"]["coords"]:
            coord_to_ids[(int(r + dr), int(c + dc))].append(idx)
    for ids in coord_to_ids.values():
        if len(ids) > 1:
            conflict_ids.update(ids)
    proposals = [proposal for idx, proposal in enumerate(proposals) if idx not in conflict_ids]

    out = g.copy()
    for proposal in proposals:
        for r, c in proposal["secondary"]["coords"]:
            out[int(r), int(c)] = int(bg)

    for proposal in proposals:
        secondary = proposal["secondary"]
        dr, dc = map(int, proposal["delta"])
        r0, c0, _r1, _c1 = map(int, secondary["bbox"])
        _paste(out, secondary["patch"], int(r0 + dr), int(c0 + dc), int(bg))

    if return_meta:
        return out, {"pairs": meta_pairs, "pair_count": len(meta_pairs)}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-shape objects and slide only the follower toward its shape-mate until they touch.

    Input structure:
    - bg = the most common color.
    - The non-bg pixels form an even number of 4-connected objects scattered on the canvas.
    - Objects come in shape pairs: each binary shape appears exactly twice up to rotation and reflection, and the two instances use different colors.
    - Paired objects are separated by background space and do not overlap.

    Rule:
    - Group objects by their rotation/flip-invariant binary mask.
    - For each pair, determine the shape's dominant axis: more occupied rows means vertical, more occupied columns means horizontal.
    - Along that dominant axis, the object with the smaller center coordinate is the primary object; the other is the follower.
    - Keep the primary fixed. Move the follower straight toward the primary along that same axis until the two objects become edge-adjacent without overlap, then erase the old follower position.
    - If the follower would leave the grid or overlap any other non-bg pixel before reaching that touching position, leave the whole pair unchanged.

    Key insight:
    - Shape matching finds the pairs, but the direction of movement is controlled by the paired shape's long-axis orientation together with which copy sits earlier on that axis.
    """
    return _solve(grid, return_meta=False)


def _paint(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _variant_for_axis(mask: np.ndarray, axis: str) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if axis == "v":
        return np.flipud(arr) if random.random() < 0.5 else arr.copy()
    return np.fliplr(arr) if random.random() < 0.5 else arr.copy()


def _grow_mask(axis: str, height: int, width: int, target_area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if axis == "v":
        col = random.randrange(width)
        for row in range(height):
            if row > 0:
                col = max(0, min(width - 1, col + random.choice([-1, 0, 1])))
            mask[row, col] = True
    else:
        row = random.randrange(height)
        for col in range(width):
            if col > 0:
                row = max(0, min(height - 1, row + random.choice([-1, 0, 1])))
            mask[row, col] = True

    while int(mask.sum()) < int(target_area):
        frontier = set()
        for r, c in np.argwhere(mask):
            for dr, dc in _NEIGHBORS4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(sorted(frontier))
        mask[nr, nc] = True
    return mask


def _random_mask(axis: str, used_keys: set[tuple[tuple[int, int], bytes]]) -> np.ndarray:
    for _ in range(200):
        if axis == "v":
            height = random.randint(4, 6)
            width = random.randint(2, min(4, height - 1))
        else:
            width = random.randint(4, 6)
            height = random.randint(2, min(4, width - 1))
        max_area = max(height + 1, min(height * width - 1, height + width + 3))
        target_area = random.randint(height + width // 2, max_area)
        mask = _grow_mask(axis, height, width, target_area)
        key = _canonical_shape_key(mask)
        if key not in used_keys:
            return mask
    raise RuntimeError(f"failed to create unique {axis}-dominant shape")


def _build_free_module(axis: str, bg: int, colors: tuple[int, int], used_keys: set[tuple[tuple[int, int], bytes]]):
    mask = _random_mask(axis, used_keys)
    used_keys.add(_canonical_shape_key(mask))

    first_mask = _variant_for_axis(mask, axis)
    second_mask = _variant_for_axis(mask, axis)
    first = _paint(first_mask, int(colors[0]), bg)
    second = _paint(second_mask, int(colors[1]), bg)

    if axis == "v":
        top_margin = random.randint(1, 2)
        bottom_margin = random.randint(1, 2)
        left_margin = random.randint(1, 2)
        right_margin = random.randint(1, 3)
        gap = random.randint(3, 6)
        h = int(first.shape[0])
        w = int(first.shape[1])
        patch = np.full((top_margin + h + gap + h + bottom_margin, left_margin + w + right_margin), bg, dtype=int)
        _paste(patch, first, int(top_margin), int(left_margin), bg)
        _paste(patch, second, int(top_margin + h + gap), int(left_margin), bg)
    else:
        top_margin = random.randint(1, 2)
        bottom_margin = random.randint(1, 3)
        left_margin = random.randint(1, 2)
        right_margin = random.randint(1, 2)
        gap = random.randint(3, 6)
        h = int(first.shape[0])
        w = int(first.shape[1])
        patch = np.full((top_margin + h + bottom_margin, left_margin + w + gap + w + right_margin), bg, dtype=int)
        _paste(patch, first, int(top_margin), int(left_margin), bg)
        _paste(patch, second, int(top_margin), int(left_margin + w + gap), bg)

    return patch


def _build_blocked_module(
    blocked_axis: str,
    bg: int,
    blocked_colors: tuple[int, int],
    blocker_colors: tuple[int, int],
    used_keys: set[tuple[tuple[int, int], bytes]],
):
    blocked_mask = _random_mask(blocked_axis, used_keys)
    used_keys.add(_canonical_shape_key(blocked_mask))
    blocker_axis = "h" if blocked_axis == "v" else "v"
    blocker_mask = _random_mask(blocker_axis, used_keys)
    used_keys.add(_canonical_shape_key(blocker_mask))

    blocked_primary = _paint(_variant_for_axis(blocked_mask, blocked_axis), int(blocked_colors[0]), bg)
    blocked_secondary = _paint(_variant_for_axis(blocked_mask, blocked_axis), int(blocked_colors[1]), bg)
    blocker_primary = _paint(_variant_for_axis(blocker_mask, blocker_axis), int(blocker_colors[0]), bg)
    blocker_secondary = _paint(_variant_for_axis(blocker_mask, blocker_axis), int(blocker_colors[1]), bg)

    if blocked_axis == "v":
        left_margin = random.randint(1, 2)
        top_margin = random.randint(1, 2)
        gap_top = random.randint(1, 2)
        gap_bottom = random.randint(1, 2)
        sub_gap = random.randint(3, 5)
        blocker_gap = random.randint(3, 5)
        h_a, w_a = map(int, blocked_primary.shape)
        h_b, w_b = map(int, blocker_primary.shape)
        r_a1 = int(top_margin)
        c_a = int(left_margin)
        r_b1 = int(r_a1 + h_a + gap_top)
        c_b1 = int(c_a)
        r_a2 = int(r_b1 + h_b + gap_bottom)
        c_b2 = int(c_b1 + w_b + blocker_gap)
        total_h = int(r_a2 + h_a + top_margin)
        total_w = int(max(c_a + w_a, c_b2 + w_b) + left_margin + 1)
        patch = np.full((total_h, total_w), bg, dtype=int)
        _paste(patch, blocked_primary, r_a1, c_a, bg)
        _paste(patch, blocked_secondary, r_a2, c_a, bg)
        _paste(patch, blocker_primary, r_b1, c_b1, bg)
        _paste(patch, blocker_secondary, r_b1, c_b2, bg)
    else:
        top_margin = random.randint(1, 2)
        left_margin = random.randint(1, 2)
        gap_left = random.randint(1, 2)
        gap_right = random.randint(1, 2)
        sub_gap = random.randint(3, 5)
        blocker_gap = random.randint(3, 5)
        h_a, w_a = map(int, blocked_primary.shape)
        h_b, w_b = map(int, blocker_primary.shape)
        r_a = int(top_margin)
        c_a1 = int(left_margin)
        r_b1 = int(r_a)
        c_b1 = int(c_a1 + w_a + gap_left)
        c_a2 = int(c_b1 + w_b + gap_right)
        r_b2 = int(r_b1 + h_b + blocker_gap)
        total_h = int(max(r_a + h_a, r_b2 + h_b) + top_margin + 1)
        total_w = int(c_a2 + w_a + left_margin)
        patch = np.full((total_h, total_w), bg, dtype=int)
        _paste(patch, blocked_primary, r_a, c_a1, bg)
        _paste(patch, blocked_secondary, r_a, c_a2, bg)
        _paste(patch, blocker_primary, r_b1, c_b1, bg)
        _paste(patch, blocker_secondary, r_b2, c_b1, bg)

    assert int(sub_gap) >= 3  # kept for branch balance and visual spacing
    return patch


def _choose_modules(bg: int, colors: list[int], used_keys: set[tuple[tuple[int, int], bytes]]) -> list[np.ndarray]:
    template = random.choice(
        [
            ("free", "v"),
            ("blocked", None),
            ("blocked", None, "free", None),
            ("free", "v", "free", "h", "free", None),
            ("blocked", None, "free", "v", "free", "h"),
        ]
    )

    modules = []
    color_idx = 0
    i = 0
    while i < len(template):
        kind = template[i]
        axis = template[i + 1]
        pair_colors_needed = 4 if kind == "blocked" else 2
        chosen = tuple(int(colors[color_idx + j]) for j in range(pair_colors_needed))
        color_idx += pair_colors_needed
        if kind == "free":
            actual_axis = random.choice(["v", "h"]) if axis is None else str(axis)
            modules.append(_build_free_module(actual_axis, bg, (chosen[0], chosen[1]), used_keys))
        else:
            blocked_axis = random.choice(["v", "h"]) if axis is None else str(axis)
            modules.append(
                _build_blocked_module(
                    blocked_axis,
                    bg,
                    (chosen[0], chosen[1]),
                    (chosen[2], chosen[3]),
                    used_keys,
                )
            )
        i += 2
    return modules


def _place_modules(bg: int, modules: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(int(module.shape[0]) for module in modules)
    max_w = max(int(module.shape[1]) for module in modules)
    min_side = max(14, max_h + 2, max_w + 2, 10 + 3 * len(modules))

    for _ in range(120):
        height = random.randint(min_side, 30)
        width = random.randint(min_side, 30)
        canvas = np.full((height, width), bg, dtype=int)
        rects: list[tuple[int, int, int, int]] = []
        ok = True
        for module in random.sample(modules, len(modules)):
            mh, mw = map(int, module.shape)
            placed = False
            for _ in range(200):
                r0 = random.randint(0, height - mh)
                c0 = random.randint(0, width - mw)
                rect = (r0, c0, r0 + mh - 1, c0 + mw - 1)
                if any(
                    not (
                        rect[2] < other[0] - 1
                        or rect[0] > other[2] + 1
                        or rect[3] < other[1] - 1
                        or rect[1] > other[3] + 1
                    )
                    for other in rects
                ):
                    continue
                _paste(canvas, module, int(r0), int(c0), bg)
                rects.append(rect)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return canvas
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        bg = random.randrange(10)
        used_keys: set[tuple[tuple[int, int], bytes]] = set()
        colors = random.sample([color for color in range(10) if color != bg], 8)
        modules = _choose_modules(bg, colors, used_keys)
        grid = _place_modules(bg, modules)
        if grid is None:
            continue
        try:
            _out, meta = _solve(grid, return_meta=True)
        except AssertionError:
            continue

        pair_count = int(meta["pair_count"])
        axes = {pair["axis"] for pair in meta["pairs"]}
        moved_count = sum(int(pair["moved"]) for pair in meta["pairs"])
        blocked_count = sum(int(pair["blocked"]) for pair in meta["pairs"])
        if pair_count < 2 or pair_count > 4:
            continue
        if axes != {"v", "h"}:
            continue
        if moved_count == 0:
            continue
        if blocked_count > 0 and moved_count + blocked_count != pair_count:
            continue
        return grid
    raise RuntimeError("failed to generate a valid seed230354328/4 sample")
