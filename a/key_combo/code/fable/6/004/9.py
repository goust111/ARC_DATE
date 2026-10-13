from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, neighbors4, paste


DIRS = ("up", "down", "left", "right")
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
PAIR_COUNT_RANGE = (2, 4)
RECT_H_RANGE = (4, 7)
RECT_W_RANGE = (4, 7)
MAX_ATTEMPTS = 500


def _normalize_coords(coords: np.ndarray | list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(list(coords), dtype=int)
    if arr.size == 0:
        return ()
    arr = arr - arr.min(axis=0, keepdims=True)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _bbox_from_norm(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    if not shape:
        return 0, 0
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _touch_side(mask: np.ndarray) -> str | None:
    sides = []
    if bool(mask[0, :].any()):
        sides.append("up")
    if bool(mask[-1, :].any()):
        sides.append("down")
    if bool(mask[:, 0].any()):
        sides.append("left")
    if bool(mask[:, -1].any()):
        sides.append("right")
    return sides[0] if len(sides) == 1 else None


def _paint_rect(grid: np.ndarray, top: int, left: int, h: int, w: int, color: int) -> None:
    grid[top : top + h, left : left + w] = int(color)


def _rect_bbox(top: int, left: int, h: int, w: int) -> tuple[int, int, int, int]:
    return int(top), int(left), int(top + h - 1), int(left + w - 1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0] - pad
        or b[2] + pad < a[0] - pad
        or a[3] + pad < b[1] - pad
        or b[3] + pad < a[1] - pad
    )


def _center_side(source_bbox: tuple[int, int, int, int], target_bbox: tuple[int, int, int, int]) -> str:
    sr = (source_bbox[0] + source_bbox[2]) / 2
    sc = (source_bbox[1] + source_bbox[3]) / 2
    tr = (target_bbox[0] + target_bbox[2]) / 2
    tc = (target_bbox[1] + target_bbox[3]) / 2
    dr = tr - sr
    dc = tc - sc
    if abs(dr) >= abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _component_infos(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = int(get_bg(grid))
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    infos: list[dict] = []
    for idx, comp in enumerate(comps):
        color = int(comp["color"])
        bbox = tuple(map(int, comp["bbox"]))
        patch = crop_bbox(grid, bbox)
        occ = np.argwhere(patch == color)
        missing_mask = patch == bg
        side = None
        missing_shape = None
        if bool(missing_mask.any()):
            side = _touch_side(missing_mask)
            if side is not None:
                missing_shape = _normalize_coords(np.argwhere(missing_mask))
        infos.append(
            {
                "idx": idx,
                "color": color,
                "bbox": bbox,
                "bbox_area": int(patch.shape[0] * patch.shape[1]),
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
                "area": int(comp["area"]),
                "shape": _normalize_coords(occ),
                "missing_shape": missing_shape,
                "side": side,
            }
        )
    return bg, infos


def _unique_pairing(infos: list[dict]) -> list[tuple[int, int]]:
    n = len(infos)
    edges: dict[int, list[int]] = {i: [] for i in range(n)}
    for i, a in enumerate(infos):
        if a["missing_shape"] is None:
            continue
        for j, b in enumerate(infos):
            if i == j:
                continue
            if (
                int(a["color"]) == int(b["color"])
                and int(a["bbox_area"]) > int(b["bbox_area"])
                and int(a["h"]) >= int(b["h"])
                and int(a["w"]) >= int(b["w"])
                and a["missing_shape"] == b["shape"]
            ):
                edges[i].append(j)

    solutions: set[tuple[tuple[int, int], ...]] = set()

    def dfs(unused: frozenset[int], pairs: list[tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return
        if not unused:
            solutions.add(tuple(sorted(pairs)))
            return
        i = min(unused)
        for j in edges.get(i, []):
            if j in unused:
                dfs(unused - {i, j}, pairs + [(i, j)])
        for j in unused:
            if j != i and i in edges.get(j, []):
                dfs(unused - {i, j}, pairs + [(j, i)])

    dfs(frozenset(range(n)), [])
    assert len(solutions) == 1, "expected a unique receiver-fragment pairing"
    result = list(next(iter(solutions)))
    assert len(result) * 2 == n, "every component must belong to exactly one pair"
    return result


def _destination(side: str, bbox: tuple[int, int, int, int], grid_shape: tuple[int, int]) -> tuple[int, int]:
    h = int(bbox[2] - bbox[0] + 1)
    w = int(bbox[3] - bbox[1] + 1)
    H, W = map(int, grid_shape)
    if side == "up":
        return 0, int(bbox[1])
    if side == "down":
        return int(H - h), int(bbox[1])
    if side == "left":
        return int(bbox[0]), 0
    assert side == "right"
    return int(bbox[0]), int(W - w)


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, infos = _component_infos(g)
    pairing = _unique_pairing(infos)

    out = np.full_like(g, bg)
    placements: list[dict] = []
    for receiver_idx, fragment_idx in pairing:
        receiver = infos[receiver_idx]
        fragment = infos[fragment_idx]
        side = receiver["side"]
        assert side in DIRS
        top, left = _destination(side, receiver["bbox"], g.shape)
        bbox = _rect_bbox(top, left, receiver["h"], receiver["w"])
        for prev in placements:
            assert not _boxes_overlap(bbox, prev["bbox"]), "moved rectangles must not overlap"
        _paint_rect(out, top, left, receiver["h"], receiver["w"], receiver["color"])
        placements.append(
            {
                "receiver": receiver,
                "fragment": fragment,
                "side": side,
                "bbox": bbox,
            }
        )
    return out, placements


def main(grid: np.ndarray) -> np.ndarray:
    """Fit each fragment back into its matching notch, then slide the completed rectangle to the edge named by that notch side.

    Input structure:
    - The grid has a dominant background color and several monochrome connected components.
    - For each color-pair relation, one larger component is a rectangle with a same-colored side notch, and one smaller component is the notch fragment translated elsewhere.
    - Each notch shape matches exactly one same-colored fragment under translation, and each completed rectangle can slide straight to one grid edge without conflicts.

    Rule:
    - Detect each receiver-fragment pair by complement fit: inside the receiver's bbox, the background cells form the translated shape of exactly one same-colored fragment.
    - Treat those missing cells as the patch that completes the receiver into a solid axis-aligned rectangle.
    - Move the completed rectangle toward the grid edge on the notch's touched side: top notch -> top edge, bottom notch -> bottom edge, left notch -> left edge, right notch -> right edge.

    Key insight:
    - The free fragment's original location is a decoy; only its shape identifies which notch it fills, and the notch side alone determines the final travel direction.
    """
    out, _ = _solve(grid)
    return out


def _grow_connected(shape_h: int, shape_w: int, side: str, *, require_self_notched: bool) -> tuple[tuple[int, int], ...] | None:
    area_max = shape_h * shape_w
    min_area = 2 if area_max >= 2 else 1
    area_choices = list(range(min_area, area_max + 1))
    random.shuffle(area_choices)
    side_cells = []
    for r in range(shape_h):
        for c in range(shape_w):
            if side == "up" and r == 0:
                side_cells.append((r, c))
            elif side == "down" and r == shape_h - 1:
                side_cells.append((r, c))
            elif side == "left" and c == 0:
                side_cells.append((r, c))
            elif side == "right" and c == shape_w - 1:
                side_cells.append((r, c))
    for area in area_choices:
        for _ in range(80):
            cells = {random.choice(side_cells)}
            while len(cells) < area:
                frontier = []
                for r, c in cells:
                    for nr, nc in neighbors4(r, c):
                        if 0 <= nr < shape_h and 0 <= nc < shape_w and (nr, nc) not in cells:
                            frontier.append((int(nr), int(nc)))
                if not frontier:
                    break
                cells.add(random.choice(frontier))
            if len(cells) != area:
                continue
            raw = tuple(sorted((int(r), int(c)) for r, c in cells))
            norm = _normalize_coords(raw)
            if require_self_notched:
                hh, ww = _bbox_from_norm(norm)
                patch = np.zeros((hh, ww), dtype=int)
                for r, c in norm:
                    patch[r, c] = 1
                side2 = _touch_side(patch == 0)
                if side2 is None:
                    continue
            return raw
    return None


def _sample_notch(rect_h: int, rect_w: int, side: str, *, require_self_notched: bool) -> tuple[tuple[int, int], ...] | None:
    for _ in range(120):
        cells: list[tuple[int, int]] = []
        if side in {"up", "down"}:
            span = random.randint(1, rect_w - 2)
            left = random.randint(1, rect_w - span - 1)
            max_depth = min(3, rect_h - 1)
            depths = [random.randint(1, max_depth) for _ in range(span)]
            for offset, depth in enumerate(depths):
                c = left + offset
                if side == "up":
                    for r in range(depth):
                        cells.append((r, c))
                else:
                    for r in range(rect_h - depth, rect_h):
                        cells.append((r, c))
        else:
            span = random.randint(1, rect_h - 2)
            top = random.randint(1, rect_h - span - 1)
            max_depth = min(3, rect_w - 1)
            depths = [random.randint(1, max_depth) for _ in range(span)]
            for offset, depth in enumerate(depths):
                r = top + offset
                if side == "left":
                    for c in range(depth):
                        cells.append((r, c))
                else:
                    for c in range(rect_w - depth, rect_w):
                        cells.append((r, c))
        shifted = tuple(sorted(set((int(r), int(c)) for r, c in cells)))
        if not shifted or len(shifted) >= rect_h * rect_w:
            continue
        if require_self_notched:
            norm = _normalize_coords(shifted)
            hh, ww = _bbox_from_norm(norm)
            patch = np.zeros((hh, ww), dtype=int)
            for r, c in norm:
                patch[r, c] = 1
            if _touch_side(patch == 0) is None:
                continue
        return shifted
    return None


def _make_receiver_patch(spec: dict, bg: int) -> np.ndarray:
    patch = np.full((spec["h"], spec["w"]), spec["color"], dtype=int)
    for r, c in spec["notch"]:
        patch[r, c] = int(bg)
    return patch


def _make_fragment_patch(spec: dict, bg: int) -> np.ndarray:
    frag = np.full(spec["fragment_shape"], bg, dtype=int)
    for r, c in spec["fragment_cells"]:
        frag[r, c] = int(spec["color"])
    return frag


def _place_output_rects(specs: list[dict], H: int, W: int) -> list[tuple[int, int]] | None:
    occupied: list[tuple[int, int, int, int]] = []
    positions = [None] * len(specs)
    order = list(range(len(specs)))
    order.sort(key=lambda i: -(specs[i]["h"] * specs[i]["w"]))
    for idx in order:
        spec = specs[idx]
        h, w, side = spec["h"], spec["w"], spec["side"]
        candidates = []
        if side == "up":
            for c in range(0, W - w + 1):
                candidates.append((0, c))
        elif side == "down":
            for c in range(0, W - w + 1):
                candidates.append((H - h, c))
        elif side == "left":
            for r in range(0, H - h + 1):
                candidates.append((r, 0))
        else:
            for r in range(0, H - h + 1):
                candidates.append((r, W - w))
        random.shuffle(candidates)
        chosen = None
        for top, left in candidates:
            bbox = _rect_bbox(top, left, h, w)
            if all(not _boxes_overlap(bbox, other, pad=1) for other in occupied):
                chosen = (top, left)
                occupied.append(bbox)
                break
        if chosen is None:
            return None
        positions[idx] = chosen
    return positions  # type: ignore[return-value]


def _candidate_input_positions(spec: dict, out_pos: tuple[int, int], H: int, W: int) -> list[tuple[int, int]]:
    top_out, left_out = out_pos
    h, w, side = spec["h"], spec["w"], spec["side"]
    options = []
    if side == "up":
        for shift in range(1, H - h + 1):
            options.append((shift, left_out))
    elif side == "down":
        for shift in range(1, H - h + 1):
            options.append((H - h - shift, left_out))
    elif side == "left":
        for shift in range(1, W - w + 1):
            options.append((top_out, shift))
    else:
        for shift in range(1, W - w + 1):
            options.append((top_out, W - w - shift))
    random.shuffle(options)
    options.sort(key=lambda rc: -_target_distance(spec["side"], rc, h, w, H, W))
    return options


def _sweep_bbox(spec: dict, in_pos: tuple[int, int], out_pos: tuple[int, int]) -> tuple[int, int, int, int]:
    in_box = _rect_bbox(in_pos[0], in_pos[1], spec["h"], spec["w"])
    out_box = _rect_bbox(out_pos[0], out_pos[1], spec["h"], spec["w"])
    return (
        min(in_box[0], out_box[0]),
        min(in_box[1], out_box[1]),
        max(in_box[2], out_box[2]),
        max(in_box[3], out_box[3]),
    )


def _target_distance(side: str, pos: tuple[int, int], h: int, w: int, H: int, W: int) -> int:
    top, left = pos
    bottom_gap = H - h - top
    right_gap = W - w - left
    if side == "up":
        return int(top)
    if side == "down":
        return int(bottom_gap)
    if side == "left":
        return int(left)
    return int(right_gap)


def _nearest_edge(side: str, pos: tuple[int, int], h: int, w: int, H: int, W: int) -> str:
    top, left = pos
    dists = {
        "up": int(top),
        "down": int(H - h - top),
        "left": int(left),
        "right": int(W - w - left),
    }
    return min(DIRS, key=lambda name: (dists[name], DIRS.index(name)))


def _choose_input_positions(specs: list[dict], out_positions: list[tuple[int, int]], H: int, W: int) -> list[tuple[int, int]] | None:
    order = list(range(len(specs)))
    order.sort(key=lambda i: len(_candidate_input_positions(specs[i], out_positions[i], H, W)))
    positions = [None] * len(specs)
    input_boxes: list[tuple[int, int, int, int]] = []
    sweep_boxes: list[tuple[int, int, int, int]] = []

    def dfs(k: int) -> bool:
        if k == len(order):
            return any(
                _nearest_edge(specs[i]["side"], positions[i], specs[i]["h"], specs[i]["w"], H, W) != specs[i]["side"]
                for i in range(len(specs))
            )
        idx = order[k]
        spec = specs[idx]
        for pos in _candidate_input_positions(spec, out_positions[idx], H, W):
            in_box = _rect_bbox(pos[0], pos[1], spec["h"], spec["w"])
            sweep = _sweep_bbox(spec, pos, out_positions[idx])
            if any(_boxes_overlap(in_box, other, pad=1) for other in input_boxes):
                continue
            if any(_boxes_overlap(sweep, other) for other in sweep_boxes):
                continue
            positions[idx] = pos
            input_boxes.append(in_box)
            sweep_boxes.append(sweep)
            if dfs(k + 1):
                return True
            input_boxes.pop()
            sweep_boxes.pop()
            positions[idx] = None
        return False

    if not dfs(0):
        return None
    return positions  # type: ignore[return-value]


def _candidate_fragment_locations(
    canvas: np.ndarray,
    patch: np.ndarray,
    receiver_bbox: tuple[int, int, int, int],
    bg: int,
    blocked_boxes: list[tuple[int, int, int, int]],
    lure_side: str | None = None,
) -> list[tuple[int, int]]:
    H, W = canvas.shape
    h, w = patch.shape
    occ = canvas != bg
    mask = patch != bg
    candidates = []
    for top in range(0, H - h + 1):
        for left in range(0, W - w + 1):
            if bool((occ[top : top + h, left : left + w] & mask).any()):
                continue
            rs = max(0, top - 1)
            re = min(H, top + h + 1)
            cs = max(0, left - 1)
            ce = min(W, left + w + 1)
            if bool(occ[rs:re, cs:ce].any()):
                continue
            frag_box = _rect_bbox(top, left, h, w)
            if any(_boxes_overlap(frag_box, box) for box in blocked_boxes):
                continue
            if lure_side is not None and _center_side(receiver_bbox, frag_box) != lure_side:
                continue
            candidates.append((top, left))
    return candidates


def _wrong_output_nearest(specs: list[dict], input_positions: list[tuple[int, int]], grid_shape: tuple[int, int], bg: int) -> np.ndarray:
    H, W = grid_shape
    out = np.full((H, W), bg, dtype=int)
    for spec, pos in zip(specs, input_positions):
        edge = _nearest_edge(spec["side"], pos, spec["h"], spec["w"], H, W)
        if edge == "up":
            top, left = 0, pos[1]
        elif edge == "down":
            top, left = H - spec["h"], pos[1]
        elif edge == "left":
            top, left = pos[0], 0
        else:
            top, left = pos[0], W - spec["w"]
        _paint_rect(out, top, left, spec["h"], spec["w"], spec["color"])
    return out


def _wrong_output_fragment_side(specs: list[dict], input_positions: list[tuple[int, int]], fragment_positions: list[tuple[int, int]], grid_shape: tuple[int, int], bg: int) -> np.ndarray:
    H, W = grid_shape
    out = np.full((H, W), bg, dtype=int)
    for spec, rpos, fpos in zip(specs, input_positions, fragment_positions):
        receiver_box = _rect_bbox(rpos[0], rpos[1], spec["h"], spec["w"])
        fragment_box = _rect_bbox(fpos[0], fpos[1], spec["fragment_shape"][0], spec["fragment_shape"][1])
        side = _center_side(receiver_box, fragment_box)
        if side == "up":
            top, left = 0, rpos[1]
        elif side == "down":
            top, left = H - spec["h"], rpos[1]
        elif side == "left":
            top, left = rpos[0], 0
        else:
            top, left = rpos[0], W - spec["w"]
        _paint_rect(out, top, left, spec["h"], spec["w"], spec["color"])
    return out


def _build_specs(n_pairs: int, palette: list[int]) -> list[dict] | None:
    directions = [random.choice(DIRS) for _ in range(n_pairs)]
    if n_pairs >= 2 and len(set(directions)) == 1:
        directions[random.randrange(n_pairs)] = random.choice([d for d in DIRS if d != directions[0]])
    specs = []
    forced_notched_fragment = random.randrange(n_pairs)
    for i in range(n_pairs):
        rect_h = random.randint(*RECT_H_RANGE)
        rect_w = random.randint(*RECT_W_RANGE)
        if directions[i] in {"up", "down"} and rect_h < 2:
            return None
        if directions[i] in {"left", "right"} and rect_w < 2:
            return None
        notch = _sample_notch(
            rect_h,
            rect_w,
            directions[i],
            require_self_notched=(i == forced_notched_fragment),
        )
        if notch is None:
            return None
        fragment_cells = _normalize_coords(notch)
        frag_h, frag_w = _bbox_from_norm(fragment_cells)
        specs.append(
            {
                "color": palette[i],
                "side": directions[i],
                "h": rect_h,
                "w": rect_w,
                "notch": notch,
                "fragment_cells": fragment_cells,
                "fragment_shape": (frag_h, frag_w),
            }
        )
    return specs


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    bg = random.randint(0, 9)
    palette = random.sample([c for c in range(10) if c != bg], n_pairs)

    specs = _build_specs(n_pairs, palette)
    if specs is None:
        return None

    out_positions = _place_output_rects(specs, H, W)
    if out_positions is None:
        return None

    input_positions = _choose_input_positions(specs, out_positions, H, W)
    if input_positions is None:
        return None

    out = np.full((H, W), bg, dtype=int)
    for spec, pos in zip(specs, out_positions):
        _paint_rect(out, pos[0], pos[1], spec["h"], spec["w"], spec["color"])

    inp = np.full((H, W), bg, dtype=int)
    for spec, pos in zip(specs, input_positions):
        receiver_patch = _make_receiver_patch(spec, bg)
        paste(inp, receiver_patch, pos[0], pos[1], transparent=bg)

    receiver_boxes = [_rect_bbox(pos[0], pos[1], spec["h"], spec["w"]) for spec, pos in zip(specs, input_positions)]
    fragment_positions = []
    for spec, pos in zip(specs, input_positions):
        fragment_patch = _make_fragment_patch(spec, bg)
        receiver_box = _rect_bbox(pos[0], pos[1], spec["h"], spec["w"])
        lure_options = [d for d in DIRS if d != spec["side"]]
        random.shuffle(lure_options)
        placed = None
        for lure_side in lure_options:
            candidates = _candidate_fragment_locations(
                inp,
                fragment_patch,
                receiver_box,
                bg,
                receiver_boxes,
                lure_side=lure_side,
            )
            if candidates:
                placed = (random.choice(candidates), lure_side)
                break
        if placed is None:
            fallback_candidates = [
                loc
                for loc in _candidate_fragment_locations(
                    inp,
                    fragment_patch,
                    receiver_box,
                    bg,
                    receiver_boxes,
                    lure_side=None,
                )
                if _center_side(receiver_box, _rect_bbox(loc[0], loc[1], fragment_patch.shape[0], fragment_patch.shape[1]))
                != spec["side"]
            ]
            if not fallback_candidates:
                return None
            fallback = random.choice(fallback_candidates)
            frag_box = _rect_bbox(fallback[0], fallback[1], fragment_patch.shape[0], fragment_patch.shape[1])
            if _center_side(receiver_box, frag_box) == spec["side"]:
                return None
            placed = (fallback, _center_side(receiver_box, frag_box))
        fragment_positions.append(placed[0])
        paste(inp, fragment_patch, placed[0][0], placed[0][1], transparent=bg)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_wrong_output_nearest(specs, input_positions, inp.shape, bg), out):
        return None
    if np.array_equal(_wrong_output_fragment_side(specs, input_positions, fragment_positions, inp.shape, bg), out):
        return None

    _, placements = _solve(inp)
    if len(placements) != n_pairs:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
