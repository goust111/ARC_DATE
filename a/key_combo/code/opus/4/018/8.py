from __future__ import annotations

from pathlib import Path
import itertools
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, find_components, get_bg


BG = Color.BLACK
DIVIDER = Color.GREY
MARKER = Color.YELLOW
PAIR_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.PINK, Color.ORANGE]
MAX_GENERATION_ATTEMPTS = 500
PROFILE_CYCLE = [
    ("double", "single"),
    ("none", "double"),
    ("single", "none"),
]
_PROFILE_INDEX = 0


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return False

    cells = {tuple(map(int, rc)) for rc in coords}
    stack = [next(iter(cells))]
    seen = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                stack.append(nxt)
    return len(seen) == len(cells)


def _build_shape_catalog() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for h, w in ((2, 2), (2, 3), (3, 2), (3, 3)):
        cells = h * w
        for area in range(3, min(6, cells) + 1):
            for combo in itertools.combinations(range(cells), area):
                mask = np.zeros((h, w), dtype=bool)
                for idx in combo:
                    mask[idx // w, idx % w] = True
                if not _is_connected(mask):
                    continue
                if not mask[0, :].any() or not mask[-1, :].any():
                    continue
                if not mask[:, 0].any() or not mask[:, -1].any():
                    continue
                sig = _mask_signature(mask)
                if sig not in seen:
                    seen.add(sig)
                    shapes.append(mask.copy())
    return shapes


SHAPE_CATALOG = _build_shape_catalog()


def _copy_object(obj: dict, dc: int = 0) -> dict:
    moved = dict(obj)
    moved["sprite_coords"] = np.asarray(obj["sprite_coords"], dtype=int).copy()
    moved["marker_coords"] = np.asarray(obj["marker_coords"], dtype=int).copy()
    moved["all_coords"] = np.asarray(obj["all_coords"], dtype=int).copy()
    moved["values"] = np.asarray(obj["values"], dtype=int).copy()
    if dc:
        moved["sprite_coords"][:, 1] += int(dc)
        if moved["marker_coords"].size:
            moved["marker_coords"][:, 1] += int(dc)
        moved["all_coords"][:, 1] += int(dc)
    return moved


def _coords_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.asarray(coords, dtype=int)}


def _object_margins(obj: dict, width: int) -> tuple[int, int]:
    cols = np.asarray(obj["all_coords"], dtype=int)[:, 1]
    return int(cols.min()), int(width - 1 - cols.max())


def _marker_candidates(shape: np.ndarray, side: str) -> list[tuple[int, int]]:
    sprite = {tuple(map(int, rc)) for rc in np.argwhere(shape)}
    h, w = shape.shape
    candidates: set[tuple[int, int]] = set()
    for r, c in sprite:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if (rr, cc) in sprite:
                continue
            if side == "left" and cc == w:
                continue
            if side == "right" and cc == -1:
                continue
            candidates.add((rr, cc))

    outer = []
    vertical = []
    for rr, cc in candidates:
        if side == "left" and cc == -1:
            outer.append((rr, cc))
        elif side == "right" and cc == w:
            outer.append((rr, cc))
        else:
            vertical.append((rr, cc))
    return outer + vertical


def _make_object(
    obj_id: int,
    *,
    side: str,
    color: int,
    shape: np.ndarray,
    top: int,
    left: int,
    marked: bool,
    role: str,
    pair_index: int | None,
    branch: str | None,
    occupied: set[tuple[int, int]],
    grid_shape: tuple[int, int],
) -> dict | None:
    H, W = grid_shape
    sprite_coords = np.argwhere(shape).astype(int)
    sprite_coords += np.array([int(top), int(left)], dtype=int)
    if np.any(sprite_coords[:, 0] < 0) or np.any(sprite_coords[:, 0] >= H):
        return None
    if np.any(sprite_coords[:, 1] < 0) or np.any(sprite_coords[:, 1] >= W):
        return None
    if any((int(r), int(c)) in occupied for r, c in sprite_coords):
        return None

    candidates = [None]
    if marked:
        candidates = _marker_candidates(shape, side)
        random.shuffle(candidates)

    for marker_rc in candidates:
        if marker_rc is None:
            marker_coords = np.zeros((0, 2), dtype=int)
            values = np.full(sprite_coords.shape[0], int(color), dtype=int)
            all_coords = sprite_coords.copy()
        else:
            marker_coords = np.array([[int(top + marker_rc[0]), int(left + marker_rc[1])]], dtype=int)
            mr, mc = map(int, marker_coords[0])
            if not (0 <= mr < H and 0 <= mc < W):
                continue
            if (mr, mc) in occupied:
                continue
            values = np.concatenate(
                [
                    np.full(sprite_coords.shape[0], int(color), dtype=int),
                    np.array([int(MARKER)], dtype=int),
                ]
            )
            all_coords = np.vstack([sprite_coords, marker_coords])

        colors = set(values.tolist()) - {int(MARKER)}
        if len(colors) != 1:
            continue
        obj = {
            "id": int(obj_id),
            "side": side,
            "color": int(color),
            "marked": bool(marked),
            "role": role,
            "pair_index": pair_index,
            "branch": branch,
            "sprite_coords": sprite_coords.copy(),
            "marker_coords": marker_coords.copy(),
            "all_coords": all_coords.copy(),
            "values": values.copy(),
        }
        return obj

    return None


def _paint_coords(out: np.ndarray, coords: np.ndarray, color: int) -> None:
    H, W = out.shape
    for r, c in np.asarray(coords, dtype=int):
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            out[rr, cc] = int(color)


def _find_divider_col(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    divider_cols = [c for c in range(g.shape[1]) if bool(np.all(g[:, c] == int(DIVIDER)))]
    assert len(divider_cols) == 1, "expected exactly one full-height grey divider column"
    return int(divider_cols[0])


def _extract_objects(grid: np.ndarray, divider_col: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    working = g.copy()
    working[:, int(divider_col)] = bg

    components = find_components(
        working,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert 4 <= len(components) <= 6, "expected 4-6 multicolor object components"

    objects: list[dict] = []
    for idx, comp in enumerate(components):
        coords = np.asarray(comp["coords"], dtype=int)
        values = working[coords[:, 0], coords[:, 1]].astype(int)
        colors = sorted(set(map(int, values)) - {int(MARKER)})
        assert len(colors) == 1, "each object must have exactly one sprite color"
        color = int(colors[0])
        sprite_coords = coords[values == color].astype(int)
        marker_coords = coords[values == int(MARKER)].astype(int)
        assert 3 <= sprite_coords.shape[0] <= 6, "sprite area must be 3-6"
        assert marker_coords.shape[0] <= 1, "expected at most one yellow marker per object"

        sprite_bbox = bbox_coords(sprite_coords)
        if sprite_bbox[3] < divider_col:
            side = "left"
        elif sprite_bbox[1] > divider_col:
            side = "right"
        else:
            raise AssertionError("sprite must lie entirely on one side of the divider")

        objects.append(
            {
                "id": int(idx),
                "side": side,
                "color": color,
                "marked": bool(marker_coords.shape[0]),
                "role": "unknown",
                "pair_index": None,
                "branch": None,
                "sprite_coords": sprite_coords.copy(),
                "marker_coords": marker_coords.copy(),
                "all_coords": coords.copy(),
                "values": values.copy(),
            }
        )

    left_colors = [obj["color"] for obj in objects if obj["side"] == "left"]
    right_colors = [obj["color"] for obj in objects if obj["side"] == "right"]
    assert 2 <= len(left_colors) <= 3 and 2 <= len(right_colors) <= 3
    assert len(set(left_colors)) == len(left_colors), "left-side sprite colors must be distinct"
    assert len(set(right_colors)) == len(right_colors), "right-side sprite colors must be distinct"
    return objects


def _occupancy(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    occ = np.full(shape, -1, dtype=int)
    for obj in objects:
        coords = np.asarray(obj["all_coords"], dtype=int)
        assert np.all(occ[coords[:, 0], coords[:, 1]] == -1), "objects must not overlap"
        occ[coords[:, 0], coords[:, 1]] = int(obj["id"])
    return occ


def _movable_ids(
    shape: tuple[int, int],
    objects: list[dict],
    occ: np.ndarray,
    *,
    dc: int,
    divider_col: int,
) -> set[int]:
    _h, w = shape
    movable = {int(obj["id"]) for obj in objects}
    by_id = {int(obj["id"]): obj for obj in objects}

    changed = True
    while changed:
        changed = False
        for obj_id in list(movable):
            coords = np.asarray(by_id[obj_id]["all_coords"], dtype=int)
            shifted_cols = coords[:, 1] + int(dc)
            if np.any(shifted_cols < 0) or np.any(shifted_cols >= w):
                movable.remove(obj_id)
                changed = True
                continue
            if np.any(shifted_cols == int(divider_col)):
                movable.remove(obj_id)
                changed = True
                continue
            labels = occ[coords[:, 0], shifted_cols]
            blockers = {int(label) for label in labels if int(label) not in {-1, obj_id}}
            if any(blocker not in movable for blocker in blockers):
                movable.remove(obj_id)
                changed = True
    return movable


def _slide_side(objects: list[dict], shape: tuple[int, int], *, dc: int, divider_col: int) -> list[dict]:
    moved = [_copy_object(obj) for obj in objects]
    width = shape[1]
    for _step in range(width):
        occ = _occupancy(shape, moved)
        movable = _movable_ids(shape, moved, occ, dc=dc, divider_col=divider_col)
        if not movable:
            return moved
        for obj in moved:
            if int(obj["id"]) in movable:
                obj["sprite_coords"][:, 1] += int(dc)
                if obj["marker_coords"].size:
                    obj["marker_coords"][:, 1] += int(dc)
                obj["all_coords"][:, 1] += int(dc)
    raise AssertionError("horizontal slide should settle within grid width")


def _sprite_bbox(obj: dict) -> tuple[int, int, int, int]:
    return bbox_coords(np.asarray(obj["sprite_coords"], dtype=int))


def _row_overlap(a: dict, b: dict) -> int:
    a0, _a1, a2, _a3 = _sprite_bbox(a)
    b0, _b1, b2, _b3 = _sprite_bbox(b)
    return max(0, min(int(a2), int(b2)) - max(int(a0), int(b0)) + 1)


def _pair_boundary_objects(left: list[dict], right: list[dict], divider_col: int) -> list[tuple[int, int]]:
    left_edge = [obj for obj in left if _sprite_bbox(obj)[3] == int(divider_col - 1)]
    right_edge = [obj for obj in right if _sprite_bbox(obj)[1] == int(divider_col + 1)]
    left_edge.sort(key=lambda obj: _sprite_bbox(obj))
    right_edge.sort(key=lambda obj: _sprite_bbox(obj))

    options: dict[int, list[tuple[int, int, int]]] = {}
    for l_obj in left_edge:
        lb = _sprite_bbox(l_obj)
        lc = (lb[0] + lb[2]) / 2.0
        for r_obj in right_edge:
            ov = _row_overlap(l_obj, r_obj)
            if ov <= 0:
                continue
            rb = _sprite_bbox(r_obj)
            rc = (rb[0] + rb[2]) / 2.0
            center_cost = int(abs(lc - rc) * 10)
            options.setdefault(int(l_obj["id"]), []).append((int(r_obj["id"]), int(ov), center_cost))

    left_ids = [int(obj["id"]) for obj in left_edge]
    option_map = {li: options.get(li, []) for li in left_ids}

    def rec(pos: int, used_right: frozenset[int]) -> tuple[tuple[int, int, int], tuple[tuple[int, int], ...]]:
        if pos == len(left_ids):
            return (0, 0, 0), ()

        li = left_ids[pos]
        best_score, best_pairs = rec(pos + 1, used_right)
        best_sig = tuple(best_pairs)

        for ri, overlap, center_cost in option_map[li]:
            if ri in used_right:
                continue
            tail_score, tail_pairs = rec(pos + 1, used_right | {ri})
            cand_score = (
                tail_score[0] + int(overlap),
                tail_score[1] + 1,
                tail_score[2] - int(center_cost),
            )
            cand_pairs = ((int(li), int(ri)),) + tail_pairs
            cand_sig = tuple(cand_pairs)
            if cand_score > best_score or (cand_score == best_score and cand_sig < best_sig):
                best_score, best_pairs, best_sig = cand_score, cand_pairs, cand_sig

        return best_score, best_pairs

    _score, pairs = rec(0, frozenset())
    return list(pairs)


def _mirror_coords(sprite_coords: np.ndarray, side: str) -> np.ndarray:
    coords = np.asarray(sprite_coords, dtype=int)
    out = coords.copy()
    if side == "left":
        c0 = int(coords[:, 1].min())
        out[:, 1] = 2 * c0 - 1 - coords[:, 1]
    else:
        c1 = int(coords[:, 1].max())
        out[:, 1] = 2 * c1 + 1 - coords[:, 1]
    return out


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert int(get_bg(g)) == int(BG), "expected black background"

    divider_col = _find_divider_col(g)
    objects = _extract_objects(g, divider_col)
    left_input = sorted(
        [_copy_object(obj) for obj in objects if obj["side"] == "left"],
        key=lambda obj: (_sprite_bbox(obj)[3], _sprite_bbox(obj)[0]),
        reverse=True,
    )
    right_input = sorted(
        [_copy_object(obj) for obj in objects if obj["side"] == "right"],
        key=lambda obj: (_sprite_bbox(obj)[1], _sprite_bbox(obj)[0]),
    )

    left_slid = _slide_side(left_input, g.shape, dc=1, divider_col=divider_col)
    right_slid = _slide_side(right_input, g.shape, dc=-1, divider_col=divider_col)

    slid_by_id = {int(obj["id"]): obj for obj in left_slid + right_slid}
    input_by_id = {int(obj["id"]): obj for obj in objects}
    pairs_idx = _pair_boundary_objects(left_slid, right_slid, divider_col)
    pairs: list[dict] = []
    paired_ids: set[int] = set()
    for li, ri in pairs_idx:
        left_obj = slid_by_id[int(li)]
        right_obj = slid_by_id[int(ri)]
        paired_ids.add(int(li))
        paired_ids.add(int(ri))
        left_marked = bool(left_obj["marked"])
        right_marked = bool(right_obj["marked"])
        if left_marked and right_marked:
            branch = "double"
        elif left_marked or right_marked:
            branch = "single"
        else:
            branch = "none"
        pairs.append(
            {
                "left_id": int(li),
                "right_id": int(ri),
                "left": left_obj,
                "right": right_obj,
                "branch": branch,
                "overlap": _row_overlap(left_obj, right_obj),
            }
        )

    analysis_objects = []
    for obj_id, slid_obj in slid_by_id.items():
        inp = input_by_id[int(obj_id)]
        delta = int(slid_obj["sprite_coords"][0, 1] - inp["sprite_coords"][0, 1])
        moved = _copy_object(slid_obj)
        moved["input_side"] = inp["side"]
        moved["slide_dc"] = delta
        analysis_objects.append(moved)

    return {
        "grid": g,
        "divider_col": int(divider_col),
        "objects": sorted(analysis_objects, key=lambda obj: int(obj["id"])),
        "pairs": pairs,
        "paired_ids": paired_ids,
    }


def _render_baseline(analysis: dict) -> np.ndarray:
    g = np.asarray(analysis["grid"], dtype=int)
    out = np.full_like(g, int(BG))
    out[:, int(analysis["divider_col"])] = int(DIVIDER)
    for obj in analysis["objects"]:
        _paint_coords(out, np.asarray(obj["sprite_coords"], dtype=int), int(obj["color"]))
    return out


def _render_output(analysis: dict) -> np.ndarray:
    g = np.asarray(analysis["grid"], dtype=int)
    out = np.full_like(g, int(BG))
    out[:, int(analysis["divider_col"])] = int(DIVIDER)

    objects = {int(obj["id"]): obj for obj in analysis["objects"]}
    final_colors = {obj_id: int(obj["color"]) for obj_id, obj in objects.items()}
    extras: list[tuple[np.ndarray, int]] = []

    for pair in analysis["pairs"]:
        left = objects[int(pair["left_id"])]
        right = objects[int(pair["right_id"])]
        branch = pair["branch"]
        if branch == "double":
            extras.append((_mirror_coords(left["sprite_coords"], "left"), int(left["color"])))
            extras.append((_mirror_coords(right["sprite_coords"], "right"), int(right["color"])))
        elif branch == "single":
            if left["marked"]:
                final_colors[int(right["id"])] = int(left["color"])
            else:
                final_colors[int(left["id"])] = int(right["color"])
        else:
            final_colors[int(left["id"])] = int(right["color"])
            final_colors[int(right["id"])] = int(left["color"])

    for obj_id, obj in objects.items():
        _paint_coords(out, np.asarray(obj["sprite_coords"], dtype=int), int(final_colors[obj_id]))

    for coords, color in extras:
        for r, c in np.asarray(coords, dtype=int):
            rr = int(r)
            cc = int(c)
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and cc != int(analysis["divider_col"]):
                out[rr, cc] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the left and right sprites toward the central divider, pair the opposite-edge neighbors, and transform each pair according to its yellow-marker pattern.

    Input structure:
    - The background is black and one full-height grey column splits the canvas into left and right halves.
    - Each side contains 2-3 small 4-connected monochrome sprites, and some sprites have one yellow 4-neighbor marker attached to them.
    - Markers travel with their sprites as part of the same rigid multicolor object, but the sprite pixels themselves still define pairing and recoloring.

    Rule:
    - Treat every left object as a rigid body that slides right and every right object as a rigid body that slides left; on each step, only objects whose next column is clear of the divider and already-settled blockers may move.
    - After all objects stop, pair a left sprite with a right sprite when they touch the divider on their respective sides and their sprite-row ranges overlap; each sprite belongs to at most one best-overlap pair.
    - For a paired sprite pair: double-marked pairs mirror each sprite outward on its own side, single-marked pairs recolor the unmarked sprite to the marked sprite's color, and unmarked pairs swap sprite colors. Unpaired sprites keep their slid positions.
    - Erase all yellow markers, keep the grey divider unchanged, and clip mirrored pixels that would fall outside the grid.

    Key insight:
    - The decisive pairing is not present in the input: it emerges only after bidirectional sliding and collision stopping, and the pair's joint marker state then selects one of three distinct transformations.
    """
    analysis = _analyze_grid(np.asarray(grid, dtype=int))
    return _render_output(analysis)


def _next_profile() -> tuple[str, ...]:
    global _PROFILE_INDEX
    profile = PROFILE_CYCLE[_PROFILE_INDEX % len(PROFILE_CYCLE)]
    _PROFILE_INDEX += 1
    branches = list(profile)
    random.shuffle(branches)
    return tuple(branches)


def _branch_marks(branch: str) -> tuple[bool, bool]:
    if branch == "double":
        return True, True
    if branch == "single":
        return random.choice([(True, False), (False, True)])
    return False, False


def _pick_distinct_pair_colors(branches: tuple[str, ...]) -> tuple[list[int], list[int]] | None:
    left_colors = random.sample(PAIR_COLORS, len(branches))
    for _ in range(50):
        right_colors = random.sample(PAIR_COLORS, len(branches))
        if all(
            left_colors[i] != right_colors[i] or branches[i] == "double"
            for i in range(len(branches))
        ):
            return left_colors, right_colors
    return None


def _choose_band_tops(band_heights: list[int], H: int) -> tuple[int, int] | None:
    top0_max = H - (band_heights[0] + band_heights[1] + 3)
    if top0_max < 1:
        return None
    top0 = random.randint(1, top0_max)
    top1_min = top0 + band_heights[0] + 2
    top1_max = H - band_heights[1] - 1
    if top1_min > top1_max:
        return None
    top1 = random.randint(top1_min, top1_max)
    return int(top0), int(top1)


def _place_object_with_choices(
    *,
    obj_id: int,
    side: str,
    color: int,
    shape: np.ndarray,
    marked: bool,
    role: str,
    pair_index: int | None,
    branch: str | None,
    top_choices: list[int],
    left_choices: list[int],
    occupied: set[tuple[int, int]],
    grid_shape: tuple[int, int],
) -> dict | None:
    tops = top_choices[:]
    lefts = left_choices[:]
    random.shuffle(tops)
    random.shuffle(lefts)
    for top in tops:
        for left in lefts:
            obj = _make_object(
                int(obj_id),
                side=side,
                color=int(color),
                shape=shape,
                top=int(top),
                left=int(left),
                marked=bool(marked),
                role=role,
                pair_index=pair_index,
                branch=branch,
                occupied=occupied,
                grid_shape=grid_shape,
            )
            if obj is not None:
                return obj
    return None


def _build_template(profile: tuple[str, ...]) -> tuple[np.ndarray, list[dict], dict] | None:
    H = random.randint(11, 16)
    W = random.choice([13, 15, 17, 19])
    divider = W // 2
    grid_shape = (H, W)

    colors = _pick_distinct_pair_colors(profile)
    if colors is None:
        return None
    left_pair_colors, right_pair_colors = colors

    left_shapes = [random.choice(SHAPE_CATALOG) for _ in profile]
    right_shapes = [random.choice(SHAPE_CATALOG) for _ in profile]
    band_heights = [max(left_shapes[i].shape[0], right_shapes[i].shape[0]) for i in range(len(profile))]
    band_tops = _choose_band_tops(band_heights, H)
    if band_tops is None:
        return None

    non_double = [i for i, branch in enumerate(profile) if branch != "double"]
    if not non_double:
        return None
    extra_sides = [random.choice(["left", "right"])]
    if len(non_double) == 2 and random.random() < 0.35:
        other = "right" if extra_sides[0] == "left" else "left"
        extra_sides.append(other)
    extra_plan = {
        side: {
            "pair_index": random.choice(non_double),
            "marked": random.random() < 0.5,
            "shape": random.choice(SHAPE_CATALOG),
        }
        for side in extra_sides
    }

    left_colors_used = set(left_pair_colors)
    right_colors_used = set(right_pair_colors)
    for side, spec in extra_plan.items():
        remaining = [c for c in PAIR_COLORS if c not in (left_colors_used if side == "left" else right_colors_used)]
        if not remaining:
            return None
        spec["color"] = random.choice(remaining)
        if side == "left":
            left_colors_used.add(int(spec["color"]))
        else:
            right_colors_used.add(int(spec["color"]))

    occupied: set[tuple[int, int]] = set((r, divider) for r in range(H))
    left_objects: list[dict] = []
    right_objects: list[dict] = []
    obj_id = 0
    pair_meta: list[dict] = []

    for i, branch in enumerate(profile):
        l_shape = left_shapes[i]
        r_shape = right_shapes[i]
        band_top = band_tops[i]
        band_height = band_heights[i]
        left_top_choices = list(range(band_top, band_top + band_height - l_shape.shape[0] + 1))
        right_top_choices = list(range(band_top, band_top + band_height - r_shape.shape[0] + 1))
        left_marked, right_marked = _branch_marks(branch)

        left_obj = _place_object_with_choices(
            obj_id=obj_id,
            side="left",
            color=left_pair_colors[i],
            shape=l_shape,
            marked=left_marked,
            role="pair",
            pair_index=i,
            branch=branch,
            top_choices=left_top_choices,
            left_choices=[divider - l_shape.shape[1]],
            occupied=occupied,
            grid_shape=grid_shape,
        )
        if left_obj is None:
            return None
        occupied |= _coords_set(left_obj["all_coords"])
        left_objects.append(left_obj)
        obj_id += 1

        right_obj = _place_object_with_choices(
            obj_id=obj_id,
            side="right",
            color=right_pair_colors[i],
            shape=r_shape,
            marked=right_marked,
            role="pair",
            pair_index=i,
            branch=branch,
            top_choices=right_top_choices,
            left_choices=[divider + 1],
            occupied=occupied,
            grid_shape=grid_shape,
        )
        if right_obj is None:
            return None
        occupied |= _coords_set(right_obj["all_coords"])
        right_objects.append(right_obj)
        obj_id += 1

        pair_meta.append({"branch": branch, "band_top": band_top, "band_height": band_height})

    for side, spec in extra_plan.items():
        shape = np.asarray(spec["shape"], dtype=bool)
        pair_index = int(spec["pair_index"])
        pair_band_top = int(pair_meta[pair_index]["band_top"])
        pair_band_height = int(pair_meta[pair_index]["band_height"])
        top_choices = list(range(pair_band_top, pair_band_top + pair_band_height - shape.shape[0] + 1))
        if side == "left":
            left_choices = list(range(0, max(1, divider - shape.shape[1] - 1)))
        else:
            left_choices = list(range(divider + 2, W - shape.shape[1] + 1))
            left_choices.sort(reverse=True)
        if not left_choices:
            return None

        obj = _place_object_with_choices(
            obj_id=obj_id,
            side=side,
            color=int(spec["color"]),
            shape=shape,
            marked=bool(spec["marked"]),
            role="extra",
            pair_index=pair_index,
            branch=None,
            top_choices=top_choices,
            left_choices=left_choices,
            occupied=occupied,
            grid_shape=grid_shape,
        )
        if obj is None:
            return None
        occupied |= _coords_set(obj["all_coords"])
        if side == "left":
            left_objects.append(obj)
        else:
            right_objects.append(obj)
        obj_id += 1

    settled_left = _slide_side(left_objects, grid_shape, dc=1, divider_col=divider)
    settled_right = _slide_side(right_objects, grid_shape, dc=-1, divider_col=divider)
    settled = settled_left + settled_right

    meta = {
        "grid_shape": grid_shape,
        "divider": divider,
        "profile": profile,
        "extra_sides": set(extra_sides),
    }
    return np.full(grid_shape, int(BG), dtype=int), settled, meta


def _shift_settled_to_input(settled: list[dict], meta: dict) -> list[dict] | None:
    H, W = meta["grid_shape"]
    divider = int(meta["divider"])
    left_side = [obj for obj in settled if obj["side"] == "left"]
    right_side = [obj for obj in settled if obj["side"] == "right"]

    def build_side(side_objects: list[dict], side: str) -> list[dict] | None:
        if not side_objects:
            return []
        pair_objs = [obj for obj in side_objects if obj["role"] == "pair"]
        extra_objs = [obj for obj in side_objects if obj["role"] == "extra"]
        if not pair_objs:
            return None

        if side == "left":
            base_max = min(_object_margins(obj, W)[0] for obj in pair_objs)
            if extra_objs:
                base_max = min(base_max, min(_object_margins(obj, W)[0] - 1 for obj in extra_objs))
        else:
            base_max = min(_object_margins(obj, W)[1] for obj in pair_objs)
            if extra_objs:
                base_max = min(base_max, min(_object_margins(obj, W)[1] - 1 for obj in extra_objs))
        if base_max < 1:
            return None

        base_shift = random.randint(1, min(4, int(base_max)))
        shifted: list[dict] = []
        for obj in side_objects:
            if obj["role"] == "pair":
                shift = base_shift
            else:
                margin = _object_margins(obj, W)[0 if side == "left" else 1]
                if margin <= base_shift:
                    return None
                shift = random.randint(base_shift + 1, int(margin))
            dc = -shift if side == "left" else shift
            moved = _copy_object(obj, dc=dc)
            shifted.append(moved)
        return shifted

    shifted_left = build_side(left_side, "left")
    shifted_right = build_side(right_side, "right")
    if shifted_left is None or shifted_right is None:
        return None

    input_objects = shifted_left + shifted_right
    occ = set((r, divider) for r in range(H))
    for obj in input_objects:
        cells = _coords_set(obj["all_coords"])
        if occ & cells:
            return None
        occ |= cells
    return input_objects


def _objects_to_grid(shape: tuple[int, int], divider: int, objects: list[dict]) -> np.ndarray:
    grid = np.full(shape, int(BG), dtype=int)
    grid[:, int(divider)] = int(DIVIDER)
    for obj in objects:
        sprite_coords = np.asarray(obj["sprite_coords"], dtype=int)
        grid[sprite_coords[:, 0], sprite_coords[:, 1]] = int(obj["color"])
        if obj["marker_coords"].size:
            marker_coords = np.asarray(obj["marker_coords"], dtype=int)
            grid[marker_coords[:, 0], marker_coords[:, 1]] = int(MARKER)
    return grid


def _validate_semantics(grid: np.ndarray, target_profile: tuple[str, ...]) -> bool:
    try:
        analysis = _analyze_grid(grid)
        output = _render_output(analysis)
    except Exception:
        return False

    if tuple(sorted(pair["branch"] for pair in analysis["pairs"])) != tuple(sorted(target_profile)):
        return False

    if len(analysis["pairs"]) != len(target_profile):
        return False

    objects = analysis["objects"]
    if sum(int(obj["slide_dc"]) != 0 for obj in objects) < len(objects):
        return False

    paired_ids = set(int(x) for x in analysis["paired_ids"])
    if sum(int(obj["id"]) not in paired_ids for obj in objects) < 1:
        return False

    left_count = sum(obj["side"] == "left" for obj in objects)
    right_count = sum(obj["side"] == "right" for obj in objects)
    if not (2 <= left_count <= 3 and 2 <= right_count <= 3):
        return False

    baseline = _render_baseline(analysis)
    baseline_occ = _coords_set(np.argwhere(baseline != int(BG)))

    for pair in analysis["pairs"]:
        left = pair["left"]
        right = pair["right"]
        if pair["branch"] == "double":
            left_mirror = _mirror_coords(left["sprite_coords"], "left")
            right_mirror = _mirror_coords(right["sprite_coords"], "right")
            left_new = [
                (int(r), int(c))
                for r, c in left_mirror
                if 0 <= int(r) < grid.shape[0]
                and 0 <= int(c) < grid.shape[1]
                and int(c) != int(analysis["divider_col"])
                and (int(r), int(c)) not in baseline_occ
            ]
            right_new = [
                (int(r), int(c))
                for r, c in right_mirror
                if 0 <= int(r) < grid.shape[0]
                and 0 <= int(c) < grid.shape[1]
                and int(c) != int(analysis["divider_col"])
                and (int(r), int(c)) not in baseline_occ
            ]
            if not left_new or not right_new:
                return False
        elif pair["branch"] == "single":
            if int(left["color"]) == int(right["color"]):
                return False
        else:
            if int(left["color"]) == int(right["color"]):
                return False

    if np.array_equal(output, grid):
        return False

    return True


def _gen_attempt(target_profile: tuple[str, ...]) -> np.ndarray | None:
    built = _build_template(target_profile)
    if built is None:
        return None
    _blank, settled, meta = built

    input_objects = _shift_settled_to_input(settled, meta)
    if input_objects is None:
        return None

    grid = _objects_to_grid(meta["grid_shape"], int(meta["divider"]), input_objects)
    if _validate_semantics(grid, target_profile):
        return grid
    return None


def generate_input() -> np.ndarray:
    target_profile = _next_profile()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(target_profile)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid paired-sliding ARC input")
