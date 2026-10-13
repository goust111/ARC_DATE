from __future__ import annotations

import random
import sys
from collections import Counter
from typing import Iterable

import numpy as np

try:
    _ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
    if _ARC_UTILS_DIR not in sys.path:
        sys.path.append(_ARC_UTILS_DIR)
    from arc_utils import Color, crop_bbox, find_components, get_bg, in_bounds, neighbors4, paste
except Exception:  # pragma: no cover - fallback for direct loading
    class Color:
        BLACK = 0
        TEAL = 8

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])

    def neighbors4(r: int, c: int):
        yield int(r) - 1, int(c)
        yield int(r) + 1, int(c)
        yield int(r), int(c) - 1
        yield int(r), int(c) + 1

    def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = map(int, bbox)
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        transparent: int | None = None,
        overwrite: bool = True,
        clip: bool = False,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        h, w = patch.shape
        if not clip and (r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]):
            raise ValueError("paste out of bounds")
        rr0 = max(0, int(r0))
        cc0 = max(0, int(c0))
        rr1 = min(dst.shape[0], int(r0) + h)
        cc1 = min(dst.shape[1], int(c0) + w)
        if rr0 >= rr1 or cc0 >= cc1:
            return dst
        pr0 = rr0 - int(r0)
        pc0 = cc0 - int(c0)
        view = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]
        dst_view = dst[rr0:rr1, cc0:cc1]
        use = np.ones_like(view, dtype=bool)
        if mask is not None:
            use &= mask[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]
        if transparent is not None:
            use &= view != int(transparent)
            if not overwrite:
                use &= dst_view == int(transparent)
        dst_view[use] = view[use]
        return dst

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        return_format: str = "dict",
    ) -> list[dict]:
        if bg is None:
            bg = get_bg(grid)
        bg = int(bg)
        seen = np.zeros_like(grid, dtype=bool)
        out: list[dict] = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if seen[r, c] or int(grid[r, c]) == bg:
                    continue
                target = int(grid[r, c])
                stack = [(r, c)]
                coords = []
                seen[r, c] = True
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for nr, nc in neighbors4(rr, cc):
                        if not in_bounds(grid.shape, nr, nc) or seen[nr, nc]:
                            continue
                        if int(grid[nr, nc]) == bg:
                            continue
                        if monochromatic and int(grid[nr, nc]) != target:
                            continue
                        seen[nr, nc] = True
                        stack.append((nr, nc))
                arr = np.array(coords, dtype=int)
                r0, c0 = arr.min(axis=0)
                r1, c1 = arr.max(axis=0)
                out.append(
                    {
                        "color": target if monochromatic else None,
                        "coords": arr,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(arr.shape[0]),
                    }
                )
        return out


BG = Color.BLACK
ANCHOR_COLOR = Color.TEAL
FRAGMENT_PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
ANGLE_TO_DIR = {
    0: (0, 1),
    90: (1, 0),
    180: (0, -1),
    270: (-1, 0),
}
MASK_SIZES = [(3, 3), (3, 4), (4, 3), (4, 4)]
MAX_GENERATION_ATTEMPTS = 400


def _rotate_cw(arr: np.ndarray, angle: int) -> np.ndarray:
    quarter_turns = (int(angle) // 90) % 4
    return np.rot90(arr, -quarter_turns)


def _coords_from_mask(mask: np.ndarray, top: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, top)
    rel = np.argwhere(mask)
    return [(r0 + int(r), c0 + int(c)) for r, c in rel]


def _touches(coords: Iterable[tuple[int, int]], occ_mask: np.ndarray) -> bool:
    for r, c in coords:
        for nr, nc in neighbors4(r, c):
            if in_bounds(occ_mask.shape, nr, nc) and bool(occ_mask[nr, nc]):
                return True
    return False


def _overlaps(coords: Iterable[tuple[int, int]], occ_mask: np.ndarray) -> bool:
    return any(bool(occ_mask[r, c]) for r, c in coords)


def _all_in_bounds(coords: Iterable[tuple[int, int]], shape: tuple[int, int]) -> bool:
    return all(in_bounds(shape, r, c) for r, c in coords)


def _paint_bool(mask: np.ndarray, coords: Iterable[tuple[int, int]]) -> None:
    for r, c in coords:
        mask[r, c] = True


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _component_info(grid: np.ndarray, comp: dict, bg: int) -> dict:
    patch = crop_bbox(grid, comp["bbox"])
    mask = patch != int(bg)
    colors = sorted(int(v) for v in np.unique(patch[mask]))
    r0, c0, _r1, _c1 = map(int, comp["bbox"])
    return {
        "bbox": tuple(map(int, comp["bbox"])),
        "top": (r0, c0),
        "patch": patch,
        "mask": mask,
        "coords": [(int(r), int(c)) for r, c in comp["coords"]],
        "colors": colors,
        "area": int(comp["area"]),
    }


def _match_angle(template_mask: np.ndarray, frag_mask: np.ndarray) -> int:
    matches = []
    for angle in (0, 90, 180, 270):
        rotated = _rotate_cw(template_mask, angle)
        if rotated.shape == frag_mask.shape and np.array_equal(rotated, frag_mask):
            matches.append(angle)
    assert len(matches) == 1, "fragment rotation must be uniquely diagnosable"
    return int(matches[0])


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    objects = [_component_info(g, comp, bg) for comp in find_components(g, bg=bg, connectivity=4, monochromatic=False)]
    assert objects, "expected at least one non-background object"

    color_counts = [len(obj["colors"]) for obj in objects]
    best = max(color_counts)
    assert best >= 3 and color_counts.count(best) == 1, "template must be the unique richest-color object"
    template = objects[color_counts.index(best)]
    template_mask = template["mask"]

    anchor = None
    fragments = []
    for obj in objects:
        if obj is template:
            continue
        is_anchor = (
            obj["colors"] == [ANCHOR_COLOR]
            and obj["patch"].shape == (2, 2)
            and obj["area"] == 4
            and bool(np.all(obj["mask"]))
        )
        if is_anchor:
            assert anchor is None, "anchor must be unique"
            anchor = obj
            continue
        assert len(obj["colors"]) == 1, "every non-template non-anchor object must be monochrome"
        angle = _match_angle(template_mask, obj["mask"])
        color = int(obj["colors"][0])
        fragments.append(
            {
                "bbox": obj["bbox"],
                "top": obj["top"],
                "mask": obj["mask"],
                "patch": _mask_to_patch(obj["mask"], color),
                "color": color,
                "angle": angle,
                "direction": ANGLE_TO_DIR[angle],
            }
        )

    assert anchor is not None, "expected one teal 2x2 anchor"
    assert 3 <= len(fragments) <= 4, "expected 3-4 fragments"
    assert len({frag["angle"] for frag in fragments}) == len(fragments), "fragment rotations must be distinct"

    anchor_mask = np.zeros_like(g, dtype=bool)
    _paint_bool(anchor_mask, anchor["coords"])

    return {
        "grid": g,
        "bg": bg,
        "template": template,
        "template_mask": template_mask,
        "anchor": anchor,
        "anchor_mask": anchor_mask,
        "fragments": fragments,
    }


def _slide_distance(
    mask: np.ndarray,
    top: tuple[int, int],
    direction: tuple[int, int],
    state_grid: np.ndarray,
    docking_mask: np.ndarray,
    bg: int,
) -> tuple[int, tuple[int, int]] | None:
    blocked = state_grid != int(bg)
    self_coords = _coords_from_mask(mask, top)
    for r, c in self_coords:
        blocked[r, c] = False

    if _touches(self_coords, docking_mask):
        return 0, (int(top[0]), int(top[1]))

    dr, dc = map(int, direction)
    for dist in range(1, max(state_grid.shape) + 1):
        new_top = (int(top[0]) + dr * dist, int(top[1]) + dc * dist)
        coords = _coords_from_mask(mask, new_top)
        if not _all_in_bounds(coords, state_grid.shape):
            return None
        if _overlaps(coords, blocked):
            return None
        if _touches(coords, docking_mask):
            return int(dist), new_top
    return None


def _simulate_solution(scene: dict) -> tuple[np.ndarray, list[dict]]:
    g = scene["grid"].copy()
    bg = int(scene["bg"])
    fragments = [frag.copy() for frag in scene["fragments"]]
    docking_mask = scene["anchor_mask"].copy()
    remaining = list(range(len(fragments)))
    moves: list[dict] = []

    while remaining:
        candidates = []
        for idx in remaining:
            frag = fragments[idx]
            result = _slide_distance(frag["mask"], frag["top"], frag["direction"], g, docking_mask, bg)
            assert result is not None, "every remaining fragment must have a clear slide path"
            dist, new_top = result
            r0, c0 = map(int, frag["top"])
            candidates.append((int(dist), int(c0), int(r0), int(idx), new_top))

        dist, _c0, _r0, idx, new_top = min(candidates)
        frag = fragments[idx]
        old_coords = _coords_from_mask(frag["mask"], frag["top"])
        for r, c in old_coords:
            g[r, c] = bg
        paste(g, frag["patch"], int(new_top[0]), int(new_top[1]), transparent=bg)
        new_coords = _coords_from_mask(frag["mask"], new_top)
        _paint_bool(docking_mask, new_coords)
        frag["top"] = (int(new_top[0]), int(new_top[1]))
        moves.append(
            {
                "idx": int(idx),
                "angle": int(frag["angle"]),
                "color": int(frag["color"]),
                "distance": int(dist),
                "top": frag["top"],
            }
        )
        remaining.remove(idx)

    return g, moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Diagnose each monochrome fragment's template rotation, reinterpret that angle as a slide direction, and stamp the fragment at its docking stop.

    Input structure:
    - A black background contains one connected multicolor template, one teal 2x2 anchor block, and 3-4 scattered monochrome fragments.
    - The template is the only object with the largest number of distinct non-background colors.
    - Each fragment is a single-color rotated copy of the template's occupied-cell shape.

    Rule:
    - Identify the unique richest-color object as the template and read every other non-anchor object as a fragment.
    - For each fragment, determine which clockwise rotation of the template shape matches it exactly; map 0/90/180/270 degrees to slide right/down/left/up.
    - Repeatedly choose the remaining fragment whose slide reaches first 4-neighbor contact with the anchor or already docked fragments using the shortest distance, breaking ties by bbox top-left left-to-right then top-to-bottom.
    - Move the fragment by translation only, erase its old position, stamp it at the stopping location, and leave the template fixed.

    Key insight:
    - The rotation diagnosis is not used to reorient the piece; it is reinterpreted as a movement command, so the rotated shape stays rotated while only its travel direction depends on the template match.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    solved, _moves = _simulate_solution(scene)
    return solved


def _random_template_mask() -> np.ndarray:
    for _ in range(300):
        h, w = random.choice(MASK_SIZES)
        area = random.randint(5, min(h * w - 1, 8))
        idxs = random.sample(range(h * w), area)
        mask = np.zeros((h, w), dtype=bool)
        for idx in idxs:
            mask[idx // w, idx % w] = True
        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        coords = np.argwhere(mask)
        seen = {tuple(map(int, coords[0]))}
        stack = [tuple(map(int, coords[0]))]
        while stack:
            r, c = stack.pop()
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        if len(seen) != int(mask.sum()):
            continue
        rotations = {(_rotate_cw(mask, angle).shape, _rotate_cw(mask, angle).tobytes()) for angle in (0, 90, 180, 270)}
        if len(rotations) != 4:
            continue
        return mask
    raise RuntimeError("failed to build an asymmetric template mask")


def _colorize_template(mask: np.ndarray, n_colors: int) -> np.ndarray:
    palette = random.sample(FRAGMENT_PALETTE, n_colors)
    patch = np.full(mask.shape, BG, dtype=int)
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    random.shuffle(cells)
    for i, (r, c) in enumerate(cells):
        patch[r, c] = int(palette[i % len(palette)])
    return patch


def _position_candidates(
    shape: tuple[int, int],
    mask: np.ndarray,
    direction: tuple[int, int],
    occupied: np.ndarray,
    support_mask: np.ndarray,
    anchor_mask: np.ndarray,
    forbid_anchor_touch: bool,
) -> list[tuple[int, int]]:
    h, w = mask.shape
    H, W = shape
    support_coords = np.argwhere(anchor_mask if support_mask is None else support_mask)
    anchor_center = support_coords.mean(axis=0) if support_coords.size else np.array([H / 2, W / 2], dtype=float)
    scored: list[tuple[float, float, int, int]] = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            coords = _coords_from_mask(mask, (r0, c0))
            if _overlaps(coords, occupied):
                continue
            if not _touches(coords, support_mask):
                continue
            if forbid_anchor_touch and _touches(coords, anchor_mask):
                continue
            prev_top = (int(r0) - int(direction[0]), int(c0) - int(direction[1]))
            prev_coords = _coords_from_mask(mask, prev_top)
            if not _all_in_bounds(prev_coords, shape):
                continue
            if _overlaps(prev_coords, occupied):
                continue
            if _touches(prev_coords, occupied):
                continue
            center = np.array([r0 + (h - 1) / 2.0, c0 + (w - 1) / 2.0])
            score = float(np.abs(center - anchor_center).sum())
            scored.append((score, random.random(), int(r0), int(c0)))
    scored.sort()
    return [(r0, c0) for _score, _noise, r0, c0 in scored[:24]]


def _path_positions(mask: np.ndarray, final_top: tuple[int, int], direction: tuple[int, int], dist: int) -> list[list[tuple[int, int]]]:
    dr, dc = map(int, direction)
    init_top = (int(final_top[0]) - dr * int(dist), int(final_top[1]) - dc * int(dist))
    return [_coords_from_mask(mask, (init_top[0] + dr * step, init_top[1] + dc * step)) for step in range(int(dist))]


def _choose_initial_positions(plans: list[dict], shape: tuple[int, int], static_blocked: np.ndarray) -> bool:
    anchor_dists = [2, 3, 4, 5]
    tie_anchor = len(plans) >= 2 and random.random() < 0.4
    if tie_anchor:
        shared = random.choice(anchor_dists[:-1])
        distance_sets = {0: [shared], 1: [shared]}
    else:
        d0, d1 = random.sample(anchor_dists, 2)
        distance_sets = {0: [d0], 1: [d1]}

    chain_base = max(distance_sets[0][0], distance_sets[1][0]) + 1
    for idx in range(2, len(plans)):
        options = list(range(chain_base, min(max(shape), chain_base + 3)))
        random.shuffle(options)
        distance_sets[idx] = options

    reserved = np.zeros(shape, dtype=bool)
    order = sorted(range(len(plans)), key=lambda idx: -max(distance_sets[idx]))

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        plan = plans[idx]
        for dist in distance_sets[idx]:
            positions = _path_positions(plan["mask"], plan["final_top"], plan["direction"], dist)
            if not positions:
                continue
            union_coords = [coord for coords in positions for coord in coords]
            if not _all_in_bounds(union_coords, shape):
                continue
            if _overlaps(union_coords, static_blocked) or _overlaps(union_coords, reserved):
                continue
            init_top = (
                int(plan["final_top"][0]) - int(plan["direction"][0]) * int(dist),
                int(plan["final_top"][1]) - int(plan["direction"][1]) * int(dist),
            )
            plan["initial_top"] = init_top
            plan["distance"] = int(dist)
            _paint_bool(reserved, union_coords)
            if backtrack(pos + 1):
                return True
            for r, c in union_coords:
                reserved[r, c] = False
            plan["initial_top"] = None
            plan["distance"] = None
        return False

    return backtrack(0)


def _find_template_top(shape: tuple[int, int], patch: np.ndarray, blocked: np.ndarray) -> tuple[int, int] | None:
    h, w = patch.shape
    H, W = shape
    for _ in range(80):
        r0 = random.randint(0, H - h)
        c0 = random.randint(0, W - w)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(H, r0 + h + 1)
        cc1 = min(W, c0 + w + 1)
        if not blocked[rr0:rr1, cc0:cc1].any():
            return int(r0), int(c0)
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(H, r0 + h + 1)
            cc1 = min(W, c0 + w + 1)
            if not blocked[rr0:rr1, cc0:cc1].any():
                return int(r0), int(c0)
    return None


def _render_scene(shape: tuple[int, int], template_patch: np.ndarray, template_top: tuple[int, int], plans: list[dict], anchor_top: tuple[int, int], use_initial: bool) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    paste(grid, template_patch, int(template_top[0]), int(template_top[1]), transparent=BG)
    grid[int(anchor_top[0]) : int(anchor_top[0]) + 2, int(anchor_top[1]) : int(anchor_top[1]) + 2] = ANCHOR_COLOR
    for plan in plans:
        top = plan["initial_top"] if use_initial else plan["final_top"]
        paste(grid, plan["patch"], int(top[0]), int(top[1]), transparent=BG)
    return grid


def _validate_generated(inp: np.ndarray, expected: np.ndarray) -> bool:
    scene = _extract_scene(inp)
    solved, moves = _simulate_solution(scene)
    if not np.array_equal(solved, expected):
        return False
    if len({move["angle"] for move in moves}) != len(moves):
        return False

    anchor_mask = scene["anchor_mask"].copy()
    state = scene["grid"].copy()
    fragments = [frag.copy() for frag in scene["fragments"]]
    remaining = list(range(len(fragments)))
    seen_anchor_contact = False
    seen_chain_contact = False
    placed_non_anchor = np.zeros_like(anchor_mask, dtype=bool)

    for move in moves:
        candidates = []
        for idx in remaining:
            frag = fragments[idx]
            result = _slide_distance(frag["mask"], frag["top"], frag["direction"], state, anchor_mask | placed_non_anchor, scene["bg"])
            if result is None:
                return False
            dist, new_top = result
            candidates.append((dist, frag["top"][1], frag["top"][0], idx, new_top))
        best = min(candidates)
        if best[3] != move["idx"] or best[0] != move["distance"] or best[4] != move["top"]:
            return False

        frag = fragments[move["idx"]]
        final_coords = _coords_from_mask(frag["mask"], move["top"])
        if _touches(final_coords, anchor_mask):
            seen_anchor_contact = True
        if _touches(final_coords, placed_non_anchor) and not _touches(final_coords, anchor_mask):
            seen_chain_contact = True

        old_coords = _coords_from_mask(frag["mask"], frag["top"])
        for r, c in old_coords:
            state[r, c] = scene["bg"]
        paste(state, frag["patch"], int(move["top"][0]), int(move["top"][1]), transparent=scene["bg"])
        _paint_bool(placed_non_anchor, final_coords)
        frag["top"] = move["top"]
        remaining.remove(move["idx"])

    return seen_anchor_contact and seen_chain_contact


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(20, 28)
    W = random.randint(20, 28)
    shape = (H, W)

    mask = _random_template_mask()
    n_fragments = random.choice([3, 4])
    angles = random.sample([0, 90, 180, 270], n_fragments)
    fragment_colors = random.sample(FRAGMENT_PALETTE, n_fragments)
    template_patch = _colorize_template(mask, 4 if n_fragments == 4 else random.choice([4, 5]))

    anchor_top = (random.randint(7, H - 9), random.randint(7, W - 9))
    anchor_mask = np.zeros(shape, dtype=bool)
    anchor_mask[anchor_top[0] : anchor_top[0] + 2, anchor_top[1] : anchor_top[1] + 2] = True

    support_specs = [("anchor", None), ("anchor", None)]
    if n_fragments == 3:
        support_specs.append(("piece", random.choice([0, 1])))
    else:
        support_specs.extend([("piece", 0), ("piece", 1)])

    plans = []
    occupied_final = anchor_mask.copy()
    final_piece_masks: list[np.ndarray] = []
    for angle, color, (support_kind, support_idx) in zip(angles, fragment_colors, support_specs):
        frag_mask = _rotate_cw(mask, angle)
        patch = _mask_to_patch(frag_mask, color)
        if support_kind == "anchor":
            support_mask = anchor_mask
            forbid_anchor_touch = False
        else:
            support_mask = final_piece_masks[int(support_idx)]
            forbid_anchor_touch = True
        candidates = _position_candidates(shape, frag_mask, ANGLE_TO_DIR[angle], occupied_final, support_mask, anchor_mask, forbid_anchor_touch)
        if not candidates:
            return None
        final_top = random.choice(candidates[: min(8, len(candidates))])
        coords = _coords_from_mask(frag_mask, final_top)
        piece_mask = np.zeros(shape, dtype=bool)
        _paint_bool(piece_mask, coords)
        final_piece_masks.append(piece_mask)
        _paint_bool(occupied_final, coords)
        plans.append(
            {
                "mask": frag_mask,
                "patch": patch,
                "angle": int(angle),
                "direction": ANGLE_TO_DIR[angle],
                "color": int(color),
                "final_top": tuple(map(int, final_top)),
                "initial_top": None,
                "distance": None,
            }
        )

    if not _choose_initial_positions(plans, shape, anchor_mask):
        return None

    blocked_for_template = occupied_final.copy()
    for plan in plans:
        _paint_bool(blocked_for_template, _coords_from_mask(plan["mask"], plan["initial_top"]))
        for coords in _path_positions(plan["mask"], plan["final_top"], plan["direction"], plan["distance"]):
            _paint_bool(blocked_for_template, coords)
    template_top = _find_template_top(shape, template_patch, blocked_for_template)
    if template_top is None:
        return None

    inp = _render_scene(shape, template_patch, template_top, plans, anchor_top, use_initial=True)
    expected = _render_scene(shape, template_patch, template_top, plans, anchor_top, use_initial=False)
    if np.array_equal(inp, expected):
        return None
    try:
        if not _validate_generated(inp, expected):
            return None
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-rotation ARC instance")
