from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
SIDES = ("up", "right", "down", "left")

GEN_BODY_DIMS = (2, 3)
GEN_CONTAINER_SIZE_RANGE = (7, 10)
GEN_CONTAINER_COUNT_RANGE = (2, 4)
GEN_CLEARANCE_FROM_BORDER = 1
GEN_CANVAS_SIZE = 30
GEN_PATCH_GAP = 2
GEN_MAX_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    H, W = g.shape
    visited = np.zeros((H, W), dtype=bool)
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    components: list[dict] = []
    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if visited[r, c] or color == int(bg):
                continue
            visited[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if visited[nr, nc] or int(g[nr, nc]) == int(bg):
                        continue
                    if monochromatic and int(g[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _component_patch(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    coords = comp["coords"]
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(comp["color"])
    return patch


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _build_key_patch(body_h: int, body_w: int, side: str, tooth_offset: int, color: int) -> np.ndarray:
    cells = {(r, c) for r in range(body_h) for c in range(body_w)}
    if side == "up":
        cells.add((-1, tooth_offset))
    elif side == "down":
        cells.add((body_h, tooth_offset))
    elif side == "left":
        cells.add((tooth_offset, -1))
    else:
        assert side == "right"
        cells.add((tooth_offset, body_w))
    cells = _normalize_cells(cells)
    coords = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    patch[coords[:, 0], coords[:, 1]] = int(color)
    return patch


def _key_info_from_cells(cells: set[tuple[int, int]]) -> dict:
    candidates = []
    for tooth in cells:
        body = cells - {tooth}
        if not body:
            continue
        body_arr = np.array(sorted(body), dtype=int)
        r0, c0, r1, c1 = bbox_coords(body_arr)
        full_rect = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        if body != full_rect:
            continue
        side = None
        if tooth[0] == r0 - 1 and c0 <= tooth[1] <= c1 and (tooth[0] + 1, tooth[1]) in body:
            side = "up"
        elif tooth[0] == r1 + 1 and c0 <= tooth[1] <= c1 and (tooth[0] - 1, tooth[1]) in body:
            side = "down"
        elif tooth[1] == c0 - 1 and r0 <= tooth[0] <= r1 and (tooth[0], tooth[1] + 1) in body:
            side = "left"
        elif tooth[1] == c1 + 1 and r0 <= tooth[0] <= r1 and (tooth[0], tooth[1] - 1) in body:
            side = "right"
        if side is None:
            continue
        candidates.append(
            {
                "tooth": tuple(map(int, tooth)),
                "side": side,
                "body_bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    assert len(candidates) == 1, "key shape must decompose into one solid body plus one tooth"
    return candidates[0]


def _key_info_from_patch(patch: np.ndarray, bg: int) -> dict:
    cells = {tuple(map(int, rc)) for rc in np.argwhere(patch != int(bg))}
    return _key_info_from_cells(cells)


def _rotate_patch_to_side(patch: np.ndarray, target_side: str, bg: int) -> tuple[np.ndarray, dict]:
    candidates: list[tuple[np.ndarray, dict]] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(patch, k).copy()
        key = (tuple(map(int, rotated.shape)), rotated.tobytes())
        if key in seen:
            continue
        seen.add(key)
        info = _key_info_from_patch(rotated, bg)
        if info["side"] == target_side:
            candidates.append((rotated, info))
    assert len(candidates) == 1, "hole side must determine a unique key rotation"
    return candidates[0]


def _container_info(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
        return None

    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0 + 1, r1):
        border.add((r, c0))
        border.add((r, c1))

    cells = {tuple(map(int, rc)) for rc in comp["coords"]}
    if not cells.issubset(border):
        return None
    missing = sorted(border - cells)
    if len(missing) > 1:
        return None
    if len(missing) == 1:
        hr, hc = missing[0]
        if (hr in (r0, r1)) and (hc in (c0, c1)):
            return None
        if hr == r0:
            side = "up"
        elif hr == r1:
            side = "down"
        elif hc == c0:
            side = "left"
        else:
            side = "right"
        return {
            "bbox": (r0, c0, r1, c1),
            "color": int(comp["color"]),
            "has_hole": True,
            "hole": (int(hr), int(hc)),
            "hole_side": side,
        }
    if len(cells) != len(border):
        return None
    return {
        "bbox": (r0, c0, r1, c1),
        "color": int(comp["color"]),
        "has_hole": False,
        "hole": None,
        "hole_side": None,
    }


def _coords_inside(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(np.all((coords[:, 0] > r0) & (coords[:, 0] < r1) & (coords[:, 1] > c0) & (coords[:, 1] < c1)))


def _draw_border(dst: np.ndarray, color: int, hole: tuple[int, int] | None) -> None:
    dst[0, :] = int(color)
    dst[-1, :] = int(color)
    dst[:, 0] = int(color)
    dst[:, -1] = int(color)
    if hole is not None:
        dst[hole] = BG


def _paste_non_bg(dst: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    coords = np.argwhere(patch != int(bg))
    for rr, cc in coords:
        dst[int(top + rr), int(left + cc)] = int(patch[int(rr), int(cc)])


def _valid_source_origins(
    patch: np.ndarray,
    outer_h: int,
    outer_w: int,
    *,
    bg: int,
    clearance: int,
) -> list[tuple[int, int]]:
    coords = np.argwhere(patch != int(bg))
    min_row = 1 + int(clearance)
    max_row = outer_h - 2 - int(clearance)
    min_col = 1 + int(clearance)
    max_col = outer_w - 2 - int(clearance)
    if max_row < min_row or max_col < min_col:
        return []
    valid = []
    for top in range(outer_h - patch.shape[0] + 1):
        for left in range(outer_w - patch.shape[1] + 1):
            ok = True
            for rr, cc in coords:
                gr = int(top + rr)
                gc = int(left + cc)
                if not (min_row <= gr <= max_row and min_col <= gc <= max_col):
                    ok = False
                    break
            if ok:
                valid.append((int(top), int(left)))
    return valid


def _valid_hole_positions(
    patch: np.ndarray,
    side: str,
    outer_h: int,
    outer_w: int,
    *,
    bg: int,
) -> list[tuple[int, int]]:
    info = _key_info_from_patch(patch, bg)
    assert info["side"] == side
    tooth_r, tooth_c = info["tooth"]
    coords = [tuple(map(int, rc)) for rc in np.argwhere(patch != int(bg))]

    if side == "up":
        border_cells = [(0, c) for c in range(1, outer_w - 1)]
    elif side == "down":
        border_cells = [(outer_h - 1, c) for c in range(1, outer_w - 1)]
    elif side == "left":
        border_cells = [(r, 0) for r in range(1, outer_h - 1)]
    else:
        border_cells = [(r, outer_w - 1) for r in range(1, outer_h - 1)]

    valid = []
    for hole_r, hole_c in border_cells:
        top = int(hole_r - tooth_r)
        left = int(hole_c - tooth_c)
        ok = True
        for rr, cc in coords:
            gr = int(top + rr)
            gc = int(left + cc)
            if not (0 <= gr < outer_h and 0 <= gc < outer_w):
                ok = False
                break
            if (gr, gc) == (hole_r, hole_c):
                continue
            if not (1 <= gr <= outer_h - 2 and 1 <= gc <= outer_w - 2):
                ok = False
                break
        if ok:
            valid.append((int(hole_r), int(hole_c)))
    return valid


def _render_container(spec: dict, *, output: bool) -> np.ndarray:
    patch = np.full((spec["outer_h"], spec["outer_w"]), BG, dtype=int)
    _draw_border(patch, spec["container_color"], spec["hole"] if spec["has_hole"] else None)
    obj_patch = spec["target_patch"] if output and spec["has_hole"] else spec["source_patch"]
    obj_origin = spec["target_origin"] if output and spec["has_hole"] else spec["source_origin"]
    _paste_non_bg(patch, obj_patch, obj_origin[0], obj_origin[1], BG)
    return patch


def _rect_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    gap: int,
) -> bool:
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])


def _place_rectangles(shapes: list[tuple[int, int]], *, canvas_h: int, canvas_w: int, gap: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(shapes)), key=lambda i: shapes[i][0] * shapes[i][1], reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(shapes)
    rects: list[tuple[int, int, int, int]] = []
    for i in order:
        h, w = shapes[i]
        candidates = []
        for top in range(canvas_h - h + 1):
            for left in range(canvas_w - w + 1):
                rect = (top, left, top + h - 1, left + w - 1)
                if all(not _rect_too_close(rect, other, gap=gap) for other in rects):
                    candidates.append((int(top), int(left)))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        placements[i] = (top, left)
        rects.append((top, left, top + h - 1, left + w - 1))
    return [pos for pos in placements if pos is not None]


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere((inp != BG) | (out != BG))
    r0, c0, r1, c1 = bbox_coords(coords)
    top_margin = random.randint(0, min(2, r0))
    left_margin = random.randint(0, min(2, c0))
    bottom_margin = random.randint(0, min(2, inp.shape[0] - 1 - r1))
    right_margin = random.randint(0, min(2, inp.shape[1] - 1 - c1))
    rs = slice(r0 - top_margin, r1 + bottom_margin + 1)
    cs = slice(c0 - left_margin, c1 + right_margin + 1)
    return inp[rs, cs].copy(), out[rs, cs].copy()


def _make_container_spec(
    *,
    container_color: int,
    key_color: int,
    has_hole: bool,
    forced_hole_side: str | None,
    must_require_rotation: bool,
) -> dict | None:
    for _ in range(GEN_MAX_ATTEMPTS):
        body_h = random.choice(GEN_BODY_DIMS)
        body_w = random.choice(GEN_BODY_DIMS)
        source_side = random.choice(SIDES)
        tooth_extent = body_w if source_side in ("up", "down") else body_h
        tooth_offset = random.randrange(tooth_extent)
        source_patch = _build_key_patch(body_h, body_w, source_side, tooth_offset, key_color)
        _key_info_from_patch(source_patch, BG)

        hole_side = forced_hole_side if forced_hole_side is not None else random.choice(SIDES)
        target_patch = None
        target_info = None
        if has_hole:
            if must_require_rotation and source_side == hole_side:
                continue
            target_patch, target_info = _rotate_patch_to_side(source_patch, hole_side, BG)
        min_outer_h = source_patch.shape[0] + 4
        min_outer_w = source_patch.shape[1] + 4
        if has_hole:
            min_outer_h = max(min_outer_h, target_patch.shape[0] + 2)
            min_outer_w = max(min_outer_w, target_patch.shape[1] + 2)
        max_outer = GEN_CONTAINER_SIZE_RANGE[1]
        if min_outer_h > max_outer or min_outer_w > max_outer:
            continue
        outer_h = random.randint(max(min_outer_h, GEN_CONTAINER_SIZE_RANGE[0]), max_outer)
        outer_w = random.randint(max(min_outer_w, GEN_CONTAINER_SIZE_RANGE[0]), max_outer)

        source_origins = _valid_source_origins(
            source_patch,
            outer_h,
            outer_w,
            bg=BG,
            clearance=GEN_CLEARANCE_FROM_BORDER,
        )
        if not source_origins:
            continue
        source_origin = random.choice(source_origins)

        spec = {
            "outer_h": int(outer_h),
            "outer_w": int(outer_w),
            "container_color": int(container_color),
            "key_color": int(key_color),
            "has_hole": bool(has_hole),
            "source_patch": source_patch,
            "source_origin": tuple(map(int, source_origin)),
            "source_side": source_side,
            "hole": None,
            "hole_side": None,
            "target_patch": source_patch,
            "target_origin": tuple(map(int, source_origin)),
        }
        if not has_hole:
            return spec

        holes = _valid_hole_positions(target_patch, hole_side, outer_h, outer_w, bg=BG)
        if not holes:
            continue
        hole = random.choice(holes)
        tooth_r, tooth_c = target_info["tooth"]
        target_origin = (int(hole[0] - tooth_r), int(hole[1] - tooth_c))
        spec.update(
            {
                "hole": tuple(map(int, hole)),
                "hole_side": hole_side,
                "target_patch": target_patch,
                "target_origin": target_origin,
            }
        )
        return spec
    return None


def _scene_pair() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.randint(*GEN_CONTAINER_COUNT_RANGE)
    locked_count = random.randint(1, n - 1)
    closed_count = n - locked_count
    assert closed_count >= 1

    palette = random.sample(list(range(1, 10)), 2 * n)
    container_colors = palette[:n]
    key_colors = palette[n:]

    locked_sides = [random.choice(SIDES) for _ in range(locked_count)]
    if locked_count > 1 and len(set(locked_sides)) == 1:
        locked_sides[1] = random.choice([side for side in SIDES if side != locked_sides[0]])

    flags = [True] * locked_count + [False] * closed_count
    random.shuffle(flags)

    specs = []
    hole_side_i = 0
    forced_rotation_needed = True
    for i, has_hole in enumerate(flags):
        forced_hole_side = locked_sides[hole_side_i] if has_hole else None
        spec = _make_container_spec(
            container_color=container_colors[i],
            key_color=key_colors[i],
            has_hole=has_hole,
            forced_hole_side=forced_hole_side,
            must_require_rotation=has_hole and forced_rotation_needed,
        )
        if spec is None:
            return None
        if has_hole:
            hole_side_i += 1
            if spec["source_side"] != spec["hole_side"]:
                forced_rotation_needed = False
        specs.append(spec)

    if forced_rotation_needed:
        return None
    locked_specs = [spec for spec in specs if spec["has_hole"]]
    if len(locked_specs) > 1 and len({spec["hole_side"] for spec in locked_specs}) < 2:
        return None

    input_patches = [_render_container(spec, output=False) for spec in specs]
    output_patches = [_render_container(spec, output=True) for spec in specs]
    shapes = [tuple(map(int, patch.shape)) for patch in input_patches]
    placements = _place_rectangles(shapes, canvas_h=GEN_CANVAS_SIZE, canvas_w=GEN_CANVAS_SIZE, gap=GEN_PATCH_GAP)
    if placements is None:
        return None

    inp = np.full((GEN_CANVAS_SIZE, GEN_CANVAS_SIZE), BG, dtype=int)
    out = np.full((GEN_CANVAS_SIZE, GEN_CANVAS_SIZE), BG, dtype=int)
    for patch_in, patch_out, (top, left) in zip(input_patches, output_patches, placements):
        _paste_non_bg(inp, patch_in, top, left, BG)
        _paste_non_bg(out, patch_out, top, left, BG)

    inp, out = _crop_pair(inp, out)
    if not _validate_scene(inp, out):
        return None
    return inp, out


def _validate_scene(inp: np.ndarray, expected: np.ndarray) -> bool:
    if np.array_equal(inp, expected):
        return False

    comps = find_components(inp, bg=BG, connectivity=4, monochromatic=True)
    container_meta = []
    object_comps = []
    for comp in comps:
        info = _container_info(comp)
        if info is None:
            object_comps.append(comp)
        else:
            container_meta.append(info)
    if len(container_meta) < 2:
        return False
    if not any(meta["has_hole"] for meta in container_meta):
        return False
    if not any(not meta["has_hole"] for meta in container_meta):
        return False
    if len(object_comps) != len(container_meta):
        return False

    required_rotation = False
    counts = [0] * len(container_meta)
    for obj in object_comps:
        inside = [i for i, meta in enumerate(container_meta) if _coords_inside(obj["coords"], meta["bbox"])]
        if len(inside) != 1:
            return False
        i = inside[0]
        counts[i] += 1
        patch = _component_patch(inp, obj, BG)
        info = _key_info_from_patch(patch, BG)
        if container_meta[i]["has_hole"]:
            variants = []
            seen = set()
            for k in range(4):
                rotated = np.rot90(patch, k).copy()
                key = (tuple(map(int, rotated.shape)), rotated.tobytes())
                if key in seen:
                    continue
                seen.add(key)
                if _key_info_from_patch(rotated, BG)["side"] == container_meta[i]["hole_side"]:
                    variants.append(key)
            if len(variants) != 1:
                return False
            if info["side"] != container_meta[i]["hole_side"]:
                required_rotation = True
    if any(count != 1 for count in counts):
        return False
    if not required_rotation:
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each key-shaped object to the lock direction and slide it into the matching border notch.

    Input structure
    - The grid has a black background with several one-pixel-thick rectangular border containers of different colors.
    - Each container encloses exactly one monochromatic key object made of a solid rectangular body plus one single-pixel tooth.
    - Some container borders are complete, while others have exactly one missing non-corner border cell acting as a lock hole.

    Rule
    - For every container with a lock hole, find the unique rotation of its inner key whose tooth points toward that hole side.
    - Erase the key from its old position, then translate the rotated key so the tooth lands exactly on the missing border cell and all other key pixels stay inside the container.
    - Leave closed containers and their inner objects unchanged.

    Key insight
    - The missing border cell is a lock, not an exit: the key must first be oriented so its single tooth faces the hole, then be inserted so the tooth alone occupies the border gap.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)

    containers = []
    others = []
    for comp in components:
        info = _container_info(comp)
        if info is None:
            others.append(comp)
        else:
            containers.append(info)
    assert containers, "expected at least one container"

    objects_by_container: list[list[dict]] = [[] for _ in containers]
    for comp in others:
        inside = [i for i, container in enumerate(containers) if _coords_inside(comp["coords"], container["bbox"])]
        assert len(inside) == 1, "each non-container object must belong to exactly one container"
        objects_by_container[inside[0]].append(comp)

    out = g.copy()
    for container, objs in zip(containers, objects_by_container):
        assert len(objs) == 1, "each container must enclose exactly one key object"
        if not container["has_hole"]:
            continue

        obj = objs[0]
        source_patch = _component_patch(g, obj, bg)
        rotated_patch, rotated_info = _rotate_patch_to_side(source_patch, container["hole_side"], bg)
        tooth_r, tooth_c = rotated_info["tooth"]
        hole_r, hole_c = container["hole"]
        target_top = int(hole_r - tooth_r)
        target_left = int(hole_c - tooth_c)

        out[obj["coords"][:, 0], obj["coords"][:, 1]] = int(bg)
        for rr, cc in np.argwhere(rotated_patch != int(bg)):
            gr = int(target_top + rr)
            gc = int(target_left + cc)
            assert 0 <= gr < out.shape[0] and 0 <= gc < out.shape[1], "inserted key must stay in bounds"
            if (gr, gc) == (hole_r, hole_c):
                assert int(out[gr, gc]) == int(bg), "hole cell must be empty before insertion"
            else:
                r0, c0, r1, c1 = container["bbox"]
                assert r0 < gr < r1 and c0 < gc < c1, "all non-tooth key pixels must remain inside the container"
                assert int(out[gr, gc]) == int(bg), "inserted key must not overwrite existing pixels"
            out[gr, gc] = int(obj["color"])

    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        made = _scene_pair()
        if made is not None:
            inp, _out = made
            return inp
    raise RuntimeError("failed to generate a valid key-in-lock ARC input")
