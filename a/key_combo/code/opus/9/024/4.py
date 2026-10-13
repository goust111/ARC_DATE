from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SPRITE_SIZE = 3
SPRITE_AREA = 9
SPRITE_COLORS_PER_OBJECT = 2
CORNERS = ("NW", "NE", "SW", "SE")
OUTWARD = {
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SW": (1, -1),
    "SE": (1, 1),
}
ROTATED_CORNER = {
    0: {"NW": "NW", "NE": "NE", "SW": "SW", "SE": "SE"},
    1: {"NW": "SW", "NE": "NW", "SW": "SE", "SE": "NE"},
    2: {"NW": "SE", "NE": "SW", "SW": "NE", "SE": "NW"},
    3: {"NW": "NE", "NE": "SE", "SW": "NW", "SE": "SW"},
}
SPRITE_PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GEN_RETRIES = 400


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            colors: set[int] = set()
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                colors.add(int(grid[rr, cc]))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(
                {
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                    "colors": colors,
                }
            )
    return components


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, clip: bool = False) -> None:
    h, w = patch.shape
    dh, dw = dst.shape
    if not clip:
        if r0 < 0 or c0 < 0 or r0 + h > dh or c0 + w > dw:
            raise ValueError("patch out of bounds")
        dst[r0 : r0 + h, c0 : c0 + w] = patch
        return

    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(dh, r0 + h)
    cc1 = min(dw, c0 + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    pr0 = rr0 - r0
    pc0 = cc0 - c0
    dst[rr0:rr1, cc0:cc1] = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]


def _corners_from_bbox(bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {
        "NW": (r0, c0),
        "NE": (r0, c1),
        "SW": (r1, c0),
        "SE": (r1, c1),
    }


def _outside_corner(corner: str, coord: tuple[int, int]) -> tuple[int, int]:
    dr, dc = OUTWARD[corner]
    return coord[0] + dr, coord[1] + dc


def _l_group(dst_corner: str, bbox: tuple[int, int, int, int]) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    sprite_corner = _corners_from_bbox(bbox)[dst_corner]
    center = _outside_corner(dst_corner, sprite_corner)
    square = [(center[0] + dr, center[1] + dc) for dr in (-1, 1) for dc in (-1, 1)]
    return center, [pos for pos in square if pos != sprite_corner]


def _rotation_steps(src_corner: str, dst_corner: str) -> int:
    for steps, mapping in ROTATED_CORNER.items():
        if mapping[src_corner] == dst_corner:
            return steps
    raise AssertionError("source corner must map to a unique destination corner")


def _scaled_bbox(sprite_bbox: tuple[int, int, int, int], scale: int) -> tuple[int, int, int, int]:
    r0, c0, _r1, _c1 = sprite_bbox
    center_r = r0 + 1
    center_c = c0 + 1
    side = SPRITE_SIZE * scale
    top = center_r - side // 2
    left = center_c - side // 2
    return top, left, top + side - 1, left + side - 1


def _clip_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    h, w = shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(h - 1, r1)
    cc1 = min(w - 1, c1)
    if rr0 > rr1 or cc0 > cc1:
        return None
    return rr0, cc0, rr1, cc1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _sprite_output_patch(sprite_patch: np.ndarray, src_corner: str, dst_corner: str, scale: int) -> np.ndarray:
    rotated = np.rot90(sprite_patch, k=_rotation_steps(src_corner, dst_corner))
    return np.kron(rotated, np.ones((scale, scale), dtype=int))


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    assert np.any(g != BG), "input must contain foreground"

    components = _find_components(g, bg=BG)
    sprites: list[dict] = []
    singletons: dict[tuple[int, int], int] = {}
    for comp in components:
        bbox = comp["bbox"]
        r0, c0, r1, c1 = bbox
        if comp["area"] == 1:
            coord = comp["coords"][0]
            singletons[coord] = int(g[coord])
            continue
        if (
            comp["area"] == SPRITE_AREA
            and r1 - r0 + 1 == SPRITE_SIZE
            and c1 - c0 + 1 == SPRITE_SIZE
            and len(comp["colors"]) == SPRITE_COLORS_PER_OBJECT
            and np.all(g[r0 : r1 + 1, c0 : c1 + 1] != BG)
        ):
            sprites.append({"bbox": bbox, "patch": _crop(g, bbox)})
            continue
        raise AssertionError("unexpected foreground component")

    assert len(sprites) in (2, 3), "expected 2 or 3 sprites"
    assert len(singletons) == 4 * len(sprites), "expected four marker singletons per sprite"

    used_markers: set[tuple[int, int]] = set()
    objects: list[dict] = []
    for sprite in sprites:
        bbox = sprite["bbox"]
        corners = _corners_from_bbox(bbox)
        matches = []
        for src_corner in CORNERS:
            singleton = _outside_corner(src_corner, corners[src_corner])
            marker_color = singletons.get(singleton)
            if marker_color is None:
                continue
            for dst_corner in CORNERS:
                if src_corner == dst_corner:
                    continue
                center, l_markers = _l_group(dst_corner, bbox)
                if len({singleton, *l_markers}) != 4:
                    continue
                if all(singletons.get(pos) == marker_color for pos in l_markers):
                    matches.append(
                        {
                            "bbox": bbox,
                            "patch": sprite["patch"],
                            "src_corner": src_corner,
                            "dst_corner": dst_corner,
                            "marker_color": marker_color,
                            "marker_coords": set([singleton, *l_markers]),
                            "center": center,
                        }
                    )
        assert len(matches) == 1, "expected a unique marker assignment per sprite"
        match = matches[0]
        assert not (used_markers & match["marker_coords"]), "markers must not be shared"
        used_markers |= match["marker_coords"]
        objects.append(match)

    assert used_markers == set(singletons), "all singleton pixels must participate in a marker system"
    return {"objects": objects, "scale": len({obj["marker_color"] for obj in objects})}


def _render_scene(parsed: dict, shape: tuple[int, int], scale: int | None = None, rotate: bool = True) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    actual_scale = parsed["scale"] if scale is None else int(scale)
    centers: list[tuple[tuple[int, int], int]] = []
    for obj in parsed["objects"]:
        patch = obj["patch"]
        if rotate:
            patch = np.rot90(patch, k=_rotation_steps(obj["src_corner"], obj["dst_corner"]))
        scaled = np.kron(patch, np.ones((actual_scale, actual_scale), dtype=int))
        top, left, _bottom, _right = _scaled_bbox(obj["bbox"], actual_scale)
        _paste(out, scaled, top, left, clip=True)
        centers.append((obj["center"], obj["marker_color"]))
    for (r, c), color in centers:
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each marked 3x3 sprite, count distinct marker colors to get one global scale, then enlarge the rotated sprites.

    Input structure:
    - The background is black.
    - There are 2 or 3 filled 3x3 sprites, each using exactly two non-black colors.
    - Every sprite has four singleton marker pixels of one marker color: three form a missing-corner square around one sprite corner, and one sits diagonally outside another corner.
    - Different sprites may reuse or vary their marker color, and markers never overlap sprites or each other.

    Rule:
    - For each sprite, the lone diagonal marker names the source corner and the missing-corner square names the destination corner.
    - Rotate that sprite by the unique non-zero quarter-turn that maps the source corner to the destination corner.
    - Count how many distinct colors are used by all marker pixels across the whole grid; this count is the common scale factor.
    - Remove the input markers, enlarge every rotated sprite by that factor around its original 3x3 center with clipping at the canvas edge, and keep one marker-colored pixel at each missing-corner square center.

    Key insight:
    - The same marker system carries two meanings at once: locally it tells each sprite how to rotate, and globally its color diversity tells every sprite how much to scale.
    """
    parsed = _parse_scene(np.asarray(grid, dtype=int))
    return _render_scene(parsed, tuple(map(int, grid.shape)))


def _is_rot_asymmetric(patch: np.ndarray) -> bool:
    return all(not np.array_equal(patch, np.rot90(patch, k=k)) for k in (1, 2, 3))


def _random_sprite_patch(color_a: int, color_b: int) -> np.ndarray:
    positions = [(r, c) for r in range(SPRITE_SIZE) for c in range(SPRITE_SIZE)]
    for _ in range(100):
        count_a = random.randint(3, 6)
        patch = np.full((SPRITE_SIZE, SPRITE_SIZE), color_b, dtype=int)
        for r, c in random.sample(positions, count_a):
            patch[r, c] = color_a
        if len(set(map(int, patch.flat))) == 2 and _is_rot_asymmetric(patch):
            return patch
    raise RuntimeError("failed to sample an asymmetric sprite")


def _sample_rotation_pairs(count: int) -> list[tuple[str, str]]:
    pairs = [(src, dst) for src in CORNERS for dst in CORNERS if src != dst]
    for _ in range(100):
        chosen = random.sample(pairs, count)
        steps = {_rotation_steps(src, dst) for src, dst in chosen}
        if len(steps) >= min(2, count):
            return chosen
    raise RuntimeError("failed to diversify rotation steps")


def _sample_marker_assignment(count: int, distinct_count: int, forbidden: set[int]) -> list[int]:
    available = [c for c in SPRITE_PALETTE if c not in forbidden]
    distinct = random.sample(available, distinct_count)
    assignment = distinct.copy()
    while len(assignment) < count:
        assignment.append(random.choice(distinct))
    random.shuffle(assignment)
    if len(set(assignment)) != distinct_count:
        raise AssertionError("marker assignment lost a required color")
    return assignment


def _occupied_cells_for_input(sprite_bbox: tuple[int, int, int, int], src_corner: str, dst_corner: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = sprite_bbox
    center, l_markers = _l_group(dst_corner, sprite_bbox)
    singleton = _outside_corner(src_corner, _corners_from_bbox(sprite_bbox)[src_corner])
    cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    cells.update(l_markers)
    cells.add(singleton)
    cells.add(center)
    return cells


def _candidate_positions(
    shape: tuple[int, int], src_corner: str, dst_corner: str
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    candidates = []
    for r0 in range(h - SPRITE_SIZE + 1):
        for c0 in range(w - SPRITE_SIZE + 1):
            bbox = (r0, c0, r0 + 2, c0 + 2)
            cells = _occupied_cells_for_input(bbox, src_corner, dst_corner)
            if all(0 <= r < h and 0 <= c < w for r, c in cells):
                candidates.append(bbox)
    random.shuffle(candidates)
    return candidates


def _valid_placement(
    new_bbox: tuple[int, int, int, int],
    src_corner: str,
    dst_corner: str,
    placed: list[dict],
    shape: tuple[int, int],
    scale: int,
) -> bool:
    new_cells = _occupied_cells_for_input(new_bbox, src_corner, dst_corner)
    new_visible_out = _clip_bbox(_scaled_bbox(new_bbox, scale), shape)
    new_center, _ = _l_group(dst_corner, new_bbox)
    if new_visible_out is None:
        return False

    for prev in placed:
        if _boxes_overlap(new_bbox, prev["bbox"], gap=5):
            return False
        if new_cells & prev["occupied_input"]:
            return False
        prev_visible = _clip_bbox(_scaled_bbox(prev["bbox"], scale), shape)
        assert prev_visible is not None
        if _boxes_overlap(new_visible_out, prev_visible, gap=1):
            return False
        prev_center = prev["center"]
        if (
            prev_visible[0] <= new_center[0] <= prev_visible[2]
            and prev_visible[1] <= new_center[1] <= prev_visible[3]
        ):
            return False
        if (
            new_visible_out[0] <= prev_center[0] <= new_visible_out[2]
            and new_visible_out[1] <= prev_center[1] <= new_visible_out[3]
        ):
            return False
    return True


def _build_input_grid(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = obj["patch"]
        center, l_markers = _l_group(obj["dst_corner"], obj["bbox"])
        singleton = _outside_corner(obj["src_corner"], _corners_from_bbox(obj["bbox"])[obj["src_corner"]])
        for r, c in [singleton, *l_markers]:
            grid[r, c] = obj["marker_color"]
        assert grid[center] == BG
    return grid


def _check_semantics(inp: np.ndarray, planned_scale: int, want_clip: bool, object_count: int) -> bool:
    parsed = _parse_scene(inp)
    if parsed["scale"] != planned_scale or len(parsed["objects"]) != object_count:
        return False

    out = main(inp)
    if not np.array_equal(out, _render_scene(parsed, inp.shape)):
        return False

    sprite_color_count = len({int(v) for v in inp.flat if int(v) != BG and int(v) not in {obj["marker_color"] for obj in parsed["objects"]}})
    if sprite_color_count == planned_scale:
        return False

    if planned_scale > 1 and np.array_equal(out, _render_scene(parsed, inp.shape, scale=1)):
        return False
    if np.array_equal(out, _render_scene(parsed, inp.shape, rotate=False)):
        return False

    clipped_count = 0
    visible_boxes = []
    for obj in parsed["objects"]:
        raw_box = _scaled_bbox(obj["bbox"], planned_scale)
        vis_box = _clip_bbox(raw_box, inp.shape)
        if vis_box is None:
            return False
        visible_boxes.append(vis_box)
        if raw_box != vis_box:
            clipped_count += 1
    for i in range(len(visible_boxes)):
        for j in range(i + 1, len(visible_boxes)):
            if _boxes_overlap(visible_boxes[i], visible_boxes[j], gap=1):
                return False
    if want_clip and clipped_count == 0:
        return False
    if not want_clip and clipped_count != 0:
        return False

    rotation_steps = {_rotation_steps(obj["src_corner"], obj["dst_corner"]) for obj in parsed["objects"]}
    if len(rotation_steps) < min(2, len(parsed["objects"])):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    object_count = 3 if random.random() < 0.6 else 2
    scale = random.randint(1, object_count)
    want_clip = scale > 1 and random.random() < 0.35

    if object_count == 3 and scale == 3:
        h = random.randint(27, 30)
        w = random.randint(27, 30)
    elif object_count == 3 or scale == 3:
        h = random.randint(24, 30)
        w = random.randint(24, 30)
    else:
        h = random.randint(20, 30)
        w = random.randint(20, 30)

    rotation_pairs = _sample_rotation_pairs(object_count)

    sprite_forbidden_pool = set()
    sprite_specs = []
    for _ in range(object_count):
        color_a, color_b = random.sample(SPRITE_PALETTE, 2)
        sprite_specs.append((color_a, color_b))
        sprite_forbidden_pool.update((color_a, color_b))

    marker_colors = _sample_marker_assignment(object_count, scale, sprite_forbidden_pool)
    objects: list[dict] = []

    for idx in range(object_count):
        src_corner, dst_corner = rotation_pairs[idx]
        bbox_candidates = _candidate_positions((h, w), src_corner, dst_corner)
        patch = _random_sprite_patch(*sprite_specs[idx])
        random.shuffle(bbox_candidates)
        chosen = None

        if want_clip:
            edge_bias = []
            interior = []
            for bbox in bbox_candidates:
                raw = _scaled_bbox(bbox, scale)
                if _clip_bbox(raw, (h, w)) != raw:
                    edge_bias.append(bbox)
                else:
                    interior.append(bbox)
            bbox_candidates = edge_bias + interior

        for bbox in bbox_candidates:
            if _valid_placement(bbox, src_corner, dst_corner, objects, (h, w), scale):
                chosen = bbox
                break
        if chosen is None:
            return None

        objects.append(
            {
                "bbox": chosen,
                "patch": patch,
                "src_corner": src_corner,
                "dst_corner": dst_corner,
                "marker_color": marker_colors[idx],
                "occupied_input": _occupied_cells_for_input(chosen, src_corner, dst_corner),
                "center": _l_group(dst_corner, chosen)[0],
            }
        )

    if want_clip:
        if not any(_clip_bbox(_scaled_bbox(obj["bbox"], scale), (h, w)) != _scaled_bbox(obj["bbox"], scale) for obj in objects):
            return None
    if len(set(marker_colors)) != scale:
        return None

    grid = _build_input_grid((h, w), objects)
    if not _check_semantics(grid, scale, want_clip, object_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
