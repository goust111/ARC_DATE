from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, find_components, in_bounds, paste


BG = Color.BLACK
FRAME = Color.GREEN
ANCHOR = Color.GREY
DIR_MARK = Color.MAROON
SPRITE_COLORS = [Color.BLUE, Color.RED, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL]
MAX_GENERATION_ATTEMPTS = 400


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _mirror_rel_pixels(rel_pixels: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return [(dr, -dc, color) for dr, dc, color in rel_pixels]


def _copy_abs_pixels(
    rel_pixels: list[tuple[int, int, int]], target: tuple[int, int]
) -> list[tuple[int, int, int]]:
    tr, tc = target
    return [(tr + dr, tc + dc, color) for dr, dc, color in rel_pixels]


def _source_data(grid: np.ndarray) -> dict:
    candidates = []
    for comp in find_components(grid, bg=BG, connectivity=4, monochromatic=False, return_format="dict"):
        coords = comp["coords"]
        vals = grid[coords[:, 0], coords[:, 1]]
        if int(np.count_nonzero(vals == ANCHOR)) == 1 and np.any(vals != ANCHOR):
            candidates.append(comp)
    assert len(candidates) == 1, "expected exactly one multicolor source sprite with one gray anchor"

    source = candidates[0]
    coords = source["coords"]
    vals = grid[coords[:, 0], coords[:, 1]]
    anchor = tuple(map(int, coords[vals == ANCHOR][0]))
    sr, sc = anchor

    rel_pixels = []
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        color = int(grid[rr, cc])
        if color == ANCHOR:
            continue
        rel_pixels.append((rr - sr, cc - sc, color))

    return {"anchor": anchor, "rel_pixels": rel_pixels}


def _rectangle_data(grid: np.ndarray) -> list[dict]:
    frame_mask = np.isin(grid, [FRAME, DIR_MARK]).astype(int)
    comps = find_components(
        frame_mask,
        bg=0,
        colors=[1],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )

    rects: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        sub = crop_bbox(grid, comp["bbox"])
        h, w = sub.shape
        if h < 5 or w < 5:
            continue

        frame_like = np.isin(sub, [FRAME, DIR_MARK])
        border = _border_mask(sub.shape)
        if not np.array_equal(frame_like, border):
            continue

        marks = np.argwhere(sub == DIR_MARK)
        if marks.shape != (1, 2):
            continue
        mr, mc = map(int, marks[0])
        if h % 2 != 1 or mr != h // 2 or mc not in (0, w - 1):
            continue

        rects.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "direction": "left" if mc == 0 else "right",
                "area": int(h * w),
            }
        )

    assert rects, "expected at least one green rectangle with one maroon side marker"
    rects.sort(key=lambda item: item["bbox"])
    return rects


def _classify_targets(
    grid: np.ndarray, source_anchor: tuple[int, int], rects: list[dict]
) -> tuple[list[dict], list[tuple[int, int]]]:
    targets = [tuple(map(int, rc)) for rc in np.argwhere(grid == ANCHOR) if tuple(map(int, rc)) != source_anchor]
    valid: list[dict] = []
    invalid: list[tuple[int, int]] = []

    for target in targets:
        owners = []
        for rect in rects:
            r0, c0, r1, c1 = rect["bbox"]
            if r0 < target[0] < r1 and c0 < target[1] < c1:
                owners.append(rect)
        assert len(owners) <= 1, "target anchor lies inside multiple rectangle interiors"
        if owners:
            valid.append({"anchor": target, "rect": owners[0]})
        else:
            invalid.append(target)

    assert valid, "expected at least one valid target anchor inside a rectangle"
    return valid, invalid


def _protected_mask(shape: tuple[int, int], rects: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        mask[r0, c0 : c1 + 1] = True
        mask[r1, c0 : c1 + 1] = True
        mask[r0 : r1 + 1, c0] = True
        mask[r0 : r1 + 1, c1] = True
    return mask


def _render_output_from_parts(
    base: np.ndarray, rel_pixels: list[tuple[int, int, int]], targets: list[dict], rects: list[dict]
) -> np.ndarray:
    out = base.copy()
    protected = _protected_mask(base.shape, rects)

    for item in targets:
        tr, tc = item["anchor"]
        out[tr, tc] = BG

    claims: dict[tuple[int, int], tuple[tuple[int, int, int, int, int], int]] = {}
    for idx, item in enumerate(targets):
        rect = item["rect"]
        rel = _mirror_rel_pixels(rel_pixels) if rect["direction"] == "left" else rel_pixels
        priority = (
            int(rect["area"]),
            int(rect["bbox"][0]),
            int(rect["bbox"][1]),
            int(item["anchor"][0]),
            int(item["anchor"][1]),
        )
        for rr, cc, color in _copy_abs_pixels(rel, item["anchor"]):
            if not in_bounds(base.shape, rr, cc) or protected[rr, cc]:
                continue
            key = (rr, cc)
            current = claims.get(key)
            if current is None or priority < current[0]:
                claims[key] = (priority, int(color))

    for (r, c), (_priority, color) in claims.items():
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the anchored multicolor sprite into rectangle-selected targets, mirroring only when the container marker sits on the left border.

    Input structure:
    - Black background with one 4-connected multicolor source sprite that contains exactly one gray anchor pixel.
    - Several green hollow rectangles with a single maroon marker on the midpoint of either the left or the right border.
    - Additional isolated gray pixels act as target anchors; some lie strictly inside a rectangle interior and others are distractors outside every rectangle bbox.

    Rule:
    - Identify the source sprite and separate its gray anchor from the non-gray sprite pixels.
    - Keep only those gray target pixels that lie strictly inside exactly one green rectangle; ignore outside distractors.
    - For each valid target, compute the translation from the source anchor to that target anchor.
    - If the owning rectangle's maroon marker is on the left border, horizontally mirror the sprite around the anchor before translating; if the marker is on the right border, keep the original orientation.
    - Erase each valid target anchor to black, copy the translated non-gray sprite pixels, and never overwrite any green frame pixel or maroon marker pixel.

    Key insight:
    - The green rectangles simultaneously choose which gray anchors are active and encode each copy's orientation, so anchor alignment alone is insufficient without rectangle containment and side-marker reading.
    """
    g = np.asarray(grid, dtype=int)
    source = _source_data(g)
    rects = _rectangle_data(g)
    targets, _invalid = _classify_targets(g, source["anchor"], rects)
    return _render_output_from_parts(g, source["rel_pixels"], targets, rects)


def _random_connected_cells(h: int, w: int, total: int) -> list[tuple[int, int]] | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set()

    while len(cells) < total:
        frontier.clear()
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    return list(cells)


def _generate_sprite() -> dict | None:
    for _ in range(120):
        h = random.randint(4, 6)
        w = random.randint(4, 6)
        total = random.randint(8, min(15, h * w))
        cells = _random_connected_cells(h, w, total)
        if cells is None:
            continue

        anchor = random.choice(cells)
        rel_shape = {(r - anchor[0], c - anchor[1]) for r, c in cells}
        if rel_shape == {(dr, -dc) for dr, dc in rel_shape}:
            continue

        non_anchor = [cell for cell in cells if cell != anchor]
        n_colors = random.randint(2, min(3, len(non_anchor)))
        palette = random.sample(SPRITE_COLORS, n_colors)
        assigned = palette[:]
        while len(assigned) < len(non_anchor):
            assigned.append(random.choice(palette))
        random.shuffle(assigned)

        patch = np.zeros((h, w), dtype=int)
        for (r, c), color in zip(non_anchor, assigned):
            patch[r, c] = int(color)
        patch[anchor] = ANCHOR

        bb = bbox(patch != BG)
        assert bb is not None
        patch = crop_bbox(patch, bb)
        ar, ac = anchor
        r0, c0, _r1, _c1 = bb
        anchor_in_patch = (ar - r0, ac - c0)

        rel_pixels = []
        rel_shape = set()
        for rr, cc in np.argwhere(patch != BG):
            r = int(rr)
            c = int(cc)
            rel = (r - anchor_in_patch[0], c - anchor_in_patch[1])
            rel_shape.add(rel)
            color = int(patch[r, c])
            if color != ANCHOR:
                rel_pixels.append((rel[0], rel[1], color))

        if rel_shape == {(dr, -dc) for dr, dc in rel_shape}:
            continue

        return {
            "patch": patch,
            "anchor_in_patch": anchor_in_patch,
            "rel_pixels": rel_pixels,
            "rel_shape": rel_shape,
        }
    return None


def _rel_extents(rel_pixels: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    rows = [dr for dr, _dc, _color in rel_pixels]
    cols = [dc for _dr, dc, _color in rel_pixels]
    return min(rows), max(rows), min(cols), max(cols)


def _boxes_far_enough(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _size_options(sprite: dict) -> list[tuple[int, int]]:
    rel_all = list(sprite["rel_shape"])
    rows = [dr for dr, _dc in rel_all]
    cols = [dc for _dr, dc in rel_all]
    sprite_h = max(rows) - min(rows) + 1
    sprite_w = max(cols) - min(cols) + 1
    options = []
    min_h = max(5, sprite_h + 3)
    if min_h % 2 == 0:
        min_h += 1
    for h in range(min_h, 12, 2):
        for w in range(max(5, sprite_w + 3), 12):
            options.append((h, w))
    random.shuffle(options)
    return options


def _choose_rect_specs(sprite: dict, rect_count: int) -> list[dict] | None:
    options = _size_options(sprite)
    by_area: dict[int, list[tuple[int, int]]] = {}
    for h, w in options:
        by_area.setdefault(h * w, []).append((h, w))
    areas = list(by_area)
    random.shuffle(areas)
    if len(areas) < rect_count:
        return None

    chosen_areas = random.sample(areas, rect_count)
    specs = []
    directions = ["left", "right"]
    while len(directions) < rect_count:
        directions.append(random.choice(["left", "right"]))
    random.shuffle(directions)
    if "left" not in directions or "right" not in directions:
        return None

    for area, direction in zip(chosen_areas, directions):
        h, w = random.choice(by_area[area])
        specs.append({"height": h, "width": w, "direction": direction})

    specs.sort(key=lambda item: item["height"] * item["width"], reverse=True)
    return specs


def _place_rectangles(
    shape: tuple[int, int], specs: list[dict]
) -> list[dict] | None:
    h, w = shape
    rects: list[dict] = []
    for spec in specs:
        rh = spec["height"]
        rw = spec["width"]
        candidates = []
        for r0 in range(0, h - rh + 1):
            for c0 in range(0, w - rw + 1):
                box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                if all(_boxes_far_enough(box, other["bbox"], gap=2) for other in rects):
                    candidates.append(box)
        if not candidates:
            return None
        r0, c0, r1, c1 = random.choice(candidates)
        rects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "direction": spec["direction"],
                "area": rh * rw,
            }
        )
    rects.sort(key=lambda item: item["bbox"])
    return rects


def _source_bbox_at(sprite: dict, top_left: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = top_left
    h, w = sprite["patch"].shape
    return r0, c0, r0 + h - 1, c0 + w - 1


def _place_source(
    shape: tuple[int, int], sprite: dict, rects: list[dict]
) -> tuple[int, int] | None:
    h, w = shape
    sh, sw = sprite["patch"].shape
    candidates = []
    for r0 in range(0, h - sh + 1):
        for c0 in range(0, w - sw + 1):
            box = _source_bbox_at(sprite, (r0, c0))
            if all(_boxes_far_enough(box, rect["bbox"], gap=2) for rect in rects):
                candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _copy_signature(
    rect: dict, rel_pixels: list[tuple[int, int, int]], target: tuple[int, int]
) -> dict:
    r0, c0, r1, c1 = rect["bbox"]
    abs_pixels = _copy_abs_pixels(rel_pixels, target)
    visible = set()
    masked = set()
    for rr, cc, color in abs_pixels:
        if not (r0 <= rr <= r1 and c0 <= cc <= c1):
            return {"ok": False}
        item = (int(rr), int(cc), int(color))
        if rr in (r0, r1) or cc in (c0, c1):
            masked.add(item)
        else:
            visible.add(item)
    return {"ok": bool(visible), "visible": visible, "masked": masked}


def _anchor_candidates(
    rect: dict,
    rel_pixels: list[tuple[int, int, int]],
    require_border_touch: bool,
) -> list[dict]:
    r0, c0, r1, c1 = rect["bbox"]
    min_dr, max_dr, min_dc, max_dc = _rel_extents(rel_pixels)
    row_lo = max(r0 + 2, r0 - min_dr)
    row_hi = min(r1 - 2, r1 - max_dr)
    col_lo = max(c0 + 2, c0 - min_dc)
    col_hi = min(c1 - 2, c1 - max_dc)

    candidates = []
    for rr in range(row_lo, row_hi + 1):
        for cc in range(col_lo, col_hi + 1):
            sig = _copy_signature(rect, rel_pixels, (rr, cc))
            if not sig["ok"]:
                continue
            if require_border_touch and not sig["masked"]:
                continue
            if not require_border_touch and sig["masked"] and random.random() < 0.65:
                continue
            candidates.append({"anchor": (rr, cc), "visible": sig["visible"], "masked": sig["masked"]})
    random.shuffle(candidates)
    return candidates


def _pick_valid_targets(scene: dict) -> bool:
    rects = scene["rects"]
    rel_pixels = scene["sprite"]["rel_pixels"]
    left_rects = [idx for idx, rect in enumerate(rects) if rect["direction"] == "left"]
    right_rects = [idx for idx, rect in enumerate(rects) if rect["direction"] == "right"]
    if not left_rects or not right_rects:
        return False

    max_valid = min(len(rects), 4)
    min_valid = 2
    valid_count = random.randint(min_valid, max_valid)
    chosen = {random.choice(left_rects), random.choice(right_rects)}
    remaining = [idx for idx in range(len(rects)) if idx not in chosen]
    random.shuffle(remaining)
    for idx in remaining[: max(0, valid_count - len(chosen))]:
        chosen.add(idx)

    chosen_list = sorted(chosen)
    mask_rect_idx = random.choice(chosen_list)
    valid_targets = []
    seen_visible: set[tuple[int, int]] = set()
    mirror_witness = False
    masked_witness = False

    for rect_idx in chosen_list:
        rect = rects[rect_idx]
        rel = _mirror_rel_pixels(rel_pixels) if rect["direction"] == "left" else rel_pixels
        require_border_touch = rect_idx == mask_rect_idx
        candidates = _anchor_candidates(rect, rel, require_border_touch=require_border_touch)
        if rect["direction"] == "left":
            filtered = []
            for cand in candidates:
                alt = _copy_signature(rect, rel_pixels, cand["anchor"])
                if alt["ok"] and alt["visible"] != cand["visible"]:
                    filtered.append(cand)
            if filtered:
                candidates = filtered
        placed = None
        for cand in candidates:
            coords_only = {(r, c) for r, c, _color in cand["visible"]}
            if coords_only & seen_visible:
                continue
            placed = cand
            seen_visible |= coords_only
            break
        if placed is None:
            return False

        if rect["direction"] == "left":
            alt = _copy_signature(rect, rel_pixels, placed["anchor"])
            if alt["ok"] and alt["visible"] != placed["visible"]:
                mirror_witness = True
        if placed["masked"]:
            masked_witness = True
        valid_targets.append({"anchor": placed["anchor"], "rect_index": rect_idx})

    if not mirror_witness or not masked_witness:
        return False

    scene["valid_targets"] = valid_targets
    return True


def _draw_rectangles(grid: np.ndarray, rects: list[dict]) -> None:
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME
        mr = (r0 + r1) // 2
        mc = c0 if rect["direction"] == "left" else c1
        grid[mr, mc] = DIR_MARK


def _can_place_invalid_anchor(grid: np.ndarray, rects: list[dict], coord: tuple[int, int]) -> bool:
    r, c = coord
    if grid[r, c] != BG:
        return False
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        if r0 <= r <= r1 and c0 <= c <= c1:
            return False
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if in_bounds(grid.shape, rr, cc) and grid[rr, cc] != BG:
                return False
    return True


def _render_input(scene: dict) -> np.ndarray:
    grid = np.zeros(scene["shape"], dtype=int)
    source = scene["sprite"]
    sr0, sc0 = scene["source_top_left"]
    paste(grid, source["patch"], sr0, sc0, transparent=BG)
    _draw_rectangles(grid, scene["rects"])
    for item in scene["valid_targets"]:
        grid[item["anchor"]] = ANCHOR
    for coord in scene["invalid_targets"]:
        grid[coord] = ANCHOR
    return grid


def _source_anchor_abs(scene: dict) -> tuple[int, int]:
    sr0, sc0 = scene["source_top_left"]
    ar, ac = scene["sprite"]["anchor_in_patch"]
    return sr0 + ar, sc0 + ac


def _target_items_for_render(scene: dict) -> list[dict]:
    items = []
    for item in scene["valid_targets"]:
        items.append({"anchor": item["anchor"], "rect": scene["rects"][item["rect_index"]]})
    return items


def _render_output(scene: dict, protect_frames: bool = True, mirror_override: dict[int, str] | None = None) -> np.ndarray:
    inp = _render_input(scene)
    rects = scene["rects"]
    rel_pixels = scene["sprite"]["rel_pixels"]
    items = []
    for item in scene["valid_targets"]:
        rect = dict(rects[item["rect_index"]])
        if mirror_override and item["rect_index"] in mirror_override:
            rect["direction"] = mirror_override[item["rect_index"]]
        items.append({"anchor": item["anchor"], "rect": rect})

    if protect_frames:
        return _render_output_from_parts(inp, rel_pixels, items, rects)

    out = inp.copy()
    for item in items:
        tr, tc = item["anchor"]
        out[tr, tc] = BG
    claims: dict[tuple[int, int], tuple[tuple[int, int, int, int, int], int]] = {}
    for item in items:
        rect = item["rect"]
        rel = _mirror_rel_pixels(rel_pixels) if rect["direction"] == "left" else rel_pixels
        priority = (
            int(rect["area"]),
            int(rect["bbox"][0]),
            int(rect["bbox"][1]),
            int(item["anchor"][0]),
            int(item["anchor"][1]),
        )
        for rr, cc, color in _copy_abs_pixels(rel, item["anchor"]):
            if not in_bounds(out.shape, rr, cc):
                continue
            key = (rr, cc)
            current = claims.get(key)
            if current is None or priority < current[0]:
                claims[key] = (priority, int(color))
    for (r, c), (_priority, color) in claims.items():
        out[r, c] = color
    return out


def _validate_scene(scene: dict) -> bool:
    inp = _render_input(scene)
    out = _render_output(scene)
    if not np.array_equal(main(inp), out):
        return False

    source_anchor = _source_anchor_abs(scene)
    if int(inp[source_anchor]) != ANCHOR:
        return False

    valid_targets = scene["valid_targets"]
    invalid_targets = scene["invalid_targets"]
    if not valid_targets or not invalid_targets:
        return False
    if len(valid_targets) + len(invalid_targets) < 3 or len(valid_targets) + len(invalid_targets) > 6:
        return False

    directions = {scene["rects"][item["rect_index"]]["direction"] for item in valid_targets}
    if directions != {"left", "right"}:
        return False

    for coord in invalid_targets:
        if int(out[coord]) != ANCHOR:
            return False

    protected = _protected_mask(inp.shape, scene["rects"])
    if not np.array_equal(out[protected], inp[protected]):
        return False

    full_rel = scene["sprite"]["rel_shape"]
    if full_rel == {(dr, -dc) for dr, dc in full_rel}:
        return False

    for item in valid_targets:
        rect = scene["rects"][item["rect_index"]]
        rel = _mirror_rel_pixels(scene["sprite"]["rel_pixels"]) if rect["direction"] == "left" else scene["sprite"]["rel_pixels"]
        sig = _copy_signature(rect, rel, item["anchor"])
        if not sig["ok"] or not sig["visible"]:
            return False

    masked_diff = _render_output(scene, protect_frames=False)
    if np.array_equal(masked_diff, out):
        return False

    mirror_rect = next(item["rect_index"] for item in valid_targets if scene["rects"][item["rect_index"]]["direction"] == "left")
    unmirrored = _render_output(scene, mirror_override={mirror_rect: "right"})
    if np.array_equal(unmirrored, out):
        return False

    visible_sets = []
    for item in valid_targets:
        rect = scene["rects"][item["rect_index"]]
        rel = _mirror_rel_pixels(scene["sprite"]["rel_pixels"]) if rect["direction"] == "left" else scene["sprite"]["rel_pixels"]
        sig = _copy_signature(rect, rel, item["anchor"])
        cells = {(r, c) for r, c, _color in sig["visible"]}
        visible_sets.append(cells)
    if any(a & b for i, a in enumerate(visible_sets) for b in visible_sets[i + 1 :]):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    sprite = _generate_sprite()
    if sprite is None:
        return None

    rect_count = random.randint(2, 4)
    specs = _choose_rect_specs(sprite, rect_count)
    if specs is None:
        return None

    rows = random.randint(18, 26)
    cols = random.randint(18, 26)
    if random.random() < 0.55:
        cols = min(26, max(18, rows + random.randint(-2, 2)))

    rects = _place_rectangles((rows, cols), specs)
    if rects is None:
        return None

    source_top_left = _place_source((rows, cols), sprite, rects)
    if source_top_left is None:
        return None

    scene = {
        "shape": (rows, cols),
        "sprite": sprite,
        "rects": rects,
        "source_top_left": source_top_left,
        "valid_targets": [],
        "invalid_targets": [],
    }

    if not _pick_valid_targets(scene):
        return None

    input_grid = _render_input(scene)

    max_invalid = min(6 - len(scene["valid_targets"]), 3)
    if max_invalid < 1:
        return None
    invalid_count = random.randint(1, max_invalid)
    candidates = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if _can_place_invalid_anchor(input_grid, rects, (r, c))
    ]
    random.shuffle(candidates)
    if len(candidates) < invalid_count:
        return None
    scene["invalid_targets"] = candidates[:invalid_count]

    if not _validate_scene(scene):
        return None
    return _render_input(scene)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchored-copy ARC input")
