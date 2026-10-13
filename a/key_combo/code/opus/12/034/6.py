from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 6, 7)
GRID_SIDE_RANGE = (18, 26)
MIN_BBOX_GAP = 2
GEN_ATTEMPTS = 400


def _parse_mask(spec: str) -> np.ndarray:
    lines = spec.split("/")
    width = len(lines[0])
    mask = np.array([[ch == "#" for ch in row] for row in lines], dtype=bool)
    assert all(len(row) == width for row in lines)
    assert mask.any()
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


SYMMETRIC_LIBRARY = {
    3: [_parse_mask("###"), _parse_mask("#/#/#")],
    4: [_parse_mask("##/##"), _parse_mask("###/.#."), _parse_mask("####")],
    5: [
        _parse_mask(".#./###/.#."),
        _parse_mask("###/.#./.#."),
        _parse_mask("#.#/###"),
        _parse_mask("#####"),
    ],
    6: [_parse_mask("##/##/##"), _parse_mask(".#./###/#.#"), _parse_mask("######")],
    7: [
        _parse_mask("###/.#./###"),
        _parse_mask("..#../#####/..#.."),
        _parse_mask("#.#/###/#.#"),
        _parse_mask("#######"),
    ],
}

ASYMMETRIC_LIBRARY = {
    3: [_parse_mask("##/#."), _parse_mask("##/.#"), _parse_mask("#./##")],
    4: [_parse_mask("#../###"), _parse_mask("##./.#./.#."), _parse_mask("#./##/.#")],
    5: [
        _parse_mask("#../#../###"),
        _parse_mask("##./.##/..#"),
        _parse_mask("###/#../#.."),
        _parse_mask("##./##./#.."),
    ],
    6: [
        _parse_mask("####/##.."),
        _parse_mask("#../###/##."),
        _parse_mask("##./###/#.."),
        _parse_mask("###/##./#.."),
    ],
    7: [
        _parse_mask(".##/###/##."),
        _parse_mask("####/###."),
        _parse_mask("###/###/#.."),
        _parse_mask("##../####/.#.."),
        _parse_mask("##./###/.##"),
    ],
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    bbox = _bbox_of_mask(mask)
    if bbox is None:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = bbox
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _is_vertically_symmetric(mask: np.ndarray) -> bool:
    return np.array_equal(mask, np.fliplr(mask))


def _flip_variant(mask: np.ndarray, rng: random.Random, preserve_vertical_symmetry: bool) -> np.ndarray:
    out = mask.copy()
    if rng.random() < 0.5:
        out = np.flipud(out)
    if not preserve_vertical_symmetry and rng.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            coords = []
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for cr, cc in coords:
                mask[cr - r0, cc - c0] = True
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )
    return objects


def _paint_mask(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int, *, clip: bool) -> None:
    h, w = mask.shape
    gh, gw = grid.shape
    for rr in range(h):
        for cc in range(w):
            if not mask[rr, cc]:
                continue
            gr = r0 + rr
            gc = c0 + cc
            if clip:
                if 0 <= gr < gh and 0 <= gc < gw:
                    grid[gr, gc] = int(color)
            else:
                if not (0 <= gr < gh and 0 <= gc < gw):
                    raise ValueError("paint out of bounds")
                grid[gr, gc] = int(color)


def _expanded_boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a[0] - pad, a[1] - pad, a[2] + pad, a[3] + pad
    br0, bc0, br1, bc1 = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _visible_patch(mask: np.ndarray, grid_shape: tuple[int, int], r0: int, c0: int) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    gh, gw = grid_shape
    h, w = mask.shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(gh, r0 + h)
    cc1 = min(gw, c0 + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return np.zeros((0, 0), dtype=bool), None
    patch = mask[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0].copy()
    bbox = _bbox_of_mask(patch)
    if bbox is None:
        return patch, None
    pr0, pc0, pr1, pc1 = bbox
    return patch, (rr0 + pr0, cc0 + pc0, rr0 + pr1, cc0 + pc1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify sprites by vertical symmetry, pair equal-area template/target objects, and stamp template shapes onto the target positions.

    Input structure:
    - A black background contains several small monochrome 4-connected sprites.
    - Some cropped sprite masks are vertically mirror-symmetric; the rest are not.
    - Each sprite has a well-defined bounding box and pixel count.

    Rule:
    - Crop every sprite to its minimal bounding box and test whether its mask equals its left-right mirror.
    - Treat symmetric sprites as templates and non-symmetric sprites as targets.
    - Pair a template with a target when their foreground pixel counts are equal.
    - For each pair, erase the template from its original location, erase the target pixels, then stamp the template mask in the target color at the target bbox's top-left corner, clipping only at the grid boundary.

    Key insight:
    - Shape, color, and position come from different sources: symmetry picks the template shape, equal area finds its partner, and the target contributes the color and anchor position.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    objects = _extract_objects(g)

    templates_by_area: dict[int, list[dict]] = {}
    targets_by_area: dict[int, list[dict]] = {}
    for obj in objects:
        bucket = templates_by_area if _is_vertically_symmetric(obj["mask"]) else targets_by_area
        bucket.setdefault(int(obj["area"]), []).append(obj)

    pairs: list[tuple[dict, dict]] = []
    for area, templates in templates_by_area.items():
        targets = targets_by_area.get(area, [])
        if len(templates) == 1 and len(targets) == 1:
            pairs.append((templates[0], targets[0]))

    for template, target in pairs:
        for r, c in template["coords"]:
            out[r, c] = BG
        for r, c in target["coords"]:
            out[r, c] = BG

    for template, target in pairs:
        r0, c0, _r1, _c1 = target["bbox"]
        _paint_mask(out, template["mask"], int(target["color"]), int(r0), int(c0), clip=True)

    return out


def _validate_library() -> None:
    for area, masks in SYMMETRIC_LIBRARY.items():
        for mask in masks:
            assert int(mask.sum()) == area
            assert _mask_connected(mask)
            assert _is_vertically_symmetric(mask)
    for area, masks in ASYMMETRIC_LIBRARY.items():
        for mask in masks:
            assert int(mask.sum()) == area
            assert _mask_connected(mask)
            assert not _is_vertically_symmetric(mask)


def _all_pair_candidates() -> list[dict]:
    combos = []
    for area in range(3, 8):
        for template in SYMMETRIC_LIBRARY[area]:
            for target in ASYMMETRIC_LIBRARY[area]:
                t_bbox_area = int(template.shape[0] * template.shape[1])
                g_bbox_area = int(target.shape[0] * target.shape[1])
                if t_bbox_area > g_bbox_area:
                    relation = "larger"
                elif t_bbox_area < g_bbox_area:
                    relation = "smaller"
                else:
                    relation = "equal"
                combos.append(
                    {
                        "area": area,
                        "template": template,
                        "target": target,
                        "relation": relation,
                    }
                )
    return combos


PAIR_CANDIDATES = _all_pair_candidates()
_validate_library()


def _choose_pairs(rng: random.Random) -> list[dict]:
    pairs = []
    used_areas = set()

    for required in ("larger", "smaller"):
        options = [c for c in PAIR_CANDIDATES if c["relation"] == required and c["area"] not in used_areas]
        if not options:
            raise RuntimeError(f"missing {required} pair candidates")
        choice = rng.choice(options).copy()
        used_areas.add(int(choice["area"]))
        pairs.append(choice)

    remaining = [c for c in PAIR_CANDIDATES if c["area"] not in used_areas]
    pairs.append(rng.choice(remaining).copy())
    rng.shuffle(pairs)

    colors = list(PALETTE)
    rng.shuffle(colors)
    for idx, pair in enumerate(pairs):
        pair["template"] = _flip_variant(pair["template"], rng, preserve_vertical_symmetry=True)
        pair["target"] = _flip_variant(pair["target"], rng, preserve_vertical_symmetry=False)
        assert _is_vertically_symmetric(pair["template"])
        assert not _is_vertically_symmetric(pair["target"])
        pair["template_color"] = int(colors[2 * idx])
        pair["target_color"] = int(colors[2 * idx + 1])
    return pairs


def _candidate_positions(
    grid_shape: tuple[int, int],
    target_mask: np.ndarray,
    template_mask: np.ndarray,
    clip_dirs: tuple[str, ...],
) -> list[tuple[int, int]]:
    gh, gw = grid_shape
    th, tw = target_mask.shape
    sh, sw = template_mask.shape

    if "bottom" in clip_dirs:
        r_values = [gh - th]
    else:
        max_h = max(th, sh)
        if gh < max_h:
            return []
        r_values = list(range(0, gh - max_h + 1))

    if "right" in clip_dirs:
        c_values = [gw - tw]
    else:
        max_w = max(tw, sw)
        if gw < max_w:
            return []
        c_values = list(range(0, gw - max_w + 1))

    return [(r0, c0) for r0 in r_values for c0 in c_values]


def _place_targets(rng: random.Random, pairs: list[dict], grid_shape: tuple[int, int]) -> bool:
    input_boxes: list[tuple[int, int, int, int]] = []
    output_boxes: list[tuple[int, int, int, int]] = []

    clip_options = []
    for idx, pair in enumerate(pairs):
        dirs = []
        if pair["template"].shape[0] > pair["target"].shape[0]:
            dirs.append("bottom")
        if pair["template"].shape[1] > pair["target"].shape[1]:
            dirs.append("right")
        if dirs:
            clip_options.append((idx, tuple(dirs)))

    chosen_clip: dict[int, tuple[str, ...]] = {}
    if clip_options and rng.random() < 0.45:
        clip_idx, dirs = rng.choice(clip_options)
        selected = [rng.choice(dirs)]
        if len(dirs) == 2 and rng.random() < 0.5:
            selected = list(dirs)
        chosen_clip[clip_idx] = tuple(selected)

    order = list(range(len(pairs)))
    order.sort(
        key=lambda i: max(
            pairs[i]["template"].shape[0] * pairs[i]["template"].shape[1],
            pairs[i]["target"].shape[0] * pairs[i]["target"].shape[1],
        ),
        reverse=True,
    )
    if chosen_clip:
        clip_idx = next(iter(chosen_clip))
        order.remove(clip_idx)
        order.insert(0, clip_idx)

    for idx in order:
        pair = pairs[idx]
        clip_dirs = chosen_clip.get(idx, ())
        candidates = _candidate_positions(grid_shape, pair["target"], pair["template"], clip_dirs)
        rng.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            th, tw = pair["target"].shape
            target_bbox = (r0, c0, r0 + th - 1, c0 + tw - 1)
            if any(_expanded_boxes_overlap(target_bbox, box, MIN_BBOX_GAP) for box in input_boxes):
                continue

            visible_mask, output_bbox = _visible_patch(pair["template"], grid_shape, r0, c0)
            if output_bbox is None:
                continue
            if not _mask_connected(_crop_mask(visible_mask)):
                continue
            if any(_expanded_boxes_overlap(output_bbox, box, MIN_BBOX_GAP) for box in output_boxes):
                continue

            pair["target_pos"] = (r0, c0)
            pair["target_bbox"] = target_bbox
            pair["output_bbox"] = output_bbox
            pair["visible_output_area"] = int(visible_mask.sum())
            pair["did_clip"] = int(visible_mask.sum()) < int(pair["template"].sum())
            input_boxes.append(target_bbox)
            output_boxes.append(output_bbox)
            placed = True
            break
        if not placed:
            return False

    for idx, pair in enumerate(pairs):
        pair["clip_dirs"] = chosen_clip.get(idx, ())
    return True


def _place_templates(rng: random.Random, pairs: list[dict], grid_shape: tuple[int, int]) -> bool:
    input_boxes = [pair["target_bbox"] for pair in pairs]
    for pair in sorted(pairs, key=lambda item: item["template"].shape[0] * item["template"].shape[1], reverse=True):
        th, tw = pair["template"].shape
        gh, gw = grid_shape
        candidates = [(r0, c0) for r0 in range(gh - th + 1) for c0 in range(gw - tw + 1)]
        rng.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            bbox = (r0, c0, r0 + th - 1, c0 + tw - 1)
            if any(_expanded_boxes_overlap(bbox, other, MIN_BBOX_GAP) for other in input_boxes):
                continue
            pair["template_pos"] = (r0, c0)
            pair["template_bbox"] = bbox
            input_boxes.append(bbox)
            placed = True
            break
        if not placed:
            return False
    return True


def _render_input(grid_shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    for pair in pairs:
        tr, tc = pair["target_pos"]
        _paint_mask(grid, pair["target"], pair["target_color"], tr, tc, clip=False)
    for pair in pairs:
        tr, tc = pair["template_pos"]
        _paint_mask(grid, pair["template"], pair["template_color"], tr, tc, clip=False)
    return grid


def _render_expected(grid_shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    for pair in pairs:
        tr, tc = pair["target_pos"]
        _paint_mask(grid, pair["template"], pair["target_color"], tr, tc, clip=True)
    return grid


def _validate_scene(inp: np.ndarray, expected: np.ndarray, pairs: list[dict]) -> bool:
    if np.array_equal(inp, expected):
        return False

    objects = _extract_objects(inp)
    if len(objects) != 6:
        return False
    if len({obj["color"] for obj in objects}) != 6:
        return False

    for i, a in enumerate(objects):
        if a["area"] < 3 or a["area"] > 7:
            return False
        for b in objects[i + 1 :]:
            if _expanded_boxes_overlap(a["bbox"], b["bbox"], MIN_BBOX_GAP):
                return False

    sym_objects = [obj for obj in objects if _is_vertically_symmetric(obj["mask"])]
    asym_objects = [obj for obj in objects if not _is_vertically_symmetric(obj["mask"])]
    if len(sym_objects) != 3 or len(asym_objects) != 3:
        return False

    sym_counts = Counter(obj["area"] for obj in sym_objects)
    asym_counts = Counter(obj["area"] for obj in asym_objects)
    if set(sym_counts) != set(asym_counts):
        return False
    if any(sym_counts[a] != 1 or asym_counts[a] != 1 for a in sym_counts):
        return False

    if not any(pair["relation"] == "larger" for pair in pairs):
        return False
    if not any(pair["relation"] == "smaller" for pair in pairs):
        return False
    if any(pair["visible_output_area"] <= 0 for pair in pairs):
        return False

    out_objects = _extract_objects(expected)
    if len(out_objects) != 3:
        return False
    if {obj["color"] for obj in out_objects} != {pair["target_color"] for pair in pairs}:
        return False
    if any(obj["color"] in {pair["template_color"] for pair in pairs} for obj in out_objects):
        return False

    return np.array_equal(main(inp), expected)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        pairs = _choose_pairs(rng)
        grid_shape = (
            rng.randint(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1]),
            rng.randint(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1]),
        )
        if not _place_targets(rng, pairs, grid_shape):
            continue
        if not _place_templates(rng, pairs, grid_shape):
            continue
        inp = _render_input(grid_shape, pairs)
        expected = _render_expected(grid_shape, pairs)
        if _validate_scene(inp, expected, pairs):
            return inp
    raise RuntimeError("failed to generate a valid symmetry-area stamping puzzle")
