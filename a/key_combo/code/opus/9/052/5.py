import random
from collections import Counter, deque

import numpy as np


BG = 0
BODY = 1
GREY = 5
RAY_COLORS = (2, 3, 4, 7)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 500


def _in_bounds(shape, r, c):
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_coords(grid, bg=BG):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _transform_coord(coord, h, w, flip, rot):
    r, c = coord
    if flip:
        c = w - 1 - c
    if rot == 0:
        return r, c
    if rot == 1:
        return c, h - 1 - r
    if rot == 2:
        return h - 1 - r, w - 1 - c
    return w - 1 - c, r


def _transform_dir(direction, flip, rot):
    dr, dc = direction
    if flip:
        dc = -dc
    if rot == 0:
        return dr, dc
    if rot == 1:
        return dc, -dr
    if rot == 2:
        return -dr, -dc
    return -dc, dr


def _all_transforms(h, w):
    out = []
    seen = set()
    for flip in (False, True):
        for rot in range(4):
            mapping = tuple(_transform_coord((r, c), h, w, flip, rot) for r in range(h) for c in range(w))
            if mapping in seen:
                continue
            seen.add(mapping)
            out_shape = (h, w) if rot % 2 == 0 else (w, h)
            out.append(
                {
                    "flip": flip,
                    "rot": rot,
                    "shape": out_shape,
                    "map_coord": lambda coord, f=flip, k=rot, hh=h, ww=w: _transform_coord(coord, hh, ww, f, k),
                    "map_dir": lambda direction, f=flip, k=rot: _transform_dir(direction, f, k),
                }
            )
    return out


def _candidate_side_data(body_bbox):
    r0, c0, r1, c1 = body_bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if w >= h:
        return [
            {"direction": (-1, 0), "edge": [(0, c) for c in range(w)]},
            {"direction": (1, 0), "edge": [(h - 1, c) for c in range(w)]},
        ]
    return [
        {"direction": (0, -1), "edge": [(r, 0) for r in range(h)]},
        {"direction": (0, 1), "edge": [(r, w - 1) for r in range(h)]},
    ]


def _parse_template(grid, comp):
    counts = Counter(int(grid[r, c]) for r, c in comp)
    if counts[BODY] < 6:
        return None
    blue = [(r, c) for r, c in comp if int(grid[r, c]) == BODY]
    r0, c0, r1, c1 = _bbox(blue)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if min(h, w) != 2 or max(h, w) < 3:
        return None
    if len(blue) != h * w:
        return None
    body_set = set(blue)
    valid_sides = []
    for side in _candidate_side_data((r0, c0, r1, c1)):
        dr, dc = side["direction"]
        rays = []
        covered = set(body_set)
        ok = True
        for lr, lc in side["edge"]:
            gr = r0 + lr
            gc = c0 + lc
            rr = gr + dr
            cc = gc + dc
            if not _in_bounds(grid.shape, rr, cc) or int(grid[rr, cc]) == BG:
                continue
            color = int(grid[rr, cc])
            if color == BODY:
                ok = False
                break
            length = 0
            cells = []
            while _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == color:
                cells.append((rr, cc))
                covered.add((rr, cc))
                length += 1
                rr += dr
                cc += dc
            if length < 2:
                ok = False
                break
            rays.append(
                {
                    "source_local": (lr, lc),
                    "source_global": (gr, gc),
                    "color": color,
                    "direction": (dr, dc),
                    "length": length,
                    "tip": cells[-1],
                }
            )
        if ok and 2 <= len(rays) <= 3 and len({ray["color"] for ray in rays}) == len(rays) and covered == set(comp):
            valid_sides.append({"direction": side["direction"], "rays": rays, "bbox": (r0, c0, r1, c1), "shape": (h, w)})
    if len(valid_sides) != 1:
        return None
    return valid_sides[0]


def _parse_target(grid, comp):
    r0, c0, r1, c1 = _bbox(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if min(h, w) != 2 or max(h, w) < 3:
        return None
    if len(comp) != h * w:
        return None
    counts = Counter(int(grid[r, c]) for r, c in comp)
    non_blue = [(color, count) for color, count in counts.items() if color != BODY]
    if len(non_blue) != 1 or non_blue[0][1] != 1 or counts[BODY] != h * w - 1:
        return None
    color = non_blue[0][0]
    embedded = next((r, c) for r, c in comp if int(grid[r, c]) == color)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    return {
        "bbox": (r0, c0, r1, c1),
        "shape": (h, w),
        "color": color,
        "embedded_global": embedded,
        "embedded_local": (embedded[0] - r0, embedded[1] - c0),
        "patch": patch,
        "coords": comp,
    }


def _parse_scene(grid):
    comps = _component_coords(grid, BG)
    template = None
    targets = []
    for comp in comps:
        candidate = _parse_template(grid, comp)
        if candidate is not None:
            if template is not None:
                raise AssertionError("expected exactly one template sprite")
            template = candidate
            continue
        target = _parse_target(grid, comp)
        if target is None:
            raise AssertionError("unexpected foreground component")
        targets.append(target)
    if template is None:
        raise AssertionError("missing template sprite")
    return template, targets


def _place_patch_checked(scene, patch, top_left, tip=None, tip_color=None):
    out = scene.copy()
    r0, c0 = top_left
    h, w = patch.shape
    for r in range(h):
        for c in range(w):
            color = int(patch[r, c])
            gr = r0 + r
            gc = c0 + c
            if not _in_bounds(out.shape, gr, gc):
                raise AssertionError("patch out of bounds")
            if tip is not None and (gr, gc) == tip:
                if color != tip_color or int(out[gr, gc]) != tip_color:
                    raise AssertionError("only the embedded pixel may overlap its own ray tip")
            elif int(out[gr, gc]) != BG:
                raise AssertionError("translated sprite must not overlap other foreground")
            out[gr, gc] = color
    return out


def _propagate(static_scene, placements, template_rays):
    overlay = set()
    for placed in placements:
        transform = placed["transform"]
        r0, c0 = placed["top_left"]
        for ray in template_rays:
            sr, sc = transform["map_coord"](ray["source_local"])
            dr, dc = transform["map_dir"](ray["direction"])
            rr = r0 + sr + dr
            cc = c0 + sc + dc
            while _in_bounds(static_scene.shape, rr, cc) and int(static_scene[rr, cc]) == BG:
                overlay.add((rr, cc))
                rr += dr
                cc += dc
    out = static_scene.copy()
    for r, c in overlay:
        out[r, c] = GREY
    return out, overlay


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each color-paired rectangle onto its matching template ray tip, then copy the template's ray layout through each rectangle's dihedral orientation.

    Input structure:
    - The background is black.
    - One template consists of a solid blue 2-by-N or N-by-2 rectangle with 2-3 colored rays extending from one long side.
    - Each other foreground component is a dense rectangle of the same size, all blue except for one embedded pixel whose color matches exactly one template ray.

    Rule:
    - Identify the template as the only component whose non-blue pixels form outward same-color rays instead of a single embedded pixel.
    - For each template ray, record its source body cell, color, direction, and tip; pair each non-template rectangle to the ray with the same color.
    - Translate every paired rectangle so its embedded pixel lands on that ray tip, then for every template ray draw a gray ray from the corresponding transformed source cell on every moved rectangle until blocked by non-black or by the grid edge.

    Key insight:
    - Each colored template ray has two roles at once: its color chooses which rectangle moves to that tip, and its source cell on the template tells every moved rectangle where a transformed gray propagation ray must start.
    """
    g = np.asarray(grid, dtype=int)
    template, targets = _parse_scene(g)
    template_rays = sorted(template["rays"], key=lambda ray: ray["color"])
    ray_by_color = {ray["color"]: ray for ray in template_rays}

    if len(targets) != len(template_rays):
        raise AssertionError("ray count must equal non-template sprite count")
    if {target["color"] for target in targets} != set(ray_by_color):
        raise AssertionError("each target must match exactly one ray color")

    body_h, body_w = template["shape"]
    transforms = _all_transforms(body_h, body_w)

    out = g.copy()
    for target in targets:
        for r, c in target["coords"]:
            out[r, c] = BG

    placements = []
    static = out.copy()
    for target in sorted(targets, key=lambda item: item["color"]):
        ray = ray_by_color[target["color"]]
        candidates = [
            transform
            for transform in transforms
            if transform["shape"] == target["shape"]
            and transform["map_coord"](ray["source_local"]) == target["embedded_local"]
        ]
        if len(candidates) != 1:
            raise AssertionError("target orientation must be uniquely determined")
        transform = candidates[0]
        tip_r, tip_c = ray["tip"]
        er, ec = target["embedded_local"]
        top_left = (tip_r - er, tip_c - ec)
        static = _place_patch_checked(static, target["patch"], top_left, tip=ray["tip"], tip_color=target["color"])
        placements.append({"top_left": top_left, "transform": transform})

    final, _overlay = _propagate(static, placements, template_rays)
    return final


def _build_template_scene(shape, top_left, ray_direction, ray_specs, canvas_shape):
    h, w = shape
    grid = np.zeros(canvas_shape, dtype=int)
    r0, c0 = top_left
    grid[r0 : r0 + h, c0 : c0 + w] = BODY
    for spec in ray_specs:
        sr, sc = spec["source_local"]
        dr, dc = ray_direction
        gr = r0 + sr
        gc = c0 + sc
        for step in range(1, spec["length"] + 1):
            rr = gr + dr * step
            cc = gc + dc * step
            if not _in_bounds(canvas_shape, rr, cc):
                raise AssertionError("ray left the canvas")
            if int(grid[rr, cc]) != BG:
                raise AssertionError("template rays must not overlap")
            grid[rr, cc] = spec["color"]
        spec["source_global"] = (gr, gc)
        spec["tip"] = (gr + dr * spec["length"], gc + dc * spec["length"])
        spec["direction"] = ray_direction
    return grid


def _target_patch(shape, embedded_local, color):
    patch = np.full(shape, BODY, dtype=int)
    patch[embedded_local] = color
    return patch


def _top_left_candidates(scene_shape, patch_shape):
    h, w = scene_shape
    ph, pw = patch_shape
    return [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]


def _footprint(top_left, patch_shape):
    r0, c0 = top_left
    h, w = patch_shape
    return {(r0 + r, c0 + c) for r in range(h) for c in range(w)}


def _padded(cells, shape):
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _choose_template_top_left(canvas_shape, shape, direction):
    h, w = shape
    max_len = 5
    margin = 1
    rows = range(margin, canvas_shape[0] - h - margin + 1)
    cols = range(margin, canvas_shape[1] - w - margin + 1)
    if direction == (-1, 0):
        rows = range(max_len + margin, canvas_shape[0] - h - margin + 1)
    elif direction == (1, 0):
        rows = range(margin, canvas_shape[0] - h - max_len - margin + 1)
    elif direction == (0, -1):
        cols = range(max_len + margin, canvas_shape[1] - w - margin + 1)
    else:
        cols = range(margin, canvas_shape[1] - w - max_len - margin + 1)
    rows = list(rows)
    cols = list(cols)
    if not rows or not cols:
        return None
    return random.choice(rows), random.choice(cols)


def _ray_position_choices(ray_count):
    if ray_count == 2:
        return [(0, 1), (0, 2), (1, 3), (0, 3)]
    return [(0, 1, 3), (0, 2, 3), (0, 1, 2), (1, 2, 3)]


def _valid_final_options(template_scene, shape, ray_specs, matched_index):
    body_h, body_w = shape
    transforms = _all_transforms(body_h, body_w)
    ray = ray_specs[matched_index]
    options = []
    for transform in transforms:
        embedded_local = transform["map_coord"](ray["source_local"])
        patch_shape = transform["shape"]
        patch = _target_patch(patch_shape, embedded_local, ray["color"])
        tip_r, tip_c = ray["tip"]
        top_left = (tip_r - embedded_local[0], tip_c - embedded_local[1])
        ok = True
        for r in range(patch_shape[0]):
            for c in range(patch_shape[1]):
                gr = top_left[0] + r
                gc = top_left[1] + c
                if not _in_bounds(template_scene.shape, gr, gc):
                    ok = False
                    break
                if (gr, gc) == ray["tip"]:
                    if int(template_scene[gr, gc]) != ray["color"]:
                        ok = False
                        break
                elif int(template_scene[gr, gc]) != BG:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            options.append(
                {
                    "transform": transform,
                    "embedded_local": embedded_local,
                    "patch_shape": patch_shape,
                    "patch": patch,
                    "top_left": top_left,
                    "color": ray["color"],
                    "matched_index": matched_index,
                }
            )
    random.shuffle(options)
    return options


def _select_final_layout(template_scene, shape, ray_specs):
    choices = [_valid_final_options(template_scene, shape, ray_specs, i) for i in range(len(ray_specs))]
    if any(not options for options in choices):
        return None

    order = sorted(range(len(ray_specs)), key=lambda i: len(choices[i]))
    picked = [None] * len(ray_specs)
    used = set()

    def rec(k):
        if k == len(order):
            return True
        idx = order[k]
        for option in choices[idx]:
            cells = _footprint(option["top_left"], option["patch_shape"])
            if cells & used:
                continue
            picked[idx] = option
            used.update(cells)
            if rec(k + 1):
                return True
            used.difference_update(cells)
            picked[idx] = None
        return False

    if not rec(0):
        return None
    return picked


def _build_output_from_config(template_scene, final_targets, ray_specs):
    static = template_scene.copy()
    placements = []
    for target in final_targets:
        static = _place_patch_checked(static, target["patch"], target["top_left"], tip=ray_specs[target["matched_index"]]["tip"], tip_color=target["color"])
        placements.append({"top_left": target["top_left"], "transform": target["transform"]})
    final, overlay = _propagate(static, placements, ray_specs)
    return static, final, overlay


def _semantic_checks(static_scene, final_targets, ray_specs, overlay):
    if not overlay:
        return False

    painted_by_target = []
    painted_by_ray = [0] * len(ray_specs)
    used_cells = set()

    for target in final_targets:
        transform = target["transform"]
        r0, c0 = target["top_left"]
        lengths = []
        for i, ray in enumerate(ray_specs):
            sr, sc = transform["map_coord"](ray["source_local"])
            dr, dc = transform["map_dir"](ray["direction"])
            rr = r0 + sr + dr
            cc = c0 + sc + dc
            cells = []
            while _in_bounds(static_scene.shape, rr, cc) and int(static_scene[rr, cc]) == BG:
                cells.append((rr, cc))
                rr += dr
                cc += dc
            lengths.append(len(cells))
            if cells:
                painted_by_ray[i] += 1
            for cell in cells:
                if cell in used_cells:
                    return False
                used_cells.add(cell)
        painted_by_target.append(lengths)

    if used_cells != overlay:
        return False
    if any(count == 0 for count in painted_by_ray):
        return False
    for matched_index, lengths in enumerate(painted_by_target):
        if sum(lengths) == 0:
            return False
        if sum(lengths[i] > 0 for i in range(len(lengths)) if i != matched_index) == 0:
            return False
    return True


def _choose_input_positions(template_scene, final_targets):
    occupied = {(int(r), int(c)) for r, c in np.argwhere(template_scene != BG)}
    forbidden = _padded(occupied, template_scene.shape)
    chosen = [None] * len(final_targets)
    order = sorted(range(len(final_targets)), key=lambda i: final_targets[i]["patch_shape"][0] * final_targets[i]["patch_shape"][1], reverse=True)

    def rec(k, blocked):
        if k == len(order):
            return True
        idx = order[k]
        target = final_targets[idx]
        candidates = _top_left_candidates(template_scene.shape, target["patch_shape"])
        random.shuffle(candidates)
        for top_left in candidates:
            if top_left == target["top_left"]:
                continue
            cells = _footprint(top_left, target["patch_shape"])
            if cells & blocked:
                continue
            chosen[idx] = top_left
            next_blocked = blocked | _padded(cells, template_scene.shape)
            if rec(k + 1, next_blocked):
                return True
            chosen[idx] = None
        return False

    if not rec(0, forbidden):
        return None
    return chosen


def _gen_attempt():
    canvas_shape = (random.randint(18, 24), random.randint(18, 24))
    shape = random.choice([(2, 4), (4, 2)])
    ray_direction = random.choice([(-1, 0), (1, 0)]) if shape[1] > shape[0] else random.choice([(0, -1), (0, 1)])
    template_tl = _choose_template_top_left(canvas_shape, shape, ray_direction)
    if template_tl is None:
        return None

    ray_count = random.choice([2, 3])
    ray_positions = list(random.choice(_ray_position_choices(ray_count)))
    colors = random.sample(list(RAY_COLORS), ray_count)

    h, w = shape
    if ray_direction == (-1, 0):
        source_cells = [(0, c) for c in ray_positions]
    elif ray_direction == (1, 0):
        source_cells = [(h - 1, c) for c in ray_positions]
    elif ray_direction == (0, -1):
        source_cells = [(r, 0) for r in ray_positions]
    else:
        source_cells = [(r, w - 1) for r in ray_positions]

    ray_specs = [
        {"source_local": source_local, "color": color, "length": random.randint(2, 5)}
        for source_local, color in zip(source_cells, colors)
    ]
    random.shuffle(ray_specs)
    template_scene = _build_template_scene(shape, template_tl, ray_direction, ray_specs, canvas_shape)

    final_targets = _select_final_layout(template_scene, shape, ray_specs)
    if final_targets is None:
        return None

    static_scene, expected_output, overlay = _build_output_from_config(template_scene, final_targets, ray_specs)
    if not _semantic_checks(static_scene, final_targets, ray_specs, overlay):
        return None

    input_positions = _choose_input_positions(template_scene, final_targets)
    if input_positions is None:
        return None

    input_grid = template_scene.copy()
    for target, input_tl in zip(final_targets, input_positions):
        input_grid = _place_patch_checked(input_grid, target["patch"], input_tl)
        target["input_top_left"] = input_tl

    solved = main(input_grid)
    if not np.array_equal(solved, expected_output):
        return None

    if np.array_equal(input_grid, expected_output):
        return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
