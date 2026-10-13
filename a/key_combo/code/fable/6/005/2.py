from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
DIR_NAME = {
    (-1, 0): "up",
    (1, 0): "down",
    (0, -1): "left",
    (0, 1): "right",
}
OPEN_TO_RAY = {
    "up": (1, 0),
    "down": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
CANONICAL_BODY = [
    [(1, 0), (1, -1), (1, 1), (2, 0)],
    [(1, 0), (2, 0), (2, -1), (2, 1)],
    [(1, -1), (1, 0), (2, 0), (2, 1), (3, 0)],
]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIRS4 if int(connectivity) == 4 else DIRS8
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if in_bounds(mask.shape, nr, nc) and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def rotate_offsets(offsets: list[tuple[int, int]], quarter_turns: int) -> list[tuple[int, int]]:
    turns = int(quarter_turns) % 4
    out = [(int(r), int(c)) for r, c in offsets]
    for _ in range(turns):
        out = [(-c, r) for r, c in out]
    return out


def sprite_variants(ray_dir: tuple[int, int]) -> list[list[tuple[int, int]]]:
    turns = {
        (-1, 0): 0,   # up
        (0, 1): 1,    # right
        (1, 0): 2,    # down
        (0, -1): 3,   # left
    }[tuple(ray_dir)]
    return [rotate_offsets(base, turns) for base in CANONICAL_BODY]


def decode_direction(body_coords: list[tuple[int, int]], marker: tuple[int, int]) -> tuple[int, int]:
    mr, mc = marker
    mean_dr = sum(r - mr for r, _ in body_coords) / len(body_coords)
    mean_dc = sum(c - mc for _, c in body_coords) / len(body_coords)
    if abs(mean_dr) == abs(mean_dc):
        raise AssertionError("sprite centroid bias is ambiguous")
    if abs(mean_dr) > abs(mean_dc):
        return (-1, 0) if mean_dr > 0 else (1, 0)
    return (0, -1) if mean_dc > 0 else (0, 1)


def make_container_patch(height: int, width: int, opening: str, wall: int, bg: int) -> np.ndarray:
    patch = np.full((height, width), bg, dtype=int)
    if opening == "up":
        patch[:, 0] = wall
        patch[:, -1] = wall
        patch[-1, :] = wall
    elif opening == "down":
        patch[:, 0] = wall
        patch[:, -1] = wall
        patch[0, :] = wall
    elif opening == "left":
        patch[0, :] = wall
        patch[-1, :] = wall
        patch[:, -1] = wall
    elif opening == "right":
        patch[0, :] = wall
        patch[-1, :] = wall
        patch[:, 0] = wall
    else:
        raise ValueError("bad opening")
    return patch


def container_from_bbox(r0: int, c0: int, height: int, width: int, opening: str) -> dict:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    if opening == "up":
        cavity = (r0, c0 + 1, r1 - 1, c1 - 1)
    elif opening == "down":
        cavity = (r0 + 1, c0 + 1, r1, c1 - 1)
    elif opening == "left":
        cavity = (r0 + 1, c0, r1 - 1, c1 - 1)
    elif opening == "right":
        cavity = (r0 + 1, c0 + 1, r1 - 1, c1)
    else:
        raise ValueError("bad opening")
    return {
        "bbox": (r0, c0, r1, c1),
        "opening": opening,
        "cavity": cavity,
        "ray_dir": OPEN_TO_RAY[opening],
    }


def classify_container(grid: np.ndarray, coords: list[tuple[int, int]], wall: int, bg: int) -> dict | None:
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = grid[r0:r1 + 1, c0:c1 + 1]
    h, w = patch.shape
    wall_mask = patch == int(wall)
    if not np.all((patch == int(wall)) | (patch == int(bg))):
        return None

    def matches(opening: str) -> bool:
        expected = np.zeros((h, w), dtype=bool)
        if opening == "up" and h >= 2 and w >= 3:
            expected[:, 0] = True
            expected[:, -1] = True
            expected[-1, :] = True
        elif opening == "down" and h >= 2 and w >= 3:
            expected[:, 0] = True
            expected[:, -1] = True
            expected[0, :] = True
        elif opening == "left" and h >= 3 and w >= 2:
            expected[0, :] = True
            expected[-1, :] = True
            expected[:, -1] = True
        elif opening == "right" and h >= 3 and w >= 2:
            expected[0, :] = True
            expected[-1, :] = True
            expected[:, 0] = True
        else:
            return False
        return bool(np.array_equal(wall_mask, expected))

    openings = [opening for opening in ("up", "down", "left", "right") if matches(opening)]
    if len(openings) != 1:
        return None
    return container_from_bbox(r0, c0, h, w, openings[0])


def hit_container_back_wall(container: dict, hit: tuple[int, int], ray_dir: tuple[int, int]) -> bool:
    r, c = hit
    r0, c0, r1, c1 = container["bbox"]
    opening = container["opening"]
    if tuple(ray_dir) != OPEN_TO_RAY[opening]:
        return False
    if opening == "up":
        return r == r1 and c0 < c < c1
    if opening == "down":
        return r == r0 and c0 < c < c1
    if opening == "left":
        return c == c1 and r0 < r < r1
    return c == c0 and r0 < r < r1


def trace_ray(grid: np.ndarray, start: tuple[int, int], ray_dir: tuple[int, int], bg: int) -> dict:
    dr, dc = ray_dir
    r, c = start[0] + dr, start[1] + dc
    path: list[tuple[int, int]] = []
    while in_bounds(grid.shape, r, c):
        if int(grid[r, c]) != int(bg):
            return {"hit": (r, c), "color": int(grid[r, c]), "path": path}
        path.append((r, c))
        r += dr
        c += dc
    return {"hit": None, "color": None, "path": path}


def parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    counts = Counter(map(int, g.ravel()))
    marker_colors = [color for color in fg_colors if counts[int(color)] == 1]
    repeated = [color for color in fg_colors if counts[int(color)] > 1]
    assert len(marker_colors) >= 1, "expected marker colors"
    assert len(repeated) == 2, "expected body color and wall color"

    assignments: list[dict] = []
    for body_color in repeated:
        wall_color = repeated[1] if repeated[0] == body_color else repeated[0]

        containers = []
        ok = True
        wall_comps = components_mask(g == int(wall_color), connectivity=4)
        if not wall_comps:
            continue
        for comp in wall_comps:
            container = classify_container(g, comp, wall_color, bg)
            if container is None:
                ok = False
                break
            containers.append(container)
        if not ok or not containers:
            continue

        sprite_mask = (g != int(bg)) & (g != int(wall_color))
        sprite_comps = components_mask(sprite_mask, connectivity=8)
        if len(sprite_comps) != len(marker_colors):
            continue

        sprites = []
        used_markers: set[int] = set()
        for comp in sprite_comps:
            comp_colors = {int(g[r, c]) for r, c in comp}
            marker_here = [color for color in marker_colors if color in comp_colors]
            if len(marker_here) != 1:
                ok = False
                break
            marker_color = int(marker_here[0])
            marker_cells = [(r, c) for r, c in comp if int(g[r, c]) == marker_color]
            body_cells = [(r, c) for r, c in comp if int(g[r, c]) == int(body_color)]
            if len(marker_cells) != 1 or not body_cells:
                ok = False
                break
            if any(int(g[r, c]) not in {int(body_color), marker_color} for r, c in comp):
                ok = False
                break
            marker = marker_cells[0]
            ray_dir = decode_direction(body_cells, marker)
            sprites.append(
                {
                    "marker_color": marker_color,
                    "marker": marker,
                    "body": body_cells,
                    "coords": list(comp),
                    "bbox": bbox_coords(comp),
                    "ray_dir": ray_dir,
                }
            )
            used_markers.add(marker_color)
        if ok and used_markers == set(map(int, marker_colors)):
            assignments.append(
                {
                    "bg": int(bg),
                    "body_color": int(body_color),
                    "wall_color": int(wall_color),
                    "marker_colors": sorted(map(int, marker_colors)),
                    "containers": sorted(containers, key=lambda item: item["bbox"]),
                    "sprites": sorted(sprites, key=lambda item: item["marker_color"]),
                }
            )

    assert len(assignments) == 1, "scene roles must be uniquely inferable"
    return assignments[0]


def analyze_rays(grid: np.ndarray, scene: dict) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(scene["bg"])
    wall_color = int(scene["wall_color"])
    containers = list(scene["containers"])
    results = []
    for sprite in scene["sprites"]:
        trace = trace_ray(g, sprite["marker"], sprite["ray_dir"], bg)
        hit = trace["hit"]
        outcome = {
            "marker_color": int(sprite["marker_color"]),
            "trace": trace,
            "result": "erase",
            "container_bbox": None,
        }
        if hit is not None and trace["color"] == wall_color:
            hits = [c for c in containers if hit_container_back_wall(c, hit, sprite["ray_dir"])]
            if len(hits) == 1:
                outcome["result"] = "fill"
                outcome["container_bbox"] = hits[0]["bbox"]
        results.append(outcome)
    return sorted(results, key=lambda item: item["marker_color"])


def render_output(grid: np.ndarray, scene: dict, outcomes: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bg = int(scene["bg"])
    sprite_by_color = {sprite["marker_color"]: sprite for sprite in scene["sprites"]}
    container_by_bbox = {container["bbox"]: container for container in scene["containers"]}
    filled_boxes: set[tuple[int, int, int, int]] = set()

    for outcome in outcomes:
        if outcome["result"] == "erase":
            for r, c in sprite_by_color[int(outcome["marker_color"])]["coords"]:
                out[r, c] = bg
        else:
            bbox = outcome["container_bbox"]
            assert bbox is not None and bbox not in filled_boxes
            filled_boxes.add(bbox)
            cr0, cc0, cr1, cc1 = container_by_bbox[bbox]["cavity"]
            out[cr0:cr1 + 1, cc0:cc1 + 1] = int(outcome["marker_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each sprite's hidden ray direction from its centroid bias and either fill the faced-open container or erase the sprite.

    Input structure:
    - One dominant background surrounds several multicolor 8-connected sprites and several monochrome U-shaped containers.
    - Every sprite uses one shared body color plus exactly one singleton marker color; marker colors are all distinct.
    - Every container uses one shared wall color and is a one-cell-thick rectangular cavity with exactly one open side.

    Rule:
    - For each sprite, compute the centroid of its body pixels relative to its marker; the ray points in the opposite cardinal direction.
    - Trace an invisible ray from the marker until it leaves the grid or hits the first non-background pixel.
    - If that first hit is the back wall reached through a container's open side, fill the whole cavity with the marker color and keep the sprite.
    - Otherwise erase that sprite and its marker to background; containers never move and failed rays do not draw anything.

    Key insight:
    - The direction is not drawn explicitly: it is encoded by the body's centroid bias around the marker, and the ray only succeeds when that decoded direction meets a container from the mouth side instead of striking a closed wall.
    """
    scene = parse_scene(grid)
    outcomes = analyze_rays(grid, scene)
    return render_output(grid, scene, outcomes)


def rect_is_free(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 0) -> bool:
    hh, ww = reserved.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def reserve_rect(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 0) -> None:
    hh, ww = reserved.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    reserved[rr0:rr1, cc0:cc1] = True


def component_cells(marker: tuple[int, int], body_offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    mr, mc = marker
    return [(mr, mc)] + [(mr + dr, mc + dc) for dr, dc in body_offsets]


def component_bbox(marker: tuple[int, int], body_offsets: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    return bbox_coords(component_cells(marker, body_offsets))


def place_component(grid: np.ndarray, marker: tuple[int, int], body_offsets: list[tuple[int, int]], marker_color: int, body_color: int) -> list[tuple[int, int]]:
    coords = component_cells(marker, body_offsets)
    mr, mc = marker
    grid[mr, mc] = marker_color
    for r, c in coords:
        if (r, c) != (mr, mc):
            grid[r, c] = body_color
    return coords


def candidate_successes(grid: np.ndarray, reserved: np.ndarray, line_reserved: np.ndarray, container: dict, ray_dir: tuple[int, int], marker_color: int, body_color: int) -> list[dict]:
    r0, c0, r1, c1 = container["bbox"]
    bg = get_bg(grid)
    h, w = grid.shape
    out: list[dict] = []
    variants = sprite_variants(ray_dir)
    if ray_dir == (1, 0):
        lanes = [(r0 - gap, c) for gap in range(1, 8) for c in range(c0 + 1, c1)]
    elif ray_dir == (-1, 0):
        lanes = [(r1 + gap, c) for gap in range(1, 8) for c in range(c0 + 1, c1)]
    elif ray_dir == (0, 1):
        lanes = [(r, c0 - gap) for gap in range(1, 8) for r in range(r0 + 1, r1)]
    else:
        lanes = [(r, c1 + gap) for gap in range(1, 8) for r in range(r0 + 1, r1)]
    random.shuffle(lanes)
    random.shuffle(variants)
    for marker in lanes:
        mr, mc = marker
        if not in_bounds((h, w), mr, mc):
            continue
        for body_offsets in variants:
            coords = component_cells(marker, body_offsets)
            if not all(in_bounds((h, w), r, c) for r, c in coords):
                continue
            if any(bool(line_reserved[r, c]) or int(grid[r, c]) != int(bg) for r, c in coords):
                continue
            br0, bc0, br1, bc1 = component_bbox(marker, body_offsets)
            if not rect_is_free(reserved, br0, bc0, br1 - br0 + 1, bc1 - bc0 + 1, pad=1):
                continue
            trace = trace_ray(grid, marker, ray_dir, bg)
            if trace["hit"] is None:
                continue
            if not hit_container_back_wall(container, trace["hit"], ray_dir):
                continue
            out.append(
                {
                    "kind": "success",
                    "marker": marker,
                    "body_offsets": body_offsets,
                    "marker_color": marker_color,
                    "body_color": body_color,
                    "ray_dir": ray_dir,
                    "target_bbox": container["bbox"],
                    "path": trace["path"],
                }
            )
    return out


def candidate_wall_misses(grid: np.ndarray, reserved: np.ndarray, line_reserved: np.ndarray, container: dict, marker_color: int, body_color: int) -> list[dict]:
    r0, c0, r1, c1 = container["bbox"]
    bg = get_bg(grid)
    h, w = grid.shape
    if container["opening"] in {"up", "down"}:
        options = [((0, 1), [(r, c0 - gap) for gap in range(1, 8) for r in range(r0, r1 + 1)]),
                   ((0, -1), [(r, c1 + gap) for gap in range(1, 8) for r in range(r0, r1 + 1)])]
    else:
        options = [((1, 0), [(r0 - gap, c) for gap in range(1, 8) for c in range(c0, c1 + 1)]),
                   ((-1, 0), [(r1 + gap, c) for gap in range(1, 8) for c in range(c0, c1 + 1)])]

    out: list[dict] = []
    random.shuffle(options)
    for ray_dir, lanes in options:
        variants = sprite_variants(ray_dir)
        random.shuffle(lanes)
        random.shuffle(variants)
        for marker in lanes:
            mr, mc = marker
            if not in_bounds((h, w), mr, mc):
                continue
            for body_offsets in variants:
                coords = component_cells(marker, body_offsets)
                if not all(in_bounds((h, w), r, c) for r, c in coords):
                    continue
                if any(bool(line_reserved[r, c]) or int(grid[r, c]) != int(bg) for r, c in coords):
                    continue
                br0, bc0, br1, bc1 = component_bbox(marker, body_offsets)
                if not rect_is_free(reserved, br0, bc0, br1 - br0 + 1, bc1 - bc0 + 1, pad=1):
                    continue
                trace = trace_ray(grid, marker, ray_dir, bg)
                if trace["hit"] is None or int(trace["color"]) != int(container["wall_color"]):
                    continue
                hr, hc = trace["hit"]
                if not (r0 <= hr <= r1 and c0 <= hc <= c1):
                    continue
                if hit_container_back_wall(container, trace["hit"], ray_dir):
                    continue
                out.append(
                    {
                        "kind": "wall_miss",
                        "marker": marker,
                        "body_offsets": body_offsets,
                        "marker_color": marker_color,
                        "body_color": body_color,
                        "ray_dir": ray_dir,
                        "target_bbox": container["bbox"],
                        "path": trace["path"],
                    }
                )
    return out


def candidate_oob_misses(grid: np.ndarray, reserved: np.ndarray, line_reserved: np.ndarray, marker_color: int, body_color: int) -> list[dict]:
    bg = get_bg(grid)
    h, w = grid.shape
    dirs = list(DIR_NAME)
    random.shuffle(dirs)
    out: list[dict] = []
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    for ray_dir in dirs:
        variants = sprite_variants(ray_dir)
        random.shuffle(variants)
        for marker in cells:
            mr, mc = marker
            if int(grid[mr, mc]) != int(bg):
                continue
            for body_offsets in variants:
                coords = component_cells(marker, body_offsets)
                if not all(in_bounds((h, w), r, c) for r, c in coords):
                    continue
                if any(bool(line_reserved[r, c]) or int(grid[r, c]) != int(bg) for r, c in coords):
                    continue
                br0, bc0, br1, bc1 = component_bbox(marker, body_offsets)
                if not rect_is_free(reserved, br0, bc0, br1 - br0 + 1, bc1 - bc0 + 1, pad=1):
                    continue
                trace = trace_ray(grid, marker, ray_dir, bg)
                if trace["hit"] is not None:
                    continue
                out.append(
                    {
                        "kind": "oob_miss",
                        "marker": marker,
                        "body_offsets": body_offsets,
                        "marker_color": marker_color,
                        "body_color": body_color,
                        "ray_dir": ray_dir,
                        "target_bbox": None,
                        "path": trace["path"],
                    }
                )
    return out


def build_expected_from_specs(grid: np.ndarray, containers: list[dict], sprite_specs: list[dict], bg: int) -> np.ndarray:
    out = grid.copy()
    by_bbox = {container["bbox"]: container for container in containers}
    for spec in sprite_specs:
        if spec["kind"] == "success":
            cr0, cc0, cr1, cc1 = by_bbox[spec["target_bbox"]]["cavity"]
            out[cr0:cr1 + 1, cc0:cc1 + 1] = int(spec["marker_color"])
        else:
            for r, c in component_cells(spec["marker"], spec["body_offsets"]):
                out[r, c] = bg
    return out


def validate_generated(grid: np.ndarray, expected: np.ndarray, sprite_specs: list[dict], containers: list[dict], bg: int) -> bool:
    try:
        scene = parse_scene(grid)
    except Exception:
        return False

    if int(scene["bg"]) != int(bg):
        return False
    outcomes = analyze_rays(grid, scene)
    if len(outcomes) != len(sprite_specs):
        return False

    specs_by_color = {int(spec["marker_color"]): spec for spec in sprite_specs}
    dirs_seen = set()
    success_count = 0
    miss_count = 0
    wall_miss_found = False
    for sprite in scene["sprites"]:
        spec = specs_by_color.get(int(sprite["marker_color"]))
        if spec is None:
            return False
        if tuple(sprite["ray_dir"]) != tuple(spec["ray_dir"]):
            return False
        dirs_seen.add(tuple(sprite["ray_dir"]))

    used_fill_boxes: set[tuple[int, int, int, int]] = set()
    for outcome in outcomes:
        spec = specs_by_color[int(outcome["marker_color"])]
        if spec["kind"] == "success":
            success_count += 1
            if outcome["result"] != "fill" or outcome["container_bbox"] != spec["target_bbox"]:
                return False
            if outcome["container_bbox"] in used_fill_boxes:
                return False
            used_fill_boxes.add(outcome["container_bbox"])
        else:
            miss_count += 1
            if outcome["result"] != "erase":
                return False
            if spec["kind"] == "wall_miss":
                hit = outcome["trace"]["hit"]
                if hit is None:
                    return False
                hit_color = int(grid[hit])
                if hit_color != int(scene["wall_color"]):
                    return False
                wall_miss_found = True
            if spec["kind"] == "oob_miss" and outcome["trace"]["hit"] is not None:
                return False

    if success_count < 1 or miss_count < 1 or not wall_miss_found:
        return False
    if len(dirs_seen) < 2:
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        h = random.randint(18, 28)
        w = random.randint(18, 28)
        palette = list(range(10))
        bg, body_color, wall_color = random.sample(palette, 3)
        marker_pool = [color for color in palette if color not in {bg, body_color, wall_color}]

        grid = np.full((h, w), bg, dtype=int)
        reserved = np.zeros((h, w), dtype=bool)
        line_reserved = np.zeros((h, w), dtype=bool)

        container_count = random.randint(2, 4)
        openings = random.sample(["up", "down", "left", "right"], k=min(container_count, 4))
        while len(openings) < container_count:
            openings.append(random.choice(["up", "down", "left", "right"]))
        random.shuffle(openings)

        containers: list[dict] = []
        ok = True
        for opening in openings:
            placed = False
            for _inner in range(80):
                if opening in {"up", "down"}:
                    height = random.randint(3, 6)
                    width = random.randint(4, 8)
                else:
                    height = random.randint(4, 8)
                    width = random.randint(3, 6)
                r0 = random.randint(1, h - height - 1)
                c0 = random.randint(1, w - width - 1)
                if not rect_is_free(reserved, r0, c0, height, width, pad=1):
                    continue
                patch = make_container_patch(height, width, opening, wall_color, bg)
                if np.count_nonzero(patch == wall_color) == 0:
                    continue
                grid[r0:r0 + height, c0:c0 + width] = patch
                reserve_rect(reserved, r0, c0, height, width, pad=1)
                container = container_from_bbox(r0, c0, height, width, opening)
                container["wall_color"] = wall_color
                containers.append(container)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        containers.sort(key=lambda item: item["bbox"])

        random.shuffle(marker_pool)
        sprite_specs: list[dict] = []

        success_container = random.choice(containers)
        success_marker = marker_pool.pop()
        success_candidates = candidate_successes(grid, reserved, line_reserved, success_container, success_container["ray_dir"], success_marker, body_color)
        if not success_candidates:
            continue
        success_spec = random.choice(success_candidates)
        place_component(grid, success_spec["marker"], success_spec["body_offsets"], success_spec["marker_color"], body_color)
        br0, bc0, br1, bc1 = component_bbox(success_spec["marker"], success_spec["body_offsets"])
        reserve_rect(reserved, br0, bc0, br1 - br0 + 1, bc1 - bc0 + 1, pad=1)
        for r, c in success_spec["path"]:
            line_reserved[r, c] = True
        sprite_specs.append(success_spec)

        miss_containers = [container for container in containers if container["bbox"] != success_container["bbox"]]
        wall_miss_container = random.choice(miss_containers or containers)
        wall_miss_marker = marker_pool.pop()
        wall_miss_candidates = candidate_wall_misses(grid, reserved, line_reserved, wall_miss_container, wall_miss_marker, body_color)
        if not wall_miss_candidates:
            continue
        wall_miss_spec = random.choice(wall_miss_candidates)
        place_component(grid, wall_miss_spec["marker"], wall_miss_spec["body_offsets"], wall_miss_spec["marker_color"], body_color)
        br0, bc0, br1, bc1 = component_bbox(wall_miss_spec["marker"], wall_miss_spec["body_offsets"])
        reserve_rect(reserved, br0, bc0, br1 - br0 + 1, bc1 - bc0 + 1, pad=1)
        for r, c in wall_miss_spec["path"]:
            line_reserved[r, c] = True
        sprite_specs.append(wall_miss_spec)

        if marker_pool and random.random() < 0.7:
            oob_marker = marker_pool.pop()
            oob_candidates = candidate_oob_misses(grid, reserved, line_reserved, oob_marker, body_color)
            if oob_candidates:
                oob_spec = random.choice(oob_candidates)
                place_component(grid, oob_spec["marker"], oob_spec["body_offsets"], oob_spec["marker_color"], body_color)
                br0, bc0, br1, bc1 = component_bbox(oob_spec["marker"], oob_spec["body_offsets"])
                reserve_rect(reserved, br0, bc0, br1 - br0 + 1, bc1 - bc0 + 1, pad=1)
                for r, c in oob_spec["path"]:
                    line_reserved[r, c] = True
                sprite_specs.append(oob_spec)

        if random.random() < 0.5 and marker_pool:
            extra_success_choices = [container for container in containers if container["bbox"] != success_container["bbox"]]
            if extra_success_choices:
                extra_container = random.choice(extra_success_choices)
                extra_marker = marker_pool.pop()
                extra_candidates = candidate_successes(grid, reserved, line_reserved, extra_container, extra_container["ray_dir"], extra_marker, body_color)
                if extra_candidates:
                    extra_spec = random.choice(extra_candidates)
                    place_component(grid, extra_spec["marker"], extra_spec["body_offsets"], extra_spec["marker_color"], body_color)
                    br0, bc0, br1, bc1 = component_bbox(extra_spec["marker"], extra_spec["body_offsets"])
                    reserve_rect(reserved, br0, bc0, br1 - br0 + 1, bc1 - bc0 + 1, pad=1)
                    for r, c in extra_spec["path"]:
                        line_reserved[r, c] = True
                    sprite_specs.append(extra_spec)

        expected = build_expected_from_specs(grid, containers, sprite_specs, bg)
        if validate_generated(grid, expected, sprite_specs, containers, bg):
            return grid

    raise RuntimeError("generate_input: failed to build a valid scene")
