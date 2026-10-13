from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
MAROON = 9
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EDGE_NAMES = ("top", "bottom", "left", "right")
PAYLOAD_COLORS = (1, 2, 3, 6, 7, 8)
GEN_ATTEMPTS = 400


SEED_SHAPES = [
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).flat)).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def ring_cells(h: int, w: int, r: int, c: int, dist: int) -> list[tuple[int, int]]:
    cells = []
    for rr in range(max(0, r - dist), min(h - 1, r + dist) + 1):
        for cc in range(max(0, c - dist), min(w - 1, c + dist) + 1):
            if chebyshev((rr, cc), (r, c)) == dist:
                cells.append((rr, cc))
    return cells


def edge_touches(coords: np.ndarray, shape: tuple[int, int]) -> set[str]:
    h, w = shape
    out = set()
    if np.any(coords[:, 0] == 0):
        out.add("top")
    if np.any(coords[:, 0] == h - 1):
        out.add("bottom")
    if np.any(coords[:, 1] == 0):
        out.add("left")
    if np.any(coords[:, 1] == w - 1):
        out.add("right")
    return out


def find_mono_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if visited[r, c] or color not in colors:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                    "edges": edge_touches(coords, g.shape),
                }
            )
    return comps


def rotate_seed_patch(patch: np.ndarray, rot_k: int) -> np.ndarray:
    return np.rot90(np.asarray(patch, dtype=int), -int(rot_k) % 4)


def rotations_all_distinct(mask: np.ndarray) -> bool:
    variants = []
    visible = np.asarray(mask, dtype=bool).copy()
    visible[1, 1] = False
    for k in range(4):
        rot = np.rot90(visible, k)
        if any(np.array_equal(rot, prev) for prev in variants):
            return False
        variants.append(rot)
    return True


def random_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def unique_shape_masks(area: int, max_side: int = 4) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out = []
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    frontier.add((r + dr, c + dc))
            frontier -= cells
            nr, nc = random.choice(tuple(frontier))
            cells.add((nr, nc))
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, c0 = min(rows), min(cols)
        norm = {(r - r0, c - c0) for r, c in cells}
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if h > max_side or w > max_side:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        key = (mask.shape, mask.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(mask)
    return out


def choose_seed_patch(color: int) -> np.ndarray:
    for _ in range(100):
        mask = random.choice(SEED_SHAPES).copy()
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randrange(4))
        if rotations_all_distinct(mask) and bbox_coords(np.argwhere(mask)) == (0, 0, 2, 2):
            patch = np.zeros((3, 3), dtype=int)
            patch[mask] = int(color)
            return patch
    raise RuntimeError("failed to create asymmetric 3x3 seed patch")


def marker_radius(marker: tuple[int, int], blockers: list[tuple[int, int]]) -> int:
    return min(chebyshev(marker, blocker) for blocker in blockers) - 1


def parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == BG

    markers = sorted([tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)], key=lambda rc: (rc[1], rc[0]))
    blockers = [tuple(map(int, rc)) for rc in np.argwhere(g == GREY)]
    assert len(markers) in (2, 3)
    assert len(blockers) >= len(markers)

    payload = set(map(int, np.unique(g))) - {bg, YELLOW, GREY, MAROON}
    comps = find_mono_components(g, payload)
    assert len(comps) == 2 * len(markers)

    edge_to_comps = {edge: [] for edge in EDGE_NAMES}
    for comp in comps:
        if len(comp["edges"]) == 1:
            edge = next(iter(comp["edges"]))
            edge_to_comps[edge].append(comp)
    cue_edges = [edge for edge in EDGE_NAMES if len(edge_to_comps[edge]) == len(markers)]
    assert len(cue_edges) == 1
    cue_edge = cue_edges[0]
    cues = edge_to_comps[cue_edge]
    seeds = [comp for comp in comps if not comp["edges"]]
    assert len(seeds) == len(markers)

    cue_by_color: dict[int, dict] = {}
    for cue in cues:
        color = int(cue["color"])
        assert color not in cue_by_color
        cue_by_color[color] = cue

    seed_entries = []
    for seed in seeds:
        color = int(seed["color"])
        assert color in cue_by_color
        patch = crop_bbox(g, seed["bbox"])
        assert patch.shape == (3, 3)
        assert np.count_nonzero(patch == color) == seed["area"]
        seed_entries.append({**seed, "patch": patch})

    seed_entries.sort(key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
    assert len({int(comp["color"]) for comp in seed_entries}) == len(seed_entries)
    return {
        "bg": bg,
        "markers": markers,
        "blockers": blockers,
        "cue_edge": cue_edge,
        "cues": cues,
        "cue_by_color": cue_by_color,
        "seeds": seed_entries,
    }


def render_scene(
    grid: np.ndarray,
    scene: dict,
    *,
    disable_rotation: bool = False,
    reverse_seed_order: bool = False,
    uniform_radius: int | None = None,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    markers = list(scene["markers"])
    blockers = list(scene["blockers"])
    seeds = list(scene["seeds"])
    if reverse_seed_order:
        seeds = list(reversed(seeds))

    radii = [marker_radius(marker, blockers) for marker in markers]
    if uniform_radius is not None:
        radii = [int(uniform_radius)] * len(markers)

    h, w = out.shape
    for (r, c), radius in zip(markers, radii):
        assert radius >= 1
        r0, r1 = r - radius, r + radius
        c0, c1 = c - radius, c + radius
        assert 0 <= r0 <= r1 < h
        assert 0 <= c0 <= c1 < w
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[region == scene["bg"]] = MAROON

    for (r, c), seed in zip(markers, seeds):
        rot_k = 0 if disable_rotation else int(scene["cue_by_color"][int(seed["color"])]["area"]) % 4
        patch = rotate_seed_patch(seed["patch"], rot_k)
        for rr in range(3):
            for cc in range(3):
                val = int(patch[rr, cc])
                if val == scene["bg"]:
                    continue
                tr, tc = r - 1 + rr, c - 1 + cc
                if out[tr, tc] in (YELLOW, GREY):
                    continue
                out[tr, tc] = val

    for comp in scene["seeds"] + scene["cues"]:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = scene["bg"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw blocker-limited maroon squares around yellow markers and place cue-rotated seed patterns at their centers.

    Input structure:
    - A black background contains 2-3 yellow marker pixels, several grey blocker pixels, 2-3 interior 3x3 monochrome seed patterns, and 2-3 same-color cue sprites touching one outer edge.
    - Seed colors are distinct, each cue has the same color as exactly one seed, and cue sprites lie on a single border while seeds stay away from the border and from the marker/blocker area.

    Rule:
    - For each yellow marker, compute the minimum Chebyshev distance to any grey blocker; fill only background cells in the centered square of radius one less than that distance with maroon.
    - Sort yellow markers left-to-right, sort seed patterns left-to-right, and pair them in that order.
    - For each paired seed, find the same-color cue sprite, rotate the seed's 3x3 bounding-box contents clockwise by (cue pixel count mod 4) quarter turns, and place the rotated pattern centered on the marker while preserving yellow markers and grey blockers.
    - Erase the original seed patterns and cue sprites back to background.

    Key insight:
    - Square size is encoded locally by nearest grey blockers, while the centered pattern orientation is encoded remotely by same-color cue-sprite pixel counts; both signals must be combined inside the same square.
    """
    scene = parse_scene(np.asarray(grid, dtype=int))
    return render_scene(np.asarray(grid, dtype=int), scene)


def candidate_marker_positions(
    h: int,
    w: int,
    radius: int,
    cue_edge: str,
    placed: list[tuple[int, int, int]],
) -> list[tuple[int, int]]:
    candidates = []
    for r in range(radius, h - radius):
        for c in range(radius, w - radius):
            if cue_edge == "top" and r - radius < 4:
                continue
            if cue_edge == "bottom" and h - 1 - (r + radius) < 4:
                continue
            if cue_edge == "left" and c - radius < 4:
                continue
            if cue_edge == "right" and w - 1 - (c + radius) < 4:
                continue
            if all(chebyshev((r, c), (pr, pc)) > radius + prad + 1 for pr, pc, prad in placed):
                candidates.append((r, c))
    return candidates


def place_markers(h: int, w: int, cue_edge: str, radii: list[int]) -> list[tuple[int, int, int]] | None:
    placed: list[tuple[int, int, int]] = []
    for radius in sorted(radii, reverse=True):
        choices = candidate_marker_positions(h, w, radius, cue_edge, placed)
        if not choices:
            return None
        r, c = random.choice(choices)
        placed.append((r, c, radius))
    if len({c for _, c, _ in placed}) != len(placed):
        return None
    return placed


def place_blockers(h: int, w: int, markers: list[tuple[int, int, int]]) -> set[tuple[int, int]] | None:
    blockers: set[tuple[int, int]] = set()
    for r, c, radius in markers:
        exact = []
        for cell in ring_cells(h, w, r, c, radius + 1):
            if cell in blockers:
                continue
            if all(
                chebyshev(cell, (orow, ocol)) >= oradius + 1
                for orow, ocol, oradius in markers
            ):
                exact.append(cell)
        if not exact:
            return None
        blockers.add(random.choice(exact))

    if random.random() < 0.6:
        extra_exact = []
        for r, c, radius in markers:
            for cell in ring_cells(h, w, r, c, radius + 1):
                if cell not in blockers and all(
                    chebyshev(cell, (orow, ocol)) >= oradius + 1
                    for orow, ocol, oradius in markers
                ):
                    extra_exact.append(cell)
        if extra_exact:
            blockers.add(random.choice(extra_exact))

    extras = []
    occupied = {(r, c) for r, c, _ in markers} | blockers
    for r in range(h):
        for c in range(w):
            if (r, c) in occupied:
                continue
            if all(chebyshev((r, c), (mr, mc)) > radius + 1 for mr, mc, radius in markers):
                extras.append((r, c))
    if extras:
        for cell in random.sample(extras, k=random.randint(0, min(3, len(extras)))):
            blockers.add(cell)
    return blockers


def box_is_clear(mask: np.ndarray, r0: int, c0: int, size: int, pad: int = 1) -> bool:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + size + pad)
    cc1 = min(w, c0 + size + pad)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def choose_seed_boxes(
    h: int,
    w: int,
    cue_edge: str,
    forbidden: np.ndarray,
    count: int,
) -> list[tuple[int, int]] | None:
    candidates = []
    for r0 in range(1, h - 3):
        for c0 in range(1, w - 3):
            if cue_edge == "top" and r0 < 6:
                continue
            if cue_edge == "bottom" and r0 > h - 9:
                continue
            if cue_edge == "left" and c0 < 6:
                continue
            if cue_edge == "right" and c0 > w - 9:
                continue
            if box_is_clear(forbidden, r0, c0, 3, pad=1):
                candidates.append((r0, c0))
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    used_cols: set[int] = set()
    for r0, c0 in candidates:
        if c0 in used_cols:
            continue
        if all(max(abs(r0 - rr), abs(c0 - cc)) >= 4 for rr, cc in chosen):
            chosen.append((r0, c0))
            used_cols.add(c0)
            if len(chosen) == count:
                return chosen
    return None


def cue_size_for_rotation(rot_k: int) -> int:
    options = {0: [4], 1: [1, 5], 2: [2, 6], 3: [3, 7]}
    return random.choice(options[int(rot_k) % 4])


def place_cues(
    h: int,
    w: int,
    cue_edge: str,
    colors: list[int],
    rotations: dict[int, int],
) -> list[dict] | None:
    cue_entries = []
    order = colors[:]
    random.shuffle(order)
    for color in order:
        area = cue_size_for_rotation(rotations[color])
        masks = unique_shape_masks(area)
        if not masks:
            return None
        mask = random.choice(masks)
        cue_entries.append({"color": color, "mask": mask, "area": area})

    gap = 2
    if cue_edge in ("top", "bottom"):
        widths = [int(entry["mask"].shape[1]) for entry in cue_entries]
        need = 2 + sum(widths) + gap * (len(widths) - 1)
        if need > w:
            return None
        extras = random_partition(w - need, len(widths) + 1)
        c = 1 + extras[0]
        for i, entry in enumerate(cue_entries):
            mask = entry["mask"]
            r0 = 0 if cue_edge == "top" else h - mask.shape[0]
            c0 = c
            entry["top_left"] = (r0, c0)
            c += mask.shape[1]
            if i < len(widths) - 1:
                c += gap + extras[i + 1]
    else:
        heights = [int(entry["mask"].shape[0]) for entry in cue_entries]
        need = 2 + sum(heights) + gap * (len(heights) - 1)
        if need > h:
            return None
        extras = random_partition(h - need, len(heights) + 1)
        r = 1 + extras[0]
        for i, entry in enumerate(cue_entries):
            mask = entry["mask"]
            r0 = r
            c0 = 0 if cue_edge == "left" else w - mask.shape[1]
            entry["top_left"] = (r0, c0)
            r += mask.shape[0]
            if i < len(heights) - 1:
                r += gap + extras[i + 1]
    return cue_entries


def build_input() -> np.ndarray | None:
    h = random.randint(19, 25)
    w = random.randint(19, 25)
    cue_edge = random.choice(EDGE_NAMES)
    count = random.randint(2, 3)

    radii = [random.randint(3, 5) for _ in range(count)]
    if len(set(radii)) == 1:
        radii[0] = 3 if radii[0] > 3 else 5
    markers = place_markers(h, w, cue_edge, radii)
    if markers is None:
        return None

    blockers = place_blockers(h, w, markers)
    if blockers is None:
        return None

    forbidden = np.zeros((h, w), dtype=bool)
    for r, c, radius in markers:
        forbidden[r - radius : r + radius + 1, c - radius : c + radius + 1] = True
    for r, c in blockers:
        forbidden[r, c] = True

    seed_boxes = choose_seed_boxes(h, w, cue_edge, forbidden, count)
    if seed_boxes is None:
        return None

    colors = random.sample(PAYLOAD_COLORS, count)
    rotations = {color: random.randrange(4) for color in colors}
    if all(rot == 0 for rot in rotations.values()):
        rotations[colors[0]] = random.choice((1, 2, 3))

    cue_entries = place_cues(h, w, cue_edge, colors, rotations)
    if cue_entries is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for r, c, _ in markers:
        grid[r, c] = YELLOW
    for r, c in blockers:
        grid[r, c] = GREY

    for (r0, c0), color in zip(seed_boxes, random.sample(colors, len(colors))):
        patch = choose_seed_patch(color)
        sub = grid[r0 : r0 + 3, c0 : c0 + 3]
        if np.any(sub != BG):
            return None
        mask = patch != BG
        sub[mask] = patch[mask]

    for entry in cue_entries:
        mask = entry["mask"]
        r0, c0 = entry["top_left"]
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = int(entry["color"])
        sub = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        if np.any((patch != BG) & (sub != BG)):
            return None
        sub[patch != BG] = patch[patch != BG]

    return grid


def validate_instance(grid: np.ndarray) -> bool:
    scene = parse_scene(grid)
    markers = scene["markers"]
    blockers = scene["blockers"]
    radii = [marker_radius(marker, blockers) for marker in markers]
    if min(radii) < 3 or len(set(radii)) == 1:
        return False

    boxes = []
    for (r, c), radius in zip(markers, radii):
        box = (r - radius, c - radius, r + radius, c + radius)
        boxes.append(box)
        if box[0] < 0 or box[1] < 0 or box[2] >= grid.shape[0] or box[3] >= grid.shape[1]:
            return False
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            if not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]):
                return False

    rotations = [int(scene["cue_by_color"][int(seed["color"])]["area"]) % 4 for seed in scene["seeds"]]
    if not any(rotations):
        return False

    out = main(grid)
    if not np.array_equal(out, render_scene(grid, scene)):
        return False
    if np.array_equal(out, render_scene(grid, scene, disable_rotation=True)):
        return False
    if np.array_equal(out, render_scene(grid, scene, reverse_seed_order=True)):
        return False
    if np.array_equal(out, render_scene(grid, scene, uniform_radius=min(radii))):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = build_input()
        if grid is not None and validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate valid ARC instance")
