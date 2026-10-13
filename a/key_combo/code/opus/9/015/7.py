from __future__ import annotations

from collections import defaultdict, deque
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np

try:
    from arc_utils import bbox, crop_bbox, find_components, get_bg, paste, random_free_location
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_task_015_7", "/home/lixinyu/snap/new/solvers/arc_utils.py"
    ).load_module()
    bbox = _ARC_UTILS.bbox
    crop_bbox = _ARC_UTILS.crop_bbox
    find_components = _ARC_UTILS.find_components
    get_bg = _ARC_UTILS.get_bg
    paste = _ARC_UTILS.paste
    random_free_location = _ARC_UTILS.random_free_location


BG = 0
BODY = 9
MARKER_COLORS = [1, 2, 3]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _coords_bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_connected(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    if not coords:
        return False
    start = coords[0]
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c, mask.shape):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == len(coords)


def _marker_direction(body_mask: np.ndarray, r: int, c: int) -> tuple[int, int]:
    h, w = body_mask.shape
    dirs = []
    if r == 0:
        dirs.append((-1, 0))
    if r == h - 1:
        dirs.append((1, 0))
    if c == 0:
        dirs.append((0, -1))
    if c == w - 1:
        dirs.append((0, 1))
    if len(dirs) != 1:
        raise ValueError("marker must lie on a unique side of its fragment bbox")
    return dirs[0]


def _assemble_fragments(
    grid: np.ndarray,
) -> tuple[list[np.ndarray], dict[int, tuple[int, int]], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg = g[g != bg]
    if fg.size == 0:
        raise ValueError("empty foreground")

    values, counts = np.unique(fg, return_counts=True)
    body = int(values[int(np.argmax(counts))])
    marker_colors = {int(v) for v in values if int(v) != body}
    if not marker_colors:
        raise ValueError("no connector markers")

    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    sprites: list[np.ndarray] = []
    markers_by_color: dict[int, list[tuple[int, tuple[int, int], tuple[int, int]]]] = defaultdict(list)

    for idx, comp in enumerate(components):
        sprite = crop_bbox(g, comp["bbox"])
        sprites.append(sprite)
        body_mask = sprite == body
        marker_locs = np.argwhere((sprite != bg) & (sprite != body))
        if marker_locs.size == 0:
            raise ValueError("fragment without markers")
        for rr, cc in marker_locs:
            r, c = int(rr), int(cc)
            direction = _marker_direction(body_mask, r, c)
            markers_by_color[int(sprite[r, c])].append((idx, (r, c), direction))

    adjacency: list[list[tuple[int, tuple[int, int]]]] = [[] for _ in sprites]
    seam_points: list[tuple[int, int, int]] = []
    for color, occurrences in markers_by_color.items():
        if len(occurrences) != 2:
            raise ValueError("each marker color must appear exactly twice")
        (a_idx, a_pos, a_dir), (b_idx, b_pos, b_dir) = occurrences
        if a_idx == b_idx or b_dir != (-a_dir[0], -a_dir[1]):
            raise ValueError("invalid docking pair")
        delta = (a_pos[0] + a_dir[0] - b_pos[0], a_pos[1] + a_dir[1] - b_pos[1])
        adjacency[a_idx].append((b_idx, delta))
        adjacency[b_idx].append((a_idx, (-delta[0], -delta[1])))

    placements: dict[int, tuple[int, int]] = {0: (0, 0)}
    q = deque([0])
    while q:
        idx = q.popleft()
        base_r, base_c = placements[idx]
        for nbr, (dr, dc) in adjacency[idx]:
            pos = (base_r + dr, base_c + dc)
            if nbr in placements:
                if placements[nbr] != pos:
                    raise ValueError("inconsistent docking constraints")
            else:
                placements[nbr] = pos
                q.append(nbr)
    if len(placements) != len(sprites):
        raise ValueError("fragments are not fully connected by marker pairs")

    for idx, sprite in enumerate(sprites):
        base_r, base_c = placements[idx]
        for rr, cc in np.argwhere((sprite != bg) & (sprite != body)):
            seam_points.append((base_r + int(rr), base_c + int(cc), int(sprite[rr, cc])))
    return sprites, placements, seam_points


def _fill_holes_from_seams(
    body_grid: np.ndarray,
    seam_points: list[tuple[int, int, int]],
) -> np.ndarray:
    out = body_grid.copy()
    ordered_points = sorted(seam_points, key=lambda x: (x[2], x[0], x[1]))
    for rr, cc in np.argwhere(out == BG):
        r, c = int(rr), int(cc)
        best = None
        best_color = None
        for sr, sc, color in ordered_points:
            dist = abs(r - sr) + abs(c - sc)
            key = (dist, sr, sc, color)
            if best is None or key < best:
                best = key
                best_color = color
        out[r, c] = int(best_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble marker-linked maroon fragments, then fill the assembled bbox gaps by nearest seam color.

    Input structure:
    - The grid has black background and 3-4 separated foreground fragments.
    - Each fragment has a maroon body plus 1-2 external connector-marker pixels in blue, red, or green.
    - Every connector color appears exactly twice, on two different fragments, and each marker sits just outside its fragment body on the side that indicates docking direction.

    Rule:
    - Detect each fragment and pair equal-colored connector markers.
    - Translate the fragments rigidly so every matched marker pair becomes edge-adjacent with opposite outward directions.
    - Remove all connector markers but remember their assembled coordinates as seam points.
    - Take the tight bounding box of the assembled maroon bodies.
    - Every black pixel inside that bbox is a gap; recolor it with the color of the Manhattan-nearest seam point, then return the tight crop.

    Key insight:
    - The connector markers first determine the spatial assembly, then their vanished positions still act as color anchors for filling the remaining bbox holes.
    """
    sprites, placements, seam_points = _assemble_fragments(grid)

    body_cells: list[tuple[int, int]] = []
    for idx, sprite in enumerate(sprites):
        base_r, base_c = placements[idx]
        for rr, cc in np.argwhere(sprite == BODY):
            body_cells.append((base_r + int(rr), base_c + int(cc)))
    if not body_cells:
        raise ValueError("no body cells")

    r0 = min(r for r, _ in body_cells)
    c0 = min(c for _, c in body_cells)
    r1 = max(r for r, _ in body_cells)
    c1 = max(c for _, c in body_cells)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)

    for idx, sprite in enumerate(sprites):
        base_r, base_c = placements[idx]
        body_patch = np.where(sprite == BODY, BODY, BG)
        rr, cc = base_r - r0, base_c - c0
        target = out[rr : rr + body_patch.shape[0], cc : cc + body_patch.shape[1]]
        mask = body_patch != BG
        if np.any((target != BG) & mask):
            raise ValueError("assembled bodies overlap")
        paste(out, body_patch, rr, cc, transparent=BG)

    shifted_seams = [(r - r0, c - c0, color) for r, c, color in seam_points]
    return _fill_holes_from_seams(out, shifted_seams)


def _random_partition(total: int, parts: int, low: int, high: int) -> list[int] | None:
    values = [low] * parts
    remaining = total - low * parts
    if remaining < 0:
        return None
    while remaining > 0:
        choices = [i for i, v in enumerate(values) if v < high]
        if not choices:
            return None
        idx = random.choice(choices)
        values[idx] += 1
        remaining -= 1
    random.shuffle(values)
    return values


def _make_canonical_layout(n: int) -> tuple[int, int, list[tuple[int, int]], list[dict]]:
    if n == 3:
        h = random.choice([5, 6])
        w = 7
        r1 = random.randint(1, h - 2)
        r2 = random.randint(1, h - 2)
        seeds = [(h // 2, 0), (h // 2, 3), (h // 2, w - 1)]
        seams = [
            {
                "color": 1,
                "a": 0,
                "b": 1,
                "p1": (r1, 2),
                "p2": (r1, 3),
                "anchor1": (r1, 1),
                "anchor2": (r1, 4),
            },
            {
                "color": 2 if random.random() < 0.5 else 3,
                "a": 1,
                "b": 2,
                "p1": (r2, 4),
                "p2": (r2, 5),
                "anchor1": (r2, 3),
                "anchor2": (r2, 6),
            },
        ]
        return h, w, seeds, seams

    h = 6
    w = 7
    row_mid = 2
    c_mid = 2
    c_left = 1
    c_right = 5
    r_top = random.randint(1, 2)
    seeds = [(1, 1), (1, 5), (4, 1), (4, 5)]
    seams = [
        {"color": 1, "a": 0, "b": 1, "p1": (r_top, c_mid), "p2": (r_top, c_mid + 1), "anchor1": (r_top, c_mid - 1), "anchor2": (r_top, c_mid + 2)},
        {"color": 2, "a": 0, "b": 2, "p1": (row_mid, c_left), "p2": (row_mid + 1, c_left), "anchor1": (row_mid - 1, c_left), "anchor2": (row_mid + 2, c_left)},
        {"color": 3, "a": 1, "b": 3, "p1": (row_mid, c_right), "p2": (row_mid + 1, c_right), "anchor1": (row_mid - 1, c_right), "anchor2": (row_mid + 2, c_right)},
    ]
    return h, w, seeds, seams


def _grow_regions(
    h: int,
    w: int,
    n: int,
    targets: list[int],
    seeds: list[tuple[int, int]],
    seams: list[dict],
) -> np.ndarray | None:
    labels = np.full((h, w), -1, dtype=int)
    reserved = set()
    region_seeds = [set([tuple(map(int, seed))]) for seed in seeds]
    forbidden = [set() for _ in range(n)]

    for seam in seams:
        p1 = tuple(map(int, seam["p1"]))
        p2 = tuple(map(int, seam["p2"]))
        reserved.add(p1)
        reserved.add(p2)
        region_seeds[seam["a"]].add(tuple(map(int, seam["anchor1"])))
        region_seeds[seam["b"]].add(tuple(map(int, seam["anchor2"])))
        for rr, cc in _neighbors4(*p1, (h, w)):
            if (rr, cc) != tuple(map(int, seam["anchor1"])):
                forbidden[seam["a"]].add((rr, cc))
        for rr, cc in _neighbors4(*p2, (h, w)):
            if (rr, cc) != tuple(map(int, seam["anchor2"])):
                forbidden[seam["b"]].add((rr, cc))

    for r, c in reserved:
        labels[r, c] = -2

    sizes = [0] * n
    for idx in range(n):
        cells = list(region_seeds[idx])
        if len(cells) > targets[idx]:
            return None
        for r, c in cells:
            if not (0 <= r < h and 0 <= c < w) or (r, c) in reserved:
                return None
            if labels[r, c] not in (-1, idx):
                return None
            labels[r, c] = idx
        sizes[idx] = len(cells)

    total_target = sum(targets)
    while sum(sizes) < total_target:
        progressed = False
        deficits = sorted(range(n), key=lambda i: (targets[i] - sizes[i], random.random()), reverse=True)
        for idx in deficits:
            if sizes[idx] >= targets[idx]:
                continue
            frontier = set()
            for r, c in np.argwhere(labels == idx):
                for rr, cc in _neighbors4(int(r), int(c), (h, w)):
                    if labels[rr, cc] == -1 and (rr, cc) not in reserved and (rr, cc) not in forbidden[idx]:
                        frontier.add((rr, cc))
            if not frontier:
                continue
            seed_r, seed_c = seeds[idx]
            ranked = sorted(frontier, key=lambda p: (abs(p[0] - seed_r) + abs(p[1] - seed_c), random.random()))
            pick = random.choice(ranked[: min(4, len(ranked))])
            labels[pick] = idx
            sizes[idx] += 1
            progressed = True
        if not progressed:
            return None

    if np.any(labels == -1):
        return None
    for idx in range(n):
        if not _mask_connected(labels == idx):
            return None
    return labels


def _carve_extra_holes(
    labels: np.ndarray,
    seams: list[dict],
    extra_count: int,
) -> tuple[np.ndarray, list[tuple[int, int]]] | tuple[None, None]:
    h, w = labels.shape
    seam_points = [tuple(map(int, seam["p1"])) + (int(seam["color"]),) for seam in seams]
    seam_points += [tuple(map(int, seam["p2"])) + (int(seam["color"]),) for seam in seams]
    forbidden_cells = {tuple(map(int, seam["p1"])) for seam in seams} | {
        tuple(map(int, seam["p2"])) for seam in seams
    }

    chosen: list[tuple[int, int]] = []
    work = labels.copy()
    for _ in range(extra_count):
        candidates = []
        for r in range(h):
            for c in range(w):
                if work[r, c] < 0 or (r, c) in forbidden_cells:
                    continue
                region = int(work[r, c])
                if np.sum(work == region) <= 8:
                    continue
                test = work.copy()
                test[r, c] = -2
                if not _mask_connected(test == region):
                    continue
                dists = sorted(abs(r - sr) + abs(c - sc) for sr, sc, _ in seam_points)
                if len(dists) >= 2 and dists[0] == dists[1]:
                    continue
                if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in chosen):
                    continue
                candidates.append((r, c))
        if not candidates:
            return None, None
        pick = random.choice(candidates)
        work[pick] = -2
        chosen.append(pick)
    return work, chosen


def _build_full_fragment_layers(
    labels: np.ndarray,
    seams: list[dict],
) -> list[np.ndarray]:
    n = int(labels.max()) + 1
    layers = [np.zeros(labels.shape, dtype=int) for _ in range(n)]
    for idx in range(n):
        layers[idx][labels == idx] = BODY
    for seam in seams:
        r1, c1 = map(int, seam["p1"])
        r2, c2 = map(int, seam["p2"])
        color = int(seam["color"])
        layers[int(seam["a"])][r1, c1] = color
        layers[int(seam["b"])][r2, c2] = color
    return layers


def _oracle_output_from_labels(labels: np.ndarray, seams: list[dict]) -> np.ndarray:
    body = np.where(labels >= 0, BODY, BG)
    seam_points = [tuple(map(int, seam["p1"])) + (int(seam["color"]),) for seam in seams]
    seam_points += [tuple(map(int, seam["p2"])) + (int(seam["color"]),) for seam in seams]
    return _fill_holes_from_seams(body, seam_points)


def _symmetry(grid: np.ndarray, rot: int, flip: bool) -> np.ndarray:
    out = np.rot90(grid, rot)
    if flip:
        out = np.fliplr(out)
    return out


def _scatter_fragments(full_layers: list[np.ndarray]) -> np.ndarray | None:
    canvas_h = random.randint(18, 24)
    canvas_w = random.randint(18, 24)
    canvas = np.zeros((canvas_h, canvas_w), dtype=int)
    shadow = np.zeros((canvas_h, canvas_w), dtype=int)

    sprites = []
    for layer in full_layers:
        bb = bbox(layer != BG)
        if bb is None:
            return None
        sprites.append(crop_bbox(layer, bb))

    order = sorted(range(len(sprites)), key=lambda i: sprites[i].shape[0] * sprites[i].shape[1], reverse=True)
    for idx in order:
        sprite = sprites[idx]
        rect = np.ones(sprite.shape, dtype=int)
        try:
            r0, c0 = random_free_location(
                shadow,
                rect,
                transparent=0,
                padding=2,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(shadow, rect, r0, c0)
        paste(canvas, sprite, r0, c0, transparent=BG)
    return canvas


def _validate_semantics(inp: np.ndarray, out: np.ndarray, n: int) -> bool:
    comps = find_components(inp, bg=BG, connectivity=4, monochromatic=False)
    if len(comps) != n:
        return False

    marker_counts = defaultdict(int)
    for comp in comps:
        sprite = crop_bbox(inp, comp["bbox"])
        body_area = int(np.sum(sprite == BODY))
        if not (8 <= body_area <= 15):
            return False
        marks = np.argwhere((sprite != BG) & (sprite != BODY))
        if len(marks) not in (1, 2):
            return False
        for rr, cc in marks:
            try:
                _marker_direction(sprite == BODY, int(rr), int(cc))
            except ValueError:
                return False
            marker_counts[int(sprite[rr, cc])] += 1

    if sorted(marker_counts.values()) != [2] * (n - 1):
        return False

    body_mask = out == BODY
    if body_mask.any():
        b = bbox(body_mask)
        if b != (0, 0, out.shape[0] - 1, out.shape[1] - 1):
            return False
    else:
        return False

    try:
        sprites, placements, seam_points = _assemble_fragments(inp)
    except Exception:
        return False

    body_cells = set()
    for idx, sprite in enumerate(sprites):
        br, bc = placements[idx]
        for rr, cc in np.argwhere(sprite == BODY):
            body_cells.add((br + int(rr), bc + int(cc)))
    if not body_cells:
        return False
    r0, c0, r1, c1 = _coords_bbox(body_cells)
    holes = []
    seam_xy = {(r, c) for r, c, _ in seam_points}
    non_seam_holes = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) not in body_cells:
                holes.append((r, c))
                if (r, c) not in seam_xy:
                    non_seam_holes += 1
                dists = sorted(abs(r - sr) + abs(c - sc) for sr, sc, _ in seam_points)
                if len(dists) >= 2 and dists[0] == dists[1]:
                    return False
    if not (3 <= len(holes) <= 6):
        return False
    if n == 3 and non_seam_holes < 1:
        return False
    return np.array_equal(main(inp), out)


def _generate_attempt() -> np.ndarray | None:
    n = 3 if random.random() < 0.6 else 4
    h, w = 5, 8
    labels = np.full((h, w), -1, dtype=int)
    seams: list[dict] = []

    if n == 3:
        labels[:, :3] = 0
        labels[:, 3:5] = 1
        labels[:, 5:] = 2
        r1, r2 = random.sample([1, 2, 3], 2)
        seams = [
            {"color": 1, "a": 0, "b": 1, "p1": (r1, 2), "p2": (r1, 3)},
            {"color": 2 if random.random() < 0.5 else 3, "a": 1, "b": 2, "p1": (r2, 4), "p2": (r2, 5)},
        ]
    else:
        labels[:, :2] = 0
        labels[:, 2:4] = 1
        labels[:, 4:6] = 2
        labels[:, 6:] = 3
        r1, r2, r3 = random.sample([1, 2, 3], 3)
        seams = [
            {"color": 1, "a": 0, "b": 1, "p1": (r1, 1), "p2": (r1, 2)},
            {"color": 2, "a": 1, "b": 2, "p1": (r2, 3), "p2": (r2, 4)},
            {"color": 3, "a": 2, "b": 3, "p1": (r3, 5), "p2": (r3, 6)},
        ]

    for seam in seams:
        r1, c1 = map(int, seam["p1"])
        r2, c2 = map(int, seam["p2"])
        labels[r1, c1] = -2
        labels[r2, c2] = -2

    if n == 3:
        labels, carved = _carve_extra_holes(labels, seams, random.choice([1, 2]))
        if labels is None or carved is None:
            return None

    layers = _build_full_fragment_layers(labels, seams)
    out = _oracle_output_from_labels(labels, seams)
    rot = random.randrange(4)
    flip = random.random() < 0.5
    layers = [_symmetry(layer, rot, flip) for layer in layers]
    out = _symmetry(out, rot, flip)

    inp = _scatter_fragments(layers)
    if inp is None:
        return None
    if not _validate_semantics(inp, out, n):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 015/7")
