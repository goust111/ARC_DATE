import random

import numpy as np


GRID_H_RANGE = (16, 25)
GRID_W_RANGE = (16, 25)
ANCHOR_COUNT_CHOICES = (2, 3)
ANCHOR_H_RANGE = (2, 5)
ANCHOR_W_RANGE = (2, 6)
MAX_RETRIES = 400
PLACEMENT_TRIES = 250
PARTICLE_COLORS_MIN = 2
PARTICLE_COLORS_MAX = 4


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _ring_capacity(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return 2 * (r1 - r0 + 1) + 2 * (c1 - c0 + 1) + 4


def _outer_ring(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    assert 0 < r0 <= r1 < h - 1
    assert 0 < c0 <= c1 < w - 1

    ring: list[tuple[int, int]] = []
    for c in range(c0 - 1, c1 + 2):
        ring.append((r0 - 1, c))
    for r in range(r0, r1 + 2):
        ring.append((r, c1 + 1))
    for c in range(c1, c0 - 2, -1):
        ring.append((r1 + 1, c))
    for r in range(r1, r0 - 1, -1):
        ring.append((r, c0 - 1))
    return ring


def _distance_to_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    r, c = cell
    r0, c0, r1, c1 = bbox
    dr = 0
    if r < r0:
        dr = r0 - r
    elif r > r1:
        dr = r - r1
    dc = 0
    if c < c0:
        dc = c0 - c
    elif c > c1:
        dc = c - c1
    return int(dr + dc)


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    anchors: list[dict] = []
    particles: list[dict] = []

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for coords in _components_of_color(g, color):
            if len(coords) == 1:
                particles.append({"color": color, "pos": coords[0]})
                continue
            r0, c0, r1, c1 = _bbox(coords)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert len(coords) == area, "foreground non-singletons must be solid rectangles"
            anchors.append({"color": color, "bbox": (r0, c0, r1, c1)})

    anchors.sort(key=lambda a: (a["bbox"][0], a["bbox"][1]))
    particles.sort(key=lambda p: p["pos"])
    assert 2 <= len(anchors) <= 3, "expected 2-3 anchor rectangles"
    return bg, anchors, particles


def _assign_particles(anchors: list[dict], particles: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = [[] for _ in anchors]
    for particle in sorted(particles, key=lambda p: p["pos"]):
        dists = [_distance_to_bbox(particle["pos"], anchor["bbox"]) for anchor in anchors]
        best = min(dists)
        winners = [idx for idx, dist in enumerate(dists) if dist == best]
        assert len(winners) == 1, "particle-to-anchor assignment must be unambiguous"
        groups[winners[0]].append(particle)
    return groups


def _render_output(shape: tuple[int, int], bg: int, anchors: list[dict], particles: list[dict]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for anchor in anchors:
        r0, c0, r1, c1 = anchor["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(anchor["color"])

    groups = _assign_particles(anchors, particles)
    for anchor, group in zip(anchors, groups):
        ring = _outer_ring(anchor["bbox"], shape)
        assert len(group) <= len(ring), "coating cannot exceed ring length"
        for (r, c), particle in zip(ring, group):
            out[r, c] = int(particle["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assign each singleton particle to its unique nearest solid rectangle anchor.
    Then clear the particles and repaint their colors clockwise on the anchor's
    outside ring, starting from the upper-left ring corner.

    When several particles map to the same anchor, their colors are consumed in
    row-major order so the multicolor coating is deterministic.
    """
    g = np.asarray(grid, dtype=int)
    bg, anchors, particles = _parse_scene(g)
    return _render_output(g.shape, bg, anchors, particles)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _sample_anchors(h: int, w: int, bg: int, num_anchors: int) -> list[dict] | None:
    colors = random.sample([c for c in range(10) if c != bg], num_anchors)
    anchors: list[dict] = []
    expanded_boxes: list[tuple[int, int, int, int]] = []

    for color in colors:
        placed = False
        for _ in range(PLACEMENT_TRIES):
            ah = random.randint(*ANCHOR_H_RANGE)
            aw = random.randint(*ANCHOR_W_RANGE)
            if ah + 2 > h or aw + 2 > w:
                continue
            r0 = random.randint(1, h - ah - 1)
            c0 = random.randint(1, w - aw - 1)
            bbox = (r0, c0, r0 + ah - 1, c0 + aw - 1)
            expanded = (r0 - 1, c0 - 1, r0 + ah, c0 + aw)
            if any(_boxes_overlap(expanded, other) for other in expanded_boxes):
                continue
            anchors.append({"color": int(color), "bbox": bbox})
            expanded_boxes.append(expanded)
            placed = True
            break
        if not placed:
            return None

    anchors.sort(key=lambda a: (a["bbox"][0], a["bbox"][1]))
    return anchors


def _anchor_buffer_mask(shape: tuple[int, int], anchors: list[dict]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for anchor in anchors:
        r0, c0, r1, c1 = anchor["bbox"]
        mask[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
    return mask


def _candidate_cells(shape: tuple[int, int], anchors: list[dict]) -> list[list[tuple[int, int]]]:
    h, w = shape
    blocked = _anchor_buffer_mask(shape, anchors)
    pools: list[list[tuple[int, int]]] = [[] for _ in anchors]
    for r in range(h):
        for c in range(w):
            if blocked[r, c]:
                continue
            dists = [_distance_to_bbox((r, c), anchor["bbox"]) for anchor in anchors]
            best = min(dists)
            winners = [idx for idx, dist in enumerate(dists) if dist == best]
            if len(winners) == 1:
                pools[winners[0]].append((r, c))
    return pools


def _mark_dilated(mask: np.ndarray, cell: tuple[int, int]) -> None:
    h, w = mask.shape
    r, c = cell
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            mask[rr, cc] = True


def _sample_counts(anchors: list[dict], pools: list[list[tuple[int, int]]]) -> list[int] | None:
    counts: list[int] = []
    num_anchors = len(anchors)
    max_particles = 8 if num_anchors == 2 else 6

    for anchor, pool in zip(anchors, pools):
        if not pool:
            return None
        ring_cap = _ring_capacity(anchor["bbox"])
        spatial_cap = max(1, len(pool) // 6)
        upper = min(ring_cap, spatial_cap, max_particles)
        if upper < 1:
            return None
        counts.append(random.randint(1, upper))

    if max(counts) == 1:
        idx = random.randrange(num_anchors)
        ring_cap = _ring_capacity(anchors[idx]["bbox"])
        spatial_cap = max(1, len(pools[idx]) // 6)
        upper = min(ring_cap, spatial_cap, max_particles)
        if upper >= 2:
            counts[idx] = random.randint(2, upper)
    return counts


def _place_particles(
    shape: tuple[int, int],
    anchors: list[dict],
    pools: list[list[tuple[int, int]]],
    counts: list[int],
    particle_palette: list[int],
) -> list[dict] | None:
    blocked = _anchor_buffer_mask(shape, anchors)
    particles: list[dict] = []

    order = sorted(range(len(anchors)), key=lambda idx: (len(pools[idx]) / counts[idx], len(pools[idx])))
    for idx in order:
        options = pools[idx][:]
        random.shuffle(options)
        chosen: list[tuple[int, int]] = []
        for cell in options:
            r, c = cell
            if blocked[r, c]:
                continue
            chosen.append(cell)
            _mark_dilated(blocked, cell)
            if len(chosen) == counts[idx]:
                break
        if len(chosen) != counts[idx]:
            return None
        for cell in chosen:
            particles.append({"color": int(random.choice(particle_palette)), "pos": cell})

    if len(particle_palette) >= 2 and len(particles) >= 2:
        used = {particle["color"] for particle in particles}
        if len(used) == 1:
            particles[-1]["color"] = int(random.choice([c for c in particle_palette if c != particles[-1]["color"]]))

    particles.sort(key=lambda p: p["pos"])
    return particles


def _build_input(shape: tuple[int, int], bg: int, anchors: list[dict], particles: list[dict]) -> np.ndarray:
    inp = np.full(shape, int(bg), dtype=int)
    for anchor in anchors:
        r0, c0, r1, c1 = anchor["bbox"]
        inp[r0 : r1 + 1, c0 : c1 + 1] = int(anchor["color"])
    for particle in particles:
        r, c = particle["pos"]
        inp[r, c] = int(particle["color"])
    return inp


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)
    num_anchors = random.choice(ANCHOR_COUNT_CHOICES)

    anchors = _sample_anchors(h, w, bg, num_anchors)
    if anchors is None:
        return None

    pools = _candidate_cells((h, w), anchors)
    counts = _sample_counts(anchors, pools)
    if counts is None:
        return None

    used_colors = {bg} | {anchor["color"] for anchor in anchors}
    particle_choices = [c for c in range(10) if c not in used_colors]
    if len(particle_choices) < PARTICLE_COLORS_MIN:
        return None
    palette_size = random.randint(PARTICLE_COLORS_MIN, min(PARTICLE_COLORS_MAX, len(particle_choices)))
    particle_palette = random.sample(particle_choices, palette_size)

    particles = _place_particles((h, w), anchors, pools, counts, particle_palette)
    if particles is None:
        return None

    inp = _build_input((h, w), bg, anchors, particles)
    if _get_bg(inp) != bg:
        return None

    expected = _render_output((h, w), bg, anchors, particles)
    try:
        parsed_bg, parsed_anchors, parsed_particles = _parse_scene(inp)
    except Exception:
        return None
    if parsed_bg != bg or len(parsed_anchors) != len(anchors) or len(parsed_particles) != len(particles):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
