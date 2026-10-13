import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 400


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[list[tuple[int, int]]] = []
    next_id = 1
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            components.append(coords)
            next_id += 1
    return labels, components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _hole_regions_for_component(coords: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    local = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        local[r - r0, c - c0] = True

    open_mask = ~local
    outside = np.zeros_like(open_mask, dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                if open_mask[r, c] and not outside[r, c]:
                    outside[r, c] = True
                    q.append((r, c))

    while q:
        rr, cc = q.popleft()
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    holes_mask = open_mask & ~outside
    _, hole_components = _label_components(holes_mask)
    return [[(r + r0, c + c0) for r, c in comp] for comp in hole_components]


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    grey_mask = g == GREY
    labels, components = _label_components(grey_mask)

    infos = []
    emitters: dict[int, dict] = {}
    receivers: dict[int, dict] = {}
    for comp_id, coords in enumerate(components, start=1):
        holes = _hole_regions_for_component(coords)
        info = {
            "id": comp_id,
            "coords": coords,
            "area": len(coords),
            "holes": holes,
            "hole": holes[0][0] if holes and len(holes) == 1 and len(holes[0]) == 1 else None,
        }
        infos.append(info)
        if info["hole"] is not None:
            emitters[comp_id] = info
        else:
            receivers[comp_id] = info

    hit_counts = {comp_id: 0 for comp_id in receivers}
    rays = []
    emitter_ids = set(emitters)
    for comp_id, info in emitters.items():
        hole_r, hole_c = info["hole"]
        for dr, dc in DIRS4:
            rr, cc = hole_r, hole_c
            result = {"source": comp_id, "dir": (dr, dc), "kind": "edge", "target": None}
            while True:
                rr += dr
                cc += dc
                if not in_bounds(g.shape, rr, cc):
                    break
                if labels[rr, cc] == comp_id or g[rr, cc] == BLACK:
                    continue
                other = int(labels[rr, cc])
                if other in emitter_ids:
                    result = {"source": comp_id, "dir": (dr, dc), "kind": "blocked", "target": other}
                    break
                if other in receivers:
                    hit_counts[other] += 1
                    result = {"source": comp_id, "dir": (dr, dc), "kind": "hit", "target": other}
                    break
                break
            rays.append(result)

    return {
        "labels": labels,
        "components": infos,
        "emitters": emitters,
        "receivers": receivers,
        "hit_counts": hit_counts,
        "rays": rays,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor holeless grey components according to how many orthogonal rays reach them from holed grey components.

    Input structure:
    - The grid is mostly black background with several disconnected grey 4-connected components.
    - Some grey components contain exactly one enclosed black hole pixel; the others contain no enclosed background.
    - Blue and red are absent from the input and are reserved as output marker colors.

    Rule:
    - Split the grey components by topology: holed components are emitters and holeless components are receivers.
    - For each emitter, use its unique enclosed hole pixel as the ray origin and cast rays up, down, left, and right.
    - While leaving an emitter, ignore the emitter's own body and background; then stop at the first different grey component encountered. A first hit on a receiver counts, while a first hit on another emitter only blocks the ray.
    - Recolor each receiver by total hits: exactly 1 hit becomes blue, 2 or more hits become red, and 0 hits stay grey. Emitters and hole pixels stay unchanged.

    Key insight:
    - Color is not the role cue. The deciding feature is topology: a single enclosed hole turns a grey component into a ray source, and only the global collision counts determine the receiver colors.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze_scene(g)

    out = g.copy()
    for comp_id, info in analysis["receivers"].items():
        hits = analysis["hit_counts"][comp_id]
        if hits == 1:
            color = BLUE
        elif hits >= 2:
            color = RED
        else:
            color = GREY
        for r, c in info["coords"]:
            out[r, c] = color
    return out


def _grow_shape(
    rng: random.Random,
    required: set[tuple[int, int]],
    allowed: set[tuple[int, int]],
    target_size: int,
) -> set[tuple[int, int]] | None:
    cells = set(required)
    while len(cells) < target_size:
        frontier = set()
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in allowed and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(rng.choice(sorted(frontier)))
    return cells


def _make_emitter_shape(rng: random.Random) -> set[tuple[int, int]] | None:
    required = {(-1, 0), (1, 0), (0, -1), (0, 1)}
    diagonals = {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    target = rng.randint(4, 8)
    return _grow_shape(rng, required, required | diagonals, target)


def _make_positive_receiver(rng: random.Random, max_h: int, max_w: int) -> set[tuple[int, int]] | None:
    allowed = {(r, c) for r in range(max_h) for c in range(max_w)}
    target = rng.randint(3, min(8, len(allowed)))
    return _grow_shape(rng, {(0, 0)}, allowed, target)


def _make_blue_receiver(rng: random.Random) -> set[tuple[int, int]] | None:
    allowed = {(r, c) for r in range(-1, 3) for c in range(0, 3)}
    target = rng.randint(3, 6)
    return _grow_shape(rng, {(0, 0)}, allowed, target)


def _place_shape(
    comp_grid: np.ndarray,
    comp_id: int,
    anchor: tuple[int, int],
    rel_cells: set[tuple[int, int]],
) -> bool:
    pts = []
    ar, ac = anchor
    for dr, dc in rel_cells:
        rr = ar + dr
        cc = ac + dc
        if not in_bounds(comp_grid.shape, rr, cc) or comp_grid[rr, cc] != 0:
            return False
        pts.append((rr, cc))
    for rr, cc in pts:
        comp_grid[rr, cc] = comp_id
    return True


def _transform_scene(comp_grid: np.ndarray, hole_grid: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    rot = rng.randrange(4)
    comp_t = np.rot90(comp_grid, rot)
    hole_t = np.rot90(hole_grid, rot)
    if rng.random() < 0.5:
        comp_t = np.fliplr(comp_t)
        hole_t = np.fliplr(hole_t)
    return comp_t, hole_t


def _components_separated(grid: np.ndarray) -> bool:
    labels, components = _label_components(grid == GREY)
    for i, comp_a in enumerate(components):
        for comp_b in components[i + 1 :]:
            for ra, ca in comp_a:
                for rb, cb in comp_b:
                    if max(abs(ra - rb), abs(ca - cb)) < 2:
                        return False
    return True


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    gap_row = rng.randint(4, 6)
    gap_col = rng.randint(4, 6)
    blue_gap = gap_col + rng.randint(3, 4)
    low_gap = gap_row + rng.randint(3, 4)
    margin = 3

    anchors = {
        1: (margin, margin),
        2: (margin, margin + gap_col),
        3: (margin + gap_row, margin),
        4: (margin + gap_row, margin + gap_col),
        5: (margin, margin + blue_gap),
        6: (margin + low_gap, margin + blue_gap + 2),
    }

    local_h = low_gap + 6
    local_w = blue_gap + 8
    comp_grid = np.zeros((local_h, local_w), dtype=int)
    hole_grid = np.zeros((local_h, local_w), dtype=int)

    emitter1 = _make_emitter_shape(rng)
    emitter2 = _make_emitter_shape(rng)
    emitter3 = _make_emitter_shape(rng)
    red_receiver = _make_positive_receiver(rng, 3, 3)
    blue_receiver = _make_blue_receiver(rng)
    zero_receiver = _make_positive_receiver(rng, 3, 4)
    shapes = [emitter1, emitter2, emitter3, red_receiver, blue_receiver, zero_receiver]
    if any(shape is None for shape in shapes):
        return None

    for comp_id, shape in enumerate(shapes, start=1):
        if not _place_shape(comp_grid, comp_id, anchors[comp_id], shape):
            return None
        if comp_id <= 3:
            hr, hc = anchors[comp_id]
            if comp_grid[hr, hc] != 0:
                return None
            hole_grid[hr, hc] = comp_id

    comp_grid, hole_grid = _transform_scene(comp_grid, hole_grid, rng)
    pad_top = rng.randint(0, 2)
    pad_bottom = rng.randint(0, 2)
    pad_left = rng.randint(0, 2)
    pad_right = rng.randint(0, 2)
    h, w = comp_grid.shape
    final = np.zeros((h + pad_top + pad_bottom, w + pad_left + pad_right), dtype=int)
    final[pad_top : pad_top + h, pad_left : pad_left + w][comp_grid > 0] = GREY
    return final


def _validate_generated(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not np.isin(grid, [BLACK, GREY]).all():
        return False
    if not _components_separated(grid):
        return False

    analysis = _analyze_scene(grid)
    emitters = analysis["emitters"]
    receivers = analysis["receivers"]
    if len(emitters) < 2 or len(receivers) < 2:
        return False
    if len(emitters) + len(receivers) < 4 or len(emitters) + len(receivers) > 7:
        return False

    for info in analysis["components"]:
        if not (3 <= info["area"] <= 8):
            return False
    for info in emitters.values():
        if info["hole"] is None:
            return False
        if len(info["holes"]) != 1 or len(info["holes"][0]) != 1:
            return False

    counts = sorted(analysis["hit_counts"].values())
    if not any(v == 0 for v in counts):
        return False
    if not any(v == 1 for v in counts):
        return False
    if not any(v >= 2 for v in counts):
        return False
    if not any(ray["kind"] == "blocked" for ray in analysis["rays"]):
        return False
    if not any(ray["kind"] == "edge" for ray in analysis["rays"]):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not (out == BLUE).any():
        return False
    if not (out == RED).any():
        return False
    if not (out == GREY).any():
        return False

    blue_components = 0
    red_components = 0
    grey_receiver_components = 0
    for comp_id, info in receivers.items():
        color = int(out[info["coords"][0][0], info["coords"][0][1]])
        hits = analysis["hit_counts"][comp_id]
        if hits == 1 and color != BLUE:
            return False
        if hits >= 2 and color != RED:
            return False
        if hits == 0 and color != GREY:
            return False
        blue_components += int(color == BLUE)
        red_components += int(color == RED)
        grey_receiver_components += int(color == GREY)
    if blue_components == 0 or red_components == 0 or grey_receiver_components == 0:
        return False

    for info in emitters.values():
        for r, c in info["coords"]:
            if out[r, c] != GREY:
                return False
        hr, hc = info["hole"]
        if out[hr, hc] != BLACK:
            return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
