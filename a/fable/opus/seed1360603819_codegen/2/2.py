import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8

SEED_COLORS = (RED, GREEN, YELLOW, AZURE)
SEED_ORDER = {
    AZURE: 0,
    GREEN: 1,
    RED: 2,
    YELLOW: 3,
}

DIRECTIONS = ("up", "down", "left", "right")
DIR_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

SEPARATOR_CHOICES = (9, 10, 11)
LEGEND_SHAPE_COL = 1
LEGEND_ROW_GAP = 1
LEGEND_OUTER_MARGIN = 1
PLACEMENT_GAP = 1
GEN_RETRIES = 500


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    arr = np.asarray(mask, dtype=int)
    return tuple(tuple(int(v) for v in row) for row in arr)


def _ring_shape(height: int, width: int) -> dict[str, object]:
    mask = np.ones((height, width), dtype=bool)
    mask[1:-1, 1:-1] = False
    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    return {
        "mask": mask,
        "interior": interior,
        "h": height,
        "w": width,
        "interior_size": len(interior),
    }


SHAPES = {
    "ring33": _ring_shape(3, 3),
    "ring34": _ring_shape(3, 4),
    "ring43": _ring_shape(4, 3),
    "ring44": _ring_shape(4, 4),
}
SHAPE_NAMES = tuple(SHAPES)

ARROW_MASKS = {
    "up": np.array([[0, 1, 0], [1, 0, 1]], dtype=bool),
    "down": np.array([[1, 0, 1], [0, 1, 0]], dtype=bool),
    "left": np.array([[0, 1], [1, 0], [0, 1]], dtype=bool),
    "right": np.array([[1, 0], [0, 1], [1, 0]], dtype=bool),
}
ARROW_KEYS = {_mask_key(mask): direction for direction, mask in ARROW_MASKS.items()}


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    grid = np.asarray(mask, dtype=bool)
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not grid[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(
    coords: list[tuple[int, int]], bbox: tuple[int, int, int, int] | None = None
) -> np.ndarray:
    if bbox is None:
        bbox = _bbox(coords)
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _interior_mask(border_mask: np.ndarray) -> np.ndarray:
    wall = np.asarray(border_mask, dtype=bool)
    h, w = wall.shape
    outside = np.zeros_like(wall, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not wall[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not wall[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return (~wall) & (~outside)


def _find_separator(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == MAGENTA)]
    assert len(cols) == 1, "expected exactly one full magenta separator column"
    return int(cols[0])


def _parse_arrow(coords: list[tuple[int, int]]) -> str:
    key = _mask_key(_mask_from_coords(coords))
    assert key in ARROW_KEYS, "unrecognized legend arrow"
    return ARROW_KEYS[key]


def _legend_mapping(grid: np.ndarray, sep: int) -> dict[tuple[tuple[int, ...], ...], str]:
    legend = grid[:, :sep]
    grey_comps = _components(legend == GREY, connectivity=4)
    orange_comps = _components(legend == ORANGE, connectivity=8)
    assert grey_comps, "legend must contain at least one grey shape"
    assert len(grey_comps) == len(orange_comps), "legend rows must pair one shape with one arrow"

    grey_entries = []
    for coords in grey_comps:
        bbox = _bbox(coords)
        grey_entries.append(
            {
                "bbox": bbox,
                "center": 0.5 * (bbox[0] + bbox[2]),
                "key": _mask_key(_mask_from_coords(coords, bbox)),
            }
        )

    orange_entries = []
    for coords in orange_comps:
        bbox = _bbox(coords)
        orange_entries.append(
            {
                "bbox": bbox,
                "center": 0.5 * (bbox[0] + bbox[2]),
                "direction": _parse_arrow(coords),
            }
        )

    used_arrows: set[int] = set()
    mapping: dict[tuple[tuple[int, ...], ...], str] = {}
    for entry in sorted(grey_entries, key=lambda item: (item["bbox"][0], item["bbox"][1])):
        r0, _c0, r1, _c1 = entry["bbox"]
        best: tuple[tuple[float, float, float], int] | None = None
        for idx, arrow in enumerate(orange_entries):
            if idx in used_arrows:
                continue
            ar0, ac0, ar1, _ac1 = arrow["bbox"]
            overlap = max(0, min(r1, ar1) - max(r0, ar0) + 1)
            center_gap = abs(float(entry["center"]) - float(arrow["center"]))
            col_gap = abs(float(ac0) - float(_c1))
            score = (0.0 if overlap > 0 else 1.0, center_gap, col_gap)
            if best is None or score < best[0]:
                best = (score, idx)
        assert best is not None, "each legend shape must have an arrow"
        used_arrows.add(best[1])
        arrow = orange_entries[best[1]]
        assert entry["key"] not in mapping, "legend shape keys must be unique"
        mapping[entry["key"]] = str(arrow["direction"])

    assert len(mapping) == len(grey_entries), "legend pairing must cover all shapes"
    return mapping


def _extract_containers(
    grid: np.ndarray, sep: int
) -> tuple[list[dict[str, object]], np.ndarray]:
    mask = grid == GREY
    mask[:, : sep + 1] = False
    comps = _components(mask, connectivity=4)
    assert comps, "main canvas must contain at least one grey container"

    owner = np.full(grid.shape, -1, dtype=int)
    containers: list[dict[str, object]] = []

    for idx, coords in enumerate(comps):
        bbox = _bbox(coords)
        border_mask = _mask_from_coords(coords, bbox)
        inside = _interior_mask(border_mask)
        assert np.any(inside), "every container must enclose a cavity"

        r0, c0, r1, c1 = bbox
        local = grid[r0 : r1 + 1, c0 : c1 + 1]
        seed_points = np.argwhere(inside & (local != BLACK))
        assert len(seed_points) == 1, "each container cavity must contain exactly one colored seed"
        sr, sc = (int(seed_points[0, 0]), int(seed_points[0, 1]))
        seed_color = int(local[sr, sc])
        assert seed_color in SEED_COLORS, "seed color must be one of the four allowed colors"

        interior_coords = [(r0 + int(rr), c0 + int(cc)) for rr, cc in np.argwhere(inside)]

        for r, c in coords:
            assert owner[r, c] == -1, "containers must not overlap"
            owner[r, c] = idx

        containers.append(
            {
                "id": idx,
                "bbox": bbox,
                "shape_key": _mask_key(border_mask),
                "seed": (r0 + sr, c0 + sc),
                "seed_color": seed_color,
                "interior": interior_coords,
            }
        )

    return containers, owner


def _trace_target(
    seed: tuple[int, int], direction: str, owner: np.ndarray, sep: int, self_id: int
) -> int | None:
    dr, dc = DIR_STEPS[direction]
    h, w = owner.shape
    r = seed[0] + dr
    c = seed[1] + dc
    canvas_left = sep + 1
    while 0 <= r < h and canvas_left <= c < w:
        hit = int(owner[r, c])
        if hit >= 0 and hit != self_id:
            return hit
        r += dr
        c += dc
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Map each grey hollow shape to a direction via the left legend, cast a ray
    from each seed along that direction, and fill the first hit container's
    cavity with the shooter's seed color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator(g)
    shape_to_direction = _legend_mapping(g, sep)
    containers, owner = _extract_containers(g, sep)

    incoming: dict[int, list[int]] = defaultdict(list)
    for container in containers:
        shape_key = container["shape_key"]
        assert shape_key in shape_to_direction, "every container shape must appear in the legend"
        direction = shape_to_direction[shape_key]
        target = _trace_target(container["seed"], direction, owner, sep, int(container["id"]))
        if target is not None:
            incoming[target].append(int(container["seed_color"]))

    out = g.copy()
    for target_id, colors in incoming.items():
        fill = min(colors, key=lambda color: SEED_ORDER[color])
        for r, c in containers[target_id]["interior"]:
            out[r, c] = fill
    return out


def _container_spec(shape_name: str, direction: str) -> dict[str, object]:
    spec = SHAPES[shape_name]
    return {
        "shape_name": shape_name,
        "direction": direction,
        "mask": spec["mask"],
        "interior": list(spec["interior"]),
        "h": int(spec["h"]),
        "w": int(spec["w"]),
        "interior_size": int(spec["interior_size"]),
    }


def _bbox_from_position(container: dict[str, object], position: tuple[int, int]) -> tuple[int, int, int, int]:
    top, left = position
    return top, left, top + int(container["h"]) - 1, left + int(container["w"]) - 1


def _bbox_in_canvas(bbox: tuple[int, int, int, int], h: int, w: int, sep: int) -> bool:
    r0, c0, r1, c1 = bbox
    return 0 <= r0 and r1 < h and sep + 1 <= c0 and c1 < w


def _bbox_clear(
    bbox: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]], gap: int = PLACEMENT_GAP
) -> bool:
    r0, c0, r1, c1 = bbox
    for or0, oc0, or1, oc1 in boxes:
        rows_separated = r1 + gap < or0 or or1 + gap < r0
        cols_separated = c1 + gap < oc0 or oc1 + gap < c0
        if not (rows_separated or cols_separated):
            return False
    return True


def _owner_from_positions(
    containers: list[dict[str, object]], positions: dict[int, tuple[int, int]], h: int, w: int
) -> np.ndarray:
    owner = np.full((h, w), -1, dtype=int)
    for idx, container in enumerate(containers):
        top, left = positions[idx]
        mask = np.asarray(container["mask"], dtype=bool)
        for rr, cc in np.argwhere(mask):
            r = top + int(rr)
            c = left + int(cc)
            assert owner[r, c] == -1, "generated containers must not overlap"
            owner[r, c] = idx
    return owner


def _trace_from_positions(
    idx: int,
    seed_local: tuple[int, int],
    containers: list[dict[str, object]],
    positions: dict[int, tuple[int, int]],
    owner: np.ndarray,
    sep: int,
) -> int | None:
    top, left = positions[idx]
    seed = (top + int(seed_local[0]), left + int(seed_local[1]))
    return _trace_target(seed, str(containers[idx]["direction"]), owner, sep, idx)


def _assign_seed_locals(
    containers: list[dict[str, object]],
    positions: dict[int, tuple[int, int]],
    requirements: dict[int, int | None],
    h: int,
    w: int,
    sep: int,
) -> dict[int, tuple[int, int]] | None:
    owner = _owner_from_positions(containers, positions, h, w)
    assignments: dict[int, tuple[int, int]] = {}
    order = list(requirements)
    random.shuffle(order)
    for idx in order:
        want = requirements[idx]
        candidates = []
        for seed_local in containers[idx]["interior"]:
            hit = _trace_from_positions(idx, seed_local, containers, positions, owner, sep)
            if want is None:
                if hit is None:
                    candidates.append(seed_local)
            elif hit == want:
                candidates.append(seed_local)
        if not candidates:
            return None
        assignments[idx] = random.choice(candidates)
    return assignments


def _simulate_hits(
    containers: list[dict[str, object]],
    positions: dict[int, tuple[int, int]],
    seeds: dict[int, tuple[int, int]],
    h: int,
    w: int,
    sep: int,
) -> tuple[dict[int, int | None], dict[int, list[int]]]:
    owner = _owner_from_positions(containers, positions, h, w)
    hits: dict[int, int | None] = {}
    incoming: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(containers)):
        target = _trace_from_positions(idx, seeds[idx], containers, positions, owner, sep)
        hits[idx] = target
        if target is not None:
            incoming[target].append(idx)
    return hits, incoming


def _side_candidates_for_hit(
    container: dict[str, object],
    target_container: dict[str, object],
    target_pos: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    h: int,
    w: int,
    sep: int,
    gap: int,
) -> list[tuple[int, int]]:
    tr0, tc0, tr1, tc1 = _bbox_from_position(target_container, target_pos)
    candidates: set[tuple[int, int]] = set()
    direction = str(container["direction"])
    if direction == "right":
        left = tc0 - gap - int(container["w"])
        for seed_local in container["interior"]:
            for hit_row in range(tr0, tr1 + 1):
                top = hit_row - int(seed_local[0])
                bbox = _bbox_from_position(container, (top, left))
                if _bbox_in_canvas(bbox, h, w, sep) and _bbox_clear(bbox, boxes):
                    candidates.add((top, left))
    elif direction == "left":
        left = tc1 + gap + 1
        for seed_local in container["interior"]:
            for hit_row in range(tr0, tr1 + 1):
                top = hit_row - int(seed_local[0])
                bbox = _bbox_from_position(container, (top, left))
                if _bbox_in_canvas(bbox, h, w, sep) and _bbox_clear(bbox, boxes):
                    candidates.add((top, left))
    elif direction == "down":
        top = tr0 - gap - int(container["h"])
        for seed_local in container["interior"]:
            for hit_col in range(tc0, tc1 + 1):
                left = hit_col - int(seed_local[1])
                bbox = _bbox_from_position(container, (top, left))
                if _bbox_in_canvas(bbox, h, w, sep) and _bbox_clear(bbox, boxes):
                    candidates.add((top, left))
    else:
        top = tr1 + gap + 1
        for seed_local in container["interior"]:
            for hit_col in range(tc0, tc1 + 1):
                left = hit_col - int(seed_local[1])
                bbox = _bbox_from_position(container, (top, left))
                if _bbox_in_canvas(bbox, h, w, sep) and _bbox_clear(bbox, boxes):
                    candidates.add((top, left))
    return list(candidates)


def _edge_candidates(
    container: dict[str, object],
    boxes: list[tuple[int, int, int, int]],
    h: int,
    w: int,
    sep: int,
    samples: int = 80,
) -> list[tuple[int, int]]:
    positions: set[tuple[int, int]] = set()
    direction = str(container["direction"])
    for _ in range(samples):
        if direction == "left":
            top = random.randint(0, h - int(container["h"]))
            left = sep + 1 + random.randint(0, 2)
        elif direction == "right":
            top = random.randint(0, h - int(container["h"]))
            left = w - int(container["w"]) - random.randint(0, 2)
        elif direction == "up":
            top = random.randint(0, 2)
            left = random.randint(sep + 1, w - int(container["w"]))
        else:
            top = h - int(container["h"]) - random.randint(0, 2)
            left = random.randint(sep + 1, w - int(container["w"]))
        bbox = _bbox_from_position(container, (top, left))
        if _bbox_in_canvas(bbox, h, w, sep) and _bbox_clear(bbox, boxes):
            positions.add((top, left))
    return list(positions)


def _legend_total_height(shape_names: list[str], shape_to_direction: dict[str, str]) -> int:
    total = 2 * LEGEND_OUTER_MARGIN
    for idx, shape_name in enumerate(shape_names):
        arrow_h = ARROW_MASKS[shape_to_direction[shape_name]].shape[0]
        total += max(int(SHAPES[shape_name]["h"]), int(arrow_h))
        if idx + 1 < len(shape_names):
            total += LEGEND_ROW_GAP
    return total


def _draw_legend(
    grid: np.ndarray,
    sep: int,
    shape_to_direction: dict[str, str],
    shape_names: list[str],
) -> None:
    order = list(shape_names)
    random.shuffle(order)
    band_heights = [
        max(int(SHAPES[name]["h"]), int(ARROW_MASKS[shape_to_direction[name]].shape[0])) for name in order
    ]
    total_height = 2 * LEGEND_OUTER_MARGIN + sum(band_heights) + LEGEND_ROW_GAP * (len(order) - 1)
    start_row = random.randint(0, grid.shape[0] - total_height)
    arrow_col = sep - 3
    cursor = start_row + LEGEND_OUTER_MARGIN
    for name, band_h in zip(order, band_heights):
        shape = SHAPES[name]
        arrow = ARROW_MASKS[shape_to_direction[name]]
        shape_top = cursor + (band_h - int(shape["h"])) // 2
        arrow_top = cursor + (band_h - arrow.shape[0]) // 2
        shape_mask = np.asarray(shape["mask"], dtype=bool)
        for rr, cc in np.argwhere(shape_mask):
            grid[shape_top + int(rr), LEGEND_SHAPE_COL + int(cc)] = GREY
        for rr, cc in np.argwhere(arrow):
            grid[arrow_top + int(rr), arrow_col + int(cc)] = ORANGE
        cursor += band_h + LEGEND_ROW_GAP


def _draw_container(
    grid: np.ndarray,
    container: dict[str, object],
    position: tuple[int, int],
    seed_local: tuple[int, int],
    color: int,
) -> None:
    top, left = position
    mask = np.asarray(container["mask"], dtype=bool)
    for rr, cc in np.argwhere(mask):
        grid[top + int(rr), left + int(cc)] = GREY
    grid[top + int(seed_local[0]), left + int(seed_local[1])] = color


def _render_grid(
    h: int,
    w: int,
    sep: int,
    shape_to_direction: dict[str, str],
    containers: list[dict[str, object]],
    positions: dict[int, tuple[int, int]],
    seeds: dict[int, tuple[int, int]],
    colors: list[int],
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    grid[:, sep] = MAGENTA
    _draw_legend(grid, sep, shape_to_direction, [str(container["shape_name"]) for container in containers])
    for idx, container in enumerate(containers):
        _draw_container(grid, container, positions[idx], seeds[idx], colors[idx])
    return grid


def _try_multi_hit(
    containers: list[dict[str, object]],
    h: int,
    w: int,
    sep: int,
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]] | None:
    indices = list(range(len(containers)))
    target_candidates = [idx for idx in indices if int(containers[idx]["interior_size"]) > 1] or indices
    random.shuffle(target_candidates)

    for target_idx in target_candidates:
        target = containers[target_idx]
        others = [idx for idx in indices if idx != target_idx]
        gaps = {idx: random.randint(2, 4) for idx in others}

        left_need = max(
            [int(containers[idx]["w"]) + gaps[idx] for idx in others if containers[idx]["direction"] == "right"]
            + [2]
        )
        right_need = max(
            [int(containers[idx]["w"]) + gaps[idx] for idx in others if containers[idx]["direction"] == "left"]
            + [2]
        )
        up_need = max(
            [int(containers[idx]["h"]) + gaps[idx] for idx in others if containers[idx]["direction"] == "down"]
            + [2]
        )
        down_need = max(
            [int(containers[idx]["h"]) + gaps[idx] for idx in others if containers[idx]["direction"] == "up"]
            + [2]
        )

        r_min = up_need
        r_max = h - int(target["h"]) - down_need
        c_min = sep + 1 + left_need
        c_max = w - int(target["w"]) - right_need
        if r_min > r_max or c_min > c_max:
            continue

        for _ in range(40):
            target_pos = (random.randint(r_min, r_max), random.randint(c_min, c_max))
            positions = {target_idx: target_pos}
            boxes = [_bbox_from_position(target, target_pos)]
            order = others[:]
            random.shuffle(order)
            ok = True
            for idx in order:
                options = _side_candidates_for_hit(
                    containers[idx], target, target_pos, boxes, h, w, sep, gaps[idx]
                )
                if not options:
                    ok = False
                    break
                pos = random.choice(options)
                positions[idx] = pos
                boxes.append(_bbox_from_position(containers[idx], pos))
            if not ok:
                continue

            requirements = {idx: target_idx for idx in others}
            requirements[target_idx] = None
            seeds = _assign_seed_locals(containers, positions, requirements, h, w, sep)
            if seeds is None:
                continue

            hits, incoming = _simulate_hits(containers, positions, seeds, h, w, sep)
            if any(len(srcs) >= 2 for srcs in incoming.values()) and hits[target_idx] is None:
                return positions, seeds
    return None


def _try_single_hit(
    containers: list[dict[str, object]],
    h: int,
    w: int,
    sep: int,
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]] | None:
    indices = list(range(len(containers)))
    target_candidates = [idx for idx in indices if int(containers[idx]["interior_size"]) > 1] or indices

    for _ in range(80):
        target_idx = random.choice(target_candidates)
        shooter_idx = random.choice([idx for idx in indices if idx != target_idx])
        target = containers[target_idx]

        r_min = 2
        r_max = h - int(target["h"]) - 3
        c_min = sep + 3
        c_max = w - int(target["w"]) - 3
        if r_min > r_max or c_min > c_max:
            continue

        target_pos = (random.randint(r_min, r_max), random.randint(c_min, c_max))
        positions = {target_idx: target_pos}
        boxes = [_bbox_from_position(target, target_pos)]

        gap = random.randint(2, 4)
        shooter_options = _side_candidates_for_hit(
            containers[shooter_idx], target, target_pos, boxes, h, w, sep, gap
        )
        if not shooter_options:
            continue
        shooter_pos = random.choice(shooter_options)
        positions[shooter_idx] = shooter_pos
        boxes.append(_bbox_from_position(containers[shooter_idx], shooter_pos))

        remaining = [idx for idx in indices if idx not in positions]
        random.shuffle(remaining)
        ok = True
        for idx in remaining:
            options = _edge_candidates(containers[idx], boxes, h, w, sep)
            if not options:
                ok = False
                break
            pos = random.choice(options)
            positions[idx] = pos
            boxes.append(_bbox_from_position(containers[idx], pos))
        if not ok:
            continue

        requirements = {idx: None for idx in indices}
        requirements[shooter_idx] = target_idx
        seeds = _assign_seed_locals(containers, positions, requirements, h, w, sep)
        if seeds is None:
            continue

        hits, incoming = _simulate_hits(containers, positions, seeds, h, w, sep)
        hit_count = sum(target is not None for target in hits.values())
        if hit_count == 1 and max((len(srcs) for srcs in incoming.values()), default=0) == 1:
            return positions, seeds
    return None


def _try_no_hit(
    containers: list[dict[str, object]],
    h: int,
    w: int,
    sep: int,
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]] | None:
    positions: dict[int, tuple[int, int]] = {}
    boxes: list[tuple[int, int, int, int]] = []
    order = list(range(len(containers)))
    random.shuffle(order)
    for idx in order:
        options = _edge_candidates(containers[idx], boxes, h, w, sep)
        if not options:
            return None
        pos = random.choice(options)
        positions[idx] = pos
        boxes.append(_bbox_from_position(containers[idx], pos))

    requirements = {idx: None for idx in range(len(containers))}
    seeds = _assign_seed_locals(containers, positions, requirements, h, w, sep)
    if seeds is None:
        return None

    hits, _incoming = _simulate_hits(containers, positions, seeds, h, w, sep)
    if all(target is None for target in hits.values()):
        return positions, seeds
    return None


def _gen_attempt() -> np.ndarray | None:
    scenario = random.choice(["multi_hit", "single_hit", "no_hit"])
    if scenario == "multi_hit":
        count = random.randint(3, 4)
    else:
        count = random.randint(2, 4)

    sep = random.choice(SEPARATOR_CHOICES)
    shape_names = random.sample(list(SHAPE_NAMES), count)
    directions = random.sample(list(DIRECTIONS), count)
    shape_to_direction = dict(zip(shape_names, directions))
    containers = [_container_spec(name, shape_to_direction[name]) for name in shape_names]

    legend_height = _legend_total_height(shape_names, shape_to_direction)
    min_h = max(legend_height + 1, 18 if scenario == "multi_hit" else 16)
    max_h = min_h + 5
    h = random.randint(min_h, max_h)

    min_main_w = 18 if scenario == "multi_hit" else 15
    main_w = random.randint(min_main_w, min_main_w + 6)
    w = sep + 1 + main_w

    if scenario == "multi_hit":
        result = _try_multi_hit(containers, h, w, sep)
    elif scenario == "single_hit":
        result = _try_single_hit(containers, h, w, sep)
    else:
        result = _try_no_hit(containers, h, w, sep)
    if result is None:
        return None

    positions, seeds = result
    colors = random.sample(list(SEED_COLORS), count)
    grid = _render_grid(h, w, sep, shape_to_direction, containers, positions, seeds, colors)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if scenario != "no_hit" and np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
