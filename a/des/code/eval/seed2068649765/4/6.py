import random
from collections import Counter

import numpy as np


BG = 0
SIDES = ("top", "left", "bottom", "right")
WALL_COLORS = (1, 2, 3)
MARKER_COLORS = (4, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _classify_components(grid: np.ndarray) -> tuple[list[dict], list[list[tuple[int, int]]]]:
    cargos: list[dict] = []
    containers: list[list[tuple[int, int]]] = []
    for comp in _components(grid != BG):
        r0, c0, r1, c1 = _bbox(comp)
        vals = [int(grid[r, c]) for r, c in comp]
        unique = set(vals)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if len(unique) == 1 and h == w and 1 <= h <= 3 and len(comp) == h * w:
            cargos.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "color": int(vals[0]),
                    "size": int(h),
                }
            )
        else:
            containers.append(comp)
    return cargos, containers


def _extract_marker_sequence(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    side: str,
    wall_color: int,
) -> list[int]:
    r0, c0, r1, c1 = rect
    seq: list[int] = []
    seen: set[int] = set()

    if side == "top":
        axis = [(r, (c0 - 1, c1 + 1)) for r in range(r0, r1 + 1)]
    elif side == "bottom":
        axis = [(r, (c0 - 1, c1 + 1)) for r in range(r1, r0 - 1, -1)]
    elif side == "left":
        axis = [((r0 - 1, r1 + 1), c) for c in range(c0, c1 + 1)]
    else:
        axis = [((r0 - 1, r1 + 1), c) for c in range(c1, c0 - 1, -1)]

    for entry in axis:
        colors: set[int] = set()
        if side in ("top", "bottom"):
            row, side_cols = entry
            for col in side_cols:
                color = int(grid[row, col])
                if color not in (BG, wall_color):
                    colors.add(color)
        else:
            side_rows, col = entry
            for row in side_rows:
                color = int(grid[row, col])
                if color not in (BG, wall_color):
                    colors.add(color)
        if len(colors) > 1:
            return []
        if colors:
            color = next(iter(colors))
            if color in seen:
                return []
            seq.append(color)
            seen.add(color)

    return seq if 1 <= len(seq) <= 3 else []


def _valid_cavity(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    rect: tuple[int, int, int, int],
    side: str,
    wall_color: int,
) -> list[int]:
    h, w = grid.shape
    r0, c0, r1, c1 = rect
    if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BG):
        return []

    if side == "top":
        if not (r0 > 0 and r1 + 1 < h and c0 > 0 and c1 + 1 < w):
            return []
        if not np.all(comp_mask[r0 : r1 + 1, c0 - 1]):
            return []
        if not np.all(comp_mask[r0 : r1 + 1, c1 + 1]):
            return []
        if not np.all(grid[r1 + 1, c0 : c1 + 1] == wall_color):
            return []
        if np.any(grid[r0 - 1, c0 : c1 + 1] != BG):
            return []
        opening = c1 - c0 + 1
    elif side == "bottom":
        if not (r1 + 1 < h and r0 > 0 and c0 > 0 and c1 + 1 < w):
            return []
        if not np.all(comp_mask[r0 : r1 + 1, c0 - 1]):
            return []
        if not np.all(comp_mask[r0 : r1 + 1, c1 + 1]):
            return []
        if not np.all(grid[r0 - 1, c0 : c1 + 1] == wall_color):
            return []
        if np.any(grid[r1 + 1, c0 : c1 + 1] != BG):
            return []
        opening = c1 - c0 + 1
    elif side == "left":
        if not (c0 > 0 and c1 + 1 < w and r0 > 0 and r1 + 1 < h):
            return []
        if not np.all(comp_mask[r0 - 1, c0 : c1 + 1]):
            return []
        if not np.all(comp_mask[r1 + 1, c0 : c1 + 1]):
            return []
        if not np.all(grid[r0 : r1 + 1, c1 + 1] == wall_color):
            return []
        if np.any(grid[r0 : r1 + 1, c0 - 1] != BG):
            return []
        opening = r1 - r0 + 1
    else:
        if not (c1 + 1 < w and c0 > 0 and r0 > 0 and r1 + 1 < h):
            return []
        if not np.all(comp_mask[r0 - 1, c0 : c1 + 1]):
            return []
        if not np.all(comp_mask[r1 + 1, c0 : c1 + 1]):
            return []
        if not np.all(grid[r0 : r1 + 1, c0 - 1] == wall_color):
            return []
        if np.any(grid[r0 : r1 + 1, c1 + 1] != BG):
            return []
        opening = r1 - r0 + 1

    if not (2 <= opening <= 4):
        return []
    return _extract_marker_sequence(grid, rect, side, wall_color)


def _analyze_container(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    vals = [int(grid[r, c]) for r, c in coords]
    wall_color = Counter(vals).most_common(1)[0][0]
    br0, bc0, br1, bc1 = _bbox(coords)
    comp_mask = np.zeros_like(grid, dtype=bool)
    for r, c in coords:
        comp_mask[r, c] = True

    best = None
    best_score = None
    for r0 in range(br0, br1 + 1):
        for r1 in range(r0, br1 + 1):
            for c0 in range(bc0, bc1 + 1):
                for c1 in range(c0, bc1 + 1):
                    rect = (r0, c0, r1, c1)
                    if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BG):
                        continue
                    for side in SIDES:
                        marker_seq = _valid_cavity(grid, comp_mask, rect, side, wall_color)
                        if not marker_seq:
                            continue
                        if side in ("top", "bottom"):
                            opening = c1 - c0 + 1
                            depth = r1 - r0 + 1
                        else:
                            opening = r1 - r0 + 1
                            depth = c1 - c0 + 1
                        score = (opening * depth, len(marker_seq), depth)
                        if best_score is None or score > best_score:
                            best_score = score
                            best = {
                                "bbox": (br0, bc0, br1, bc1),
                                "cavity": rect,
                                "side": side,
                                "markers": marker_seq,
                                "opening": int(opening),
                                "depth": int(depth),
                                "wall_color": int(wall_color),
                            }
    assert best is not None, "failed to detect a valid container cavity"
    return best


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    cargos, container_comps = _classify_components(grid)
    containers = [_analyze_container(grid, comp) for comp in container_comps]
    cargos.sort(key=lambda item: item["bbox"])
    containers.sort(key=lambda item: item["bbox"])
    for idx, cargo in enumerate(cargos):
        cargo["id"] = idx
    return cargos, containers


def _stack_origin(container: dict, size: int) -> tuple[int, int]:
    r0, c0, r1, c1 = container["cavity"]
    if container["side"] in ("top", "bottom"):
        col = c0 + (container["opening"] - size) // 2
        return 0, col
    row = r0 + (container["opening"] - size) // 2
    return row, 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move square cargos into open containers.

    Cavity width decides which cargos can pass through the opening.
    Matching marker colors on the opening edges then decide the bottom-to-top
    stacking order inside each container.
    """
    g = np.asarray(grid, dtype=int)
    cargos, containers = _parse_scene(g)
    out = g.copy()

    cargos_by_color: dict[int, list[dict]] = {}
    for cargo in cargos:
        cargos_by_color.setdefault(int(cargo["color"]), []).append(cargo)

    assignments: dict[int, list[dict]] = {i: [] for i in range(len(containers))}
    used_ids: set[int] = set()
    for ci, container in enumerate(containers):
        for color in container["markers"]:
            candidates = [
                cargo
                for cargo in cargos_by_color.get(int(color), [])
                if cargo["size"] <= container["opening"] and cargo["id"] not in used_ids
            ]
            if not candidates:
                continue
            cargo = sorted(candidates, key=lambda item: (item["size"], item["bbox"]))[0]
            assignments[ci].append(cargo)
            used_ids.add(cargo["id"])

    for cargo_id in used_ids:
        cargo = cargos[cargo_id]
        r0, c0, r1, c1 = cargo["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for ci, container in enumerate(containers):
        r0, c0, r1, c1 = container["cavity"]
        side = container["side"]
        if side == "top":
            cursor = r1
            for cargo in assignments[ci]:
                size = cargo["size"]
                _, col = _stack_origin(container, size)
                row = cursor - size + 1
                out[row : cursor + 1, col : col + size] = cargo["color"]
                cursor -= size
        elif side == "bottom":
            cursor = r0
            for cargo in assignments[ci]:
                size = cargo["size"]
                _, col = _stack_origin(container, size)
                out[cursor : cursor + size, col : col + size] = cargo["color"]
                cursor += size
        elif side == "left":
            cursor = c1
            for cargo in assignments[ci]:
                size = cargo["size"]
                row, _ = _stack_origin(container, size)
                col = cursor - size + 1
                out[row : row + size, col : cursor + 1] = cargo["color"]
                cursor -= size
        else:
            cursor = c0
            for cargo in assignments[ci]:
                size = cargo["size"]
                row, _ = _stack_origin(container, size)
                out[row : row + size, cursor : cursor + size] = cargo["color"]
                cursor += size

    return out


def _build_container_patch(
    opening: int,
    depth: int,
    wall_color: int,
    marker_colors: list[int],
    style: str,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    extra = rng.randint(1, 2) if style == "asym" else 0
    arm_side = rng.choice(["left", "right"])

    h = depth + 1 + extra
    w = opening + 2
    patch = np.zeros((h, w), dtype=int)
    cavity = np.zeros((h, w), dtype=bool)
    start_r = extra

    patch[start_r : start_r + depth + 1, 0] = wall_color
    patch[start_r : start_r + depth + 1, w - 1] = wall_color
    patch[start_r + depth, :] = wall_color
    cavity[start_r : start_r + depth, 1 : w - 1] = True

    if extra:
        patch[:extra, 0 if arm_side == "left" else w - 1] = wall_color

    for idx, color in enumerate(marker_colors):
        row = start_r + idx
        patch[row, 0] = color
        patch[row, w - 1] = color

    return patch, cavity


def _rotate_container(
    patch: np.ndarray,
    cavity: np.ndarray,
    side: str,
) -> tuple[np.ndarray, tuple[int, int, int, int], str]:
    k = {"top": 0, "left": 1, "bottom": 2, "right": 3}[side]
    rpatch = np.rot90(patch, k)
    rcavity = np.rot90(cavity, k)
    coords = np.argwhere(rcavity)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return rpatch, (int(r0), int(c0), int(r1), int(c1)), side


def _bbox_clear(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    hh, ww = occupied.shape
    r0 = max(0, top - pad)
    r1 = min(hh, top + h + pad)
    c0 = max(0, left - pad)
    c1 = min(ww, left + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _paint_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    sub = grid[top : top + h, left : left + w]
    mask = patch != BG
    sub[mask] = patch[mask]
    occupied[top : top + h, left : left + w][mask] = True


def _container_space_ok(
    grid_shape: tuple[int, int],
    top: int,
    left: int,
    patch_shape: tuple[int, int],
    cavity_bbox: tuple[int, int, int, int],
    side: str,
) -> bool:
    h, w = patch_shape
    gh, gw = grid_shape
    if top < 1 or left < 1 or top + h >= gh - 1 or left + w >= gw - 1:
        return False

    r0, c0, r1, c1 = cavity_bbox
    world_r0 = top + r0
    world_c0 = left + c0
    world_r1 = top + r1
    world_c1 = left + c1
    cargo_room = 5
    if side == "top":
        return world_r0 >= cargo_room
    if side == "bottom":
        return gh - world_r1 - 1 >= cargo_room
    if side == "left":
        return world_c0 >= cargo_room
    return gw - world_c1 - 1 >= cargo_room


def _candidate_positions(
    grid_shape: tuple[int, int],
    container: dict,
    size: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    h, w = grid_shape
    br0, bc0, br1, bc1 = container["bbox"]
    cr0, cc0, cr1, cc1 = container["cavity"]
    cr0 += container["top"]
    cc0 += container["left"]
    cr1 += container["top"]
    cc1 += container["left"]
    br0 += container["top"]
    bc0 += container["left"]
    br1 += container["top"]
    bc1 += container["left"]

    positions: list[tuple[int, int]] = []
    if container["side"] == "top":
        r_start = max(1, cr0 - size - 5)
        r_end = cr0 - size - 1
        c_start = max(1, min(bc0, cc0) - 2)
        c_end = min(w - size - 1, max(bc1, cc1) - size + 3)
        for r in range(r_start, r_end + 1):
            for c in range(c_start, c_end + 1):
                positions.append((r, c))
    elif container["side"] == "bottom":
        r_start = cr1 + 2
        r_end = min(h - size - 1, max(br1, cr1) + 5)
        c_start = max(1, min(bc0, cc0) - 2)
        c_end = min(w - size - 1, max(bc1, cc1) - size + 3)
        for r in range(r_start, r_end + 1):
            for c in range(c_start, c_end + 1):
                positions.append((r, c))
    elif container["side"] == "left":
        c_start = max(1, cc0 - size - 5)
        c_end = cc0 - size - 1
        r_start = max(1, min(br0, cr0) - 2)
        r_end = min(h - size - 1, max(br1, cr1) - size + 3)
        for r in range(r_start, r_end + 1):
            for c in range(c_start, c_end + 1):
                positions.append((r, c))
    else:
        c_start = cc1 + 2
        c_end = min(w - size - 1, max(bc1, cc1) + 5)
        r_start = max(1, min(br0, cr0) - 2)
        r_end = min(h - size - 1, max(br1, cr1) - size + 3)
        for r in range(r_start, r_end + 1):
            for c in range(c_start, c_end + 1):
                positions.append((r, c))

    rng.shuffle(positions)
    if positions:
        return positions

    fallback = [(r, c) for r in range(1, h - size) for c in range(1, w - size)]
    rng.shuffle(fallback)
    return fallback


def _square_clear(occupied: np.ndarray, top: int, left: int, size: int, pad: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    r1 = min(h, top + size + pad)
    c0 = max(0, left - pad)
    c1 = min(w, left + size + pad)
    return not occupied[r0:r1, c0:c1].any()


def _paint_square(grid: np.ndarray, occupied: np.ndarray, top: int, left: int, size: int, color: int) -> None:
    grid[top : top + size, left : left + size] = color
    occupied[top : top + size, left : left + size] = True


def _marker_count_partition(n_containers: int, rng: random.Random) -> list[int]:
    total = rng.randint(n_containers, min(6, 3 * n_containers))
    counts = [1] * n_containers
    for _ in range(total - n_containers):
        choices = [idx for idx, count in enumerate(counts) if count < 3]
        counts[rng.choice(choices)] += 1
    rng.shuffle(counts)
    return counts


def _container_spec(marker_count: int, marker_colors: list[int], rng: random.Random) -> dict:
    opening = rng.randint(2, 4)
    while True:
        sizes = []
        for _ in range(marker_count):
            if opening == 2 and rng.random() < 0.4:
                sizes.append(3)
            else:
                sizes.append(rng.randint(1, 3))
        fit_sum = sum(size for size in sizes if size <= opening)
        depth = max(4, marker_count + rng.randint(0, 2), fit_sum + rng.randint(1, 2))
        if depth <= 9:
            break
    return {
        "opening": int(opening),
        "depth": int(depth),
        "marker_colors": marker_colors,
        "cargo_sizes": sizes,
        "wall_color": int(rng.choice(WALL_COLORS)),
        "style": "asym" if rng.random() < 0.5 else "u",
        "side": rng.choice(SIDES),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_containers = rng.choice([1, 2, 2, 3])
    marker_counts = _marker_count_partition(n_containers, rng)
    total_markers = sum(marker_counts)
    marker_palette = list(MARKER_COLORS)
    rng.shuffle(marker_palette)
    marker_palette = marker_palette[:total_markers]

    base = 18 + 3 * n_containers
    h = rng.randint(base, 30)
    w = rng.randint(base, 30)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    specs = []
    offset = 0
    for marker_count in marker_counts:
        colors = marker_palette[offset : offset + marker_count]
        offset += marker_count
        specs.append(_container_spec(marker_count, colors, rng))

    containers: list[dict] = []
    for spec in specs:
        patch, cavity = _build_container_patch(
            spec["opening"],
            spec["depth"],
            spec["wall_color"],
            list(spec["marker_colors"]),
            spec["style"],
            rng,
        )
        patch, cavity_bbox, _ = _rotate_container(patch, cavity, spec["side"])
        placed = False
        candidate_positions = [
            (top, left)
            for top in range(1, h - patch.shape[0])
            for left in range(1, w - patch.shape[1])
        ]
        rng.shuffle(candidate_positions)
        for top, left in candidate_positions:
            if not _container_space_ok((h, w), top, left, patch.shape, cavity_bbox, spec["side"]):
                continue
            if not _bbox_clear(occupied, top, left, patch.shape[0], patch.shape[1], pad=2):
                continue
            _paint_patch(grid, occupied, patch, top, left)
            containers.append(
                {
                    "bbox": (0, 0, patch.shape[0] - 1, patch.shape[1] - 1),
                    "cavity": cavity_bbox,
                    "side": spec["side"],
                    "markers": list(spec["marker_colors"]),
                    "opening": spec["opening"],
                    "cargo_sizes": list(spec["cargo_sizes"]),
                    "top": top,
                    "left": left,
                }
            )
            placed = True
            break
        if not placed:
            return None

    if not any(
        size <= container["opening"]
        for container in containers
        for size in container["cargo_sizes"]
    ):
        containers[0]["cargo_sizes"][0] = 1

    cargo_specs = []
    for container in containers:
        for color, size in zip(container["markers"], container["cargo_sizes"]):
            cargo_specs.append(
                {
                    "color": int(color),
                    "size": int(size),
                    "container": container,
                }
            )

    distractor_count = rng.randint(1, 3)
    for _ in range(distractor_count):
        cargo_specs.append(
            {
                "color": int(rng.choice(WALL_COLORS)),
                "size": rng.randint(1, 3),
                "container": rng.choice(containers),
            }
        )

    rng.shuffle(cargo_specs)
    for cargo in cargo_specs:
        positions = _candidate_positions((h, w), cargo["container"], cargo["size"], rng)
        placed = False
        for top, left in positions:
            if _square_clear(occupied, top, left, cargo["size"], pad=1):
                _paint_square(grid, occupied, top, left, cargo["size"], cargo["color"])
                placed = True
                break
        if not placed:
            return None

    try:
        parsed_cargos, parsed_containers = _parse_scene(grid)
        if len(parsed_containers) != len(containers):
            return None
        expected_markers = sorted(color for container in containers for color in container["markers"])
        found_markers = sorted(color for container in parsed_containers for color in container["markers"])
        if expected_markers != found_markers:
            return None
        out = main(grid)
        if np.array_equal(out, grid):
            return None
        if len(parsed_cargos) != len(cargo_specs):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
