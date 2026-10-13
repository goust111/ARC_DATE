from __future__ import annotations

from itertools import combinations
import numpy as np


BG = 0
SHAPE_COLORS = (1, 2, 3)
WALL = 5
MAX_ATTEMPTS = 400
RNG = np.random.default_rng()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _find_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in colors or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return comps


def _enclosed_bg_count(grid: np.ndarray, comp: dict) -> int:
    r0, c0, r1, c1 = comp["bbox"]
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    empty = crop == BG
    h, w = crop.shape
    outside = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))

    return int(np.count_nonzero(empty & ~outside))


def _is_u_container_patch(patch: np.ndarray) -> bool:
    h, w = patch.shape
    if h < 3 or w < 3:
        return False
    mask = np.zeros_like(patch, dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True
    mask[-1, :] = True
    return bool(np.all(patch[mask] == WALL) and np.all(patch[~mask] == BG))


def _detect_u_containers(grid: np.ndarray) -> list[dict]:
    comps = _find_components(grid, {WALL})
    containers = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if _is_u_container_patch(patch):
            containers.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "width": int(c1 - c0 - 1),
                }
            )
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each matching U-shaped container using the enclosed black area of the same-colored shape.

    Input structure:
    - Black background with several disconnected monochrome closed shapes in the upper area.
    - Each shape uses a non-grey color and encloses one or more black cells inside its own bounding box.
    - Grey U-shaped containers appear below; each has two vertical walls, one bottom bar, and a black rectangular cavity.

    Rule:
    - For every non-grey shape, count the black cells inside its bounding box that cannot reach the box border through 4-connected black paths.
    - Measure each grey container by its cavity width, namely the number of columns strictly between the two side walls.
    - Match shapes to containers whose cavity width equals that enclosed-area count, fill the entire cavity with the shape color, and then erase the original shape.

    Key insight:
    - The visible outline of a shape is only a clue source; the true matching key is the hidden enclosed black area recovered by connectivity inside that shape's own bounding box.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    shape_comps = _find_components(g, set(SHAPE_COLORS))
    containers = _detect_u_containers(g)

    area_to_shape: dict[int, dict] = {}
    for comp in shape_comps:
        area = _enclosed_bg_count(g, comp)
        area_to_shape[area] = comp

    width_to_container: dict[int, dict] = {}
    for container in containers:
        width_to_container[int(container["width"])] = container

    for area, comp in area_to_shape.items():
        container = width_to_container[area]
        r0, c0, r1, c1 = container["bbox"]
        out[r0:r1, c0 + 1 : c1] = int(comp["color"])
        for r, c in comp["coords"]:
            out[r, c] = BG

    return out


def _component_is_single(patch: np.ndarray, color: int) -> bool:
    comps = _find_components(patch, {color})
    return len(comps) == 1 and all(int(v) in (BG, color) for v in patch.flatten())


def _shape_patch_valid(patch: np.ndarray, color: int, target_area: int) -> bool:
    if not _component_is_single(patch, color):
        return False
    comp = _find_components(patch, {color})[0]
    if not (8 <= comp["area"] <= 16):
        return False
    return _enclosed_bg_count(patch, comp) == target_area


def _grow_connected_fill(h: int, w: int, count: int) -> set[tuple[int, int]]:
    if count == 0:
        return set()
    interior = {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)}
    boundary_adj = [p for p in interior if p[0] in (1, h - 2) or p[1] in (1, w - 2)]
    seed = boundary_adj[int(RNG.integers(0, len(boundary_adj)))]
    chosen = {seed}
    while len(chosen) < count:
        frontier = []
        for r, c in chosen:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in interior and (nr, nc) not in chosen:
                    frontier.append((nr, nc))
        if not frontier:
            return set()
        nr, nc = frontier[int(RNG.integers(0, len(frontier)))]
        chosen.add((nr, nc))
    return chosen


def _corner_mutation_variants(patch: np.ndarray, color: int, target_area: int) -> list[np.ndarray]:
    h, w = patch.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    variants = [patch]
    for k in (1, 2):
        for removed in combinations(corners, k):
            variant = patch.copy()
            for r, c in removed:
                variant[r, c] = BG
            if _shape_patch_valid(variant, color, target_area):
                variants.append(variant)
    return variants


def _build_shape_patch(target_area: int, color: int) -> tuple[np.ndarray, bool]:
    candidates = []
    for h in range(3, 5):
        for w in range(3, 6):
            interior = (h - 2) * (w - 2)
            extra = interior - target_area
            pixels = 2 * h + 2 * w - 4 + extra
            if extra >= 0 and 8 <= pixels <= 16:
                candidates.append((h, w, extra))
    assert candidates

    for _ in range(120):
        h, w, extra = candidates[int(RNG.integers(0, len(candidates)))]
        patch = np.zeros((h, w), dtype=int)
        patch[0, :] = color
        patch[-1, :] = color
        patch[:, 0] = color
        patch[:, -1] = color
        for r, c in _grow_connected_fill(h, w, extra):
            patch[r, c] = color

        variants = _corner_mutation_variants(patch, color, target_area)
        patch = variants[int(RNG.integers(0, len(variants)))]
        patch = np.rot90(patch, int(RNG.integers(0, 4)))
        if int(RNG.integers(0, 2)):
            patch = np.fliplr(patch)

        if not _shape_patch_valid(patch, color, target_area):
            continue

        comp = _find_components(patch, {color})[0]
        r0, c0, r1, c1 = comp["bbox"]
        crop = patch[r0 : r1 + 1, c0 : c1 + 1]
        border_bg = bool(
            np.any(crop[0, :] == BG)
            or np.any(crop[-1, :] == BG)
            or np.any(crop[:, 0] == BG)
            or np.any(crop[:, -1] == BG)
        )
        interior_area = max(0, crop.shape[0] - 2) * max(0, crop.shape[1] - 2)
        irregular = border_bg or target_area < interior_area
        return patch, irregular
    raise RuntimeError("failed to build shape patch")


def _u_patch(width: int, interior_h: int) -> np.ndarray:
    patch = np.zeros((interior_h + 1, width + 2), dtype=int)
    patch[:, 0] = WALL
    patch[:, -1] = WALL
    patch[-1, :] = WALL
    return patch


def _horizontal_positions(widths: list[int], total_w: int, min_gap: int) -> list[int] | None:
    need = sum(widths) + min_gap * (len(widths) - 1)
    if need > total_w:
        return None
    slack = total_w - need
    parts = RNG.multinomial(slack, np.ones(len(widths) + 1) / (len(widths) + 1))
    positions: list[int] = []
    c = int(parts[0])
    for i, width in enumerate(widths):
        positions.append(c)
        c += width
        if i < len(widths) - 1:
            c += min_gap + int(parts[i + 1])
    return positions


def _expected_output(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    shapes = _find_components(grid, set(SHAPE_COLORS))
    containers = _detect_u_containers(grid)
    width_to_container = {int(container["width"]): container for container in containers}
    for comp in shapes:
        area = _enclosed_bg_count(grid, comp)
        container = width_to_container[area]
        r0, c0, r1, c1 = container["bbox"]
        out[r0:r1, c0 + 1 : c1] = int(comp["color"])
        for r, c in comp["coords"]:
            out[r, c] = BG
    return out


def _validate_sample(grid: np.ndarray) -> bool:
    shapes = _find_components(grid, set(SHAPE_COLORS))
    containers = _detect_u_containers(grid)
    if len(shapes) != 3 or len(containers) != 3:
        return False

    shape_areas = [_enclosed_bg_count(grid, comp) for comp in shapes]
    if sorted(shape_areas) != sorted(set(shape_areas)):
        return False
    if any(area < 1 or area > 5 for area in shape_areas):
        return False

    widths = [int(container["width"]) for container in containers]
    if sorted(widths) != sorted(set(widths)):
        return False
    if sorted(widths) != sorted(shape_areas):
        return False

    irregular_flags = []
    shape_pixels = []
    for comp, area in zip(shapes, shape_areas):
        r0, c0, r1, c1 = comp["bbox"]
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        border_bg = bool(
            np.any(crop[0, :] == BG)
            or np.any(crop[-1, :] == BG)
            or np.any(crop[:, 0] == BG)
            or np.any(crop[:, -1] == BG)
        )
        interior_area = max(0, crop.shape[0] - 2) * max(0, crop.shape[1] - 2)
        irregular_flags.append(border_bg or area < interior_area)
        shape_pixels.append(int(comp["area"]))
    if not any(irregular_flags):
        return False

    by_area = sorted(zip(shape_areas, shape_pixels))
    if all(by_area[i][1] < by_area[i + 1][1] for i in range(len(by_area) - 1)):
        return False

    shapes_lr = [area for area, _c in sorted(((shape_areas[i], shapes[i]["bbox"][1]) for i in range(3)), key=lambda x: x[1])]
    widths_lr = [int(container["width"]) for container in sorted(containers, key=lambda d: d["bbox"][1])]
    if shapes_lr == widths_lr:
        return False

    expected = _expected_output(grid)
    actual = main(grid)
    return np.array_equal(actual, expected)


def _generate_attempt() -> np.ndarray | None:
    areas = list(map(int, RNG.choice(np.arange(1, 6), size=3, replace=False)))
    colors = list(SHAPE_COLORS)
    RNG.shuffle(colors)

    shape_infos = []
    irregular_count = 0
    for area, color in zip(areas, colors):
        patch, irregular = _build_shape_patch(area, color)
        irregular_count += int(irregular)
        shape_infos.append(
            {
                "area": area,
                "color": color,
                "patch": patch,
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
                "pixels": int(np.count_nonzero(patch)),
            }
        )
    if irregular_count == 0:
        return None

    by_area = sorted((info["area"], info["pixels"]) for info in shape_infos)
    if all(by_area[i][1] < by_area[i + 1][1] for i in range(len(by_area) - 1)):
        return None

    bottom_widths = [info["area"] + 2 for info in shape_infos]
    max_need = max(sum(bottom_widths) + 4, sum(info["w"] for info in shape_infos) + 4)
    if max_need > 30:
        return None
    total_w = int(RNG.integers(max(18, max_need), min(30, max_need + 6) + 1))
    total_h = int(RNG.integers(14, 20))

    shape_order = list(range(3))
    RNG.shuffle(shape_order)
    shape_widths = [shape_infos[i]["w"] for i in shape_order]
    shape_cols = _horizontal_positions(shape_widths, total_w, min_gap=2)
    if shape_cols is None:
        return None

    container_order = list(range(3))
    for _ in range(20):
        RNG.shuffle(container_order)
        shape_areas_lr = [shape_infos[i]["area"] for i in shape_order]
        container_widths_lr = [shape_infos[i]["area"] for i in container_order]
        if shape_areas_lr != container_widths_lr:
            break
    else:
        return None

    container_widths = [shape_infos[i]["area"] + 2 for i in container_order]
    container_cols = _horizontal_positions(container_widths, total_w, min_gap=2)
    if container_cols is None:
        return None

    grid = np.zeros((total_h, total_w), dtype=int)

    for idx, c0 in zip(shape_order, shape_cols):
        info = shape_infos[idx]
        top_band = 7 - info["h"]
        if top_band < 0:
            return None
        r0 = int(RNG.integers(0, top_band + 1))
        patch = info["patch"]
        mask = patch != BG
        if np.any(grid[r0 : r0 + info["h"], c0 : c0 + info["w"]][mask] != BG):
            return None
        grid[r0 : r0 + info["h"], c0 : c0 + info["w"]][mask] = patch[mask]

    for idx, c0 in zip(container_order, container_cols):
        width = shape_infos[idx]["area"]
        interior_h = int(RNG.integers(2, 5))
        patch = _u_patch(width, interior_h)
        ph = patch.shape[0]
        min_top = max(8, total_h - 6)
        max_top = total_h - ph
        if min_top > max_top:
            return None
        r0 = int(RNG.integers(min_top, max_top + 1))
        if np.any((grid[r0 : r0 + ph, c0 : c0 + patch.shape[1]] != BG) & (patch != BG)):
            return None
        mask = patch != BG
        grid[r0 : r0 + ph, c0 : c0 + patch.shape[1]][mask] = patch[mask]

    if max(grid.shape) > 30:
        return None
    if not _validate_sample(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to construct a valid sample")
