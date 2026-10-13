from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]

BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500


def neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def find_components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def component_colors(grid: np.ndarray, coords: list[Coord]) -> set[int]:
    return {int(grid[r, c]) for r, c in coords}


def path_degrees(coords: list[Coord]) -> dict[Coord, int]:
    cells = set(coords)
    return {
        cell: sum((nbr in cells) for nbr in neighbors4(*cell))
        for cell in cells
    }


def ordered_path(coords: list[Coord]) -> tuple[list[Coord], list[Coord]]:
    cells = set(coords)
    assert len(cells) == len(coords), "path cells must be unique"
    degrees = path_degrees(coords)
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must not branch"

    start = endpoints[0]
    prev = None
    cur = start
    path: list[Coord] = []
    while True:
        path.append(cur)
        nxt = [nbr for nbr in neighbors4(*cur) if nbr in cells and nbr != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path traversal must be unique"
        prev, cur = cur, nxt[0]

    assert len(path) == len(coords), "path traversal must visit every cell"
    return path, endpoints


def first_occurrence_order(values: list[int]) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            order.append(int(value))
    return order


def mapping_from_sequence(values: list[int]) -> dict[int, int]:
    order = first_occurrence_order(values)
    return {order[i]: order[-1 - i] for i in range(len(order))}


def transform_from_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g != bg)
    assert components, "expected foreground components"

    path_candidates: list[list[Coord]] = []
    for comp in components:
        colors = component_colors(g, comp)
        if len(colors) <= 1:
            continue
        try:
            ordered_path(comp)
        except AssertionError:
            continue
        path_candidates.append(comp)

    assert len(path_candidates) == 1, "expected exactly one multicolor non-branching path"
    path_coords = path_candidates[0]
    path, _endpoints = ordered_path(path_coords)
    path_values = [int(g[r, c]) for r, c in path]
    mapping = mapping_from_sequence(path_values)
    path_colors = set(mapping)

    regions = [comp for comp in components if comp is not path_coords]
    assert len(regions) == len(path_colors), "expected one region per path color"

    region_colors = set()
    for region in regions:
        colors = component_colors(g, region)
        assert len(colors) == 1, "regions must be monochrome"
        color = next(iter(colors))
        assert color in path_colors, "region color must come from the path palette"
        region_colors.add(color)
    assert region_colors == path_colors, "region colors must match the path colors exactly"

    out = g.copy()
    for region in regions:
        color = int(g[region[0][0], region[0][1]])
        mapped = mapping[color]
        for r, c in region:
            out[r, c] = mapped
    for r, c in path:
        out[r, c] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a multicolor path to build a reversed first-seen color mapping, recolor the matching remote regions, then erase the path.

    Input structure:
    - Black background contains one 4-connected one-pixel-wide multicolor path with exactly two endpoints and no branches.
    - The path uses 3 or 4 distinct colors, and each of those colors also appears on exactly one separate monochrome 4-connected region.
    - The monochrome regions are disconnected from each other and from the path.

    Rule:
    - Start from the endpoint with smaller row index, breaking ties by smaller column index, and traverse the path to the other endpoint.
    - Record the order in which new colors first appear on that walk, reverse that first-appearance list, and pair each original color with the color in the same reversed position.
    - Recolor every monochrome region by that color mapping, keep each region's shape fixed, and finally erase the whole path to background.

    Key insight:
    - The path is an instruction carrier rather than the transformed object: its traversal order defines a color permutation that is applied somewhere else before the path disappears.
    """
    return transform_from_grid(grid)


def crop_patch(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def is_simple_path_mask(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if len(coords) < 2:
        return False
    degrees = path_degrees(coords)
    return (
        all(degree in (1, 2) for degree in degrees.values())
        and sum(degree == 1 for degree in degrees.values()) == 2
    )


def random_blob_patch(color: int) -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        if h * w < 5:
            continue
        area = random.randint(5, min(15, h * w))
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = sorted(
                {
                    nbr
                    for r, c in cells
                    for nbr in neighbors4(r, c)
                    if 0 <= nbr[0] < h and 0 <= nbr[1] < w and nbr not in cells
                }
            )
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if mask.shape[0] == 1 or mask.shape[1] == 1:
            continue
        if is_simple_path_mask(mask):
            continue
        patch = np.zeros((h, w), dtype=int)
        patch[mask] = color
        return crop_patch(patch)
    return None


def patch_coords(patch: np.ndarray) -> list[Coord]:
    return [tuple(map(int, rc)) for rc in np.argwhere(patch != BG)]


def free_locations(canvas: np.ndarray, patch: np.ndarray) -> list[Coord]:
    h, w = canvas.shape
    ph, pw = patch.shape
    coords = patch_coords(patch)
    options: list[Coord] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            ok = True
            for pr, pc in coords:
                rr, cc = r0 + pr, c0 + pc
                if canvas[rr, cc] != BG:
                    ok = False
                    break
                for nr in range(rr - 1, rr + 2):
                    for nc in range(cc - 1, cc + 2):
                        if not in_bounds(canvas.shape, nr, nc):
                            continue
                        if canvas[nr, nc] != BG:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                options.append((r0, c0))
    return options


def paste_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    dst[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]


def random_path(zone_h: int, zone_w: int, length: int) -> list[Coord] | None:
    starts = [(r, c) for r in range(zone_h) for c in range(zone_w)]
    random.shuffle(starts)

    def dfs(path: list[Coord], used: set[Coord]) -> list[Coord] | None:
        if len(path) == length:
            turns = 0
            for i in range(1, len(path) - 1):
                dr1 = path[i][0] - path[i - 1][0]
                dc1 = path[i][1] - path[i - 1][1]
                dr2 = path[i + 1][0] - path[i][0]
                dc2 = path[i + 1][1] - path[i][1]
                turns += (dr1, dc1) != (dr2, dc2)
            return path if turns >= 1 else None

        cr, cc = path[-1]
        candidates = []
        for nr, nc in neighbors4(cr, cc):
            if not (0 <= nr < zone_h and 0 <= nc < zone_w):
                continue
            if (nr, nc) in used:
                continue
            if sum((nbr in used) for nbr in neighbors4(nr, nc)) != 1:
                continue
            candidates.append((nr, nc))
        random.shuffle(candidates)
        for nxt in candidates:
            used.add(nxt)
            path.append(nxt)
            found = dfs(path, used)
            if found is not None:
                return found
            path.pop()
            used.remove(nxt)
        return None

    for start in starts:
        found = dfs([start], {start})
        if found is not None:
            return found
    return None


def sequence_reverse_mapping_differs(values: list[int]) -> bool:
    return mapping_from_sequence(values) != mapping_from_sequence(list(reversed(values)))


def random_path_sequence(colors: list[int], length: int) -> list[int] | None:
    n = len(colors)
    for _ in range(200):
        order = random.sample(colors, n)
        last_new = random.randint(n, length - 1)
        first_positions = [0] + sorted(random.sample(range(1, last_new), n - 2)) + [last_new]
        seq = [None] * length
        for i, pos in enumerate(first_positions):
            seq[pos] = order[i]
        for i, pos in enumerate(first_positions):
            end = first_positions[i + 1] if i + 1 < n else length
            allowed = order[: i + 1]
            for j in range(pos + 1, end):
                seq[j] = random.choice(allowed)
        if seq[-1] == order[-1]:
            seq[-1] = random.choice(order[:-1])
        repeated_before_last = any(
            seq[i] in seq[:i] for i in range(1, first_positions[-1])
        )
        if not repeated_before_last:
            continue
        if mapping_from_sequence(seq) == {c: c for c in order}:
            continue
        if not sequence_reverse_mapping_differs(seq):
            continue
        return [int(x) for x in seq]
    return None


def zone_spec(h: int, w: int, layout: str) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if layout == "tb":
        path_zone = (1, 1, h // 2 - 2, w - 2)
        region_zone = (h // 2 + 1, 1, h - 2, w - 2)
    else:
        path_zone = (1, 1, h - 2, w // 2 - 2)
        region_zone = (1, w // 2 + 1, h - 2, w - 2)
    return path_zone, region_zone


def zone_shape(zone: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = zone
    return r1 - r0 + 1, c1 - c0 + 1


def translate(coords: list[Coord], offset_r: int, offset_c: int) -> list[Coord]:
    return [(r + offset_r, c + offset_c) for r, c in coords]


def apply_mapping_to_regions(
    grid: np.ndarray,
    path: list[Coord],
    region_infos: list[dict],
    mapping: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    for info in region_infos:
        mapped = mapping[info["color"]]
        for r, c in info["coords"]:
            out[r, c] = mapped
    for r, c in path:
        out[r, c] = BG
    return out


def validate_instance(
    grid: np.ndarray,
    path: list[Coord],
    sequence: list[int],
    region_infos: list[dict],
) -> bool:
    try:
        assert grid.ndim == 2
        assert max(grid.shape) <= 30
        assert 3 <= len(set(sequence)) <= 4
        assert 6 <= len(path) <= 12
        assert len(path) == len(set(path))

        path_cells = set(path)
        degrees = path_degrees(path)
        assert all(degree in (1, 2) for degree in degrees.values())
        endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
        assert len(endpoints) == 2
        ordered, ordered_endpoints = ordered_path(path)
        assert ordered_endpoints == endpoints
        assert ordered == path or ordered == list(reversed(path))

        smallest_endpoint = min(endpoints)
        assert ordered[0] == smallest_endpoint

        for (r, c), color in zip(ordered, sequence):
            assert int(grid[r, c]) == color

        path_palette = set(sequence)
        assert len(region_infos) == len(path_palette)
        assert {info["color"] for info in region_infos} == path_palette
        for info in region_infos:
            assert 5 <= len(info["coords"]) <= 15
            assert len({int(grid[r, c]) for r, c in info["coords"]}) == 1
            assert not (path_cells & set(info["coords"]))

        mapping = mapping_from_sequence(sequence)
        reverse_mapping = mapping_from_sequence(list(reversed(sequence)))
        assert mapping != reverse_mapping
        assert any(mapping[color] != color for color in mapping)

        expected = apply_mapping_to_regions(grid, ordered, region_infos, mapping)
        wrong = apply_mapping_to_regions(grid, ordered, region_infos, reverse_mapping)
        assert not np.array_equal(expected, wrong)
        actual = main(grid)
        assert np.array_equal(actual, expected)
        assert np.any(expected != grid)
        return True
    except AssertionError:
        return False


def generate_attempt() -> np.ndarray | None:
    h = random.randint(14, 16)
    w = random.randint(14, 16)
    layout = random.choice(["tb", "lr"])
    path_zone, region_zone = zone_spec(h, w, layout)
    path_h, path_w = zone_shape(path_zone)
    region_h, region_w = zone_shape(region_zone)
    if min(path_h, path_w, region_h, region_w) < 5:
        return None

    n_colors = random.choice([3, 4])
    colors = random.sample(PALETTE, n_colors)
    path_len = random.randint(6, 12)
    rel_path_unordered = random_path(path_h, path_w, path_len)
    if rel_path_unordered is None:
        return None
    rel_path, _endpoints = ordered_path(rel_path_unordered)
    sequence = random_path_sequence(colors, path_len)
    if sequence is None:
        return None

    region_canvas = np.zeros((region_h, region_w), dtype=int)
    region_infos: list[dict] = []
    for color in colors:
        patch = random_blob_patch(color)
        if patch is None:
            return None
        options = free_locations(region_canvas, patch)
        if not options:
            return None
        r0, c0 = random.choice(options)
        paste_patch(region_canvas, patch, r0, c0)
        coords = translate(patch_coords(patch), region_zone[0] + r0, region_zone[1] + c0)
        region_infos.append({"color": color, "coords": coords})

    grid = np.zeros((h, w), dtype=int)
    paste_patch(grid, region_canvas, region_zone[0], region_zone[1])

    abs_path = translate(rel_path, path_zone[0], path_zone[1])
    for (r, c), color in zip(abs_path, sequence):
        if grid[r, c] != BG:
            return None
        grid[r, c] = color

    if not validate_instance(grid, abs_path, sequence, region_infos):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
