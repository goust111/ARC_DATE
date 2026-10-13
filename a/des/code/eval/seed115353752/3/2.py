import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _draw_border(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _candidate_boxes(blocked: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    grid_h, grid_w = map(int, blocked.shape)
    out: list[tuple[int, int]] = []
    for r0 in range(grid_h - h + 1):
        r_lo = max(0, r0 - 1)
        r_hi = min(grid_h, r0 + h + 1)
        for c0 in range(grid_w - w + 1):
            c_lo = max(0, c0 - 1)
            c_hi = min(grid_w, c0 + w + 1)
            if not bool(blocked[r_lo:r_hi, c_lo:c_hi].any()):
                out.append((int(r0), int(c0)))
    return out


def _block_box(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    grid_h, grid_w = map(int, blocked.shape)
    blocked[max(0, r0 - 1) : min(grid_h, r0 + h + 1), max(0, c0 - 1) : min(grid_w, c0 + w + 1)] = True


def _transform_point(pt: tuple[int, int], turns: int, flip: bool) -> tuple[int, int]:
    r, c = map(int, pt)
    for _ in range(turns % 4):
        r, c = c, -r
    if flip:
        c = -c
    return int(r), int(c)


def _random_connected_shape(rng: random.Random, size: int, radius: int = 3) -> set[tuple[int, int]]:
    for _ in range(100):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if (nr, nc) in cells:
                        continue
                    if abs(nr) > radius or abs(nc) > radius:
                        continue
                    frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) == size:
            return cells
    raise RuntimeError("failed to build connected fragment shape")


def _make_fragment_spec(container_color: int, body_color: int, rng: random.Random) -> dict:
    size = rng.randint(2, 7)
    cells = _random_connected_shape(rng, size)
    marker = rng.choice(tuple(cells))
    turns = rng.randrange(4)
    flip = bool(rng.randrange(2))

    rel_cells = [_transform_point((r - marker[0], c - marker[1]), turns, flip) for r, c in cells]
    rows = [r for r, _ in rel_cells]
    cols = [c for _, c in rel_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    marker_local = (-min_r, -min_c)
    fragment_pixels = []
    for rr, cc in rel_cells:
        local_r = int(rr - min_r)
        local_c = int(cc - min_c)
        color = int(container_color) if (rr, cc) == (0, 0) else int(body_color)
        fragment_pixels.append((local_r, local_c, color))
    fragment_pixels.sort()

    frag_h = int(max_r - min_r + 1)
    frag_w = int(max_c - min_c + 1)
    slack_top = rng.randint(0, 3)
    slack_bottom = rng.randint(0, 3)
    slack_left = rng.randint(0, 3)
    slack_right = rng.randint(0, 3)
    interior_h = int(frag_h + slack_top + slack_bottom)
    interior_w = int(frag_w + slack_left + slack_right)

    return {
        "container_color": int(container_color),
        "body_color": int(body_color),
        "fragment_pixels": fragment_pixels,
        "fragment_h": int(frag_h),
        "fragment_w": int(frag_w),
        "marker_local": (int(marker_local[0]), int(marker_local[1])),
        "anchor_local": (int(marker_local[0] + slack_top), int(marker_local[1] + slack_left)),
        "interior_h": int(interior_h),
        "interior_w": int(interior_w),
        "outer_h": int(interior_h + 2),
        "outer_w": int(interior_w + 2),
    }


def _detect_containers(grid: np.ndarray) -> dict[int, dict]:
    g = np.asarray(grid, dtype=int)
    containers: dict[int, dict] = {}
    for color in (int(c) for c in np.unique(g) if int(c) != BG):
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            box_h = int(r1 - r0 + 1)
            box_w = int(c1 - c0 + 1)
            if box_h < 3 or box_w < 3:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros_like(sub, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not bool((sub[border] == color).all()):
                continue
            interior = sub[1:-1, 1:-1]
            if interior.size == 0:
                continue
            if not bool(((interior == BG) | (interior == color)).all()):
                continue
            anchor_offsets = np.argwhere(interior == color)
            if int(anchor_offsets.shape[0]) != 1:
                continue
            border_count = int(2 * (box_h + box_w) - 4)
            same_color_count = int(np.count_nonzero(sub == color))
            comp_area = int(comp.shape[0])
            if same_color_count != border_count + 1:
                continue
            if comp_area not in (border_count, border_count + 1):
                continue
            anchor_r = int(r0 + 1 + anchor_offsets[0, 0])
            anchor_c = int(c0 + 1 + anchor_offsets[0, 1])
            if color in containers:
                raise AssertionError("container colors must be unique")
            containers[color] = {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "anchor": (int(anchor_r), int(anchor_c)),
            }
    return containers


def _container_occupied(shape: tuple[int, int], containers: dict[int, dict]) -> np.ndarray:
    h, w = map(int, shape)
    occupied = np.zeros((h, w), dtype=bool)
    for info in containers.values():
        r0, c0, r1, c1 = info["bbox"]
        occupied[r0, c0 : c1 + 1] = True
        occupied[r1, c0 : c1 + 1] = True
        occupied[r0 : r1 + 1, c0] = True
        occupied[r0 : r1 + 1, c1] = True
        ar, ac = info["anchor"]
        occupied[ar, ac] = True
    return occupied


def _extract_fragments(grid: np.ndarray, containers: dict[int, dict]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    occupied = _container_occupied(g.shape, containers)
    fragments: list[dict] = []
    used_match_colors: set[int] = set()
    for comp in _components((g != BG) & (~occupied)):
        colors = Counter(int(g[r, c]) for r, c in comp)
        assert len(colors) == 2, "each fragment must have one marker color plus one body color"

        marker_color = None
        body_color = None
        for color, count in colors.items():
            if count == 1 and color in containers:
                marker_color = int(color)
            else:
                body_color = int(color)

        assert marker_color is not None, "fragment marker must match a container color"
        assert body_color is not None, "fragment body color missing"
        assert colors[marker_color] == 1, "fragment marker must be unique"
        assert colors[body_color] == int(comp.shape[0] - 1), "fragment body must be monochrome"
        assert marker_color not in used_match_colors, "at most one fragment may target each container"
        used_match_colors.add(marker_color)

        marker_idx = next(i for i, (r, c) in enumerate(comp) if int(g[r, c]) == marker_color)
        marker_r = int(comp[marker_idx, 0])
        marker_c = int(comp[marker_idx, 1])
        pixels = []
        for r, c in comp:
            pixels.append((int(r - marker_r), int(c - marker_c), int(g[r, c])))
        fragments.append({"match_color": int(marker_color), "pixels": pixels})
    return fragments


def _paint_fragment(out: np.ndarray, container: dict, fragment: dict) -> None:
    r0, c0, r1, c1 = container["bbox"]
    anchor_r, anchor_c = container["anchor"]
    for dr, dc, color in fragment["pixels"]:
        rr = int(anchor_r + dr)
        cc = int(anchor_c + dc)
        assert r0 < rr < r1 and c0 < cc < c1, "translated fragment must stay inside the target container"
        out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each outside fragment into the unique container with the same marker color.

    Containers are rectangular 1-cell-thick colored frames with one matching-color
    anchor pixel inside. Each outside fragment is a 4-connected multicolor object
    with exactly one marker pixel whose color matches one container border. The
    fragment is translated so that its marker lands on the container anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    containers = _detect_containers(g)
    assert containers, "expected at least one container"
    fragments = _extract_fragments(g, containers)
    assert fragments, "expected at least one fragment"

    out = np.zeros_like(g)
    for info in containers.values():
        r0, c0, r1, c1 = info["bbox"]
        color = int(info["color"])
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        ar, ac = info["anchor"]
        out[ar, ac] = color

    for fragment in fragments:
        _paint_fragment(out, containers[int(fragment["match_color"])], fragment)

    return out


GEN_RETRIES = 400
GRID_MIN = 20
GRID_MAX = 32
PAIR_COUNTS = (2, 3, 4)


def _expected_output(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec in specs:
        r0, c0 = spec["container_top_left"]
        _draw_border(out, r0, c0, spec["outer_h"], spec["outer_w"], spec["container_color"])
        ar, ac = spec["anchor_abs"]
        out[ar, ac] = spec["container_color"]

    for spec in specs:
        r0, c0 = spec["container_top_left"]
        r1 = int(r0 + spec["outer_h"] - 1)
        c1 = int(c0 + spec["outer_w"] - 1)
        anchor_r, anchor_c = spec["anchor_abs"]
        marker_r, marker_c = spec["marker_local"]
        for frag_r, frag_c, color in spec["fragment_pixels"]:
            rr = int(anchor_r + frag_r - marker_r)
            cc = int(anchor_c + frag_c - marker_c)
            if not (r0 < rr < r1 and c0 < cc < c1):
                raise AssertionError("generated fragment escapes its container")
            out[rr, cc] = int(color)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice(PAIR_COUNTS)
    colors = list(PALETTE)
    rng.shuffle(colors)
    container_colors = colors[:pair_count]
    body_colors = colors[pair_count : 2 * pair_count]

    specs = [_make_fragment_spec(container_colors[i], body_colors[i], rng) for i in range(pair_count)]

    total_area = sum(spec["outer_h"] * spec["outer_w"] + spec["fragment_h"] * spec["fragment_w"] for spec in specs)
    min_side = max(GRID_MIN, int(total_area**0.5) + 6, 14 + 2 * pair_count)
    if min_side > GRID_MAX:
        return None

    grid_h = rng.randint(min_side, GRID_MAX)
    grid_w = rng.randint(min_side, GRID_MAX)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    for idx in sorted(range(pair_count), key=lambda i: specs[i]["outer_h"] * specs[i]["outer_w"], reverse=True):
        spec = specs[idx]
        options = _candidate_boxes(blocked, spec["outer_h"], spec["outer_w"])
        if not options:
            return None
        r0, c0 = rng.choice(options)
        spec["container_top_left"] = (int(r0), int(c0))
        _draw_border(grid, r0, c0, spec["outer_h"], spec["outer_w"], spec["container_color"])
        anchor_r = int(r0 + 1 + spec["anchor_local"][0])
        anchor_c = int(c0 + 1 + spec["anchor_local"][1])
        spec["anchor_abs"] = (anchor_r, anchor_c)
        grid[anchor_r, anchor_c] = spec["container_color"]
        _block_box(blocked, r0, c0, spec["outer_h"], spec["outer_w"])

    for idx in sorted(range(pair_count), key=lambda i: specs[i]["fragment_h"] * specs[i]["fragment_w"], reverse=True):
        spec = specs[idx]
        options = _candidate_boxes(blocked, spec["fragment_h"], spec["fragment_w"])
        if not options:
            return None
        r0, c0 = rng.choice(options)
        spec["fragment_top_left"] = (int(r0), int(c0))
        for frag_r, frag_c, color in spec["fragment_pixels"]:
            grid[int(r0 + frag_r), int(c0 + frag_c)] = int(color)
        _block_box(blocked, r0, c0, spec["fragment_h"], spec["fragment_w"])

    expected = _expected_output(grid.shape, specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
