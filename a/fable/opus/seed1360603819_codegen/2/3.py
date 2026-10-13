import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
TARGET_COLORS = (RED, GREEN, YELLOW)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _count_holes(mask_crop: np.ndarray) -> int:
    holes = 0
    empty = ~np.asarray(mask_crop, dtype=bool)
    h, w = empty.shape
    for comp in _components(empty):
        touches_edge = np.any(
            (comp[:, 0] == 0)
            | (comp[:, 0] == h - 1)
            | (comp[:, 1] == 0)
            | (comp[:, 1] == w - 1)
        )
        if not touches_edge:
            holes += 1
    return holes


def _legend_mapping(grid: np.ndarray) -> tuple[int, dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    legend_row = None
    for r in range(g.shape[0]):
        if np.any(g[r] != BLACK):
            legend_row = r
            break
    assert legend_row is not None, "missing legend row"

    cols = [int(c) for c in np.flatnonzero(g[legend_row] != BLACK)]
    assert len(cols) == 3, "legend row must contain exactly three key pixels"
    colors = [int(g[legend_row, c]) for c in cols]
    assert set(colors) == set(TARGET_COLORS), "legend colors must be red/green/yellow"
    return int(legend_row), {index + 1: color for index, color in enumerate(colors)}


def _anchor_positions(grid: np.ndarray, start_row: int) -> dict[int, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    anchors: dict[int, tuple[int, int]] = {}
    for color in TARGET_COLORS:
        pts = np.argwhere(g[start_row:] == color)
        assert pts.shape[0] <= 1, "each anchor color may appear at most once below the legend"
        if pts.shape[0] == 1:
            r, c = map(int, pts[0])
            anchors[int(color)] = (int(start_row + r), c)
    return anchors


def _grey_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    objs = []
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        mask_crop = (g[r0 : r1 + 1, c0 : c1 + 1] == GREY)
        holes = _count_holes(mask_crop)
        rel = comp - np.array([r0, c0], dtype=int)
        objs.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (h, w),
                "coords": comp.copy(),
                "rel_coords": rel,
                "center": (r0 + h // 2, c0 + w // 2),
                "holes": int(holes),
            }
        )
    objs.sort(key=lambda obj: obj["bbox"])
    return objs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each grey block's holes, use the top legend to map hole-count -> color,
    then move the whole block so its center lands on the matching anchor pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    legend_row, legend = _legend_mapping(g)
    anchors = _anchor_positions(g, legend_row + 1)
    objects = _grey_objects(g)

    out = g.copy()
    out[out == GREY] = BLACK

    for obj in objects:
        assert obj["holes"] in legend, "grey blocks must have 1-3 holes"
        target_color = legend[obj["holes"]]
        anchor = anchors.get(target_color)
        if anchor is None:
            dest_coords = obj["coords"]
        else:
            dr = int(anchor[0] - obj["center"][0])
            dc = int(anchor[1] - obj["center"][1])
            dest_coords = obj["coords"] + np.array([dr, dc], dtype=int)
            assert np.all(dest_coords[:, 0] >= 0) and np.all(dest_coords[:, 0] < g.shape[0])
            assert np.all(dest_coords[:, 1] >= 0) and np.all(dest_coords[:, 1] < g.shape[1])
        out[dest_coords[:, 0], dest_coords[:, 1]] = GREY

    return out


GEN_HEIGHT_RANGE = (20, 28)
GEN_WIDTH_RANGE = (22, 30)
GEN_SIDE_MARGIN = 1
GEN_TOP_MARGIN = 3
GEN_RECT_PAD = 1
GEN_ATTEMPTS = 400
GEN_SIZES = ((5, 5), (5, 7), (7, 5), (7, 7))


def _pick_holes(height: int, width: int, count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    center = (height // 2, width // 2)
    cells = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if (r, c) != center
    ]
    for _ in range(24):
        rng.shuffle(cells)
        chosen: list[tuple[int, int]] = []

        def backtrack(index: int) -> bool:
            if len(chosen) == count:
                return True
            if len(cells) - index < count - len(chosen):
                return False
            for i in range(index, len(cells)):
                cell = cells[i]
                if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) > 1 for rr, cc in chosen):
                    chosen.append(cell)
                    if backtrack(i + 1):
                        return True
                    chosen.pop()
            return False

        if backtrack(0):
            return chosen.copy()
    return None


def _object_spec(hole_count: int, rng: random.Random) -> dict:
    sizes = list(GEN_SIZES)
    rng.shuffle(sizes)
    for height, width in sizes:
        holes = _pick_holes(height, width, hole_count, rng)
        if holes is not None:
            return {
                "hole_count": int(hole_count),
                "height": int(height),
                "width": int(width),
                "holes": holes,
            }
    raise RuntimeError("failed to build object spec")


def _can_place(used: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    h, w = used.shape
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    if r0 < GEN_TOP_MARGIN or c0 < GEN_SIDE_MARGIN:
        return False
    if r1 > h - 1 - GEN_SIDE_MARGIN or c1 > w - 1 - GEN_SIDE_MARGIN:
        return False
    rr0 = max(0, r0 - GEN_RECT_PAD)
    cc0 = max(0, c0 - GEN_RECT_PAD)
    rr1 = min(h - 1, r1 + GEN_RECT_PAD)
    cc1 = min(w - 1, c1 + GEN_RECT_PAD)
    return not bool(used[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_used(used: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    h, w = used.shape
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    rr0 = max(0, r0 - GEN_RECT_PAD)
    cc0 = max(0, c0 - GEN_RECT_PAD)
    rr1 = min(h - 1, r1 + GEN_RECT_PAD)
    cc1 = min(w - 1, c1 + GEN_RECT_PAD)
    used[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _random_rect(used: np.ndarray, height: int, width: int, rng: random.Random) -> tuple[int, int] | None:
    h, w = used.shape
    candidates = []
    for r0 in range(GEN_TOP_MARGIN, h - height - GEN_SIDE_MARGIN + 2):
        for c0 in range(GEN_SIDE_MARGIN, w - width - GEN_SIDE_MARGIN + 2):
            if _can_place(used, r0, c0, height, width):
                candidates.append((r0, c0))
    if not candidates:
        return None
    return rng.choice(candidates)


def _paint_object(grid: np.ndarray, spec: dict, r0: int, c0: int) -> None:
    height = spec["height"]
    width = spec["width"]
    grid[r0 : r0 + height, c0 : c0 + width] = GREY
    for hr, hc in spec["holes"]:
        grid[r0 + hr, c0 + hc] = BLACK


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    counts_pool = [1, 2, 3]
    scenario = rng.choice(("all_move", "none_move", "mixed"))
    if scenario == "mixed":
        n_objects = rng.randint(2, 3)
    else:
        n_objects = rng.randint(1, 3)

    object_counts = rng.sample(counts_pool, n_objects)
    specs = [_object_spec(k, rng) for k in object_counts]

    grid_h = rng.randint(*GEN_HEIGHT_RANGE)
    grid_w = rng.randint(*GEN_WIDTH_RANGE)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    legend_colors = list(TARGET_COLORS)
    rng.shuffle(legend_colors)
    legend = {1: legend_colors[0], 2: legend_colors[1], 3: legend_colors[2]}
    legend_start = rng.randint(1, grid_w - 6)
    for i, color in enumerate(legend_colors):
        grid[0, legend_start + 2 * i] = color

    move_counts: set[int]
    if scenario == "all_move":
        move_counts = set(object_counts)
    elif scenario == "none_move":
        move_counts = set()
    else:
        move_size = rng.randint(1, n_objects - 1)
        move_counts = set(rng.sample(object_counts, move_size))

    used = np.zeros((grid_h, grid_w), dtype=bool)
    used[:GEN_TOP_MARGIN, :] = True

    placed_specs = []
    for spec in specs:
        spec = dict(spec)
        spec["target_color"] = legend[spec["hole_count"]]
        spec["will_move"] = spec["hole_count"] in move_counts
        placed_specs.append(spec)

    rng.shuffle(placed_specs)

    for spec in placed_specs:
        if not spec["will_move"]:
            continue
        dest = _random_rect(used, spec["height"], spec["width"], rng)
        if dest is None:
            return None
        dr0, dc0 = dest
        spec["dest_top_left"] = dest
        spec["anchor"] = (dr0 + spec["height"] // 2, dc0 + spec["width"] // 2)
        _mark_used(used, dr0, dc0, spec["height"], spec["width"])

    for spec in placed_specs:
        src = _random_rect(used, spec["height"], spec["width"], rng)
        if src is None:
            return None
        sr0, sc0 = src
        spec["src_top_left"] = src
        _mark_used(used, sr0, sc0, spec["height"], spec["width"])

    for spec in placed_specs:
        if spec["will_move"]:
            ar, ac = spec["anchor"]
            grid[ar, ac] = spec["target_color"]

    for spec in placed_specs:
        sr0, sc0 = spec["src_top_left"]
        _paint_object(grid, spec, sr0, sc0)

    try:
        out = main(grid)
    except AssertionError:
        return None

    expected_anchors = {spec["target_color"] for spec in placed_specs if spec["will_move"]}
    actual_anchors = {color for color in TARGET_COLORS if np.count_nonzero(grid[1:, :] == color) == 1}
    if actual_anchors != expected_anchors:
        return None

    for spec in placed_specs:
        h = spec["height"]
        w = spec["width"]
        sr0, sc0 = spec["src_top_left"]
        before = (grid[sr0 : sr0 + h, sc0 : sc0 + w] == GREY).sum()
        after = (out[sr0 : sr0 + h, sc0 : sc0 + w] == GREY).sum()
        if spec["will_move"] and after >= before:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
