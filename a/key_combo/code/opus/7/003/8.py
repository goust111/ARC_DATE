import random
import sys
from collections import deque
from itertools import combinations
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, find_components, get_bg, neighbors4


BG = Color.BLACK
WALL = Color.BLUE
SPRITE_COLORS = (Color.RED, Color.GREEN, Color.ORANGE)
MARKER = Color.YELLOW
GUIDE = Color.GREY

GRID_MIN = 16
GRID_MAX = 22
GEN_ATTEMPTS = 400


def _is_hollow_sprite(grid: np.ndarray, comp: dict) -> bool:
    if int(comp["area"]) != 8:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return False
    tile = crop_bbox(grid, comp["bbox"])
    col = int(comp["color"])
    if int(tile[1, 1]) != int(BG):
        return False
    ring = tile.copy()
    ring[1, 1] = col
    return bool(np.all(ring == col))


def _find_sprites(grid: np.ndarray) -> list[dict]:
    sprites = []
    for comp in find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=SPRITE_COLORS,
        return_format="dict",
    ):
        if _is_hollow_sprite(grid, comp):
            r0, c0, r1, c1 = comp["bbox"]
            sprites.append(
                {
                    "color": int(comp["color"]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "center": (int(r0 + 1), int(c0 + 1)),
                }
            )
    sprites.sort(key=lambda s: (s["bbox"][0], s["bbox"][1], s["color"]))
    return sprites


def _select_sprite(grid: np.ndarray, sprites: list[dict]) -> dict:
    guide_box = bbox(grid == GUIDE)
    assert guide_box is not None, "guide mask missing"
    gr0, gc0, gr1, gc1 = guide_box
    chosen = []
    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        if gr0 <= r0 and gc0 <= c0 and r1 <= gr1 and c1 <= gc1:
            chosen.append(sprite)
    assert len(chosen) == 1, "guide bbox must select exactly one sprite"
    return chosen[0]


def _reachable_non_walls(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = grid.shape
    sr, sc = start
    assert 0 <= sr < h and 0 <= sc < w
    seen = {(int(sr), int(sc))}
    q = deque([(int(sr), int(sc))])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(grid[nr, nc]) != int(WALL):
                seen.add((int(nr), int(nc)))
                q.append((int(nr), int(nc)))
    return seen


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    sprites = _find_sprites(g)
    assert len(sprites) == 3, "expected exactly three hollow sprites"
    assert {int(s["color"]) for s in sprites} == set(map(int, SPRITE_COLORS))

    selected = _select_sprite(g, sprites)
    reach = _reachable_non_walls(g, selected["center"])
    markers = {tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)}
    reachable_markers = markers & reach
    return {
        "sprites": sprites,
        "selected": selected,
        "reach": reach,
        "markers": markers,
        "reachable_markers": reachable_markers,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the guide-framed hollow sprite, then repeat it using the count of reachable marker pixels.

    Input structure:
    - Black background contains exactly three monochrome 3x3 hollow-square sprites in red, green, and orange.
    - Blue wall segments partition the board into regions, with a few single-cell black gaps left in the walls.
    - Sparse gray guide pixels form one bounding box, and yellow singleton markers are scattered across the regions.

    Rule:
    - Find the three hollow 3x3 sprites and compute the bbox of all gray guide pixels.
    - Choose the unique sprite whose 3x3 bbox is fully contained in that guide bbox.
    - Starting from the chosen sprite's center, flood through every non-blue cell; blue wall pixels are the only blockers.
    - Count how many yellow marker pixels lie in that reachable set, crop the chosen 3x3 sprite, and concatenate that tile horizontally that many times.

    Key insight:
    - The guide mask chooses both which sprite to output and which region supplies the flood-fill start, so the wall-gap reachability determines the repetition count.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    selected = info["selected"]
    count = len(info["reachable_markers"])
    assert count >= 1, "at least one reachable marker is required"
    tile = crop_bbox(g, selected["bbox"])
    return np.tile(tile, (1, count))


def _range_inclusive(lo: int, hi: int) -> list[int]:
    if lo > hi:
        return []
    return list(range(int(lo), int(hi) + 1))


def _choice(seq):
    return random.choice(list(seq))


def _build_cross_layout() -> dict:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    split_r = random.randint(5, h - 6)
    split_c = random.randint(5, w - 6)

    grid = np.full((h, w), BG, dtype=int)
    grid[split_r, :] = WALL
    grid[:, split_c] = WALL

    chambers = {
        "TL": (0, 0, split_r - 1, split_c - 1),
        "TR": (0, split_c + 1, split_r - 1, w - 1),
        "BL": (split_r + 1, 0, h - 1, split_c - 1),
        "BR": (split_r + 1, split_c + 1, h - 1, w - 1),
    }
    edges = {
        "top": ("TL", "TR", (random.randint(1, split_r - 1), split_c)),
        "bottom": ("BL", "BR", (random.randint(split_r + 1, h - 2), split_c)),
        "left": ("TL", "BL", (split_r, random.randint(1, split_c - 1))),
        "right": ("TR", "BR", (split_r, random.randint(split_c + 1, w - 2))),
    }
    return {"grid": grid, "chambers": chambers, "edges": edges}


def _build_t_layout() -> dict:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    split_r = random.randint(5, h - 6)
    split_c = random.randint(5, w - 6)

    grid = np.full((h, w), BG, dtype=int)
    grid[:, split_c] = WALL
    grid[split_r, : split_c] = WALL

    chambers = {
        "TL": (0, 0, split_r - 1, split_c - 1),
        "BL": (split_r + 1, 0, h - 1, split_c - 1),
        "R": (0, split_c + 1, h - 1, w - 1),
    }
    edges = {
        "top": ("TL", "R", (random.randint(1, split_r - 1), split_c)),
        "bottom": ("BL", "R", (random.randint(split_r + 1, h - 2), split_c)),
        "left": ("TL", "BL", (split_r, random.randint(1, split_c - 1))),
    }
    return {"grid": grid, "chambers": chambers, "edges": edges}


def _reachable_chambers(chambers: dict, edges: dict, open_edges: set[str], start: str) -> set[str]:
    graph = {name: set() for name in chambers}
    for edge_name in open_edges:
        a, b, _gap = edges[edge_name]
        graph[a].add(b)
        graph[b].add(a)
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in graph[cur]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def _pick_open_edges(layout: dict, selected: str) -> tuple[set[str], set[str]]:
    chamber_names = set(layout["chambers"])
    valid = []
    edge_names = list(layout["edges"])
    for k in range(1, len(edge_names) + 1):
        for subset in combinations(edge_names, k):
            chosen = set(subset)
            reach = _reachable_chambers(layout["chambers"], layout["edges"], chosen, selected)
            if 2 <= len(reach) < len(chamber_names):
                valid.append((chosen, reach))
    assert valid
    return random.choice(valid)


def _pick_sprite_positions(chamber: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = chamber
    rows = _range_inclusive(r0 + 1, r1 - 3)
    cols = _range_inclusive(c0 + 1, c1 - 3)
    return [(r, c) for r in rows for c in cols]


def _paint_sprite(grid: np.ndarray, top_left: tuple[int, int], color: int) -> tuple[int, int, int, int]:
    r, c = top_left
    grid[r : r + 3, c : c + 3] = color
    grid[r + 1, c + 1] = BG
    return (int(r), int(c), int(r + 2), int(c + 2))


def _guide_cells_for_bbox(sprite_bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = sprite_bbox
    gr0, gc0, gr1, gc1 = r0 - 1, c0 - 1, r1 + 1, c1 + 1
    cells = {(gr0, gc0), (gr0, gc1), (gr1, gc0), (gr1, gc1)}
    extras = [
        (gr0, gc0 + 2),
        (gr1, gc0 + 2),
        (gr0 + 2, gc0),
        (gr0 + 2, gc1),
    ]
    random.shuffle(extras)
    for cell in extras[: random.randint(0, 2)]:
        cells.add(cell)
    return cells


def _pick_marker_cell(
    chamber: tuple[int, int, int, int],
    reserved_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int]:
    r0, c0, r1, c1 = chamber
    candidates = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            blocked = False
            for br0, bc0, br1, bc1 in reserved_boxes:
                if br0 <= r <= br1 and bc0 <= c <= bc1:
                    blocked = True
                    break
            if not blocked:
                candidates.append((r, c))
    assert candidates, "no free marker cell"
    return _choice(candidates)


def _expected_output_from_info(grid: np.ndarray, info: dict) -> np.ndarray:
    tile = crop_bbox(grid, info["selected"]["bbox"])
    return np.tile(tile, (1, len(info["reachable_markers"])))


def _validate_generated(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    info = _analyze(g)
    marker_total = len(info["markers"])
    reachable_total = len(info["reachable_markers"])
    assert 2 <= marker_total <= 4
    assert 1 <= reachable_total < marker_total

    guide_box = bbox(g == GUIDE)
    assert guide_box is not None
    gr0, gc0, gr1, gc1 = guide_box
    assert gr1 - gr0 + 1 >= 5 and gc1 - gc0 + 1 >= 5

    out = main(g)
    expected = _expected_output_from_info(g, info)
    assert np.array_equal(out, expected)
    assert out.shape == (3, 3 * reachable_total)


def _random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.asarray(out, dtype=int)


def _gen_attempt() -> np.ndarray | None:
    layout = _choice([_build_cross_layout, _build_t_layout])()
    grid = layout["grid"].copy()
    chambers = layout["chambers"]
    selected = _choice(chambers.keys())
    open_edges, reachable = _pick_open_edges(layout, selected)

    for edge_name in open_edges:
        _a, _b, gap = layout["edges"][edge_name]
        grid[gap] = BG

    chamber_names = list(chambers)
    if len(chamber_names) == 3:
        sprite_chambers = chamber_names
    else:
        others = [name for name in chamber_names if name != selected]
        sprite_chambers = [selected] + random.sample(others, 2)
    random.shuffle(sprite_chambers)
    color_map = {name: color for name, color in zip(sprite_chambers, random.sample(list(SPRITE_COLORS), 3))}

    sprite_boxes = {}
    reserved_by_chamber = {name: [] for name in chambers}
    for name in sprite_chambers:
        positions = _pick_sprite_positions(chambers[name])
        if not positions:
            return None
        top_left = _choice(positions)
        box = _paint_sprite(grid, top_left, color_map[name])
        sprite_boxes[name] = box
        reserved_by_chamber[name].append(box)

    selected_box = sprite_boxes[selected]
    guide_cells = _guide_cells_for_bbox(selected_box)
    for r, c in guide_cells:
        if int(grid[r, c]) != int(BG):
            return None
        grid[r, c] = GUIDE
    sr0, sc0, sr1, sc1 = selected_box
    reserved_by_chamber[selected].append((sr0 - 1, sc0 - 1, sr1 + 1, sc1 + 1))

    reachable_other = list(sorted(reachable - {selected}))
    unreachable = list(sorted(set(chambers) - reachable))
    if not reachable_other or not unreachable:
        return None

    marker_chambers = {random.choice(reachable_other), random.choice(unreachable)}
    remaining = [name for name in chambers if name not in marker_chambers]
    target_total = random.randint(2, len(chambers))
    random.shuffle(remaining)
    for name in remaining:
        if len(marker_chambers) >= target_total:
            break
        if random.random() < 0.65 or len(marker_chambers) + len(remaining) <= target_total:
            marker_chambers.add(name)
    if len(marker_chambers) < 2:
        return None

    marker_cells = {}
    for name in marker_chambers:
        try:
            cell = _pick_marker_cell(chambers[name], reserved_by_chamber[name])
        except AssertionError:
            return None
        if int(grid[cell]) != int(BG):
            return None
        grid[cell] = MARKER
        marker_cells[name] = cell

    if len(marker_cells) != len(marker_chambers):
        return None

    reachable_marked_other = [name for name in reachable_other if name in marker_chambers]
    unreachable_marked = [name for name in unreachable if name in marker_chambers]
    if not reachable_marked_other or not unreachable_marked:
        return None

    transformed = _random_symmetry(grid)
    try:
        _validate_generated(transformed)
    except AssertionError:
        return None
    return transformed


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 8")
