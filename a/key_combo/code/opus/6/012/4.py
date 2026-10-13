from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WALL = 2
MARKER = 9
ANCHOR_COLORS = {7, 8}
SPRITE_COLORS = {1, 3, 4, 5, 6}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _component_from(
    grid: np.ndarray,
    start: tuple[int, int],
    visited: np.ndarray,
) -> dict:
    color = int(grid[start])
    q: deque[tuple[int, int]] = deque([start])
    visited[start] = True
    cells: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        cells.append((int(r), int(c)))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if not visited[nr, nc] and int(grid[nr, nc]) == color:
                    visited[nr, nc] = True
                    q.append((nr, nc))

    coords = np.array(sorted(cells), dtype=int)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return {
        "color": color,
        "coords": coords,
        "coord_set": {tuple(rc) for rc in cells},
        "area": int(coords.shape[0]),
        "bbox": (r0, c0, r1, c1),
        "top_row": r0,
        "left_col": c0,
    }


def _find_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    visited = np.zeros(grid.shape, dtype=bool)
    comps: list[dict] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if visited[r, c] or int(grid[r, c]) not in colors:
                continue
            comps.append(_component_from(grid, (r, c), visited))
    return comps


def _wall_column(grid: np.ndarray) -> int:
    wall_cells = np.argwhere(grid == WALL)
    assert wall_cells.size > 0, "expected a red separator wall"
    counts = np.bincount(wall_cells[:, 1], minlength=grid.shape[1])
    col = int(np.argmax(counts))
    assert int(counts[col]) >= grid.shape[0] - 2, "wall must be a tall vertical separator"
    return col


def _direction_from_marker(grid: np.ndarray, coords: np.ndarray) -> tuple[int, int]:
    dirs: set[tuple[int, int]] = set()
    H, W = grid.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) == MARKER:
                dirs.add((-dr, -dc))
    assert len(dirs) == 1, "each sprite must touch markers on exactly one side"
    return next(iter(dirs))


def _perp_width(coords: np.ndarray, direction: tuple[int, int]) -> int:
    if direction[0] != 0:
        return int(coords[:, 1].max() - coords[:, 1].min() + 1)
    return int(coords[:, 0].max() - coords[:, 0].min() + 1)


def _priority(sprite: dict, wall_col: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = sprite["bbox"]
    dr, dc = sprite["direction"]
    if dc == 1 and c1 < wall_col:
        dist = wall_col - c1 - 1
        return (0, int(dist), int(r0), int(c0))
    if dc == -1 and c0 > wall_col:
        dist = c0 - wall_col - 1
        return (0, int(dist), int(r0), int(c0))
    return (1, abs(int((c0 + c1) // 2) - wall_col), int(r0), int(c0))


def _touches4(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    if len(a) > len(b):
        a, b = b, a
    for r, c in a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in b:
                return True
    return False


def _move_sprite(
    coords: np.ndarray,
    direction: tuple[int, int],
    width: int,
    wall_mask: np.ndarray,
    anchor_cells: set[tuple[int, int]],
) -> tuple[np.ndarray, set[tuple[int, int]], dict]:
    pos = coords.copy()
    H, W = wall_mask.shape
    swept: set[tuple[int, int]] = set()
    touched_wall = False
    passed_wall = False
    stopped_by_wall = False
    stopped_by_anchor = False

    while True:
        nxt = pos.copy()
        nxt[:, 0] += int(direction[0])
        nxt[:, 1] += int(direction[1])
        if not bool(
            np.all((0 <= nxt[:, 0]) & (nxt[:, 0] < H) & (0 <= nxt[:, 1]) & (nxt[:, 1] < W))
        ):
            break

        next_set = {tuple(rc) for rc in nxt.tolist()}
        if next_set & anchor_cells:
            stopped_by_anchor = True
            break

        wall_hits = [(int(r), int(c)) for r, c in next_set if bool(wall_mask[int(r), int(c)])]
        if wall_hits:
            touched_wall = True
            if width == 1:
                for r, c in wall_hits:
                    wall_mask[r, c] = False
                passed_wall = True
                pos = nxt
                swept.update(next_set)
                continue
            stopped_by_wall = True
            break

        pos = nxt
        swept.update(next_set)

    meta = {
        "touched_wall": bool(touched_wall),
        "passed_wall": bool(passed_wall),
        "stopped_by_wall": bool(stopped_by_wall),
        "stopped_by_anchor": bool(stopped_by_anchor),
    }
    return pos, swept, meta


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    wall_col = _wall_column(g)
    anchors = _find_components(g, ANCHOR_COLORS)
    sprites = _find_components(g, SPRITE_COLORS)
    assert 4 <= len(sprites) <= 6, "expected 4-6 movable sprites"

    parsed = []
    for idx, sprite in enumerate(sprites):
        direction = _direction_from_marker(g, sprite["coords"])
        width = _perp_width(sprite["coords"], direction)
        parsed.append(
            {
                **sprite,
                "id": idx,
                "direction": direction,
                "width": int(width),
            }
        )

    return {
        "grid": g,
        "wall_col": int(wall_col),
        "anchors": anchors,
        "sprites": parsed,
    }


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    scene = _parse_scene(grid)
    g = scene["grid"]
    wall_mask = g == WALL
    anchor_cells: set[tuple[int, int]] = set()
    anchor_blocks: list[dict] = []
    for anchor in scene["anchors"]:
        anchor_cells |= anchor["coord_set"]
        anchor_blocks.append(
            {
                "id": f"anchor-{len(anchor_blocks)}",
                "kind": "anchor",
                "color": int(anchor["color"]),
                "coords": anchor["coords"],
                "coord_set": set(anchor["coord_set"]),
                "area": int(anchor["area"]),
                "top_row": int(anchor["top_row"]),
                "left_col": int(anchor["left_col"]),
            }
        )

    final_by_id: dict[int, np.ndarray] = {}
    sprite_meta: dict[int, dict] = {}
    for sprite in sorted(scene["sprites"], key=lambda sp: _priority(sp, scene["wall_col"])):
        final_coords, swept, move_meta = _move_sprite(
            sprite["coords"],
            sprite["direction"],
            sprite["width"],
            wall_mask,
            anchor_cells,
        )
        final_by_id[int(sprite["id"])] = final_coords
        sprite_meta[int(sprite["id"])] = {
            **move_meta,
            "swept": swept,
            "direction": tuple(sprite["direction"]),
            "width": int(sprite["width"]),
            "orig_color": int(sprite["color"]),
            "area": int(sprite["area"]),
            "bbox": tuple(sprite["bbox"]),
        }

    sprite_blocks: list[dict] = []
    for sprite in scene["sprites"]:
        coords = final_by_id[int(sprite["id"])]
        coord_set = {tuple(rc) for rc in coords.tolist()}
        sprite_blocks.append(
            {
                "id": int(sprite["id"]),
                "kind": "sprite",
                "color": int(sprite["color"]),
                "coords": coords,
                "coord_set": coord_set,
                "area": int(coords.shape[0]),
                "top_row": int(coords[:, 0].min()),
                "left_col": int(coords[:, 1].min()),
            }
        )

    all_blocks = sprite_blocks + anchor_blocks
    recolors: dict[int, int] = {}
    recolor_sources: dict[int, str] = {}
    for block in sprite_blocks:
        neighbors = [
            other
            for other in all_blocks
            if other["id"] != block["id"] and _touches4(block["coord_set"], other["coord_set"])
        ]
        if neighbors:
            best = max(neighbors, key=lambda b: (int(b["area"]), -int(b["top_row"]), -int(b["left_col"])))
            recolors[int(block["id"])] = int(best["color"])
            recolor_sources[int(block["id"])] = str(best["kind"])
        else:
            recolors[int(block["id"])] = int(block["color"])
            recolor_sources[int(block["id"])] = "self"

    out = np.full_like(g, BG)
    out[g == MARKER] = MARKER
    out[wall_mask] = WALL
    for anchor in anchor_blocks:
        coords = anchor["coords"]
        out[coords[:, 0], coords[:, 1]] = int(anchor["color"])
    for block in sprite_blocks:
        coords = block["coords"]
        out[coords[:, 0], coords[:, 1]] = int(recolors[int(block["id"])])

    meta = {
        "scene": scene,
        "sprites": sprite_meta,
        "recolors": recolors,
        "recolor_sources": recolor_sources,
        "final_by_id": final_by_id,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push sprites away from maroon markers, let thin ones punch through the red wall, then recolor them by their new neighbors.

    Input structure:
    - Black background contains one red vertical separator wall, several monochrome sprites, maroon single-pixel markers, and two 2x2 bottom anchor blocks.
    - Every movable sprite is a 4-connected component in blue/green/yellow/grey/pink and touches maroon on exactly one cardinal side.
    - The anchors are fixed orange or teal 2x2 blocks, and the separator wall is one cell thick.

    Rule:
    - Keep all maroon markers unchanged and keep every red wall pixel unless a width-1 sprite later punches through it.
    - For each sprite, move it straight away from its marker; when a moving sprite next steps into the wall, width 1 deletes the hit wall pixel and continues, while width >= 2 stops one cell before the wall.
    - If a sprite never meets the wall, or after punching through it, it keeps moving until the next step would leave the grid or hit an anchor block.
    - Process sprites from nearest-to-wall to furthest-to-wall, then recolor each stopped sprite to the color of its largest 4-neighbor block among other stopped sprites and anchors; ties go to the neighbor whose top row is smaller.

    Key insight:
    - Marker side sets the translation ray, sprite width decides whether the wall is a barrier or a door, and that landing side changes which neighbor color the sprite inherits.
    """
    out, _meta = _simulate(grid)
    return out


def _cells_to_array(cells: list[tuple[int, int]]) -> np.ndarray:
    return np.array(sorted((int(r), int(c)) for r, c in cells), dtype=int)


def _paint_cells(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _bar(row: int, c0: int, length: int) -> np.ndarray:
    return _cells_to_array([(int(row), int(c0 + k)) for k in range(int(length))])


def _rect(r0: int, c0: int, h: int, w: int) -> np.ndarray:
    return _cells_to_array(
        [(int(r0 + dr), int(c0 + dc)) for dr in range(int(h)) for dc in range(int(w))]
    )


def _mirror_h(grid: np.ndarray) -> np.ndarray:
    return np.fliplr(grid)


def _add_sprite(
    grid: np.ndarray,
    color: int,
    coords: np.ndarray,
    marker: tuple[int, int],
    reserved_paths: set[tuple[int, int]],
    reserved_outputs: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    coord_set = {tuple(rc) for rc in coords.tolist()}
    mr, mc = marker
    if not (0 <= mr < H and 0 <= mc < W):
        return False
    if any(not (0 <= int(r) < H and 0 <= int(c) < W) for r, c in coord_set):
        return False
    if any(int(grid[r, c]) != BG for r, c in coord_set):
        return False
    if int(grid[mr, mc]) != BG:
        return False
    if marker in coord_set:
        return False

    probe = grid.copy()
    _paint_cells(probe, coords, color)
    probe[mr, mc] = MARKER
    try:
        direction = _direction_from_marker(probe, coords)
    except AssertionError:
        return False
    width = _perp_width(coords, direction)
    wall_mask = probe == WALL
    anchor_cells = set()
    for anchor in _find_components(probe, ANCHOR_COLORS):
        anchor_cells |= anchor["coord_set"]
    final_coords, swept, _meta = _move_sprite(
        coords,
        direction,
        width,
        wall_mask,
        anchor_cells,
    )
    swept_no_wall = {cell for cell in swept if not bool((probe == WALL)[cell])}
    if (swept_no_wall & reserved_paths) or ({tuple(rc) for rc in final_coords.tolist()} & reserved_outputs):
        return False

    _paint_cells(grid, coords, color)
    grid[mr, mc] = MARKER
    reserved_paths.update(swept_no_wall)
    reserved_outputs.update({tuple(rc) for rc in final_coords.tolist()})
    return True


def _build_case(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(16, 20)
    W = rng.randint(20, 24)
    wall_col = rng.randint(W // 2 - 1, W // 2 + 1)
    if wall_col < 8 or W - wall_col - 1 < 8:
        return None

    grid = np.full((H, W), BG, dtype=int)
    grid[:, wall_col] = WALL

    left_anchor_color = rng.choice([7, 8])
    right_anchor_color = rng.choice([7, 8])
    grid[H - 2 : H, 1:3] = left_anchor_color
    grid[H - 2 : H, W - 3 : W - 1] = right_anchor_color

    colors = rng.sample(sorted(SPRITE_COLORS), 5)
    a_color, b_color, c_color, d_color, e_color = colors

    reserved_paths: set[tuple[int, int]] = set()
    reserved_outputs: set[tuple[int, int]] = set()

    len_a = rng.randint(5, 6)
    len_b = rng.randint(4, 5)
    rightmost_a = wall_col - rng.randint(2, 4)
    rightmost_b = wall_col - rng.randint(2, 4)
    a_coords = _bar(H - 1, rightmost_a - len_a + 1, len_a)
    b_coords = _bar(H - 2, rightmost_b - len_b + 1, len_b)
    if a_coords[:, 1].min() <= 1 or b_coords[:, 1].min() <= 1:
        return None
    a_marker_col = int(a_coords[:, 1].min()) - 1
    b_marker_col = int(b_coords[:, 1].min()) - 1
    if a_marker_col in set(map(int, b_coords[:, 1])) or b_marker_col in set(map(int, a_coords[:, 1])):
        return None

    c_top = rng.randint(4, max(5, H // 2))
    c_len = rng.randint(2, 4)
    c_right = wall_col - rng.randint(2, 4)
    c_coords = _rect(c_top, c_right - c_len + 1, 2, c_len)
    if c_coords[:, 1].min() <= 1 or c_top + 1 >= H - 5:
        return None

    d_row_candidates = [r for r in range(1, max(2, c_top - 1)) if r not in {H - 1, H - 2}]
    if not d_row_candidates:
        return None
    d_row = rng.choice(d_row_candidates)
    d_len = rng.randint(4, 6)
    d_left = wall_col + rng.randint(2, 4)
    d_coords = _bar(d_row, d_left, d_len)
    if d_coords[:, 1].max() >= W - 2:
        return None

    e_top = rng.randint(2, H - 8)
    e_coords = _rect(e_top, W - 3, 2, 2)

    specs = [
        (a_color, a_coords, (H - 1, a_marker_col)),
        (b_color, b_coords, (H - 2, b_marker_col)),
        (c_color, c_coords, (c_top, int(c_coords[:, 1].min()) - 1)),
        (d_color, d_coords, (d_row, int(d_coords[:, 1].max()) + 1)),
        (e_color, e_coords, (e_top - 1, W - 3)),
    ]

    for color, coords, marker in specs:
        if not _add_sprite(grid, color, coords, marker, reserved_paths, reserved_outputs):
            return None

    if rng.random() < 0.5:
        grid = _mirror_h(grid)
    return grid


def _validate_case(grid: np.ndarray) -> bool:
    try:
        out, meta = _simulate(grid)
    except AssertionError:
        return False

    scene = meta["scene"]
    if len(scene["sprites"]) != 5:
        return False

    sprite_meta = list(meta["sprites"].values())
    if sum(1 for m in sprite_meta if m["passed_wall"]) < 2:
        return False
    if not any(m["stopped_by_wall"] for m in sprite_meta):
        return False
    if not any((not m["touched_wall"]) and m["stopped_by_anchor"] for m in sprite_meta):
        return False

    recolor_sources = meta["recolor_sources"]
    recolors = meta["recolors"]
    changed = [sid for sid, m in meta["sprites"].items() if int(recolors[sid]) != int(m["orig_color"])]
    unchanged = [sid for sid, m in meta["sprites"].items() if int(recolors[sid]) == int(m["orig_color"])]
    if not changed or not unchanged:
        return False
    if not any(recolor_sources[sid] == "sprite" for sid in changed):
        return False
    if not any(recolor_sources[sid] == "anchor" for sid in changed):
        return False

    if np.array_equal(grid, out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        grid = _build_case(rng)
        if grid is None:
            continue
        if _validate_case(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
