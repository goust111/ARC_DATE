from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random

import numpy as np


BG = 0
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ORTHOGONAL = {
    "up": ("left", "right"),
    "down": ("left", "right"),
    "left": ("up", "down"),
    "right": ("up", "down"),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
ROTATION_BY_DIR = {"up": 0, "left": 1, "down": 2, "right": 3}
PALETTE = tuple(range(1, 10))


@dataclass(frozen=True)
class Triangle:
    color: int
    direction: str
    size: int
    bbox: tuple[int, int, int, int]
    apex: tuple[int, int]
    cells: tuple[tuple[int, int], ...]
    base_cells: frozenset[tuple[int, int]]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def triangle_mask(direction: str, size: int) -> np.ndarray:
    size = int(size)
    assert size >= 2
    if direction == "up":
        h, w = size, 2 * size - 1
        center = size - 1
        mask = np.zeros((h, w), dtype=bool)
        for r in range(h):
            mask[r, center - r : center + r + 1] = True
        return mask
    if direction == "down":
        return np.flipud(triangle_mask("up", size))
    if direction == "right":
        h, w = 2 * size - 1, size
        center = size - 1
        mask = np.zeros((h, w), dtype=bool)
        for c in range(w):
            reach = size - 1 - c
            mask[center - reach : center + reach + 1, c] = True
        return mask
    if direction == "left":
        return np.fliplr(triangle_mask("right", size))
    raise ValueError(f"unknown direction: {direction}")


def make_triangle_patch(direction: str, size: int, color: int) -> np.ndarray:
    mask = triangle_mask(direction, size)
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in neighbors4(rr, cc):
                    if not in_bounds((h, w), nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((int(nr), int(nc)))
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def detect_triangle(component: dict) -> Triangle:
    coords = component["coords"]
    color = int(component["color"])
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    patch = np.zeros((h, w), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True

    matches: list[tuple[str, int]] = []
    if w == 2 * h - 1:
        for direction in ("up", "down"):
            if np.array_equal(patch, triangle_mask(direction, h)):
                matches.append((direction, h))
    if h == 2 * w - 1:
        for direction in ("left", "right"):
            if np.array_equal(patch, triangle_mask(direction, w)):
                matches.append((direction, w))
    assert len(matches) == 1, "every non-ignition component must be one canonical solid triangle"
    direction, size = matches[0]

    if direction == "up":
        apex = (r0, c0 + size - 1)
        base_cells = frozenset((r1, c) for c in range(c0, c1 + 1))
    elif direction == "down":
        apex = (r1, c0 + size - 1)
        base_cells = frozenset((r0, c) for c in range(c0, c1 + 1))
    elif direction == "right":
        apex = (r0 + size - 1, c1)
        base_cells = frozenset((r, c0) for r in range(r0, r1 + 1))
    else:
        apex = (r0 + size - 1, c0)
        base_cells = frozenset((r, c1) for r in range(r0, r1 + 1))

    cells = tuple(sorted((int(r), int(c)) for r, c in coords))
    return Triangle(
        color=color,
        direction=direction,
        size=int(size),
        bbox=(r0, c0, r1, c1),
        apex=tuple(map(int, apex)),
        cells=cells,
        base_cells=base_cells,
    )


def parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = int(get_bg(g))
    assert bg == BG, "the puzzle uses a black background"

    components = find_components(g, bg)
    assert len(components) >= 3, "expected several foreground objects"
    assert len({int(comp["color"]) for comp in components}) == len(components), "all foreground colors must be unique"

    ignition_components = [comp for comp in components if int(comp["area"]) == 1]
    assert len(ignition_components) == 1, "expected exactly one ignition pixel"
    ignition_comp = ignition_components[0]
    ignition = tuple(map(int, ignition_comp["coords"][0]))
    ignition_color = int(ignition_comp["color"])

    triangles: list[Triangle] = [detect_triangle(comp) for comp in components if comp is not ignition_comp]
    assert triangles, "expected at least one triangle"
    by_color = {tri.color: tri for tri in triangles}
    assert len(by_color) == len(triangles)

    lookup: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for cell in tri.cells:
            assert cell not in lookup, "triangles must not overlap"
            lookup[cell] = tri.color

    start_candidates: list[int] = []
    for tri in triangles:
        dr, dc = DIRS[tri.direction]
        base_outer = {(r - dr, c - dc) for r, c in tri.base_cells}
        if ignition in base_outer:
            start_candidates.append(int(tri.color))
    assert len(start_candidates) == 1, "ignition pixel must uniquely pick a triangle base"
    start_color = start_candidates[0]

    return {
        "grid": g,
        "bg": bg,
        "ignition": ignition,
        "ignition_color": ignition_color,
        "triangles": triangles,
        "by_color": by_color,
        "lookup": lookup,
        "start_color": start_color,
    }


def _solve_with_info(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    scene = parse_scene(grid)
    g = scene["grid"]
    out = g.copy()
    lookup = scene["lookup"]
    by_color = scene["by_color"]
    ignition = scene["ignition"]

    current_color = int(scene["start_color"])
    ignited = [current_color]
    ignited_set = {current_color}
    segments: list[dict] = []

    for _ in range(len(by_color) + 1):
        tri = by_color[current_color]
        dr, dc = DIRS[tri.direction]
        r = tri.apex[0] + dr
        c = tri.apex[1] + dc
        path: list[tuple[int, int]] = []
        hit_color: int | None = None
        hit_coord: tuple[int, int] | None = None

        while in_bounds(g.shape, r, c):
            coord = (int(r), int(c))
            if coord in lookup:
                hit_color = int(lookup[coord])
                hit_coord = coord
                break
            assert coord != ignition, "the ignition pixel must stay out of the forward ray corridor"
            assert int(g[coord]) == BG, "rays should travel through background until a triangle is hit"
            path.append(coord)
            r += dr
            c += dc

        for rr, cc in path:
            out[rr, cc] = int(tri.color)

        if hit_color is None:
            segments.append(
                {
                    "source_color": int(tri.color),
                    "path_len": int(len(path)),
                    "hit_color": None,
                    "hit_kind": None,
                    "hit_coord": None,
                }
            )
            terminal = {"kind": "boundary"}
            return out, {
                "ignited_sequence": ignited,
                "segments": segments,
                "terminal": terminal,
            }

        target = by_color[hit_color]
        if hit_coord in target.base_cells:
            assert hit_color not in ignited_set, "a triangle may ignite at most once"
            hit_kind = "base"
            ignited.append(int(hit_color))
            ignited_set.add(int(hit_color))
            segments.append(
                {
                    "source_color": int(tri.color),
                    "path_len": int(len(path)),
                    "hit_color": int(hit_color),
                    "hit_kind": hit_kind,
                    "hit_coord": hit_coord,
                }
            )
            current_color = int(hit_color)
            continue

        hit_kind = "apex" if hit_coord == target.apex else "side"
        for rr, cc in target.cells:
            out[rr, cc] = int(tri.color)
        segments.append(
            {
                "source_color": int(tri.color),
                "path_len": int(len(path)),
                "hit_color": int(hit_color),
                "hit_kind": hit_kind,
                "hit_coord": hit_coord,
            }
        )
        terminal = {
            "kind": "absorb",
            "mode": hit_kind,
            "target_color": int(hit_color),
            "incoming_color": int(tri.color),
        }
        return out, {
            "ignited_sequence": ignited,
            "segments": segments,
            "terminal": terminal,
        }

    raise AssertionError("chain reaction exceeded the number of triangles")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the ignition-driven arrow chain, relaying only through base hits and recoloring the first non-base hit triangle.

    Input structure:
    - The grid has a black background, several solid monochrome triangles, and exactly one unique-color ignition pixel.
    - Each triangle points up, down, left, or right; the ignition pixel sits directly outside the base side of exactly one triangle.
    - Triangle colors are all distinct, so every triangle is an individually identifiable object.

    Rule:
    - Start from the triangle selected by the ignition pixel and launch a straight ray of that triangle's color from the cell just beyond its apex in the pointing direction.
    - When the ray first reaches another triangle, a base-side hit ignites that triangle and the chain continues from its own apex; a side or apex hit stops the chain and recolors the whole struck triangle to the incoming ray color.
    - If a ray segment reaches the canvas border without hitting any triangle, it is drawn all the way to the edge and the chain stops; no triangle may ignite more than once.

    Key insight:
    - A triangle's orientation matters twice: it chooses both the outgoing ray direction and the one side that can accept the relay, so the chain length is decided by hit-face geometry rather than by simple nearest-object distance.
    """
    out, _info = _solve_with_info(grid)
    return out


def _place_patch(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    padding: int = 1,
    forbidden: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return None

    cells = [(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(patch != BG)]
    cell_set = set(cells)
    if forbidden is not None and any(cell in forbidden for cell in cell_set):
        return None

    for rr, cc in cells:
        if int(grid[rr, cc]) != BG:
            return None
        for nr in range(rr - padding, rr + padding + 1):
            for nc in range(cc - padding, cc + padding + 1):
                if (nr, nc) in occupied and (nr, nc) not in cell_set:
                    return None

    for rr, cc in cells:
        grid[rr, cc] = int(patch[rr - r0, cc - c0])
        occupied.add((rr, cc))
    return cells


def _rotate_grid_for_direction(grid: np.ndarray, direction: str) -> np.ndarray:
    return np.rot90(grid, k=ROTATION_BY_DIR[direction]).copy()


def _core_rows_for_branch(
    chain_sizes: list[int],
    branch: str,
    terminal_size: int | None,
    terminal_side: str | None,
    term_offset: int | None,
    relay_gaps: list[int],
    term_gap: int,
) -> dict | None:
    n = len(chain_sizes)

    if branch == "boundary":
        last_apex = random.randint(2, 5)
        terminal_info = {"top": None, "direction": None, "offset": None}
    elif branch == "apex":
        assert terminal_size is not None
        top = random.randint(0, 2)
        apex_hit = top + terminal_size - 1
        last_apex = apex_hit + term_gap + 1
        terminal_info = {"top": top, "direction": "down", "offset": None}
    else:
        assert terminal_size is not None and terminal_side is not None and term_offset is not None
        top = random.randint(0, 2)
        if terminal_side == "right":
            hit_row = top + 2 * terminal_size - 2 - term_offset
        else:
            hit_row = top + terminal_size - 1 + term_offset
        last_apex = hit_row + term_gap + 1
        terminal_info = {"top": top, "direction": terminal_side, "offset": term_offset}

    apex_rows = [0] * n
    base_rows = [0] * n
    apex_rows[-1] = int(last_apex)
    base_rows[-1] = int(last_apex + chain_sizes[-1] - 1)
    for idx in range(n - 2, -1, -1):
        apex_rows[idx] = int(base_rows[idx + 1] + relay_gaps[idx] + 1)
        base_rows[idx] = int(apex_rows[idx] + chain_sizes[idx] - 1)

    ignition_row = int(base_rows[0] + 1)
    min_height = ignition_row + 2
    if min_height > 30:
        return None

    height = min_height + random.randint(0, min(4, 30 - min_height))
    return {
        "height": int(height),
        "apex_rows": apex_rows,
        "terminal": terminal_info,
    }


def _build_canonical_up_scene() -> tuple[np.ndarray, dict] | None:
    branch = random.choices(
        population=["boundary", "apex", "side"],
        weights=[3, 3, 4],
        k=1,
    )[0]
    chain_len = random.randint(2, 4)
    chain_sizes = [random.randint(3, 4) for _ in range(chain_len)]
    relay_gaps = [random.randint(1, 4) for _ in range(chain_len - 1)]
    term_gap = random.randint(1, 4)

    terminal_size: int | None = None
    terminal_side: str | None = None
    term_offset: int | None = None
    if branch == "apex":
        terminal_size = random.randint(3, 4)
    elif branch == "side":
        terminal_size = random.randint(3, 5)
        terminal_side = random.choice(["left", "right"])
        term_offset = random.randint(1, terminal_size - 2)

    layout = _core_rows_for_branch(
        chain_sizes,
        branch,
        terminal_size,
        terminal_side,
        term_offset,
        relay_gaps,
        term_gap,
    )
    if layout is None:
        return None

    height = int(layout["height"])
    max_half = max(chain_sizes) - 1
    min_width = max(13, 2 * max_half + 7)
    if branch == "side" and terminal_size is not None:
        min_width = max(min_width, 2 * terminal_size + 3)
    if min_width > 30:
        return None
    width = random.randint(min_width, 30)

    axis_lo = max_half + 2
    axis_hi = width - max_half - 3
    if branch == "side" and terminal_size is not None and term_offset is not None:
        axis_lo = max(axis_lo, term_offset + 1)
        axis_hi = min(axis_hi, width - terminal_size + term_offset - 1)
    if axis_lo > axis_hi:
        return None
    axis = random.randint(axis_lo, axis_hi)

    extra_random = random.randint(0, 2)
    core_triangle_count = chain_len + (0 if branch == "boundary" else 1)
    total_triangles = core_triangle_count + 1 + extra_random
    if total_triangles + 1 > len(PALETTE):
        return None
    colors = random.sample(PALETTE, total_triangles + 1)
    ignition_color = colors[0]
    chain_colors = colors[1 : 1 + chain_len]
    color_cursor = 1 + chain_len
    terminal_color: int | None = None
    if branch != "boundary":
        terminal_color = colors[color_cursor]
        color_cursor += 1
    same_dir_color = colors[color_cursor]
    color_cursor += 1
    extra_colors = colors[color_cursor:]

    grid = np.zeros((height, width), dtype=int)
    occupied: set[tuple[int, int]] = set()
    plan = {
        "branch": branch,
        "chain_colors": [int(c) for c in chain_colors],
        "terminal_color": None if terminal_color is None else int(terminal_color),
        "terminal_mode": None if branch == "boundary" else ("apex" if branch == "apex" else "side"),
        "distractor_colors": [int(same_dir_color), *map(int, extra_colors)],
    }

    apex_rows = layout["apex_rows"]
    for apex_row, size, color in zip(apex_rows, chain_sizes, chain_colors):
        r0 = int(apex_row)
        c0 = int(axis - (size - 1))
        cells = _place_patch(
            grid,
            occupied,
            make_triangle_patch("up", size, color),
            r0,
            c0,
            padding=1,
        )
        if cells is None:
            return None

    if branch == "apex":
        assert terminal_size is not None and terminal_color is not None
        top = int(layout["terminal"]["top"])
        c0 = int(axis - (terminal_size - 1))
        cells = _place_patch(
            grid,
            occupied,
            make_triangle_patch("down", terminal_size, terminal_color),
            top,
            c0,
            padding=1,
        )
        if cells is None:
            return None
    elif branch == "side":
        assert terminal_size is not None and terminal_color is not None and terminal_side is not None and term_offset is not None
        top = int(layout["terminal"]["top"])
        c0 = int(axis - term_offset)
        cells = _place_patch(
            grid,
            occupied,
            make_triangle_patch(terminal_side, terminal_size, terminal_color),
            top,
            c0,
            padding=1,
        )
        if cells is None:
            return None

    start_size = chain_sizes[0]
    ignition = (int(apex_rows[0] + start_size), int(axis))
    if not in_bounds(grid.shape, *ignition) or int(grid[ignition]) != BG:
        return None
    triangle_neighbors = [
        (nr, nc)
        for nr, nc in neighbors4(*ignition)
        if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != BG
    ]
    if triangle_neighbors != [(apex_rows[0] + start_size - 1, axis)]:
        return None

    corridor_forbidden = {(r, axis) for r in range(height)}
    ignition_reserved = {ignition}
    for nr in range(ignition[0] - 1, ignition[0] + 2):
        for nc in range(ignition[1] - 1, ignition[1] + 2):
            if in_bounds(grid.shape, nr, nc):
                ignition_reserved.add((nr, nc))

    mandatory_same_dir_size = random.randint(3, 4)
    mandatory_same_dir_patch = make_triangle_patch("up", mandatory_same_dir_size, same_dir_color)
    same_dir_forbidden = corridor_forbidden | ignition_reserved
    placed_same_dir = False
    for _ in range(150):
        side = random.choice([-1, 1])
        center_col = axis + side * random.randint(mandatory_same_dir_size, mandatory_same_dir_size + 4)
        c0 = center_col - (mandatory_same_dir_size - 1)
        r0 = random.randint(0, height - mandatory_same_dir_patch.shape[0])
        cells = _place_patch(
            grid,
            occupied,
            mandatory_same_dir_patch,
            r0,
            c0,
            padding=1,
            forbidden=same_dir_forbidden,
        )
        if cells is not None:
            placed_same_dir = True
            break
    if not placed_same_dir:
        return None

    for color in extra_colors:
        placed = False
        for _ in range(200):
            direction = random.choice(["up", "down", "left", "right"])
            size = random.randint(3, 4)
            patch = make_triangle_patch(direction, size, color)
            r0 = random.randint(0, height - patch.shape[0])
            c0 = random.randint(0, width - patch.shape[1])
            cells = _place_patch(
                grid,
                occupied,
                patch,
                r0,
                c0,
                padding=1,
                forbidden=corridor_forbidden | ignition_reserved,
            )
            if cells is not None:
                placed = True
                break
        if not placed:
            return None

    grid[ignition] = int(ignition_color)
    return grid, plan


def _validate_generated_instance(grid: np.ndarray, plan: dict) -> bool:
    out, info = _solve_with_info(grid)
    if np.array_equal(out, grid):
        return False

    if info["ignited_sequence"] != plan["chain_colors"]:
        return False
    if len(info["segments"]) != len(plan["chain_colors"]):
        return False
    if sum(int(seg["path_len"]) for seg in info["segments"]) < 2:
        return False

    terminal = info["terminal"]
    if plan["branch"] == "boundary":
        if terminal["kind"] != "boundary":
            return False
    else:
        if terminal["kind"] != "absorb":
            return False
        if terminal["target_color"] != plan["terminal_color"]:
            return False
        if terminal["mode"] != plan["terminal_mode"]:
            return False

    segments = info["segments"]
    if not all(seg["hit_kind"] == "base" for seg in segments[:-1]):
        return False
    if segments[-1]["hit_kind"] == "base":
        return False

    scene = parse_scene(grid)
    triangles = scene["by_color"]
    for color in plan["distractor_colors"]:
        tri = triangles[int(color)]
        original_values = {int(grid[r, c]) for r, c in tri.cells}
        output_values = {int(out[r, c]) for r, c in tri.cells}
        if original_values != {int(color)} or output_values != {int(color)}:
            return False

    if plan["branch"] == "side":
        hit_row, hit_col = segments[-1]["hit_coord"]
        target = triangles[int(plan["terminal_color"])]
        if (hit_row, hit_col) == target.apex or (hit_row, hit_col) in target.base_cells:
            return False

    if plan["branch"] == "apex":
        target = triangles[int(plan["terminal_color"])]
        if segments[-1]["hit_coord"] != target.apex:
            return False

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    return True


def _build_instance() -> np.ndarray | None:
    local = _build_canonical_up_scene()
    if local is None:
        return None
    grid, plan = local
    final_direction = random.choice(["up", "down", "left", "right"])
    rotated = _rotate_grid_for_direction(grid, final_direction)
    if not _validate_generated_instance(rotated, plan):
        return None
    return rotated


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-chain relay instance")
