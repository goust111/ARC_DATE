from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5


@dataclass(frozen=True)
class Domino:
    coords: tuple[tuple[int, int], tuple[int, int]]
    red: tuple[int, int]
    green: tuple[int, int]


@dataclass(frozen=True)
class TriangleInstruction:
    coords: tuple[tuple[int, int], ...]
    tip: tuple[int, int]
    line_color: int


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    nbrs = neighbors4 if connectivity == 4 else neighbors8
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in nbrs(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def triangle_sprites() -> list[np.ndarray]:
    sprites: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for size in (2, 3):
        base = np.zeros((size, size), dtype=int)
        for r in range(size):
            for c in range(size):
                if r <= c:
                    base[r, c] = GREY
        for k in range(4):
            sprite = np.rot90(base, k=k)
            key = tuple(tuple(int(v) for v in row) for row in sprite)
            if key not in seen:
                seen.add(key)
                sprites.append(sprite.copy())
    return sprites


TRIANGLE_SPRITES = triangle_sprites()


def sprite_cells(sprite: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    pts = []
    for r, c in np.argwhere(sprite == GREY):
        pts.append((top + int(r), left + int(c)))
    return pts


def diagonal_orientation(blue_coords: np.ndarray) -> tuple[int, int]:
    diffs = {int(r - c) for r, c in blue_coords}
    sums = {int(r + c) for r, c in blue_coords}
    if len(diffs) == 1:
        return 1, 1
    if len(sums) == 1:
        return 1, -1
    raise AssertionError("blue object must be a diagonal segment")


def diagonal_key(coord: tuple[int, int], slope: tuple[int, int]) -> int:
    r, c = coord
    return r - c if slope == (1, 1) else r + c


def component_distance(coords_a: list[tuple[int, int]] | tuple[tuple[int, int], ...] | np.ndarray,
                       coords_b: list[tuple[int, int]] | tuple[tuple[int, int], ...] | np.ndarray) -> int:
    pts_a = [tuple(map(int, p)) for p in coords_a]
    pts_b = [tuple(map(int, p)) for p in coords_b]
    return min(chebyshev(a, b) for a in pts_a for b in pts_b)


def point_to_component_distance(point: tuple[int, int], coords: list[tuple[int, int]] | tuple[tuple[int, int], ...] | np.ndarray) -> int:
    return min(chebyshev(point, tuple(map(int, p))) for p in coords)


def triangle_tip(triangle_coords: np.ndarray, blue_coords: np.ndarray) -> tuple[int, int]:
    pts = [tuple(map(int, p)) for p in triangle_coords]
    blue = [tuple(map(int, p)) for p in blue_coords]
    distances = {pt: min(chebyshev(pt, bp) for bp in blue) for pt in pts}
    max_dist = max(distances.values())
    tips = [pt for pt, dist in distances.items() if dist == max_dist]
    assert len(tips) == 1, "triangle tip must be unique"
    return tips[0]


def full_diagonal_cells(shape: tuple[int, int], tip: tuple[int, int], slope: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = slope
    r, c = tip
    while in_bounds(shape, r - dr, c - dc):
        r -= dr
        c -= dc
    pts = []
    while in_bounds(shape, r, c):
        pts.append((r, c))
        r += dr
        c += dc
    return pts


def parse_dominoes(grid: np.ndarray) -> list[Domino]:
    mask = (grid == RED) | (grid == GREEN)
    out: list[Domino] = []
    for comp in components(mask, connectivity=4):
        coords = [tuple(map(int, p)) for p in comp]
        if len(coords) != 2:
            raise AssertionError("red/green foreground must form only dominoes")
        colors = {int(grid[r, c]) for r, c in coords}
        if colors != {RED, GREEN}:
            raise AssertionError("each domino must contain one red and one green")
        red = next(pt for pt in coords if int(grid[pt]) == RED)
        green = next(pt for pt in coords if int(grid[pt]) == GREEN)
        out.append(Domino(coords=(coords[0], coords[1]), red=red, green=green))
    return out


def extract_instructions(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], list[TriangleInstruction], list[Domino]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    blue_coords = np.argwhere(g == BLUE)
    assert 5 <= len(blue_coords) <= 8, "blue diagonal must have length 5-8"
    slope = diagonal_orientation(blue_coords)
    blue_keys = [diagonal_key((int(r), int(c)), slope) for r, c in blue_coords]
    assert len(set(blue_keys)) == 1, "blue pixels must lie on one diagonal"

    if slope == (1, 1):
        ordered = sorted((int(r), int(c)) for r, c in blue_coords)
        assert all(ordered[i + 1] == (ordered[i][0] + 1, ordered[i][1] + 1) for i in range(len(ordered) - 1))
    else:
        ordered = sorted(((int(r), int(c)) for r, c in blue_coords), key=lambda rc: (rc[0], -rc[1]))
        assert all(ordered[i + 1] == (ordered[i][0] + 1, ordered[i][1] - 1) for i in range(len(ordered) - 1))

    triangle_comps = components(g == GREY, connectivity=4)
    assert len(triangle_comps) in (2, 3), "expected 2-3 grey triangles"

    dominoes = parse_dominoes(g)
    assert len(dominoes) == len(triangle_comps), "each triangle needs one domino"

    blue_key = blue_keys[0]
    instructions: list[TriangleInstruction] = []
    used_tip_rows: set[int] = set()
    used_line_keys: set[int] = set()

    for comp in triangle_comps:
        tip = triangle_tip(comp, blue_coords)
        assert tip[0] not in used_tip_rows, "tip rows must be unique"
        used_tip_rows.add(tip[0])

        tri_pts = [tuple(map(int, p)) for p in comp]
        assert component_distance(tri_pts, blue_coords) == 1, "triangle must touch blue by adjacency only"

        line_key = diagonal_key(tip, slope)
        assert line_key != blue_key, "new line must be offset from the blue segment"
        assert line_key not in used_line_keys, "parallel guide lines must be distinct"
        used_line_keys.add(line_key)

        domino_distances = [component_distance(tri_pts, domino.coords) for domino in dominoes]
        nearest = min(domino_distances)
        winners = [i for i, dist in enumerate(domino_distances) if dist == nearest]
        assert len(winners) == 1, "each triangle must have a unique nearest domino"
        domino = dominoes[winners[0]]
        assert 2 <= nearest <= 3, "domino must be nearby but not touching"

        red_distance = point_to_component_distance(domino.red, tri_pts)
        green_distance = point_to_component_distance(domino.green, tri_pts)
        assert red_distance != green_distance, "domino color choice must be unambiguous"
        line_color = RED if red_distance < green_distance else GREEN

        instructions.append(
            TriangleInstruction(
                coords=tuple(sorted(tri_pts)),
                tip=tip,
                line_color=line_color,
            )
        )

    instructions.sort(key=lambda item: (item.tip[0], item.tip[1]))
    return blue_coords, slope, instructions, dominoes


def render_output(grid: np.ndarray,
                  blue_coords: np.ndarray,
                  slope: tuple[int, int],
                  instructions: list[TriangleInstruction],
                  dominoes: list[Domino]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for instruction in instructions:
        for r, c in full_diagonal_cells(out.shape, instruction.tip, slope):
            out[r, c] = instruction.line_color
    out[out == GREY] = BLACK
    for r, c in blue_coords:
        out[int(r), int(c)] = BLUE
    for domino in dominoes:
        rr, rc = domino.red
        gr, gc = domino.green
        out[rr, rc] = RED
        out[gr, gc] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend parallel diagonals through grey triangle tips, color them from nearby dominoes, and erase the guides.

    Input structure:
    - The grid contains a blue diagonal segment on a black background.
    - Two or three grey right-triangle guide objects touch that diagonal without overlapping it.
    - Near each triangle there is one red-green orthogonal domino that stays in place.

    Rule:
    - Keep the blue diagonal unchanged and determine its slope.
    - For each grey triangle, find the unique triangle pixel farthest from the blue diagonal; that tip seeds a full diagonal parallel to the blue segment.
    - Pair the triangle with its unique nearest red-green domino, compare the red and green cells' Chebyshev distances to the triangle, and color the new diagonal with the nearer color.
    - Draw the new diagonals in ascending tip-row order, erase all remaining grey pixels, and preserve the original domino cells.

    Key insight:
    - The triangle answers where to draw a parallel diagonal, while the nearby domino answers whether that diagonal should be red or green.
    """
    blue_coords, slope, instructions, dominoes = extract_instructions(np.asarray(grid, dtype=int))
    return render_output(grid, blue_coords, slope, instructions, dominoes)


def _triangle_candidates(grid: np.ndarray,
                         blue_coords: np.ndarray,
                         slope: tuple[int, int],
                         used_tip_rows: set[int],
                         used_line_keys: set[int]) -> list[tuple[np.ndarray, tuple[int, int], int]]:
    blue_key = diagonal_key(tuple(map(int, blue_coords[0])), slope)
    candidates: list[tuple[np.ndarray, tuple[int, int], int]] = []
    for sprite in TRIANGLE_SPRITES:
        h, w = sprite.shape
        rel_cells = [tuple(map(int, p)) for p in np.argwhere(sprite == GREY)]
        top_lefts: set[tuple[int, int]] = set()
        for br, bc in blue_coords:
            br = int(br)
            bc = int(bc)
            for sr, sc in rel_cells:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        r0 = br + dr - sr
                        c0 = bc + dc - sc
                        if 0 <= r0 <= grid.shape[0] - h and 0 <= c0 <= grid.shape[1] - w:
                            top_lefts.add((r0, c0))
        for r0, c0 in top_lefts:
            cells = sprite_cells(sprite, r0, c0)
            if any(int(grid[r, c]) != BLACK for r, c in cells):
                continue
            if not any(any(in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BLUE for nr, nc in neighbors8(r, c)) for r, c in cells):
                continue
            if any(any(in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREY for nr, nc in neighbors8(r, c)) for r, c in cells):
                continue
            comp = np.array(cells, dtype=int)
            if component_distance(cells, blue_coords) != 1:
                continue
            try:
                tip = triangle_tip(comp, blue_coords)
            except AssertionError:
                continue
            line_key = diagonal_key(tip, slope)
            if tip[0] in used_tip_rows or line_key in used_line_keys or line_key == blue_key:
                continue
            candidates.append((comp, tip, line_key))
    random.shuffle(candidates)
    return candidates


def _occupied_too_close(grid: np.ndarray, cells: list[tuple[int, int]], allowed_colors: set[int]) -> bool:
    for r, c in cells:
        for nr, nc in neighbors8(r, c):
            if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) not in allowed_colors:
                return True
    return False


def _domino_candidates(grid: np.ndarray,
                       triangles: list[np.ndarray],
                       target_index: int,
                       desired_color: int) -> list[tuple[tuple[int, int], tuple[int, int], int, int]]:
    tri = [tuple(map(int, p)) for p in triangles[target_index]]
    candidates: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            for dr, dc in ((1, 0), (0, 1)):
                a = (r, c)
                b = (r + dr, c + dc)
                if not in_bounds(grid.shape, *b):
                    continue
                pair = tuple(sorted((a, b)))
                if pair in seen:
                    continue
                seen.add(pair)
                if any(int(grid[rr, cc]) != BLACK for rr, cc in pair):
                    continue
                d_self = component_distance(pair, tri)
                if not (2 <= d_self <= 3):
                    continue
                other_ds = [
                    component_distance(pair, [tuple(map(int, p)) for p in triangles[j]])
                    for j in range(len(triangles))
                    if j != target_index
                ]
                if other_ds and d_self >= min(other_ds):
                    continue
                if _occupied_too_close(grid, list(pair), {BLACK, BLUE}):
                    continue
                a_dist = point_to_component_distance(a, tri)
                b_dist = point_to_component_distance(b, tri)
                if a_dist == b_dist:
                    continue
                closer, farther = (a, b) if a_dist < b_dist else (b, a)
                if desired_color == RED:
                    red_cell, green_cell = closer, farther
                else:
                    green_cell, red_cell = closer, farther
                candidates.append((red_cell, green_cell, d_self, abs(a_dist - b_dist)))
    candidates.sort(key=lambda item: (item[2], -item[3], item[0], item[1]))
    return candidates


def _force_color_mix(count: int) -> list[int]:
    colors = [random.choice((RED, GREEN)) for _ in range(count)]
    if count >= 2 and random.random() < 0.7:
        colors[0] = RED
        colors[1] = GREEN
        random.shuffle(colors)
    return colors


def _build_attempt() -> np.ndarray | None:
    height = random.randint(14, 18)
    width = random.randint(14, 18)
    grid = np.zeros((height, width), dtype=int)

    length = random.randint(5, 8)
    start_r = random.randint(2, height - length - 3)
    start_c = random.randint(2, width - length - 3)
    for i in range(length):
        grid[start_r + i, start_c + i] = BLUE
    blue_coords = np.argwhere(grid == BLUE)
    slope = (1, 1)

    triangle_count = random.choice((2, 3))
    triangles: list[np.ndarray] = []
    used_tip_rows: set[int] = set()
    used_line_keys: set[int] = set()
    for _ in range(triangle_count):
        candidates = _triangle_candidates(grid, blue_coords, slope, used_tip_rows, used_line_keys)
        if not candidates:
            return None
        comp, tip, line_key = random.choice(candidates[: min(40, len(candidates))])
        for r, c in comp:
            grid[int(r), int(c)] = GREY
        triangles.append(comp.copy())
        used_tip_rows.add(tip[0])
        used_line_keys.add(line_key)

    desired_colors = _force_color_mix(triangle_count)
    order = list(range(triangle_count))
    order.sort(key=lambda idx: len(_domino_candidates(grid, triangles, idx, desired_colors[idx])))
    for idx in order:
        candidates = _domino_candidates(grid, triangles, idx, desired_colors[idx])
        if not candidates:
            return None
        red_cell, green_cell, _dist, _gap = random.choice(candidates[: min(12, len(candidates))])
        grid[red_cell] = RED
        grid[green_cell] = GREEN

    if random.random() < 0.5:
        grid = grid[:, ::-1].copy()

    return grid


def _semantic_validate_input(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], list[TriangleInstruction], list[Domino]]:
    colors = set(map(int, np.unique(grid)))
    assert colors <= {BLACK, BLUE, RED, GREEN, GREY}
    blue_coords, slope, instructions, dominoes = extract_instructions(grid)
    assert len(instructions) == len(dominoes)
    for domino in dominoes:
        assert chebyshev(domino.red, domino.green) == 1
        assert domino.red[0] == domino.green[0] or domino.red[1] == domino.green[1]
    for instruction in instructions:
        line_cells = full_diagonal_cells(grid.shape, instruction.tip, slope)
        assert len(line_cells) >= 4
    return blue_coords, slope, instructions, dominoes


def _semantic_validate_output(grid: np.ndarray,
                              out: np.ndarray,
                              blue_coords: np.ndarray,
                              slope: tuple[int, int],
                              instructions: list[TriangleInstruction],
                              dominoes: list[Domino]) -> None:
    assert out.shape == grid.shape
    assert GREY not in set(map(int, np.unique(out)))

    protected: set[tuple[int, int]] = {tuple(map(int, p)) for p in blue_coords}
    protected |= {domino.red for domino in dominoes}
    protected |= {domino.green for domino in dominoes}

    layered: dict[tuple[int, int], int] = {}
    for instruction in instructions:
        for cell in full_diagonal_cells(grid.shape, instruction.tip, slope):
            layered[cell] = instruction.line_color

    for r, c in protected:
        assert int(out[r, c]) == int(grid[r, c])

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            cell = (r, c)
            if cell in protected:
                continue
            if cell in layered:
                assert int(out[r, c]) == layered[cell]
            elif int(grid[r, c]) == GREY:
                assert int(out[r, c]) == BLACK
            else:
                assert int(out[r, c]) == int(grid[r, c])


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_attempt()
        if grid is None:
            continue
        try:
            _semantic_validate_input(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
