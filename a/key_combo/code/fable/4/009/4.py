from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

_COLORS = tuple(range(10))
_GEN_RETRIES = 600


def _background(g: np.ndarray) -> int:
    return int(Counter(map(int, g.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_for_color(g: np.ndarray, color: int) -> list[np.ndarray]:
    height, width = map(int, g.shape)
    mask = g == int(color)
    seen = np.zeros(g.shape, dtype=bool)
    comps: list[np.ndarray] = []

    for start_r, start_c in np.argwhere(mask):
        sr, sc = int(start_r), int(start_c)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[Coord] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _is_rectangular_frame(coords: np.ndarray, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    cells = {(int(r), int(c)) for r, c in coords}
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    return cells == expected


def _outside_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.ones(shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    return mask


def _ranked_scatter_colors(g: np.ndarray, bg: int, frame_color: int, box: BBox) -> list[int]:
    values = [
        int(v)
        for v in g[_outside_mask(g.shape, box)].ravel()
        if int(v) not in (int(bg), int(frame_color))
    ]
    counts = Counter(values)
    assert counts, "expected colored scatter pixels outside the frame"
    assert len(set(counts.values())) == len(counts), "scatter color frequencies must be distinct"
    return sorted(counts, key=lambda color: counts[color], reverse=True)


def _parse_scene(g: np.ndarray) -> dict[str, object]:
    bg = _background(g)
    candidates: list[dict[str, object]] = []

    for color in set(map(int, g.ravel())) - {bg}:
        for coords in _components_for_color(g, color):
            box = _bbox(coords)
            if not _is_rectangular_frame(coords, box):
                continue

            r0, c0, r1, c1 = box
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            seed_local = np.argwhere(inner != int(bg))
            if seed_local.shape[0] != 1:
                continue

            sr, sc = map(int, seed_local[0])
            seed_abs = (r0 + 1 + sr, c0 + 1 + sc)
            seed_color = int(g[seed_abs])
            if seed_color == int(color):
                continue

            ranked = _ranked_scatter_colors(g, bg, int(color), box)
            if ranked[0] != seed_color:
                continue

            candidates.append(
                {
                    "bg": int(bg),
                    "frame_color": int(color),
                    "box": box,
                    "seed": seed_abs,
                    "ranked": ranked,
                }
            )

    assert len(candidates) == 1, "expected one uniquely identifiable framed seed cavity"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the framed seed into Chebyshev-distance square rings colored by outside scatter frequency rank.

    Input structure:
    - A dominant background contains one one-pixel-thick rectangular frame of a
      non-background color.
    - The frame interior is otherwise background and contains exactly one seed
      pixel whose color also appears among the outside scatter.
    - Outside the whole frame rectangle are isolated scatter pixels in several
      colors; their color frequencies are all distinct, and the seed color is
      the most frequent outside scatter color.

    Rule:
    - Count only non-background, non-frame scatter pixels outside the frame
      rectangle and rank their colors by decreasing frequency.
    - For each cell inside the frame, compute its Chebyshev distance d from the
      seed pixel and paint it with the color ranked d+1, if that rank exists.
    - Leave frame pixels and all outside scatter pixels exactly as they were;
      interior cells whose distance has no ranked color remain background.

    Key insight:
    - The scatter positions are decoys: their counts form the palette, and the
      seed-centered Chebyshev distance is the index into that frequency ranking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    scene = _parse_scene(g)
    bg = int(scene["bg"])
    r0, c0, r1, c1 = scene["box"]  # type: ignore[assignment]
    seed_r, seed_c = scene["seed"]  # type: ignore[assignment]
    ranked = [int(color) for color in scene["ranked"]]  # type: ignore[iteration-over-annotation]

    out = g.copy()
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            dist = max(abs(r - int(seed_r)), abs(c - int(seed_c)))
            out[r, c] = ranked[dist] if dist < len(ranked) else bg
    return out


# --------------- generator ---------------


def _frame_positions(height: int, width: int, frame_h: int, frame_w: int, side: str) -> list[Coord]:
    positions: list[Coord] = []
    for r0 in range(1, height - frame_h):
        for c0 in range(1, width - frame_w):
            top = r0
            bottom = height - (r0 + frame_h)
            left = c0
            right = width - (c0 + frame_w)
            if side == "left" and left <= 2 and right >= 5:
                positions.append((r0, c0))
            elif side == "right" and right <= 2 and left >= 5:
                positions.append((r0, c0))
            elif side == "top" and top <= 2 and bottom >= 5:
                positions.append((r0, c0))
            elif side == "bottom" and bottom <= 2 and top >= 5:
                positions.append((r0, c0))
    return positions


def _draw_frame(g: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    g[r0 : r1 + 1, c0] = int(color)
    g[r0 : r1 + 1, c1] = int(color)
    g[r0, c0 : c1 + 1] = int(color)
    g[r1, c0 : c1 + 1] = int(color)


def _scatter_candidates(shape: tuple[int, int], box: BBox) -> list[Coord]:
    height, width = map(int, shape)
    r0, c0, r1, c1 = box
    cells: list[Coord] = []
    for r in range(height):
        for c in range(width):
            if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
                continue
            cells.append((r, c))
    return cells


def _place_isolated_points(g: np.ndarray, candidates: list[Coord], ranked: list[int], counts: list[int]) -> bool:
    height, width = map(int, g.shape)
    free = set(candidates)
    for color, count in zip(ranked, counts):
        for _ in range(int(count)):
            if not free:
                return False
            r, c = random.choice(tuple(free))
            g[r, c] = int(color)
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if 0 <= rr < height and 0 <= cc < width:
                        free.discard((rr, cc))
    return True


def _sample_rank_count(max_distance: int, available_colors: int) -> int:
    max_rank = min(int(max_distance) + 1, int(available_colors))
    if max_rank < 2:
        return 0
    if max_distance >= 3 and random.random() < 0.45:
        return random.randint(2, min(max_rank, max_distance))
    return random.randint(2, max_rank)


def _off_center_seed(inner_h: int, inner_w: int) -> Coord:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    centers = {(inner_h // 2, inner_w // 2)}
    if inner_h % 2 == 0:
        centers.add((inner_h // 2 - 1, inner_w // 2))
    if inner_w % 2 == 0:
        centers.add((inner_h // 2, inner_w // 2 - 1))
    if inner_h % 2 == 0 and inner_w % 2 == 0:
        centers.add((inner_h // 2 - 1, inner_w // 2 - 1))
    if random.random() < 0.7:
        choices = [cell for cell in cells if cell not in centers]
        return random.choice(choices)
    return random.choice(cells)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(17, 30)
    width = random.randint(17, 30)
    bg = random.choice(_COLORS)
    grid = np.full((height, width), int(bg), dtype=int)

    frame_color = random.choice([color for color in _COLORS if color != bg])
    palette = [color for color in _COLORS if color not in (bg, frame_color)]

    inner_h = random.randint(5, min(10, height - 7))
    inner_w = random.randint(5, min(10, width - 7))
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    sides = ["left", "right", "top", "bottom"]
    side = random.choice(sides)
    positions = _frame_positions(height, width, frame_h, frame_w, side)
    if not positions:
        return None
    r0, c0 = random.choice(positions)
    box = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
    _draw_frame(grid, box, frame_color)

    seed_dr, seed_dc = _off_center_seed(inner_h, inner_w)
    seed = (r0 + 1 + seed_dr, c0 + 1 + seed_dc)
    max_distance = max(seed_dr, inner_h - 1 - seed_dr, seed_dc, inner_w - 1 - seed_dc)
    rank_count = _sample_rank_count(max_distance, len(palette))
    if rank_count < 2:
        return None

    ranked = random.sample(palette, rank_count)
    grid[seed] = int(ranked[0])

    count_pool = list(range(3, 4 + rank_count * 2))
    if len(count_pool) < rank_count:
        return None
    counts = sorted(random.sample(count_pool, rank_count), reverse=True)

    candidates = _scatter_candidates(grid.shape, box)
    random.shuffle(candidates)
    if not _place_isolated_points(grid, candidates, ranked, counts):
        return None

    try:
        scene = _parse_scene(grid)
        out = main(grid)
    except AssertionError:
        return None

    if scene["box"] != box or tuple(scene["seed"]) != seed:  # type: ignore[arg-type]
        return None
    if [int(c) for c in scene["ranked"]] != ranked:  # type: ignore[iteration-over-annotation]
        return None
    if np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
