from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

BODY_COLORS = (BLUE, RED, GREEN)
SIZE_OPTIONS = (
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 4),
    (4, 3),
    (2, 5),
    (5, 2),
)
GEN_RETRY_LIMIT = 400


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in deltas:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _cavity_from_component(local_grid: np.ndarray, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if r0 == 0 or c0 == 0 or r1 == local_grid.shape[0] - 1 or c1 == local_grid.shape[1] - 1:
        return None
    if coords.shape[0] != h * w:
        return None
    if not np.all(local_grid[r0 : r1 + 1, c0 : c1 + 1] == BLACK):
        return None
    if not np.all(local_grid[r0 - 1, c0 : c1 + 1] == GREY):
        return None
    if not np.all(local_grid[r1 + 1, c0 : c1 + 1] == GREY):
        return None
    if not np.all(local_grid[r0 : r1 + 1, c0 - 1] == GREY):
        return None
    if not np.all(local_grid[r0 : r1 + 1, c1 + 1] == GREY):
        return None
    return {"bbox": (r0, c0, r1, c1), "size": (h, w)}


def _parse_grid(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    grey_coords = np.argwhere(g == GREY)
    assert grey_coords.size > 0, "template frame missing"
    frame_bbox = _bbox(grey_coords)
    r0, c0, r1, c1 = frame_bbox
    frame_crop = g[r0 : r1 + 1, c0 : c1 + 1]

    cavities: list[dict] = []
    for comp in _components(frame_crop == BLACK, connectivity=4):
        cavity = _cavity_from_component(frame_crop, comp)
        if cavity is None:
            continue
        cr0, cc0, cr1, cc1 = cavity["bbox"]
        cavity["bbox"] = (r0 + cr0, c0 + cc0, r0 + cr1, c0 + cc1)
        cavities.append(cavity)

    assert len(cavities) == 3, f"expected 3 cavities, found {len(cavities)}"
    cavity_sizes = [c["size"] for c in cavities]
    assert len(set(cavity_sizes)) == 3, "cavity sizes must be unique"

    outside_mask = (g != BLACK) & (g != GREY)
    outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    sprites: list[dict] = []
    for comp in _components(outside_mask, connectivity=4):
        sr0, sc0, sr1, sc1 = _bbox(comp)
        patch = g[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        assert np.all((patch != BLACK) & (patch != GREY)), "sprite bboxes must be fully filled"
        marker_coords = np.argwhere(patch == YELLOW)
        assert marker_coords.shape == (1, 2), "each sprite must contain exactly one yellow marker"
        body_colors = sorted(set(map(int, patch.flat)) - {YELLOW})
        assert len(body_colors) == 1, "each sprite must have exactly one body color"
        body_color = int(body_colors[0])
        sprites.append(
            {
                "bbox": (sr0, sc0, sr1, sc1),
                "coords": comp,
                "size": patch.shape,
                "patch": patch,
                "marker_offset": (int(marker_coords[0, 0]), int(marker_coords[0, 1])),
                "body_color": body_color,
            }
        )

    assert len(sprites) == 3, f"expected 3 sprites, found {len(sprites)}"
    sprite_sizes = [tuple(map(int, s["size"])) for s in sprites]
    assert len(set(sprite_sizes)) == 3, "sprite sizes must be unique"
    assert set(sprite_sizes) == set(cavity_sizes), "sprite sizes must match cavity sizes"
    return frame_bbox, cavities, sprites


def _embed_sprites(grid: np.ndarray, cavities: list[dict], sprites: list[dict]) -> tuple[np.ndarray, list[dict]]:
    out = np.asarray(grid, dtype=int).copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = BLACK

    sprite_by_size = {tuple(map(int, sprite["size"])): sprite for sprite in sprites}
    markers: list[dict] = []
    for cavity in cavities:
        sprite = sprite_by_size[cavity["size"]]
        r0, c0, r1, c1 = cavity["bbox"]
        patch = sprite["patch"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert patch.shape == (h, w)
        out[r0 : r1 + 1, c0 : c1 + 1] = patch
        mr, mc = sprite["marker_offset"]
        markers.append({"coord": (r0 + mr, c0 + mc), "body_color": sprite["body_color"]})
    return out, markers


def _chebyshev_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    step_r = _sign(er - sr)
    step_c = _sign(ec - sc)
    row_dist = abs(er - sr)
    col_dist = abs(ec - sc)

    r, c = sr, sc
    path: list[tuple[int, int]] = []
    for _ in range(min(row_dist, col_dist)):
        r += step_r
        c += step_c
        path.append((r, c))
    if row_dist > col_dist:
        for _ in range(row_dist - col_dist):
            r += step_r
            path.append((r, c))
    else:
        for _ in range(col_dist - row_dist):
            c += step_c
            path.append((r, c))
    assert path and path[-1] == end
    return path


def _paint_paths(out: np.ndarray, markers: list[dict]) -> list[int]:
    ordered = sorted(markers, key=lambda item: (item["coord"][0], item["coord"][1]))
    counts: list[int] = []
    for i in range(len(ordered) - 1):
        start = ordered[i]["coord"]
        end = ordered[i + 1]["coord"]
        color = ordered[i]["body_color"]
        painted = 0
        for r, c in _chebyshev_path(start, end):
            if out[r, c] == BLACK:
                out[r, c] = color
                painted += 1
        counts.append(painted)
    return counts


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[int]]:
    _frame_bbox, cavities, sprites = _parse_grid(grid)
    out, markers = _embed_sprites(grid, cavities, sprites)
    counts = _paint_paths(out, markers)
    return out, markers, counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Embed each uniquely size-matched sprite into its cavity, then connect the relocated yellow markers with body-colored Chebyshev paths.

    Input structure:
    - A black background contains one grey template region whose bbox encloses exactly three black rectangular cavities.
    - Outside the template bbox sit three filled rectangular sprites of distinct sizes and body colors, each with exactly one yellow marker pixel.
    - Additional black openings may appear inside the template bbox, but only the fully grey-enclosed black rectangles are cavities.

    Rule:
    - Find the grey template bbox and the three internal black rectangles that are completely surrounded by grey.
    - Find the three external sprites as 4-connected non-black/non-grey components, and match each sprite to the unique cavity with the same bbox size.
    - Move each whole sprite into its cavity by aligning top-left corners, which relocates the yellow marker pixels.
    - Sort the relocated markers by row and connect consecutive pairs with Chebyshev-shortest king-move paths: diagonal steps first, then any remaining straight steps.
    - Color each path with the body color of its start sprite, drawing only onto black background cells and leaving non-background cells unchanged.

    Key insight:
    - The connection endpoints are not the input markers' original positions; they are the markers after size-matched embedding, so the solver must complete the inlay step before the path step is even well defined.
    """
    out, _markers, _counts = _solve(np.asarray(grid, dtype=int))
    return out


def _distribute(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _make_sprite(size: tuple[int, int], body_color: int) -> np.ndarray:
    h, w = size
    patch = np.full((h, w), body_color, dtype=int)
    mr = random.randrange(h)
    mc = random.randrange(w)
    patch[mr, mc] = YELLOW
    return patch


def _choose_template_geometry() -> tuple[int, int, int, int, str, int, int]:
    layout = random.choice(["right", "bottom"])
    template_h = random.randint(12, 16)
    template_w = random.randint(16, 19)
    if layout == "right":
        grid_h = random.randint(max(20, template_h + 4), 27)
        grid_w = random.randint(max(24, template_w + 7), 29)
        top = random.randint(2, grid_h - template_h - 2)
        left = random.randint(2, 4)
    else:
        grid_h = random.randint(max(24, template_h + 7), 29)
        grid_w = random.randint(max(20, template_w + 4), 27)
        top = random.randint(2, 4)
        left = random.randint(2, grid_w - template_w - 2)
    return grid_h, grid_w, template_h, template_w, layout, top, left


def _place_cavities(template_h: int, template_w: int) -> tuple[list[dict], tuple[int, int]]:
    corridor_w = random.choice([1, 2])
    corridor_c0 = (template_w - corridor_w) // 2
    corridor = (corridor_c0, corridor_c0 + corridor_w - 1)

    sizes = random.sample(SIZE_OPTIONS, 3)
    heights = [h for h, _w in sizes]
    needed_rows = 2 + sum(heights) + 2
    if needed_rows > template_h:
        raise ValueError("template too short")
    extras = _distribute(template_h - needed_rows, 4)

    side_pattern = random.choice([["left", "right", "left"], ["right", "left", "right"]])
    row = 1 + extras[0]
    cavities: list[dict] = []
    for i, ((h, w), side) in enumerate(zip(sizes, side_pattern)):
        if side == "left":
            c_min = 1
            c_max = corridor[0] - w - 1
        else:
            c_min = corridor[1] + 2
            c_max = template_w - w - 1
        if c_min > c_max:
            raise ValueError("template too narrow for cavity")
        col = random.randint(c_min, c_max)
        cavities.append({"bbox": (row, col, row + h - 1, col + w - 1), "size": (h, w), "side": side})
        row += h + 1 + extras[i + 1]
    return cavities, corridor


def _place_external_sprites(grid: np.ndarray, patches: list[np.ndarray], frame_bbox: tuple[int, int, int, int], layout: str) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = frame_bbox
    order = sorted(range(len(patches)), key=lambda i: -(patches[i].shape[0] * patches[i].shape[1]))
    occupied = grid != BLACK

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        if layout == "right":
            row_min = 1
            row_max = h - ph - 1
            col_min = c1 + 2
            col_max = w - pw - 1
        else:
            row_min = r1 + 2
            row_max = h - ph - 1
            col_min = 1
            col_max = w - pw - 1
        if row_min > row_max or col_min > col_max:
            return False

        candidates: list[tuple[int, int]] = []
        for rr in range(row_min, row_max + 1):
            for cc in range(col_min, col_max + 1):
                rr0 = max(0, rr - 1)
                cc0 = max(0, cc - 1)
                rr1 = min(h, rr + ph + 1)
                cc1 = min(w, cc + pw + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((rr, cc))
        if not candidates:
            return False
        rr, cc = random.choice(candidates)
        grid[rr : rr + ph, cc : cc + pw] = patch
        occupied[rr : rr + ph, cc : cc + pw] = True
    return True


def _attempt_generate() -> np.ndarray | None:
    grid_h, grid_w, template_h, template_w, layout, top, left = _choose_template_geometry()
    grid = np.zeros((grid_h, grid_w), dtype=int)

    try:
        cavities_local, corridor = _place_cavities(template_h, template_w)
    except ValueError:
        return None

    body_colors = list(BODY_COLORS)
    random.shuffle(body_colors)
    sprites_local: list[dict] = []
    for cavity, body_color in zip(cavities_local, body_colors):
        patch = _make_sprite(cavity["size"], body_color)
        sprites_local.append({"size": cavity["size"], "patch": patch, "body_color": body_color})

    frame_bbox = (top, left, top + template_h - 1, left + template_w - 1)
    r0, c0, r1, c1 = frame_bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    corr0 = c0 + corridor[0]
    corr1 = c0 + corridor[1]
    grid[r0 : r1 + 1, corr0 : corr1 + 1] = BLACK
    for cavity in cavities_local:
        cr0, cc0, cr1, cc1 = cavity["bbox"]
        grid[r0 + cr0 : r0 + cr1 + 1, c0 + cc0 : c0 + cc1 + 1] = BLACK

    patches = [sprite["patch"] for sprite in sprites_local]
    if not _place_external_sprites(grid, patches, frame_bbox, layout):
        return None

    try:
        parsed_bbox, cavities, sprites = _parse_grid(grid)
        assert parsed_bbox == frame_bbox
        embedded, markers = _embed_sprites(grid, cavities, sprites)
        before_path = embedded.copy()
        counts = _paint_paths(embedded, markers)
        marker_rows = [item["coord"][0] for item in markers]
        assert len(set(marker_rows)) == 3, "marker row order must be unique"
        assert all(count > 0 for count in counts), "each path segment must paint visible background"
        assert not np.array_equal(before_path, embedded), "path step must change the output"
        assert np.array_equal(main(grid), embedded), "generator instance must match solver"
        return grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
