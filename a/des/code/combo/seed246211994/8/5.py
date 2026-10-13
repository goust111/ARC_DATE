from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass(frozen=True)
class Sprite:
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    border_color: int
    center_color: int


@dataclass(frozen=True)
class Scene:
    bg: int
    sprites: tuple[Sprite, ...]
    target_ids: tuple[int, int]
    replacements: tuple[int, int]


def neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _parse_sprite(grid: np.ndarray, coords: np.ndarray) -> Sprite | None:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or h % 2 == 0 or w % 2 == 0:
        return None
    if int(coords.shape[0]) != h * w:
        return None

    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    cr = h // 2
    cc = w // 2
    center_color = int(patch[cr, cc])
    border_color = int(patch[0, 0])
    if center_color == border_color:
        return None

    for rr in range(h):
        for cc0 in range(w):
            want = center_color if (rr, cc0) == (cr, cc) else border_color
            if int(patch[rr, cc0]) != want:
                return None

    return Sprite(
        bbox=(r0, c0, r1, c1),
        center=(r0 + cr, c0 + cc),
        border_color=border_color,
        center_color=center_color,
    )


def _path_endpoints(coords: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    cells = {(int(r), int(c)) for r, c in coords}
    endpoints: list[tuple[int, int]] = []
    for r, c in cells:
        deg = sum((nr, nc) in cells for nr, nc in neighbors4(r, c, h, w))
        if deg == 1:
            endpoints.append((r, c))
    return endpoints


def _analyze_scene(grid: np.ndarray) -> Scene:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    sprites: list[Sprite] = []
    for coords in components_mask((g != bg) & (g != GREY)):
        if int(coords.shape[0]) == 1:
            continue
        sprite = _parse_sprite(g, coords)
        assert sprite is not None, "all multi-cell non-grey objects must be sprites"
        sprites.append(sprite)
    assert len(sprites) >= 4, "scene needs at least two targets and two sources"

    cell_to_sprite: dict[tuple[int, int], int] = {}
    for sid, sprite in enumerate(sprites):
        r0, c0, r1, c1 = sprite.bbox
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                cell_to_sprite[(rr, cc)] = sid

    path_candidates: list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]] = []
    for coords in components_mask(g == GREY):
        endpoints = _path_endpoints(coords, g.shape)
        if len(endpoints) != 2:
            continue
        touched: list[int] = []
        ok = True
        for ep in endpoints:
            adj = {
                cell_to_sprite[(nr, nc)]
                for nr, nc in neighbors4(ep[0], ep[1], h, w)
                if (nr, nc) in cell_to_sprite
            }
            if len(adj) != 1:
                ok = False
                break
            touched.append(next(iter(adj)))
        if ok and touched[0] != touched[1]:
            path_candidates.append((coords, (touched[0], touched[1]), tuple(endpoints)))

    assert len(path_candidates) == 1, "expected exactly one valid grey path"
    _, target_ids, _ = path_candidates[0]

    replacements: list[int] = []
    for tid in target_ids:
        key = sprites[tid].center_color
        matches = [sid for sid, sprite in enumerate(sprites) if sid != tid and sprite.border_color == key]
        assert len(matches) == 1, "each target center color must match exactly one other sprite border color"
        replacements.append(sprites[matches[0]].center_color)

    return Scene(
        bg=bg,
        sprites=tuple(sprites),
        target_ids=target_ids,
        replacements=(replacements[0], replacements[1]),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the two sprites touched by the grey path endpoints, replace each kept sprite's
    center by the center of the unique sprite whose border color matches that target center,
    and delete every other non-grey object.
    """
    g = np.asarray(grid, dtype=int)
    scene = _analyze_scene(g)

    out = np.full_like(g, scene.bg)
    out[g == GREY] = GREY

    for tid, new_center in zip(scene.target_ids, scene.replacements, strict=True):
        sprite = scene.sprites[tid]
        r0, c0, r1, c1 = sprite.bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1]
        cr, cc = sprite.center
        out[cr, cc] = new_center

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 25)
GRID_W_RANGE = (18, 25)
TOTAL_SPRITES_CHOICES = (4, 5, 6)
SPRITE_SHAPES = ((3, 3), (3, 5), (5, 3), (5, 5))
CONTROL_COUNT_RANGE = (2, 4)
MAX_ATTEMPTS = 500


def _pick(rng: np.random.Generator, seq: tuple | list):
    return seq[int(rng.integers(0, len(seq)))]


def _sprite_patch(h: int, w: int, border: int, center: int) -> np.ndarray:
    patch = np.full((h, w), border, dtype=int)
    patch[h // 2, w // 2] = center
    return patch


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _can_place_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, gap: int) -> bool:
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - gap)
    rr1 = min(H, r0 + h + gap)
    cc0 = max(0, c0 - gap)
    cc1 = min(W, c0 + w + gap)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _endpoint_options(spec: dict[str, int], H: int, W: int) -> list[tuple[str, tuple[int, int]]]:
    r0, c0, h, w = spec["r0"], spec["c0"], spec["h"], spec["w"]
    rm = r0 + h // 2
    cm = c0 + w // 2
    options = [
        ("U", (r0 - 1, cm)),
        ("D", (r0 + h, cm)),
        ("L", (rm, c0 - 1)),
        ("R", (rm, c0 + w)),
    ]
    return [(side, cell) for side, cell in options if 0 <= cell[0] < H and 0 <= cell[1] < W]


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("segment must be axis-aligned")


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    filtered = [points[0]]
    for p in points[1:]:
        if p != filtered[-1]:
            filtered.append(p)
    cells: list[tuple[int, int]] = []
    for i in range(len(filtered) - 1):
        seg = _segment_cells(filtered[i], filtered[i + 1])
        if i > 0:
            seg = seg[1:]
        cells.extend(seg)
    if len(set(cells)) != len(cells):
        return None
    return cells


def _path_block_mask(targets: list[dict[str, int]], H: int, W: int) -> np.ndarray:
    blocked = np.zeros((H, W), dtype=bool)
    for spec in targets:
        r0, c0, h, w = spec["r0"], spec["c0"], spec["h"], spec["w"]
        rr0 = max(0, r0 - 1)
        rr1 = min(H, r0 + h + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(W, c0 + w + 1)
        blocked[rr0:rr1, cc0:cc1] = True
    return blocked


def _build_path(
    rng: np.random.Generator,
    targets: list[dict[str, int]],
    H: int,
    W: int,
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]] | None:
    blocked = _path_block_mask(targets, H, W)
    blocked[start] = False
    blocked[end] = False

    candidates: list[list[tuple[int, int]]] = [
        [start, (start[0], end[1]), end],
        [start, (end[0], start[1]), end],
    ]

    row_order = rng.permutation(np.arange(1, H - 1))
    col_order = rng.permutation(np.arange(1, W - 1))
    for row in row_order[:8]:
        candidates.append([start, (int(row), start[1]), (int(row), end[1]), end])
    for col in col_order[:8]:
        candidates.append([start, (start[0], int(col)), (end[0], int(col)), end])

    for idx in rng.permutation(len(candidates)):
        cells = _polyline_cells(candidates[int(idx)])
        if cells is None:
            continue
        if any(blocked[r, c] for r, c in cells[1:-1]):
            continue
        return cells
    return None


def _place_targets_and_path(rng: np.random.Generator, H: int, W: int) -> tuple[list[dict[str, int]], list[tuple[int, int]]] | None:
    for _ in range(200):
        occupied = np.zeros((H, W), dtype=bool)
        targets: list[dict[str, int]] = []
        for _idx in range(2):
            placed = False
            for _ in range(120):
                h, w = _pick(rng, SPRITE_SHAPES)
                r0 = int(rng.integers(2, H - h - 1))
                c0 = int(rng.integers(2, W - w - 1))
                if not _can_place_bbox(occupied, r0, c0, h, w, gap=2):
                    continue
                targets.append({"r0": r0, "c0": c0, "h": h, "w": w})
                occupied[r0 : r0 + h, c0 : c0 + w] = True
                placed = True
                break
            if not placed:
                break
        if len(targets) != 2:
            continue

        c0r = targets[0]["r0"] + targets[0]["h"] // 2
        c0c = targets[0]["c0"] + targets[0]["w"] // 2
        c1r = targets[1]["r0"] + targets[1]["h"] // 2
        c1c = targets[1]["c0"] + targets[1]["w"] // 2
        if abs(c0r - c1r) + abs(c0c - c1c) < min(H, W) // 2:
            continue

        opts0 = _endpoint_options(targets[0], H, W)
        opts1 = _endpoint_options(targets[1], H, W)
        pairings = [(a, b) for a in opts0 for b in opts1]
        for idx in rng.permutation(len(pairings)):
            (_, start), (_, end) = pairings[int(idx)]
            path = _build_path(rng, targets, H, W, start, end)
            if path is not None and len(path) >= 4:
                return targets, path
    return None


def _control_candidates(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    path_set = {(int(r), int(c)) for r, c in np.argwhere(grid == GREY)}
    candidates: list[tuple[int, int]] = []
    for r, c in path_set:
        if sum((nr, nc) in path_set for nr, nc in neighbors4(r, c, H, W)) < 2:
            continue
        for nr, nc in neighbors4(r, c, H, W):
            if int(grid[nr, nc]) != bg:
                continue
            grey_neighbors = 0
            other_neighbors = 0
            for rr, cc in neighbors4(nr, nc, H, W):
                if int(grid[rr, cc]) == GREY:
                    grey_neighbors += 1
                elif int(grid[rr, cc]) != bg:
                    other_neighbors += 1
            if grey_neighbors == 1 and other_neighbors == 0:
                candidates.append((nr, nc))
    return list(dict.fromkeys(candidates))


def _select_control_points(
    rng: np.random.Generator,
    candidates: list[tuple[int, int]],
    want: int,
) -> list[tuple[int, int]] | None:
    selected: list[tuple[int, int]] = []
    for idx in rng.permutation(len(candidates)):
        cell = candidates[int(idx)]
        if any(abs(cell[0] - r) + abs(cell[1] - c) == 1 for r, c in selected):
            continue
        selected.append(cell)
        if len(selected) == want:
            return selected
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))

    bg_choices = [c for c in range(10) if c != GREY]
    bg = int(_pick(rng, bg_choices))
    palette = [c for c in range(10) if c not in {bg, GREY}]

    total_sprites = int(_pick(rng, TOTAL_SPRITES_CHOICES))
    if total_sprites > len(palette):
        return None

    border_colors = [int(v) for v in rng.choice(np.array(palette, dtype=int), size=total_sprites, replace=False)]
    source_ids = [int(v) for v in rng.choice(np.arange(2, total_sprites), size=2, replace=False)]

    center_colors: list[int] = [0] * total_sprites
    center_colors[0] = border_colors[source_ids[0]]
    center_colors[1] = border_colors[source_ids[1]]
    source_to_target = {source_ids[0]: 0, source_ids[1]: 1}
    for idx in range(2, total_sprites):
        choices = [c for c in palette if c != border_colors[idx]]
        if idx in source_to_target:
            tgt = source_to_target[idx]
            choices = [c for c in choices if c != border_colors[tgt]]
        center_colors[idx] = int(_pick(rng, choices))

    placed = _place_targets_and_path(rng, H, W)
    if placed is None:
        return None
    targets, path = placed

    specs: list[dict[str, int]] = []
    for idx in range(total_sprites):
        h, w = _pick(rng, SPRITE_SHAPES)
        specs.append(
            {
                "h": h,
                "w": w,
                "border": border_colors[idx],
                "center": center_colors[idx],
                "r0": -1,
                "c0": -1,
            }
        )

    for tid in range(2):
        specs[tid]["h"] = targets[tid]["h"]
        specs[tid]["w"] = targets[tid]["w"]
        specs[tid]["r0"] = targets[tid]["r0"]
        specs[tid]["c0"] = targets[tid]["c0"]

    grid = np.full((H, W), bg, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    for tid in range(2):
        spec = specs[tid]
        patch = _sprite_patch(spec["h"], spec["w"], spec["border"], spec["center"])
        _paste(grid, patch, spec["r0"], spec["c0"])
        occupied[spec["r0"] : spec["r0"] + spec["h"], spec["c0"] : spec["c0"] + spec["w"]] = True

    for r, c in path:
        grid[r, c] = GREY
        occupied[r, c] = True

    for idx in range(2, total_sprites):
        spec = specs[idx]
        placed_ok = False
        for _ in range(150):
            r0 = int(rng.integers(1, H - spec["h"]))
            c0 = int(rng.integers(1, W - spec["w"]))
            if not _can_place_bbox(occupied, r0, c0, spec["h"], spec["w"], gap=1):
                continue
            spec["r0"] = r0
            spec["c0"] = c0
            patch = _sprite_patch(spec["h"], spec["w"], spec["border"], spec["center"])
            _paste(grid, patch, r0, c0)
            occupied[r0 : r0 + spec["h"], c0 : c0 + spec["w"]] = True
            placed_ok = True
            break
        if not placed_ok:
            return None

    want_controls = int(rng.integers(CONTROL_COUNT_RANGE[0], CONTROL_COUNT_RANGE[1] + 1))
    candidates = _control_candidates(grid, bg)
    controls = _select_control_points(rng, candidates, want_controls)
    if controls is None:
        return None

    control_choices = [c for c in palette]
    for cell in controls:
        grid[cell] = int(_pick(rng, control_choices))

    try:
        scene = _analyze_scene(grid)
    except AssertionError:
        return None

    if len(scene.target_ids) != 2:
        return None
    for tid, src_center in zip(scene.target_ids, scene.replacements, strict=True):
        if src_center == scene.sprites[tid].center_color:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
