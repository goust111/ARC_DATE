import random
from collections import Counter, defaultdict, deque

import numpy as np


GRID_SIZE = 30
SPRITE_SIZE = 5
GEN_ATTEMPTS = 400
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
CORNERS = {
    "tl": (0, 0),
    "tr": (0, GRID_SIZE - 1),
    "bl": (GRID_SIZE - 1, 0),
    "br": (GRID_SIZE - 1, GRID_SIZE - 1),
}
BLOCK_ORIGINS = {
    "tl": (0, 0),
    "tr": (0, GRID_SIZE - SPRITE_SIZE),
    "bl": (GRID_SIZE - SPRITE_SIZE, 0),
    "br": (GRID_SIZE - SPRITE_SIZE, GRID_SIZE - SPRITE_SIZE),
}


def _neighbors(r: int, c: int, shape: tuple[int, int], conn8: bool) -> list[tuple[int, int]]:
    out = []
    dirs = DIR8 if conn8 else DIR4
    h, w = shape
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _components(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, mask.shape, conn8):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _corner_from_pos(pos: tuple[int, int]) -> str:
    for name, coord in CORNERS.items():
        if pos == coord:
            return name
    raise AssertionError("corner pixel not found")


def _facing_corner(local: tuple[int, int], corner_name: str) -> bool:
    r, c = local
    if corner_name == "tl":
        return r == 0 or c == 0
    if corner_name == "tr":
        return r == 0 or c == SPRITE_SIZE - 1
    if corner_name == "bl":
        return r == SPRITE_SIZE - 1 or c == 0
    if corner_name == "br":
        return r == SPRITE_SIZE - 1 or c == SPRITE_SIZE - 1
    raise AssertionError("unknown corner")


def _parse_sprites(grid: np.ndarray) -> tuple[int, dict[int, str], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE)

    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])
    corner_colors: dict[int, str] = {}
    work = g.copy()
    for pos in CORNERS.values():
        color = int(g[pos])
        assert color != bg, "corner labels must be foreground"
        assert color not in corner_colors, "corner label colors must be distinct"
        corner_colors[color] = _corner_from_pos(pos)
        work[pos] = bg

    sprites = []
    for pts in _components(work != bg, conn8=True):
        r0, c0, r1, c1 = _bbox(pts)
        assert r1 - r0 + 1 == SPRITE_SIZE and c1 - c0 + 1 == SPRITE_SIZE, "sprites must be 5x5"
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = {int(v) for v in np.unique(patch) if int(v) != bg}
        marker_colors = colors & set(corner_colors)
        body_colors = colors - set(corner_colors)
        assert len(marker_colors) == 1 and len(body_colors) == 1, "sprite must have one body and one marker color"
        marker_color = next(iter(marker_colors))
        body_color = next(iter(body_colors))
        mask = patch != bg
        marker_cells = np.argwhere(patch == marker_color)
        assert len(marker_cells) == 1, "sprite must contain exactly one marker pixel"
        marker_pos = tuple(map(int, marker_cells[0]))
        body_mask = patch == body_color
        assert np.any(body_mask), "sprite body must be non-empty"
        sprites.append(
            {
                "bbox": (r0, c0),
                "mask": mask,
                "body_mask": body_mask,
                "marker_color": marker_color,
                "body_color": body_color,
                "marker_pos": marker_pos,
            }
        )

    body_colors = [sprite["body_color"] for sprite in sprites]
    assert len(body_colors) == len(set(body_colors)), "each sprite body color must be unique"
    assert sprites, "expected at least one sprite"
    return bg, corner_colors, sprites


def main(grid: np.ndarray) -> np.ndarray:
    bg, corner_colors, sprites = _parse_sprites(np.asarray(grid, dtype=int))
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)

    by_marker: dict[int, list[dict]] = defaultdict(list)
    for sprite in sprites:
        by_marker[int(sprite["marker_color"])].append(sprite)

    for marker_color, group in by_marker.items():
        corner_name = corner_colors[int(marker_color)]
        block_r0, block_c0 = BLOCK_ORIGINS[corner_name]
        union_mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
        for sprite in group:
            union_mask |= sprite["mask"]
        out[block_r0 : block_r0 + SPRITE_SIZE, block_c0 : block_c0 + SPRITE_SIZE][union_mask] = marker_color

        planned_detail: dict[tuple[int, int], int] = {}
        for idx, sprite in enumerate(group):
            marker_pos = sprite["marker_pos"]
            if not _facing_corner(marker_pos, corner_name):
                continue
            covered = any(
                other["body_mask"][marker_pos]
                for j, other in enumerate(group)
                if j != idx
            )
            if not covered:
                continue
            previous = planned_detail.get(marker_pos)
            color = int(sprite["body_color"])
            assert previous is None or previous == color, "ambiguous visible body color"
            planned_detail[marker_pos] = color

        for (rr, cc), color in planned_detail.items():
            out[block_r0 + rr, block_c0 + cc] = color

    return out


def _grow_connected_mask(
    rng: random.Random,
    required: set[tuple[int, int]],
    min_area: int,
    max_area: int,
) -> np.ndarray | None:
    req = set(required)
    for cell in list(req):
        rr, cc = cell
        if not (0 <= rr < SPRITE_SIZE and 0 <= cc < SPRITE_SIZE):
            return None
    if not req:
        req.add((rng.randrange(SPRITE_SIZE), rng.randrange(SPRITE_SIZE)))
    req_list = list(req)
    rng.shuffle(req_list)
    cells = {req_list[0]}
    for target_r, target_c in req_list[1:]:
        start_r, start_c = min(
            cells,
            key=lambda pt: abs(pt[0] - target_r) + abs(pt[1] - target_c),
        )
        cur_r, cur_c = start_r, start_c
        while cur_r != target_r:
            cur_r += 1 if target_r > cur_r else -1
            cells.add((cur_r, cur_c))
        while cur_c != target_c:
            cur_c += 1 if target_c > cur_c else -1
            cells.add((cur_r, cur_c))
    if len(cells) > max_area:
        return None
    target = rng.randint(max(len(cells), min_area), max_area)

    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors(r, c, (SPRITE_SIZE, SPRITE_SIZE), conn8=False):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(rng.choice(sorted(frontier)))

    mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_marker_pos(rng: random.Random, mask: np.ndarray, prefer_side: str | None) -> tuple[int, int]:
    cells = [tuple(map(int, pt)) for pt in np.argwhere(mask)]
    if prefer_side is not None:
        side_cells = [pt for pt in cells if _facing_corner(pt, prefer_side)]
        if side_cells:
            return rng.choice(side_cells)
    return rng.choice(cells)


def _bbox_cover_requirements(
    rng: random.Random,
    extra: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]]:
    required = set(extra or ())
    required.add((0, rng.randrange(SPRITE_SIZE)))
    required.add((SPRITE_SIZE - 1, rng.randrange(SPRITE_SIZE)))
    required.add((rng.randrange(SPRITE_SIZE), 0))
    required.add((rng.randrange(SPRITE_SIZE), SPRITE_SIZE - 1))
    return required


def _make_sprite_masks(
    rng: random.Random,
    groups: list[int],
) -> list[dict] | None:
    by_group: dict[int, list[int]] = defaultdict(list)
    for idx, group_id in enumerate(groups):
        by_group[group_id].append(idx)

    sprites: list[dict | None] = [None] * len(groups)
    exception_group = None
    candidate_groups = [gid for gid, idxs in by_group.items() if len(idxs) >= 2]
    if candidate_groups and rng.random() < 0.6:
        exception_group = rng.choice(candidate_groups)

    for gid, idxs in by_group.items():
        used_marker_positions: set[tuple[int, int]] = set()
        if gid == exception_group:
            a_idx, b_idx = rng.sample(idxs, 2)
            if gid == 0:
                facing_choices = [(0, c) for c in range(SPRITE_SIZE)] + [(r, 0) for r in range(1, SPRITE_SIZE)]
            elif gid == 1:
                facing_choices = [(0, c) for c in range(SPRITE_SIZE)] + [(r, SPRITE_SIZE - 1) for r in range(1, SPRITE_SIZE)]
            elif gid == 2:
                facing_choices = [(SPRITE_SIZE - 1, c) for c in range(SPRITE_SIZE)] + [(r, 0) for r in range(SPRITE_SIZE - 1)]
            else:
                facing_choices = [(SPRITE_SIZE - 1, c) for c in range(SPRITE_SIZE)] + [(r, SPRITE_SIZE - 1) for r in range(SPRITE_SIZE - 1)]
            shared_cell = rng.choice(facing_choices)

            mask_a = _grow_connected_mask(rng, _bbox_cover_requirements(rng, {shared_cell}), 7, 13)
            mask_b = _grow_connected_mask(rng, _bbox_cover_requirements(rng, {shared_cell}), 7, 13)
            if mask_a is None or mask_b is None:
                return None
            marker_a = shared_cell
            other_cells_b = [pt for pt in np.argwhere(mask_b) if tuple(map(int, pt)) != shared_cell]
            if not other_cells_b:
                return None
            marker_b = tuple(map(int, rng.choice(other_cells_b)))
            sprites[a_idx] = {"mask": mask_a, "marker_pos": marker_a, "group": gid}
            sprites[b_idx] = {"mask": mask_b, "marker_pos": marker_b, "group": gid}
            used_marker_positions.add(marker_a)
            used_marker_positions.add(marker_b)

        for idx in idxs:
            if sprites[idx] is not None:
                continue
            mask = _grow_connected_mask(rng, _bbox_cover_requirements(rng), 7, 13)
            if mask is None:
                return None
            for _ in range(30):
                marker_pos = _random_marker_pos(rng, mask, None)
                if marker_pos not in used_marker_positions:
                    break
            else:
                return None
            used_marker_positions.add(marker_pos)
            sprites[idx] = {"mask": mask, "marker_pos": marker_pos, "group": gid}

    return [sprite for sprite in sprites if sprite is not None]


def _sample_group_ids(rng: random.Random, n: int) -> list[int]:
    groups = [rng.randrange(4) for _ in range(n)]
    if n >= 3 and len(set(groups)) == n and rng.random() < 0.7:
        groups[-1] = groups[0]
    if len(set(groups)) == 1 and n >= 2 and rng.random() < 0.7:
        groups[-1] = (groups[0] + rng.randrange(1, 4)) % 4
    return groups


def _non_overlapping_positions(rng: random.Random, n: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(5, 21) for c in range(5, 21)]
    rng.shuffle(candidates)
    placed: list[tuple[int, int]] = []
    for r, c in candidates:
        ok = True
        for pr, pc in placed:
            if not (r + SPRITE_SIZE + 1 <= pr or pr + SPRITE_SIZE + 1 <= r or c + SPRITE_SIZE + 1 <= pc or pc + SPRITE_SIZE + 1 <= c):
                ok = False
                break
        if ok:
            placed.append((r, c))
            if len(placed) == n:
                return placed
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(10))
    bg = rng.choice(palette)
    palette.remove(bg)
    rng.shuffle(palette)
    corner_cols = palette[:4]
    body_cols = palette[4:]
    n_sprites = rng.randint(2, len(body_cols))
    groups = _sample_group_ids(rng, n_sprites)
    sprite_specs = _make_sprite_masks(rng, groups)
    if sprite_specs is None:
        return None
    positions = _non_overlapping_positions(rng, n_sprites)
    if positions is None:
        return None

    grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    for name, color in zip(("tl", "tr", "bl", "br"), corner_cols):
        grid[CORNERS[name]] = color

    rng.shuffle(body_cols)
    for idx, spec in enumerate(sprite_specs):
        r0, c0 = positions[idx]
        body_color = body_cols[idx]
        marker_color = corner_cols[spec["group"]]
        patch = np.full((SPRITE_SIZE, SPRITE_SIZE), bg, dtype=int)
        patch[spec["mask"]] = body_color
        patch[spec["marker_pos"]] = marker_color
        grid[r0 : r0 + SPRITE_SIZE, c0 : c0 + SPRITE_SIZE] = patch

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)):
        return None
    if len({int(v) for v in np.unique(out)}) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
