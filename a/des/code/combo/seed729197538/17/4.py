import random
from collections import Counter

import numpy as np


BG = 8  # azure
DIR_TOP = 0
DIR_RIGHT = 1
DIR_BOTTOM = 2
DIR_LEFT = 3
DIR_TO_K = {
    DIR_TOP: 0,
    DIR_RIGHT: 3,   # clockwise 90
    DIR_BOTTOM: 2,
    DIR_LEFT: 1,    # counter-clockwise 90
}
PALETTE = [0, 1, 2, 3, 4, 5, 6, 7, 9]
NEI4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEI4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_patch(grid: np.ndarray, coords: np.ndarray, bg: int = BG) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    mask = np.zeros_like(patch, dtype=bool)
    for r, c in coords:
        rr = int(r - r0)
        cc = int(c - c0)
        patch[rr, cc] = int(grid[r, c])
        mask[rr, cc] = True
    return patch, mask, (r0, c0, r1, c1)


def _region_from_pos(r: int, c: int, h: int, w: int) -> int:
    x = (c + 0.5) / w
    y = (r + 0.5) / h
    scores = np.array(
        [
            min(x, 1.0 - x) - y,
            x - max(y, 1.0 - y),
            y - max(x, 1.0 - x),
            min(y, 1.0 - y) - x,
        ],
        dtype=float,
    )
    best = np.flatnonzero(scores == scores.max())
    assert len(best) == 1 and float(scores[best[0]]) > 0.0
    return int(best[0])


def _rotated_marker_pos(mask: np.ndarray, marker_rc: tuple[int, int], k: int) -> tuple[int, int]:
    marker_mask = np.zeros_like(mask, dtype=bool)
    marker_mask[marker_rc] = True
    rotated = np.rot90(marker_mask, k)
    pts = np.argwhere(rotated)
    assert pts.shape == (1, 2)
    return int(pts[0, 0]), int(pts[0, 1])


def _paste_checked(out: np.ndarray, patch: np.ndarray, mask: np.ndarray, top: int, left: int, bg: int = BG) -> None:
    h, w = patch.shape
    oh, ow = out.shape
    assert 0 <= top <= oh - h and 0 <= left <= ow - w
    for rr, cc in np.argwhere(mask):
        r = int(top + rr)
        c = int(left + cc)
        assert out[r, c] == bg
        out[r, c] = int(patch[rr, cc])


def _classify_scene(grid: np.ndarray) -> tuple[dict[int, tuple[int, int]], list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = _components(g != BG)
    anchors: dict[int, tuple[int, int]] = {}
    objects = []
    for coords in comps:
        if len(coords) == 1:
            r, c = map(int, coords[0])
            if r in (0, h - 1) or c in (0, w - 1):
                color = int(g[r, c])
                assert color not in anchors
                anchors[color] = (r, c)
                continue
        assert np.all(
            (coords[:, 0] > 0)
            & (coords[:, 0] < h - 1)
            & (coords[:, 1] > 0)
            & (coords[:, 1] < w - 1)
        )
        objects.append(coords)
    assert anchors and objects and len(anchors) == len(objects)
    return anchors, objects


def _object_spec_from_grid(grid: np.ndarray, coords: np.ndarray) -> dict:
    patch, mask, bbox = _coords_to_patch(grid, coords, bg=BG)
    color_counts = Counter(int(grid[r, c]) for r, c in coords)
    marker_colors = [color for color, count in color_counts.items() if count == 1]
    assert len(marker_colors) == 1
    marker_color = int(marker_colors[0])
    marker_abs = next((int(r), int(c)) for r, c in coords if int(grid[r, c]) == marker_color)
    r0, c0, _r1, _c1 = bbox
    marker_rel = (marker_abs[0] - r0, marker_abs[1] - c0)
    direction = _region_from_pos(marker_rel[0], marker_rel[1], patch.shape[0], patch.shape[1])
    k = DIR_TO_K[direction]
    rot_patch = np.rot90(patch, k)
    rot_mask = np.rot90(mask, k)
    rot_marker = _rotated_marker_pos(mask, marker_rel, k)
    return {
        "patch": patch,
        "mask": mask,
        "marker_color": marker_color,
        "marker_rel": marker_rel,
        "direction": direction,
        "rot_patch": rot_patch,
        "rot_mask": rot_mask,
        "rot_marker": rot_marker,
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    anchors, objects = _classify_scene(g)

    specs = [_object_spec_from_grid(g, coords) for coords in objects]
    marker_colors = {spec["marker_color"] for spec in specs}
    assert marker_colors == set(anchors)

    out = np.full_like(g, BG)
    for spec in specs:
        anchor_r, anchor_c = anchors[spec["marker_color"]]
        top = int(anchor_r - spec["rot_marker"][0])
        left = int(anchor_c - spec["rot_marker"][1])
        _paste_checked(out, spec["rot_patch"], spec["rot_mask"], top, left, bg=BG)
    return out


def _inflate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    out = mask.copy()
    pts = np.argwhere(mask)
    for r, c in pts:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _random_connected_mask(rng: random.Random) -> np.ndarray | None:
    for _ in range(80):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        max_area = min(h * w, 10)
        min_area = max(4, (h * w + 2) // 3)
        area = rng.randint(min_area, max_area)
        seed = (rng.randrange(h), rng.randrange(w))
        cells = {seed}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEI4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        coords = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = _bbox(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            mask[int(r - r0), int(c - c0)] = True
        if mask.shape[0] <= 5 and mask.shape[1] <= 5 and mask.sum() >= 4:
            return mask
    return None


def _choose_marker_cell(mask: np.ndarray, direction: int, rng: random.Random) -> tuple[int, int] | None:
    candidates = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        try:
            region = _region_from_pos(int(r), int(c), h, w)
        except AssertionError:
            continue
        if region == direction:
            score = {
                DIR_TOP: -r,
                DIR_RIGHT: c,
                DIR_BOTTOM: r,
                DIR_LEFT: -c,
            }[direction]
            candidates.append((score, int(r), int(c)))
    if not candidates:
        return None
    best_score = max(item[0] for item in candidates)
    best = [(r, c) for score, r, c in candidates if score == best_score]
    return rng.choice(best)


def _split_counts(total: int, parts: int, rng: random.Random) -> list[int] | None:
    if total < 2 * parts:
        return None
    counts = [2] * parts
    rest = total - 2 * parts
    for _ in range(rest):
        counts[rng.randrange(parts)] += 1
    rng.shuffle(counts)
    return counts


def _paint_object(mask: np.ndarray, marker_rc: tuple[int, int], marker_color: int, body_palette: list[int], rng: random.Random) -> np.ndarray:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) != marker_rc]
    rng.shuffle(coords)
    body_count = len(coords)

    max_body_colors = min(3, len(body_palette), body_count // 2 if body_count >= 2 else 1)
    choices = [k for k in range(1, max_body_colors + 1) if 2 * k <= body_count]
    num_body_colors = rng.choice(choices) if choices else 1
    counts = _split_counts(body_count, num_body_colors, rng)
    assert counts is not None
    colors = rng.sample(body_palette, num_body_colors)

    patch = np.full(mask.shape, BG, dtype=int)
    patch[marker_rc] = int(marker_color)
    idx = 0
    ordered = sorted(coords, key=lambda rc: (rc[0] + rc[1], rc[0], rc[1], rng.random()))
    for color, count in zip(colors, counts):
        for _ in range(count):
            r, c = ordered[idx]
            patch[r, c] = int(color)
            idx += 1
    return patch


def _build_object(direction: int, marker_color: int, body_palette: list[int], rng: random.Random) -> dict | None:
    for _ in range(120):
        mask = _random_connected_mask(rng)
        if mask is None:
            continue
        marker_rc = _choose_marker_cell(mask, direction, rng)
        if marker_rc is None:
            continue
        patch = _paint_object(mask, marker_rc, marker_color, body_palette, rng)
        k = DIR_TO_K[direction]
        rot_patch = np.rot90(patch, k)
        rot_mask = np.rot90(mask, k)
        rot_marker = _rotated_marker_pos(mask, marker_rc, k)
        return {
            "patch": patch,
            "mask": mask,
            "marker_color": marker_color,
            "marker_rel": marker_rc,
            "direction": direction,
            "rot_patch": rot_patch,
            "rot_mask": rot_mask,
            "rot_marker": rot_marker,
        }
    return None


def _border_cells(h: int, w: int) -> list[tuple[int, int]]:
    cells = []
    for c in range(w):
        cells.append((0, c))
        cells.append((h - 1, c))
    for r in range(1, h - 1):
        cells.append((r, 0))
        cells.append((r, w - 1))
    return cells


def _anchor_neighbors(rc: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = rc
    return {
        (r, c),
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    }


def _fits_on_mask(blocked: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = blocked.shape
    ph, pw = mask.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    for rr, cc in np.argwhere(mask):
        if blocked[int(top + rr), int(left + cc)]:
            return False
    return True


def _place_outputs(specs: list[dict], h: int, w: int, rng: random.Random) -> tuple[dict[int, tuple[int, int]], np.ndarray] | None:
    order = list(range(len(specs)))
    rng.shuffle(order)
    blocked = np.zeros((h, w), dtype=bool)
    anchors: dict[int, tuple[int, int]] = {}
    forbidden_anchor_cells: set[tuple[int, int]] = set()
    border = _border_cells(h, w)

    for idx in order:
        spec = specs[idx]
        candidates = border[:]
        rng.shuffle(candidates)
        placed = False
        ph, pw = spec["rot_patch"].shape
        mr, mc = spec["rot_marker"]
        for anchor in candidates:
            if anchor in forbidden_anchor_cells:
                continue
            top = int(anchor[0] - mr)
            left = int(anchor[1] - mc)
            if not _fits_on_mask(blocked, spec["rot_mask"], top, left):
                continue
            for rr, cc in np.argwhere(spec["rot_mask"]):
                blocked[int(top + rr), int(left + cc)] = True
            anchors[spec["marker_color"]] = anchor
            forbidden_anchor_cells.update(
                {
                    (ar, ac)
                    for ar, ac in _anchor_neighbors(anchor)
                    if 0 <= ar < h and 0 <= ac < w and (ar in (0, h - 1) or ac in (0, w - 1))
                }
            )
            placed = True
            break
        if not placed:
            return None
    return anchors, blocked


def _place_inputs(specs: list[dict], h: int, w: int, occupied_output: np.ndarray, rng: random.Random) -> list[tuple[int, int]] | None:
    blocked = _inflate(occupied_output, radius=1)
    blocked[0, :] = True
    blocked[h - 1, :] = True
    blocked[:, 0] = True
    blocked[:, w - 1] = True
    placements: list[tuple[int, int]] = [(-1, -1)] * len(specs)

    order = list(range(len(specs)))
    rng.shuffle(order)
    for idx in order:
        spec = specs[idx]
        candidates = []
        ph, pw = spec["patch"].shape
        for top in range(1, h - ph):
            for left in range(1, w - pw):
                if _fits_on_mask(blocked, spec["mask"], top, left):
                    candidates.append((top, left))
        if not candidates:
            return None
        top, left = rng.choice(candidates)
        placements[idx] = (top, left)
        obj_occ = np.zeros((h, w), dtype=bool)
        for rr, cc in np.argwhere(spec["mask"]):
            obj_occ[int(top + rr), int(left + cc)] = True
        blocked |= _inflate(obj_occ, radius=1)
    return placements


def _render_input(specs: list[dict], placements: list[tuple[int, int]], anchors: dict[int, tuple[int, int]], h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for color, (r, c) in anchors.items():
        grid[int(r), int(c)] = int(color)
    for spec, (top, left) in zip(specs, placements):
        for rr, cc in np.argwhere(spec["mask"]):
            grid[int(top + rr), int(left + cc)] = int(spec["patch"][rr, cc])
    return grid


def _render_output(specs: list[dict], anchors: dict[int, tuple[int, int]], h: int, w: int) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    for spec in specs:
        anchor_r, anchor_c = anchors[spec["marker_color"]]
        top = int(anchor_r - spec["rot_marker"][0])
        left = int(anchor_c - spec["rot_marker"][1])
        _paste_checked(out, spec["rot_patch"], spec["rot_mask"], top, left, bg=BG)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 20)
    w = rng.randint(14, 20)
    n_objects = rng.randint(2, 4)

    marker_colors = rng.sample(PALETTE, n_objects)
    body_palette = [color for color in PALETTE if color not in marker_colors]
    directions = rng.sample([DIR_TOP, DIR_RIGHT, DIR_BOTTOM, DIR_LEFT], n_objects)

    specs = []
    for direction, marker_color in zip(directions, marker_colors):
        spec = _build_object(direction, marker_color, body_palette, rng)
        if spec is None:
            return None
        specs.append(spec)

    placed_outputs = _place_outputs(specs, h, w, rng)
    if placed_outputs is None:
        return None
    anchors, output_occ = placed_outputs

    placements = _place_inputs(specs, h, w, output_occ, rng)
    if placements is None:
        return None

    grid = _render_input(specs, placements, anchors, h, w)
    expected = _render_output(specs, anchors, h, w)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
