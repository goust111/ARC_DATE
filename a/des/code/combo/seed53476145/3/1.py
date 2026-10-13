import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
GEN_ATTEMPTS = 300
PLACE_ATTEMPTS = 120


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


TEMPLATE_LIBRARY_3 = [
    _mask_from_rows(["111", "101", "111"]),
    _mask_from_rows(["011", "101", "111"]),
    _mask_from_rows(["110", "101", "111"]),
    _mask_from_rows(["111", "101", "011"]),
    _mask_from_rows(["111", "101", "110"]),
]

TEMPLATE_LIBRARY_4 = [
    _mask_from_rows(["1111", "1001", "1001", "1111"]),
    _mask_from_rows(["0111", "1001", "1001", "1111"]),
    _mask_from_rows(["1110", "1001", "1001", "1111"]),
    _mask_from_rows(["1111", "1001", "1001", "0111"]),
    _mask_from_rows(["1111", "1001", "1001", "1110"]),
]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    neigh = N4 if connectivity == 4 else N8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in neigh:
                    nr, nc = cr + dr, cc + dc
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


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _shape_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask.astype(bool))


def _flood_background(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _hole_components(mask: np.ndarray) -> list[np.ndarray]:
    bg_reachable = _flood_background(mask)
    holes = (~mask) & (~bg_reachable)
    return _components(holes, connectivity=4)


def _outer_boundary_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    exterior = np.zeros_like(padded, dtype=bool)
    q = deque([(0, 0)])
    exterior[0, 0] = True
    ph, pw = padded.shape
    while q:
        r, c = q.popleft()
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ph and 0 <= nc < pw and not padded[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            pr, pc = r + 1, c + 1
            for dr, dc in N8:
                if exterior[pr + dr, pc + dc]:
                    boundary[r, c] = True
                    break
    return boundary


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert cols, "expected a full-height grey separator column"
    return int(cols[-1])


def _parse_legend(left: np.ndarray) -> dict[tuple[str, ...], int]:
    template_comps = sorted(
        _components(left == GREY, connectivity=8),
        key=lambda pts: (_bbox(pts)[0], _bbox(pts)[1]),
    )
    fill_comps = []
    for coords in _components((left != BG) & (left != GREY), connectivity=4):
        vals = {int(left[r, c]) for r, c in coords}
        assert len(vals) == 1, "legend fill block must be monochrome"
        fill_comps.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "color": vals.pop(),
            }
        )

    used_fill = set()
    mapping: dict[tuple[str, ...], int] = {}
    for coords in template_comps:
        r0, c0, r1, c1 = _bbox(coords)
        mask = _mask_from_coords(coords)
        assert mask.shape in {(3, 3), (4, 4)}, "template must be 3x3 or 4x4"
        candidates = []
        for idx, comp in enumerate(fill_comps):
            if idx in used_fill:
                continue
            fr0, fc0, fr1, fc1 = comp["bbox"]
            if fc0 != c1 + 1 or fc1 != fc0:
                continue
            if max(r0, fr0) <= min(r1, fr1):
                candidates.append((idx, comp))
        assert len(candidates) == 1, "each template must have exactly one adjacent fill block"
        idx, comp = candidates[0]
        used_fill.add(idx)
        key = _shape_key(mask)
        assert key not in mapping, "legend shapes must be unique"
        mapping[key] = int(comp["color"])

    assert mapping, "legend mapping must be non-empty"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each hollow sprite's outer contour to the legend and fill its internal holes.
    The output contains only the scene to the right of the separator column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep = _find_separator_col(g)
    legend = g[:, :sep]
    scene = g[:, sep + 1 :].copy()
    mapping = _parse_legend(legend)

    out = scene.copy()
    for coords in _components(scene != BG, connectivity=4):
        mask = _mask_from_coords(coords)
        key = _shape_key(_outer_boundary_mask(mask))
        assert key in mapping, "scene sprite must match a legend template"
        fill_color = mapping[key]
        r0, c0, _r1, _c1 = _bbox(coords)
        for hole in _hole_components(mask):
            out[hole[:, 0] + r0, hole[:, 1] + c0] = int(fill_color)

    return out


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask, connectivity=4)
    return len(comps) == 1


def _choose_templates(rng: random.Random) -> list[np.ndarray]:
    count = rng.randint(2, 4)
    picks = [
        rng.choice(TEMPLATE_LIBRARY_3),
        rng.choice(TEMPLATE_LIBRARY_4),
    ]
    pool = [mask for mask in TEMPLATE_LIBRARY_3 + TEMPLATE_LIBRARY_4 if all(not np.array_equal(mask, seen) for seen in picks)]
    rng.shuffle(pool)
    picks.extend(pool[: count - 2])
    rng.shuffle(picks)
    return [mask.copy() for mask in picks]


def _scene_template_indices(rng: random.Random, n_templates: int, n_sprites: int) -> list[int]:
    base = list(range(n_templates))
    rng.shuffle(base)
    indices = base[: min(n_templates, n_sprites)]
    while len(indices) < n_sprites:
        indices.append(rng.randrange(n_templates))
    rng.shuffle(indices)
    return indices


def _legend_layout(rng: random.Random, templates: list[np.ndarray]) -> tuple[int, list[tuple[int, int]]]:
    gap_sizes = [rng.randint(1, 2) for _ in range(max(0, len(templates) - 1))]
    top_margin = rng.randint(0, 2)
    bottom_margin = rng.randint(0, 2)
    rows: list[tuple[int, int]] = []
    cur = top_margin
    for idx, mask in enumerate(templates):
        col = rng.randint(0, 1)
        rows.append((cur, col))
        cur += mask.shape[0]
        if idx < len(gap_sizes):
            cur += gap_sizes[idx]
    height = cur + bottom_margin
    return height, rows


def _place_scene_sprites(
    rng: random.Random,
    scene_h: int,
    scene_w: int,
    sprite_masks: list[np.ndarray],
    sprite_colors: list[int],
) -> np.ndarray | None:
    scene = np.zeros((scene_h, scene_w), dtype=int)
    occupied = np.zeros((scene_h, scene_w), dtype=bool)
    order = list(range(len(sprite_masks)))
    rng.shuffle(order)
    for idx in order:
        mask = sprite_masks[idx]
        color = int(sprite_colors[idx])
        h, w = mask.shape
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            r0 = rng.randint(0, scene_h - h)
            c0 = rng.randint(0, scene_w - w)
            rs = max(0, r0 - 1)
            re = min(scene_h, r0 + h + 1)
            cs = max(0, c0 - 1)
            ce = min(scene_w, c0 + w + 1)
            if occupied[rs:re, cs:ce].any():
                continue
            scene[r0 : r0 + h, c0 : c0 + w][mask] = color
            occupied[r0 : r0 + h, c0 : c0 + w][mask] = True
            placed = True
            break
        if not placed:
            return None
    return scene


def _generate_once(rng: random.Random) -> np.ndarray | None:
    templates = _choose_templates(rng)
    legend_h, legend_slots = _legend_layout(rng, templates)
    n_templates = len(templates)
    n_sprites = rng.randint(2, 4)

    colors = rng.sample(COLOR_POOL, n_templates + n_sprites)
    fill_colors = colors[:n_templates]
    sprite_colors = colors[n_templates:]
    sprite_template_ids = _scene_template_indices(rng, n_templates, n_sprites)
    sprite_masks = [templates[idx].copy() for idx in sprite_template_ids]

    min_scene_h = max(mask.shape[0] for mask in sprite_masks) + 2
    scene_h = max(legend_h, min_scene_h) + rng.randint(0, 4)
    scene_w = rng.randint(10, 16)
    if scene_w < max(mask.shape[1] for mask in sprite_masks) + 2:
        return None

    scene = _place_scene_sprites(rng, scene_h, scene_w, sprite_masks, sprite_colors)
    if scene is None:
        return None

    legend_w = 6
    grid = np.zeros((scene_h, legend_w + 1 + scene_w), dtype=int)
    sep = legend_w
    grid[:, sep] = GREY
    grid[:, sep + 1 :] = scene

    for idx, mask in enumerate(templates):
        r0, c0 = legend_slots[idx]
        h, w = mask.shape
        grid[r0 : r0 + h, c0 : c0 + w][mask] = GREY
        fill_r0 = r0 + (h - 2) // 2
        fill_c = c0 + w
        grid[fill_r0 : fill_r0 + 2, fill_c] = int(fill_colors[idx])

    legend_map = {_shape_key(mask): color for mask, color in zip(templates, fill_colors)}
    assert len(legend_map) == len(templates)
    assert len(_components(scene != BG, connectivity=4)) == n_sprites
    for coords in _components(scene != BG, connectivity=4):
        mask = _mask_from_coords(coords)
        assert _is_connected(mask)
        assert _hole_components(mask), "every sprite must contain at least one hole"
        key = _shape_key(_outer_boundary_mask(mask))
        assert key in legend_map

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        grid = _generate_once(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
