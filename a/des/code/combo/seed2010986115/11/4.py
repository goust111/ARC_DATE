from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
WALL = 5
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 300


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _extract_palette(grid: np.ndarray) -> tuple[dict[int, list[int]], int]:
    g = np.asarray(grid, dtype=int)
    scene_rows = np.flatnonzero(np.any(g == WALL, axis=1))
    assert scene_rows.size > 0, "expected a grey-walled scene"
    scene_start = int(scene_rows[0])

    mapping: dict[int, list[int]] = {}
    for r in range(scene_start):
        row = g[r]
        if np.all(row == BG):
            continue
        key = int(row[0])
        assert key not in (BG, WALL), "palette key must be a visible non-grey color"
        pattern: list[int] = []
        c = 1
        while c < g.shape[1] and int(row[c]) != BG:
            value = int(row[c])
            assert value != WALL, "grey is reserved for walls"
            pattern.append(value)
            c += 1
        assert pattern, "palette row must contain a non-empty pattern unit"
        assert np.all(row[c:] == BG), "palette row must be zero-padded after the pattern"
        assert key not in mapping, "palette keys must be unique"
        mapping[key] = pattern

    assert len(mapping) >= 2, "expected at least two palette rows"
    return mapping, scene_start


def _find_cavities(
    grid: np.ndarray, scene_start: int, mapping: dict[int, list[int]]
) -> list[dict[str, object]]:
    scene = np.asarray(grid[scene_start:], dtype=int)
    h, w = scene.shape
    cavities: list[dict[str, object]] = []

    for comp in _components(scene != WALL):
        if _touches_border(comp, h, w):
            continue
        seed_colors = {
            int(scene[r, c])
            for r, c in comp
            if int(scene[r, c]) not in (BG, WALL)
        }
        assert len(seed_colors) == 1, "each cavity must contain exactly one seed color"
        seed_color = next(iter(seed_colors))
        assert seed_color in mapping, "seed color must exist in the palette"
        cavities.append({"seed": seed_color, "pixels": comp})

    assert cavities, "expected at least one enclosed cavity"
    return cavities


def main(grid: np.ndarray) -> np.ndarray:
    """
    Look up each cavity seed color in the top palette and tile the matching
    horizontal pattern across that cavity row by row.
    """
    g = np.asarray(grid, dtype=int)
    mapping, scene_start = _extract_palette(g)
    cavities = _find_cavities(g, scene_start, mapping)

    out = g.copy()
    for cavity in cavities:
        pattern = mapping[int(cavity["seed"])]
        rows: dict[int, list[int]] = defaultdict(list)
        for r, c in cavity["pixels"]:
            rows[int(r)].append(int(c))
        for r in sorted(rows):
            cols = sorted(rows[r])
            for idx, c in enumerate(cols):
                out[scene_start + r, c] = pattern[idx % len(pattern)]

    return out


def _single_component(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _has_wall_block(mask: np.ndarray) -> bool:
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return False
    return bool(
        np.any(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:])
    )


def _make_stair_mask(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 5)
    w = rng.randint(4, 6)
    mask = np.ones((h, w), dtype=bool)
    side = rng.choice(["left", "right"])
    order = list(range(h))
    if rng.random() < 0.5:
        order.reverse()
    trim = 0
    max_trim = w - 2
    changed = False
    for i, r in enumerate(order):
        if i > 0 and trim < max_trim and rng.random() < 0.75:
            trim += 1
            changed = True
        if side == "left":
            mask[r, :trim] = False
        else:
            mask[r, w - trim :] = False
    if not changed:
        trim = 1
        if side == "left":
            mask[order[-1], :trim] = False
        else:
            mask[order[-1], w - trim :] = False
    if rng.random() < 0.5:
        mask = mask.T
    return mask


def _make_interior_mask(rng: random.Random) -> np.ndarray:
    if rng.random() < 0.45:
        h = rng.randint(2, 4)
        w = rng.randint(3, 5)
        return np.ones((h, w), dtype=bool)
    return _make_stair_mask(rng)


def _choose_anchor(points: list[tuple[int, int]], axis: str) -> tuple[int, int]:
    if axis == "left":
        target = min(c for _, c in points)
        cand = sorted((r, c) for r, c in points if c == target)
    elif axis == "right":
        target = max(c for _, c in points)
        cand = sorted((r, c) for r, c in points if c == target)
    elif axis == "top":
        target = min(r for r, _ in points)
        cand = sorted((r, c) for r, c in points if r == target)
    else:
        target = max(r for r, _ in points)
        cand = sorted((r, c) for r, c in points if r == target)
    return cand[len(cand) // 2]


def _make_cavity_spec(rng: random.Random, seed_color: int) -> dict[str, object] | None:
    for _ in range(60):
        mask = _make_interior_mask(rng)
        ih, iw = mask.shape
        interior = np.zeros((ih + 2, iw + 2), dtype=bool)
        interior[1 : ih + 1, 1 : iw + 1] = mask

        wall = np.zeros_like(interior)
        for r, c in np.argwhere(interior):
            rr = int(r)
            cc = int(c)
            for nr, nc in neighbors8(rr, cc):
                if 0 <= nr < interior.shape[0] and 0 <= nc < interior.shape[1] and not interior[nr, nc]:
                    wall[nr, nc] = True

        if not _single_component(interior):
            continue
        if not _single_component(wall):
            continue
        if _has_wall_block(wall):
            continue

        enclosed = [
            comp
            for comp in _components(~wall)
            if not _touches_border(comp, wall.shape[0], wall.shape[1])
        ]
        interior_cells = [tuple(map(int, rc)) for rc in np.argwhere(interior)]
        if len(enclosed) != 1 or set(enclosed[0]) != set(interior_cells):
            continue

        wall_points = [tuple(map(int, rc)) for rc in np.argwhere(wall)]
        seed = rng.choice(interior_cells)
        return {
            "h": int(wall.shape[0]),
            "w": int(wall.shape[1]),
            "wall": wall,
            "seed": seed,
            "seed_color": int(seed_color),
            "anchors": {
                "left": _choose_anchor(wall_points, "left"),
                "right": _choose_anchor(wall_points, "right"),
                "top": _choose_anchor(wall_points, "top"),
                "bottom": _choose_anchor(wall_points, "bottom"),
            },
        }
    return None


def _make_palette(rng: random.Random) -> tuple[dict[int, list[int]], list[int]]:
    k = rng.randint(2, 4)
    keys = rng.sample(PALETTE_COLORS, k)
    mapping: dict[int, list[int]] = {}
    for key in keys:
        plen = rng.randint(2, 5)
        choices = rng.sample(PALETTE_COLORS, rng.randint(2, min(4, len(PALETTE_COLORS))))
        pattern = [rng.choice(choices) for _ in range(plen)]
        if len(set(pattern)) == 1:
            pattern[-1] = choices[1] if choices[1] != pattern[0] else choices[0]
        mapping[key] = pattern
    return mapping, keys


def _build_scene_vertical(
    rng: random.Random, specs: list[dict[str, object]], min_width: int
) -> np.ndarray | None:
    gaps = [rng.randint(2, 4) for _ in range(len(specs) - 1)]
    top_margin = rng.randint(0, 1)
    bottom_margin = rng.randint(0, 1)
    scene_h = top_margin + sum(int(spec["h"]) for spec in specs) + sum(gaps) + bottom_margin
    if scene_h > 20:
        return None

    max_w = max(int(spec["w"]) for spec in specs)
    scene_w = max(min_width, max_w + rng.randint(5, 8))
    if scene_w > 28:
        return None

    scene = np.zeros((scene_h, scene_w), dtype=int)
    placed: list[tuple[int, int, int, int]] = []
    top = top_margin
    for i, spec in enumerate(specs):
        sh = int(spec["h"])
        sw = int(spec["w"])
        left = rng.randint(0, scene_w - sw)

        wall_mask = spec["wall"]
        patch = scene[top : top + sh, left : left + sw]
        patch[np.asarray(wall_mask, dtype=bool)] = WALL

        seed_r, seed_c = spec["seed"]
        scene[top + seed_r, left + seed_c] = int(spec["seed_color"])
        placed.append((top, left, sh, sw))
        top += sh + (gaps[i] if i < len(gaps) else 0)

    for i in range(len(specs) - 1):
        top_a, left_a, sh_a, _sw_a = placed[i]
        top_b, left_b, _sh_b, _sw_b = placed[i + 1]
        gap_row = rng.randint(top_a + sh_a, top_b - 1)
        ar, ac = specs[i]["anchors"]["bottom"]
        br, bc = specs[i + 1]["anchors"]["top"]
        ra = top_a + ar
        ca = left_a + ac
        rb = top_b + br
        cb = left_b + bc
        scene[ra : gap_row + 1, ca] = WALL
        c0, c1 = sorted((ca, cb))
        scene[gap_row, c0 : c1 + 1] = WALL
        scene[gap_row : rb + 1, cb] = WALL

    return scene


def _build_scene_horizontal(
    rng: random.Random, specs: list[dict[str, object]], min_width: int
) -> np.ndarray | None:
    gaps = [rng.randint(2, 4) for _ in range(len(specs) - 1)]
    left_margin = rng.randint(0, 1)
    right_margin = rng.randint(0, 1)
    scene_w = left_margin + sum(int(spec["w"]) for spec in specs) + sum(gaps) + right_margin
    scene_w = max(scene_w, min_width)
    if scene_w > 28:
        return None

    max_h = max(int(spec["h"]) for spec in specs)
    scene_h = max_h + rng.randint(5, 8)
    if scene_h > 20:
        return None

    scene = np.zeros((scene_h, scene_w), dtype=int)
    placed: list[tuple[int, int, int, int]] = []
    left = left_margin
    for i, spec in enumerate(specs):
        sh = int(spec["h"])
        sw = int(spec["w"])
        top = rng.randint(0, scene_h - sh)

        wall_mask = spec["wall"]
        patch = scene[top : top + sh, left : left + sw]
        patch[np.asarray(wall_mask, dtype=bool)] = WALL

        seed_r, seed_c = spec["seed"]
        scene[top + seed_r, left + seed_c] = int(spec["seed_color"])
        placed.append((top, left, sh, sw))
        left += sw + (gaps[i] if i < len(gaps) else 0)

    for i in range(len(specs) - 1):
        top_a, left_a, _sh_a, sw_a = placed[i]
        top_b, left_b, _sh_b, _sw_b = placed[i + 1]
        gap_col = rng.randint(left_a + sw_a, left_b - 1)
        ar, ac = specs[i]["anchors"]["right"]
        br, bc = specs[i + 1]["anchors"]["left"]
        ra = top_a + ar
        ca = left_a + ac
        rb = top_b + br
        cb = left_b + bc
        scene[ra, ca : gap_col + 1] = WALL
        r0, r1 = sorted((ra, rb))
        scene[r0 : r1 + 1, gap_col] = WALL
        scene[rb, gap_col : cb + 1] = WALL

    return scene


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        mapping, scene_start = _extract_palette(grid)
        cavities = _find_cavities(grid, scene_start, mapping)
        scene = grid[scene_start:]
        if len(_components(scene == WALL)) != 1:
            return False
        out = main(grid)
        if np.array_equal(out, grid):
            return False
        for cavity in cavities:
            pixels = cavity["pixels"]
            if any(int(out[scene_start + r, c]) in (BG, WALL) for r, c in pixels):
                return False
        return True
    except AssertionError:
        return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette, keys = _make_palette(rng)
    cavity_count = rng.choice([2, 3])
    chosen_keys = [rng.choice(keys) for _ in range(cavity_count)]
    if len(set(chosen_keys)) == 1:
        chosen_keys[-1] = rng.choice([k for k in keys if k != chosen_keys[0]])

    specs: list[dict[str, object]] = []
    for key in chosen_keys:
        spec = _make_cavity_spec(rng, key)
        if spec is None:
            return None
        specs.append(spec)

    min_width = max(len(pattern) + 3 for pattern in palette.values())
    if rng.random() < 0.5:
        scene = _build_scene_vertical(rng, specs, min_width)
    else:
        scene = _build_scene_horizontal(rng, specs, min_width)
    if scene is None:
        return None

    palette_h = 2 * len(palette) - 1
    gap_rows = rng.randint(0, 2)
    total_h = palette_h + gap_rows + scene.shape[0]
    total_w = scene.shape[1]
    grid = np.zeros((total_h, total_w), dtype=int)

    row = 0
    for idx, (key, pattern) in enumerate(palette.items()):
        grid[row, 0] = key
        grid[row, 1 : 1 + len(pattern)] = pattern
        row += 1
        if idx < len(palette) - 1:
            row += 1

    grid[palette_h + gap_rows :, :] = scene
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
