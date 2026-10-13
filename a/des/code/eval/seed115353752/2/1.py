from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTIONS = ("up", "right", "down", "left")
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATE_RETRIES = 400

TEMPLATE_LIBRARY = {
    3: [
        np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    ],
    4: [
        np.array(
            [[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 0], [1, 0, 1, 1]],
            dtype=bool,
        ),
        np.array(
            [[1, 1, 1, 1], [0, 1, 0, 1], [0, 1, 1, 1], [1, 1, 0, 1]],
            dtype=bool,
        ),
        np.array(
            [[1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 1, 0], [1, 0, 1, 1]],
            dtype=bool,
        ),
        np.array(
            [[1, 1, 1, 1], [1, 1, 0, 1], [0, 1, 0, 1], [1, 1, 1, 1]],
            dtype=bool,
        ),
    ],
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected(mask: np.ndarray) -> bool:
    count = int(mask.sum())
    return count > 0 and len(_components(mask)) == 1


def _rotate_template(mask: np.ndarray, direction: str) -> np.ndarray:
    if direction == "up":
        return mask
    if direction == "right":
        return np.rot90(mask, -1)
    if direction == "down":
        return np.rot90(mask, 2)
    return np.rot90(mask, 1)


def _marker_offsets(points: list[tuple[int, int]], marker: tuple[int, int]) -> tuple[float, float]:
    coords = np.array(points, dtype=float)
    center = coords.mean(axis=0)
    return float(marker[0] - center[0]), float(marker[1] - center[1])


def _direction_from_region(points: list[tuple[int, int]], marker: tuple[int, int]) -> str:
    dr, dc = _marker_offsets(points, marker)
    if abs(dr) >= abs(dc):
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def _validate_templates() -> None:
    for size, masks in TEMPLATE_LIBRARY.items():
        for mask in masks:
            assert mask.shape == (size, size)
            assert bool(mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1])
            assert _is_connected(mask)
            rotations = {np.rot90(mask, k).tobytes() for k in range(4)}
            assert len(rotations) > 1


def _detect_palette(
    grid: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray], str, int]:
    h, w = grid.shape
    colored = (grid != BLACK) & (grid != GREY)
    side_to_comps: dict[str, list[list[tuple[int, int]]]] = {
        "top": [],
        "bottom": [],
        "left": [],
        "right": [],
    }
    touch_map = {"top": 0, "bottom": h - 1, "left": 0, "right": w - 1}

    for comp in _components(colored):
        r0, c0, r1, c1 = _bbox(comp)
        touches = []
        if r0 == touch_map["top"]:
            touches.append("top")
        if r1 == touch_map["bottom"]:
            touches.append("bottom")
        if c0 == touch_map["left"]:
            touches.append("left")
        if c1 == touch_map["right"]:
            touches.append("right")
        if not touches:
            continue
        assert len(touches) == 1
        side_to_comps[touches[0]].append(comp)

    palette_sides = [side for side, comps in side_to_comps.items() if len(comps) >= 2]
    assert len(palette_sides) == 1
    side = palette_sides[0]
    edge_comps = side_to_comps[side]
    assert edge_comps

    spans = set()
    for comp in edge_comps:
        r0, c0, r1, c1 = _bbox(comp)
        spans.add((r1 - r0 + 1) if side in ("top", "bottom") else (c1 - c0 + 1))
    assert len(spans) == 1
    k = spans.pop()

    templates: dict[int, np.ndarray] = {}
    for comp in edge_comps:
        vals = {int(grid[r, c]) for r, c in comp}
        assert len(vals) == 1
        color = vals.pop()
        assert color not in (BLACK, GREY)
        r0, c0, r1, c1 = _bbox(comp)
        patch = (grid[r0 : r1 + 1, c0 : c1 + 1] == color)
        assert patch.shape == (k, k)
        templates[color] = patch.copy()

    if side == "top":
        core = grid[k:, :].copy()
    elif side == "bottom":
        core = grid[:-k, :].copy()
    elif side == "left":
        core = grid[:, k:].copy()
    else:
        core = grid[:, :-k].copy()
    return core, templates, side, k


def _stamp(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = grid.shape
    mh, mw = mask.shape
    for r in range(mh):
        for c in range(mw):
            if not mask[r, c]:
                continue
            gr, gc = top + r, left + c
            if 0 <= gr < h and 0 <= gc < w and grid[gr, gc] != GREY:
                grid[gr, gc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    根据封闭区域内标记像素的颜色选择边缘模板，并按标记相对区域中心的方位旋转后盖印。
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    core, templates, _side, _k = _detect_palette(g)
    h, w = core.shape
    passable = core != GREY
    out = core.copy()

    for comp in _components(passable):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue

        markers = [(r, c) for r, c in comp if int(core[r, c]) not in (BLACK, GREY)]
        assert len(markers) == 1
        marker = markers[0]
        color = int(core[marker])
        assert color in templates

        direction = _direction_from_region(comp, marker)
        stamp = _rotate_template(templates[color], direction)
        _stamp(out, stamp, marker[0], marker[1], color)

    return out


def _make_corner_notch(mask: np.ndarray, corner: str, nh: int, nw: int) -> np.ndarray:
    test = mask.copy()
    if corner == "tl":
        test[:nh, :nw] = False
    elif corner == "tr":
        test[:nh, -nw:] = False
    elif corner == "bl":
        test[-nh:, :nw] = False
    else:
        test[-nh:, -nw:] = False
    return test


def _make_region_mask(
    k: int,
    direction: str,
    rng: random.Random,
    crowded: bool,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    size_hi = k + (3 if crowded else 5)
    for _ in range(120):
        h = rng.randint(k + 2, size_hi)
        w = rng.randint(k + 2, size_hi)

        if direction in ("up", "down"):
            ar = 0 if direction == "up" else h - k
            center_c = (w - k) // 2
            c_choices = [c for c in range(w - k + 1) if abs(c - center_c) <= 1]
            ac = rng.choice(c_choices)
        else:
            ac = 0 if direction == "left" else w - k
            center_r = (h - k) // 2
            r_choices = [r for r in range(h - k + 1) if abs(r - center_r) <= 1]
            ar = rng.choice(r_choices)

        mask = np.ones((h, w), dtype=bool)
        anchor = np.zeros((h, w), dtype=bool)
        anchor[ar : ar + k, ac : ac + k] = True

        corners = ["tl", "tr", "bl", "br"]
        rng.shuffle(corners)
        target_cuts = rng.randint(1, 3)
        cuts = 0
        for corner in corners:
            max_h = max(1, min(h - 2, h // 2 + 1))
            max_w = max(1, min(w - 2, w // 2 + 1))
            nh = rng.randint(1, max_h)
            nw = rng.randint(1, max_w)
            test = _make_corner_notch(mask, corner, nh, nw)
            if np.any(anchor & ~test):
                continue
            if int(test.sum()) < k * k + 4:
                continue
            if not _is_connected(test):
                continue
            mask = test
            cuts += 1
            if cuts >= target_cuts:
                break

        if cuts == 0:
            continue

        marker = (ar, ac)
        dr, dc = _marker_offsets(list(map(tuple, np.argwhere(mask))), marker)
        if direction == "up" and not (dr < 0 and abs(dr) > abs(dc) + 0.4):
            continue
        if direction == "down" and not (dr > 0 and abs(dr) > abs(dc) + 0.4):
            continue
        if direction == "left" and not (dc < 0 and abs(dc) > abs(dr) + 0.4):
            continue
        if direction == "right" and not (dc > 0 and abs(dc) > abs(dr) + 0.4):
            continue
        if _direction_from_region(list(map(tuple, np.argwhere(mask))), marker) != direction:
            continue
        return mask, marker
    return None


def _place_regions(
    core_h: int,
    core_w: int,
    specs: list[dict],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    occ = np.zeros((core_h, core_w), dtype=bool)
    placements: list[tuple[int, int]] = [(-1, -1)] * len(specs)
    order = sorted(range(len(specs)), key=lambda i: int(np.prod(specs[i]["mask"].shape)), reverse=True)

    for idx in order:
        mask = specs[idx]["mask"]
        h, w = mask.shape
        candidates = []
        for r0 in range(1, core_h - h):
            for c0 in range(1, core_w - w):
                if occ[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        placements[idx] = (r0, c0)
        occ[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True

    return placements


def _draw_region(
    core: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    color: int,
    marker: tuple[int, int],
) -> None:
    h, w = core.shape
    walls: set[tuple[int, int]] = set()
    for rr, cc in np.argwhere(mask):
        for dr, dc in DIR4:
            nr, nc = int(rr) + dr, int(cc) + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                continue
            gr, gc = top + int(rr) + dr, left + int(cc) + dc
            if 0 <= gr < h and 0 <= gc < w:
                walls.add((gr, gc))
    for gr, gc in walls:
        core[gr, gc] = GREY
    core[top + marker[0], left + marker[1]] = color


def _choose_starts(length: int, size: int, count: int, rng: random.Random) -> list[int] | None:
    lo = 1
    hi = length - size - 1
    if hi < lo:
        return None
    for _ in range(200):
        starts: list[int] = []
        positions = list(range(lo, hi + 1))
        rng.shuffle(positions)
        for pos in positions:
            if all(pos >= old + size + 1 or old >= pos + size + 1 for old in starts):
                starts.append(pos)
                if len(starts) == count:
                    return sorted(starts)
    return None


def _assemble_input(
    core: np.ndarray,
    side: str,
    k: int,
    templates: dict[int, np.ndarray],
    rng: random.Random,
) -> np.ndarray | None:
    core_h, core_w = core.shape
    colors = list(templates)
    axis_len = core_w if side in ("top", "bottom") else core_h
    starts = _choose_starts(axis_len, k, len(colors), rng)
    if starts is None:
        return None

    order = colors[:]
    rng.shuffle(order)
    if side in ("top", "bottom"):
        out = np.full((core_h + k, core_w), BLACK, dtype=int)
        if side == "top":
            out[k:, :] = core
            row0 = 0
        else:
            out[:core_h, :] = core
            row0 = core_h
        for color, start in zip(order, starts):
            patch = templates[color]
            block = out[row0 : row0 + k, start : start + k]
            block[patch] = color
    else:
        out = np.full((core_h, core_w + k), BLACK, dtype=int)
        if side == "left":
            out[:, k:] = core
            col0 = 0
        else:
            out[:, :core_w] = core
            col0 = core_w
        for color, start in zip(order, starts):
            patch = templates[color]
            block = out[start : start + k, col0 : col0 + k]
            block[patch] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    k = rng.choice([3, 4])
    side = rng.choice(["top", "bottom", "left", "right"])
    core_h = rng.randint(22, 30)
    core_w = rng.randint(22, 30)

    axis_len = core_w if side in ("top", "bottom") else core_h
    max_regions = min(4, max(0, (axis_len - 1) // (k + 1)))
    if max_regions < 2:
        return None
    region_count = rng.randint(2, max_regions)
    crowded = region_count >= 4

    directions = list(DIRECTIONS)
    rng.shuffle(directions)
    directions = directions[:region_count]
    colors = rng.sample(list(PALETTE_COLORS), region_count)

    template_choices = TEMPLATE_LIBRARY[k][:]
    rng.shuffle(template_choices)
    templates = {color: template_choices[i % len(template_choices)].copy() for i, color in enumerate(colors)}

    specs: list[dict] = []
    for color, direction in zip(colors, directions):
        built = _make_region_mask(k, direction, rng, crowded)
        if built is None:
            return None
        mask, marker = built
        specs.append({"color": color, "direction": direction, "mask": mask, "marker": marker})

    placements = _place_regions(core_h, core_w, specs, rng)
    if placements is None:
        return None

    core = np.full((core_h, core_w), BLACK, dtype=int)
    for spec, (r0, c0) in zip(specs, placements):
        _draw_region(core, spec["mask"], r0, c0, spec["color"], spec["marker"])

    expected = core.copy()
    for spec, (r0, c0) in zip(specs, placements):
        stamp = _rotate_template(templates[spec["color"]], spec["direction"])
        mr, mc = spec["marker"]
        _stamp(expected, stamp, r0 + mr, c0 + mc, spec["color"])

    inp = _assemble_input(core, side, k, templates, rng)
    if inp is None:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")


_validate_templates()
