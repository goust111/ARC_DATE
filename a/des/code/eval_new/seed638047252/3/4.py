from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
KIND_ORDER = ("h", "v", "hv")
AREA_BANDS = {
    2: ((4, 9), (11, 18)),
    3: ((4, 8), (10, 14), (16, 22)),
}
GENERATION_ATTEMPTS = 300
SHAPE_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(np.asarray(coords, dtype=int))

    comps.sort(key=lambda coords: (int(coords[:, 0].min()), int(coords[:, 1].min())))
    return comps


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _classify_mask(mask: np.ndarray) -> str:
    has_h = bool(np.array_equal(mask, np.flipud(mask)))
    has_v = bool(np.array_equal(mask, np.fliplr(mask)))
    if has_h and has_v:
        return "hv"
    if has_h:
        return "h"
    if has_v:
        return "v"
    return "none"


def _extract_components(grid: np.ndarray) -> list[dict[str, object]]:
    comps = _components(grid != BACKGROUND)
    info: list[dict[str, object]] = []

    for coords in comps:
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "each connected region must be monochrome"
        color = next(iter(colors))

        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        local = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
        shifted = coords - np.array([r0, c0], dtype=int)
        local[shifted[:, 0], shifted[:, 1]] = True

        kind = _classify_mask(local)
        assert kind != "none", "every sprite must have at least one symmetry axis"
        info.append(
            {
                "coords": coords,
                "color": color,
                "area": int(coords.shape[0]),
                "kind": kind,
            }
        )

    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each sprite according to the largest-area sprite that shares its symmetry-axis type.

    Input structure:
    - The background is black.
    - The grid contains several separated monochrome connected sprites, each with a different color.
    - Every sprite belongs to exactly one of three symmetry classes after cropping to its bounding box:
      horizontal-only, vertical-only, or both horizontal and vertical.

    Rule:
    - Find every non-background connected sprite and crop it to its minimal bounding box.
    - Test whether the cropped mask is symmetric across the bounding-box horizontal midline, vertical midline, or both, and group sprites by that symmetry type.
    - Inside each group, find the sprite with the largest area and keep its color as the group's target color.
    - Repaint every sprite in that group with the target color while preserving each sprite's original shape and position.

    Key insight:
    The grouping rule is hidden in each sprite's intrinsic reflection symmetry, and the recoloring color is not arbitrary: it is taken from the largest sprite inside the same symmetry group.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sprites = _extract_components(g)
    out = g.copy()

    for kind in KIND_ORDER:
        group = [sprite for sprite in sprites if sprite["kind"] == kind]
        if not group:
            continue
        target_color = int(max(group, key=lambda sprite: int(sprite["area"]))["color"])
        for sprite in group:
            coords = sprite["coords"]
            out[coords[:, 0], coords[:, 1]] = target_color

    return out


def _orbit(kind: str, h: int, w: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    if kind == "h":
        cells = {(r, c), (h - 1 - r, c)}
    elif kind == "v":
        cells = {(r, c), (r, w - 1 - c)}
    else:
        cells = {
            (rr, cc)
            for rr in {r, h - 1 - r}
            for cc in {c, w - 1 - c}
        }
    return tuple(sorted(cells))


def _all_orbits(kind: str, h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    seen: dict[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] = {}
    for r in range(h):
        for c in range(w):
            orb = _orbit(kind, h, w, r, c)
            seen[orb] = orb
    return list(seen)


def _seed_orbit(kind: str, h: int, w: int) -> tuple[tuple[int, int], ...]:
    if kind == "h":
        return _orbit(kind, h, w, (h - 1) // 2, random.randrange(w))
    if kind == "v":
        return _orbit(kind, h, w, random.randrange(h), (w - 1) // 2)
    return _orbit(kind, h, w, (h - 1) // 2, (w - 1) // 2)


def _orbit_adjacent(mask: np.ndarray, orb: tuple[tuple[int, int], ...]) -> bool:
    h, w = map(int, mask.shape)
    for r, c in orb:
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc]:
                return True
    return False


def _paint_orbit(mask: np.ndarray, orb: tuple[tuple[int, int], ...]) -> None:
    for r, c in orb:
        mask[r, c] = True


def _generate_mask(kind: str, area_lo: int, area_hi: int) -> np.ndarray | None:
    dim_ranges = {
        "h": ((3, 7), (4, 8)),
        "v": ((4, 8), (3, 7)),
        "hv": ((3, 8), (3, 8)),
    }
    h_range, w_range = dim_ranges[kind]

    for _ in range(SHAPE_ATTEMPTS):
        raw_h = random.randint(*h_range)
        raw_w = random.randint(*w_range)
        mask = np.zeros((raw_h, raw_w), dtype=bool)

        seed = _seed_orbit(kind, raw_h, raw_w)
        if len(seed) > area_hi:
            continue
        _paint_orbit(mask, seed)

        used = {seed}
        stalled = False
        while True:
            area = int(mask.sum())
            candidates = [
                orb
                for orb in _all_orbits(kind, raw_h, raw_w)
                if orb not in used and area + len(orb) <= area_hi and _orbit_adjacent(mask, orb)
            ]
            if area >= area_lo and (not candidates or random.random() < 0.35):
                break
            if not candidates:
                stalled = True
                break
            orb = random.choice(candidates)
            _paint_orbit(mask, orb)
            used.add(orb)

        if stalled:
            continue

        trimmed = _trim_mask(mask)
        if int(trimmed.sum()) < area_lo or int(trimmed.sum()) > area_hi:
            continue
        if len(_components(trimmed)) != 1:
            continue
        if _classify_mask(trimmed) != kind:
            continue
        return trimmed

    return None


def _place_masks(masks: list[np.ndarray]) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    dims = [(int(mask.shape[0]), int(mask.shape[1])) for mask in masks]
    total_box_area = sum((h + 2) * (w + 2) for h, w in dims)
    min_side = max(
        max(h for h, _ in dims) + 2,
        max(w for _, w in dims) + 2,
        int(np.ceil(total_box_area ** 0.5)),
    )

    candidates = [
        (h, w)
        for h in range(max(18, min_side), 31)
        for w in range(max(18, min_side), 31)
        if h * w >= total_box_area
    ]
    random.shuffle(candidates)
    order = sorted(range(len(masks)), key=lambda i: int(masks[i].sum()), reverse=True)

    for h, w in candidates[:80]:
        blocked = np.zeros((h, w), dtype=bool)
        positions: list[tuple[int, int] | None] = [None] * len(masks)
        ok = True

        for idx in order:
            mask = masks[idx]
            mh, mw = map(int, mask.shape)
            valid_positions: list[tuple[int, int]] = []
            for r0 in range(h - mh + 1):
                for c0 in range(w - mw + 1):
                    rs = max(0, r0 - 1)
                    re = min(h, r0 + mh + 1)
                    cs = max(0, c0 - 1)
                    ce = min(w, c0 + mw + 1)
                    if blocked[rs:re, cs:ce].any():
                        continue
                    valid_positions.append((r0, c0))

            if not valid_positions:
                ok = False
                break

            r0, c0 = random.choice(valid_positions)
            positions[idx] = (r0, c0)
            rs = max(0, r0 - 1)
            re = min(h, r0 + mh + 1)
            cs = max(0, c0 - 1)
            ce = min(w, c0 + mw + 1)
            blocked[rs:re, cs:ce] = True

        if ok:
            return (h, w), [pos for pos in positions if pos is not None]

    return None


def _render(shape: tuple[int, int], specs: list[dict[str, object]], by_group: bool) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    group_colors = {
        kind: int(max((spec for spec in specs if spec["kind"] == kind), key=lambda spec: int(spec["area"]))["color"])
        for kind in KIND_ORDER
    }

    for spec in specs:
        r0, c0 = spec["pos"]
        mask = spec["mask"]
        h, w = map(int, mask.shape)
        color = group_colors[spec["kind"]] if by_group else int(spec["color"])
        region = grid[r0 : r0 + h, c0 : c0 + w]
        region[mask] = color

    return grid


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    specs: list[dict[str, object]] = []

    for kind in KIND_ORDER:
        count = random.randint(2, 3)
        bands = list(AREA_BANDS[count])
        random.shuffle(bands)
        group_specs: list[dict[str, object]] = []

        for area_lo, area_hi in bands:
            mask = _generate_mask(kind, area_lo, area_hi)
            if mask is None:
                return None
            group_specs.append(
                {
                    "kind": kind,
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )

        areas = [int(spec["area"]) for spec in group_specs]
        if len(areas) != len(set(areas)):
            return None
        specs.extend(group_specs)

    if len(specs) > 9:
        return None

    random.shuffle(specs)
    placed = _place_masks([spec["mask"] for spec in specs])
    if placed is None:
        return None

    shape, positions = placed
    colors = random.sample(range(1, 10), len(specs))
    for spec, pos, color in zip(specs, positions, colors):
        spec["pos"] = pos
        spec["color"] = int(color)

    inp = _render(shape, specs, by_group=False)
    out = _render(shape, specs, by_group=True)

    extracted = _extract_components(inp)
    if len(extracted) != len(specs):
        return None
    counts = {kind: sum(1 for spec in extracted if spec["kind"] == kind) for kind in KIND_ORDER}
    if any(counts[kind] < 2 for kind in KIND_ORDER):
        return None
    if not np.array_equal(main(inp), out):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        built = _build_case()
        if built is not None:
            inp, _out = built
            return inp
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
