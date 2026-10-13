from __future__ import annotations

import numpy as np


_RNG = np.random.default_rng()
_DIRS = ("up", "right", "down", "left")
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": np.array(pts, dtype=int)})
    return comps


def _legend_sizes(grid: np.ndarray, bg: int) -> dict[int, int]:
    entries = [int(grid[r, -1]) for r in range(grid.shape[0]) if int(grid[r, -1]) != bg]
    assert entries, "expected a non-empty legend in the rightmost column"
    assert len(entries) == len(set(entries)), "legend colors must be unique"
    return {color: i + 1 for i, color in enumerate(entries)}


def _pattern(direction: str, arm_len: int) -> np.ndarray:
    pat = np.zeros((arm_len, arm_len), dtype=bool)
    if direction == "up":
        pat[-1, :] = True
        pat[:, -1] = True
    elif direction == "right":
        pat[-1, :] = True
        pat[:, 0] = True
    elif direction == "down":
        pat[0, :] = True
        pat[:, 0] = True
    elif direction == "left":
        pat[0, :] = True
        pat[:, -1] = True
    else:
        raise ValueError(f"unknown direction: {direction}")
    return pat


def _parse_l(grid: np.ndarray, comp: dict[str, object]) -> dict[str, int]:
    coords = np.asarray(comp["coords"], dtype=int)
    color = int(comp["color"])
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == w and h >= 2, "every object must be a square-bbox L"
    arm_len = h
    sub = grid[r0 : r1 + 1, c0 : c1 + 1] == color
    matches = [direction for direction in _DIRS if np.array_equal(sub, _pattern(direction, arm_len))]
    assert len(matches) == 1, "every component must match exactly one L orientation"
    return {
        "color": color,
        "direction": matches[0],
        "arm_len": arm_len,
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
    }


def _paint_rect(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    h, w = out.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(h - 1, int(r1))
    cc1 = min(w - 1, int(c1))
    if rr0 <= rr1 and cc0 <= cc1:
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a same-color rectangle out of each L object using the right-column color legend.

    Input structure:
    - The background is the most common color.
    - The rightmost column contains a vertical legend of unique non-background colors; reading those legend pixels from top to bottom assigns sizes 1, 2, 3, ...
    - Elsewhere the grid contains several monochrome L-shaped objects, each formed by one full bbox edge row and one full bbox edge column of equal length, so each L leaves exactly one bbox corner missing.

    Rule:
    - For each L object, find the matching legend color and read its size S from the legend order.
    - Decode the L direction from its missing bbox corner: missing top-left means grow up, missing top-right means grow right, missing bottom-right means grow down, and missing bottom-left means grow left.
    - Paint an S-deep solid rectangle of the same color adjacent to that side, spanning the full row-or-column extent of the L's square bbox.
    - Keep the original L pixels and keep the legend column unchanged.

    Key insight:
    - The L geometry does not encode the depth directly; it only reveals which side is open. The actual growth depth comes from a separate color-to-rank lookup in the legend column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2

    bg = _background_color(g)
    size_map = _legend_sizes(g, bg)

    work = g.copy()
    work[:, -1] = int(bg)
    objects = [_parse_l(work, comp) for comp in _components(work, bg)]

    out = g.copy()
    for obj in objects:
        color = int(obj["color"])
        depth = int(size_map[color])
        r0, c0, r1, c1 = (int(obj[k]) for k in ("r0", "c0", "r1", "c1"))
        direction = str(obj["direction"])
        if direction == "up":
            _paint_rect(out, r0 - depth, c0, r0 - 1, c1, color)
        elif direction == "right":
            _paint_rect(out, r0, c1 + 1, r1, c1 + depth, color)
        elif direction == "down":
            _paint_rect(out, r1 + 1, c0, r1 + depth, c1, color)
        else:
            _paint_rect(out, r0, c0 - depth, r1, c0 - 1, color)

    out[:, -1] = g[:, -1]
    return out


def _spec(color: int, direction: str, arm_len: int, depth: int) -> dict[str, int | str]:
    if direction in {"up", "down"}:
        foot_h = arm_len + depth
        foot_w = arm_len
    else:
        foot_h = arm_len
        foot_w = arm_len + depth
    return {
        "color": int(color),
        "direction": direction,
        "arm_len": int(arm_len),
        "depth": int(depth),
        "foot_h": int(foot_h),
        "foot_w": int(foot_w),
    }


def _bbox_origin(spec: dict[str, int | str]) -> tuple[int, int]:
    direction = str(spec["direction"])
    depth = int(spec["depth"])
    if direction == "up":
        return depth, 0
    if direction == "left":
        return 0, depth
    return 0, 0


def _fill_bounds(spec: dict[str, int | str], top: int, left: int) -> tuple[int, int, int, int]:
    direction = str(spec["direction"])
    arm_len = int(spec["arm_len"])
    depth = int(spec["depth"])
    if direction == "up":
        return top, left, top + depth - 1, left + arm_len - 1
    if direction == "right":
        return top, left + arm_len, top + arm_len - 1, left + arm_len + depth - 1
    if direction == "down":
        return top + arm_len, left, top + arm_len + depth - 1, left + arm_len - 1
    return top, left, top + arm_len - 1, left + depth - 1


def _can_place(occ: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occ.shape[0], top + h + 1)
    c1 = min(occ.shape[1], left + w + 1)
    return not bool(np.any(occ[r0:r1, c0:c1]))


def _place_specs(specs: list[dict[str, int | str]], height: int, usable_width: int) -> list[dict[str, int | str]] | None:
    occ = np.zeros((height, usable_width), dtype=bool)
    placed: list[dict[str, int | str] | None] = [None] * len(specs)
    order = sorted(
        range(len(specs)),
        key=lambda i: int(specs[i]["foot_h"]) * int(specs[i]["foot_w"]),
        reverse=True,
    )
    for idx in order:
        spec = specs[idx]
        foot_h = int(spec["foot_h"])
        foot_w = int(spec["foot_w"])
        if foot_h > height or foot_w > usable_width:
            return None
        for _ in range(300):
            top = int(_RNG.integers(0, height - foot_h + 1))
            left = int(_RNG.integers(0, usable_width - foot_w + 1))
            if not _can_place(occ, top, left, foot_h, foot_w):
                continue
            occ[top : top + foot_h, left : left + foot_w] = True
            item = dict(spec)
            item["top"] = top
            item["left"] = left
            placed[idx] = item
            break
        else:
            return None
    return [dict(item) for item in placed if item is not None]


def _draw_l(grid: np.ndarray, spec: dict[str, int | str], bg: int) -> None:
    top = int(spec["top"])
    left = int(spec["left"])
    arm_len = int(spec["arm_len"])
    color = int(spec["color"])
    box_r0, box_c0 = _bbox_origin(spec)
    patch = np.full((arm_len, arm_len), int(bg), dtype=int)
    patch[_pattern(str(spec["direction"]), arm_len)] = int(color)
    grid[top + box_r0 : top + box_r0 + arm_len, left + box_c0 : left + box_c0 + arm_len] = np.where(
        patch != int(bg),
        patch,
        grid[top + box_r0 : top + box_r0 + arm_len, left + box_c0 : left + box_c0 + arm_len],
    )


def _build_grids(
    height: int,
    width: int,
    bg: int,
    specs: list[dict[str, int | str]],
    legend_order: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), int(bg), dtype=int)
    out = np.full((height, width), int(bg), dtype=int)

    for spec in specs:
        _draw_l(inp, spec, bg)
        _draw_l(out, spec, bg)
        fr0, fc0, fr1, fc1 = _fill_bounds(spec, int(spec["top"]), int(spec["left"]))
        out[fr0 : fr1 + 1, fc0 : fc1 + 1] = int(spec["color"])

    legend_top = int(_RNG.integers(0, height - len(legend_order) + 1))
    for offset, color in enumerate(legend_order):
        inp[legend_top + offset, -1] = int(color)
        out[legend_top + offset, -1] = int(color)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    bg = int(_RNG.integers(0, 10))
    palette = [color for color in range(10) if color != bg]

    count = int(_RNG.choice([3, 4, 5]))
    colors = [int(c) for c in _RNG.choice(palette, size=count, replace=False)]
    legend_order = [int(c) for c in _RNG.permutation(colors)]
    size_map = {color: i + 1 for i, color in enumerate(legend_order)}

    directions = list(_RNG.permutation(_DIRS))
    while len(directions) < count:
        directions.extend(list(_RNG.permutation(_DIRS)))
    directions = directions[:count]

    arm_lengths = [int(_RNG.integers(2, 7)) for _ in range(count)]
    if len(set(arm_lengths)) == 1:
        arm_lengths[0] += 1

    specs = [
        _spec(color, direction, arm_len, size_map[color])
        for color, direction, arm_len in zip(colors, directions, arm_lengths)
    ]

    min_h = max(max(int(spec["foot_h"]) for spec in specs) + 1, len(legend_order) + 1, 12)
    min_usable_w = max(max(int(spec["foot_w"]) for spec in specs) + 1, 10)
    total_input_fg = sum(2 * int(spec["arm_len"]) - 1 for spec in specs) + len(legend_order)
    total_foot_area = sum(int(spec["foot_h"]) * int(spec["foot_w"]) for spec in specs)

    for _ in range(120):
        if min_h > 30 or min_usable_w > 28:
            return None
        height = int(_RNG.integers(min_h, 31))
        usable_width = int(_RNG.integers(min_usable_w, 29))
        width = usable_width + 2  # one blank spacer column plus the legend column
        if height * usable_width < total_foot_area + 12 * count:
            continue
        if height * width <= 2 * total_input_fg:
            continue
        placed = _place_specs(specs, height, usable_width)
        if placed is None:
            continue

        inp, expected = _build_grids(height, width, bg, placed, legend_order)
        if _background_color(inp) != bg:
            continue
        if np.array_equal(inp, expected):
            continue
        if not np.array_equal(main(inp), expected):
            continue
        return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after 300 attempts")
