import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 300


def _majority_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _side_coords(layer: dict, side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = layer["bbox"]
    if side == "top":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    return [(r, c1) for r in range(r0, r1 + 1)]


def _extract_layers(grid: np.ndarray, bg: int) -> list[dict]:
    comps = _components(grid != bg)
    assert comps, "expected at least one layer"

    layers = []
    for pts in comps:
        pts_set = set(pts)
        colors = {int(grid[r, c]) for r, c in pts}
        assert len(colors) == 1, "each layer must be monochrome in the input"
        color = next(iter(colors))

        r0, c0, r1, c1 = _bbox(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "layer bbox too small"

        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r - r0, c - c0] = True

        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True

        assert not np.any(comp_mask & ~border_mask), "layer pixels must lie on the bbox border"

        side_missing = {}
        for side in SIDES:
            coords = _side_coords({"bbox": (r0, c0, r1, c1)}, side)
            missing = [(rr, cc) for rr, cc in coords if (rr, cc) not in pts_set]
            side_missing[side] = missing

        gap_sides = [side for side, missing in side_missing.items() if len(missing) == 1]
        assert len(gap_sides) == 1, "each layer must have exactly one one-cell side gap"
        gap_side = gap_sides[0]
        gap = side_missing[gap_side][0]

        if gap_side in ("top", "bottom"):
            assert c0 < gap[1] < c1, "gap must be non-corner on a horizontal side"
        else:
            assert r0 < gap[0] < r1, "gap must be non-corner on a vertical side"

        for side in SIDES:
            if side == gap_side:
                continue
            assert len(side_missing[side]) == 0, "non-gap sides must be intact"

        layers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": color,
                "gap_side": gap_side,
                "gap": gap,
            }
        )

    layers.sort(key=lambda layer: (layer["bbox"][2] - layer["bbox"][0] + 1) * (layer["bbox"][3] - layer["bbox"][1] + 1), reverse=True)

    for outer, inner in zip(layers, layers[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert (ir0, ic0, ir1, ic1) == (or0 + 2, oc0 + 2, or1 - 2, oc1 - 2), "layers must be separated by a one-cell black gap"

    return layers


def _current_side_color(grid: np.ndarray, layer: dict) -> int:
    coords = _side_coords(layer, layer["gap_side"])
    core_coords = coords[1:-1] if len(coords) > 2 else coords

    def collect(points: list[tuple[int, int]]) -> list[int]:
        colors = []
        for r, c in points:
            if (r, c) == layer["gap"]:
                continue
            val = int(grid[r, c])
            if val != BG:
                colors.append(val)
        return colors

    colors = collect(core_coords)
    if not colors:
        colors = collect(coords)
    assert colors, "gap side must keep at least one colored pixel"
    color, _ = Counter(colors).most_common(1)[0]
    return int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate colors inward through the side gaps of nested rectangular frames.
    """
    g = np.asarray(grid, dtype=int)
    bg = _majority_color(g)
    assert bg == BG, "expected black background"

    layers = _extract_layers(g, bg)
    out = g.copy()

    for outer, inner in zip(layers, layers[1:]):
        incoming = _current_side_color(out, outer)
        for r, c in _side_coords(inner, outer["gap_side"]):
            if (r, c) != inner["gap"]:
                out[r, c] = incoming

    final_color = _current_side_color(out, layers[-1])
    r0, c0, r1, c1 = layers[-1]["bbox"]
    out[r0 + 1 : r1, c0 + 1 : c1] = final_color
    return out


def _choose_sides(rng: random.Random, n_layers: int) -> list[str]:
    sides = [rng.choice(SIDES)]
    cascade_branch = rng.choice([False, True])
    repeated = False
    for idx in range(1, n_layers):
        prev = sides[-1]
        if cascade_branch:
            if idx == n_layers - 1 and not repeated:
                side = prev
            else:
                side = prev if rng.random() < 0.5 else rng.choice([s for s in SIDES if s != prev])
            repeated = repeated or side == prev
        else:
            side = rng.choice([s for s in SIDES if s != prev])
        sides.append(side)
    return sides


def _gap_coord(rng: random.Random, bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0, rng.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return r1, rng.randint(c0 + 1, c1 - 1)
    if side == "left":
        return rng.randint(r0 + 1, r1 - 1), c0
    return rng.randint(r0 + 1, r1 - 1), c1


def _draw_layer(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, side: str, gap: tuple[int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[gap] = BG


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_layers = rng.randint(2, 5)
    inner_h = rng.randint(5, 8)
    inner_w = rng.randint(5, 8)
    outer_h = inner_h + 4 * (n_layers - 1)
    outer_w = inner_w + 4 * (n_layers - 1)

    if outer_h > 24 or outer_w > 24:
        return None

    margin_top = rng.randint(1, 4)
    margin_bottom = rng.randint(1, 4)
    margin_left = rng.randint(1, 4)
    margin_right = rng.randint(1, 4)
    h = outer_h + margin_top + margin_bottom
    w = outer_w + margin_left + margin_right

    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)
    colors = rng.sample(PALETTE, n_layers)
    sides = _choose_sides(rng, n_layers)

    layers = []
    for idx in range(n_layers):
        inset = 2 * idx
        bbox = (
            margin_top + inset,
            margin_left + inset,
            margin_top + outer_h - 1 - inset,
            margin_left + outer_w - 1 - inset,
        )
        gap = _gap_coord(rng, bbox, sides[idx])
        _draw_layer(grid, bbox, colors[idx], sides[idx], gap)
        layers.append({"bbox": bbox, "color": colors[idx], "gap_side": sides[idx], "gap": gap})

    try:
        parsed = _extract_layers(grid, BG)
    except AssertionError:
        return None

    if len(parsed) != n_layers:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid nested-frame input")
