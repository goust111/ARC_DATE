from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
SEED_COLOR = 4
PORT_COLOR = 7
BORDER_COLORS = (1, 2, 3, 5, 6, 8, 9)
MAX_GRID = 30
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _border_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _rotate_offset(dr: int, dc: int, turns: int) -> tuple[int, int]:
    for _ in range(turns % 4):
        dr, dc = dc, -dr
    return dr, dc


def _in_ring(
    r: int,
    c: int,
    outer_box: tuple[int, int, int, int],
    inner_box: tuple[int, int, int, int],
) -> bool:
    or0, oc0, or1, oc1 = outer_box
    ir0, ic0, ir1, ic1 = inner_box
    if not (or0 < r < or1 and oc0 < c < oc1):
        return False
    if ir0 <= r <= ir1 and ic0 <= c <= ic1:
        return False
    return True


def _extract_structure(
    grid: np.ndarray,
) -> tuple[list[dict[str, object]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(np.min(g)) >= 0
    assert PORT_COLOR in set(map(int, g.ravel()))
    comps = _connected_components(g != BG)

    layers: list[dict[str, object]] = []
    seed_coords: list[tuple[int, int]] = []
    for coords in comps:
        colors = {int(g[r, c]) for r, c in coords}
        if colors == {SEED_COLOR}:
            seed_coords.extend(coords)
            continue

        if PORT_COLOR in colors and len(colors - {PORT_COLOR}) == 1:
            color = next(iter(colors - {PORT_COLOR}))
            assert color not in {BG, SEED_COLOR, PORT_COLOR}
            box = _bbox(coords)
            assert set(coords) == _border_cells(box), "layer must be a 1-pixel rectangle border"
            ports = [(r, c) for r, c in coords if int(g[r, c]) == PORT_COLOR]
            assert len(ports) == 1, "each layer must have one port"
            layers.append({"bbox": box, "color": int(color), "port": ports[0]})
            continue

        raise AssertionError("unexpected foreground component")

    assert len(layers) >= 2, "need at least two nested layers"
    assert seed_coords, "seed pattern is missing"

    layers.sort(key=lambda layer: -_bbox_area(layer["bbox"]))  # outer to inner
    for i in range(len(layers) - 1):
        outer = layers[i]["bbox"]
        inner = layers[i + 1]["bbox"]
        or0, oc0, or1, oc1 = outer
        ir0, ic0, ir1, ic1 = inner
        assert or0 + or1 == ir0 + ir1 and oc0 + oc1 == ic0 + ic1, "layers must be concentric"
        assert or0 < ir0 < ir1 < or1 and oc0 < ic0 < ic1 < oc1, "layers must be strictly nested"
        assert ir0 - or0 >= 2 and ic0 - oc0 >= 2, "layers need a visible gap"

    inner_box = layers[-1]["bbox"]
    ir0, ic0, ir1, ic1 = inner_box
    for r, c in seed_coords:
        assert ir0 + 1 < r < ir1 - 1 and ic0 + 1 < c < ic1 - 1, "seed must stay detached from the border"

    seed_coords.sort()
    return layers, seed_coords


def _apply_transfers(
    grid: np.ndarray,
    layers: list[dict[str, object]],
    seed_coords: list[tuple[int, int]],
) -> tuple[np.ndarray, list[bool], list[int]]:
    out = np.asarray(grid, dtype=int).copy()
    source_port = layers[-1]["port"]
    sr, sc = source_port
    rel = [(r - sr, c - sc) for r, c in seed_coords]
    clipped_flags: list[bool] = []
    kept_counts: list[int] = []
    h, w = out.shape

    for turns in range(1, len(layers)):
        target_idx = len(layers) - 1 - turns
        target = layers[target_idx]
        outer_box = target["bbox"]
        inner_box = layers[target_idx + 1]["bbox"]
        tr, tc = target["port"]
        kept = 0
        for dr, dc in rel:
            rr, cc = _rotate_offset(dr, dc, turns)
            nr, nc = tr + rr, tc + cc
            if 0 <= nr < h and 0 <= nc < w and _in_ring(nr, nc, outer_box, inner_box):
                out[nr, nc] = int(target["color"])
                kept += 1
        clipped_flags.append(kept != len(rel))
        kept_counts.append(kept)

    return out, clipped_flags, kept_counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect nested rectangle layers with orange ports, then copy the innermost yellow seed
    outward one layer at a time. Each outward step rotates the seed 90 degrees clockwise and
    recolors it to the target layer's border color, while clipping anything outside that ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    layers, seed_coords = _extract_structure(g)
    out, _clipped, _kept = _apply_transfers(g, layers, seed_coords)
    return out


def _random_odd(rng: np.random.Generator, low: int, high: int) -> int:
    choices = [v for v in range(low, high + 1) if v % 2 == 1]
    assert choices
    return int(choices[int(rng.integers(len(choices)))])


def _sample_margin_pair(rng: np.random.Generator, extra: int) -> tuple[int, int] | None:
    options = []
    for a in range(1, min(4, extra - 1) + 1):
        for b in range(1, min(4, extra - a) + 1):
            options.append((a, b))
    if not options:
        return None
    return options[int(rng.integers(len(options)))]


def _port_position(box: tuple[int, int, int, int], side: int) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if side == 0:
        return r0, mid_c
    if side == 1:
        return mid_r, c1
    if side == 2:
        return r1, mid_c
    return mid_r, c0


def _grow_mask(
    rng: np.random.Generator,
    height: int,
    width: int,
    area: int,
) -> np.ndarray | None:
    if area > height * width:
        return None
    for _ in range(50):
        start = (int(rng.integers(height)), int(rng.integers(width)))
        cells = {start}
        frontier = set(_neighbors4(*start))
        while len(cells) < area:
            choices = [
                (r, c)
                for r, c in frontier
                if 0 <= r < height and 0 <= c < width and (r, c) not in cells
            ]
            if not choices:
                break
            nr, nc = choices[int(rng.integers(len(choices)))]
            cells.add((nr, nc))
            frontier.remove((nr, nc))
            frontier.update(_neighbors4(nr, nc))
        if len(cells) != area:
            continue
        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    return None


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return mask.shape, mask.astype(np.uint8).tobytes()


def _is_asymmetric(mask: np.ndarray) -> bool:
    base = _tight_mask(mask)
    seen = {_mask_signature(base)}
    variants = []
    for turns in range(4):
        rot = np.rot90(base, -turns)
        variants.append(rot)
        variants.append(np.fliplr(rot))
    for variant in variants:
        sig = _mask_signature(_tight_mask(variant))
        seen.add(sig)
    return len(seen) == 8


def _sample_seed_coords(
    rng: np.random.Generator,
    inner_box: tuple[int, int, int, int],
) -> list[tuple[int, int]] | None:
    ir0, ic0, ir1, ic1 = inner_box
    safe_h = ir1 - ir0 - 3
    safe_w = ic1 - ic0 - 3
    if safe_h < 2 or safe_w < 2:
        return None

    for _ in range(80):
        patch_h = int(rng.integers(2, min(5, safe_h) + 1))
        patch_w = int(rng.integers(2, min(5, safe_w) + 1))
        max_area = min(7, patch_h * patch_w - 1)
        if max_area < 3:
            continue
        area = int(rng.integers(3, max_area + 1))
        mask = _grow_mask(rng, patch_h, patch_w, area)
        if mask is None or not _is_asymmetric(mask):
            continue
        top = int(rng.integers(ir0 + 2, ir1 - patch_h))
        left = int(rng.integers(ic0 + 2, ic1 - patch_w))
        coords = [
            (top + r, left + c)
            for r in range(patch_h)
            for c in range(patch_w)
            if mask[r, c]
        ]
        coords.sort()
        return coords
    return None


def _build_input(
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, object]], list[tuple[int, int]], bool] | None:
    n_layers = int(rng.integers(2, 6))
    gaps = [int(rng.integers(1, 5)) for _ in range(n_layers - 1)]
    total_inset = sum(gap + 1 for gap in gaps)
    max_inner_h = 28 - 2 * total_inset
    max_inner_w = 28 - 2 * total_inset
    if max_inner_h < 7 or max_inner_w < 7:
        return None

    inner_h = _random_odd(rng, 7, max_inner_h)
    inner_w = _random_odd(rng, 7, max_inner_w)
    outer_h = inner_h + 2 * total_inset
    outer_w = inner_w + 2 * total_inset

    row_margins = _sample_margin_pair(rng, MAX_GRID - outer_h)
    col_margins = _sample_margin_pair(rng, MAX_GRID - outer_w)
    if row_margins is None or col_margins is None:
        return None
    top, bottom = row_margins
    left, right = col_margins

    grid = np.full((outer_h + top + bottom, outer_w + left + right), BG, dtype=int)
    colors = [int(c) for c in rng.choice(BORDER_COLORS, size=n_layers, replace=False)]
    force_clip = bool(int(rng.integers(2)))

    outer_box = (top, left, top + outer_h - 1, left + outer_w - 1)
    layers: list[dict[str, object]] = []
    offset = 0
    for idx in range(n_layers):
        if idx > 0:
            offset += gaps[idx - 1] + 1
        box = (
            outer_box[0] + offset,
            outer_box[1] + offset,
            outer_box[2] - offset,
            outer_box[3] - offset,
        )
        side = int(rng.integers(4))
        port = _port_position(box, side)
        layers.append({"bbox": box, "color": colors[idx], "port": port})

        r0, c0, r1, c1 = box
        grid[r0, c0 : c1 + 1] = colors[idx]
        grid[r1, c0 : c1 + 1] = colors[idx]
        grid[r0 : r1 + 1, c0] = colors[idx]
        grid[r0 : r1 + 1, c1] = colors[idx]
        grid[port] = PORT_COLOR

    seed_coords = _sample_seed_coords(rng, layers[-1]["bbox"])
    if seed_coords is None:
        return None
    for r, c in seed_coords:
        grid[r, c] = SEED_COLOR

    return grid, layers, seed_coords, force_clip


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    built = _build_input(rng)
    if built is None:
        return None
    grid, layers, seed_coords, force_clip = built

    expected, clipped_flags, kept_counts = _apply_transfers(grid, layers, seed_coords)
    if any(count == 0 for count in kept_counts):
        return None
    if any(np.sum(expected == int(layer["color"])) == 0 for layer in layers[:-1]):
        return None
    if force_clip != any(clipped_flags):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
