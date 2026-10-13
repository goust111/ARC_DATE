from __future__ import annotations

import random

import numpy as np


_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_BG = 0
_COLORS = tuple(range(1, 10))
_GEN_RETRIES = 400


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in _NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _monochrome_components(
    grid: np.ndarray,
    bg: int,
    blocked: np.ndarray | None = None,
) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool) if blocked is None else blocked.copy()
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _shape_signature(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], bytes]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.uint8)
    for r, c in coords:
        mask[r - r0, c - c0] = 1
    return tuple(map(int, mask.shape)), mask.tobytes()


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _mask_has_hole(mask: np.ndarray) -> bool:
    if not np.any(~mask):
        return False
    h, w = mask.shape
    for comp in _mask_components(~mask):
        if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in comp):
            return True
    return False


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[dict[int, int], np.ndarray]] = []
    for coords in _mask_components(grid != int(bg)):
        r0, c0, r1, c1 = _bbox(coords)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if ww != 2 or len(coords) != hh * ww:
            continue
        if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == int(bg)):
            continue
        mapping: dict[int, int] = {}
        ok = True
        for rr in range(hh):
            left = int(patch[rr, 0])
            right = int(patch[rr, 1])
            if left == right or left in mapping:
                ok = False
                break
            mapping[left] = right
        if not ok:
            continue
        mask = np.zeros_like(grid, dtype=bool)
        for r, c in coords:
            mask[r, c] = True
        candidates.append((mapping, mask))
    assert len(candidates) == 1, "expected a unique edge legend"
    return candidates[0]


def _extract_container_hole(
    grid: np.ndarray,
    comp: dict,
    bg: int,
) -> list[tuple[int, int]] | None:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    if np.any((patch != color) & (patch != int(bg))):
        return None
    if not (
        np.all(patch[0, :] == color)
        and np.all(patch[-1, :] == color)
        and np.all(patch[:, 0] == color)
        and np.all(patch[:, -1] == color)
    ):
        return None
    holes = _mask_components(patch == int(bg))
    if len(holes) != 1:
        return None
    hole = holes[0]
    hh, ww = patch.shape
    if any(r in (0, hh - 1) or c in (0, ww - 1) for r, c in hole):
        return None
    return [(r0 + r, c0 + c) for r, c in hole]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each free key shape into the matching container cavity and recolor it via the edge legend.

    Input structure:
    - The background is the dominant black color.
    - One edge contains a small n×2 legend block; each row is a horizontal color pair
      whose left cell is a container color and whose right cell is the fill color for that container.
    - Away from the legend are several monochrome objects: some are rectangular outer shells
      with one enclosed background cavity, and the rest are solid key shapes on the background.

    Rule:
    - Read the edge legend as a container-color to result-color mapping.
    - For each shell object, extract its unique enclosed cavity shape.
    - Find a solid key object whose shape matches that cavity exactly up to translation only.
    - Remove the matched key from its original location and repaint the cavity cells with the
      mapped result color, leaving the shell and legend pixels unchanged.

    Key insight:
    The key is matched against the cavity's negative-space silhouette, then recolored by an
    independent visible legend before being embedded into the shell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    mapping, legend_mask = _find_legend(g, bg)

    containers_by_shape: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    keys_by_shape: dict[tuple[tuple[int, int], bytes], list[dict]] = {}

    for comp in _monochrome_components(g, bg, blocked=legend_mask):
        hole_coords = _extract_container_hole(g, comp, bg)
        if hole_coords is not None:
            color = int(comp["color"])
            assert color in mapping, "every container color must appear in the legend"
            sig = _shape_signature(hole_coords)
            containers_by_shape.setdefault(sig, []).append(
                {"color": color, "hole_coords": hole_coords}
            )
            continue

        sig = _shape_signature(comp["coords"])
        r0, c0, r1, c1 = _bbox(comp["coords"])
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp["coords"]:
            mask[r - r0, c - c0] = True
        assert not _mask_has_hole(mask), "keys must be solid shapes"
        keys_by_shape.setdefault(sig, []).append(comp)

    assert containers_by_shape, "expected at least one container"
    assert keys_by_shape, "expected at least one key"
    assert set(containers_by_shape) == set(keys_by_shape), "container cavities and key shapes must match"

    out = g.copy()
    for sig, containers in containers_by_shape.items():
        keys = keys_by_shape.get(sig, [])
        assert len(keys) == len(containers), "shape counts must match between keys and cavities"
        for key in keys:
            for r, c in key["coords"]:
                out[r, c] = int(bg)
        for container in containers:
            fill_color = int(mapping[int(container["color"])])
            for r, c in container["hole_coords"]:
                out[r, c] = fill_color

    return out


def _connected_mask(h: int, w: int, area: int) -> np.ndarray:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in _NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        assert frontier, "connected growth must have a frontier"
        cells.add(random.choice(tuple(frontier)))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _crop_mask(mask)


def _random_key_shape() -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        max_area = min(h * w, 9)
        min_area = min(max_area, max(3, (h * w + 1) // 2))
        area = random.randint(min_area, max_area)
        mask = _connected_mask(h, w, area)
        if _mask_has_hole(mask):
            continue
        if min(mask.shape) == 1 and random.random() < 0.7:
            continue
        if bool(mask.all()) and random.random() < 0.65:
            continue
        return mask
    raise RuntimeError("failed to generate a cavity shape")


def _container_patch(container_color: int, hole_mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    hole_h, hole_w = hole_mask.shape
    top = random.randint(1, 2)
    bottom = random.randint(1, 3)
    left = random.randint(1, 2)
    right = random.randint(1, 3)
    patch = np.full((hole_h + top + bottom, hole_w + left + right), container_color, dtype=int)
    window = patch[top : top + hole_h, left : left + hole_w]
    window[hole_mask] = _BG
    return patch, top, left


def _key_patch(key_color: int, hole_mask: np.ndarray) -> np.ndarray:
    patch = np.full(hole_mask.shape, _BG, dtype=int)
    patch[hole_mask] = key_color
    return patch


def _boxes_intersect(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded_box(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    h: int,
    w: int,
    gap: int = 1,
) -> tuple[int, int, int, int]:
    return (
        max(0, r0 - gap),
        max(0, c0 - gap),
        min(h - 1, r1 + gap),
        min(w - 1, c1 + gap),
    )


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    reserved: list[tuple[int, int, int, int]],
) -> tuple[int, int]:
    h, w = grid.shape
    ph, pw = patch.shape
    max_r = h - ph - 1
    max_c = w - pw - 1
    assert max_r >= 1 and max_c >= 1, "patch must fit with a one-cell margin"
    positions = [(r, c) for r in range(1, max_r + 1) for c in range(1, max_c + 1)]
    random.shuffle(positions)
    for r0, c0 in positions:
        box = _expanded_box(r0, c0, r0 + ph - 1, c0 + pw - 1, h, w)
        if any(_boxes_intersect(box, other) for other in reserved):
            continue
        region = grid[r0 : r0 + ph, c0 : c0 + pw]
        if np.any((patch != _BG) & (region != _BG)):
            continue
        region[patch != _BG] = patch[patch != _BG]
        reserved.append(box)
        return r0, c0
    raise RuntimeError("failed to place patch")


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    region = grid[r0 : r0 + ph, c0 : c0 + pw]
    region[patch != _BG] = patch[patch != _BG]


def _legend_block(
    container_colors: list[int],
    result_colors: list[int],
) -> np.ndarray:
    pairs = list(zip(container_colors, result_colors, strict=True))
    random.shuffle(pairs)
    legend = np.zeros((len(pairs), 2), dtype=int)
    for idx, (src, dst) in enumerate(pairs):
        legend[idx, 0] = src
        legend[idx, 1] = dst
    return legend


def _place_legend(
    grid: np.ndarray,
    legend: np.ndarray,
    side: str,
) -> tuple[int, int, int, int]:
    h, w = grid.shape
    lh, lw = legend.shape
    assert lw == 2
    if side == "top":
        r0 = 0
        c0 = random.randint(0, w - lw)
    elif side == "bottom":
        r0 = h - lh
        c0 = random.randint(0, w - lw)
    elif side == "left":
        r0 = random.randint(0, h - lh)
        c0 = 0
    else:
        assert side == "right"
        r0 = random.randint(0, h - lh)
        c0 = w - lw
    grid[r0 : r0 + lh, c0 : c0 + lw] = legend
    return r0, c0, r0 + lh - 1, c0 + lw - 1


def _make_case() -> tuple[np.ndarray, np.ndarray]:
    n = random.randint(1, 3)
    min_side = 14 + 4 * (n - 1)
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)

    colors = random.sample(_COLORS, 3 * n)
    container_colors = colors[:n]
    result_colors = colors[n : 2 * n]
    key_colors = colors[2 * n :]

    input_grid = np.full((h, w), _BG, dtype=int)
    expected = np.full((h, w), _BG, dtype=int)

    legend = _legend_block(container_colors, result_colors)
    side = random.choice(["top", "bottom", "left", "right"])
    legend_box = _place_legend(input_grid, legend, side)
    _paint_patch(expected, legend, legend_box[0], legend_box[1])
    reserved = [_expanded_box(*legend_box, h, w)]

    shape_sigs: set[tuple[tuple[int, int], bytes]] = set()
    specs: list[dict] = []
    for idx in range(n):
        for _ in range(200):
            hole_mask = _random_key_shape()
            sig = (tuple(map(int, hole_mask.shape)), hole_mask.astype(np.uint8).tobytes())
            if sig in shape_sigs:
                continue
            shape_sigs.add(sig)
            container_patch, hole_r0, hole_c0 = _container_patch(container_colors[idx], hole_mask)
            key_patch = _key_patch(key_colors[idx], hole_mask)
            specs.append(
                {
                    "hole_mask": hole_mask,
                    "container_patch": container_patch,
                    "key_patch": key_patch,
                    "result_color": result_colors[idx],
                    "hole_r0": hole_r0,
                    "hole_c0": hole_c0,
                }
            )
            break
        else:
            raise RuntimeError("failed to produce unique shapes")

    order = sorted(
        [(spec["container_patch"].size, "container", idx) for idx, spec in enumerate(specs)]
        + [(spec["key_patch"].size, "key", idx) for idx, spec in enumerate(specs)],
        reverse=True,
    )

    for _size, kind, idx in order:
        spec = specs[idx]
        patch = spec["container_patch"] if kind == "container" else spec["key_patch"]
        r0, c0 = _place_patch(input_grid, patch, reserved)
        if kind == "container":
            spec["container_pos"] = (r0, c0)
        else:
            spec["key_pos"] = (r0, c0)

    for spec in specs:
        cr0, cc0 = spec["container_pos"]
        container_patch = spec["container_patch"]
        _paint_patch(expected, container_patch, cr0, cc0)

        hole_mask = spec["hole_mask"]
        hr0 = cr0 + spec["hole_r0"]
        hc0 = cc0 + spec["hole_c0"]
        hole_region = expected[hr0 : hr0 + hole_mask.shape[0], hc0 : hc0 + hole_mask.shape[1]]
        hole_region[hole_mask] = spec["result_color"]

    return input_grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        try:
            inp, out = _make_case()
        except RuntimeError:
            continue
        try:
            if np.array_equal(main(inp), out):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
