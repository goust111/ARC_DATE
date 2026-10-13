from __future__ import annotations

import hashlib
import random
from typing import Iterable

import numpy as np


GREEN = 3
BLUE = 1
RED = 2
BLACK = 0

GRID_SIDES = (13, 15)
TEMPLATE_SIZES = ((3, 3), (3, 4), (4, 3), (4, 4))
DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
CORNER_NAMES = ("tl", "tr", "bl", "br")
MAX_GENERATION_ATTEMPTS = 600

_SEEN_INPUTS: set[str] = set()
_SEEN_OUTPUTS: set[str] = set()


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _ray_cells(center: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = center
    dr, dc = direction
    out = [(r, c)]
    while True:
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        out.append((nr, nc))
        r, c = nr, nc
        if r == 0 or r == h - 1 or c == 0 or c == w - 1:
            break
    return out


def _template_component(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    mask = (grid == BLUE) | (grid == RED)
    comps = _component_coords(mask)
    assert len(comps) == 1, "expected exactly one blue/red template component"
    bbox = _bbox_from_coords(comps[0])
    template = _crop_bbox(grid, bbox)
    assert np.all((template == BLUE) | (template == RED)), "template bbox must be fully tiled by blue/red"
    return template, bbox


def _template_anchor(template: np.ndarray) -> tuple[int, int]:
    h, w = template.shape
    return (h - 1) // 2, (w - 1) // 2


def _stamp_endpoint(
    out: np.ndarray,
    template: np.ndarray,
    endpoint: tuple[int, int],
) -> None:
    h, w = template.shape
    ar, ac = _template_anchor(template)
    top = endpoint[0] - ar
    left = endpoint[1] - ac
    for r in range(h):
        for c in range(w):
            rr = top + r
            cc = left + c
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                value = int(template[r, c])
                out[rr, cc] = GREEN if value == BLUE else RED


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire a diagonal ray from the center and stamp the corner template at the ray endpoint.

    Input structure:
    - The grid is an odd square on a black background with a green center pixel.
    - All other green pixels lie strictly inside one diagonal quadrant relative to the center and serve as direction indicators.
    - One corner contains a single fully tiled blue/red rectangular template component that does not overlap the ray source.

    Rule:
    - Use the non-center green pixels to determine one diagonal direction from the center.
    - Extend a green diagonal ray from the center to the first corner-edge endpoint in that direction.
    - Extract the blue/red template by its tight bounding box, align the template's floored center to the ray endpoint, and stamp it there with blue recolored to green and red kept red, clipping out-of-bounds cells.
    - Erase the original blue/red template and erase every original green indicator pixel that is not on the final ray.

    Key insight:
    - The scattered green pixels do not survive directly; they only encode a quadrant, whose ray endpoint becomes the geometric anchor for a recolored copy of the corner template.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and h == w and h % 2 == 1, "expected an odd square grid"

    center = (h // 2, w // 2)
    assert int(g[center]) == GREEN, "expected a green center pixel"

    marker_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == GREEN) if tuple(map(int, rc)) != center]
    assert marker_coords, "expected off-center green markers"

    directions = {(_sign(r - center[0]), _sign(c - center[1])) for r, c in marker_coords}
    assert len(directions) == 1, "green markers must indicate exactly one diagonal quadrant"
    direction = next(iter(directions))
    assert direction[0] != 0 and direction[1] != 0, "markers must be off the center axes"

    ray = _ray_cells(center, direction, g.shape)
    ray_set = set(ray)
    endpoint = ray[-1]

    template, template_bbox = _template_component(g)

    out = g.copy()
    for r, c in ray:
        out[r, c] = GREEN
    _stamp_endpoint(out, template, endpoint)

    tr0, tc0, tr1, tc1 = template_bbox
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = BLACK

    for r, c in marker_coords:
        if (r, c) not in ray_set:
            out[r, c] = BLACK

    return out


def _hash_grid(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _corner_bounds(side: int, th: int, tw: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, side - tw
    if corner == "bl":
        return side - th, 0
    if corner == "br":
        return side - th, side - tw
    raise ValueError(f"unknown corner {corner}")


def _corner_from_direction(direction: tuple[int, int]) -> str:
    return {
        (-1, -1): "tl",
        (-1, 1): "tr",
        (1, -1): "bl",
        (1, 1): "br",
    }[direction]


def _opposite_corner(corner: str) -> str:
    return {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}[corner]


def _quadrant_cells(
    side: int,
    center: tuple[int, int],
    direction: tuple[int, int],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    dr, dc = direction
    cells = []
    for r in range(side):
        for c in range(side):
            if (r, c) in forbidden or (r, c) == center:
                continue
            if _sign(r - center[0]) == dr and _sign(c - center[1]) == dc:
                cells.append((r, c))
    return cells


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = [
        mask,
        np.rot90(mask, 1),
        np.rot90(mask, 2),
        np.rot90(mask, 3),
        np.fliplr(mask),
        np.flipud(mask),
        np.fliplr(np.rot90(mask, 1)),
        np.flipud(np.rot90(mask, 1)),
    ]
    uniq: list[np.ndarray] = []
    keys: set[bytes] = set()
    for arr in variants:
        key = arr.astype(np.uint8).tobytes()
        if key not in keys:
            keys.add(key)
            uniq.append(arr.copy())
    return uniq


def _is_asymmetric(mask: np.ndarray) -> bool:
    return len(_dihedral_variants(mask)) == 8


def _random_connected_mask(h: int, w: int) -> np.ndarray | None:
    area = h * w
    indices = list(range(area))
    for _ in range(120):
        count = random.randint(max(4, area // 2), area - 1)
        chosen = set(random.sample(indices, count))
        mask = np.zeros((h, w), dtype=bool)
        for idx in chosen:
            mask[idx // w, idx % w] = True
        comps = _component_coords(mask)
        if len(comps) != 1:
            continue
        if not _is_asymmetric(mask):
            continue
        return mask
    return None


def _random_template(h: int, w: int) -> np.ndarray | None:
    area = h * w
    for _ in range(200):
        red_mask = _random_connected_mask(h, w)
        if red_mask is None:
            continue
        red_count = int(red_mask.sum())
        if red_count <= 0 or red_count >= area:
            continue
        template = np.where(red_mask, RED, BLUE).astype(int)
        if BLUE not in template or RED not in template:
            continue
        if not _is_asymmetric(template == RED):
            continue
        if not _is_asymmetric(template == BLUE):
            continue
        return template
    return None


def _place_template(grid: np.ndarray, template: np.ndarray, corner: str) -> tuple[int, int, int, int]:
    th, tw = template.shape
    r0, c0 = _corner_bounds(grid.shape[0], th, tw, corner)
    grid[r0 : r0 + th, c0 : c0 + tw] = template
    return r0, c0, r0 + th - 1, c0 + tw - 1


def _build_markers(
    side: int,
    direction: tuple[int, int],
    ray: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    center = (side // 2, side // 2)
    ray_markers = ray[1:-1]
    if len(ray_markers) < 2:
        return []

    on_ray_count = 1 if len(ray_markers) == 2 else random.randint(1, min(2, len(ray_markers)))
    on_ray = random.sample(ray_markers, on_ray_count)

    quadrant = _quadrant_cells(side, center, direction, forbidden | set(ray))
    off_ray_candidates = [cell for cell in quadrant if cell not in set(on_ray)]
    if len(off_ray_candidates) < 2:
        return []

    total_count = random.randint(3, 5)
    off_ray_count = total_count - on_ray_count
    if off_ray_count < 2:
        off_ray_count = 2
        total_count = on_ray_count + off_ray_count
    if off_ray_count > len(off_ray_candidates):
        return []

    off_ray = random.sample(off_ray_candidates, off_ray_count)
    markers = on_ray + off_ray
    random.shuffle(markers)
    return markers


def _expected_output(
    side: int,
    direction: tuple[int, int],
    template: np.ndarray,
    template_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    out = np.zeros((side, side), dtype=int)
    center = (side // 2, side // 2)
    ray = _ray_cells(center, direction, out.shape)
    for r, c in ray:
        out[r, c] = GREEN
    _stamp_endpoint(out, template, ray[-1])
    tr0, tc0, tr1, tc1 = template_bbox
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = BLACK
    return out


def _validate_instance(
    inp: np.ndarray,
    out: np.ndarray,
    direction: tuple[int, int],
    template_bbox: tuple[int, int, int, int],
    markers: list[tuple[int, int]],
) -> bool:
    side = inp.shape[0]
    center = (side // 2, side // 2)
    ray = _ray_cells(center, direction, inp.shape)
    ray_set = set(ray)

    if int(inp[center]) != GREEN:
        return False
    if not markers or not any(m in ray_set for m in markers) or not any(m not in ray_set for m in markers):
        return False
    if any(_sign(r - center[0]) != direction[0] or _sign(c - center[1]) != direction[1] for r, c in markers):
        return False

    mask = (inp == BLUE) | (inp == RED)
    comps = _component_coords(mask)
    if len(comps) != 1:
        return False
    bbox = _bbox_from_coords(comps[0])
    if bbox != template_bbox:
        return False
    template = _crop_bbox(inp, bbox)
    if not np.all((template == BLUE) | (template == RED)):
        return False

    if np.array_equal(inp, out):
        return False
    if not np.array_equal(main(inp), out):
        return False

    expected_green = set(ray)
    actual_green = {tuple(map(int, rc)) for rc in np.argwhere(out == GREEN)}
    red_cells = {tuple(map(int, rc)) for rc in np.argwhere(out == RED)}
    if not expected_green.issubset(actual_green | red_cells):
        return False
    if any(m not in ray_set and int(out[m]) != BLACK for m in markers):
        return False

    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    side = random.choice(GRID_SIDES)
    direction = random.choice(DIRECTIONS)
    th, tw = random.choice(TEMPLATE_SIZES)
    template = _random_template(th, tw)
    if template is None:
        return None

    center = (side // 2, side // 2)
    endpoint_corner = _corner_from_direction(direction)
    template_corner_choices = [c for c in CORNER_NAMES if c != endpoint_corner]
    if _opposite_corner(endpoint_corner) in template_corner_choices and random.random() < 0.55:
        template_corner = _opposite_corner(endpoint_corner)
    else:
        template_corner = random.choice(template_corner_choices)

    inp = np.zeros((side, side), dtype=int)
    inp[center] = GREEN
    template_bbox = _place_template(inp, template, template_corner)

    ray = _ray_cells(center, direction, inp.shape)
    forbidden = {
        (r, c)
        for r in range(template_bbox[0], template_bbox[2] + 1)
        for c in range(template_bbox[1], template_bbox[3] + 1)
    }
    if any(cell in forbidden for cell in ray):
        return None

    markers = _build_markers(side, direction, ray, forbidden)
    if not markers:
        return None
    for r, c in markers:
        inp[r, c] = GREEN

    out = _expected_output(side, direction, template, template_bbox)
    if not _validate_instance(inp, out, direction, template_bbox, markers):
        return None

    in_sig = _hash_grid(inp)
    out_sig = _hash_grid(out)
    if in_sig in _SEEN_INPUTS or out_sig in _SEEN_OUTPUTS:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sample = _gen_attempt()
        if sample is None:
            continue
        inp, out = sample
        _SEEN_INPUTS.add(_hash_grid(inp))
        _SEEN_OUTPUTS.add(_hash_grid(out))
        return inp
    raise RuntimeError("failed to generate a valid unique input")
