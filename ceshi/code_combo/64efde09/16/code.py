# Combo variant of 64efde09 (index 16)
# combo_spec: D1/3+D8/0
# Name: 尺寸匹配+宽矩形模板

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Original task: every sprite was an exact dihedral copy of one thin Nx2 template, so the solver could match full
#      color arrays, recover a template cell id, then copy marker-labeled rays from that exact port to every transform.
#    - Combo task: sprites only agree on rectangle size, not on internal coloring, and the rectangle is now N x M with
#      M in {3, 4}. A concrete example is three 8x3 / 3x8 dense rectangles that all have one visibly dominant long edge,
#      but whose interiors are colored differently. A marker that hits the 3rd cell on the active edge of one sprite
#      should trigger rays from the 3rd active-edge cell of every other equally sized sprite, even when one sprite is
#      rotated and therefore appears as 3x8 instead of 8x3.
#    - The changed step is the correspondence rule: instead of "same template cell under exact transform", the port is
#      now "same geometric index along the active long edge inside the same size class". The size class is unordered:
#      (8, 3) and (3, 8) are the same class.
# b) Complexity calibration
#    - The width change is visually obvious because sprites are 3 or 4 cells thick instead of 2.
#    - The size-only match is also visible because same-sized sprites no longer have identical interiors; relying on
#      full-color equality would fail on these samples.
#    - The design keeps only one rectangle size class per puzzle so humans see repeated evidence for the same rule rather
#      than several unrelated correspondences. At the same time, the generator forces both portrait and landscape
#      orientations to appear so the unordered size match is genuinely exercised.
# c) Hidden assumption audit
#    - Original `main()` assumed one sprite could serve as the exact template for all others. That no longer holds, so
#      the new solver never compares full sprite arrays across components.
#    - Original port ids came from an `ids` array pushed through dihedral transforms. That breaks under size-only
#      matching, so the new solver replaces template-cell ids with a clockwise geometric ordering of the active edge.
#    - Once correspondence is size-only, active edges must have the same length within a size class. The generator
#      therefore guarantees that the active edge is always a long side; `main()` asserts that property explicitly.
# d) Placement legality
#    - Sprites remain dense rectangles, axis-aligned, non-overlapping, and inside the grid.
#    - Each sprite is placed so its active side faces a boundary, leaving enough background clearance for both markers
#      and output rays. Final validation still calls `main()` and rejects accidental conflicts.
# e) Visual observability
#    - A human can inspect a sample and notice two coupled facts: same-sized rectangles behave together even though their
#      color layouts differ, and the relevant port is the same offset along the edge with the strongest repeated color.
#    - Including both N x M and M x N sprites makes the size-only grouping visible without relying on hidden color ids.
# f) Visual cleanliness
#    - Markers stay isolated on the outer border, and rays always travel outward toward the nearest boundary instead of
#      weaving through the middle of the canvas.
#    - Padding between sprites keeps bodies and rays visually separable.
# g) Change inferability
#    - Marker colors are unique and kept outside the sprite palette, so each ray family has a single visible source.
#    - The sprite palette is shared across all sprites but their arrangements differ, which makes "match by size, not by
#      exact texture" inferable from a few examples.
# h) New solver assumptions guaranteed by the generator
#    - Every sprite is a non-square dense rectangle whose active side is the unique highest-scoring edge among all four.
#    - That active side is always a long side, so every sprite in a size class has the same number of valid ports.
#    - At least one portrait and one landscape orientation are present, and at least one same-size pair is not a
#      dihedral color match, preventing degeneration back to the original exact-template task.


def _dihedral_images(arr: np.ndarray) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(arr, k)
        images.append(rotated)
        images.append(np.fliplr(rotated))
    return images


def _is_dihedral_match(a: np.ndarray, b: np.ndarray) -> bool:
    return any(img.shape == b.shape and np.array_equal(img, b) for img in _dihedral_images(a))


def _edge_specs(sub: np.ndarray) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    h, w = sub.shape
    assert h != w, "sprites must be non-square rectangles"
    return [
        ((0, -1), [(r, 0) for r in range(h)]),
        ((1, 0), [(h - 1, c) for c in range(w)]),
        ((0, 1), [(r, w - 1) for r in range(h - 1, -1, -1)]),
        ((-1, 0), [(0, c) for c in range(w - 1, -1, -1)]),
    ]


def _active_side(sub: np.ndarray) -> tuple[tuple[int, int], list[tuple[int, int]], int]:
    scored = []
    for direction, coords in _edge_specs(sub):
        vals = [int(sub[r, c]) for r, c in coords]
        scored.append((max(Counter(vals).values()), direction, list(coords)))

    scored.sort(key=lambda item: item[0], reverse=True)
    assert scored[0][0] > scored[1][0], "active side must be structurally unique across all edges"
    return scored[0][1], scored[0][2], scored[0][0]


def _size_key(sub: np.ndarray) -> tuple[int, int]:
    h, w = map(int, sub.shape)
    return (max(h, w), min(h, w))


def _first_marker_on_ray(
    g: np.ndarray,
    bg: int,
    markers: dict[tuple[int, int], int],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> int | None:
    H, W = g.shape
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W:
        if int(g[r, c]) != bg:
            return markers.get((r, c))
        r += dr
        c += dc
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate boundary-marker ray labels across every equally sized wide rectangle, even when their internal colors differ.

    Input structure:
    - The background is the most frequent color.
    - Non-background objects are dense non-square rectangles plus isolated single-pixel markers.
    - Rectangles belong to a shared unordered size class `(N, M)` with `M` equal to 3 or 4, but different sprites may
      appear as `N x M` or `M x N`, and their interior color arrangements need not match each other.

    Rule:
    - For each rectangle, compare all four edges and choose the unique edge whose largest same-color multiplicity is
      strictly highest; that edge is the active side.
    - Scan outward from each active-side cell. If the first non-background cell on that ray is a marker, that marker
      labels the geometric index of that port along the active edge within the sprite's size class.
    - Copy each discovered marker color outward from the same active-edge index of every rectangle in that size class
      until the ray hits a non-background cell or exits the grid.

    Key insight:
    - The marker is no longer tied to one exact colored template cell. It identifies an offset along the active long edge
      of a rectangle size class, so the label follows size and edge geometry rather than full sprite texture.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False)

    sprites_raw: list[dict] = []
    markers: dict[tuple[int, int], int] = {}
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = int(comp["area"])
        if area == 1:
            r, c = map(int, comp["coords"][0])
            markers[(r, c)] = int(g[r, c])
        else:
            assert area == h * w, "sprite components must be dense rectangles"
            sprites_raw.append({"bbox": (r0, c0, r1, c1), "sub": crop_bbox(g, (r0, c0, r1, c1))})

    assert len(sprites_raw) >= 2, "expected multiple rectangular sprites"
    assert markers, "expected isolated control markers"
    assert len(set(markers.values())) == len(markers), "control marker colors must be unique"

    sprites = []
    group_port_lengths: dict[tuple[int, int], int] = {}
    for sprite in sprites_raw:
        direction, active_coords, _score = _active_side(sprite["sub"])
        key = _size_key(sprite["sub"])
        assert len(active_coords) == key[0], "size-only matching expects the active edge to be the long side"
        if key in group_port_lengths:
            assert group_port_lengths[key] == len(active_coords), "size class must have consistent port count"
        else:
            group_port_lengths[key] = len(active_coords)
        sprites.append(
            {
                **sprite,
                "size_key": key,
                "direction": direction,
                "active_coords": active_coords,
            }
        )

    marker_to_port: dict[int, tuple[tuple[int, int], int]] = {}
    port_to_color: dict[tuple[tuple[int, int], int], int] = {}
    group_to_port_colors: dict[tuple[int, int], dict[int, int]] = {}
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        key = sprite["size_key"]
        for port_index, (lr, lc) in enumerate(sprite["active_coords"]):
            marker_color = _first_marker_on_ray(g, bg, markers, r0 + lr, c0 + lc, dr, dc)
            if marker_color is None:
                continue
            port_key = (key, port_index)
            if marker_color in marker_to_port:
                assert marker_to_port[marker_color] == port_key
            if port_key in port_to_color:
                assert port_to_color[port_key] == marker_color
            marker_to_port[marker_color] = port_key
            port_to_color[port_key] = marker_color
            group_to_port_colors.setdefault(key, {})[port_index] = marker_color

    assert set(marker_to_port) == set(markers.values()), "every marker must label one size-class port"

    out = g.copy()
    H, W = g.shape
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for port_index, color in group_to_port_colors.get(sprite["size_key"], {}).items():
            lr, lc = sprite["active_coords"][port_index]
            r = r0 + lr + dr
            c = c0 + lc + dc
            while 0 <= r < H and 0 <= c < W:
                if int(g[r, c]) != bg:
                    break
                assert int(out[r, c]) in (bg, color), "control rays must not conflict"
                out[r, c] = color
                r += dr
                c += dc

    return out


# --------------- generator ---------------

_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (24, 30)
_GRID_W_RANGE = (30, 34)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_WIDTH_CHOICES = (3, 4)
_TEMPLATE_COLOR_COUNT = (3, 4)
_ACTIVE_MAJORITY_RANGE = (5, 7)
_SPRITE_COUNT_RANGE = (3, 4)
_NUM_MARKERS = 3
_PLACEMENT_PAD = 2
_PLACEMENT_CLEARANCE = 2
_PLACEMENT_RETRIES = 140
_SPRITE_RETRIES = 80
_GEN_RETRIES = 500


def _build_edge_with_max_count(length: int, colors: list[int], target: int, anchor: int | None = None) -> list[int] | None:
    if not (1 <= target <= length):
        return None
    if len(colors) < 2 and target < length:
        return None
    if length > len(colors) * target:
        return None

    dominant = random.choice(colors) if anchor is None else anchor
    others = [c for c in colors if c != dominant]
    if target < length and not others:
        return None

    counts = {dominant: target}
    remaining = length - target
    supply: list[int] = []
    for color in others:
        supply.extend([color] * target)
    random.shuffle(supply)
    if remaining > len(supply):
        return None
    for color in supply[:remaining]:
        counts[color] = counts.get(color, 0) + 1

    edge: list[int] = []
    for color, cnt in counts.items():
        edge.extend([color] * cnt)
    random.shuffle(edge)
    assert max(Counter(edge).values()) == target
    return edge


def _design_sprite(n: int, m: int, palette: list[int], orientation: str) -> tuple[np.ndarray, tuple[int, int]] | None:
    long_len = n
    active_score = random.randint(_ACTIVE_MAJORITY_RANGE[0], min(long_len, _ACTIVE_MAJORITY_RANGE[1]))
    anchor = random.choice(palette)
    min_other = (long_len + len(palette) - 1) // len(palette)
    max_other = active_score - 1
    if min_other > max_other:
        return None

    if orientation == "portrait":
        active_direction = random.choice([(0, -1), (0, 1)])
        columns: list[list[int]] = []
        active_col = _build_edge_with_max_count(long_len, palette, active_score, anchor=anchor)
        if active_col is None:
            return None
        columns.append(active_col)
        for _ in range(m - 1):
            target = random.randint(min_other, max_other)
            col = _build_edge_with_max_count(long_len, palette, target)
            if col is None:
                return None
            columns.append(col)
        sprite = np.column_stack(columns).astype(int)
        if active_direction == (0, 1):
            sprite = np.fliplr(sprite)
    else:
        active_direction = random.choice([(-1, 0), (1, 0)])
        rows: list[list[int]] = []
        active_row = _build_edge_with_max_count(long_len, palette, active_score, anchor=anchor)
        if active_row is None:
            return None
        rows.append(active_row)
        for _ in range(m - 1):
            target = random.randint(min_other, max_other)
            row = _build_edge_with_max_count(long_len, palette, target)
            if row is None:
                return None
            rows.append(row)
        sprite = np.array(rows, dtype=int)
        if active_direction == (1, 0):
            sprite = np.flipud(sprite)

    if len(set(map(int, sprite.flat))) < min(3, len(palette)):
        return None

    direction, active_coords, _score = _active_side(sprite)
    if direction != active_direction:
        return None
    if len(active_coords) != long_len:
        return None
    return sprite, active_direction


def _random_sprite_position(H: int, W: int, h: int, w: int, dr: int, dc: int) -> tuple[int, int]:
    pad = _PLACEMENT_CLEARANCE
    if dr == 0 and dc == -1:
        return random.randint(0, H - h), random.randint(pad, W - w)
    if dr == 0 and dc == 1:
        return random.randint(0, H - h), random.randint(0, W - w - pad - 1)
    if dr == -1 and dc == 0:
        return random.randint(pad, H - h), random.randint(0, W - w)
    return random.randint(0, H - h - pad - 1), random.randint(0, W - w)


def _occupied_set(placements: list[tuple]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        for r in range(h):
            for c in range(w):
                cells.add((r0 + r, c0 + c))
    return cells


def _place_sprites(H: int, W: int, sprites: list[tuple[np.ndarray, tuple[int, int]]]) -> list[tuple] | None:
    placements: list[tuple] = []
    pad = _PLACEMENT_PAD
    for sub, direction in sprites:
        h, w = sub.shape
        dr, dc = direction
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            r0, c0 = _random_sprite_position(H, W, h, w, dr, dc)
            occ = _occupied_set(placements)
            if any(
                (r0 + r, c0 + c) in occ
                for r in range(-pad, h + pad)
                for c in range(-pad, w + pad)
            ):
                continue
            _direction, active_coords, _score = _active_side(sub)
            placements.append((r0, c0, sub, direction, active_coords))
            placed = True
            break
        if not placed:
            return None
    return placements


def _boundary_position(r: int, c: int, dr: int, dc: int, H: int, W: int) -> tuple[int, int]:
    if dr == 0 and dc == -1:
        return r, 0
    if dr == 0 and dc == 1:
        return r, W - 1
    if dr == -1 and dc == 0:
        return 0, c
    return H - 1, c


def _place_markers(
    grid: np.ndarray,
    sprite_placements: list[tuple],
    long_len: int,
    palette: list[int],
) -> bool:
    H, W = grid.shape
    if long_len < _NUM_MARKERS:
        return False

    labeled_indices = random.sample(list(range(long_len)), _NUM_MARKERS)
    marker_colors = random.sample([c for c in _NON_BG_COLORS if c not in palette], _NUM_MARKERS)

    for color, port_index in zip(marker_colors, labeled_indices):
        r0, c0, _sub, direction, active_coords = random.choice(sprite_placements)
        dr, dc = direction
        lr, lc = active_coords[port_index]
        marker_r, marker_c = _boundary_position(r0 + lr, c0 + lc, dr, dc, H, W)

        cr, cc = r0 + lr + dr, c0 + lc + dc
        while (cr, cc) != (marker_r, marker_c) and 0 <= cr < H and 0 <= cc < W:
            if int(grid[cr, cc]) != _BG:
                return False
            cr += dr
            cc += dc

        if int(grid[marker_r, marker_c]) != _BG:
            return False
        grid[marker_r, marker_c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_TEMPLATE_N_RANGE)
    m = random.choice(_TEMPLATE_WIDTH_CHOICES)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)
    palette = random.sample(_NON_BG_COLORS, random.randint(*_TEMPLATE_COLOR_COUNT))

    num_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    orientations = ["portrait", "landscape"] + [random.choice(["portrait", "landscape"]) for _ in range(num_sprites - 2)]
    random.shuffle(orientations)

    sprites: list[tuple[np.ndarray, tuple[int, int]]] = []
    for orientation in orientations:
        built = None
        for _ in range(_SPRITE_RETRIES):
            candidate = _design_sprite(n, m, palette, orientation)
            if candidate is None:
                continue
            sub, direction = candidate
            if any(_is_dihedral_match(sub, prior_sub) for prior_sub, _prior_direction in sprites):
                continue
            built = (sub, direction)
            break
        if built is None:
            return None
        sprites.append(built)

    placements = _place_sprites(H, W, sprites)
    if placements is None:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    for r0, c0, sub, _direction, _active_coords in placements:
        h, w = sub.shape
        grid[r0:r0 + h, c0:c0 + w] = sub

    if not _place_markers(grid, placements, n, palette):
        return None

    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
