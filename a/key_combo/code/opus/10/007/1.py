from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np


BG = 0
RED = 2
RECT_FILL = 5
MARKER_COLORS = (1, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDE_ORDER = ("up", "down", "left", "right")
OPPOSITE_SIDE = {"up": "down", "down": "up", "left": "right", "right": "left"}
GEN_RETRIES = 400
_GEN_COUNTER = 0


@dataclass(frozen=True)
class Component:
    coords: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    area: int
    colors: tuple[int, ...]


@dataclass(frozen=True)
class SpriteSpec:
    color: int
    body: tuple[tuple[int, int], ...]
    height: int
    width: int
    side: str
    anchor: int


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def find_components(grid: np.ndarray, mask: np.ndarray | None = None) -> list[Component]:
    if mask is None:
        mask = grid != BG
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[Component] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            colors = set()
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                colors.add(int(grid[rr, cc]))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(
                Component(
                    coords=tuple(coords),
                    bbox=bbox_coords(coords),
                    area=len(coords),
                    colors=tuple(sorted(colors)),
                )
            )
    return out


def is_connected(coords: Iterable[tuple[int, int]]) -> bool:
    pts = set(coords)
    if not pts:
        return False
    q = deque([next(iter(pts))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen == pts


def normalize_shape(coords: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = list(coords)
    r0, c0, _, _ = bbox_coords(pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def body_bbox(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    h = 1 + max(r for r, _ in shape)
    w = 1 + max(c for _, c in shape)
    return h, w


def parse_sprite(component: Component, grid: np.ndarray) -> SpriteSpec:
    colors = set(component.colors)
    assert RED in colors and len(colors) == 2, "each source sprite must contain one red indicator and one body color"
    body_color = next(color for color in colors if color != RED)
    body = tuple((r, c) for r, c in component.coords if int(grid[r, c]) == body_color)
    red_cells = [(r, c) for r, c in component.coords if int(grid[r, c]) == RED]
    assert len(red_cells) == 1
    assert 3 <= len(body) <= 6 and is_connected(body)
    red_r, red_c = red_cells[0]

    r0, c0, r1, c1 = bbox_coords(body)
    norm_body = normalize_shape(body)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    neighbors = {(red_r + dr, red_c + dc) for dr, dc in DIR4}
    assert neighbors & set(body), "indicator must be 4-neighbor adjacent to the body"

    side = None
    anchor = None
    if red_r == r0 - 1 and c0 <= red_c <= c1:
        side = "up"
        anchor = red_c - c0
    elif red_r == r1 + 1 and c0 <= red_c <= c1:
        side = "down"
        anchor = red_c - c0
    elif red_c == c0 - 1 and r0 <= red_r <= r1:
        side = "left"
        anchor = red_r - r0
    elif red_c == c1 + 1 and r0 <= red_r <= r1:
        side = "right"
        anchor = red_r - r0
    assert side is not None and anchor is not None, "indicator must sit just outside one side of the body bbox"

    return SpriteSpec(color=body_color, body=norm_body, height=h, width=w, side=side, anchor=anchor)


def choose_target_component(grid: np.ndarray) -> Component:
    components = find_components(grid, grid != BG)
    assert len(components) >= 3
    components = sorted(components, key=lambda comp: comp.area, reverse=True)
    target = components[0]
    assert target.area > components[1].area
    r0, c0, r1, c1 = target.bbox
    assert target.area == (r1 - r0 + 1) * (c1 - c0 + 1), "target must be a filled rectangle with embedded markers"
    return target


def anchor_for_mode(spec: SpriteSpec, mode: str) -> int:
    if mode == "correct":
        return spec.anchor
    if mode == "topleft":
        return 0
    if mode == "center":
        return spec.width // 2 if spec.side in {"up", "down"} else spec.height // 2
    raise ValueError(f"unknown mode: {mode}")


def stamp_top_left(marker: tuple[int, int], spec: SpriteSpec, *, mode: str, reverse: bool = False) -> tuple[int, int]:
    mr, mc = marker
    side = OPPOSITE_SIDE[spec.side] if reverse else spec.side
    anchor = anchor_for_mode(spec, mode) if not reverse else spec.anchor

    if side == "up":
        return mr + 1, mc - anchor
    if side == "down":
        return mr - spec.height, mc - anchor
    if side == "left":
        return mr - anchor, mc + 1
    if side == "right":
        return mr - anchor, mc - spec.width
    raise ValueError(f"unknown side: {side}")


def render_from_parts(
    base: np.ndarray,
    markers: dict[int, tuple[int, int]],
    sprites: dict[int, SpriteSpec],
    *,
    mode: str = "correct",
    reverse: bool = False,
) -> np.ndarray:
    out = base.copy()
    fill = int(np.bincount(out.ravel()).argmax())
    order = sorted(markers.items(), key=lambda item: (item[1][0] + item[1][1], item[1][0], item[1][1]))
    h, w = out.shape

    for color, marker in order:
        spec = sprites[color]
        top, left = stamp_top_left(marker, spec, mode=mode, reverse=reverse)
        for dr, dc in spec.body:
            rr = top + dr
            cc = left + dc
            if 0 <= rr < h and 0 <= cc < w and out[rr, cc] == fill:
                out[rr, cc] = color
        mr, mc = marker
        out[mr, mc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the grey target rectangle and stamp each matching external sprite onto the marker-indicated side.

    Input structure:
    - The background is black.
    - One large filled grey rectangle contains 2-3 isolated marker pixels of distinct colors.
    - Outside the rectangle, each marker color has one same-colored 4-connected sprite body plus exactly one adjacent red indicator pixel just outside the body's bounding box.

    Rule:
    - Find the large filled rectangle and crop to its bounding box for the output base.
    - Match each interior marker to the external sprite with the same non-red color.
    - Use the red pixel's side relative to the sprite body bbox to choose which side of the marker receives the stamp, and align the stamp as if the omitted red pixel itself moved onto the marker.
    - Stamp only the sprite body color, preserve the marker pixel, and if stamps would conflict, earlier markers in top-left Manhattan order keep priority.

    Key insight:
    - The red pixel is not part of the copied pattern; it only supplies the viewing side and the along-edge alignment needed to place the matching body unambiguously.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    target = choose_target_component(g)
    crop = crop_bbox(g, target.bbox)
    fill = int(np.bincount(crop.ravel()).argmax())
    assert fill == RECT_FILL

    markers: dict[int, tuple[int, int]] = {}
    for r, c in np.argwhere(crop != fill):
        color = int(crop[r, c])
        assert color not in {BG, RED, fill}
        assert color not in markers
        markers[color] = (int(r), int(c))
    assert 2 <= len(markers) <= 3

    sprites: dict[int, SpriteSpec] = {}
    for component in find_components(g, g != BG):
        if component == target:
            continue
        spec = parse_sprite(component, g)
        assert spec.color in markers
        assert spec.color not in sprites
        sprites[spec.color] = spec
    assert set(sprites) == set(markers)

    sizes = {(spec.height, spec.width) for spec in sprites.values()}
    assert len(sizes) == len(sprites), "all sprite bbox sizes must be distinct"
    return render_from_parts(crop, markers, sprites)


def all_shapes() -> dict[tuple[int, int], list[tuple[tuple[int, int], ...]]]:
    shapes: dict[tuple[int, int], set[tuple[tuple[int, int], ...]]] = {}
    cells = [(r, c) for r in range(3) for c in range(3)]
    for mask in range(1 << len(cells)):
        pts = [cells[i] for i in range(len(cells)) if (mask >> i) & 1]
        if not (3 <= len(pts) <= 6):
            continue
        if not is_connected(pts):
            continue
        shape = normalize_shape(pts)
        h, w = body_bbox(shape)
        if (h, w) not in {(1, 3), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)}:
            continue
        shapes.setdefault((h, w), set()).add(shape)
    return {key: sorted(value) for key, value in shapes.items()}


SHAPES_BY_SIZE = all_shapes()
SIZE_OPTIONS = tuple(sorted(SHAPES_BY_SIZE))
DIR_PATTERNS = (
    ("up", "down"),
    ("left", "right"),
    ("up", "left", "right"),
    ("down", "left", "up"),
    ("up", "down", "right"),
    ("left", "right", "down"),
)


def side_anchors(shape: tuple[tuple[int, int], ...], side: str) -> list[int]:
    h, w = body_bbox(shape)
    if side == "up":
        return sorted({c for r, c in shape if r == 0})
    if side == "down":
        return sorted({c for r, c in shape if r == h - 1})
    if side == "left":
        return sorted({r for r, c in shape if c == 0})
    if side == "right":
        return sorted({r for r, c in shape if c == w - 1})
    raise ValueError(f"unknown side: {side}")


def marker_from_bbox(top: int, left: int, spec: SpriteSpec) -> tuple[int, int]:
    if spec.side == "up":
        return top - 1, left + spec.anchor
    if spec.side == "down":
        return top + spec.height, left + spec.anchor
    if spec.side == "left":
        return top + spec.anchor, left - 1
    if spec.side == "right":
        return top + spec.anchor, left + spec.width
    raise ValueError(f"unknown side: {spec.side}")


def build_output_layout(
    rect_h: int, rect_w: int, specs: list[SpriteSpec]
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], ...]]] | None:
    marker_margin = 2
    occupied = np.zeros((rect_h, rect_w), dtype=bool)
    reserved = np.zeros((rect_h, rect_w), dtype=bool)
    markers: list[tuple[int, int] | None] = [None] * len(specs)
    bodies: list[tuple[tuple[int, int], ...] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda i: -(len(specs[i].body)))

    def with_halo(cells: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
        out = set()
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rect_h and 0 <= cc < rect_w:
                        out.add((rr, cc))
        return out

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        spec = specs[idx]
        candidates = []
        for top in range(rect_h - spec.height + 1):
            for left in range(rect_w - spec.width + 1):
                marker = marker_from_bbox(top, left, spec)
                mr, mc = marker
                if not (marker_margin <= mr < rect_h - marker_margin):
                    continue
                if not (marker_margin <= mc < rect_w - marker_margin):
                    continue
                body_cells = tuple(sorted((top + dr, left + dc) for dr, dc in spec.body))
                if marker in body_cells:
                    continue
                cells = set(body_cells) | {marker}
                if any(occupied[r, c] for r, c in cells):
                    continue
                halo = with_halo(cells)
                if any(reserved[r, c] for r, c in halo):
                    continue
                candidates.append((marker, body_cells, halo))
        random.shuffle(candidates)

        for marker, body_cells, halo in candidates:
            markers[idx] = marker
            bodies[idx] = body_cells
            for r, c in set(body_cells) | {marker}:
                occupied[r, c] = True
            for r, c in halo:
                reserved[r, c] = True
            if dfs(pos + 1):
                return True
            markers[idx] = None
            bodies[idx] = None
            for r, c in set(body_cells) | {marker}:
                occupied[r, c] = False
            for r, c in halo:
                reserved[r, c] = False
        return False

    if not dfs(0):
        return None
    return [m for m in markers if m is not None], [b for b in bodies if b is not None]


def make_sprite_patch(spec: SpriteSpec) -> np.ndarray:
    red_pos = marker_from_bbox(0, 0, spec)
    all_cells = set(spec.body) | {red_pos}
    r0, c0, r1, c1 = bbox_coords(all_cells)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in spec.body:
        patch[r - r0, c - c0] = spec.color
    patch[red_pos[0] - r0, red_pos[1] - c0] = RED
    return patch


def try_place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    rect_box: tuple[int, int, int, int],
    occupied: np.ndarray,
    *,
    padding: int,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    mask = patch != BG
    candidates = []
    for top in range(h - ph + 1):
        for left in range(w - pw + 1):
            if not (top + ph - 1 < rect_box[0] - 1 or top > rect_box[2] + 1 or left + pw - 1 < rect_box[1] - 1 or left > rect_box[3] + 1):
                continue
            ok = True
            for rr in range(ph):
                for cc in range(pw):
                    if not mask[rr, cc]:
                        continue
                    gr, gc = top + rr, left + cc
                    if occupied[gr, gc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                candidates.append((top, left))
    random.shuffle(candidates)

    for top, left in candidates:
        touched = []
        ok = True
        for rr in range(ph):
            for cc in range(pw):
                if not mask[rr, cc]:
                    continue
                gr, gc = top + rr, left + cc
                for dr in range(-padding, padding + 1):
                    for dc in range(-padding, padding + 1):
                        nr, nc = gr + dr, gc + dc
                        if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
                touched.append((gr, gc))
            if not ok:
                break
        if not ok:
            continue
        for rr in range(ph):
            for cc in range(pw):
                if mask[rr, cc]:
                    grid[top + rr, left + cc] = patch[rr, cc]
        for gr, gc in touched:
            for dr in range(-padding, padding + 1):
                for dc in range(-padding, padding + 1):
                    nr, nc = gr + dr, gc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        occupied[nr, nc] = True
        return True
    return False


def intended_output(
    rect_h: int,
    rect_w: int,
    fill: int,
    markers: list[tuple[int, int]],
    specs: list[SpriteSpec],
) -> np.ndarray:
    base = np.full((rect_h, rect_w), fill, dtype=int)
    marker_map = {spec.color: marker for spec, marker in zip(specs, markers)}
    sprite_map = {spec.color: spec for spec in specs}
    for spec, marker in zip(specs, markers):
        base[marker] = spec.color
    return render_from_parts(base, marker_map, sprite_map)


def choose_specs() -> list[SpriteSpec] | None:
    global _GEN_COUNTER
    dirs = DIR_PATTERNS[_GEN_COUNTER % len(DIR_PATTERNS)]
    _GEN_COUNTER += 1

    colors = random.sample(MARKER_COLORS, len(dirs))
    sizes = random.sample(SIZE_OPTIONS, len(dirs))
    specs: list[SpriteSpec] = []
    for color, side, size in zip(colors, dirs, sizes):
        candidates = []
        for shape in SHAPES_BY_SIZE[size]:
            h, w = size
            if side in {"up", "down"} and w == 1:
                continue
            if side in {"left", "right"} and h == 1:
                continue
            anchors = side_anchors(shape, side)
            if not anchors:
                continue
            for anchor in anchors:
                candidates.append(SpriteSpec(color=color, body=shape, height=h, width=w, side=side, anchor=anchor))
        if not candidates:
            return None
        # Prefer off-corner anchors when possible so the red pixel's orthogonal offset matters.
        scored = sorted(
            candidates,
            key=lambda spec: (
                abs(spec.anchor - ((spec.width - 1) / 2 if spec.side in {"up", "down"} else (spec.height - 1) / 2)),
                -len(spec.body),
                random.random(),
            ),
            reverse=True,
        )
        specs.append(scored[0])
    return specs


def reject_if_shortcuts_match(inp: np.ndarray) -> bool:
    target = choose_target_component(inp)
    crop = crop_bbox(inp, target.bbox)
    fill = RECT_FILL
    markers = {int(crop[r, c]): (int(r), int(c)) for r, c in np.argwhere(crop != fill)}
    sprites: dict[int, SpriteSpec] = {}
    for component in find_components(inp, inp != BG):
        if component == target:
            continue
        spec = parse_sprite(component, inp)
        sprites[spec.color] = spec

    correct = render_from_parts(crop, markers, sprites)
    for mode in ("topleft", "center"):
        if np.array_equal(correct, render_from_parts(crop, markers, sprites, mode=mode)):
            return True
    if np.array_equal(correct, render_from_parts(crop, markers, sprites, reverse=True)):
        return True
    return False


def validate_semantics(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False
    if not np.array_equal(out, expected):
        return False
    if reject_if_shortcuts_match(inp):
        return False

    comps = sorted(find_components(inp, inp != BG), key=lambda comp: comp.area, reverse=True)
    target = comps[0]
    if len(comps) not in {3, 4}:
        return False
    if int(np.bincount(crop_bbox(inp, target.bbox).ravel()).argmax()) != RECT_FILL:
        return False
    sprite_sizes = set()
    for comp in comps[1:]:
        spec = parse_sprite(comp, inp)
        sprite_sizes.add((spec.height, spec.width))
    return len(sprite_sizes) == len(comps) - 1


def gen_attempt() -> np.ndarray | None:
    specs = choose_specs()
    if specs is None:
        return None

    rect_h = random.randint(8, 10)
    rect_w = random.randint(8, 10)
    layout = build_output_layout(rect_h, rect_w, specs)
    if layout is None:
        return None
    markers, _bodies = layout
    expected = intended_output(rect_h, rect_w, RECT_FILL, markers, specs)

    grid_h = rect_h + random.randint(6, 10)
    grid_w = rect_w + random.randint(6, 10)
    if max(grid_h, grid_w) > 30:
        return None
    top = random.randint(3, grid_h - rect_h - 3)
    left = random.randint(3, grid_w - rect_w - 3)

    inp = np.zeros((grid_h, grid_w), dtype=int)
    inp[top : top + rect_h, left : left + rect_w] = RECT_FILL
    for spec, marker in zip(specs, markers):
        inp[top + marker[0], left + marker[1]] = spec.color

    occupied = np.zeros_like(inp, dtype=bool)
    occupied[max(0, top - 1) : min(grid_h, top + rect_h + 1), max(0, left - 1) : min(grid_w, left + rect_w + 1)] = True
    rect_box = (top, left, top + rect_h - 1, left + rect_w - 1)

    patches = [make_sprite_patch(spec) for spec in specs]
    order = list(range(len(specs)))
    random.shuffle(order)
    for idx in order:
        if not try_place_patch(inp, patches[idx], rect_box, occupied, padding=1):
            return None

    if not validate_semantics(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid direction-stamp puzzle input")
